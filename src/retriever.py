"""Hybrid retrieval: embed the question, pull the most similar daily narratives,
and optionally constrain by structured metadata parsed from the question.

The structured pre-filter is what makes this a *context layer* rather than a plain
semantic search: a question like "when was my load dangerously high" narrows to
high_risk ACWR days before ranking, improving precision on a small corpus.
"""
from __future__ import annotations

import re

from .config import RETRIEVAL
from .embeddings import get_embedder
from .vectorstore import get_vector_store

# phrase -> metadata filter. Deliberately simple and transparent.
_INTENT_FILTERS = {
    r"\b(overtrain\w*|overreach\w*|too much|high(est)? load|load.{0,12}high|"
    r"danger\w*|spik\w*|injury risk|high.?risk|peak load)\b":
        {"acwr_band": {"$in": ["caution", "high_risk"]}},
    r"\b(detrain\w*|undertrain\w*|too little|taper\w*|rest week|easy week)\b":
        {"acwr_band": {"$in": ["detraining"]}},
    r"(under-?recover\w*|hrv.{0,12}(suppress|low)\w*|suppress\w*.{0,6}hrv|low hrv|"
    r"fatigue\w*|run.?down)":
        {"hrv_state": {"$in": ["suppressed"]}},
    r"(worst sleep|poor sleep|bad sleep|sleep.{0,12}(worst|bad|poor)\w*|"
    r"slept.{0,12}(worst|bad|poorly|least)|least sleep|sleep deprivation|sleep debt)":
        {"poor_sleep": 1},
    r"\b(rest day|day off)\b":
        {"is_rest_day": 1},
    r"\b(should i rest|need to rest|not ready|unready)\b":
        {"readiness_flag": {"$in": ["rest", "monitor"]}},
}


def infer_filter(question: str) -> dict | None:
    q = question.lower()
    for pattern, flt in _INTENT_FILTERS.items():
        if re.search(pattern, q):
            return flt
    return None


class Retriever:
    def __init__(self, embedder=None, store=None, embed_backend: str | None = None,
                 force_fallback: bool = False):
        self.embedder = embedder or get_embedder(embed_backend)
        self.store = store or get_vector_store(force_fallback=force_fallback)

    def index(self, docs: list[dict]) -> int:
        """Embed and upsert rendered daily narratives. Returns count indexed."""
        if not docs:
            return 0
        texts = [d["text"] for d in docs]
        embeddings = self.embedder.embed(texts)
        self.store.add(
            ids=[d["id"] for d in docs],
            embeddings=embeddings,
            documents=texts,
            metadatas=[d["metadata"] for d in docs],
        )
        return len(docs)

    def retrieve(self, question: str, top_k: int | None = None,
                 use_structured_filter: bool = True,
                 athlete_id: str | None = None) -> list[dict]:
        top_k = top_k or RETRIEVAL.top_k
        q_emb = self.embedder.embed([question])[0]

        # Scope to one athlete so a shared store can hold a whole cohort without
        # cross-contaminating retrieval. This filter is always applied; the intent
        # filter is layered on top of it.
        base = {"athlete_id": athlete_id} if athlete_id else {}
        intent = infer_filter(question) if use_structured_filter else None
        where = {**base, **intent} if intent else (base or None)

        results = self.store.query(q_emb, top_k=top_k, where=where)
        # If the intent filter was too aggressive and found nothing, fall back to
        # the athlete scope alone so we never return an empty context.
        if not results and intent:
            results = self.store.query(q_emb, top_k=top_k, where=(base or None))
        return [r for r in results if r["score"] >= RETRIEVAL.min_similarity]
