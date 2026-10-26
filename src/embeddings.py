"""Embedding backends with graceful degradation.

Resolution order (see config.ProviderConfig.resolve_embedding):
  1. sentence_transformers  — local model, best quality, no API key.
  2. openai                 — hosted embeddings when a key is set.
  3. hashing                — deterministic bag-of-words hashing vector; zero deps,
                              so the demo always runs. Quality is modest but the
                              retrieval plumbing is identical, which is the point.

All backends expose embed(texts) -> list[list[float]] with L2-normalised vectors so
cosine similarity reduces to a dot product downstream.
"""
from __future__ import annotations

import logging
import math
import re
from functools import lru_cache

from .config import PROVIDERS

log = logging.getLogger("embeddings")

_TOKEN = re.compile(r"[a-z0-9]+")


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class HashingEmbedder:
    """Dependency-free embedder: hashed bag-of-words into a fixed-width vector.

    Not state of the art, but deterministic and offline. It lets the whole RAG
    pipeline run and be tested without any model download or API key.
    """

    backend = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN.findall(text.lower()):
                h = hash(tok)
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
                vec[idx] += sign
            out.append(_normalise(vec))
        return out


class SentenceTransformerEmbedder:
    backend = "sentence_transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OpenAIEmbedder:
    backend = "openai"

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [_normalise(list(d.embedding)) for d in resp.data]


@lru_cache(maxsize=4)
def get_embedder(forced: str | None = None):
    backend = PROVIDERS.resolve_embedding(forced)
    try:
        if backend == "sentence_transformers":
            return SentenceTransformerEmbedder(PROVIDERS.embedding_model)
        if backend == "openai":
            return OpenAIEmbedder()
    except Exception as exc:  # pragma: no cover - fallback path
        log.warning("embedding backend %s failed to init (%s); using hashing", backend, exc)
    return HashingEmbedder()
