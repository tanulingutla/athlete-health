"""Vector store abstraction: ChromaDB when installed, a pure-Python JSON store
otherwise. Both support add + metadata-filtered similarity search.

The fallback keeps the exact same interface so retriever.py and rag.py never know
which backend is live — the sort of seam you want when a reviewer may or may not have
a vector DB installed.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .config import CHROMA_DIR, VECTOR_STORE_PATH, _module_available

log = logging.getLogger("vectorstore")


def _cosine(a: list[float], b: list[float]) -> float:
    # vectors are L2-normalised upstream, so a dot product is the cosine similarity
    return sum(x * y for x, y in zip(a, b))


class JsonVectorStore:
    """A transparent, file-backed vector store. Great for inspection and tests."""

    backend = "json_fallback"

    def __init__(self, path: Path = VECTOR_STORE_PATH):
        self.path = Path(path)
        self._docs: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._docs = json.loads(self.path.read_text(encoding="utf-8"))

    def reset(self) -> None:
        self._docs = {}
        if self.path.exists():
            self.path.unlink()

    def add(self, ids, embeddings, documents, metadatas) -> None:
        for i, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._docs[i] = {"embedding": emb, "document": doc, "metadata": meta}
        self.path.write_text(json.dumps(self._docs), encoding="utf-8")

    def query(self, embedding, top_k, where: dict | None = None) -> list[dict]:
        scored = []
        for doc_id, rec in self._docs.items():
            if where and not _matches(rec["metadata"], where):
                continue
            scored.append({
                "id": doc_id,
                "score": _cosine(embedding, rec["embedding"]),
                "document": rec["document"],
                "metadata": rec["metadata"],
            })
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._docs)


class ChromaVectorStore:
    backend = "chroma"

    def __init__(self, collection: str = "daily_narratives"):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        name = self._col.name
        self._client.delete_collection(name)
        self._col = self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        self._col.upsert(ids=list(ids), embeddings=[list(e) for e in embeddings],
                         documents=list(documents), metadatas=list(metadatas))

    def query(self, embedding, top_k, where: dict | None = None) -> list[dict]:
        res = self._col.query(query_embeddings=[list(embedding)], n_results=top_k,
                              where=where or None)
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append({"id": i, "score": 1.0 - dist, "document": doc, "metadata": meta})
        return out

    def count(self) -> int:
        return self._col.count()


def _matches(metadata: dict, where: dict) -> bool:
    """Minimal where-filter supporting equality and {'$in': [...]} for the fallback."""
    for key, cond in where.items():
        val = metadata.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and val not in cond["$in"]:
                return False
            if "$eq" in cond and val != cond["$eq"]:
                return False
        elif val != cond:
            return False
    return True


def get_vector_store(force_fallback: bool = False):
    if not force_fallback and _module_available("chromadb"):
        try:
            return ChromaVectorStore()
        except Exception as exc:  # pragma: no cover
            log.warning("chroma init failed (%s); using json fallback", exc)
    return JsonVectorStore()
