"""Shared pytest helpers — fake embedder + fake vector store keep tests offline."""
from __future__ import annotations

import hashlib
from typing import Any

from gnews_agent.storage.vector_store import VectorHit


class FakeEmbedder:
    """Deterministic 8-dim hash-based embedder — no model download, no network."""

    def __init__(self, embed_model: str = "fake-embed", embed_dim: int = 8) -> None:
        self.embed_model = embed_model
        self.embed_dim = embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [b / 255.0 for b in digest[: self.embed_dim]]
            # L2-normalise so cosine math behaves.
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


class FakeVectorStore:
    """In-memory list-of-dicts vector store for ingestion-flow tests."""

    def __init__(self, *, embed_model: str = "fake-embed", embed_dim: int = 8) -> None:
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.records: list[dict[str, Any]] = []

    def upsert(self, article_id: int, embedding: list[float], metadata: dict[str, Any]) -> None:
        self.records.append({"article_id": article_id, "embedding": embedding, "metadata": metadata})

    def query(self, embedding, *, k=10, where=None):
        # Stage 2 will exercise this; Stage 1 tests only need upsert/count.
        return [VectorHit(r["article_id"], 1.0, r["metadata"]) for r in self.records[:k]]

    def count(self) -> int:
        return len(self.records)

    def delete(self, article_id: int) -> None:
        self.records = [r for r in self.records if r["article_id"] != article_id]
