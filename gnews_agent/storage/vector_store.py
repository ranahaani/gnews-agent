"""Backend-agnostic vector store contract.

v1 ships a single concrete backend (``backends/chroma.py``); LanceDB and
Qdrant slot in behind the same protocol later via optional extras.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class VectorHit:
    """Single semantic search result — ids hydrate against SQLite separately."""

    article_id: int
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """Minimal surface required by ingestion + query layers."""

    def upsert(self, article_id: int, embedding: list[float], metadata: dict[str, Any]) -> None:
        ...

    def query(
        self,
        embedding: list[float],
        *,
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        ...

    def count(self) -> int:
        ...

    def delete(self, article_id: int) -> None:
        ...


def make_vector_store(
    backend: str,
    *,
    persist_path: str,
    embed_model: str,
    embed_dim: int,
) -> VectorStore:
    """Construct the configured backend lazily so optional extras stay opt-in."""
    if backend == "chroma":
        from gnews_agent.storage.backends.chroma import ChromaVectorStore

        return ChromaVectorStore(
            persist_path=persist_path,
            embed_model=embed_model,
            embed_dim=embed_dim,
        )
    if backend == "lance":
        raise NotImplementedError("LanceDB backend lands post-v1 — install `gnews-agent[lance]`")
    if backend == "qdrant":
        raise NotImplementedError("Qdrant backend lands post-v1 — install `gnews-agent[qdrant]`")
    raise ValueError(f"unknown vector backend: {backend!r}")
