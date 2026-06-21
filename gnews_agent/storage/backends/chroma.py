"""ChromaDB vector-store backend (default for v1).

Pinned to ``chromadb>=0.5,<0.6`` because Chroma's persistence model has had
breaking changes between minor versions. Embedding dimension is recorded on
every record's metadata; the store refuses cross-model queries via
:class:`EmbeddingDimMismatchError`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gnews_agent.exceptions import EmbeddingDimMismatchError
from gnews_agent.storage.vector_store import VectorHit


class ChromaVectorStore:
    """Persistent ChromaDB collection scoped to a single embedding model."""

    collection_name = "articles"

    def __init__(self, *, persist_path: str, embed_model: str, embed_dim: int) -> None:
        # Import lazily so chromadb stays out of the import graph for users
        # who only need the deduplicator/utility helpers.
        import chromadb

        self._embed_model = embed_model
        self._embed_dim = embed_dim
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embed_model": embed_model, "embed_dim": str(embed_dim)},
        )
        self._check_model_lock()

    def _check_model_lock(self) -> None:
        stored_model = (self._collection.metadata or {}).get("embed_model")
        if stored_model and stored_model != self._embed_model:
            raise EmbeddingDimMismatchError(
                f"vector collection was created with embed_model={stored_model!r} "
                f"but this instance is configured for embed_model={self._embed_model!r}. "
                "Use a separate vector_path or delete the existing collection."
            )

    def upsert(self, article_id: int, embedding: list[float], metadata: dict[str, Any]) -> None:
        if len(embedding) != self._embed_dim:
            raise EmbeddingDimMismatchError(
                f"embedding has dim={len(embedding)} but collection expects dim={self._embed_dim}"
            )
        payload = {
            **metadata,
            "embed_model": self._embed_model,
            "embed_dim": self._embed_dim,
        }
        self._collection.upsert(
            ids=[str(article_id)],
            embeddings=[embedding],
            metadatas=[payload],
            documents=[metadata.get("document", "")],
        )

    def query(
        self,
        embedding: list[float],
        *,
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        if len(embedding) != self._embed_dim:
            raise EmbeddingDimMismatchError(
                f"query embedding dim={len(embedding)} but collection expects {self._embed_dim}"
            )
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where or None,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        hits: list[VectorHit] = []
        for idx, raw_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else 0.0
            # Chroma returns cosine distance; flip to similarity.
            similarity = 1.0 - float(distance)
            hits.append(
                VectorHit(
                    article_id=int(raw_id),
                    score=similarity,
                    metadata=dict(metadatas[idx]) if idx < len(metadatas) else {},
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def delete(self, article_id: int) -> None:
        self._collection.delete(ids=[str(article_id)])
