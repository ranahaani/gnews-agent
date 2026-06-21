"""Embedding backends.

Default: ``sentence-transformers`` in-process (``all-MiniLM-L6-v2``, 384-dim,
~80MB model, no sidecar). The model is loaded lazily on first ``embed`` call
so importing the package stays cheap — important for the CLI ``--help`` path
and for cold-starting the MCP server.

Optional: OpenAI ``text-embedding-3-small`` (1536-dim) via
``gnews-agent[openai]`` — gated behind the import.
"""
from __future__ import annotations

import os
from typing import Protocol


class Embedder(Protocol):
    embed_model: str
    embed_dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Default in-process embedder. Model load is deferred to first use."""

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2", embed_dim: int = 384) -> None:
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self._model = None  # lazy

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embed_model)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [vec.tolist() for vec in vectors]


class OpenAIEmbedder:
    """OpenAI embedding backend — opt-in via ``gnews-agent[openai]``."""

    def __init__(
        self,
        embed_model: str = "text-embedding-3-small",
        embed_dim: int = 1536,
        api_key: str | None = None,
    ) -> None:
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _load(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAI embedder requires `pip install gnews-agent[openai]`"
                ) from e
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._load()
        response = client.embeddings.create(model=self.embed_model, input=texts)
        return [item.embedding for item in response.data]


def make_embedder(*, backend: str, embed_model: str, embed_dim: int) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(embed_model=embed_model, embed_dim=embed_dim)
    if backend == "openai":
        return OpenAIEmbedder(embed_model=embed_model, embed_dim=embed_dim)
    raise ValueError(f"unknown embed backend: {backend!r}")
