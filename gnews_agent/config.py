"""Runtime configuration for :class:`NewsMemory`."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_state_dir() -> Path:
    return Path(os.path.expanduser("~/.gnews_agent"))


@dataclass(frozen=True)
class NewsMemoryConfig:
    """Pure-data configuration for a :class:`NewsMemory` instance.

    Defaults are tuned for zero-infra local use: SQLite under ``~/.gnews_agent``,
    ChromaDB sibling directory, sentence-transformers in-process embedding model.
    """

    db_path: Path = field(default_factory=lambda: _default_state_dir() / "news.db")
    vector_path: Path = field(default_factory=lambda: _default_state_dir() / "chroma")
    vector_backend: str = "chroma"  # "chroma" | "lance" (opt-in) | "qdrant" (opt-in)
    embed_backend: str = "sentence-transformers"  # | "openai"
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384
    llm_provider: str | None = None      # e.g. "openai", "anthropic" — required only for brief/sentiment
    llm_model: str | None = None         # e.g. "gpt-4o-mini"
    language: str = "en"
    country: str = "US"
    fetch_min_interval_seconds: float = 1.0
    max_fetch_results: int = 50          # keep <=100 so temporal filters stay precise (see GNews 0.8.2)

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_path.mkdir(parents=True, exist_ok=True)
