"""Stage 0 smoke tests — verifies the package imports and the public surface is locked."""
from __future__ import annotations

import pytest

import gnews_agent
from gnews_agent import (
    EmbeddingDimMismatchError,
    GNewsAgentError,
    LLMKeyMissingError,
    NewsMemory,
    NewsMemoryConfig,
    __version__,
)


def test_version_is_string():
    assert isinstance(__version__, str) and __version__.count(".") == 2


def test_public_exports_present():
    expected = {
        "NewsMemory",
        "NewsMemoryConfig",
        "GNewsAgentError",
        "LLMKeyMissingError",
        "EmbeddingDimMismatchError",
        "__version__",
    }
    assert expected.issubset(set(gnews_agent.__all__))


def test_exception_hierarchy():
    assert issubclass(LLMKeyMissingError, GNewsAgentError)
    assert issubclass(EmbeddingDimMismatchError, GNewsAgentError)


def test_config_defaults_resolve(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = NewsMemoryConfig(
        db_path=tmp_path / "news.db",
        vector_path=tmp_path / "chroma",
    )
    assert cfg.embed_dim == 384
    assert cfg.embed_model == "all-MiniLM-L6-v2"
    assert cfg.fetch_min_interval_seconds == 1.0
    assert cfg.max_fetch_results <= 100  # keep temporal filters precise (see GNews 0.8.2)


def test_news_memory_raises_for_unwired_methods(tmp_path):
    cfg = NewsMemoryConfig(db_path=tmp_path / "news.db", vector_path=tmp_path / "chroma")
    memory = NewsMemory(config=cfg)
    for method in ("ingest", "search", "timeline", "brief", "sentiment", "stats"):
        with pytest.raises(NotImplementedError):
            getattr(memory, method)("OpenAI") if method != "stats" else getattr(memory, method)()
