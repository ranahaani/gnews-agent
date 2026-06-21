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


def test_news_memory_unwired_methods_raise(tmp_path):
    """Stage 2/3/5 methods still raise; ingest/stats are implemented (Stage 1)."""
    from tests.conftest import FakeEmbedder, FakeVectorStore
    from gnews_agent.storage.sqlite_store import SqliteStore
    from gnews_agent.ingestion.fetcher import Fetcher
    from tests.test_fetcher import FakeGNews

    cfg = NewsMemoryConfig(
        db_path=tmp_path / "news.db",
        vector_path=tmp_path / "chroma",
        embed_model="fake-embed",
        embed_dim=8,
    )
    memory = NewsMemory(
        config=cfg,
        fetcher=Fetcher(gnews_client=FakeGNews(), min_interval_seconds=0),
        embedder=FakeEmbedder(),
        sqlite_store=SqliteStore(cfg.db_path),
        vector_store=FakeVectorStore(),
    )
    with pytest.raises(NotImplementedError):
        memory.brief("OpenAI")
    with pytest.raises(NotImplementedError):
        memory.sentiment("OpenAI")
    with pytest.raises(NotImplementedError):
        memory.monitor(["OpenAI"])
