"""brief() + sentiment() — LLM mocked, no network."""
from __future__ import annotations

import datetime as _dt

import pytest

from gnews_agent import NewsMemory, NewsMemoryConfig
from gnews_agent.ai.client import LLMClient, LLMConfig
from gnews_agent.exceptions import LLMKeyMissingError
from gnews_agent.ingestion.fetcher import Fetcher
from gnews_agent.storage.sqlite_store import SqliteStore
from gnews_agent.storage.vector_store import VectorHit

from tests.conftest import FakeEmbedder, FakeVectorStore
from tests.test_fetcher import FakeGNews


class StubLLM(LLMClient):
    """LLMClient that returns scripted JSON payloads instead of calling LiteLLM."""

    def __init__(self, payloads: list[dict]) -> None:
        super().__init__(LLMConfig(provider="ollama", model="stub"))
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def complete_json(self, prompt: str):
        self.prompts.append(prompt)
        return self.payloads.pop(0) if self.payloads else {}


class ScriptedVectorStore(FakeVectorStore):
    def __init__(self):
        super().__init__()
        self.scripted: list[VectorHit] = []

    def query(self, embedding, *, k=10, where=None):
        return list(self.scripted[:k])


def _make_memory(tmp_path, *, llm):
    cfg = NewsMemoryConfig(
        db_path=tmp_path / "news.db",
        vector_path=tmp_path / "chroma",
        embed_model="fake-embed",
        embed_dim=8,
        llm_provider="ollama",
        llm_model="stub",
    )
    # Date the fixture article as "today" so days-filtered search always
    # returns it — otherwise the test rots whenever the clock advances past
    # the hard-coded date.
    today_rfc822 = _dt.datetime.now(_dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    items = [{
        "title": "OpenAI ships GPT-5",
        "url": "https://reuters.com/openai-gpt5",
        "description": "OpenAI today announced GPT-5.",
        "published date": today_rfc822,
        "publisher": {"title": "Reuters", "href": "https://reuters.com"},
    }]
    vectors = ScriptedVectorStore()
    mem = NewsMemory(
        config=cfg,
        fetcher=Fetcher(gnews_client=FakeGNews(items=items), min_interval_seconds=0),
        embedder=FakeEmbedder(),
        sqlite_store=SqliteStore(cfg.db_path),
        vector_store=vectors,
        llm_client=llm,
    )
    mem.ingest("OpenAI")
    article_ids = sorted(int(r["article_id"]) for r in vectors.records)
    vectors.scripted = [VectorHit(article_id=article_ids[0], score=0.9, metadata={})]
    return mem


def test_brief_returns_structured_response(tmp_path):
    llm = StubLLM([{
        "summary": "OpenAI announced GPT-5 today [reuters](https://reuters.com/openai-gpt5).",
        "citations": ["https://reuters.com/openai-gpt5"],
        "sentiment": "neutral",
    }])
    mem = _make_memory(tmp_path, llm=llm)
    brief = mem.brief("OpenAI", days=7, max_articles=5)
    assert "GPT-5" in brief["summary"]
    assert brief["citations"] == ["https://reuters.com/openai-gpt5"]
    assert brief["sentiment"] == "neutral"
    assert brief["article_count"] == 1


def test_brief_empty_corpus_skips_llm(tmp_path):
    llm = StubLLM(payloads=[])  # empty — would IndexError if called
    cfg = NewsMemoryConfig(
        db_path=tmp_path / "news.db",
        vector_path=tmp_path / "chroma",
        embed_model="fake-embed",
        embed_dim=8,
        llm_provider="ollama",
        llm_model="stub",
    )
    mem = NewsMemory(
        config=cfg,
        fetcher=Fetcher(gnews_client=FakeGNews(items=[]), min_interval_seconds=0),
        embedder=FakeEmbedder(),
        sqlite_store=SqliteStore(cfg.db_path),
        vector_store=ScriptedVectorStore(),
        llm_client=llm,
    )
    brief = mem.brief("Nothing", days=7)
    assert brief["article_count"] == 0
    assert "No articles" in brief["summary"]


def test_sentiment_returns_score(tmp_path):
    llm = StubLLM([{
        "overall": "positive",
        "score": 0.6,
        "rationale": "Markets reacted favourably.",
    }])
    mem = _make_memory(tmp_path, llm=llm)
    s = mem.sentiment("OpenAI", days=14)
    assert s["overall"] == "positive"
    assert s["score"] == 0.6
    assert s["article_count"] == 1


def test_brief_without_llm_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = NewsMemoryConfig(
        db_path=tmp_path / "news.db",
        vector_path=tmp_path / "chroma",
        embed_model="fake-embed",
        embed_dim=8,
        llm_provider="openai",   # provider needs OPENAI_API_KEY
        llm_model="gpt-4o-mini",
    )
    today_rfc822 = _dt.datetime.now(_dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    items = [{
        "title": "OpenAI ships GPT-5",
        "url": "https://reuters.com/openai-gpt5",
        "description": "OpenAI today announced GPT-5.",
        "published date": today_rfc822,
        "publisher": {"title": "Reuters", "href": "https://reuters.com"},
    }]
    vectors = ScriptedVectorStore()
    mem = NewsMemory(
        config=cfg,
        fetcher=Fetcher(gnews_client=FakeGNews(items=items), min_interval_seconds=0),
        embedder=FakeEmbedder(),
        sqlite_store=SqliteStore(cfg.db_path),
        vector_store=vectors,
    )
    mem.ingest("OpenAI")
    article_id = int(vectors.records[0]["article_id"])
    vectors.scripted = [VectorHit(article_id=article_id, score=0.9, metadata={})]
    with pytest.raises(LLMKeyMissingError):
        mem.brief("OpenAI", days=7, max_articles=3)
