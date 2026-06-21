"""Real brief() + sentiment() against whichever LLM provider key is set."""
from __future__ import annotations

import pytest

from gnews_agent import NewsMemory, NewsMemoryConfig


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def memory(tmp_path_factory, llm_provider_choice):
    provider, _env, model = llm_provider_choice
    state = tmp_path_factory.mktemp("brief_state")
    cfg = NewsMemoryConfig(
        db_path=state / "news.db",
        vector_path=state / "chroma",
        max_fetch_results=8,
        fetch_min_interval_seconds=0,
        llm_provider=provider,
        llm_model=model,
    )
    mem = NewsMemory(config=cfg)
    mem.ingest("artificial intelligence")
    return mem


def test_real_brief_returns_cited_summary(memory):
    brief = memory.brief("artificial intelligence", days=14, max_articles=5)
    assert "summary" in brief and isinstance(brief["summary"], str)
    assert len(brief["summary"]) > 50, brief
    assert brief["article_count"] >= 1
    assert "sentiment" in brief
    assert brief["sentiment"] in {"positive", "negative", "neutral", "mixed"}
    # At least one citation should be present and look like a URL.
    if brief.get("citations"):
        assert any(c.startswith("http") for c in brief["citations"])


def test_real_sentiment_returns_score(memory):
    s = memory.sentiment("artificial intelligence", days=14)
    assert s["overall"] in {"positive", "negative", "neutral", "mixed"}
    assert -1.0 <= s["score"] <= 1.0
    assert s["article_count"] >= 1
