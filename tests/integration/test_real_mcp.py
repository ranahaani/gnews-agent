"""FastMCP server build smoke — instantiates the app + lists registered tools.

A full stdio round-trip needs a co-process; this smoke confirms the app
constructs, the five tools register, and tool callables are wired through
the `_memory` singleton.
"""
from __future__ import annotations

import pytest

from gnews_agent import NewsMemory, NewsMemoryConfig
from mcp_server import server as mcp_server


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_memory(tmp_path_factory):
    state = tmp_path_factory.mktemp("mcp_state")
    cfg = NewsMemoryConfig(
        db_path=state / "news.db",
        vector_path=state / "chroma",
        max_fetch_results=5,
        fetch_min_interval_seconds=0,
    )
    return NewsMemory(config=cfg)


def test_build_app_registers_tools_and_resources(real_memory, monkeypatch):
    import asyncio

    monkeypatch.setattr(mcp_server, "_memory", real_memory)
    app = mcp_server.build_app()
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    expected = {"search_news", "get_brief", "get_sentiment", "get_timeline", "monitor_topic"}
    assert expected.issubset(names), f"missing tools: {expected - names}"


def test_search_news_tool_callable_against_real_memory(real_memory, monkeypatch):
    monkeypatch.setattr(mcp_server, "_memory", real_memory)
    real_memory.ingest("artificial intelligence")
    hits = mcp_server.search_news("artificial intelligence", days=14, limit=3, semantic=True)
    assert isinstance(hits, list)
    if hits:
        assert "url" in hits[0]


def test_monitor_topic_real_rejects_loopback(real_memory, monkeypatch):
    monkeypatch.setattr(mcp_server, "_memory", real_memory)
    from gnews_agent.exceptions import WebhookSecurityError
    with pytest.raises(WebhookSecurityError):
        mcp_server.monitor_topic(["AI"], threshold=5, webhook_url="http://127.0.0.1:8000/hook")
