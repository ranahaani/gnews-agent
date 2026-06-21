"""MCP tool handler logic — exercised without booting the FastMCP app.

Tool body functions in ``mcp_server.server`` are kept as plain callables that
read from a module-level ``_memory`` singleton, so they can be tested by
swapping that singleton for a stub.
"""
from __future__ import annotations

import pytest

from mcp_server import server as mcp_server


class StubMemory:
    def __init__(self):
        self.calls: list[tuple] = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return [{"title": "x", "url": "https://example.com", "score": 0.5}]

    def brief(self, topic, **kwargs):
        self.calls.append(("brief", topic, kwargs))
        return {"summary": "ok", "citations": [], "sentiment": "neutral", "article_count": 0}

    def sentiment(self, topic, **kwargs):
        self.calls.append(("sentiment", topic, kwargs))
        return {"overall": "neutral", "score": 0.0, "article_count": 0}

    def timeline(self, topic, **kwargs):
        self.calls.append(("timeline", topic, kwargs))
        return [{"date": "2026-06-16", "count": 1}]


@pytest.fixture(autouse=True)
def stub_memory(monkeypatch):
    stub = StubMemory()
    monkeypatch.setattr(mcp_server, "_memory", stub)
    return stub


def test_search_news(stub_memory):
    result = mcp_server.search_news("GPT-5", days=3, limit=5, semantic=True)
    assert result and result[0]["url"] == "https://example.com"
    _, q, kw = stub_memory.calls[0]
    assert q == "GPT-5"
    assert kw["days"] == 3
    assert kw["limit"] == 5


def test_get_timeline(stub_memory):
    out = mcp_server.get_timeline("OpenAI", days=14)
    assert out[0]["date"] == "2026-06-16"
    _, t, kw = stub_memory.calls[0]
    assert t == "OpenAI"
    assert kw["days"] == 14


def test_get_brief(stub_memory):
    out = mcp_server.get_brief("OpenAI", days=5)
    assert out["summary"] == "ok"


def test_monitor_topic_validates_webhook(stub_memory):
    out = mcp_server.monitor_topic(["OpenAI"], threshold=3, webhook_url="https://hooks.example.com/x")
    assert out["status"] == "registered"
    assert out["webhook_url"] == "https://hooks.example.com/x"


def test_monitor_topic_rejects_private_ip(stub_memory):
    from gnews_agent.exceptions import WebhookSecurityError
    with pytest.raises(WebhookSecurityError):
        mcp_server.monitor_topic(["OpenAI"], threshold=3, webhook_url="https://10.0.0.1/x")


def test_get_memory_raises_when_uninitialised(monkeypatch):
    monkeypatch.setattr(mcp_server, "_memory", None)
    with pytest.raises(RuntimeError):
        mcp_server._get_memory()
