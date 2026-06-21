"""CLI smoke tests — each subcommand wires through to NewsMemory and emits JSON."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from gnews_agent import cli


class StubMemory:
    """Stand-in NewsMemory that records calls so CLI wiring can be asserted."""

    def __init__(self):
        self.calls: list[tuple] = []

    def stats(self):
        self.calls.append(("stats",))
        return {"total_articles": 0, "vector_count": 0, "embed_model": "fake", "embed_dim": 8}

    def ingest(self, topic, method="get_news"):
        self.calls.append(("ingest", topic, method))
        return {topic: {"fetched": 1, "new": 1, "skipped": 0, "status": "success"}}

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return [{"title": "hit", "url": "https://example.com", "score": 0.9, "search_mode": "semantic"}]

    def brief(self, topic, **kwargs):
        self.calls.append(("brief", topic, kwargs))
        return {"summary": "ok", "citations": [], "sentiment": "neutral", "article_count": 0}

    def sentiment(self, topic, **kwargs):
        self.calls.append(("sentiment", topic, kwargs))
        return {"overall": "neutral", "score": 0.0, "article_count": 0}

    def timeline(self, topic, **kwargs):
        self.calls.append(("timeline", topic, kwargs))
        return [{"date": "2026-06-16", "count": 1}]


@pytest.fixture
def runner(monkeypatch):
    stub = StubMemory()
    monkeypatch.setattr(cli, "_build_memory", lambda *args, **kwargs: stub)
    return CliRunner(), stub


def test_version_flag(runner):
    cli_runner, _ = runner
    result = cli_runner.invoke(cli.main, ["--version"])
    assert result.exit_code == 0
    assert "gnews-agent" in result.output


def test_stats(runner):
    cli_runner, stub = runner
    result = cli_runner.invoke(cli.main, ["stats"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["embed_model"] == "fake"
    assert stub.calls == [("stats",)]


def test_ingest_dispatches_topic_and_method(runner):
    cli_runner, stub = runner
    result = cli_runner.invoke(cli.main, ["ingest", "OpenAI", "--method", "get_news"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["OpenAI"]["new"] == 1
    assert stub.calls == [("ingest", "OpenAI", "get_news")]


def test_search_forwards_flags(runner):
    cli_runner, stub = runner
    result = cli_runner.invoke(cli.main, ["search", "GPT-5", "--days", "7", "--limit", "3", "--keyword"])
    assert result.exit_code == 0, result.output
    hits = json.loads(result.output)
    assert hits and hits[0]["title"] == "hit"
    _, query, kwargs = stub.calls[0]
    assert query == "GPT-5"
    assert kwargs["days"] == 7
    assert kwargs["limit"] == 3
    assert kwargs["semantic"] is False


def test_brief(runner):
    cli_runner, stub = runner
    result = cli_runner.invoke(cli.main, ["brief", "OpenAI", "--days", "3", "--max-articles", "5"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"] == "ok"
    _, topic, kwargs = stub.calls[0]
    assert topic == "OpenAI"
    assert kwargs["days"] == 3
    assert kwargs["max_articles"] == 5
    assert kwargs["include_citations"] is True


def test_timeline(runner):
    cli_runner, stub = runner
    result = cli_runner.invoke(cli.main, ["timeline", "OpenAI", "--days", "30"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["date"] == "2026-06-16"
