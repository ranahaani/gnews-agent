"""FastMCP server exposing :class:`NewsMemory` over stdio + HTTP.

Tools are registered against a module-level FastMCP app; tool bodies call
through a process-singleton ``NewsMemory`` injected at startup by
``run(memory_factory=..., transport=..., port=...)``. ``get_entities`` is
intentionally absent — deferred to v2 per the locked PRD decision.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from gnews_agent import NewsMemory
from gnews_agent.exceptions import WebhookSecurityError
from mcp_server.security import validate_webhook


logger = logging.getLogger(__name__)

_memory: NewsMemory | None = None


def _get_memory() -> NewsMemory:
    if _memory is None:
        raise RuntimeError("MCP server not initialised — call run() with a memory_factory")
    return _memory


# ---------------------------------------------------------------------------
# tool implementations (kept as plain functions so they are unit-testable
# without a running MCP server)
# ---------------------------------------------------------------------------

def search_news(
    query: str,
    days: int = 7,
    country: str = "US",
    language: str = "en",
    semantic: bool = True,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the top ``limit`` articles for ``query`` from the local store."""
    return _get_memory().search(
        query, days=days, country=country, language=language,
        semantic=semantic, limit=limit,
    )


def get_brief(
    topic: str,
    days: int = 7,
    max_articles: int = 20,
    include_citations: bool = True,
) -> dict[str, Any]:
    """Cited LLM brief on ``topic`` (requires LLM key)."""
    return _get_memory().brief(
        topic, days=days, max_articles=max_articles, include_citations=include_citations,
    )


def get_sentiment(topic: str, days: int = 14, timeline: bool = False) -> dict[str, Any]:
    """Sentiment scoring for ``topic`` (requires LLM key)."""
    return _get_memory().sentiment(topic, days=days, timeline=timeline)


def get_timeline(
    topic: str,
    start_date: str | None = None,
    end_date: str | None = None,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """Day-by-day article counts for ``topic`` (keyless)."""
    return _get_memory().timeline(topic, start=start_date, end=end_date, days=days)


def monitor_topic(
    topics: list[str],
    threshold: int = 5,
    webhook_url: str | None = None,
) -> dict[str, Any]:
    """Record a monitor target. v1 is stateless — webhook URL is validated
    and the request acknowledged; the polling loop lands in a later iteration.
    """
    validated_url = validate_webhook(webhook_url, allow_http=False) if webhook_url else None
    return {
        "topics": topics,
        "threshold": threshold,
        "webhook_url": validated_url,
        "status": "registered",
        "note": "v1 acknowledges the registration only; polling loop ships with the scheduler",
    }


def list_latest(topic: str, limit: int = 10) -> list[dict[str, Any]]:
    """Resource backing ``news://latest/{topic}``."""
    return _get_memory().search(topic, semantic=False, limit=limit)


# ---------------------------------------------------------------------------
# FastMCP app construction
# ---------------------------------------------------------------------------

def build_app():
    """Build the FastMCP app, register tools + resources, return it."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MCP server requires fastmcp — `pip install fastmcp>=2.0`") from exc

    mcp = FastMCP("gnews-agent")

    @mcp.tool()
    def search_news_tool(  # noqa: D401
        query: str,
        days: int = 7,
        country: str = "US",
        language: str = "en",
        semantic: bool = True,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search over locally-stored articles."""
        return search_news(query, days, country, language, semantic, limit)

    @mcp.tool()
    def get_brief_tool(
        topic: str,
        days: int = 7,
        max_articles: int = 20,
        include_citations: bool = True,
    ) -> dict[str, Any]:
        """Cited LLM brief on a topic. Requires LLM key."""
        return get_brief(topic, days, max_articles, include_citations)

    @mcp.tool()
    def get_sentiment_tool(topic: str, days: int = 14, timeline: bool = False) -> dict[str, Any]:
        """Sentiment scoring for a topic. Requires LLM key."""
        return get_sentiment(topic, days, timeline)

    @mcp.tool()
    def get_timeline_tool(
        topic: str,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Day-by-day article counts (keyless)."""
        return get_timeline(topic, start_date, end_date, days)

    @mcp.tool()
    def monitor_topic_tool(
        topics: list[str],
        threshold: int = 5,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        """Register a topic monitor. Webhook URL is SSRF-validated."""
        try:
            return monitor_topic(topics, threshold, webhook_url)
        except WebhookSecurityError as exc:
            return {"status": "rejected", "reason": str(exc)}

    @mcp.resource("news://latest/{topic}")
    def latest_resource(topic: str) -> list[dict[str, Any]]:
        return list_latest(topic, limit=10)

    @mcp.resource("news://timeline/{topic}")
    def timeline_resource(topic: str) -> list[dict[str, Any]]:
        return get_timeline(topic, days=7)

    @mcp.resource("news://sentiment/{topic}")
    def sentiment_resource(topic: str) -> dict[str, Any]:
        return get_sentiment(topic, days=7)

    return mcp


def run(
    *,
    memory_factory: Callable[[], NewsMemory],
    transport: str = "stdio",
    port: int = 8000,
) -> None:
    """Boot the MCP server. Called from ``gnews-agent serve``."""
    global _memory
    _memory = memory_factory()
    app = build_app()
    if transport == "stdio":
        app.run()  # FastMCP default = stdio
    elif transport == "http":
        # FastMCP 2.x uses .run(transport="streamable-http", port=...); 3.x adds .run(transport="http").
        try:
            app.run(transport="http", port=port)
        except TypeError:
            app.run(transport="streamable-http", port=port)
    else:  # pragma: no cover
        raise ValueError(f"unknown transport: {transport!r}")
