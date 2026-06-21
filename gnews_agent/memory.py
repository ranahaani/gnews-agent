"""Public :class:`NewsMemory` facade — Stage 0 stub.

Wire-up of ingestion / storage / AI layers lands in subsequent stages. This
file exists so ``from gnews_agent import NewsMemory`` works on a fresh install
and so the public API surface is locked from day one.
"""
from __future__ import annotations

from typing import Any

from gnews_agent.config import NewsMemoryConfig


class NewsMemory:
    """Persistent semantic memory over GNews articles.

    All real behaviour is implemented in later stages — Stage 0 only locks the
    public surface. Calling any method raises ``NotImplementedError`` so that
    integration tests fail loudly until the method is wired up.
    """

    def __init__(self, config: NewsMemoryConfig | None = None, **overrides: Any) -> None:
        self.config = config or NewsMemoryConfig(**overrides)
        self.config.ensure_dirs()

    def ingest(self, topics: str | list[str], **kwargs: Any) -> dict[str, int]:
        raise NotImplementedError("NewsMemory.ingest lands in Stage 1")

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("NewsMemory.search lands in Stage 2")

    def timeline(self, topic: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("NewsMemory.timeline lands in Stage 2")

    def brief(self, topic: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.brief lands in Stage 3")

    def sentiment(self, topic: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.sentiment lands in Stage 3")

    def monitor(self, topics: list[str], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.monitor lands alongside the MCP server in Stage 5")

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.stats lands in Stage 1")

    def export(self, path: str, **kwargs: Any) -> str:
        raise NotImplementedError("NewsMemory.export is post-v1")
