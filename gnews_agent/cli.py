"""``gnews-agent`` CLI.

Thin click facade over :class:`NewsMemory`. Each subcommand:

* constructs a ``NewsMemory`` from CLI args + env defaults,
* calls one method,
* prints the result as JSON to stdout.

JSON-by-default keeps the CLI scriptable from shells / agents. ``--pretty``
re-renders the same payload with indentation for human reading.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

from gnews_agent import NewsMemory, NewsMemoryConfig, __version__


def _build_memory(
    db_path: Path | None,
    vector_path: Path | None,
    country: str,
    language: str,
    llm_provider: str | None,
    llm_model: str | None,
) -> NewsMemory:
    cfg = NewsMemoryConfig(
        **{k: v for k, v in {
            "db_path": db_path,
            "vector_path": vector_path,
            "country": country,
            "language": language,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        }.items() if v is not None}
    )
    return NewsMemory(config=cfg)


def _emit(payload: Any, pretty: bool) -> None:
    if pretty:
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        click.echo(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# top-level group
# ---------------------------------------------------------------------------

@click.group(help="gnews-agent — persistent news intelligence for AI agents.")
@click.version_option(__version__, prog_name="gnews-agent")
@click.option("--db-path", type=click.Path(path_type=Path), default=None,
              help="SQLite path (default: ~/.gnews_agent/news.db).")
@click.option("--vector-path", type=click.Path(path_type=Path), default=None,
              help="Vector store dir (default: ~/.gnews_agent/chroma).")
@click.option("--country", default="US", show_default=True)
@click.option("--language", default="en", show_default=True)
@click.option("--llm-provider", default=lambda: os.environ.get("GNEWS_AGENT_LLM_PROVIDER"))
@click.option("--llm-model",    default=lambda: os.environ.get("GNEWS_AGENT_LLM_MODEL"))
@click.option("--verbose", "-v", is_flag=True, help="Enable INFO logging.")
@click.pass_context
def main(
    ctx: click.Context,
    db_path: Path | None,
    vector_path: Path | None,
    country: str,
    language: str,
    llm_provider: str | None,
    llm_model: str | None,
    verbose: bool,
) -> None:
    """Top-level CLI group. Subcommands lazy-build a NewsMemory from these flags."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    ctx.ensure_object(dict)
    ctx.obj["memory_factory"] = lambda: _build_memory(
        db_path, vector_path, country, language, llm_provider, llm_model,
    )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

@main.command(help="Show counts and embedding info for the local store.")
@click.option("--pretty/--no-pretty", default=True)
@click.pass_context
def stats(ctx: click.Context, pretty: bool) -> None:
    memory = ctx.obj["memory_factory"]()
    _emit(memory.stats(), pretty)


@main.command(help="Fetch + persist articles for TOPIC.")
@click.argument("topic")
@click.option("--method", default="get_news",
              type=click.Choice([
                  "get_news", "get_top_news", "get_news_by_topic",
                  "get_news_by_location", "get_news_by_site",
              ]))
@click.option("--pretty/--no-pretty", default=True)
@click.pass_context
def ingest(ctx: click.Context, topic: str, method: str, pretty: bool) -> None:
    memory = ctx.obj["memory_factory"]()
    _emit(memory.ingest(topic, method=method), pretty)


@main.command(help="Semantic search over stored articles.")
@click.argument("query")
@click.option("--days", type=int, default=None)
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--country", default=None)
@click.option("--language", default=None)
@click.option("--semantic/--keyword", default=True)
@click.option("--pretty/--no-pretty", default=True)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    days: int | None,
    limit: int,
    country: str | None,
    language: str | None,
    semantic: bool,
    pretty: bool,
) -> None:
    memory = ctx.obj["memory_factory"]()
    hits = memory.search(
        query,
        days=days,
        limit=limit,
        country=country,
        language=language,
        semantic=semantic,
    )
    _emit(hits, pretty)


@main.command(help="LLM-cited brief on TOPIC (requires LLM key).")
@click.argument("topic")
@click.option("--days", type=int, default=7, show_default=True)
@click.option("--max-articles", type=int, default=20, show_default=True)
@click.option("--no-citations", is_flag=True)
@click.option("--pretty/--no-pretty", default=True)
@click.pass_context
def brief(
    ctx: click.Context,
    topic: str,
    days: int,
    max_articles: int,
    no_citations: bool,
    pretty: bool,
) -> None:
    memory = ctx.obj["memory_factory"]()
    payload = memory.brief(
        topic,
        days=days,
        max_articles=max_articles,
        include_citations=not no_citations,
    )
    _emit(payload, pretty)


@main.command(help="Sentiment for TOPIC over the last --days days.")
@click.argument("topic")
@click.option("--days", type=int, default=14, show_default=True)
@click.option("--timeline", is_flag=True)
@click.option("--pretty/--no-pretty", default=True)
@click.pass_context
def sentiment(ctx: click.Context, topic: str, days: int, timeline: bool, pretty: bool) -> None:
    memory = ctx.obj["memory_factory"]()
    _emit(memory.sentiment(topic, days=days, timeline=timeline), pretty)


@main.command(help="Day-by-day article counts for TOPIC.")
@click.argument("topic")
@click.option("--days", type=int, default=30, show_default=True)
@click.option("--pretty/--no-pretty", default=True)
@click.pass_context
def timeline(ctx: click.Context, topic: str, days: int, pretty: bool) -> None:
    memory = ctx.obj["memory_factory"]()
    _emit(memory.timeline(topic, days=days), pretty)


@main.command(help="Run the FastMCP server (stdio or HTTP).")
@click.option("--transport", type=click.Choice(["stdio", "http"]), default="stdio", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.pass_context
def serve(ctx: click.Context, transport: str, port: int) -> None:
    # Lazy import so ``gnews-agent --help`` doesn't pay the FastMCP cost.
    from mcp_server.server import run

    factory = ctx.obj["memory_factory"]
    run(memory_factory=factory, transport=transport, port=port)


if __name__ == "__main__":
    main(obj={})
