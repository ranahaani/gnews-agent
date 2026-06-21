"""Click-powered CLI — Stage 0 stub.

Full subcommand wiring lands in Stage 4. Stage 0 provides the entrypoint so
``pip install -e .`` registers the ``gnews-agent`` console script immediately.
"""
from __future__ import annotations

import click

from gnews_agent import __version__


@click.group(help="gnews-agent — persistent news intelligence for AI agents.")
@click.version_option(__version__, prog_name="gnews-agent")
def main() -> None:
    """Top-level CLI group. Subcommands are wired up in later stages."""


@main.command()
def stats() -> None:
    """Print stored-article statistics (Stage 1)."""
    raise click.UsageError("stats lands in Stage 1")


@main.command()
@click.argument("topic")
def ingest(topic: str) -> None:
    """Fetch + store articles for TOPIC (Stage 1)."""
    raise click.UsageError("ingest lands in Stage 1")


@main.command()
@click.argument("query")
def search(query: str) -> None:
    """Semantic search over stored articles (Stage 2)."""
    raise click.UsageError("search lands in Stage 2")


@main.command()
@click.argument("topic")
def brief(topic: str) -> None:
    """LLM-cited brief on TOPIC (Stage 3, requires LLM key)."""
    raise click.UsageError("brief lands in Stage 3")


@main.command()
@click.option("--transport", type=click.Choice(["stdio", "http"]), default="stdio")
@click.option("--port", type=int, default=8000)
def serve(transport: str, port: int) -> None:
    """Run the FastMCP server (Stage 5)."""
    raise click.UsageError("serve lands in Stage 5")


if __name__ == "__main__":
    main()
