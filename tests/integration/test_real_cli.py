"""CLI end-to-end via `python -m gnews_agent` — the shape a user actually sees."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


pytestmark = pytest.mark.integration


def _run(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gnews_agent.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_version_prints_semver():
    result = _run(["--version"])
    assert result.returncode == 0, result.stderr
    assert "gnews-agent" in result.stdout


def test_cli_stats_emits_json(tmp_path):
    result = _run([
        "--db-path", str(tmp_path / "news.db"),
        "--vector-path", str(tmp_path / "chroma"),
        "stats", "--no-pretty",
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["total_articles"] == 0


def test_cli_ingest_then_search(tmp_path):
    common = [
        "--db-path", str(tmp_path / "news.db"),
        "--vector-path", str(tmp_path / "chroma"),
    ]
    ingest = _run(common + ["ingest", "artificial intelligence", "--no-pretty"])
    assert ingest.returncode == 0, ingest.stderr
    ingest_payload = json.loads(ingest.stdout)
    assert ingest_payload["artificial intelligence"]["status"] == "success"

    search = _run(common + ["search", "artificial intelligence", "--limit", "3", "--no-pretty"])
    assert search.returncode == 0, search.stderr
    hits = json.loads(search.stdout)
    assert isinstance(hits, list)
    assert len(hits) >= 1
