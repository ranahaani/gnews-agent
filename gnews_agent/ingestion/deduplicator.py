"""Article deduplication helpers.

Composite key = ``sha256(title_slug + "|" + publisher_norm)``. URL is
intentionally excluded — Google News surfaces the same article under multiple
URL variations (locale params, tracking suffixes, redirector vs resolved).
Including URL in the key over-stored badly.

Reuters + BBC covering the same event = 2 distinct publishers → 2 distinct
composite keys → both rows kept. That is the intended behaviour: journalism
signal benefits from preserving multi-outlet framings. Cosine similarity is
*not* applied at ingestion time; it is reserved for query-time re-ranking.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit


_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_DASH_RE = re.compile(r"[^a-z0-9\-]+")
_MULTI_DASH_RE = re.compile(r"-{2,}")


def title_slug(title: str) -> str:
    """Lowercase, NFKD-normalise, collapse whitespace, replace spaces with dashes.

    Strips punctuation that would otherwise produce drifting slugs across
    publishers ("Trump's plan" vs "Trump’s plan" vs "Trump's plan").
    Returns empty string for empty/None input rather than raising — the caller
    decides whether an empty slug is acceptable.
    """
    if not title:
        return ""
    normalised = unicodedata.normalize("NFKD", title)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower().strip()
    spaced = _WHITESPACE_RE.sub(" ", lowered)
    dashed = spaced.replace(" ", "-")
    cleaned = _NON_WORD_DASH_RE.sub("", dashed)
    collapsed = _MULTI_DASH_RE.sub("-", cleaned).strip("-")
    return collapsed


def publisher_norm(publisher: str | None) -> str:
    """Lowercase + strip + collapse whitespace. Empty input → ``"unknown"``."""
    if not publisher:
        return "unknown"
    normalised = unicodedata.normalize("NFKD", publisher)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    return _WHITESPACE_RE.sub(" ", ascii_only.lower().strip()) or "unknown"


def composite_key(title: str, publisher: str | None) -> str:
    """Stable dedup key for the (title, publisher) pair."""
    payload = f"{title_slug(title)}|{publisher_norm(publisher)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def url_hash(url: str) -> str:
    """Hash the canonicalised URL for the ``articles.url_hash`` backstop.

    Canonicalisation: lowercase host, drop fragment, strip ``utm_*`` query
    params, strip trailing slash from path. Intentionally light — the URL
    UNIQUE constraint exists as a fallback only; the real dedup signal is the
    composite key above.
    """
    if not url:
        return ""
    parts = urlsplit(url.strip())
    host = parts.hostname.lower() if parts.hostname else ""
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    path = parts.path.rstrip("/") or "/"
    query = "&".join(
        kv for kv in parts.query.split("&")
        if kv and not kv.lower().startswith("utm_")
    )
    canonical = urlunsplit((parts.scheme.lower(), netloc, path, query, ""))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
