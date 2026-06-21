"""Chroma where-clause builder — must wrap multi-field filters in $and."""
from __future__ import annotations

from gnews_agent.memory import _build_where_clause


def test_no_filters_returns_none():
    assert _build_where_clause(country=None, language=None) is None


def test_single_field_returns_flat_dict():
    assert _build_where_clause(country="US", language=None) == {"country": "US"}
    assert _build_where_clause(country=None, language="en") == {"language": "en"}


def test_two_fields_wrap_in_and():
    """Regression: Chroma rejects {'country': 'US', 'language': 'en'} flat —
    requires $and. Discovered against real Chroma in integration tests."""
    where = _build_where_clause(country="US", language="en")
    assert where == {"$and": [{"country": "US"}, {"language": "en"}]}
