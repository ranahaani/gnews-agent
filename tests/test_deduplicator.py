"""Dedup key correctness — composite key drops URL by design (see system design §4)."""
from __future__ import annotations

from gnews_agent.ingestion.deduplicator import (
    composite_key,
    publisher_norm,
    title_slug,
    url_hash,
)


class TestTitleSlug:
    def test_basic(self):
        assert title_slug("OpenAI ships GPT-5") == "openai-ships-gpt-5"

    def test_case_and_whitespace(self):
        assert title_slug("  OpenAI  Ships   GPT-5  ") == "openai-ships-gpt-5"

    def test_unicode_smart_quotes(self):
        # Smart quotes are stripped to ASCII — same slug as straight quote.
        a = title_slug("Trump’s plan announced")
        b = title_slug("Trump's plan announced")
        assert a == b == "trumps-plan-announced"

    def test_punctuation_dropped(self):
        assert title_slug("OpenAI: ships GPT-5!") == "openai-ships-gpt-5"

    def test_empty_input(self):
        assert title_slug("") == ""
        assert title_slug(None) == ""  # type: ignore[arg-type]


class TestPublisherNorm:
    def test_basic(self):
        assert publisher_norm("Reuters") == "reuters"
        assert publisher_norm("  BBC News  ") == "bbc news"

    def test_empty_means_unknown(self):
        assert publisher_norm(None) == "unknown"
        assert publisher_norm("") == "unknown"
        assert publisher_norm("   ") == "unknown"


class TestCompositeKey:
    def test_same_publisher_same_title_dedups(self):
        k1 = composite_key("OpenAI ships GPT-5", "Reuters")
        k2 = composite_key("openai ships gpt-5", "REUTERS")
        assert k1 == k2

    def test_different_publishers_same_title_kept_separate(self):
        """Critical: Reuters + BBC covering the same event must produce
        different composite keys so both rows are stored."""
        reuters = composite_key("OpenAI ships GPT-5", "Reuters")
        bbc = composite_key("OpenAI ships GPT-5", "BBC News")
        assert reuters != bbc

    def test_different_titles_same_publisher_kept_separate(self):
        k1 = composite_key("OpenAI ships GPT-5", "Reuters")
        k2 = composite_key("OpenAI delays GPT-5", "Reuters")
        assert k1 != k2

    def test_url_not_in_key(self):
        """Same (title, publisher) must produce same key regardless of URL.

        Google News surfaces the same article under multiple URL variants;
        the dedup key intentionally excludes URL to catch those collisions.
        """
        # This is implicit in the API — composite_key takes no url. The test
        # documents the design choice for future readers.
        assert "url" not in composite_key.__code__.co_varnames


class TestUrlHash:
    def test_canonical_form_drops_utm(self):
        a = url_hash("https://example.com/story?utm_source=twitter&id=1")
        b = url_hash("https://example.com/story?id=1")
        assert a == b

    def test_trailing_slash_normalised(self):
        a = url_hash("https://example.com/story/")
        b = url_hash("https://example.com/story")
        assert a == b

    def test_fragment_dropped(self):
        a = url_hash("https://example.com/story#top")
        b = url_hash("https://example.com/story")
        assert a == b

    def test_host_lowercased(self):
        a = url_hash("https://Example.COM/story")
        b = url_hash("https://example.com/story")
        assert a == b
