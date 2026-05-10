"""Tests for src/wikipedia_fetcher.py — Kiwix client, HTML extraction, smart truncation.

Run integration tests (require live Kiwix server) with:
    pytest tests/test_wikipedia_fetcher.py -m integration
"""

import json
import os
import sys

import pytest
from unittest.mock import patch, MagicMock


class TestSmartTruncate:
    """Test the smart_truncate function and its helpers."""

    def _get_func(self):
        from src.wikipedia_fetcher import (
            smart_truncate, _split_sections,
            _accumulate_by_sections, _accumulate_by_paragraphs,
        )
        return smart_truncate, _split_sections, _accumulate_by_sections, _accumulate_by_paragraphs

    def test_section_level_truncation(self):
        st, ss, sa_s, sa_p = self._get_func()
        # Use ==Header== format (no spaces around text) to match wiki regex
        text = "==Intro==\n" + "A " * 200 + "\n\n==Details==\n" + "B " * 500 + "\n\n==Outro==\n" + "C " * 100
        result = st(text, max_words=400, min_words=250)
        assert result is not None
        words = len(result.split())
        assert 250 <= words <= 400

    def test_paragraph_level_fallback(self):
        st, ss, sa_s, sa_p = self._get_func()
        # No section headers — should fall back to paragraph splitting
        text = "\n\n".join(["Word " * (30 + i) for i in range(10)])
        result = st(text, max_words=250, min_words=100)
        assert result is not None
        words = len(result.split())
        assert 100 <= words <= 250

    def test_too_short_returns_none(self):
        st, ss, sa_s, sa_p = self._get_func()
        text = "Short text"
        result = st(text, max_words=600, min_words=250)
        assert result is None

    def test_split_sections(self):
        st, ss, sa_s, sa_p = self._get_func()
        text = "==Header 1==\nBody 1\n\n==Header 2==\nBody 2"
        sections = ss(text)
        assert len(sections) == 2
        assert "Header 1" in sections[0][0]
        assert "Body 1" in sections[0][1]

    def test_split_sections_no_header(self):
        st, ss, sa_s, sa_p = self._get_func()
        text = "Just a plain paragraph with no headers."
        sections = ss(text)
        assert len(sections) == 1
        assert sections[0][0] is None

    def test_accumulate_by_sections(self):
        st, ss, sa_s, sa_p = self._get_func()
        text = "==S1==\n" + "A " * 100 + "\n\n==S2==\n" + "B " * 200
        result = sa_s(text, max_words=200, min_words=50)
        assert result is not None
        assert "==S1==" in result

    def test_accumulate_by_paragraphs(self):
        st, ss, sa_s, sa_p = self._get_func()
        text = "\n\n".join(["Para " + str(i) + " word" * 20 for i in range(5)])
        result = sa_p(text, max_words=150, min_words=50)
        assert result is not None

    def test_empty_text(self):
        st, ss, sa_s, sa_p = self._get_func()
        assert st("", max_words=600, min_words=250) is None


class TestExtractWikiText:
    """Test HTML-to-text extraction."""

    def test_basic_extraction(self):
        from src.wikipedia_fetcher import extract_wiki_text
        html = '<div id="mw-content-text"><div class="mw-parser-output">Hello world</div></div>'
        result = extract_wiki_text(html)
        assert "Hello world" in result

    def test_script_removal(self):
        from src.wikipedia_fetcher import extract_wiki_text
        html = '<div id="mw-content-text"><script>alert("xss")</script>Real text</div>'
        result = extract_wiki_text(html)
        assert "alert" not in result
        assert "Real text" in result

    def test_footer_removal(self):
        from src.wikipedia_fetcher import extract_wiki_text
        html = '<div id="mw-content-text">Article body\n\nThis article is issued from Wikipedia</div>'
        result = extract_wiki_text(html)
        assert "Article body" in result
        assert "issued from Wikipedia" not in result

    def test_blank_line_collapse(self):
        from src.wikipedia_fetcher import extract_wiki_text
        html = '<div id="mw-content-text">Line 1\n\n\n\n\nLine 2</div>'
        result = extract_wiki_text(html)
        assert "\n\n\n" not in result


class TestKiwixClient:
    """Test the KiwixClient class (mocked)."""

    def test_init(self):
        from src.wikipedia_fetcher import KiwixClient
        client = KiwixClient(base_url="http://test", zim_name="test_zim")
        assert client.base_url == "http://test"
        assert client.zim_name == "test_zim"

    def test_search_parses_titles(self):
        from src.wikipedia_fetcher import KiwixClient
        mock_resp = MagicMock()
        mock_resp.text = '<a href="/content/test_zim/Article%20One">A1</a><a href="/content/other_zim/Nope">N</a>'
        mock_resp.raise_for_status = lambda: None

        client = KiwixClient(base_url="http://test", zim_name="test_zim")
        with patch.object(client, "_get", return_value=mock_resp):
            titles = client.search("query", count=5)
        # Titles are URL-decoded by the href parsing
        assert any("Article" in t for t in titles)
        assert "Nope" not in titles

    def test_search_empty_result(self):
        from src.wikipedia_fetcher import KiwixClient
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status = lambda: None

        client = KiwixClient(base_url="http://test", zim_name="test_zim")
        with patch.object(client, "_get", return_value=mock_resp):
            titles = client.search("nope", count=5)
        assert titles == []

    def test_skip_patterns(self):
        from src.wikipedia_fetcher import KiwixClient
        # Verify skip patterns actually catch list pages
        skip_list = ["List of universities", "Glossary of terms", "Index of plants"]
        for title in skip_list:
            assert any(skip in title for skip in KiwixClient.SKIP_PATTERNS), f"Should skip: {title}"

    def test_context_manager(self):
        from src.wikipedia_fetcher import KiwixClient
        with KiwixClient() as client:
            assert client is not None


class TestOrchestratorPipelineIntegration:
    """Integration tests — fetch real articles from Kiwix and run the full
    orchestrator pipeline (fetch → clean → re-enforce max_words).

    Requires a live Kiwix server. Marked with @pytest.mark.integration so they
    are skipped by default in CI.
    """

    def _load_config(self):
        """Load the real project config.json."""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.mark.integration
    def test_fetch_clean_truncate_within_limit(self):
        """Fetch several random articles and verify they are within the word limit
        after the full orchestrator pipeline: fetch → clean → post-clean truncation.
        """
        from src.wikipedia_fetcher import KiwixClient, smart_truncate, hard_truncate
        from src.orchestrator import clean_content

        config = self._load_config()

        # Use the same article_filter as krystof/johi profiles
        max_words = 300
        min_words = 50
        content_lang = "de"  # test with German Wikipedia

        kiwix_cfg = config.get("kiwix_servers", {}).get(content_lang, {})
        base_url = kiwix_cfg.get("base_url", "http://192.168.100.52:8080")
        zim_name = kiwix_cfg.get("zim_name", "wikipedia_de_all_nopic_2026-01")

        client = KiwixClient(base_url=base_url, zim_name=zim_name)

        # Fetch 5 random articles and run through the full pipeline
        num_articles = 5
        for i in range(num_articles):
            title, text = client.get_random_article(
                min_words=min_words,
                max_words=max_words,
            )

            assert title != "Error", f"Attempt {i+1}: failed to fetch article"
            pre_clean_words = len(text.split())

            # Step: clean content (orchestrator step)
            cleaned = clean_content(text)
            post_clean_words = len(cleaned.split())

            # Step: re-enforce max_words after cleaning (orchestrator step)
            if post_clean_words > max_words:
                truncated = smart_truncate(
                    cleaned, max_words=max_words, min_words=min_words
                ) or hard_truncate(cleaned, max_words=max_words)
            else:
                truncated = cleaned

            final_words = len(truncated.split())

            # max_words is the hard guarantee — cleaning can slightly reduce words
            # below min_words (removes references, footers) but that's fine.
            assert final_words <= max_words, (
                f"Article {i+1} ('{title}') has {final_words} words "
                f"(pre-clean: {pre_clean_words}, post-clean: {post_clean_words}), "
                f"expected ≤ {max_words}"
            )

        client.close()

    @pytest.mark.integration
    def test_fetch_clean_truncate_italian(self):
        """Same pipeline test but with Italian Wikipedia (johi profile)."""
        from src.wikipedia_fetcher import KiwixClient, smart_truncate, hard_truncate
        from src.orchestrator import clean_content

        config = self._load_config()

        max_words = 300
        min_words = 50
        content_lang = "it"

        kiwix_cfg = config.get("kiwix_servers", {}).get(content_lang, {})
        base_url = kiwix_cfg.get("base_url", "http://192.168.100.52:8080")
        zim_name = kiwix_cfg.get("zim_name", "wikipedia_it_all_nopic_2026-02")

        client = KiwixClient(base_url=base_url, zim_name=zim_name)

        for i in range(5):
            title, text = client.get_random_article(
                min_words=min_words,
                max_words=max_words,
            )

            assert title != "Error", f"Attempt {i+1}: failed to fetch article"

            cleaned = clean_content(text)
            final = (
                smart_truncate(cleaned, max_words=max_words, min_words=min_words)
                or hard_truncate(cleaned, max_words=max_words)
            )
            final_words = len(final.split())

            assert final_words <= max_words, (
                f"Article {i+1} ('{title}') has {final_words} words, "
                f"expected ≤ {max_words}"
            )

        client.close()
