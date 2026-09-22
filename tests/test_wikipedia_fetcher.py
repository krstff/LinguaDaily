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
        # Intro: ~280 words (fits within min=250, max=400)
        # Details: ~350 words (would exceed max when added to Intro)
        # Outro: ~100 words
        text = "==Intro==\n" + "A " * 280 + "\n\n==Details==\n" + "B " * 350 + "\n\n==Outro==\n" + "C " * 100
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

    def test_random_title_via_redirect(self):
        """_fetch_random_title parses the /random 302 redirect Location."""
        from src.wikipedia_fetcher import KiwixClient
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "/content/test_zim/Some%20Article"}

        client = KiwixClient(base_url="http://test", zim_name="test_zim")
        with patch.object(client, "_get", return_value=mock_resp):
            title, meta = client._fetch_random_title()
        assert title == "Some Article"
        assert meta == {}

    def test_random_title_404_raises(self):
        from src.wikipedia_fetcher import KiwixClient, WikiFetchError
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        client = KiwixClient(base_url="http://test", zim_name="test_zim")
        with patch.object(client, "_get", return_value=mock_resp):
            with pytest.raises(WikiFetchError):
                client._fetch_random_title()


class TestWikipediaClient:
    """Test the online WikipediaClient (REST API, mocked HTTP)."""

    def _client(self, lang="de"):
        from src.wikipedia_fetcher import WikipediaClient
        return WikipediaClient(language=lang)

    def test_init(self):
        client = self._client("de")
        assert client.base_url == "https://de.wikipedia.org"
        assert client.language == "de"

    def test_fetch_random_title(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {
            "query": {"random": [{"id": 1, "ns": 0, "title": "Quantum physics"}]},
        }

        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            title, meta = client._fetch_random_title()
        assert title == "Quantum physics"
        assert meta == {}
        # Hits the classic API with main-namespace random (default lang 'de')
        url = mock_get.call_args[0][0]
        assert url == "https://de.wikipedia.org/w/api.php"
        params = mock_get.call_args[1]["params"]
        assert params["list"] == "random"
        assert params["rnnamespace"] == "0"

    def test_fetch_random_title_empty_raises(self):
        from src.wikipedia_fetcher import WikiFetchError
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"query": {}}

        with patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(WikiFetchError):
                client._fetch_random_title()

    def test_get_article_parses_api_response(self):
        client = self._client()
        html = '<div class="mw-parser-output"><p>Text</p></div>'
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"parse": {"title": "Quantum physics",
                                            "text": {"*": html}}}

        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            result = client.get_article("Quantum physics")
        assert result == html
        url = mock_get.call_args[0][0]
        assert url == "https://de.wikipedia.org/w/api.php"
        params = mock_get.call_args[1]["params"]
        assert params["action"] == "parse"
        assert params["page"] == "Quantum physics"
        assert params["prop"] == "text"

    def test_get_article_api_error_raises(self):
        from src.wikipedia_fetcher import WikiFetchError
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"error": {"code": "missingtitle", "info": "x"}}

        with patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(WikiFetchError, match="missingtitle"):
                client.get_article("Nonexistent page")

    def test_inherits_skip_patterns(self):
        from src.wikipedia_fetcher import WikipediaClient
        client = self._client()
        for title in ["List of universities", "Glossary of terms"]:
            assert any(skip in title for skip in client.SKIP_PATTERNS)

    def test_context_manager(self):
        with self._client() as client:
            assert client is not None

    def test_random_article_full_loop(self):
        """Shared loop: skip disambiguation pages, accept a good article."""
        from src.wikipedia_fetcher import WikipediaClient

        client = self._client("en")

        # First random: a page that turns out to be a disambiguation page
        # (caught by the text heuristics, like Kiwix). Second: a real
        # article with enough prose paragraphs.
        random_titles = ["Foo (disambiguation)".replace(" (disambiguation)", "")]
        titles = ["List of places" , "Good Article"]  # 1st skipped (title pattern)
        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None, **kw):
            calls["n"] += 1
            if params and params.get("list") == "random":
                t = titles[min(calls["n"] - 1, len(titles) - 1)]
                resp = MagicMock()
                resp.raise_for_status = lambda: None
                resp.json = lambda: {"query": {"random": [{"id": 1, "ns": 0, "title": t}]}}
                return resp
            # action=parse responses
            if params and params.get("page") == "Good Article":
                html = ('<div class="mw-parser-output">'
                        + ("<p>" + "meaningful prose sentence here " * 15 + "</p>") * 10
                        + "</div>")
                resp = MagicMock()
                resp.raise_for_status = lambda: None
                resp.json = lambda: {"parse": {"title": "Good Article",
                                               "text": {"*": html}}}
                return resp
            # The "List of places" article — too few prose paragraphs
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json = lambda: {"parse": {"title": "List of places",
                                           "text": {"*": '<div class="mw-parser-output"><p>short</p></div>'}}}
            return resp

        with patch.object(client.session, "get", side_effect=fake_get):
            title, text = client.get_random_article(min_words=50, max_words=600)

        assert title == "Good Article"
        assert len(text.split()) >= 50


class TestRandomArticleResilience:
    """Transient-error handling in the shared random-article loop."""

    def _client(self):
        from src.wikipedia_fetcher import WikipediaClient
        return WikipediaClient(language="en")

    @staticmethod
    def _net_error(retry_after=None):
        import requests
        e = requests.HTTPError("429 Too Many Requests")
        resp = MagicMock()
        resp.headers = {"Retry-After": str(retry_after)} if retry_after else {}
        e.response = resp
        return e

    def test_429_storm_aborts_early(self):
        """Persistent network failures stop after 5 consecutive errors."""
        client = self._client()
        with patch.object(client, "_fetch_random_title",
                          side_effect=self._net_error()), \
             patch("time.sleep") as mock_sleep:
            title, text = client.get_random_article()
        assert title == "Error"
        assert "consecutive network errors" in text
        # 5 attempts: sleep between 1-4, abort on 5th without sleeping
        assert mock_sleep.call_count == 4

    def test_retries_and_recovers(self):
        """A few network errors followed by success still delivers the article."""
        client = self._client()
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] <= 3:
                raise self._net_error()
            return "Good Article", {}

        paras = "<p>" + "meaningful prose sentence here " * 15 + "</p>"
        html = '<div class="mw-parser-output">' + paras * 10 + "</div>"

        with patch.object(client, "_fetch_random_title", side_effect=flaky), \
             patch.object(client, "get_article", return_value=html), \
             patch("time.sleep"):
            title, text = client.get_random_article(min_words=50)
        assert title == "Good Article"
        assert len(text.split()) >= 50

    def test_exhausts_attempts_reports_last_skip(self):
        client = self._client()
        calls = {"n": 0}

        def short_titles():
            calls["n"] += 1
            return f"Stub {calls['n']}", {}

        paras = "<p>tiny " * 3 + "</p>"
        html = '<div class="mw-parser-output">' + paras + "</div>"

        with patch.object(client, "_fetch_random_title", side_effect=short_titles), \
             patch.object(client, "get_article", return_value=html):
            title, text = client.get_random_article(max_attempts=3, min_words=250)
        assert title == "Error"
        assert "last skip" in text


class TestResolveBackend:
    """Test resolve_wikipedia_backend selection policy."""

    def test_auto_with_kiwix_entry_uses_kiwix(self):
        from src.wikipedia_fetcher import resolve_wikipedia_backend
        config = {"kiwix_servers": {"de": {"base_url": "http://k", "zim_name": "de_zim"}}}
        backend, params = resolve_wikipedia_backend(config, "de")
        assert backend == "kiwix"
        assert params == {"base_url": "http://k", "zim_name": "de_zim"}

    def test_auto_without_kiwix_entry_uses_online(self):
        from src.wikipedia_fetcher import resolve_wikipedia_backend
        config = {"kiwix_servers": {"de": {"base_url": "http://k", "zim_name": "de_zim"}}}
        backend, params = resolve_wikipedia_backend(config, "cs")
        assert backend == "online"
        assert params == {"language": "cs"}

    def test_empty_kiwix_servers_uses_online(self):
        from src.wikipedia_fetcher import resolve_wikipedia_backend
        backend, params = resolve_wikipedia_backend({}, "de")
        assert backend == "online"

    def test_explicit_online_overrides_kiwix(self):
        from src.wikipedia_fetcher import resolve_wikipedia_backend
        config = {
            "wikipedia": {"backend": "online"},
            "kiwix_servers": {"de": {"base_url": "http://k", "zim_name": "de_zim"}},
        }
        backend, params = resolve_wikipedia_backend(config, "de")
        assert backend == "online"

    def test_explicit_kiwix_without_entry_falls_back_online(self):
        from src.wikipedia_fetcher import resolve_wikipedia_backend
        config = {"wikipedia": {"backend": "kiwix"}}
        backend, params = resolve_wikipedia_backend(config, "de")
        assert backend == "online"

    def test_legacy_top_level_kiwix_block(self):
        from src.wikipedia_fetcher import resolve_wikipedia_backend
        config = {"kiwix": {"base_url": "http://legacy", "zim_name": "legacy_zim"}}
        backend, params = resolve_wikipedia_backend(config, "de")
        assert backend == "kiwix"
        assert params["base_url"] == "http://legacy"

    def test_language_is_case_insensitive(self):
        from src.wikipedia_fetcher import resolve_wikipedia_backend
        config = {"kiwix_servers": {"de": {"base_url": "http://k", "zim_name": "de_zim"}}}
        backend, _ = resolve_wikipedia_backend(config, "DE")
        assert backend == "kiwix"


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
