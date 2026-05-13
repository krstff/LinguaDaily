"""Tests for src/news_fetcher.py — RSS fetching, topic mapping, filtering."""

import pytest
from unittest.mock import patch, MagicMock


class TestNewsFetcherInit:
    """Test NewsFetcher initialization and feed resolution."""

    def test_default_init(self):
        from src.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        assert fetcher._feeds is not None

    def test_custom_feeds(self):
        from src.news_fetcher import NewsFetcher
        custom = {"Custom": ["https://example.com/feed.xml"]}
        fetcher = NewsFetcher(feeds=custom)
        assert "Custom" in fetcher._feeds

    def test_resolve_feeds_exact_match(self):
        from src.news_fetcher import NewsFetcher, FEED_CATALOGUE
        fetcher = NewsFetcher()
        feeds = fetcher._resolve_feeds("Technology")
        assert isinstance(feeds, list)
        assert len(feeds) > 0

    def test_resolve_feeds_case_insensitive(self):
        from src.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        feeds_upper = fetcher._resolve_feeds("technology")
        feeds_lower = fetcher._resolve_feeds("Technology")
        assert feeds_upper == feeds_lower

    def test_resolve_feeds_unknown_topic_fallback(self):
        from src.news_fetcher import NewsFetcher, DEFAULT_FEEDS
        fetcher = NewsFetcher()
        feeds = fetcher._resolve_feeds("NonexistentTopic")
        assert feeds == DEFAULT_FEEDS


class TestHtmlToText:
    """Test HTML-to-text extraction."""

    def test_basic(self):
        from src.news_fetcher import NewsFetcher
        result = NewsFetcher._html_to_text("<p>Hello</p>")
        assert "Hello" in result

    def test_nested_tags(self):
        from src.news_fetcher import NewsFetcher
        html = "<div><p><strong>Bold</strong> and <em>italic</em></p></div>"
        result = NewsFetcher._html_to_text(html)
        assert "Bold" in result
        assert "italic" in result

    def test_empty_html(self):
        from src.news_fetcher import NewsFetcher
        assert NewsFetcher._html_to_text("") == ""

    def test_none_html(self):
        from src.news_fetcher import NewsFetcher
        assert NewsFetcher._html_to_text(None) == ""

    def test_blank_line_collapse(self):
        from src.news_fetcher import NewsFetcher
        html = "<p>A</p><br><br><p>B</p>"
        result = NewsFetcher._html_to_text(html)
        # Should not have excessive blank lines
        assert "\n\n\n" not in result


class TestFilterArticle:
    """Test article length filtering."""

    @pytest.fixture
    def fetcher(self):
        from src.news_fetcher import NewsFetcher
        return NewsFetcher()

    def test_article_too_short(self, fetcher):
        article = {
            "title": "Short",
            "body_html": "<p>Not enough words here.</p>",
            "link": "http://example.com",
        }
        result = fetcher._filter_article(article, min_words=250, max_words=600)
        assert result is None

    def test_article_perfect_length(self, fetcher):
        article = {
            "title": "Just Right",
            "body_html": "<p>" + "word " * 300 + "</p>",
            "link": "http://example.com",
        }
        result = fetcher._filter_article(article, min_words=250, max_words=600)
        assert result is not None
        title, text = result
        assert title == "Just Right"
        assert 250 <= len(text.split()) <= 600

    def test_article_truncated(self, fetcher):
        article = {
            "title": "Too Long",
            "body_html": "<p>" + "word " * 1000 + "</p>",
            "link": "http://example.com",
        }
        result = fetcher._filter_article(article, min_words=250, max_words=600)
        assert result is not None
        title, text = result
        assert len(text.split()) <= 600

    def test_article_truncated_with_paragraphs(self, fetcher):
        article = {
            "title": "Multi Para",
            "body_html": "\n\n".join([f"<p>word{i} " * 100 + "</p>" for i in range(10)]),
            "link": "http://example.com",
        }
        result = fetcher._filter_article(article, min_words=250, max_words=600)
        assert result is not None
        title, text = result
        assert 250 <= len(text.split()) <= 600


class TestFetchFeed:
    """Test RSS feed fetching (mocked)."""

    @pytest.fixture
    def fetcher(self):
        from src.news_fetcher import NewsFetcher
        return NewsFetcher()

    @patch("src.news_fetcher.feedparser")
    def test_fetch_feed_success(self, mock_fp, fetcher):
        body_html = "<p>" + "word " * 300 + "</p>"
        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda key, default=None: {
            "title": "Test Article",
            "content": [{"type": "html", "value": body_html}],
            "description": "",
            "link": "http://example.com/article",
            "published": "2026-01-01",
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_fp.parse.return_value = mock_feed

        with patch.object(fetcher._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = "<rss></rss>"
            mock_resp.raise_for_status = lambda: None
            mock_get.return_value = mock_resp

            articles = fetcher._fetch_feed("https://example.com/feed.xml", max_items=10)
        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"

    @patch("src.news_fetcher.feedparser")
    def test_fetch_feed_failure(self, mock_fp, fetcher):
        import requests
        with patch.object(fetcher._session, "get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            articles = fetcher._fetch_feed("https://broken.com/feed.xml")
        assert articles == []

    @patch("src.news_fetcher.feedparser")
    def test_fetch_feed_prefers_content_encoded(self, mock_fp, fetcher):
        """content:encoded should be preferred over description."""
        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda key, default=None: {
            "title": "Test",
            "content": [{"type": "html", "value": "<p>Full content here</p>"}],
            "description": "Short teaser",
            "link": "",
            "published": "",
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_fp.parse.return_value = mock_feed

        with patch.object(fetcher._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = "<rss></rss>"
            mock_resp.raise_for_status = lambda: None
            mock_get.return_value = mock_resp

            articles = fetcher._fetch_feed("https://example.com/feed.xml")
        body_html = articles[0]["body_html"]
        assert "Full content" in body_html


class TestFetchByTopic:
    """End-to-end topic fetching (mocked)."""

    @patch("src.news_fetcher.feedparser")
    def test_fetch_by_topic_returns_article(self, mock_fp):
        from src.news_fetcher import NewsFetcher

        # Create mock articles with sufficient length
        mock_entries = []
        for i in range(5):
            body_html = f"<p>word{i} " * 300 + "</p>"
            entry_data = {
                "title": f"Article {i}",
                "content": [{"type": "html", "value": body_html}],
                "description": "",
                "link": f"http://example.com/{i}",
                "published": "",
            }
            entry = MagicMock()
            entry.get.side_effect = lambda key, default=None, d=entry_data: d.get(key, default)
            mock_entries.append(entry)

        mock_feed = MagicMock()
        mock_feed.entries = mock_entries
        mock_fp.parse.return_value = mock_feed

        with NewsFetcher() as fetcher:
            with patch.object(fetcher._session, "get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.text = "<rss></rss>"
                mock_resp.raise_for_status = lambda: None
                mock_get.return_value = mock_resp

                title, text = fetcher.fetch_by_topic("Technology", min_words=250, max_words=600)
        assert title is not None
        assert len(text.split()) >= 250

    @patch("src.news_fetcher.feedparser")
    def test_fetch_by_topic_no_eligible_articles(self, mock_fp):
        from src.news_fetcher import NewsFetcher

        # All articles too short
        entry_data = {
            "title": "Too Short",
            "content": [{"type": "html", "value": "<p>Brief.</p>"}],
            "description": "",
            "link": "",
            "published": "",
        }
        entry = MagicMock()
        entry.get.side_effect = lambda key, default=None, d=entry_data: d.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [entry]
        mock_fp.parse.return_value = mock_feed

        with NewsFetcher() as fetcher:
            with patch.object(fetcher._session, "get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.text = "<rss></rss>"
                mock_resp.raise_for_status = lambda: None
                mock_get.return_value = mock_resp

                title, text = fetcher.fetch_by_topic("Technology", min_words=250, max_words=600)
        assert title is None


class TestFeedCatalogue:
    """Test that the feed catalogue covers all expected topics."""

    def test_all_topics_have_feeds(self):
        from src.news_fetcher import FEED_CATALOGUE, DEFAULT_FEEDS
        # Check that every topic in the default config has a mapping
        expected_topics = [
            "Technology", "Science", "Mathematics", "History", "Art", "Music",
            "Philosophy", "Literature", "Architecture", "Biology", "Physics",
            "Chemistry", "Geography", "Astronomy", "Psychology", "Economics",
            "Politics", "Medicine", "Culture",
        ]
        # FEED_CATALOGUE is structured as {lang: {topic: [urls]}}
        for lang, topics in FEED_CATALOGUE.items():
            for topic in expected_topics:
                assert topic in topics, f"Missing feed mapping for topic: {topic} (lang: {lang})"

    def test_feeds_are_http_urls(self):
        from src.news_fetcher import FEED_CATALOGUE
        # FEED_CATALOGUE is structured as {lang: {topic: [urls]}}
        for lang, topics in FEED_CATALOGUE.items():
            for topic, feeds in topics.items():
                for url in feeds:
                    assert url.startswith("http"), f"{lang}/{topic}: URL should start with http: {url}"
