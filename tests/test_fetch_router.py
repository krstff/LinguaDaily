"""Tests for src/fetch_router.py — content source dispatching."""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open


class TestFetchRouterDispatch:
    """Test that the router dispatches to the correct fetcher."""

    def test_dispatch_wikipedia(self):
        from src.fetch_router import fetch_article
        config = {"kiwix": {}, "profiles": {}}

        with patch("src.fetch_router._fetch_wikipedia") as mock_wiki:
            mock_wiki.return_value = ("Wiki Title", "Some content")
            title, text = fetch_article("wikipedia", "Tech", config)
        mock_wiki.assert_called_once_with(config, learning_language=None, article_filter=None)
        assert title == "Wiki Title"

    def test_dispatch_wikipedia_with_options(self):
        from src.fetch_router import fetch_article
        config = {"kiwix": {}, "profiles": {}}

        with patch("src.fetch_router._fetch_wikipedia") as mock_wiki:
            mock_wiki.return_value = ("Wiki Title", "Some content")
            title, text = fetch_article(
                "wikipedia", "Tech", config,
                learning_language="de",
                article_filter={"min_words": 300},
            )
        mock_wiki.assert_called_once_with(
            config, learning_language="de", article_filter={"min_words": 300}
        )
        assert title == "Wiki Title"

    def test_dispatch_news(self):
        from src.fetch_router import fetch_article
        config = {"sources": {"news": {}}, "article_filter": {}}

        with patch("src.fetch_router._fetch_news") as mock_news:
            mock_news.return_value = ("News Title", "Some content")
            title, text = fetch_article("news", "Politics", config)
        mock_news.assert_called_once()
        assert title == "News Title"

    def test_unknown_source_fallback(self):
        from src.fetch_router import fetch_article
        config = {"kiwix": {}, "profiles": {}}

        with patch("src.fetch_router._fetch_wikipedia") as mock_wiki:
            mock_wiki.return_value = ("Fallback", "Content")
            title, text = fetch_article("unknown_source", "Topic", config)
        # Should fall back to wikipedia
        mock_wiki.assert_called_once()


class TestFetchWikipedia:
    """Test Wikipedia fetching via direct import."""

    def test_success(self):
        from src.fetch_router import _fetch_wikipedia
        config = {}

        mock_article = ("Test Title", "Some content")

        with patch("wikipedia_fetcher.KiwixClient") as MockClient, \
             patch("wikipedia_fetcher.load_fetcher_config") as mock_load:
            mock_load.return_value = {
                "base_url": "http://localhost:8080",
                "zim_name": "wikipedia_de",
                "article_filter": {"min_words": 250, "max_words": 600},
            }
            instance = MagicMock()
            instance.get_random_article.return_value = mock_article
            MockClient.return_value.__enter__ = MagicMock(return_value=instance)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            title, text = _fetch_wikipedia(config, learning_language="de")
        assert title == "Test Title"
        assert text == "Some content"

    def test_passes_article_filter_overrides(self):
        from src.fetch_router import _fetch_wikipedia
        config = {}
        article_filter = {"min_words": 300, "max_words": 500}

        with patch("wikipedia_fetcher.load_fetcher_config") as mock_load:
            mock_load.return_value = {
                "base_url": "http://localhost:8080",
                "zim_name": "wikipedia_en",
                "article_filter": {"min_words": 250, "max_words": 600},
            }
            instance = MagicMock()
            instance.get_random_article.return_value = ("Title", "Content")
            with patch("wikipedia_fetcher.KiwixClient") as MockClient:
                MockClient.return_value.__enter__ = MagicMock(return_value=instance)
                MockClient.return_value.__exit__ = MagicMock(return_value=False)

                _fetch_wikipedia(config, article_filter=article_filter)
        # Verify the filter overrides were merged into settings before calling
        assert instance.get_random_article.call_args[1]["min_words"] == 300
        assert instance.get_random_article.call_args[1]["max_words"] == 500

    def test_error_returns_none(self):
        from src.fetch_router import _fetch_wikipedia
        config = {}

        with patch("wikipedia_fetcher.load_fetcher_config") as mock_load:
            mock_load.return_value = {
                "base_url": "http://localhost:8080",
                "zim_name": "wikipedia_en",
                "article_filter": {"min_words": 250, "max_words": 600},
            }
            with patch("wikipedia_fetcher.KiwixClient") as MockClient:
                MockClient.return_value.__enter__.side_effect = Exception("Connection refused")

                title, text = _fetch_wikipedia(config)
            assert title is None
            assert text is None

    def test_resolves_learning_language(self):
        """load_fetcher_config should be called with the learning_language."""
        from src.fetch_router import _fetch_wikipedia
        config = {}

        with patch("wikipedia_fetcher.load_fetcher_config") as mock_load:
            mock_load.return_value = {
                "base_url": "http://localhost:8080",
                "zim_name": "wikipedia_de",
                "article_filter": {"min_words": 250, "max_words": 600},
            }
            instance = MagicMock()
            instance.get_random_article.return_value = ("Title", "Content")
            with patch("wikipedia_fetcher.KiwixClient") as MockClient:
                MockClient.return_value.__enter__ = MagicMock(return_value=instance)
                MockClient.return_value.__exit__ = MagicMock(return_value=False)

                _fetch_wikipedia(config, learning_language="de")
        mock_load.assert_called_once_with(learning_language="de")


class TestFetchNews:
    """Test news fetching."""

    def test_success(self):
        from src.fetch_router import _fetch_news
        config = {
            "sources": {"news": {"feeds": {}, "categories": {}}},
            "article_filter": {"min_words": 250, "max_words": 600},
        }

        with patch("news_fetcher.NewsFetcher") as MockNF:
            instance = MagicMock()
            instance.fetch_by_topic.return_value = ("News Title", "News content")
            MockNF.return_value = instance

            title, text = _fetch_news("Politics", config)
        assert title == "News Title"

    def test_default_article_filter(self):
        """Should use defaults when article_filter is missing."""
        from src.fetch_router import _fetch_news
        config = {
            "sources": {"news": {}},
            # No article_filter key
        }

        with patch("news_fetcher.NewsFetcher") as MockNF:
            instance = MagicMock()
            instance.fetch_by_topic.return_value = ("Title", "Content")
            MockNF.return_value = instance

            title, text = _fetch_news("Topic", config)
        assert title == "Title"
        # Should use default min/max_words since no article_filter in config
        call_kwargs = instance.fetch_by_topic.call_args[1]
        assert call_kwargs["min_words"] == 250
        assert call_kwargs["max_words"] == 600

    def test_picks_random_topic_when_none(self):
        """When topic is None, should pick a random topic."""
        from src.fetch_router import _fetch_news
        config = {
            "sources": {"news": {}},
            "article_filter": {},
        }

        with patch("news_fetcher.NewsFetcher") as MockNF:
            instance = MagicMock()
            instance.pick_random_topic.return_value = "Technology"
            instance.fetch_by_topic.return_value = ("Tech News", "Content")
            MockNF.return_value = instance

            title, text = _fetch_news(None, config)
        # Should have called pick_random_topic and then fetch with the result
        instance.pick_random_topic.assert_called_once()
        instance.fetch_by_topic.assert_called_once_with(
            "Technology", min_words=250, max_words=600
        )
        assert title == "Tech News"

    def test_returns_none_when_no_topics_available(self):
        """When topic is None and no topics exist, should return (None, None)."""
        from src.fetch_router import _fetch_news
        config = {
            "sources": {"news": {}},
            "article_filter": {},
        }

        with patch("news_fetcher.NewsFetcher") as MockNF:
            instance = MagicMock()
            instance.pick_random_topic.return_value = None  # No topics available
            MockNF.return_value = instance

            title, text = _fetch_news(None, config)
        assert title is None
        assert text is None
        # fetch_by_topic should NOT have been called
        instance.fetch_by_topic.assert_not_called()


class TestCli:
    """Test CLI entry points."""

    @patch("src.fetch_router.fetch_article")
    def test_cli_output(self, mock_fetch):
        from src.fetch_router import main
        mock_fetch.return_value = ("CLI Test", "word " * 300)

        with patch("sys.argv", ["fetch_router.py", "--source", "wikipedia", "Topic"]), \
             patch("src.config.load_config") as mock_load_cfg:
            mock_load_cfg.return_value = {"kiwix": {}, "profiles": {}}
            with patch("builtins.print") as mock_print:
                main()

        args, kwargs = mock_print.call_args
        output = json.loads(args[0])
        assert output["title"] == "CLI Test"
        assert output["source"] == "wikipedia"
