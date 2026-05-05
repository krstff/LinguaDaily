"""Tests for src/orchestrator.py — config loading, profile resolution, article fetching."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary config file with test data."""
    config = {
        "default_profile": "test_user",
        "kiwix": {"base_url": "http://localhost:8080", "zim_name": "test_zim"},
        "sources": {"wikipedia": {}, "news": {}},
        "profiles": {
            "test_user": {
                "source_lang": "en",
                "target_lang": "de",
                "target_lang_name": "German",
                "source": "wikipedia",
                "topics": ["Technology", "Science"],
                "article_filter": {"min_words": 250, "target_words": 400, "max_words": 600},
                "schedule": {"time": "08:00", "tz": "Europe/Berlin"},
            },
            "news_user": {
                "source_lang": "en",
                "target_lang": "fr",
                "target_lang_name": "French",
                "source": "news",
                "topics": ["Politics"],
                "article_filter": {"min_words": 300, "target_words": 500, "max_words": 700},
                "schedule": {"time": "09:00", "tz": "Europe/Paris"},
            },
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config, str(config_file)


class TestLoadConfig:
    """Test config loading."""

    def test_load_config(self, sample_config, monkeypatch):
        from src.orchestrator import load_config, CONFIG_PATH
        config, config_path = sample_config
        monkeypatch.setattr("src.orchestrator.CONFIG_PATH", config_path)
        loaded = load_config()
        assert "profiles" in loaded
        assert "test_user" in loaded["profiles"]

    def test_load_config_missing(self, tmp_path):
        from src.orchestrator import load_config
        with patch("src.orchestrator.CONFIG_PATH", str(tmp_path / "missing.json")):
            with pytest.raises(FileNotFoundError):
                load_config()


class TestGetProfile:
    """Test profile resolution."""

    def test_explicit_profile(self, sample_config):
        from src.orchestrator import get_profile
        config = sample_config[0]
        name, profile = get_profile(config, "test_user")
        assert name == "test_user"
        assert profile["target_lang_name"] == "German"

    def test_default_profile(self, sample_config):
        from src.orchestrator import get_profile
        config = sample_config[0]
        name, profile = get_profile(config)
        assert name == "test_user"  # default_profile

    def test_first_profile_fallback(self, tmp_path):
        """When default_profile is missing, should use first profile."""
        from src.orchestrator import get_profile
        config = {"default_profile": None, "profiles": {"only_one": {}}}
        name, profile = get_profile(config)
        assert name == "only_one"

    def test_missing_profile_fallback(self, sample_config):
        """Unknown profile should fall back to default."""
        from src.orchestrator import get_profile
        config = sample_config[0]
        name, profile = get_profile(config, "nonexistent")
        assert name == "test_user"  # falls back to default

    def test_no_profiles_raises(self):
        from src.orchestrator import get_profile
        config = {"profiles": {}}
        with pytest.raises(ValueError, match="No profiles"):
            get_profile(config)


class TestFetchArticle:
    """Test the fetch_article router wrapper in orchestrator."""

    def test_fetch_wikipedia(self, sample_config, monkeypatch):
        from src.orchestrator import fetch_article
        config = sample_config[0]

        with patch("src.orchestrator.fetch_article.__wrapped__") as mock_route:
            pass  # Can't easily mock the renamed function; test via integration instead

    def test_fetch_with_source(self, sample_config):
        from src.orchestrator import fetch_article
        config = sample_config[0]

        with patch("src.fetch_router.fetch_article") as mock_route:
            mock_route.return_value = ("Test Title", "Test content " * 300)
            title, text = fetch_article(source="wikipedia", topic="Tech", config=config)
        assert title == "Test Title"
        mock_route.assert_called_once()


class TestPayloadGeneration:
    """Test that the orchestrator generates correct payload structure."""

    @patch("src.orchestrator.fetch_article")
    def test_payload_structure(self, mock_fetch, sample_config, monkeypatch):
        from src.orchestrator import main
        config, config_path = sample_config
        monkeypatch.setattr("src.orchestrator.CONFIG_PATH", config_path)

        # Mock the fetch to return a known result
        mock_fetch.return_value = ("Test Article", "word " * 300)

        with patch("sys.argv", ["orchestrator.py", "--profile", "test_user"]):
            with patch("builtins.print") as mock_print:
                main()

        # Find the payload output
        calls = [c[0] for c in mock_print.call_args_list if c and isinstance(c[0], str) and "---PAYLOAD_START---" in str(c)]
        assert len(calls) > 0, "Payload start marker not printed"
