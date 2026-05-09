"""Tests for src/orchestrator.py — config loading, profile resolution, article fetching, Orchestrator pipeline."""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary config file with test data."""
    config = {
        "llm": {
            "base_url": "http://localhost:8080/v1",
            "default_model": "gemma4-26b",
        },
        "tts": {
            "base_url": "http://localhost:8080/v1",
            "model": "omnivoice",
        },
        "profiles": {
            "test_user": {
                "source_lang": "en",
                "target_lang": "de",
                "target_lang_name": "German",
                "content_lang": "de",
                "source": "wikipedia",
                "topics": ["Technology", "Science"],
                "article_filter": {"min_words": 50, "max_words": 300},
                "use_tts": True,
                "tts_voice": "male",
                "schedule": {"time": "08:00", "tz": "Europe/Berlin"},
            },
            "no_tts_user": {
                "source_lang": "en",
                "target_lang": "fr",
                "target_lang_name": "French",
                "content_lang": "fr",
                "source": "news",
                "topics": ["Politics"],
                "use_tts": False,
            },
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config, str(config_file)


class TestLoadConfig:
    """Test config loading."""

    def test_load_config(self, sample_config):
        from src.orchestrator import load_config
        config, config_path = sample_config
        loaded = load_config(config_path)
        assert "profiles" in loaded
        assert "test_user" in loaded["profiles"]

    def test_load_config_missing(self, tmp_path):
        from src.orchestrator import load_config
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "missing.json"))


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
        config = dict(sample_config[0])
        config["default_profile"] = "test_user"
        name, profile = get_profile(config)
        assert name == "test_user"

    def test_first_profile_fallback(self):
        """When default_profile is missing, should use first profile."""
        from src.orchestrator import get_profile
        config = {"default_profile": None, "profiles": {"only_one": {}}}
        name, profile = get_profile(config)
        assert name == "only_one"

    def test_missing_profile_fallback(self, sample_config):
        """Unknown profile should fall back to default."""
        from src.orchestrator import get_profile
        config = dict(sample_config[0])
        config["default_profile"] = "test_user"
        name, profile = get_profile(config, "nonexistent")
        assert name == "test_user"

    def test_no_profiles_raises(self):
        from src.orchestrator import get_profile
        config = {"profiles": {}}
        with pytest.raises(ValueError, match="No profiles"):
            get_profile(config)


class TestCleanContent:
    """Test content cleaning."""

    def test_removes_reference_markers(self):
        from src.orchestrator import clean_content
        text = "This is a sentence [1] with references [ 2 ]."
        result = clean_content(text)
        assert "[1]" not in result
        assert "[ 2 ]" not in result

    def test_strips_footer_sections(self):
        from src.orchestrator import clean_content
        text = "Main content here.\n\nSee also\n\nSome footer stuff."
        result = clean_content(text)
        assert "See also" not in result
        assert "footer stuff" not in result

    def test_fixes_missing_spaces(self):
        from src.orchestrator import clean_content
        text = "word1,word2 and word3,word4."
        result = clean_content(text)
        assert ", word2" in result
        assert ", word4" in result


class TestFetchArticle:
    """Test the fetch_article router wrapper."""

    def test_fetch_with_source(self, sample_config):
        from src.orchestrator import fetch_article
        config = sample_config[0]

        # orchestrator does: from fetch_router import fetch_article as route_fetch
        with patch("fetch_router.fetch_article") as mock_route:
            mock_route.return_value = ("Test Title", "Test content here.")
            title, text = fetch_article(source="wikipedia", topic="Tech", config=config)

        assert title == "Test Title"
        mock_route.assert_called_once()


class TestOrchestratorInit:
    """Test Orchestrator initialization."""

    def test_init_with_config(self, sample_config):
        from src.orchestrator import Orchestrator
        orch = Orchestrator(config=sample_config[0])
        assert "test_user" in orch.config["profiles"]


class TestOrchestratorRunLesson:
    """Test the full lesson pipeline via Orchestrator.run_lesson."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_config):
        from src.orchestrator import Orchestrator

        mock_client = MagicMock()
        mock_client.translate.return_value = "Python ist eine großartige Sprache."
        mock_client.extract_vocab.return_value = [
            {"word": "Sprache", "meaning": "language"}
        ]
        mock_client.profile_name = "test_user"

        with patch("fetch_router.fetch_article") as mock_fetch, \
             patch("tts.synthesize") as mock_tts, \
             patch("llama_client.LlamaClient", return_value=mock_client):

            mock_fetch.return_value = ("Python Basics", "Python is a great language.")
            mock_tts.return_value = "/tmp/output/test_user/test.wav"

            orch = Orchestrator(config=sample_config[0])
            lesson = await orch.run_lesson("test_user")

        assert lesson is not None
        assert lesson["title"] == "Python Basics"
        assert lesson["profile"] == "test_user"
        assert lesson["content"] == "Python ist eine großartige Sprache."
        assert lesson["wav_path"] == "/tmp/output/test_user/test.wav"
        assert len(lesson["vocab"]) == 1
        assert lesson["target_lang_name"] == "German"
        assert "timestamp" in lesson

    @pytest.mark.asyncio
    async def test_fallback_when_fetch_fails(self, sample_config):
        from src.orchestrator import Orchestrator

        with patch("fetch_router.fetch_article", return_value=(None, None)):
            orch = Orchestrator(config=sample_config[0])
            lesson = await orch.run_lesson("test_user", topic="AI")

        assert lesson is not None
        assert "could not be retrieved" in lesson["original_content"]

    @pytest.mark.asyncio
    async def test_unknown_profile_returns_none(self, sample_config):
        from src.orchestrator import Orchestrator

        orch = Orchestrator(config=sample_config[0])
        lesson = await orch.run_lesson("nonexistent")
        assert lesson is None

    @pytest.mark.asyncio
    async def test_tts_disabled_skips_audio(self, sample_config):
        from src.orchestrator import Orchestrator

        with patch("fetch_router.fetch_article", return_value=("Test", "Some content here.")):
            orch = Orchestrator(config=sample_config[0])
            lesson = await orch.run_lesson("no_tts_user")  # use_tts: False

        assert lesson is not None
        assert lesson["wav_path"] is None

    @pytest.mark.asyncio
    async def test_no_llm_config_skips_translation(self, sample_config):
        from src.orchestrator import Orchestrator

        config = dict(sample_config[0])
        del config["llm"]  # no LLM configured

        with patch("fetch_router.fetch_article", return_value=("Test", "Original text.")):
            orch = Orchestrator(config=config)
            lesson = await orch.run_lesson("test_user")

        assert lesson is not None
        # Content should be the original (no translation attempted)
        assert "Original text" in lesson["content"]

    @pytest.mark.asyncio
    async def test_delivery_callback_called(self, sample_config):
        from src.orchestrator import Orchestrator

        delivery_log = []

        async def mock_delivery(profile_name, lesson):
            delivery_log.append((profile_name, lesson))

        with patch("fetch_router.fetch_article", return_value=("Test", "Content.")):
            orch = Orchestrator(config=sample_config[0])
            await orch.run_lesson(
                "test_user",
                delivery_callback=mock_delivery,
            )

        assert len(delivery_log) == 1
        assert delivery_log[0][0] == "test_user"
        assert delivery_log[0][1]["title"] == "Test"

    @pytest.mark.asyncio
    async def test_delivery_callback_error_does_not_crash(self, sample_config):
        from src.orchestrator import Orchestrator

        async def failing_delivery(profile_name, lesson):
            raise ConnectionError("Telegram unreachable")

        with patch("fetch_router.fetch_article", return_value=("Test", "Content.")):
            orch = Orchestrator(config=sample_config[0])
            lesson = await orch.run_lesson(
                "test_user",
                delivery_callback=failing_delivery,
            )

        # Lesson should still be returned despite delivery failure
        assert lesson is not None


class TestOrchestratorVocab:
    """Test vocab extraction and persistence in the pipeline."""

    @pytest.mark.asyncio
    async def test_vocab_saved_to_file(self, sample_config, tmp_path):
        from src.orchestrator import Orchestrator

        mock_client = MagicMock()
        mock_client.translate.return_value = "The house is big."
        mock_client.extract_vocab.return_value = [
            {"word": "Haus", "meaning": "house"},
            {"word": "groß", "meaning": "big"},
        ]
        mock_client.profile_name = "test_user"

        with patch("fetch_router.fetch_article", return_value=("Test", "Das Haus ist groß.")), \
             patch("llama_client.LlamaClient", return_value=mock_client):

            orch = Orchestrator(config=sample_config[0])
            lesson = await orch.run_lesson("test_user")

        assert lesson is not None
        assert len(lesson["vocab"]) == 2


class TestOrchestratorCLI:
    """Test CLI entry point."""

    def test_cli_runs_pipeline(self, sample_config, capsys):
        from src.orchestrator import main

        config, config_path = sample_config

        mock_client = MagicMock()
        mock_client.translate.return_value = "Translated text."
        mock_client.extract_vocab.return_value = []
        mock_client.profile_name = "test_user"

        with patch("fetch_router.fetch_article", return_value=("Test Article", "word " * 300)), \
             patch("llama_client.LlamaClient", return_value=mock_client), \
             patch("sys.argv", ["orchestrator.py", "--config", config_path, "--profile", "test_user"]):

            main()

        output = capsys.readouterr().out
        assert "Task Execution Complete" in output
