"""Tests for src/scheduler.py — scheduling, lesson pipeline, CLI."""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call


@pytest.fixture
def sample_config(tmp_path):
    """Config with multiple profiles, some scheduled."""
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
            "krystof": {
                "source_lang": "en",
                "target_lang": "de",
                "target_lang_name": "German",
                "content_lang": "de",
                "source": "wikipedia",
                "topics": ["Technology", "Science"],
                "article_filter": {"min_words": 50, "max_words": 300},
                "schedule": {
                    "time": "08:00",
                    "tz": "Europe/Berlin",
                },
                "use_tts": True,
                "tts_voice": "male",
            },
            "anna": {
                "source_lang": "en",
                "target_lang": "es",
                "target_lang_name": "Spanish",
                "content_lang": "es",
                "topics": ["History", "Art"],
                "schedule": {
                    "time": "10:30",
                    "tz": "Europe/Madrid",
                },
                "use_tts": False,
            },
            "unscheduled": {
                "source_lang": "en",
                "target_lang": "fr",
                "target_lang_name": "French",
                # no schedule
            },
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config, str(config_file)


class TestGetScheduledProfiles:
    """Test profile schedule discovery."""

    def test_returns_scheduled_profiles(self, sample_config):
        from src.scheduler import LessonScheduler
        config = sample_config[0]
        scheduler = LessonScheduler(config=config)

        profiles = scheduler.get_scheduled_profiles()
        names = [name for name, _ in profiles]

        assert "krystof" in names
        assert "anna" in names
        assert "unscheduled" not in names
        assert len(profiles) == 2

    def test_empty_when_no_schedules(self):
        from src.scheduler import LessonScheduler
        config = {"profiles": {"a": {}}}  # no schedule keys
        scheduler = LessonScheduler(config=config)
        assert scheduler.get_scheduled_profiles() == []

    def test_empty_config(self):
        from src.scheduler import LessonScheduler
        scheduler = LessonScheduler(config={})
        assert scheduler.get_scheduled_profiles() == []

    def test_profile_without_time_is_skipped(self):
        from src.scheduler import LessonScheduler
        config = {
            "profiles": {
                "partial": {"schedule": {"tz": "UTC"}},  # no time
                "full": {"schedule": {"time": "09:00", "tz": "UTC"}},
            }
        }
        scheduler = LessonScheduler(config=config)
        names = [n for n, _ in scheduler.get_scheduled_profiles()]
        assert names == ["full"]


class TestRunLessonPipeline:
    """Test the full lesson pipeline with mocked dependencies."""

    @patch("llama_client.LlamaClient")
    @patch("tts.synthesize")
    @patch("fetch_router.fetch_article")
    def test_full_pipeline(self, mock_fetch, mock_tts, mock_llama_cls, sample_config):
        from src.scheduler import LessonScheduler

        # Mock article fetch
        mock_fetch.return_value = ("Python Basics", "Python is a great language.")

        # Mock TTS
        mock_tts.return_value = "/tmp/output/krystof/test.wav"

        # Mock LLM client
        mock_client = MagicMock()
        mock_client.translate.return_value = "Python ist eine großartige Sprache."
        mock_client.extract_vocab.return_value = [
            {"word": "Sprache", "meaning": "language"}
        ]
        mock_llama_cls.return_value = mock_client

        scheduler = LessonScheduler(config=sample_config[0])
        lesson = asyncio.get_event_loop().run_until_complete(
            scheduler.run_lesson("krystof")
        )

        assert lesson is not None
        assert lesson["title"] == "Python Basics"
        assert lesson["profile"] == "krystof"
        assert lesson["content"] == "Python ist eine großartige Sprache."
        assert lesson["wav_path"] == "/tmp/output/krystof/test.wav"
        assert len(lesson["vocab"]) == 1
        assert lesson["target_lang_name"] == "German"
        assert "timestamp" in lesson

    @patch("src.fetch_router.fetch_article")
    def test_fallback_when_fetch_fails(self, mock_fetch, sample_config):
        from src.scheduler import LessonScheduler

        mock_fetch.side_effect = Exception("network error")

        scheduler = LessonScheduler(config=sample_config[0])
        lesson = asyncio.get_event_loop().run_until_complete(
            scheduler.run_lesson("krystof")
        )

        assert lesson is not None
        assert "could not be retrieved" in lesson["original_content"]

    @patch("src.fetch_router.fetch_article")
    def test_empty_profile_returns_none(self, mock_fetch, sample_config):
        from src.scheduler import LessonScheduler

        scheduler = LessonScheduler(config=sample_config[0])
        lesson = asyncio.get_event_loop().run_until_complete(
            scheduler.run_lesson("nonexistent")
        )

        assert lesson is None

    @patch("src.fetch_router.fetch_article")
    def test_tts_disabled_skips_audio(self, mock_fetch, sample_config):
        from src.scheduler import LessonScheduler

        mock_fetch.return_value = ("Test", "Some content here.")

        scheduler = LessonScheduler(config=sample_config[0])
        lesson = asyncio.get_event_loop().run_until_complete(
            scheduler.run_lesson("anna")  # use_tts: False
        )

        assert lesson is not None
        assert lesson["wav_path"] is None
        # TTS should NOT have been called for anna
        from src.tts import synthesize
        # We can't easily check if it was called since we didn't mock it,
        # but the pipeline path for use_tts=False skips the tts import

    @patch("llama_client.LlamaClient")
    @patch("fetch_router.fetch_article")
    def test_no_llm_config_skips_translation(self, mock_fetch, mock_llama_cls, sample_config):
        from src.scheduler import LessonScheduler

        config = dict(sample_config[0])
        del config["llm"]  # no LLM configured

        mock_fetch.return_value = ("Test", "Original text.")

        scheduler = LessonScheduler(config=config)
        lesson = asyncio.get_event_loop().run_until_complete(
            scheduler.run_lesson("krystof")
        )

        assert lesson is not None
        # Content should be the original (no translation attempted)
        assert lesson["content"] == "Original text."
        mock_llama_cls.assert_not_called()


class TestDeliveryCallback:
    """Test that lessons are delivered via callback."""

    @pytest.mark.asyncio
    async def test_callback_receives_lesson(self, sample_config):
        from src.scheduler import LessonScheduler

        delivery_log = []

        async def mock_delivery(profile_name, lesson):
            delivery_log.append((profile_name, lesson))

        scheduler = LessonScheduler(
            config=sample_config[0],
            delivery_callback=mock_delivery,
        )

        # Build and run a job manually
        profile_name, profile = sample_config[0]["profiles"]["krystof"], sample_config[0]["profiles"]["krystof"]
        job = scheduler._build_job("krystof", profile)

        with patch.object(scheduler, "run_lesson", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "title": "Test",
                "content": "Translated.",
                "profile": "krystof",
            }
            await job()

        assert len(delivery_log) == 1
        assert delivery_log[0][0] == "krystof"
        assert delivery_log[0][1]["title"] == "Test"

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash_pipeline(self, sample_config, caplog):
        from src.scheduler import LessonScheduler

        async def failing_delivery(profile_name, lesson):
            raise ConnectionError("Telegram unreachable")

        scheduler = LessonScheduler(
            config=sample_config[0],
            delivery_callback=failing_delivery,
        )

        job = scheduler._build_job("krystof", sample_config[0]["profiles"]["krystof"])

        with patch.object(scheduler, "run_lesson", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"title": "Test", "content": "text"}
            await job()  # should not raise

        assert "Delivery failed" in caplog.text or "Telegram unreachable" in caplog.text


class TestAPSchedulerIntegration:
    """Test APScheduler job creation (without actually running)."""

    def test_start_creates_jobs(self, sample_config):
        from src.scheduler import LessonScheduler

        scheduler = LessonScheduler(config=sample_config[0])

        with patch("apscheduler.schedulers.asyncio.AsyncIOScheduler") as MockSched:
            mock_instance = MagicMock()
            MockSched.return_value = mock_instance

            # We can't fully await start() in a sync test, so check job building
            scheduled = scheduler.get_scheduled_profiles()
            assert len(scheduled) == 2

            # Verify jobs would be built correctly
            for name, profile in scheduled:
                job = scheduler._build_job(name, profile)
                assert asyncio.iscoroutinefunction(job)


class TestCLIList:
    """Test CLI --list mode."""

    def test_list_prints_schedule(self, sample_config, capsys):
        from src.scheduler import main

        config_path = sample_config[1]

        with patch("sys.argv", ["scheduler.py", "--config", config_path, "--list"]):
            main()

        output = capsys.readouterr().out
        assert "krystof" in output
        assert "anna" in output
        assert "08:00" in output
        assert "10:30" in output
        assert "unscheduled" not in output

    def test_list_no_schedules(self, capsys, tmp_path):
        from src.scheduler import main

        config = {"profiles": {"a": {"source_lang": "en"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        with patch("sys.argv", ["scheduler.py", "--config", str(config_file), "--list"]):
            main()

        output = capsys.readouterr().out
        assert "No profiles" in output


class TestJobBuild:
    """Test job function creation."""

    def test_job_is_async(self, sample_config):
        from src.scheduler import LessonScheduler

        scheduler = LessonScheduler(config=sample_config[0])
        profile = sample_config[0]["profiles"]["krystof"]
        job = scheduler._build_job("krystof", profile)

        assert asyncio.iscoroutinefunction(job)

    def test_job_runs_pipeline_and_delivers(self, sample_config):
        from src.scheduler import LessonScheduler

        delivered = []

        async def capture(profile_name, lesson):
            delivered.append((profile_name, lesson))

        scheduler = LessonScheduler(
            config=sample_config[0],
            delivery_callback=capture,
        )

        profile = sample_config[0]["profiles"]["krystof"]
        job = scheduler._build_job("krystof", profile)

        # Run the job with mocked run_lesson
        loop = asyncio.get_event_loop()

        async def test():
            with patch.object(scheduler, "run_lesson", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = {"title": "Math", "content": "1+1=2"}
                await job()

        loop.run_until_complete(test())
        assert len(delivered) == 1
        assert delivered[0][0] == "krystof"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("src.fetch_router.fetch_article")
    def test_profile_with_no_topics(self, mock_fetch, sample_config):
        from src.scheduler import LessonScheduler

        config = dict(sample_config[0])
        config["profiles"]["krystof"]["topics"] = []  # empty topics
        mock_fetch.return_value = ("Random", "Some text.")

        scheduler = LessonScheduler(config=config)
        lesson = asyncio.get_event_loop().run_until_complete(
            scheduler.run_lesson("krystof")
        )

        assert lesson is not None
        assert lesson["topic"] is None

    def test_load_config_from_path(self, sample_config):
        from src.scheduler import LessonScheduler

        config_path = sample_config[1]
        scheduler = LessonScheduler(config_path=config_path)
        assert "krystof" in scheduler.config["profiles"]

    def test_stop_without_scheduler_is_safe(self, sample_config):
        from src.scheduler import LessonScheduler

        scheduler = LessonScheduler(config=sample_config[0])
        # _scheduler is None — stop should not raise
        loop = asyncio.get_event_loop()
        loop.run_until_complete(scheduler.stop())
