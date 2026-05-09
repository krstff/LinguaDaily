"""Tests for src/scheduler.py — scheduling, job building, CLI."""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


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
        profile = sample_config[0]["profiles"]["krystof"]
        job = scheduler._build_job("krystof", profile)

        mock_lesson = {
            "title": "Test",
            "content": "Translated.",
            "profile": "krystof",
            "word_count": 10,
            "vocab": [],
        }

        async def fake_run_lesson(pname, delivery_callback=None):
            # Simulate Orchestrator.run_lesson calling the delivery callback
            if delivery_callback:
                await delivery_callback(pname, mock_lesson)
            return mock_lesson

        with patch("orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_lesson = fake_run_lesson
            MockOrch.return_value = mock_orch
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

        profile = sample_config[0]["profiles"]["krystof"]
        job = scheduler._build_job("krystof", profile)

        mock_lesson = {
            "title": "Test",
            "content": "text",
            "profile": "krystof",
            "word_count": 10,
            "vocab": [],
        }

        async def fake_run_lesson(pname, delivery_callback=None):
            try:
                if delivery_callback:
                    await delivery_callback(pname, mock_lesson)
            except Exception as e:
                # Orchestrator.run_lesson catches delivery errors internally
                import logging
                logger = logging.getLogger("src.scheduler")
                logger.error("[%s] Delivery failed: %s", pname, e)
            return mock_lesson

        with patch("orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_lesson = fake_run_lesson
            MockOrch.return_value = mock_orch
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

    @pytest.mark.asyncio
    async def test_job_runs_orchestrator_and_delivers(self, sample_config):
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

        mock_lesson = {
            "title": "Math",
            "content": "1+1=2",
            "profile": "krystof",
            "word_count": 5,
            "vocab": [],
        }

        async def fake_run_lesson(pname, delivery_callback=None):
            if delivery_callback:
                await delivery_callback(pname, mock_lesson)
            return mock_lesson

        with patch("orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_lesson = fake_run_lesson
            MockOrch.return_value = mock_orch
            await job()

        assert len(delivered) == 1
        assert delivered[0][0] == "krystof"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_load_config_from_path(self, sample_config):
        from src.scheduler import LessonScheduler

        config_path = sample_config[1]
        scheduler = LessonScheduler(config_path=config_path)
        assert "krystof" in scheduler.config["profiles"]

    @pytest.mark.asyncio
    async def test_stop_without_scheduler_is_safe(self, sample_config):
        from src.scheduler import LessonScheduler

        scheduler = LessonScheduler(config=sample_config[0])
        # _scheduler is None — stop should not raise
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_orchestrator_failure_logs_error(self, sample_config, caplog):
        from src.scheduler import LessonScheduler

        scheduler = LessonScheduler(config=sample_config[0])
        profile = sample_config[0]["profiles"]["krystof"]
        job = scheduler._build_job("krystof", profile)

        with patch("orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_lesson = AsyncMock(return_value=None)
            MockOrch.return_value = mock_orch
            await job()

        assert "returned no result" in caplog.text


class TestSchedulerConfig:
    """Test scheduler configuration handling."""

    def test_init_with_config(self, sample_config):
        from src.scheduler import LessonScheduler
        scheduler = LessonScheduler(config=sample_config[0])
        assert "krystof" in scheduler.config["profiles"]

    def test_init_loads_from_path(self, sample_config):
        from src.scheduler import LessonScheduler
        _, config_path = sample_config
        scheduler = LessonScheduler(config_path=config_path)
        assert "anna" in scheduler.config["profiles"]

    def test_delivery_callback_stored(self, sample_config):
        from src.scheduler import LessonScheduler

        async def callback(profile_name, lesson):
            pass

        scheduler = LessonScheduler(
            config=sample_config[0],
            delivery_callback=callback,
        )
        assert scheduler.delivery_callback is callback
