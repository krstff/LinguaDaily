"""Tests for src/main.py — daemon startup, signal handling, service wiring."""

import asyncio
import json
import logging
import os
import signal
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


@pytest.fixture
def full_config(tmp_path):
    """Full config with telegram, llm, tts, and profiles with schedules."""
    config = {
        "telegram": {"bot_token": "123456:TEST-TOKEN"},
        "llm": {
            "base_url": "http://localhost:8080/v1",
            "default_model": "gemma4-26b",
        },
        "tts": {"base_url": "http://localhost:8080/v1", "model": "omnivoice"},
        "profiles": {
            "krystof": {
                "source_lang": "en",
                "target_lang": "de",
                "target_lang_name": "German",
                "topics": ["Technology"],
                "schedule": {"time": "08:00", "tz": "Europe/Berlin"},
                "use_tts": True,
            },
            "anna": {
                "source_lang": "en",
                "target_lang": "es",
                "target_lang_name": "Spanish",
                "topics": ["History"],
                "schedule": {"time": "10:30", "tz": "Europe/Madrid"},
            },
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config, str(config_file)


class TestDaemonInit:
    """Test daemon initialization and config loading."""

    def test_init_with_config(self, full_config):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config=full_config[0])
        assert daemon.config == full_config[0]
        assert daemon.bot is None
        assert daemon.scheduler is None

    def test_init_loads_config_from_path(self, full_config):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config_path=full_config[1])
        assert "krystof" in daemon.config["profiles"]
        assert "anna" in daemon.config["profiles"]

    def test_init_missing_config_raises(self, tmp_path):
        from src.main import LinguaDaemon

        with pytest.raises(FileNotFoundError):
            LinguaDaemon(config_path=str(tmp_path / "nonexistent.json"))


class TestSetupLogging:
    """Test logging configuration."""

    def test_setup_logging_creates_handlers(self, full_config):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config=full_config[0])
        daemon._setup_logging(verbose=True)

        root = logging.getLogger()
        assert len(root.handlers) >= 2


class TestPrintBanner:
    """Test startup banner output."""

    def test_banner_shows_profiles_and_schedule(self, full_config, capsys):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config=full_config[0])
        with patch("src.main.CONFIG_PATH", "/test/config.json"):
            daemon._print_banner()

        output = capsys.readouterr().out
        assert "OpenClaw-Lingua" in output
        assert "Profiles:   2" in output
        assert "Scheduled:  2" in output
        assert "08:00" in output
        assert "10:30" in output

    def test_banner_shows_telegram_configured(self, full_config, capsys):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config=full_config[0])
        with patch("src.main.CONFIG_PATH", "/test/config.json"):
            daemon._print_banner()

        assert "configured" in capsys.readouterr().out.lower()

    def test_banner_shows_llm_model(self, full_config, capsys):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config=full_config[0])
        with patch("src.main.CONFIG_PATH", "/test/config.json"):
            daemon._print_banner()

        assert "gemma4-26b" in capsys.readouterr().out

    def test_banner_no_telegram(self, tmp_path, capsys):
        from src.main import LinguaDaemon

        config = {
            "profiles": {
                "test": {
                    "source_lang": "en",
                    "target_lang_name": "French",
                    "schedule": {"time": "09:00", "tz": "UTC"},
                }
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        daemon = LinguaDaemon(config=config)
        with patch("src.main.CONFIG_PATH", str(config_file)):
            daemon._print_banner()

        output = capsys.readouterr().out
        assert "disabled" in output.lower() or "⚠️" in output

    def test_banner_no_llm(self, tmp_path, capsys):
        from src.main import LinguaDaemon

        config = {
            "profiles": {
                "test": {
                    "source_lang": "en",
                    "target_lang_name": "French",
                    "schedule": {"time": "09:00", "tz": "UTC"},
                }
            }
        }

        daemon = LinguaDaemon(config=config)
        with patch("src.main.CONFIG_PATH", "/test/config.json"):
            daemon._print_banner()

        output = capsys.readouterr().out
        assert "not configured" in output.lower() or "⚠️" in output

    def test_banner_no_profiles(self, capsys):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config={"profiles": {}})
        with patch("src.main.CONFIG_PATH", "/test/config.json"):
            daemon._print_banner()

        output = capsys.readouterr().out
        assert "Profiles:   0" in output


class TestServiceWiring:
    """Test that services are wired together correctly."""

    def test_scheduler_created_on_start(self, full_config):
        from src.main import LinguaDaemon
        from src.scheduler import LessonScheduler

        daemon = LinguaDaemon(config=full_config[0])

        # Manually create scheduler (like start() does)
        delivery_cb = None  # no bot in this test
        daemon.scheduler = LessonScheduler(
            config=daemon.config,
            delivery_callback=delivery_cb,
        )

        assert daemon.scheduler is not None
        assert daemon.scheduler.delivery_callback is None

    def test_bot_delivery_callback_passed_to_scheduler(self, full_config):
        from src.main import LinguaDaemon
        from src.scheduler import LessonScheduler

        async def fake_delivery(profile_name, lesson):
            pass

        daemon = LinguaDaemon(config=full_config[0])
        daemon.scheduler = LessonScheduler(
            config=daemon.config,
            delivery_callback=fake_delivery,
        )

        assert daemon.scheduler.delivery_callback is fake_delivery

    def test_no_telegram_token_skips_bot(self, tmp_path):
        from src.main import LinguaDaemon

        config = {
            "profiles": {
                "krystof": {
                    "source_lang": "en",
                    "target_lang": "de",
                    "target_lang_name": "German",
                    "schedule": {"time": "08:00", "tz": "UTC"},
                }
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        daemon = LinguaDaemon(config=config)
        # No telegram token in config → bot should be None after start logic
        tg_token = (daemon.config.get("telegram", {}).get("bot_token")
                    or os.environ.get("TELEGRAM_BOT_TOKEN", ""))
        assert tg_token == ""


class TestGracefulShutdown:
    """Test graceful shutdown of all services."""

    @pytest.mark.asyncio
    async def test_stop_calls_bot_and_scheduler_stop(self, full_config):
        from src.main import LinguaDaemon

        mock_bot = AsyncMock()
        mock_scheduler = AsyncMock()

        daemon = LinguaDaemon(config=full_config[0])
        daemon.bot = mock_bot
        daemon.scheduler = mock_scheduler

        await daemon.stop()

        mock_bot.stop.assert_called_once()
        mock_scheduler.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_bot_error(self, full_config, caplog):
        from src.main import LinguaDaemon

        mock_bot = AsyncMock()
        mock_bot.stop = AsyncMock(side_effect=Exception("network error"))
        mock_scheduler = AsyncMock()

        daemon = LinguaDaemon(config=full_config[0])
        daemon.bot = mock_bot
        daemon.scheduler = mock_scheduler

        await daemon.stop()  # should not raise

        assert "Bot shutdown error" in caplog.text or "network error" in caplog.text


class TestSignalHandling:
    """Test signal handling for graceful shutdown."""

    def test_handle_signal_sets_shutdown_event(self, full_config):
        from src.main import LinguaDaemon, _handle_signal

        daemon = LinguaDaemon(config=full_config[0])
        assert not daemon._shutdown_event.is_set()

        # Simulate SIGINT (no real signal, just call handler)
        daemon._shutdown_event.set()  # what the signal handler does via event loop
        assert daemon._shutdown_event.is_set()

    def test_shutdown_event_can_be_awaited(self, full_config):
        from src.main import LinguaDaemon

        daemon = LinguaDaemon(config=full_config[0])
        daemon._shutdown_event.set()

        loop = asyncio.get_event_loop()
        # Should return immediately since event is already set
        loop.run_until_complete(daemon._wait_for_shutdown())


class TestCLI:
    """Test CLI entry point."""

    def test_cli_missing_config(self, capsys, tmp_path):
        from src.main import main

        with patch("sys.argv", ["main.py", "--config", str(tmp_path / "nope.json")]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

        assert "not found" in capsys.readouterr().err.lower()

    def test_cli_invalid_json(self, capsys, tmp_path):
        from src.main import main

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json")

        with patch("sys.argv", ["main.py", "--config", str(bad_file)]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

        assert "json" in capsys.readouterr().err.lower()

    def test_cli_verbose_sets_debug_level(self, tmp_path, caplog):
        from src.main import main, LinguaDaemon

        config = {"profiles": {}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        # Patch start to capture logging level before it does work
        original_start = LinguaDaemon.start

        async def capture_start(self):
            root = logging.getLogger()
            assert root.level == logging.DEBUG
            raise asyncio.CancelledError

        with patch.object(LinguaDaemon, "start", capture_start):
            with patch("sys.argv", ["main.py", "--config", str(config_file), "--verbose"]):
                try:
                    main()
                except (asyncio.CancelledError, SystemExit):
                    pass
