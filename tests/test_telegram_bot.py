"""Tests for src/telegram_bot.py — config, mapping, history DB, tutor routing."""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary config with Telegram + profile settings."""
    config = {
        "telegram": {
            "bot_token": "123456:TEST-TOKEN",
        },
        "llm": {
            "base_url": "http://localhost:8080/v1",
            "default_model": "gemma4-26b",
        },
        "profiles": {
            "krystof": {
                "source_lang": "en",
                "target_lang": "de",
                "target_lang_name": "German",
                "telegram_chat_id": 111222333,
                "schedule": {
                    "time": "08:00",
                    "tz": "Europe/Berlin",
                },
                "use_tts": True,
            },
            "anna": {
                "source_lang": "en",
                "target_lang": "es",
                "target_lang_name": "Spanish",
                "telegram_chat_id": 444555666,
                "schedule": {
                    "time": "10:00",
                    "tz": "Europe/Madrid",
                },
            },
            "unregistered": {
                "source_lang": "en",
                "target_lang": "fr",
                "target_lang_name": "French",
                # no telegram_chat_id
            },
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config, str(config_file)


@pytest.fixture
def mock_aiogram():
    """Patch aiogram.Bot to avoid needing a real Telegram connection."""
    with patch("aiogram.Bot") as MockBot:
        bot_instance = AsyncMock()
        bot_instance.session.close = AsyncMock()
        MockBot.return_value = bot_instance
        yield bot_instance


# ── ChatHistoryDB tests ─────────────────────────────────────────────

class TestChatHistoryDB:
    """Test SQLite conversation history storage."""

    def test_init_creates_tables(self, tmp_path):
        from src.telegram_bot import ChatHistoryDB
        db_path = str(tmp_path / "test_history.db")
        db = ChatHistoryDB(db_path)

        # Verify table exists
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'"
        )
        assert cursor.fetchone() is not None
        db.close()

    def test_add_and_get_history(self, tmp_path):
        from src.telegram_bot import ChatHistoryDB
        db = ChatHistoryDB(str(tmp_path / "test.db"))

        db.add_message("user123", "krystof", "user", "What is Hallo?")
        db.add_message("user123", "krystof", "assistant", "Hallo means hello.")

        history = db.get_history("user123", "krystof")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "What is Hallo?"}
        assert history[1] == {"role": "assistant", "content": "Hallo means hello."}

        db.close()

    def test_history_isolation_between_users(self, tmp_path):
        from src.telegram_bot import ChatHistoryDB
        db = ChatHistoryDB(str(tmp_path / "test.db"))

        db.add_message("user123", "krystof", "user", "msg from user1")
        db.add_message("user456", "krystof", "user", "msg from user2")

        h1 = db.get_history("user123", "krystof")
        h2 = db.get_history("user456", "krystof")

        assert len(h1) == 1
        assert h1[0]["content"] == "msg from user1"
        assert len(h2) == 1
        assert h2[0]["content"] == "msg from user2"
        db.close()

    def test_max_turns_limits_history(self, tmp_path):
        from src.telegram_bot import ChatHistoryDB
        db = ChatHistoryDB(str(tmp_path / "test.db"))

        for i in range(20):
            db.add_message("user1", "krystof", "user", f"q{i}")
            db.add_message("user1", "krystof", "assistant", f"a{i}")

        history = db.get_history("user1", "krystof", max_turns=3)
        # 3 turns = 6 messages (most recent)
        assert len(history) == 6
        # First should be most recent of the kept set
        assert history[0]["content"] == "q17"
        db.close()

    def test_clear_history(self, tmp_path):
        from src.telegram_bot import ChatHistoryDB
        db = ChatHistoryDB(str(tmp_path / "test.db"))

        db.add_message("user1", "krystof", "user", "msg1")
        db.add_message("user1", "krystof", "assistant", "reply1")
        db.add_message("user2", "anna", "user", "msg2")

        db.clear_history("user1", "krystof")

        assert len(db.get_history("user1", "krystof")) == 0
        assert len(db.get_history("user2", "anna")) == 1
        db.close()


# ── TelegramBot init & mapping tests ────────────────────────────────

class TestTelegramBotInit:
    """Test bot initialization and profile mapping."""

    def test_init_loads_mapping(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        assert bot.chat_id_to_profile[111222333] == "krystof"
        assert bot.chat_id_to_profile[444555666] == "anna"
        # unregistered has no chat_id
        assert len(bot.chat_id_to_profile) == 2
        bot.db.close()

    def test_init_no_telegram_config(self):
        """Bot should handle missing telegram config gracefully."""
        from src.telegram_bot import TelegramBot
        config = {"profiles": {}}
        bot = TelegramBot(config=config)
        assert bot.bot_token == ""
        assert bot.chat_id_to_profile == {}
        bot.db.close()

    def test_resolve_profile(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        assert bot.resolve_profile(111222333) == "krystof"
        assert bot.resolve_profile(999999999) is None
        bot.db.close()

    def test_register_user(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        bot.register_user(777888999, "unregistered")
        assert bot.resolve_profile(777888999) == "unregistered"
        assert bot.profile_to_chat_id["unregistered"] == 777888999
        bot.db.close()

    def test_register_overwrites_existing(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # krystof was mapped to 111222333
        bot.register_user(777888999, "krystof")
        assert bot.profile_to_chat_id["krystof"] == 777888999
        bot.db.close()


# ── Lesson delivery tests ───────────────────────────────────────────

class TestDeliverLesson:
    """Test lesson delivery to Telegram."""

    @pytest.mark.asyncio
    async def test_deliver_lesson_sends_text(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        lesson = {
            "title": "Python Basics",
            "content": "Python is a programming language.",
            "original_content": "Das ist Python.",
            "source_lang": "en",
            "target_lang_name": "German",
            "content_lang": "de",
        }

        await bot.deliver_lesson("krystof", lesson)

        # Should have called send_message twice: original + translation
        assert mock_aiogram.send_message.call_count == 2
        calls = mock_aiogram.send_message.call_args_list

        # Message 1: original text
        msg1_kwargs = calls[0][1]
        assert msg1_kwargs["chat_id"] == 111222333
        assert "Python Basics" in msg1_kwargs["text"]
        assert "Original" in msg1_kwargs["text"]

        # Message 2: translation + vocabulary
        msg2_kwargs = calls[1][1]
        assert msg2_kwargs["chat_id"] == 111222333
        assert "Translation" in msg2_kwargs["text"]
        assert "German" in msg2_kwargs["text"]

        bot.db.close()

    @pytest.mark.asyncio
    async def test_deliver_lesson_no_chat_id(self, sample_config, mock_aiogram, caplog):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        lesson = {"title": "Test", "content": "content"}
        await bot.deliver_lesson("unregistered", lesson)

        mock_aiogram.send_message.assert_not_called()
        assert "No Telegram chat_id" in caplog.text or "skipping delivery" in caplog.text
        bot.db.close()

    @pytest.mark.asyncio
    async def test_deliver_lesson_truncates_long_content(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        lesson = {
            "title": "Long Article",
            "content": "A" * 5000,
            "original_content": "B" * 5000,
            "source_lang": "en",
            "target_lang_name": "German",
            "content_lang": "de",
        }

        await bot.deliver_lesson("krystof", lesson)

        assert mock_aiogram.send_message.call_count == 2
        # Both messages should be under Telegram limit
        for call in mock_aiogram.send_message.call_args_list:
            text = call[1]["text"]
            assert len(text) <= 4096
            assert "\u2026" in text or "..." in text  # truncated
        bot.db.close()


# ── Tutor chat tests ────────────────────────────────────────────────

class TestTutorChat:
    """Test tutor message routing."""

    @pytest.mark.asyncio
    async def test_tutor_chat_routes_to_llm(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Mock LlamaClient.tutor_chat
        with patch.object(bot, "_get_llama_client") as mock_get:
            mock_client = MagicMock()
            mock_client.tutor_chat.return_value = "Hallo means hello in German."
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "What does Hallo mean?")

            # Should call tutor_chat with correct profile settings
            mock_client.tutor_chat.assert_called_once()
            call_kwargs = mock_client.tutor_chat.call_args[1]
            assert call_kwargs["message"] == "What does Hallo mean?"
            assert call_kwargs["language_name"] == "German"
            assert call_kwargs["native_lang"] == "en"

            # Should send reply on Telegram
            mock_aiogram.send_message.assert_called_once()
            sent = mock_aiogram.send_message.call_args[1]["text"]
            assert "Hallo means hello" in sent

        bot.db.close()

    @pytest.mark.asyncio
    async def test_tutor_chat_unregistered_user(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_tutor_message(999999999, "Hello!")

        mock_aiogram.send_message.assert_called_once()
        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "not registered" in sent.lower() or "register" in sent.lower()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_tutor_chat_stores_history(self, sample_config, mock_aiogram, tmp_path):
        from src.telegram_bot import TelegramBot, ChatHistoryDB
        config = sample_config[0]
        bot = TelegramBot(config=config)
        # Use a temp DB to avoid cross-test pollution
        bot.db = ChatHistoryDB(str(tmp_path / "test.db"))

        with patch.object(bot, "_get_llama_client") as mock_get:
            mock_client = MagicMock()
            mock_client.tutor_chat.return_value = "Great question!"
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "What is Konjunktiv?")

            # Verify history was stored
            history = bot.db.get_history(111222333, "krystof")
            assert len(history) == 2
            assert history[0] == {"role": "user", "content": "What is Konjunktiv?"}
            assert history[1] == {"role": "assistant", "content": "Great question!"}

        bot.db.close()

    @pytest.mark.asyncio
    async def test_tutor_chat_truncates_long_reply(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        with patch.object(bot, "_get_llama_client") as mock_get:
            mock_client = MagicMock()
            mock_client.tutor_chat.return_value = "A" * 5000
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "Tell me everything about German grammar.")

            sent = mock_aiogram.send_message.call_args[1]["text"]
            assert len(sent) <= 4000
            assert sent.endswith("...")

        bot.db.close()

    @pytest.mark.asyncio
    async def test_tutor_chat_fallback_on_llm_error(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        with patch.object(bot, "_get_llama_client") as mock_get:
            mock_client = MagicMock()
            mock_client.tutor_chat.return_value = None  # LLM failure
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "Hello tutor")

            sent = mock_aiogram.send_message.call_args[1]["text"]
            assert "unavailable" in sent.lower()

        bot.db.close()


# ── Command handler tests ───────────────────────────────────────────

class TestCommands:
    """Test Telegram command handlers."""

    @pytest.mark.asyncio
    async def test_command_start_registered(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_start(111222333)

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "krystof" in sent
        assert "German" in sent
        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_start_unregistered(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_start(999999999)

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "not registered" in sent.lower() or "register" in sent.lower()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_register_success(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_register(777888999, "unregistered")

        assert bot.resolve_profile(777888999) == "unregistered"
        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "Registered" in sent or "✅" in sent
        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_register_bad_profile(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_register(777888999, "nonexistent")

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "not found" in sent.lower()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_register_no_arg(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_register(777888999, "")

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "Usage" in sent or "usage" in sent.lower()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_history_clear(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Add some history first
        bot.db.add_message(111222333, "krystof", "user", "test")

        await bot.handle_history_clear(111222333)

        assert len(bot.db.get_history(111222333, "krystof")) == 0
        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "cleared" in sent.lower()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_status(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_status(111222333)

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "krystof" in sent
        assert "German" in sent
        assert "08:00" in sent
        assert "Europe/Berlin" in sent
        bot.db.close()


# ── Bot lifecycle tests ─────────────────────────────────────────────

class TestBotLifecycle:
    """Test bot start/stop and edge cases."""

    @pytest.mark.asyncio
    async def test_stop_closes_resources(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.stop()

        # DB should be closed (conn.closed would raise on use)
        with pytest.raises(Exception):  # sqlite3 exception on closed db
            bot.db.get_history("x", "y")

    @pytest.mark.asyncio
    async def test_no_bot_token(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        del config["telegram"]  # no token
        bot = TelegramBot(config=config)
        assert bot.bot_token == ""
        bot.db.close()

    @pytest.mark.asyncio
    async def test_deliver_lesson_no_audio(self, sample_config, mock_aiogram):
        """Lesson without wav_path should still deliver text (2 messages)."""
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        lesson = {
            "title": "No Audio",
            "content": "Just text.",
            "original_content": "Original text.",
            "source_lang": "en",
            "target_lang_name": "German",
            "content_lang": "de",
            # no wav_path
        }

        await bot.deliver_lesson("krystof", lesson)
        # Two text messages: original + translation (no audio call)
        assert mock_aiogram.send_message.call_count == 2
        bot.db.close()


# ── Environment variable fallback ──────────────────────────────────

class TestEnvFallback:
    """Test environment variable configuration."""

    def test_telegram_token_from_env(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        del config["telegram"]  # no telegram section

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token-123"}):
            bot = TelegramBot(config=config)
            assert bot.bot_token == "env-token-123"
            bot.db.close()
