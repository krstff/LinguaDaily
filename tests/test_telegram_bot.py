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
                "learning_language": "de",
                "native_language": "en",
                "telegram_chat_id": 111222333,
                "schedule": {
                    "time": "08:00",
                    "tz": "Europe/Berlin",
                },
                "use_tts": True,
            },
            "anna": {
                "learning_language": "es",
                "native_language": "en",
                "telegram_chat_id": 444555666,
                "schedule": {
                    "time": "10:00",
                    "tz": "Europe/Madrid",
                },
            },
            "unregistered": {
                "learning_language": "fr",
                "native_language": "en",
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

        # Verify tables exist
        for table in ("chat_history", "latest_lesson", "lesson_log"):
            cursor = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            assert cursor.fetchone() is not None, f"missing table {table}"
        db.close()

    def test_lesson_log_lifecycle(self, tmp_path):
        from src.telegram_bot import ChatHistoryDB
        db = ChatHistoryDB(str(tmp_path / "test.db"))

        lid = db.log_lesson("krystof", "Some Title")
        assert lid >= 1

        # First marking wins, second is a no-op (idempotent)
        assert db.mark_lesson_finished(lid, 12345) is True
        assert db.mark_lesson_finished(lid, 99999) is False

        row = db.conn.execute(
            "SELECT profile, title, delivered_at, finished_at, finished_by "
            "FROM lesson_log WHERE id=?",
            (lid,),
        ).fetchone()
        assert row[0] == "krystof"
        assert row[1] == "Some Title"
        assert row[2] is not None
        assert row[3] is not None
        assert row[4] == "12345"
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

        assert "krystof" in bot.chat_id_to_profiles[111222333]
        assert "anna" in bot.chat_id_to_profiles[444555666]
        # unregistered has no chat_id
        assert len(bot.chat_id_to_profiles) == 2
        bot.db.close()

    def test_init_no_telegram_config(self):
        """Bot should handle missing telegram config gracefully."""
        from src.telegram_bot import TelegramBot
        config = {"profiles": {}}
        bot = TelegramBot(config=config)
        assert bot.bot_token == ""
        assert bot.chat_id_to_profiles == {}
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
            "learning_language_name": "German",
            "native_language": "English",
            "vocab": [],
        }

        await bot.deliver_lesson("krystof", lesson)

        # Should have called send_message 3x: original + translation + ack
        assert mock_aiogram.send_message.call_count == 3
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
        assert "English" in msg2_kwargs["text"]

        # Message 3: lesson-ack — plain text + button (no effect; the
        # streak effect is reserved for the post-click confirmation)
        msg3_kwargs = calls[2][1]
        assert msg3_kwargs["chat_id"] == 111222333
        assert "message_effect_id" not in msg3_kwargs
        button = msg3_kwargs["reply_markup"].inline_keyboard[0][0]
        assert button.text == "✅ Finished"
        assert button.callback_data.startswith("ld:krystof:")

        # A lesson_log row was recorded
        row = bot.db.conn.execute(
            "SELECT COUNT(*) FROM lesson_log WHERE profile='krystof' AND title='Python Basics'"
        ).fetchone()
        assert row[0] >= 1

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
            "learning_language_name": "German",
            "native_language": "English",
            "vocab": [],
        }

        await bot.deliver_lesson("krystof", lesson)

        # original + translation + short ack message
        assert mock_aiogram.send_message.call_count == 3
        # Long messages should be under Telegram limit and truncated
        for call in mock_aiogram.send_message.call_args_list[:2]:
            text = call[1]["text"]
            assert len(text) <= 4096
            assert "\u2026" in text or "..." in text  # truncated
        bot.db.close()


# ── Tutor chat tests ────────────────────────────────────────────────

async def _await_tutor_task(bot, chat_id):
    """Wait for the background tutor reply task to finish."""
    entry = bot._in_flight.get(chat_id)
    assert entry is not None, "no tutor task registered"
    await entry["task"]
    assert chat_id not in bot._in_flight, "task did not clean up _in_flight"


class TestTutorChat:
    """Test tutor message routing."""

    @pytest.mark.asyncio
    async def test_tutor_chat_routes_to_llm(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Mock LlamaClient.tutor_chat_stream
        with patch.object(bot, "_get_llama_client") as mock_get:
            mock_client = MagicMock()
            mock_client.tutor_chat_stream = AsyncMock(
                return_value="Hallo means hello in German.")
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "What does Hallo mean?")
            await _await_tutor_task(bot, 111222333)

            # Should call tutor_chat_stream with correct profile settings
            mock_client.tutor_chat_stream.assert_awaited_once()
            call_kwargs = mock_client.tutor_chat_stream.call_args[1]
            assert call_kwargs["message"] == "What does Hallo mean?"
            assert call_kwargs["language_name"] == "German"
            assert call_kwargs["native_lang"] == "en"

            # Should post the reply on Telegram (edit of the Thinking… msg)
            sent = (mock_aiogram.edit_message_text.call_args[1]["text"]
                    if mock_aiogram.edit_message_text.called
                    else mock_aiogram.send_message.call_args[1]["text"])
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
            mock_client.tutor_chat_stream = AsyncMock(return_value="Great question!")
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "What is Konjunktiv?")
            await _await_tutor_task(bot, 111222333)

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
            mock_client.tutor_chat_stream = AsyncMock(return_value="A" * 5000)
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "Tell me everything about German grammar.")
            await _await_tutor_task(bot, 111222333)

            # With telegramify-markdown: truncated to 4096, sent with entities
            call = mock_aiogram.edit_message_text.call_args \
                if mock_aiogram.edit_message_text.called \
                else mock_aiogram.send_message.call_args
            sent = call[1]["text"]
            assert len(sent) <= 4096
            assert sent.endswith("...")
            # Should use entities parameter (not parse_mode)
            call_kwargs = call[1]
            assert "entities" in call_kwargs or not call_kwargs.get("parse_mode")

        bot.db.close()

    @pytest.mark.asyncio
    async def test_tutor_chat_fallback_on_llm_error(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        with patch.object(bot, "_get_llama_client") as mock_get:
            mock_client = MagicMock()
            mock_client.tutor_chat_stream = AsyncMock(return_value=None)  # LLM failure
            mock_get.return_value = mock_client

            await bot.handle_tutor_message(111222333, "Hello tutor")
            await _await_tutor_task(bot, 111222333)

            sent = (mock_aiogram.edit_message_text.call_args[1]["text"]
                    if mock_aiogram.edit_message_text.called
                    else mock_aiogram.send_message.call_args[1]["text"])
            assert "unavailable" in sent.lower()

        bot.db.close()

    @pytest.mark.asyncio
    async def test_tutor_chat_rejected_while_in_flight(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Simulate a reply still being generated
        bot._in_flight[111222333] = {
            "task": asyncio.get_running_loop().create_task(asyncio.sleep(10)),
            "message_id": 42,
        }

        await bot.handle_tutor_message(111222333, "another question")

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "still working" in sent
        assert "/stop" in sent
        bot._in_flight[111222333]["task"].cancel()
        bot.db.close()


# ── /stop command tests ─────────────────────────────────────────────

class TestStopCommand:
    """Test the /stop cancel command."""

    @pytest.mark.asyncio
    async def test_stop_nothing_in_flight(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_stop(111222333)

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "Nothing to stop" in sent
        bot.db.close()

    @pytest.mark.asyncio
    async def test_stop_cancels_task_and_deletes_placeholder(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        loop = asyncio.get_running_loop()
        task = loop.create_task(asyncio.sleep(10))
        bot._in_flight[111222333] = {"task": task, "message_id": 42}

        await bot.handle_stop(111222333)

        # Task cancelled and placeholder removed
        assert task.cancelling() or task.cancelled()
        mock_aiogram.delete_message.assert_awaited_once()
        delete_args = mock_aiogram.delete_message.call_args
        assert delete_args[0][:2] == (111222333, 42) or \
            delete_args[1] == {"chat_id": 111222333, "message_id": 42}
        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "Stopped" in sent

        task.cancel()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_stop_twice_sends_nothing_to_stop(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        loop = asyncio.get_running_loop()
        bot._in_flight[111222333] = {
            "task": loop.create_task(asyncio.sleep(10)), "message_id": 42,
        }

        await bot.handle_stop(111222333)
        mock_aiogram.send_message.reset_mock()
        await bot.handle_stop(111222333)

        # Second /stop while cancellation is in progress → not "Stopped" again
        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "Nothing to stop" in sent
        bot._in_flight[111222333]["task"].cancel()
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
    async def test_register_user_success(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        bot.register_user(777888999, "unregistered")

        assert bot.resolve_profile(777888999) == "unregistered"
        bot.db.close()

    @pytest.mark.asyncio
    async def test_register_user_multiple_profiles(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Register a second profile for an existing chat
        bot.register_user(111222333, "anna")

        cid_profiles = bot.chat_id_to_profiles.get(111222333, [])
        assert "krystof" in cid_profiles
        assert "anna" in cid_profiles
        bot.db.close()

    @pytest.mark.asyncio
    async def test_profiles_menu_has_switch_buttons(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # One chat with two profiles
        bot.register_user(111222333, "anna")

        await bot.handle_profiles(111222333)

        kwargs = mock_aiogram.send_message.call_args[1]
        assert "Your profiles" in kwargs["text"]
        markup = kwargs["reply_markup"]
        buttons = markup.inline_keyboard
        assert len(buttons) == 2
        callbacks = [b.callback_data for row in buttons for b in row]
        assert "sw:111222333:krystof" in callbacks
        assert "sw:111222333:anna" in callbacks
        bot.db.close()

    @pytest.mark.asyncio
    async def test_profiles_menu_single_profile_no_buttons(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Chat with only one profile — menu is sent as plain text
        await bot.handle_profiles(111222333)

        kwargs = mock_aiogram.send_message.call_args[1]
        markup = kwargs.get("reply_markup")
        assert markup is None or not markup.inline_keyboard
        bot.db.close()

    @pytest.mark.asyncio
    async def test_switch_callback_switches_and_deletes_menu(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        bot.register_user(111222333, "anna")
        assert bot.resolve_profile(111222333) == "krystof"

        menu_message = MagicMock()
        menu_message.delete = AsyncMock()

        ok = await bot.handle_switch_callback(111222333, "anna", menu_message)

        assert ok is True
        assert bot.resolve_profile(111222333) == "anna"
        menu_message.delete.assert_awaited_once()
        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "Switched to" in sent
        assert "anna" in sent
        bot.db.close()

    @pytest.mark.asyncio
    async def test_switch_callback_invalid_profile(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        menu_message = MagicMock()
        menu_message.delete = AsyncMock()

        ok = await bot.handle_switch_callback(111222333, "ghost", menu_message)

        assert ok is False
        # Profile unchanged, menu not deleted, no confirmation sent
        assert bot.resolve_profile(111222333) == "krystof"
        menu_message.delete.assert_not_awaited()
        mock_aiogram.send_message.assert_not_awaited()
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

    @pytest.mark.asyncio
    async def test_command_another_lesson_requests_lesson(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        with patch("orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_lesson = AsyncMock(return_value={"title": "Test"})
            MockOrch.return_value = mock_orch

            await bot.handle_another_lesson(111222333)

            # Confirmation message sent before pipeline starts
            sent = mock_aiogram.send_message.call_args[1]["text"]
            assert "Requesting a new lesson" in sent
            assert "krystof" in sent

            # Orchestrator called with correct profile and delivery callback
            mock_orch.run_lesson.assert_called_once()
            positional, kwargs = mock_orch.run_lesson.call_args
            assert positional[0] == "krystof"
            assert kwargs["delivery_callback"] == bot.deliver_lesson

        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_another_lesson_cooldown(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Manually set a recent timestamp to trigger cooldown
        import time
        bot._last_lesson_request[111222333] = time.time()

        with patch("orchestrator.Orchestrator") as MockOrch:
            await bot.handle_another_lesson(111222333)

            # Should have sent cooldown message, NOT started pipeline
            sent = mock_aiogram.send_message.call_args[1]["text"]
            assert "Please wait" in sent
            MockOrch.assert_not_called()

        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_another_lesson_unregistered(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        await bot.handle_another_lesson(999999999)

        sent = mock_aiogram.send_message.call_args[1]["text"]
        assert "not registered" in sent.lower() or "register" in sent.lower()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_command_another_lesson_cooldown_expires(self, sample_config, mock_aiogram):
        from src.telegram_bot import TelegramBot
        from src.config import TG_LESSON_COOLDOWN_SECS
        config = sample_config[0]
        bot = TelegramBot(config=config)

        # Set timestamp well before cooldown window
        import time
        bot._last_lesson_request[111222333] = time.time() - TG_LESSON_COOLDOWN_SECS - 1

        with patch("orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_lesson = AsyncMock(return_value={"title": "Test"})
            MockOrch.return_value = mock_orch

            await bot.handle_another_lesson(111222333)

            # Cooldown expired — should proceed
            mock_orch.run_lesson.assert_called_once()

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
        """Lesson without wav_path should still deliver text (3 messages)."""
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        lesson = {
            "title": "No Audio",
            "content": "Just text.",
            "original_content": "Original text.",
            "learning_language_name": "German",
            "native_language": "English",
            "vocab": [],
            # no wav_path
        }

        await bot.deliver_lesson("krystof", lesson)
        # Three text messages: original + translation + ack (no audio call)
        assert mock_aiogram.send_message.call_count == 3
        bot.db.close()

    @pytest.mark.asyncio
    async def test_deliver_lesson_ack_disabled(self, sample_config, mock_aiogram):
        """lesson_ack: false suppresses the ack message and DB log row."""
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        config["lesson_ack"] = False
        bot = TelegramBot(config=config)

        before = bot.db.conn.execute("SELECT COUNT(*) FROM lesson_log").fetchone()[0]
        lesson = {
            "title": "No Ack",
            "content": "content",
            "original_content": "original",
            "learning_language_name": "German",
            "native_language": "English",
            "vocab": [],
        }

        await bot.deliver_lesson("krystof", lesson)
        assert mock_aiogram.send_message.call_count == 2
        after = bot.db.conn.execute("SELECT COUNT(*) FROM lesson_log").fetchone()[0]
        assert after == before  # no log row written
        bot.db.close()

    @pytest.mark.asyncio
    async def test_deliver_lesson_ack_language(self, sample_config, mock_aiogram):
        """Ack text follows the profile's LEARNING language (krystof learns
        German, natively English) — not the native language."""
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)

        lesson = {
            "title": "Titel",
            "content": "content",
            "original_content": "original",
            "learning_language_name": "German",
            "native_language": "English",
            "vocab": [],
        }

        await bot.deliver_lesson("krystof", lesson)
        ack_text = mock_aiogram.send_message.call_args_list[2][1]["text"]
        assert "Klicke unten" in ack_text  # German, not English

        # Switching the learning language switches the ack text too
        config["profiles"]["krystof"]["learning_language"] = "cs"
        bot2 = TelegramBot(config=config)
        await bot2.deliver_lesson("krystof", lesson)
        ack_text = mock_aiogram.send_message.call_args_list[-1][1]["text"]
        assert "Klikni dole" in ack_text  # Czech
        bot.db.close()
        bot2.db.close()


# ── Lesson-ack click flow ───────────────────────────────────────────

class TestLessonAckDone:
    """Test the 'Finished' button post-click effect (edit + delete)."""

    @pytest.mark.asyncio
    async def test_handle_lesson_done_delete_send_delete(
        self, sample_config, mock_aiogram, monkeypatch
    ):
        """Click → delete ack, send confirmation (with effect), delete it."""
        import types as _types
        import src.telegram_bot as tb_mod
        from src.telegram_bot import TelegramBot
        monkeypatch.setattr(tb_mod, "TG_LESSON_ACK_DELETE_DELAY_SECS", 0)
        config = sample_config[0]
        bot = TelegramBot(config=config)

        class FakeMessage:
            def __init__(self, chat_id=111222333):
                self.chat = _types.SimpleNamespace(id=chat_id)
                self.edited = None
                self.deleted = False

            async def edit_text(self, text=None, reply_markup=None, **kw):
                self.edited = (text, reply_markup)

            async def delete(self):
                self.deleted = True

        old_msg = FakeMessage()
        new_msg = FakeMessage()
        mock_aiogram.send_message.return_value = new_msg
        # deterministic streak → heart tier
        monkeypatch.setattr(TelegramBot, "_lesson_streak_days",
                            lambda self, profile: 28)

        assert await bot.handle_lesson_done("krystof", old_msg) is True

        # old ack message was deleted, not edited
        assert old_msg.deleted is True
        assert old_msg.edited is None
        # confirmation sent as a NEW message: short text in the LEARNING
        # language (krystof learns German) with the streak effect
        kwargs = mock_aiogram.send_message.call_args[1]
        assert kwargs["chat_id"] == 111222333
        assert kwargs["text"] == "🎉 Gut gemacht!"
        # streak 28 → month-tier effect (value comes from config)
        assert kwargs["message_effect_id"] == tb_mod.LESSON_ACK_EFFECT_MONTH_STREAK
        # and deleted again after the delay
        assert new_msg.deleted is True
        bot.db.close()

    @pytest.mark.asyncio
    async def test_handle_lesson_done_delete_falls_back_to_edit(
        self, sample_config, mock_aiogram, monkeypatch
    ):
        """If the ack message can't be deleted (e.g. group permissions),
        fall back to editing it in place and skip the extra message."""
        import types as _types
        import src.telegram_bot as tb_mod
        from src.telegram_bot import TelegramBot
        monkeypatch.setattr(tb_mod, "TG_LESSON_ACK_DELETE_DELAY_SECS", 0)
        config = sample_config[0]
        bot = TelegramBot(config=config)

        class NoDeleteMessage:
            def __init__(self):
                self.chat = _types.SimpleNamespace(id=111222333)
                self.edited = None

            async def edit_text(self, text=None, reply_markup=None, **kw):
                self.edited = (text, reply_markup)

            async def delete(self):
                raise RuntimeError("delete not allowed")

        msg = NoDeleteMessage()
        assert await bot.handle_lesson_done("krystof", msg) is True

        text, markup = msg.edited
        assert "🎉" in text
        assert markup.inline_keyboard == []  # button removed
        mock_aiogram.send_message.assert_not_called()
        bot.db.close()

    @pytest.mark.asyncio
    async def test_handle_lesson_done_send_failure(
        self, sample_config, mock_aiogram, monkeypatch
    ):
        """Ack deleted but confirmation send fails → False, nothing to delete."""
        import types as _types
        import src.telegram_bot as tb_mod
        from src.telegram_bot import TelegramBot
        monkeypatch.setattr(tb_mod, "TG_LESSON_ACK_DELETE_DELAY_SECS", 0)
        config = sample_config[0]
        bot = TelegramBot(config=config)

        class FakeMessage:
            def __init__(self):
                self.chat = _types.SimpleNamespace(id=111222333)
                self.deleted = False

            async def delete(self):
                self.deleted = True

        msg = FakeMessage()
        mock_aiogram.send_message.side_effect = RuntimeError("telegram down")

        assert await bot.handle_lesson_done("krystof", msg) is False
        assert msg.deleted is True
        bot.db.close()

    @pytest.mark.asyncio
    async def test_handle_lesson_done_none_message(self, sample_config):
        from src.telegram_bot import TelegramBot
        config = sample_config[0]
        bot = TelegramBot(config=config)
        assert await bot.handle_lesson_done("krystof", None) is False
        bot.db.close()




# ── Streak-based ack effects ────────────────────────────────────────

class TestLessonAckEffect:
    """Test completion streak counting and effect tier selection."""

    @pytest.mark.parametrize(
        ("streak", "tier"),
        [
            (0, "default"),
            (1, "default"),
            (6, "default"),
            (7, "week"),
            (13, "default"),
            (14, "week"),
            (27, "default"),
            (28, "month"),
            (35, "week"),
            (56, "month"),
        ],
    )
    def test_effect_tiers(self, streak, tier):
        from src.config import (
            LESSON_ACK_EFFECT_DEFAULT,
            LESSON_ACK_EFFECT_MONTH_STREAK,
            LESSON_ACK_EFFECT_WEEK_STREAK,
        )
        from src.telegram_bot import TelegramBot
        expected = {
            "default": LESSON_ACK_EFFECT_DEFAULT,
            "week": LESSON_ACK_EFFECT_WEEK_STREAK,
            "month": LESSON_ACK_EFFECT_MONTH_STREAK,
        }[tier]
        assert TelegramBot._lesson_ack_effect(streak) == expected

    @staticmethod
    def _mark_day(db, profile, day, title):
        """Log a lesson for the profile and mark it finished on `day`."""
        lid = db.log_lesson(profile, title)
        db.conn.execute(
            "UPDATE lesson_log SET finished_at=? WHERE id=?",
            (f"{day.isoformat()} 08:00:00", lid),
        )
        return lid

    def test_streak_counts_consecutive_finished_days(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        from src.telegram_bot import ChatHistoryDB, TelegramBot
        db = ChatHistoryDB(str(tmp_path / "test.db"))
        today = datetime.now(timezone.utc).date()

        # finished today + yesterday + 2 days ago → streak 3
        for i in range(3):
            day = today - timedelta(days=i)
            self._mark_day(db, "p", day, f"t{i}")
        # an unmarked lesson must not affect the streak
        db.log_lesson("p", "unmarked")

        bot = TelegramBot(config={"profiles": {}})
        orig_db = bot.db
        bot.db = db
        try:
            assert bot._lesson_streak_days("p") == 3
        finally:
            orig_db.close()
            db.close()

    def test_streak_gap_resets(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        from src.telegram_bot import ChatHistoryDB, TelegramBot
        db = ChatHistoryDB(str(tmp_path / "test.db"))
        today = datetime.now(timezone.utc).date()

        # finished yesterday and 3 days ago, but NOT 2 days ago → streak 1
        for i in (1, 3):
            day = today - timedelta(days=i)
            self._mark_day(db, "p", day, f"t{i}")

        bot = TelegramBot(config={"profiles": {}})
        orig_db = bot.db
        bot.db = db
        try:
            assert bot._lesson_streak_days("p") == 1
        finally:
            orig_db.close()
            db.close()

    def test_streak_empty(self, tmp_path):
        from src.telegram_bot import ChatHistoryDB, TelegramBot
        db = ChatHistoryDB(str(tmp_path / "test.db"))
        bot = TelegramBot(config={"profiles": {}})
        orig_db = bot.db
        bot.db = db
        try:
            assert bot._lesson_streak_days("p") == 0
        finally:
            orig_db.close()
            db.close()


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
