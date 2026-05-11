#!/usr/bin/env python3
"""
Telegram bot for LinguaDaily standalone daemon.

Handles two flows:
  1. Lesson delivery — receives a lesson dict from the scheduler/orchestrator,
     sends translated article + TTS audio to the user's Telegram chat.
  2. Tutor chat — routes user messages to llama_client.tutor_chat() with
     per-user conversation history stored in SQLite.

Config shape (config.json):
    {
      "telegram": {
        "bot_token": "123456:ABC-DEF..."
      },
      "profiles": {
        "krystof": {
          "telegram_chat_id": 123456789   // optional, pre-mapped user
        }
      }
    }

Usage (import):
    from src.telegram_bot import TelegramBot
    bot = TelegramBot(config)
    bot.deliver_lesson(profile_name, lesson_dict)
    await bot.start()          # long-running, or use as context manager

Usage (CLI):
    python3 src/telegram_bot.py --config config.json
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Optional

from config import CONFIG_PATH, DATA_DIR, load_config

CHAT_DB_PATH = DATA_DIR / "chat_history.db"

logger = logging.getLogger(__name__)

# ── SQLite conversation history ─────────────────────────────────────


class ChatHistoryDB:
    """Lightweight SQLite store for per-user tutor conversation history."""

    def __init__(self, db_path: str = CHAT_DB_PATH):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        # WAL mode for better concurrent read/write performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys and busy timeout (helps in multi-process scenarios)
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                profile TEXT NOT NULL,
                role TEXT NOT NULL,       -- 'user' or 'assistant'
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_profile
            ON chat_history (user_id, profile)
        """)
        self.conn.commit()

    def get_history(
        self, user_id: str, profile: str, max_turns: int = 10
    ) -> list[dict]:
        """Return recent conversation history as OpenAI-style messages."""
        rows = self.conn.execute("""
            SELECT role, content FROM chat_history
            WHERE user_id = ? AND profile = ?
            ORDER BY id DESC LIMIT ?
        """, (str(user_id), profile, max_turns * 2)).fetchall()

        # Reverse to chronological order
        rows.reverse()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def add_message(self, user_id: str, profile: str, role: str, content: str):
        """Store a single message in the conversation history."""
        self.conn.execute(
            "INSERT INTO chat_history (user_id, profile, role, content) VALUES (?, ?, ?, ?)",
            (str(user_id), profile, role, content),
        )
        self.conn.commit()

    def clear_history(self, user_id: str, profile: str):
        """Clear all history for a user+profile pair."""
        self.conn.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND profile = ?",
            (str(user_id), profile),
        )
        self.conn.commit()

    def purge_old_entries(self, max_age_days: int = 30):
        """
        Delete entries older than max_age_days to prevent unbounded DB growth.

        Parameters
        ----------
        max_age_days : int
            Entries older than this many days are deleted. Default: 30.
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        rows = self.conn.execute(
            "SELECT COUNT(*) FROM chat_history WHERE created_at < ?",
            (cutoff.isoformat(),),
        ).fetchone()
        deleted_count = rows[0] if rows else 0
        if deleted_count > 0:
            self.conn.execute(
                "DELETE FROM chat_history WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            self.conn.commit()
            logger.debug("Purged %d old chat entries (>=%d days)", deleted_count, max_age_days)
        return deleted_count

    def close(self):
        self.conn.close()


# ── Telegram Bot ────────────────────────────────────────────────────

class TelegramBot:
    """Telegram bot handler for lesson delivery and tutor chat."""

    def __init__(self, config=None, profile_name=None):
        if config is None:
            config = self._load_config()

        self.config = config
        self.profile_name = profile_name

        tg_cfg = config.get("telegram", {})
        self.bot_token = tg_cfg.get("bot_token", "") or os.environ.get(
            "TELEGRAM_BOT_TOKEN", ""
        )

        # Resolve chat ID → profile mapping from config
        self.chat_id_to_profile: dict[int, str] = {}
        self.profile_to_chat_id: dict[str, int] = {}
        self._build_mapping()

        # Conversation history database
        self.db = ChatHistoryDB()

        # LLM client (lazy-init)
        self._llama_client: Optional["LlamaClient"] = None

        # aiogram bot instance
        self._bot = None

    # ── Config / mapping ───────────────────────────────────────────

    def _load_config(self):
        try:
            return load_config()
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return {}

    def _build_mapping(self):
        """Build bidirectional chat_id ↔ profile mapping from config."""
        profiles = self.config.get("profiles", {})
        for name, profile in profiles.items():
            chat_id = profile.get("telegram_chat_id")
            if chat_id:
                chat_int = int(chat_id)
                self.chat_id_to_profile[chat_int] = name
                self.profile_to_chat_id[name] = chat_int
                logger.info("Mapped profile '%s' → Telegram chat %d", name, chat_int)
            else:
                logger.debug("Profile '%s' has no telegram_chat_id — skipping mapping", name)

    def resolve_profile(self, chat_id: int) -> Optional[str]:
        """Look up which profile a Telegram user belongs to."""
        return self.chat_id_to_profile.get(int(chat_id))

    def register_user(self, chat_id: int, profile_name: str):
        """Register a new chat_id → profile mapping at runtime."""
        self.chat_id_to_profile[int(chat_id)] = profile_name
        self.profile_to_chat_id[profile_name] = int(chat_id)

    # ── LLM client ─────────────────────────────────────────────────

    def _get_llama_client(self, profile_name: str):
        if self._llama_client is None or self._llama_client.profile_name != profile_name:
            from llama_client import LlamaClient
            self._llama_client = LlamaClient(
                config=self.config, profile_name=profile_name
            )
        return self._llama_client

    # ── Lesson delivery ────────────────────────────────────────────

    # Telegram message size limit
    _TG_MAX_MSG_LEN = 4096
    _TG_SAFE_TRUNCATE = 3900  # leave room for appended suffixes

    def _truncate_for_telegram(self, text: str, suffix: str = "\n…") -> str:
        """Truncate text to fit Telegram's 4096 char limit."""
        if len(text) <= self._TG_MAX_MSG_LEN:
            return text
        return text[:self._TG_SAFE_TRUNCATE] + suffix

    async def deliver_lesson(self, profile_name: str, lesson: dict):
        """
        Deliver a completed lesson to the user's Telegram chat as four messages:
          1. Original article text (content language)
          2. Translation
          3. Vocabulary list
          4. TTS audio sent as an audio file

        Parameters
        ----------
        profile_name : str
            Profile whose Telegram chat receives this lesson.
        lesson : dict
            Lesson payload with keys: title, content (translated text),
            original_content, vocab, wav_path (optional audio file path).
        """
        from aiogram.methods import SendAudio

        chat_id = self.profile_to_chat_id.get(profile_name)
        if not chat_id:
            logger.warning("No Telegram chat_id for profile '%s' — skipping delivery",
                          profile_name)
            return

        bot = await self._get_aiogram_bot()
        if bot is None:
            logger.error("Telegram bot not initialized — cannot deliver lesson")
            return

        title = lesson.get("title", "Language Lesson")
        original_content = lesson.get("original_content", "")
        translated_content = lesson.get("content", "")
        vocab = lesson.get("vocab", [])
        wav_path = lesson.get("wav_path")
        source_lang = lesson.get("source_lang", "?")
        target_lang = lesson.get("target_lang_name", "?")
        content_lang = lesson.get("content_lang", "?")

        # ── Message 1: Original text ──────────────────────────────
        msg1 = f"📰 {title}\n\n"
        msg1 += f"Original ({content_lang})\n\n"
        msg1 += self._truncate_for_telegram(original_content)

        try:
            await bot.send_message(chat_id=chat_id, text=msg1)
            logger.info("Delivered original text for '%s' to chat %d",
                        title, chat_id)
        except Exception as e:
            logger.error("Failed to send original text: %s", e)

        # ── Message 2: Translation ────────────────────────────────
        msg2 = f"🌐 Translation ({target_lang})\n\n"
        msg2 += self._truncate_for_telegram(translated_content, "\n…")

        try:
            await bot.send_message(chat_id=chat_id, text=msg2)
            logger.info("Delivered translation for '%s' to chat %d",
                        title, chat_id)
        except Exception as e:
            logger.error("Failed to send translation message: %s", e)

        # ── Message 3: Vocabulary ─────────────────────────────────
        if vocab:
            vocab_lines = []
            for entry in vocab:
                if isinstance(entry, dict):
                    word = entry.get("word", "")
                    meaning = entry.get("meaning", entry.get("definition", ""))
                    example = entry.get("example", "")
                    if example:
                        vocab_lines.append(f"  • {word} — {meaning}\n    «{example}»")
                    else:
                        vocab_lines.append(f"  • {word} — {meaning}")
                else:
                    vocab_lines.append(f"  • {entry}")

            msg3 = f"📝 Vocabulary ({len(vocab)} words)\n"
            msg3 += "\n".join(vocab_lines)

            if len(msg3) > self._TG_MAX_MSG_LEN:
                logger.warning("Vocab message exceeds Telegram limit for '%s', truncating",
                               title)
                msg3 = msg3[:self._TG_SAFE_TRUNCATE] + "\n…"

            try:
                await bot.send_message(chat_id=chat_id, text=msg3)
                logger.info("Delivered vocabulary for '%s' to chat %d",
                            title, chat_id)
            except Exception as e:
                logger.error("Failed to send vocabulary message: %s", e)

        # ── Message 4: TTS audio ──────────────────────────────────
        if wav_path and os.path.isfile(wav_path):
            try:
                from aiogram.types.input_file import FSInputFile
                audio_file = FSInputFile(
                    path=wav_path,
                    filename=os.path.basename(wav_path),
                )
                await bot(SendAudio(
                    chat_id=chat_id,
                    audio=audio_file,
                    caption=f"🔊 {title}",
                ))
                logger.info("Delivered audio for '%s' to chat %d", title, chat_id)
            except Exception as e:
                logger.error("Failed to send audio: %s", e)

    # ── Tutor chat handler ─────────────────────────────────────────

    async def handle_tutor_message(self, chat_id: int, text: str):
        """
        Route a user message to the LLM tutor and reply on Telegram.

        Parameters
        ----------
        chat_id : int
            Telegram chat ID of the user.
        text : str
            The user's message.
        """
        profile_name = self.resolve_profile(chat_id)
        if not profile_name:
            bot = await self._get_aiogram_bot()
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ You are not registered to a profile.\n"
                    "Use `/register <profile>` or ask an admin to add your "
                    "Telegram ID to config.json."
                ),
            )
            return

        profile = self.config.get("profiles", {}).get(profile_name, {})
        language_name = profile.get("target_lang_name", "German")
        native_lang = profile.get("source_lang", "English")

        # Get conversation history
        history = self.db.get_history(chat_id, profile_name, max_turns=10)

        # Call LLM tutor
        client = self._get_llama_client(profile_name)
        reply = client.tutor_chat(
            message=text,
            language_name=language_name,
            native_lang=native_lang,
            history=history,
            max_history=10,
        )

        if not reply:
            reply = "⚠️ The tutor is currently unavailable. Please try again later."

        # Store in history (best-effort — don't block the reply on DB errors)
        try:
            self.db.add_message(chat_id, profile_name, "user", text)
            self.db.add_message(chat_id, profile_name, "assistant", reply)
        except sqlite3.Error as e:
            logger.error("Failed to write chat history for %s: %s", chat_id, e)

        # Send reply (truncate for Telegram limit)
        bot = await self._get_aiogram_bot()
        if len(reply) > 4000:
            reply = reply[:3997] + "..."

        await bot.send_message(chat_id=chat_id, text=reply)

    # ── Command handlers ───────────────────────────────────────────

    async def handle_start(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if profile_name:
            profile = self.config.get("profiles", {}).get(profile_name, {})
            lang = profile.get("target_lang_name", "a language")
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"👋 Welcome to Lingua!\n\n"
                    f"You are registered as *{profile_name}* — learning {lang}.\n\n"
                    f"Send me a message and I'll tutor you in {lang}.\n"
                    f"Lessons will be delivered automatically at your scheduled time.\n\n"
                    f"Commands:\n"
                    f"/start — Show this message\n"
                    f"/register <profile> — Register to a profile\n"
                    f"/history clear — Clear chat history\n"
                    f"/status — Show current status"
                ),
                parse_mode="Markdown",
            )
        else:
            available = list(self.config.get("profiles", {}).keys())
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "👋 Welcome to Lingua!\n\n"
                    "You are not registered yet. To start:\n"
                    f"/register <profile>\n\n"
                    f"Available profiles: {', '.join(available) if available else 'none'}"
                ),
            )

    async def handle_register(self, chat_id: int, args: str):
        bot = await self._get_aiogram_bot()
        profile_name = args.strip()
        profiles = self.config.get("profiles", {})

        if not profile_name:
            available = list(profiles.keys())
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "Usage: /register <profile>\n\n"
                    f"Available profiles: {', '.join(available) if available else 'none'}"
                ),
            )
            return

        if profile_name not in profiles:
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"❌ Profile '{profile_name}' not found.\n\n"
                    f"Available: {', '.join(profiles.keys())}"
                ),
            )
            return

        self.register_user(chat_id, profile_name)
        profile = profiles[profile_name]
        lang = profile.get("target_lang_name", "a language")

        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ Registered as *{profile_name}*!\n\n"
                f"You will receive {lang} lessons and can chat with your tutor.\n\n"
                f"Send me a message to start tutoring!"
            ),
            parse_mode="Markdown",
        )

    async def handle_history_clear(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if not profile_name:
            await bot.send_message(
                chat_id=chat_id, text="⚠️ You are not registered. Use /register first."
            )
            return

        self.db.clear_history(chat_id, profile_name)
        await bot.send_message(
            chat_id=chat_id, text="🗑️ Conversation history cleared."
        )

    async def handle_status(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if profile_name:
            profile = self.config.get("profiles", {}).get(profile_name, {})
            lang = profile.get("target_lang_name", "?")
            schedule = profile.get("schedule", {})
            time_str = schedule.get("time", "not set")
            tz = schedule.get("tz", "not set")

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"📊 *Status for {profile_name}*\n\n"
                    f"Learning: {lang}\n"
                    f"Schedule: {time_str} ({tz})\n"
                    f"TTS: {'✅' if profile.get('use_tts') else '❌'}\n"
                    f"Telegram ID: {chat_id}"
                ),
                parse_mode="Markdown",
            )
        else:
            await bot.send_message(
                chat_id=chat_id, text="⚠️ Not registered. Use /register <profile>."
            )

    # ── aiogram integration ────────────────────────────────────────

    async def _get_aiogram_bot(self):
        """Lazy-init the aiogram Bot instance."""
        if self._bot is None:
            from aiogram import Bot
            if not self.bot_token:
                logger.error("No Telegram bot token configured")
                return None
            self._bot = Bot(token=self.bot_token)
        return self._bot

    async def start(self):
        """Start the Telegram bot (long-running, polls for updates)."""
        from aiogram import Dispatcher, types
        from aiogram.filters import Command

        bot = await self._get_aiogram_bot()
        if bot is None:
            logger.error("Cannot start — no bot token configured")
            return

        # Purge old chat history entries on startup (prevents unbounded DB growth)
        try:
            purged = self.db.purge_old_entries(max_age_days=30)
            if purged > 0:
                logger.info("Purged %d stale chat history entries on startup", purged)
        except Exception as e:
            logger.warning("Chat history purge failed (non-fatal): %s", e)

        dp = Dispatcher()

        # ── Command handlers ──
        @dp.message(Command("start"))
        async def cmd_start(message: types.Message):
            await self.handle_start(message.chat.id)

        @dp.message(Command("register"))
        async def cmd_register(message: types.Message):
            args = message.text.split(maxsplit=1)
            payload = args[1] if len(args) > 1 else ""
            await self.handle_register(message.chat.id, payload)

        @dp.message(Command("history"))
        async def cmd_history(message: types.Message):
            args = message.text.split(maxsplit=1)
            subcommand = args[1] if len(args) > 1 else ""
            if subcommand.strip() == "clear":
                await self.handle_history_clear(message.chat.id)
            else:
                await self._get_aiogram_bot().send_message(
                    message.chat.id, text="Usage: /history clear"
                )

        @dp.message(Command("status"))
        async def cmd_status(message: types.Message):
            await self.handle_status(message.chat.id)

        # ── All other messages → tutor chat ──
        @dp.message(lambda msg: True)  # catch-all
        async def tutor_catch_all(message: types.Message):
            if message.text:
                await self.handle_tutor_message(message.chat.id, message.text)

        logger.info("Starting Telegram bot polling...")
        try:
            await dp.start_polling(bot)
        finally:
            await bot.session.close()
            self.db.close()

    async def stop(self):
        """Gracefully stop the bot and close resources."""
        if self._bot:
            await self._bot.session.close()
        self.db.close()


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """CLI to run the Telegram bot as a standalone process.

    Usage:
        python3 src/telegram_bot.py
        python3 src/telegram_bot.py --config config.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="LinguaDaily Telegram Bot")
    parser.add_argument("--config", "-c", default=CONFIG_PATH,
                        help="Path to config.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config(args.config)

    bot = TelegramBot(config=config)

    async def run():
        try:
            await bot.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await bot.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
