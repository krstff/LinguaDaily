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
import html
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from telegramify_markdown import convert as md_convert, split_entities

from config import (
    CONFIG_PATH,
    DATA_DIR,
    DEFAULT_LEARNING_LANGUAGE,
    DEFAULT_NATIVE_LANGUAGE,
    FLASHCARD_DEFAULT_CARD_COUNT,
    FLASHCARD_DEFAULT_QUIZ_COUNT,
    LESSON_ACK_DONE_TEXT,
    LESSON_ACK_EFFECT_DEFAULT,
    LESSON_ACK_EFFECT_MONTH_STREAK,
    LESSON_ACK_EFFECT_WEEK_STREAK,
    LESSON_ACK_TEXT,
    TG_HISTORY_PURGE_DAYS,
    TG_LESSON_ACK_DELETE_DELAY_SECS,
    TG_LESSON_ACK_DEFAULT,
    TG_LESSON_COOLDOWN_SECS,
    TG_MAX_MSG_LEN,
    TG_SAFE_TRUNCATE,
    resolve_language_name,
    load_config,
)

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
        # ── Latest lesson per profile (for tutor context) ──
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS latest_lesson (
                profile TEXT PRIMARY KEY,
                title TEXT,
                original_content TEXT,
                translated_content TEXT,
                vocab_json TEXT,
                delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # ── One row per delivered lesson (stats: completion tracking) ──
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lesson_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile TEXT NOT NULL,
                title TEXT,
                delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP,
                finished_by TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lesson_log_profile
            ON lesson_log (profile, delivered_at)
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

    def purge_old_entries(self, max_age_days: int = TG_HISTORY_PURGE_DAYS):
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

    def store_lesson(self, profile: str, lesson: dict):
        """Persist the latest delivered lesson for a profile.

        Overwrites any previous lesson so the tutor always sees the most
        recent one.  Vocabulary is stored as a JSON string to keep the
        schema flat.
        """
        title = lesson.get("title", "")
        original = lesson.get("original_content", "")
        translated = lesson.get("content", "")
        vocab = json.dumps(lesson.get("vocab", []), ensure_ascii=False)

        self.conn.execute(
            """
            INSERT INTO latest_lesson (profile, title, original_content,
                                       translated_content, vocab_json,
                                       delivered_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(profile) DO UPDATE SET
                title = excluded.title,
                original_content = excluded.original_content,
                translated_content = excluded.translated_content,
                vocab_json = excluded.vocab_json,
                delivered_at = excluded.delivered_at
            """,
            (profile, title, original, translated, vocab),
        )
        self.conn.commit()

    def log_lesson(self, profile: str, title: str) -> int:
        """Insert a lesson_log row for a delivered lesson. Returns its id."""
        cur = self.conn.execute(
            "INSERT INTO lesson_log (profile, title, delivered_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (profile, title),
        )
        self.conn.commit()
        return cur.lastrowid

    def mark_lesson_finished(self, lesson_id: int, user_id) -> bool:
        """Mark a lesson as finished (read) by the given Telegram user.

        Idempotent — only the first call wins (finished_at is NULL only
        once). Returns True if this call performed the marking.
        """
        cur = self.conn.execute(
            """
            UPDATE lesson_log
            SET finished_at = CURRENT_TIMESTAMP, finished_by = ?
            WHERE id = ? AND finished_at IS NULL
            """,
            (str(user_id), lesson_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_latest_lesson(self, profile: str) -> Optional[dict]:
        """Return the most recent lesson dict for a profile, or None."""
        row = self.conn.execute(
            ("SELECT title, original_content, translated_content, vocab_json, delivered_at "
             "FROM latest_lesson WHERE profile = ?"),
            (profile,),
        ).fetchone()
        if not row:
            return None
        return {
            "title": row[0],
            "original_content": row[1] or "",
            "translated_content": row[2] or "",
            "vocab": json.loads(row[3]) if row[3] else [],
            "delivered_at": row[4],
        }

    def close(self):
        self.conn.close()


# ── Telegram Bot ────────────────────────────────────────────────────

class TelegramBot:
    """Telegram bot handler for lesson delivery and tutor chat."""

    def __init__(self, config=None, profile_name=None):
        if config is None:
            config = load_config(fallback={})

        self.config = config
        self.profile_name = profile_name

        tg_cfg = config.get("telegram", {})
        self.bot_token = tg_cfg.get("bot_token", "") or os.environ.get(
            "TELEGRAM_BOT_TOKEN", ""
        )

        # Resolve chat ID ↔ profile mappings from config.
        # A single chat ID can map to multiple profiles (learning several languages).
        self.chat_id_to_profiles: dict[int, list[str]] = {}
        self.profile_to_chat_id: dict[str, int] = {}
        self.selected_profile: dict[int, str] = {}  # active profile per chat
        self._build_mapping()

        # Conversation history database
        self.db = ChatHistoryDB()

        # LLM client (lazy-init)
        self._llama_client: Optional["LlamaClient"] = None

        # aiogram bot instance
        self._bot = None

        # Study handler (flashcards + quiz, lazy-init in start())
        self.study_handler: Optional["StudyHandler"] = None

        # Cooldown tracking for /another command (chat_id → timestamp)
        self._last_lesson_request: dict[int, float] = {}

        # Post-lesson "Finished" ack button (config: lesson_ack, default on)
        self.lesson_ack_enabled = bool(
            config.get("lesson_ack", TG_LESSON_ACK_DEFAULT))

        # In-flight tutor replies (chat_id → {"task", "message_id"}) —
        # lets /stop cancel a running LLM generation and remove the
        # "Thinking…" placeholder.
        self._in_flight: dict[int, dict] = {}

    # ── Config / mapping ───────────────────────────────────────────

    def _build_mapping(self):
        """Build bidirectional chat_id ↔ profile mappings from current config.

        A single Telegram chat ID can be shared by multiple profiles
        (e.g. one user learning German + Italian).  Each profile still
        maps to exactly one chat ID, so lessons are delivered per-profile.
        """
        profiles = self.config.get("profiles", {})
        for name, profile in profiles.items():
            chat_id = profile.get("telegram_chat_id")
            if chat_id:
                chat_int = int(chat_id)
                self.chat_id_to_profiles.setdefault(chat_int, []).append(name)
                self.profile_to_chat_id[name] = chat_int
                logger.info("Mapped profile '%s' → Telegram chat %d", name, chat_int)
            else:
                logger.debug("Profile '%s' has no telegram_chat_id — skipping mapping", name)

    def reload_config(self):
        """Reload config from disk and rebuild all in-memory mappings.

        Preserves the selected_profile state where possible (keeps a user's
        active profile selection even if the profile list changes).
        """
        old_selected = dict(self.selected_profile)

        # Reload config from disk
        self.config = load_config()

        # Drop the cached LLM client so the next tutor request rebuilds it
        # with the fresh config (e.g. after a model change in the web UI).
        # Without this, the old model stays in use until a full restart.
        self._llama_client = None

        # Reset mappings
        self.chat_id_to_profiles: dict[int, list[str]] = {}
        self.profile_to_chat_id: dict[str, int] = {}
        self._build_mapping()

        # Restore selected profiles only if they still exist
        self.selected_profile: dict[int, str] = {}
        for cid, pname in old_selected.items():
            if pname in self.profile_to_chat_id:
                self.selected_profile[cid] = pname

        logger.info("Telegram bot config reloaded — %d profile(s) mapped",
                    len(self.profile_to_chat_id))

    def resolve_profile(self, chat_id: int) -> Optional[str]:
        """Return the active profile for a Telegram user.

        If the user has multiple profiles, returns whichever they selected
        via /profiles.  If only one profile exists it is returned automatically.
        Returns None if the chat ID has no profiles at all.
        """
        cid = int(chat_id)
        profiles = self.chat_id_to_profiles.get(cid, [])
        if not profiles:
            return None
        # Return explicitly selected profile (if still valid)
        sel = self.selected_profile.get(cid)
        if sel and sel in profiles:
            return sel
        # Default to first profile
        return profiles[0]

    def select_profile(self, chat_id: int, profile_name: str) -> bool:
        """Set the active profile for a chat ID. Returns True on success."""
        cid = int(chat_id)
        profiles = self.chat_id_to_profiles.get(cid, [])
        if profile_name in profiles:
            self.selected_profile[cid] = profile_name
            return True
        return False

    def register_user(self, chat_id: int, profile_name: str):
        """Register a new chat_id → profile mapping at runtime.

        Appends to the list so one chat ID can have multiple profiles.
        """
        cid = int(chat_id)
        self.chat_id_to_profiles.setdefault(cid, []).append(profile_name)
        self.profile_to_chat_id[profile_name] = cid

    # ── LLM client ─────────────────────────────────────────────────

    def _get_llama_client(self, profile_name: str):
        if self._llama_client is None or self._llama_client.profile_name != profile_name:
            from llama_client import LlamaClient
            self._llama_client = LlamaClient(
                config=self.config, profile_name=profile_name
            )
        return self._llama_client

    # ── Lesson delivery ────────────────────────────────────────────



    def _truncate_for_telegram(self, text: str, suffix: str = "\n…") -> str:
        """Truncate text to fit Telegram's message length limit."""
        if len(text) <= TG_MAX_MSG_LEN:
            return text
        return text[:TG_SAFE_TRUNCATE] + suffix

    def _strip_markdown(self, text: str) -> str:
        """Strip common markdown formatting that LLMs may add to output.

        Telegram lesson messages use HTML parse_mode, so literal ** and __
        from markdown would render as visible characters. Strip them here.
        """
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        return text

    def _escape_html(self, text: str) -> str:
        """Escape text for Telegram HTML parse mode.

        Telegram HTML parser requires &, <, > to be escaped as entities.
        (This is a subset of full HTML escaping — enough for TG.)
        """
        return html.escape(text, quote=False)

    def _highlight_words(self, text: str, words: list) -> str:
        """Underline exact words/phrases inside already-escaped text.

        IMPORTANT: text must already be HTML-escaped before calling this,
        so that any <b> tags we add are NOT re-escaped later.

        The `words` list contains exact strings (from the LLM's highlight_*
        fields) that appear verbatim in the text. Longest matches first
        to avoid partial overlaps.
        """
        if not words:
            return text

        # Escape special regex chars and sort longest-first
        escaped = [re.escape(w) for w in words if str(w).strip()]
        escaped.sort(key=len, reverse=True)

        pattern = re.compile(
            r'(' + '|'.join(escaped) + r')',
            re.IGNORECASE,
        )

        def _replace(match):
            return f"<b>{match.group(0)}</b>"

        return pattern.sub(_replace, text)

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

        title = self._strip_markdown(lesson.get("title", "Language Lesson"))
        original_content = self._strip_markdown(lesson.get("original_content", ""))
        translated_content = self._strip_markdown(lesson.get("content", ""))
        vocab = lesson.get("vocab", [])
        wav_path = lesson.get("wav_path")
        learning_language_name = lesson.get(
            "learning_language_name", "?")
        native_language = lesson.get("native_language", "?")

        # Extract highlight words from LLM-provided lists.
        # Falls back to word/meaning fields for backward compatibility
        # with older vocab entries that lack highlight_* fields.
        vocab_source_words: list[str] = []
        vocab_target_words: list[str] = []
        for entry in vocab:
            if isinstance(entry, dict):
                # Prefer LLM-provided exact highlight forms
                src = entry.get("highlight_source")
                tgt = entry.get("highlight_target")
                if src and isinstance(src, list):
                    vocab_source_words.extend(str(w) for w in src if str(w).strip())
                elif (w := entry.get("word", "")):
                    vocab_source_words.append(w)
                if tgt and isinstance(tgt, list):
                    vocab_target_words.extend(str(w) for w in tgt if str(w).strip())
                elif (m := entry.get("meaning", entry.get("definition", ""))):
                    vocab_target_words.append(m)
            else:
                vocab_source_words.append(str(entry))

        # ── Message 1: TTS audio (sent first) ────────────────────
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

        # ── Message 2: Original text (source words highlighted) ───
        # Escape first, THEN highlight — so <b> tags are not re-escaped
        safe_original = self._escape_html(original_content)
        highlighted_original = self._highlight_words(safe_original, vocab_source_words)
        msg1 = f"📰 <b>{self._escape_html(title)}</b>\n\n"
        msg1 += f"Original ({self._escape_html(learning_language_name)})\n\n"
        msg1 += self._truncate_for_telegram(highlighted_original)

        try:
            await bot.send_message(
                chat_id=chat_id,
                text=msg1,
                parse_mode="HTML",
            )
            logger.info("Delivered original text for '%s' to chat %d",
                        title, chat_id)
        except Exception as e:
            logger.error("Failed to send original text: %s", e)

        # ── Message 3: Translation (meaning words highlighted) ───
        safe_translation = self._escape_html(translated_content)
        highlighted_translation = self._highlight_words(
            safe_translation, vocab_target_words)
        msg2 = f"🌐 Translation ({self._escape_html(native_language)})\n\n"
        msg2 += "<blockquote expandable> " + self._truncate_for_telegram(highlighted_translation, '\n…') + "</blockquote>"

        try:
            await bot.send_message(
                chat_id=chat_id,
                text=msg2,
                parse_mode="HTML",
            )
            logger.info("Delivered translation for '%s' to chat %d",
                        title, chat_id)
        except Exception as e:
            logger.error("Failed to send translation message: %s", e)

        # ── Message 4: Vocabulary (bold words, italic examples) ───
        if vocab:
            vocab_lines = []
            for entry in vocab:
                if isinstance(entry, dict):
                    word = self._escape_html(entry.get("word", ""))
                    meaning = self._escape_html(
                        entry.get("meaning", entry.get("definition", "")))
                    example = entry.get("example", "")
                    if example:
                        # word in bold, meaning plain, example sentence in italic
                        vocab_lines.append(
                            f"  • <b>{word}</b> — <tg-spoiler>{meaning}</tg-spoiler>\n"
                            f"    <i>{self._escape_html(example)}</i>")
                    else:
                        vocab_lines.append(f"  • <b>{word}</b> — {meaning}")
                else:
                    vocab_lines.append(
                        f"  • <b>{self._escape_html(str(entry))}</b>")

            msg3 = f"📝 <b>Vocabulary ({len(vocab)} words)</b>\n"
            msg3 += "\n".join(vocab_lines)

            if len(msg3) > TG_MAX_MSG_LEN:
                logger.warning("Vocab message exceeds Telegram limit for '%s', truncating",
                               title)
                msg3 = msg3[:TG_SAFE_TRUNCATE] + "\n…"

            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg3,
                    parse_mode="HTML",
                )
                logger.info("Delivered vocabulary for '%s' to chat %d",
                            title, chat_id)
            except Exception as e:
                logger.error("Failed to send vocabulary message: %s", e)

        # Persist lesson so the tutor has context
        try:
            self.db.store_lesson(profile_name, lesson)
            logger.info("Stored lesson for '%s' (profile: %s)",
                        title, profile_name)
        except sqlite3.Error as e:
            logger.error("Failed to store lesson for '%s': %s", profile_name, e)

        # ── Message 5: "Finished" ack button (completion tracking) ──
        if self.lesson_ack_enabled:
            lesson_id = None
            try:
                lesson_id = self.db.log_lesson(profile_name, title)
            except sqlite3.Error as e:
                logger.error("Failed to log lesson for '%s': %s", profile_name, e)
            if lesson_id is not None:
                await self._send_lesson_ack(bot, chat_id, profile_name, lesson_id)

        # Auto-switch tutor context to the profile that just received a lesson
        self.select_profile(chat_id, profile_name)

    async def _send_lesson_ack(
        self, bot, chat_id: int, profile_name: str, lesson_id: int,
    ):
        """Send the post-lesson acknowledgement message (message 5).

        Plain text + a single "Finished" button (no effect — the streak
        effect is reserved for the done confirmation in
        handle_lesson_done). The click is recorded in lesson_log.
        """
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        # Messages are in the learner's LEARNING language (the lesson's
        # language), not their native one.
        lang = (self.config.get("profiles", {}).get(profile_name, {})
                .get("learning_language", DEFAULT_LEARNING_LANGUAGE))
        text = LESSON_ACK_TEXT.get(lang, LESSON_ACK_TEXT["en"])
        button = InlineKeyboardButton(
            text="✅ Finished",
            callback_data=f"ld:{profile_name}:{lesson_id}",
        )
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[button]]),
            )
            logger.info("Sent lesson-ack message (lesson %d) to chat %d",
                        lesson_id, chat_id)
        except Exception as e:
            logger.error("Failed to send lesson-ack message: %s", e)

    def _lesson_streak_days(self, profile_name: str) -> int:
        """Count consecutive days (ending today) with at least one finished
        lesson for the profile.

        Today counts only if a lesson was already marked finished (e.g. a
        second daily lesson); otherwise the count starts at yesterday.
        """
        rows = self.db.conn.execute(
            "SELECT DISTINCT date(finished_at) FROM lesson_log "
            "WHERE profile = ? AND finished_at IS NOT NULL "
            "ORDER BY date(finished_at) DESC",
            (profile_name,),
        ).fetchall()
        days = {r[0] for r in rows}
        d = datetime.now(timezone.utc).date()
        streak = 0
        if d.isoformat() not in days:
            d -= timedelta(days=1)
        while d.isoformat() in days:
            streak += 1
            d -= timedelta(days=1)
        return streak

    @staticmethod
    def _lesson_ack_effect(streak_days: int) -> str:
        """Pick the ack message effect from the completion streak.

        Every 28th consecutive finished day → heart, every 7th → confetti,
        everything else → fire (checked modulo, so streaks keep cycling).
        """
        if streak_days >= 28 and streak_days % 28 == 0:
            return LESSON_ACK_EFFECT_MONTH_STREAK
        if streak_days >= 7 and streak_days % 7 == 0:
            return LESSON_ACK_EFFECT_WEEK_STREAK
        return LESSON_ACK_EFFECT_DEFAULT

    async def handle_lesson_done(self, profile_name: str, message) -> bool:
        """Post-click effect for the lesson-ack message.

        Deletes the ack message (with its button), sends a short
        confirmation as a NEW message with the streak-based effect
        (fire / confetti / heart) — effects are a send-time property,
        so editing cannot change them — then deletes that one again after
        TG_LESSON_ACK_DELETE_DELAY_SECS.
        Falls back to in-place editing if the first delete is not allowed
        (e.g. a group where the bot lacks delete rights).
        Returns True if the confirmation was shown.
        """
        if message is None:
            return False
        from aiogram.types import InlineKeyboardMarkup

        lang = (self.config.get("profiles", {}).get(profile_name, {})
                .get("learning_language", DEFAULT_LEARNING_LANGUAGE))
        done_text = LESSON_ACK_DONE_TEXT.get(lang, LESSON_ACK_DONE_TEXT["en"])

        # 1) Remove the ack message (with its button)
        try:
            await message.delete()
        except Exception as e:
            logger.warning("Could not delete lesson-ack message: %s "
                           "— falling back to in-place edit", e)
            try:
                await message.edit_text(
                    text=done_text,
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[]),
                )
                return True
            except Exception as e2:
                logger.warning("Could not edit lesson-ack message either: %s", e2)
                return False

        # 2) Send the confirmation as a new message with the streak effect
        bot = await self._get_aiogram_bot()
        streak = self._lesson_streak_days(profile_name)
        effect = self._lesson_ack_effect(streak)
        try:
            sent = await bot.send_message(
                chat_id=message.chat.id,
                text=done_text,
                message_effect_id=effect,
            )
            logger.info(
                "Lesson-done confirmation sent (profile '%s', streak %d, "
                "effect %s)",
                profile_name, streak, effect)
        except Exception as e:
            logger.error("Failed to send lesson-done confirmation: %s", e)
            return False

        # 3) Delete the confirmation after a short delay
        try:
            await asyncio.sleep(TG_LESSON_ACK_DELETE_DELAY_SECS)
            await sent.delete()
            logger.info("Lesson-done confirmation deleted (profile '%s')",
                        profile_name)
            return True
        except Exception as e:
            logger.info("Keeping lesson-done confirmation (delete failed): %s", e)
            return True

    # ── Tutor chat handler ─────────────────────────────────────────

    async def _edit_tutor_reply(
        self, bot, chat_id: int, message_id: int, reply: str
    ):
        """Replace a placeholder message with the actual tutor reply.

        Edits the existing message in-place so the user sees the thinking
        indicator replaced by the LLM response without an extra message.
        Falls back to sending a new message if editing fails (e.g. too old).
        For multi-chunk replies the first chunk is edited, remaining chunks
        are sent as new messages.
        """
        # Truncate very long replies before conversion
        if len(reply) > 4096:
            reply = reply[:4093] + "..."

        try:
            text, entities = md_convert(reply)
        except Exception as e:
            logger.warning("telegramify-markdown convert failed: %s — sending plain", e)
            await self._fallback_edit(bot, chat_id, message_id, reply)
            return

        entity_dicts = [e.to_dict() for e in entities]
        chunks = list(split_entities(text, entity_dicts, max_utf16_len=4096))

        for i, (chunk_text, chunk_entities) in enumerate(chunks):
            if not chunk_text.strip():
                continue

            try:
                if i == 0:
                    # First chunk: edit the placeholder message in-place
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=chunk_text,
                        entities=chunk_entities,
                    )
                else:
                    # Additional chunks: send as new messages
                    await bot.send_message(
                        chat_id=chat_id,
                        text=chunk_text,
                        entities=chunk_entities,
                    )
            except Exception as e:
                logger.error("Failed to send/edit tutor reply chunk: %s", e)
                # Fallback: try sending as plain text
                await self._fallback_edit(bot, chat_id, message_id, chunk_text)
                break  # stop processing further chunks after fallback

    async def _fallback_edit(
        self, bot, chat_id: int, message_id: int, text: str
    ):
        """Fallback when editing the placeholder message fails.

        Tries to delete the stale placeholder and send a new message instead.
        """
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception as e:
            logger.debug("Could not delete placeholder message: %s", e)
        try:
            await bot.send_message(chat_id=chat_id, text=text)
        except Exception as fallback_err:
            logger.error("Fallback send also failed: %s", fallback_err)

    async def _send_tutor_reply_with_entities(
        self, bot, chat_id: int, reply: str
    ):
        """Send a tutor reply using telegramify-markdown entities.

        Converts markdown from the LLM to Telegram MessageEntity objects,
        sending plain text + entities (no parse_mode needed).
        Splits long messages respecting entity boundaries and UTF-16 limits.
        Caps at ~4096 chars total to avoid spamming with many messages.
        """
        # Truncate very long replies before conversion
        if len(reply) > 4096:
            reply = reply[:4093] + "..."

        try:
            text, entities = md_convert(reply)
        except Exception as e:
            logger.warning("telegramify-markdown convert failed: %s — sending plain", e)
            await bot.send_message(chat_id=chat_id, text=reply)
            return

        entity_dicts = [e.to_dict() for e in entities]
        chunks = list(split_entities(text, entity_dicts, max_utf16_len=4096))

        for chunk_text, chunk_entities in chunks:
            if not chunk_text.strip():
                continue
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk_text,
                    entities=chunk_entities,
                )
            except Exception as e:
                logger.error("Failed to send tutor reply chunk: %s", e)
                # Fallback: send as plain text
                try:
                    await bot.send_message(chat_id=chat_id, text=chunk_text)
                except Exception as fallback_err:
                    logger.error("Fallback send also failed: %s", fallback_err)

    async def handle_tutor_message(self, chat_id: int, text: str):
        """
        Route a user message to the LLM tutor and reply on Telegram.

        Shows an immediate "thinking" message that is replaced with the
        actual LLM reply once ready — so the user sees feedback even when
        the model needs time to load or RAG takes a while.

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
                    "⚠️ You are not registered to a profile. Ask an admin "
                    "to add your Telegram chat ID via the web UI."
                ),
            )
            return

        # Reject new messages while a reply is still being generated
        existing = self._in_flight.get(int(chat_id))
        if existing is not None and self._tutor_task_running(existing["task"]):
            bot = await self._get_aiogram_bot()
            await bot.send_message(
                chat_id=chat_id,
                text="⏳ I'm still working on the previous message — "
                     "send /stop to cancel it.",
            )
            return

        profile = self.config.get("profiles", {}).get(profile_name, {})
        learning_language = profile.get("learning_language", DEFAULT_LEARNING_LANGUAGE)
        language_name = resolve_language_name(learning_language)
        native_lang = profile.get("native_language", DEFAULT_NATIVE_LANGUAGE)

        # ── Send immediate "thinking" placeholder ────────────────
        bot = await self._get_aiogram_bot()
        thinking_msg = await bot.send_message(
            chat_id=chat_id,
            text="Thinking…",
        )
        thinking_message_id = thinking_msg.message_id

        # Get conversation history
        history = self.db.get_history(chat_id, profile_name, max_turns=10)

        # Fetch the latest delivered lesson for tutor context
        lesson = None
        try:
            lesson = self.db.get_latest_lesson(profile_name)
        except sqlite3.Error as e:
            logger.error("Failed to fetch lesson for '%s': %s", profile_name, e)

        # Run the (streaming) LLM call as a task so the event loop stays
        # free — /stop can cancel it, aborting server-side generation.
        task = asyncio.create_task(self._run_tutor_reply(
            chat_id=chat_id,
            profile_name=profile_name,
            text=text,
            language_name=language_name,
            native_lang=native_lang,
            history=history,
            lesson=lesson,
            message_id=thinking_message_id,
        ))
        self._in_flight[int(chat_id)] = {
            "task": task,
            "message_id": thinking_message_id,
        }

    async def _run_tutor_reply(
        self,
        chat_id: int,
        profile_name: str,
        text: str,
        language_name: str,
        native_lang: str,
        history: list,
        lesson: Optional[dict],
        message_id: int,
    ):
        """Background task: stream the tutor reply and post it to Telegram.

        If cancelled via /stop, the streaming connection is closed so the
        model server stops generating; the placeholder was already deleted
        by the /stop handler, so nothing is sent here.
        """
        bot = await self._get_aiogram_bot()
        try:
            client = self._get_llama_client(profile_name)
            reply = await client.tutor_chat_stream(
                message=text,
                language_name=language_name,
                native_lang=native_lang,
                history=history,
                max_history=10,
                lesson=lesson,
            )

            if not reply:
                reply = "⚠️ The tutor is currently unavailable. Please try again later."

            # Store in history (best-effort — don't block the reply on DB errors)
            try:
                self.db.add_message(chat_id, profile_name, "user", text)
                self.db.add_message(chat_id, profile_name, "assistant", reply)
            except sqlite3.Error as e:
                logger.error("Failed to write chat history for %s: %s", chat_id, e)

            # ── Replace thinking message with actual reply ───────
            await self._edit_tutor_reply(bot, chat_id, message_id, reply)
        except asyncio.CancelledError:
            logger.info("[chat %d] Tutor reply cancelled via /stop", chat_id)
            raise
        except Exception as e:
            logger.error("[chat %d] Tutor reply failed: %s", chat_id, e, exc_info=True)
            try:
                await self._edit_tutor_reply(
                    bot, chat_id, message_id,
                    "⚠️ The tutor is currently unavailable. Please try again later.",
                )
            except Exception:
                pass  # e.g. placeholder already removed by /stop
        finally:
            self._in_flight.pop(int(chat_id), None)

    @staticmethod
    def _tutor_task_running(task) -> bool:
        """True while a tutor task is running or being cancelled."""
        # task.cancelling() (3.11+) covers the brief window between cancel()
        # being requested and the task actually finishing (avoids double /stop)
        cancelling = getattr(task, "cancelling", lambda: False)()
        return not task.done() and not cancelling

    async def handle_stop(self, chat_id: int):
        """Cancel the in-flight tutor reply for this chat (if any).

        Cancelling the task closes the streaming HTTP connection, which
        makes the inference server abort the generation.  The "Thinking…"
        placeholder is deleted so the exchange disappears from the chat.
        """
        bot = await self._get_aiogram_bot()
        entry = self._in_flight.get(int(chat_id))
        if entry is None or not self._tutor_task_running(entry["task"]):
            await bot.send_message(
                chat_id=chat_id,
                text="⏹ Nothing to stop — no pending reply.",
            )
            return

        entry["task"].cancel()
        try:
            await bot.delete_message(chat_id, entry["message_id"])
        except Exception as e:
            logger.debug("Could not delete thinking message: %s", e)
        await bot.send_message(
            chat_id=chat_id,
            text="⏹ Stopped.",
        )

    # ── Command handlers ───────────────────────────────────────────

    async def handle_start(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if profile_name:
            profile = self.config.get("profiles", {}).get(profile_name, {})
            lang = resolve_language_name(
                profile.get("learning_language", DEFAULT_LEARNING_LANGUAGE))
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"👋 Welcome to Lingua!\n\n"
                    f"You are registered as <b>{self._escape_html(profile_name)}</b> — learning {self._escape_html(lang)}.\n\n"
                    f"Send me a message and I'll tutor you in {self._escape_html(lang)}.\n"
                    f"Lessons will be delivered automatically at your scheduled time.\n\n"
                    f"Commands:\n"
                    f"/start — Show this message\n"
                    f"/another — Request another daily lesson\n"
                    f"/flashcards [N] — Browse vocabulary as flashcards (default 10)\n"
                    f"/quiz [N]       — Multiple-choice quiz (default 10 questions)\n"
                    f"/chatid — Show your Telegram Chat ID\n"
                    f"/profiles — List & switch your profiles\n"
                    f"/stop — Cancel a pending tutor reply\n"
                    f"/history clear — Clear chat history\n"
                    f"/stats — Show your stats"
                ),
                parse_mode="HTML",
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "👋 Welcome to Lingua!\n\n"
                    "You are not registered yet. Ask an admin to add your "
                    "Telegram chat ID to config.json via the web UI."
                ),
            )

    async def handle_history_clear(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if not profile_name:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ Not registered. Ask an admin via the web UI."
            )
            return

        self.db.clear_history(chat_id, profile_name)
        await bot.send_message(
            chat_id=chat_id, text="🗑️ Conversation history cleared."
        )

    async def handle_stats(self, chat_id: int):
        """/stats — current status + learning statistics for the profile."""
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if not profile_name:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ Not registered. Ask an admin via the web UI."
            )
            return

        profile = self.config.get("profiles", {}).get(profile_name, {})
        lang = resolve_language_name(
            profile.get("learning_language", "?"))
        schedule = profile.get("schedule", {})
        time_str = schedule.get("time", "not set")
        tz = schedule.get("tz", "not set")

        from stats import profile_stats
        try:
            stats = profile_stats(profile_name, config=self.config)
        except Exception as e:
            logger.error("Failed to compute stats for '%s': %s", profile_name, e)
            stats = None

        lines = [
            f"📊 <b>Stats for {self._escape_html(profile_name)}</b>",
            "",
            f"Learning: {self._escape_html(lang)}",
            f"Schedule: {self._escape_html(time_str)} ({self._escape_html(tz)})",
            f"TTS: {'✅' if profile.get('use_tts') else '❌'}",
        ]

        if stats:
            L = stats["lessons"]
            V = stats["vocab"]
            lines += [
                "",
                "📖 <b>Lessons</b>",
                f"Delivered: {L['delivered']}",
                f"Finished: {L['finished']}"
                + (f" ({L['completion_rate']}%)" if L['completion_rate'] is not None else ""),
                f"Current streak: {L['current_streak']} day(s)",
                f"Best streak: {L['best_streak']} day(s)",
                "",
                "📚 <b>Vocabulary</b>",
                f"Words: {V['total_words']}",
                f"Quiz accuracy: "
                + (f"{V['quiz_accuracy']}% ({V['quiz_correct']}/{V['quiz_attempts']})"
                   if V['quiz_accuracy'] is not None else "no data yet"),
                f"Avg mastery: {V['avg_mastery']:.2f} "
                f"· mastered (≥80%): {V['mastered_words']}",
            ]
            if L["last_lesson"]:
                lines.append(f"Last lesson: {L['last_lesson']}")

        lines.append(f"\nTelegram ID: {chat_id}")

        await bot.send_message(
            chat_id=chat_id,
            text="\n".join(lines),
            parse_mode="HTML",
        )

    async def handle_chat_id(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"🆔 Your Telegram Chat ID is:\n\n"
                f"<code>{chat_id}</code>\n\n"
                f"Send this to your admin so they can register you via the web UI."
            ),
            parse_mode="HTML",
        )

    async def handle_profiles(self, chat_id: int):
        """List all profiles available for this chat ID (with switch buttons)."""
        await self._send_profile_menu(chat_id)

    async def _send_profile_menu(self, chat_id: int, header: str = ""):
        """Send the profile list as a message with one inline button per profile.

        Clicking a button switches the active profile and removes the menu
        message (see handle_switch_callback).
        """
        bot = await self._get_aiogram_bot()
        profiles = self.chat_id_to_profiles.get(int(chat_id), [])
        if not profiles:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ No profiles found for this chat. Ask an admin to register you.",
            )
            return

        active = self.resolve_profile(chat_id)
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        lines = []
        rows = []
        for p in profiles:
            profile_cfg = self.config.get("profiles", {}).get(p, {})
            lang = resolve_language_name(
                profile_cfg.get("learning_language", "?"))
            marker = " ◀ active" if p == active else ""
            lines.append(f"  • <b>{self._escape_html(p)}</b> — {self._escape_html(lang)}{marker}")
            if len(profiles) > 1:
                btn_text = f"{'✅ ' if p == active else ''}{p} ({lang})"[:64]
                rows.append([InlineKeyboardButton(
                    text=btn_text,
                    callback_data=f"sw:{chat_id}:{p}",
                )])

        footer = ("Tap a button to switch your active profile."
                  if rows else "You have only one profile — it is active.")
        text = header + (
            f"👤 Your profiles:\n\n"
            + "\n".join(lines) + "\n\n"
            + footer
        )
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows) if rows else None,
        )

    async def handle_switch_callback(
        self, chat_id: int, profile_name: str, message=None,
    ) -> bool:
        """Switch the active profile after a user tapped an inline button.

        Removes the menu message so it disappears, then sends a short
        confirmation. Returns False if the profile is not valid for this chat.
        """
        bot = await self._get_aiogram_bot()
        cid = int(chat_id)
        if profile_name not in self.chat_id_to_profiles.get(cid, []):
            return False

        self.select_profile(cid, profile_name)

        # Remove the menu message so it disappears after switching
        if message is not None:
            try:
                await message.delete()
            except Exception as e:
                logger.debug("Could not delete profile menu message: %s", e)

        lang = resolve_language_name(
            self.config.get("profiles", {}).get(profile_name, {}).get(
                "learning_language", "?"))
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ Switched to <b>{self._escape_html(profile_name)}</b> ({self._escape_html(lang)})\n\n"
                f"Tutor messages will now use this profile."
            ),
            parse_mode="HTML",
        )
        return True

    async def handle_another_lesson(self, chat_id: int):
        """Trigger an on-demand lesson for the user's active profile.

        Enforces a cooldown window (TG_LESSON_COOLDOWN_SECS) per chat to
        prevent spam.  The full pipeline runs asynchronously in the background;
        the user only sees a confirmation message here.
        """
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if not profile_name:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ Not registered. Ask an admin to add your Telegram chat ID.",
            )
            return

        cid = int(chat_id)
        now = time.time()
        last = self._last_lesson_request.get(cid, 0)
        remaining = TG_LESSON_COOLDOWN_SECS - (now - last)

        if remaining > 0:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            await bot.send_message(
                chat_id=chat_id,
                text=f"⏳ Please wait {mins}m {secs}s before requesting another lesson.",
            )
            return

        self._last_lesson_request[cid] = now
        await bot.send_message(
            chat_id=chat_id,
            text=f"📖 Requesting a new lesson for <b>{self._escape_html(profile_name)}</b>…",
            parse_mode="HTML",
        )

        from orchestrator import Orchestrator
        orch = Orchestrator(config=self.config)
        try:
            await orch.run_lesson(
                profile_name,
                delivery_callback=self.deliver_lesson,
            )
        except Exception as e:
            logger.error("[%s] On-demand lesson failed: %s", profile_name, e, exc_info=True)

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

        # Suppress aiogram's verbose INFO logs for callback queries
        logging.getLogger("aiogram.event").setLevel(logging.WARNING)
        logging.getLogger("aiogram.dispatcher").setLevel(logging.WARNING)

        bot = await self._get_aiogram_bot()
        if bot is None:
            logger.error("Cannot start — no bot token configured")
            return

        # Purge old chat history entries on startup (prevents unbounded DB growth)
        try:
            purged = self.db.purge_old_entries(max_age_days=TG_HISTORY_PURGE_DAYS)
            if purged > 0:
                logger.info("Purged %d stale chat history entries on startup", purged)
        except Exception as e:
            logger.warning("Chat history purge failed (non-fatal): %s", e)

        dp = Dispatcher()

        # ── Study integration (flashcards + quiz) ────────────────
        try:
            from flashcards import StudyHandler
            self.study_handler = StudyHandler(
                config=self.config, telegram_bot=self
            )
            logger.info("Study handler initialised (flashcards + quiz)")
        except Exception as e:
            logger.warning("Study module not available: %s", e)

        # ── Command handlers ──
        @dp.message(Command("start"))
        async def cmd_start(message: types.Message):
            await self.handle_start(message.chat.id)

        @dp.message(Command("history"))
        async def cmd_history(message: types.Message):
            args = message.text.split(maxsplit=1)
            subcommand = args[1] if len(args) > 1 else ""
            if subcommand.strip() == "clear":
                await self.handle_history_clear(message.chat.id)
            else:
                bot = await self._get_aiogram_bot()
                await bot.send_message(
                    message.chat.id, text="Usage: /history clear"
                )

        @dp.message(Command("stats"))
        async def cmd_stats(message: types.Message):
            await self.handle_stats(message.chat.id)

        @dp.message(Command("chatid"))
        async def cmd_chatid(message: types.Message):
            await self.handle_chat_id(message.chat.id)

        @dp.message(Command("profiles"))
        async def cmd_profiles(message: types.Message):
            await self.handle_profiles(message.chat.id)

        @dp.message(Command("stop"))
        async def cmd_stop(message: types.Message):
            await self.handle_stop(message.chat.id)

        @dp.message(Command("another"))
        async def cmd_another(message: types.Message):
            await self.handle_another_lesson(message.chat.id)

        # ── Flashcard command ────────────────────────────────────
        @dp.message(Command("flashcards"))
        async def cmd_flashcards(message: types.Message):
            if self.study_handler is None:
                await message.answer("⚠️ Study module not available.")
                return
            profile_name = self.resolve_profile(message.chat.id)
            if not profile_name:
                await message.answer(
                    "⚠️ Not registered. Ask an admin to add your Telegram chat ID."
                )
                return

            # Parse optional count argument: /flashcards 15
            args = message.text.split(maxsplit=1)
            count = FLASHCARD_DEFAULT_CARD_COUNT
            if len(args) > 1:
                try:
                    count = int(args[1].strip())
                    count = max(1, min(count, 50))
                except ValueError:
                    pass

            await self.study_handler.start_flashcards(
                chat_id=message.chat.id,
                profile_name=profile_name,
                count=count,
            )

        # ── Quiz command ─────────────────────────────────────────
        @dp.message(Command("quiz"))
        async def cmd_quiz(message: types.Message):
            if self.study_handler is None:
                await message.answer("⚠️ Study module not available.")
                return
            profile_name = self.resolve_profile(message.chat.id)
            if not profile_name:
                await message.answer(
                    "⚠️ Not registered. Ask an admin to add your Telegram chat ID."
                )
                return

            # Parse optional count argument: /quiz 20
            args = message.text.split(maxsplit=1)
            count = FLASHCARD_DEFAULT_QUIZ_COUNT
            if len(args) > 1:
                try:
                    count = int(args[1].strip())
                    count = max(2, min(count, 50))
                except ValueError:
                    pass

            await self.study_handler.start_quiz(
                chat_id=message.chat.id,
                profile_name=profile_name,
                count=count,
            )

        # ── Profile switch callback (inline buttons) ────────────
        @dp.callback_query(lambda c: c.data and c.data.startswith("sw:"))
        async def switch_callback(callback_query: types.CallbackQuery):
            parts = callback_query.data.split(":", 2)
            if len(parts) != 3:
                await callback_query.answer()
                return
            menu_chat_id, profile_name = int(parts[1]), parts[2]
            # Only the menu owner may use it
            if int(callback_query.from_user.id) != menu_chat_id:
                await callback_query.answer("⚠️ This menu is not for you", show_alert=True)
                return
            if not await self.handle_switch_callback(
                    menu_chat_id, profile_name, callback_query.message):
                await callback_query.answer(
                    "⚠️ This profile is no longer available", show_alert=True)
                return
            await callback_query.answer()

        # ── Lesson "Finished" ack callback (inline button) ────────
        @dp.callback_query(lambda c: c.data and c.data.startswith("ld:"))
        async def lesson_done_callback(callback_query: types.CallbackQuery):
            parts = callback_query.data.split(":", 2)
            if len(parts) != 3:
                await callback_query.answer()
                return
            profile_name, lesson_id_raw = parts[1], parts[2]
            try:
                lesson_id = int(lesson_id_raw)
            except ValueError:
                await callback_query.answer()
                return
            if callback_query.message is None:
                await callback_query.answer()
                return
            chat_id = int(callback_query.message.chat.id)
            # Only a user with this profile registered in their chat may mark it
            if profile_name not in self.chat_id_to_profiles.get(chat_id, []):
                await callback_query.answer(
                    "⚠️ This lesson is not for you", show_alert=True)
                return
            marked = self.db.mark_lesson_finished(
                lesson_id, callback_query.from_user.id)
            if not marked:
                await callback_query.answer("Already marked ✓")
                try:
                    await callback_query.message.delete()
                except Exception:
                    pass
                return
            logger.info(
                "Lesson %d marked finished by user %d (profile '%s')",
                lesson_id, callback_query.from_user.id, profile_name)
            await callback_query.answer()
            await self.handle_lesson_done(profile_name, callback_query.message)

        # ── Study callback queries (flashcards + quiz) ───────────
        @dp.callback_query(lambda c: c.data and (c.data.startswith("fc:") or c.data.startswith("qz:")))
        async def study_callback(callback_query: types.CallbackQuery):
            if self.study_handler is None:
                return
            # Active quiz/flashcard sessions and the post-quiz result
            # buttons (retry_missed / new_quiz / to_flashcards) are all
            # handled inside the study handler.
            await self.study_handler.handle_callback(callback_query)

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
