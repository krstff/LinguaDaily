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
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

from config import CONFIG_PATH, DATA_DIR, resolve_language_name, load_config

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
        via /switch.  If only one profile exists it is returned automatically.
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

    # Telegram message size limit
    _TG_MAX_MSG_LEN = 4096
    _TG_SAFE_TRUNCATE = 3900  # leave room for appended suffixes

    def _truncate_for_telegram(self, text: str, suffix: str = "\n…") -> str:
        """Truncate text to fit Telegram's 4096 char limit."""
        if len(text) <= self._TG_MAX_MSG_LEN:
            return text
        return text[:self._TG_SAFE_TRUNCATE] + suffix

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

    def _markdown_to_telegram_html(self, text: str) -> str:
        """Convert common markdown formatting to Telegram HTML tags.

        Handles:
          # / ## / ### headers  → <b>header</b>
          **bold**              → <b>bold</b>
          *italic*              → <i>italic</i>
          `code`                → <code>code</code>
          > blockquote          → <i>blockquote</i>
          - / * / 1. lists     → escaped as-is (Telegram renders plain text)

        Uses a placeholder strategy so that our own HTML tags are never
        re-escaped by the final HTML-escape pass.
        """
        # ── Normalise Unicode asterisks to ASCII —───────────────
        # LLMs sometimes emit ∗ (U+2217), ✱ (U+2731), * (U+00B7)
        text = text.replace('\u2217', '*').replace('\u2731', '*')
        text = text.replace('\u00b7', '*').replace('\u2042', '*')

        replacements: dict[str, str] = {}
        counter = 0

        def _store(content: str, tag: str) -> str:
            nonlocal counter
            key = f'\x00TG{counter}\x00'
            replacements[key] = f'{tag}{self._escape_html(content)}</{tag[1:]}'
            counter += 1
            return key

        # ── Step 1: Extract block-level markdown (headers, blockquotes) ──
        # Headers: # / ## / ### → <b>text</b> (Telegram has no <h3>)
        def _replace_header(m: re.Match):
            return _store(m.group(2).strip(), '<b>')
        text = re.sub(r'^(#{1,6})\s+(.+)$', _replace_header, text, flags=re.MULTILINE)

        # Blockquotes: > text → <i>text</i>
        def _replace_blockquote(m: re.Match):
            return _store(m.group(1).strip(), '<i>')
        text = re.sub(r'^>\s*(.+)$', _replace_blockquote, text, flags=re.MULTILINE)

        # ── Step 2: Extract inline markdown (bold, italic, code) ───
        _inline_re = re.compile(
            r'\*\*(.+?)\*\*'           # **bold**
            r'|(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)'  # *italic*
            r'|`(.+?)`',                            # `code`
        )

        def _replace_inline(m: re.Match):
            tag_map = {1: '<b>', 2: '<i>', 3: '<code>'}
            content = m.group(1) or m.group(2) or m.group(3)
            return _store(content, tag_map[m.lastindex])

        text = _inline_re.sub(_replace_inline, text)

        # ── Step 3: HTML-escape the remaining plain text ─────────
        text = self._escape_html(text)

        # ── Step 4: Restore all placeholders (contain real <b>/<i> tags) ─
        for key, value in replacements.items():
            text = text.replace(self._escape_html(key), value)

        return text

    def _format_text_for_telegram(self, text: str) -> str:
        """Escape plain text for Telegram HTML messages."""
        return self._escape_html(text)

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

            if len(msg3) > self._TG_MAX_MSG_LEN:
                logger.warning("Vocab message exceeds Telegram limit for '%s', truncating",
                               title)
                msg3 = msg3[:self._TG_SAFE_TRUNCATE] + "\n…"

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

        # Auto-switch tutor context to the profile that just received a lesson
        self.select_profile(chat_id, profile_name)

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
                    "⚠️ You are not registered to a profile. Ask an admin "
                    "to add your Telegram chat ID via the web UI."
                ),
            )
            return

        profile = self.config.get("profiles", {}).get(profile_name, {})
        learning_language = profile.get("learning_language", "de")
        language_name = resolve_language_name(learning_language)
        native_lang = profile.get("native_language", "en")

        # Get conversation history
        history = self.db.get_history(chat_id, profile_name, max_turns=10)

        # Fetch the latest delivered lesson for tutor context
        lesson = None
        try:
            lesson = self.db.get_latest_lesson(profile_name)
        except sqlite3.Error as e:
            logger.error("Failed to fetch lesson for '%s': %s", profile_name, e)

        # Call LLM tutor
        client = self._get_llama_client(profile_name)
        reply = client.tutor_chat(
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

        # Send reply formatted for Telegram HTML (truncate for limit)
        bot = await self._get_aiogram_bot()
        if len(reply) > 4000:
            reply = reply[:3997] + "..."

        # Convert markdown from LLM to Telegram HTML tags, then escape rest
        tg_reply = self._markdown_to_telegram_html(reply)

        await bot.send_message(
            chat_id=chat_id,
            text=tg_reply,
            parse_mode="HTML",
        )

    # ── Command handlers ───────────────────────────────────────────

    async def handle_start(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if profile_name:
            profile = self.config.get("profiles", {}).get(profile_name, {})
            lang = resolve_language_name(
                profile.get("learning_language", "de"))
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"👋 Welcome to Lingua!\n\n"
                    f"You are registered as <b>{self._escape_html(profile_name)}</b> — learning {self._escape_html(lang)}.\n\n"
                    f"Send me a message and I'll tutor you in {self._escape_html(lang)}.\n"
                    f"Lessons will be delivered automatically at your scheduled time.\n\n"
                    f"Commands:\n"
                    f"/start — Show this message\n"
                    f"/flashcards [N] — Browse vocabulary as flashcards (default 10)\n"
                    f"/quiz [N]       — Multiple-choice quiz (default 10 questions)\n"
                    f"/chatid — Show your Telegram Chat ID\n"
                    f"/profiles — List your profiles\n"
                    f"/switch &lt;name&gt; — Switch active profile\n"
                    f"/history clear — Clear chat history\n"
                    f"/status — Show current status"
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

    async def handle_status(self, chat_id: int):
        bot = await self._get_aiogram_bot()
        profile_name = self.resolve_profile(chat_id)
        if profile_name:
            profile = self.config.get("profiles", {}).get(profile_name, {})
            lang = resolve_language_name(
                profile.get("learning_language", "?"))
            schedule = profile.get("schedule", {})
            time_str = schedule.get("time", "not set")
            tz = schedule.get("tz", "not set")

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"📊 <b>Status for {self._escape_html(profile_name)}</b>\n\n"
                    f"Learning: {self._escape_html(lang)}\n"
                    f"Schedule: {self._escape_html(time_str)} ({self._escape_html(tz)})\n"
                    f"TTS: {'✅' if profile.get('use_tts') else '❌'}\n"
                    f"Telegram ID: {chat_id}"
                ),
                parse_mode="HTML",
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ Not registered. Ask an admin via the web UI."
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
        """List all profiles available for this chat ID."""
        bot = await self._get_aiogram_bot()
        profiles = self.chat_id_to_profiles.get(int(chat_id), [])
        if not profiles:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ No profiles found for this chat. Ask an admin to register you.",
            )
            return

        active = self.resolve_profile(chat_id)
        lines = []
        for p in profiles:
            profile_cfg = self.config.get("profiles", {}).get(p, {})
            lang = resolve_language_name(
                profile_cfg.get("learning_language", "?"))
            marker = " ◀ active" if p == active else ""
            lines.append(f"  • <b>{self._escape_html(p)}</b> — {self._escape_html(lang)}{marker}")

        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"👤 Your profiles:\n\n"
                + "\n".join(lines) + "\n\n"
                + "Use <code>/switch &lt;name&gt;</code> to change active profile."
            ),
            parse_mode="HTML",
        )

    async def handle_switch(self, chat_id: int, args: str):
        """Switch the active profile for this chat ID."""
        bot = await self._get_aiogram_bot()
        target = args.strip().lower() if args else ""
        profiles = self.chat_id_to_profiles.get(int(chat_id), [])

        if not profiles:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ No profiles found for this chat.",
            )
            return

        # Exact match first, then case-insensitive prefix (only if no exact match)
        matched = None
        for p in profiles:
            if p.lower() == target:
                matched = p
                break
        if not matched:
            for p in profiles:
                if p.lower().startswith(target):
                    matched = p
                    break

        if not matched:
            names = ", ".join(f"<code>{p}</code>" for p in profiles)
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"❌ Unknown profile. Available:\n\n"
                    f"{names}\n\n"
                    f"Usage: <code>/switch &lt;name&gt;</code>"
                ),
                parse_mode="HTML",
            )
            return

        self.select_profile(chat_id, matched)
        lang = resolve_language_name(
            self.config.get("profiles", {}).get(matched, {}).get(
                "learning_language", "?"))
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ Switched to <b>{self._escape_html(matched)}</b> ({self._escape_html(lang)})\n\n"
                f"Tutor messages will now use this profile."
            ),
            parse_mode="HTML",
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

        # Suppress aiogram's verbose INFO logs for callback queries
        logging.getLogger("aiogram.event").setLevel(logging.WARNING)
        logging.getLogger("aiogram.dispatcher").setLevel(logging.WARNING)

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

        # ── Study integration (flashcards + quiz) ────────────────
        try:
            from flashcards import StudyHandler, DEFAULT_CARD_COUNT, DEFAULT_QUIZ_COUNT
            self.study_handler = StudyHandler(
                config=self.config, telegram_bot=self
            )
            logger.info("Study handler initialised (flashcards + quiz)")
        except Exception as e:
            logger.warning("Study module not available: %s", e)
            DEFAULT_CARD_COUNT = 10
            DEFAULT_QUIZ_COUNT = 10

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

        @dp.message(Command("status"))
        async def cmd_status(message: types.Message):
            await self.handle_status(message.chat.id)

        @dp.message(Command("chatid"))
        async def cmd_chatid(message: types.Message):
            await self.handle_chat_id(message.chat.id)

        @dp.message(Command("profiles"))
        async def cmd_profiles(message: types.Message):
            await self.handle_profiles(message.chat.id)

        @dp.message(Command("switch"))
        async def cmd_switch(message: types.Message):
            args = message.text.split(maxsplit=1)
            subcommand = args[1] if len(args) > 1 else ""
            if not subcommand.strip():
                await self.handle_profiles(message.chat.id)
            else:
                await self.handle_switch(message.chat.id, subcommand)

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
            count = DEFAULT_CARD_COUNT
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
            count = DEFAULT_QUIZ_COUNT
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

        # ── Study callback queries (flashcards + quiz) ───────────
        @dp.callback_query(lambda c: c.data and (c.data.startswith("fc:") or c.data.startswith("qz:")))
        async def study_callback(callback_query: types.CallbackQuery):
            if self.study_handler is None:
                return

            # Handle post-quiz result buttons (retry_missed / new_quiz)
            # These arrive after the main session is destroyed
            data = callback_query.data
            if data.startswith("qz:") and self.study_handler:
                parts = data.split(":", 4)
                if len(parts) >= 3:
                    result_chat_id = int(parts[1])
                    # Handle both old format (no token) and new format (with token)
                    action = parts[3] if len(parts) >= 4 else parts[2]
                    # Check for a results-mode session
                    result_session = self.study_handler._sessions.get(result_chat_id)
                    if result_session and result_session.get("mode") == "quiz_results":
                        # Always use currently active profile (respects /switch)
                        active_profile = self.resolve_profile(result_chat_id)
                        if not active_profile:
                            await callback_query.answer("⚠️ No profile found")
                            return

                        if action == "retry_missed":
                            await callback_query.answer()
                            from flashcards import VocabLoader

                            missed = result_session.get("missed_words", [])
                            if missed:
                                # Get learning language from current active profile config
                                profile_cfg = self.config.get("profiles", {}).get(
                                    active_profile, {})
                                lang = profile_cfg.get("learning_language", "de")
                                loader = VocabLoader(
                                    profile=active_profile,
                                    learning_language=lang,
                                )
                                all_entries = loader.all_entries()
                                questions = self.study_handler._build_questions(
                                    missed, all_entries)
                                self.study_handler._sessions[result_chat_id] = {
                                    "mode": "quiz",
                                    "profile": active_profile,
                                    "questions": questions,
                                    "index": 0,
                                    "created_at": time.time(),
                                    "message_id": None,
                                    "score": 0,
                                    "answered": False,
                                    "missed_words": [],
                                    "answer_log": [],
                                }
                                await self.study_handler._render_question(result_chat_id)
                                return
                        elif action == "new_quiz":
                            await callback_query.answer()
                            self.study_handler._end_session(result_chat_id)
                            await self.study_handler.start_quiz(
                                chat_id=result_chat_id,
                                profile_name=active_profile,
                                count=result_session.get("questions_count", DEFAULT_QUIZ_COUNT),
                            )
                            return

                        elif action == "to_flashcards":
                            await callback_query.answer()
                            self.study_handler._end_session(result_chat_id)
                            await self.study_handler.start_flashcards(
                                chat_id=result_chat_id,
                                profile_name=result_session["profile"],
                                count=DEFAULT_CARD_COUNT,
                            )
                            return

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
