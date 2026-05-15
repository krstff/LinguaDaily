#!/usr/bin/env python3
"""
Flashcard / Quiz module for LinguaDaily Telegram bot.

Two study modes:
  1. Flashcards — browse vocabulary with reveal/hide and ← → navigation
  2. Quiz       — multiple-choice (forward & reverse) with instant feedback,
                  auto-advance, score tracking, and missed-word review

Both use spaced-repetition word selection from the per-profile vocab file.

Usage (import):
    from src.flashcards import StudyHandler
    handler = StudyHandler(config, telegram_bot_instance)
    await handler.start_flashcards(chat_id, profile_name)
    await handler.start_quiz(chat_id, profile_name)

Integrates with TelegramBot by registering its command + callback handlers.
"""

import asyncio
import csv
import html
import logging
import random
import secrets
import time
from datetime import date
from typing import Optional

from config import (
    DEFAULT_LEARNING_LANGUAGE,
    FLASHCARD_DEFAULT_CARD_COUNT,
    FLASHCARD_DEFAULT_QUIZ_COUNT,
    FLASHCARD_REVIEW_COOLDOWN_DAYS,
    FLASHCARD_SESSION_TIMEOUT_SECS,
    FLASHCARD_QUIZ_AUTO_ADVANCE_SECS,
    FLASHCARD_QUIZ_DISTRACTORS,
    PROJECT_DIR,
    resolve_language_name,
)

logger = logging.getLogger(__name__)


# ── Vocabulary Loader ───────────────────────────────────────────────

class VocabLoader:
    """Reads the per-profile vocabulary CSV and supports spaced-repetition queries."""

    def __init__(self, profile: str, learning_language: str):
        self.profile = profile
        self.learning_language = learning_language
        self._data_dir = PROJECT_DIR / "data" / profile
        self._csv_path = self._data_dir / "vocabulary.csv"

    # ── Parsing ───────────────────────────────────────────────────

    def _parse_vocab(self) -> list[dict]:
        """Parse CSV vocabulary file into list of entry dicts."""
        if not self._csv_path.exists():
            return []
        with open(self._csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [{
                "word": row.get("word", "").strip(),
                "meaning": row.get("meaning", "").strip(),
                "frequency": int(row.get("frequency", 1) or 1),
                "last_seen": row.get("last_seen", "").strip() or None,
            } for row in reader]

    def all_entries(self) -> list[dict]:
        """Return all vocabulary entries (for distractor generation)."""
        return self._parse_vocab()

    # ── Writing / exposure tracking ───────────────────────────────

    def record_exposure(self, words: list[str]):
        """Increment frequency and update last_seen for a set of words.

        Called after a flashcard or quiz session to track which words the user saw.
        """
        if not self._csv_path.exists():
            return

        entries = self._parse_vocab()
        today = date.today().isoformat()
        word_map = {e["word"]: e for e in entries}

        updated = False
        for w in words:
            if w in word_map:
                word_map[w]["frequency"] += 1
                word_map[w]["last_seen"] = today
                updated = True

        if updated:
            with open(self._csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["word", "meaning", "frequency", "last_seen"])
                writer.writeheader()
                writer.writerows(entries)

    # ── Spaced-repetition selection ───────────────────────────────

    def pick_review_words(self, count: int = FLASHCARD_DEFAULT_CARD_COUNT) -> list[dict]:
        """Select words for review using spaced-repetition heuristics.

        Priority order:
          1. Words NOT seen in the last FLASHCARD_REVIEW_COOLDOWN_DAYS days (oldest first)
          2. Words with lowest frequency (seen fewest times)
          3. Random shuffle within each tier to avoid deterministic ordering
        """
        entries = self._parse_vocab()
        if not entries:
            return []

        today = date.today()

        # Split into "due" (overdue for review) and "not due yet"
        due: list[dict] = []
        not_due: list[dict] = []

        for entry in entries:
            if entry["last_seen"]:
                try:
                    last = date.fromisoformat(entry["last_seen"])
                    days_since = (today - last).days
                except ValueError:
                    days_since = 999  # treat unparseable dates as very old
            else:
                days_since = 999  # never seen in a review session

            if days_since >= FLASHCARD_REVIEW_COOLDOWN_DAYS:
                due.append(entry)
            else:
                not_due.append(entry)

        # Sort "due" by last_seen ascending (oldest first), then by frequency asc
        due.sort(key=lambda e: (e["last_seen"] or "", e["frequency"]))
        # Shuffle "not due" for variety, weighted toward low frequency
        random.shuffle(not_due)
        not_due.sort(key=lambda e: e["frequency"])

        # Pick from due first, fill remaining from not_due
        selected = due[:count]
        remaining = count - len(selected)
        if remaining > 0:
            selected.extend(not_due[:remaining])

        return selected


# ── Study Handler (Flashcards + Quiz) ───────────────────────────────

class StudyHandler:
    """Manages interactive flashcard and quiz sessions inside Telegram.

    Session types are stored in mode: "flashcards" | "quiz"
    All session state lives in self._sessions[chat_id].
    """

    def __init__(self, config: dict, telegram_bot):
        self.config = config
        self.bot = telegram_bot  # TelegramBot instance

        # Per-user session state: chat_id → session dict
        self._sessions: dict[int, dict] = {}

    # ── Session lifecycle ─────────────────────────────────────────

    def _get_session(self, chat_id: int) -> Optional[dict]:
        """Return the active session or None if expired/missing."""
        session = self._sessions.get(chat_id)
        if not session:
            return None
        if time.time() - session["created_at"] > FLASHCARD_SESSION_TIMEOUT_SECS:
            del self._sessions[chat_id]
            return None
        # Refresh timeout
        session["created_at"] = time.time()
        return session

    def _next_generation(self) -> int:
        """Incrementing counter to invalidate stale async tasks."""
        self._gen_counter = getattr(self, "_gen_counter", 0) + 1
        return self._gen_counter

    def _end_session(self, chat_id: int):
        """Remove an active session, cancelling any pending auto-advance task."""
        session = self._sessions.pop(chat_id, None)
        if session:
            task = session.get("_auto_advance_task")
            if task and not task.done():
                task.cancel()
            # Record exposure for all words shown in this session
            self._record_session_exposure(session)

    def _record_session_exposure(self, session: dict):
        """Update vocabulary frequency/last_seen for words seen in a session."""
        mode = session.get("mode")
        profile = session.get("profile")
        if not profile:
            return

        words: list[str] = []
        if mode == "flashcards":
            words = [w["word"] for w in session.get("words", [])]
        elif mode == "quiz":
            words = [q["entry"]["word"] for q in session.get("questions", [])]

        if not words:
            return

        # Get learning language from config
        profile_cfg = self.config.get("profiles", {}).get(profile, {})
        lang = profile_cfg.get("learning_language", DEFAULT_LEARNING_LANGUAGE)

        try:
            loader = VocabLoader(profile=profile, learning_language=lang)
            loader.record_exposure(words)
        except Exception as e:
            logger.debug("Failed to record exposure for %s: %s", profile, e)

    async def _send(self, chat_id: int, text: str, reply_markup=None):
        """Helper: send a message via the Telegram bot."""
        bot = await self.bot._get_aiogram_bot()
        return await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _edit(self, chat_id: int, message_id: int, text: str, reply_markup=None):
        """Helper: edit a message via the Telegram bot."""
        bot = await self.bot._get_aiogram_bot()
        return await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    # ── Public API: Flashcards ────────────────────────────────────

    async def start_flashcards(self, chat_id: int, profile_name: str, count: int = FLASHCARD_DEFAULT_CARD_COUNT):
        """Start a flashcard review session."""
        # Save old message_id before ending session so we can delete it
        old_session = self._sessions.get(chat_id)
        old_msg_id = old_session.get("message_id") if old_session else None
        self._end_session(chat_id)  # close any existing session

        profile_cfg = self.config.get("profiles", {}).get(profile_name, {})
        lang = profile_cfg.get("learning_language", DEFAULT_LEARNING_LANGUAGE)
        loader = VocabLoader(profile=profile_name, learning_language=lang)
        words = loader.pick_review_words(count=count)

        if not words:
            await self._send(chat_id, "📚 No vocabulary found yet. Complete a lesson first!")
            return

        self._sessions[chat_id] = {
            "mode": "flashcards",
            "profile": profile_name,
            "words": words,
            "index": 0,
            "created_at": time.time(),
            "message_id": None,
            "revealed": False,
            "_token": secrets.token_urlsafe(4)[:6],  # short token for stale-button detection
        }

        # Delete old quiz/flashcard message if one was left behind
        if old_msg_id:
            try:
                from aiogram.methods import DeleteMessage
                bot = await self.bot._get_aiogram_bot()
                await bot(DeleteMessage(chat_id=chat_id, message_id=old_msg_id))
            except Exception as e:
                logger.debug("Failed to delete old message: %s", e)

        await self._render_flashcard(chat_id)

    # ── Public API: Quiz ──────────────────────────────────────────

    async def start_quiz(self, chat_id: int, profile_name: str, count: int = FLASHCARD_DEFAULT_QUIZ_COUNT):
        """Start a quiz session.

        The quiz alternates between forward (word→meaning) and reverse
        (meaning→word) questions for balanced practice.
        """
        # Save old message_id before ending session so we can delete it
        old_session = self._sessions.get(chat_id)
        old_msg_id = old_session.get("message_id") if old_session else None
        self._end_session(chat_id)  # close any existing session

        profile_cfg = self.config.get("profiles", {}).get(profile_name, {})
        lang = profile_cfg.get("learning_language", DEFAULT_LEARNING_LANGUAGE)
        loader = VocabLoader(profile=profile_name, learning_language=lang)
        words = loader.pick_review_words(count=count)
        all_entries = loader.all_entries()

        if len(words) < 2:
            await self._send(
                chat_id,
                "📚 Not enough vocabulary for a quiz. Need at least 2 words — complete more lessons!",
            )
            return

        # Build question queue: alternate forward / reverse
        questions = self._build_questions(words, all_entries)

        self._sessions[chat_id] = {
            "mode": "quiz",
            "profile": profile_name,
            "questions": questions,      # list of {entry, direction, choices, correct_idx}
            "index": 0,                  # current question index
            "created_at": time.time(),
            "message_id": None,
            "score": 0,
            "answered": False,           # has user answered current question?
            "missed_words": [],          # entries got wrong (for review)
            "answer_log": [],            # list of {entry, correct: bool} for summary
            "_auto_advance_task": None,  # track task to cancel on manual next
            "generation": self._next_generation(),  # bump to invalidate stale tasks
            "_token": secrets.token_urlsafe(4)[:6],  # short token for stale-button detection
        }

        # Delete old quiz/flashcard message if one was left behind
        if old_msg_id:
            try:
                from aiogram.methods import DeleteMessage
                bot = await self.bot._get_aiogram_bot()
                await bot(DeleteMessage(chat_id=chat_id, message_id=old_msg_id))
            except Exception as e:
                logger.debug("Failed to delete old quiz message: %s", e)

        await self._render_question(chat_id)

    # ── Callback router ───────────────────────────────────────────

    async def handle_callback(self, callback_query):
        """Process inline-button callback queries.

        Callback data format:
          fc:<chat_id>:<token>:<action>   — flashcard actions
          qz:<chat_id>:<token>:<answer>   — quiz answer selection
          qz:<chat_id>:<token>:<action>   — quiz control (skip, stop)
        """
        data = callback_query.data

        # ── Flashcard callbacks ───────────────────────────────────
        if data.startswith("fc:"):
            return await self._handle_flashcard_callback(callback_query, data)

        # ── Quiz callbacks ────────────────────────────────────────
        if data.startswith("qz:"):
            return await self._handle_quiz_callback(callback_query, data)

        return False  # not our callback

    # ── Flashcard internals ───────────────────────────────────────

    async def _handle_flashcard_callback(self, callback_query, data: str):
        # Format: fc:<chat_id>:<token>:<action>
        parts = data.split(":", 4)
        if len(parts) < 4:
            return False

        chat_id = int(parts[1])
        token = parts[2]
        action = parts[3]

        if chat_id != callback_query.message.chat.id:
            await callback_query.answer("⚠️ Not your session")
            return True

        session = self._get_session(chat_id)
        if not session or session["mode"] != "flashcards":
            await callback_query.answer("⚠️ Session expired. Start a new one with /flashcards")
            return True

        # Verify token matches — prevents stale buttons from old sessions
        if session.get("_token") != token:
            await callback_query.answer()
            return True

        if action == "next":
            self._flashcard_navigate(session, +1)
            await callback_query.answer()
            await self._render_flashcard(chat_id)

        elif action == "prev":
            self._flashcard_navigate(session, -1)
            await callback_query.answer()
            await self._render_flashcard(chat_id)

        elif action == "reveal":
            session["revealed"] = True
            await callback_query.answer()
            await self._render_flashcard(chat_id)

        elif action == "hide":
            session["revealed"] = False
            await callback_query.answer()
            await self._render_flashcard(chat_id)

        elif action == "stop":
            await callback_query.answer()
            # Delete the flashcard message to avoid clutter
            msg_id = session.get("message_id")
            self._end_session(chat_id)
            if msg_id:
                try:
                    from aiogram.methods import DeleteMessage
                    bot = await self.bot._get_aiogram_bot()
                    await bot(DeleteMessage(chat_id=chat_id, message_id=msg_id))
                except Exception as e:
                    logger.debug("Failed to delete flashcard message: %s", e)
            words = session["words"]
            idx = session["index"] + 1
            await self._send(
                chat_id,
                f"📚 Flashcard session ended.\n"
                f"Reviewed {idx} / {len(words)} cards.",
            )

        return True

    def _flashcard_navigate(self, session: dict, direction: int):
        total = len(session["words"])
        session["index"] = (session["index"] + direction) % total
        session["revealed"] = False  # always start hidden

    async def _render_flashcard(self, chat_id: int):
        session = self._sessions.get(chat_id)
        if not session:
            return

        word_entry = session["words"][session["index"]]
        total = len(session["words"])
        idx = session["index"] + 1
        revealed = session.get("revealed", False)

        card_text, reply_markup = self._build_flashcard_message(
            word_entry=word_entry, index=idx, total=total,
            chat_id=chat_id, revealed=revealed,
            token=session.get("_token", ""),
        )

        msg_id = session.get("message_id")
        if msg_id:
            try:
                await self._edit(chat_id, msg_id, card_text, reply_markup)
                return
            except Exception as e:
                logger.debug("Flashcard edit failed (%s), sending new", e)

        msg = await self._send(chat_id, card_text, reply_markup)
        session["message_id"] = msg.message_id

    def _build_flashcard_message(
        self, word_entry: dict, index: int, total: int,
        chat_id: int, revealed: bool, token: str = "",
    ) -> tuple[str, "InlineKeyboardMarkup"]:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        word = html.escape(word_entry["word"])
        meaning = html.escape(word_entry["meaning"] or "(unknown)")
        freq = word_entry.get("frequency", 1)
        last_seen = word_entry.get("last_seen") or "never"

        if revealed:
            meaning_line = f"<b>Meaning:</b> {meaning}"
        else:
            meaning_line = "<i>Press 👁 Reveal to see the meaning</i>"

        card_text = (
            f"📚 <b>Flashcard {index}/{total}</b>\n\n"
            f"<b>{word}</b>\n\n"
            f"{meaning_line}\n\n"
            f"<i>Seen {freq}x · Last: {last_seen}</i>"
        )

        reveal_btn = "🔙 Hide" if revealed else "👁 Reveal"
        reveal_act = "hide" if revealed else "reveal"

        keyboard = [
            [
                InlineKeyboardButton(text="⏹ Stop", callback_data=f"fc:{chat_id}:{token}:stop"),
            ],
            [
                InlineKeyboardButton(text="⬅️ Prev", callback_data=f"fc:{chat_id}:{token}:prev"),
                InlineKeyboardButton(text=reveal_btn, callback_data=f"fc:{chat_id}:{token}:{reveal_act}"),
                InlineKeyboardButton(text="Next ➡️", callback_data=f"fc:{chat_id}:{token}:next"),
            ],
        ]

        return card_text, InlineKeyboardMarkup(inline_keyboard=keyboard)

    # ── Quiz internals ────────────────────────────────────────────

    def _build_questions(self, words: list[dict], all_entries: list[dict]) -> list[dict]:
        """Build a quiz question queue from selected words.

        Alternates forward (word→meaning) and reverse (meaning→word).
        Each question has 4 choices: 1 correct + 3 distractors.
        """
        questions = []
        # Build lookup sets for fast distractor picking
        all_meanings = [e["meaning"] for e in all_entries if e["meaning"]]
        all_words = [e["word"] for e in all_entries]

        for i, entry in enumerate(words):
            # Alternate direction: even=forward, odd=reverse
            direction = "forward" if i % 2 == 0 else "reverse"

            if direction == "forward":
                # Question: "What does X mean?"  Choices: meanings
                correct_answer = entry["meaning"]
                distractors = self._pick_distractors(
                    pool=all_meanings,
                    exclude={correct_answer},
                    count=FLASHCARD_QUIZ_DISTRACTORS,
                )
                choices = self._shuffle_with_correct(correct_answer, distractors)

            else:
                # Question: "How do you say X?"  Choices: words
                correct_answer = entry["word"]
                distractors = self._pick_distractors(
                    pool=all_words,
                    exclude={correct_answer},
                    count=FLASHCARD_QUIZ_DISTRACTORS,
                )
                choices = self._shuffle_with_correct(correct_answer, distractors)

            correct_idx = choices.index(correct_answer)

            questions.append({
                "entry": entry,
                "direction": direction,
                "choices": choices,       # list of 4 strings
                "correct_idx": correct_idx,
            })

        return questions

    def _pick_distractors(self, pool: list[str], exclude: set[str], count: int) -> list[str]:
        """Pick random distractors from a pool, excluding given values."""
        available = [d for d in pool if d not in exclude]
        # Deduplicate while preserving randomness
        available = list(dict.fromkeys(available))  # unique, preserve order
        return random.sample(available, min(count, len(available)))

    def _shuffle_with_correct(self, correct: str, distractors: list[str]) -> list[str]:
        """Combine correct answer + distractors and shuffle."""
        choices = [correct] + distractors
        random.shuffle(choices)
        return choices

    async def _handle_quiz_callback(self, callback_query, data: str):
        # Format: qz:<chat_id>:<token>:<action_or_idx>
        parts = data.split(":", 4)
        if len(parts) < 4:
            return False

        chat_id = int(parts[1])
        token = parts[2]
        action_or_idx = parts[3]

        if chat_id != callback_query.message.chat.id:
            await callback_query.answer("⚠️ Not your session")
            return True

        session = self._get_session(chat_id)
        if not session or session["mode"] != "quiz":
            await callback_query.answer("⚠️ Session expired. Start a new one with /quiz")
            return True

        # Verify token matches — prevents stale buttons from old sessions
        if session.get("_token") != token:
            await callback_query.answer()
            return True

        # ── Control actions ───────────────────────────────────────
        if action_or_idx == "skip":
            session["answered"] = False
            session["answer_log"].append({
                "entry": session["questions"][session["index"]]["entry"],
                "correct": False,  # skipped counts as miss
            })
            session["missed_words"].append(
                session["questions"][session["index"]]["entry"])
            await self._quiz_next(chat_id)
            return True

        if action_or_idx == "stop":
            await callback_query.answer()
            # Cancel any pending auto-advance and end session
            old_task = session.pop("_auto_advance_task", None)
            if old_task and not old_task.done():
                old_task.cancel()
            # Delete the quiz message to avoid clutter
            msg_id = session.get("message_id")
            self._end_session(chat_id)
            if msg_id:
                try:
                    from aiogram.methods import DeleteMessage
                    bot = await self.bot._get_aiogram_bot()
                    await bot(DeleteMessage(chat_id=chat_id, message_id=msg_id))
                except Exception as e:
                    logger.debug("Failed to delete quiz message: %s", e)
            return True

        if action_or_idx == "retry_missed":
            await callback_query.answer()
            missed = session.pop("missed_words", [])
            if missed:
                # Start a new quiz with just the missed words
                profile_cfg = self.config.get("profiles", {}).get(session["profile"], {})
                lang = profile_cfg.get("learning_language", DEFAULT_LEARNING_LANGUAGE)
                all_entries = VocabLoader(
                    profile=session["profile"], learning_language=lang
                ).all_entries()
                questions = self._build_questions(missed, all_entries)
                session.update({
                    "questions": questions,
                    "index": 0,
                    "score": 0,
                    "answered": False,
                    "missed_words": [],
                    "answer_log": [],
                    "message_id": None,
                })
                await self._render_question(chat_id)
            else:
                await self._send(chat_id, "🎉 No missed words to retry!")
            return True

        if action_or_idx == "new_quiz":
            await callback_query.answer()
            self._end_session(chat_id)
            await self.start_quiz(
                chat_id=chat_id,
                profile_name=session["profile"],
                count=len(session.get("questions", [])),
            )
            return True

        # ── Answer selection (numeric index) ──────────────────────
        try:
            answer_idx = int(action_or_idx)
        except ValueError:
            await callback_query.answer()
            return True

        if session.get("answered"):
            # Already answered this question — ignore extra taps
            await callback_query.answer()
            return True

        # Dismiss Telegram's loading indicator immediately
        await callback_query.answer()

        session["answered"] = True

        question = session["questions"][session["index"]]
        is_correct = (answer_idx == question["correct_idx"])

        if is_correct:
            session["score"] += 1
        else:
            session["missed_words"].append(question["entry"])

        session["answer_log"].append({
            "entry": question["entry"],
            "correct": is_correct,
        })

        # Show feedback (highlight correct/wrong buttons)
        await self._render_question_feedback(chat_id, answer_idx)

        # Auto-advance after a short delay
        # Cancel any pending auto-advance from previous question
        old_task = session.pop("_auto_advance_task", None)
        if old_task and not old_task.done():
            old_task.cancel()

        task = asyncio.create_task(self._quiz_auto_advance(chat_id))
        session["_auto_advance_task"] = task
        return True

    async def _render_question(self, chat_id: int):
        """Render the current quiz question (fresh, no feedback)."""
        session = self._sessions.get(chat_id)
        if not session:
            return

        question = session["questions"][session["index"]]
        total = len(session["questions"])
        q_num = session["index"] + 1

        q_text, reply_markup = self._build_question_message(
            question=question,
            q_num=q_num,
            total=total,
            score=session["score"],
            chat_id=chat_id,
            feedback=None,
            answer_log=session.get("answer_log"),
            token=session.get("_token", ""),
        )

        msg_id = session.get("message_id")
        if msg_id:
            try:
                await self._edit(chat_id, msg_id, q_text, reply_markup)
                return
            except Exception as e:
                logger.debug("Quiz edit failed (%s), sending new", e)

        msg = await self._send(chat_id, q_text, reply_markup)
        session["message_id"] = msg.message_id

    async def _render_question_feedback(self, chat_id: int, chosen_idx: int):
        """Re-render the question with feedback (correct/wrong highlighting)."""
        session = self._sessions.get(chat_id)
        if not session:
            return

        question = session["questions"][session["index"]]
        total = len(session["questions"])
        q_num = session["index"] + 1

        correct = question["correct_idx"]
        is_correct = (chosen_idx == correct)

        q_text, reply_markup = self._build_question_message(
            question=question,
            q_num=q_num,
            total=total,
            score=session["score"],
            chat_id=chat_id,
            feedback={
                "chosen": chosen_idx,
                "correct": correct,
                "is_correct": is_correct,
            },
            answer_log=session.get("answer_log"),
            token=session.get("_token", ""),
        )

        try:
            msg_id = session["message_id"]
            await self._edit(chat_id, msg_id, q_text, reply_markup)
        except Exception as e:
            logger.debug("Quiz feedback edit failed (%s)", e)

    def _build_question_message(
        self, question: dict, q_num: int, total: int, score: int,
        chat_id: int, feedback: Optional[dict],
        answer_log: Optional[list] = None, token: str = "",
    ) -> tuple[str, "InlineKeyboardMarkup"]:
        """Build the quiz question text and inline keyboard.

        Parameters
        ----------
        feedback : dict or None
            If provided: {chosen, correct, is_correct} — buttons are highlighted.
            If None: fresh question, all buttons active.
        """
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        entry = question["entry"]
        direction = question["direction"]
        choices = question["choices"]

        # ── Question text ─────────────────────────────────────────
        if direction == "forward":
            prompt_word = html.escape(entry["word"])
            question_label = "What does this mean?"
        else:
            prompt_meaning = html.escape(entry["meaning"])
            question_label = "How do you say this?"

        # Positional progress bar: each dot = one question in order
        answer_log = answer_log or []
        dots = []
        for i in range(total):
            if i < len(answer_log):
                dots.append("🔵" if answer_log[i]["correct"] else "⚪")
            elif i == q_num - 1:  # current question (0-indexed)
                dots.append("🟡")
            else:
                dots.append("⚫")
        score_bar = "".join(dots)

        if direction == "forward":
            q_text = (
                f"❓ <b>Quiz {q_num}/{total}</b>  Score: {score}/{total}\n\n"
                f"{question_label}\n\n"
                f"<b>{prompt_word}</b>"
            )
        else:
            q_text = (
                f"❓ <b>Quiz {q_num}/{total}</b>  Score: {score}/{total}\n\n"
                f"{question_label}\n\n"
                f"<i>{prompt_meaning}</i>"
            )

        # ── Answer buttons ────────────────────────────────────────
        choice_buttons = []
        for i, choice in enumerate(choices):
            btn_text = html.escape(choice)

            if feedback:
                chosen = feedback["chosen"]
                correct = feedback["correct"]
                is_correct = feedback["is_correct"]

                if i == correct and not is_correct:
                    # Wrong answer chosen — highlight the correct one green
                    btn_text = f"✅ {btn_text}"
                elif i == chosen and not is_correct:
                    # This is the wrong choice the user picked
                    btn_text = f"❌ {btn_text}"
                elif i == chosen and is_correct:
                    btn_text = f"✅ {btn_text}"

            # Letters A/B/C/D for readability
            letter = chr(ord("A") + i) if i < 26 else str(i + 1)
            display_text = f"{letter}) {btn_text}"

            choice_buttons.append(
                InlineKeyboardButton(
                    text=display_text,
                    callback_data=f"qz:{chat_id}:{token}:{i}",
                )
            )

        # 2x2 grid for up to 4 choices
        rows = []
        for j in range(0, len(choice_buttons), 2):
            rows.append(choice_buttons[j:j + 2])

        # Skip / Stop row
        if feedback:
            # After answering: no extra buttons, auto-advance handles it
            pass
        else:
            rows.append([
                InlineKeyboardButton(
                    text="⏭ Skip",
                    callback_data=f"qz:{chat_id}:{token}:skip",
                ),
                InlineKeyboardButton(
                    text="⏹ Stop",
                    callback_data=f"qz:{chat_id}:{token}:stop",
                ),
            ])

        # Append score bar to message
        q_text += f"\n\n{score_bar}"

        reply_markup = InlineKeyboardMarkup(inline_keyboard=rows)
        return q_text, reply_markup

    async def _quiz_auto_advance(self, chat_id: int):
        """Wait briefly then advance to the next question or finish."""
        session = self._sessions.get(chat_id)
        if not session:
            return
        my_gen = session.get("generation")
        try:
            await asyncio.sleep(FLASHCARD_QUIZ_AUTO_ADVANCE_SECS)
        except asyncio.CancelledError:
            return
        # Only advance if this is still the current generation (prevents stale tasks
        # from a stopped quiz from hitting a brand new quiz session)
        session = self._sessions.get(chat_id)
        if not session or session["mode"] != "quiz" or session.get("generation") != my_gen:
            return
        await self._quiz_next(chat_id)

    async def _quiz_next(self, chat_id: int):
        """Advance to the next quiz question or show results."""
        session = self._sessions.get(chat_id)
        if not session or session["mode"] != "quiz":
            return

        if session["index"] + 1 >= len(session["questions"]):
            await self._quiz_finish(chat_id)
        else:
            session["index"] += 1
            session["answered"] = False
            await self._render_question(chat_id)

    async def _quiz_finish(self, chat_id: int):
        """Show quiz results and offer retry options."""
        session = self._sessions.get(chat_id)
        if not session or session["mode"] != "quiz":
            return

        total = len(session["questions"])
        score = session["score"]
        pct = round(score / total * 100) if total > 0 else 0
        missed = session.get("missed_words", [])

        # Emoji based on performance
        if pct == 100:
            emoji = "🏆"
        elif pct >= 80:
            emoji = "🌟"
        elif pct >= 60:
            emoji = "👍"
        elif pct >= 40:
            emoji = "💪"
        else:
            emoji = "📚"

        # Build results text
        result_text = (
            f"{emoji} <b>Quiz Complete!</b>\n\n"
            f"<b>Score:</b> {score}/{total} ({pct}%)\n\n"
        )

        if missed:
            missed_lines = []
            for entry in missed[:10]:  # cap at 10 to fit message limit
                w = html.escape(entry["word"])
                m = html.escape(entry["meaning"] or "(unknown)")
                missed_lines.append(f"  • <b>{w}</b> — {m}")

            result_text += (
                f"<b>Words to review ({len(missed)}):</b>\n"
                + "\n".join(missed_lines) + "\n\n"
            )
        else:
            result_text += "Perfect score! No words to review. 🎉\n\n"

        # Buttons: retry missed, new quiz, flashcards
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        token = secrets.token_urlsafe(4)[:6]
        buttons = []
        if missed:
            buttons.append([
                InlineKeyboardButton(
                    text="🔄 Retry missed",
                    callback_data=f"qz:{chat_id}:{token}:retry_missed",
                ),
            ])
        buttons.append([
            InlineKeyboardButton(
                text="🆕 New quiz",
                callback_data=f"qz:{chat_id}:{token}:new_quiz",
            ),
            InlineKeyboardButton(
                text="📚 Flashcards",
                callback_data=f"qz:{chat_id}:{token}:to_flashcards",
            ),
        ])

        reply_markup = InlineKeyboardMarkup(inline_keyboard=buttons)

        # Send as new message (don't edit — results are a separate screen)
        await self._send(chat_id, result_text, reply_markup)

        # Clean up the session but keep profile for retry/new_quiz
        profile = session["profile"]
        missed_copy = list(missed)
        self._end_session(chat_id)

        # Store minimal state for post-quiz actions (retry_missed / new_quiz)
        self._sessions[chat_id] = {
            "mode": "quiz_results",
            "profile": profile,
            "missed_words": missed_copy,
            "created_at": time.time(),
            "questions_count": total,
            "_token": token,  # prevents stale-button attacks
        }
