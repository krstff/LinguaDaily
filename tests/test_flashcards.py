"""Tests for src/flashcards.py — post-quiz results actions + VocabLoader DB ownership."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest


def _cb(data: str, chat_id: int) -> MagicMock:
    """Build a fake aiogram callback query."""
    cb = MagicMock()
    cb.data = data
    cb.message.chat.id = chat_id
    cb.answer = AsyncMock()
    return cb


@pytest.fixture
def study_env(tmp_path, monkeypatch):
    """StudyHandler + fake bot + a shared vocab DB pointed at tmp_path.

    The shared DB path is monkeypatched so the handler's VocabLoader
    (which uses the process-wide shared instance by default) never
    touches the real data/chat_history.db.
    """
    # Patch the top-level vocab_db module — that's what flashcards.py
    # imports (src/ is on sys.path), not the src.vocab_db package form.
    import vocab_db
    from src.flashcards import StudyHandler

    monkeypatch.setattr(vocab_db, "DEFAULT_DB_PATH", tmp_path / "shared.db")
    db = vocab_db.get_shared_db()
    db.add_words("krystof", [
        {"word": "Haus", "meaning": "house"},
        {"word": "Auto", "meaning": "car"},
        {"word": "Tisch", "meaning": "table"},
    ])

    config = {"profiles": {"krystof": {"learning_language": "de"}}}
    bot = MagicMock()
    aiogram_bot = AsyncMock()
    bot._get_aiogram_bot = AsyncMock(return_value=aiogram_bot)

    handler = StudyHandler(config, bot)
    yield db, bot, aiogram_bot, handler


def _results_session(token="abc123", missed=None, questions_count=3):
    return {
        "mode": "quiz_results",
        "profile": "krystof",
        "missed_words": missed if missed is not None else [
            {"word": "Haus", "meaning": "house"}],
        "created_at": time.time(),
        "questions_count": questions_count,
        "_token": token,
    }


class TestResultsActions:
    """Post-quiz result buttons (moved from telegram_bot.study_callback)."""

    @pytest.mark.asyncio
    async def test_retry_missed_rebuilds_quiz(self, study_env):
        db, bot, aiogram, handler = study_env
        handler._sessions[1] = _results_session()
        cb = _cb("qz:1:abc123:retry_missed", 1)
        assert await handler.handle_callback(cb) is True

        session = handler._sessions[1]
        assert session["mode"] == "quiz"
        assert session["profile"] == "krystof"
        assert len(session["questions"]) == 1
        assert session["questions"][0]["entry"]["word"] == "Haus"
        assert session["score"] == 0
        cb.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_retry_missed_token_mismatch_ends_session(self, study_env):
        db, bot, aiogram, handler = study_env
        handler._sessions[1] = _results_session()
        cb = _cb("qz:1:WRONG:retry_missed", 1)
        assert await handler.handle_callback(cb) is True
        assert 1 not in handler._sessions
        cb.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_new_quiz_starts_full_quiz(self, study_env):
        db, bot, aiogram, handler = study_env
        handler._sessions[1] = _results_session(questions_count=3)
        cb = _cb("qz:1:abc123:new_quiz", 1)
        assert await handler.handle_callback(cb) is True

        session = handler._sessions[1]
        assert session["mode"] == "quiz"
        assert len(session["questions"]) == 3  # all 3 words

    @pytest.mark.asyncio
    async def test_to_flashcards_starts_flashcards(self, study_env):
        db, bot, aiogram, handler = study_env
        handler._sessions[1] = _results_session()
        cb = _cb("qz:1:abc123:to_flashcards", 1)
        assert await handler.handle_callback(cb) is True
        assert handler._sessions[1]["mode"] == "flashcards"

    @pytest.mark.asyncio
    async def test_unknown_action_on_results_is_noop(self, study_env):
        db, bot, aiogram, handler = study_env
        handler._sessions[1] = _results_session()
        cb = _cb("qz:1:abc123:stop", 1)
        assert await handler.handle_callback(cb) is True
        assert handler._sessions[1]["mode"] == "quiz_results"  # untouched


class TestActiveQuizStillWorks:
    """Regression: active-quiz callbacks are unaffected by the results move."""

    def _active_session(self, handler, token="abc123"):
        import vocab_db
        entries = vocab_db.get_shared_db().get_entries("krystof")
        questions = handler._build_questions(entries[:2], entries)
        handler._sessions[1] = {
            "mode": "quiz",
            "profile": "krystof",
            "questions": questions,
            "index": 0,
            "created_at": time.time(),
            "message_id": None,
            "score": 0,
            "answered": False,
            "missed_words": [],
            "answer_log": [],
            "_auto_advance_task": None,
            "generation": 1,
            "_token": token,
        }

    @pytest.mark.asyncio
    async def test_answer_is_recorded(self, study_env):
        db, bot, aiogram, handler = study_env
        self._active_session(handler)
        cb = _cb("qz:1:abc123:0", 1)
        assert await handler.handle_callback(cb) is True
        session = handler._sessions[1]
        assert session["answered"] is True
        assert len(session["answer_log"]) == 1
        # Cancel the pending auto-advance so the test loop can close
        task = session.get("_auto_advance_task")
        if task:
            task.cancel()

    @pytest.mark.asyncio
    async def test_stop_ends_session(self, study_env):
        db, bot, aiogram, handler = study_env
        self._active_session(handler)
        cb = _cb("qz:1:abc123:stop", 1)
        assert await handler.handle_callback(cb) is True
        assert 1 not in handler._sessions

    @pytest.mark.asyncio
    async def test_results_action_string_ignored_in_active_quiz(self, study_env):
        """retry_missed/new_quiz are no-op in an active quiz (buttons
        don't exist there) — must not crash or mutate the session."""
        db, bot, aiogram, handler = study_env
        self._active_session(handler)
        cb = _cb("qz:1:abc123:retry_missed", 1)
        assert await handler.handle_callback(cb) is True
        assert handler._sessions[1]["mode"] == "quiz"
        assert handler._sessions[1]["answered"] is False


class TestVocabLoaderOwnership:
    def test_external_db_not_closed_by_loader(self, tmp_path):
        from src.flashcards import VocabLoader
        from src.vocab_db import VocabDB
        db = VocabDB(str(tmp_path / "external.db"))
        loader = VocabLoader("krystof", "de", db=db)
        loader.close()
        assert db.word_count("krystof") == 0  # externally owned → open
        db.close()

    def test_explicit_db_path_is_owned(self, tmp_path):
        from src.flashcards import VocabLoader
        db_path = str(tmp_path / "own.db")
        loader = VocabLoader("krystof", "de", db_path=db_path)
        loader.close()
        # connection closed → new instance sees an empty db
        loader2 = VocabLoader("krystof", "de", db_path=db_path)
        assert loader2._db.word_count("krystof") == 0
        loader2.close()

    def test_default_uses_shared_db(self, tmp_path, monkeypatch):
        import vocab_db
        from src.flashcards import VocabLoader
        monkeypatch.setattr(vocab_db, "DEFAULT_DB_PATH", tmp_path / "shared.db")
        loader = VocabLoader("krystof", "de")
        assert loader._db is vocab_db.get_shared_db()
        loader.close()
        assert vocab_db.get_shared_db() is loader._db  # not closed
