"""Tests for src/stats.py — per-profile statistics."""

from datetime import datetime, timedelta, timezone

import pytest


def _mark_day(db, profile, day, title, finished=True):
    """Insert a lesson_log row for a specific date (UTC)."""
    lid = db.log_lesson(profile, title)
    ts = f"{day.isoformat()} 08:00:00"
    db.conn.execute(
        "UPDATE lesson_log SET delivered_at=? WHERE id=?", (ts, lid))
    if finished:
        db.conn.execute(
            "UPDATE lesson_log SET finished_at=? WHERE id=?", (ts, lid))
    db.conn.commit()  # release the write transaction for other connections


@pytest.fixture
def db(tmp_path):
    from src.telegram_bot import ChatHistoryDB
    from src.vocab_db import VocabDB
    path = str(tmp_path / "test.db")
    chat = ChatHistoryDB(path)
    # VocabDB shares the same file
    vocab = VocabDB(path)
    yield path, chat, vocab
    chat.close()
    vocab.close()


CONFIG = {"profiles": {
    "p": {"learning_language": "de", "native_language": "en", "enabled": True},
}}


class TestStreaks:
    def test_empty(self):
        from src.stats import _streaks
        assert _streaks(set()) == (0, 0)

    def test_single_day(self):
        from src.stats import _streaks
        today = datetime.now(timezone.utc).date().isoformat()
        assert _streaks({today}) == (1, 1)

    def test_current_streak_ignores_missing_today(self):
        from src.stats import _streaks
        today = datetime.now(timezone.utc).date()
        days = {(today - timedelta(days=i)).isoformat() for i in range(1, 4)}
        # yesterday, 2d, 3d → current streak 3 (today not finished yet)
        assert _streaks(days) == (3, 3)

    def test_gap_breaks_current(self):
        from src.stats import _streaks
        today = datetime.now(timezone.utc).date()
        days = {(today - timedelta(days=i)).isoformat() for i in (1, 3)}
        # yesterday + 3 days ago, gap at 2d → current 1, best 1
        assert _streaks(days) == (1, 1)

    def test_best_streak_finds_longest_run(self):
        from src.stats import _streaks
        today = datetime.now(timezone.utc).date()
        # run of 5 ending 10 days ago, then 2 days ending yesterday
        days = {(today - timedelta(days=i)).isoformat() for i in range(10, 15)}
        days |= {(today - timedelta(days=i)).isoformat() for i in (1, 2)}
        current, best = _streaks(days)
        assert current == 2
        assert best == 5


class TestProfileStats:
    def test_external_db_not_closed(self, db):
        from src.stats import profile_stats
        path, chat, vocab = db
        s = profile_stats("p", config=CONFIG, db=vocab)
        assert s["lessons"]["delivered"] == 0
        assert vocab.word_count("p") == 0  # externally owned → still usable

    def test_empty_profile(self, db):
        from src.stats import profile_stats
        path, chat, vocab = db
        s = profile_stats("p", config=CONFIG, db_path=path)
        assert s["lessons"]["delivered"] == 0
        assert s["lessons"]["finished"] == 0
        assert s["lessons"]["completion_rate"] is None
        assert s["lessons"]["current_streak"] == 0
        assert s["vocab"]["total_words"] == 0
        assert s["vocab"]["quiz_accuracy"] is None
        assert len(s["recent_activity"]) == 14

    def test_lessons_and_vocab(self, db):
        from src.stats import profile_stats
        path, chat, vocab = db
        today = datetime.now(timezone.utc).date()

        _mark_day(chat, "p", today - timedelta(days=2), "old")
        _mark_day(chat, "p", today - timedelta(days=1), "yesterday", finished=False)
        _mark_day(chat, "p", today, "today")

        vocab.add_words("p", [{"word": "Haus", "meaning": "house"},
                              {"word": "Auto", "meaning": "car"}])
        vocab.record_exposure("p", ["Haus"], outcomes=[
            ("Haus", True), ("Haus", False)])

        s = profile_stats("p", config=CONFIG, db_path=path)
        L, V = s["lessons"], s["vocab"]

        assert L["delivered"] == 3
        assert L["finished"] == 2
        assert L["completion_rate"] == round(100 * 2 / 3, 1)
        # Streak counts FINISHED days: today + 2d ago, but yesterday was
        # only delivered → streak is 1 (chain broken by yesterday)
        assert L["current_streak"] == 1
        assert L["best_streak"] == 1
        assert L["first_lesson"] == (today - timedelta(days=2)).isoformat()
        assert L["last_lesson"] == today.isoformat()

        assert V["total_words"] == 2
        assert V["total_exposures"] == 3  # Haus 1+1, Auto 1
        assert V["quiz_attempts"] == 2
        assert V["quiz_correct"] == 1
        assert V["quiz_accuracy"] == 50.0
        # avg mastery: Haus (1 correct, 1 wrong) = 0.5, Auto never quizzed = 0
        assert V["avg_mastery"] == pytest.approx(0.25, abs=0.001)

        # activity chart has today's row with 1 delivered + 1 finished
        today_row = [d for d in s["recent_activity"]
                     if d["date"] == today.isoformat()][0]
        assert today_row["delivered"] == 1
        assert today_row["finished"] == 1

        assert len(s["recent_lessons"]) == 3

        # top words ordered by frequency (Haus seen twice, Auto once)
        assert s["top_words"][0]["word"] == "Haus"
        assert s["top_words"][0]["frequency"] == 2
        assert s["top_words"][1]["word"] == "Auto"
