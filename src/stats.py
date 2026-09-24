#!/usr/bin/env python3
"""
Stats module for LinguaDaily.

Computes per-profile learning statistics from the shared SQLite store
(data/chat_history.db):
  - lessons:  delivered / finished counts from lesson_log
  - streaks:  current + best consecutive-day completion streaks
  - vocab:    word count, exposures, quiz accuracy, mastery from vocab

Used by both the web UI (/stats page + /api/stats API) and the Telegram
/stats command.

Usage:
    from src.stats import profile_stats
    stats = profile_stats("krystof", config)
"""

import logging
from datetime import date, datetime, timedelta, timezone

from config import load_config, resolve_language_name
from vocab_db import VocabDB, get_shared_db

logger = logging.getLogger(__name__)

# Days shown in the activity chart.
RECENT_DAYS = 14

# Mastery threshold for the "mastered words" counter.
MASTERED_THRESHOLD = 0.8


def _streaks(finished_dates: set[str]) -> tuple[int, int]:
    """Return (current_streak, best_streak) for a set of ISO dates on which
    at least one lesson was finished.

    The current streak counts consecutive days ending today (or yesterday —
    today still in progress doesn't break the streak).
    """
    if not finished_dates:
        return 0, 0

    days = sorted(finished_dates, reverse=True)

    # Current streak: start at today, step to yesterday if today is empty
    current = 0
    d = datetime.now(timezone.utc).date()
    if d.isoformat() not in finished_dates:
        d -= timedelta(days=1)
    while d.isoformat() in finished_dates:
        current += 1
        d -= timedelta(days=1)

    # Best streak: walk the sorted dates, counting consecutive runs
    best = 1
    run = 1
    for prev, cur in zip(days, days[1:]):
        prev_d, cur_d = date.fromisoformat(prev), date.fromisoformat(cur)
        if (prev_d - cur_d).days == 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return current, best


def _profile_stats_queries(conn, profile_name: str) -> dict:
    """Run all stats queries for one profile and return the raw values.

    The caller must hold the connection lock (``VocabDB.lock``) — this
    queries ``conn`` directly, bypassing VocabDB's per-method locking.
    """
    # ── Lessons ────────────────────────────────────────────────────
    delivered, finished = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN finished_at IS NOT NULL "
        "THEN 1 ELSE 0 END) FROM lesson_log WHERE profile = ?",
        (profile_name,),
    ).fetchone()
    delivered = delivered or 0
    finished = finished or 0

    first_lesson, last_lesson = conn.execute(
        "SELECT MIN(delivered_at), MAX(delivered_at) FROM lesson_log "
        "WHERE profile = ?",
        (profile_name,),
    ).fetchone()
    last_finished = conn.execute(
        "SELECT MAX(finished_at) FROM lesson_log "
        "WHERE profile = ? AND finished_at IS NOT NULL",
        (profile_name,),
    ).fetchone()[0]

    finished_dates = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT date(finished_at) FROM lesson_log "
            "WHERE profile = ? AND finished_at IS NOT NULL",
            (profile_name,),
        ).fetchall()
    }
    current_streak, best_streak = _streaks(finished_dates)

    # ── Vocab ──────────────────────────────────────────────────────
    vocab_row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(frequency), 0), "
        "COALESCE(SUM(total_correct), 0), COALESCE(SUM(total_wrong), 0), "
        "COALESCE(AVG(mastery_score), 0), "
        "SUM(CASE WHEN mastery_score >= ? THEN 1 ELSE 0 END), "
        "MAX(last_seen) "
        "FROM vocab WHERE profile = ?",
        (MASTERED_THRESHOLD, profile_name),
    ).fetchone()
    (total_words, total_exposures, quiz_correct, quiz_wrong,
     avg_mastery, mastered_words, last_reviewed) = vocab_row
    quiz_attempts = quiz_correct + quiz_wrong
    quiz_accuracy = round(100 * quiz_correct / quiz_attempts, 1) \
        if quiz_attempts else None

    # ── Recent activity (last RECENT_DAYS days) ────────────────────
    today = datetime.now(timezone.utc).date()
    activity = []
    for i in range(RECENT_DAYS - 1, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN finished_at IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM lesson_log WHERE profile = ? AND date(delivered_at) = ?",
            (profile_name, day),
        ).fetchone()
        activity.append({
            "date": day,
            "delivered": row[0] or 0,
            "finished": row[1] or 0,
        })

    # ── Top words (most frequently encountered) ───────────────────
    top_words = [
        {"word": r[0], "meaning": r[1], "frequency": r[2]}
        for r in conn.execute(
            "SELECT word, meaning, frequency FROM vocab WHERE profile = ? "
            "ORDER BY frequency DESC, id LIMIT 10",
            (profile_name,),
        ).fetchall()
    ]

    # ── Recent finished lessons (for the list view) ───────────────
    recent_lessons = [
        {
            "title": r[0],
            "delivered_at": r[1],
            "finished_at": r[2],
        }
        for r in conn.execute(
            "SELECT title, delivered_at, finished_at FROM lesson_log "
            "WHERE profile = ? ORDER BY delivered_at DESC LIMIT 10",
            (profile_name,),
        ).fetchall()
    ]

    return {
        "delivered": delivered,
        "finished": finished,
        "first_lesson": first_lesson,
        "last_lesson": last_lesson,
        "last_finished": last_finished,
        "current_streak": current_streak,
        "best_streak": best_streak,
        "total_words": total_words,
        "total_exposures": total_exposures,
        "quiz_correct": quiz_correct,
        "quiz_attempts": quiz_attempts,
        "quiz_accuracy": quiz_accuracy,
        "avg_mastery": avg_mastery,
        "mastered_words": mastered_words,
        "last_reviewed": last_reviewed,
        "activity": activity,
        "top_words": top_words,
        "recent_lessons": recent_lessons,
    }


def profile_stats(profile_name: str, config: dict | None = None,
                  db=None, db_path=None) -> dict:
    """Compute the full stats dict for one profile.

    Args:
        profile_name: profile key from config["profiles"]
        config: config dict (loaded from disk if None)
        db: an existing VocabDB to query (not closed here)
        db_path: open a dedicated database (closed here); with neither
            db nor db_path the process-wide shared store is used
    """
    if config is None:
        config = load_config()
    profile_cfg = (config.get("profiles", {}) or {}).get(profile_name, {})

    if db is not None:
        owns_db = False
    elif db_path is not None:
        db, owns_db = VocabDB(db_path), True
    else:
        db, owns_db = get_shared_db(), False

    try:
        # All queries run under the connection lock: the shared database
        # is used concurrently by the bot loop and the web UI threads.
        with db.lock:
            data = _profile_stats_queries(db.conn, profile_name)
    finally:
        if owns_db:
            db.close()

    learning_language = profile_cfg.get("learning_language", "")
    return {
        "profile": profile_name,
        "learning_language": learning_language,
        "learning_language_name": resolve_language_name(learning_language),
        "native_language": profile_cfg.get("native_language", ""),
        "enabled": profile_cfg.get("enabled", True),
        "lessons": {
            "delivered": data["delivered"],
            "finished": data["finished"],
            "completion_rate": round(
                100 * data["finished"] / data["delivered"], 1)
                if data["delivered"] else None,
            "current_streak": data["current_streak"],
            "best_streak": data["best_streak"],
            "first_lesson": (data["first_lesson"] or "")[:10] or None,
            "last_lesson": (data["last_lesson"] or "")[:10] or None,
            "last_finished": (data["last_finished"] or "")[:10] or None,
        },
        "vocab": {
            "total_words": data["total_words"],
            "total_exposures": data["total_exposures"],
            "quiz_attempts": data["quiz_attempts"],
            "quiz_correct": data["quiz_correct"],
            "quiz_accuracy": data["quiz_accuracy"],
            "avg_mastery": round(data["avg_mastery"], 3),
            "mastered_words": data["mastered_words"],
            "last_reviewed": data["last_reviewed"],
        },
        "recent_activity": data["activity"],
        "top_words": data["top_words"],
        "recent_lessons": data["recent_lessons"],
    }
