#!/usr/bin/env python3
"""
SQLite vocabulary store for LinguaDaily.

Replaces the legacy per-profile data/<profile>/vocabulary.csv files. All
profiles share one table in the main chat-history database
(data/chat_history.db) so lesson stats, SRS data and lesson completion
data live in a single place.

Schema:
    vocab (
        id, profile, word, word_key (lowercase, dedup key), meaning,
        frequency, last_seen, total_correct, total_wrong, mastery_score
    )

Usage:
    from src.vocab_db import VocabDB
    db = VocabDB()                      # default: data/chat_history.db
    db.add_words("krystof", [{"word": "Haus", "meaning": "house"}])
    entries = db.get_entries("krystof")

One-time CSV migration (idempotent, no-op when no CSVs remain):
    python3 -c "from src.vocab_db import VocabDB; VocabDB().migrate_csv()"
"""

import csv
import logging
import os
import sqlite3
import threading
from datetime import date
from pathlib import Path

from config import PROJECT_DIR

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = PROJECT_DIR / "data" / "chat_history.db"

# Bayesian mastery update (shared with flashcards' SRS selection).
# Prior of (1, 1) prevents 0/1 extremes with few data points.
def update_mastery(total_correct: int, total_wrong: int, was_correct: bool):
    """Return (new_correct, new_wrong, new_mastery_score)."""
    if was_correct:
        total_correct += 1
    else:
        total_wrong += 1
    mastery = (total_correct + 1) / (total_correct + total_wrong + 2)
    return total_correct, total_wrong, round(mastery, 4)


class VocabDB:
    """SQLite-backed per-profile vocabulary storage with SRS counters.

    Connections are opened with ``check_same_thread=False`` and every
    operation is serialized through a re-entrant lock, so one instance
    can be shared across threads (the daemon runs the Telegram bot in
    the asyncio loop and the web UI in Flask threads — see
    :func:`get_shared_db`).
    """

    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        os.makedirs(self.db_path.parent, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path),
                                    check_same_thread=False)
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()

    def _init_schema(self):
        with self._lock:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vocab (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile TEXT NOT NULL,
                word TEXT NOT NULL,
                word_key TEXT NOT NULL,
                meaning TEXT NOT NULL DEFAULT '',
                frequency INTEGER NOT NULL DEFAULT 1,
                last_seen TEXT,
                total_correct INTEGER NOT NULL DEFAULT 0,
                total_wrong INTEGER NOT NULL DEFAULT 0,
                mastery_score REAL NOT NULL DEFAULT 0.0,
                UNIQUE(profile, word_key)
            )
        """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vocab_profile ON vocab(profile)")
            # lesson_log is owned by ChatHistoryDB (telegram_bot.py) —
            # mirrored here (IF NOT EXISTS, same DDL) so a VocabDB-created
            # database is usable standalone by stats queries.
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

    # ── Reads ──────────────────────────────────────────────────────

    def get_entries(self, profile: str) -> list[dict]:
        """All vocabulary entries for a profile (insertion order)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT word, meaning, frequency, last_seen, total_correct, "
                "total_wrong, mastery_score FROM vocab "
                "WHERE profile = ? ORDER BY id",
                (profile,),
            ).fetchall()
        return [
            {
                "word": r[0],
                "meaning": r[1],
                "frequency": r[2],
                "last_seen": r[3] or None,
                "total_correct": r[4],
                "total_wrong": r[5],
                "mastery_score": r[6],
            }
            for r in rows
        ]

    def word_count(self, profile: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM vocab WHERE profile = ?", (profile,)
            ).fetchone()
        return row[0]

    # ── Writes ─────────────────────────────────────────────────────

    def add_words(self, profile: str, words) -> int:
        """Persist lesson vocabulary (dicts {word, meaning} or plain
        strings).

        New words are inserted with frequency 1. Words that already exist
        are treated as a re-encounter: frequency +1 and last_seen bumped
        (the original word form and meaning are kept).
        Returns the number of words inserted or refreshed.
        """
        today = date.today().isoformat()
        touched = 0
        with self._lock:
            for entry in words:
                if isinstance(entry, dict):
                    w_raw = str(entry.get("word", "")).strip()
                    meaning = str(entry.get("meaning", "")).strip()
                else:
                    w_raw = str(entry).strip()
                    meaning = ""
                w = w_raw.lower()
                if not w:
                    continue
                cur = self.conn.execute(
                    "INSERT INTO vocab (profile, word, word_key, meaning, "
                    "frequency, last_seen) "
                    "VALUES (?, ?, ?, ?, 1, ?) "
                    "ON CONFLICT(profile, word_key) DO UPDATE SET "
                    "frequency = frequency + 1, last_seen = excluded.last_seen",
                    (profile, w_raw, w, meaning, today),
                )
                touched += cur.rowcount
            self.conn.commit()
        return touched

    def record_exposure(
        self,
        profile: str,
        words: list[str],
        outcomes: list[tuple[str, bool]] | None = None,
    ):
        """Persist a study session: bump frequency/last_seen for every
        word shown, and apply quiz outcomes (word, was_correct) to the
        Bayesian mastery counters.
        """
        if not words and not outcomes:
            return
        today = date.today().isoformat()
        with self._lock:
            for w in words:
                self.conn.execute(
                    "UPDATE vocab SET frequency = frequency + 1, last_seen = ? "
                    "WHERE profile = ? AND word_key = ?",
                    (today, profile, str(w).strip().lower()),
                )
            if outcomes:
                for w, was_correct in outcomes:
                    row = self.conn.execute(
                        "SELECT total_correct, total_wrong FROM vocab "
                        "WHERE profile = ? AND word_key = ?",
                        (profile, str(w).strip().lower()),
                    ).fetchone()
                    if not row:
                        continue
                    c, wrong, mastery = update_mastery(row[0], row[1], was_correct)
                    self.conn.execute(
                        "UPDATE vocab SET total_correct = ?, total_wrong = ?, "
                        "mastery_score = ? WHERE profile = ? AND word_key = ?",
                        (c, wrong, mastery, profile, str(w).strip().lower()),
                    )
            self.conn.commit()

    # ── Legacy CSV migration ───────────────────────────────────────

    def migrate_csv(self) -> dict[str, int]:
        """One-time migration: import every data/<profile>/vocabulary.csv
        into the vocab table, then delete the CSV files.

        Idempotent — a second run finds no CSVs and does nothing.
        Returns {profile: rows_imported}.
        """
        data_dir = PROJECT_DIR / "data"
        results: dict[str, int] = {}
        for csv_path in sorted(data_dir.glob("*/vocabulary.csv")):
            profile = csv_path.parent.name
            imported = 0
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = (row.get("word") or "").strip()
                    if not word:
                        continue
                    with self._lock:
                        cur = self.conn.execute(
                            "INSERT INTO vocab (profile, word, word_key, meaning, "
                            "frequency, last_seen, total_correct, total_wrong, "
                            "mastery_score) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                            "ON CONFLICT(profile, word_key) DO UPDATE SET "
                            "frequency = MAX(vocab.frequency, excluded.frequency), "
                            "total_correct = MAX(vocab.total_correct, "
                            "excluded.total_correct), "
                            "total_wrong = MAX(vocab.total_wrong, excluded.total_wrong)",
                            (
                                profile,
                                word,
                                word.lower(),
                                (row.get("meaning") or "").strip(),
                                int(row.get("frequency") or 1),
                                (row.get("last_seen") or "").strip() or None,
                                int(row.get("total_correct") or 0),
                                int(row.get("total_wrong") or 0),
                                float(row.get("mastery_score") or 0.0),
                            ),
                        )
                        imported += max(cur.rowcount, 0)
                        self.conn.commit()
            csv_path.unlink()
            results[profile] = imported
            logger.info("Migrated %d vocab rows for '%s' (removed %s)",
                        imported, profile, csv_path)
        return results

    @property
    def lock(self) -> threading.RLock:
        """The connection lock. Code that uses ``self.conn`` directly
        (e.g. stats.py querying lesson_log) must hold it."""
        return self._lock

    def close(self):
        with self._lock:
            self.conn.close()


# ── Process-wide shared instance ───────────────────────────────────
#
# The daemon runs the Telegram bot (asyncio loop) and the web UI
# (Flask threads) in one process, so all components share a single
# VocabDB connection — it is thread-safe by construction. Callers that
# pass an explicit db_path (tests, one-off CLIs) get dedicated
# instances instead.

_shared_db: "VocabDB | None" = None


def get_shared_db() -> "VocabDB":
    """Return the process-wide VocabDB (created on first use)."""
    global _shared_db
    if _shared_db is None:
        _shared_db = VocabDB()
    return _shared_db


def reset_shared_db() -> None:
    """Close and drop the process-wide VocabDB (teardown/tests)."""
    global _shared_db
    if _shared_db is not None:
        _shared_db.close()
        _shared_db = None
