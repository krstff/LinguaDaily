#!/usr/bin/env python3
"""
Vocabulary processor for LinguaDaily standalone daemon.

Persists vocabulary extracted by the LLM into the shared SQLite store
(data/chat_history.db, table `vocab`) via VocabDB — frequency starts at 1
with today's date; duplicates (case-insensitive) are skipped.

Usage (import):
    from src.processor import LinguaProcessor
    proc = LinguaProcessor(profile="krystof")  # uses the shared store
    proc.update_vocab(vocab_list)  # list of {word, meaning} dicts
"""

from config import (
    DEFAULT_LEARNING_LANGUAGE,
    DEFAULT_PROFILE_NAME,
    resolve_language_name,
)

from vocab_db import VocabDB, get_shared_db


class LinguaProcessor:
    """Manages vocabulary persistence for a single profile (SQLite).

    Pass ``db`` to share an existing VocabDB (it will not be closed by
    ``close()``), or ``db_path`` for a dedicated connection. With
    neither, the process-wide shared store is used.
    """

    def __init__(
        self,
        learning_language=DEFAULT_LEARNING_LANGUAGE,
        profile=DEFAULT_PROFILE_NAME,
        db=None,
        db_path=None,
    ):
        self.learning_language = learning_language
        self.learning_language_name = resolve_language_name(learning_language)
        self.profile = profile
        if db is not None:
            self.db = db
            self._owns_db = False
        elif db_path is not None:
            self.db = VocabDB(db_path)
            self._owns_db = True
        else:
            self.db = get_shared_db()
            self._owns_db = False

    # ── Public API ─────────────────────────────────────────────────

    def update_vocab(self, words) -> int:
        """
        Persist lesson vocabulary words (from llama_client.extract_vocab).

        Accepts dicts {word, meaning} or plain strings. New words start at
        frequency 1; words seen in earlier lessons get frequency +1
        (case-insensitive matching). Returns the number of words inserted
        or refreshed.
        """
        return self.db.add_words(self.profile, words)

    def close(self):
        """Close the database connection (only if we own it)."""
        if self._owns_db:
            self.db.close()


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """CLI for testing vocabulary updates.

    Usage:
        python3 src/processor.py --profile krystof --words '{"word":"Haus","meaning":"house"}'
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Vocabulary processor")
    parser.add_argument("--profile", "-p", default="default", help="Profile name")
    parser.add_argument("--lang", "-l", default="de",
                        help="Learning language code (e.g. de, it)")
    parser.add_argument("--words", "-w", default=None,
                        help='JSON array of {word, meaning} dicts')
    args = parser.parse_args()

    proc = LinguaProcessor(learning_language=args.lang, profile=args.profile)

    if args.words:
        words = json.loads(args.words)
        added = proc.update_vocab(words)
        print(f"Added {added} new word(s) for '{args.profile}' "
              f"(total {proc.db.word_count(args.profile)})")


if __name__ == "__main__":
    main()
