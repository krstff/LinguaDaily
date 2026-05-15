#!/usr/bin/env python3
"""
Vocabulary processor for LinguaDaily standalone daemon.

Manages per-profile vocabulary CSV files — reading existing entries,
appending new words extracted by the LLM, and tracking frequency / last-seen date.

Usage (import):
    from src.processor import LinguaProcessor
    proc = LinguaProcessor(profile="krystof")
    proc.update_vocab(vocab_list)  # list of {word, meaning} dicts

Vocabulary file format (data/<profile>/vocabulary.csv):
    word,meaning,frequency,last_seen
    der Anwohner,resident,1,2026-05-14
"""

import csv
import os
from datetime import date

from config import (
    DEFAULT_LEARNING_LANGUAGE,
    DEFAULT_PROFILE_NAME,
    PROJECT_DIR,
    resolve_language_name,
)


class LinguaProcessor:
    """Manages vocabulary persistence for a single profile."""

    def __init__(
        self,
        learning_language=DEFAULT_LEARNING_LANGUAGE,
        profile=DEFAULT_PROFILE_NAME,
        vocab_path=None,
    ):
        self.learning_language = learning_language
        self.learning_language_name = resolve_language_name(learning_language)
        self.profile = profile

        # Resolve vocab_path: explicit > per-profile default
        if vocab_path:
            if not os.path.isabs(vocab_path):
                self.vocab_path = PROJECT_DIR / vocab_path
            else:
                self.vocab_path = vocab_path
        else:
            self.vocab_path = PROJECT_DIR / "data" / profile / "vocabulary.csv"

    # ── File I/O ───────────────────────────────────────────────────

    def _ensure_vocab_file(self):
        """Create the vocabulary CSV file if it doesn't exist."""
        if os.path.exists(self.vocab_path):
            return

        os.makedirs(os.path.dirname(self.vocab_path) or ".", exist_ok=True)
        with open(self.vocab_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["word", "meaning", "frequency", "last_seen"])

    def _read_existing_vocab(self):
        """Read existing vocabulary entries as {word_lower: row_dict}."""
        existing = {}
        if not os.path.exists(self.vocab_path):
            return existing

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get("word", "").strip().lower()
                if not word:
                    continue
                existing[word] = {
                    "word": row.get("word", "").strip(),
                    "meaning": row.get("meaning", "").strip(),
                    "frequency": int(row.get("frequency", 1) or 1),
                    "last_seen": row.get("last_seen", "").strip() or None,
                }
        return existing

    # ── Public API ─────────────────────────────────────────────────

    def update_vocab(self, words):
        """
        Append new vocabulary words to the CSV vocabulary file.

        Parameters
        ----------
        words : list
            List of dicts {word, meaning} (from llama_client.extract_vocab)
            or plain strings. Duplicate words are skipped; new words get
            frequency 1 and today's date.
        """
        os.makedirs(os.path.dirname(self.vocab_path) or ".", exist_ok=True)
        self._ensure_vocab_file()

        today = date.today().isoformat()
        existing = self._read_existing_vocab()

        # Read all current rows, add new ones, rewrite
        all_rows = list(existing.values())
        added = 0

        for entry in words:
            if isinstance(entry, dict):
                w_raw = str(entry.get("word", "")).strip()
                w = w_raw.lower()
                m = entry.get("meaning", "(new)")
            else:
                w_raw = str(entry).strip()
                w = w_raw.lower()
                m = "(new)"

            if not w or w in existing:
                continue

            all_rows.append({
                "word": w_raw,
                "meaning": m,
                "frequency": 1,
                "last_seen": today,
            })
            added += 1

        if added > 0:
            with open(self.vocab_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["word", "meaning", "frequency", "last_seen"])
                writer.writeheader()
                writer.writerows(all_rows)


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
        proc.update_vocab(words)
        print(f"Updated vocabulary for '{args.profile}' -> {proc.vocab_path}")


if __name__ == "__main__":
    main()
