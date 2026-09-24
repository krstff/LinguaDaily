# Processor (Vocabulary) Guide

`src/processor.py` manages per-profile vocabulary in the shared SQLite store. It appends new words extracted by the LLM and tracks frequency and last-seen date.

## What it does

The processor is a thin persistence layer — it doesn't call the LLM or fetch content. Its sole responsibility is persisting vocabulary into the `vocab` table of `data/chat_history.db` (via `src/vocab_db.py`).

```
Orchestrator
    └── llama_client.extract_vocab() → [{"word": "Haus", "meaning": "house"}]
        └── processor.update_vocab(vocab_list) → inserts into vocab table
```

## Storage Schema

All profiles share one table (the legacy per-profile `data/<profile>/vocabulary.csv`
files were migrated to SQLite and removed):

```sql
CREATE TABLE vocab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    word TEXT NOT NULL,
    word_key TEXT NOT NULL,          -- lower(word), dedup key
    meaning TEXT NOT NULL DEFAULT '',
    frequency INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT,                  -- ISO date of last exposure
    total_correct INTEGER NOT NULL DEFAULT 0,   -- quiz SRS counters
    total_wrong INTEGER NOT NULL DEFAULT 0,
    mastery_score REAL NOT NULL DEFAULT 0.0,
    UNIQUE(profile, word_key)
);
```

- **Word / Meaning**: from LLM extraction (meaning in the user's native language);
  the first-seen form and meaning are kept on re-encounters
- **Frequency**: starts at 1, increments on every re-encounter in a new
  lesson's vocabulary **and** on each flashcard/quiz exposure
- **SRS counters**: updated by `VocabDB.record_exposure()` (Bayesian mastery)

## API

```python
from src.processor import LinguaProcessor

# Create processor for a profile
proc = LinguaProcessor(
    learning_language="de",      # language code (display name kept for context)
    profile="krystof",
)

# Optional: custom database path (tests)
proc = LinguaProcessor(profile="krystof", db_path="/tmp/test.db")

# Update with words from LLM (list of {word, meaning} dicts)
vocab = [{"word": "Haus", "meaning": "house"}, {"word": "Auto", "meaning": "car"}]
added = proc.update_vocab(vocab)   # returns number of NEW words added

# Plain strings also work
proc.update_vocab(["Hallo", "Welt"])
```

Duplicate words (case-insensitive) are skipped; new words get frequency 1
and today's date.
