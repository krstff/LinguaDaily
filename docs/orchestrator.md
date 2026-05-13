# Orchestrator Guide

`src/orchestrator.py` is the central lesson pipeline controller. It coordinates every step of a single lesson — from fetching an article to delivering it via Telegram — and provides reusable utility functions used throughout the codebase.

## Architecture

```
Orchestrator.run_lesson(profile_name, delivery_callback)
    │
    ├── 1. _fetch_and_clean()              ← fetch_router → clean_content()
    ├── 2. ┌─ _generate_tts_async()        ← tts.synthesize() (thread pool)
    │      └─ _translate_async()           ← llama_client.translate() (thread pool)
    │         ↑↑ both run in parallel via asyncio.gather()
    ├── 3. _extract_and_save_vocab()       ← llama_client.extract_vocab() + processor
    └── 4. delivery_callback()             ← telegram_bot.deliver_lesson()
```

Steps 2a (TTS) and 2b (Translation) run **in parallel** via `asyncio.gather()` — both only need the original cleaned content, so they don't depend on each other. Step 3 (vocab extraction) runs after translation because it needs the translated text for context.

Each step degrades gracefully — if any step fails, the lesson continues without that feature.

## Pipeline Steps

### 1. Fetch & Clean (`_fetch_and_clean`)

Fetches an article via `fetch_router.py` using the profile's `source`, `learning_language`, and `article_filter`. Then runs `clean_content()` to strip Wikipedia artifacts (reference markers, footer sections, broken wiki links).

After cleaning, re-enforces `max_words` by running `smart_truncate()` + `hard_truncate()` as a safety net — cleaning can inflate word count by splitting merged tokens from adjacent capitalized words.

**On failure**: Falls back to placeholder text like *"A wikipedia article about X could not be retrieved."*

### 2. TTS & Translation (parallel)

Both steps run concurrently via `asyncio.gather()`:

#### 2a. Generate TTS (`_generate_tts_async`)

Calls `tts.synthesize()` with the original (untranslated) content in the `learning_language`. Audio is saved to `output/<profile>/<uuid>.wav`. Runs in a thread pool executor since TTS is synchronous I/O.

**On failure**: Lesson continues without audio (`wav_path` = `None`). If profile has `"use_tts": false`, this step is skipped entirely.

#### 2b. Translate (`_translate_async`)

Calls `llama_client.translate()` to translate from the article's language (`learning_language`) to the user's native language (`native_language`). This is the **opposite** direction of what might seem intuitive — the lesson text is delivered in the learner's native language so they understand the content, while the audio is in the target language.

**On failure**: Falls back to the original untranslated text. If no LLM is configured (`config["llm"]` missing), this step is skipped entirely.

### 3. Extract & Save Vocab (`_extract_and_save_vocab`)

Runs **after** translation because it needs both the original and translated text for context. Calls `llama_client.extract_vocab()` to pull useful vocabulary words from the original article (in the learning language). Then persists them via `processor.update_vocab()` to a per-profile markdown file at `data/<profile>/vocabulary.md`.

**On failure**: Continues with empty vocab list. If no LLM configured, skipped entirely.

### 4. Deliver (`delivery_callback`)

If a delivery callback was provided (e.g., `TelegramBot.deliver_lesson`), the completed lesson dict is passed to it. The callback sends the original text, translation, vocabulary list, and TTS audio to the user's Telegram chat. The bot also persists the lesson in SQLite for tutor context injection.

**On failure**: Logs error, lesson still returned successfully.

## Orchestrator Class API

```python
from src.orchestrator import Orchestrator

# Create with config
orch = Orchestrator(config=config)  # dict from config.json
orch = Orchestrator(config_path="/path/to/config.json")  # or load from file

# Run a lesson for one profile
lesson = await orch.run_lesson("krystof")

# With a delivery callback
async def my_callback(profile_name, lesson):
    print(f"Delivered: {lesson['title']}")

lesson = await orch.run_lesson(
    "krystof",
    delivery_callback=my_callback,  # optional: deliver after pipeline
)
```

### `run_lesson()` signature

```python
async def run_lesson(self, profile_name: str,
                     delivery_callback: Optional[Callable] = None) -> Optional[dict]:
```

**Parameters:**
- `profile_name` — the profile to run a lesson for (must exist in config)
- `delivery_callback` — async callable `(profile_name, lesson)` → None, called after pipeline completes

**Returns:** A lesson dict or `None` on total failure.

### Return value

```python
{
    "profile": "krystof",
    "title": "Quantum Computing",
    "content": "Translated article text...",       # translated (or original if LLM failed)
    "original_content": "Original article text...",
    "learning_language": "de",
    "learning_language_name": "German",
    "native_language": "en",
    "wav_path": "/workspace/output/krystof/lingua_xxx.wav",  # or None
    "vocab": [{"word": "Quanten", "meaning": "quantum"}],     # or []
    "word_count": 245,
    "timestamp": "2026-05-09T08:00:00.123456"
}
```

## Utility Functions (module-level)

These are standalone functions used by other modules:

### `get_profile(config, profile_name=None)`

Resolves a profile from config with fallback chain: explicit name → `default_profile` → first profile. Returns `(name, dict)`. Raises `ValueError` if no profiles exist.

```python
from src.orchestrator import get_profile
name, profile = get_profile(config, "krystof")  # explicit
name, profile = get_profile(config)             # uses default_profile or first
```

### `clean_content(text)`

Cleans Wikipedia-extracted text: removes reference markers (`[1]`), footer sections (`See also`, `References`, German equivalents), fixes missing spaces from broken wiki links (e.g., `lower+Upper` → `lower Upper`), normalizes whitespace.

```python
from src.orchestrator import clean_content
clean = clean_content(raw_wikipedia_text)
```

## CLI Usage

Run a single lesson pipeline from the command line:

```bash
# Default profile (uses default_profile from config)
conda run -n lingua python src/orchestrator.py

# Specific profile
conda run -n lingua python src/orchestrator.py --profile krystof

# Override TTS URL
conda run -n lingua python src/orchestrator.py --tts-url http://192.168.1.50:8080/v1

# Custom config path
conda run -n lingua python src/orchestrator.py --config /path/to/config.json
```

### CLI arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--profile`, `-p` | User profile name | Config's `default_profile` |
| `--config`, `-c` | Path to config file | `config.json` in project root |
| `--tts-url` | Override TTS base_url | Config's `tts.base_url` |

> **Note:** The CLI always wires up a `TelegramBot` delivery callback, so lessons are delivered to Telegram when running via CLI.

## Graceful Degradation

The pipeline is designed so that **no single failure stops the lesson**:

| If this fails... | The lesson... |
|-----------------|---------------|
| Article fetch | Uses placeholder text |
| Content cleaning | Continues with raw text |
| TTS generation | Delivers without audio |
| LLM translation | Uses original untranslated text |
| Vocab extraction | Skips vocab, delivers normally |
| Delivery callback | Logs error, lesson still prepared |

## How the Scheduler Uses It

The scheduler's worker processes queued profiles and delegates to Orchestrator:

```python
# Inside scheduler._worker():
from orchestrator import Orchestrator
orch = Orchestrator(config=self.config)
lesson = await orch.run_lesson(
    profile_name,
    delivery_callback=self.delivery_callback,
)
```

This keeps the scheduler thin (cron + queue management only) and the orchestrator as the single source of truth for the lesson pipeline.
