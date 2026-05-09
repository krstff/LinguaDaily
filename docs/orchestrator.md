# Orchestrator Guide

`src/orchestrator.py` is the central lesson pipeline controller. It coordinates every step of a single lesson — from fetching an article to delivering it via Telegram — and provides reusable utility functions used throughout the codebase.

## Architecture

```
Orchestrator.run_lesson(profile_name)
    │
    ├── 1. _fetch_and_clean()     ← fetch_router → clean_content()
    ├── 2. _generate_tts()        ← tts.synthesize()
    ├── 3. _translate()           ← llama_client.translate()
    ├── 4. _extract_and_save_vocab() ← llama_client.extract_vocab() + processor.update_vocab()
    └── 5. delivery_callback()    ← telegram_bot.deliver_lesson()
```

Each step is a separate async method, making the pipeline easy to test and extend. No single step is required — if any step fails, the lesson degrades gracefully and continues.

## Pipeline Steps

### 1. Fetch & Clean (`_fetch_and_clean`)

Fetches an article via `fetch_router.py` using the profile's `source`, `content_lang`, and `article_filter`. Picks a random topic from the profile's `topics` list (or uses a specific topic if provided). Then runs `clean_content()` to strip Wikipedia artifacts.

**On failure**: Falls back to placeholder text like *"A wikipedia article about X could not be retrieved."*

### 2. Generate TTS (`_generate_tts`)

Calls `tts.synthesize()` with the original (untranslated) content in the `content_lang`. Audio is saved to `output/<profile>/<uuid>.wav`.

**On failure**: Lesson continues without audio (`wav_path` = `None`). If profile has `"use_tts": false`, this step is skipped entirely.

### 3. Translate (`_translate`)

Calls `llama_client.translate()` to translate from the article's language (`content_lang`) to the user's native language (`source_lang`). This is the **opposite** direction of what might seem intuitive — the lesson is delivered in the learner's native language so they understand the content, while the audio is in the target language.

**On failure**: Falls back to the original untranslated text. If no LLM is configured (`config["llm"]` missing), this step is skipped entirely.

### 4. Extract & Save Vocab (`_extract_and_save_vocab`)

Calls `llama_client.extract_vocab()` to pull useful vocabulary words from the original article (in the target/learning language). Then persists them via `processor.update_vocab()` to a per-profile markdown file at `data/<profile>/vocabulary.md`.

**On failure**: Continues with empty vocab list. If no LLM configured, skipped entirely.

### 5. Deliver (`delivery_callback`)

If a delivery callback was provided (e.g., `TelegramBot.deliver_lesson`), the completed lesson dict is passed to it. The callback sends the translated text and TTS audio to the user's Telegram chat.

**On failure**: Logs error, lesson still returned successfully.

## Orchestrator Class API

```python
from src.orchestrator import Orchestrator

# Create with config
orch = Orchestrator(config=config)  # dict from config.json
orch = Orchestrator(config_path="/path/to/config.json")  # or load from file

# Run a lesson for one profile
lesson = await orch.run_lesson("krystof")

# With a specific topic and delivery callback
async def my_callback(profile_name, lesson):
    print(f"Delivered: {lesson['title']}")

lesson = await orch.run_lesson(
    "krystof",
    topic="Quantum Physics",          # optional: override random topic
    delivery_callback=my_callback,    # optional: deliver after pipeline
)
```

### `run_lesson()` return value

Returns a dict (or `None` on total failure):

```python
{
    "profile": "krystof",
    "title": "Quantum Computing",
    "content": "Translated article text...",       # translated (or original)
    "original_content": "Original article text...",
    "topic": "Technology",
    "source_lang": "en",
    "target_lang": "de",
    "target_lang_name": "German",
    "content_lang": "de",
    "wav_path": "/workspace/output/krystof/lingua_xxx.wav",  # or None
    "vocab": [{"word": "Quanten", "meaning": "quantum"}],     # or []
    "word_count": 245,
    "timestamp": "2026-05-09T08:00:00.123456"
}
```

## Utility Functions (module-level)

These are standalone functions used by other modules:

### `load_config(path=None)`

Loads config from JSON file. Defaults to `config.json` in project root.

```python
from src.orchestrator import load_config
config = load_config()
config = load_config("/custom/path.json")
```

### `get_profile(config, profile_name=None)`

Resolves a profile from config with fallback chain: explicit name → `default_profile` → first profile. Returns `(name, dict)`. Raises `ValueError` if no profiles exist.

```python
from src.orchestrator import get_profile
name, profile = get_profile(config, "krystof")  # explicit
name, profile = get_profile(config)             # uses default_profile or first
```

### `clean_content(text)`

Cleans Wikipedia-extracted text: removes reference markers (`[1]`), footer sections (`See also`, `References`), fixes missing spaces from broken wiki links, normalizes whitespace.

```python
from src.orchestrator import clean_content
clean = clean_content(raw_wikipedia_text)
```

### `fetch_article(source, topic, config, content_lang=None, article_filter=None)`

Wrapper around `fetch_router.fetch_article()`. Fetches an article from the given source (wikipedia/news). Returns `(title, text)` or `(None, None)`.

```python
from src.orchestrator import fetch_article
title, text = fetch_article(
    source="wikipedia",
    topic="Physics",
    config=config,
    content_lang="de",
)
```

## CLI Usage

Run a single lesson pipeline from the command line:

```bash
# Default profile, random topic
conda run -n lingua python src/orchestrator.py

# Specific profile
conda run -n lingua python src/orchestrator.py --profile krystof

# Profile + specific topic
conda run -n lingua python src/orchestrator.py --profile anna "Quantum Physics"

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
| `topic` (positional) | Topic to search for | Random from profile topics |

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

The scheduler no longer contains pipeline logic. Instead, each scheduled job creates an `Orchestrator` and delegates:

```python
# Inside scheduler._build_job():
from orchestrator import Orchestrator
orch = Orchestrator(config=self.config)
lesson = await orch.run_lesson(profile_name, delivery_callback=self.delivery_callback)
```

This keeps the scheduler thin (cron management only) and the orchestrator as the single source of truth for the lesson pipeline.
