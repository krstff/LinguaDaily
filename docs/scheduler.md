# Lesson Scheduler Guide

The scheduler runs daily lessons automatically using APScheduler. Each profile with a `schedule` section gets its own cron job that fires the full lesson pipeline and delivers the result via Telegram.

## Architecture

```
Scheduler (APScheduler)
    │
    ├── 08:00 Europe/Berlin → krystof
    │   └── fetch → clean → TTS → translate → vocab → deliver_lesson()
    │
    └── 10:30 Europe/Madrid → anna
        └── fetch → clean → skip TTS → translate → vocab → deliver_lesson()
```

## Pipeline Steps

When a scheduled job fires, it runs these steps in order:

| Step | Source | What happens on failure |
|------|--------|------------------------|
| 1. Fetch article | `fetch_router.py` | Falls back to placeholder text |
| 2. Clean content | `orchestrator.clean_content()` | Continues with raw text |
| 3. Generate TTS | `tts.synthesize()` | Skips audio, delivers text only |
| 4. Translate | `llama_client.translate()` | Uses original untranslated text |
| 5. Extract vocab | `llama_client.extract_vocab()` | Continues with empty vocab list |
| 6. Deliver | callback (e.g., `bot.deliver_lesson`) | Logs error, lesson still prepared |

**No single point of failure** — the pipeline degrades gracefully at every step.

## Configuration

Each profile needs a `schedule` section:

```json
{
  "profiles": {
    "krystof": {
      "source_lang": "en",
      "target_lang": "de",
      "target_lang_name": "German",
      "topics": ["Technology", "Science", "History"],
      "article_filter": {
        "min_words": 50,
        "max_words": 300
      },
      "schedule": {
        "time": "08:00",
        "tz": "Europe/Berlin"
      },
      "use_tts": true,
      "tts_voice": "male"
    },
    "anna": {
      "source_lang": "en",
      "target_lang": "es",
      "target_lang_name": "Spanish",
      "topics": ["History", "Art"],
      "schedule": {
        "time": "10:30",
        "tz": "Europe/Madrid"
      },
      "use_tts": false
    }
  }
}
```

### Schedule fields

| Field | Required | Example | Description |
|-------|----------|---------|-------------|
| `time` | Yes | `"08:00"` | Daily fire time (HH:MM, 24-hour) |
| `tz` | No | `"Europe/Berlin"` | Timezone (default: UTC) |

Profiles without a `schedule` section are **not scheduled** — they can still be used for on-demand tutor chat.

## Delivery Callback

The scheduler is channel-agnostic. You pass it any async callable that accepts `(profile_name, lesson)`:

```python
from src.scheduler import LessonScheduler
from src.telegram_bot import TelegramBot

bot = TelegramBot(config=config)
scheduler = LessonScheduler(
    config=config,
    delivery_callback=bot.deliver_lesson  # <-- the callback
)
await scheduler.start()
```

### Lesson dict structure

The callback receives this dict:

```python
{
    "profile": "krystof",
    "title": "Quantum Computing",
    "content": "Translated article text...",      # translated (or original if LLM failed)
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

## Concurrency

- `max_instances=1` per profile — if a lesson takes longer than the interval, the next run is **skipped** (not queued)
- Different profiles run independently and can overlap

## CLI

```bash
# List all scheduled profiles
conda run -n lingua python src/scheduler.py --list
# Output:
# Profile                Time   Timezone                   Language
# ----------------------------------------------------------------------
# krystof              08:00   Europe/Berlin              German
# anna                 10:30   Europe/Madrid              Spanish

# Start scheduler (blocks until Ctrl+C)
conda run -n lingua python src/scheduler.py
```

## Import API

```python
from src.scheduler import LessonScheduler

scheduler = LessonScheduler(
    config=config,                          # dict from config.json
    delivery_callback=bot.deliver_lesson,   # async callable(profile_name, lesson)
)

# Inspect scheduled profiles
scheduled = scheduler.get_scheduled_profiles()
# → [("krystof", {...}), ("anna", {...})]

# Run a single lesson manually (for testing)
lesson = await scheduler.run_lesson("krystof")

# Start/stop
await scheduler.start()   # blocks until cancelled
await scheduler.stop()    # graceful shutdown
```
