# Lesson Scheduler Guide

The scheduler manages daily cron jobs using APScheduler with a **serial FIFO queue**. Each profile with a `schedule` section gets its own cron trigger. When a trigger fires, the profile is pushed onto a shared queue. A single background worker processes lessons **one at a time**, guaranteeing no overlap even when multiple profiles share the same schedule time.

## Architecture

```
Scheduler (APScheduler)
    │
    ├── 08:00 Europe/Berlin → krystof ─┐
    ├── 08:00 Europe/Berlin → johi     ├─→ FIFO Queue ─→ Worker (serial)
    └── 10:30 Europe/Madrid → anna ────┘              │
                                                       ▼
                                              Orchestrator.run_lesson()
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                       ▼                       ▼
                     fetch→clean→TTS       translate→vocab         deliver_lesson()
```

The scheduler is **thin** — it only handles cron scheduling, queuing, and profile discovery. All pipeline logic lives in `Orchestrator`.

## Enabled / Disabled Profiles

Profiles have an `enabled` field (default: `true`). Only enabled profiles with a `schedule` section are scheduled. The Web UI provides a toggle button per profile.

```json
{
  "profiles": {
    "krystof": {
      "enabled": true,       ← scheduled (if schedule.time is set)
      "schedule": { "time": "08:00", "tz": "Europe/Berlin" }
    },
    "johi": {
      "enabled": false,      ← NOT scheduled even though schedule exists
      "schedule": { "time": "09:00", "tz": "Europe/Berlin" }
    }
  }
}
```

Profiles without `enabled` set default to `true` (backward compatible). Disabled profiles still work for tutor chat — they're just excluded from the scheduler.

## Queuing Behavior

Multiple profiles can share the same schedule time (e.g., both `krystof` and `johi` at `08:00`). When their triggers fire:

1. Each trigger **pushes** its profile onto a FIFO queue (instant, non-blocking).
2. A single background worker pulls profiles from the queue and runs them **sequentially**.
3. If three profiles share the same time, they run one after another — no overlap, no resource contention.

### Queue ordering
- Profiles are processed in **FIFO order** (first enqueued = first run).
- When multiple triggers fire at the exact same second, the order is determined by APScheduler's internal scheduling order (roughly config-definition order).
- If you need a specific order, stagger times by a few minutes (e.g., `08:00`, `08:03`, `08:06`).

### Queue depth
- The queue is unbounded. If triggers fire faster than lessons complete (unlikely with daily schedules), items accumulate and are processed in order.
- There is **no skip logic** — every enqueued lesson will eventually run.

## Pipeline Steps (delegated to Orchestrator)

See [Orchestrator Guide](orchestrator.md) for full details. When a scheduled job fires:

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

Each profile needs a `schedule` section and must be enabled:

```json
{
  "profiles": {
    "krystof": {
      "native_language": "en",
      "learning_language": "de",
      "source": "wikipedia",
      "article_filter": {
        "min_words": 50,
        "max_words": 300
      },
      "enabled": true,
      "schedule": {
        "time": "08:00",
        "tz": "Europe/Berlin"
      },
      "use_tts": true,
      "tts_voice": "male"
    },
    "anna": {
      "native_language": "en",
      "learning_language": "es",
      "source": "wikipedia",
      "enabled": false,       ← disabled — won't be scheduled
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

Profiles without a `schedule` section or with `"enabled": false` are **not scheduled** — they can still be used for on-demand tutor chat.

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

The callback receives this dict (returned by `Orchestrator.run_lesson()`):

```python
{
    "profile": "krystof",
    "title": "Quantum Computing",
    "content": "Translated article text...",      # translated (or original if LLM failed)
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

## Concurrency & Serial Execution

- **Single worker, FIFO queue** — all lessons run sequentially. Even if multiple cron triggers fire at the same time, only one lesson runs at any given moment.
- **No overlap** — LLM API calls, TTS requests, and file I/O never compete between profiles.
- **No skipped runs** — every trigger is enqueued; unlike the old `max_instances=1` model, nothing is dropped if a previous lesson takes longer than expected.

## CLI

```bash
# List all scheduled profiles (only enabled ones)
conda run -n lingua python src/scheduler.py --list
# Output:
# Profile                Time   Timezone                   Language
# ----------------------------------------------------------------------
# krystof              08:00   Europe/Berlin              German

# Start scheduler (blocks at cron times, Ctrl+C to stop)
conda run -n lingua python src/scheduler.py

# Run all jobs immediately, then keep daemon alive
conda run -n lingua python src/scheduler.py --run-now

# Run all jobs once and exit (testing/debugging)
conda run -n lingua python src/scheduler.py --once
```

### CLI flags

| Flag | Shorthand | Behavior |
|------|-----------|----------|
| `--list` | `-l` | Print schedule table and exit |
| `--run-now` | `-n` | Push all enabled profiles onto the queue immediately, then run as normal daemon. Useful for "test now + stay alive" |
| `--once` | `-o` | Run all enabled profiles once via the queue worker, wait for completion, then exit. Ideal for testing the full pipeline without keeping a daemon running |
| `--config` | `-c` | Override path to config.json (default: `config.json`) |

## Import API

```python
from src.scheduler import LessonScheduler

scheduler = LessonScheduler(
    config=config,                          # dict from config.json
    delivery_callback=bot.deliver_lesson,   # async callable(profile_name, lesson)
)

# Inspect scheduled profiles (only enabled + with schedule)
scheduled = scheduler.get_scheduled_profiles()
# → [("krystof", {...})]  (disabled profiles excluded)

# Start/stop
await scheduler.start()                   # blocks until cancelled
await scheduler.start(immediate_run=True)  # run now + stay alive
await scheduler.stop()                    # graceful shutdown

# One-shot (run all enabled profiles, wait for completion, shut down)
await scheduler.run_once()
```
