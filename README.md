# OpenClaw-Lingua

An autonomous multi-user language immersion agent for OpenClaw.

## Overview

OpenClaw-Lingua automates the process of language learning by:
1. **Fetching** daily content from a local Kiwix/Wikipedia ZIM server.
2. **Translating** content into each user's target language.
3. **Tracking** vocabulary usage via per-user Markdown databases.
4. **Delivering** lessons to each user on their schedule.

## Features

- **Automated Daily Lessons** (via OpenClaw Cron — one per user)
- **Per-User Vocabulary Tracking** (separate `data/<profile>/vocabulary.md`)
- **Multi-Channel Delivery** (Telegram, WhatsApp — per user)
- **Contextual Tutoring** (Interactive Q&A via OpenClaw Agent)
- **Multi-User Profiles** (independent language pairs, topics, schedules)

## Architecture

```
config.json                          → shared config + per-user profiles
kiwix server (shared)
tts server (OmniVoice / local llama)
data/
  krystof/
    vocabulary.md                    → Krystof's vocab
  anna/
    vocabulary.md                    → Anna's vocab
output/
  krystof/
    lingua_*.wav                     → TTS audio per run

orchestrator.py  →  wikipedia_fetcher.py  →  tts.py  →  processor.py  →  vocabulary.md
     (entry)          (Kiwix client)         (speech gen)   (prep for LLM)    (per-user DB)
```

## Quick Start

### 1. Configure profiles

Edit `config.json`:

```json
{
  "default_profile": "krystof",
  "kiwix": {
    "base_url": "http://192.168.100.52:8080",
    "zim_name": "wikipedia_en_all_maxi_2026-02"
  },
  "profiles": {
    "krystof": {
      "source_lang": "en",
      "target_lang": "de",
      "target_lang_name": "German",
      "source": "wikipedia",
      "topics": [
        "Technology", "Science", "Mathematics",
        "History", "Art", "Music",
        "Philosophy", "Literature", "Architecture"
      ],
      "article_filter": {
        "min_words": 250,
        "target_words": 400,
        "max_words": 600
      },
      "schedule": {
        "time": "08:00",
        "tz": "Europe/Berlin"
      }
    }
  }
}
```

> **Note:** Delivery routing (channel, recipient) is configured in your OpenClaw cron job — not in this file. Keep `config.json` clean for public repos.

### 2. Add a new user

```bash
# Create data directory for the new user
mkdir -p data/anna

# Add profile to config.json profiles map
# Set up a cron job (see below)
```

### 3. Test a run

```bash
cd /path/to/openclaw-lingua
python3 src/orchestrator.py --profile krystof
```

#### Running locally (outside Docker)

The TTS server defaults to `http://llama-swap:8080/v1` (internal Docker DNS). When running on your host machine, override it:

```bash
# Full pipeline with local TTS URL:
python3 src/orchestrator.py --profile krystof --tts-url http://192.168.100.60:8080/v1

# Standalone TTS test:
python3 src/tts.py --tts-url http://192.168.100.60:8080/v1 --lang de "Hallo Welt"
```

### 4. Set up cron jobs

Create one cron job per profile in OpenClaw, targeting the profile's scheduled time and delivery channel. See [SKILL.md](skill/SKILL.md) for details.

## Article Length & Smart Truncation

Wikipedia articles are filtered and truncated so learners get coherent, bite-sized passages (~research-paper abstract length) instead of walls of text or mid-sentence cuts.

### How it works

`wikipedia_fetcher.py` uses a **two-pass smart truncation** strategy:

1. **Section-level** — Splits the article on Wikipedia section headers (`==Section==`) and greedily accumulates complete sections until reaching `max_words`. This preserves the article's structure and headings.
2. **Paragraph-level (fallback)** — If the first section alone is too long (common with lead/intro sections), falls back to splitting on blank-line-separated paragraphs and accumulating those instead.

Articles that are too short (< `min_words`) are skipped entirely.

### Configuring word targets

All thresholds are controlled per-profile from `config.json`:

```json
{
  "profiles": {
    "krystof": {
      "article_filter": {
        "min_words": 250,
        "target_words": 400,
        "max_words": 600
      }
    }
  }
}
```

| Setting | Meaning |
|---------|---------|
| `min_words` | Articles shorter than this are **skipped** (stubs, disambiguation pages) |
| `target_words` | The ideal length the truncation aims for |
| `max_words` | Hard ceiling — truncation stops before exceeding this |

### Where truncation happens

| File | Role |
|------|------|
| `src/wikipedia_fetcher.py` | **Only place** that does truncation — `smart_truncate()` + `get_random_article()` |
| `src/orchestrator.py` | Calls the fetcher; no length logic here |
| `src/processor.py` | Passes text through unchanged |

> **Note:** If you adjust the targets, only edit `config.json`. No code changes needed.

## TTS Configuration

Speech is generated via a local OmniVoice-compatible server (LLaMA.cpp with Omnivoice model).

### Per-Profile Voice Selection

Each profile can choose its own voice in `config.json`:

```json
{
  "profiles": {
    "krystof": {
      "tts_voice": "male"
    },
    "anna": {
      "tts_voice": "female"
    }
  }
}
```

Default voice is **male** when not specified.

### CLI Overrides

| Flag | Script | Purpose |
|------|--------|---------|
| `--tts-url <url>` | orchestrator, tts.py | Override TTS server address (for local runs) |
| `--voice <name>` | tts.py | Override voice for a single run |

## Documentation

- [Daemon (main.py)](docs/daemon.md) — Startup, service wiring, signal handling, systemd/Docker
- [Telegram Bot Guide](docs/telegram-bot.md) — Setup, registration, commands, tutor chat
- [Lesson Scheduler Guide](docs/scheduler.md) — Pipeline steps, config, delivery callback API
- [SKILL.md](skill/SKILL.md) — Full usage, API reference, and cron integration details
