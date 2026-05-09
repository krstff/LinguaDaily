# LinguaDaily

A standalone language-learning daemon that delivers daily lessons via Telegram — fetching articles, translating them with a local LLM, generating TTS audio, and providing interactive tutoring.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  main.py (daemon)                    │
│                                                      │
│  ┌──────────────────┐    ┌──────────────────────┐   │
│  │   scheduler.py   │    │   telegram_bot.py    │   │
│  │                  │    │                      │   │
│  │  per-profile     │    │  • lesson delivery   │   │
│  │  daily cron jobs │────▶• tutor chat          │   │
│  │                  │    │  • /register         │   │
│  └────────┬─────────┘    │  • /status           │   │
│           │              └──────────┬───────────┘   │
│           ▼                         │               │
│  ┌──────────────────────────────┐   │               │
│  │     Lesson Pipeline          │◀──┘               │
│  │                              │                   │
│  │  1. fetch_router.py          │                   │
│  │     → wikipedia_fetcher      │                   │
│  │     → news_fetcher           │                   │
│  │                              │                   │
│  │  2. orchestrator.clean()     │                   │
│  │                              │                   │
│  │  3. tts.py (OmniVoice)       │                   │
│  │                              │                   │
│  │  4. llama_client.py          │                   │
│  │     → translate              │                   │
│  │     → extract_vocab          │                   │
│  └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
conda create -n lingua python=3.11 -y
conda run -n lingua pip install aiogram openai pytest pytest-asyncio apscheduler
```

### 2. Configure `config.json`

```json
{
  "llm": {
    "base_url": "http://localhost:8080/v1",
    "default_model": "gemma4-26b"
  },
  "tts": {
    "base_url": "http://localhost:8080/v1",
    "model": "omnivoice"
  },
  "telegram": {
    "bot_token": "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
  },
  "profiles": {
    "krystof": {
      "source_lang": "en",
      "target_lang": "de",
      "target_lang_name": "German",
      "content_lang": "de",
      "source": "wikipedia",
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
      "tts_voice": "male",
      "telegram_chat_id": 111222333
    }
  }
}
```

### 3. Start the daemon

```bash
conda run -n lingua python src/main.py --config config.json
```

The startup banner shows all configured profiles, schedules, and service status:

```
============================================================
  LinguaDaily Standalone Daemon
============================================================
  Config:     /workspace/config.json
  Profiles:   1 (krystof)
  Scheduled:  1 daily lesson(s)
    • krystof          08:00 (Europe/Berlin) → German
  Telegram:   ✅ configured (token: ...ST-TOKEN)
  LLM:        gemma4-26b @ http://localhost:8080/v1
============================================================
```

## Testing Individual Components

See [TESTING.md](docs/testing.md) for standalone commands to test each component without the full daemon.

Quick examples:

```bash
# Test LLM endpoint health (no real model calls in tests)
conda run -n lingua python src/llama_client.py health --config config.json

# Run a single lesson pipeline manually
conda run -n lingua python src/orchestrator.py --profile krystof

# List scheduled profiles
conda run -n lingua python src/scheduler.py --list

# Run the Telegram bot standalone (for testing)
conda run -n lingua python src/telegram_bot.py --config config.json
```

## Components

| File | Role |
|------|------|
| `src/main.py` | Daemon entry — wires scheduler + Telegram bot, signal handling |
| `src/scheduler.py` | APScheduler daily lessons per profile (fetch → translate → deliver) |
| `src/telegram_bot.py` | aiogram 3.x bot — lesson delivery + interactive tutor chat |
| `src/llama_client.py` | Local LLM client — translate, extract vocab, tutor chat |
| `src/orchestrator.py` | Content fetching pipeline (article → clean → TTS → payload) |
| `src/fetch_router.py` | Routes fetch requests to wikipedia or news sources |
| `src/wikipedia_fetcher.py` | Kiwix/ZIM client for offline Wikipedia articles |
| `src/news_fetcher.py` | RSS feed fetching for current events |
| `src/tts.py` | OmniVoice TTS wrapper (OpenAI-compatible API) |
| `src/processor.py` | Vocab tracking and Markdown generation |

## Documentation

- [Daemon (main.py)](docs/daemon.md) — Startup, service wiring, signal handling, systemd/Docker
- [Telegram Bot Guide](docs/telegram-bot.md) — Setup, registration, commands, tutor chat
- [Lesson Scheduler Guide](docs/scheduler.md) — Pipeline steps, config, delivery callback API
- [Testing Guide](docs/testing.md) — Run and test each component standalone

## Tests

```bash
conda run -n lingua pytest tests/ -v
```

All new components are fully tested (88 tests, all mocked — zero real LLM calls during testing).
