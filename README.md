# LinguaDaily

A standalone language-learning daemon that delivers daily lessons via Telegram — fetching articles, translating them with a local LLM, generating TTS audio, and providing interactive tutoring.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    main.py (daemon)                        │
│                                                           │
│  ┌─────────────────────┐    ┌──────────────────────────┐ │
│  │   scheduler.py      │    │   telegram_bot.py        │ │
│  │                     │    │                          │ │
│  │  per-profile cron   │    │  • lesson delivery       │ │
│  │  jobs (APScheduler) │    │  • tutor chat (SQLite)   │ │
│  │                     │    │  • /register, /status    │ │
│  └─────────┬───────────┘    └──────────┬───────────────┘ │
│            │                           │                 │
│            ▼                           │                 │
│  ┌─────────────────────────────────────┤◀────────────────┘ │
│  │         orchestrator.py             │                   │
│  │                                     │                   │
│  │  Orchestrator.run_lesson():         │                   │
│  │    1. fetch_router → article        │                   │
│  │    2. clean_content()               │                   │
│  │    3. tts.py (OmniVoice)            │                   │
│  │    4. llama_client.translate()      │                   │
│  │    5. llama_client.extract_vocab()  │                   │
│  │    6. processor.update_vocab()      │                   │
│  │    7. delivery_callback()           │                   │
│  └─────────────────────────────────────┘                   │
└───────────────────────────────────────────────────────────┘
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

Quick examples:

```bash
# Test LLM endpoint health
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
| `src/orchestrator.py` | **Lesson pipeline** — fetch → clean → TTS → translate → vocab → deliver |
| `src/scheduler.py` | APScheduler daily lessons per profile (delegates to orchestrator) |
| `src/telegram_bot.py` | aiogram 3.x bot — lesson delivery + interactive tutor chat |
| `src/llama_client.py` | Local LLM client — translate, extract vocab, tutor chat |
| `src/processor.py` | Vocabulary persistence — markdown file management |
| `src/fetch_router.py` | Routes fetch requests to wikipedia or news sources |
| `src/wikipedia_fetcher.py` | Kiwix/ZIM client for offline Wikipedia articles |
| `src/news_fetcher.py` | RSS feed fetching for current events |
| `src/tts.py` | OmniVoice TTS wrapper (OpenAI-compatible API) |

## Documentation

- [Daemon (main.py)](docs/daemon.md) — Startup, service wiring, signal handling, systemd/Docker
- [Orchestrator Guide](docs/orchestrator.md) — Pipeline steps, utility functions, CLI usage
- [Processor (Vocabulary)](docs/processor.md) — Vocab markdown file management
- [Lesson Scheduler Guide](docs/scheduler.md) — Schedule config, delivery callback API
- [Telegram Bot Guide](docs/telegram-bot.md) — Setup, registration, commands, tutor chat

## Tests

```bash
conda run -n lingua pytest tests/ -v
```

122 passing tests across 6 test files (all mocked — zero real LLM calls during testing).

| Test file | Count | What it covers |
|-----------|-------|----------------|
| `test_llama_client.py` | 20 | Model resolution, translate, vocab extraction, tutor chat, health check |
| `test_telegram_bot.py` | 29 | Bot init, lesson delivery, tutor chat, commands, history DB |
| `test_scheduler.py` | 20 | Schedule discovery, job building, delivery callback, CLI |
| `test_main.py` | 20 | Daemon startup, service wiring, signal handling, CLI |
| `test_orchestrator.py` | 18 | Config/profile utils, clean_content, full pipeline, CLI |
| `test_processor.py` | 15 | Vocab file init, read/update/dedup, markdown persistence |
