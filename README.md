# LinguaDaily

A standalone language-learning daemon that delivers daily lessons via Telegram — fetching articles, translating them with a local LLM, generating TTS audio, and providing interactive tutoring. Includes a web UI for profile management and model selection.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    main.py (daemon)                                │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │ scheduler.py │  │ telegram_bot.py  │  │ web_ui.py           │ │
│  │              │  │                  │  │                     │ │
│  │ cron jobs    │  │ • lesson delivery│  │ • Dashboard         │ │
│  │ serial queue │  │ • tutor chat     │  │ • Model selection   │ │
│  │              │  │ • lesson context │  │ • Config editor     │ │
│  └──────┬───────┘  │    (SQLite)      │  │ • Live log viewer   │ │
│         │          └──────────────────┘  └─────────────────────┘ │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   orchestrator.py                            │ │
│  │                                                              │ │
│  │  Orchestrator.run_lesson():                                  │ │
│  │    1. fetch_router → article                                 │ │
│  │    2. clean_content()                                        │ │
│  │    3+4. tts.py + llama_client.translate() (parallel)         │ │
│  │    5. llama_client.extract_vocab()                           │ │
│  │    6. processor.update_vocab()                               │ │
│  │    7. delivery_callback()                                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
conda create -n lingua python=3.11 -y
conda run -n lingua pip install aiogram openai pytest pytest-asyncio apscheduler flask jinja2
```

### 2. Configure `config.json`

```json
{
  "default_profile": "krystof",
  "llm": {
    "base_url": "http://llama-swap:8080/v1",
    "default_model": "gemma-4-26B-language",
    "api_key": "",
    "timeout": 600
  },
  "tts": {
    "base_url": "http://llama-swap:8080/v1",
    "model": "omnivoice",
    "api_key": ""
  },
  "telegram": {
    "bot_token": "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
  },
  "profiles": {
    "krystof": {
      "native_language": "en",
      "learning_language": "de",
      "source": "wikipedia",
      "article_filter": {
        "min_words": 50,
        "max_words": 300
      },
      "use_tts": true,
      "tts_voice": "male",
      "telegram_chat_id": 111222333,
      "enabled": true,
      "schedule": {
        "time": "08:00",
        "tz": "Europe/Berlin"
      }
    }
  }
}
```

### 3. Start the daemon

```bash
# Daemon only (scheduler + Telegram bot)
conda run -n lingua python src/main.py --config config.json

# Daemon + Web UI
conda run -n lingua python src/main.py --config config.json --web-ui

# Web UI standalone (no scheduler/bot)
conda run -n lingua python src/web_ui.py --host 127.0.0.1 --port 8089
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
  LLM:        gemma-4-26B-language @ http://llama-swap:8080/v1
============================================================
```

## Web UI

The web UI is a lightweight admin panel for managing profiles, selecting models, and viewing logs.

### Dashboard (`/`)
- Profile table with enable/disable toggle per profile
- Model selection panel — choose translation, tutoring, and TTS models from dropdowns (populated by calling `/v1/models` on your LLM server)
- Add/Edit/Delete profiles via modal form

### Logs (`/logs`)
- Live tail of `lingua.log` with color-coded severity levels
- Auto-refresh every 3 seconds

### Config (`/config`)
- Raw JSON editor for `config.json` with validation and backup

### Model Selection
The dashboard has a **Model Selection** panel that queries your LLM server's `/v1/models` endpoint and populates three dropdowns:

| Dropdown | Config key | Used by |
|----------|-----------|---------|
| Translation Model | `llm.translate_model` | Article translation + vocabulary extraction |
| Tutoring Model | `llm.tutor_model` | Interactive tutor chat |
| TTS Model | `tts.model` | Text-to-speech synthesis |

Leave a dropdown on "→ default" to use the global `default_model`. Changes are saved globally and apply to all profiles.

### Standalone

```bash
# Defaults (localhost:8089, no auth)
conda run -n lingua python src/web_ui.py

# Remote access with basic auth
conda run -n lingua python src/web_ui.py --host 0.0.0.0 --port 8089 --password mypass
```

## Config Reference

### Global settings

| Key | Type | Description |
|-----|------|-------------|
| `default_profile` | string | Default profile for CLI tools |
| `llm.base_url` | string | OpenAI-compatible API base URL (e.g. `http://llama-swap:8080/v1`) |
| `llm.default_model` | string | Fallback model when no task-specific override is set |
| `llm.translate_model` | string *(optional)* | Model for translation + vocab extraction |
| `llm.tutor_model` | string *(optional)* | Model for interactive tutoring |
| `tts.base_url` | string | TTS API base URL |
| `tts.model` | string | TTS model name (default: `omnivoice`) |
| `telegram.bot_token` | string | Telegram BotFather token |

### Profile settings

| Key | Type | Description |
|-----|------|-------------|
| `native_language` | string | User's native language code (`en`, `de`, …) |
| `learning_language` | string | Language the user is learning |
| `source` | string | Content source: `wikipedia` or `news` |
| `article_filter.min_words` | int | Minimum article length |
| `article_filter.max_words` | int | Maximum article length |
| `use_tts` | bool | Generate TTS audio (default: `true`) |
| `tts_voice` | string | Voice name (`male`, `female`) |
| `telegram_chat_id` | string/int | Telegram chat ID for lesson delivery + tutor chat |
| `enabled` | bool | Whether the profile is scheduled (default: `true`). Toggle from Web UI. |
| `schedule.time` | string | Daily lesson time, HH:MM (24h) |
| `schedule.tz` | string | Timezone, e.g. `Europe/Berlin` |

## Environment Check

Before starting the daemon, run the environment health check to verify your setup:

```bash
# Full check (config + packages + network connectivity)
conda run -n lingua python src/env_check.py --config config.json

# Quick check (skip network — fast offline validation)
conda run -n lingua python src/env_check.py --config config.json --quick
```

The env check validates:
- **Config file**: exists, valid JSON, correct structure
- **Profiles**: language codes, chat IDs (numeric, no conflicts), source validity, article filter ranges, schedule times
- **Python packages**: all 6 required dependencies installed with version info
- **Directories**: data dir, output dir, per-profile vocab files
- **LLM endpoint**: `/v1/models` reachable, configured models exist on server
- **Kiwix servers**: HTTP root + `/random` article fetch works
- **Telegram bot**: `getMe` call confirms token is valid and returns bot username
- **TTS endpoint**: `/v1/models` reachable
- **News RSS feeds**: accessible (if using news source)

A sample config (`config.sample.json`) with profiles "alice" and "bob" is included for reference.

## Testing Individual Components

Quick examples:

```bash
# Test LLM endpoint health
conda run -n lingua python src/llama_client.py health --config config.json

# Run a single lesson pipeline manually
conda run -n lingua python src/orchestrator.py --profile krystof

# List scheduled profiles
conda run -n lingua python src/scheduler.py --list

# Run all scheduled jobs once and exit (quick test)
conda run -n lingua python src/scheduler.py --once

# Run all jobs now + keep daemon alive
conda run -n lingua python src/scheduler.py --run-now

# Run the Telegram bot standalone (for testing)
conda run -n lingua python src/telegram_bot.py --config config.json

# Web UI standalone
conda run -n lingua python src/web_ui.py --host 127.0.0.1 --port 8089
```

## Components

| File | Role |
|------|------|
| `src/main.py` | Daemon entry — wires scheduler + Telegram bot, signal handling, optional Web UI |
| `src/orchestrator.py` | **Lesson pipeline** — fetch → clean → TTS → translate → vocab → deliver |
| `src/scheduler.py` | APScheduler daily lessons per profile (delegates to orchestrator) |
| `src/telegram_bot.py` | aiogram 3.x bot — lesson delivery + interactive tutor chat with lesson context |
| `src/web_ui.py` | Flask admin panel — dashboard, model selection, config editor, log viewer |
| `src/llama_client.py` | Local LLM client — translate, extract vocab, tutor chat (with lesson injection) |
| `src/processor.py` | Vocabulary persistence — markdown file management |
| `src/fetch_router.py` | Routes fetch requests to wikipedia or news sources |
| `src/wikipedia_fetcher.py` | Kiwix/ZIM client for offline Wikipedia articles |
| `src/news_fetcher.py` | RSS feed fetching for current events |
| `src/tts.py` | OmniVoice TTS wrapper (OpenAI-compatible API) |
| `src/env_check.py` | Deployment health check — config, packages, connectivity validation |

## Documentation

- [Daemon (main.py)](docs/daemon.md) — Startup, service wiring, signal handling, systemd/Docker
- [Web UI](docs/webui-design.md) — Dashboard, model selection, config editor, log viewer
- [Orchestrator Guide](docs/orchestrator.md) — Pipeline steps, utility functions, CLI usage
- [Processor (Vocabulary)](docs/processor.md) — Vocab markdown file management
- [Lesson Scheduler Guide](docs/scheduler.md) — Schedule config, enabled/disabled profiles, delivery callback API
- [Telegram Bot Guide](docs/telegram-bot.md) — Setup, commands, tutor chat with lesson context
- [LLM Client Guide](docs/llama-client.md) — Model resolution, translate, vocab extraction, tutor chat
- [TTS Module Guide](docs/tts.md) — OmniVoice wrapper, text sanitization
- [Wikipedia Fetcher Guide](docs/wikipedia-fetcher.md) — Kiwix/ZIM client, HTML extraction, smart truncation

## Tests

```bash
conda run -n lingua pytest tests/ -v
```

172 passing tests across 9 test files (all mocked — zero real LLM calls during testing).

| Test file | Count | What it covers |
|-----------|-------|----------------|
| `test_fetch_router.py` | 11 | Source routing (wikipedia/news), CLI |
| `test_llama_client.py` | 20 | Model resolution, translate, vocab extraction, tutor chat, health check |
| `test_main.py` | 20 | Daemon startup, service wiring, signal handling, CLI |
| `test_news_fetcher.py` | 21 | Feed loading, topic resolution, article truncation |
| `test_orchestrator.py` | 21 | Config/profile utils, clean_content, full pipeline, CLI |
| `test_processor.py` | 15 | Vocab file init, read/update/dedup, markdown persistence |
| `test_scheduler.py` | 17 | Schedule discovery, job building, delivery callback, CLI |
| `test_telegram_bot.py` | 28 | Bot init, lesson delivery, tutor chat, commands, history DB |
| `test_wikipedia_fetcher.py` | 19 | KiwixClient, extract_wiki_text, smart/hard truncation |
