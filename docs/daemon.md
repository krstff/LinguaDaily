# Daemon (main.py) Guide

`src/main.py` is the standalone daemon entry point. It starts the Telegram bot and lesson scheduler as concurrent async tasks, handles signals for graceful shutdown, and logs everything to both console and file.

## What it does on startup

```
============================================================
  OpenClaw-Lingua Standalone Daemon
============================================================
  Config:     /workspace/config.json
  Profiles:   2 (krystof, anna)
  Scheduled:  2 daily lesson(s)
    • krystof          08:00 (Europe/Berlin) → German
    • anna             10:30 (Europe/Madrid) → Spanish
  Telegram:   ✅ configured (token: ...ST-TOKEN)
  LLM:        gemma4-26b @ http://localhost:8080/v1
============================================================
```

## Architecture

```
main.py (async event loop)
    │
    ├── Telegram Bot task     ← polls for user messages, delivers lessons
    │   └── delivery_callback → scheduler feeds lessons here
    │
    └── Scheduler task        ← fires daily cron jobs per profile
        └── run_lesson()      ← fetch → clean → TTS → translate → vocab
```

Both tasks run concurrently via `asyncio.create_task()`. The daemon blocks until a shutdown signal (SIGINT/SIGTERM) or any task fails.

## Service Wiring

The daemon wires services together automatically:

```python
# Telegram bot created if token is configured
if tg_token:
    bot = TelegramBot(config=config)

# Scheduler gets bot.deliver_lesson as delivery callback
delivery_callback = bot.deliver_lesson if bot else None
scheduler = LessonScheduler(config=config, delivery_callback=delivery_callback)
```

**No Telegram?** The scheduler still runs — lessons are prepared and logged but not delivered.

**No LLM?** Lessons use the original untranslated text — TTS and fetch still work.

## Logging

| Destination | Level | Format |
|-------------|-------|--------|
| Console (stdout) | INFO (or DEBUG with `--verbose`) | `[time] [name] LEVEL: message` |
| File (`lingua.log`) | DEBUG (always) | `[datetime] [name] LEVEL: message` |

## Signal Handling

| Signal | Source | Action |
|--------|--------|--------|
| SIGINT | Ctrl+C, `kill -2` | Graceful shutdown |
| SIGTERM | `kill <pid>`, systemd stop | Graceful shutdown |

On shutdown: bot polling stops → scheduler jobs complete or cancel → all resources closed.

## CLI Usage

```bash
# Default config (config.json in project root)
conda run -n lingua python src/main.py

# Custom config
conda run -n lingua python src/main.py --config /path/to/config.json

# Debug logging
conda run -n lingua python src/main.py --verbose

# Run in background (with nohup or systemd)
nohup conda run -n lingua python src/main.py > /dev/null 2>&1 &
```

## Config Validation

The daemon validates config on startup:

| Condition | Behavior |
|-----------|----------|
| File not found | Exit with error (code 1) |
| Invalid JSON | Exit with error (code 1) |
| No profiles | Warning, continues |
| No Telegram token | Warning, bot disabled |
| No LLM config | Warning, translation/tutor disabled |

## Running as a Service

### systemd unit example

```ini
[Unit]
Description=OpenClaw-Lingua Daemon
After=network.target

[Service]
Type=simple
User=lingua
WorkingDirectory=/workspace
ExecStart=/opt/conda/envs/lingua/bin/python src/main.py --config /workspace/config.json
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker (future)

The daemon is designed to run as PID 1 in a container — signal handling works correctly with `--init` flag or when using systemd-style supervisors.

## Import API

```python
from src.main import LinguaDaemon

daemon = LinguaDaemon(config=config)
await daemon.start()   # blocks until shutdown
await daemon.stop()    # graceful shutdown
```
