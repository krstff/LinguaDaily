# Telegram Bot Guide

One Telegram bot serves all users. Each user is identified by their Telegram chat ID and mapped to a profile with its own language pair, schedule, and tutor history.

## Architecture

```
Telegram Bot (one token)
    │
    ├── User A (chat_id=111222333) → profile "krystof" → German lessons + tutor
    ├── User B (chat_id=444555666) → profile "anna"    → Spanish lessons + tutor  
    └── User C sends /register unregistered             → French lessons + tutor
```

Each user gets:
- Independent language pair (from their profile config)
- Scheduled lessons delivered to their chat
- Isolated conversation history with the tutor (SQLite, keyed by `chat_id` + `profile`)

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g., "Lingua Tutor")
4. Choose a username (must end in `bot`, e.g., `lingua_tutor_bot`)
5. Copy the **API token** — you'll need it below

### 2. Configure `config.json`

Add the bot token and map your Telegram chat IDs to profiles:

```json
{
  "telegram": {
    "bot_token": "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
  },
  "profiles": {
    "krystof": {
      "source_lang": "en",
      "target_lang": "de",
      "target_lang_name": "German",
      "telegram_chat_id": 111222333,
      "schedule": { "time": "08:00", "tz": "Europe/Berlin" }
    },
    "anna": {
      "source_lang": "en",
      "target_lang": "es",
      "target_lang_name": "Spanish", 
      "telegram_chat_id": 444555666,
      "schedule": { "time": "10:00", "tz": "Europe/Madrid" }
    },
    "unregistered": {
      "source_lang": "en",
      "target_lang": "fr",
      "target_lang_name": "French"
      // no telegram_chat_id — user will self-register via /register
    }
  }
}
```

### 3. Find Your Telegram Chat ID

Send any message to your bot, then visit:

```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

Look for `"chat": { "id": 123456789 }` in the response. That's your chat ID.

**Alternatively**, start the bot and send it `/start` — if you're not pre-registered, it'll tell you to use `/register <profile>`.

### 4. Start the Bot

```bash
# Standalone (for testing)
conda run -n lingua python src/telegram_bot.py --config config.json

# As part of the daemon (once main.py is built)
conda run -n lingua python src/main.py
```

## Usage

### Commands

| Command | What it does |
|---------|-------------|
| `/start` | Show welcome message + current registration status |
| `/register <profile>` | Register your chat to a profile (e.g., `/register krystof`) |
| `/history clear` | Clear your tutor conversation history |
| `/status` | Show your schedule, language pair, and Telegram ID |

### Tutor Chat

Just send any message — the bot routes it to the LLM tutor with your profile's language settings:

```
You: What does "Konjunktiv" mean in German?
Bot: Konjunktiv is the subjunctive mood in German...
```

Conversation history persists across sessions (stored in `data/chat_history.db`).

### Scheduled Lessons

Lessons are delivered automatically at each profile's scheduled time. The scheduler triggers the orchestrator pipeline, which fetches content, generates TTS audio, and delivers both to your chat:

1. 📖 Text message with translated article
2. 🔊 Audio file (WAV) of the original text read aloud

## User Registration Options

### Option A: Config-based (pre-set)

Add `telegram_chat_id` to each profile in `config.json`. Best for known users — no setup needed on their end.

```json
"krystof": { "telegram_chat_id": 111222333 }
```

### Option B: Self-service (runtime)

Leave `telegram_chat_id` out and let users register themselves with `/register <profile>`. Best for shared bots or unknown users. The mapping is kept in memory while the bot runs.

### Both

Use both — pre-register some users in config, allow others to self-register via `/register`.

## Conversation History

Stored in SQLite at `data/chat_history.db`:

| Column | Example |
|--------|---------|
| `user_id` | `111222333` (Telegram chat ID) |
| `profile` | `krystof` |
| `role` | `user` or `assistant` |
| `content` | The message text |

History is **isolated per user+profile** — User A never sees User B's conversations. Recent history (last 10 turns) is sent to the LLM tutor for context on each chat message.

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TELEGRAM_BOT_TOKEN` | Bot token (fallback if not in config) | `123456:ABC...` |
