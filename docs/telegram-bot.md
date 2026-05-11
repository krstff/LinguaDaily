# Telegram Bot Guide

One Telegram bot serves all users. Each user is identified by their Telegram chat ID and mapped to a profile with its own language pair, schedule, and tutor history.

## Architecture

```
Telegram Bot (one token)
    │
    ├── User A (chat_id=111222333) → profile "krystof" → German lessons + tutor
    ├── User B (chat_id=444555666) → profile "anna"    → Spanish lessons + tutor
```

Each user gets:
- Independent language pair (from their profile config)
- Scheduled lessons delivered to their chat
- Isolated conversation history with the tutor (SQLite, keyed by `chat_id` + `profile`)
- **Lesson context** — the tutor sees the most recent delivered lesson when answering questions

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
      "native_language": "en",
      "learning_language": "de",
      "telegram_chat_id": 111222333,
      "schedule": { "time": "08:00", "tz": "Europe/Berlin" }
    },
    "anna": {
      "native_language": "en",
      "learning_language": "es",
      "telegram_chat_id": 444555666,
      "schedule": { "time": "10:00", "tz": "Europe/Madrid" }
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

### 4. Start the Bot

```bash
# Standalone (for testing)
conda run -n lingua python src/telegram_bot.py --config config.json

# As part of the daemon
conda run -n lingua python src/main.py
```

## Usage

### Commands

| Command | What it does |
|---------|-------------|
| `/start` | Show welcome message + current registration status |
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

1. 📰 Text message with original article
2. 🌐 Translation
3. 📝 Vocabulary list
4. 🔊 Audio file (WAV) of the original text read aloud

## User Registration

### Config-based (pre-set)

Add `telegram_chat_id` to each profile in `config.json`. This is the only supported method — users must be registered via config or the Web UI before they can use the bot.

```json
"krystof": { "telegram_chat_id": 111222333 }
```

Unregistered users who message the bot receive a notice to ask an admin to add them via the web UI.

## Conversation History

Stored in SQLite at `data/chat_history.db`:

| Column | Example |
|--------|---------|
| `user_id` | `111222333` (Telegram chat ID) |
| `profile` | `krystof` |
| `role` | `user` or `assistant` |
| `content` | The message text |

History is **isolated per user+profile** — User A never sees User B's conversations. Recent history (last 10 turns) is sent to the LLM tutor for context on each chat message. Old entries are auto-purged after 30 days.

## Lesson Context Injection

When a user messages the tutor, it has access to their **most recently delivered lesson**. The lesson is persisted in SQLite (`latest_lesson` table) when `deliver_lesson()` runs, and injected into the tutor's system prompt:

```
=== TODAY'S LESSON ===
Title: Berlin
Delivered: 2026-05-11 08:00:00

Original article (German):
Berlin ist die Hauptstadt von Deutschland.

Translation (English):
Berlin is the capital of Germany.

Vocabulary (3 words):
  Hauptstadt — capital city
  ...
=== END LESSON ===
```

This means the tutor can answer questions directly about the lesson content, vocabulary, and translations. Content is truncated to 2000 chars each to stay within reasonable token limits.

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TELEGRAM_BOT_TOKEN` | Bot token (fallback if not in config) | `123456:ABC...` |
