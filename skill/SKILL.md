# Lingua Skill

Language immersion tutor — fetches real Wikipedia articles from a local Kiwix server, translates them, and tracks vocabulary.

## Architecture

```
orchestrator.py  →  wikipedia_fetcher.py  →  tts.py  →  output/<profile>/lingua_*.wav
     (entry)          (Kiwix client, lang-aware)   (OmniVoice TTS)
                                              ↓
                                    processor.py  →  data/<profile>/vocabulary.md
                                         (prep for LLM)          (per-user vocab DB)
```

**Flow:** Fetch native-language article → generate TTS audio → Agent adds translation + glossary → deliver text + voice message.

## Multi-User Profiles

Each user gets their own profile in `config.json` with separate language pair, topics, article length settings, schedule, and vocabulary file.

```
config.json                          → shared config + profiles map
data/
  krystof/
    vocabulary.md                    → Krystof's vocab
  anna/
    vocabulary.md                    → Anna's vocab
```

## Content Source

Pulls random Wikipedia articles from a **local Kiwix/ZIM server** instead of scraping the web. This works offline and gives consistent, readable prose for language learning.

**Default server:** `http://192.168.100.52:8080` (English Wikipedia, ~19M articles)

Configuration in `config.json` (shared global setting):
```json
{
  "kiwix": {
    "base_url": "http://192.168.100.52:8080",
    "zim_name": "wikipedia_en_all_maxi_2026-02"
  }
}
```

## Configuration

### config.json structure

```json
{
  "default_profile": "krystof",
  "kiwix_servers": {
    "en": {
      "base_url": "http://192.168.100.52:8080",
      "zim_name": "wikipedia_en_all_maxi_2026-02"
    },
    "de": {
      "base_url": "http://192.168.100.52:8080",
      "zim_name": "wikipedia_de_all_maxi_2026-04"
    }
  },
  "tts": {
    "base_url": "http://localhost:8080/v1",
    "api_key": "***",
    "model": "omnivoice",
    "default_voice": "female"
  },
  "profiles": {
    "krystof": {
      "source_lang": "en",
      "target_lang": "de",
      "target_lang_name": "German",
      "content_lang": "de",
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

### kiwix_servers

Maps language codes → Kiwix server config. The orchestrator picks the right server based on each profile's `content_lang`. Add entries for every language you have a ZIM file for.

### tts

Global TTS settings for OmniVoice. All profiles share these unless overridden later.

### Per-profile: content_lang

`content_lang` specifies which language the fetched article should be in. This is the language the learner is studying — the TTS reads it aloud, and the Agent translates *into* the user's native language (`source_lang`).

Example: Krystof speaks English (`source_lang: en`) and is learning German (`target_lang: de`). So `content_lang: de` — articles are fetched in German, read aloud in German, and translated to English.

### Per-profile settings

| Setting | Description |
|---------|-------------|
| `source_lang` | Source language code (default: `en`) |
| `target_lang` | Target language code |
| `target_lang_name` | Display name (e.g. `German`) |
| `source` | Content source (`wikipedia` now; planned: `news`, `custom`) |
| `topics` | Topics to search for (random pick per run) |
| `article_filter` | Word count thresholds for article filtering |
| `schedule.time` | Daily lesson delivery time |
| `schedule.tz` | Timezone for scheduling |

> **Delivery** is configured in your OpenClaw cron job (`delivery.channel`, `delivery.to`), not here.

### Adding a new user

1. Add a profile entry under `profiles` in `config.json`
2. Create the data directory: `mkdir -p data/<profile_name>`
3. Set up a cron job (see below)

## Usage

### Via orchestrator (full pipeline)

```bash
python3 src/orchestrator.py                      # default profile, random topic
python3 src/orchestrator.py --profile krystof    # specific profile
python3 src/orchestrator.py --profile anna       # another user
python3 src/orchestrator.py "quantum physics"    # default profile, specific topic
python3 src/orchestrator.py --profile anna "art"  # profile + topic
```

Outputs a JSON payload between `---PAYLOAD_START---` / `---PAYLOAD_END---` markers containing the article text and user profile, ready for LLM translation.

### Via fetcher directly

```bash
python3 src/wikipedia_fetcher.py                    # random article (JSON)
python3 src/wikipedia_fetcher.py "black holes"      # search + pick random result
python3 src/wikipedia_fetcher.py --config config.json  # explicit config path
```

### As a Python module

```python
from src.wikipedia_fetcher import KiwixClient

client = KiwixClient(
    base_url="http://192.168.100.52:8080",
    zim_name="wikipedia_en_all_maxi_2026-02"
)

# Random article
title, text = client.get_random_article()

# Search-based
titles = client.search("quantum computing", count=5)
html = client.get_article(titles[0])
```

## Article Filtering

The fetcher automatically filters out:
- List/glossary/index pages (not good for learning)
- Disambiguation pages
- Very short stubs (below `min_words`)
- Extremely long pages (above `max_words`) — smart-truncated

Prefers readable prose articles within the per-profile word range.

### Smart Truncation

`wikipedia_fetcher.py` uses a **two-pass smart truncation** strategy:

1. **Section-level** — Splits on Wikipedia section headers and accumulates complete sections
2. **Paragraph-level (fallback)** — Falls back to blank-line-separated paragraphs

All thresholds are controlled per-profile via `article_filter` in `config.json`.

## Capabilities

- **Fetch Content**: Local Kiwix server (offline Wikipedia ZIM files) or RSS news feeds. Multi-language support via `kiwix_servers` map.
- **TTS Audio**: Generates voice readings of the fetched article using local OmniVoice server — no tokens, fast.
- **Translate**: LLM provides native-language glossary/summary (not full translation) since content is already in the target language.
- **Vocabulary Tracking**: Auto-updates per-profile `data/<profile>/vocabulary.md`.
- **Dual-Language Delivery**: Original text + translated summary + voice message for immersive learning.
- **Multi-User**: Separate profiles with independent vocab, topics, schedules, and language pairs.

## Cron Integration

Set up one cron job per profile in OpenClaw:

```
Name: Lingua Daily Lesson — <profile>
Schedule: daily at <profile.schedule.time> <profile.schedule.tz>
Command: python3 src/orchestrator.py --profile <profile_name>
Delivery: announce to <profile.delivery.channel>:<profile.delivery.to>
```

The Agent reads the JSON payload and:
1. **Translates** the article from `content_lang` into the user's native language (`source_lang`) — providing a glossary/summary rather than a full translation.
2. **Attaches** the TTS audio file using `MEDIA:<wav_path>` so the recipient gets a voice message alongside the text lesson.
3. Updates the vocabulary database.

## Payload Format

The orchestrator outputs a JSON payload with profile context:

```json
{
  "profile": "krystof",
  "title": "Some Article",
  "content": "... (in content_lang) ...",
  "topic": "Technology",
  "content_lang": "de",
  "source_lang": "en",
  "target_lang": "de",
  "target_lang_name": "German",
  "vocab_path": "/absolute/path/to/data/krystof/vocabulary.md",
  "wav_path": "/absolute/path/to/output/krystof/lingua_a1b2c3d4.wav"
}
```

The Agent uses this to:
1. **Translate** the article from `content_lang` into the user's native language (`source_lang`)
2. **Attach** the TTS audio via `MEDIA:<wav_path>` (if `wav_path` is present)
3. Update the vocabulary database
4. Route delivery to the correct recipient
