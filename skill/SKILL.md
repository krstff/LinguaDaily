# Lingua Skill

Language immersion tutor — fetches real Wikipedia articles from a local Kiwix server, translates them, and tracks vocabulary.

## Architecture

```
orchestrator.py  →  wikipedia_fetcher.py  →  processor.py  →  data/<profile>/vocabulary.md
     (entry)          (Kiwix client)         (prep for LLM)          (per-user vocab DB)
```

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

- **Fetch Content**: Local Kiwix server (offline Wikipedia ZIM files).
- **Translate**: LLM translates fetched article into target language.
- **Vocabulary Tracking**: Auto-updates per-profile `data/<profile>/vocabulary.md`.
- **Dual-Language Delivery**: Original + translated text for comparison learning.
- **Multi-User**: Separate profiles with independent vocab, topics, and schedules.

## Cron Integration

Set up one cron job per profile in OpenClaw:

```
Name: Lingua Daily Lesson — <profile>
Schedule: daily at <profile.schedule.time> <profile.schedule.tz>
Command: python3 src/orchestrator.py --profile <profile_name>
Delivery: announce to <profile.delivery.channel>:<profile.delivery.to>
```

The Agent reads the JSON payload, translates via LLM using the profile's target language, and delivers the dual-language lesson to the correct recipient.

## Payload Format

The orchestrator outputs a JSON payload with profile context:

```json
{
  "profile": "krystof",
  "title": "Some Article",
  "content": "...",
  "topic": "Technology",
  "source_lang": "en",
  "target_lang": "de",
  "target_lang_name": "German",
  "vocab_path": "/absolute/path/to/data/krystof/vocabulary.md"
}
```

The Agent uses this to:
1. Translate in the correct target language
2. Update the right vocabulary file
3. Route delivery to the correct recipient
