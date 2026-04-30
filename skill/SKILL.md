# Lingua Skill

Language immersion tutor — fetches real Wikipedia articles from a local Kiwix server, translates them, and tracks vocabulary.

## Architecture

```
orchestrator.py  →  wikipedia_fetcher.py  →  processor.py  →  vocabulary.md
     (entry)          (Kiwix client)         (prep for LLM)    (vocab DB)
```

## Content Source

Pulls random Wikipedia articles from a **local Kiwix/ZIM server** instead of scraping the web. This works offline and gives consistent, readable prose for language learning.

**Default server:** `http://192.168.100.52:8080` (English Wikipedia, ~19M articles)

Configuration in `config.json`:
```json
{
  "kiwix": {
    "base_url": "http://192.168.100.52:8080",
    "zim_name": "wikipedia_en_all_maxi_2026-02"
  }
}
```

## Usage

### Via orchestrator (full pipeline)

```bash
python3 src/orchestrator.py                    # random topic
python3 src/orchestrator.py "quantum physics"  # search for topic
```

Outputs a JSON payload between `---PAYLOAD_START---` / `---PAYLOAD_END---` markers containing the article text, ready for LLM translation.

### Via fetcher directly

```bash
python3 src/wikipedia_fetcher.py               # random article (JSON)
python3 src/wikipedia_fetcher.py "black holes" # search + pick random result
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
- Very short stubs (< 300 words)
- Extremely long pages (> 3000 words) — truncated

Prefers readable prose articles in the 300–3000 word range.

## Capabilities

- **Fetch Content**: Local Kiwix server (offline Wikipedia ZIM files).
- **Translate**: LLM translates fetched article into target language.
- **Vocabulary Tracking**: Auto-updates `data/vocabulary.md` with new words and frequency counts.
- **Dual-Language Delivery**: Original + translated text for comparison learning.

## Configuration

| Setting | Location | Description |
|---------|----------|-------------|
| `source_lang` | `config.json` | Source language code (default: `en`) |
| `target_lang` | `config.json` | Target language code (default: `de`) |
| `target_lang_name` | `config.json` | Display name (default: `German`) |
| `kiwix.base_url` | `config.json` | Kiwix server URL |
| `kiwix.zim_name` | `config.json` | ZIM file name |
| `topics` | `config.json` | Topics to search for (random pick) |

## Cron Integration

Set up a daily cron job in OpenClaw:

```
Schedule: daily at 08:00 CET
Command: python3 src/orchestrator.py
```

The Agent reads the JSON payload, translates via LLM, and delivers the dual-language lesson.
