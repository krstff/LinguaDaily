# Wikipedia Fetcher Guide

`src/wikipedia_fetcher.py` fetches articles from a local **Kiwix/ZIM server** (offline Wikipedia). It handles HTML extraction, quality filtering, smart truncation, and multi-language support.

This is the most complex source module (~300+ lines) because it deals with raw Kiwix HTTP responses, multi-language skip patterns, disambiguation page detection, and coherent text truncation.

## Architecture

```
Kiwix Server (ZIM reader)
    │
    ├── /random?content=ZIMNAME  → 302 redirect to article path
    ├── /search?pattern=...      → HTML list of matching titles
    └── /content/ZIMNAME/Title   → Full article HTML
          │
          ▼
    extract_wiki_text()          ← BeautifulSoup: strip chrome, infoboxes, footers
          │
          ▼
    smart_truncate()             ← 3-pass: sections → paragraphs → sentences
          │
          ▼
    (title, clean_text)          ← returned to fetch_router / orchestrator
```

## KiwixClient Class

Thin HTTP client for the Kiwix Server REST API. Handles UTF-8 encoding quirks, session management, and article quality filtering.

### Initialization

```python
from src.wikipedia_fetcher import KiwixClient

# Direct initialization
client = KiwixClient(
    base_url="http://192.168.100.52:8080",
    zim_name="wikipedia_en_all_maxi_2026-02",
)

# Context manager (auto-closes session)
with KiwixClient(base_url=..., zim_name=...) as client:
    title, text = client.get_random_article()
```

### Config-driven initialization

```python
from src.wikipedia_fetcher import load_fetcher_config

settings = load_fetcher_config(
    config_path="config.json",
    content_lang="de",  # resolves kiwix_servers["de"] from config
)
client = KiwixClient(base_url=settings["base_url"], zim_name=settings["zim_name"])
```

### Config shape

```json
{
  "kiwix_servers": {
    "en": {
      "base_url": "http://192.168.100.52:8080",
      "zim_name": "wikipedia_en_all_maxi_2026-02"
    },
    "de": {
      "base_url": "http://192.168.100.53:8080",
      "zim_name": "wikipedia_de_all_maxi_2026-02"
    }
  },
  "article_filter": {
    "min_words": 250,
    "max_words": 600
  }
}
```

The `load_fetcher_config()` function resolves the correct Kiwix server for a given `content_lang`. Falls back to legacy top-level `"kiwix"` block if `"kiwix_servers"` is not present.

### Public Methods

#### `get_random_article(max_attempts=10, min_words=250, max_words=600)`

Fetches a random readable article suitable for language learning. This is the main entry point used by `fetch_router.py`.

**Process:**
1. Hits `/random?content=ZIMNAME` → gets 302 redirect to article path
2. Extracts title from the Location header
3. Filters out lists, glossaries, disambiguation pages (multi-language patterns)
4. Fetches full HTML via `/content/ZIMNAME/Title`
5. Checks for enough prose paragraphs (not just tables/infoboxes)
6. Extracts clean text via `extract_wiki_text()`
7. Re-checks for disambiguation pages and table-heavy content in the extracted text
8. Truncates to `max_words` using `smart_truncate()` if needed
9. Returns `(title, text)` or falls back after 10 attempts

**Returns:** `(title: str, text: str)` — or `("Error", "...")` on failure.

#### `search(pattern, count=5, offset=0)`

Searches the ZIM file and returns a list of decoded article titles.

```python
titles = client.search("quantum computing", count=10)
# → ["Quantum computing", "Quantum mechanics", "Quantum entanglement", ...]
```

Uses `content=` parameter (not `book=`) to avoid Kiwix's "confusion-of-tongues" error when multiple language ZIMs are loaded.

#### `get_article(title)`

Fetches full article HTML for a given title. Handles URL-encoding correctly (decodes first, then re-encodes to avoid double-encoding `%C3%A4` → `%25C3%25A4`).

### Skip Patterns & Footer Markers

Defined as class-level constants (`SKIP_PATTERNS`, `FOOTER_MARKERS`) covering 7 languages: English, German, Spanish, Italian, Hungarian, French, Polish, Czech.

These are **hardcoded** in the source (see §2.G of `.work-in-progress.md` for a future improvement to make them config-driven).

## HTML Extraction

### `extract_wiki_text(html, skip_infoboxes=True)`

Extracts readable prose from Wikipedia/Kiwix HTML. Targets the main content area (`#mw-content-text > .mw-parser-output`) and strips:
- `<script>`, `<style>`, `<noscript>`, `<nav>` tags
- Reference sections (`.reflist`, `.mw-references-wrap`)
- Infoboxes, navboxes, vcard tables
- Data-heavy tables (>10 rows, >70% short cells)
- Footer noise (Creative Commons attribution text in 8 languages)

**Post-processing:** Inserts double-newline paragraph boundaries after sentence-ending punctuation + newline + uppercase letter, so `smart_truncate()` can split on paragraph boundaries.

### `_has_enough_prose(html, min_paragraphs=5)`

Quick filter — checks if the article has at least 5 `<p>` tags with ≥15 words each. Catches pure list/infobox pages early before fetching full HTML.

### `_is_table_heavy(text, min_prose_lines=8)`

Checks if extracted text is mostly short lines (<10 words per line). Returns `True` (skip this article) if fewer than 8 prose-length lines exist. Catches disambiguation pages and table-heavy content that slipped through HTML-level filtering.

## Smart Truncation

### `smart_truncate(text, max_words=600, min_words=250)`

Truncates text to at most `max_words` at a coherent structural boundary. Three-pass strategy:

| Pass | Strategy | When it wins |
|------|----------|-------------|
| 1 | **Section-level** — accumulates complete `==Header==` sections | Article has section headers and fits within limits |
| 2 | **Paragraph-level** — accumulates `\n\n`-separated paragraphs | No section headers, or single huge lead section |
| 3 | **Sentence-level** — accumulates `.!?`-terminated sentences | Dense text with no paragraph breaks (e.g., bibliography-heavy) |

Each pass greedily accumulates chunks until adding the next would exceed `max_words`. Returns the truncated text if ≥ `min_words`, or `None` to fall through to the next pass.

**Returns:** Truncated text string, or `None` if no usable chunk ≥ min_words could be produced.

### `hard_truncate(text, max_words=600)`

Last-resort fallback — cuts at a word boundary regardless of sentence/paragraph structure. Appends `"..."` to indicate truncation. Called when all three smart passes fail.

## CLI Usage

```bash
# Fetch a random article (default settings)
conda run -n lingua python src/wikipedia_fetcher.py

# With config and language
conda run -n lingua python src/wikipedia_fetcher.py --config config.json --content-lang de

# Custom word limits
conda run -n lingua python src/wikipedia_fetcher.py --min-words 100 --max-words 400
```

Outputs structured JSON:
```json
{
  "title": "Quantum Computing",
  "text": "...",
  "source": "Kiwix (wikipedia_en_all_maxi_2026-02)",
  "word_count": 342
}
```

## UTF-8 Encoding Quirk

Kiwix Server omits the `charset` parameter in its `Content-Type` header, so Python's `requests` defaults to ISO-8859-1 (HTTP/1.1 fallback) even though actual content is UTF-8. The client forces `resp.encoding = 'utf-8'` after every request to avoid mojibake on non-ASCII characters.

## Integration with fetch_router.py

The orchestrator doesn't call `wikipedia_fetcher.py` directly — it goes through `fetch_router.py`:

```python
# In orchestrator._run_pipeline():
from fetch_router import fetch_article as _fetch
title, content = _fetch(
    source="wikipedia",
    topic=None,
    config=self.config,
    learning_language=learning_language,
    article_filter=article_filter,
)
```

`fetch_router.py` imports `KiwixClient` directly (no subprocess), loads config via `load_fetcher_config()`, and delegates to `client.get_random_article()`.
