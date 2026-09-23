#!/usr/bin/env python3
"""
Wiktionary word-lookup client for the tutor — offline (Kiwix/ZIM) or online.

Same two-backend pattern as the Wikipedia article pipeline
(see wikipedia_fetcher.py):
  * KiwixClient     — local Kiwix Server (ZIM files), fully offline
  * WiktionaryClient— {lang}.wiktionary.org via the classic MediaWiki API

Backend resolution mirrors resolve_wikipedia_backend():
  * wiktionary.backend = "auto"   (default) — Kiwix if a ZIM is configured
    for the language, otherwise online
  * = "kiwix"  — offline only (falls back to online if unconfigured)
  * = "online" — wiktionary.org only

The heavy lifting (Kiwix HTTP quirks, URL-encoding, UTF-8, user agent) is
inherited from the article pipeline — this module adds only:
  * resolve_wiktionary_backend()
  * WiktionaryClient (online, .wiktionary.org domain)
  * extract_wiktionary_entry()  — HTML → compact reference text
  * fetch_wiktionary_entry()    — title-fallback lookup (term → lemma)
  * get_dictionary_reference()  — multi-term reference block for the tutor

Usage:
    python3 src/wiktionary_client.py --term gehen --lang de --config config.json
"""

import json
import logging
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import KIWIX_DEFAULT_BASE_URL
from wikipedia_fetcher import (
    KiwixClient,
    WikipediaClient,
    WikiFetchError,
)

logger = logging.getLogger(__name__)

# Hard caps so a dictionary reference never bloats the tutor prompt
ENTRY_MAX_CHARS = 2000
REFERENCE_MAX_CHARS = 4000
MAX_TERMS = 3

# Part-of-speech lines that mark an inflected-form stub page (e.g. de:
# "Konjugierte Form" for "geht") — in that case prefer the lemma entry
INFLICTED_FORM_RE = re.compile(
    r"konjugierte form|deklination des|forme conjuguée|flect\w* form"
    r"|inflected form|skloňovan\w*|deklinov\w*", re.IGNORECASE,
)


# ── Backend resolution (mirror of resolve_wikipedia_backend) ─────────

def resolve_wiktionary_backend(config, language=None):
    """
    Resolve which Wiktionary backend to use for a language.

    Returns (backend, params):
      ("kiwix",  {"base_url": str, "zim_name": str})
      ("online", {"language": str})

    Config shape:
      {
        "wiktionary": {
          "backend": "auto" | "kiwix" | "online",
          "servers": {
            "de": {"base_url": "http://host:8080",
                    "zim_name": "de.wiktionary_..."}
          }
        }
      }
    """
    lang = (language or "de").lower()
    wt = (config or {}).get("wiktionary") or {}
    backend = (wt.get("backend") or "auto").lower()

    servers = wt.get("servers") or {}
    srv = servers.get(lang) or {}
    # A server entry only counts if it names a ZIM file
    has_kiwix = bool(srv.get("zim_name"))

    if backend == "online" or not (has_kiwix and backend in ("auto", "kiwix")):
        return "online", {"language": lang}

    return "kiwix", {
        "base_url": srv.get("base_url", KIWIX_DEFAULT_BASE_URL),
        "zim_name": srv.get("zim_name"),
    }


# ── Online client (subclass of the Wikipedia article client) ─────────

class WiktionaryClient(WikipediaClient):
    """Client for an online {lang}.wiktionary.org wiki."""

    def __init__(self, language="de", timeout=15):
        super().__init__(language=language, timeout=timeout)
        self.base_url = f"https://{self.language}.wiktionary.org"

    def get_entry_html(self, title):
        """Fetch the rendered content HTML for a word's page.

        Raises WikiFetchError when the page does not exist.
        """
        data = self._api(
            action="parse", page=title, prop="text",
            format="json", redirects=1,
        )
        if data.get("error"):
            code = data["error"].get("code", "unknown")
            raise WikiFetchError(f"wiktionary API error for '{title}': {code}")
        text = (data.get("parse") or {}).get("text", {}).get("*")
        if not text:
            raise WikiFetchError(f"wiktionary API returned no content for '{title}'")
        return text


# ── HTML → compact reference text ────────────────────────────────────

def extract_wiktionary_entry(html, max_chars=ENTRY_MAX_CHARS):
    """
    Extract a compact reference block from a Wiktionary page's HTML.

    Keeps what the tutor needs (unlike the article extractor, which
    removes tables — a wiktionary page *is* mostly tables and lists):
      * part of speech (first h3, e.g. "Verb, unregelmäßig, intransitiv")
      * definition list items (translation list items are excluded)
      * inflection tables (conjugation / declension), condensed
      * English translations (spans marked lang="en")

    Returns a string, or None when the page has no usable content.
    Works on both Kiwix-served HTML and the MediaWiki API's
    action=parse prop=text output (same .mw-parser-output structure).
    """
    soup = BeautifulSoup(html, "html.parser")

    content = None
    for selector in ("#mw-content-text .mw-parser-output", ".mw-parser-output"):
        content = soup.select_one(selector)
        if content:
            break
    if content is None:
        content = soup

    # Remove chrome + maintenance template boxes
    for tag in content.find_all(["script", "style", "link", "meta"]):
        tag.decompose()
    for tag in content.find_all(class_=re.compile(
            r"mw-editsection|noprint|mw-jump-link|reflist|mbox|ambox|ueberarbeiten")):
        tag.decompose()
    toc = content.find(id="toc")
    if toc:
        toc.decompose()

    parts = []

    # 1. Part of speech — first h3 ("Verb, unregelmäßig, intransitiv")
    h3 = content.find("h3")
    if h3:
        pos = h3.get_text(" ", strip=True)
        if pos:
            parts.append(f"Part of speech: {pos}")

    # 2. English translations (MediaWiki marks them with lang="en") —
    #    highest value for translation questions, so they go first
    en = []
    for span in content.find_all("span", lang="en"):
        t = span.get_text(" ", strip=True)
        if t:
            en.append(t)
    if en:
        parts.append("English: " + ", ".join(list(dict.fromkeys(en))[:15]))

    # 3. Definitions — list items, excluding translation-list entries
    #    (those contain a "→" arrow, a <span lang="xx"> value, or an
    #    interwiki language link with class "extiw")
    defs = []
    for li in content.find_all("li"):
        txt = li.get_text(" ", strip=True)
        if "→" in txt:
            continue
        if li.find(class_="extiw") or li.find("span", lang=True):
            continue
        if txt.endswith(":") or "[?" in txt:
            continue  # section headers, audio placeholders
        if "[" in txt and "]" in txt:
            continue  # reference citations
        if 3 <= len(txt) <= 100 and len(txt.split()) <= 18:
            defs.append(txt)
        if len(defs) >= 8:
            break
    if defs:
        parts.append("Definitions: " + " | ".join(dict.fromkeys(defs)))

    # 4. Inflection tables (conjugation / declension) — condensed rows.
    #    Prefer tables marked for inflection (de: class "inflection-table");
    #    fall back to any table with >= 3 rows.
    inflection = ""

    def _table_rows(table):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c[:60] for c in cells if c]
            if cells:
                rows.append(" | ".join(cells)[:140])
        return rows

    tables = []
    for table in content.find_all("table"):
        classes = " ".join(table.get("class") or []).lower()
        if not re.search(r"inflection|conjugation|declension|flexion|declina", classes):
            continue
        rows = _table_rows(table)
        if len(rows) >= 3:
            tables.append("\n".join(rows[:20]))
    if not tables:  # fallback: any substantial table
        for table in content.find_all("table"):
            rows = _table_rows(table)
            if len(rows) >= 3:
                tables.append("\n".join(rows[:20]))
    if tables:
        inflection = "\n".join(tables[:2])
        parts.append("Inflection:\n" + inflection[:900])

    text = "\n".join(parts).strip()

    # Disambiguation/stub pages have no part of speech, no translations and
    # no inflection — nothing usable for the tutor
    if not (h3 and h3.get_text(strip=True)) and not en and not inflection:
        return None
    if len(text) < 20:
        return None
    return text[:max_chars]
    if len(text) < 20:
        return None
    return text[:max_chars]


# ── Lookup with title fallbacks ──────────────────────────────────────

def _candidate_titles(term, lemma):
    """Title variants to try, deduplicated: term → capitalized → lemma."""
    term = (term or "").strip()
    lemma = (lemma or "").strip()
    # Case-sensitive dedup: "haus" and "Haus" are DIFFERENT candidates
    # (German nouns are capitalized — the user often types lowercase)
    seen, out = set(), []
    for t in (term, term.capitalize(), lemma, lemma.capitalize()):
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fetch_wiktionary_entry(term, lemma="", language="de", config=None, timeout=15):
    """
    Fetch one word's reference text from the configured Wiktionary backend.

    Tries title variants (as-given, capitalized, lemma hint) to handle
    German noun capitalization and inflected forms ("geht" → lemma "gehen").

    Returns the extracted reference text, or None when no page exists or
    the backend is unreachable.
    """
    term = (term or "").strip()
    if not term:
        return None

    backend, params = resolve_wiktionary_backend(config, language)
    titles = _candidate_titles(term, lemma)
    if not titles:
        return None

    if backend == "kiwix":
        client = KiwixClient(base_url=params["base_url"], zim_name=params["zim_name"])
        fetch = client.get_article
    else:
        client = WiktionaryClient(language=params["language"], timeout=timeout)
        fetch = client.get_entry_html

    try:
        for title in titles:
            try:
                html = fetch(title)
            except WikiFetchError:
                continue  # page missing — try next title variant
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    continue
                logger.warning("Wiktionary (kiwix) HTTP error for %r: %s", title, e)
                return None
            except requests.RequestException as e:
                logger.warning("Wiktionary backend unreachable (%s): %s", backend, e)
                return None
            entry = extract_wiktionary_entry(html)
            if entry:
                # If the hit is an inflected-form stub ("geht" → "Konjugierte
                # Form des Verbs gehen") and a lemma hint was given, re-fetch
                # the lemma — it carries the full conjugation/declension.
                pos = ""
                for line in entry.splitlines():
                    if line.startswith("Part of speech:"):
                        pos = line.split(":", 1)[1]
                        break
                if lemma and lemma.strip() != term \
                        and INFLICTED_FORM_RE.search(pos):
                    for lemma_title in _candidate_titles(lemma, ""):
                        try:
                            lemma_html = fetch(lemma_title)
                        except (WikiFetchError, requests.HTTPError):
                            continue
                        except requests.RequestException:
                            return entry  # backend flaky — keep what we have
                        lemma_entry = extract_wiktionary_entry(lemma_html)
                        if lemma_entry:
                            logger.info(
                                "Wiktionary [%s] stub %r → lemma %r (%d chars)",
                                backend, title, lemma_title, len(lemma_entry),
                            )
                            return lemma_entry
                logger.info("Wiktionary [%s] %r → %d chars", backend, title, len(entry))
                return entry
        logger.debug("Wiktionary: no usable page for %r (tried %s)", term, titles)
        return None
    finally:
        client.close()


# ── Multi-term reference block for the tutor ─────────────────────────

def get_dictionary_reference(terms, lemma="", language="de", config=None,
                             max_terms=MAX_TERMS, timeout=15):
    """
    Build a dictionary reference block for the tutor prompt.

    ``terms`` — words/phrases extracted from the user's message.
    Returns a joined reference block (""-separated per word), capped at
    REFERENCE_MAX_CHARS, or None when no term resolves.
    """
    if not terms:
        return None
    unique = list(dict.fromkeys(t.strip() for t in terms if isinstance(t, str) and t.strip()))
    blocks = []
    for term in unique[:max_terms]:
        entry = fetch_wiktionary_entry(term, lemma, language, config, timeout=timeout)
        if entry:
            blocks.append(f"=== {term} ===\n{entry}")
    if not blocks:
        return None
    return "\n\n".join(blocks)[:REFERENCE_MAX_CHARS]


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Look up a word in Wiktionary")
    parser.add_argument("--term", required=True, help="word to look up")
    parser.add_argument("--lemma", default="", help="optional base form")
    parser.add_argument("--lang", default="de", help="language code (e.g. de)")
    parser.add_argument("--config", default=None, help="path to config.json")
    args = parser.parse_args()

    config = {}
    if args.config:
        p = Path(args.config)
        if p.exists():
            config = json.loads(p.read_text(encoding="utf-8"))

    entry = get_dictionary_reference([args.term], args.lemma, args.lang, config)
    if entry is None:
        print(json.dumps({"term": args.term, "found": False}, ensure_ascii=False))
        sys.exit(1)
    print(entry)


if __name__ == "__main__":
    main()
