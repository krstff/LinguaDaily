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
  * analyze_wiktionary_entry()  — HTML → completeness/pointer signals
  * fetch_wiktionary_entry()    — title fallback + stub/lemma resolution
  * get_dictionary_reference()  — multi-term reference block for the tutor

Stub/lemma resolution (fetch_wiktionary_entry)
----------------------------------------------
Wiktionary has two page types, and some words resolve to the wrong one:
  * full entries (lemma): definitions + translations + inflection tables
  * pointer/stub pages: "this word is a form of X" — data lives on X
Multilingual pages also contain sections for many languages; the page may
have no content at all for the lookup language.  Resolution:

  0. Language scope:  target-language sections only (heading match);
                      no match + headings exist → page irrelevant → None
  1. Complete?       inflection table OR substantive definitions (scoped)
                      → done, extract
  2. Pointer?        Tier 1: form-of-* standardized classes (scoped) → link
                      Tier 2: word links inside the scoped sections, outside
                      translation tables (incl. "lásd/Siehe/see" navboxes)
                      → unique/majority candidate → fetch it (ONE hop),
                      re-check 0+1, merge with a form note
  3. Otherwise       keep what we have (stub note), never crash

The LLM "lemma" hint is a tiebreaker only, never the driver.

Usage:
    python3 src/wiktionary_client.py --term gehen --lang de --config config.json
"""

import json
import logging
import re
import sys
import unicodedata
from collections import Counter
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


# ── Language scoping (step 0) ─────────────────────────────────────────

# Folded language-name patterns per TARGET language.  A page section is
# "for us" when its h2 heading (text or extiw link title) contains one of
# the folded patterns.  Patterns are stored folded (diacritics stripped,
# casefolded).
#   * de/cs/hu/it/es — the endonym as written on that language's own wiki
#     ("Haus (Deutsch)", bare "čeština", bare "Magyar", …)
#   * en — the English exonyms, because on en.wiktionary.org the
#     language sections are headed "English", "Czech", "German", …
LANG_HEADINGS = {
    "de": ("deutsch",),
    "en": ("english", "czech", "german", "hungarian", "spanish", "italian"),
    "cs": ("cestina", "cesky"),
    "hu": ("magyar",),
    "it": ("italiano",),
    "es": ("espanol", "castellano"),
}

# Every language name known to the scoping table — Tier 2 word links whose
# text is one of these are language mentions, not lemma candidates.
_ALL_LANGUAGE_NAMES = {p for pats in LANG_HEADINGS.values() for p in pats}

# Link title prefixes (folded, with trailing colon) that are never words.
_EXCLUDED_LINK_PREFIXES = (
    "appendix:", "category:", "datei:", "draft:", "file:", "flexion:",
    "help:", "hilfe:", "index:", "kategorie:", "q:", "reim:", "rhymes:",
    "schablone:", "school:", "soubor:", "template:", "talk:", "topic:",
    "user:", "wikipedia:", "wikislovník:", "wikisótar:", "projet:",
    "proyecto:", "progetto:", "discusión:", "discussion:",
)

_INFLECTION_TABLE_RE = re.compile(
    r"inflection|conjugation|declension|flexion|declina", re.IGNORECASE)
_TRANSLATIONS_ANCESTOR_RE = re.compile(r"translations", re.IGNORECASE)
_FORM_OF_RE = re.compile(r"form-of|use-with-mention", re.IGNORECASE)
# Labeled boilerplate lines: "IPA: …", "Hyphenation: …", "angličtina: …"
_LABELED_LINE_RE = re.compile(r"^[\w\-'() ]{1,25}\s*:\s*")
_H2_RE = re.compile(r"<h2[^>]*>.*?</h2>", re.DOTALL | re.IGNORECASE)
_WIKI_HREF_RE = re.compile(r"^(?:/wiki/|\./)")


def _fold(text):
    """Casefold + diacritic-fold (NFD, strip combining marks) + squash."""
    s = unicodedata.normalize("NFD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.casefold()).strip()


def _split_sections(html):
    """Split a Wiktionary page into (top_html, sections).

    ``sections`` — list of {html, heading_text, extiw_title}; each ``html``
    runs from one <h2> up to the next.  ``top_html`` is everything before
    the first <h2> ("" when the page has none).
    """
    matches = list(_H2_RE.finditer(html or ""))
    if not matches:
        return (html or ""), []
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(html)
        soup = BeautifulSoup(m.group(0), "html.parser")
        h2 = soup.find("h2")
        extiw = h2.find("a", class_="extiw") if h2 else None
        sections.append({
            "html": html[m.start():end],
            "heading_text": h2.get_text(" ", strip=True) if h2 else "",
            "extiw_title": extiw.get("title", "") if extiw and extiw.get("title") else "",
        })
    return html[:matches[0].start()], sections


def _heading_matches_language(heading_text, extiw_title, language):
    patterns = LANG_HEADINGS.get((language or "").lower())
    if not patterns:
        return False
    for text in (heading_text, extiw_title):
        if not text:
            continue
        folded = _fold(text)
        if any(p in folded for p in patterns):
            return True
    return False


def _target_sections(top_html, sections, language):
    """Step 0 — the sections of a page relevant to the target language.

    Returns a list of section HTML strings, or None when the page is
    irrelevant for this language (headings exist, none match).  The top
    section is included iff at least one heading matched, or the page has
    no headings at all (then the whole page is the language's edition).
    """
    matched = [
        s for s in sections
        if _heading_matches_language(s["heading_text"], s["extiw_title"], language)
    ]
    if matched:
        out = [top_html] if top_html.strip() else []
        return out + [s["html"] for s in matched]
    if not sections:
        return [top_html] if top_html.strip() else []
    # No heading matches.  Several language sections without one for us =
    # multilingual page missing our language (hu.wikt "men": Dán, Feröeri, …)
    # → irrelevant.  A SINGLE language section is unambiguous (de.wikt
    # "nosit (Tschechisch)") → use it.
    if len(sections) == 1:
        out = [top_html] if top_html.strip() else []
        return out + [sections[0]["html"]]
    return None


# ── HTML → compact reference text ────────────────────────────────────

def extract_wiktionary_entry(html, max_chars=ENTRY_MAX_CHARS):
    """
    Extract a compact reference block from Wiktionary section HTML.

    Keeps what the tutor needs (unlike the article extractor, which
    removes tables — a wiktionary page *is* mostly tables and lists):
      * part of speech (first h3, e.g. "Verb, unregelmäßig, intransitiv")
      * definition list items (translation list items are excluded)
      * inflection tables (conjugation / declension), condensed
      * English translations (spans marked lang="en")

    Accepts either a full page or pre-scoped section HTML (both render to
    .mw-parser-output-style fragments).  Returns a string, or None when
    the content has no usable part (disambiguation/stub pages).  Works on
    both Kiwix-served HTML and the MediaWiki API's action=parse output.
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
    #    interwiki language link with class "extiw") and form-of pointer
    #    boilerplate ("third-person singular present indicative", …)
    defs = []
    for li in content.find_all("li"):
        txt = li.get_text(" ", strip=True)
        if "→" in txt:
            continue
        if li.find(class_="extiw") or li.find("span", lang=True):
            continue
        if li.find(class_=_FORM_OF_RE):
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

    # 4. Inflection tables (conjugation / declension) — condensed rows
    #    with the mood/tense cell inherited on continuation rows (rowspan
    #    flattening), so every row is self-contained for the model.
    #    Class hints: de/hu "inflection-table", cs "konjugace", en
    #    "conjugation"/"declension"; fallback: any substantial table.
    #    Budget 1500 chars / up to 4 tables (cs verbs need present +
    #    imperative + past + participle tables; full hu verb ≈ 750).
    inflection = ""

    def _table_rows(table):
        rows = []
        prev_first = ""
        prev_count = 0
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c[:60] for c in cells if c]
            if not cells:
                continue
            if prev_count and len(cells) < prev_count and prev_first:
                cells = [prev_first] + cells
            prev_count = len(cells)
            prev_first = cells[0]
            rows.append(" | ".join(cells)[:140])
        return rows

    tables = []
    for table in content.find_all("table"):
        classes = " ".join(table.get("class") or []).lower()
        if not re.search(
                r"inflection|conjugation|declension|flexion|declina|konjugace",
                classes):
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
        inflection = "\n".join(tables[:4])
        parts.append("Inflection:\n" + inflection[:1500])

    text = "\n".join(parts).strip()

    # Disambiguation/stub pages have no part of speech, no translations and
    # no inflection — nothing usable for the tutor
    if not (h3 and h3.get_text(strip=True)) and not en and not inflection:
        return None
    if len(text) < 20:
        return None
    return text[:max_chars]


# ── Structured signal pass (steps 1 + 2) ─────────────────────────────

def _is_substantive_definition(li):
    """True when a <li> is a real definition the tutor can use."""
    text = re.sub(r"\s+", " ", li.get_text(" ", strip=True)).strip()
    if len(text) < 15:
        return False
    if "→" in text or text.endswith(":"):
        return False
    if _LABELED_LINE_RE.match(text):
        return False  # pronunciation/label boilerplate, not a definition
    # IPA / pronunciation lines are symbol-heavy, not word-heavy
    letters = sum(1 for c in text if c.isalpha())
    if letters * 2 < len(text):
        return False
    # Inflection pointer lines end with the lemma link
    # ("… des Verbs gehen", "… form of <a>pracovat</a>")
    for a in reversed(li.find_all("a", href=True)):
        atext = a.get_text(" ", strip=True)
        if atext and text.endswith(atext) and _WIKI_HREF_RE.match(a["href"]):
            return False
    return True


def analyze_wiktionary_entry(html):
    """
    Structured signal pass over already language-scoped section HTML.

    Returns a dict:
      has_inflection_table        — bool
      has_substantive_definitions — bool (translations never count)
      form_of_lemma               — str | None (Tier 1 standardized classes)
      word_link_counts            — Counter of Tier 2 candidate word links
    """
    soup = BeautifulSoup(html, "html.parser")

    has_table = False
    for table in soup.find_all("table"):
        classes = " ".join(table.get("class") or [])
        if _INFLECTION_TABLE_RE.search(classes) and table.find("tr"):
            has_table = True
            break

    has_defs = False
    for li in soup.find_all("li"):
        if li.find(class_="extiw") or li.find("span", lang=True):
            continue
        if li.find_parent(class_=_TRANSLATIONS_ANCESTOR_RE):
            continue
        if li.find(class_=_FORM_OF_RE):
            continue  # form-of pointer boilerplate, not a definition
        if _is_substantive_definition(li):
            has_defs = True
            break

    # Tier 1 — standardized form-of classes (en/hu/… editions)
    form_of = None
    for span in soup.find_all("span", class_=["form-of-definition", "use-with-mention"]):
        link_span = span.find("span", class_="form-of-definition-link")
        a = (link_span.find("a") if link_span else None) or span.find("a")
        if a is None:
            continue
        title = (a.get("title") or a.get_text(" ", strip=True)).split("#", 1)[0].strip()
        if not title or _fold(title).startswith(_EXCLUDED_LINK_PREFIXES):
            continue
        form_of = title
        break

    # Tier 2 — plain word links (shape heuristic for editions without the
    # classes, e.g. German stubs and "lásd/Siehe" navboxes)
    counts = Counter()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not _WIKI_HREF_RE.match(href):
            continue
        if "action=edit" in href:
            continue
        classes = a.get("class") or []
        if "new" in classes:
            continue  # red link — page does not exist
        if a.find_parent(class_=_TRANSLATIONS_ANCESTOR_RE):
            continue
        title = (a.get("title") or a.get_text(" ", strip=True)).split("#", 1)[0].strip()
        if not title:
            continue
        folded = _fold(title)
        if folded.startswith(_EXCLUDED_LINK_PREFIXES):
            continue
        if folded in _ALL_LANGUAGE_NAMES:
            continue
        counts[title] += 1

    return {
        "has_inflection_table": has_table,
        "has_substantive_definitions": has_defs,
        "form_of_lemma": form_of,
        "word_link_counts": counts,
    }


def _pick_lemma_candidate(signals, term, lemma=""):
    """
    Choose a lemma candidate from the signal pass.

    Tier 1 (form-of classes) wins outright; Tier 2 needs exactly one
    distinct candidate, a clear majority (>= 3 and >= 2x the runner-up),
    or the LLM lemma hint as a tiebreaker among the candidates.
    Returns a title, or None.
    """
    term = (term or "").strip()
    lemma = (lemma or "").strip()

    form_of = signals.get("form_of_lemma")
    if form_of and _fold(form_of) != _fold(term):
        return form_of

    counts = signals.get("word_link_counts") or Counter()
    if term:
        tf = _fold(term)
        counts = Counter({t: n for t, n in counts.items() if _fold(t) != tf})
    if not counts:
        return None

    ranked = counts.most_common()
    if len(ranked) == 1:
        return ranked[0][0]

    top_name, top_count = ranked[0]
    if top_count >= 3 and top_count >= 2 * ranked[1][1]:
        return top_name

    if lemma:
        lf = _fold(lemma)
        for t in counts:
            if _fold(t) == lf:
                return t
    return None


# ── Stub/lemma resolution (steps 0–3) ────────────────────────────────

def _scoped_signals(html, language):
    """Steps 0+1+2 signals for one page: (scoped_html, signals) or (None, None)."""
    top, sections = _split_sections(html)
    scoped = _target_sections(top, sections, language)
    if scoped is None:
        return None, None
    scoped_html = "\n".join(scoped)
    return scoped_html, analyze_wiktionary_entry(scoped_html)


def _is_complete(signals):
    return signals["has_inflection_table"] or signals["has_substantive_definitions"]


def _resolve_entry(html, term, lemma, language, fetch):
    """
    Resolve one fetched page to a reference text (steps 0–3).

    ``fetch`` is the backend's HTML fetch (used for the single lemma hop);
    pass None to disable hopping.  Returns a text block, or None.
    """
    scoped_html, signals = _scoped_signals(html, language)
    if scoped_html is None:
        return None  # step 0: page has no target-language section

    # Step 1 — complete entry, done
    if _is_complete(signals):
        text = extract_wiktionary_entry(scoped_html)
        if text:
            return text

    # Step 2 — pointer page: resolve the lemma, at most ONE hop
    candidate = _pick_lemma_candidate(signals, term, lemma)
    if candidate and fetch is not None:
        try:
            cand_html = fetch(candidate)
        except (WikiFetchError, requests.HTTPError):
            cand_html = None  # candidate missing — keep what we have
        except requests.RequestException as e:
            logger.warning("Wiktionary lemma hop failed for %r: %s", candidate, e)
            cand_html = None
        if cand_html is not None:
            cand_scoped, cand_signals = _scoped_signals(cand_html, language)
            if cand_scoped is not None and _is_complete(cand_signals):
                cand_text = extract_wiktionary_entry(cand_scoped)
                if cand_text:
                    logger.info("Wiktionary pointer %r → %r", term, candidate)
                    return f"Form: {term} — see {candidate}\n{cand_text}"

    # Step 3 — keep what we have (with a pointer note when we have one)
    text = extract_wiktionary_entry(scoped_html)
    if text:
        if candidate:
            text += f"\nNote: {term} appears to be a form of {candidate}."
        return text
    if candidate:
        return f"Form: {term} — see {candidate}"
    return None


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
    German noun capitalization and inflected forms ("geht" → lemma "gehen"),
    then resolves pointer/stub pages to the lemma entry (one hop max).

    Returns the extracted reference text, or None when no page exists, the
    page is irrelevant for the language, or the backend is unreachable.
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
            entry = _resolve_entry(html, term, lemma, language, fetch)
            if entry:
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
