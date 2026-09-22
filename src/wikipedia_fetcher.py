#!/usr/bin/env python3
"""
Fetch a random Wikipedia article — offline (Kiwix/ZIM) or online (wikipedia.org).

Two backends share one extraction pipeline (see WikiClientBase):
  * KiwixClient     — local Kiwix Server (ZIM files), fully offline
  * WikipediaClient — wikipedia.org via the classic MediaWiki API
                      (/w/api.php?action=query&list=random + action=parse)

Backend resolution (see resolve_wikipedia_backend):
  * wikipedia.backend = "auto"   (default) — Kiwix if configured for the
    language, otherwise online
  * wikipedia.backend = "kiwix"  — offline only (falls back to online per
    language if no Kiwix server is configured)
  * wikipedia.backend = "online" — wikipedia.org only

Uses requests + BeautifulSoup for clean HTTP and HTML handling.

Usage:
    python3 src/wikipedia_fetcher.py                  # random article
    python3 src/wikipedia_fetcher.py --learning-language de
    python3 src/wikipedia_fetcher.py --config config.json
"""

import json
import logging
import os
import re
import sys
import time
from urllib.parse import unquote

logger = logging.getLogger(__name__)

import requests
from bs4 import BeautifulSoup

from config import (
    KIWIX_DEFAULT_BASE_URL,
    KIWIX_DEFAULT_ZIM_NAME,
    ARTICLE_FILTER_DEFAULTS,
    CLEAN_WORD_BUFFER,
)


# ── Paragraph-break detection regex ────────────────────────────────
# Matches sentence-ending punctuation + newline + uppercase letter (Latin-
# extended range covering DE, ES, IT, HU, FR, PL diacritics).  Used to
# insert proper paragraph breaks in Wikipedia text.
_PARAGRAPH_BREAK_RE = re.compile(
    r'([.!?])\n'
    r'([A-Z'
    r'\u00C0\u0104\u0126\u0138\u015A\u017D'
    r'\u0181\u0182\u0184\u0186\u0193'
    r'\u01A0\u01A2\u01B5\u01BF\u01C5\u01C7'
    r'\u01C9\u01CA\u01CC\u01CE\u01D0\u01D2'
    r'\u01D4\u01D6\u01D8\u01DA\u01DC\u01DE'
    r'\u01E0\u01E2\u01E4\u01E6\u01E8\u01EA'
    r'\u01EC\u01EE\u01F1\u01F3\u01F5\u01F7'
    r'\u01F9\u01FB\u01FD\u01FF'
    r'])',
)

# ── Shared Wikipedia client base ────────────────────────────────────

# Wikipedia's robot policy asks bots to identify themselves with contact
# info. Override via the WIKI_USER_AGENT env var.
WIKI_USER_AGENT = os.environ.get(
    "WIKI_USER_AGENT",
    "LinguaDaily/1.0 (personal daily-lesson bot; contact: administrator@localhost)",
)


class WikiFetchError(Exception):
    """Fatal fetch error (endpoint missing/unavailable) — abort the retry loop."""


class WikiClientBase:
    """Shared behavior for Wikipedia content clients (Kiwix / online).

    Subclasses implement two hooks:
      _fetch_random_title() -> (title, meta) | None
          Fetch one random candidate title. Return None to retry with a
          different article. Raise WikiFetchError for fatal errors
          (e.g. a Kiwix server without the /random endpoint).
      get_article(title) -> str
          Fetch the full article HTML for a title.

    Everything else (title filtering, prose checks, text extraction,
    disambiguation detection, word-count limits, smart truncation) is
    shared, so both backends produce identical output.
    """

    # Patterns to skip — English + common translations (DE, ES, IT, HU, FR, PL)
    SKIP_PATTERNS = [
        # English
        "List of", "list of", "List_of",
        "Glossary", "Glossary_of",
        "Index of", "index of", "Index_of",
        "Table of", "Table_of",
        "Bibliography", "Bibliography_of",
        "Outline of", "Outline_of",
        # German
        "Liste der", "Liste von", "Liste (",
        "Begriffsklärung", "Siehe auch",
        "Tafel der", "Verzeichnis",
        # Spanish
        "Lista de", "Anexo:Lista",
        "Glosario", "Índice de",
        "Tabla de",
        # Italian
        "Elenco di", "Elenco dei",
        "Glossario", "Indice di",
        # Hungarian
        "Listája", "-listák", "Jegyzék",
        "Táblázat", "Szójegyzék",
        # French
        "Liste de", "Liste des",
        "Glossaire", "Index de",
        "Table de",
        # Polish
        "Lista", "Wykaz", "Słownik",
        # Czech
        "Seznam", "Seznamy", "Přehled", "Tabulka",
        "Glosář", "Rejstřík",
    ]
    # Footer noise — English + translations (DE, ES, IT, HU, FR, PL) + live Wikipedia
    FOOTER_MARKERS = [
        # English (Kiwix footer)
        "This article is issued from Wikipedia",
        "Creative Commons",
        "Additional terms may apply",
        # English (live Wikipedia page footer)
        "This page was last edited",
        # German
        "Dieser Artikel wurde aus Wikipedia extrahiert",
        # Spanish
        "Este artículo fue extraído de Wikipedia",
        # Italian
        "Questo articolo è stato estratto da Wikipedia",
        # Hungarian
        "Ez a szócikk a Wikipédiából származik",
        # French
        "Cet article est issu de Wikipédia",
        # Polish
        "Artykuł pochodzi z Wikipedii",
        # Czech
        "Tento článek byl extrahován z Wikipedie",
    ]
    # Disambiguation page detection — English + common translations
    DISAMBIG_PATTERNS = [
        "may refer to",              # EN
        "kann sich beziehen auf",     # DE
        "puede referirse a",          # ES
        "può riferirsi a",            # IT
        "lehet több jelentése is",    # HU
        "peut faire référence à",     # FR
        "může znamenat",              # CS
        "viz rozcestník",             # CS (disambiguation page)
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": WIKI_USER_AGENT})

    # ── Random article (shared pipeline) ────────────────────────

    def get_random_article(self, max_attempts=20, min_words=250, max_words=600):
        """
        Fetch a random readable article suitable for language learning.

        Filters out lists, glossaries, disambiguation pages, and stubs.
        Truncates longer articles to max_words using coherent
        section/paragraph boundaries.

        Returns (title, text) — title is "Error" when no suitable
        article could be fetched.

        Raises nothing — all failures are reported as an "Error" title so
        callers can distinguish them from real content and abort the lesson.
        """
        # Abort early when the backend is persistently unreachable/rate-limited
        # (e.g. 429 storms) instead of burning all max_attempts on dead ends.
        max_consecutive_net_errors = 5
        consecutive_net_errors = 0
        last_skip = "no candidate drawn"

        for attempt in range(1, max_attempts + 1):
            try:
                result = self._fetch_random_title()
            except WikiFetchError as e:
                # Fatal backend error (e.g. ZIM not loaded)
                logger.warning("Random article fetch aborted (attempt %d/%d): %s",
                               attempt, max_attempts, e)
                return "Error", str(e)
            except requests.RequestException as e:
                # Transient network/HTTP error (429, timeout, ...) — respect
                # Retry-After when present (capped), then retry
                consecutive_net_errors += 1
                delay = 1
                resp = getattr(e, "response", None)
                if resp is not None:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = min(int(float(retry_after)), 10)
                        except (TypeError, ValueError):
                            pass
                logger.debug("Random article fetch: network error on attempt %d/%d "
                             "(consecutive: %d): %s — retrying in %ds",
                             attempt, max_attempts, consecutive_net_errors, e, delay)
                if consecutive_net_errors >= max_consecutive_net_errors:
                    logger.warning("Random article fetch aborted: %d consecutive "
                                   "network errors (last: %s)",
                                   consecutive_net_errors, e)
                    return "Error", (f"Backend unreachable after {consecutive_net_errors} "
                                     f"consecutive network errors (last: {e})")
                time.sleep(delay)
                continue
            consecutive_net_errors = 0

            if result is None:
                last_skip = "no candidate title returned"
                continue
            title, meta = result

            # Skip disambiguation pages flagged by the backend
            # (e.g. REST API 'type': 'disambiguation')
            if meta.get("disambiguation"):
                last_skip = f"{title!r} is a disambiguation page"
                logger.debug("Attempt %d/%d: %s", attempt, max_attempts, last_skip)
                continue

            # Quick title filter
            if any(skip in title for skip in self.SKIP_PATTERNS):
                last_skip = f"{title!r} matches a skip pattern"
                logger.debug("Attempt %d/%d: %s", attempt, max_attempts, last_skip)
                continue

            # Fetch the full article HTML via the backend
            try:
                html = self.get_article(title)
            except Exception as e:
                last_skip = f"{title!r}: article fetch failed ({e})"
                logger.debug("Attempt %d/%d: %s", attempt, max_attempts, last_skip)
                continue

            # Skip articles that are mostly tables/infoboxes with no prose
            if not _has_enough_prose(html):
                last_skip = f"{title!r} has not enough prose"
                logger.debug("Attempt %d/%d: %s", attempt, max_attempts, last_skip)
                continue

            text = extract_wiki_text(html)

            # Disambiguation page filter (multi-language patterns)
            if any(pat in text[:500] for pat in self.DISAMBIG_PATTERNS):
                last_skip = f"{title!r} looks like a disambiguation page"
                logger.debug("Attempt %d/%d: %s", attempt, max_attempts, last_skip)
                continue

            # Skip articles that are mostly table/infobox data (short lines)
            if _is_table_heavy(text):
                last_skip = f"{title!r} is mostly tables"
                logger.debug("Attempt %d/%d: %s", attempt, max_attempts, last_skip)
                continue

            word_count = len(text.split())

            # Add buffer: cleaning (parentheses removal, reference stripping, etc.)
            # removes words on average. Raise the minimum so that after cleaning
            # we still deliver at least min_words.
            effective_min = int(min_words / (1 - CLEAN_WORD_BUFFER))

            # Too short — skip
            if word_count < effective_min:
                last_skip = f"{title!r} too short ({word_count} < {effective_min} words)"
                logger.debug("Attempt %d/%d: %s", attempt, max_attempts, last_skip)
                continue

            html_title = _get_title_from_html(html) or title

            # Within max_words — return as-is (no truncation needed)
            if word_count <= max_words:
                return html_title, text.strip()

            # Too long — truncate at a coherent boundary
            truncated = smart_truncate(text, max_words=max_words, min_words=min_words)
            if truncated:
                return html_title, truncated
            # hard-truncate as last resort
            return html_title, hard_truncate(text, max_words=max_words)

        logger.warning("Random article fetch failed after %d attempts (last skip: %s)",
                       max_attempts, last_skip)
        return "Error", (f"Could not fetch a suitable random article after "
                         f"{max_attempts} attempts (last skip: {last_skip})")

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self):
        """Close the underlying session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Kiwix client (offline) ──────────────────────────────────────────

class KiwixClient(WikiClientBase):
    """Thin client for a Kiwix Server (ZIM reader)."""

    def __init__(self, base_url="http://192.168.100.52:8080", zim_name="wikipedia_en_all_maxi_2026-02"):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.zim_name = zim_name

    # ── HTTP helpers ──────────────────────────────────────────────

    def _get(self, path, params=None, timeout=15, allow_redirects=True):
        """GET a Kiwix endpoint and return the response.

        Kiwix Server omits the charset parameter in its Content-Type header,
        so requests defaults to ISO-8859-1 (HTTP/1.1 fallback) even though
        the actual content is UTF-8. Force UTF-8 after each request to avoid
        mojibake on non-ASCII characters.
        """
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=timeout, allow_redirects=allow_redirects)
        # Force UTF-8 — Kiwix serves ZIM content as UTF-8 but doesn't declare
        # it in Content-Type, so requests defaults to ISO-8859-1 (HTTP/1.1 fallback).
        resp.encoding = 'utf-8'
        return resp

    # ── Public API ────────────────────────────────────────────────

    def search(self, pattern, count=5, offset=0):
        """Search the ZIM file. Returns list of article titles.

        Uses 'content=' instead of 'book=' to avoid Kiwix's
        'confusion-of-tongues' error when multiple language ZIMs are loaded.
        """
        params = {
            "content": self.zim_name,
            "pattern": pattern,
            "offset": str(offset),
            "count": str(count),
        }
        resp = self._get("/search", params=params)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        prefix = f"/content/{self.zim_name}/"
        seen = set()
        titles = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href.startswith(prefix):
                title_encoded = href[len(prefix):]
                # Decode to human-readable form so skip_titles comparison works
                # (history stores decoded titles like "50 Tore in 50 Spielen",
                #  but Kiwix returns URL-encoded ones like "50_Tore_in_50_Spielen")
                title = unquote(title_encoded)
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
        return titles

    def get_article(self, title):
        """Fetch full article HTML for a given title. Returns HTML string."""
        # URL-encode the title (handles spaces, underscores, special chars).
        # Titles from Kiwix search results are already URL-encoded, so first
        # decode to get the raw title, then re-encode to avoid double-encoding
        # (%C3%A4 → %25C3%25A4) which causes 404 errors.
        from urllib.parse import quote
        raw = unquote(title)
        encoded = quote(raw, safe="_")
        resp = self._get(f"/content/{self.zim_name}/{encoded}")
        resp.raise_for_status()
        return resp.text

    def _fetch_random_title(self):
        """Random title via the Kiwix /random endpoint (302 redirect).

        Returns (title, {}) — or None to retry. Raises WikiFetchError if
        the server has no /random endpoint.
        """
        resp = self._get("/random", params={"content": self.zim_name},
                         timeout=15, allow_redirects=False)
        if resp.status_code == 404:
            raise WikiFetchError("/random endpoint not available on this Kiwix server.")

        # Follow the redirect — Location header contains the article path
        location = resp.headers.get("Location", "")
        if not location:
            return None

        # Extract title from the redirect URL: /content/ZIMNAME/Title
        prefix = f"/content/{self.zim_name}/"
        if location.startswith(prefix):
            title_raw = location[len(prefix):]
        elif location.startswith("/"):
            # Some versions return just /Title
            title_raw = location.lstrip("/")
        else:
            title_raw = location

        # URL-decode the title
        return unquote(title_raw), {}


# ── Online Wikipedia client (classic MediaWiki API) ────────────────

class WikipediaClient(WikiClientBase):
    """Client for online Wikipedia (wikipedia.org).

    Uses the classic MediaWiki API, which is available on every language
    wiki (the newer REST endpoints like /api/rest_v1/page/random only
    exist on a handful of wikis):

      GET /w/api.php?action=query&list=random&rnnamespace=0
          Random main-namespace article title.
      GET /w/api.php?action=parse&page={title}&prop=text
          Content-only HTML (the same mw-parser-output structure Kiwix
          serves), so the shared extraction pipeline works unchanged.

    Disambiguation pages are filtered by the shared pipeline's text
    heuristics (same as the Kiwix backend).

    No per-language configuration needed — the domain is derived from
    the language code (e.g. "de" → https://de.wikipedia.org).
    """

    def __init__(self, language="en", timeout=15):
        super().__init__()
        self.language = language.lower()
        self.timeout = timeout
        self.base_url = f"https://{self.language}.wikipedia.org"

    def _api(self, **params):
        """Call the classic MediaWiki API (available on every language wiki)."""
        resp = self.session.get(
            f"{self.base_url}/w/api.php", params=params, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _fetch_random_title(self):
        """Random main-namespace article title via the classic API."""
        data = self._api(
            action="query", list="random", rnnamespace="0",
            rnlimit="1", format="json",
        )
        random = (data.get("query") or {}).get("random") or []
        if not random:
            raise WikiFetchError(
                f"Wikipedia API returned no random page for '{self.language}'"
            )
        return random[0]["title"], {}

    def get_article(self, title):
        """Fetch content-only article HTML (mw-parser-output) for a title."""
        data = self._api(
            action="parse", page=title, prop="text", format="json",
        )
        if data.get("error"):
            code = data["error"].get("code", "unknown")
            raise WikiFetchError(
                f"Wikipedia API error for '{title}': {code}"
            )
        text = (data.get("parse") or {}).get("text", {}).get("*")
        if not text:
            raise WikiFetchError(
                f"Wikipedia API returned no content for '{title}'"
            )
        return text


# ── HTML extraction helpers ─────────────────────────────────────────

def _get_title_from_html(html):
    """Extract page title from HTML (prefers <h1>, falls back to <title>)."""
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title and title.get_text(strip=True):
        return title.get_text(strip=True)
    return None


def _has_enough_prose(html, min_paragraphs=5):
    """
    Check if the article has enough prose paragraphs (not just tables/infoboxes).

    Looks for <p> tags with meaningful text (>= 15 words) to determine if
    the article is suitable for language learning.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = None
    for selector in [
        "#mw-content-text > .mw-parser-output",
        "#mw-content-text",
        "#bodyContent",
    ]:
        content = soup.select_one(selector)
        if content:
            break
    if not content:
        content = soup

    prose_paras = 0
    for p in content.find_all("p"):
        text = p.get_text(strip=True)
        if len(text.split()) >= 15:
            prose_paras += 1
    return prose_paras >= min_paragraphs


def _is_table_heavy(text, min_prose_lines=8):
    """
    Check if the extracted text is mostly short lines (indicating table/infobox data).

    Returns True if there are fewer than min_prose_lines of lines with >= 10 words.
    This catches pure lists/tables while allowing articles with many wiki-links.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    prose_lines = sum(1 for l in lines if len(l.split()) >= 10)
    return prose_lines < min_prose_lines


def extract_wiki_text(html, skip_infoboxes=True):
    """
    Extract readable text from Wikipedia/Kiwix HTML.

    Targets the main content area to avoid navigation chrome, then strips
    common footer noise.

    Parameters
    ----------
    html : str
        Raw article HTML.
    skip_infoboxes : bool
        If True, removes infoboxes and data-heavy tables so that only
        prose paragraphs are returned. Use False to get everything.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Prefer the article body over the full page
    content = None
    for selector in [
        "#mw-content-text > .mw-parser-output",
        "#mw-content-text",
        "#bodyContent .mw-parser-output",
        "#bodyContent",
    ]:
        content = soup.select_one(selector)
        if content:
            break

    # Fallback: parse the whole page
    if not content:
        content = soup

    # Remove script/style/nav/sidebar noise
    for tag in content.find_all(["script", "style", "noscript", "nav", ".mw-hidden-catlinks", ".reflist", ".mw-references-wrap"]):
        tag.decompose()

    if skip_infoboxes:
        # Remove infoboxes (they produce vertical one-word-per-line noise)
        for table in content.find_all("table"):
            if table is None:
                continue
            try:
                classes = table.get("class", []) or []
                if any(cls.startswith(("infobox", "vcard", "navbox", "ambox", "metadata")) for cls in classes):
                    table.decompose()
            except Exception:
                pass

        # Remove data-heavy tables (tables with many rows but little prose)
        for table in content.find_all("table"):
            if table is None:
                continue
            try:
                rows = table.find_all("tr")
                if len(rows) > 10:
                    cell_count = len(table.find_all(["td", "th"]))
                    if cell_count > 0:
                        short_cells = sum(1 for c in table.find_all(["td", "th"]) if len(c.get_text(strip=True).split()) < 8)
                        if short_cells / cell_count > 0.7:
                            table.decompose()
            except Exception:
                pass

    # Ensure heading elements (h2, h3, etc.) get their own paragraph boundary
    # by inserting blank lines before and after them. This way get_text(separator="\n")
    # will produce \n\n around headings, creating proper paragraph breaks.
    for tag in content.find_all(["h2", "h3", "h4", "h5", "h6"]):
        if tag.previous_sibling is None or not isinstance(tag.previous_sibling, str) or not tag.previous_sibling.strip():
            tag.insert_before(soup.new_string("\n"))
        if tag.next_sibling is None or not isinstance(tag.next_sibling, str) or not tag.next_sibling.strip():
            tag.insert_after(soup.new_string("\n"))

    # Get clean text
    text = content.get_text(separator="\n", strip=True)

    # Collapse blank lines → keep paragraph boundaries as double-newlines
    lines = text.split("\n")
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            prev_blank = True
        else:
            if prev_blank:
                cleaned.append("")  # blank line creates \n\n boundary when joined
            cleaned.append(stripped)
            prev_blank = False
    text = "\n".join(cleaned)

    # Post-process: Wikipedia HTML often produces single-newline boundaries
    # between paragraphs (no real blank lines). Insert double-newlines after
    # sentence-ending punctuation followed by a newline and an uppercase letter,
    # so smart_truncate can split on paragraph boundaries.
    text = _PARAGRAPH_BREAK_RE.sub(r'\1\n\n\2', text)

    # Remove footer noise
    for marker in KiwixClient.FOOTER_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].rstrip()

    return text


# ── Smart truncation ──────────────────────────────────────────────

def hard_truncate(text, max_words=600):
    """
    Hard-truncate text to max_words at a word boundary.

    Last-resort fallback when smart_truncate fails — guarantees the output
    never exceeds max_words, even if it cuts mid-sentence.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


def smart_truncate(text, max_words=600, min_words=250):
    """
    Truncate article text to at most max_words at a coherent boundary.

    Strategy (three-pass): tries sections → paragraphs → sentences.
    Each pass greedily accumulates chunks until adding the next one would
    exceed max_words. The first pass that produces >= min_words wins.
    This gives the best structural fidelity while staying under the cap.

    Returns the truncated text, or None if no usable chunk >= min_words
    could be produced.
    """
    # Pass 1: section-level splitting (best structural coherence)
    result = _accumulate_by_sections(text, max_words, min_words)
    if result:
        return result

    # Pass 2: paragraph-level splitting (fallback for articles with no
    # section markers or a single huge lead section)
    result = _accumulate_by_paragraphs(text, max_words, min_words)
    if result:
        return result

    # Pass 3: sentence-level splitting (last resort for dense text with
    # no paragraph breaks — e.g. bibliography-heavy Wikipedia articles)
    result = _accumulate_by_sentences(text, max_words, min_words)
    return result


def _split_sections(text):
    """
    Split text on Wikipedia-style section headers (==Header==).
    Returns a list of (header_line_or_None, body_text) tuples.
    Non-header lines before any header are grouped under None.
    """
    sections = []
    current_header = None
    current_body_lines = []

    for line in text.split("\n"):
        if re.match(r'^={2,}', line.strip()) and line.strip().endswith('='):
            if current_body_lines or current_header is not None:
                body = "\n".join(current_body_lines).strip()
                if body:
                    sections.append((current_header, body))
            current_header = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    body = "\n".join(current_body_lines).strip()
    if body:
        sections.append((current_header, body))

    return sections


def _accumulate_by_sections(text, max_words, min_words):
    """Accumulate complete sections until hitting max_words."""
    sections = _split_sections(text)
    if not sections:
        return None

    accumulated = []
    total = 0
    for header, body in sections:
        body_words = len(body.split())
        header_words = len(header.split()) if header else 0
        chunk_words = header_words + body_words

        if total + chunk_words > max_words:
            break

        if header:
            accumulated.append(header)
        accumulated.append(body)
        total += chunk_words

    result = "\n\n".join(accumulated).strip()
    if total >= min_words:
        return result
    return None


def _accumulate_by_sentences(text, max_words, min_words):
    r"""Accumulate sentences (split on [.!?]\s+) until hitting max_words."""
    # Split on sentence-ending punctuation followed by whitespace/newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return None

    accumulated = []
    total = 0
    for sent in sentences:
        sent_words = len(sent.split())

        if total + sent_words > max_words:
            break

        accumulated.append(sent)
        total += sent_words

    result = " ".join(accumulated).strip()
    if total >= min_words:
        return result
    return None


def _accumulate_by_paragraphs(text, max_words, min_words):
    """Accumulate complete paragraphs (blank-line separated) until hitting max_words."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return None

    accumulated = []
    total = 0
    for para in paragraphs:
        para_words = len(para.split())

        if total + para_words > max_words:
            break

        accumulated.append(para)
        total += para_words

    result = "\n\n".join(accumulated).strip()
    if total >= min_words:
        return result
    return None


# ── CLI helpers ───────────────────────────────────────────────────

def parse_cli_args(args):
    config_path = None
    learning_language = None
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 1
        elif arg == "--learning-language" and i + 1 < len(args):
            learning_language = args[i + 1]
            i += 1
        elif arg in ("--min-words", "--max-words") and i + 1 < len(args):
            overrides[arg.lstrip("-").replace("-", "_")] = int(args[i + 1])
            i += 1
        i += 1
    return config_path, learning_language, overrides


def resolve_wikipedia_backend(config, language=None):
    """
    Resolve which Wikipedia backend to use for a language.

    Returns (backend, params):
      ("kiwix",  {"base_url": str, "zim_name": str})
      ("online", {"language": str})

    Selection policy:
      * config["wikipedia"]["backend"] == "online" → always wikipedia.org
      * == "kiwix" → Kiwix if a server is configured for the language,
        otherwise online (fallback so the pipeline never breaks)
      * default "auto" → Kiwix if a server is configured for the language,
        otherwise online

    Server resolution: kiwix_servers[language], falling back to the legacy
    top-level 'kiwix' block.
    """
    lang = (language or "en").lower()
    backend = ((config.get("wikipedia") or {}).get("backend") or "auto").lower()

    kiwix_servers = config.get("kiwix_servers") or {}
    kiwix_cfg = kiwix_servers.get(lang) or config.get("kiwix") or {}
    has_kiwix = bool(kiwix_cfg)

    if backend == "online" or not (has_kiwix and backend in ("auto", "kiwix")):
        return "online", {"language": lang}

    return "kiwix", {
        "base_url": kiwix_cfg.get("base_url", KIWIX_DEFAULT_BASE_URL),
        "zim_name": kiwix_cfg.get("zim_name", KIWIX_DEFAULT_ZIM_NAME),
    }


def load_fetcher_config(config_path=None, learning_language=None):
    """
    Load fetcher configuration from config.json.

    Parameters
    ----------
    config_path : str or None
        Path to config.json. Defaults to project root.
    learning_language : str or None
        Language code (e.g. "de", "en"). Resolves the Wikipedia backend
        (Kiwix / online) for this language — see resolve_wikipedia_backend.

    Returns
    -------
    dict with keys: backend ("kiwix"|"online"), language, base_url,
    zim_name (Kiwix only), article_filter.
    """
    if config_path is None:
        from config import CONFIG_PATH as config_path

    settings = {
        "backend": "kiwix",
        "language": (learning_language or "en").lower(),
        "base_url": KIWIX_DEFAULT_BASE_URL,
        "zim_name": KIWIX_DEFAULT_ZIM_NAME,
        "article_filter": ARTICLE_FILTER_DEFAULTS.copy(),
    }

    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        backend, params = resolve_wikipedia_backend(config, learning_language)
        settings["backend"] = backend
        if backend == "kiwix":
            settings["base_url"] = params["base_url"]
            settings["zim_name"] = params["zim_name"]

        af = config.get("article_filter", {})
        for key in settings["article_filter"]:
            settings["article_filter"][key] = af.get(key, settings["article_filter"][key])

    return settings


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    config_path, learning_language, overrides = parse_cli_args(sys.argv[1:])
    settings = load_fetcher_config(config_path,
                                   learning_language=learning_language)

    af = settings["article_filter"]
    # CLI overrides take precedence
    if overrides:
        af.update(overrides)
    min_words = af["min_words"]
    max_words = af["max_words"]

    if settings["backend"] == "online":
        client = WikipediaClient(language=settings["language"])
        source_label = f"wikipedia.org ({settings['language']})"
    else:
        client = KiwixClient(base_url=settings["base_url"],
                             zim_name=settings["zim_name"])
        source_label = f"Kiwix ({settings['zim_name']})"

    with client:
        title, text = client.get_random_article(
            min_words=min_words,
            max_words=max_words,
        )

    # Output as structured payload
    result = {
        "title": title,
        "text": text,
        "source": source_label,
        "word_count": len(text.split()),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
