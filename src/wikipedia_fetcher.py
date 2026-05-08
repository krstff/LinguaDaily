#!/usr/bin/env python3
"""
Fetch a random Wikipedia article from a local Kiwix/ZIM server.

Uses requests + BeautifulSoup for clean HTTP and HTML handling.

Usage:
    python3 src/wikipedia_fetcher.py                  # random article
    python3 src/wikipedia_fetcher.py "quantum"        # search-based fetch
    python3 src/wikipedia_fetcher.py --config config.json  # read Kiwix settings from config
"""

import json
import os
import random
import sys

import requests
from bs4 import BeautifulSoup

DEFAULT_BASE_URL = "http://192.168.100.52:8080"
DEFAULT_ZIM_NAME = "wikipedia_en_all_maxi_2026-02"
DEFAULT_ARTICLE_FILTER = {"min_words": 250, "target_words": 400, "max_words": 600}


# ── Kiwix client ────────────────────────────────────────────────────

class KiwixClient:
    """Thin client for a Kiwix Server (ZIM reader)."""

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
    # Footer noise — English + translations (DE, ES, IT, HU, FR, PL)
    FOOTER_MARKERS = [
        # English
        "This article is issued from Wikipedia",
        "Creative Commons",
        "Additional terms may apply",
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
    CONTENT_SELECTOR = "#mw-content-text, #bodyContent, .mw-parser-output"

    def __init__(self, base_url="http://192.168.100.52:8080", zim_name="wikipedia_en_all_maxi_2026-02"):
        self.base_url = base_url.rstrip("/")
        self.zim_name = zim_name
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OpenClaw-Lingua/1.0"})

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
        titles = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href.startswith(prefix):
                titles.append(href[len(prefix):])
        return titles

    def get_article(self, title):
        """Fetch full article HTML for a given title. Returns a Response."""
        # URL-encode the title (handles spaces, underscores, special chars).
        # Titles from Kiwix search results are already URL-encoded, so first
        # decode to get the raw title, then re-encode to avoid double-encoding
        # (%C3%A4 → %25C3%25A4) which causes 404 errors.
        from urllib.parse import quote, unquote
        raw = unquote(title)
        encoded = quote(raw, safe="_")
        resp = self._get(f"/content/{self.zim_name}/{encoded}")
        resp.raise_for_status()
        return resp

    # ── Random article (via /random endpoint) ────────────────────

    def get_random_article(self, max_attempts=10, min_words=250, target_words=400, max_words=600):
        """
        Fetch a random readable article suitable for language learning.

        Uses the Kiwix /random endpoint (follows 302 redirect to get the
        actual article path). Filters out lists, glossaries, disambiguation
        pages, and stubs. Prefers articles in the target word range;
        smart-truncates longer ones to complete sections.
        """
        for _ in range(max_attempts):
            # Kiwix /random?content=ZIMNAME returns a 302 redirect to the article
            resp = self._get("/random", params={"content": self.zim_name}, timeout=15, allow_redirects=False)
            if resp.status_code == 404:
                return "Error", "/random endpoint not available on this Kiwix server."

            # Follow the redirect — Location header contains the article path
            location = resp.headers.get("Location", "")
            if not location:
                continue

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
            from urllib.parse import unquote
            title = unquote(title_raw)

            # Quick title filter
            if any(skip in title for skip in self.SKIP_PATTERNS):
                continue

            # Fetch the full article HTML via content endpoint
            article_resp = self.get_article(title)

            # Skip articles that are mostly tables/infoboxes with no prose
            if not _has_enough_prose(article_resp.text):
                continue

            text = extract_wiki_text(article_resp.text)

            # Disambiguation page filter (multi-language patterns)
            disambig_patterns = [
                "may refer to",              # EN
                "kann sich beziehen auf",     # DE
                "puede referirse a",          # ES
                "può riferirsi a",            # IT
                "lehet több jelentése is",    # HU
                "peut faire référence à",     # FR
                "může znamenat",              # CS
                "viz rozcestník",             # CS (disambiguation page)
            ]
            if any(pat in text[:500] for pat in disambig_patterns):
                continue

            # Skip articles that are mostly table/infobox data (short lines)
            if _is_table_heavy(text):
                continue

            word_count = len(text.split())

            # Too short — skip
            if word_count < min_words:
                continue

            # Perfect length — keep as-is
            if word_count <= min_words:
                html_title = _get_title_from_html(article_resp.text)
                return html_title or title, text

            # In range but above min — smart-truncate to target
            if word_count <= max_words:
                truncated = smart_truncate(text, target_words=target_words, max_words=max_words, min_words=min_words)
                if truncated:
                    html_title = _get_title_from_html(article_resp.text)
                    return html_title or title, truncated
                # If truncation failed, return as-is
                html_title = _get_title_from_html(article_resp.text)
                return html_title or title, text

            # Too long — smart-truncate to a coherent chunk
            truncated = smart_truncate(text, target_words=target_words, max_words=max_words, min_words=min_words)
            if truncated:
                html_title = _get_title_from_html(article_resp.text)
                return html_title or title, truncated

        return "Error", "Could not fetch a suitable random article after multiple attempts."

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self):
        """Close the underlying session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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
    import re
    text = re.sub(r'([.!?])\n([A-Z\u00C0\u0104\u0126\u0138\u015A\u017D\u0181\u0182\u0184\u0186\u0193\u01A0\u01A2\u01B5\u01BF\u01C5\u01C7\u01C9\u01CA\u01CC\u01CE\u01D0\u01D2\u01D4\u01D6\u01D8\u01DA\u01DC\u01DE\u01E0\u01E2\u01E4\u01E6\u01E8\u01EA\u01EC\u01EE\u01F1\u01F3\u01F5\u01F7\u01F9\u01FB\u01FD\u01FF])', r'\1\n\n\2', text)

    # Remove footer noise
    for marker in KiwixClient.FOOTER_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].rstrip()

    return text


# ── Smart truncation ──────────────────────────────────────────────

def smart_truncate(text, target_words=400, max_words=600, min_words=250):
    """
    Truncate article text to a coherent chunk of roughly target_words.

    Strategy (three-pass):
      1. Split on Wikipedia section markers ("==...==" lines) and greedily
         accumulate complete sections until hitting target_words.
      2. If the first section itself is too long, fall back to splitting on
         blank-line-separated paragraphs and accumulating those instead.
      3. Last resort: split on sentence boundaries and accumulate sentences.

    Returns the truncated text, or None if no usable chunk >= min_words
    could be produced.
    """
    # Pass 1: section-level splitting
    result = _accumulate_by_sections(text, target_words, max_words, min_words)
    if result:
        return result

    # Pass 2: paragraph-level splitting (fallback for articles with no
    # section markers or a single huge lead section)
    result = _accumulate_by_paragraphs(text, target_words, max_words, min_words)
    if result:
        return result

    # Pass 3: sentence-level splitting (last resort for dense text with
    # no paragraph breaks — e.g. bibliography-heavy Wikipedia articles)
    result = _accumulate_by_sentences(text, target_words, max_words, min_words)
    return result


def _split_sections(text):
    """
    Split text on Wikipedia-style section headers (==Header==).
    Returns a list of (header_line_or_None, body_text) tuples.
    Non-header lines before any header are grouped under None.
    """
    import re
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


def _accumulate_by_sections(text, target_words, max_words, min_words):
    """Accumulate complete sections until near target_words."""
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


def _accumulate_by_sentences(text, target_words, max_words, min_words):
    r"""Accumulate sentences (split on [.!?]\s+) until near target_words."""
    import re
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


def _accumulate_by_paragraphs(text, target_words, max_words, min_words):
    """Accumulate complete paragraphs (blank-line separated) until near target_words."""
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
    content_lang = None
    search_query = None
    overrides = {}
    i = 0
    positional_args = []
    while i < len(args):
        arg = args[i]
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 1
        elif arg == "--content-lang" and i + 1 < len(args):
            content_lang = args[i + 1]
            i += 1
        elif arg in ("--min-words", "--target-words", "--max-words") and i + 1 < len(args):
            overrides[arg.lstrip("-").replace("-", "_")] = int(args[i + 1])
            i += 1
        elif not arg.startswith("-"):
            positional_args.append(arg)
        i += 1
    search_query = positional_args[0] if positional_args else None
    return config_path, content_lang, search_query, overrides


def load_fetcher_config(config_path=None, content_lang=None):
    """
    Load fetcher configuration from config.json.

    Parameters
    ----------
    config_path : str or None
        Path to config.json. Defaults to project root.
    content_lang : str or None
        Language code (e.g. "de", "en"). If given, resolves Kiwix server
        from kiwix_servers[content_lang]. Falls back to legacy top-level
        'kiwix' block if not found.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config.json")

    settings = {
        "base_url": DEFAULT_BASE_URL,
        "zim_name": DEFAULT_ZIM_NAME,
        "article_filter": DEFAULT_ARTICLE_FILTER.copy(),
    }

    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Resolve Kiwix server: prefer kiwix_servers[content_lang], fall back to legacy 'kiwix'
        wiki_cfg = {}
        if content_lang and "kiwix_servers" in config:
            wiki_cfg = config["kiwix_servers"].get(content_lang, {})

        # Fall back to legacy top-level kiwix block
        if not wiki_cfg:
            wiki_cfg = config.get("kiwix", {})

        settings["base_url"] = wiki_cfg.get("base_url", settings["base_url"])
        settings["zim_name"] = wiki_cfg.get("zim_name", settings["zim_name"])

        af = config.get("article_filter", {})
        for key in settings["article_filter"]:
            settings["article_filter"][key] = af.get(key, settings["article_filter"][key])

    return settings


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    config_path, content_lang, search_query, overrides = parse_cli_args(sys.argv[1:])
    settings = load_fetcher_config(config_path, content_lang=content_lang)

    base_url = settings["base_url"]
    zim_name = settings["zim_name"]
    af = settings["article_filter"]
    # CLI overrides take precedence
    if overrides:
        af.update(overrides)
    min_words = af["min_words"]
    target_words = af["target_words"]
    max_words = af["max_words"]

    with KiwixClient(base_url=base_url, zim_name=zim_name) as client:
        if search_query:
            # Search mode — try up to 15 results until we find a prose-rich article
            titles = client.search(search_query, count=50)
            if not titles:
                print(json.dumps({"error": f"No results for '{search_query}'"}))
                sys.exit(1)

            title = None
            text = ""
            random.shuffle(titles)
            for attempt, chosen in enumerate(titles):
                # Quick title filter
                if any(skip in chosen for skip in KiwixClient.SKIP_PATTERNS):
                    continue
                resp = client.get_article(chosen)

                # Skip articles that are mostly tables/infoboxes with no prose
                if not _has_enough_prose(resp.text):
                    continue

                text = extract_wiki_text(resp.text)
                disambig_patterns = [
                    "may refer to",              # EN
                    "kann sich beziehen auf",     # DE
                    "puede referirse a",          # ES
                    "può riferirsi a",            # IT
                    "lehet több jelentése is",    # HU
                    "peut faire référence à",     # FR
                    "může znamenat",              # CS
                    "viz rozcestník",             # CS
                ]
                if any(pat in text[:500] for pat in disambig_patterns):
                    continue

                # Skip articles that are mostly table/infobox data (short lines)
                if _is_table_heavy(text):
                    continue

                word_count = len(text.split())
                if word_count < min_words:
                    continue

                title = _get_title_from_html(resp.text) or chosen
                break

            if not title:
                print(json.dumps({"error": f"No suitable prose article found for '{search_query}' after trying {len(titles)} results."}))
                sys.exit(1)

            # Smart-truncate if needed
            word_count = len(text.split())
            if word_count > max_words:
                truncated = smart_truncate(text, target_words, max_words, min_words)
                if truncated:
                    text = truncated
        else:
            # Random mode
            title, text = client.get_random_article(
                min_words=min_words,
                target_words=target_words,
                max_words=max_words,
            )

        # Output as structured payload
        result = {
            "title": title,
            "text": text,
            "source": f"Kiwix ({zim_name})",
            "word_count": len(text.split()),
        }
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
