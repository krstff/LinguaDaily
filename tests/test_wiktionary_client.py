#!/usr/bin/env python3
"""
Tests for src/wiktionary_client.py — backend resolution, HTML extraction,
title-fallback lookup, and the multi-term reference block.

Extraction tests run against trimmed fixtures that mirror the structure of
real de.wiktionary.org pages (verb "gehen", noun "Haus", disambiguation stub).
Lookup tests mock the Kiwix / online clients — no network access needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

FIXTURES = Path(__file__).parent / "fixtures"
VERB_HTML = (FIXTURES / "wiktionary_verb.html").read_text(encoding="utf-8")
NOUN_HTML = (FIXTURES / "wiktionary_noun.html").read_text(encoding="utf-8")
STUB_HTML = (FIXTURES / "wiktionary_stub.html").read_text(encoding="utf-8")


def _http_404():
    err = requests.HTTPError("404 Client Error")
    err.response = MagicMock(status_code=404)
    return err


# ── Backend resolution ────────────────────────────────────────────────

class TestResolveBackend:
    def test_auto_with_kiwix_server(self):
        from src.wiktionary_client import resolve_wiktionary_backend
        config = {"wiktionary": {"backend": "auto", "servers": {
            "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
        backend, params = resolve_wiktionary_backend(config, "de")
        assert backend == "kiwix"
        assert params["base_url"] == "http://k:8080"
        assert params["zim_name"] == "de.wiktionary"

    def test_auto_without_server_falls_back_online(self):
        from src.wiktionary_client import resolve_wiktionary_backend
        config = {"wiktionary": {"backend": "auto", "servers": {}}}
        backend, params = resolve_wiktionary_backend(config, "de")
        assert backend == "online"
        assert params["language"] == "de"

    def test_auto_server_without_zim_name_counts_as_missing(self):
        from src.wiktionary_client import resolve_wiktionary_backend
        config = {"wiktionary": {"backend": "auto", "servers": {
            "de": {"base_url": "http://k:8080"}}}}
        backend, _ = resolve_wiktionary_backend(config, "de")
        assert backend == "online"

    def test_explicit_kiwix_without_server_falls_back_online(self):
        from src.wiktionary_client import resolve_wiktionary_backend
        config = {"wiktionary": {"backend": "kiwix", "servers": {}}}
        backend, _ = resolve_wiktionary_backend(config, "de")
        assert backend == "online"

    def test_explicit_online_ignores_server(self):
        from src.wiktionary_client import resolve_wiktionary_backend
        config = {"wiktionary": {"backend": "online", "servers": {
            "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
        backend, _ = resolve_wiktionary_backend(config, "de")
        assert backend == "online"

    def test_empty_and_none_config(self):
        from src.wiktionary_client import resolve_wiktionary_backend
        for config in ({}, None):
            backend, _ = resolve_wiktionary_backend(config, "cs")
            assert backend == "online"

    def test_language_case_insensitive(self):
        from src.wiktionary_client import resolve_wiktionary_backend
        config = {"wiktionary": {"backend": "auto", "servers": {
            "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
        backend, _ = resolve_wiktionary_backend(config, "DE")
        assert backend == "kiwix"


# ── Title candidates ──────────────────────────────────────────────────

class TestCandidateTitles:
    def test_basic_chain(self):
        from src.wiktionary_client import _candidate_titles
        # case-sensitive: capitalized variant is still tried (German nouns)
        assert _candidate_titles("gehen", "gehen") == ["gehen", "Gehen"]

    def test_lemma_and_capitalization(self):
        from src.wiktionary_client import _candidate_titles
        assert _candidate_titles("haus", "Haus") == ["haus", "Haus"]

    def test_inflected_form_with_lemma(self):
        from src.wiktionary_client import _candidate_titles
        assert _candidate_titles("geht", "gehen") == ["geht", "Geht", "gehen", "Gehen"]

    def test_empty(self):
        from src.wiktionary_client import _candidate_titles
        assert _candidate_titles("", "") == []
        assert _candidate_titles(None, None) == []


# ── HTML extraction ───────────────────────────────────────────────────

class TestExtract:
    def test_verb_page(self):
        from src.wiktionary_client import extract_wiktionary_entry
        out = extract_wiktionary_entry(VERB_HTML)
        assert out is not None
        # Part of speech
        assert "Verb, unregelmäßig, intransitiv" in out
        # English translations (highest value)
        assert "English: walk" in out
        # Core definitions kept
        assert "sich fortbewegen" in out
        # Conjugation table kept (past tense visible for conjugation Qs)
        assert "gehe" in out
        assert "ging" in out
        assert "Inflection:" in out
        # Noise removed
        assert "→" not in out
        assert "Bairisch" not in out          # dialect table
        assert "Überarbeitung" not in out     # maintenance box
        assert "Ripuarisch" not in out

    def test_noun_page(self):
        from src.wiktionary_client import extract_wiktionary_entry
        out = extract_wiktionary_entry(NOUN_HTML)
        assert out is not None
        assert "Substantiv, n" in out
        assert "English: house" in out
        assert "Häusern" in out  # Dativ Plural — case forms present

    def test_stub_page_returns_none(self):
        from src.wiktionary_client import extract_wiktionary_entry
        assert extract_wiktionary_entry(STUB_HTML) is None

    def test_empty_html(self):
        from src.wiktionary_client import extract_wiktionary_entry
        assert extract_wiktionary_entry("") is None

    def test_output_capped(self):
        from src.wiktionary_client import extract_wiktionary_entry
        out = extract_wiktionary_entry(VERB_HTML, max_chars=100)
        assert out is None or len(out) <= 100

    def test_no_raw_html_leaks(self):
        from src.wiktionary_client import extract_wiktionary_entry
        out = extract_wiktionary_entry(VERB_HTML)
        assert "<" not in out and ">" not in out


# ── Fetch with title fallbacks ────────────────────────────────────────

class TestFetchEntry:
    def test_kiwix_404_then_success(self):
        """First title 404 (inflected form), lemma title resolves."""
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = [_http_404(), VERB_HTML]
        with patch.object(wc, "KiwixClient", return_value=fake):
            config = {"wiktionary": {"backend": "auto", "servers": {
                "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
            out = wc.fetch_wiktionary_entry("geht", "gehen", "de", config)
        assert out is not None
        assert "sich fortbewegen" in out
        # Two titles attempted: "geht" → 404, "Geht" → VERB_HTML
        assert fake.get_article.call_count == 2

    def test_online_backend(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_entry_html.return_value = NOUN_HTML
        with patch.object(wc, "WiktionaryClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("Haus", "", "de", {})
        assert out is not None
        assert "English: house" in out

    def test_all_404_returns_none(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = _http_404()
        with patch.object(wc, "KiwixClient", return_value=fake):
            config = {"wiktionary": {"backend": "auto", "servers": {
                "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
            out = wc.fetch_wiktionary_entry("nosit", "", "de", config)
        assert out is None

    def test_connection_error_returns_none(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = requests.ConnectionError("refused")
        with patch.object(wc, "KiwixClient", return_value=fake):
            config = {"wiktionary": {"backend": "auto", "servers": {
                "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
            out = wc.fetch_wiktionary_entry("gehen", "", "de", config)
        assert out is None

    def test_wikifetch_error_treated_as_missing(self):
        from src import wiktionary_client as wc
        # Use the exception class from the same module the code catches
        fake = MagicMock()
        fake.get_entry_html.side_effect = wc.WikiFetchError("page missing")
        with patch.object(wc, "WiktionaryClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("nosit", "", "de", {})
        assert out is None

    def test_empty_term(self):
        from src import wiktionary_client as wc
        assert wc.fetch_wiktionary_entry("", "", "de", {}) is None
        assert wc.fetch_wiktionary_entry(None, "", "de", {}) is None

    def test_inflected_stub_prefers_lemma(self):
        """'geht' page is a Konjugierte-Form stub → fetch lemma 'gehen'."""
        from src import wiktionary_client as wc
        stub = (
            "<div class='mw-parser-output'><h2>geht</h2>"
            "<h3>Konjugierte Form</h3>"
            "<ul><li>3. Person Singular des Verbs gehen</li></ul></div>"
        )
        fake = MagicMock()
        fake.get_article.side_effect = [stub, VERB_HTML]
        with patch.object(wc, "KiwixClient", return_value=fake):
            config = {"wiktionary": {"backend": "auto", "servers": {
                "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
            out = wc.fetch_wiktionary_entry("geht", "gehen", "de", config)
        assert out is not None
        # Full verb entry (conjugation table), not the stub
        assert "gehe" in out and "ging" in out
        assert fake.get_article.call_args_list[-1].args[0] == "gehen"

    def test_stub_without_lemma_kept(self):
        from src import wiktionary_client as wc
        stub = (
            "<div class='mw-parser-output'><h2>geht</h2>"
            "<h3>Konjugierte Form</h3>"
            "<ul><li>3. Person Singular des Verbs gehen</li></ul></div>"
        )
        fake = MagicMock()
        fake.get_article.side_effect = [stub]
        with patch.object(wc, "KiwixClient", return_value=fake):
            config = {"wiktionary": {"backend": "auto", "servers": {
                "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
            out = wc.fetch_wiktionary_entry("geht", "", "de", config)
        assert out is not None
        assert "Konjugierte Form" in out

    def test_client_closed(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = _http_404()
        with patch.object(wc, "KiwixClient", return_value=fake):
            config = {"wiktionary": {"backend": "auto", "servers": {
                "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
            wc.fetch_wiktionary_entry("nosit", "", "de", config)
        fake.close.assert_called_once()


# ── Multi-term reference block ────────────────────────────────────────

class TestDictionaryReference:
    def test_multiple_terms(self):
        from src import wiktionary_client as wc
        with patch.object(wc, "fetch_wiktionary_entry", side_effect=[
                "=== verb entry ===", "=== noun entry ==="]) as m:
            out = wc.get_dictionary_reference(["gehen", "Haus"], "", "de", {})
        assert out is not None
        assert "=== gehen ===" in out
        assert "=== Haus ===" in out
        assert m.call_count == 2

    def test_dedup_and_cap(self):
        from src import wiktionary_client as wc
        with patch.object(wc, "fetch_wiktionary_entry", return_value="entry") as m:
            out = wc.get_dictionary_reference(
                ["a", "a", "b", "c", "d"], "", "de", {})
        assert m.call_count == 3  # capped at MAX_TERMS
        assert "=== a ===" in out and "=== d ===" not in out

    def test_no_terms(self):
        from src import wiktionary_client as wc
        assert wc.get_dictionary_reference([], "", "de", {}) is None
        assert wc.get_dictionary_reference(None, "", "de", {}) is None

    def test_no_entry_found(self):
        from src import wiktionary_client as wc
        with patch.object(wc, "fetch_wiktionary_entry", return_value=None):
            assert wc.get_dictionary_reference(["nosit"], "", "de", {}) is None

    def test_junk_terms_filtered(self):
        from src import wiktionary_client as wc
        with patch.object(wc, "fetch_wiktionary_entry", return_value="entry") as m:
            out = wc.get_dictionary_reference(["  ", "", None, "Haus"], "", "de", {})
        assert out is not None
        assert m.call_count == 1
