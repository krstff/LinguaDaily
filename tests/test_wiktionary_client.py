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
DE_STUB_VERB_HTML = (FIXTURES / "wiktionary_de_stub_verb.html").read_text(encoding="utf-8")
DE_STUB_NOUN_HTML = (FIXTURES / "wiktionary_de_stub_noun.html").read_text(encoding="utf-8")
EN_FORMOF_HTML = (FIXTURES / "wiktionary_en_formof.html").read_text(encoding="utf-8")
EN_FULL_HTML = (FIXTURES / "wiktionary_en_full.html").read_text(encoding="utf-8")
HU_PARTIAL_HTML = (FIXTURES / "wiktionary_hu_partial.html").read_text(encoding="utf-8")
HU_FULL_HTML = (FIXTURES / "wiktionary_hu_full.html").read_text(encoding="utf-8")
HU_FOREIGN_HTML = (FIXTURES / "wiktionary_hu_foreign_only.html").read_text(encoding="utf-8")
SINGLE_LANG_HTML = (FIXTURES / "wiktionary_single_lang.html").read_text(encoding="utf-8")


def _kiwix_config(lang="de"):
    return {"wiktionary": {"backend": "auto", "servers": {
        lang: {"base_url": "http://k:8080", "zim_name": f"{lang}.wiktionary"}}}}


def _scoped_signals(html, language):
    """Step 0 (scoping) + the signal pass, as the resolution code uses it."""
    from src import wiktionary_client as wc
    top, sections = wc._split_sections(html)
    scoped = wc._target_sections(top, sections, language)
    assert scoped is not None
    return wc.analyze_wiktionary_entry("\n".join(scoped))


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

    def test_rowspan_mood_inherited_on_continuation_rows(self):
        # "du | gehst" (2 cells) must inherit the mood from the row above
        from src.wiktionary_client import extract_wiktionary_entry
        html = (
            "<div class='mw-parser-output'><h2>x (Deutsch)</h2><h3>Verb</h3>"
            "<table class='inflection-table'>"
            "<tr><th>Person</th><th>Wortform</th></tr>"
            "<tr><td>Präsens</td><td>ich</td><td>gehe</td></tr>"
            "<tr><td>du</td><td>gehst</td></tr>"
            "<tr><td>Präteritum</td><td>ich</td><td>ging</td></tr>"
            "<tr><td>du</td><td>gingst</td></tr>"
            "</table></div>"
        )
        out = extract_wiktionary_entry(html)
        assert "Präsens | du | gehst" in out
        assert "Präteritum | du | gingst" in out

    def test_all_conjugation_tables_kept_including_past_tense(self):
        # cs verbs use class="konjugace"; the past-tense table is the THIRD
        # one and must not be cut (tutors answer past-tense questions)
        from src.wiktionary_client import extract_wiktionary_entry
        def cs_table(first_row):
            return (
                "<table class='konjugace verbum'>"
                "<tr><td>osoba</td><td>1.</td><td>2.</td><td>3.</td></tr>"
                f"<tr><td>{first_row}</td><td>a</td><td>b</td><td>c</td></tr>"
                "<tr><td>x</td><td>y</td><td>z</td><td>w</td></tr>"
                "</table>"
            )
        html = (
            "<div class='mw-parser-output'><h2>pracovat (čeština)</h2>"
            "<h3>Verbum</h3>"
            + cs_table("přítomný čas")
            + cs_table("imperativ")
            + cs_table("minulý čas: pracoval, pracovala, pracovali")
            + cs_table("pracující")
            + "</div>"
        )
        out = extract_wiktionary_entry(html)
        assert "přítomný čas" in out
        assert "imperativ" in out
        assert "pracovali" in out   # past tense (3rd table)
        assert "pracující" in out   # participle (4th table)


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
        """'geht' page is a pointer stub → one hop to the lemma 'gehen'."""
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = [DE_STUB_VERB_HTML, VERB_HTML]
        with patch.object(wc, "KiwixClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("geht", "", "de", _kiwix_config())
        assert out is not None
        # Form note + full verb entry (conjugation table), not the stub
        assert out.startswith("Form: geht — see gehen")
        assert "gehe" in out and "ging" in out
        assert fake.get_article.call_count == 2
        assert fake.get_article.call_args_list[-1].args[0] == "gehen"

    def test_stub_lemma_hint_not_required(self):
        """Tier 2 word links find the lemma without the LLM hint."""
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = [DE_STUB_NOUN_HTML, NOUN_HTML]
        with patch.object(wc, "KiwixClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("Häuser", "", "de", _kiwix_config())
        assert out is not None
        assert out.startswith("Form: Häuser — see Haus")
        assert "English: house" in out

    def test_stub_candidate_missing_kept_with_note(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = [DE_STUB_VERB_HTML, _http_404()]
        with patch.object(wc, "KiwixClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("geht", "", "de", _kiwix_config())
        assert out is not None
        assert "Konjugierte Form" in out
        assert "Note: geht appears to be a form of gehen." in out

    def test_client_closed(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = _http_404()
        with patch.object(wc, "KiwixClient", return_value=fake):
            config = {"wiktionary": {"backend": "auto", "servers": {
                "de": {"base_url": "http://k:8080", "zim_name": "de.wiktionary"}}}}
            wc.fetch_wiktionary_entry("nosit", "", "de", config)
        fake.close.assert_called_once()


# ── Step 0 — language scoping ─────────────────────────────────────────


class TestSectionScoping:
    def test_no_h2_whole_page_is_target(self):
        from src import wiktionary_client as wc
        top, sections = wc._split_sections(SINGLE_LANG_HTML)
        assert sections == []
        scoped = wc._target_sections(top, sections, "de")
        assert scoped == [top]
        assert "Wohngebäude" in scoped[0]

    def test_heading_text_match_includes_top_section(self):
        from src import wiktionary_client as wc
        top, sections = wc._split_sections(DE_STUB_VERB_HTML)
        assert len(sections) == 1
        html = "\n".join(wc._target_sections(top, sections, "de"))
        assert "Konjugierte Form" in html
        assert "Wort des Tages" in html  # top section included when a heading matched

    def test_multi_section_page_without_language_returns_none(self):
        from src import wiktionary_client as wc
        top, sections = wc._split_sections(HU_FOREIGN_HTML)
        # men page: Dán/Feröeri/Korni/Norvég/Üzbég/Svéd/Mandarin — no Magyar
        assert wc._target_sections(top, sections, "hu") is None
        assert wc._target_sections(top, sections, "de") is None

    def test_single_foreign_section_is_unambiguous(self):
        # de.wikt "nosit (Tschechisch)": one section, no German — the page
        # is unambiguously about the word, use it
        from src import wiktionary_client as wc
        html = (
            "<div class='mw-parser-output'>"
            "<h2>nosit (Tschechisch)</h2>"
            "<h3>Verb</h3><ul><li>to carry, to bear</li></ul>"
            "<table class='inflection-table'><tr><th>a</th></tr>"
            "<tr><td>b</td></tr><tr><td>c</td></tr></table>"
            "</div>"
        )
        top, sections = wc._split_sections(html)
        scoped = wc._target_sections(top, sections, "de")
        assert scoped is not None
        assert "to carry, to bear" in "".join(scoped)

    def test_extiw_title_match(self):
        from src import wiktionary_client as wc
        html = (
            "<div><h2><a class='extiw' "
            "href='https://hu.wikipedia.org/wiki/Magyar_nyelv' "
            "title='magyar_nyelv'>Magyar</a></h2><p>tartalom</p></div>"
        )
        top, sections = wc._split_sections(html)
        scoped = wc._target_sections(top, sections, "hu")
        assert scoped is not None
        assert "tartalom" in "".join(scoped)

    def test_diacritic_folding(self):
        from src import wiktionary_client as wc
        html = "<div><h2>Čeština</h2><p>x</p></div>"
        top, sections = wc._split_sections(html)
        assert wc._target_sections(top, sections, "cs") is not None

    def test_en_wiki_uses_exonym_headings(self):
        from src import wiktionary_client as wc
        top, sections = wc._split_sections(EN_FORMOF_HTML)
        html = "\n".join(wc._target_sections(top, sections, "en"))
        assert "form-of-definition" in html  # the Czech section is in scope
        assert "pracuję" not in html          # the Polish section is not

    def test_unknown_language_returns_none(self):
        from src import wiktionary_client as wc
        top, sections = wc._split_sections(HU_FOREIGN_HTML)
        assert wc._target_sections(top, sections, "xx") is None


# ── Step 1 — completeness signals ─────────────────────────────────────


class TestCompletenessSignals:
    def test_de_stub_verb_not_complete(self):
        # pointer <li>s end with the lemma link → not definitions
        sig = _scoped_signals(DE_STUB_VERB_HTML, "de")
        assert not sig["has_inflection_table"]
        assert not sig["has_substantive_definitions"]

    def test_translations_only_not_complete(self):
        # menni page: translations table + empty <li>s → NOT a complete entry
        sig = _scoped_signals(HU_PARTIAL_HTML, "hu")
        assert not sig["has_inflection_table"]
        assert not sig["has_substantive_definitions"]

    def test_inflection_table_complete(self):
        sig = _scoped_signals(HU_FULL_HTML, "hu")
        assert sig["has_inflection_table"]
        assert sig["has_substantive_definitions"]

    def test_substantive_definitions_complete(self):
        sig = _scoped_signals(SINGLE_LANG_HTML, "de")
        assert sig["has_substantive_definitions"]

    def test_ipa_line_not_substantive(self):
        from src import wiktionary_client as wc
        html = ("<div class='mw-parser-output'><h2>x (Deutsch)</h2>"
                "<ul><li>[ˈɡeːt]</li></ul></div>")
        assert not wc.analyze_wiktionary_entry(html)["has_substantive_definitions"]

    def test_labeled_boilerplate_not_substantive(self):
        from src import wiktionary_client as wc
        html = (
            "<div class='mw-parser-output'><h2>x (Deutsch)</h2><ul>"
            "<li>IPA ( key ) : [ˈprat͡sujɛ]</li>"
            "<li>Hyphenation: pra‧cu‧je</li>"
            "</ul></div>"
        )
        assert not wc.analyze_wiktionary_entry(html)["has_substantive_definitions"]

    def test_short_line_not_substantive(self):
        from src import wiktionary_client as wc
        html = ("<div class='mw-parser-output'><h2>x (Deutsch)</h2>"
                "<ul><li>to go</li></ul></div>")
        assert not wc.analyze_wiktionary_entry(html)["has_substantive_definitions"]

    def test_form_of_boilerplate_not_substantive(self):
        # en pointer pages describe forms ("third-person singular present
        # indicative") — that boilerplate must not count as a definition
        sig = _scoped_signals(EN_FORMOF_HTML, "en")
        assert not sig["has_substantive_definitions"]
        assert not sig["has_inflection_table"]

    def test_translation_li_not_substantive(self):
        from src import wiktionary_client as wc
        html = (
            "<div class='mw-parser-output'><h2>x (Deutsch)</h2>"
            "<table class='translations'><tr><td><ul>"
            "<li>german: <a class='extiw' href='https://de.wiktionary.org/wiki/Y' "
            "title='w:Y'>Y</a> — a perfectly long translation value</li>"
            "</ul></td></tr></table></div>"
        )
        assert not wc.analyze_wiktionary_entry(html)["has_substantive_definitions"]


# ── Step 2 — pointer resolution ───────────────────────────────────────


class TestPointerResolution:
    def test_tier1_form_of_class(self):
        sig = _scoped_signals(EN_FORMOF_HTML, "en")
        assert sig["form_of_lemma"] == "pracovat"

    def test_tier1_foreign_section_ignored(self):
        # form-of classes exist ONLY in the Mandarin section of the men page
        from src import wiktionary_client as wc
        top, sections = wc._split_sections(HU_FOREIGN_HTML)
        mandarin = next(s for s in sections if s["heading_text"].startswith("Mandarin"))
        assert wc.analyze_wiktionary_entry(mandarin["html"])["form_of_lemma"] == "mėn"
        # …but Hungarian scoping finds no section at all
        assert wc._target_sections(top, sections, "hu") is None

    def test_tier2_unique_candidate(self):
        from src import wiktionary_client as wc
        sig = _scoped_signals(HU_PARTIAL_HTML, "hu")
        assert wc._pick_lemma_candidate(sig, "menni") == "megy"

    def test_tier2_majority_beats_junk(self):
        from src import wiktionary_client as wc
        # de stub: gehen ×3 vs hegt (Ähnliche Wörter) + Zeitraum (WOTD junk) ×1
        sig = _scoped_signals(DE_STUB_VERB_HTML, "de")
        assert "hegt" in sig["word_link_counts"]
        assert "Zeitraum" in sig["word_link_counts"]
        assert wc._pick_lemma_candidate(sig, "geht") == "gehen"

    def test_tier2_translations_links_excluded(self):
        sig = _scoped_signals(HU_PARTIAL_HTML, "hu")
        assert "go" not in sig["word_link_counts"]
        assert "to" not in sig["word_link_counts"]

    def test_tier2_excludes_redlinks_and_namespaces(self):
        from src import wiktionary_client as wc
        html = (
            "<div class='mw-parser-output'><h2>x (Deutsch)</h2><ul>"
            "<li><a class='new' href='/wiki/DoesNotExist' title='DoesNotExist'>DoesNotExist</a></li>"
            "<li><a href='/wiki/Appendix:Glossary' title='Appendix:Glossary'>Appendix</a></li>"
            "<li><a href='/wiki/File:X.ogg' title='File:X.ogg'>sound</a></li>"
            "<li><a href='/wiki/lemma' title='lemma'>lemma</a></li>"
            "</ul></div>"
        )
        sig = wc.analyze_wiktionary_entry(html)
        assert wc._pick_lemma_candidate(sig, "x") == "lemma"

    def test_tier2_self_link_excluded(self):
        from src import wiktionary_client as wc
        html = (
            "<div class='mw-parser-output'><h2>geht (Deutsch)</h2><p>"
            "<a href='/wiki/geht' title='geht'>geht</a> · "
            "<a href='/wiki/gehen' title='gehen'>gehen</a></p></div>"
        )
        sig = wc.analyze_wiktionary_entry(html)
        assert wc._pick_lemma_candidate(sig, "geht") == "gehen"

    def test_tier2_no_candidate_when_tied(self):
        from src import wiktionary_client as wc
        html = (
            "<div class='mw-parser-output'><h2>x (Deutsch)</h2><ul>"
            "<li><a href='/wiki/candidateA' title='candidateA'>A</a></li>"
            "<li><a href='/wiki/candidateB' title='candidateB'>B</a></li>"
            "</ul></div>"
        )
        sig = wc.analyze_wiktionary_entry(html)
        assert wc._pick_lemma_candidate(sig, "x") is None

    def test_tier2_lemma_hint_is_tiebreaker(self):
        from src import wiktionary_client as wc
        html = (
            "<div class='mw-parser-output'><h2>x (Deutsch)</h2><ul>"
            "<li><a href='/wiki/candidateA' title='candidateA'>A</a></li>"
            "<li><a href='/wiki/candidateB' title='candidateB'>B</a></li>"
            "</ul></div>"
        )
        sig = wc.analyze_wiktionary_entry(html)
        assert wc._pick_lemma_candidate(sig, "x", "candidateB") == "candidateB"


# ── Steps 0–3 — resolution orchestration ──────────────────────────────


class TestResolveEntry:
    def test_complete_page_no_hop(self):
        from src import wiktionary_client as wc
        fetch = MagicMock()
        out = wc._resolve_entry(NOUN_HTML, "Haus", "", "de", fetch)
        fetch.assert_not_called()
        assert out is not None
        assert "English: house" in out

    def test_stub_hops_to_lemma_with_form_note(self):
        from src import wiktionary_client as wc
        fetch = MagicMock(side_effect=[VERB_HTML])
        out = wc._resolve_entry(DE_STUB_VERB_HTML, "geht", "", "de", fetch)
        fetch.assert_called_once_with("gehen")
        assert out is not None
        assert out.startswith("Form: geht — see gehen")
        assert "sich fortbewegen" in out
        assert "Inflection:" in out

    def test_single_lang_page_complete(self):
        from src import wiktionary_client as wc
        out = wc._resolve_entry(SINGLE_LANG_HTML, "Haus", "", "de", MagicMock())
        assert out is not None
        assert "Wohngebäude" in out
        assert "Inflection:" in out

    def test_foreign_only_page_returns_none(self):
        from src import wiktionary_client as wc
        assert wc._resolve_entry(HU_FOREIGN_HTML, "men", "", "hu", MagicMock()) is None

    def test_candidate_missing_keeps_stub_with_note(self):
        from src import wiktionary_client as wc
        fetch = MagicMock(side_effect=[requests.HTTPError("404")])
        out = wc._resolve_entry(DE_STUB_VERB_HTML, "geht", "", "de", fetch)
        assert out is not None
        assert "Konjugierte Form" in out
        assert "Note: geht appears to be a form of gehen." in out

    def test_candidate_wrong_language_keeps_original(self):
        from src import wiktionary_client as wc
        fetch = MagicMock(side_effect=[HU_FOREIGN_HTML])  # no German section
        out = wc._resolve_entry(DE_STUB_VERB_HTML, "geht", "", "de", fetch)
        assert out is not None
        assert "Konjugierte Form" in out
        assert "Note: geht appears to be a form of gehen." in out

    def test_candidate_incomplete_no_further_hop(self):
        # candidate page is itself a pointer → stop after ONE hop
        from src import wiktionary_client as wc
        fetch = MagicMock(side_effect=[HU_PARTIAL_HTML])
        out = wc._resolve_entry(HU_PARTIAL_HTML, "menni", "", "hu", fetch)
        assert fetch.call_count == 1
        assert out == "Form: menni — see megy"

    def test_connection_error_mid_hop_keeps_original(self):
        from src import wiktionary_client as wc
        fetch = MagicMock(side_effect=requests.ConnectionError("refused"))
        out = wc._resolve_entry(DE_STUB_VERB_HTML, "geht", "", "de", fetch)
        assert out is not None
        assert "Konjugierte Form" in out

    def test_fetch_none_disables_hop(self):
        from src import wiktionary_client as wc
        out = wc._resolve_entry(DE_STUB_VERB_HTML, "geht", "", "de", None)
        assert out is not None
        assert "Konjugierte Form" in out
        assert "Note: geht appears to be a form of gehen." in out

    def test_kiwix_hu_partial_hops_to_full(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = [HU_PARTIAL_HTML, HU_FULL_HTML]
        with patch.object(wc, "KiwixClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("menni", "", "hu", _kiwix_config("hu"))
        assert out is not None
        assert out.startswith("Form: menni — see megy")
        assert "Helyet változtat" in out

    def test_online_en_formof_hops_to_pracovat(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_entry_html.side_effect = [EN_FORMOF_HTML, EN_FULL_HTML]
        with patch.object(wc, "WiktionaryClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("pracuje", "", "en", {})
        assert out is not None
        assert out.startswith("Form: pracuje — see pracovat")
        assert "to work" in out
        assert fake.get_entry_html.call_args_list[-1].args[0] == "pracovat"

    def test_kiwix_foreign_only_returns_none(self):
        from src import wiktionary_client as wc
        fake = MagicMock()
        fake.get_article.side_effect = [HU_FOREIGN_HTML, _http_404()]
        with patch.object(wc, "KiwixClient", return_value=fake):
            out = wc.fetch_wiktionary_entry("men", "", "hu", _kiwix_config("hu"))
        assert out is None


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
