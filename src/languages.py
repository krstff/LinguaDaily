#!/usr/bin/env python3
"""
Single source of truth for ALL language-dependent strings in LinguaDaily.

Adding a new language = editing ONLY this file:

  1. LANGUAGE_NAMES
     Add the ISO 639-1 code → display name.  Used everywhere a language
     name is shown or injected into LLM prompts (Telegram, web UI, …).

  2. LESSON_ACK_TEXT / LESSON_ACK_DONE_TEXT
     Post-lesson messages shown in the learner's LEARNING language.
     Copy an existing table entry and translate.  (Optional — unknown
     codes fall back to "en".)

  3. CONVO_STRINGS
     Conversation-practice UI strings shown in the learner's NATIVE
     language.  Copy the "en" table and translate.  (Optional — unknown
     codes fall back to "en".)

  4. PERIOD_DECIMAL_LANGUAGES
     Add the code ONLY if the language writes numbers with "." as the
     decimal separator.  All other supported languages automatically
     get comma decimals (European convention) — see the table below.

  5. NEWS_FEED_CATALOGUE (optional)
     Add a {topic: [feed URLs]} table for the language, with topics in
     that language (e.g. "Tecnología" for es).  Without one, English
     feeds are used.

  6. LANG_HEADINGS (optional)
     Add the folded heading pattern(s) used to scope multilingual
     Wiktionary pages to the language (see the table's comments).

Nothing else in the codebase needs to change: LLM prompts are
language-agnostic templates that receive the display name via
{language_name} formatting.

To get a translation, paste JUST this file into an LLM.
"""

# ── Language codes → display names ────────────────────────────────
#
# Used to resolve `learning_language_name` automatically from the
# `learning_language` code.  The user only needs to set the ISO code
# ("de", "it", …) in config.json; the human-readable name is computed
# here so it can never go out of sync.
#
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "fr": "French",
    "pt": "Portuguese",
    "ru": "Russian",
    "hu": "Hungarian",
    "cs": "Czech",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "tr": "Turkish",
}

# All language codes the app knows about.
SUPPORTED_LANGUAGE_CODES = frozenset(LANGUAGE_NAMES)


def resolve_language_name(lang_code: str) -> str:
    """Return a human-readable language name for an ISO code.

    Falls back to the code itself if the mapping is unknown.
    """
    return LANGUAGE_NAMES.get(lang_code, lang_code)


# ── Post-lesson acknowledgement texts ─────────────────────────────
#
# Shown per learner's LEARNING language (the language of the lesson,
# not their native one).  Unknown language codes fall back to "en".
#
LESSON_ACK_TEXT: dict[str, str] = {
    "en": "📖 You've reached the end of your lesson.\nClick below when you're done reading 👇",
    "de": "📖 Du bist am Ende deiner Lektion.\nKlicke unten, wenn du fertig bist 👇",
    "cs": "📖 Jsi u konce lekce.\nKlikni dole, když budeš hotový 👇",
    "hu": "📖 Elérkeztél a lecke végére.\nKattints lent, ha kész vagy 👇",
    "it": "📖 Sei arrivato alla fine della lezione.\nClicca qui sotto quando hai finito 👇",
    "es": "📖 Has llegado al final de la lección.\nHaz clic abajo cuando termines 👇",
    "fr": "📖 Tu es à la fin de ta leçon.\nClique ci-dessous quand tu as fini 👇",
}
# Short "good job" confirmation shown (with the streak effect) after the
# "Finished" click — also in the learning language.
LESSON_ACK_DONE_TEXT: dict[str, str] = {
    "en": "🎉 Good job!",
    "de": "🎉 Gut gemacht!",
    "cs": "🎉 Dobře!",
    "hu": "🎉 Jól csináltad!",
    "it": "🎉 Ben fatto!",
    "es": "🎉 ¡Bien hecho!",
    "fr": "🎉 Bien joué !",
}


def lesson_ack_text(lang_code: str) -> str:
    """Post-lesson acknowledgement text for the learning language."""
    return LESSON_ACK_TEXT.get(lang_code, LESSON_ACK_TEXT["en"])


def lesson_ack_done_text(lang_code: str) -> str:
    """Post-"Finished" confirmation text for the learning language."""
    return LESSON_ACK_DONE_TEXT.get(lang_code, LESSON_ACK_DONE_TEXT["en"])


# ── Conversation practice UI strings ──────────────────────────────
#
# Keyed by the profile's `native_language` — the language the user
# understands.  English is the fallback for any code without a table.
#
CONVO_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "tts_disabled": (
            "⚠️ TTS is disabled for this profile — "
            "conversation practice needs audio."),
        "no_llm": (
            "⚠️ No LLM model configured — set <code>llm.default_model</code> "
            "(optionally a small fast one via <code>llm.task_models.convo</code>)."),
        "no_lesson": "⚠️ No lesson yet — run /another first, then try /convo.",
        "preparing": "🎧 Preparing your conversation…",
        "llm_unavailable": (
            "⚠️ Could not start the conversation — LLM unavailable. Try again."),
        "no_session": "No active conversation — start one with /convo.",
        "busy": "⏳ Still working on the last answer — one moment.",
        "transcribing": "🎧 Transcribing…",
        "processing_error": "⚠️ Something went wrong — try recording again.",
        "download_failed": "⚠️ Could not download your voice message.",
        "ffmpeg_missing": (
            "⚠️ Could not process the audio — is ffmpeg installed on the daemon?"),
        "stt_failed": "⚠️ Transcription failed — is the STT endpoint up? Try again.",
        "empty_transcript": "🤔 I couldn't catch anything — try again?",
        "tutor_unavailable": "⚠️ The tutor is unavailable — try recording again.",
        "tts_failed": "(⚠️ TTS failed — text only)",
        "turn_header": "✅ <b>Turn {turn}/{total}</b> · score: <b>{score}/100</b>",
        "summary_title": "🏁 <b>Conversation complete!</b>",
        "summary_topic": "Topic",
        "summary_avg": "Average score",
        "summary_note": "(Scores are based on the words you used.)",
        "caption_header": "🗣 <b>Conversation {turn}/{total}</b> · topic",
        "caption_prompt": (
            "🎤 Record your answer as a voice message — or just type it."),
        "partner": "🗣 Partner",
        "you": "🎤 You",
    },
    "cs": {
        "tts_disabled": (
            "⚠️ TTS je pro tento profil vypnutý — "
            "cvičení konverzace potřebuje zvuk."),
        "no_llm": (
            "⚠️ Není nakonfigurovaný žádný LLM model — nastavte "
            "<code>llm.default_model</code> (volitelně malý rychlý přes "
            "<code>llm.task_models.convo</code>)."),
        "no_lesson": "⚠️ Zatím žádná lekce — nejdřív spusťte /another, pak zkuste /convo.",
        "preparing": "🎧 Připravuji vaši konverzaci…",
        "llm_unavailable": (
            "⚠️ Nedaří se nastartovat konverzaci — LLM není dostupné. Zkuste to znovu."),
        "no_session": "Žádná aktivní konverzace — spusťte ji příkazem /convo.",
        "busy": "⏳ Stále zpracovávám poslední odpověď — chvilku.",
        "transcribing": "🎧 Přepisuji…",
        "processing_error": "⚠️ Něco se pokazilo — zkuste to nahrát znovu.",
        "download_failed": "⚠️ Nedaří se stáhnout vaši hlasovou zprávu.",
        "ffmpeg_missing": "⚠️ Zvuk se nepodařilo zpracovat — je na démonu nainstalovaný ffmpeg?",
        "stt_failed": "⚠️ Přepis selhal — je STT endpoint dostupný? Zkuste to znovu.",
        "empty_transcript": "🤔 Nic jsem nepochopil — zkuste to znovu?",
        "tutor_unavailable": "⚠️ Tutořitel není dostupný — zkuste to nahrát znovu.",
        "tts_failed": "(⚠️ TTS selhalo — jen text)",
        "turn_header": "✅ <b>Kolo {turn}/{total}</b> · skóre: <b>{score}/100</b>",
        "summary_title": "🏁 <b>Konverzace dokončena!</b>",
        "summary_topic": "Téma",
        "summary_avg": "Průměrné skóre",
        "summary_note": "(Skóre je založeno na použitých slovech.)",
        "caption_header": "🗣 <b>Konverzace {turn}/{total}</b> · téma",
        "caption_prompt": (
            "🎤 Nahrajte svou odpověď jako hlasovou zprávu — nebo ji prostě napište."),
        "partner": "🗣 Partner",
        "you": "🎤 Vy",
    },
}


def convo_text(lang_code: str, key: str, **fmt) -> str:
    """Localized conversation-UI string for the profile's native language.

    Falls back to English for unknown codes or missing keys.
    """
    table = CONVO_STRINGS.get(lang_code) or CONVO_STRINGS["en"]
    template = table.get(key) or CONVO_STRINGS["en"][key]
    return template.format(**fmt) if fmt else template


# ── Language number formatting ────────────────────────────────────
#
# Languages that write numbers with "." as the decimal separator.  Every
# OTHER supported language uses the European comma convention, so the
# comma set is derived from LANGUAGE_NAMES instead of being maintained
# separately (adding a language above gets the right behaviour for free).
PERIOD_DECIMAL_LANGUAGES = frozenset({"en"})

# Used by TTS number-to-words conversion.
COMMA_DECIMAL_LANGUAGES = frozenset(LANGUAGE_NAMES) - PERIOD_DECIMAL_LANGUAGES


# ── News feeds (fallback catalogue) ───────────────────────────────
#
# Per-language topic → RSS feed catalogue used by news_fetcher.py when
# config.json has no sources.news.feeds.  Topic names are localized per
# language.  Unknown languages fall back to the "en" table.
NEWS_FEED_CATALOGUE: dict[str, dict[str, list[str]]] = {
    "en": {
        "Technology": [
            "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "https://www.theregister.com/security/headlines.atom",
        ],
        "Science": ["https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"],
        "Mathematics": ["https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"],
        "History": ["https://feeds.bbci.co.uk/news/world/rss.xml"],
        "Art": ["https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"],
        "Music": ["https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"],
        "Philosophy": ["https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"],
        "Literature": ["https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"],
        "Architecture": ["https://feeds.bbci.co.uk/news/world/rss.xml"],
        "Biology": ["https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"],
        "Physics": ["https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"],
        "Chemistry": ["https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"],
        "Geography": [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://www.nationalgeographic.com/news/",
        ],
        "Astronomy": [
            "https://www.nasa.gov/rss/dyn/breaking_news.rss",
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        ],
        "Psychology": ["https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"],
        "Economics": ["https://feeds.bbci.co.uk/news/business/rss.xml"],
        "Politics": [
            "https://feeds.bbci.co.uk/news/politics/rss.xml",
            "https://feeds.bbci.co.uk/news/world/rss.xml",
        ],
        "Medicine": ["https://feeds.bbci.co.uk/news/health/rss.xml"],
        "Culture": ["https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"],
    },
}
# General fallback feeds when no topic-specific feed is found.
NEWS_DEFAULT_FEEDS = ["https://feeds.bbci.co.uk/news/rss.xml"]


# ── Wiktionary language scoping ───────────────────────────────────
#
# Folded language-name patterns per TARGET language (used by
# wiktionary_client.py).  A page section is "for us" when its h2
# heading (text or extiw link title) contains one of the folded
# patterns.  Patterns are stored folded (diacritics stripped,
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
