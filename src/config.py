#!/usr/bin/env python3
"""
Shared path resolution and config loading for LinguaDaily modules.

All source files should import from here instead of duplicating
SCRIPT_DIR / PROJECT_DIR / CONFIG_PATH boilerplate or their own
_load_config() functions.

Usage:
    from src.config import PROJECT_DIR, CONFIG_PATH, DATA_DIR, load_config

    config = load_config()                     # default path
    config = load_config("/custom/config.json") # override
"""

import json
import pathlib

# ── Language code → display name mapping ────────────────────
#
# Used to resolve `learning_language_name` automatically from
# the `learning_language` code.  The user only needs to set
# the ISO code ("de", "it", …) in config.json; the human-readable
# name is computed here so it can never go out of sync.
#
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "fr": "French",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
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


def resolve_language_name(lang_code: str) -> str:
    """Return a human-readable language name for an ISO code.

    Falls back to the code itself if the mapping is unknown.
    """
    return LANGUAGE_NAMES.get(lang_code, lang_code)


# ── Path resolution (computed once at import time) ───────────

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "output"
LOG_FILE    = PROJECT_DIR / "lingua.log"


# ── Centralised defaults ───────────────────────────────────────
#
# Every module-level default lives here so callers never hardcode
# magic values.  Override per-module via config.json or env vars.

# ── Kiwix / Wikipedia ──────────────────────────────────────────
KIWIX_DEFAULT_BASE_URL = "http://192.168.100.52:8080"
KIWIX_DEFAULT_ZIM_NAME = "wikipedia_en_all_maxi_2026-02"
ARTICLE_FILTER_DEFAULTS = {"min_words": 250, "max_words": 600}

# ── News feeds (fallback catalogue) ────────────────────────────
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
NEWS_DEFAULT_FEEDS = ["https://feeds.bbci.co.uk/news/rss.xml"]

# ── LLM ────────────────────────────────────────────────────────
LLM_DEFAULT_BASE_URL   = "http://localhost:8080/v1"
LLM_DEFAULT_MODEL      = "gemma-4-26B-language"
LLM_DEFAULT_TIMEOUT    = 600

# ── TTS ────────────────────────────────────────────────────────
TTS_DEFAULT_MODEL       = "omnivoice"
TTS_DEFAULT_VOICE       = "male"
TTS_DEFAULT_NUM_STEP    = 16
TTS_DEFAULT_MAX_AGE_DAYS = 7
TTS_DEFAULT_MAX_FILES   = 10

# ── Flashcards / Quiz ──────────────────────────────────────────
FLASHCARD_SESSION_TIMEOUT_SECS = 300
FLASHCARD_DEFAULT_CARD_COUNT   = 10
FLASHCARD_DEFAULT_QUIZ_COUNT   = 10
FLASHCARD_REVIEW_COOLDOWN_DAYS = 3
FLASHCARD_QUIZ_AUTO_ADVANCE_SECS = 2
FLASHCARD_QUIZ_DISTRACTORS     = 3

# ── Telegram ───────────────────────────────────────────────────
TG_MAX_MSG_LEN      = 4096
TG_SAFE_TRUNCATE    = 3900
TG_HISTORY_PURGE_DAYS = 30

# ── Profile defaults (fallbacks when config is silent) ──────────
DEFAULT_LEARNING_LANGUAGE = "de"
DEFAULT_NATIVE_LANGUAGE   = "en"
DEFAULT_PROFILE_NAME      = "default"


# ── Config loader ────────────────────────────────────────────────────

def load_config(path=None, fallback=None):
    """Load and return the project config as a dict.

    Args:
        path: Optional path to a JSON config file. Defaults to
              ``config.json`` in the project root.
        fallback: Value to return on any error (default: raise).

    Returns:
        dict with the parsed configuration, or ``fallback`` on error.

    Raises:
        FileNotFoundError: if the config file does not exist and no fallback.
        json.JSONDecodeError: if the file is not valid JSON and no fallback.
    """
    target = pathlib.Path(path) if path else CONFIG_PATH
    try:
        with open(target, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        if fallback is not None:
            return fallback
        raise
