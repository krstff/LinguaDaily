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
