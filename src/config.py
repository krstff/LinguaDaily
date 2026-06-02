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
TG_MAX_MSG_LEN           = 4096
TG_SAFE_TRUNCATE         = 3900
TG_HISTORY_PURGE_DAYS    = 30
TG_LESSON_COOLDOWN_SECS  = 600   # minutes between /another requests

# ── RAG ───────────────────────────────────────────────────
RAG_DEFAULT_QDRANT_URL    = "http://localhost:6333"
RAG_DEFAULT_COLLECTION    = "linguadaily_docs"
RAG_DEFAULT_EMBED_MODEL   = "nomic-embed-text"
RAG_DEFAULT_CHUNK_SIZE    = 500
RAG_DEFAULT_CHUNK_OVERLAP = 100

# ── Profile defaults (fallbacks when config is silent) ──────────
DEFAULT_LEARNING_LANGUAGE = "de"
DEFAULT_NATIVE_LANGUAGE   = "en"
DEFAULT_PROFILE_NAME      = "default"


# ── Shared OpenAI client (singleton) ──────────────────────────────
#
# All modules that talk to llama.cpp share ONE OpenAI client instance.
# This avoids multiple HTTP connection pools fighting over the same
# server — especially important when llama-swap is loading/unloading
# models and transient timeouts trigger independent retries from
# separate clients ("zombie" duplicate requests).

def get_openai_client(base_url: str = None, api_key: str = "none", timeout: float = 60):
    """
    Get or create the shared OpenAI-compatible client for llama.cpp.

    Only ONE instance is ever created (module-level singleton) regardless
    of how many times this function is called. The base_url, api_key, and
    timeout are used on first creation only — subsequent calls return the
    same instance.

    Parameters
    ----------
    base_url : str or None
        API base URL (default: from config or LLM_DEFAULT_BASE_URL).
    api_key : str
        API key (default: "none" for local llama.cpp).
    timeout : float
        Default request timeout in seconds (default: 60).

    Returns
    -------
    OpenAI client instance, or None if the package is not installed.
    """
    import logging

    try:
        from openai import OpenAI
    except ImportError:
        logger_cfg = logging.getLogger("lingua")
        logger_cfg.warning("'openai' package not installed — LLM calls will fail.")
        return None

    if not hasattr(get_openai_client, "_instance") or get_openai_client._instance is None:
        resolved_url = base_url or get_llm_base_url()
        get_openai_client._instance = OpenAI(
            base_url=resolved_url,
            api_key=api_key or "none",
            timeout=timeout,
        )
    return get_openai_client._instance


def reset_openai_client():
    """Reset the shared OpenAI client (for tests / config reload)."""
    if hasattr(get_openai_client, "_instance"):
        get_openai_client._instance = None


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


def get_llm_base_url(path=None) -> str:
    """Return the LLM base_url from config.json, falling back to default."""
    import os
    cfg = load_config(path, fallback={})
    return (
        (cfg.get("llm", {}) or {}).get("base_url")
        or os.environ.get("LLAMA_BASE_URL")
        or LLM_DEFAULT_BASE_URL
    )


def get_rag_config(path=None) -> dict:
    """Return resolved RAG config: config.json values merged with defaults.

    Returns a dict with keys:
        qdrant_url, collection_name, embedding_model,
        chunk_size, chunk_overlap, embedding_base_url
    """
    import os
    cfg = load_config(path, fallback={})
    rag = cfg.get("rag", {}) or {}

    return {
        "qdrant_url": (
            rag.get("qdrant_url")
            or os.environ.get("QDRANT_URL")
            or RAG_DEFAULT_QDRANT_URL
        ),
        "collection_name": rag.get("collection_name", RAG_DEFAULT_COLLECTION),
        "embedding_model": (
            rag.get("embedding_model")
            or os.environ.get("EMBEDDING_MODEL")
            or RAG_DEFAULT_EMBED_MODEL
        ),
        "chunk_size": rag.get("chunk_size", RAG_DEFAULT_CHUNK_SIZE),
        "chunk_overlap": rag.get("chunk_overlap", RAG_DEFAULT_CHUNK_OVERLAP),
        "embedding_base_url": rag.get("embedding_base_url") or get_llm_base_url(path),
    }
