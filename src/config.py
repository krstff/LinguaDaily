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

# ── Path resolution (computed once at import time) ───────────────────

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "output"
LOG_FILE    = PROJECT_DIR / "lingua.log"


# ── Config loader ────────────────────────────────────────────────────

def load_config(path=None):
    """Load and return the project config as a dict.

    Args:
        path: Optional path to a JSON config file. Defaults to
              ``config.json`` in the project root.

    Returns:
        dict with the parsed configuration.

    Raises:
        FileNotFoundError: if the config file does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    target = pathlib.Path(path) if path else CONFIG_PATH
    with open(target, encoding="utf-8") as f:
        return json.load(f)
