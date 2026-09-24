"""Pytest configuration — add src/ to sys.path for local module imports."""

import os
import sys

# Add project src/ directory to sys.path so modules can import each other
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


import pytest


def _reset_client_caches():
    """Clear the shared OpenAI client caches in all loaded config modules."""
    for module_name in ("config", "src.config"):
        if module_name in sys.modules:
            mod = sys.modules[module_name]
            if hasattr(mod, "reset_openai_client"):
                mod.reset_openai_client()
            if hasattr(mod, "reset_embedding_client"):
                mod.reset_embedding_client()


@pytest.fixture(autouse=True)
def reset_shared_openai_client():
    """Reset the shared OpenAI client caches before/after each test."""
    _reset_client_caches()
    yield
    _reset_client_caches()


@pytest.fixture(autouse=True)
def isolate_shared_vocab_db(tmp_path, monkeypatch):
    """Ensure tests never leak into (or read from) the real shared VocabDB.

    The shared instance is created lazily on first use, so redirecting
    ``DEFAULT_DB_PATH`` to a per-test tmp dir before any test runs is
    sufficient: any accidental shared-DB access lands in tmp.

    Note: production code imports the top-level ``vocab_db`` module
    (src/ is on sys.path), while some tests import ``src.vocab_db`` —
    these are distinct module objects, so both are patched + reset.
    """
    import vocab_db
    from src import vocab_db as pkg_vocab_db
    for mod in (vocab_db, pkg_vocab_db):
        mod.reset_shared_db()
        monkeypatch.setattr(mod, "DEFAULT_DB_PATH", tmp_path / "shared_vocab.db")
    yield
    vocab_db.reset_shared_db()
    pkg_vocab_db.reset_shared_db()
