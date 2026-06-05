"""Pytest configuration — add src/ to sys.path for local module imports."""

import os
import sys

# Add project src/ directory to sys.path so modules can import each other
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


import pytest


@pytest.fixture(autouse=True)
def reset_shared_openai_client():
    """Reset the shared OpenAI client singleton before/after each test."""
    for module_name in ("config", "src.config"):
        if module_name in sys.modules:
            mod = sys.modules[module_name]
            if hasattr(mod, "get_openai_client") and hasattr(mod.get_openai_client, "_instance"):
                mod.get_openai_client._instance = None
    yield
    for module_name in ("config", "src.config"):
        if module_name in sys.modules:
            mod = sys.modules[module_name]
            if hasattr(mod, "get_openai_client") and hasattr(mod.get_openai_client, "_instance"):
                mod.get_openai_client._instance = None
