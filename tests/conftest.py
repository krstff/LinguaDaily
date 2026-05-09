"""Pytest configuration — add src/ to sys.path for local module imports."""

import os
import sys

# Add project src/ directory to sys.path so modules can import each other
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
