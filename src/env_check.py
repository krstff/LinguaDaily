#!/usr/bin/env python3
"""
Environment health check for LinguaDaily.

Validates configuration, checks service connectivity, and verifies that
all required Python packages are installed. Designed to be run before
starting the daemon so setup problems are caught early.

Usage:
    python3 src/env_check.py                  # full check
    python3 src/env_check.py --quick          # skip network checks
    python3 src/env_check.py --config path    # custom config file

Exit codes:
    0 — all checks passed (warnings only)
    1 — one or more errors found
"""

import importlib
import json
import os
import sys
from urllib.request import urlopen
from urllib.error import URLError

from config import CONFIG_PATH, DATA_DIR, PROJECT_DIR, load_config


# ── Required Python packages ────────────────────────────────────────

REQUIRED_PACKAGES = {
    "openai":         "LLM client (translation, vocab, tutor)",
    "requests":       "HTTP client (Kiwix, TTS, news RSS)",
    "beautifulsoup4": "HTML parsing (Wikipedia article extraction)",
    "feedparser":     "RSS feed parsing (news source)",
    "aiogram":        "Telegram bot framework",
    "apscheduler":   "Lesson scheduler",
}

# ── Config schema validation rules ──────────────────────────────────

def validate_config(config):
    """Return (errors, warnings) from config structure checks."""
    errors = []
    warnings = []

    # --- Top-level keys ---
    if not config.get("profiles"):
        errors.append("No 'profiles' section in config.json")
    else:
        profiles = config["profiles"]
        default = config.get("default_profile")
        if default and default not in profiles:
            warnings.append(
                f"default_profile '{default}' is not in profiles list"
            )

        for name, profile in profiles.items():
            _validate_profile(errors, warnings, name, profile)

    # --- LLM section ---
    llm = config.get("llm")
    if llm:
        if not llm.get("base_url"):
            errors.append("llm.base_url is missing — LLM calls will fail")
        if not llm.get("default_model"):
            warnings.append("llm.default_model is not set (LLM client has a fallback)")
    else:
        warnings.append(
            "No 'llm' section — translation, vocab extraction, and tutor chat disabled"
        )

    # --- Telegram section ---
    tg = config.get("telegram")
    if tg:
        if not tg.get("bot_token"):
            errors.append("telegram.bot_token is empty — bot cannot start")
    else:
        warnings.append("No 'telegram' section — lesson delivery and tutor chat disabled")

    # --- Kiwix servers (needed for wikipedia source) ---
    wiki_profiles = [
        (n, p) for n, p in config.get("profiles", {}).items()
        if p.get("source") == "wikipedia"
    ]
    if wiki_profiles:
        kiwix = config.get("kiwix_servers", {})
        if not kiwix:
            errors.append(
                "Profiles use 'wikipedia' source but 'kiwix_servers' is empty/missing"
            )
        for name, profile in wiki_profiles:
            learning_lang = profile.get("learning_language")
            if learning_lang and learning_lang not in kiwix:
                errors.append(
                    f"Profile '{name}' uses learning_language='{learning_lang}' "
                    f"but no kiwix_servers entry for '{learning_lang}'"
                )
            elif learning_lang and learning_lang in kiwix:
                srv = kiwix[learning_lang]
                if not srv.get("base_url"):
                    errors.append(
                        f"kiwix_servers['{learning_lang}'].base_url is empty"
                    )
                if not srv.get("zim_name"):
                    errors.append(
                        f"kiwix_servers['{learning_lang}'].zim_name is empty"
                    )

    # --- TTS section ---
    tts_profiles = [
        (n, p) for n, p in config.get("profiles", {}).items()
        if p.get("use_tts")
    ]
    if tts_profiles:
        tts = config.get("tts")
        if not tts:
            warnings.append(
                f"Profiles use TTS ('{', '.join(n for n,_ in tts_profiles)}') "
                "but no 'tts' section — audio generation will fail"
            )
        elif not tts.get("base_url"):
            errors.append("tts.base_url is missing — TTS calls will fail")

    # --- Schedule sanity ---
    for name, profile in config.get("profiles", {}).items():
        sched = profile.get("schedule", {})
        time_str = sched.get("time", "")
        if time_str:
            parts = time_str.split(":")
            if len(parts) != 2:
                errors.append(
                    f"Profile '{name}' schedule.time '{time_str}' is not HH:MM"
                )
            else:
                try:
                    h, m = int(parts[0]), int(parts[1])
                    if not (0 <= h <= 23 and 0 <= m <= 59):
                        errors.append(
                            f"Profile '{name}' schedule.time '{time_str}' is out of range"
                        )
                except ValueError:
                    errors.append(
                        f"Profile '{name}' schedule.time '{time_str}' contains non-numeric values"
                    )

    return errors, warnings


def _validate_profile(errors, warnings, name, profile):
    """Validate a single profile dict."""
    if not profile.get("telegram_chat_id"):
        warnings.append(
            f"Profile '{name}' has no telegram_chat_id — "
            "lessons won't be delivered via Telegram"
        )
    if not profile.get("native_language"):
        errors.append(f"Profile '{name}' is missing 'native_language'")
    if not profile.get("learning_language"):
        errors.append(f"Profile '{name}' is missing 'learning_language'")


# ── Connectivity checks ─────────────────────────────────────────────

def check_http(url, label, timeout=5):
    """Return (ok: bool, message: str)."""
    try:
        resp = urlopen(url.rstrip("/") + "/", timeout=timeout)
        if 200 <= resp.status < 400:
            return True, f"✅ {label}: {url} reachable (HTTP {resp.status})"
        return False, f"❌ {label}: {url} returned HTTP {resp.status}"
    except URLError as e:
        return False, f"❌ {label}: {url} unreachable ({e.reason})"
    except Exception as e:
        return False, f"❌ {label}: {url} error — {e}"


def check_llm(config):
    """Check LLM endpoint connectivity."""
    llm = config.get("llm", {})
    base_url = llm.get("base_url")
    if not base_url:
        return []
    results = []
    # Check the /v1/models endpoint (standard OpenAI-compatible)
    ok, msg = check_http(base_url.rstrip("/") + "/models", "LLM")
    results.append(msg)
    return results


def check_kiwix(config):
    """Check all configured Kiwix servers."""
    results = []
    kiwix = config.get("kiwix_servers", {})
    for lang, srv in kiwix.items():
        url = srv.get("base_url")
        if not url:
            continue
        ok, msg = check_http(url, f"Kiwix ({lang})")
        results.append(msg)
    return results


def check_telegram(config):
    """Validate the Telegram bot token by calling getMe."""
    tg = config.get("telegram", {})
    token = tg.get("bot_token", "")
    if not token:
        return []
    results = []
    url = f"https://api.telegram.org/bot{token}/getMe"
    ok, msg = check_http(url, "Telegram API")
    results.append(msg)
    return results


def check_tts(config):
    """Check TTS endpoint connectivity."""
    tts = config.get("tts", {})
    base_url = tts.get("base_url")
    if not base_url:
        return []
    results = []
    ok, msg = check_http(base_url.rstrip("/") + "/models", "TTS")
    results.append(msg)
    return results


# ── Package checks ──────────────────────────────────────────────────

def check_packages():
    """Return list of (ok: bool, message: str) for each required package."""
    results = []
    for pkg_name, purpose in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg_name)
            version = getattr(mod, "__version__", "?")
            results.append((True, f"✅ {pkg_name} ({version}) — {purpose}"))
        except ImportError:
            results.append((False, f"❌ {pkg_name} NOT INSTALLED — {purpose}"))
    return results


# ── File / directory checks ────────────────────────────────────────

def check_files(config):
    """Check that key directories and files exist or can be created."""
    results = []

    # Project root
    if PROJECT_DIR.exists():
        results.append((True, f"✅ Project root: {PROJECT_DIR}"))
    else:
        results.append((False, f"❌ Project root missing: {PROJECT_DIR}"))

    # Data directory
    if DATA_DIR.exists():
        results.append((True, f"✅ Data dir: {DATA_DIR}"))
    elif DATA_DIR.parent.exists():
        results.append((True, f"⚠️  Data dir will be created on first run: {DATA_DIR}"))
    else:
        results.append((False, f"❌ Cannot create data dir: {DATA_DIR}"))

    # Per-profile data dirs and vocab files
    for name, profile in config.get("profiles", {}).items():
        pdir = DATA_DIR / name
        vfile = pdir / "vocabulary.md"
        if pdir.exists():
            if vfile.exists():
                results.append((True, f"  ✅ {name}: data dir + vocabulary.md OK"))
            else:
                results.append((True, f"  ⚠️  {name}: data dir exists, vocabulary.md will be created on first lesson"))
        else:
            results.append((True, f"  ℹ️  {name}: data dir will be created on first lesson"))

    return results


# ── Main entry point ───────────────────────────────────────────────

def run(config_path=None, skip_network=False):
    """
    Run the full environment check.

    Parameters
    ----------
    config_path : str or None
        Path to config.json. Defaults to project root.
    skip_network : bool
        If True, skip all HTTP connectivity checks (fast offline mode).

    Returns
    -------
    int
        0 if no errors (warnings are OK), 1 if errors found.
    """
    print("=" * 60)
    print("  LinguaDaily — Environment Health Check")
    print("=" * 60)
    all_errors = []
    all_warnings = []

    # ── 1. Load config ────────────────────────────────────────────
    print("\n📋 Config file...")
    try:
        config = load_config(config_path)
        print(f"  ✅ Loaded: {CONFIG_PATH if not config_path else config_path}")
    except FileNotFoundError as e:
        print(f"  ❌ Config not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON in config: {e}")
        return 1

    # ── 2. Config validation ──────────────────────────────────────
    print("\n🔍 Config structure...")
    cfg_errors, cfg_warnings = validate_config(config)
    for e in cfg_errors:
        print(f"  ❌ {e}")
        all_errors.append(e)
    for w in cfg_warnings:
        print(f"  ⚠️  {w}")
        all_warnings.append(w)
    if not cfg_errors and not cfg_warnings:
        print("  ✅ Config structure OK")

    # ── 3. File / directory checks ────────────────────────────────
    print("\n📁 Files & directories...")
    for ok, msg in check_files(config):
        print(f"  {msg}")
        if not ok:
            all_errors.append(msg)

    # ── 4. Package checks ─────────────────────────────────────────
    print("\n📦 Python packages...")
    for ok, msg in check_packages():
        print(f"  {msg}")
        if not ok:
            all_errors.append(msg)

    # ── 5. Connectivity (optional) ────────────────────────────────
    if not skip_network:
        print("\n🌐 Service connectivity...")
        for msg in check_llm(config):
            print(f"  {msg}")
            if "❌" in msg:
                all_errors.append(msg)

        for msg in check_kiwix(config):
            print(f"  {msg}")
            if "❌" in msg:
                all_errors.append(msg)

        for msg in check_telegram(config):
            print(f"  {msg}")
            if "❌" in msg:
                all_errors.append(msg)

        for msg in check_tts(config):
            print(f"  {msg}")
            if "❌" in msg:
                all_errors.append(msg)
    else:
        print("\n🌐 Service connectivity... (skipped — use without --quick to test)")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not all_errors and not all_warnings:
        print("  🚀 All checks passed — environment is healthy!")
    else:
        if all_errors:
            print(f"  🚨 {len(all_errors)} error(s) found:")
            for e in all_errors:
                print(f"    • {e}")
        if all_warnings:
            print(f"\n  ⚠️  {len(all_warnings)} warning(s):")
            for w in all_warnings:
                print(f"    • {w}")
    print("=" * 60)

    return 1 if all_errors else 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LinguaDaily environment check")
    parser.add_argument("--config", "-c", default=None, help="Path to config.json")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Skip network connectivity checks")
    args = parser.parse_args()

    sys.exit(run(config_path=args.config, skip_network=args.quick))


if __name__ == "__main__":
    main()
