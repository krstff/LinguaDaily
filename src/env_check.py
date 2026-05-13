#!/usr/bin/env python3
"""
Environment health check for LinguaDaily.

Validates the entire deployment setup before starting the daemon:
  • Config file exists and is valid JSON
  • Config structure (profiles, LLM, Telegram, Kiwix, TTS, schedules)
  • Language codes resolve to known languages
  • Chat IDs are numeric, non-empty, no accidental duplicates
  • Article filter ranges make sense
  • Required Python packages are installed
  • Data directories exist or can be created
  • LLM endpoint is reachable and serves the configured models
  • Kiwix servers respond and can serve articles
  • Telegram bot token is valid (bot exists on Telegram)
  • TTS endpoint is reachable
  • News RSS feeds are accessible (if using news source)

Usage:
    python3 src/env_check.py                  # full check (with network)
    python3 src/env_check.py --quick          # skip network checks
    python3 src/env_check.py --config path    # custom config file

Exit codes:
    0 — all checks passed (warnings only)
    1 — one or more errors found
"""

import importlib
import json
import re
import sys
from urllib.request import urlopen
from urllib.error import URLError

from config import (
    CONFIG_PATH, DATA_DIR, PROJECT_DIR, load_config,
    resolve_language_name, LANGUAGE_NAMES,
)


# ── Required Python packages ────────────────────────────────────────

REQUIRED_PACKAGES = {
    "openai":         "LLM client (translation, vocab, tutor)",
    "requests":       "HTTP client (Kiwix, TTS, news RSS)",
    "beautifulsoup4": "HTML parsing (Wikipedia article extraction)",
    "feedparser":     "RSS feed parsing (news source)",
    "aiogram":        "Telegram bot framework",
    "apscheduler":   "Lesson scheduler",
}

# Valid sources a profile can use
VALID_SOURCES = {"wikipedia", "news"}


# ── Config schema validation ────────────────────────────────────────

def validate_config(config):
    """Return (errors, warnings) from config structure checks."""
    errors = []
    warnings = []

    # ── Top-level keys ────────────────────────────────────────────
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
            _validate_profile(errors, warnings, name, profile, config)

    # ── Check for duplicate chat IDs ──────────────────────────────
    _check_duplicate_chat_ids(config, errors, warnings)

    # ── LLM section ───────────────────────────────────────────────
    llm = config.get("llm")
    if llm:
        if not llm.get("base_url"):
            errors.append("llm.base_url is missing — LLM calls will fail")
        if not llm.get("default_model"):
            warnings.append(
                "llm.default_model is not set (LLM client has a hardcoded fallback)"
            )
    else:
        warnings.append(
            "No 'llm' section — translation, vocab extraction, and tutor chat disabled"
        )

    # ── Telegram section ──────────────────────────────────────────
    tg = config.get("telegram")
    if tg:
        if not tg.get("bot_token"):
            errors.append("telegram.bot_token is empty — bot cannot start")
        else:
            # Validate token format: <numeric_id>:<64-char-string>
            token = tg["bot_token"]
            if not re.match(r'^\d{10,}:.{30,50}$', token):
                warnings.append(
                    "telegram.bot_token format looks unusual — "
                    "expected <digits>:<alphanumeric string>"
                )
    else:
        warnings.append("No 'telegram' section — lesson delivery and tutor chat disabled")

    # ── Kiwix servers (needed for wikipedia source) ───────────────
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

    # ── TTS section ───────────────────────────────────────────────
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

    # ── News feeds (needed for news source) ───────────────────────
    news_profiles = [
        (n, p) for n, p in config.get("profiles", {}).items()
        if p.get("source") == "news"
    ]
    if news_profiles:
        feeds_cfg = config.get("sources", {}).get("news", {}).get("feeds", {})
        if not feeds_cfg:
            warnings.append(
                f"Profiles use 'news' source ('{', '.join(n for n,_ in news_profiles)}') "
                "but no news feeds configured in sources.news.feeds"
            )

    return errors, warnings


def _validate_profile(errors, warnings, name, profile, config):
    """Validate a single profile dict."""

    # ── Required fields ───────────────────────────────────────────
    if not profile.get("native_language"):
        errors.append(f"Profile '{name}' is missing 'native_language'")
    if not profile.get("learning_language"):
        errors.append(f"Profile '{name}' is missing 'learning_language'")

    # ── Language code validation ──────────────────────────────────
    for field in ("native_language", "learning_language"):
        code = profile.get(field)
        if code and code not in LANGUAGE_NAMES:
            warnings.append(
                f"Profile '{name}'.{field}='{code}' — unknown language code "
                f"(not in LANGUAGE_NAMES; will fall back to using the code as-is)"
            )

    # ── Telegram chat ID ──────────────────────────────────────────
    chat_id = profile.get("telegram_chat_id")
    if chat_id is not None and chat_id != "":
        # Must be numeric (int or string of digits)
        try:
            int(chat_id)
        except (ValueError, TypeError):
            errors.append(
                f"Profile '{name}' telegram_chat_id='{chat_id}' is not a valid integer"
            )

    if not chat_id:
        warnings.append(
            f"Profile '{name}' has no telegram_chat_id — "
            "lessons won't be delivered via Telegram"
        )

    # ── Source validation ─────────────────────────────────────────
    source = profile.get("source")
    if source and source not in VALID_SOURCES:
        errors.append(
            f"Profile '{name}' source='{source}' — must be one of {sorted(VALID_SOURCES)}"
        )

    # ── Article filter sanity ─────────────────────────────────────
    af = profile.get("article_filter", {})
    min_w = af.get("min_words")
    max_w = af.get("max_words")
    if min_w is not None and max_w is not None:
        if min_w >= max_w:
            errors.append(
                f"Profile '{name}' article_filter: min_words ({min_w}) >= "
                f"max_words ({max_w}) — no articles will pass the filter"
            )

    # ── Schedule validation ───────────────────────────────────────
    sched = profile.get("schedule", {})
    time_str = sched.get("time", "")
    if time_str:
        parts = time_str.split(":")
        if len(parts) != 2:
            errors.append(
                f"Profile '{name}' schedule.time '{time_str}' is not HH:MM format"
            )
        else:
            try:
                h, m = int(parts[0]), int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    errors.append(
                        f"Profile '{name}' schedule.time '{time_str}' is out of range "
                        f"(hours 0-23, minutes 0-59)"
                    )
            except ValueError:
                errors.append(
                    f"Profile '{name}' schedule.time '{time_str}' contains non-numeric values"
                )


def _check_duplicate_chat_ids(config, errors, warnings):
    """Warn if multiple profiles share the same Telegram chat ID."""
    chat_map = {}  # chat_id -> [profile_names]
    for name, profile in config.get("profiles", {}).items():
        cid = profile.get("telegram_chat_id")
        if cid:
            chat_map.setdefault(cid, []).append(name)

    for cid, profiles in chat_map.items():
        if len(profiles) > 1:
            warnings.append(
                f"Multiple profiles share chat_id={cid}: "
                f"{', '.join(profiles)} — this is valid (multi-language per user) "
                f"but ensure schedule times don't overlap"
            )


# ── Connectivity checks ─────────────────────────────────────────────

def _http_get_json(url, label, timeout=5):
    """HTTP GET returning (ok: bool, message: str, json_data: dict|None)."""
    try:
        resp = urlopen(url, timeout=timeout)
        status = resp.status
        if 200 <= status < 300:
            body = resp.read()
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                data = None
            return True, f"✅ {label}: {url} (HTTP {status})", data
        return False, f"❌ {label}: {url} returned HTTP {status}", None
    except URLError as e:
        return False, f"❌ {label}: {url} unreachable ({e.reason})", None
    except Exception as e:
        return False, f"❌ {label}: {url} error — {e}", None


def _http_get(url, label, timeout=5):
    """Simple HTTP GET returning (ok: bool, message: str)."""
    ok, msg, _ = _http_get_json(url, label, timeout)
    return ok, msg


def check_llm(config):
    """Check LLM endpoint: connectivity + model availability."""
    llm = config.get("llm", {})
    base_url = llm.get("base_url")
    if not base_url:
        return []

    results = []
    base = base_url.rstrip("/")

    # 1. Check /v1/models endpoint
    ok, msg, models_data = _http_get_json(base + "/models", "LLM (/v1/models)")
    results.append(msg)

    if not ok:
        return results

    # 2. Extract available model IDs
    available_models = set()
    if isinstance(models_data, dict):
        for m in models_data.get("data", []):
            mid = m.get("id", "")
            if mid:
                available_models.add(mid)

    # 3. Check that configured models exist on the server
    models_to_check = {}
    default_model = llm.get("default_model")
    if default_model:
        models_to_check["llm.default_model"] = default_model
    translate_model = llm.get("translate_model")
    if translate_model:
        models_to_check["llm.translate_model"] = translate_model
    tutor_model = llm.get("tutor_model")
    if tutor_model:
        models_to_check["llm.tutor_model"] = tutor_model

    # Also check per-profile model overrides
    for pname, profile in config.get("profiles", {}).items():
        for key in ("llm_model", "llm_translate_model", "llm_tutor_model"):
            val = profile.get(key)
            if val:
                models_to_check[f"profiles.{pname}.{key}"] = val

    for cfg_key, model_name in models_to_check.items():
        if model_name not in available_models:
            results.append(
                f"⚠️  Model '{model_name}' ({cfg_key}) not found on LLM server — "
                f"available: {', '.join(sorted(available_models)[:5])}{'...' if len(available_models) > 5 else ''}"
            )

    return results


def check_kiwix(config):
    """Check all configured Kiwix servers."""
    results = []
    kiwix = config.get("kiwix_servers", {})
    for lang, srv in kiwix.items():
        url = srv.get("base_url")
        if not url:
            continue

        # Check root endpoint
        ok, msg = _http_get(url.rstrip("/") + "/", f"Kiwix ({lang})")
        results.append(msg)

        # Try to fetch a random article (verifies ZIM is actually loaded)
        zim_name = srv.get("zim_name", "")
        if ok and zim_name:
            random_url = f"{url.rstrip('/')}/random?content={zim_name}"
            rok, rmsg = _http_get(random_url, f"Kiwix ({lang}) /random")
            if not rok:
                results.append(
                    f"⚠️  Kiwix ({lang}): root OK but /random failed — "
                    f"ZIM file may not be loaded or server misconfigured"
                )

    return results


def check_telegram(config):
    """Validate the Telegram bot token by calling getMe."""
    tg = config.get("telegram", {})
    token = tg.get("bot_token", "")
    if not token:
        return []

    results = []
    url = f"https://api.telegram.org/bot{token}/getMe"
    ok, msg, data = _http_get_json(url, "Telegram Bot API (getMe)")
    results.append(msg)

    if ok and data:
        bot_info = data.get("result", {})
        if not bot_info:
            results.append(
                f"⚠️  Telegram getMe returned no 'result' — token may be invalid"
            )
        else:
            bot_name = bot_info.get("username", "unknown")
            is_bot = bot_info.get("is_bot", False)
            if is_bot:
                results[-1] = f"✅ Telegram Bot: @{bot_name} (token valid)"
            else:
                results.append(
                    f"⚠️  Telegram entity '@{bot_name}' is not a bot — token may be wrong"
                )

    # Check that at least one profile has a chat_id configured
    profiles_with_chat = [
        n for n, p in config.get("profiles", {}).items()
        if p.get("telegram_chat_id")
    ]
    if not profiles_with_chat:
        results.append(
            "⚠️  No profiles have telegram_chat_id set — bot has no users to deliver to"
        )

    return results


def check_tts(config):
    """Check TTS endpoint connectivity."""
    tts = config.get("tts", {})
    base_url = tts.get("base_url")
    if not base_url:
        return []

    results = []
    # Try the /v1/models endpoint (OpenAI-compatible)
    ok, msg = _http_get(base_url.rstrip("/") + "/models", "TTS")
    results.append(msg)
    return results


def check_news_feeds(config):
    """Check that configured news RSS feeds are accessible."""
    results = []
    feeds_cfg = config.get("sources", {}).get("news", {}).get("feeds", {})

    # Only check feeds for languages actually used by profiles
    used_langs = set()
    for profile in config.get("profiles", {}).values():
        if profile.get("source") == "news":
            ll = profile.get("learning_language")
            if ll:
                used_langs.add(ll)

    # Check native language feeds too (articles are in learning_language but
    # feeds may be keyed by the article's source language)
    for lang, topics in feeds_cfg.items():
        if lang not in used_langs:
            continue
        for topic, urls in topics.items():
            for url in urls[:1]:  # check first feed per topic (not all)
                ok, msg = _http_get(url, f"RSS ({lang}/{topic})")
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
        results.append((False, f"❌ Cannot create data dir: parent of {DATA_DIR} missing"))

    # Per-profile data dirs and vocab files
    for name, profile in config.get("profiles", {}).items():
        pdir = DATA_DIR / name
        vfile = pdir / "vocabulary.md"
        if pdir.exists():
            if vfile.exists():
                results.append((True, f"  ✅ {name}: data dir + vocabulary.md OK"))
            else:
                results.append(
                    (True, f"  ℹ️  {name}: data dir exists, vocabulary.md created on first lesson")
                )
        else:
            results.append(
                (True, f"  ℹ️  {name}: data dir will be created on first lesson")
            )

    # Output directory for TTS
    from config import OUTPUT_DIR
    if OUTPUT_DIR.exists():
        results.append((True, f"✅ Output dir: {OUTPUT_DIR}"))
    elif OUTPUT_DIR.parent.exists():
        results.append((True, f"ℹ️  Output dir will be created on first run: {OUTPUT_DIR}"))

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
        cfg_file = CONFIG_PATH if not config_path else config_path
        print(f"  ✅ Loaded: {cfg_file}")
    except FileNotFoundError as e:
        print(f"  ❌ Config not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON in config: {e}")
        return 1

    # ── 2. Config structure validation ────────────────────────────
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

        for msg in check_news_feeds(config):
            print(f"  {msg}")
            if "❌" in msg:
                all_errors.append(msg)
    else:
        print("\n🌐 Service connectivity... (skipped — run without --quick to test)")

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

    parser = argparse.ArgumentParser(
        description="LinguaDaily environment health check",
        epilog="Exit code 0 = healthy, 1 = errors found",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to config.json (default: config.json in project root)",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Skip network connectivity checks (offline mode)",
    )
    args = parser.parse_args()

    sys.exit(run(config_path=args.config, skip_network=args.quick))


if __name__ == "__main__":
    main()
