#!/usr/bin/env python3
"""
LinguaDaily Web UI — lightweight admin panel for config management and log reading.

Standalone usage:
    python src/web_ui.py --host 127.0.0.1 --port 8089

Integrated into main.py:
    python src/main.py --web-ui

Templates live in src/templates/ (Jinja2 + HTMX).
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path

# ── Ensure src/ is on path for standalone execution ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (CONFIG_PATH, LOG_FILE, PROJECT_DIR,
                        resolve_language_name, load_config)

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    print("Error: flask is required. Install with: pip install flask", file=sys.stderr)
    sys.exit(1)


# ── Globals (set at startup or via create_app) ─────────────────────

_config_path = CONFIG_PATH
_log_file = LOG_FILE
_password = None
_scheduler_ref = None  # weak reference to LessonScheduler for hot-reload

# Path to templates directory (next to this file)
_TEMPLATE_DIR = str(Path(__file__).resolve().parent / "templates")


# ── Authentication helper ────────────────────────────────

def require_auth(f):
    """Optional basic auth decorator. Only active if password is set."""
    @wraps(f)
    def decorated(*args, **kwargs):
        global _password
        if not _password:
            return f(*args, **kwargs)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            import base64
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                user, pwd = decoded.split(":", 1)
                if pwd == _password:
                    return f(*args, **kwargs)
            except Exception:
                pass
        response = jsonify({"message": "Authentication required"})
        response.status_code = 401
        response.headers["WWW-Authenticate"] = 'Basic realm="LinguaDaily Admin"'
        return response
    return decorated


# ── Flask application factory ────────────────────────────

def create_app(config_path=None, log_file=None, password=None,
               scheduler=None):
    """Create and configure the Flask web UI app.

    Args:
        config_path: Path to config.json (default: project root)
        log_file:    Path to lingua.log (default: project root)
        password:    Optional basic-auth password for remote access
        scheduler:   Optional LessonScheduler instance for hot-reload support
    """
    global _config_path, _log_file, _password, _scheduler_ref

    if config_path:
        _config_path = Path(config_path)
    if log_file:
        _log_file = Path(log_file)
    if password is not None:
        _password = password
    _scheduler_ref = scheduler
    """Create and configure the Flask web UI app.

    Args:
        config_path: Path to config.json (default: project root)
        log_file:    Path to lingua.log (default: project root)
        password:    Optional basic-auth password for remote access
    """


    app = Flask(__name__, template_folder=_TEMPLATE_DIR)
    app.secret_key = "lingua-webui"  # minimal, no sessions used

    # ── Global context: Kiwix languages for profile dropdowns ──
    @app.context_processor
    def inject_kiwix_languages():
        try:
            cfg = load_config(_config_path)
        except Exception:
            cfg = {}
        servers = cfg.get("kiwix_servers", {})
        # Sorted list of language codes that have Kiwix server entries
        langs = sorted(servers.keys())
        return {"kiwix_languages": langs}

    # ── Hot-reload endpoint ──────────────────────────────
    @app.route("/api/reload", methods=["POST"])
    @require_auth
    def reload_config():
        """Reload config from disk and refresh scheduler jobs.

        Called after any config/profile change so the running daemon
        picks up new profiles, schedule changes, and enable/disable toggles
        without requiring a restart.
        """
        global _scheduler_ref
        try:
            # Re-read config to confirm it's valid JSON
            load_config(_config_path)

            if _scheduler_ref is not None:
                _scheduler_ref.reload_config()
                _scheduler_ref.reload_jobs()
                return jsonify({"message": "Config reloaded — scheduler jobs updated"})
            else:
                return jsonify({"message": "Config validated (no scheduler attached)"})
        except Exception as e:
            return jsonify({"message": f"Reload failed: {e}"}), 500

    # ── Dashboard ────────────────────────────────────────
    @app.route("/")
    @require_auth
    def dashboard():
        try:
            config = load_config(_config_path)
        except Exception as e:
            return render_template("dashboard.html", active="dashboard",
                                   profiles={}, profile_count=0,
                                   scheduled_count=0, lang_count=0,
                                   error=str(e)), 500

        profiles = config.get("profiles", {})
        scheduled = sum(
            1 for p in profiles.values()
            if p.get("enabled", True) and (p.get("schedule") or {}).get("time")
        )
        langs = set(
            p.get("learning_language")
            for p in profiles.values()
            if p.get("learning_language"))

        return render_template("dashboard.html", active="dashboard",
                               profiles=profiles,
                               profile_count=len(profiles),
                               scheduled_count=scheduled,
                               lang_count=len(langs))

    # ── Logs viewer ──────────────────────────────────────
    @app.route("/logs")
    @require_auth
    def logs_page():
        return render_template("logs.html", active="logs")

    @app.route("/api/logs/tail")
    @require_auth
    def log_tail():
        lines = request.args.get("lines", 500, type=int)
        level_filter = request.args.get("level", "ALL").upper()

        try:
            if not _log_file.exists():
                return "Log file not found."

            with open(_log_file, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            # Take last N lines
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines

            # Colorize and filter
            output = []
            for line in tail:
                stripped = line.rstrip("\n")
                # Determine level from log format: "2026-05-09 13:06:01,450 [module] LEVEL:"
                level_class = ""
                if "[lingua]" in stripped or "[scheduler]" in stripped or "[telegram_bot]" in stripped:
                    for lvl in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                        if f" {lvl}:" in stripped:
                            level_class = lvl.lower()
                            break

                # Apply level filter
                if level_filter != "ALL":
                    has_level = any(f" {level_filter}:" in stripped for _ in [1])
                    if not has_level:
                        continue

                # Escape HTML
                safe = (stripped
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))

                if level_class:
                    output.append(f'<span class="{level_class}">{safe}</span>')
                else:
                    output.append(safe)

            return "\n".join(output)

        except Exception as e:
            return f"Error reading log: {e}"

    # ── Config editor ────────────────────────────────────
    @app.route("/config")
    @require_auth
    def config_page():
        try:
            with open(_config_path, encoding="utf-8") as f:
                raw = f.read()
            # Escape for HTML textarea (Jinja2 autoescapes, but we pass raw)
            content = render_template("config.html", active="config", config_text=raw)
        except Exception as e:
            content = render_template("config.html", active="config",
                                      config_text="", error=str(e))

        return content

    @app.route("/api/config/save", methods=["POST"])
    @require_auth
    def save_config():
        raw = request.form.get("config_json", "")

        # Validate JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            return jsonify({"message": f"Invalid JSON: {e}"}), 400

        # Write with backup
        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)

            formatted = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
            with open(_config_path, "w", encoding="utf-8") as f:
                f.write(formatted)

            return jsonify({"message": "Config saved", "config_json": formatted})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    # ── Sources page ────────────────────────────────────────
    @app.route("/sources")
    @require_auth
    def sources_page():
        try:
            config = load_config(_config_path)
        except Exception as e:
            return render_template("sources.html", active="sources",
                                   kiwix_servers={}, news_feeds={},
                                   feed_languages=[],
                                   error=str(e)), 500

        kiwix_servers = config.get("kiwix_servers", {})
        feeds_raw = (config.get("sources", {}) or {}).get("news", {}) or {}
        news_feeds = feeds_raw.get("feeds", {})

        # Collect available languages from both Kiwix servers and news feeds
        all_langs = set(kiwix_servers.keys())
        all_langs.update(news_feeds.keys())
        feed_languages = sorted(all_langs)

        return render_template("sources.html", active="sources",
                               kiwix_servers=kiwix_servers,
                               news_feeds=news_feeds,
                               feed_languages=feed_languages)

    # ── Kiwix server CRUD ───────────────────────────────
    @app.route("/api/sources/kiwix", methods=["POST"])
    @require_auth
    def api_kiwix():
        action = request.form.get("action", "add")
        config = load_config(_config_path)
        kiwix = config.setdefault("kiwix_servers", {})

        lang = request.form.get("lang", "").strip().lower()
        if not lang:
            return jsonify({"message": "Language code is required"}), 400

        base_url = request.form.get("base_url", "").strip()
        zim_name = request.form.get("zim_name", "").strip()

        server_data = {}
        if base_url:
            server_data["base_url"] = base_url
        if zim_name:
            server_data["zim_name"] = zim_name

        if action == "add":
            if lang in kiwix:
                return jsonify({"message": f"Server for '{lang}' already exists"}), 409
            kiwix[lang] = server_data
        elif action == "edit":
            if lang not in kiwix:
                return jsonify({"message": f"Server for '{lang}' not found"}), 404
            kiwix[lang] = server_data

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)
            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            return jsonify({"message": f"Kiwix server '{lang}' saved"})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    @app.route("/api/sources/kiwix/<lang>", methods=["DELETE"])
    @require_auth
    def delete_kiwix(lang):
        config = load_config(_config_path)
        kiwix = config.get("kiwix_servers", {})

        lang_lower = lang.lower()
        if lang_lower not in kiwix:
            return jsonify({"message": f"Server for '{lang}' not found"}), 404

        del kiwix[lang_lower]

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)
            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            return jsonify({"message": f"Kiwix server '{lang}' removed"})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    # ── Add language support (scaffolds Kiwix + feeds) ──────
    @app.route("/api/sources/language", methods=["POST"])
    @require_auth
    def add_language():
        """Add a new language with Kiwix server entry and empty feeds bucket.

        Form fields: lang (required), base_url (optional), zim_name (optional)
        """
        lang = request.form.get("lang", "").strip().lower()
        if not lang or len(lang) != 2:
            return jsonify({"message": "Enter a valid two-letter language code"}), 400

        config = load_config(_config_path)

        # Check for existing Kiwix entry
        kiwix = config.setdefault("kiwix_servers", {})
        if lang in kiwix:
            return jsonify({"message": f"Kiwix server for '{lang}' already exists"}), 409

        # Add Kiwix server (with optional URL/ZIM from form, or empty)
        base_url = request.form.get("base_url", "").strip()
        zim_name = request.form.get("zim_name", "").strip()
        kiwix[lang] = {}
        if base_url:
            kiwix[lang]["base_url"] = base_url
        if zim_name:
            kiwix[lang]["zim_name"] = zim_name

        # Add empty feeds bucket
        sources = config.setdefault("sources", {})
        news = sources.setdefault("news", {})
        feeds = news.setdefault("feeds", {})
        feeds.setdefault(lang, {})

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)
            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            return jsonify({"message": f"Language '{lang}' added"})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    # ── RSS news feeds CRUD (language-keyed) ────────────────
    @app.route("/api/sources/news-feeds", methods=["POST"])
    @require_auth
    def api_news_feeds():
        action = request.form.get("action", "add")
        config = load_config(_config_path)
        sources = config.setdefault("sources", {})
        news = sources.setdefault("news", {})
        feeds = news.setdefault("feeds", {})

        lang = request.form.get("lang", "en").strip().lower()
        topic = request.form.get("topic", "").strip()
        if not topic:
            return jsonify({"message": "Topic is required"}), 400

        # Ensure language bucket exists
        lang_feeds = feeds.setdefault(lang, {})

        urls_raw = request.form.get("urls", "").strip()
        urls = [u.strip() for u in urls_raw.splitlines() if u.strip()] if urls_raw else []

        if action == "add":
            if topic in lang_feeds:
                return jsonify({"message": f"Topic '{topic}' already exists for '{lang}'"}), 409
            lang_feeds[topic] = urls
        elif action == "edit":
            if topic not in lang_feeds:
                return jsonify({"message": f"Topic '{topic}' not found for '{lang}'"}), 404
            lang_feeds[topic] = urls

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)
            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            return jsonify({"message": f"News feeds for '{lang}' / '{topic}' saved"})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    @app.route("/api/sources/news-feeds/<lang>/<topic>", methods=["DELETE"])
    @require_auth
    def delete_news_feed(lang, topic):
        config = load_config(_config_path)
        feeds = (config.get("sources", {}) or {}).get("news", {}).get("feeds", {})
        lang_lower = lang.lower()

        lang_feeds = feeds.get(lang_lower, {})
        if topic not in lang_feeds:
            return jsonify({"message": f"Topic '{topic}' not found for '{lang}'"}), 404

        del lang_feeds[topic]

        # Clean up empty language buckets
        if not lang_feeds and lang_lower in feeds:
            del feeds[lang_lower]

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)
            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            return jsonify({"message": f"News feeds for '{lang}' / '{topic}' removed"})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    # ── Profile CRUD API ─────────────────────────────────
    @app.route("/api/profiles", methods=["POST"])
    @require_auth
    def api_profiles():
        action = request.form.get("action", "add")
        config = load_config(_config_path)
        profiles = config.setdefault("profiles", {})

        name = request.form.get("name", "").strip()
        if not name:
            return jsonify({"message": "Profile name is required"}), 400

        profile_data = {
            "telegram_chat_id": request.form.get(
                "telegram_chat_id", "").strip() or None,
            "native_language": request.form.get("native_language", "en"),
            "learning_language": request.form.get("learning_language", ""),
            "source": request.form.get("source", "wikipedia"),
            "article_filter": {
                "min_words": int(request.form.get("min_words", 30)),
                "max_words": int(request.form.get("max_words", 150)),
            },
            "use_tts": request.form.get("use_tts") == "on",
            "tts_voice": request.form.get("tts_voice", "male"),
            "enabled": request.form.get("enabled") != "false",
        }

        schedule_time = request.form.get("schedule_time", "").strip()
        schedule_tz = request.form.get("schedule_tz", "Europe/Berlin").strip()
        if schedule_time:
            profile_data["schedule"] = {"time": schedule_time, "tz": schedule_tz}

        if action == "add":
            if name in profiles:
                return jsonify({"message": f"Profile '{name}' already exists"}), 409
            profiles[name] = profile_data

        elif action == "edit":
            old_name = request.form.get("edit_name", "").strip()
            if old_name and old_name in profiles:
                del profiles[old_name]
            # After deleting the old entry, name is free unless another profile
            # already uses it (rename collision)
            if name in profiles:
                return jsonify({"message": f"Profile '{name}' conflicts"}), 409
            profiles[name] = profile_data

        # Persist
        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)

            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")

            return jsonify({"message": f"Profile '{name}' saved"})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    @app.route("/api/profiles/<name>", methods=["DELETE"])
    @require_auth
    def delete_profile(name):
        config = load_config(_config_path)
        profiles = config.get("profiles", {})

        if name not in profiles:
            return jsonify({"message": f"Profile '{name}' not found"}), 404

        del profiles[name]

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)

            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")

            return jsonify({"message": f"Profile '{name}' deleted"})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    @app.route("/api/profiles/<name>/toggle", methods=["POST"])
    @require_auth
    def toggle_profile(name):
        """Toggle the enabled state of a profile."""
        config = load_config(_config_path)
        profiles = config.get("profiles", {})

        if name not in profiles:
            return jsonify({"message": f"Profile '{name}' not found"}), 404

        # Toggle: default to True if key doesn't exist yet
        current = profiles[name].get("enabled", True)
        profiles[name]["enabled"] = not current
        new_state = profiles[name]["enabled"]

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)

            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")

            state_word = "enabled" if new_state else "disabled"
            return jsonify({"message": f"Profile '{name}' {state_word}", "enabled": new_state})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    @app.route("/api/profiles/<name>/run-lesson", methods=["POST"])
    @require_auth
    def run_lesson(name):
        """Trigger a single lesson for the given profile, delivered to Telegram.

        Returns immediately with status; the lesson pipeline runs in a
        background thread so the HTTP response is fast.
        """
        import asyncio as _asyncio
        import threading as _threading

        config = load_config(_config_path)
        profiles = config.get("profiles", {})
        if name not in profiles:
            return jsonify({"message": f"Profile '{name}' not found"}), 404

        profile_cfg = profiles[name]
        chat_id = profile_cfg.get("telegram_chat_id")
        if not chat_id:
            return jsonify({"message": f"Profile '{name}' has no telegram_chat_id"}), 400

        # Run lesson pipeline + delivery in a background thread
        def _run():
            bot = None
            try:
                from orchestrator import Orchestrator
                from telegram_bot import TelegramBot
                orch = Orchestrator(config=config)
                bot = TelegramBot(config=config)
                lesson = _asyncio.run(orch.run_lesson(name, delivery_callback=bot.deliver_lesson))
                if lesson:
                    title = lesson.get("title", "?")
                    print(f"[web-ui] Lesson delivered for '{name}': {title}")
                else:
                    print(f"[web-ui] Lesson pipeline returned no result for '{name}'")
            except Exception as e:
                print(f"[web-ui] Error running lesson for '{name}': {e}")
            finally:
                # Clean up aiogram session + DB to avoid "Unclosed client session"
                try:
                    if bot and bot._bot:
                        _asyncio.run(bot.stop())
                except Exception:
                    pass
                try:
                    if bot:
                        bot.db.close()
                except Exception:
                    pass

        _threading.Thread(target=_run, daemon=True).start()
        return jsonify({"message": f"Lesson started for '{name}' — check logs for progress"})

    # ── Model management API ─────────────────────────────
    @app.route("/api/models/fetch")
    @require_auth
    def fetch_models():
        """Fetch available models from the OpenAI-compatible API.

        Queries both the LLM base_url and TTS base_url endpoints,
        returning a merged list of unique model IDs.
        """
        config = load_config(_config_path)
        llm_cfg = config.get("llm", {})
        tts_cfg = config.get("tts", {})

        urls = []
        llm_url = llm_cfg.get("base_url", "")
        if llm_url:
            urls.append(llm_url)
        tts_url = tts_cfg.get("base_url", "")
        if tts_url and tts_url != llm_url:
            urls.append(tts_url)

        # Fallback to env vars
        import os as _os
        env_url = _os.environ.get("LLAMA_BASE_URL", "")
        if env_url and env_url not in urls:
            urls.append(env_url)

        if not urls:
            return jsonify({"llm_models": [], "tts_models": [], "error": "No API URLs configured"})

        llm_models = []
        tts_models = []
        errors = []

        try:
            from openai import OpenAI
        except ImportError:
            return jsonify({"llm_models": [], "tts_models": [], "error": "openai package not installed"})

        for url in urls:
            api_key = llm_cfg.get("api_key", "") or "none"
            try:
                client = OpenAI(base_url=url, api_key=api_key, timeout=10)
                models = client.models.list()
                for m in models:
                    model_id = getattr(m, "id", str(m))
                    if model_id not in llm_models:
                        llm_models.append(model_id)
                    # TTS models typically have "tts" or "voice" in the ID, but
                    # we also include all models since OmniVoice may use any
                    if model_id not in tts_models:
                        tts_models.append(model_id)
            except Exception as e:
                errors.append(f"{url}: {e}")

        return jsonify({
            "llm_models": sorted(llm_models),
            "tts_models": sorted(tts_models),
            "errors": errors,
        })

    @app.route("/api/models/save", methods=["POST"])
    @require_auth
    def save_models():
        """Save global model selections to config.json.

        Expects JSON body:
        {
            "translate_model": "model-name",
            "tutor_model": "model-name",
            "tts_model": "omnivoice"
        }
        """
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"message": "Invalid JSON body"}), 400

        config = load_config(_config_path)
        llm_cfg = config.setdefault("llm", {})
        tts_cfg = config.setdefault("tts", {})

        changed = []

        if "translate_model" in data:
            val = data["translate_model"]
            # Empty string means "use default"
            if val:
                llm_cfg["translate_model"] = val
            elif "translate_model" in llm_cfg:
                del llm_cfg["translate_model"]
            changed.append("translate_model")

        if "tutor_model" in data:
            val = data["tutor_model"]
            if val:
                llm_cfg["tutor_model"] = val
            elif "tutor_model" in llm_cfg:
                del llm_cfg["tutor_model"]
            changed.append("tutor_model")

        if "tts_model" in data:
            val = data["tts_model"]
            tts_cfg["model"] = val if val else "omnivoice"
            changed.append("tts_model")

        try:
            backup = _config_path.with_suffix(".json.bak")
            if _config_path.exists():
                shutil.copy2(_config_path, backup)

            with open(_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")

            return jsonify({"message": f"Saved: {', '.join(changed)}", "changed": changed})
        except Exception as e:
            return jsonify({"message": f"Write error: {e}"}), 500

    # ── Model settings helper for dashboard template ─────
    @app.route("/api/models/current")
    @require_auth
    def current_models():
        """Return current model selections from config."""
        config = load_config(_config_path)
        llm_cfg = config.get("llm", {})
        tts_cfg = config.get("tts", {})
        return jsonify({
            "default_model": llm_cfg.get("default_model", ""),
            "translate_model": llm_cfg.get("translate_model", ""),
            "tutor_model": llm_cfg.get("tutor_model", ""),
            "tts_model": tts_cfg.get("model", "omnivoice"),
        })

    return app


# ── Standalone entry point ───────────────────────────────

def main():
    """Run the web UI standalone.

    Usage:
        python src/web_ui.py                     # defaults
        python src/web_ui.py --host 0.0.0.0      # bind all interfaces
        python src/web_ui.py --port 9090         # custom port
        python src/web_ui.py --password mypass   # enable basic auth
    """
    parser = argparse.ArgumentParser(description="LinguaDaily Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8089, help="Port (default: 8089)")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--password", default=None, help="Basic auth password for remote access")
    args = parser.parse_args()

    app = create_app(
        config_path=args.config,
        password=args.password,
    )

    print("=" * 50)
    print("  LinguaDaily Web UI")
    print("=" * 50)
    print(f"  URL:      http://{args.host}:{args.port}")
    print(f"  Config:   {_config_path}")
    print(f"  Log file: {_log_file}")
    if args.password:
        print("  Auth:     basic (password set)")
    else:
        print("  Auth:     none (localhost only)")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
