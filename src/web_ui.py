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

def create_app(config_path=None, log_file=None, password=None):
    """Create and configure the Flask web UI app.

    Args:
        config_path: Path to config.json (default: project root)
        log_file:    Path to lingua.log (default: project root)
        password:    Optional basic-auth password for remote access
    """
    global _config_path, _log_file, _password

    if config_path:
        _config_path = Path(config_path)
    if log_file:
        _log_file = Path(log_file)
    if password is not None:
        _password = password

    app = Flask(__name__, template_folder=_TEMPLATE_DIR)
    app.secret_key = "lingua-webui"  # minimal, no sessions used

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
            if not name or name in profiles:
                # If saving back to same name, it's already deleted above — re-add
                profiles[name] = profile_data
            else:
                return jsonify({"message": f"Profile '{name}' conflicts"}), 409

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
