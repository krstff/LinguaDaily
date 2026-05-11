#!/usr/bin/env python3
"""
LinguaDaily standalone daemon entry point.

Wires together the Telegram bot (interactive tutor + lesson delivery) and
the lesson scheduler (daily automated pipeline). Runs both concurrently as
a single async process with graceful signal handling.

Usage:
    conda run -n lingua python src/main.py --config config.json
"""

import asyncio
import json
import logging
import os
import signal
import sys

from config import (CONFIG_PATH, DATA_DIR, LOG_FILE,
                    resolve_language_name, load_config)

logger = logging.getLogger("lingua")


class LinguaDaemon:
    """Top-level daemon managing scheduler + Telegram bot."""

    def __init__(self, config=None, config_path=None):
        if config is None:
            config = self._load_config(config_path)

        self.config = config
        self.bot = None
        self.scheduler = None
        self.web_server = None
        self._shutdown_event = asyncio.Event()

    def _load_config(self, path=None):
        """Load config from JSON file (delegates to shared loader)."""
        return load_config(path)

    def _setup_logging(self, verbose=False):
        """Configure logging to both console and file."""
        level = logging.DEBUG if verbose else logging.INFO

        # Root logger
        root = logging.getLogger()
        root.setLevel(level)

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(console)

        # File handler (rotating would be ideal, but keeping it simple)
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # always full detail in file
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        ))
        root.addHandler(file_handler)

        # Silence werkzeug request logs (GET /api/logs/tail floods the console)
        # but keep errors/warnings visible
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def _print_banner(self, web_ui_host=None, web_ui_port=None):
        """Print startup banner with config summary."""
        profiles = self.config.get("profiles", {})
        scheduled = [(n, p["schedule"]) for n, p in profiles.items()
                     if p.get("schedule") and p["schedule"].get("time")]

        print("=" * 60)
        print("  LinguaDaily Standalone Daemon")
        print("=" * 60)
        print(f"  Config:     {CONFIG_PATH}")
        print(f"  Profiles:   {len(profiles)} ({', '.join(profiles.keys())})")
        print(f"  Scheduled:  {len(scheduled)} daily lesson(s)")

        if scheduled:
            for name, sched in scheduled:
                profile = profiles[name]
                lang = resolve_language_name(
                    profile.get("learning_language", "?"))
                print(f"    • {name:<16} {sched['time']} ({sched.get('tz', 'UTC')}) → {lang}")

        tg_token = self.config.get("telegram", {}).get("bot_token")
        if tg_token:
            print(f"  Telegram:   ✅ configured (token: ...{tg_token[-6:]})")
        else:
            print("  Telegram:   ⚠️  no bot token — tutor chat disabled")

        llm_cfg = self.config.get("llm", {})
        if llm_cfg:
            print(f"  LLM:        {llm_cfg.get('default_model', '?')} "
                  f"@ {llm_cfg.get('base_url', '?')}")
        else:
            print("  LLM:        ⚠️  not configured — translation/tutor disabled")

        if web_ui_host and web_ui_port:
            print(f"  Web UI:     http://{web_ui_host}:{web_ui_port}")

        print("=" * 60)

    async def start(self, web_ui_host=None, web_ui_port=None):
        """Start all services concurrently."""
        self._print_banner(web_ui_host, web_ui_port)

        # ── Start Telegram bot (if configured) ─────────────────────
        tg_token = self.config.get("telegram", {}).get("bot_token") or os.environ.get(
            "TELEGRAM_BOT_TOKEN", ""
        )

        if tg_token:
            logger.info("Starting Telegram bot...")
            try:
                from telegram_bot import TelegramBot
                self.bot = TelegramBot(config=self.config)
            except Exception as e:
                logger.error("Failed to init Telegram bot: %s", e)
                self.bot = None

        # ── Start lesson scheduler ─────────────────────────────────
        delivery_callback = (self.bot.deliver_lesson if self.bot else None)

        logger.info("Starting lesson scheduler...")
        from scheduler import LessonScheduler
        self.scheduler = LessonScheduler(
            config=self.config,
            delivery_callback=delivery_callback,
        )

        # ── Start web UI (if requested) ────────────────────────────
        if web_ui_host and web_ui_port:
            logger.info("Starting Web UI on %s:%s...", web_ui_host, web_ui_port)
            self._start_web_ui(web_ui_host, web_ui_port)

        # ── Run concurrently ───────────────────────────────────────
        tasks = []

        # Telegram bot (polling loop)
        if self.bot:
            task = asyncio.create_task(self._run_bot(), name="telegram-bot")
            tasks.append(task)

        # Scheduler (blocking cron loop)
        task = asyncio.create_task(self._run_scheduler(), name="scheduler")
        tasks.append(task)

        logger.info("All services started — %d active task(s)", len(tasks))

        # Wait for shutdown signal or any task to fail
        shutdown_task = asyncio.create_task(self._wait_for_shutdown(), name="shutdown-watcher")
        try:
            done, pending = await asyncio.wait(
                [shutdown_task] + tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If a task failed (not cancelled), log it
            for t in done:
                if t.cancelled():
                    continue
                if t.exception():
                    logger.error("Task '%s' failed: %s",
                               t.get_name(), t.exception())

        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _run_bot(self):
        """Run the Telegram bot polling loop."""
        try:
            await self.bot.start()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Telegram bot crashed: %s", e, exc_info=True)
            # Keep running — scheduler can still work without Telegram

    async def _run_scheduler(self):
        """Run the lesson scheduler loop."""
        try:
            await self.scheduler.start()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Scheduler crashed: %s", e, exc_info=True)

    async def _wait_for_shutdown(self):
        """Block until shutdown event is set."""
        await self._shutdown_event.wait()

    def _start_web_ui(self, host, port):
        """Start the web UI in a background thread."""
        try:
            from web_ui import create_app
            import threading

            web_password = self.config.get("web_ui", {}).get("password")
            flask_app = create_app(
                config_path=str(CONFIG_PATH),
                log_file=str(LOG_FILE),
                password=web_password,
            )

            from werkzeug.serving import make_server
            self.web_server = make_server(host, port, flask_app, threaded=True)
            thread = threading.Thread(target=self.web_server.serve_forever, daemon=True)
            thread.start()
            logger.info("Web UI started on http://%s:%d", host, port)
        except ImportError:
            logger.error("Flask not installed — web UI unavailable (pip install flask)")
        except Exception as e:
            logger.error("Failed to start Web UI: %s", e)

    async def stop(self):
        """Gracefully shut down all services."""
        logger.info("Shutting down...")

        # Shut down web server
        if self.web_server:
            try:
                self.web_server.shutdown()
            except Exception as e:
                logger.warning("Web UI shutdown error: %s", e)

        # Cancel the bot task
        if self.bot:
            try:
                await self.bot.stop()
            except Exception as e:
                logger.warning("Bot shutdown error: %s", e)

        # Cancel the scheduler
        if self.scheduler:
            try:
                await self.scheduler.stop()
            except Exception as e:
                logger.warning("Scheduler shutdown error: %s", e)

        logger.info("Shutdown complete.")


# ── Signal handling ────────────────────────────────────────────────

def _handle_signal(daemon, signum, frame):
    """Signal handler for SIGINT/SIGTERM."""
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating shutdown...", sig_name)
    asyncio.get_event_loop().call_soon_threadsafe(
        daemon._shutdown_event.set
    )


# ── CLI entry point ───────────────────────────────────────────────

def main():
    """CLI to start the Lingua daemon.

    Usage:
        python3 src/main.py                        # default config
        python3 src/main.py --config config.json   # custom config
        python3 src/main.py --verbose              # debug logging
    """
    import argparse

    parser = argparse.ArgumentParser(description="LinguaDaily Daemon")
    parser.add_argument("--config", "-c", default=CONFIG_PATH,
                        help="Path to config.json")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--web-ui", action="store_true", default=False,
                        help="Start the admin web UI")
    parser.add_argument("--web-host", default="0.0.0.0",
                        help="Web UI bind address (default: 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=8089,
                        help="Web UI port (default: 8089)")
    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in config: {e}", file=sys.stderr)
        sys.exit(1)

    profiles = config.get("profiles", {})
    if not profiles:
        print("Warning: no profiles defined in config.json", file=sys.stderr)

    # Create daemon
    daemon = LinguaDaemon(config=config, config_path=args.config)
    daemon._setup_logging(verbose=args.verbose)

    logger.info("Lingua daemon starting up...")

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: _handle_signal(daemon, s, f))

    # Run
    try:
        asyncio.run(daemon.start(
            web_ui_host=args.web_host if args.web_ui else None,
            web_ui_port=args.web_port if args.web_ui else None,
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard")


if __name__ == "__main__":
    main()
