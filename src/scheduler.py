#!/usr/bin/env python3
"""
Lesson scheduler for LinguaDaily standalone daemon.

Schedules per-profile lesson runs using APScheduler. Each profile with a
`schedule.time` / `schedule.tz` gets a daily cron job that fires the full
lesson pipeline via Orchestrator.run_lesson() and delivers the result via
a pluggable callback (e.g., TelegramBot.deliver_lesson).

Config shape:
    {
      "profiles": {
        "krystof": {
          "schedule": {
            "time": "08:00",
            "tz": "Europe/Berlin"
          },
          ...
        }
      }
    }

Usage (import):
    from src.scheduler import LessonScheduler
    scheduler = LessonScheduler(config, delivery_callback=bot.deliver_lesson)
    await scheduler.start()

Usage (CLI — dry-run listing):
    python3 src/scheduler.py --config config.json --list
"""

import asyncio
import json
import logging
import os
import sys
from typing import Callable, Optional

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")

logger = logging.getLogger(__name__)


class LessonScheduler:
    """Schedules daily lessons per profile via APScheduler."""

    def __init__(
        self,
        config=None,
        delivery_callback=None,
        config_path=None,
    ):
        """
        Parameters
        ----------
        config : dict or None
            Full config.json contents. Loaded from default path if None.
        delivery_callback : callable or None
            Async callable(profile_name: str, lesson: dict) -> None.
            Called after each lesson pipeline completes. If None, lessons
            are logged but not delivered anywhere.
        config_path : str or None
            Override path to config.json.
        """
        if config is None:
            config = self._load_config(config_path)

        self.config = config
        self.delivery_callback = delivery_callback
        self._scheduler = None

    # ── Config helpers ─────────────────────────────────────────────

    @staticmethod
    def _load_config(path=None):
        """Load config from JSON file."""
        target = path or CONFIG_PATH
        with open(target, encoding="utf-8") as f:
            return json.load(f)

    def get_scheduled_profiles(self) -> list[tuple[str, dict]]:
        """
        Return list of (profile_name, profile_config) for all profiles
        that have a schedule configured.
        """
        results = []
        for name, profile in self.config.get("profiles", {}).items():
            schedule = profile.get("schedule")
            if schedule and schedule.get("time"):
                results.append((name, profile))
        return results

    # ── APScheduler integration ────────────────────────────────────

    def _build_job(self, profile_name: str, profile: dict):
        """Build an async job function for one profile."""
        async def job():
            logger.info("=" * 60)
            logger.info("SCHEDULED LESSON — Profile: %s", profile_name)
            logger.info("=" * 60)

            # Delegate full pipeline to Orchestrator
            from orchestrator import Orchestrator
            orch = Orchestrator(config=self.config)

            lesson = await orch.run_lesson(
                profile_name,
                delivery_callback=self.delivery_callback,
            )

            if lesson:
                logger.info("[%s] Lesson prepared: '%s' (%d words, %d vocab)",
                           profile_name,
                           lesson.get("title", "?"),
                           lesson.get("word_count", 0),
                           len(lesson.get("vocab", [])),
                           )
            else:
                logger.error("[%s] Lesson pipeline returned no result",
                            profile_name)

        return job

    async def start(self):
        """Start the APScheduler (blocks until cancelled)."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduled = self.get_scheduled_profiles()
        if not scheduled:
            logger.warning("No profiles with schedules found — scheduler idle")
            return

        self._scheduler = AsyncIOScheduler(timezone="UTC")

        for profile_name, profile in scheduled:
            schedule = profile["schedule"]
            time_str = schedule["time"]  # "HH:MM"
            tz = schedule.get("tz", "UTC")

            hours, minutes = time_str.split(":")

            trigger = CronTrigger(
                hour=hours,
                minute=minutes,
                timezone=tz,
            )

            job = self._build_job(profile_name, profile)
            self._scheduler.add_job(
                job,
                trigger=trigger,
                id=f"lesson_{profile_name}",
                name=f"Lesson for {profile_name}",
                replace_existing=True,
                max_instances=1,  # don't overlap runs
            )

            logger.info("Scheduled '%s' at %s (%s)",
                       profile_name, time_str, tz)

        self._scheduler.start()
        logger.info("Scheduler started — %d active job(s)", len(scheduled))

        try:
            # Block forever (or until shutdown event)
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.info("Scheduler task cancelled")
        finally:
            self._scheduler.shutdown()

    async def stop(self):
        """Gracefully stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)

    # ── CLI helpers ────────────────────────────────────────────────

    @staticmethod
    def print_schedule(config=None, config_path=None):
        """Print all scheduled profiles (for --list mode)."""
        if config is None:
            config = LessonScheduler._load_config(config_path)

        scheduled = LessonScheduler(config).get_scheduled_profiles()

        if not scheduled:
            print("No profiles with schedules configured.")
            return

        print(f"{'Profile':<20} {'Time':>8} {'Timezone':<25} {'Language'}")
        print("-" * 70)
        for name, profile in scheduled:
            sched = profile["schedule"]
            time_str = sched.get("time", "?")
            tz = sched.get("tz", "UTC")
            lang = profile.get("target_lang_name", "?")
            print(f"{name:<20} {time_str:>8} {tz:<25} {lang}")


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """CLI for listing/running the scheduler.

    Usage:
        python3 src/scheduler.py --list                  # show schedule
        python3 src/scheduler.py --config config.json     # start (blocks)
    """
    import argparse

    parser = argparse.ArgumentParser(description="LinguaDaily Lesson Scheduler")
    parser.add_argument("--config", "-c", default=CONFIG_PATH,
                        help="Path to config.json")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List scheduled profiles and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    if args.list:
        LessonScheduler.print_schedule(config=config)
        return

    scheduler = LessonScheduler(config=config)

    async def run():
        try:
            await scheduler.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await scheduler.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
