#!/usr/bin/env python3
"""
Lesson scheduler for LinguaDaily standalone daemon.

Schedules per-profile lesson runs using APScheduler with a **serial FIFO queue**.
Each profile with `schedule.time` / `schedule.tz` gets a daily cron trigger. When
a trigger fires the profile is pushed onto an internal queue, and a single
background worker processes lessons **one at a time**. This guarantees no overlap
even when multiple profiles share the same schedule time.

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

Usage (CLI):
    python3 src/scheduler.py --list        # show schedule
    python3 src/scheduler.py               # start daemon (blocks at cron times)
    python3 src/scheduler.py --run-now     # run all jobs now, then stay alive
    python3 src/scheduler.py --once        # run all jobs once and exit
"""

import asyncio
import json
import logging
import os
import sys
from typing import Callable, Optional

from config import CONFIG_PATH, load_config

logger = logging.getLogger(__name__)


class LessonScheduler:
    """
    Schedules daily lessons per profile via APScheduler with a serial queue.

    Each profile gets an independent cron trigger. When a trigger fires, the
    profile is pushed onto a FIFO queue. A single background worker pulls from
    that queue and runs lessons **one at a time**, guaranteeing no overlap even
    when multiple profiles share the same schedule time.
    """

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
        self._job_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    # ── Config helpers ─────────────────────────────────────────────

    @staticmethod
    def _load_config(path=None):
        """Load config from JSON file (delegates to shared loader)."""
        return load_config(path)

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
        """
        Build an async job function for one profile.

        The cron trigger only pushes the profile onto the internal queue;
        the actual lesson work is done by the single background worker.
        """
        async def job():
            logger.info("[%s] Scheduled trigger fired — enqueuing", profile_name)
            await self._job_queue.put((profile_name, profile))
        return job

    async def _worker(self):
        """
        Background worker that processes queued profiles one at a time.

        Pulls (profile_name, profile) tuples from the FIFO queue and runs the
        full lesson pipeline sequentially. This ensures that even when multiple
        cron triggers fire at the same time, lessons never overlap.
        """
        while True:
            profile_name, _profile = await self._job_queue.get()
            try:
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
            except Exception as e:
                logger.error("[%s] Worker error: %s", profile_name, e, exc_info=True)
            finally:
                self._job_queue.task_done()

    async def _enqueue_all_profiles(self):
        """
        Push all scheduled profiles onto the job queue immediately.

        Used by --run-now and --once to trigger lessons right away instead of
        waiting for the next cron fire.
        """
        scheduled = self.get_scheduled_profiles()
        if not scheduled:
            logger.warning("No profiles with schedules found — nothing to enqueue")
            return
        for profile_name, profile in scheduled:
            await self._job_queue.put((profile_name, profile))
            logger.info("[%s] Enqueued for immediate run", profile_name)
        logger.info("%d profile(s) enqueued for immediate execution", len(scheduled))

    async def start(self, immediate_run: bool = False):
        """
        Start the APScheduler and the background worker.

        Parameters
        ----------
        immediate_run : bool
            If True, push all scheduled profiles onto the queue right away so
            lessons run immediately instead of waiting for the next cron fire.
        """
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
            )

            logger.info("Scheduled '%s' at %s (%s)",
                       profile_name, time_str, tz)

        # Start the single background worker that processes lessons sequentially
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Background worker started — processing queue (serial)")

        self._scheduler.start()
        logger.info("Scheduler started — %d active job(s), 1 serial worker", len(scheduled))

        # Optional: run all jobs immediately on startup
        if immediate_run:
            await self._enqueue_all_profiles()

        try:
            # Block forever (or until shutdown event)
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.info("Scheduler task cancelled")
        finally:
            if self._scheduler and self._scheduler.running:
                self._scheduler.shutdown()

    async def stop(self):
        """Gracefully stop the scheduler and background worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    async def run_once(self):
        """
        Run all scheduled profiles once and shut down.

        Starts the worker, enqueues every scheduled profile, waits for the
        queue to drain, then stops everything cleanly. Useful for testing.
        """
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduled = self.get_scheduled_profiles()
        if not scheduled:
            logger.warning("No profiles with schedules found — nothing to run")
            return

        # Start the worker (no APScheduler cron needed for one-shot)
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Background worker started — one-shot mode")

        # Enqueue all profiles
        await self._enqueue_all_profiles()

        # Wait for every enqueued item to be processed
        logger.info("Waiting for queue to drain...")
        await self._job_queue.join()
        logger.info("Queue drained — all lessons complete")

        # Shut down
        await self.stop()

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
        python3 src/scheduler.py                          # start daemon (blocks)
        python3 src/scheduler.py --run-now                # run now, then stay alive
        python3 src/scheduler.py --once                   # run once and exit
    """
    import argparse

    parser = argparse.ArgumentParser(description="LinguaDaily Lesson Scheduler")
    parser.add_argument("--config", "-c", default=CONFIG_PATH,
                        help="Path to config.json")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List scheduled profiles and exit")
    parser.add_argument("--run-now", "-n", action="store_true",
                        help="Run all scheduled jobs immediately, then keep daemon alive")
    parser.add_argument("--once", "-o", action="store_true",
                        help="Run all scheduled jobs once and exit (for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config(args.config)

    if args.list:
        LessonScheduler.print_schedule(config=config)
        return

    # Wire up Telegram delivery callback when running jobs (--run-now / --once)
    delivery_callback = None
    tg_token = config.get("telegram", {}).get("bot_token") or os.environ.get(
        "TELEGRAM_BOT_TOKEN", ""
    )
    if tg_token and (args.run_now or args.once):
        try:
            from telegram_bot import TelegramBot
            bot = TelegramBot(config=config)
            delivery_callback = bot.deliver_lesson
            logger.info("Telegram delivery callback wired up")
        except Exception as e:
            logger.warning("Failed to init Telegram bot for delivery: %s", e)

    scheduler = LessonScheduler(
        config=config,
        delivery_callback=delivery_callback,
    )

    if args.once:
        # One-shot: run all profiles and exit
        asyncio.run(scheduler.run_once())
        return

    async def run():
        try:
            await scheduler.start(immediate_run=args.run_now)
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
