#!/usr/bin/env python3
"""
Lesson scheduler for LinguaDaily standalone daemon.

Schedules per-profile lesson runs using APScheduler. Each profile with a
`schedule.time` / `schedule.tz` gets a daily cron job that fires the full
lesson pipeline (fetch → clean → TTS → translate) and delivers the result
via a pluggable callback (e.g., TelegramBot.deliver_lesson).

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
import random
import sys
from datetime import datetime
from typing import Callable, Optional

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")

logger = logging.getLogger(__name__)


class LessonScheduler:
    """Schedules and runs daily lessons per profile."""

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

    # ── Lesson pipeline ────────────────────────────────────────────

    async def run_lesson(self, profile_name: str) -> Optional[dict]:
        """
        Run the full lesson pipeline for one profile.

        Steps:
          1. Fetch an article (random topic from profile)
          2. Clean content (strip wiki artifacts)
          3. Generate TTS audio (if enabled)
          4. Translate via LLM
          5. Extract vocabulary via LLM
          6. Return lesson dict

        Parameters
        ----------
        profile_name : str
            Profile to run the lesson for.

        Returns
        -------
        dict or None
            Lesson payload ready for delivery, or None on failure.
        """
        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            logger.error("Profile '%s' not found — skipping", profile_name)
            return None

        profile = profiles[profile_name]

        # ── Step 1: Pick topic and fetch article ───────────────────
        topics = profile.get("topics", [])
        if topics:
            topic = random.choice(topics)
        else:
            topic = None

        source = profile.get("source", "wikipedia")
        content_lang = profile.get(
            "content_lang", profile.get("target_lang", "en")
        )
        article_filter = profile.get("article_filter")

        logger.info("[%s] Fetching %s article (topic: %s)...",
                    profile_name, source, topic or "random")

        title = None
        content = None

        try:
            from fetch_router import fetch_article as route_fetch
            title, content = route_fetch(
                source=source,
                topic=topic,
                config=self.config,
                content_lang=content_lang,
                article_filter=article_filter,
            )
        except Exception as e:
            logger.error("[%s] Article fetch failed: %s", profile_name, e)

        if not content:
            logger.warning("[%s] No article fetched — using fallback",
                          profile_name)
            title = f"Article about {topic or 'general topic'}"
            content = (f"A {source} article about {topic or 'a general topic'} "
                      "could not be retrieved from the local server.")

        word_count = len(content.split())
        logger.info("[%s] Fetched '%s' (%d words)", profile_name, title, word_count)

        # ── Step 2: Clean content ──────────────────────────────────
        try:
            from orchestrator import clean_content
            content = clean_content(content)
        except Exception as e:
            logger.warning("[%s] Content cleaning failed: %s", profile_name, e)

        # ── Step 3: Generate TTS audio ─────────────────────────────
        wav_path = None
        use_tts = profile.get("use_tts", True)
        if use_tts and self.config.get("tts"):
            logger.info("[%s] Generating TTS (lang: %s)...",
                       profile_name, content_lang)
            try:
                from tts import synthesize
                output_dir = os.path.join(PROJECT_DIR, "output", profile_name)
                os.makedirs(output_dir, exist_ok=True)

                wav_path = synthesize(
                    text=content,
                    language_id=content_lang,
                    config=self.config,
                    output_dir=output_dir,
                    voice=profile.get("tts_voice", "male"),
                )
                if wav_path:
                    logger.info("[%s] TTS audio: %s", profile_name, wav_path)
            except Exception as e:
                logger.warning("[%s] TTS failed (lesson continues without audio): %s",
                              profile_name, e)
        elif not use_tts:
            logger.info("[%s] TTS disabled — skipping", profile_name)

        # ── Step 4: Translate via LLM ──────────────────────────────
        translated = content  # fallback: original text
        source_lang = profile.get("source_lang", "en")
        target_lang = profile.get("target_lang", "de")
        target_lang_name = profile.get("target_lang_name", "?")

        if self.config.get("llm"):
            logger.info("[%s] Translating (%s → %s)...",
                       profile_name, source_lang, target_lang)
            try:
                from llama_client import LlamaClient
                client = LlamaClient(
                    config=self.config, profile_name=profile_name
                )
                # Translate from content_lang (article language) to target_lang
                translated = client.translate(
                    text=content,
                    source_lang=content_lang,
                    target_lang=target_lang,
                )
                if translated:
                    logger.info("[%s] Translation complete", profile_name)
                else:
                    logger.warning("[%s] LLM translation returned empty — "
                                  "using original text", profile_name)
            except Exception as e:
                logger.warning("[%s] Translation failed (using original): %s",
                              profile_name, e)

        # ── Step 5: Extract vocabulary ─────────────────────────────
        vocab = []
        if self.config.get("llm"):
            try:
                from llama_client import LlamaClient
                client = LlamaClient(
                    config=self.config, profile_name=profile_name
                )
                vocab = client.extract_vocab(
                    original_text=content,
                    translated_text=translated,
                    source_lang=target_lang,
                    target_lang=source_lang,
                    max_words=15,
                )
                if vocab:
                    logger.info("[%s] Extracted %d vocabulary words",
                               profile_name, len(vocab))
            except Exception as e:
                logger.warning("[%s] Vocab extraction failed: %s",
                              profile_name, e)

        # ── Build lesson dict ──────────────────────────────────────
        lesson = {
            "profile": profile_name,
            "title": title or "Language Lesson",
            "content": translated,
            "original_content": content,
            "topic": topic,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "target_lang_name": target_lang_name,
            "content_lang": content_lang,
            "wav_path": wav_path,
            "vocab": vocab,
            "word_count": word_count,
            "timestamp": datetime.now().isoformat(),
        }

        return lesson

    # ── APScheduler integration ────────────────────────────────────

    def _build_job(self, profile_name: str, profile: dict):
        """Build an async job function for one profile."""
        async def job():
            logger.info("=" * 60)
            logger.info("SCHEDULED LESSON — Profile: %s", profile_name)
            logger.info("=" * 60)

            lesson = await self.run_lesson(profile_name)

            if lesson and self.delivery_callback:
                try:
                    await self.delivery_callback(profile_name, lesson)
                    logger.info("[%s] Lesson delivered successfully",
                              profile_name)
                except Exception as e:
                    logger.error("[%s] Delivery failed: %s",
                               profile_name, e)
            elif lesson:
                logger.info("[%s] Lesson prepared (no delivery callback)",
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
