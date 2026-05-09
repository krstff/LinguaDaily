#!/usr/bin/env python3
"""
Lesson orchestrator for LinguaDaily standalone daemon.

Central pipeline controller that coordinates the full lesson flow:
  fetch → clean → TTS → translate → extract vocab → deliver

Usage (import):
    from src.orchestrator import Orchestrator, clean_content, get_profile, load_config
    orch = Orchestrator(config)
    lesson = orch.run_lesson(profile_name, delivery_callback=bot.deliver_lesson)

Usage (CLI — standalone run for one profile):
    python3 src/orchestrator.py --profile krystof
    python3 src/orchestrator.py --profile anna "quantum physics"
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Resolve paths ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")


# ── Config helpers ──────────────────────────────────────────────────

def load_config(path=None):
    """Load config from a JSON file. Defaults to config.json in project root."""
    target = path or CONFIG_PATH
    with open(target, encoding="utf-8") as f:
        return json.load(f)


def get_profile(config, profile_name=None):
    """
    Resolve a profile from the config.

    Priority:
      1. profile_name argument
      2. config['default_profile']
      3. first profile in config['profiles']

    Returns (profile_name, profile_dict) or raises ValueError.
    """
    if profile_name and profile_name in config["profiles"]:
        return profile_name, config["profiles"][profile_name]

    if profile_name:
        logger.warning("Profile '%s' not found, falling back to default.", profile_name)

    default = config.get("default_profile")
    if default and default in config["profiles"]:
        return default, config["profiles"][default]

    # Last resort: first profile
    profiles = config.get("profiles", {})
    if profiles:
        first = next(iter(profiles))
        return first, profiles[first]

    raise ValueError("No profiles defined in config.json")


# ── Content cleaning ────────────────────────────────────────────────

def clean_content(text):
    """Clean up extracted article text for display and TTS.

    Strips Wikipedia reference markers ([5], [ 1 ], etc.), removes wiki
    footer sections (Siehe auch, Literatur, Weblinks …), fixes missing
    spaces from link-adjacent words, and normalizes whitespace.
    """
    # ── Remove inline citation/reference markers like [5], [ 1 ] ──
    text = re.sub(r'\s*\[\s*\d+\s*\]\s*', '', text)

    # ── Strip Wikipedia footer sections and everything after them ──
    footer_headers = [
        'Siehe auch', 'Literatur', 'Weblinks', 'Quellen',
        'Einzelnachweise', 'Fußnoten', 'Anmerkungen',
        'Commons :',
        # English
        'See also', 'References', 'External links', 'Notes',
        'Bibliography', 'Further reading',
    ]
    for header in footer_headers:
        idx = text.find(header)
        if idx != -1:
            text = text[:idx]
            break

    # ── Fix missing spaces from adjacent wiki links ──
    # lower+Upper ("vergebenund" → "vergeben und")
    text = re.sub(r'([a-zäöüß])([A-ZÄÖÜ])', r'\1 \2', text)
    # punctuation without trailing space ("gestartet,das" → "gestartet, das")
    text = re.sub(r'([,.:;!?)\»])([A-Za-zÄÖÜäöü])', r'\1 \2', text)

    # ── Normalize whitespace ──
    # 1. Collapse 3+ newlines into double newline (paragraph boundary)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 2. Within each paragraph block, join lines broken by wiki HTML links.
    #    Only join if the preceding line ends mid-sentence (no terminal punctuation).
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    for para in paragraphs:
        lines = [l.strip() for l in para.split('\n') if l.strip()]
        if len(lines) <= 1:
            cleaned_paragraphs.append(para.strip())
            continue

        joined_lines = [lines[0]]
        for line in lines[1:]:
            prev = joined_lines[-1]
            # Join if previous line does NOT end with sentence-ending punctuation
            if not re.search(r'[.!?\"\)\»]$' , prev):
                joined_lines[-1] = prev + " " + line
            else:
                joined_lines.append(line)
        cleaned_paragraphs.append(" ".join(joined_lines))

    text = '\n\n'.join(cleaned_paragraphs)

    # 3. Collapse multiple spaces into single spaces.
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    return text


# ── Article fetching wrapper ────────────────────────────────────────

def fetch_article(source="wikipedia", topic=None, config=None,
                  content_lang=None, article_filter=None):
    """
    Fetch an article from the given content source via the router.

    Parameters
    ----------
    source : str
        Content source identifier ("wikipedia", "news").
    topic : str or None
        Topic to search/filter by.
    config : dict or None
        Full config.json contents. Loaded from disk if None.
    content_lang : str or None
        Language code for the desired content (used to pick the right
        Kiwix server when source is wikipedia).

    Returns
    -------
    (title, text) or (None, None) on failure.
    """
    if config is None:
        config = load_config()

    from fetch_router import fetch_article as route_fetch
    return route_fetch(source, topic, config, content_lang=content_lang,
                       article_filter=article_filter)


# ── Orchestrator class ──────────────────────────────────────────────

class Orchestrator:
    """
    Central lesson pipeline controller.

    Coordinates fetching, cleaning, TTS, translation, vocabulary extraction,
    and delivery for a single profile's lesson.

    Usage:
        orch = Orchestrator(config)
        lesson = orch.run_lesson("krystof", delivery_callback=bot.deliver_lesson)
    """

    def __init__(self, config=None, config_path=None):
        if config is None:
            config = load_config(config_path)
        self.config = config
        self._llama_client = None
        self._processor = None

    # ── Lazy dependencies ──────────────────────────────────────────

    def _get_llama_client(self, profile_name):
        """Get or create a LlamaClient for the given profile."""
        if (self._llama_client is None or
                self._llama_client.profile_name != profile_name):
            from llama_client import LlamaClient
            self._llama_client = LlamaClient(
                config=self.config,
                profile_name=profile_name,
            )
        return self._llama_client

    def _get_processor(self, profile_name):
        """Get or create a LinguaProcessor for the given profile."""
        if (self._processor is None or
                self._processor.profile != profile_name):
            from processor import LinguaProcessor
            target_lang_name = self.config.get("profiles", {}).get(
                profile_name, {}
            ).get("target_lang_name", "?")
            self._processor = LinguaProcessor(
                target_lang_name=target_lang_name,
                profile=profile_name,
            )
        return self._processor

    # ── Pipeline steps ─────────────────────────────────────────────

    def _fetch_and_clean(self, profile_name, topic=None):
        """Step 1+2: Fetch article and clean content."""
        profiles = self.config.get("profiles", {})
        profile = profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")

        source = profile.get("source", "wikipedia")
        content_lang = profile.get(
            "content_lang", profile.get("target_lang", "en")
        )
        article_filter = profile.get("article_filter")

        logger.info("[%s] Fetching %s article (topic: %s)...",
                    profile_name, source, topic or "random")

        title, content = fetch_article(
            source=source,
            topic=topic,
            config=self.config,
            content_lang=content_lang,
            article_filter=article_filter,
        )

        if not content:
            logger.warning("[%s] No article fetched — using fallback",
                          profile_name)
            title = f"Article about {topic or 'general topic'}"
            content = (f"A {source} article about {topic or 'a general topic'} "
                      "could not be retrieved from the local server.")

        # Clean content
        content = clean_content(content)
        word_count = len(content.split())
        logger.info("[%s] Fetched '%s' (%d words)", profile_name, title, word_count)

        return title, content, source, content_lang

    def _generate_tts(self, profile_name, content, content_lang):
        """Step 3: Generate TTS audio for the original content."""
        profile = self.config.get("profiles", {}).get(profile_name, {})
        use_tts = profile.get("use_tts", True)

        if not use_tts or not self.config.get("tts"):
            logger.info("[%s] TTS disabled — skipping", profile_name)
            return None

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
                return wav_path
        except Exception as e:
            logger.warning("[%s] TTS failed (lesson continues without audio): %s",
                          profile_name, e)

        return None

    def _translate(self, profile_name, content, source_lang, target_lang):
        """Step 4: Translate content via LLM."""
        profile = self.config.get("profiles", {}).get(profile_name, {})
        # source_lang here is the article's language (content_lang)
        # target_lang is the user's native language

        if not self.config.get("llm"):
            logger.info("[%s] LLM not configured — skipping translation",
                       profile_name)
            return content

        logger.info("[%s] Translating (%s → %s)...",
                    profile_name, source_lang, target_lang)
        try:
            client = self._get_llama_client(profile_name)
            translated = client.translate(
                text=content,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if translated:
                logger.info("[%s] Translation complete", profile_name)
                return translated
        except Exception as e:
            logger.warning("[%s] Translation failed (using original): %s",
                          profile_name, e)

        logger.warning("[%s] Falling back to original text", profile_name)
        return content

    def _extract_and_save_vocab(self, profile_name, original_content,
                                translated_content, source_lang, target_lang):
        """Step 5: Extract vocabulary via LLM and persist to markdown."""
        if not self.config.get("llm"):
            logger.info("[%s] LLM not configured — skipping vocab extraction",
                       profile_name)
            return []

        logger.info("[%s] Extracting vocabulary...", profile_name)
        try:
            client = self._get_llama_client(profile_name)
            # source_lang for vocab = language the user is learning (= target_lang of profile)
            # target_lang for vocab = user's native language (for definitions)
            vocab = client.extract_vocab(
                original_text=original_content,
                translated_text=translated_content,
                source_lang=target_lang,
                target_lang=source_lang,
                max_words=15,
            )

            if vocab:
                logger.info("[%s] Extracted %d vocabulary words",
                           profile_name, len(vocab))
                # Persist to markdown file
                processor = self._get_processor(profile_name)
                processor.update_vocab(vocab)
                logger.info("[%s] Vocabulary saved to %s",
                           profile_name, processor.vocab_path)
            return vocab
        except Exception as e:
            logger.warning("[%s] Vocab extraction failed: %s",
                          profile_name, e)

        return []

    def _build_lesson(self, profile_name, title, original_content,
                      translated_content, topic, source_lang, target_lang,
                      content_lang, wav_path, vocab):
        """Build the final lesson dict."""
        from datetime import datetime
        profile = self.config.get("profiles", {}).get(profile_name, {})
        return {
            "profile": profile_name,
            "title": title or "Language Lesson",
            "content": translated_content,
            "original_content": original_content,
            "topic": topic,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "target_lang_name": profile.get("target_lang_name", "?"),
            "content_lang": content_lang,
            "wav_path": wav_path,
            "vocab": vocab,
            "word_count": len(original_content.split()),
            "timestamp": datetime.now().isoformat(),
        }

    # ── Main pipeline ──────────────────────────────────────────────

    async def run_lesson(self, profile_name: str, topic: Optional[str] = None,
                         delivery_callback: Optional[Callable] = None) -> Optional[dict]:
        """
        Run the full lesson pipeline for one profile.

        Steps:
          1. Fetch article (random topic from profile if not specified)
          2. Clean content
          3. Generate TTS audio (if enabled)
          4. Translate via LLM
          5. Extract vocabulary via LLM and save to markdown
          6. Deliver via callback (if provided)

        Parameters
        ----------
        profile_name : str
            Profile to run the lesson for.
        topic : str or None
            Override topic. Picks random from profile topics if None.
        delivery_callback : callable or None
            Async callable(profile_name: str, lesson: dict) -> None.

        Returns
        -------
        dict or None
            Lesson payload ready for delivery, or None on failure.
        """
        logger.info("=" * 60)
        logger.info("LESSON PIPELINE — Profile: %s", profile_name)
        logger.info("=" * 60)

        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            logger.error("Profile '%s' not found — skipping", profile_name)
            return None

        profile = profiles[profile_name]

        # Resolve topic
        if topic is None and profile.get("topics"):
            topic = random.choice(profile["topics"])

        source_lang = profile.get("source_lang", "en")
        target_lang = profile.get("target_lang", "de")

        try:
            # Step 1+2: Fetch and clean
            title, content, source, content_lang = self._fetch_and_clean(
                profile_name, topic=topic)

            # Step 3: TTS
            wav_path = self._generate_tts(profile_name, content, content_lang)

            # Step 4: Translate
            translated = self._translate(
                profile_name, content, source_lang=content_lang,
                target_lang=source_lang)

            # Step 5: Extract and save vocab
            vocab = self._extract_and_save_vocab(
                profile_name, content, translated,
                source_lang=source_lang, target_lang=target_lang)

            # Build lesson dict
            lesson = self._build_lesson(
                profile_name, title, content, translated, topic,
                source_lang, target_lang, content_lang, wav_path, vocab)

            # Step 6: Deliver via callback
            if delivery_callback and lesson:
                try:
                    await delivery_callback(profile_name, lesson)
                    logger.info("[%s] Lesson delivered successfully",
                              profile_name)
                except Exception as e:
                    logger.error("[%s] Delivery failed: %s",
                               profile_name, e)

            return lesson

        except Exception as e:
            logger.error("[%s] Pipeline failed: %s", profile_name, e, exc_info=True)
            return None


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """
    Standalone orchestrator run — fetch content, translate, and deliver a lesson.

    Usage:
        python3 src/orchestrator.py                      # default profile, random topic
        python3 src/orchestrator.py --profile krystof    # specific profile
        python3 src/orchestrator.py --profile anna "quantum physics"  # profile + topic
    """
    parser = argparse.ArgumentParser(description="LinguaDaily orchestrator")
    parser.add_argument("--profile", "-p", help="User profile name")
    parser.add_argument("--config", "-c", default=CONFIG_PATH,
                        help="Path to config file")
    parser.add_argument("--tts-url", default=None,
                        help="Override TTS base_url")
    parser.add_argument("topic", nargs="?", default=None, help="Topic to search for")
    args = parser.parse_args()

    print("--- LinguaDaily: Task Execution Started ---")

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    if args.tts_url:
        config.setdefault("tts", {})["base_url"] = args.tts_url

    profile_name, profile = get_profile(config, args.profile)

    print(f"Profile: {profile_name}")
    print(f"Source: {profile.get('source', 'wikipedia')}")
    print(f"Content Language: {profile.get('content_lang', 'en')}")
    print(f"Target Topic: {args.topic or '(random from profile)'}")
    print(f"Languages: {profile['source_lang']} (native) → "
          f"{profile['target_lang_name']} (learning)")

    # Run the pipeline (async)
    orch = Orchestrator(config=config)

    async def _run():
        return await orch.run_lesson(profile_name, topic=args.topic)

    lesson = asyncio.run(_run())

    if lesson:
        print(f"\n✅ Lesson complete: {lesson['title']}")
        print(f"   Words: {lesson['word_count']}")
        print(f"   Vocab: {len(lesson.get('vocab', []))} words extracted")
        if lesson.get("wav_path"):
            print(f"   Audio: {lesson['wav_path']}")
    else:
        print("\n❌ Lesson pipeline failed — check logs for details.")

    print("--- Task Execution Complete ---")


if __name__ == "__main__":
    main()
