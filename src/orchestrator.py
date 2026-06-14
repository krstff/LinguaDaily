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
import logging
import os
import re
import sys
from typing import Callable, Optional

from config import (
    CONFIG_PATH,
    DEFAULT_LEARNING_LANGUAGE,
    DEFAULT_NATIVE_LANGUAGE,
    PROJECT_DIR,
    TTS_DEFAULT_VOICE,
    resolve_language_name,
    load_config,
)

logger = logging.getLogger(__name__)


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

    # ── Remove parenthetical content (birth dates, disambiguation notes, etc.) ──
    text = re.sub(r'\([^)]*\)', '', text)

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

        # Build paragraph: join wiki-link-broken lines with spaces,
        # keep sentence boundaries and section headers on separate lines.
        parts = []  # list of (text, break_after) tuples
        for i, line in enumerate(lines):
            if i == 0:
                parts.append((line, False))
            else:
                prev_text = parts[-1][0]
                prev_ends_sentence = bool(re.search(r'[.!?"\)\»]$' , prev_text))
                # Section header heuristic: first line of block, short (≤6 words),
                # doesn't look like a wiki link fragment (not "The", "A", etc.)
                is_header = (i == 1 and len(prev_text.split()) <= 6 and
                             not prev_text.lower().startswith(("the ", "a ", "an ", "(")) and
                             not prev_text.lower().endswith((" in", " on", " at", " of", " by")) and
                             (len(lines) > 1 and lines[1][0].isupper()))
                if prev_ends_sentence or is_header:
                    # Mark previous line to break after it, then start new content
                    parts[-1] = (prev_text, True)
                    parts.append((line, False))
                else:
                    # Join with space (wiki link fragment continuation)
                    parts[-1] = (prev_text + " " + line, False)
        # Now build the paragraph text from parts
        para_lines = []
        for idx, (txt, break_after) in enumerate(parts):
            if idx == 0:
                para_lines.append(txt)
            elif parts[idx - 1][1]:  # previous had break_after=True
                para_lines.append(txt)
            else:
                para_lines[-1] += " " + txt
        cleaned_paragraphs.append("\n".join(para_lines))

    text = '\n\n'.join(cleaned_paragraphs)

    # 3. Collapse multiple spaces into single spaces.
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    return text


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
            learning_language = self.config.get("profiles", {}).get(
                profile_name, {}
            ).get("learning_language", "?")
            self._processor = LinguaProcessor(
                learning_language=learning_language,
                profile=profile_name,
            )
        return self._processor

    # ── Pipeline steps ─────────────────────────────────────────────

    def _fetch_and_clean(self, profile_name):
        """Step 1+2: Fetch a random article and clean content."""
        profiles = self.config.get("profiles", {})
        profile = profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")

        source = profile.get("source", "wikipedia")
        learning_language = profile.get("learning_language", DEFAULT_LEARNING_LANGUAGE)
        article_filter = profile.get("article_filter")

        logger.info("[%s] Fetching random %s article (lang: %s)...",
                    profile_name, source, learning_language)

        from fetch_router import fetch_article as _fetch
        title, content = _fetch(
            source=source,
            topic=None,
            config=self.config,
            learning_language=learning_language,
            article_filter=article_filter,
        )

        if not content:
            logger.error("[%s] No article fetched from %s — aborting pipeline",
                         profile_name, source)
            return None

        # Clean content
        content = clean_content(content)
        word_count = len(content.split())

        # Re-enforce max_words after cleaning (cleaning can inflate word count
        # by splitting merged tokens from wiki links / adjacent capitalized words)
        max_words = article_filter.get("max_words") if article_filter else None
        if max_words and word_count > max_words:
            logger.info("[%s] Post-clean word count %d exceeds max %d — re-truncating",
                        profile_name, word_count, max_words)
            from wikipedia_fetcher import smart_truncate, hard_truncate
            min_words = article_filter.get("min_words", 50) if article_filter else 50
            content = smart_truncate(content, max_words=max_words, min_words=min_words) or \
                      hard_truncate(content, max_words=max_words)
            word_count = len(content.split())

        logger.info("[%s] Fetched '%s' (%d words)", profile_name, title, word_count)

        return title, content, source, learning_language

    def _generate_tts(self, profile_name, content, learning_language):
        """Step 3: Generate TTS audio for the original content."""
        profile = self.config.get("profiles", {}).get(profile_name, {})
        use_tts = profile.get("use_tts", True)

        if not use_tts or not self.config.get("tts"):
            logger.info("[%s] TTS disabled — skipping", profile_name)
            return None

        logger.info("[%s] Generating TTS (lang: %s)...",
                    profile_name, learning_language)
        try:
            from tts import synthesize
            output_dir = os.path.join(PROJECT_DIR, "output", profile_name)
            os.makedirs(output_dir, exist_ok=True)

            wav_path = synthesize(
                text=content,
                language_id=learning_language,
                config=self.config,
                output_dir=output_dir,
                voice=profile.get("tts_voice", TTS_DEFAULT_VOICE),
            )
            if wav_path:
                logger.info("[%s] TTS audio: %s", profile_name, wav_path)
                return wav_path
        except Exception as e:
            logger.warning("[%s] TTS failed (lesson continues without audio): %s",
                          profile_name, e)

        return None

    def _translate(self, profile_name, content,
                    learning_language, native_language):
        """Step 4: Translate content via LLM.

        Translates the article from `learning_language` (what the user is
        studying) into `native_language` (what the user already understands).
        """
        profile = self.config.get("profiles", {}).get(profile_name, {})

        if not self.config.get("llm"):
            logger.info("[%s] LLM not configured — skipping translation",
                       profile_name)
            return content

        word_count = len(content.split())
        model = self._get_llama_client(profile_name).resolve_model("translate")
        logger.info("[%s] Translating (%s → %s, %d words, model: %s)...",
                    profile_name, learning_language, native_language,
                    word_count, model)
        try:
            client = self._get_llama_client(profile_name)
            translated = client.translate(
                text=content,
                source_lang=learning_language,
                target_lang=native_language,
            )
            if translated:
                logger.info("[%s] Translation complete (%d words output)",
                           profile_name, len(translated.split()))
                return translated
        except Exception as e:
            error_msg = str(e)
            if any(kw in error_msg.lower() for kw in ("connection", "refused", "timeout", "unreachable", "network")):
                logger.warning("[%s] Translation failed: LLM unreachable", profile_name)
            else:
                logger.warning("[%s] Translation failed: %s", profile_name, error_msg[:100])

        logger.warning("[%s] Falling back to original text", profile_name)
        return content

    def _extract_and_save_vocab(self, profile_name, original_content,
                                translated_content,
                                learning_language, native_language):
        """Step 5: Extract vocabulary via LLM and persist to markdown.

        Vocab is extracted from the article in `learning_language` with
        definitions provided in `native_language`.
        """
        if not self.config.get("llm"):
            logger.info("[%s] LLM not configured — skipping vocab extraction",
                       profile_name)
            return []

        orig_words = len(original_content.split())
        trans_words = len(translated_content.split()) if translated_content else 0
        model = self._get_llama_client(profile_name).resolve_model("vocab")
        logger.info("[%s] Extracting vocabulary (orig: %d words, trans: %d words, model: %s)...",
                    profile_name, orig_words, trans_words, model)
        try:
            client = self._get_llama_client(profile_name)
            # source_lang for vocab = language the user is learning
            # target_lang for vocab = user's native language (for definitions)
            vocab = client.extract_vocab(
                original_text=original_content,
                translated_text=translated_content,
                source_lang=learning_language,
                target_lang=native_language,
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
            error_msg = str(e)
            if any(kw in error_msg.lower() for kw in ("connection", "refused", "timeout", "unreachable", "network")):
                logger.warning("[%s] Vocab extraction failed: LLM unreachable", profile_name)
            else:
                logger.warning("[%s] Vocab extraction failed: %s", profile_name, error_msg[:100])

        return []

    def _build_lesson(self, profile_name, title, original_content,
                      translated_content,
                      learning_language, native_language,
                      wav_path, vocab):
        """Build the final lesson dict."""
        from datetime import datetime
        return {
            "profile": profile_name,
            "title": title or "Language Lesson",
            "content": translated_content,
            "original_content": original_content,
            "learning_language": learning_language,
            "learning_language_name": resolve_language_name(learning_language),
            "native_language": native_language,
            "wav_path": wav_path,
            "vocab": vocab,
            "word_count": len(original_content.split()),
            "timestamp": datetime.now().isoformat(),
        }

    # ── Main pipeline ──────────────────────────────────────────────

    async def _generate_tts_async(self, profile_name, content,
                                   learning_language):
        """Async wrapper for TTS generation (runs in executor thread)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_tts, profile_name, content,
            learning_language)

    async def _translate_async(self, profile_name, content,
                               learning_language, native_language):
        """Async wrapper for translation (runs in executor thread)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._translate, profile_name, content,
            learning_language, native_language)

    async def run_lesson(self, profile_name: str,
                         delivery_callback: Optional[Callable] = None) -> Optional[dict]:
        """
        Run the full lesson pipeline for one profile.

        Steps:
          1. Fetch a random article
          2. Clean content
          3+4. Generate TTS audio AND Translate via LLM (in parallel)
          5. Extract vocabulary via LLM and save to markdown
          6. Deliver via callback (if provided)

        Parameters
        ----------
        profile_name : str
            Profile to run the lesson for.
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

        native_language = profile.get("native_language", DEFAULT_NATIVE_LANGUAGE)
        learning_language = profile.get("learning_language", DEFAULT_LEARNING_LANGUAGE)

        try:
            # Step 1+2: Fetch and clean
            result = self._fetch_and_clean(profile_name)
            if result is None:
                # Article fetch failed — abort pipeline, skip all LLM calls
                logger.warning("[%s] Pipeline aborted — no article available", profile_name)
                return None
            title, content, source, fetch_lang = result

            # Step 3+4: TTS and Translate in parallel (both need only original content)
            logger.info("[%s] Running TTS + Translation in parallel...", profile_name)
            wav_path, translated = await asyncio.gather(
                self._generate_tts_async(profile_name, content,
                                         learning_language),
                self._translate_async(profile_name, content,
                                      learning_language=learning_language,
                                      native_language=native_language),
            )

            # Step 5: Extract and save vocab (depends on translated text)
            vocab = self._extract_and_save_vocab(
                profile_name, content, translated,
                learning_language=learning_language,
                native_language=native_language)

            # Build lesson dict
            lesson = self._build_lesson(
                profile_name, title, content, translated,
                learning_language, native_language, wav_path, vocab)

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
    Standalone orchestrator run — fetch a random article, translate, and deliver.

    Usage:
        python3 src/orchestrator.py                      # default profile
        python3 src/orchestrator.py --profile krystof    # specific profile
    """
    parser = argparse.ArgumentParser(description="LinguaDaily orchestrator")
    parser.add_argument("--profile", "-p", help="User profile name")
    parser.add_argument("--config", "-c", default=CONFIG_PATH,
                        help="Path to config file")
    parser.add_argument("--tts-url", default=None,
                        help="Override TTS base_url")
    args = parser.parse_args()

    # Configure logging for CLI runs
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

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
    learning_language = profile.get('learning_language', '?')
    print(f"Learning language: {resolve_language_name(learning_language)} ({learning_language})")
    print(f"Native language:   {profile.get('native_language', '?')}")

    # Show LLM config
    llm_cfg = config.get("llm", {})
    if llm_cfg:
        orch_temp = Orchestrator(config=config)
        client_temp = orch_temp._get_llama_client(profile_name)
        print(f"LLM: {client_temp.default_model} @ {client_temp.base_url}")
        print(f"   translate model: {client_temp.resolve_model('translate')}")
        print(f"   vocab model:     {client_temp.resolve_model('vocab')}")
    else:
        print("LLM: not configured")

    # Run the pipeline (async)
    orch = Orchestrator(config=config)

    # Wire up Telegram delivery callback
    from telegram_bot import TelegramBot
    tg_bot = TelegramBot(config=config, profile_name=profile_name)

    async def _run():
        try:
            return await orch.run_lesson(
                profile_name,
                delivery_callback=tg_bot.deliver_lesson,
            )
        finally:
            await tg_bot.stop()  # close the aiohttp session

    lesson = asyncio.run(_run())

    if lesson:
        print(f"\n✅ Lesson complete: {lesson['title']}")
        print(f"   Words: {lesson['word_count']}")
        print(f"   Vocab: {len(lesson.get('vocab', []))} words extracted")
        if lesson.get("wav_path"):
            print(f"   Audio: {lesson['wav_path']}")

        # Show original text snippet
        original = lesson.get("original_content", "")[:300]
        print(f"\n📄 Original ({lesson.get('learning_language_name', '?')}):")
        print(f"   {original}{'...' if len(original) == 300 else ''}")

        # Show translated text snippet
        translated = lesson.get("content", "")[:300]
        print(f"\n🌐 Translated ({lesson.get('native_language', '?')}):")
        print(f"   {translated}{'...' if len(translated) == 300 else ''}")
    else:
        print("\n❌ Lesson pipeline failed — check logs for details.")

    print("--- Task Execution Complete ---")


if __name__ == "__main__":
    main()
