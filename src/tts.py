#!/usr/bin/env python3
"""
TTS wrapper for local OmniVoice server.

Usage (import):
    from src.tts import synthesize
    wav_path = synthesize(text, language_id="de", config=cfg)

Usage (CLI):
    python3 src/tts.py --config config.json --lang de "Hallo Welt"
"""

import json
import logging
import os
import re
import sys
import uuid

from config import (
    DEFAULT_LEARNING_LANGUAGE,
    OUTPUT_DIR,
    TTS_DEFAULT_MAX_AGE_DAYS,
    TTS_DEFAULT_MAX_FILES,
    TTS_DEFAULT_MODEL,
    TTS_DEFAULT_NUM_STEP,
    TTS_DEFAULT_VOICE,
    load_config,
)

logger = logging.getLogger(__name__)


def sanitize_for_tts(text):
    """
    Clean text extracted from Wikipedia/Kiwix for TTS consumption.

    Wikipedia articles contain reference markers, special symbols,
    and encoding artifacts that make TTS output sound garbled.
    This function strips or normalises those artifacts while preserving
    the article's prose content.

    Returns cleaned text suitable for speech synthesis.
    """
    if not text:
        return text

    # 1. Remove inline reference markers: [1], [2], [a], [b], etc.
    text = re.sub(r'\[[\da-zA-Z]+\]', '', text)

    # 2. Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')

    # 3. Remove common Wikipedia artifacts that TTS can't pronounce:
    #    - Up/down arrows (↑ ↓) from edit links / section anchors
    #    - Other Unicode symbols that make no sense spoken aloud
    text = re.sub(r'[\u2190-\u2195\u21A8-\u21B5]', '', text)

    # 4. Collapse multiple blank lines into at most two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. Strip leading/trailing whitespace
    text = text.strip()

    return text


def _get_client(config):
    """Get the shared OpenAI-compatible client for TTS.

    Uses the module-level singleton from config so all callers share
    ONE connection pool to llama.cpp. Returns None if openai is not installed.
    """
    from config import get_openai_client

    tts_cfg = config.get("tts", {})
    base_url = tts_cfg.get("base_url")
    api_key = tts_cfg.get("api_key", "")
    return get_openai_client(base_url=base_url, api_key=api_key or "none")


def _cleanup_old_files(
    output_dir,
    max_age_days=TTS_DEFAULT_MAX_AGE_DAYS,
    max_files=TTS_DEFAULT_MAX_FILES,
):
    """Remove old TTS WAV files from a profile output directory.

    Keeps the most recent `max_files` WAVs and deletes anything older than
    `max_age_days`. Both conditions are checked — a file is removed if it
    exceeds the age limit OR the directory has more than max_files total.

    Parameters
    ----------
    output_dir : str
        Directory containing WAV files to clean up.
    max_age_days : int
        Delete files older than this many days (0 = disable age limit).
    max_files : int
        Keep at most this many recent files (-1 = disable count limit).
    """
    if not os.path.isdir(output_dir):
        return

    import time
    now = time.time()

    wav_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(".wav")
    ]

    if not wav_files:
        return

    # Sort by modification time, newest first
    wav_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )

    removed = 0
    for idx, wav in enumerate(wav_files):
        filepath = os.path.join(output_dir, wav)
        age_days = (now - os.path.getmtime(filepath)) / 86400
        should_remove = False

        # Age limit: remove files older than max_age_days
        if max_age_days > 0 and age_days > max_age_days:
            should_remove = True

        # Count limit: keep only the most recent max_files (list sorted newest-first)
        if max_files > 0 and idx >= max_files:
            should_remove = True

        if should_remove:
            try:
                os.remove(filepath)
                removed += 1
            except OSError as e:
                logger.warning("TTS cleanup: failed to remove %s: %s", wav, e)

    if removed:
        logger.info("TTS cleanup: removed %d old file(s) from %s",
                    removed, output_dir)


def synthesize(
    text,
    language_id=DEFAULT_LEARNING_LANGUAGE,
    config=None,
    output_dir=None,
    voice=None,
):
    """
    Generate speech from text using the local OmniVoice server.

    Parameters
    ----------
    text : str
        Text to synthesize (should be in the target/content language).
    language_id : str
        ISO language code for TTS (e.g. "de", "en", "cs").
    config : dict or None
        Full config.json contents. Loaded from default path if None.
    output_dir : str or None
        Directory to write the WAV file. Defaults to project/output/.
    voice : str or None
        Voice name (e.g. "male", "female"). Falls back to config tts.default_voice.

    Returns
    -------
    str or None
        Absolute path to the generated WAV file, or None on failure.
    """
    if not text or not text.strip():
        return None

    # Clean Wikipedia artifacts before sending to TTS
    text = sanitize_for_tts(text)
    if not text.strip():
        return None

    if config is None:
        try:
            config = load_config()
        except Exception as e:
            logger.error("Error loading config for TTS: %s", e)
            return None

    if output_dir is None:
        # Per-profile output directory
        profile = config.get("default_profile", "default")
        output_dir = os.path.join(OUTPUT_DIR, profile)

    os.makedirs(output_dir, exist_ok=True)

    tts_cfg = config.get("tts", {})
    model = tts_cfg.get("model", TTS_DEFAULT_MODEL)
    if voice is None:
        voice = tts_cfg.get("default_voice", TTS_DEFAULT_VOICE)

    # Read cleanup settings from config (tts.max_age_days, tts.max_files)
    max_age_days = int(tts_cfg.get("max_age_days", TTS_DEFAULT_MAX_AGE_DAYS))
    max_files = int(tts_cfg.get("max_files", TTS_DEFAULT_MAX_FILES))

    filename = f"lingua_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(output_dir, filename)

    try:
        client = _get_client(config)
        if client is None:
            return None
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            extra_body={"language_id": language_id, "num_step": TTS_DEFAULT_NUM_STEP},
        ) as response:
            response.stream_to_file(filepath)

        # Verify the file was actually written and is non-empty
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            # Clean up old files after successful generation
            _cleanup_old_files(output_dir, max_age_days=max_age_days,
                               max_files=max_files)
            return filepath
        else:
            logger.warning("TTS: generated file is empty or missing")
            return None
    except Exception as e:
        logger.error("TTS error: %s", e)
        # Clean up partial file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OmniVoice TTS wrapper")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--lang", "-l", default="de", help="Language code for TTS")
    parser.add_argument("--config", "-c", default=None, help="Path to config.json")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--voice", "-v", default=None, help="Voice name (e.g. male, female)")
    parser.add_argument("--tts-url", default=None, help="Override TTS base_url (e.g. http://192.168.100.60:8080/v1)")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    if args.tts_url:
        config.setdefault("tts", {})["base_url"] = args.tts_url
    wav_path = synthesize(args.text, language_id=args.lang, config=config, output_dir=args.output_dir, voice=args.voice)

    if wav_path:
        print(json.dumps({"wav_path": wav_path}))
    else:
        print(json.dumps({"error": "TTS generation failed"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
