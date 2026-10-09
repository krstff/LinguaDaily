#!/usr/bin/env python3
"""
Speech-to-text client for LinguaDaily.

Talks to any OpenAI-compatible /v1/audio/transcriptions endpoint —
e.g. a local whisper.cpp `whisper-server` deployment (fits the
self-hosted setup) or a hosted API.  This module never downloads or
bundles models; the endpoint connection is configured by the operator.

Config (config.json):
    {
      "stt": {
        "base_url": "http://localhost:9002/v1",
        "api_key": "",
        "model": "whisper",
        "timeout": 60
      }
    }

Usage (import):
    from src.stt import is_configured, transcribe, voice_to_wav
    if is_configured(config):
        wav = voice_to_wav("/path/to/voice.ogg")
        text = transcribe(wav, language="de", config=config)

Usage (CLI):
    python3 src/stt.py --config config.json --lang de /path/to/audio.wav
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# 16 kHz mono WAV is the sweet spot for most STT backends
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TIMEOUT = 60


def get_stt_config(config: dict) -> dict:
    """Resolve STT settings from a config dict.

    Returns {base_url, api_key, model, timeout}.
    """
    stt = config.get("stt", {}) or {}
    return {
        "base_url": stt.get("base_url", "") or "",
        "api_key": stt.get("api_key", "") or "none",
        "model": stt.get("model", "") or "",
        "timeout": float(stt.get("timeout", DEFAULT_TIMEOUT)),
    }


def is_configured(config: dict) -> bool:
    """True when an STT endpoint (base_url + model) is configured."""
    cfg = get_stt_config(config)
    return bool(cfg["base_url"] and cfg["model"])


def voice_to_wav(src_path: str, dst_dir: Optional[str] = None,
                 sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[str]:
    """Convert any ffmpeg-readable audio (e.g. Telegram OGG/Opus voice
    notes) to a mono WAV file for STT.

    Returns the WAV path, or None on failure (ffmpeg missing / decode error).
    """
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found — cannot convert audio for STT "
                     "(install ffmpeg on the daemon host)")
        return None

    if not os.path.isfile(src_path):
        return None

    dst_dir = dst_dir or os.path.dirname(src_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path) + ".wav")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src_path,
        "-ar", str(sample_rate), "-ac", "1",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
            return dst_path
        logger.error("ffmpeg produced no output for %s", src_path)
        return None
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("ffmpeg conversion failed for %s: %s", src_path, e)
        return None


def transcribe(audio_path: str, language: Optional[str] = None,
               config: Optional[dict] = None,
               timeout: Optional[float] = None) -> Optional[str]:
    """Transcribe an audio file via the configured STT endpoint.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (WAV recommended).
    language : str or None
        ISO 639-1 code of the spoken language (e.g. "de"). Forces the
        recognizer language when provided.
    config : dict or None
        Full config.json contents (loaded from default path if None).
    timeout : float or None
        Override the configured request timeout.

    Returns
    -------
    str or None
        Transcribed text (may be empty string), or None on failure.
    """
    from config import get_openai_client, load_config

    if config is None:
        config = load_config(fallback={})

    stt = get_stt_config(config)
    if not stt["base_url"] or not stt["model"]:
        logger.error("STT not configured (stt.base_url / stt.model missing)")
        return None

    if not audio_path or not os.path.isfile(audio_path):
        logger.error("STT: audio file not found: %s", audio_path)
        return None

    client = get_openai_client(
        base_url=stt["base_url"],
        api_key=stt["api_key"],
        timeout=timeout or stt["timeout"],
    )
    if client is None:
        logger.error("STT: openai client unavailable")
        return None

    try:
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=stt["model"],
                file=f,
                language=language or None,
            )
        text = (getattr(result, "text", "") or "").strip()
        logger.info("STT transcribed %s (%d chars, lang=%s)",
                    os.path.basename(audio_path), len(text), language or "auto")
        return text
    except Exception as e:
        error_msg = str(e)
        if any(kw in error_msg.lower() for kw in
               ("connection", "refused", "timeout", "unreachable", "network")):
            logger.error("STT unreachable (%s)", error_msg[:80])
        else:
            logger.error("STT transcription failed: %s", e)
        return None


def main():
    """CLI for testing the STT endpoint.

    Usage:
        python3 src/stt.py --config config.json --lang de /path/to/audio.wav
    """
    import argparse

    parser = argparse.ArgumentParser(description="LinguaDaily STT test")
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument("--lang", default=None, help="ISO language code (e.g. de)")
    parser.add_argument("--config", "-c", default=None, help="Path to config.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(message)s")

    from config import load_config
    config = load_config(args.config) if args.config else load_config(fallback={})

    if not is_configured(config):
        print("❌ STT not configured — add an 'stt' section to config.json",
              file=sys.stderr)
        sys.exit(1)

    # Convert non-WAV inputs first
    path = args.audio
    if not path.lower().endswith(".wav"):
        wav = voice_to_wav(path)
        if not wav:
            print("❌ Could not convert audio to WAV", file=sys.stderr)
            sys.exit(1)
        path = wav

    text = transcribe(path, language=args.lang, config=config)
    if text is None:
        print("❌ Transcription failed — check the STT endpoint", file=sys.stderr)
        sys.exit(1)
    print(text)


if __name__ == "__main__":
    main()
