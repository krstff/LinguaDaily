"""Tests for src/stt.py — config resolution + transcription client (mocked).

No network calls: the OpenAI client is mocked.
"""

import shutil
from unittest.mock import MagicMock

import pytest

import stt


# ── Config resolution ───────────────────────────────────────────────

class TestConfig:
    def test_get_stt_config(self):
        cfg = stt.get_stt_config(
            {"stt": {"base_url": "http://x/v1", "model": "w", "timeout": 5}})
        assert cfg["base_url"] == "http://x/v1"
        assert cfg["model"] == "w"
        assert cfg["api_key"] == "none"
        assert cfg["timeout"] == 5

    def test_get_stt_config_defaults(self):
        cfg = stt.get_stt_config({"stt": {"base_url": "http://x", "model": "w"}})
        assert cfg["timeout"] == stt.DEFAULT_TIMEOUT
        assert cfg["api_key"] == "none"

    def test_is_configured(self):
        assert stt.is_configured(
            {"stt": {"base_url": "http://x", "model": "w"}})
        assert not stt.is_configured({})
        assert not stt.is_configured({"stt": {"base_url": "http://x"}})
        assert not stt.is_configured({"stt": {"model": "w"}})


# ── transcribe() ────────────────────────────────────────────────────

class TestTranscribe:
    def test_unconfigured_returns_none(self):
        assert stt.transcribe("/tmp/x.wav", config={}) is None

    def test_missing_file_returns_none(self, tmp_path):
        cfg = {"stt": {"base_url": "http://x/v1", "model": "w"}}
        assert stt.transcribe(str(tmp_path / "nope.wav"), config=cfg) is None

    def test_success(self, tmp_path, monkeypatch):
        import config as config_mod

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFFfake")
        fake = MagicMock()
        fake.audio.transcriptions.create.return_value = MagicMock(
            text="Hallo Welt")
        monkeypatch.setattr(config_mod, "get_openai_client",
                            lambda **kw: fake)

        text = stt.transcribe(
            str(audio), language="de",
            config={"stt": {"base_url": "http://x/v1", "model": "whisper"}})

        assert text == "Hallo Welt"
        kwargs = fake.audio.transcriptions.create.call_args.kwargs
        assert kwargs["model"] == "whisper"
        assert kwargs["language"] == "de"

    def test_endpoint_error_returns_none(self, tmp_path, monkeypatch):
        import config as config_mod

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFFfake")
        fake = MagicMock()
        fake.audio.transcriptions.create.side_effect = Exception(
            "Connection refused")
        monkeypatch.setattr(config_mod, "get_openai_client",
                            lambda **kw: fake)

        assert stt.transcribe(
            str(audio),
            config={"stt": {"base_url": "http://x/v1", "model": "w"}},
        ) is None

    def test_no_language_when_none(self, tmp_path, monkeypatch):
        import config as config_mod

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFFfake")
        fake = MagicMock()
        fake.audio.transcriptions.create.return_value = MagicMock(text="ok")
        monkeypatch.setattr(config_mod, "get_openai_client",
                            lambda **kw: fake)

        stt.transcribe(
            str(audio),
            config={"stt": {"base_url": "http://x/v1", "model": "w"}})

        kwargs = fake.audio.transcriptions.create.call_args.kwargs
        assert kwargs["language"] is None


# ── voice_to_wav() ──────────────────────────────────────────────────

class TestVoiceToWav:
    def test_missing_ffmpeg_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        src = tmp_path / "a.ogg"
        src.write_bytes(b"OggS")
        assert stt.voice_to_wav(str(src)) is None

    def test_missing_source_returns_none(self, tmp_path):
        assert stt.voice_to_wav(str(tmp_path / "nope.ogg")) is None
