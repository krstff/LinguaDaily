"""Tests for src/convo.py — conversation practice session flow.

LLM / STT / TTS layers are mocked; no network or model calls happen.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import convo as convo_mod
from convo import ConvoHandler, _clamp_turns, _level_label


# ── Helpers ─────────────────────────────────────────────────────────

def _lesson():
    return {
        "title": "Quantencomputer",
        "original_content": "Ein Quantencomputer ist ...",
        "translated_content": "A quantum computer is ...",
        "vocab": [],
        "delivered_at": "2025-01-01",
    }


def _config(**profile_overrides):
    profile = {
        "learning_language": "de",
        "native_language": "en",
        "target_level": "A2",
        "use_tts": True,
    }
    profile.update(profile_overrides)
    return {
        "stt": {"base_url": "http://stt:9002/v1", "model": "whisper"},
        "tts": {"base_url": "http://tts:8080/v1", "model": "omnivoice"},
        "llm": {"base_url": "http://llm:8080/v1", "default_model": "small-model"},
        "profiles": {"krystof": profile},
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    """ConvoHandler + fake bot with TTS stubbed to a real file."""
    config = _config()
    bot = MagicMock()
    bot.config = config
    aiogram_bot = AsyncMock()
    bot._get_aiogram_bot = AsyncMock(return_value=aiogram_bot)
    # send_message returns a message object whose delete() is awaitable
    aiogram_bot.send_message.return_value = MagicMock(delete=AsyncMock())
    bot.db = MagicMock()
    bot.db.get_latest_lesson = MagicMock(return_value=_lesson())

    handler = ConvoHandler(config=config, telegram_bot=bot)

    # TTS stub: return a real (tiny) file so FSInputFile has something
    wav_file = tmp_path / "line.wav"
    wav_file.write_bytes(b"RIFF....")
    handler._synth_line = lambda *a, **k: str(wav_file)

    # Keep temp audio out of the real output/ dir
    monkeypatch.setattr(convo_mod, "OUTPUT_DIR", tmp_path)

    return handler, bot, aiogram_bot, tmp_path


def _voice_message(chat_id=1):
    msg = MagicMock()
    msg.chat.id = chat_id
    # aiogram 3: bot.download() takes the Voice object (reads file_id)
    msg.voice = MagicMock(file_id="voice-file-id")
    return msg


def _sent_texts(aiogram_bot):
    return [
        c.kwargs.get("text", "")
        for c in aiogram_bot.send_message.await_args_list
        if c.kwargs.get("text")
    ]


def _audio_captions(aiogram_bot):
    # Direct invocations (bot(SendAudio(...))) are recorded in mock_calls
    return [
        c.args[0].caption
        for c in aiogram_bot.mock_calls
        if c.args and getattr(c.args[0], "caption", None)
    ]


def _stub_stt(monkeypatch, tmp_path, transcript):
    wav = tmp_path / "user.wav"
    wav.write_bytes(b"RIFF....")
    monkeypatch.setattr(convo_mod, "voice_to_wav", lambda *a, **k: str(wav))
    monkeypatch.setattr(convo_mod, "transcribe", lambda *a, **k: transcript)


# ── Pure helpers ────────────────────────────────────────────────────

class TestHelpers:
    def test_level_label(self):
        assert _level_label("A2") == "A2"
        assert _level_label("b1") == "B1"
        assert "simple" in _level_label("original")

    def test_clamp_turns(self):
        assert _clamp_turns(None) == convo_mod.CONVO_DEFAULT_TURNS
        assert _clamp_turns(1) == convo_mod.CONVO_MIN_TURNS
        assert _clamp_turns(99) == convo_mod.CONVO_MAX_TURNS
        assert _clamp_turns(6) == 6


# ── start_convo ─────────────────────────────────────────────────────

class TestStartConvo:
    @pytest.mark.asyncio
    async def test_requires_stt_config(self, env):
        handler, bot, aiogram_bot, _ = env
        del handler.config["stt"]
        await handler.start_convo(1, "krystof")
        assert 1 not in handler._sessions
        assert any("Speech-to-text" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_requires_lesson(self, env):
        handler, bot, aiogram_bot, _ = env
        bot.db.get_latest_lesson = MagicMock(return_value=None)
        await handler.start_convo(1, "krystof")
        assert 1 not in handler._sessions
        assert any("No lesson yet" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_requires_tts_enabled(self, env):
        handler, bot, aiogram_bot, _ = env
        handler.config["profiles"]["krystof"]["use_tts"] = False
        await handler.start_convo(1, "krystof")
        assert 1 not in handler._sessions
        assert any("TTS" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_sends_opening_line_with_spoilers(self, env):
        handler, bot, aiogram_bot, _ = env
        handler._generate_opening = lambda *a, **k: {
            "reply": "Hast du den Artikel über Quantencomputer gelesen?",
            "translation": "Did you read the article about quantum computers?",
        }
        await handler.start_convo(1, "krystof")

        session = handler._sessions[1]
        assert session["turns_total"] == convo_mod.CONVO_DEFAULT_TURNS
        assert session["topic"] == "Quantencomputer"
        assert session["lines"] == []

        captions = _audio_captions(aiogram_bot)
        assert len(captions) == 1
        assert "<tg-spoiler>" in captions[0]
        assert "Quantencomputer" in captions[0]
        assert "Hast du den Artikel" in captions[0]

    @pytest.mark.asyncio
    async def test_respects_turns_argument(self, env):
        handler, bot, aiogram_bot, _ = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof", turns=6)
        assert handler._sessions[1]["turns_total"] == 6

    @pytest.mark.asyncio
    async def test_llm_failure_no_session(self, env):
        handler, bot, aiogram_bot, _ = env
        handler._generate_opening = lambda *a, **k: None
        await handler.start_convo(1, "krystof")
        assert 1 not in handler._sessions
        assert any("Could not start" in t for t in _sent_texts(aiogram_bot))


# ── end_convo (used by /stop) ───────────────────────────────────────

class TestEndConvo:
    @pytest.mark.asyncio
    async def test_ends_active_session(self, env):
        handler, bot, aiogram_bot, _ = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        assert 1 in handler._sessions

        assert handler.end_convo(1) is True
        assert 1 not in handler._sessions
        # Second call: nothing left to end
        assert handler.end_convo(1) is False

    @pytest.mark.asyncio
    async def test_no_session_returns_false(self, env):
        handler, bot, aiogram_bot, _ = env
        assert handler.end_convo(1) is False

    @pytest.mark.asyncio
    async def test_voice_after_stop_is_rejected(self, env):
        handler, bot, aiogram_bot, tmp_path = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        handler.end_convo(1)

        await handler.handle_voice(_voice_message(1))
        assert any("No active conversation" in t for t in _sent_texts(aiogram_bot))


# ── Voice handling ──────────────────────────────────────────────────

class TestVoiceFlow:
    @pytest.mark.asyncio
    async def test_voice_without_session_gets_hint(self, env):
        handler, bot, aiogram_bot, _ = env
        await handler.handle_voice(_voice_message(42))
        assert any("/convo" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_expired_session_gets_hint(self, env):
        handler, bot, aiogram_bot, _ = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        handler._sessions[1]["created_at"] = time.time() - 720  # > timeout
        await handler.handle_voice(_voice_message(1))
        assert 1 not in handler._sessions
        assert any("/convo" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_busy_session_rejects(self, env):
        handler, bot, aiogram_bot, _ = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        handler._sessions[1]["busy"] = True
        await handler.handle_voice(_voice_message(1))
        assert any("Still working" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_empty_transcript_keeps_session(self, env, monkeypatch):
        handler, bot, aiogram_bot, tmp_path = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        _stub_stt(monkeypatch, tmp_path, "")

        await handler.handle_voice(_voice_message(1))

        assert len(handler._sessions[1]["lines"]) == 0
        assert any("couldn't catch" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_stt_failure_keeps_session(self, env, monkeypatch):
        handler, bot, aiogram_bot, tmp_path = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        _stub_stt(monkeypatch, tmp_path, None)  # endpoint down

        await handler.handle_voice(_voice_message(1))

        assert len(handler._sessions[1]["lines"]) == 0
        assert any("Transcription failed" in t for t in _sent_texts(aiogram_bot))

    @pytest.mark.asyncio
    async def test_scores_and_advances(self, env, monkeypatch):
        handler, bot, aiogram_bot, tmp_path = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        _stub_stt(monkeypatch, tmp_path, "Ja, ich habe es gelesen.")
        handler._generate_next = lambda *a, **k: {
            "feedback": "Sehr gut.",
            "score": 90,
            "reply": "Sehr interessant.",
            "translation": "Very interesting.",
        }

        await handler.handle_voice(_voice_message(1))

        session = handler._sessions[1]
        assert session["lines"] == [
            {"transcript": "Ja, ich habe es gelesen.", "score": 90}
        ]
        texts = _sent_texts(aiogram_bot)
        assert any("90/100" in t and "Ja, ich habe es gelesen." in t for t in texts)
        # next line audio with turn 2 caption
        captions = _audio_captions(aiogram_bot)
        assert any("Conversation 2/6" in c for c in captions)

    @pytest.mark.asyncio
    async def test_score_is_clamped(self, env, monkeypatch):
        handler, bot, aiogram_bot, tmp_path = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        await handler.start_convo(1, "krystof")
        _stub_stt(monkeypatch, tmp_path, "Etwas.")
        handler._generate_next = lambda *a, **k: {
            "feedback": "?", "score": 250, "reply": "Ok", "translation": "Ok",
        }
        await handler.handle_voice(_voice_message(1))
        assert handler._sessions[1]["lines"][0]["score"] == 100

    @pytest.mark.asyncio
    async def test_final_turn_sends_summary_and_ends(self, env, monkeypatch):
        handler, bot, aiogram_bot, tmp_path = env
        handler._generate_opening = lambda *a, **k: {"reply": "Hi", "translation": "Hi"}
        handler._generate_next = lambda *a, **k: {
            "feedback": "Gut.", "score": 80, "reply": "Weiter", "translation": "More",
        }
        await handler.start_convo(1, "krystof")
        handler._sessions[1]["turns_total"] = 2

        _stub_stt(monkeypatch, tmp_path, "Eins.")
        await handler.handle_voice(_voice_message(1))
        assert 1 in handler._sessions  # still going (1/2)

        _stub_stt(monkeypatch, tmp_path, "Zwei, sehr gut.")
        await handler.handle_voice(_voice_message(1))

        assert 1 not in handler._sessions  # session ended
        texts = _sent_texts(aiogram_bot)
        assert any("Conversation complete" in t for t in texts)
        assert any("80/100" in t for t in texts)  # average score
        assert any("Quantencomputer" in t for t in texts)

    @pytest.mark.asyncio
    async def test_lesson_topic_used_in_prompts(self, env, monkeypatch):
        handler, bot, aiogram_bot, tmp_path = env
        captured = {}

        def fake_opening(client, lesson, lang_code, native, level):
            captured["lesson_title"] = lesson.get("title")
            captured["level"] = level
            return {"reply": "Hi", "translation": "Hi"}

        handler._generate_opening = fake_opening
        await handler.start_convo(1, "krystof")
        assert captured["lesson_title"] == "Quantencomputer"
        assert captured["level"] == "A2"


# ── Model resolution ────────────────────────────────────────────────

class TestModelResolution:
    def test_global_task_model_override(self):
        config = _config()
        config["llm"]["task_models"] = {"convo": "tiny-model"}
        from llama_client import LlamaClient
        client = LlamaClient(config=config, profile_name="krystof")
        assert client.resolve_model("convo") == "tiny-model"
        assert client.resolve_model("tutor") == "small-model"

    def test_profile_task_override_wins(self):
        config = _config()
        config["llm"]["task_models"] = {"convo": "tiny-model"}
        config["profiles"]["krystof"]["llm_convo_model"] = "profile-model"
        from llama_client import LlamaClient
        client = LlamaClient(config=config, profile_name="krystof")
        assert client.resolve_model("convo") == "profile-model"
