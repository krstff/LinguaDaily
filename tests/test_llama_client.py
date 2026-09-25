"""Tests for src/llama_client.py — model resolution, config loading, API calls."""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary config file with LLM settings."""
    config = {
        "llm": {
            "base_url": "http://localhost:8080/v1",
            "default_model": "gemma4-26b",
            "api_key": "",
        },
        "profiles": {
            "krystof": {
                "source_lang": "en",
                "target_lang": "de",
                "target_lang_name": "German",
            },
            "anna": {
                "source_lang": "en",
                "target_lang": "es",
                "target_lang_name": "Spanish",
                "llm_model": "mistral-7b",
            },
            "custom_models": {
                "source_lang": "en",
                "target_lang": "fr",
                "target_lang_name": "French",
                "llm_translate_model": "gemma4-26b",
                "llm_tutor_model": "mistral-7b",
            },
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config, str(config_file)


class TestLlamaClientInit:
    """Test LlamaClient initialization and config loading."""

    def test_init_with_config(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config)
        assert client.base_url == "http://localhost:8080/v1"
        assert client.default_model == "gemma4-26b"

    def test_init_with_profile(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config, profile_name="anna")
        assert client.profile_name == "anna"
        assert client.profile.get("llm_model") == "mistral-7b"

    def test_init_no_profile(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config, profile_name="krystof")
        assert client.profile.get("llm_model") is None  # no override

    def test_init_missing_profile(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config, profile_name="nonexistent")
        assert client.profile == {}

    def test_init_no_default_model(self, sample_config):
        """No hardcoded fallback: missing llm.default_model → None."""
        from src.llama_client import LlamaClient
        config = {"llm": {"base_url": "http://localhost:8080/v1"}}
        with patch.dict(os.environ, {"LLAMA_MODEL": "env-model"}, clear=False):
            client = LlamaClient(config=config)
        assert client.default_model is None

    def test_no_model_translate_returns_none(self, sample_config):
        """Without a configured model, LLM calls are skipped (not silent
        fallback to a hardcoded model)."""
        from src.llama_client import LlamaClient
        config = {"llm": {"base_url": "http://localhost:8080/v1"}}
        client = LlamaClient(config=config)
        assert client.resolve_model("translate") is None
        assert client.translate("Hallo Welt") is None
        assert client.simplify_language("text", level="A1") is None
        assert client.extract_vocab("text") == []


class TestModelResolution:
    """Test model resolution priority chain."""

    def test_default_model(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config, profile_name="krystof")
        assert client.resolve_model("translate") == "gemma4-26b"
        assert client.resolve_model("tutor") == "gemma4-26b"

    def test_profile_generic_override(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config, profile_name="anna")
        # Generic llm_model applies to all tasks
        assert client.resolve_model("translate") == "mistral-7b"
        assert client.resolve_model("tutor") == "mistral-7b"

    def test_profile_task_override_beats_generic(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config, profile_name="custom_models")
        assert client.resolve_model("translate") == "gemma4-26b"
        assert client.resolve_model("tutor") == "mistral-7b"

    def test_deprecated_task_models_ignored(self, sample_config):
        """llm.translate_model / tutor_model / simplify_model are deprecated
        and no longer affect model resolution."""
        from src.llama_client import LlamaClient
        config = dict(sample_config[0])
        config["llm"]["translate_model"] = "deprecated-model"
        config["llm"]["tutor_model"] = "deprecated-model"
        client = LlamaClient(config=config, profile_name="krystof")
        assert client.resolve_model("translate") == "gemma4-26b"
        assert client.resolve_model("tutor") == "gemma4-26b"

    @patch("config.get_openai_client")
    def test_explicit_model_param(self, mock_get_client, sample_config):
        """Explicit model arg in _chat overrides resolution."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test"))]
        )
        mock_get_client.return_value = mock_openai

        client._chat([{"role": "user", "content": "hi"}], model="explicit-model")
        mock_openai.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "explicit-model"


class TestTranslate:
    """Test translation calls."""

    @patch("openai.OpenAI")
    def test_translate_sends_correct_messages(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello World"))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client.translate("Hallo Welt", source_lang="de", target_lang="en")

        assert result == "Hello World"
        call_args = mock_instance.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        assert messages[0]["role"] == "system"
        assert "de" in messages[0]["content"]
        assert "en" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hallo Welt"
        # Translation should use low temperature
        assert call_args["temperature"] == 0.1

    def test_translate_returns_none_on_error(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config)
        # No openai client → returns None
        result = client.translate("test")
        assert result is None


class TestExtractVocab:
    """Test vocabulary extraction."""

    @patch("openai.OpenAI")
    def test_vocab_parses_json(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        vocab_json = json.dumps([
            {"word": "Hallo", "meaning": "Hello"},
            {"word": "Welt", "meaning": "World"},
        ])
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=vocab_json))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client.extract_vocab("Hallo Welt", source_lang="de", target_lang="en")

        assert len(result) == 2
        assert result[0]["word"] == "Hallo"
        assert result[1]["meaning"] == "World"

    @patch("openai.OpenAI")
    def test_vocab_handles_code_fences(self, MockOpenAI, sample_config):
        """Response wrapped in markdown code fences should be parsed."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        vocab_json = json.dumps([{"word": "test", "meaning": "test"}])
        fenced = f"```json\n{vocab_json}\n```"
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=fenced))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client.extract_vocab("test", source_lang="de", target_lang="en")

        assert len(result) == 1
        assert result[0]["word"] == "test"

    @patch("openai.OpenAI")
    def test_vocab_empty_on_bad_json(self, MockOpenAI, sample_config):
        """Garbage response returns empty list, not exception."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not json at all"))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client.extract_vocab("test", source_lang="de", target_lang="en")
        assert result == []

    @patch("openai.OpenAI")
    def test_vocab_respects_max_words(self, MockOpenAI, sample_config):
        """Should trim to max_words."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        words = [{"word": f"w{i}", "meaning": f"m{i}"} for i in range(50)]
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(words)))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client.extract_vocab("text", max_words=10)
        assert len(result) == 10


class TestTutorChat:
    """Test tutoring chat."""

    @patch("openai.OpenAI")
    def test_tutor_chat_sends_history(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="That means 'hello' in German."))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        history = [
            {"role": "user", "content": "What does Hallo mean?"},
            {"role": "assistant", "content": "Hallo means hello in German."},
        ]
        result = client.tutor_chat(
            "How do you say goodbye?",
            language_name="German",
            native_lang="English",
            history=history,
        )

        assert result == "That means 'hello' in German."
        call_args = mock_instance.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        # system + 2 history + current message = 4
        assert len(messages) == 4
        assert messages[-1]["content"] == "How do you say goodbye?"

    @patch("openai.OpenAI")
    def test_tutor_chat_trims_history(self, MockOpenAI, sample_config):
        """Should limit history to max_history turns."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OK"))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        # 6 turns = 12 messages (user/assistant pairs)
        flat_history = []
        for i in range(6):
            flat_history.append({"role": "user", "content": f"msg{i}"})
            flat_history.append({"role": "assistant", "content": f"reply{i}"})

        client.tutor_chat("new msg", history=flat_history, max_history=2)
        call_args = mock_instance.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        # system (1) + last 4 history messages + current (1) = 6
        assert len(messages) == 6

    @patch("openai.OpenAI")
    def test_tutor_chat_uses_higher_temperature(self, MockOpenAI, sample_config):
        """Tutoring should use higher temperature for creative responses."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Great question!"))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        client.tutor_chat("What is Konjunktiv?")
        call_args = mock_instance.chat.completions.create.call_args[1]
        assert call_args["temperature"] == 0.7


class TestTutorChatStream:
    """Streaming tutor variant (used by the Telegram bot / /stop)."""

    @staticmethod
    def _make_client(sample_config, chunks, stream_state):
        import src.llama_client as lc_mod
        from src.llama_client import LlamaClient

        client = LlamaClient(config=sample_config[0], profile_name="krystof")
        client._prepare_tutor_messages = lambda **kw: (
            [{"role": "user", "content": "hi"}], "gemma4-26b")

        class FakeStream:
            def __init__(self):
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._i >= len(chunks):
                    raise StopAsyncIteration
                token = chunks[self._i]
                self._i += 1
                if stream_state is not None:
                    await asyncio.sleep(stream_state.pop("delay", 0.001))
                return MagicMock(choices=[MagicMock(delta=MagicMock(content=token))])

            async def close(self):
                if stream_state is not None:
                    stream_state["closed"] = True

        async def fake_create(**kw):
            fake_create.stream_kwargs = kw
            return FakeStream()

        return client, fake_create

    def test_stream_concatenates_chunks(self, sample_config):

        client, fake_create = self._make_client(
            sample_config, ["Hello ", "world"], None)

        with patch("config.get_async_openai_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create = fake_create
            mock_get.return_value = mock_client

            reply = asyncio.run(client.tutor_chat_stream(message="hi"))

        assert reply == "Hello world"
        assert fake_create.stream_kwargs["stream"] is True

    def test_cancellation_closes_stream(self, sample_config):
        """Cancelling the task must close the stream (→ server aborts gen)."""

        state = {"delay": 0.01, "closed": False}
        client, fake_create = self._make_client(
            sample_config, ["x "] * 100, state)

        with patch("config.get_async_openai_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create = fake_create
            mock_get.return_value = mock_client

            async def run():
                task = asyncio.create_task(client.tutor_chat_stream(message="hi"))
                await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            asyncio.run(run())

        assert state["closed"], "stream must be closed on cancellation"

    def test_client_none_returns_none(self, sample_config):
        from src.llama_client import LlamaClient

        client = LlamaClient(config=sample_config[0], profile_name="krystof")
        client._prepare_tutor_messages = lambda **kw: ([("user", "hi")], "m")

        with patch("config.get_async_openai_client", return_value=None):
            assert asyncio.run(client.tutor_chat_stream(message="hi")) is None


class TestIntentRewrite:
    """Intent classification + retrieval-query rewriting."""

    @patch("openai.OpenAI")
    def test_classify_intent_returns_query(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "intent": "vocab_query",
                "query": "meaning of the German word 'Hallo' in English",
            })))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client._classify_intent(
            "what does Hallo mean?", "German",
            history=[{"role": "user", "content": "teach me greetings"}],
        )
        assert result["intent"] == "vocab_query"
        assert result["query"] == "meaning of the German word 'Hallo' in English"
        assert result["terms"] == []
        assert result["lemma"] == ""

    @patch("openai.OpenAI")
    def test_classify_intent_extracts_terms_and_lemma(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "intent": "grammar_query",
                "query": "past tense conjugation of the German verb gehen",
                "terms": ["geht"],
                "lemma": "gehen",
            })))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client._classify_intent("what is the past tense of geht?", "German")
        assert result["terms"] == ["geht"]
        assert result["lemma"] == "gehen"

    @patch("openai.OpenAI")
    def test_classify_intent_terms_junk_filtered(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "intent": "vocab_query",
                "query": "q",
                "terms": ["Haus", "", 42, "Hund", "Esel", "Pferd"],
                "lemma": 7,
            })))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client._classify_intent("Haus, Hund, Esel?", "German")
        # non-strings and empties dropped, capped at 3
        assert result["terms"] == ["Haus", "Hund", "Esel"]
        assert result["lemma"] == ""

    @patch("openai.OpenAI")
    def test_classify_intent_chitchat_null_query(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "intent": "chitchat", "query": None,
            })))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client._classify_intent("hi there!", "German")
        assert result["intent"] == "chitchat"
        assert result["query"] == ""
        assert result["terms"] == []

    @patch("openai.OpenAI")
    def test_classify_intent_bad_json_falls_back(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="sorry, cannot parse"))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client._classify_intent("what is the subjunctive?", "German")
        assert result["intent"] == "chitchat"
        assert result["query"] == ""
        assert result["terms"] == []

    @patch("src.rag_service.get_rag_service")
    def test_fetch_rag_context_uses_rewritten_query(self, mock_get_rag, sample_config):
        from src.llama_client import LlamaClient
        client = LlamaClient(config=sample_config[0])

        mock_rag = MagicMock()
        mock_rag.embed_text.return_value = [0.1] * 8
        mock_rag.query_knowledge_base.return_value = [{"text": "chunk1"}]
        mock_get_rag.return_value = mock_rag

        refs = client._fetch_rag_context(
            "what is it?",
            search_query="usage of the German dative case",
        )

        mock_rag.embed_text.assert_called_once_with("usage of the German dative case")
        assert refs == ["chunk1"]

    @patch("src.rag_service.get_rag_service")
    def test_fetch_rag_context_falls_back_to_message(self, mock_get_rag, sample_config):
        from src.llama_client import LlamaClient
        client = LlamaClient(config=sample_config[0])

        mock_rag = MagicMock()
        mock_rag.embed_text.return_value = [0.1] * 8
        mock_rag.query_knowledge_base.return_value = []
        mock_get_rag.return_value = mock_rag

        client._fetch_rag_context("What is Konjunktiv II?", search_query="")
        mock_rag.embed_text.assert_called_once_with("What is Konjunktiv II?")


class TestTutorDictionary:
    """Word-level questions get dictionary references injected."""

    def _mock_llm(self, intent_json, reply="The answer."):
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=intent_json))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=reply))]),
        ]
        return mock_instance

    @patch("src.wiktionary_client.get_dictionary_reference")
    @patch("src.rag_service.get_rag_service")
    @patch("openai.OpenAI")
    def test_terms_trigger_dictionary_lookup(self, MockOpenAI, mock_get_rag,
                                            mock_dict, sample_config):
        from src.llama_client import LlamaClient
        mock_instance = self._mock_llm(json.dumps({
            "intent": "grammar_query",
            "query": "past tense of gehen",
            "terms": ["geht"],
            "lemma": "gehen",
        }))
        MockOpenAI.return_value = mock_instance
        mock_rag = MagicMock()
        mock_rag.embed_text.return_value = [0.1] * 8
        mock_rag.query_knowledge_base.return_value = [{"text": "chunk"}]
        mock_get_rag.return_value = mock_rag
        mock_dict.return_value = "=== geht ===\npast: ging"

        client = LlamaClient(config=sample_config[0])
        client.tutor_chat("Was ist das Präteritum von geht?", language_name="German")

        mock_dict.assert_called_once()
        args, _ = mock_dict.call_args
        assert args[0] == ["geht"]      # terms
        assert args[1] == "gehen"       # lemma
        assert args[2] == "de"          # language code derived from name

        # Dictionary reference in the system prompt; RAG is NOT queried
        # when a dictionary reference was found
        final_messages = mock_instance.chat.completions.create.call_args_list[1][1]["messages"]
        system = final_messages[0]["content"]
        assert "=== geht ===" in system
        assert "past: ging" in system
        assert "chunk" not in system
        mock_rag.query_knowledge_base.assert_not_called()

    @patch("src.wiktionary_client.get_dictionary_reference")
    @patch("src.rag_service.get_rag_service")
    @patch("openai.OpenAI")
    def test_rag_used_when_dictionary_not_found(self, MockOpenAI, mock_get_rag,
                                                mock_dict, sample_config):
        from src.llama_client import LlamaClient
        mock_instance = self._mock_llm(json.dumps({
            "intent": "vocab_query",
            "query": "meaning of qqqxyz",
            "terms": ["qqqxyz"],
            "lemma": None,
        }))
        MockOpenAI.return_value = mock_instance
        mock_rag = MagicMock()
        mock_rag.embed_text.return_value = [0.1] * 8
        mock_rag.query_knowledge_base.return_value = [{"text": "chunk"}]
        mock_get_rag.return_value = mock_rag
        mock_dict.return_value = None  # wiktionary has no entry

        client = LlamaClient(config=sample_config[0])
        client.tutor_chat("What does qqqxyz mean?", language_name="German")

        # Dictionary produced nothing → RAG fallback runs
        mock_rag.query_knowledge_base.assert_called_once()
        final_messages = mock_instance.chat.completions.create.call_args_list[1][1]["messages"]
        assert "chunk" in final_messages[0]["content"]

    @patch("src.wiktionary_client.get_dictionary_reference")
    @patch("src.rag_service.get_rag_service")
    @patch("openai.OpenAI")
    def test_dictionary_failure_does_not_break_chat(self, MockOpenAI, mock_get_rag,
                                                    mock_dict, sample_config):
        from src.llama_client import LlamaClient
        mock_instance = self._mock_llm(json.dumps({
            "intent": "vocab_query",
            "query": "meaning of nosit",
            "terms": ["nosit"],
            "lemma": None,
        }), reply="It means nose.")
        MockOpenAI.return_value = mock_instance
        mock_rag = MagicMock()
        mock_rag.embed_text.return_value = [0.1] * 8
        mock_rag.query_knowledge_base.return_value = []
        mock_get_rag.return_value = mock_rag
        mock_dict.side_effect = RuntimeError("kiwix down")

        client = LlamaClient(config=sample_config[0])
        reply = client.tutor_chat("What does nosit mean?", language_name="German")

        assert reply == "It means nose."

    @patch("src.wiktionary_client.get_dictionary_reference")
    @patch("src.rag_service.get_rag_service")
    @patch("openai.OpenAI")
    def test_chitchat_skips_dictionary(self, MockOpenAI, mock_get_rag,
                                       mock_dict, sample_config):
        from src.llama_client import LlamaClient
        mock_instance = self._mock_llm(json.dumps({
            "intent": "chitchat", "query": None, "terms": [], "lemma": None,
        }))
        MockOpenAI.return_value = mock_instance
        mock_get_rag.return_value = MagicMock()

        client = LlamaClient(config=sample_config[0])
        client.tutor_chat("Hi!", language_name="German")

        mock_dict.assert_not_called()
        mock_get_rag.assert_not_called()

    @patch("src.wiktionary_client.get_dictionary_reference")
    @patch("src.rag_service.get_rag_service")
    @patch("openai.OpenAI")
    def test_no_terms_skips_dictionary(self, MockOpenAI, mock_get_rag,
                                       mock_dict, sample_config):
        from src.llama_client import LlamaClient
        mock_instance = self._mock_llm(json.dumps({
            "intent": "grammar_query",
            "query": "how to use the dative case",
            "terms": [],
            "lemma": None,
        }))
        MockOpenAI.return_value = mock_instance
        mock_rag = MagicMock()
        mock_rag.embed_text.return_value = [0.1] * 8
        mock_rag.query_knowledge_base.return_value = []
        mock_get_rag.return_value = mock_rag

        client = LlamaClient(config=sample_config[0])
        client.tutor_chat("How does the dative case work?", language_name="German")

        mock_dict.assert_not_called()


class TestHealthCheck:
    """Test health check."""

    @patch("openai.OpenAI")
    def test_health_ok(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OK"))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        assert client.health_check() is True

    @patch("openai.OpenAI")
    def test_health_fail(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = Exception("connection refused")
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        assert client.health_check() is False


class TestSimplifyLanguage:
    """Test text simplification to CEFR levels."""

    @patch("openai.OpenAI")
    def test_simplify_sends_correct_prompt(self, MockOpenAI, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Vereinfachter Text."))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        result = client.simplify_language("Komplexer Text.", language="de", level="B1")

        assert result == "Vereinfachter Text."
        call_args = mock_instance.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        assert messages[0]["role"] == "system"
        assert "B1" in messages[0]["content"]
        assert "German" in messages[0]["content"]
        assert messages[1]["content"] == "Komplexer Text."
        # Should use low temperature for deterministic simplification
        assert call_args["temperature"] == 0.1

    @patch("openai.OpenAI")
    def test_simplify_uses_general_model(self, MockOpenAI, sample_config):
        """Simplify uses the single general model (llm.default_model)."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Simple."))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        client.simplify_language("text", level="A1")

        call_args = mock_instance.chat.completions.create.call_args[1]
        assert call_args["model"] == "gemma4-26b"

    def test_simplify_returns_none_on_error(self, sample_config):
        from src.llama_client import LlamaClient
        config = sample_config[0]
        client = LlamaClient(config=config)
        result = client.simplify_language("test", level="A2")
        assert result is None

    @patch("openai.OpenAI")
    def test_simplify_various_levels(self, MockOpenAI, sample_config):
        """Should handle all CEFR levels."""
        from src.llama_client import LlamaClient
        config = sample_config[0]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OK"))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            result = client.simplify_language("text", level=level)
            assert result == "OK"
            call_args = mock_instance.chat.completions.create.call_args[1]
            assert level in call_args["messages"][0]["content"]


class TestCLI:
    """Test CLI entry point."""

    @patch("openai.OpenAI")
    def test_cli_health(self, MockOpenAI, sample_config, capsys):
        from src.llama_client import main
        config, config_path = sample_config
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OK"))]
        )
        MockOpenAI.return_value = mock_instance

        with patch("sys.argv", ["llama_client.py", "health", "--config", config_path]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "healthy" in captured.out
