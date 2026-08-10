"""Tests for src/llama_client.py — model resolution, config loading, API calls."""

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
    def test_simplify_uses_simplify_model(self, MockOpenAI, sample_config):
        """Should resolve the 'simplify' model task."""
        from src.llama_client import LlamaClient
        config = dict(sample_config[0])
        config["llm"]["simplify_model"] = "simplifier-model"
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Simple."))]
        )
        MockOpenAI.return_value = mock_instance

        client = LlamaClient(config=config)
        client.simplify_language("text", level="A1")

        call_args = mock_instance.chat.completions.create.call_args[1]
        assert call_args["model"] == "simplifier-model"

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
