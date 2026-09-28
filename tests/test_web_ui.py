"""Tests for src/web_ui.py — model management endpoints.

config.json is the single source of truth for model names: saving an empty
value must REMOVE the key (no hardcoded defaults), and /api/models/current
must report exactly what is in the config.
"""

import json


def _make_app(tmp_path):
    config = {
        "llm": {
            "base_url": "http://localhost:8080/v1",
            "default_model": "model-a",
            "api_key": "",
        },
        "tts": {"base_url": "http://localhost:8080/v1", "model": "tts-x"},
        "stt": {"base_url": "http://localhost:9002/v1", "model": "whisper"},
        "rag": {"embedding_model": "embed-x"},
        "profiles": {},
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))

    from src.web_ui import create_app
    app = create_app(config_path=str(config_file))
    app.config["TESTING"] = True
    return app, config_file


class TestModelsSave:
    """POST /api/models/save — config.json stays the source of truth."""

    def test_save_sets_values(self, tmp_path):
        app, config_file = _make_app(tmp_path)
        with app.test_client() as c:
            res = c.post("/api/models/save", json={
                "default_model": "model-b",
                "tts_model": "tts-y",
                "stt_model": "whisper-large",
                "embedding_model": "embed-y",
            })
            assert res.status_code == 200

        cfg = json.loads(config_file.read_text())
        assert cfg["llm"]["default_model"] == "model-b"
        assert cfg["tts"]["model"] == "tts-y"
        assert cfg["stt"]["model"] == "whisper-large"
        assert cfg["rag"]["embedding_model"] == "embed-y"

    def test_save_empty_removes_key(self, tmp_path):
        """Empty values delete the key — nothing hidden is substituted."""
        app, config_file = _make_app(tmp_path)
        with app.test_client() as c:
            res = c.post("/api/models/save", json={
                "default_model": "",
                "tts_model": "",
                "stt_model": "",
                "embedding_model": "",
            })
            assert res.status_code == 200

        cfg = json.loads(config_file.read_text())
        assert "default_model" not in cfg["llm"]
        assert "model" not in cfg["tts"]
        assert "model" not in cfg["stt"]
        assert "embedding_model" not in cfg["rag"]


class TestModelsCurrent:
    """GET /api/models/current — reports the config verbatim."""

    def test_current_reports_config(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with app.test_client() as c:
            res = c.get("/api/models/current")
            assert res.status_code == 200
            data = res.get_json()
            assert data["default_model"] == "model-a"
            assert data["tts_model"] == "tts-x"
            assert data["stt_model"] == "whisper"
            assert data["embedding_model"] == "embed-x"

    def test_current_empty_when_unconfigured(self, tmp_path):
        config = {"llm": {"base_url": "http://localhost:8080/v1"}, "profiles": {}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        from src.web_ui import create_app
        app = create_app(config_path=str(config_file))
        app.config["TESTING"] = True
        with app.test_client() as c:
            data = c.get("/api/models/current").get_json()
            # No hardcoded fallbacks — empty strings, not default names
            assert data["default_model"] == ""
            assert data["tts_model"] == ""
            assert data["stt_model"] == ""
            assert data["embedding_model"] == ""


class TestModelsFetch:
    """GET /api/models/fetch — lists models per endpoint (mocked OpenAI)."""

    def test_fetch_includes_stt_models(self, tmp_path, monkeypatch):
        import openai

        class FakeModel:
            def __init__(self, id):
                self.id = id

        class FakeModels:
            def __init__(self, ids):
                self._ids = ids

            def list(self):
                return [FakeModel(i) for i in self._ids]

        class FakeOpenAI:
            def __init__(self, base_url=None, api_key=None, timeout=None):
                self.base_url = base_url

            @property
            def models(self):
                # STT endpoint (port 9002) has its own model list;
                # LLM/TTS share port 8080 and reuse one cached query.
                if "9002" in (self.base_url or ""):
                    return FakeModels(["whisper-2", "whisper-1"])
                return FakeModels(["model-b", "model-a"])

        monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

        app, _ = _make_app(tmp_path)
        with app.test_client() as c:
            data = c.get("/api/models/fetch").get_json()

        assert data["stt_models"] == ["whisper-1", "whisper-2"]
        assert data["llm_models"] == ["model-a", "model-b"]
        assert data["tts_models"] == ["model-a", "model-b"]
        assert data["errors"] == []

    def test_fetch_stt_error_is_reported(self, tmp_path, monkeypatch):
        import openai

        class FakeModel:
            def __init__(self, id):
                self.id = id

        class FakeModels:
            def list(self):
                return [FakeModel("model-a")]

        class FakeOpenAI:
            def __init__(self, base_url=None, api_key=None, timeout=None):
                self.base_url = base_url

            @property
            def models(self):
                if "9002" in (self.base_url or ""):
                    raise RuntimeError("STT endpoint down")
                return FakeModels()

        monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

        app, _ = _make_app(tmp_path)
        with app.test_client() as c:
            data = c.get("/api/models/fetch").get_json()

        assert data["stt_models"] == []
        assert data["llm_models"] == ["model-a"]
        assert any("9002" in e for e in data["errors"])


class TestVocab:
    """Vocabulary browse + delete endpoints (shared DB is isolated to tmp)."""

    @staticmethod
    def _make_vocab_app(tmp_path):
        config = {
            "profiles": {
                "krystof": {"learning_language": "de"},
                "johi": {"learning_language": "it"},
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        from src.web_ui import create_app
        app = create_app(config_path=str(config_file))
        app.config["TESTING"] = True
        return app

    @staticmethod
    def _seed():
        from src.vocab_db import get_shared_db
        db = get_shared_db()
        db.add_words("krystof", [
            {"word": "Haus", "meaning": "house"},
            {"word": "Auto", "meaning": "car"},
            {"word": "Politiker", "meaning": "politician"},
        ])
        db.add_words("johi", [{"word": "ciao", "meaning": "hello"}])
        return db

    def test_page_renders_with_profiles_and_counts(self, tmp_path):
        app = self._make_vocab_app(tmp_path)
        self._seed()
        with app.test_client() as c:
            html = c.get("/vocab").data.decode()
        for frag in ('id="vocab-tbody"', 'id="profile-select"',
                     'id="delete-selected-btn"', 'Clear all for profile'):
            assert frag in html, frag
        assert "krystof" in html and "johi" in html

    def test_api_list(self, tmp_path):
        app = self._make_vocab_app(tmp_path)
        self._seed()
        with app.test_client() as c:
            data = c.get("/api/vocab?profile=krystof&per_page=2").get_json()
        assert data["total"] == 3
        assert data["pages"] == 2
        assert len(data["entries"]) == 2
        assert all("id" in e for e in data["entries"])

    def test_api_requires_profile(self, tmp_path):
        app = self._make_vocab_app(tmp_path)
        with app.test_client() as c:
            assert c.get("/api/vocab").status_code == 400

    def test_api_search(self, tmp_path):
        app = self._make_vocab_app(tmp_path)
        self._seed()
        with app.test_client() as c:
            data = c.get("/api/vocab?profile=krystof&search=polit").get_json()
        assert data["total"] == 1
        assert data["entries"][0]["word"] == "Politiker"

    def test_api_delete_selected(self, tmp_path):
        app = self._make_vocab_app(tmp_path)
        db = self._seed()
        _, entries = db.search_entries("krystof")
        ids = [e["id"] for e in entries if e["word"] in ("Haus", "Auto")]
        assert len(ids) == 2
        with app.test_client() as c:
            res = c.post("/api/vocab/delete",
                         json={"profile": "krystof", "ids": ids})
            assert res.status_code == 200
            assert res.get_json()["deleted"] == 2
        assert db.word_count("krystof") == 1
        assert db.word_count("johi") == 1

    def test_api_delete_requires_ids(self, tmp_path):
        app = self._make_vocab_app(tmp_path)
        self._seed()
        with app.test_client() as c:
            assert c.post("/api/vocab/delete",
                          json={"profile": "krystof", "ids": []}).status_code == 400

    def test_api_clear_profile(self, tmp_path):
        app = self._make_vocab_app(tmp_path)
        db = self._seed()
        with app.test_client() as c:
            res = c.post("/api/vocab/krystof/clear")
            assert res.status_code == 200
            assert res.get_json()["deleted"] == 3
        assert db.word_count("krystof") == 0
        assert db.word_count("johi") == 1


class TestBroadcast:
    """POST /api/broadcast — sends a message to every known chat ID."""

    @staticmethod
    def _make_broadcast_app(tmp_path, profiles, token="123:TEST"):
        config = {"profiles": profiles}
        if token is not None:
            config["telegram"] = {"bot_token": token}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        from src.web_ui import create_app
        app = create_app(config_path=str(config_file))
        app.config["TESTING"] = True
        return app

    @staticmethod
    def _fake_bot(monkeypatch, sent, fail_chats=()):
        """Monkeypatch aiogram.Bot with a fake that records send_message calls."""
        import aiogram

        class FakeSession:
            async def close(self):
                pass

        class FakeBot:
            def __init__(self, token=None):
                self.session = FakeSession()

            async def send_message(self, chat_id, text):
                if chat_id in fail_chats:
                    raise RuntimeError(f"blocked chat {chat_id}")
                sent.append((chat_id, text))

        monkeypatch.setattr(aiogram, "Bot", FakeBot)

    def test_broadcast_sends_to_all_unique_chats(self, tmp_path, monkeypatch):
        """Duplicate chat IDs across profiles are sent to only once."""
        app = self._make_broadcast_app(tmp_path, {
            "alice": {"telegram_chat_id": 111},
            "bob": {"telegram_chat_id": 222},
            "alice_de": {"telegram_chat_id": 111},  # same chat as alice
            "nochat": {},
        })
        self._fake_bot(monkeypatch, sent := [])

        with app.test_client() as c:
            res = c.post("/api/broadcast", json={"message": "hello everyone"})

        assert res.status_code == 200
        data = res.get_json()
        assert data["sent"] == 2
        assert data["failed"] == []
        assert sorted(chat for chat, _ in sent) == [111, 222]
        assert all(text == "hello everyone" for _, text in sent)

    def test_broadcast_reports_failures(self, tmp_path, monkeypatch):
        app = self._make_broadcast_app(tmp_path, {
            "alice": {"telegram_chat_id": 111},
            "bob": {"telegram_chat_id": 222},
        })
        self._fake_bot(monkeypatch, sent := [], fail_chats={222})

        with app.test_client() as c:
            res = c.post("/api/broadcast", json={"message": "hi"})

        assert res.status_code == 200
        data = res.get_json()
        assert data["sent"] == 1
        assert len(data["failed"]) == 1
        assert data["failed"][0]["chat_id"] == 222

    def test_broadcast_empty_message_rejected(self, tmp_path):
        app = self._make_broadcast_app(
            tmp_path, {"alice": {"telegram_chat_id": 111}})
        with app.test_client() as c:
            assert c.post("/api/broadcast", json={"message": "   "}).status_code == 400

    def test_broadcast_too_long_rejected(self, tmp_path):
        app = self._make_broadcast_app(
            tmp_path, {"alice": {"telegram_chat_id": 111}})
        with app.test_client() as c:
            res = c.post("/api/broadcast", json={"message": "x" * 4097})
            assert res.status_code == 400

    def test_broadcast_without_token_rejected(self, tmp_path):
        app = self._make_broadcast_app(
            tmp_path, {"alice": {"telegram_chat_id": 111}}, token=None)
        with app.test_client() as c:
            res = c.post("/api/broadcast", json={"message": "hi"})
            assert res.status_code == 400
            assert "token" in res.get_json()["message"].lower()

    def test_broadcast_without_chats_rejected(self, tmp_path, monkeypatch):
        app = self._make_broadcast_app(tmp_path, {"nochat": {}})
        self._fake_bot(monkeypatch, sent := [])
        with app.test_client() as c:
            res = c.post("/api/broadcast", json={"message": "hi"})
            assert res.status_code == 400
            assert sent == []

    def test_dashboard_shows_broadcast_panel(self, tmp_path):
        app = self._make_broadcast_app(tmp_path, {
            "alice": {"telegram_chat_id": 111},
            "bob": {"telegram_chat_id": 222},
        })
        with app.test_client() as c:
            html = c.get("/").data.decode()
        assert 'id="broadcast-message"' in html
        assert 'id="broadcast-btn"' in html
        assert "(2 chat(s))" in html
