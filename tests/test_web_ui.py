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
                "embedding_model": "embed-y",
            })
            assert res.status_code == 200

        cfg = json.loads(config_file.read_text())
        assert cfg["llm"]["default_model"] == "model-b"
        assert cfg["tts"]["model"] == "tts-y"
        assert cfg["rag"]["embedding_model"] == "embed-y"

    def test_save_empty_removes_key(self, tmp_path):
        """Empty values delete the key — nothing hidden is substituted."""
        app, config_file = _make_app(tmp_path)
        with app.test_client() as c:
            res = c.post("/api/models/save", json={
                "default_model": "",
                "tts_model": "",
                "embedding_model": "",
            })
            assert res.status_code == 200

        cfg = json.loads(config_file.read_text())
        assert "default_model" not in cfg["llm"]
        assert "model" not in cfg["tts"]
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
            assert data["embedding_model"] == ""
