"""Tests for src/processor.py — vocabulary storage (SQLite via VocabDB)."""

import pytest


class TestProcessorInit:
    """Test LinguaProcessor initialization."""

    def test_default_init(self, tmp_path):
        from src.processor import LinguaProcessor
        proc = LinguaProcessor(learning_language="de", profile="test",
                               db_path=str(tmp_path / "test.db"))
        assert proc.learning_language == "de"
        assert proc.learning_language_name == "German"
        assert proc.profile == "test"
        proc.close()

    def test_explicit_db_path(self, tmp_path):
        from src.processor import LinguaProcessor
        db = tmp_path / "custom.db"
        proc = LinguaProcessor(profile="test", db_path=str(db))
        assert str(db) in str(proc.db.db_path)
        proc.close()

    def test_external_db_not_closed_by_processor(self, tmp_path):
        from src.processor import LinguaProcessor
        from src.vocab_db import VocabDB
        db = VocabDB(str(tmp_path / "external.db"))
        proc = LinguaProcessor(profile="test", db=db)
        proc.update_vocab(["hello"])
        proc.close()
        assert db.word_count("test") == 1  # externally owned → still open
        db.close()


class TestUpdateVocab:
    """Test vocabulary updates against a temporary database."""

    @pytest.fixture
    def processor(self, tmp_path):
        from src.processor import LinguaProcessor
        proc = LinguaProcessor(profile="test",
                               db_path=str(tmp_path / "test.db"))
        yield proc
        proc.close()

    def test_add_new_word_string(self, processor):
        processor.update_vocab(["hello"])
        entries = {e["word"].lower(): e for e in processor.db.get_entries("test")}
        assert "hello" in entries

    def test_add_new_word_dict(self, processor):
        processor.update_vocab([{"word": "bonjour", "meaning": "greeting"}])
        entries = {e["word"].lower(): e for e in processor.db.get_entries("test")}
        assert "bonjour" in entries
        assert entries["bonjour"]["meaning"] == "greeting"

    def test_reencounter_increments_frequency(self, processor):
        processor.update_vocab(["hello"])
        assert processor.update_vocab(["hello"]) == 1  # refreshed, not new
        entries = {e["word"].lower(): e for e in processor.db.get_entries("test")}
        assert entries["hello"]["frequency"] == 2
        assert processor.db.word_count("test") == 1

    def test_skip_empty_words(self, processor):
        processor.update_vocab(["", "  ", "hello"])
        entries = {e["word"].lower(): e for e in processor.db.get_entries("test")}
        assert "" not in entries
        assert "hello" in entries

    def test_case_insensitive_reencounter(self, processor):
        processor.update_vocab(["Hello"])
        processor.update_vocab(["hello"])  # same word, different case
        entries = processor.db.get_entries("test")
        assert len(entries) == 1
        assert entries[0]["word"] == "Hello"  # original form kept
        assert entries[0]["frequency"] == 2

    def test_multiple_words_batch(self, processor):
        assert processor.update_vocab([
            {"word": "Haus", "meaning": "house"},
            {"word": "Auto", "meaning": "car"},
            {"word": "Buch", "meaning": "book"},
        ]) == 3
        assert processor.db.word_count("test") == 3

    def test_todays_date_recorded(self, processor):
        from datetime import date
        processor.update_vocab([{"word": "test", "meaning": "test"}])
        entries = processor.db.get_entries("test")
        assert entries[0]["last_seen"] == date.today().isoformat()

    def test_profiles_isolated(self, processor):
        processor.update_vocab(["only-test"])
        assert processor.db.word_count("other-profile") == 0
        assert processor.db.word_count("test") == 1
