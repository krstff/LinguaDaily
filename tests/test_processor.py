"""Tests for src/processor.py — content processing, vocabulary tracking."""

import os
import pytest
from unittest.mock import patch


class TestProcessorInit:
    """Test LinguaProcessor initialization."""

    def test_default_init(self):
        from src.processor import LinguaProcessor
        proc = LinguaProcessor(target_lang_name="German", profile="test")
        assert proc.target_lang_name == "German"
        assert proc.profile == "test"

    def test_explicit_vocab_path(self, tmp_path):
        from src.processor import LinguaProcessor
        vocab = tmp_path / "vocab.md"
        proc = LinguaProcessor(profile="test", vocab_path=str(vocab))
        assert str(vocab) in proc.vocab_path

    def test_default_vocab_path(self, tmp_path, monkeypatch):
        from src.processor import LinguaProcessor
        # Set up a temp project dir
        monkeypatch.setattr("src.processor.__file__", str(tmp_path / "processor.py"))
        proc = LinguaProcessor(profile="test_user")
        assert "test_user" in proc.vocab_path


class TestProcessDailyContent:
    """Test content processing for LLM translation."""

    def test_process_returns_structured_data(self):
        from src.processor import LinguaProcessor
        proc = LinguaProcessor(target_lang_name="German", profile="test")
        result = proc.process_daily_content("Hello world", source_lang="en", title="Test")
        assert "source_text" in result
        assert result["source_text"] == "Hello world"
        assert result["target_lang"] == "German"
        assert "instruction" in result

    def test_instruction_contains_target_lang(self):
        from src.processor import LinguaProcessor
        proc = LinguaProcessor(target_lang_name="French", profile="test")
        result = proc.process_daily_content("Bonjour", source_lang="en")
        assert "French" in result["instruction"]

    def test_preserves_title(self):
        from src.processor import LinguaProcessor
        proc = LinguaProcessor(profile="test")
        result = proc.process_daily_content("Content", title="My Title")
        assert result["title"] == "My Title"


class TestVocabFile:
    """Test vocabulary file management."""

    @pytest.fixture
    def processor(self, tmp_path):
        from src.processor import LinguaProcessor
        vocab = tmp_path / "vocab.md"
        return LinguaProcessor(profile="test", vocab_path=str(vocab))

    def test_ensure_vocab_file_creates_if_missing(self, processor):
        assert not os.path.exists(processor.vocab_path)
        processor._ensure_vocab_file()
        assert os.path.exists(processor.vocab_path)

    def test_ensure_vocab_file_skips_existing(self, processor):
        with open(processor.vocab_path, "w") as f:
            f.write("existing content")
        processor._ensure_vocab_file()
        with open(processor.vocab_path) as f:
            assert "existing content" in f.read()

    def test_read_existing_vocab(self, processor):
        # Write a sample vocab file
        with open(processor.vocab_path, "w") as f:
            f.write("# Vocab\n\n| Word | Meaning | Frequency | Last Seen |\n")
            f.write("|---|---|---|---|\n")
            f.write("| hello | greeting | 3 | 2026-01-01 |\n")
            f.write("| world | earth | 1 | 2026-01-02 |\n")

        vocab = processor._read_existing_vocab()
        assert "hello" in vocab
        assert vocab["hello"] == 3
        assert "world" in vocab
        assert vocab["world"] == 1

    def test_read_existing_vocab_empty_file(self, processor):
        with open(processor.vocab_path, "w") as f:
            f.write("# Empty\n")
        vocab = processor._read_existing_vocab()
        assert vocab == {}

    def test_read_existing_vocab_missing_file(self, processor):
        # File doesn't exist yet
        vocab = processor._read_existing_vocab()
        assert vocab == {}


class TestUpdateVocab:
    """Test vocabulary updates."""

    @pytest.fixture
    def processor(self, tmp_path):
        from src.processor import LinguaProcessor
        vocab = tmp_path / "vocab.md"
        return LinguaProcessor(profile="test", vocab_path=str(vocab))

    def test_add_new_word_string(self, processor):
        processor._ensure_vocab_file()
        processor.update_vocab(["hello"])
        vocab = processor._read_existing_vocab()
        assert "hello" in vocab

    def test_add_new_word_dict(self, processor):
        processor._ensure_vocab_file()
        processor.update_vocab([{"word": "bonjour", "meaning": "greeting", "freq": 1}])
        vocab = processor._read_existing_vocab()
        assert "bonjour" in vocab

    def test_skip_duplicate(self, processor):
        processor._ensure_vocab_file()
        processor.update_vocab(["hello"])
        processor.update_vocab(["hello"])  # duplicate
        vocab = processor._read_existing_vocab()
        assert vocab.get("hello", 0) == 1  # Should still be 1, not incremented

    def test_skip_empty_words(self, processor):
        processor._ensure_vocab_file()
        processor.update_vocab(["", "  ", "hello"])
        vocab = processor._read_existing_vocab()
        assert "" not in vocab
        assert "hello" in vocab

    def test_case_insensitive_dedup(self, processor):
        processor._ensure_vocab_file()
        processor.update_vocab(["Hello"])
        processor.update_vocab(["hello"])  # same word, different case
        vocab = processor._read_existing_vocab()
        assert len(vocab) == 1
