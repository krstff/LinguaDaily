"""Tests for src/vocab_db.py — SQLite vocabulary store + CSV migration."""

import csv
import pytest


@pytest.fixture
def db(tmp_path):
    from src.vocab_db import VocabDB
    d = VocabDB(str(tmp_path / "vocab.db"))
    yield d
    d.close()


class TestVocabDB:
    def test_add_and_read(self, db):
        db.add_words("p", [{"word": "Haus", "meaning": "house"}])
        entries = db.get_entries("p")
        assert len(entries) == 1
        assert entries[0]["word"] == "Haus"
        assert entries[0]["meaning"] == "house"
        assert entries[0]["frequency"] == 1
        assert entries[0]["mastery_score"] == 0.0

    def test_reencounter_case_insensitive(self, db):
        db.add_words("p", [{"word": "Haus", "meaning": "house"}])
        assert db.add_words("p", [{"word": "haus", "meaning": "doubled"}]) == 1
        entry = db.get_entries("p")[0]
        assert db.word_count("p") == 1
        assert entry["frequency"] == 2
        assert entry["word"] == "Haus"  # original form kept
        assert entry["meaning"] == "house"  # original meaning kept

    def test_record_exposure_bumps_frequency(self, db):
        db.add_words("p", ["haus", "auto"])
        db.record_exposure("p", ["Haus", "auto"])
        entries = {e["word"]: e for e in db.get_entries("p")}
        assert entries["haus"]["frequency"] == 2
        assert entries["auto"]["frequency"] == 2

    def test_record_exposure_outcomes_update_mastery(self, db):
        db.add_words("p", ["haus"])
        # 3 correct, 1 wrong → mastery (3+1)/(4+2) = 0.6667
        db.record_exposure("p", [], outcomes=[
            ("haus", True), ("haus", True), ("haus", True), ("haus", False),
        ])
        entry = db.get_entries("p")[0]
        assert entry["total_correct"] == 3
        assert entry["total_wrong"] == 1
        assert entry["mastery_score"] == pytest.approx(0.6667, abs=0.001)

    def test_outcomes_without_words(self, db):
        db.add_words("p", ["haus"])
        db.record_exposure("p", [], outcomes=[("haus", True)])
        entry = db.get_entries("p")[0]
        assert entry["total_correct"] == 1
        assert entry["frequency"] == 1  # not bumped (no exposure)

    def test_migration_imports_and_removes_csv(self, db, tmp_path, monkeypatch):
        from src import vocab_db
        # Point the migration at a temp data dir with two profile CSVs
        data_dir = tmp_path / "data"
        for profile, rows in {
            "krystof": [("Haus", "house", "3", "2026-01-01", "5", "1", "0.75")],
            "johi": [("ciao", "hello", "1", "2026-01-02", "", "", "")],
        }.items():
            pdir = data_dir / profile
            pdir.mkdir(parents=True)
            with open(pdir / "vocabulary.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["word", "meaning", "frequency", "last_seen",
                            "total_correct", "total_wrong", "mastery_score"])
                for row in rows:
                    w.writerow(row)
        monkeypatch.setattr(vocab_db, "PROJECT_DIR", tmp_path)

        result = db.migrate_csv()
        assert result == {"johi": 1, "krystof": 1}

        # CSVs are gone
        assert not list((data_dir).glob("*/vocabulary.csv"))

        # Data survived (including SRS counters)
        k = db.get_entries("krystof")[0]
        assert k["frequency"] == 3
        assert k["last_seen"] == "2026-01-01"
        assert k["total_correct"] == 5
        assert k["mastery_score"] == pytest.approx(0.75)

    def test_migration_idempotent(self, db, tmp_path, monkeypatch):
        from src import vocab_db
        monkeypatch.setattr(vocab_db, "PROJECT_DIR", tmp_path)
        assert db.migrate_csv() == {}  # no CSVs → nothing happens


class TestSharedInstance:
    """Process-wide shared VocabDB (bot loop + web UI threads)."""

    def test_get_shared_db_returns_same_instance(self, tmp_path, monkeypatch):
        from src import vocab_db
        monkeypatch.setattr(vocab_db, "DEFAULT_DB_PATH",
                            tmp_path / "shared.db")
        a = vocab_db.get_shared_db()
        b = vocab_db.get_shared_db()
        assert a is b
        assert (tmp_path / "shared.db").exists()

    def test_reset_shared_db_closes_and_recreates(self, tmp_path, monkeypatch):
        from src import vocab_db
        monkeypatch.setattr(vocab_db, "DEFAULT_DB_PATH",
                            tmp_path / "shared.db")
        a = vocab_db.get_shared_db()
        a.add_words("p", ["haus"])
        vocab_db.reset_shared_db()
        b = vocab_db.get_shared_db()
        assert a is not b  # fresh instance (data persists in the file)
        assert b.word_count("p") == 1


class TestUpdateMastery:
    def test_correct_increases(self):
        from src.vocab_db import update_mastery
        c, w, m = update_mastery(0, 0, True)
        assert (c, w) == (1, 0)
        assert m == pytest.approx(2 / 3, abs=0.001)

    def test_wrong_increases(self):
        from src.vocab_db import update_mastery
        c, w, m = update_mastery(0, 0, False)
        assert (c, w) == (0, 1)
        assert m == pytest.approx(1 / 3, abs=0.001)
