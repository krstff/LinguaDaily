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


class TestSearchAndDelete:
    """search_entries / delete_by_ids / delete_profile / profile_counts."""

    @pytest.fixture
    def seeded(self, db):
        db.add_words("krystof", [
            {"word": "Haus", "meaning": "house"},
            {"word": "Auto", "meaning": "car"},
            {"word": "Politiker", "meaning": "politician"},
        ])
        db.add_words("johi", [{"word": "ciao", "meaning": "hello"}])
        # Bump Haus frequency so sorting is meaningful
        db.record_exposure("krystof", ["Haus", "Haus"])
        db.record_exposure("krystof", [], outcomes=[("Haus", True)])
        return db

    def test_profile_counts(self, seeded):
        assert seeded.profile_counts() == {"krystof": 3, "johi": 1}

    def test_search_returns_total_and_entries_with_id(self, seeded):
        total, entries = seeded.search_entries("krystof")
        assert total == 3
        assert len(entries) == 3
        assert all("id" in e and e["id"] > 0 for e in entries)

    def test_search_filters_case_insensitive_word_and_meaning(self, seeded):
        total, _ = seeded.search_entries("krystof", search="polit")
        assert total == 1
        total, entries = seeded.search_entries("krystof", search="HOUSE")
        assert total == 1
        assert entries[0]["word"] == "Haus"
        total, _ = seeded.search_entries("krystof", search="zzz-nope")
        assert total == 0

    def test_search_scoped_to_profile(self, seeded):
        total, _ = seeded.search_entries("johi", search="ciao")
        assert total == 1
        total, _ = seeded.search_entries("krystof", search="ciao")
        assert total == 0

    def test_search_pagination(self, db):
        # Zero-padded so lexicographic word order == numeric order
        db.add_words("p", [{"word": f"w{i:02d}", "meaning": f"m{i}"} for i in range(25)])
        total, page1 = db.search_entries("p", sort="word", limit=10, offset=0)
        assert total == 25
        assert [e["word"] for e in page1] == [f"w{i:02d}" for i in range(10)]
        total, page3 = db.search_entries("p", sort="word", limit=10, offset=20)
        assert [e["word"] for e in page3] == [f"w{i:02d}" for i in range(20, 25)]

    def test_search_sort_and_order(self, seeded):
        # Haus has frequency 3 (2 exposures + 1 add), others 1
        _, entries = seeded.search_entries("krystof", sort="frequency", order="desc")
        assert entries[0]["word"] == "Haus"
        _, entries = seeded.search_entries("krystof", sort="word", order="asc")
        assert [e["word"] for e in entries] == sorted(
            ["Haus", "Auto", "Politiker"], key=str.lower)

    def test_search_invalid_sort_falls_back_to_id(self, seeded):
        total, entries = seeded.search_entries("krystof", sort="DROP TABLE")
        assert total == 3  # no SQL injection, just default ordering

    def test_delete_by_ids(self, seeded):
        _, entries = seeded.search_entries("krystof")
        by_word = {e["word"]: e["id"] for e in entries}
        deleted = seeded.delete_by_ids("krystof", [by_word["Haus"], by_word["Auto"]])
        assert deleted == 2
        assert seeded.word_count("krystof") == 1
        # johi untouched
        assert seeded.word_count("johi") == 1

    def test_delete_by_ids_scoped_to_profile(self, seeded):
        """An id from another profile must not be deleted via this profile."""
        _, johi_entries = seeded.search_entries("johi")
        johi_id = johi_entries[0]["id"]
        deleted = seeded.delete_by_ids("krystof", [johi_id])
        assert deleted == 0
        assert seeded.word_count("johi") == 1

    def test_delete_by_ids_empty_and_unknown(self, seeded):
        assert seeded.delete_by_ids("krystof", []) == 0
        assert seeded.delete_by_ids("krystof", [999999]) == 0
        assert seeded.word_count("krystof") == 3

    def test_delete_profile(self, seeded):
        assert seeded.delete_profile("krystof") == 3
        assert seeded.word_count("krystof") == 0
        assert seeded.profile_counts() == {"johi": 1}


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
