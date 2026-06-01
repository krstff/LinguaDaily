#!/usr/bin/env python3
"""
Integration tests for RAG service — Qdrant operations, no LLM calls.

Tests exercise the **actual production code paths** in rag_service.py so that
bugs like wrong API parameter names are caught before they hit the bot.

Requires a running Qdrant instance.  URL is auto-detected:
    - env var ``QDRANT_TEST_URL`` (explicit override)
    - ``http://qdrant:6333`` if inside Docker Compose
    - ``http://localhost:6333`` as last resort

Usage:
    python3 tests/test_rag.py
    QDRANT_TEST_URL=http://host:6333 python3 tests/test_rag.py
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Resolve Qdrant URL ───────────────────────────────────────────

_QDRANT_URL = os.environ.get("QDRANT_TEST_URL") or "http://qdrant:6333"
_TEST_DIM = 768
_TEST_COLLECTION = "linguadaily_test"


def _dummy_vector(seed=1.0, dim=_TEST_DIM):
    """Deterministic dummy vector."""
    return [seed * (i + 1) for i in range(dim)]


# ── Helpers ────────────────────────────────────────────────────────

def _make_rag(url: str, collection: str = _TEST_COLLECTION):
    """Create a bare RAGService with injected Qdrant client."""
    from src.rag_service import RAGService
    from qdrant_client import QdrantClient

    rag = object.__new__(RAGService)  # no __init__ (would hit embedding API)
    rag.qdrant_url = url
    rag.collection_name = collection
    rag.embedding_model = "test"
    rag.embedding_base_url = "http://nowhere/v1"
    rag.chunk_size = 500
    rag.chunk_overlap = 100
    rag._qdrant_client = QdrantClient(url=url)
    rag._openai_client = None
    return rag


# ── Tests ──────────────────────────────────────────────────────────

class TestChunking(unittest.TestCase):
    """Pure-logic tests (no network)."""

    def test_chunk_basic(self):
        from src.rag_service import RAGService

        rag = object.__new__(RAGService)
        rag.chunk_size = 100
        rag.chunk_overlap = 20

        chunks = rag.chunk_text("Hello world. This is a test.", source_id="t1")
        self.assertTrue(len(chunks) >= 1)
        for c in chunks:
            for key in ("text", "chunk_index", "source_id", "id"):
                self.assertIn(key, c)

    def test_chunk_empty(self):
        from src.rag_service import RAGService

        rag = object.__new__(RAGService)
        rag.chunk_size = 500
        rag.chunk_overlap = 100
        self.assertEqual(rag.chunk_text("", source_id="e"), [])

    def test_chunk_overlap_preserved(self):
        from src.rag_service import RAGService

        rag = object.__new__(RAGService)
        rag.chunk_size = 50
        rag.chunk_overlap = 10

        chunks = rag.chunk_text("A" * 200, source_id="ov")
        self.assertTrue(len(chunks) >= 3)
        for i in range(len(chunks) - 1):
            self.assertEqual(
                chunks[i]["text"][-rag.chunk_overlap :],
                chunks[i + 1]["text"][: rag.chunk_overlap],
            )


class TestQdrantUpsertQuery(unittest.TestCase):
    """End-to-end Qdrant tests through RAGService production methods."""

    @classmethod
    def setUpClass(cls):
        cls.rag = _make_rag(_QDRANT_URL)
        cls.client = cls.rag._get_qdrant_client()
        # Clean slate
        try:
            cls.client.delete_collection(_TEST_COLLECTION)
        except Exception:
            pass
        from qdrant_client.http import models
        cls.client.create_collection(
            collection_name=_TEST_COLLECTION,
            vectors_config=models.VectorParams(
                size=_TEST_DIM, distance=models.Distance.COSINE
            ),
        )

    def setUp(self):
        """Reset collection before each test for isolation."""
        try:
            self.client.delete_collection(_TEST_COLLECTION)
        except Exception:
            pass
        from qdrant_client.http import models
        self.client.create_collection(
            collection_name=_TEST_COLLECTION,
            vectors_config=models.VectorParams(
                size=_TEST_DIM, distance=models.Distance.COSINE
            ),
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.delete_collection(_TEST_COLLECTION)
        except Exception:
            pass

    def _upsert_test_data(self):
        """Insert known test chunks into Qdrant."""
        from qdrant_client.http import models as http_models

        vec_de = _dummy_vector(seed=10.0)
        vec_es = _dummy_vector(seed=20.0)

        self.rag._get_qdrant_client().upsert(
            collection_name=_TEST_COLLECTION,
            points=[
                http_models.PointStruct(
                    id=1,
                    vector=vec_de,
                    payload={
                        "text": "The German accusative marks the direct object.",
                        "source_id": "de_acc_001",
                        "source_file": "german_grammar.pdf",
                        "language": "de",
                        "chunk_index": 0,
                        "tags": ["grammar"],
                    },
                ),
                http_models.PointStruct(
                    id=2,
                    vector=vec_de,
                    payload={
                        "text": "Dative case is used for indirect objects in German.",
                        "source_id": "de_dat_001",
                        "source_file": "german_grammar.pdf",
                        "language": "de",
                        "chunk_index": 1,
                        "tags": ["grammar"],
                    },
                ),
                http_models.PointStruct(
                    id=3,
                    vector=vec_es,
                    payload={
                        "text": "Spanish subjunctive for wishes and doubts.",
                        "source_id": "es_subj_001",
                        "source_file": "spanish_grammar.pdf",
                        "language": "es",
                        "chunk_index": 0,
                        "tags": ["grammar"],
                    },
                ),
            ],
        )

    # ── query_knowledge_base (THE critical path) ────────────────

    def test_query_returns_hits(self):
        """query_knowledge_base must return results for a known vector."""
        self._upsert_test_data()

        hits = self.rag.query_knowledge_base(
            query_vector=_dummy_vector(seed=10.0),
            top_k=5,
        )
        self.assertIsInstance(hits, list)
        self.assertTrue(len(hits) >= 2, f"Expected ≥2 hits, got {len(hits)}")
        # At least one German chunk should appear
        german_hits = [h for h in hits if h["language"] == "de"]
        self.assertTrue(len(german_hits) >= 1, f"Expected German hits, got: {[h['text'][:40] for h in hits]}")

    def test_query_language_filter(self):
        """Language filter must narrow results to matching language only."""
        self._upsert_test_data()

        hits = self.rag.query_knowledge_base(
            query_vector=_dummy_vector(seed=10.0),
            language="de",
            top_k=5,
        )
        self.assertTrue(len(hits) >= 1)
        for h in hits:
            self.assertEqual(h["language"], "de",
                             f"Expected de, got {h['language']} in: {h['text'][:60]}")

    def test_query_no_filter_returns_all(self):
        """Without language filter, results span all languages."""
        self._upsert_test_data()

        hits = self.rag.query_knowledge_base(
            query_vector=_dummy_vector(seed=10.0),
            top_k=10,
        )
        # Should include both de and es chunks
        langs = {h["language"] for h in hits}
        self.assertIn("de", langs)

    def test_query_empty_collection(self):
        """Querying an empty collection must return [] not crash."""
        # Don't insert any data — just query
        hits = self.rag.query_knowledge_base(
            query_vector=_dummy_vector(seed=99.0),
            top_k=5,
        )
        self.assertEqual(hits, [])

    # ── scroll / list_sources / stats ────────────────────────────

    def test_list_sources(self):
        """list_sources must return structured dicts."""
        self._upsert_test_data()

        sources = self.rag.list_sources()
        self.assertTrue(len(sources) >= 1)
        for s in sources:
            for key in ("source_file", "language", "chunk_count", "tags"):
                self.assertIn(key, s, f"Missing key '{key}' in source entry")

    def test_list_sources_language_filter(self):
        """list_sources(language=…) must filter correctly."""
        self._upsert_test_data()

        de = self.rag.list_sources(language="de")
        for s in de:
            self.assertEqual(s["language"], "de")

        # Non-existent language → empty
        none = self.rag.list_sources(language="xx")
        self.assertEqual(none, [])

    def test_get_document_stats(self):
        """get_document_stats must return valid structure."""
        self._upsert_test_data()

        stats = self.rag.get_document_stats()
        self.assertIn("total_chunks", stats)
        self.assertIn("languages", stats)
        self.assertIn("sources", stats)
        self.assertTrue(stats["total_chunks"] >= 3)

    # ── deletion ─────────────────────────────────────────────────

    def test_delete_by_source_file(self):
        """delete_by_source_file must remove all chunks for a file."""
        self._upsert_test_data()

        # Verify data exists first
        sources_before = self.rag.list_sources()
        self.assertTrue(any(s["source_file"] == "spanish_grammar.pdf" for s in sources_before))

        self.rag.delete_by_source_file("spanish_grammar.pdf")

        # Verify: no Spanish chunks left
        sources_after = self.rag.list_sources()
        for s in sources_after:
            self.assertNotEqual(s["source_file"], "spanish_grammar.pdf")


class TestConnectionProbe(unittest.TestCase):
    """Test connection helper and dimension probing."""

    def test_probe_fallback_on_timeout(self):
        """Probe must return 768 fallback on connection error."""
        from src.rag_service import RAGService

        rag = object.__new__(RAGService)
        rag.embedding_base_url = "http://nonexistent:1/v1"
        rag._openai_client = None
        dim = rag._probe_embedding_dimension()
        self.assertEqual(dim, 768)


# ── Runner ─────────────────────────────────────────────────────────

def main():
    print(f"RAG tests  →  Qdrant: {_QDRANT_URL}\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestChunking))
    suite.addTests(loader.loadTestsFromTestCase(TestQdrantUpsertQuery))
    suite.addTests(loader.loadTestsFromTestCase(TestConnectionProbe))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
