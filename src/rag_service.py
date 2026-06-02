#!/usr/bin/env python3
"""
RAG Service — standalone retrieval-augmented generation module for LinguaDaily.

Uses an OpenAI-compatible API endpoint (llama.cpp, Ollama, etc.) for embeddings
and Qdrant as the vector store.  Shares the same server as LLM/TTS by default.

All config defaults live in src/config.py — this module just consumes them.

Usage:
    from src.rag_service import RAGService

    rag = RAGService()
    chunks = rag.get_contextual_chunks(
        "What is the subjunctive in Spanish?", language="es", top_k=5
    )
"""

import hashlib
import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger("lingua")


# ── Upload progress tracker (shared across all RAGService instances) ──
#
# Structure: dict[source_file] -> {
#     "status": "extracting" | "chunking" | "embedding" | "upserting" | "done" | "error",
#     "total_batches": int,
#     "completed_batches": int,
#     "message": str,
#     "started_at": float (timestamp),
# }
_upload_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()


class RAGService:
    """Interface between LinguaDaily and the Qdrant knowledge base."""

    def __init__(self):
        from src.config import get_rag_config
        rcfg = get_rag_config()

        self.qdrant_url = rcfg["qdrant_url"]
        self.collection_name = rcfg["collection_name"]
        self.embedding_model = rcfg["embedding_model"]
        self.embedding_base_url = rcfg["embedding_base_url"]
        self.chunk_size = rcfg["chunk_size"]
        self.chunk_overlap = rcfg["chunk_overlap"]

        self.embed_batch_size = rcfg.get("embed_batch_size", 32)

        self.embed_delay_secs = rcfg.get("embed_delay_secs", 0.5)

        self._qdrant_client = None
        # Global lock to prevent concurrent embedding requests from overwhelming llama-swap
        self._embed_lock = threading.Lock()

    # ── Progress tracking helpers (module-level, shared across instances) ──

    @staticmethod
    def _set_progress(source_file: str, **kwargs):
        """Update upload progress for a source file."""
        with _progress_lock:
            if source_file not in _upload_progress:
                _upload_progress[source_file] = {
                    "status": "extracting",
                    "total_batches": 0,
                    "completed_batches": 0,
                    "message": "",
                    "started_at": time.time(),
                }
            _upload_progress[source_file].update(kwargs)

    @staticmethod
    def get_progress(source_file: str = None) -> dict:
        """Get upload progress.

        If source_file is given, return progress for that file.
        If source_file is None, return progress for all active uploads.
        """
        with _progress_lock:
            if source_file:
                entry = _upload_progress.get(source_file)
                if entry:
                    return dict(entry)  # copy
                return {"status": "not_found", "message": f"No progress for '{source_file}'"}
            else:
                result = {}
                for name, entry in _upload_progress.items():
                    if entry["status"] not in ("done", "error"):
                        result[name] = dict(entry)
                return result

    @staticmethod
    def clear_progress(source_file: str):
        """Clear progress for a completed upload."""
        with _progress_lock:
            _upload_progress.pop(source_file, None)

    # ── Lazy init helpers ────────────────────────────────────────

    def _get_qdrant_client(self):
        """Get a Qdrant client (lazy). Does NOT probe embeddings.

        Embedding probing is deferred until an actual upsert/query needs it,
        so page loads and read-only operations stay fast.
        """
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "qdrant-client is required for RAG. "
                    "Install with: pip install qdrant-client"
                )

            self._qdrant_client = QdrantClient(url=self.qdrant_url)
        return self._qdrant_client

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists, probing embeddings only when needed.

        Called by upsert/query methods — NOT during page loads or stats reads.
        """
        from qdrant_client.http import models

        client = self._get_qdrant_client()
        collections = client.get_collections()
        names = [c.name for c in collections.collections]

        if self.collection_name not in names:
            # Collection doesn't exist — must probe to know dimension
            dim = self._probe_embedding_dimension()
            logger.info("Creating Qdrant collection '%s' (dim=%d)…", self.collection_name, dim)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )
        else:
            # Collection exists — check dimension only if we can probe quickly
            info = client.get_collection(self.collection_name)
            existing_dim = info.config.params.vectors.size
            try:
                dim = self._probe_embedding_dimension()
                if existing_dim != dim:
                    logger.warning(
                        "Dimension mismatch: collection=%d, model=%d — recreating '%s'",
                        existing_dim, dim, self.collection_name,
                    )
                    client.delete_collection(self.collection_name)
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=dim, distance=models.Distance.COSINE
                        ),
                    )
            except Exception as e:
                logger.debug("Skipping dimension check (embedding API unavailable): %s", e)
                # Keep existing collection — dim is fine if it was working before

    def _get_embedding_client(self):
        """Get the dedicated embedding client (no retries, long timeout).

        Uses a separate client from the main LLM client because embeddings
        need different settings:
          - NO auto-retries (retries during congestion double llama-swap load)
          - Longer timeout (larger batches take more time)
        """
        from config import get_embedding_client
        return get_embedding_client(base_url=self.embedding_base_url)

    def _probe_embedding_dimension(self) -> int:
        """Probe the embedding dimension by sending a test vector."""
        try:
            client = self._get_embedding_client()
            resp = client.embeddings.create(
                model=self.embedding_model,
                input=["dimension_probe"],
            )
            return resp.data[0].embedding.__len__()
        except Exception as e:
            logger.warning("Could not probe embedding dimension (%s) — defaulting to 768", e)
            return 768

    # ── Public API: Embeddings ─────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Convert text into a vector embedding via the OpenAI-compatible API."""
        client = self._get_embedding_client()
        resp = client.embeddings.create(model=self.embedding_model, input=[text])
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = None, source_file: str = "") -> list[list[float]]:
        """Embed multiple texts in small batches with congestion-aware backpressure.

        llama-swap has a limited internal SSE send buffer.  Sending requests faster
        than llama-swap can drain its buffer causes:
            '[WARN] handleAPIEvents sendBuffer full, dropped message'
        which freezes the model provider for ALL clients (chat, TTS, etc).

        Once this happens, llama-swap enters a degraded state where it can still
        serve ONE request every ~60 s but collapses under rapid-fire traffic.

        llama.cpp uses a slot-based architecture (n_parallel slots, default ~4).
        When all slots are busy, requests queue indefinitely with no timeout.
        A hanging request will block the caller forever unless we enforce our own.

        This method uses strict sequential processing + per-request timeouts:
          1. Small batch sizes (default 16 texts per request)
          2. Strictly sequential — waits for full response before sending next
          3. Per-batch timeout (120s) — prevents indefinite hangs from llama.cpp
             slot queue
          4. Congestion detection via response time vs. baseline
          5. Recovery mode with exponential backoff (15s → 30s → 60s)
          6. Global threading lock — only ONE caller sends at a time

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        batch_size : int or None
            Maximum number of texts per API call.  Defaults to self.embed_batch_size.

        Returns
        -------
        list[list[float]]
            Embeddings in the same order as input texts.
        """
        if batch_size is None:
            batch_size = self.embed_batch_size

        client = self._get_embedding_client()
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Per-batch timeout — prevents indefinite hangs from llama.cpp slot queue
        BATCH_TIMEOUT_SECS = 120.0

        # Recovery mode parameters
        RECOVERY_INITIAL_DELAY = 15.0   # seconds — first recovery wait after congestion
        RECOVERY_MAX_DELAY = 60.0       # hard cap on recovery delay
        CONGESTION_THRESHOLD = 2.0      # response time > 2× baseline = congested
        RECOVERY_OK_THRESHOLD = 1.3     # probe response < 1.3× baseline = recovered

        with self._embed_lock:
            baseline_elapsed: Optional[float] = None
            recovery_delay = RECOVERY_INITIAL_DELAY
            in_recovery = False
            completed = 0  # track successfully embedded batches (for progress, excludes retries)

            for i, offset in enumerate(range(0, len(texts), batch_size)):
                batch = texts[offset : offset + batch_size]

                # Wait before sending (except the very first batch)
                if i > 0:
                    wait_label = "RECOVERY" if in_recovery else "normal"
                    delay = recovery_delay if in_recovery else self.embed_delay_secs
                    logger.info(
                        "Embed — waiting %.1f s before batch %d/%d (%s)",
                        delay, i + 1, total_batches, wait_label,
                    )
                    time.sleep(delay)

                # Send with per-batch timeout to prevent indefinite hangs
                start = time.monotonic()
                try:
                    resp = client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                        timeout=BATCH_TIMEOUT_SECS,
                    )
                except Exception as e:
                    elapsed_err = time.monotonic() - start
                    error_msg = str(e)
                    # Check if this is a timeout
                    if any(kw in error_msg.lower() for kw in ("timeout", "timed out")):
                        logger.error(
                            "Embed batch %d/%d — TIMEOUT after %.1fs (llama.cpp slot queue likely full). "
                            "Entering recovery mode (delay=%.1fs)",
                            i + 1, total_batches, elapsed_err, RECOVERY_INITIAL_DELAY,
                        )
                        in_recovery = True
                        recovery_delay = RECOVERY_INITIAL_DELAY
                        continue  # retry this batch after recovery delay
                    else:
                        logger.error(
                            "Embed batch %d/%d — ERROR after %.1fs: %s",
                            i + 1, total_batches, elapsed_err, error_msg[:200],
                        )
                        raise

                elapsed = time.monotonic() - start

                embeddings = sorted(resp.data, key=lambda d: d.index)
                all_embeddings.extend([e.embedding for e in embeddings])
                completed += 1

                # ── Report progress to UI ────────────────────────────
                if source_file:
                    pct = int(completed / total_batches * 100)
                    self._set_progress(
                        source_file,
                        status="embedding",
                        total_batches=total_batches,
                        completed_batches=completed,
                        message=f"Embedding: {completed}/{total_batches} batches ({pct}%)",
                    )

                # ── Congestion detection via response-time vs. baseline ──
                if i == 0 and baseline_elapsed is None:
                    # First batch establishes the baseline (cold start is normal)
                    baseline_elapsed = elapsed
                    logger.info(
                        "Embed batch 1/%d — %.1fs (baseline) — next delay: %.2fs",
                        total_batches, elapsed, self.embed_delay_secs,
                    )
                else:
                    ratio_to_baseline = elapsed / baseline_elapsed if baseline_elapsed else 1.0

                    if in_recovery:
                        # We're in recovery mode — this batch is a probe.
                        if ratio_to_baseline < RECOVERY_OK_THRESHOLD:
                            logger.info(
                                "Embed batch %d/%d — %.1fs (×%.1f baseline) — "
                                "✅ RECOVERED, resuming normal pace",
                                i + 1, total_batches, elapsed, ratio_to_baseline,
                            )
                            in_recovery = False
                            recovery_delay = RECOVERY_INITIAL_DELAY
                        else:
                            recovery_delay = min(recovery_delay * 2, RECOVERY_MAX_DELAY)
                            logger.warning(
                                "Embed batch %d/%d — %.1fs (×%.1f baseline) — "
                                "❌ still congested, next recovery delay: %.1fs",
                                i + 1, total_batches, elapsed, ratio_to_baseline,
                                recovery_delay,
                            )
                    elif ratio_to_baseline > CONGESTION_THRESHOLD:
                        in_recovery = True
                        logger.warning(
                            "Embed batch %d/%d — %.1fs (×%.1f baseline) — "
                            "🚨 CONGESTION DETECTED, entering recovery mode (delay=%.1fs)",
                            i + 1, total_batches, elapsed, ratio_to_baseline,
                            recovery_delay,
                        )
                    else:
                        logger.debug(
                            "Embed batch %d/%d — %.1fs (×%.1f baseline)",
                            i + 1, total_batches, elapsed, ratio_to_baseline,
                        )

        return all_embeddings

    # ── PDF text cleanup ──────────────────────────────────────────

    @staticmethod
    def clean_pdf_text(text: str) -> str:
        """Remove common PDF extraction artifacts from raw text.

        Handles:
          • Lone page numbers on their own line (e.g. '42', '103')
          • Repeated running headers / footers (same line appearing 3+ times)
          • Widows & orphans — single short words on their own line (< 5 chars)
          • Lines that are only punctuation, dashes, dots, etc.
          • Excessive blank lines (collapse runs of 3+ into 2)
          • Hyphenation artifacts from line-break word splits

        Parameters
        ----------
        text : str
            Raw extracted text (from pdfplumber, etc.).

        Returns
        -------
        str
            Cleaned text suitable for chunking and embedding.
        """
        lines = text.split("\n")
        cleaned: list[str] = []

        # ── Count occurrences to detect repeated headers/footers ──
        line_counts: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if stripped:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        repeated_lines = {
            line for line, count in line_counts.items() if count >= 3 and len(line) < 200
        }

        for line in lines:
            stripped = line.strip()

            # Skip empty lines (handled later)
            if not stripped:
                cleaned.append("")
                continue

            # Skip repeated running headers / footers
            if stripped in repeated_lines:
                continue

            # Skip lone page numbers (1-4 digits)
            if re.fullmatch(r'\d{1,4}', stripped):
                continue

            # Skip lines that are only punctuation, dashes, dots, asterisks
            if re.fullmatch(r'[\s\u2014\u2013\-\*\.=~\\|/;:!,?@#%^&()+\[\]{}<>"\']{1,}', stripped):
                continue

            # Skip widows/orphans — single very short words (< 5 chars) on their own line
            if len(stripped.split()) == 1 and len(stripped) < 5:
                word = stripped.rstrip(".,;:!?'")
                if not re.fullmatch(r'IV{0,3}|V?I{0,3}\.?', word.upper()) and word.lower() not in (
                    "yes", "no", "the", "and", "for", "but", "not", "all", "can",
                    "had", "has", "was", "are", "his", "her", "its", "our",
                ):
                    continue

            cleaned.append(stripped)

        # ── Post-processing on joined text ────────────────────────
        result = "\n".join(cleaned)

        # Fix hyphenation artifacts: word split across lines with trailing hyphen
        result = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', result)

        # Collapse runs of 3+ blank lines into exactly 2
        result = re.sub(r'\n{4,}', '\n\n\n', result)

        # Remove trailing whitespace from each line
        result = "\n".join(line.rstrip() for line in result.split("\n"))

        # Collapse multiple spaces within lines to single space
        result = re.sub(r'(?<!\n) {2,}(?!\n)', ' ', result)

        return result.strip()

    # ── Chunking ───────────────────────────────────────────────────

    def chunk_text(self, text: str, source_id: str = "") -> list[dict]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if not chunk_text:
                start = end
                continue

            # Try to break at a sentence boundary
            if end < len(text):
                next_newline = chunk_text.rfind("\n\n")
                next_period = chunk_text.rfind(". ")
                break_point = max(next_newline, next_period)
                if break_point > self.chunk_size // 2:
                    chunk_text = chunk_text[:break_point].strip()

            chunk_id = hashlib.sha256(f"{source_id}:{idx}".encode()).hexdigest()[:16]

            chunks.append({
                "text": chunk_text,
                "chunk_index": idx,
                "source_id": source_id,
                "id": chunk_id,
            })

            start = end - self.chunk_overlap
            idx += 1

        return chunks

    # ── Upsert / Indexing ─────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[dict],
        language: str = "",
        source_file: str = "",
        tags: list[str] = None,
    ) -> int:
        """Embed and upsert chunks into Qdrant."""
        if not chunks:
            return 0

        self._ensure_collection()
        client = self._get_qdrant_client()
        texts = [c["text"] for c in chunks]

        # Start progress tracking
        total_batches = (len(texts) + self.embed_batch_size - 1) // self.embed_batch_size
        self._set_progress(
            source_file,
            status="embedding",
            total_batches=total_batches,
            completed_batches=0,
            message=f"Embedding: 0/{total_batches} batches (0%)",
        )

        embeddings = self.embed_batch(texts, batch_size=self.embed_batch_size, source_file=source_file)

        from qdrant_client.http import models

        # Report upserting progress
        self._set_progress(
            source_file,
            status="upserting",
            message=f"Upserting {len(chunks)} chunks into Qdrant…",
        )

        points = []
        for i, chunk in enumerate(chunks):
            payload = {
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "source_id": chunk["source_id"],
                "language": language,
                "source_file": source_file,
                "tags": tags or [],
            }
            points.append(
                models.PointStruct(
                    id=int(chunk["id"], 16),
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Upserted %d chunks for '%s' (lang=%s)", len(points), source_file, language)

        # Mark as done
        self._set_progress(
            source_file,
            status="done",
            completed_batches=total_batches,
            message=f"Done — {len(points)} chunks indexed",
        )

        return len(points)

    # ── Querying ───────────────────────────────────────────────────

    def query_knowledge_base(
        self,
        query_vector: list[float],
        top_k: int = 5,
        language: str = "",
        tags: list[str] = None,
    ) -> list[dict]:
        """Semantic search in Qdrant."""
        self._ensure_collection()
        client = self._get_qdrant_client()

        from qdrant_client.http import models as http_models

        # Build filter conditions
        must_conditions = []
        if language:
            must_conditions.append(
                http_models.FieldCondition(key="language", match=http_models.MatchValue(value=language))
            )
        if tags:
            for tag in tags:
                must_conditions.append(
                    http_models.FieldCondition(key="tags", match=http_models.MatchValue(value=tag))
                )

        search_filter = http_models.Filter(must=must_conditions) if must_conditions else None

        # Detect the correct API by inspecting method signatures at runtime.
        import inspect

        results = None
        error_log = []

        # Strategy 1: query_points() — try all known filter param names
        if hasattr(client, "query_points"):
            sig = inspect.signature(client.query_points)
            params = set(sig.parameters.keys())
            for filter_kw in ("query_filter", "filter", "search_filter"):
                if filter_kw in params and "query" in params:
                    kwargs = {
                        "collection_name": self.collection_name,
                        "query": query_vector,
                        "limit": top_k,
                    }
                    if search_filter:
                        kwargs[filter_kw] = search_filter
                    try:
                        results = client.query_points(**kwargs)
                        break
                    except Exception as e:
                        error_log.append(f"query_points({filter_kw}=): {e}")

        # Strategy 2: search() with query_vector=
        if results is None and hasattr(client, "search"):
            sig = inspect.signature(client.search)
            params = set(sig.parameters.keys())
            for filter_kw in ("query_filter", "filter", "search_filter"):
                if "query_vector" in params:
                    kwargs = {
                        "collection_name": self.collection_name,
                        "query_vector": query_vector,
                        "limit": top_k,
                    }
                    if search_filter and filter_kw in params:
                        kwargs[filter_kw] = search_filter
                    try:
                        results = client.search(**kwargs)
                        break
                    except Exception as e:
                        error_log.append(f"search({filter_kw}=): {e}")

        # Strategy 3: query_points() without filter (last resort)
        if results is None and hasattr(client, "query_points"):
            try:
                results = client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                )
            except Exception as e:
                error_log.append(f"query_points(no filter): {e}")

        if results is None:
            raise RuntimeError(
                f"No working Qdrant search method found. Tried:\n" +
                "\n".join("  - " + e for e in error_log)
            )

        hits = []
        for hit in getattr(results, "points", results):
            payload = getattr(hit, "payload", {}) or {}
            hits.append({
                "text": payload.get("text", ""),
                "source_id": payload.get("source_id", ""),
                "source_file": payload.get("source_file", ""),
                "language": payload.get("language", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "score": getattr(hit, "score", 0),
            })

        return hits

    def get_contextual_chunks(
        self,
        query_text: str,
        language: str = "",
        tags: list[str] = None,
        top_k: int = 5,
    ) -> list[str]:
        """High-level wrapper: embed query + retrieve relevant text chunks."""
        query_vector = self.embed_text(query_text)
        hits = self.query_knowledge_base(
            query_vector=query_vector, top_k=top_k, language=language, tags=tags,
        )
        return [h["text"] for h in hits]

    # ── Deletion / Management ──────────────────────────────────────

    def delete_by_source(self, source_id: str) -> bool:
        """Delete all chunks belonging to a source document (by chunk-level source_id)."""
        client = self._get_qdrant_client()

        from qdrant_client.http import models
        from qdrant_client import models as qm

        client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id))
                ])
            ),
        )
        logger.info("Deleted chunks for source_id='%s'", source_id)
        return True

    def delete_by_source_file(self, source_file: str) -> int:
        """Delete all chunks belonging to a source file. Returns count of deleted points."""
        client = self._get_qdrant_client()

        from qdrant_client.http import models
        from qdrant_client import models as qm

        result = client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    qm.FieldCondition(key="source_file", match=qm.MatchValue(value=source_file))
                ])
            ),
        )
        deleted = getattr(result, "deleted", 0) if result else 0
        logger.info("Deleted %d chunks for source_file='%s'", deleted, source_file)
        return deleted

    def delete_by_language(self, language: str) -> bool:
        """Delete all chunks for a given language."""
        client = self._get_qdrant_client()

        from qdrant_client.http import models
        from qdrant_client import models as qm

        client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    qm.FieldCondition(key="language", match=qm.MatchValue(value=language))
                ])
            ),
        )
        logger.info("Deleted all chunks for language='%s'", language)
        return True

    def _scroll_sources(self, language: str = "") -> list[dict]:
        """Scroll only the payload fields needed for stats/sources.

        Fetches minimal payload (language, source_file, tags, chunk_index)
        in paginated batches of 256. No vectors, no full text content.
        """
        client = self._get_qdrant_client()
        from qdrant_client.http import models

        filter_must = []
        if language:
            filter_must.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )
        search_filter = models.Filter(must=filter_must) if filter_must else None

        offset = None
        all_points: list[dict] = []
        batch_size = 256

        while True:
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=["language", "source_file", "tags", "chunk_index"],
                with_vectors=False,
                scroll_filter=search_filter,
            )

            points = getattr(scroll_result, "points", scroll_result[0]) if isinstance(scroll_result, tuple) else getattr(scroll_result, "points", [])
            if not points:
                break

            for pt in points:
                payload = getattr(pt, "payload", {}) or {}
                all_points.append(payload)

            last = points[-1]
            offset = getattr(last, "id", None)
            if len(points) < batch_size:
                break

        return all_points

    def get_document_stats(self) -> dict:
        """Return stats about indexed documents.

        Uses server-side count for total_chunks (instant),
        then scrolls only needed payload fields for breakdowns.
        """
        client = self._get_qdrant_client()

        # Server-side count — no scrolling needed
        try:
            count_result = client.count(collection_name=self.collection_name, exact=True)
            total_chunks = getattr(count_result, "count", 0)
        except Exception as e:
            logger.warning("Failed to count chunks: %s", e)
            total_chunks = 0

        languages: dict[str, int] = {}
        sources: dict[str, int] = {}

        try:
            for payload in self._scroll_sources():
                lang = payload.get("language", "unknown")
                source = payload.get("source_file", "unknown")
                languages[lang] = languages.get(lang, 0) + 1
                sources[source] = sources.get(source, 0) + 1
        except Exception as e:
            logger.warning("Failed to get document stats: %s", e)

        return {"total_chunks": total_chunks, "languages": languages, "sources": sources}

    def list_sources(self, language: str = "") -> list[dict]:
        """List all indexed source documents."""
        sources: dict[str, dict] = {}

        try:
            for payload in self._scroll_sources(language=language):
                lang = payload.get("language", "")
                source_file = payload.get("source_file", "unknown")
                tags = payload.get("tags", [])

                if source_file not in sources:
                    sources[source_file] = {
                        "source_file": source_file,
                        "language": lang,
                        "chunk_count": 0,
                        "tags": list(set(tags)) if tags else [],
                    }
                sources[source_file]["chunk_count"] += 1
        except Exception as e:
            logger.warning("Failed to list sources: %s", e)

        return list(sources.values())

    def get_config(self) -> dict:
        """Return current RAG configuration for the UI."""
        return {
            "qdrant_url": self.qdrant_url,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embed_batch_size": self.embed_batch_size,
            "embed_delay_secs": self.embed_delay_secs,
        }

    def test_connection(self) -> dict:
        """Test Qdrant connectivity."""
        results = {"qdrant": False, "details": ""}
        details = []

        try:
            client = self._get_qdrant_client()
            info = client.get_collections()
            results["qdrant"] = True
            details.append(f"Qdrant OK ({len(info.collections)} collections)")
        except Exception as e:
            details.append(f"Qdrant failed: {e}")

        results["details"] = "; ".join(details)
        return results


# ── Convenience singleton ───────────────────────────────────────────

_default_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the default RAG service instance."""
    global _default_service
    if _default_service is None:
        _default_service = RAGService()
    return _default_service
