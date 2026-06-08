#!/usr/bin/env python3
"""
RAG Document Management Web UI — Flask Blueprint for LinguaDaily.

Handles document upload, language tagging, embedding, indexing, browsing,
deletion and RAG settings (Qdrant URL, embedding model) management.

Routes are mounted at /documents/* and registered via register_rag_ui().

Usage (inside main.py or any Flask app):
    from rag_ui import register_rag_ui
    register_rag_ui(app, config_path)
"""

import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG_PATH, DATA_DIR, load_config

logger = logging.getLogger("lingua")

try:
    from flask import Blueprint, jsonify, render_template, request
except ImportError:
    print("Error: flask is required. Install with: pip install flask", file=sys.stderr)
    sys.exit(1)


# ── Globals ────────────────────────────────────────────────────────

_documents_dir = DATA_DIR / "documents"


def register_rag_ui(app, config_path=None):
    """Register the RAG document management routes on a Flask app."""
    _documents_dir.mkdir(parents=True, exist_ok=True)

    def _get_rag():
        """Lazy-init RAGService, always reading fresh config."""
        from src.rag_service import RAGService
        return RAGService()

    bp = Blueprint(
        "rag_docs",
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
    )

    # ── Pages ────────────────────────────────────────────────────

    @bp.route("/documents")
    def documents_page():
        """Main documents management page."""
        try:
            rag = _get_rag()
            stats = rag.get_document_stats()
            sources = rag.list_sources()
            rag_config = rag.get_config()
        except Exception as e:
            logger.warning("Failed to load document data: %s", e)
            stats = {"total_chunks": 0, "languages": {}, "sources": {}}
            sources = []
            rag_config = {}

        available_languages = []
        try:
            config = load_config(CONFIG_PATH)
            available_languages = sorted(config.get("kiwix_servers", {}).keys())
        except Exception:
            pass

        return render_template(
            "documents.html",
            active="documents",
            stats=stats,
            sources=sources,
            available_languages=available_languages,
            rag_config=rag_config,
        )

    # ── Settings API ─────────────────────────────────────────────

    @bp.route("/api/documents/settings", methods=["GET"])
    def get_settings():
        """Return current RAG settings from config."""
        try:
            config = load_config(CONFIG_PATH)
            rag_cfg = config.get("rag", {})
            return jsonify(rag_cfg)
        except Exception as e:
            return jsonify({"message": f"Failed to read config: {e}"}), 500

    @bp.route("/api/documents/settings", methods=["POST"])
    def save_settings():
        """Save RAG settings to config.json.

        JSON body:
        {
            "qdrant_url": "http://localhost:6333",
            "embedding_model": "nomic-embed-text",
            "embedding_base_url": "",       // empty = reuse LLM URL
            "chunk_size": 500,
            "chunk_overlap": 100
        }
        """
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"message": "Invalid JSON body"}), 400

        try:
            config = load_config(CONFIG_PATH)
            rag_cfg = config.setdefault("rag", {})

            changed = []

            if "qdrant_url" in data:
                rag_cfg["qdrant_url"] = data["qdrant_url"].strip() or "http://localhost:6333"
                changed.append("qdrant_url")

            if "chunk_size" in data:
                try:
                    rag_cfg["chunk_size"] = int(data["chunk_size"])
                    changed.append("chunk_size")
                except (ValueError, TypeError):
                    pass

            if "chunk_overlap" in data:
                try:
                    rag_cfg["chunk_overlap"] = int(data["chunk_overlap"])
                    changed.append("chunk_overlap")
                except (ValueError, TypeError):
                    pass

            if "embed_batch_size" in data:
                try:
                    rag_cfg["embed_batch_size"] = int(data["embed_batch_size"])
                    changed.append("embed_batch_size")
                except (ValueError, TypeError):
                    pass

            if "embed_delay_secs" in data:
                try:
                    rag_cfg["embed_delay_secs"] = float(data["embed_delay_secs"])
                    changed.append("embed_delay_secs")
                except (ValueError, TypeError):
                    pass

            backup = CONFIG_PATH.with_suffix(".json.bak")
            if CONFIG_PATH.exists():
                shutil.copy2(CONFIG_PATH, backup)

            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")

            return jsonify({
                "message": f"RAG settings saved ({', '.join(changed)})",
                "changed": changed,
            })

        except Exception as e:
            logger.error("Failed to save RAG settings: %s", e, exc_info=True)
            return jsonify({"message": f"Write error: {e}"}), 500

    @bp.route("/api/documents/settings/test", methods=["POST"])
    def test_connection():
        """Test Qdrant + embedding API connectivity."""
        try:
            # Force fresh config read by creating new service
            from src.rag_service import RAGService
            rag = RAGService()
            result = rag.test_connection()
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "qdrant": False,
                "embeddings": False,
                "details": f"Error: {e}",
            }), 500

    @bp.route("/api/documents/query", methods=["POST"])
    def query_rag():
        """Test RAG retrieval with a free-text query.

        JSON body:
        {
            "query": "What is the subjunctive?",
            "language": "es",       // optional filter
            "top_k": 5              // optional, default 5
        }

        Returns hits with score, source file, chunk index, and text.
        """
        data = request.get_json(force=True, silent=True)
        if not data or not data.get("query"):
            return jsonify({"message": "Missing 'query' field"}), 400

        query_text = data["query"].strip()
        language = data.get("language", "").strip().lower()
        try:
            top_k = int(data.get("top_k", 5))
        except (ValueError, TypeError):
            top_k = 5

        try:
            from src.rag_service import RAGService
            rag = RAGService()
            query_vector = rag.embed_text(query_text)
            hits = rag.query_knowledge_base(
                query_vector=query_vector,
                top_k=top_k,
                language=language if language else None,
            )
            return jsonify({
                "query": query_text,
                "top_k": top_k,
                "hits": hits,
            })
        except ImportError as e:
            return jsonify({"message": f"Missing dependency: {e}"}), 500
        except Exception as e:
            from src.rag_service import DimensionMismatchError
            if isinstance(e, DimensionMismatchError):
                logger.warning("Query blocked — dimension mismatch: %s", e)
                return jsonify({"message": str(e), "dimension_mismatch": True}), 409
            logger.error("RAG test query failed: %s", e, exc_info=True)
            return jsonify({"message": f"Query failed: {e}"}), 500

    # ── Upload API ───────────────────────────────────────────────

    @bp.route("/api/documents/upload", methods=["POST"])
    def upload_document():
        """Upload and index a document."""
        if "file" not in request.files:
            return jsonify({"message": "No file provided"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"message": "No file selected"}), 400

        language = request.form.get("language", "").strip().lower()
        if not language:
            return jsonify({"message": "Language is required"}), 400

        tags_raw = request.form.get("tags", "").strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

        safe_name = _sanitize_filename(uploaded.filename)
        dest = _documents_dir / safe_name
        uploaded.save(str(dest))

        try:
            rag = _get_rag()
            # Start progress tracking
            rag._set_progress(
                safe_name,
                status="extracting",
                message=f"Extracting text from '{safe_name}'…",
            )

            upserted = rag.ingest_file(
                filepath=dest,
                source_file=safe_name,
                language=language,
                tags=tags,
            )

            source_id = hashlib.sha256(safe_name.encode()).hexdigest()[:16]
            return jsonify({
                "message": f"Uploaded '{safe_name}' — {upserted} chunks indexed",
                "source_id": source_id,
                "chunks": upserted,
                "language": language,
            })

        except ValueError as e:
            # No text extracted — clean up the saved file
            os.remove(str(dest))
            return jsonify({"message": str(e)}), 400
        except ImportError as e:
            return jsonify({"message": f"Missing dependency: {e}"}), 500
        except Exception as e:
            from src.rag_service import DimensionMismatchError
            if isinstance(e, DimensionMismatchError):
                # Clean up the saved file since indexing was blocked
                os.remove(str(dest))
                logger.warning("Upload blocked — dimension mismatch: %s", e)
                return jsonify({"message": str(e), "dimension_mismatch": True}), 409
            logger.error("Upload failed for '%s': %s", safe_name, e, exc_info=True)
            return jsonify({"message": f"Processing failed: {e}"}), 500

    # ── Delete API ───────────────────────────────────────────────

    @bp.route("/api/documents/<source_file>", methods=["DELETE"])
    def delete_document(source_file):
        """Delete a document from Qdrant and remove the raw file."""
        try:
            rag = _get_rag()

            all_sources = rag.list_sources()
            target = None
            for s in all_sources:
                if s["source_file"] == source_file:
                    target = s
                    break

            if not target:
                return jsonify({"message": f"Document '{source_file}' not found"}), 404

            deleted_count = rag.delete_by_source_file(source_file)

            raw_path = _documents_dir / source_file
            if raw_path.exists():
                os.remove(str(raw_path))

            return jsonify({
                "message": f"Deleted '{source_file}' ({deleted_count} chunks removed)"
            })

        except Exception as e:
            logger.error("Delete failed for '%s': %s", source_file, e, exc_info=True)
            return jsonify({"message": f"Delete failed: {e}"}), 500

    # ── Re-index API ─────────────────────────────────────────────

    @bp.route("/api/documents/<source_file>/reindex", methods=["POST"])
    def reindex_document(source_file):
        """Re-process a document: delete old chunks, re-extract and re-embed."""
        raw_path = _documents_dir / source_file
        if not raw_path.exists():
            return jsonify({"message": f"Raw file '{source_file}' not found"}), 404

        language = request.form.get("language", "").strip().lower()
        if not language:
            return jsonify({"message": "Language is required"}), 400

        tags_raw = request.form.get("tags", "").strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

        try:
            rag = _get_rag()

            all_sources = rag.list_sources()
            for s in all_sources:
                if s["source_file"] == source_file:
                    rag.delete_by_source(s["source_id"])
                    break

            # Start progress tracking (mirrors upload flow)
            rag._set_progress(
                source_file,
                status="extracting",
                message=f"Re-indexing '{source_file}'…",
            )

            upserted = rag.ingest_file(
                filepath=raw_path,
                source_file=source_file,
                language=language,
                tags=tags,
            )

            return jsonify({
                "message": f"Re-indexed '{source_file}' — {upserted} chunks",
                "chunks": upserted,
            })

        except Exception as e:
            logger.error("Re-index failed for '%s': %s", source_file, e, exc_info=True)
            return jsonify({"message": f"Re-index failed: {e}"}), 500

    # ── Stats / List APIs ────────────────────────────────────────

    @bp.route("/api/documents/stats")
    def document_stats():
        try:
            rag = _get_rag()
            return jsonify(rag.get_document_stats())
        except Exception as e:
            return jsonify({"message": f"Failed to get stats: {e}"}), 500

    @bp.route("/api/documents/sources")
    def list_sources():
        try:
            rag = _get_rag()
            lang = request.args.get("language", "").strip().lower()
            return jsonify({"sources": rag.list_sources(language=lang)})
        except Exception as e:
            return jsonify({"message": f"Failed to list sources: {e}"}), 500

    # ── Dimension mismatch API ────────────────────────────────

    @bp.route("/api/documents/dimension-check", methods=["GET"])
    def dimension_check():
        """Check if the embedding model dimension matches the collection."""
        try:
            rag = _get_rag()
            mismatch = rag.check_dimension_mismatch()
            return jsonify({"mismatch": mismatch})
        except Exception as e:
            return jsonify({"message": f"Check failed: {e}"}), 500

    @bp.route("/api/documents/reindex-all", methods=["POST"])
    def reindex_all_documents():
        """Recreate the collection and re-process all saved documents from disk.

        Used after changing the embedding model (dimension mismatch).
        """
        data = request.get_json(force=True, silent=True) or {}
        if data.get("confirm") != "REINDEX_ALL":
            return jsonify({"message": "Confirmation required."}), 400

        try:
            rag = _get_rag()
            result = rag.reindex_all_documents(documents_dir=str(_documents_dir))
            return jsonify({
                "message": f"Re-indexed {result['indexed']}/{result['total']} documents",
                **result,
            })
        except Exception as e:
            logger.error("Re-index all failed: %s", e, exc_info=True)
            return jsonify({"message": f"Failed: {e}"}), 500

    @bp.route("/api/documents/upload-progress")
    def upload_progress():
        """Get upload progress for all active uploads.

        Returns a dict of source_file -> progress info:
        {
            "file.pdf": {
                "status": "embedding",
                "total_batches": 19,
                "completed_batches": 7,
                "message": "Embedding: 7/19 batches (37%)",
                "started_at": 1234567890.0
            }
        }
        """
        try:
            rag = _get_rag()
            return jsonify(rag.get_progress())
        except Exception as e:
            return jsonify({"message": f"Failed to get progress: {e}"}), 500

    app.register_blueprint(bp)
    logger.info("RAG document management UI registered at /documents")
    return bp


# ── Helpers ────────────────────────────────────────────────────────────

def _sanitize_filename(filename: str) -> str:
    """Make a filename safe for filesystem storage."""
    import re
    safe = re.sub(r"[^\w.\-]", "_", filename)
    return Path(safe).name



