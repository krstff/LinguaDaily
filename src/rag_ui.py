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
    global _documents_dir

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
            text = _extract_text(dest)

            if not text.strip():
                os.remove(str(dest))
                return jsonify({"message": "No text extracted from file"}), 400

            source_id = hashlib.sha256(safe_name.encode()).hexdigest()[:16]
            chunks = rag.chunk_text(text, source_id=source_id)
            upserted = rag.upsert_chunks(
                chunks=chunks,
                language=language,
                source_file=safe_name,
                tags=tags,
            )

            return jsonify({
                "message": f"Uploaded '{safe_name}' — {upserted} chunks indexed",
                "source_id": source_id,
                "chunks": upserted,
                "language": language,
            })

        except ImportError as e:
            return jsonify({"message": f"Missing dependency: {e}"}), 500
        except Exception as e:
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

            rag.delete_by_source(target["source_id"])

            raw_path = _documents_dir / source_file
            if raw_path.exists():
                os.remove(str(raw_path))

            return jsonify({
                "message": f"Deleted '{source_file}' ({target['chunk_count']} chunks removed)"
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

            text = _extract_text(raw_path)
            source_id = hashlib.sha256(source_file.encode()).hexdigest()[:16]
            chunks = rag.chunk_text(text, source_id=source_id)
            upserted = rag.upsert_chunks(
                chunks=chunks,
                language=language,
                source_file=source_file,
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

    app.register_blueprint(bp)
    logger.info("RAG document management UI registered at /documents")
    return bp


# ── Helpers ────────────────────────────────────────────────────────────

def _sanitize_filename(filename: str) -> str:
    """Make a filename safe for filesystem storage."""
    import re
    safe = re.sub(r"[^\w.\-]", "_", filename)
    return Path(safe).name


def _extract_text(filepath: Path) -> str:
    """Extract text from a file based on extension."""
    ext = filepath.suffix.lower()

    if ext == ".txt":
        return filepath.read_text(encoding="utf-8", errors="replace")

    elif ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber required for PDF files. Install with: pip install pdfplumber")

        text_parts = []
        with pdfplumber.open(str(filepath)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    elif ext == ".docx":
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx required for DOCX files. Install with: pip install python-docx")

        doc = Document(str(filepath))
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())

    else:
        raise ValueError(f"Unsupported file type: {ext}")
