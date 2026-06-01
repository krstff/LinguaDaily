# Plan: RAG-Infused Language Tutoring (Modular Approach)

## Objective
Integrate Retrieval-Augmented Generation (RAG) into LinguaDaily to provide grounded, textbook-accurate tutoring. RAG logic resides in a dedicated module (`src/rag_service.py`), with a separate Web UI for document management (`src/rag_ui.py`).

## Architecture Overview

### Embedding Backend
- **OpenAI-compatible API only** — same llama.cpp/Ollama server used for LLM + TTS
- Calls `/v1/embeddings` with the model selected on the Dashboard (e.g. `nomic-embed-text`)
- No local sentence-transformers dependency — one server handles everything

### Vector Store
- **Qdrant** (self-hosted, Docker) — configurable URL via Documents page
- Collection auto-created on first use; dimension probed from embedding API

### Document Management Web UI (`/documents`)
- Separate Flask Blueprint (`src/rag_ui.py`), not mixed into `web_ui.py`
- Settings panel: Qdrant URL, chunk size, chunk overlap + Test Connection button
- Upload form: file picker + language dropdown (from config) + optional tags
- Indexed documents table: filter by language, re-index, delete
- Raw files stored in `data/documents/`

### Model Selection (Dashboard)
- Embedding model selector added to the Dashboard model panel alongside Translate/Tutor/TTS
- Saved under `"rag": {"embedding_model": "..."}` in `config.json`
- Populated from the same `/v1/models` list as other models

## Config Schema (`config.json`)

```json
{
  "llm": {
    "base_url": "http://localhost:8080/v1",
    "translate_model": "...",
    "tutor_model": "..."
  },
  "rag": {
    "embedding_model": "nomic-embed-text",
    "qdrant_url": "http://localhost:6333",
    "chunk_size": 500,
    "chunk_overlap": 100
  }
}
```

The embedding API URL always reuses `llm.base_url` — no separate config needed.

---

## Implementation Status

### Phase 1: Foundation ✅ DONE

- [x] **`src/rag_service.py`** — Embedding (API), chunking, Qdrant upsert/query/delete
- [x] **`scripts/ingest_textbooks.py`** — CLI to ingest PDF/TXT/DOCX files
- [x] **`src/rag_ui.py`** — Flask Blueprint for `/documents` page + API endpoints
- [x] **`src/templates/documents.html`** — Documents management UI (settings, upload, table)
- [x] **Dashboard integration** — Embedding model selector in model panel (`base.html`, `dashboard.html`, `web_ui.py`)
- [x] **Navigation** — "Documents" link added to nav bar (`base.html`)
- [x] **`main.py` wiring** — `register_rag_ui()` called when web UI starts
- [x] **`requirements.txt`** — qdrant-client, pdfplumber, python-docx added
- [x] **Qdrant compatibility fix** — `query_points(query_filter=)` → `search(query_filter=)` + client-side filtering in `list_sources()`

### Phase 2: Real-time Tutor Integration ✅ DONE

- [x] **`src/llama_client.py`** — RAG-integrated tutor flow:
  - `_classify_intent()` — lightweight LLM call (temp=0.0) returns `chitchat` / `grammar_query` / `vocab_query`
  - `_fetch_rag_context()` — queries RAG knowledge base by language code, graceful fallback to empty list
  - `tutor_chat()` — routes through intent classifier, injects `[Reference Material]` block into system prompt for educational queries
- [x] **Graceful degradation** — if Qdrant is down or no docs indexed, tutor works normally without RAG
- [x] **Logging** — intent label + chunk count logged per message for debugging

### Phase 3: Super Lesson — SKIPPED (not doing this)

### Phase 4: Optimization & Scaling

- [ ] Context management ("Session Context Packets") to keep chat clean
- [ ] Refine intent router accuracy
- [ ] RAG-verified grading for user responses in chat/quiz

---

## Tutor Flow (Phase 2)

```
User message
    │
    ▼
_classify_intent()          ← lightweight LLM call (temp=0.0)
    │                         returns: chitchat | grammar_query | vocab_query
    ├── chitchat ──────────────→ normal tutor (no RAG)
    │
    ├── grammar/vocab ────────→ _fetch_rag_context()
    │                              ↓
    │                          get_contextual_chunks(language=code, top_k=5)
    │                              ↓
    │                          0–5 text chunks
    │
    ▼
tutor_chat() ────────────────→ system prompt + [Reference Material] block
                                   + today's lesson (if any)
                                   + conversation history
```

### External Dependencies

| Service | Setup | Configured Where |
|---|---|---|
| **Qdrant** | `docker run -d -p 6333:6333 qdrant/qdrant` | Documents page → Qdrant URL |
| **Embedding model** | Load on llama.cpp/Ollama server (e.g. `nomic-embed-text`) | Dashboard → Embedding Model dropdown |

## File Inventory

| File | Purpose |
|---|---|
| `src/rag_service.py` | Core RAG module — embedding, chunking, Qdrant operations |
| `src/rag_ui.py` | Flask Blueprint — `/documents` page + API endpoints |
| `src/templates/documents.html` | Documents management UI template |
| `scripts/ingest_textbooks.py` | CLI ingestion script (PDF/TXT/DOCX → Qdrant) |
| `src/web_ui.py` | Dashboard model APIs extended for embedding_model |
| `src/templates/base.html` | Nav link + JS for embedding model in model panel |
| `src/templates/dashboard.html` | Embedding model selector added to model grid |
| `src/main.py` | Wires rag_ui Blueprint into Flask app |
| `requirements.txt` | Added qdrant-client, pdfplumber, python-docx |
