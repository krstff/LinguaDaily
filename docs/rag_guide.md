# LinguaDaily RAG Guide

> **RAG is fully optional.** The tutor chat works perfectly without it — RAG simply adds textbook grounding for grammar and vocabulary questions. If Qdrant is unavailable or no documents are indexed, the tutor falls back to its own knowledge with zero disruption.

Retrieval-Augmented Generation (RAG) grounds the tutor chat in your own textbooks and learning materials. The tutor queries a vector store before answering grammar or vocabulary questions, so replies are accurate to the source material you've ingested.

---

## Architecture at a Glance

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  llama.cpp   │     │   Qdrant     │     │ Telegram /   │
│  (embeddings)│◄────│  (vector DB) │     │ Web UI       │
└──────────────┘     └──────────────┘     └──────────────┘
       ▲                      ▲                    │
       │ embed_text()         │ query_chunks()     │ upload docs
       └──────────────────────┴────────────────────┘
                     │
                     ▼
              src/rag_service.py
                     │
                     ▼
              src/llama_client.py  (tutor_chat → intent router → RAG)
```

**One server handles everything:** the same llama.cpp instance that runs your LLM and TTS also serves embeddings via `/v1/embeddings`. No separate embedding model process needed.

---

## 1. Provision Qdrant

Qdrant is the vector database. Run it with Docker:

```bash
docker run -d \
  --name linguadaily-qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_data:/qdrant/storage:rw \
  qdrant/qdrant
```

The collection `linguadaily_docs` is auto-created on first document upload. Embedding dimension is probed from the API response — no manual config needed.

**Persistent storage:** adjust the `-v` mount path to wherever you want data stored. Without it, documents vanish when the container is removed.

---

## 2. Load an Embedding Model

The embedding model must be loaded on the same llama.cpp server that handles LLM requests. This is selected via the **Dashboard → Model Selection** panel.

### llama.cpp (server mode)

```bash
# Example: load nomic-embed-text-v1.5-Q4_K_M.gguf alongside your main model
llama-server -m models/nomic-embed-text-v1.5-Q4_K_M.gguf \
  --embedding \
  --port 8080 \
  --host 0.0.0.0
```

If you run multiple models, use a model-swap proxy (e.g., `llama-gateway`) that routes `/v1/embeddings` to the embedding model and `/v1/chat/completions` to the LLM model.

---

## 3. Configure via Web UI

### Set the Embedding Model

Navigate to **Dashboard → Model Selection** and pick the embedding model from the fourth dropdown:

```
┌─────────────┬─────────────┬──────────┬──────────────────┐
│ Translation │ Tutoring    │  TTS     │ Embedding (RAG)  │
└─────────────┴─────────────┴──────────┴──────────────────┘
```

This saves `"embedding_model": "..."` under the `"rag"` key in `config.json`.

### Configure Qdrant URL

Navigate to **Documents** and set:

| Field | Value |
|---|---|
| **Qdrant URL** | `http://localhost:6333` (or your remote host) |
| **Chunk Size** | `500` (characters per chunk, default) |
| **Chunk Overlap** | `100` (overlap between chunks, default) |

Click **Test Connection** to verify Qdrant is reachable. This only tests Qdrant — not the embedding API.

---

## 4. Upload Documents

### Via Web UI

1. Go to **Documents**
2. In the upload form:
   - Pick a file (PDF, TXT, or DOCX)
   - Select the **language** of the document (e.g., German for German textbooks)
   - Optionally add comma-separated tags (e.g., `grammar`, `B1`)
3. Click **Upload & Index**

The file is stored in `data/documents/` and its text chunks are embedded and upserted into Qdrant immediately. The document table below shows all indexed sources with their language, chunk count, and tags.

---

## 5. How the Tutor Uses RAG

When a user sends a message to the tutor, the flow is:

```
User: "What's the difference between accusative and dative?"
    │
    ▼
[Intent Router] → classifies as "grammar_query"
    │
    ▼
[RAG Query]     → searches Qdrant for German grammar chunks, returns top 5
    │
    ▼
[System Prompt] → augmented with [Reference Material] block + today's lesson
    │
    ▼
[Tutor Reply]   → grounded answer from textbook content
```

**Chitchat is skipped:** messages like "Hello", "Thanks!", or "Tell me a joke" bypass RAG entirely — no extra latency, no wasted tokens.

### Intent Categories

| Intent | Triggers RAG? | Examples |
|---|---|---|
| `chitchat` | ❌ No | Greetings, opinions, small talk |
| `grammar_query` | ✅ Yes | Conjugation, cases, tenses, syntax |
| `vocab_query` | ✅ Yes | Word meanings, translations, idioms, usage |

### Graceful Degradation

If any part of RAG fails, the tutor continues normally:

- **Qdrant down** → no chunks fetched, tutor answers from its own knowledge
- **No docs indexed** → empty results, tutor works as usual
- **Embedding API timeout** → logged at DEBUG level, no user-visible error
- **Wrong language filter** → fewer/no matches, tutor still responds

There is zero risk of the tutor breaking because RAG is unavailable.

---

## 6. Managing Documents

### Re-index a Document

If you update a file on disk, click **Re-index** in the documents table to re-process it with current chunking settings.

### Delete a Document

Click **Delete** to remove all chunks from Qdrant and delete the raw file from `data/documents/`.

### Filter by Language

Use the language dropdown above the document table to show only documents for a specific learning language. Useful when managing materials for multiple languages.

---

## 7. Configuration Reference

All RAG settings live under `"rag"` in `config.json`:

```json
{
  "rag": {
    "qdrant_url": "http://localhost:6333",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 500,
    "chunk_overlap": 100
  }
}
```

| Key | Description | Default |
|---|---|---|
| `qdrant_url` | Qdrant server address | `http://localhost:6333` |
| `embedding_model` | Model name for `/v1/embeddings` calls | `nomic-embed-text` |
| `chunk_size` | Characters per text chunk | `500` |
| `chunk_overlap` | Overlap between consecutive chunks | `100` |

The embedding API URL always reuses `llm.base_url` — no separate config needed.

### Environment Variable Overrides

| Variable | Overrides |
|---|---|
| `QDRANT_URL` | `rag.qdrant_url` |
| `EMBEDDING_MODEL` | `rag.embedding_model` |

---

## 8. Recommended Embedding Models

These work well with llama.cpp (GGUF) or Ollama:

| Model | Dimensions | Notes |
|---|---|---|
| `nomic-embed-text` | 768 | Good all-rounder, fast |
| `granite-embedding` | 768 | IBM's model, solid multilingual support |
| `bge-m3` | 1024 | Strong multilingual, larger |
| `mxbai-embed-large` | 1024 | High quality, moderate size |

**Important:** the embedding model must support the OpenAI-compatible `/v1/embeddings` API format. Both llama.cpp and Ollama do this natively.

---

## File Inventory

| File | Purpose |
|---|---|
| `src/rag_service.py` | Core RAG: embedding, chunking, Qdrant upsert/query/delete |
| `src/rag_ui.py` | Flask Blueprint for `/documents` page + API endpoints |
| `src/templates/documents.html` | Documents management UI |
| `src/llama_client.py` | Tutor chat with intent router + RAG grounding |
| `src/config.py` | Shared defaults (`get_rag_config()`, `get_llm_base_url()`) |
