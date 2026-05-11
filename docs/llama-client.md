# LLM Client Guide

`src/llama_client.py` is the interface to local **llama.cpp** models via an OpenAI-compatible API. It handles translation, vocabulary extraction, and interactive tutoring — the three core language-learning tasks.

Designed for a default single-model setup with optional per-task and per-profile model overrides for future extensibility.

## Architecture

```
LlamaClient
    │
    ├── translate()          ← system prompt + user text → translation
    ├── extract_vocab()      ← system prompt + original + translated → JSON array
    └── tutor_chat()         ← system prompt + history + message → tutor reply
          │
          ▼
    _chat()                  ← OpenAI chat.completions.create()
          │
          ▼
    Local LLM (llama.cpp / vLLM) @ http://localhost:8080/v1
```

All three public methods use the same `_chat()` core — they differ only in system prompt, temperature, and input structure.

## Configuration

### Config shape

```json
{
  "llm": {
    "base_url": "http://localhost:8080/v1",
    "default_model": "gemma-4-26B-language",
    "api_key": "",
    "timeout": 600
  },
  "profiles": {
    "krystof": {
      "llm_model": "other-model",           // optional: override for all tasks
      "llm_translate_model": "...",         // optional: separate model for translation
      "llm_tutor_model": "..."              // optional: separate model for tutoring
    }
  }
}
```

### Environment variable fallbacks

| Config key | Env var | Default |
|-----------|---------|---------|
| `llm.base_url` | `LLAMA_BASE_URL` | `http://localhost:8080/v1` |
| `llm.default_model` | `LLAMA_MODEL` | `gemma-4-26B-language` |
| `llm.timeout` | `LLAMA_TIMEOUT` | `600` (seconds) |

### Model resolution priority

`resolve_model(task)` determines which model to use for a given task. Priority (highest first):

1. **Profile-level task-specific override** — e.g., `profile.llm_translate_model`
2. **Profile-level generic override** — `profile.llm_model`
3. **LLM-level task default** — `llm.translate_model`, `llm.tutor_model` (future extensibility)
4. **Global default model** — `llm.default_model`

```python
client = LlamaClient(config=config, profile_name="krystof")
print(client.resolve_model("translate"))  # follows priority chain
print(client.resolve_model("vocab"))
print(client.resolve_model("tutor"))
```

## Public API

### Initialization

```python
from src.llama_client import LlamaClient

# With config dict
client = LlamaClient(config=config, profile_name="krystof")

# Auto-load from default config.json
client = LlamaClient()  # no profile overrides
```

### `translate(text, source_lang, target_lang)`

Translates text from one language to another. Uses low temperature (0.1) for deterministic output.

```python
translated = client.translate(
    text="Hallo Welt, wie geht es dir?",
    source_lang="de",
    target_lang="en",
)
# → "Hello World, how are you?"
```

**System prompt:** Instructs the model to preserve structure (headings, paragraphs), keep technical terms accurate, and output **only** the translation — no commentary or summaries.

### `extract_vocab(original_text, translated_text, source_lang, target_lang, max_words)`

Extracts useful vocabulary words from an article. Returns a list of dicts with `word`, `meaning`, and `example` fields.

```python
vocab = client.extract_vocab(
    original_text="Der Quantencomputer nutzt...",
    translated_text="The quantum computer uses...",
    source_lang="de",       # language being learned
    target_lang="en",       # user's native language (for definitions)
    max_words=15,
)
# → [
# →   {"word": "Quantencomputer", "meaning": "quantum computer", "example": "..."},
# →   {"word": "nutzt", "meaning": "uses/employs", "example": "..."},
# → ]
```

The LLM receives both the original text and its translation for context. Output is expected as a JSON array — the client strips markdown code fences if present and parses with `json.loads()`. Returns empty list on parse failure.

### `tutor_chat(message, language_name, native_lang, history, max_history)`

Handles interactive tutoring chat messages. Uses higher temperature (0.7) for natural conversation.

```python
reply = client.tutor_chat(
    message="Wie sagt man 'hello' auf Deutsch?",
    language_name="German",
    native_lang="English",
    history=[
        {"role": "user", "content": "What is the dative case?"},
        {"role": "assistant", "content": "The dative case..."},
    ],
    max_history=10,  # trim to last 10 turns (20 messages)
)
```

**System prompt:** Instructs the model to explain grammar/vocabulary clearly, use examples in the target language with translations, be encouraging and patient, and keep responses concise.

### `health_check()`

Checks if the LLM endpoint is reachable by sending a simple "Reply with exactly 'OK'" message.

```python
is_healthy = client.health_check()  # True or False
```

## Internal Methods

### `_chat(messages, model, temperature)`

Core chat completion wrapper. Handles:
- Lazy OpenAI client initialization
- Model name resolution (falls back to `self.default_model`)
- Connection/timeout error detection (short log message instead of huge traceback)
- Returns `None` on any failure

### `_get_client()`

Lazy-initializes the `openai.OpenAI` client with configured `base_url`, `api_key`, and `timeout`. Cached — called once per client instance.

## CLI Usage

```bash
# Health check
conda run -n lingua python src/llama_client.py health

# Translate text
conda run -n lingua python src/llama_client.py translate "Hallo Welt" --source de --target en

# Extract vocabulary
conda run -n lingua python src/llama_client.py vocab "Der Quantencomputer nutzt..." --source de --target en

# Tutor chat
conda run -n lingua python src/llama_client.py chat "Wie sagt man hello?" --lang German --native English

# With profile override
conda run -n lingua python src/llama_client.py translate "text" --profile krystof

# Override LLM URL
conda run -n lingua python src/llama_client.py health --llm-url http://192.168.1.50:8080/v1
```

### CLI arguments

| Flag | Description | Default |
|------|-------------|---------|
| `command` | `translate`, `vocab`, `chat`, or `health` | (required) |
| `text` | Input text / question | (required except for health) |
| `--source`, `-s` | Source language code | `de` |
| `--target`, `-t` | Target language code | `en` |
| `--lang` | Language name (for tutor) | `German` |
| `--native` | Native language (for tutor) | `English` |
| `--profile`, `-p` | Profile for model overrides | None |
| `--config`, `-c` | Path to config.json | Default path |
| `--llm-url` | Override LLM base_url | Config value |

## Error Handling

All public methods return `None` (translate, tutor_chat) or empty list (extract_vocab) on failure. Errors are logged at appropriate levels:

| Error type | Behavior |
|-----------|----------|
| Connection refused / timeout | Logs short error message, returns None/[] |
| LLM returns empty response | Returns None/[] silently |
| JSON parse error (vocab) | Logs warning with raw output snippet, returns [] |
| `openai` package not installed | Logs error on first call, all methods return None/[] |
