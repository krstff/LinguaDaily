# TTS Module Guide

`src/tts.py` is a wrapper for local **OmniVoice** TTS via an OpenAI-compatible API. It generates WAV audio files from text and handles Wikipedia artifact cleaning so TTS output sounds natural.

## Architecture

```
text (original article in learning_language)
    │
    ▼
sanitize_for_tts()              ← strip reference markers, fix encoding artifacts
    │
    ▼
synthesize()                    ← OpenAI audio.speech.create()
    │
    ▼
output/<profile>/<uuid>.wav     ← streaming response saved to disk
```

## API

### `synthesize(text, language_id, config, output_dir, voice)`

Main entry point. Generates speech from text using the local OmniVoice server.

```python
from src.tts import synthesize

wav_path = synthesize(
    text="Hallo Welt, wie geht es dir?",
    language_id="de",           # ISO language code for TTS
    config=config,              # full config.json dict (or None to load default)
    output_dir="/workspace/output/krystof",  # optional: defaults to output/<default_profile>/
    voice="male",               # optional: defaults to tts.default_voice from config
)

# wav_path → "/workspace/output/krystof/lingua_a1b2c3d4.wav"  or None on failure
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | (required) | Text to synthesize (should be in the target/learning language) |
| `language_id` | str | `"de"` | ISO language code for TTS engine |
| `config` | dict or None | loaded from default path | Full config.json contents |
| `output_dir` | str or None | `output/<default_profile>/` | Directory to write the WAV file |
| `voice` | str or None | `tts.default_voice` from config | Voice name (e.g., `"male"`, `"female"`) |

### Return value

- **Success:** Absolute path to the generated WAV file (string)
- **Failure:** `None` — logs error via `logger.error()`, cleans up partial files

### Config shape

```json
{
  "tts": {
    "base_url": "http://llama-swap:8080/v1",
    "model": "omnivoice",
    "api_key": "",
    "default_voice": "male"
  },
  "default_profile": "krystof"
}
```

## Text Sanitization

### `sanitize_for_tts(text)`

Cleans Wikipedia-extracted text for TTS consumption. Wikipedia articles contain artifacts that make TTS output sound garbled — this function strips or normalizes them:

| Operation | Example |
|-----------|---------|
| Remove reference markers | `[1]`, `[ 2 ]`, `[a]` → removed |
| Replace non-breaking spaces | `\xa0` → regular space |
| Remove Unicode arrows/symbols | `↑`, `↓`, `↔` → removed |
| Collapse multiple blank lines | 3+ newlines → 2 (paragraph boundary) |

```python
from src.tts import sanitize_for_tts

clean = sanitize_for_tts("This is text[1] with ↑ arrows\n\n\nand extra newlines.")
# → "This is text with and extra newlines."
```

## Error Handling

All errors are logged via the standard `logging` module (not `print(stderr)`):

| Scenario | Behavior | Log level |
|----------|----------|-----------|
| `openai` package not installed | Returns `None` gracefully | `logger.warning()` |
| Config load failure | Returns `None` | `logger.error()` |
| Generated file is empty/missing | Returns `None`, cleans up partial file | `logger.warning()` |
| Any other exception | Returns `None`, cleans up partial file | `logger.error()` |

## CLI Usage

```bash
# Basic usage
conda run -n lingua python src/tts.py --config config.json --lang de "Hallo Welt"

# With voice selection
conda run -n lingua python src/tts.py --config config.json --lang de --voice female "Bonjour le monde"

# Override TTS URL
conda run -n lingua python src/tts.py --tts-url http://192.168.100.60:8080/v1 --lang fr "Bonjour"

# Custom output directory
conda run -n lingua python src/tts.py --config config.json --lang de --output-dir /tmp/tts "Test"
```

Outputs structured JSON on success:
```json
{"wav_path": "/workspace/output/krystof/lingua_a1b2c3d4.wav"}
```

On failure:
```json
{"error": "TTS generation failed"}
```

### CLI arguments

| Flag | Shorthand | Description | Default |
|------|-----------|-------------|---------|
| `text` | — | Text to synthesize | (required) |
| `--lang`, `-l` | | Language code for TTS | `de` |
| `--config`, `-c` | | Path to config.json | Default path |
| `--output-dir`, `-o` | | Output directory | `output/<default_profile>/` |
| `--voice`, `-v` | | Voice name | Config's `tts.default_voice` |
| `--tts-url` | | Override TTS base_url | Config's `tts.base_url` |

## Integration with Orchestrator

The orchestrator calls TTS as part of the lesson pipeline (step 2a, runs in parallel with translation):

```python
# Inside orchestrator._generate_tts():
from tts import synthesize
wav_path = synthesize(
    text=content,              # original article text (in learning_language)
    language_id=learning_language,   # e.g., "de" for German articles
    config=self.config,
    output_dir=output_dir,      # output/<profile>/
    voice=profile.get("tts_voice", "male"),
)
```

The WAV path is included in the lesson dict and sent to Telegram as an audio file via `TelegramBot.deliver_lesson()`. The TTS model is selected from `tts.model` in config (default: `omnivoice`) and can be changed via the Web UI model selection panel.
