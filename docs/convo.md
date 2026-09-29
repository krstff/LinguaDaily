# Conversation Practice (convo)

Short spoken conversations for speaking + listening practice, started with
`/convo` in Telegram. The topic is the profile's latest delivered lesson and
the language level follows the profile's `target_level` (CEFR).

## How it works

```
/convo
  │
  ├─ LLM (task "convo") ──► opening line + translation (JSON)
  ├─ TTS ─────────────────► audio sent with spoiler-hidden text + translation
  │
  user records a voice message
  │
  ├─ ffmpeg ──────────────► OGG/Opus → 16 kHz WAV
  ├─ STT endpoint ────────► transcript
  ├─ LLM (task "convo") ──► feedback + score (0-100) + next line + translation
  ├─ TTS ─────────────────► next audio + spoilers
  │
  … repeats for N turns (default 6, /convo 8) …
  │
  └─ summary: the full exchange (partner + user lines),
              per-turn scores + average
```

User-facing messages are localized to the profile's `native_language`
(the language the user understands — see `_STRINGS` in `src/convo.py`;
English is the fallback for languages without a table).

Each turn is ONE LLM call (feedback + next line together). The prompt
includes the full exchange history (partner + learner lines, plus the
opening line) so the partner responds coherently — with a capable model
this keeps the conversation natural. A small fast model can still be
pointed at it via `llm.task_models.convo` or a per-profile
`llm_convo_model` override.

The final summary lists the whole conversation: for each turn the partner's
line (the one the user answered) followed by the user's transcribed answer
with its score.

Scores are based on the **transcribed words** (grammar / word choice), not on
acoustics — the LLM never hears the audio.

## Partner behaviour by level

* The partner's lines (opening and every follow-up) **always end with a
  question** so the learner always knows what to say next.
* A1/A2 (and `original`) learners get minimal feedback: the most important
  error, if any — at that stage it is enough to understand the language.
* B1+ learners use a dedicated prompt variant (`CONVO_NEXT_SYSTEM_TIPS`)
  whose feedback additionally carries **one short tip on how to say the
  idea more naturally or accurately**, with a better phrasing in the target
  language when possible.  A separate prompt (instead of a conditional
  instruction in one prompt) keeps the behaviour deterministic even on
  small models.

## Configuration

```json
{
  "stt": {
    "base_url": "http://localhost:9002/v1",
    "api_key": "",
    "model": "whisper",
    "timeout": 60
  },
  "llm": {
    "task_models": { "convo": "small-fast-model" }
  },
  "convo": { "turns": 6 }
}
```

* `stt` — any OpenAI-compatible `/v1/audio/transcriptions` endpoint.
  **Required** for `/convo`; without it the command reports that STT is
  not configured.
* `llm.task_models.convo` — optional small model for the conversation
  (falls back to `llm.default_model`).
* `convo.turns` — default number of turns (2–8), overridable per call.
* `/stop` ends an active conversation at any time.
* Audio is ephemeral: TTS lines are deleted right after being sent and
  voice notes right after transcription. Session end (summary, `/stop`,
  timeout, or a new `/convo`) sweeps any stragglers, so nothing
  accumulates in `output/<profile>/convo/`.
* The profile needs `use_tts: true` and a working `tts` section.
* `ffmpeg` must be installed on the daemon host (Telegram voice notes are
  OGG/Opus and are converted to 16 kHz mono WAV before transcription).

## STT endpoint (whisper.cpp example)

`whisper-server` exposes an OpenAI-compatible API that `src/stt.py` speaks
to directly. Example llama-swap entry:

```yml
models:
  whisper:
    name: "Whisper STT"
    cmd: |
      docker run --rm --name ${MODEL_ID} \
      -p 9002:8000 \
      --network ai_stack \
      -v /models/whisper:/app/models \
      ghcr.io/ggml-org/whisper.cpp:latest \
      -m /app/models/ggml-small.bin -l de
    proxy: "http://whisper:9002"
    checkEndpoint: "/v1/models"
```

(Any OpenAI-compatible STT server works — only `base_url` + `model` matter.)

## Telegram

* `/convo [N]` — start a conversation (N = turns, default from config).
* Voice messages are routed to the active conversation; text messages keep
  going to the tutor. A voice message without an active session gets a
  short hint.
* Sessions expire after 10 minutes of inactivity (`CONVO_SESSION_TIMEOUT_SECS`).

## Files

* `src/convo.py` — `ConvoHandler`: session state, prompts, voice pipeline
* `src/stt.py` — OpenAI-compatible transcription client + ffmpeg conversion
* `src/llama_client.py` — `chat_json()` (JSON-tolerant LLM call),
  `resolve_model("convo")` task resolution
* `src/telegram_bot.py` — `/convo` command + voice-message routing
