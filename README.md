# LinguaDaily

A lightweight language-learning daemon that delivers daily language lessons via Telegram — fetching articles, translating them with a *local LLM*, generating TTS audio, and providing chat tutoring. Includes a simple web UI for configuration.

## Features

<table>
  <tr>
    <td><b>Daily language lessons</b></td>
    <td><b>Tutor chat</b></td>
    <td><b>Vocab quiz and flashcards</b></td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/48bfcb3e-7dc7-487d-8ffd-bfd872ec39ef" width="320" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/976848d0-f73c-41e9-8dad-227ddbd3575c" width="250" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/61a6d65f-661c-4870-9779-5bdc3529c511" width="320" controls></video>
    </td>
  </tr>
  <tr>
    <td>Recieve articles, TTS, translation and vocabulary daily.</td>
    <td>Ask questions about your lesson or grammar in general.</td>
    <td>Train your vocab knowledge with simple in-chat games.</td>
  </tr>
</table>

## Quick start

### 1. Install dependencies

```bash
conda create lingua -y
conda run -n lingua pip install -r requirements.txt
conda activate lingua
```

### 2. Configure `config.json`

*See config.sample.json*\
I recommend handling profiles through the web UI.

### 3. Start the daemon

```bash
# Daemon only (scheduler + Telegram bot)
python src/main.py --config config.json

# Daemon + Web UI
python src/main.py --config config.json --web-ui

# Web UI standalone (no scheduler/bot)
python src/web_ui.py --host 127.0.0.1 --port 8089
```

The startup banner shows all configured profiles, schedules, and service status:

```
============================================================
  LinguaDaily Standalone Daemon
============================================================
  Config:     /workspace/config.json
  Profiles:   1 (krystof)
  Scheduled:  1 daily lesson(s)
    • krystof          08:00 (Europe/Berlin) → German
  Telegram:   ✅ configured (token: ...ST-TOKEN)
  LLM:        gemma-4-26B-language @ http://llama-swap:8080/v1
============================================================
```
## Connections

This project relies heavily on self hosted services (eg. [Kiwix](https://wiki.kiwix.org/wiki/Main_Page) for wiki articles, locally deployed LLM and TTS, Qdrant for RAG). Altough RSS feed fetching is also supported and any OpenAI API compatible LLM should also work. All connections are setup in the config file. Sources and models can be selected and edited through the web UI.

### Telegram

- Find BotFather: Open Telegram and search for @BotFather.
- Start the Chat: Hit the Start button and send the command /newbot.
- Atfer creating the bot, save the bot token and pass it to the config.
- A command /chatid should print your ChatId to use in your config.

***Security risks***
- I highly recommend using some sort of env passing for the bot token.
- The LLM calls are not sanitized - watch out for prompt injection with the tutor chat.

My command setup:
```
start - Get usega information
another - Requests another article.
flashcards - Show flashcards.
quiz - Play a quiz.
history - Clears chat history on server.
status - Get information about profile status.
chatid - Get your chat id.
profiles - Lists all available profiles.
switch - Switch between profile chats.
```

### Web UI

A very simple web UI is available for easy of use - updates the config.json where everything is configured. \
Features CRUD ops for profiles and articles sources, logs and config editing.

## Yapping

With local LLMs becoming increasingly capable and me being very interested in AI, I wanted to create something I would actually personally use.
At first, I wanted a simple skill for OpenClaw that would help me with learning languages. However, after realizing how bloated and annoying it is to use, I decided to just do a rewrite with the help of:

### Pi.dev

The aim of this project was never about coding an app therefore I don't really care about simply using Pi.dev together with Qwen3.6 27B to basically build the entire project. To me learning how to actually leverage local AIs in a useful way is way more valuable than coding a *simple* app. \
My focus went into how the LLM calling works, which models to use together so I don't get any OOM and learning that instead of wasting thousands of tokens with an unnecessarily complicated harness just to schedule a cron job, I can simply ask Qwen to show me how to do it in a more efficient way. Why make so many calls to a LLM when I only need it for a few steps...

### Choices

My server runs two RTX 3060s 12gb giving me 24gb of VRAM. As of right now the most capable open-weight models for this purpose seem to be: 
- **Gemma4** for translation. I use the MoE 26b Q4_K_M quant with thinking mode turned off. Thinking mode is not really necessary for translation and it is much slower. The dense version would probably be better but slower and most importantly take up more VRAM.
- **OmniVoice** for TTS. It is fast, the sound quality is great and it takes at most about 4gb of VRAM. Sometimes there are issues when English names are in the text as the model forgets that it should not be speaking English for the next few words. 

I recommend doing your own research on model selection. [https://euroeval.com/leaderboards/](https://euroeval.com/leaderboards/)

I went with **Kiwix** because I did not want to rely on an online Wiki API. Setting it up in its own LXC takes less time than downloading the .zim file itself :)

Personally, I use [llama-swap](https://github.com/mostlygeek/llama-swap) for model deployment with the following config for this project:
```yml
models:
  # Split so that the TTS fits on one of the cards
  gemma-4-26B-language:
    cmd: llama-server --port ${PORT} --model /models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -c 32000 -ngl 99 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-p 0.95 --top-k 64 --metrics -ts 37,63 --chat-template-kwargs '{"enable_thinking":false}'
  
  omnivoice:
    name: "OmniVoice TTS"
    cmd: |
      docker run --rm --name ${MODEL_ID} \
      -p 8880:8880 \
      --network ai_stack \
      --gpus 'device=0' \
      --env 'MODEL_ID=k2-fsa/OmniVoice' \
      --env 'DEVICE=cuda:0' \
      -v /models/:/app/models \
      diogod2r/omnivoice-fastapi:latest
    cmdStop: docker stop ${MODEL_ID}
    proxy: "http://omnivoice:8880"
    checkEndpoint: "/health"
  
  granite-embedding:
    cmd: llama-server --port ${PORT} --model /models/granite-embedding-311m-multilingual-r2.gguf -ngl 99 --metrics --embeddings -c 32768 --batch-size 2048 --ubatch-size 512
```

## Environment Check

Before starting the daemon, run the environment health check to verify your setup and connections:

```bash
python src/env_check.py --config config.json
```

## Documentation
So i don't forget how this works :))
- [Daemon (main.py)](docs/daemon.md) — Startup, service wiring, signal handling, systemd/Docker
- [Web UI](docs/webui-design.md) — Dashboard, model selection, config editor, log viewer
- [Orchestrator Guide](docs/orchestrator.md) — Pipeline steps, utility functions, CLI usage
- [Processor (Vocabulary)](docs/processor.md) — Vocab markdown file management
- [Lesson Scheduler Guide](docs/scheduler.md) — Schedule config, enabled/disabled profiles, delivery callback API
- [Telegram Bot Guide](docs/telegram-bot.md) — Setup, commands, tutor chat with lesson context
- [LLM Client Guide](docs/llama-client.md) — Model resolution, translate, vocab extraction, tutor chat
- [TTS Module Guide](docs/tts.md) — OmniVoice wrapper, text sanitization
- [Wikipedia Fetcher Guide](docs/wikipedia-fetcher.md) — Kiwix/ZIM client, HTML extraction, smart truncation
- [RAG Guide](docs/rag_guide.md) — Optional textbook grounding for the tutor chat via Qdrant
