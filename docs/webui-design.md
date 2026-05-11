# Web UI Guide

The Web UI is a lightweight Flask admin panel for managing profiles, selecting models, viewing logs, and editing the raw config. It can run standalone or integrated into the main daemon.

## Tech Stack

- **Python + Flask** — reuses `src/config.py` (`load_config`, path constants)
- **HTMX** — dynamic behavior (live log tail, in-place edits, toggle buttons) via HTML attributes
- **Gruvbox-inspired dark theme** — terminal/nvim aesthetic, embedded CSS (~200 lines)
- **Jinja2 templates** — live in `src/templates/`

## Architecture

```
src/web_ui.py   (standalone or importable from main.py)
  ├── GET  /          → Dashboard (profiles + model selection)
  ├── GET  /logs      → Live log viewer (tail style, auto-scroll, color-coded levels)
  ├── GET  /config    → Full config.json editor with validation
  ├── POST /api/*     → CRUD for profiles, model settings, config save
  └── Embedded CSS    → Gruvbox palette, no external framework
```

## Navigation

Top nav bar: **Dashboard** | **Logs** | **Config**

## Pages

### Dashboard (`/`)

The main management page with three sections:

#### Stats row
Card-style counters for total profiles, scheduled (enabled + has schedule), and target languages.

#### Model Selection panel
Three dropdowns populated by calling `/v1/models` on your LLM server:

| Dropdown | Config key | Used by |
|----------|-----------|---------|
| Translation Model | `llm.translate_model` | Article translation + vocabulary extraction |
| Tutoring Model | `llm.tutor_model` | Interactive tutor chat |
| TTS Model | `tts.model` | Text-to-speech synthesis |

Leave a dropdown on "→ default" to use the global `default_model`. Changes apply globally (all profiles) and are saved with the **Save Models** button.

#### Profile table
Columns: Name, Enabled toggle, Language pair, Chat ID, Schedule, TTS voice, Source type, Actions.

- **Enabled column**: ✓ Enabled / ✗ Disabled toggle button — controls whether the profile is scheduled (HTMX POST to `/api/profiles/<name>/toggle`)
- **Disabled rows** are visually dimmed
- **Edit** opens a modal form pre-populated from the row's `data-*` attributes
- **Delete** with HTMX confirmation dialog

#### Add Profile button
Opens a modal form with fields: name, Telegram chat ID, native/learning language, source type, schedule time/timezone, TTS voice, word limits, enabled checkbox.

### Logs (`/logs`)

- Bottom 500 lines of `lingua.log`
- Color-coded severity: DEBUG=gray, INFO=green, WARNING=yellow, ERROR=red
- Auto-refresh every 3s via HTMX swap
- Level filter dropdown (ALL / INFO / WARNING / ERROR)

### Config (`/config`)

- Raw JSON editor showing full `config.json`
- Save validates JSON and writes back with backup (`config.json.bak`)
- For advanced edits: LLM URLs, Kiwix servers, TTS settings, feed catalogue

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profiles` | POST | Add or edit a profile (form-encoded) |
| `/api/profiles/<name>` | DELETE | Delete a profile |
| `/api/profiles/<name>/toggle` | POST | Toggle enabled/disabled state |
| `/api/models/fetch` | GET | Fetch available models from LLM server (`/v1/models`) |
| `/api/models/save` | POST | Save model selections to config.json (JSON body) |
| `/api/models/current` | GET | Return current model selections from config |
| `/api/config/save` | POST | Save raw config.json (form-encoded) |
| `/api/logs/tail` | GET | Tail log file with optional level filter |

## Model Selection Details

The model selection panel works in two steps:

1. **Fetch**: On page load, JS calls `/api/models/fetch`. The backend queries `llm.base_url/v1/models` and `tts.base_url/v1/models`, merges results, and returns sorted lists.
2. **Populate**: Dropdowns are filled with available model IDs. Current selections from config are loaded via `/api/models/current` and the matching options are selected.
3. **Save**: Clicking "Save Models" POSTs `{translate_model, tutor_model, tts_model}` to `/api/models/save`. The backend writes these into `llm.translate_model`, `llm.tutor_model`, and `tts.model` in config.json.

Model resolution follows the existing priority chain (see [LLM Client Guide](llama-client.md)):
1. Profile-level override (`profile.llm_translate_model`)
2. Profile-level generic (`profile.llm_model`)
3. **Global task default** (`llm.translate_model`, `llm.tutor_model`) ← set by Web UI
4. Global default model (`llm.default_model`)

## Gruvbox Palette (dark)

```css
--bg:         #282828   /* background */
--bg0:        #1d2021   /* darker bg */
--bg1:        #3c3836   /* table rows, inputs */
--fg:         #ebdbb2   /* text */
--fg_dim:     #928374   /* muted text */
--green:      #b8bb26   /* success, INFO */
--red:        #fb4934   /* errors, delete */
--yellow:     #fabd2f   /* warnings */
--blue:       #83a598   /* links, active nav */
--purple:     #d3869b   /* accents */
```

## Integration with main.py

The daemon optionally starts the Web UI as an asyncio task:

```bash
# Daemon + Web UI on port 8089
python src/main.py --web-ui
```

Standalone usage:

```bash
# Defaults (localhost:8089, no auth)
python src/web_ui.py --host 127.0.0.1 --port 8089

# Remote access with basic auth
python src/web_ui.py --host 0.0.0.0 --port 8089 --password mypass
```

## Security

- Localhost-only binding by default (does not expose bot token or API keys)
- Optional basic auth via `--password` flag for remote access
- All API endpoints protected by the same auth mechanism

## Application Factory

```python
from src.web_ui import create_app

app = create_app(
    config_path="/path/to/config.json",  # optional
    log_file="/path/to/lingua.log",      # optional
    password="mypass",                   # optional: enables basic auth
)
```
