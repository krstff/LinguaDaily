# LinguaDaily Web UI — Design Spec

## Tech Stack
- **Python + Flask** — reuses existing `src/config.py` (`load_config`, path constants)
- **HTMX** — dynamic behavior (live log tail, in-place edits) via HTML attributes, zero JS framework
- **Gruvbox-inspired dark theme** — terminal/nvim aesthetic

## Architecture
```
src/web_ui.py   (standalone file, importable from main.py)
  ├── GET  /          → Dashboard (profiles table + stats)
  ├── GET  /logs      → Live log viewer (tail style, auto-scroll, color-coded levels)
  ├── POST /api/*     → CRUD for config.json profiles
  ├── GET  /config    → Full config.json editor with validation
  └── Embedded CSS    → ~100 lines, gruvbox palette, no external framework
```

## Navigation
- Top nav bar with buttons: **Dashboard** | **Logs** | **Config**
- No URL editing needed — all navigation via clickable buttons in the header

## Pages

### Dashboard (`/`)
- Table of profiles: name, source/target lang, Telegram chat ID, schedule time, TTS voice, source type
- Actions per row: Edit | Enable/Disable | Delete
- "Add Profile" button opens inline form
- Changes persist directly to `config.json`

### Logs (`/logs`)
- Bottom 500 lines of `lingua.log`
- Color-coded severity: DEBUG=gray, INFO=green, WARNING=yellow, ERROR=red
- Auto-refresh every 3s via HTMX swap
- Toggle for follow mode (auto-scroll to bottom)
- Filter dropdown by log level

### Config (`/config`)
- Raw JSON editor showing full `config.json`
- Save validates JSON and writes back with backup (`config.json.bak`)
- For advanced edits: LLM URLs, Kiwix servers, TTS settings, feed catalogue

## Gruvbox Palette (dark)
```css
--bg:         #282828   /* background */
--bg0:        #1d2021   /* darker bg */
--bg1:        #3c3836   /* table rows, inputs */
--fg:         #ebdbb2   /* text */
--fg_dark:    #928374   /* muted text */
--green:      #b8bb26   /* success, INFO */
--red:        #fb4934   /* errors, delete */
--yellow:     #fabd2f   /* warnings */
--blue:       #83a598   /* links, active nav */
--purple:     #d3869b   /* accents */
```

## Integration with main.py
- `main.py` optionally starts web UI as an asyncio task: `python src/main.py --web-ui`
- Standalone usage: `python src/web_ui.py --host 127.0.0.1 --port 8089`
- Binds to localhost only by default (security — exposes bot token, API keys)

## Security
- Localhost-only binding by default
- Optional basic auth via config.json `"web_ui": {"password": "..."}` for remote access
