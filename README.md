# QhySync

QhySync is a polished chat-with-avatar experience built on Gradio. It combines:

- An expressive transparent neon avatar panel with listening/thinking/speaking states.
- Streaming assistant responses through the OpenAI-compatible API when configured.
- A deterministic offline companion mode when no API key is available.
- SQLite-backed conversations, messages, and durable per-conversation memory.
- Working personas and editable system instructions.
- JSON export and new-chat controls.
- An MCP bridge panel that accurately reflects authorization as a separate upstream step.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app listens on `0.0.0.0:7860` by default. Set `PORT` to change the port.

## Optional model configuration

```bash
export OPENAI_API_KEY=...
export QHYSYNC_MODEL=gpt-4o-mini
# Optional OpenAI-compatible proxy:
export OPENAI_API_BASE=https://your-proxy.example/v1
```

Without `OPENAI_API_KEY` or the legacy `OPENAI_KEY`, the app remains fully usable in offline mode.

## Persistence

The default database is `qhysync.sqlite3` in the project directory. Override it with `QHYSYNC_DB=/path/to/qhysync.sqlite3`.

## MCP authorization

The MCP panel intentionally does not claim that authorization is complete. Clicking **Review authorization** changes the UI to an explicit **Authorization pending** state, matching the supplied authorization reference: the upstream provider must verify the client before MCP resources are exposed.
