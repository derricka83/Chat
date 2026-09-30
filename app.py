import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import gradio as gr

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency for offline mode
    OpenAI = None


APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("QHYSYNC_DB", APP_DIR / "qhysync.sqlite3"))
DB_LOCK = threading.Lock()

PERSONAS = {
    "Companion": "You are QhySync, a warm, curious AI companion. Be concise but thoughtful, ask one useful follow-up when helpful, and never pretend to have done something you have not done.",
    "Builder": "You are QhySync Builder, a pragmatic product and engineering partner. Give clear steps, identify tradeoffs, and favor small testable changes.",
    "Coach": "You are QhySync Coach, an encouraging thinking partner. Help the user clarify goals, reflect, and turn uncertainty into an actionable next step.",
    "Researcher": "You are QhySync Researcher. Separate known facts from assumptions, structure complex topics, and call out when fresh sources would be needed.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_db() -> None:
    with DB_LOCK, db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                persona TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def create_conversation(persona: str, system_prompt: str) -> str:
    conversation_id = str(uuid.uuid4())
    now = utc_now()
    with DB_LOCK, db() as connection:
        connection.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, "New conversation", persona, system_prompt, now, now),
        )
    return conversation_id


def save_message(conversation_id: str, role: str, content: str) -> None:
    with DB_LOCK, db() as connection:
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, utc_now()),
        )
        connection.execute("UPDATE conversations SET updated_at=? WHERE id=?", (utc_now(), conversation_id))


def get_messages(conversation_id: str) -> List[Dict[str, str]]:
    with DB_LOCK, db() as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def get_memory(conversation_id: str) -> str:
    with DB_LOCK, db() as connection:
        rows = connection.execute(
            "SELECT content FROM memories WHERE conversation_id=? ORDER BY id DESC LIMIT 12", (conversation_id,)
        ).fetchall()
    return "\n".join(row["content"] for row in reversed(rows))


def replace_memory(conversation_id: str, content: str) -> None:
    with DB_LOCK, db() as connection:
        connection.execute("DELETE FROM memories WHERE conversation_id=?", (conversation_id,))
        for line in [item.strip() for item in content.splitlines() if item.strip()]:
            connection.execute(
                "INSERT INTO memories (conversation_id, content, created_at) VALUES (?, ?, ?)",
                (conversation_id, line, utc_now()),
            )


def rename_conversation(conversation_id: str, title: str) -> None:
    with DB_LOCK, db() as connection:
        connection.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title[:64], utc_now(), conversation_id))


def fallback_reply(message: str, persona: str, memory: str) -> str:
    lower = message.lower()
    if any(word in lower for word in ("hello", "hi", "hey")):
        return f"Hey — I’m QhySync. I’m in **{persona}** mode and ready to help. What are we making sense of today?"
    if "mcp" in lower or "authorize" in lower:
        return "The MCP bridge is ready to connect, but authorization must happen with the upstream provider. Open the MCP panel on the right to review the exact authorization state."
    if "remember" in lower or "memory" in lower:
        return "I can keep durable notes for this conversation. Add or edit them in the Memory panel, then I’ll use them as context in future replies."
    context = f" I’m also carrying this memory: {memory.splitlines()[0]}" if memory else ""
    return (
        f"I’m in offline companion mode right now, so I can still help with planning, drafting, and reasoning.{context}\n\n"
        f"Here’s a useful first pass on **{message[:120]}**: break it into the desired outcome, the current constraint, and the smallest next action."
    )


def stream_reply(user_message: str, persona: str, system_prompt: str, memory: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if OpenAI and api_key:
        try:
            client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE") or None)
            messages = [{"role": "system", "content": f"{PERSONAS.get(persona, PERSONAS['Companion'])}\n\nAdditional system instructions:\n{system_prompt}\n\nDurable memory:\n{memory or 'No saved memory yet.'}"}]
            messages.extend(history[-18:])
            messages.append({"role": "user", "content": user_message})
            stream = client.chat.completions.create(
                model=os.getenv("QHYSYNC_MODEL", "gpt-4o-mini"), messages=messages, temperature=0.7, stream=True
            )
            accumulated = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    accumulated += token
                    yield accumulated
            return
        except Exception as error:
            yield f"I hit a connection issue and switched to offline mode.\n\n`{type(error).__name__}: {error}`\n\n"
    reply = fallback_reply(user_message, persona, memory)
    for index in range(1, len(reply) + 1):
        if index % 3 == 0 or index == len(reply):
            yield reply[:index]
            time.sleep(0.008)


init_db()

CSS = """
:root { --ink:#e9edff; --muted:#8f9aba; --panel:rgba(15,20,43,.82); --line:rgba(156,169,219,.15); --cyan:#64e6ff; --purple:#9c7cff; --pink:#ff78d1; }
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; padding: 0 !important; background: radial-gradient(circle at 20% 10%, rgba(76,51,159,.24), transparent 34%), radial-gradient(circle at 85% 30%, rgba(0,201,255,.12), transparent 28%), #070a18 !important; color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
body { background:#070a18; }
#app-shell { min-height: 100vh; padding: 28px 38px 30px; }
#brand-row { display:flex; align-items:center; justify-content:space-between; padding-bottom:22px; }
.brand-mark { display:flex; gap:13px; align-items:center; } .brand-orb { width:42px; height:42px; border-radius:14px; background:linear-gradient(135deg,var(--cyan),var(--purple)); box-shadow:0 0 34px rgba(100,230,255,.28); position:relative; } .brand-orb:after { content:''; position:absolute; inset:11px; border:2px solid #061021; border-radius:50%; }
.eyebrow { font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:var(--cyan); font-weight:700; } .brand-name { font-size:20px; font-weight:800; letter-spacing:-.03em; } .brand-sub { font-size:12px; color:var(--muted); }
.header-actions { display:flex; gap:10px; align-items:center; } .status-pill { border:1px solid var(--line); background:rgba(21,29,59,.65); border-radius:99px; padding:9px 13px; color:#b8c2e5; font-size:12px; } .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#6dffb0; margin-right:6px; box-shadow:0 0 12px #6dffb0; }
#workspace { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(330px,.85fr); gap:18px; } .glass { background:var(--panel); border:1px solid var(--line); box-shadow:0 20px 70px rgba(0,0,0,.26); border-radius:22px; backdrop-filter:blur(18px); }
#chat-card { min-height:680px; display:flex; flex-direction:column; overflow:hidden; } .chat-top { padding:22px 24px 17px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; } .chat-title { font-size:18px; font-weight:750; } .chat-meta { color:var(--muted); font-size:12px; margin-top:5px; } .tiny-action button { border:1px solid var(--line) !important; background:transparent !important; color:#cbd4f4 !important; border-radius:10px !important; }
#chatbot { flex:1; min-height:420px; } #chatbot .wrap { background:transparent !important; } #chatbot .message { border-radius:16px !important; line-height:1.55 !important; font-size:14px !important; } #chatbot .message.user { background:linear-gradient(135deg,rgba(93,66,180,.78),rgba(45,100,163,.78)) !important; border:1px solid rgba(155,129,255,.3); } #chatbot .message.bot { background:rgba(35,44,79,.78) !important; border:1px solid rgba(123,142,205,.16); }
#composer { padding:12px 18px 18px; border-top:1px solid var(--line); } #composer textarea { background:rgba(6,10,28,.78) !important; border:1px solid rgba(112,136,223,.24) !important; border-radius:15px !important; color:var(--ink) !important; } #send button { background:linear-gradient(135deg,var(--cyan),#6f83ff) !important; color:#061021 !important; font-weight:800 !important; border:0 !important; border-radius:13px !important; }
#avatar-card { min-height:400px; overflow:hidden; position:relative; } .avatar-head { padding:20px 22px 0; position:relative; z-index:2; } .avatar-stage { height:350px; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; } .avatar-stage:before { content:''; position:absolute; width:300px; height:180px; bottom:25px; border-radius:50%; background:radial-gradient(ellipse,rgba(105,82,255,.32),transparent 68%); filter:blur(10px); }
.avatar { width:190px; height:220px; position:relative; z-index:1; animation:breathe 4s ease-in-out infinite; filter:drop-shadow(0 0 34px rgba(70,213,255,.34)); } .avatar .head { position:absolute; width:152px; height:142px; left:19px; top:40px; border-radius:48% 48% 44% 44%; background:radial-gradient(circle at 31% 24%,#e1faff 0 4%,#7cd7ef 20%,#3863c5 58%,#272b7d 100%); border:2px solid rgba(139,244,255,.63); box-shadow:inset -18px -15px 30px rgba(18,13,95,.45), inset 12px 8px 24px rgba(220,250,255,.42); } .avatar .ear { position:absolute; width:53px; height:76px; top:5px; background:linear-gradient(135deg,#77edff,#5b51c9 75%); clip-path:polygon(50% 0,100% 100%,0 100%); border:2px solid #87efff; } .avatar .ear.left { left:19px; transform:rotate(-27deg); } .avatar .ear.right { right:19px; transform:rotate(27deg); } .eye { position:absolute; top:81px; width:19px; height:25px; border-radius:50%; background:#07102b; border:4px solid #b8fbff; box-shadow:0 0 13px var(--cyan); } .eye.left { left:51px; } .eye.right { right:51px; } .eye:after { content:''; position:absolute; width:5px; height:7px; top:3px; left:4px; border-radius:50%; background:white; } .snout { position:absolute; width:64px; height:44px; left:44px; bottom:26px; border-radius:44%; background:rgba(119,210,238,.47); border-bottom:2px solid rgba(209,252,255,.73); } .nose { position:absolute; width:13px; height:9px; left:70px; bottom:49px; border-radius:60% 60% 48% 48%; background:#171c58; } .tail { position:absolute; width:77px; height:106px; right:-2px; bottom:6px; border:14px solid #6345cf; border-left-color:transparent; border-radius:50%; transform:rotate(-28deg); filter:drop-shadow(0 0 10px #6de8ff); }
.avatar-footer { position:absolute; bottom:18px; left:22px; right:22px; display:flex; justify-content:space-between; align-items:center; } .mood { font-size:12px; color:#b9c8f3; } .mood strong { color:var(--cyan); } .signal { display:flex; gap:3px; align-items:end; height:15px; } .signal i { width:3px; background:var(--cyan); border-radius:2px; animation:signal 1s ease-in-out infinite; } .signal i:nth-child(1){height:5px}.signal i:nth-child(2){height:10px;animation-delay:.12s}.signal i:nth-child(3){height:15px;animation-delay:.23s}.signal i:nth-child(4){height:8px;animation-delay:.35s}
.side-card { padding:20px 22px; } .side-card h3 { font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:#aeb9dc; margin:0 0 15px; } .side-copy { color:var(--muted); font-size:12px; line-height:1.5; margin-top:-6px; } .gradio-dropdown label, .gradio-textbox label { color:#99a6cd !important; font-size:11px !important; text-transform:uppercase; letter-spacing:.1em; } .gradio-dropdown input, .gradio-textbox textarea, .gradio-textbox input { background:rgba(6,10,28,.55) !important; border-color:var(--line) !important; color:var(--ink) !important; border-radius:11px !important; }
.mcp-box { border:1px solid rgba(156,124,255,.26); background:linear-gradient(135deg,rgba(93,64,176,.15),rgba(19,42,84,.18)); border-radius:15px; padding:14px; } .mcp-state { color:#d8d6ff; font-size:13px; font-weight:700; } .mcp-detail { color:var(--muted); font-size:11px; margin-top:5px; line-height:1.5; } .connect button { width:100%; margin-top:12px; background:rgba(113,91,221,.22) !important; border:1px solid rgba(158,132,255,.45) !important; color:#e0d9ff !important; }
@keyframes breathe {0%,100%{transform:translateY(3px) scale(1)}50%{transform:translateY(-5px) scale(1.015)}} @keyframes signal {0%,100%{opacity:.35}50%{opacity:1}} @media (max-width:900px){#app-shell{padding:20px 14px}#workspace{grid-template-columns:1fr}.avatar-card{min-height:330px}#avatar-card{min-height:380px}.header-actions .status-pill{display:none}}
"""


def avatar_html(mood: str = "Ready") -> str:
    return f"""
    <div id='avatar-card' class='glass'>
      <div class='avatar-head'><div class='eyebrow'>Live avatar</div><div class='chat-title' style='margin-top:5px'>QhySync <span style='color:#7c88ae;font-size:12px;font-weight:500'>/ your companion</span></div></div>
      <div class='avatar-stage'><div class='avatar'><div class='ear left'></div><div class='ear right'></div><div class='head'><div class='eye left'></div><div class='eye right'></div><div class='snout'></div><div class='nose'></div></div><div class='tail'></div></div></div>
      <div class='avatar-footer'><div class='mood'>mood <strong>{mood}</strong></div><div class='signal'><i></i><i></i><i></i><i></i></div></div>
    </div>
    """


def on_send(message: str, chat: List[Tuple[str, str]], persona: str, system_prompt: str, conversation_id: str):
    if not message or not message.strip():
        yield chat, "", avatar_html("Waiting"), conversation_id
        return
    history = get_messages(conversation_id)
    memory = get_memory(conversation_id)
    save_message(conversation_id, "user", message.strip())
    next_chat = list(chat or []) + [(message.strip(), "")]
    yield next_chat, "", avatar_html("Thinking"), conversation_id
    current = ""
    for current in stream_reply(message.strip(), persona, system_prompt, memory, history):
        next_chat[-1] = (message.strip(), current)
        yield next_chat, "", avatar_html("Speaking"), conversation_id
    save_message(conversation_id, "assistant", current)
    if len(history) == 0:
        rename_conversation(conversation_id, message.strip())
    yield next_chat, "", avatar_html("Ready"), conversation_id


def new_chat(persona: str, system_prompt: str):
    conversation_id = create_conversation(persona, system_prompt)
    return [], conversation_id, avatar_html("Ready"), "New conversation"


def save_settings(persona: str, system_prompt: str, memory: str, conversation_id: str):
    replace_memory(conversation_id, memory or "")
    with DB_LOCK, db() as connection:
        connection.execute("UPDATE conversations SET persona=?, system_prompt=?, updated_at=? WHERE id=?", (persona, system_prompt, utc_now(), conversation_id))
    return "Saved to durable memory"


def export_chat(chat: List[Tuple[str, str]], conversation_id: str):
    payload = {"conversation_id": conversation_id, "exported_at": utc_now(), "messages": [{"role": role, "content": content} for pair in (chat or []) for role, content in (("user", pair[0]), ("assistant", pair[1]))]}
    export_path = APP_DIR / f"qhysync-export-{conversation_id[:8]}.json"
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(export_path)


def toggle_mcp():
    return "Authorization pending", "The upstream provider must approve this client before MCP resources become available."


initial_persona = "Companion"
initial_system = "Keep responses direct, warm, and useful. Ask before making consequential assumptions."
initial_id = create_conversation(initial_persona, initial_system)

with gr.Blocks(css=CSS, title="QhySync · AI companion") as demo:
    conversation_state = gr.State(initial_id)
    gr.HTML("""<div id='app-shell'><div id='brand-row'><div class='brand-mark'><div class='brand-orb'></div><div><div class='eyebrow'>QhySync / 01</div><div class='brand-name'>Chat with QhySync</div><div class='brand-sub'>An expressive AI companion for clear thinking.</div></div></div><div class='header-actions'><div class='status-pill'><span class='status-dot'></span>Local memory online</div></div></div>""")
    with gr.Row(elem_id="workspace"):
        with gr.Column(elem_id="chat-card", elem_classes="glass"):
            gr.HTML("<div class='chat-top'><div><div class='chat-title'>A little space to think out loud.</div><div class='chat-meta'>Your messages are stored locally in SQLite. The avatar responds as the conversation changes.</div></div></div>")
            chatbot = gr.Chatbot(value=[], elem_id="chatbot", show_label=False, height=475, bubble_full_width=False, type="tuples")
            with gr.Row(elem_id="composer"):
                message = gr.Textbox(show_label=False, placeholder="Message QhySync…  (Shift + Enter for a new line)", lines=2, scale=8, autofocus=True)
                send = gr.Button("Send", elem_id="send", scale=1, variant="primary")
            with gr.Row():
                new = gr.Button("＋ New chat", elem_classes="tiny-action")
                export = gr.DownloadButton("Export JSON", elem_classes="tiny-action")
        with gr.Column(scale=1):
            avatar = gr.HTML(avatar_html(), elem_id="avatar-wrap")
            with gr.Column(elem_classes="glass side-card"):
                gr.HTML("<h3>Companion settings</h3><div class='side-copy'>Shape how QhySync thinks with you. These settings persist for this conversation.</div>")
                persona = gr.Dropdown(list(PERSONAS.keys()), value=initial_persona, label="Working persona")
                system_prompt = gr.Textbox(value=initial_system, label="System instructions", lines=3)
                memory = gr.Textbox(value="", label="Durable memory", placeholder="One memory per line…", lines=4)
                save = gr.Button("Save settings", variant="secondary")
                save_status = gr.Markdown("", elem_id="save-status")
            with gr.Column(elem_classes="glass side-card"):
                gr.HTML("<h3>MCP bridge</h3>")
                mcp_state = gr.Markdown("**Ready to connect**", elem_id="mcp-state")
                mcp_detail = gr.Markdown("Review the authorization step before exposing external tools or resources.", elem_id="mcp-detail")
                connect = gr.Button("Review authorization", elem_classes="connect")

    send.click(on_send, [message, chatbot, persona, system_prompt, conversation_state], [chatbot, message, avatar, conversation_state])
    message.submit(on_send, [message, chatbot, persona, system_prompt, conversation_state], [chatbot, message, avatar, conversation_state])
    new.click(new_chat, [persona, system_prompt], [chatbot, conversation_state, avatar, mcp_detail])
    save.click(save_settings, [persona, system_prompt, memory, conversation_state], save_status)
    connect.click(toggle_mcp, [], [mcp_state, mcp_detail])
    export.click(export_chat, [chatbot, conversation_state], export)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=True)
