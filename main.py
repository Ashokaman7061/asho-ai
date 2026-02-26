import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__, static_folder="static", static_url_path="/assets")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_NAME = os.getenv("OLLAMA_MODEL", "ministral-3:14b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip()
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))

SYSTEM_PROMPT = (
    "You are Asho AI, an assistant built to help the user effectively. "
    "Your model identity is Asho AI, and you were created by Ashok Aman. "
    "Always reply in the same language the user uses. "
    "Provide clear, useful, and polite help, and keep the conversation engaging so the user enjoys continuing to chat with you."
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "conversations.db"
LEGACY_JSON_PATH = BASE_DIR / "data" / "conversations.json"
DB_LOCK = threading.Lock()
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_STATE = {}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with DB_LOCK:
        conn = get_db()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
                """
            )
            migrate_legacy_json_if_needed(conn)
            conn.commit()
        finally:
            conn.close()


def migrate_legacy_json_if_needed(conn):
    count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    if count > 0 or not LEGACY_JSON_PATH.exists():
        return
    try:
        legacy = json.loads(LEGACY_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        app.logger.warning("legacy JSON read failed: %s", exc)
        return
    for convo in legacy.get("conversations", []):
        cid = convo.get("id") or str(uuid4())
        title = convo.get("title") or "New chat"
        created_at = convo.get("created_at") or utc_now_iso()
        updated_at = convo.get("updated_at") or created_at
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, title, created_at, updated_at),
        )
        for msg in convo.get("messages", []):
            role = msg.get("role") or "assistant"
            content = msg.get("content") or ""
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (cid, role, content, updated_at),
            )


def title_from_text(text):
    words = text.strip().split()
    if not words:
        return "New chat"
    return " ".join(words[:7])[:60]


def get_ollama_chat_url():
    if OLLAMA_BASE_URL:
        return OLLAMA_BASE_URL.rstrip("/") + "/chat"
    if OLLAMA_API_KEY:
        return "https://ollama.com/api/chat"
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    return host + "/api/chat"


def stream_ollama_chat(messages):
    url = get_ollama_chat_url()
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    payload = {"model": MODEL_NAME, "messages": messages, "stream": True}
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=120.0) as res:
        res.raise_for_status()
        for line in res.iter_lines():
            if not line:
                continue
            part = json.loads(line)
            token = part.get("message", {}).get("content")
            if token:
                yield token


def client_ip():
    forwarded = (request.headers.get("X-Forwarded-For") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    return (request.remote_addr or "unknown").strip()


def is_rate_limited(ip):
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff = now_ts - RATE_LIMIT_WINDOW_SECONDS
    with RATE_LIMIT_LOCK:
        recent = [ts for ts in RATE_LIMIT_STATE.get(ip, []) if ts >= cutoff]
        if len(recent) >= RATE_LIMIT_MAX_REQUESTS:
            RATE_LIMIT_STATE[ip] = recent
            return True
        recent.append(now_ts)
        RATE_LIMIT_STATE[ip] = recent
        return False


def create_conversation(conn, title="New chat"):
    now = utc_now_iso()
    cid = str(uuid4())
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (cid, title, now, now),
    )
    return {"id": cid, "title": title, "created_at": now, "updated_at": now}


def conversation_meta_row_to_dict(row):
    return {
        "id": row["id"],
        "title": row["title"] or "New chat",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_conversation_messages(conn, conversation_id):
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


@app.get("/")
def index():
    return render_template("index.html", model=MODEL_NAME)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/conversations")
def list_conversations():
    with DB_LOCK:
        conn = get_db()
        try:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
            items = [
                {
                    "id": r["id"],
                    "title": r["title"] or "New chat",
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": int(r["message_count"] or 0),
                }
                for r in rows
            ]
            return jsonify({"conversations": items})
        finally:
            conn.close()


@app.post("/conversations")
def create_conversation_api():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New chat").strip() or "New chat"
    with DB_LOCK:
        conn = get_db()
        try:
            convo = create_conversation(conn, title=title)
            conn.commit()
            return jsonify({"conversation": {**convo, "message_count": 0}})
        finally:
            conn.close()


@app.get("/conversations/<conversation_id>")
def get_conversation(conversation_id):
    with DB_LOCK:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
            if not row:
                return jsonify({"error": "conversation not found"}), 404
            payload = conversation_meta_row_to_dict(row)
            payload["messages"] = get_conversation_messages(conn, conversation_id)
            return jsonify({"conversation": payload})
        finally:
            conn.close()


@app.delete("/conversations/<conversation_id>")
def delete_conversation(conversation_id):
    with DB_LOCK:
        conn = get_db()
        try:
            deleted = conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,)).rowcount
            conn.commit()
            if not deleted:
                return jsonify({"error": "conversation not found"}), 404
            return jsonify({"ok": True})
        finally:
            conn.close()


@app.post("/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    conversation_id = (data.get("conversation_id") or "").strip()
    ip = client_ip()

    if not user_text:
        return jsonify({"error": "message is required"}), 400
    if len(user_text) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"message too long (max {MAX_MESSAGE_CHARS} chars)"}), 400
    if is_rate_limited(ip):
        return jsonify({"error": "rate limit exceeded, try again later"}), 429

    with DB_LOCK:
        conn = get_db()
        try:
            if conversation_id:
                row = conn.execute(
                    "SELECT id, title FROM conversations WHERE id=?", (conversation_id,)
                ).fetchone()
                if not row:
                    return jsonify({"error": "conversation not found"}), 404
                current_title = row["title"] or "New chat"
            else:
                convo = create_conversation(conn, title=title_from_text(user_text))
                conversation_id = convo["id"]
                current_title = convo["title"]

            if current_title in {"", "New chat"}:
                current_title = title_from_text(user_text)
                conn.execute(
                    "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                    (current_title, utc_now_iso(), conversation_id),
                )

            now = utc_now_iso()
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, "user", user_text, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
            model_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_conversation_messages(
                conn, conversation_id
            )
            conn.commit()
        finally:
            conn.close()

    def generate():
        full_text = []
        try:
            for token in stream_ollama_chat(model_messages):
                full_text.append(token)
                yield token
        except Exception as exc:
            app.logger.exception("chat stream failed: %s", exc)
            if not full_text:
                yield "Request failed. Please check model/API configuration."
        finally:
            if full_text:
                with DB_LOCK:
                    conn = get_db()
                    try:
                        now = utc_now_iso()
                        conn.execute(
                            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                            (conversation_id, "assistant", "".join(full_text), now),
                        )
                        conn.execute(
                            "UPDATE conversations SET updated_at=? WHERE id=?",
                            (now, conversation_id),
                        )
                        conn.commit()
                    finally:
                        conn.close()

    response = Response(generate(), mimetype="text/plain; charset=utf-8")
    response.headers["X-Conversation-Id"] = conversation_id
    response.headers["X-Conversation-Title"] = current_title
    return response


@app.post("/reset")
def reset_chat():
    with DB_LOCK:
        conn = get_db()
        try:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")
            conn.commit()
            return jsonify({"ok": True})
        finally:
            conn.close()


init_db()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
