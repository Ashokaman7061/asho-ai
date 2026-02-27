import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from flask import Flask, Response, jsonify, render_template, request, session
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__, static_folder="static", static_url_path="/assets")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("COOKIE_SECURE", "1") == "1"
app.config["SESSION_PERMANENT"] = True
app.permanent_session_lifetime = timedelta(days=int(os.getenv("SESSION_DAYS", "30")))

MODEL_NAME = os.getenv("OLLAMA_MODEL", "ministral-3:14b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "1") == "1"
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_STT_URL = os.getenv(
    "SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text"
).strip()
SARVAM_TTS_URL = os.getenv(
    "SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech/stream"
).strip()

SYSTEM_PROMPT = (
    "You are Asho AI, an assistant built to help the user effectively. "
    "Your model identity is Asho AI, and you were created by Ashok Aman. "
    "Always reply in the same language the user uses. "
    "Provide clear, useful, and polite help, and keep the conversation engaging so the user enjoys continuing to chat with you. "
    "Be strictly honest: never fabricate facts. If you do not know something, clearly say you do not know. "
    "Use current real-time date/time context provided in system messages; do not rely on stale training-time dates."
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
                    user_sub TEXT NOT NULL,
                    title TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT,
                    display_name TEXT,
                    provider TEXT NOT NULL DEFAULT 'local',
                    provider_sub TEXT,
                    session_nonce INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
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
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_sub TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            ensure_schema_columns(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_sub ON conversations(user_sub)"
            )
            migrate_legacy_json_if_needed(conn)
            conn.commit()
        finally:
            conn.close()


def ensure_schema_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if "user_sub" not in cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN user_sub TEXT NOT NULL DEFAULT 'public'")
    if "pinned" not in cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
    user_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "session_nonce" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN session_nonce INTEGER NOT NULL DEFAULT 0")


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


def get_user_profile(conn, user_sub):
    row = conn.execute(
        "SELECT profile_json FROM user_profiles WHERE user_sub=?",
        (user_sub,),
    ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row["profile_json"] or "{}")
    except Exception:
        return {}


def save_user_profile(conn, user_sub, profile):
    conn.execute(
        """
        INSERT INTO user_profiles (user_sub, profile_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_sub) DO UPDATE SET
            profile_json=excluded.profile_json,
            updated_at=excluded.updated_at
        """,
        (user_sub, json.dumps(profile, ensure_ascii=False), utc_now_iso()),
    )


def update_user_profile_from_message(profile, text):
    msg = (text or "").strip()
    if not msg:
        return profile
    p = dict(profile or {})

    lc = msg.lower()
    # Language preference heuristic
    if any(w in lc for w in ["kya", "kaise", "nahi", "haan", "mujhe", "tum", "karna"]):
        p["preferred_language"] = "hinglish"
    elif all(ord(c) < 128 for c in msg):
        p.setdefault("preferred_language", "english")

    # Detail preference heuristic
    if any(w in lc for w in ["step by step", "detail", "explain", "samjhao"]):
        p["detail_level"] = "detailed"
    elif any(w in lc for w in ["short", "brief", "jaldi", "one line"]):
        p["detail_level"] = "concise"

    # Topics of interest (very simple)
    topics = set(p.get("topics", []))
    for topic in ["deployment", "render", "flask", "ui", "security", "auth", "database", "api"]:
        if topic in lc:
            topics.add(topic)
    if topics:
        p["topics"] = sorted(topics)

    return p


def build_profile_context(profile):
    if not profile:
        return ""
    lines = ["User preference memory (for better help quality):"]
    lang = profile.get("preferred_language")
    if lang:
        lines.append(f"- Preferred language style: {lang}")
    detail = profile.get("detail_level")
    if detail:
        lines.append(f"- Preferred response detail: {detail}")
    topics = profile.get("topics") or []
    if topics:
        lines.append(f"- Frequent topics: {', '.join(topics[:8])}")
    lines.append("Use this only to improve helpfulness and communication style.")
    return "\n".join(lines)


def maybe_auto_rename_title(conn, conversation_id, current_title):
    rows = conn.execute(
        "SELECT content FROM messages WHERE conversation_id=? AND role='user' ORDER BY id ASC LIMIT 3",
        (conversation_id,),
    ).fetchall()
    if len(rows) < 3:
        return current_title

    combined = " ".join((r["content"] or "").strip() for r in rows).strip()
    auto_title = title_from_text(combined)
    if not auto_title or auto_title == "New chat" or auto_title == current_title:
        return current_title

    now = utc_now_iso()
    conn.execute(
        "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
        (auto_title, now, conversation_id),
    )
    return auto_title


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


def sarvam_headers():
    return {
        "api-subscription-key": SARVAM_API_KEY,
    }


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
    user_sub = current_user_sub()
    conn.execute(
        "INSERT INTO conversations (id, user_sub, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (cid, user_sub, title, now, now),
    )
    return {"id": cid, "title": title, "pinned": 0, "created_at": now, "updated_at": now}


def conversation_meta_row_to_dict(row):
    return {
        "id": row["id"],
        "title": row["title"] or "New chat",
        "pinned": int(row["pinned"] or 0),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_conversation_messages(conn, conversation_id):
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def build_model_messages_from_conversation(conn, conversation_id, profile_context=""):
    model_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_conversation_messages(
        conn, conversation_id
    )
    realtime_context = (
        f"Current UTC datetime: {utc_now_iso()}. "
        "Use this as the authoritative current time reference in this conversation."
    )
    parts = [model_messages[0], {"role": "system", "content": realtime_context}]
    if profile_context:
        parts.append({"role": "system", "content": profile_context})
    parts.extend(model_messages[1:])
    return parts


def is_auth_enabled():
    return AUTH_REQUIRED


def current_user_sub():
    if not AUTH_REQUIRED:
        return "public"
    user = session.get("user")
    if not user:
        return None
    return user.get("sub")


def require_user():
    sub = current_user_sub()
    if not sub:
        return None
    if not AUTH_REQUIRED:
        return sub
    try:
        user_id = int(sub.split(":", 1)[1])
    except Exception:
        session.pop("user", None)
        return None
    expected_nonce = int((session.get("user") or {}).get("session_nonce") or 0)
    with DB_LOCK:
        conn = get_db()
        try:
            row = conn.execute("SELECT session_nonce FROM users WHERE id=?", (user_id,)).fetchone()
        finally:
            conn.close()
    if not row or int(row["session_nonce"] or 0) != expected_nonce:
        session.pop("user", None)
        return None
    return sub


def session_user_payload(
    user_sub, email=None, name=None, provider="local", picture=None, session_nonce=0
):
    return {
        "sub": user_sub,
        "email": email,
        "name": name,
        "provider": provider,
        "picture": picture,
        "session_nonce": int(session_nonce or 0),
    }


def login_user(user_payload):
    session.clear()
    session.permanent = True
    session["user"] = user_payload


@app.get("/")
def index():
    return render_template(
        "index.html",
        model=MODEL_NAME,
        google_client_id=GOOGLE_CLIENT_ID,
        auth_enabled=is_auth_enabled(),
        google_enabled=bool(GOOGLE_CLIENT_ID),
    )


@app.get("/auth/me")
def auth_me():
    user = session.get("user")
    return jsonify({"auth_enabled": is_auth_enabled(), "user": user})


@app.post("/auth/google")
def auth_google():
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "google auth disabled"}), 503
    payload = request.get_json(silent=True) or {}
    credential = (payload.get("credential") or "").strip()
    if not credential:
        return jsonify({"error": "credential is required"}), 400
    try:
        info = id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except Exception as exc:
        app.logger.warning("google token verify failed: %s", exc)
        return jsonify({"error": "invalid google token"}), 401
    provider_sub = info.get("sub")
    email = (info.get("email") or "").strip().lower()
    name = info.get("name")
    picture = info.get("picture")
    with DB_LOCK:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT id FROM users WHERE provider='google' AND provider_sub=?",
                (provider_sub,),
            ).fetchone()
            if not row and email:
                row = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
                if row:
                    conn.execute(
                        "UPDATE users SET provider='google', provider_sub=?, display_name=? WHERE id=?",
                        (provider_sub, name, row["id"]),
                    )
            if not row:
                now = utc_now_iso()
                conn.execute(
                    """
                    INSERT INTO users (email, password_hash, display_name, provider, provider_sub, session_nonce, created_at)
                    VALUES (?, NULL, ?, 'google', ?, 0, ?)
                    """,
                    (email or f"google_{provider_sub}@example.local", name, provider_sub, now),
                )
                user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                session_nonce = 0
            else:
                user_id = row["id"]
                nonce_row = conn.execute(
                    "SELECT session_nonce FROM users WHERE id=?", (user_id,)
                ).fetchone()
                session_nonce = int((nonce_row["session_nonce"] if nonce_row else 0) or 0)
            conn.commit()
        finally:
            conn.close()
    login_user(
        session_user_payload(
        f"user:{user_id}",
        email=email,
        name=name,
        provider="google",
        picture=picture,
        session_nonce=session_nonce,
        )
    )
    return jsonify({"ok": True, "user": session["user"]})


@app.post("/auth/register")
def auth_register():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    name = (payload.get("name") or "").strip()
    if "@" not in email or len(email) > 254:
        return jsonify({"error": "valid email is required"}), 400
    if len(password) < 8:
        return jsonify({"error": "password must be at least 8 characters"}), 400
    password_hash = generate_password_hash(password)
    now = utc_now_iso()
    with DB_LOCK:
        conn = get_db()
        try:
            existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if existing:
                return jsonify({"error": "email already registered"}), 409
            conn.execute(
                """
                INSERT INTO users (email, password_hash, display_name, provider, provider_sub, session_nonce, created_at)
                VALUES (?, ?, ?, 'local', NULL, 0, ?)
                """,
                (email, password_hash, name or email.split("@")[0], now),
            )
            user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()
        finally:
            conn.close()
    login_user(
        session_user_payload(
        f"user:{user_id}",
        email=email,
        name=name or email.split("@")[0],
        provider="local",
        session_nonce=0,
        )
    )
    return jsonify({"ok": True, "user": session["user"]})


@app.post("/auth/login")
def auth_login():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if "@" not in email:
        return jsonify({"error": "valid email is required"}), 400
    with DB_LOCK:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT id, password_hash, display_name, provider, session_nonce FROM users WHERE email=?",
                (email,),
            ).fetchone()
        finally:
            conn.close()
    if not row or not row["password_hash"]:
        return jsonify({"error": "invalid email or password"}), 401
    if not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "invalid email or password"}), 401
    login_user(
        session_user_payload(
        f"user:{row['id']}",
        email=email,
        name=row["display_name"] or email.split("@")[0],
        provider=row["provider"] or "local",
        session_nonce=int(row["session_nonce"] or 0),
        )
    )
    return jsonify({"ok": True, "user": session["user"]})


@app.post("/auth/logout")
def auth_logout():
    session.pop("user", None)
    return jsonify({"ok": True})


@app.post("/auth/logout_all")
def auth_logout_all():
    sub = current_user_sub()
    if not sub or ":" not in sub:
        return jsonify({"error": "auth required"}), 401
    user_id = int(sub.split(":", 1)[1])
    with DB_LOCK:
        conn = get_db()
        try:
            conn.execute(
                "UPDATE users SET session_nonce = session_nonce + 1 WHERE id=?",
                (user_id,),
            )
            conn.commit()
        finally:
            conn.close()
    session.pop("user", None)
    return jsonify({"ok": True})


@app.post("/auth/delete")
def auth_delete():
    sub = current_user_sub()
    if not sub or ":" not in sub:
        return jsonify({"error": "auth required"}), 401
    user_id = int(sub.split(":", 1)[1])
    user_sub = f"user:{user_id}"
    with DB_LOCK:
        conn = get_db()
        try:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_sub=?)",
                (user_sub,),
            )
            conn.execute("DELETE FROM conversations WHERE user_sub=?", (user_sub,))
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn.commit()
        finally:
            conn.close()
    session.pop("user", None)
    return jsonify({"ok": True})


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/terms")
def terms_page():
    return render_template("terms.html")


@app.get("/privacy")
def privacy_page():
    return render_template("privacy.html")


@app.get("/conversations")
def list_conversations():
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    with DB_LOCK:
        conn = get_db()
        try:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.pinned, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.user_sub = ?
                GROUP BY c.id
                ORDER BY c.pinned DESC, c.updated_at DESC
                """,
                (user_sub,),
            ).fetchall()
            items = [
                {
                    "id": r["id"],
                    "title": r["title"] or "New chat",
                    "pinned": int(r["pinned"] or 0),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": int(r["message_count"] or 0),
                }
                for r in rows
            ]
            return jsonify({"conversations": items})
        finally:
            conn.close()


@app.get("/conversations/export")
def export_conversations():
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    with DB_LOCK:
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE user_sub=? ORDER BY updated_at DESC",
                (user_sub,),
            ).fetchall()
            conversations = []
            for row in rows:
                messages = get_conversation_messages(conn, row["id"])
                conversations.append(
                    {
                        "id": row["id"],
                        "title": row["title"] or "New chat",
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "messages": messages,
                    }
                )
            return jsonify({"exported_at": utc_now_iso(), "conversations": conversations})
        finally:
            conn.close()


@app.post("/conversations")
def create_conversation_api():
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
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
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    with DB_LOCK:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT id, title, pinned, created_at, updated_at FROM conversations WHERE id=? AND user_sub=?",
                (conversation_id, user_sub),
            ).fetchone()
            if not row:
                return jsonify({"error": "conversation not found"}), 404
            payload = conversation_meta_row_to_dict(row)
            payload["messages"] = get_conversation_messages(conn, conversation_id)
            return jsonify({"conversation": payload})
        finally:
            conn.close()


@app.post("/conversations/<conversation_id>/pin")
def pin_conversation(conversation_id):
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    data = request.get_json(silent=True) or {}
    pinned = 1 if bool(data.get("pinned")) else 0
    with DB_LOCK:
        conn = get_db()
        try:
            updated = conn.execute(
                "UPDATE conversations SET pinned=?, updated_at=? WHERE id=? AND user_sub=?",
                (pinned, utc_now_iso(), conversation_id, user_sub),
            ).rowcount
            conn.commit()
            if not updated:
                return jsonify({"error": "conversation not found"}), 404
            return jsonify({"ok": True, "pinned": pinned})
        finally:
            conn.close()


@app.post("/conversations/<conversation_id>/rename")
def rename_conversation(conversation_id):
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400
    title = title[:80]
    with DB_LOCK:
        conn = get_db()
        try:
            updated = conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=? AND user_sub=?",
                (title, utc_now_iso(), conversation_id, user_sub),
            ).rowcount
            conn.commit()
            if not updated:
                return jsonify({"error": "conversation not found"}), 404
            return jsonify({"ok": True, "title": title})
        finally:
            conn.close()


@app.post("/conversations/<conversation_id>/regenerate")
def regenerate_conversation(conversation_id):
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    data = request.get_json(silent=True) or {}
    edited_message = (data.get("message") or "").strip()
    edit_requested = "message" in data
    if edit_requested and not edited_message:
        return jsonify({"error": "message is required"}), 400
    if edited_message and len(edited_message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"message too long (max {MAX_MESSAGE_CHARS} chars)"}), 400

    with DB_LOCK:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT id, title FROM conversations WHERE id=? AND user_sub=?",
                (conversation_id, user_sub),
            ).fetchone()
            if not row:
                return jsonify({"error": "conversation not found"}), 404
            current_title = row["title"] or "New chat"
            last_user = conn.execute(
                "SELECT id, content FROM messages WHERE conversation_id=? AND role='user' ORDER BY id DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if not last_user:
                return jsonify({"error": "no user message found"}), 400

            last_user_id = int(last_user["id"])
            if edit_requested:
                conn.execute(
                    "UPDATE messages SET content=?, created_at=? WHERE id=?",
                    (edited_message, utc_now_iso(), last_user_id),
                )
                profile = get_user_profile(conn, user_sub)
                profile = update_user_profile_from_message(profile, edited_message)
                save_user_profile(conn, user_sub, profile)
            conn.execute(
                "DELETE FROM messages WHERE conversation_id=? AND id>?",
                (conversation_id, last_user_id),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (utc_now_iso(), conversation_id),
            )
            current_title = maybe_auto_rename_title(conn, conversation_id, current_title)
            profile = get_user_profile(conn, user_sub)
            profile_context = build_profile_context(profile)
            model_messages = build_model_messages_from_conversation(
                conn, conversation_id, profile_context=profile_context
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
            app.logger.exception("regenerate stream failed: %s", exc)
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


@app.delete("/conversations/<conversation_id>")
def delete_conversation(conversation_id):
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    with DB_LOCK:
        conn = get_db()
        try:
            deleted = conn.execute(
                "DELETE FROM conversations WHERE id=? AND user_sub=?", (conversation_id, user_sub)
            ).rowcount
            conn.commit()
            if not deleted:
                return jsonify({"error": "conversation not found"}), 404
            return jsonify({"ok": True})
        finally:
            conn.close()


@app.post("/chat")
def chat_api():
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
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
                    "SELECT id, title FROM conversations WHERE id=? AND user_sub=?",
                    (conversation_id, user_sub),
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
            profile = get_user_profile(conn, user_sub)
            profile = update_user_profile_from_message(profile, user_text)
            save_user_profile(conn, user_sub, profile)
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
            profile_context = build_profile_context(profile)
            model_messages = build_model_messages_from_conversation(
                conn, conversation_id, profile_context=profile_context
            )
            current_title = maybe_auto_rename_title(conn, conversation_id, current_title)
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


@app.post("/voice/stt")
def voice_stt_api():
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    if not SARVAM_API_KEY:
        return jsonify({"error": "sarvam api key not configured"}), 503
    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "audio file is required"}), 400
    filename = file.filename or "audio.webm"
    content_type = file.mimetype or "audio/webm"
    files = {"file": (filename, file.stream, content_type)}
    data = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "language_code": request.form.get("language_code", "unknown"),
        "with_diarization": request.form.get("with_diarization", "false"),
        "num_speakers": request.form.get("num_speakers", "2"),
    }
    try:
        res = httpx.post(
            SARVAM_STT_URL,
            headers=sarvam_headers(),
            data=data,
            files=files,
            timeout=180.0,
        )
        res.raise_for_status()
        payload = res.json()
    except Exception as exc:
        app.logger.exception("sarvam stt failed: %s", exc)
        return jsonify({"error": "stt request failed"}), 502
    text = (
        payload.get("transcript")
        or payload.get("text")
        or payload.get("result", {}).get("transcript")
        or ""
    ).strip()
    return jsonify({"text": text, "raw": payload})


@app.post("/voice/tts")
def voice_tts_api():
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    if not SARVAM_API_KEY:
        return jsonify({"error": "sarvam api key not configured"}), 503
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    req_payload = {
        "text": text,
        "target_language_code": payload.get("target_language_code", "hi-IN"),
        "speaker": payload.get("speaker", "shubh"),
        "model": payload.get("model", "bulbul:v3"),
        "pace": float(payload.get("pace", 1.1)),
        "speech_sample_rate": int(payload.get("speech_sample_rate", 22050)),
        "output_audio_codec": payload.get("output_audio_codec", "mp3"),
        "enable_preprocessing": bool(payload.get("enable_preprocessing", True)),
    }

    def generate():
        try:
            with httpx.stream(
                "POST",
                SARVAM_TTS_URL,
                headers={**sarvam_headers(), "Content-Type": "application/json"},
                json=req_payload,
                timeout=180.0,
            ) as res:
                res.raise_for_status()
                for chunk in res.iter_bytes():
                    if chunk:
                        yield chunk
        except Exception as exc:
            app.logger.exception("sarvam tts failed: %s", exc)
            return

    return Response(generate(), mimetype="audio/mpeg")


@app.post("/reset")
def reset_chat():
    user_sub = require_user()
    if not user_sub:
        return jsonify({"error": "auth required"}), 401
    with DB_LOCK:
        conn = get_db()
        try:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_sub=?)",
                (user_sub,),
            )
            conn.execute("DELETE FROM conversations WHERE user_sub=?", (user_sub,))
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
