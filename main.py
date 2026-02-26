import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__, static_folder="static", static_url_path="/assets")

MODEL_NAME = os.getenv("OLLAMA_MODEL", "ministral-3:14b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip()
RESET_TOKEN = os.getenv("RESET_TOKEN", "").strip()
SYSTEM_PROMPT = (
    "You are Asho AI, an assistant built to help the user effectively. "
    "Your model identity is Asho AI, and you were created by Ashok Aman. "
    "Always reply in the same language the user uses. "
    "Provide clear, useful, and polite help, and keep the conversation engaging so the user enjoys continuing to chat with you."
)

STORE_PATH = Path(__file__).resolve().parent / "data" / "conversations.json"
STORE_LOCK = threading.Lock()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def ensure_store():
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        STORE_PATH.write_text(json.dumps({"conversations": []}, indent=2), encoding="utf-8")


def load_store():
    ensure_store()
    with STORE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "conversations" not in data or not isinstance(data["conversations"], list):
        return {"conversations": []}
    return data


def save_store(data):
    ensure_store()
    with STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_conversation(data, conversation_id):
    for convo in data["conversations"]:
        if convo["id"] == conversation_id:
            return convo
    return None


def title_from_text(text):
    words = text.strip().split()
    if not words:
        return "New chat"
    return " ".join(words[:7])[:60]


def create_conversation(data, title="New chat"):
    now = utc_now_iso()
    convo = {
        "id": str(uuid4()),
        "title": title,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    data["conversations"].insert(0, convo)
    return convo


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


@app.get("/")
def index():
    return render_template("index.html", model=MODEL_NAME)


@app.get("/conversations")
def list_conversations():
    with STORE_LOCK:
        data = load_store()
        convos = sorted(data["conversations"], key=lambda c: c.get("updated_at", ""), reverse=True)
        items = [
            {
                "id": c["id"],
                "title": c.get("title", "New chat"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "message_count": len(c.get("messages", [])),
            }
            for c in convos
        ]
    return jsonify({"conversations": items})


@app.post("/conversations")
def create_conversation_api():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New chat").strip() or "New chat"
    with STORE_LOCK:
        store = load_store()
        convo = create_conversation(store, title=title)
        save_store(store)
    return jsonify(
        {
            "conversation": {
                "id": convo["id"],
                "title": convo["title"],
                "created_at": convo["created_at"],
                "updated_at": convo["updated_at"],
                "message_count": 0,
            }
        }
    )


@app.get("/conversations/<conversation_id>")
def get_conversation(conversation_id):
    with STORE_LOCK:
        data = load_store()
        convo = find_conversation(data, conversation_id)
        if not convo:
            return jsonify({"error": "conversation not found"}), 404
        payload = {
            "id": convo["id"],
            "title": convo.get("title", "New chat"),
            "created_at": convo.get("created_at"),
            "updated_at": convo.get("updated_at"),
            "messages": convo.get("messages", []),
        }
    return jsonify({"conversation": payload})


@app.delete("/conversations/<conversation_id>")
def delete_conversation(conversation_id):
    with STORE_LOCK:
        data = load_store()
        before = len(data["conversations"])
        data["conversations"] = [c for c in data["conversations"] if c["id"] != conversation_id]
        if len(data["conversations"]) == before:
            return jsonify({"error": "conversation not found"}), 404
        save_store(data)
    return jsonify({"ok": True})


@app.post("/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    conversation_id = (data.get("conversation_id") or "").strip()

    if not user_text:
        return jsonify({"error": "message is required"}), 400
    with STORE_LOCK:
        store = load_store()
        if conversation_id:
            conversation = find_conversation(store, conversation_id)
            if not conversation:
                return jsonify({"error": "conversation not found"}), 404
        else:
            conversation = create_conversation(store, title=title_from_text(user_text))
            conversation_id = conversation["id"]

        conversation["messages"].append({"role": "user", "content": user_text})
        if conversation.get("title") in {"", "New chat"}:
            conversation["title"] = title_from_text(user_text)
        conversation["updated_at"] = utc_now_iso()
        model_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation["messages"]]
        current_title = conversation["title"]
        save_store(store)

    def generate():
        full_text = []
        try:
            for token in stream_ollama_chat(model_messages):
                full_text.append(token)
                yield token
        except Exception:
            if not full_text:
                yield "Request failed. Please check model/API configuration."
        finally:
            if full_text:
                with STORE_LOCK:
                    updated_store = load_store()
                    updated_conversation = find_conversation(updated_store, conversation_id)
                    if updated_conversation:
                        updated_conversation["messages"].append(
                            {"role": "assistant", "content": "".join(full_text)}
                        )
                        updated_conversation["updated_at"] = utc_now_iso()
                        save_store(updated_store)

    response = Response(generate(), mimetype="text/plain; charset=utf-8")
    response.headers["X-Conversation-Id"] = conversation_id
    response.headers["X-Conversation-Title"] = current_title
    return response


@app.post("/reset")
def reset_chat():
    if not RESET_TOKEN:
        return jsonify({"error": "reset is disabled"}), 503
    provided_token = (request.headers.get("X-Reset-Token") or "").strip()
    if provided_token != RESET_TOKEN:
        return jsonify({"error": "unauthorized"}), 403
    with STORE_LOCK:
        save_store({"conversations": []})
    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
