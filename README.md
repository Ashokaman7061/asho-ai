# Asho AI (Flask)

## Local Run

1. Create and activate venv.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.
```powershell
pip install -r requirements.txt
```

3. Create env file from template.
```powershell
Copy-Item .env.example .env
```

4. Set environment variables.
```powershell
$env:FLASK_DEBUG="1"
$env:FLASK_SECRET_KEY="change-me-strong-secret"
$env:COOKIE_SECURE="0"
$env:GOOGLE_CLIENT_ID="your-google-client-id.apps.googleusercontent.com"
$env:GOOGLE_SEARCH_API_KEY=""
$env:GOOGLE_SEARCH_CX=""
$env:SEARCH_MAX_RESULTS="5"
$env:OLLAMA_MODEL="ministral-3:14b-cloud"
$env:OLLAMA_API_KEY=""
```

5. Run app.
```powershell
python main.py
```

## Deploy On Render (Global Access)

1. Push code to GitHub.
2. In Render: `New` -> `Web Service` -> connect GitHub repo.
3. Use:
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn main:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
4. Add environment variables in Render:
- `FLASK_DEBUG=0`
- `FLASK_SECRET_KEY=<strong-random-secret>`
- `COOKIE_SECURE=1`
- `GOOGLE_CLIENT_ID=<google-oauth-client-id>`
- `GOOGLE_SEARCH_API_KEY=<google-custom-search-api-key>`
- `GOOGLE_SEARCH_CX=<programmable-search-engine-id>`
- `SEARCH_MAX_RESULTS=5`
- `OLLAMA_MODEL=ministral-3:14b-cloud`
- `OLLAMA_API_KEY=<your-ollama-api-key>`
- `MAX_MESSAGE_CHARS=4000`
- `RATE_LIMIT_WINDOW_SECONDS=60`
- `RATE_LIMIT_MAX_REQUESTS=20`
5. Deploy and open generated `onrender.com` URL.
6. Optional custom domain: add domain in Render settings and configure DNS records.

## Important Note About Ollama

If `OLLAMA_API_KEY` is set, app uses Ollama Cloud API (`https://ollama.com/api/chat` by default).
If API key is empty, app uses local/remote Ollama host via `OLLAMA_HOST` (default `http://127.0.0.1:11434`).

## Storage

Conversations are stored in SQLite: `data/conversations.db`.
On first startup, legacy `data/conversations.json` data is auto-migrated.

## Google Login

If `GOOGLE_CLIENT_ID` is set, users must sign in with Google and each user sees only their own conversations.
If empty, app stays public mode and uses shared `public` conversations.

## Real-Time Web Search

Set `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_CX` to enable live Google Programmable Search.
For queries needing fresh data (latest/current/today/news/price etc.), app auto-fetches web snippets and provides them to the model.
