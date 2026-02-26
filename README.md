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

3. Set environment variables.
```powershell
$env:FLASK_DEBUG="1"
$env:RESET_TOKEN="change-me"
$env:OLLAMA_MODEL="ministral-3:14b-cloud"
```

4. Run app.
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
- `RESET_TOKEN=<strong-random-secret>`
- `OLLAMA_MODEL=ministral-3:14b-cloud`
5. Deploy and open generated `onrender.com` URL.
6. Optional custom domain: add domain in Render settings and configure DNS records.

## Important Note About Ollama

This app uses `ollama.chat`. If Ollama model runtime is not reachable from your deployed server, chat responses will fail.

For production, use one of these:
- Host Ollama on the same server with enough CPU/GPU/RAM.
- Or switch to a hosted LLM API backend.
