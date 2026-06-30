# Voice Agent App

Single-agent dashboard built inside the forked `pipecat-adk` repo.

## Structure

- `client/`: lightweight dashboard UI served by the backend
- `server/`: FastAPI backend and Pipecat ADK runtime

## Current Runtime

- LLM: Gemini
- STT: Deepgram
- TTS: Deepgram
- Voice transport: SmallWebRTC prebuilt UI mounted at `/voice-console/`

## Local Run

1. Configure environment:
   - Copy `server/.env.example` to `server/.env`
   - Fill in `GEMINI_API_KEY`, `STT_API_KEY`, and `TTS_API_KEY`
   - `STT_PROVIDER` and `TTS_PROVIDER` are expected to remain `deepgram`

2. Install backend dependencies:

```bash
cd apps/voice-agent/server
pip install -r requirements.txt
```

If you use the repo virtualenv:

```bash
/home/think41/pipecat-adk/.venv/bin/pip install -r requirements.txt
```

3. Start the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or with the repo virtualenv:

```bash
/home/think41/pipecat-adk/.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open:
   - Dashboard: `http://localhost:8000/`
   - Health: `http://localhost:8000/health`
   - Voice console: `http://localhost:8000/voice-console/`

## Verification Order

1. `GET /health`
2. Open the dashboard at `/`
3. Submit one chat message through the dashboard
4. Open `/voice-console/` and test microphone flow

If the backend fails on startup with a Deepgram import error, reinstall dependencies from `server/requirements.txt` because the Deepgram SDK is required by the runtime.

If `/api/chat` returns a runtime error, check these first:
- `GEMINI_API_KEY` is present
- `STT_API_KEY` is present
- `TTS_API_KEY` is present, or you intentionally reuse `STT_API_KEY`
- `ADK_DATABASE_URL` points to a reachable Postgres instance
