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

2. Install backend dependencies:

```bash
cd apps/voice-agent/server
pip install -r requirements.txt
```

3. Start the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open:
   - Dashboard: `http://localhost:8000/`
   - Health: `http://localhost:8000/health`
   - Voice console: `http://localhost:8000/voice-console/`
