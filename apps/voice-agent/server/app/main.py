from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from app.api.routes import router, small_webrtc_handler
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await small_webrtc_handler.close()


settings = get_settings()
client_root = Path(__file__).resolve().parents[2] / "client"

app = FastAPI(title="Voice Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/voice-console", SmallWebRTCPrebuiltUI)
app.mount("/", StaticFiles(directory=client_root, html=True), name="client")
