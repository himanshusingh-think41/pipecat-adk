import os
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.genai import types
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramHttpTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat_adk import AdkLLMService, SessionParams, VqlTTSMixin

from app.core.config import get_settings
from app.runtime.agent import build_app
from app.runtime.debug_observer import AdkDebugObserver

logger.remove(0)
logger.add(sys.stderr, level="INFO")


class AdkDeepgramTTSService(VqlTTSMixin, DeepgramHttpTTSService):
    """Deepgram HTTP TTS with Vql turn_id pinning for interruption tracking."""


class VoiceAgentLLMService(AdkLLMService):
    """ADK service that injects current local date and time into every user turn."""

    async def _build_user_event(self, text: str) -> types.Content:
        return types.Content(
            role="user",
            parts=[
                types.Part(text=_build_runtime_context()),
                types.Part(text=text),
            ],
        )


def _configure_google_api_key() -> None:
    settings = get_settings()
    settings.validate_runtime_settings()
    os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key)
    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)


def _validate_audio_providers() -> None:
    get_settings().validate_runtime_settings()


def _build_runtime_context() -> str:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return (
        "<system>"
        f"Current local date is {now.strftime('%B %d, %Y')}. "
        f"Current local time is {now.strftime('%I:%M %p')} IST. "
        "When the user asks about today, date, time, now, or current day, use this context "
        "instead of guessing from model knowledge."
        "</system>"
    )


async def run_bot(webrtc_connection, *, session_id: str, user_id: str = "local-user"):
    """Run the voice agent pipeline for one WebRTC connection."""

    _configure_google_api_key()
    _validate_audio_providers()
    settings = get_settings()
    app = build_app()

    session_service = DatabaseSessionService(db_url=settings.adk_database_url)
    session_params = SessionParams(
        app_name=app.name,
        user_id=user_id,
        session_id=session_id,
    )
    existing = await session_service.get_session(**session_params.model_dump())
    if not existing:
        await session_service.create_session(**session_params.model_dump())

    llm = VoiceAgentLLMService(
        session_service=session_service,
        session_params=session_params,
        app=app,
    )
    context_aggregator = llm.create_context_aggregator()

    stt = DeepgramSTTService(
        api_key=settings.stt_api_key,
        settings=DeepgramSTTService.Settings(
            language=Language.EN_US,
            model="nova-3-general",
            interim_results=True,
            punctuate=True,
            smart_format=True,
        ),
    )

    async with aiohttp.ClientSession() as aiohttp_session:
        tts = AdkDeepgramTTSService(
            api_key=settings.effective_tts_api_key,
            aiohttp_session=aiohttp_session,
            settings=DeepgramHttpTTSService.Settings(
                voice="aura-2-thalia-en",
            ),
            sample_rate=24000,
        )

        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        rtvi = RTVIProcessor()

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi), AdkDebugObserver()],
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi_processor):
            await rtvi_processor.set_bot_ready()
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(_transport, _participant, _reason):
            await task.cancel()

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client):
            logger.info("Client connected")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)


def _build_session_service() -> DatabaseSessionService:
    return DatabaseSessionService(db_url=get_settings().adk_database_url)


async def _ensure_adk_session(
    session_service: DatabaseSessionService,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
) -> None:
    existing = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if existing is None:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )


async def generate_agent_response(
    *,
    session_id: str,
    user_text: str,
    user_id: str = "local-user",
) -> str:
    """Generate a text response through the same ADK app used by the voice runtime."""

    _configure_google_api_key()
    app = build_app()
    session_service = _build_session_service()
    await _ensure_adk_session(
        session_service,
        app_name=app.name,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(app=app, session_service=session_service)
    message = types.Content(
        role="user",
        parts=[
            types.Part(text=_build_runtime_context()),
            types.Part(text=user_text),
        ],
    )
    response_parts: list[str] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        content: Optional[types.Content] = getattr(event, "content", None)
        if not content or not content.parts:
            continue
        for part in content.parts:
            if getattr(part, "text", None):
                response_parts.append(part.text)

    response_text = "".join(response_parts).strip()
    return response_text or "I heard you, but I could not produce a response."
