"""Basic assistant bot using pipecat-adk with small-webrtc-prebuilt.

This example demonstrates:
- Google ADK agents for conversation management
- Pipecat for audio/video pipeline
- Automatic interruption handling via AdkInterruptionPlugin
- WebRTC transport via small-webrtc-prebuilt
"""

import os
import sys

from dotenv import load_dotenv
from google.adk.sessions import DatabaseSessionService
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.frames.frames import LLMRunFrame

from pipecat_adk import AdkLLMService, VqlTTSMixin, SessionParams
from agent import app
from debug_observer import AdkDebugObserver

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")


class AdkGoogleTTSService(VqlTTSMixin, GoogleTTSService):
    """GoogleTTSService with Vql turn_id pinning for [HEARD] tracking."""
    pass


async def run_bot(webrtc_connection):
    """Run the assistant bot with the given WebRTC connection."""

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Create session service backed by a local SQLite file
    db_path = os.path.join(os.path.dirname(__file__), "sessions.db")
    session_service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db_path}")

    # Create session parameters
    session_params = SessionParams(
        app_name=app.name,
        user_id="user",
        session_id="session-002",
    )
    # Create session only if it doesn't already exist (DB persists across restarts)
    existing = await session_service.get_session(**session_params.model_dump())
    if not existing:
        await session_service.create_session(**session_params.model_dump())

    # Create ADK-based LLM service with the app from agent.py.
    llm = AdkLLMService(
        session_service=session_service,
        session_params=session_params,
        app=app,
    )

    # Create context aggregators
    context_aggregator = llm.create_context_aggregator()

    # Create STT service (speech-to-text)
    stt = GoogleSTTService(
        params=GoogleSTTService.InputParams(
            languages=Language.EN_US,
            model="latest_long",
            enable_automatic_punctuation=True,
            enable_interim_results=True,
        )
    )

    # Create TTS service — VqlTTSMixin pins context_id to turn_id so
    # VqlAssistantContextAggregator can pass turn_id via TTSStoppedFrame
    # without storing any state.
    tts = AdkGoogleTTSService(
        voice_id="en-IN-Chirp3-HD-Achird",
        params=GoogleTTSService.InputParams(language=Language.EN_IN),
    )

    # Create transport with audio enabled
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection, params=transport_params
    )

    rtvi = RTVIProcessor()

    # Create pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Audio input from user
            rtvi,
            stt,  # Speech-to-text
            context_aggregator.user(),  # Package user input for ADK
            llm,  # ADK agent
            tts,  # Text-to-speech
            transport.output(),  # Audio output to user
            context_aggregator.assistant(),  # Track what was spoken; push VqlTurnCompletedFrame
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True
        ),
        observers=[RTVIObserver(rtvi), AdkDebugObserver()],
    )

    # Handle client connection
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Signal bot is ready to receive messages
        await rtvi.set_bot_ready()
        # Initialize the conversation
        await task.queue_frames([LLMRunFrame()])

    # Handle participant disconnection
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    # Add transport event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
