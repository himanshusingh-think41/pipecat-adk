"""Frame definitions for the ADK-Pipecat bridge."""

from dataclasses import dataclass

from pipecat.frames.frames import SystemFrame


@dataclass
class AdkContextFrame(SystemFrame):
    """Carries invocation_id from user aggregator to LLM service.

    Pushed by AdkUserContextAggregator after persisting the user event.
    AdkBasedLLMService handles it by calling runner.run_async(invocation_id=...).
    """

    invocation_id: str


@dataclass
class AdkAudioContextCompletedFrame(SystemFrame):
    """Signals that a TTS audio context played to completion without interruption.

    Pushed downstream by the make_adk_aware_tts factory when
    on_audio_context_completed fires. AdkAssistantContextAggregator handles
    this to remove the context from its per-context accumulation buffer,
    meaning no [HEARD] event is needed for it.
    """

    context_id: str
