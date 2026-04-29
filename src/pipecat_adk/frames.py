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


@dataclass
class AdkTTSSpeakingTextFrame(SystemFrame):
    """Carries TTS text immediately before audio synthesis begins.

    Pushed by make_adk_aware_tts at the start of run_tts, before any audio
    frames are yielded. Because it is a SystemFrame it bypasses the audio
    context queue and reaches AdkAssistantContextAggregator immediately,
    giving the aggregator the heard-text to use in a [HEARD] event if the
    bot is interrupted mid-speech.

    TTSTextFrame (the standard pipecat frame) is intentionally appended
    *after* all audio, so it arrives too late for interruption tracking.
    This frame fills that gap.
    """

    context_id: str
    text: str
