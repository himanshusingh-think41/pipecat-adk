"""AdkTTSMixin: pins TTS context_id to ADK invocation_id for provenance tracking."""

from typing import Optional

from pipecat.processors.frame_processor import FrameDirection

from .frames import AdkLLMFullResponseStartFrame


class AdkTTSMixin:
    """Mixin for TTS services that pins _turn_context_id to the ADK invocation_id.

    Apply to any concrete TTS service:

        class MyTTSService(AdkTTSMixin, ElevenLabsTTSService):
            pass

    When AdkBasedLLMService pushes AdkLLMFullResponseStartFrame(invocation_id="inv1"),
    this mixin sets _turn_context_id = "inv1" instead of generating a UUID. All
    TTSTextFrame and TTSAudioRawFrame instances for that response carry
    context_id = "inv1", enabling AdkAssistantContextAggregator to correlate
    played audio with the originating ADK invocation.

    TTSSpeakFrame audio is unaffected: TTSSpeakFrame temporarily clears
    _turn_context_id and generate_context_id generates a fresh UUID, which is
    never registered in the aggregator's _invocations map and is silently ignored.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_adk_invocation_id: Optional[str] = None

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        if (
            isinstance(frame, AdkLLMFullResponseStartFrame)
            and frame.invocation_id
            and direction == FrameDirection.DOWNSTREAM
        ):
            self._pending_adk_invocation_id = frame.invocation_id
        await super().process_frame(frame, direction)
        if isinstance(frame, AdkLLMFullResponseStartFrame):
            self._pending_adk_invocation_id = None

    def create_context_id(self) -> str:
        # On a new turn (_turn_context_id is None) for an ADK invocation: use
        # invocation_id as context_id. The check `not self._turn_context_id`
        # ensures TTSSpeakFrame calls (which reset _turn_context_id) fall through
        # to super() and get a fresh UUID.
        if self._pending_adk_invocation_id and not self._turn_context_id:
            return self._pending_adk_invocation_id
        return super().create_context_id()
