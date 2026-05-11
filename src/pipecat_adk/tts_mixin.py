"""VqlTTSMixin: pins TTS _turn_context_id to the Vql turn_id.

Pipecat's TTSService normally calls create_context_id() on every
LLMFullResponseStartFrame to generate a fresh UUID for the turn.  This mixin
overrides that block to use VqlLLMFullResponseStartFrame.turn_id instead, so
context_id == turn_id throughout the TTS/transport/aggregator pipeline.

Usage::

    class MyTTSService(VqlTTSMixin, ElevenLabsTTSService):
        pass

No new member variables are introduced.  The existing _turn_context_id
attribute (owned by pipecat's TTSService) is set directly from the frame.
"""

from pipecat.frames.frames import LLMFullResponseStartFrame
from pipecat.processors.frame_processor import FrameDirection

from .frames import VqlLLMFullResponseStartFrame


class VqlTTSMixin:
    """Mixin for TTS services that pins _turn_context_id to the Vql turn_id.

    Apply before the concrete TTS service in the MRO:

        class AdkGoogleTTSService(VqlTTSMixin, GoogleTTSService):
            pass

    When AdkLLMService pushes VqlLLMFullResponseStartFrame(turn_id="t1"),
    this mixin intercepts it and sets self._turn_context_id = "t1" directly,
    bypassing create_context_id().  All TTSTextFrame and TTSAudioRawFrame
    instances for that response carry context_id == turn_id, enabling
    VqlAssistantContextAggregator to receive the turn_id via TTSStoppedFrame
    without storing any state of its own.

    create_context_id() is effectively dead code for the LLM-response path.
    It may still be called for TTSSpeakFrame-driven utterances; that is safe
    because those utterances are not tracked by VqlAssistantContextAggregator.
    """

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMFullResponseStartFrame):
            # COPIED from TTSService.process_frame LLMFullResponseStartFrame branch
            # @ pipecat 06233f53e (tts_service.py:681-686)
            # CHANGED: assert frame is VqlLLMFullResponseStartFrame — this mixin
            #          requires AdkLLMService to be the LLM service; set
            #          self._turn_context_id = frame.turn_id instead of calling
            #          create_context_id(), which makes turn_id the canonical TTS
            #          context id for the entire downstream pipeline.
            if not isinstance(frame, VqlLLMFullResponseStartFrame):
                raise RuntimeError(
                    f"VqlTTSMixin requires VqlLLMFullResponseStartFrame but received "
                    f"{type(frame).__name__}.  Ensure AdkLLMService (or another Vql "
                    f"LLM service) is used in this pipeline."
                )
            self._llm_response_started = True  # type: ignore[attr-defined]
            self._turn_context_id = frame.turn_id  # type: ignore[attr-defined]
            await self.on_turn_context_created(self._turn_context_id)  # type: ignore[attr-defined]
            await self.push_frame(frame, direction)
        else:
            await super().process_frame(frame, direction)  # type: ignore[misc]
