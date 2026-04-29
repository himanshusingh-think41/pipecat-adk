"""TTS factory for ADK-aware audio context tracking.

Wraps any Pipecat TTSService subclass to:
1. Push AdkTTSSpeakingTextFrame before audio synthesis begins so
   AdkAssistantContextAggregator can record heard-text for [HEARD] events
   even when the bot is interrupted mid-speech.
2. Push AdkAudioContextCompletedFrame when an audio context finishes
   cleanly, so the aggregator can clear that context without writing [HEARD].
"""

from typing import AsyncGenerator

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection

from .frames import AdkAudioContextCompletedFrame, AdkTTSSpeakingTextFrame


def make_adk_aware_tts(base_class: type) -> type:
    """Wrap a TTSService subclass for ADK interruption tracking.

    The returned class is identical to base_class except:

    - run_tts pushes AdkTTSSpeakingTextFrame (context_id, text) before
      yielding any audio. This frame bypasses the audio context queue
      (it's a SystemFrame) and reaches AdkAssistantContextAggregator
      immediately, so [HEARD] events contain the actual spoken text even
      when the bot is interrupted after only partial audio has played.

    - on_audio_context_completed pushes AdkAudioContextCompletedFrame so
      the aggregator clears the context without writing [HEARD].

    Args:
        base_class: Any TTSService subclass (Google, ElevenLabs, Cartesia, etc.)

    Returns:
        A new class that inherits from base_class with the overrides applied.

    Example::

        from pipecat.services.cartesia import CartesiaTTSService
        from pipecat_adk import make_adk_aware_tts

        TTSService = make_adk_aware_tts(CartesiaTTSService)
        tts = TTSService(api_key=..., voice_id=...)
    """

    class _AdkAwareTTS(base_class):
        async def run_tts(self, text: str, context_id: str = "") -> AsyncGenerator[Frame, None]:
            # Push text immediately — before any audio frame — so the assistant
            # aggregator has it in its buffer at interruption time.
            # SystemFrame routes directly via push_frame (not through the audio
            # context queue), so it arrives downstream before audio starts.
            if context_id and text.strip():
                await self.push_frame(
                    AdkTTSSpeakingTextFrame(context_id=context_id, text=text),
                    FrameDirection.DOWNSTREAM,
                )
            async for frame in super().run_tts(text, context_id):
                yield frame

        async def on_audio_context_completed(self, context_id: str) -> None:
            await self.push_frame(
                AdkAudioContextCompletedFrame(context_id=context_id),
                FrameDirection.DOWNSTREAM,
            )
            await super().on_audio_context_completed(context_id)

    _AdkAwareTTS.__name__ = f"AdkAware{base_class.__name__}"
    _AdkAwareTTS.__qualname__ = f"AdkAware{base_class.__qualname__}"
    return _AdkAwareTTS
