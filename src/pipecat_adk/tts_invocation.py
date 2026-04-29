"""TTS factory for ADK-aware audio context tracking.

Wraps any Pipecat TTSService subclass to push AdkAudioContextCompletedFrame
when an audio context finishes playing. The assistant aggregator uses this
signal to clear completed contexts from its [HEARD] tracking buffer.
"""

from pipecat.processors.frame_processor import FrameDirection

from .frames import AdkAudioContextCompletedFrame


def make_adk_aware_tts(base_class: type) -> type:
    """Wrap a TTSService subclass to signal audio context completion.

    The returned class is identical to base_class except that
    on_audio_context_completed pushes AdkAudioContextCompletedFrame
    downstream before delegating to the base implementation.

    Args:
        base_class: Any TTSService subclass (Google, ElevenLabs, Cartesia, etc.)

    Returns:
        A new class that inherits from base_class with the override applied.

    Example::

        from pipecat.services.cartesia import CartesiaTTSService
        from pipecat_adk import make_adk_aware_tts

        TTSService = make_adk_aware_tts(CartesiaTTSService)
        tts = TTSService(api_key=..., voice_id=...)
    """

    class _AdkAwareTTS(base_class):
        async def on_audio_context_completed(self, context_id: str) -> None:
            await self.push_frame(
                AdkAudioContextCompletedFrame(context_id=context_id),
                FrameDirection.DOWNSTREAM,
            )
            await super().on_audio_context_completed(context_id)

    _AdkAwareTTS.__name__ = f"AdkAware{base_class.__name__}"
    _AdkAwareTTS.__qualname__ = f"AdkAware{base_class.__qualname__}"
    return _AdkAwareTTS
