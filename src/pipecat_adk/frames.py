"""Frame definitions for the ADK-Pipecat bridge."""

from dataclasses import dataclass

from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame, SystemFrame


@dataclass
class AdkContextFrame(SystemFrame):
    """Carries invocation_id from user aggregator to LLM service.

    Pushed by AdkUserContextAggregator after persisting the user event.
    AdkBasedLLMService handles it by calling runner.run_async(invocation_id=...).
    """

    invocation_id: str


@dataclass
class AdkLLMFullResponseStartFrame(LLMFullResponseStartFrame):
    """LLMFullResponseStartFrame carrying the ADK invocation_id.

    Pushed by AdkBasedLLMService at the start of each runner.run_async call.
    AdkTTSMixin uses invocation_id to pin the TTS context_id, creating the
    provenance chain: invocation_id → TTS context_id → TTSTextFrame.context_id.
    """

    invocation_id: str = ""


@dataclass
class AdkLLMTextFrame(LLMTextFrame):
    """LLMTextFrame carrying the ADK invocation_id; excluded from LLMContext.

    append_to_context=False prevents LLMAssistantAggregator from accumulating
    this frame — only TTSTextFrame (actually-played audio) contributes to the
    assistant context via AdkAssistantContextAggregator's per-invocation map.
    """

    invocation_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.append_to_context = False
