"""Frame definitions for the Vql-Pipecat bridge.

Vql-prefixed frames are the pipecat-layer abstractions that flow between
VqlUserContextAggregator, the LLM service, VqlTTSMixin, and
VqlAssistantContextAggregator.  They carry turn_id but never expose ADK
internals (invocation_id stays private to AdkLLMService).
"""

from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    SystemFrame,
)


@dataclass
class VqlContextFrame(Frame):
    """Carries turn_id and user text from user aggregator to LLM service.

    VqlUserContextAggregator pushes this after a user turn completes.
    AdkLLMService handles it by building the ADK Content and calling
    runner.run_async(new_message=content).

    Must be a plain Frame (not SystemFrame) so that it is routed through
    __process_frame_task rather than processed inline in __input_frame_task.
    This allows VqlInterruptionFrame (a SystemFrame) to be handled by
    __input_frame_task concurrently, which calls _start_interruption() and
    cancels __process_frame_task — killing _run_adk() mid-stream.
    """

    turn_id: str = ""
    text: str = ""


@dataclass
class VqlInterruptionFrame(InterruptionFrame):
    """InterruptionFrame annotated with the turn_id that is being interrupted.

    Broadcast by VqlUserContextAggregator in _on_user_turn_started instead of
    the plain InterruptionFrame.  Carries the *previous* turn's turn_id so that
    VqlAssistantContextAggregator can attribute the partial [HEARD] text to the
    correct turn without storing any state of its own.
    """

    turn_id: str = ""


@dataclass
class VqlLLMFullResponseStartFrame(LLMFullResponseStartFrame):
    """LLMFullResponseStartFrame carrying the pipecat-layer turn_id.

    Pushed by AdkLLMService at the start of each runner.run_async response.
    VqlTTSMixin reads turn_id to pin _turn_context_id, creating the provenance
    chain: turn_id → TTS context_id → TTSTextFrame.context_id → TTSStoppedFrame.context_id.

    invocation_id is intentionally absent — it is ADK-internal and lives only
    in AdkLLMService._turn_invocation_map.
    """

    turn_id: str = ""


@dataclass
class VqlLLMTextFrame(LLMTextFrame):
    """LLMTextFrame carrying the pipecat-layer turn_id; excluded from LLMContext.

    append_to_context=False prevents LLMAssistantAggregator from accumulating
    this frame — only TTSTextFrame (actually-played audio) contributes to the
    assistant context via VqlAssistantContextAggregator.
    """

    turn_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.append_to_context = False


@dataclass
class VqlTurnCompletedFrame(SystemFrame):
    """Upstream signal from VqlAssistantContextAggregator to AdkLLMService.

    Pushed upstream when a bot turn finishes (cleanly or interrupted).
    AdkLLMService uses turn_id to look up the ADK invocation_id and, when
    interrupted=True, writes the [HEARD] event to the ADK session.

    Fields:
        turn_id:     The pipecat turn identifier for this completed turn.
        text:        The text that was actually spoken (from accumulated TTSTextFrame).
        interrupted: True if user interrupted mid-turn; False for clean completion.
    """

    turn_id: str = ""
    text: str = ""
    interrupted: bool = False
