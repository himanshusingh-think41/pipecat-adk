"""Context aggregators for the Vql-Pipecat bridge.

The Vql aggregators own the pipecat side of the conversation boundary:
- VqlUserContextAggregator: mints a turn_id per user turn, pushes VqlContextFrame.
- VqlAssistantContextAggregator: accumulates spoken TTS text, pushes VqlTurnCompletedFrame
  upstream to AdkLLMService (which then decides whether to write a [HEARD] event).

Neither aggregator writes to the ADK session — that responsibility stays entirely
in AdkLLMService.  The session_service / session_params that the old aggregators
needed are gone; see docs/turn-id-propagation.md for the full flow.
"""

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from .frames import (
    VqlContextFrame,
    VqlInterruptionFrame,
    VqlTurnCompletedFrame,
)


class VqlUserContextAggregator(LLMUserAggregator):
    """User context aggregator for Vql pipelines.

    Generates a fresh turn_id (UUID hex) on each user turn and pushes
    VqlContextFrame(turn_id, text) so AdkLLMService can route to the ADK runner.
    No ADK session writes happen here — the text travels to AdkLLMService via the frame.

    State exception (the only self._ variable):
        _prev_turn_id: the turn_id of the most recent completed turn.  Stored so
        that broadcast_interruption() can annotate VqlInterruptionFrame with the
        turn that is being cut off, letting VqlAssistantContextAggregator track
        the [HEARD] text without needing its own state.

    Extension point: override push_aggregation() to customize VqlContextFrame.
    """

    def __init__(self, *, params: Optional[LLMUserAggregatorParams] = None) -> None:
        super().__init__(context=LLMContext(), params=params)
        self._prev_turn_id: Optional[str] = None

    async def push_aggregation(self) -> str:
        """Push VqlContextFrame(turn_id, text) — skip Pipecat LLMContext entirely."""
        if len(self._aggregation) == 0:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()

        turn_id = uuid4().hex
        await self.push_frame(VqlContextFrame(turn_id=turn_id, text=aggregation))
        self._prev_turn_id = turn_id
        return aggregation

    async def broadcast_interruption(self) -> None:
        # COPIED from FrameProcessor.broadcast_interruption @ pipecat 06233f53e
        # CHANGED: when _prev_turn_id is set, emit VqlInterruptionFrame(turn_id=prev_turn_id)
        #          instead of plain InterruptionFrame.  This carries the interrupted turn's
        #          id downstream so VqlAssistantContextAggregator never needs to store it.
        #          Falls back to super() when there is no previous turn (first user turn).
        if self._prev_turn_id is None:
            await super().broadcast_interruption()
            return

        logger.debug(f"{self}: broadcasting Vql interruption for turn_id={self._prev_turn_id}")
        # Access the name-mangled queue-reset method.  This is the standard Python
        # pattern when a single-inheritance override needs to replicate a parent
        # private helper without copying the entire method body.
        self._FrameProcessor__reset_process_task()  # type: ignore[attr-defined]
        await self.stop_all_metrics()
        await self.broadcast_frame(VqlInterruptionFrame, turn_id=self._prev_turn_id)


class VqlAssistantContextAggregator(LLMAssistantAggregator):
    """Assistant context aggregator for Vql pipelines.

    Accumulates TTS text (what was actually spoken) and, when a turn ends,
    pushes VqlTurnCompletedFrame(turn_id, text, interrupted) upstream.
    AdkLLMService receives this and decides whether to write a [HEARD] event.

    No state for turn_id — it flows through frames:
    - VqlInterruptionFrame.turn_id (interrupted turns)
    - TTSStoppedFrame.context_id = turn_id (clean turns, via VqlTTSMixin)

    No ADK session writes here; no _invocations dict.
    """

    def __init__(self, *, params: Optional[LLMAssistantAggregatorParams] = None) -> None:
        # LLMContext() satisfies the parent but is intentionally unused;
        # ADK owns conversation state, so push_aggregation never writes to context.
        super().__init__(context=LLMContext(), params=params)

    # ------------------------------------------------------------------
    # Core frame routing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, VqlInterruptionFrame):
            # COPIED behavior from LLMAssistantAggregator._handle_interruptions
            # @ pipecat 06233f53e
            # CHANGED: pass turn_id from frame (no self._current_turn_id state);
            #          _trigger_assistant_turn_stopped pushes VqlTurnCompletedFrame
            #          upstream instead of writing to LLMContext.
            await self._trigger_assistant_turn_stopped(
                turn_id=frame.turn_id, interrupted=True
            )
            await self.reset()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStoppedFrame) and frame.context_id:
            # context_id == turn_id here because VqlTTSMixin pins _turn_context_id
            # to VqlLLMFullResponseStartFrame.turn_id (see tts_mixin.py).
            await self._trigger_assistant_turn_stopped(
                turn_id=frame.context_id, interrupted=False
            )
            await self.push_frame(frame, direction)

        else:
            await super().process_frame(frame, direction)

    # ------------------------------------------------------------------
    # Turn completion — carries turn_id as a parameter, never stored
    # ------------------------------------------------------------------

    async def _trigger_assistant_turn_stopped(  # type: ignore[override]
        self, *, turn_id: str = "", interrupted: bool = False
    ) -> None:
        # COPIED signature pattern from LLMAssistantAggregator._trigger_assistant_turn_stopped
        # @ pipecat 06233f53e
        # CHANGED: accepts turn_id parameter (propagated from the triggering frame,
        #          not from self state); pushes VqlTurnCompletedFrame upstream instead
        #          of writing aggregation to LLMContext.
        text = self.aggregation_string()
        await self.reset()
        if text and turn_id:
            await self.push_frame(
                VqlTurnCompletedFrame(turn_id=turn_id, text=text, interrupted=interrupted),
                FrameDirection.UPSTREAM,
            )

    # ------------------------------------------------------------------
    # Overrides to prevent parent from writing to LLMContext
    # ------------------------------------------------------------------

    async def push_aggregation(self) -> str:
        """Return accumulated text without writing to LLMContext.

        ADK already holds the full assistant response in its session journal.
        Writing to Pipecat's LLMContext would duplicate state that ADK owns.
        """
        if not self._aggregation:
            return ""
        result = self.aggregation_string()
        await self.reset()
        return result

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame) -> None:
        # No-op: TTSStoppedFrame.context_id drives turn completion in this pipeline,
        # not LLMFullResponseEndFrame.  The LLM response may end before all TTS audio
        # has played; we need to wait until TTS stops to know what was actually spoken.
        pass

    async def _handle_interruptions(self, frame) -> None:
        # No-op: VqlInterruptionFrame (intercepted in process_frame above) carries
        # the turn_id and drives interruption handling.  Plain InterruptionFrame
        # should not reach here in a correctly wired Vql pipeline.
        pass

    async def _handle_push_aggregation(self) -> None:
        # No-op: TTSSpeakFrame-driven utterances (greetings etc.) are managed by
        # the ADK agent and do not need pipecat context updates.
        pass

    # ------------------------------------------------------------------
    # Function call no-ops — ADK manages tool calls in its session journal;
    # letting these frames into LLMContext produces malformed context entries.
    # ------------------------------------------------------------------

    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame) -> None:
        pass

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame) -> None:
        pass

    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame) -> None:
        pass


@dataclass
class VqlContextAggregatorPair:
    """Pair of Vql user and assistant aggregators for ADK pipelines."""

    _user: VqlUserContextAggregator
    _assistant: VqlAssistantContextAggregator

    def user(self) -> VqlUserContextAggregator:
        return self._user

    def assistant(self) -> VqlAssistantContextAggregator:
        return self._assistant
