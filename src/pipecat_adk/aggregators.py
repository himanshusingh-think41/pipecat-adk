"""Context aggregators for the ADK-Pipecat bridge.

ADK owns conversation history. These aggregators:
- Persist user speech to the ADK session and push AdkContextFrame
- Track per-invocation TTS text using a context_id → accumulation map
- Write [HEARD] events to the ADK session when an invocation ends interrupted
- Prevent ADK function call frames from polluting Pipecat's LLMContext
"""

from dataclasses import dataclass, field
from typing import Optional

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai.types import Content, Part
from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection

from .frames import AdkContextFrame, AdkLLMFullResponseStartFrame
from .heard import HEARD_FORMAT
from .types import SessionParams


@dataclass
class _InvocationAccumulation:
    """Per-invocation state: TTS text accumulated from actually-played audio."""

    texts: list[str] = field(default_factory=list)

    def heard_text(self) -> str:
        return " ".join(self.texts).strip()


class AdkUserContextAggregator(LLMUserAggregator):
    """Persists user speech to ADK session and triggers the LLM service.

    Overrides push_aggregation to bypass Pipecat's LLMContext entirely.
    After persisting the user event, pushes AdkContextFrame(invocation_id)
    so AdkBasedLLMService knows which pre-persisted event to resume.

    Override _build_user_event to inject application-specific context
    (section info, code diffs, timing messages, etc.) into the user event.
    """

    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        *,
        params: Optional[LLMUserAggregatorParams] = None,
    ) -> None:
        """Initialize the user context aggregator.

        Args:
            session_service: ADK session service for persisting events.
            session_params: Session identification (app_name, user_id, session_id).
            params: Optional aggregator params (turn strategies, idle timeout, etc.).
        """
        super().__init__(context=LLMContext(), params=params)
        self.session_service = session_service
        self.session_params = session_params

    async def push_aggregation(self) -> str:
        """Persist user speech to ADK session and push AdkContextFrame.

        Does NOT call super() — ADK owns conversation state, so we skip
        the standard LLMContext / LLMContextFrame path entirely.
        """
        if len(self._aggregation) == 0:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()

        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            user_id=self.session_params.user_id,
            session_id=self.session_params.session_id,
        )
        if session is None:
            raise RuntimeError(
                f"ADK session not found: {self.session_params.session_id}"
            )

        event = await self._build_user_event(aggregation, session)
        await self.session_service.append_event(session, event)
        logger.debug(f"Persisted user event invocation_id={event.invocation_id}")

        await self.push_frame(AdkContextFrame(invocation_id=event.invocation_id))
        return aggregation

    async def _build_user_event(self, text: str, session) -> Event:
        """Build the ADK Event for the user's speech.

        Override in a subclass to wrap text in XML tags, append extra context
        parts (code diffs, timing messages), or add a state_delta.

        Args:
            text: The aggregated transcription of what the user said.
            session: The current ADK Session (read state from session.state).

        Returns:
            An Event ready to be passed to session_service.append_event.
        """
        return Event(
            invocation_id=Event.new_id(),
            author="user",
            content=Content(role="user", parts=[Part(text=text)]),
        )


class AdkAssistantContextAggregator(LLMAssistantAggregator):
    """Tracks per-invocation TTS text; writes [HEARD] events on interrupted turns.

    Maintains a dict keyed by invocation_id (== TTS context_id) tracking the
    TTSTextFrame text chunks that actually played (forwarded upstream by transport).

    Lifecycle:
    - AdkLLMFullResponseStartFrame: creates an entry for the invocation
    - TTSTextFrame(context_id=X): appends text to entry X (upstream, after audio plays)
    - InterruptionFrame: writes [HEARD] for any entry with accumulated text; clears map
    - BotStoppedSpeakingFrame: clears map (clean turn, no [HEARD] needed)
    - TTSStoppedFrame(context_id=X): pops entry X without [HEARD] (unit-test clean turns;
      in production, BotStoppedSpeakingFrame handles cleanup instead)

    Also no-ops function call frame handlers so ADK function calls don't
    pollute Pipecat's LLMContext.
    """

    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        *,
        params: Optional[LLMAssistantAggregatorParams] = None,
    ) -> None:
        """Initialize the assistant context aggregator.

        Args:
            session_service: ADK session service for writing [HEARD] events.
            session_params: Session identification (app_name, user_id, session_id).
            params: Optional aggregator params.
        """
        # LLMContext() is required by the parent class but intentionally unused:
        # ADK owns conversation state, so push_aggregation() is a no-op.
        super().__init__(context=LLMContext(), params=params)
        self.session_service = session_service
        self.session_params = session_params
        # Keyed by invocation_id (== TTS context_id).
        self._invocations: dict[str, _InvocationAccumulation] = {}

    async def push_aggregation(self) -> str:
        """No-op: ADK already has the full assistant response in its session.

        Skips adding to LLMContext and pushing LLMContextFrame. Still resets
        _aggregation so state stays clean.
        """
        if not self._aggregation:
            return ""
        await self.reset()
        return ""

    async def _handle_text(self, frame: TextFrame) -> None:
        """Route TTSTextFrame to the per-invocation map; skip super() for all TTS text."""
        if isinstance(frame, TTSTextFrame):
            state = self._invocations.get(frame.context_id)
            if state is not None and frame.text:
                state.texts.append(frame.text)
            elif state is None:
                logger.debug(
                    f"AdkAssistantContextAggregator: dropped TTSTextFrame with "
                    f"unknown context_id={frame.context_id!r} (stale or non-ADK TTS)"
                )
            return  # Do NOT call super(); do NOT populate _aggregation.
        await super()._handle_text(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, AdkLLMFullResponseStartFrame):
            if frame.invocation_id in self._invocations:
                logger.warning(
                    f"AdkAssistantContextAggregator: overwriting existing entry for "
                    f"invocation_id={frame.invocation_id}"
                )
            self._invocations[frame.invocation_id] = _InvocationAccumulation()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Clean turn end — full response in ADK session, no [HEARD] needed.
            self._invocations.clear()
        elif isinstance(frame, TTSStoppedFrame) and frame.context_id:
            # Unit-test clean-turn signal: pop entry without writing [HEARD].
            # In production, BotStoppedSpeakingFrame handles clean-turn cleanup;
            # TTSStoppedFrame is not sent by the TTS service on interruption.
            self._invocations.pop(frame.context_id, None)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._invocations.clear()

        await super().process_frame(frame, direction)

    async def _handle_interruptions(self, frame: InterruptionFrame) -> None:
        """Write [HEARD] for all active invocations with accumulated text, then clear."""
        for inv_id, state in list(self._invocations.items()):
            if state.heard_text():
                await self._write_heard_event(state.heard_text(), inv_id)
        self._invocations.clear()
        await super()._handle_interruptions(frame)

    async def _write_heard_event(self, heard_text: str, invocation_id: str) -> None:
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            user_id=self.session_params.user_id,
            session_id=self.session_params.session_id,
        )
        if session is None:
            logger.warning(
                f"Cannot write [HEARD] event: session not found ({self.session_params.session_id})"
            )
            return

        event = Event(
            invocation_id=Event.new_id(),
            author="user",
            content=Content(
                role="user",
                parts=[Part(text=HEARD_FORMAT.format(
                    invocation_id=invocation_id,
                    heard_text=heard_text,
                ))],
            ),
        )
        await self.session_service.append_event(session, event)
        logger.info(
            f"Wrote [HEARD] event for invocation_id={invocation_id} "
            f"session={self.session_params.session_id}"
        )

    # ADK manages function call lifecycle internally. These no-ops prevent
    # function call frames from being added to LLMContext (which would cause
    # "No function call event found" errors). The frames still flow upstream/
    # downstream via AdkBasedLLMService to inform STTMuteFilter etc.

    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame) -> None:
        pass

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame) -> None:
        pass

    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame) -> None:
        pass


@dataclass
class AdkContextAggregatorPair:
    """Pair of user and assistant aggregators for ADK pipelines."""

    _user: AdkUserContextAggregator
    _assistant: AdkAssistantContextAggregator

    def user(self) -> AdkUserContextAggregator:
        return self._user

    def assistant(self) -> AdkAssistantContextAggregator:
        return self._assistant
