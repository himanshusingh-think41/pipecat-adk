"""Context aggregators for the ADK-Pipecat bridge.

ADK owns conversation history. These aggregators:
- Persist user speech to the ADK session and push AdkContextFrame
- Track what TTS text was spoken during the current bot turn
- Write [HEARD] events to the ADK session on interruption
- Prevent ADK function call frames from polluting Pipecat's LLMContext
"""

from dataclasses import dataclass
from typing import Optional

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai.types import Content, Part
from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection

from .frames import AdkContextFrame
from .types import SessionParams


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
    """Tracks spoken assistant text for the current bot turn.

    Accumulates text from TTSTextFrame into a flat buffer. At turn end:
    - Interrupted (InterruptionFrame): writes a [HEARD] event to the ADK session,
      then clears the buffer. AdkInterruptionPlugin reads [HEARD] markers in
      before_model_callback and truncates the preceding model event deterministically.
    - Clean (BotStoppedSpeakingFrame without prior interruption): clears the buffer
      with no [HEARD] event — the full response is already in the ADK session.

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
        super().__init__(context=LLMContext(), params=params)
        self.session_service = session_service
        self.session_params = session_params
        # Text chunks spoken this bot turn, in order of TTS emission.
        self._spoken_text: list[str] = []

    async def push_aggregation(self) -> str:
        """No-op: ADK already has the full assistant response in its session.

        Skips adding to LLMContext and pushing LLMContextFrame. Still resets
        _aggregation so state stays clean.
        """
        if not self._aggregation:
            return ""
        aggregation = self.aggregation_string()
        await self.reset()
        return aggregation

    async def _handle_text(self, frame) -> None:
        """Accumulate spoken text alongside parent tracking."""
        await super()._handle_text(frame)
        if isinstance(frame, TTSTextFrame) and frame.text:
            self._spoken_text.append(frame.text)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, BotStoppedSpeakingFrame):
            # Turn ended cleanly — full response already in ADK session, no [HEARD] needed.
            self._spoken_text.clear()
        await super().process_frame(frame, direction)

    async def _handle_interruptions(self, frame: InterruptionFrame) -> None:
        """Write a [HEARD] event for whatever was spoken before the interruption."""
        heard_text = " ".join(self._spoken_text).strip()
        if heard_text:
            await self._write_heard_event(heard_text)
        self._spoken_text.clear()
        await super()._handle_interruptions(frame)

    async def _write_heard_event(self, heard_text: str) -> None:
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
                parts=[Part(text=f'<system>[HEARD] Agent was interrupted. Candidate only heard: "{heard_text}"</system>')],
            ),
        )
        await self.session_service.append_event(session, event)
        logger.info(f"Wrote [HEARD] event for session {self.session_params.session_id}")

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
