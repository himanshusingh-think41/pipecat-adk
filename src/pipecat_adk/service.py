"""ADK-based LLM service for Pipecat pipelines.

Replaces the standard LLM API call with Google ADK's runner.run_async,
bridging ADK's session-based agent invocations into Pipecat's frame pipeline.

Key design points:
- AdkLLMService is the only layer that knows the ADK session and invocation_id.
- turn_id (Vql layer) is mapped to invocation_id (ADK layer) here, in _turn_invocation_map.
- Session writes (including [HEARD] events) are centralised in this class.
- runner.run_async receives new_message directly — no pre-persisted events, no ResumabilityConfig.
"""

import asyncio
from typing import Any, Optional
from uuid import uuid4

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai.types import Content, Part
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import LLMService

from .aggregators import (
    VqlAssistantContextAggregator,
    VqlContextAggregatorPair,
    VqlUserContextAggregator,
)
from .frames import (
    VqlContextFrame,
    VqlFunctionCallInProgressFrame,
    VqlFunctionCallResultFrame,
    VqlFunctionCallsStartedFrame,
    VqlLLMFullResponseEndFrame,
    VqlLLMFullResponseStartFrame,
    VqlLLMTextFrame,
    VqlTurnCompletedFrame,
)
from .heard import HEARD_FORMAT
from .types import SessionParams


class AdkLLMService(LLMService):
    """LLM service that drives a Google ADK agent instead of a direct LLM call.

    Receives VqlContextFrame(turn_id, text) from VqlUserContextAggregator, calls
    runner.run_async(new_message=content) and streams the response as Vql frames.

    The mapping between pipecat's turn_id and ADK's invocation_id lives exclusively
    here in _turn_invocation_map.  No other layer knows or exposes invocation_id.

    Can be constructed two ways:

    1. Pass an ``App`` directly::

        llm = AdkLLMService(
            app=app,
            session_service=session_service,
            session_params=session_params,
        )

    2. Pass ``agent`` + optional ``plugins`` — the App is built internally::

        llm = AdkLLMService(
            agent=agent,
            plugins=[AdkInterruptionPlugin()],
            session_service=session_service,
            session_params=session_params,
        )

    Extension points for subclasses:
    - _build_user_event(text): customise the Content sent to the runner
    - _on_state_delta(state_delta): forward ADK state deltas to clients
    - _persist_and_run(content, state_delta): inject a programmatic event and run
    - process_frame: handle domain-specific frames by calling _persist_and_run
    """

    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        app: Optional[App] = None,
        agent: Optional[BaseAgent] = None,
        plugins: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ADK-based LLM service.

        Provide either ``app`` or ``agent`` (not both).

        Args:
            session_service: ADK session service (must already have the session created).
            session_params: Session identification (app_name, user_id, session_id).
            app: A pre-built ADK App.
            agent: The root ADK agent. Mutually exclusive with ``app``.
            plugins: ADK plugins (e.g. AdkInterruptionPlugin). Only used when
                building the App from ``agent``; ignored when ``app`` is provided.
            **kwargs: Additional keyword arguments passed to LLMService.
        """
        super().__init__(**kwargs)
        self.session_service = session_service
        self.session_params = session_params

        if app is not None and agent is not None:
            raise ValueError("Provide either 'app' or 'agent', not both.")
        if app is None and agent is None:
            raise ValueError("Either 'app' or 'agent' must be provided.")

        if app is None:
            app = App(
                name=session_params.app_name,
                root_agent=agent,
                plugins=plugins or [],
            )

        self.runner = Runner(app=app, session_service=session_service)

        # Maps pipecat turn_id → ADK invocation_id.
        # Populated when the first ADK event for a turn arrives; entries are
        # removed when VqlTurnCompletedFrame is received from the assistant
        # aggregator.  Only AdkLLMService knows both IDs.
        self._turn_invocation_map: dict[str, str] = {}

    def can_generate_metrics(self) -> bool:
        return True

    def create_context_aggregator(
        self,
        *,
        user_params: Optional[Any] = None,
        assistant_params: Optional[Any] = None,
    ) -> VqlContextAggregatorPair:
        """Create matching user and assistant aggregators for this service.

        Args:
            user_params: Optional LLMUserAggregatorParams override.
            assistant_params: Optional LLMAssistantAggregatorParams override.

        Returns:
            VqlContextAggregatorPair with .user() and .assistant() accessors.
        """
        user = VqlUserContextAggregator(params=user_params)
        assistant = VqlAssistantContextAggregator(params=assistant_params)
        return VqlContextAggregatorPair(_user=user, _assistant=assistant)

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        self._turn_invocation_map.clear()
        try:
            await self.runner.close()
        except Exception as e:
            logger.warning(f"Error closing ADK runner on stop: {e}")

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        self._turn_invocation_map.clear()
        try:
            await self.runner.close()
        except Exception as e:
            logger.warning(f"Error closing ADK runner on cancel: {e}")

    async def _process_context(self, context: LLMContext) -> None:
        """No-op: ADK context arrives via VqlContextFrame, not LLMContext."""
        pass

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        # Bypass LLMService's skip_tts injection.
        await AIService.push_frame(self, frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, VqlTurnCompletedFrame) and direction == FrameDirection.UPSTREAM:
            # Consumed here — do not propagate further upstream.
            invocation_id = self._turn_invocation_map.pop(frame.turn_id, None)
            if frame.interrupted and frame.text and invocation_id:
                await self._write_heard_event(invocation_id, frame.text)
            elif frame.interrupted and not invocation_id:
                logger.warning(
                    f"VqlTurnCompletedFrame(interrupted) for turn_id={frame.turn_id!r} "
                    f"but no invocation_id found in map — [HEARD] not written"
                )

        elif isinstance(frame, VqlContextFrame):
            content = await self._build_user_event(frame.text)
            await self._run_adk(frame.turn_id, content)

        elif not isinstance(frame, VqlTurnCompletedFrame):
            # Forward everything else (VqlTurnCompletedFrame upstream was handled above).
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # ADK runner
    # ------------------------------------------------------------------

    async def _run_adk(
        self,
        turn_id: str,
        content: Content,
        state_delta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Call runner.run_async with new_message and stream the response as Vql frames.

        ADK generates invocation_id internally; we learn it from the first event
        and store it in _turn_invocation_map[turn_id].

        Args:
            turn_id: The Vql turn identifier for this invocation.
            content: The user Content to pass as new_message.
            state_delta: Optional state changes forwarded to runner.run_async.
        """
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cache_read_input_tokens = 0
        reasoning_tokens = 0
        invocation_id = ""
        start_frame_pushed = False

        try:
            ttfb_stopped = False
            async for event in self.runner.run_async(
                user_id=self.session_params.user_id,
                session_id=self.session_params.session_id,
                new_message=content,
                state_delta=state_delta,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

                # Learn invocation_id from first event; store turn_id → invocation_id.
                if event.invocation_id and not invocation_id:
                    invocation_id = event.invocation_id
                    self._turn_invocation_map[turn_id] = invocation_id
                    logger.debug(
                        f"ADK invocation_id={invocation_id!r} "
                        f"mapped to turn_id={turn_id!r}"
                    )

                # Push start frame on first event so invocation_id is known.
                if not start_frame_pushed:
                    await self.push_frame(
                        VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id=invocation_id)
                    )
                    start_frame_pushed = True

                # Final event is authoritative for token counts (cumulative in SSE mode).
                if event.usage_metadata and not event.partial:
                    prompt_tokens = event.usage_metadata.prompt_token_count or 0
                    completion_tokens = event.usage_metadata.candidates_token_count or 0
                    total_tokens = event.usage_metadata.total_token_count or 0
                    cache_read_input_tokens = event.usage_metadata.cached_content_token_count or 0
                    reasoning_tokens = event.usage_metadata.thoughts_token_count or 0

                await self._push_frames_from_event(event, turn_id=turn_id, invocation_id=invocation_id)

        except asyncio.TimeoutError as e:
            logger.warning(f"{self} ADK timeout: {e}")
            await self._call_event_handler("on_completion_timeout")
            await self.push_error(error_msg="ADK runner timed out", exception=e)
        except Exception as e:
            logger.exception(f"{self} ADK error: {e}")
            await self.push_error(error_msg=f"ADK runner error: {e}", exception=e)
        finally:
            # Guarantee a matched start/end pair even if the runner threw before yielding.
            if not start_frame_pushed:
                await self.push_frame(
                    VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id=invocation_id)
                )
            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
            )
            await self.stop_processing_metrics()
            await self.push_frame(
                VqlLLMFullResponseEndFrame(turn_id=turn_id, invocation_id=invocation_id)
            )

    async def _push_frames_from_event(self, event: Event, *, turn_id: str, invocation_id: str) -> None:
        """Convert one ADK event into Pipecat frames.

        State deltas are forwarded to _on_state_delta before any text frames.
        Text is only emitted for partial events (final events repeat streamed text).
        Function call/response pairs are pushed both upstream and downstream.
        """
        if event.actions and event.actions.state_delta:
            await self._on_state_delta(event.actions.state_delta)

        if not event.content or not event.content.parts:
            return

        if event.partial:
            text_parts = [
                p.text
                for p in event.content.parts
                if p.text is not None and not getattr(p, "thought", False)
            ]
            if text_parts:
                text = "".join(text_parts)
                if text:
                    await self.push_frame(
                        VqlLLMTextFrame(text=text, turn_id=turn_id, invocation_id=invocation_id)
                    )

        for part in event.content.parts:
            if part.function_call:
                await self._handle_function_call(
                    part.function_call, turn_id=turn_id, invocation_id=invocation_id
                )
            elif part.function_response:
                await self._handle_function_response(
                    part.function_response, turn_id=turn_id, invocation_id=invocation_id
                )

    # ------------------------------------------------------------------
    # Session write: [HEARD] events (only session-write in the library)
    # ------------------------------------------------------------------

    async def _write_heard_event(self, invocation_id: str, heard_text: str) -> None:
        """Write a [HEARD] event to the ADK session for the given invocation.

        This is the only place in the library that writes directly to the ADK
        session.  The event format is consumed by AdkInterruptionPlugin's
        before_model_callback on the next LLM call.
        """
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            user_id=self.session_params.user_id,
            session_id=self.session_params.session_id,
        )
        if session is None:
            logger.warning(
                f"Cannot write [HEARD]: session not found "
                f"({self.session_params.session_id})"
            )
            return

        event = Event(
            invocation_id=Event.new_id(),
            author="user",
            content=Content(
                role="user",
                parts=[
                    Part(
                        text=HEARD_FORMAT.format(
                            invocation_id=invocation_id,
                            heard_text=heard_text,
                        )
                    )
                ],
            ),
        )
        await self.session_service.append_event(session, event)
        logger.info(
            f"Wrote [HEARD] for invocation_id={invocation_id!r} "
            f"session={self.session_params.session_id!r}"
        )

    # ------------------------------------------------------------------
    # Function call / response helpers
    # ------------------------------------------------------------------

    async def _handle_function_call(self, func_call, *, turn_id: str, invocation_id: str) -> None:
        assert func_call.id, "ADK function call must have an id"
        assert func_call.name, "ADK function call must have a name"

        func_call_from_llm = FunctionCallFromLLM(
            tool_call_id=func_call.id,
            function_name=func_call.name,
            arguments=func_call.args or {},
            context=None,
        )
        started = VqlFunctionCallsStartedFrame(
            function_calls=[func_call_from_llm],
            turn_id=turn_id,
            invocation_id=invocation_id,
        )
        await self.push_frame(started, FrameDirection.UPSTREAM)
        await self.push_frame(started, FrameDirection.DOWNSTREAM)

        in_progress = VqlFunctionCallInProgressFrame(
            tool_call_id=func_call.id,
            function_name=func_call.name,
            arguments=func_call.args,
            turn_id=turn_id,
            invocation_id=invocation_id,
        )
        await self.push_frame(in_progress, FrameDirection.UPSTREAM)
        await self.push_frame(in_progress, FrameDirection.DOWNSTREAM)

    async def _handle_function_response(self, func_response, *, turn_id: str, invocation_id: str) -> None:
        assert func_response.id, "ADK function response must have an id"
        assert func_response.name, "ADK function response must have a name"

        result = VqlFunctionCallResultFrame(
            tool_call_id=func_response.id,
            function_name=func_response.name,
            arguments=None,
            result=func_response.response or {},
            turn_id=turn_id,
            invocation_id=invocation_id,
        )
        await self.push_frame(result, FrameDirection.UPSTREAM)
        await self.push_frame(result, FrameDirection.DOWNSTREAM)

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    async def _build_user_event(self, text: str) -> Content:
        """Build the Content to pass as new_message to runner.run_async.

        Override in a subclass to wrap text in XML tags, append extra context
        parts (code diffs, timing messages), or otherwise enrich the message.
        Subclasses may access self.session_service / self.session_params to
        read session state.

        Args:
            text: The aggregated transcription of what the user said.

        Returns:
            A Content object passed directly to runner.run_async(new_message=...).
        """
        return Content(role="user", parts=[Part(text=text)])

    async def _on_state_delta(self, state_delta: dict) -> None:
        """Called for every ADK event that carries a state_delta.

        Override to forward state updates to clients (e.g. RTVI state-sync).
        Guaranteed to be called before any text frames from the same event.

        Args:
            state_delta: The state_delta dict from event.actions.state_delta.
        """
        pass

    async def _persist_and_run(
        self,
        content: Content,
        state_delta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Inject a programmatic event and run the agent.

        Use this from process_frame overrides to inject system events
        (silence warnings, domain events, etc.) and trigger the agent.
        A synthetic turn_id is generated; the resulting VqlTurnCompletedFrame
        will carry it but the content is not tracked for [HEARD] purposes.

        Args:
            content: The Content for the event (role="user" with system parts).
            state_delta: Optional state changes forwarded to runner.run_async.
        """
        turn_id = f"injected_{uuid4().hex}"
        await self._run_adk(turn_id, content, state_delta)
