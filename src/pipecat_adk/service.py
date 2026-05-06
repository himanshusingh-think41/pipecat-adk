"""ADK-based LLM service for Pipecat pipelines.

Replaces the standard LLM API call with Google ADK's runner.run_async,
bridging ADK's session-based agent invocations into Pipecat's frame pipeline.
"""

from typing import Any, Optional, Union

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai.types import Content, Part
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import LLMService

from .aggregators import (
    AdkAssistantContextAggregator,
    AdkContextAggregatorPair,
    AdkUserContextAggregator,
)
from .frames import AdkContextFrame, AdkLLMFullResponseStartFrame, AdkLLMTextFrame
from .types import SessionParams


class AdkBasedLLMService(LLMService):
    """LLM service that drives a Google ADK agent instead of a direct LLM call.

    Receives AdkContextFrame from AdkUserContextAggregator (the user event has
    already been persisted) and calls runner.run_async(invocation_id=...) to
    resume the invocation.

    Can be constructed two ways:

    1. Pass an ``App`` directly — full control over ADK App config::

        llm = AdkBasedLLMService(
            app=app,
            session_service=session_service,
            session_params=session_params,
        )

    2. Pass ``agent`` + optional ``plugins`` — the App is built internally::

        llm = AdkBasedLLMService(
            agent=agent,
            plugins=[AdkInterruptionPlugin()],
            session_service=session_service,
            session_params=session_params,
        )

    Either way, the App must have ``ResumabilityConfig(is_resumable=True)``; the
    service validates this and raises ``ValueError`` if it is missing.

    Extension points for subclasses:
    - _on_state_delta(state_delta): called for every ADK event that carries state
    - _persist_and_run(content, state_delta): inject a system event and run the agent
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
            app: A pre-built ADK App. Must have ResumabilityConfig(is_resumable=True).
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
                resumability_config=ResumabilityConfig(is_resumable=True),
                plugins=plugins or [],
            )
        else:
            if not (app.resumability_config and app.resumability_config.is_resumable):
                raise ValueError(
                    "The App passed to AdkBasedLLMService must have "
                    "ResumabilityConfig(is_resumable=True). "
                    "See docs/invocation-id-and-resumability.md for why this is required."
                )

        self.runner = Runner(app=app, session_service=session_service)

    def create_context_aggregator(
        self,
        *,
        user_params: Optional[Any] = None,
        assistant_params: Optional[Any] = None,
    ) -> AdkContextAggregatorPair:
        """Create matching user and assistant aggregators for this service.

        Args:
            user_params: Optional LLMUserAggregatorParams override.
            assistant_params: Optional LLMAssistantAggregatorParams override.

        Returns:
            AdkContextAggregatorPair with .user() and .assistant() accessors.
        """
        user = AdkUserContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            params=user_params,
        )
        assistant = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            params=assistant_params,
        )
        return AdkContextAggregatorPair(_user=user, _assistant=assistant)

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        try:
            await self.runner.close()
        except Exception as e:
            logger.warning(f"Error closing ADK runner on stop: {e}")

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        try:
            await self.runner.close()
        except Exception as e:
            logger.warning(f"Error closing ADK runner on cancel: {e}")

    async def _process_context(self, context: LLMContext) -> None:
        """No-op: ADK context arrives via AdkContextFrame, not LLMContext."""
        pass

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        # Bypass LLMService's skip_tts injection — we manage TTS skipping explicitly
        # when needed via frame.skip_tts on individual frames.
        await AIService.push_frame(self, frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, AdkContextFrame):
            await self._run_adk(frame.invocation_id)
        else:
            await self.push_frame(frame, direction)

    async def _run_adk(self, invocation_id: str) -> None:
        """Resume a pre-persisted ADK invocation and stream the response.

        The caller must have already created and persisted an Event with
        invocation_id to the session before calling this method.

        Args:
            invocation_id: The invocation_id of the pre-persisted user event.
        """
        await self.push_frame(AdkLLMFullResponseStartFrame(invocation_id=invocation_id))
        await self.start_ttfb_metrics()

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cache_read_input_tokens = 0
        reasoning_tokens = 0

        try:
            async for event in self.runner.run_async(
                user_id=self.session_params.user_id,
                session_id=self.session_params.session_id,
                invocation_id=invocation_id,
                new_message=None,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                await self.stop_ttfb_metrics()

                # Final event is authoritative for token counts (cumulative in SSE mode)
                if event.usage_metadata and not event.partial:
                    prompt_tokens = event.usage_metadata.prompt_token_count or 0
                    completion_tokens = event.usage_metadata.candidates_token_count or 0
                    total_tokens = event.usage_metadata.total_token_count or 0
                    cache_read_input_tokens = event.usage_metadata.cached_content_token_count or 0
                    reasoning_tokens = event.usage_metadata.thoughts_token_count or 0

                await self._push_frames_from_event(event)

        except Exception as e:
            logger.exception(f"{self} ADK error: {e}")
        finally:
            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())

    async def _push_frames_from_event(self, event: Event) -> None:
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
                        AdkLLMTextFrame(text=text, invocation_id=event.invocation_id)
                    )

        for part in event.content.parts:
            if part.function_call:
                await self._handle_function_call(part.function_call)
            elif part.function_response:
                await self._handle_function_response(part.function_response)

    async def _handle_function_call(self, func_call) -> None:
        assert func_call.id, "ADK function call must have an id"
        assert func_call.name, "ADK function call must have a name"

        func_call_from_llm = FunctionCallFromLLM(
            tool_call_id=func_call.id,
            function_name=func_call.name,
            arguments=func_call.args or {},
            context=None,
        )
        started = FunctionCallsStartedFrame(function_calls=[func_call_from_llm])
        await self.push_frame(started, FrameDirection.UPSTREAM)
        await self.push_frame(started, FrameDirection.DOWNSTREAM)

        in_progress = FunctionCallInProgressFrame(
            tool_call_id=func_call.id,
            function_name=func_call.name,
            arguments=func_call.args,
        )
        await self.push_frame(in_progress, FrameDirection.UPSTREAM)
        await self.push_frame(in_progress, FrameDirection.DOWNSTREAM)

    async def _handle_function_response(self, func_response) -> None:
        assert func_response.id, "ADK function response must have an id"
        assert func_response.name, "ADK function response must have a name"

        result = FunctionCallResultFrame(
            tool_call_id=func_response.id,
            function_name=func_response.name,
            arguments=None,
            result=func_response.response or {},
        )
        await self.push_frame(result, FrameDirection.UPSTREAM)
        await self.push_frame(result, FrameDirection.DOWNSTREAM)

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
        """Create an event, persist it to ADK, then run the agent.

        Use this from process_frame overrides to inject system events
        (silence warnings, domain events, etc.) and trigger the agent.

        Args:
            content: The Content for the event (role="user" with system parts).
            state_delta: Optional state changes to persist alongside the content.
        """
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            user_id=self.session_params.user_id,
            session_id=self.session_params.session_id,
        )
        if session is None:
            raise RuntimeError(f"ADK session not found: {self.session_params.session_id}")

        event_kwargs: dict[str, Any] = {
            "invocation_id": Event.new_id(),
            "author": "user",
            "content": content,
        }
        if state_delta:
            event_kwargs["actions"] = EventActions(state_delta=state_delta)

        event = Event(**event_kwargs)
        await self.session_service.append_event(session, event)
        await self._run_adk(event.invocation_id)
