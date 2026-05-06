"""Concise debug observer for pipecat-adk pipelines.

Logs frame activity at INFO level. Add to PipelineTask observers=[AdkDebugObserver()].
"""

from dataclasses import fields, is_dataclass
from enum import Enum, auto
from typing import Dict, Optional, Set, Tuple, Type

from google.adk.events.event import Event
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndTaskFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

from pipecat_adk.frames import AdkContextFrame, AdkLLMFullResponseStartFrame, AdkLLMTextFrame


class FrameEndpoint(Enum):
    SOURCE = auto()
    DESTINATION = auto()


class AdkDebugObserver(BaseObserver):
    """Concise frame observer for pipecat-adk pipeline debugging.

    Logs frame activity at INFO level with minimal verbosity.
    Format: FrameName field=value field2=value2
    """

    EXCLUDE_FIELDS: Set[str] = {"audio", "image", "images", "name", "pts", "metadata"}

    # Allowlist of fields per frame type. Empty set = log no fields (just the name).
    # Order matters: more specific subclasses MUST come before parent classes.
    FRAME_FIELD_ALLOWLIST: Dict[Type[Frame], Set[str]] = {
        UserStartedSpeakingFrame: {"id"},
        UserStoppedSpeakingFrame: {"id"},
        BotStartedSpeakingFrame: {"id"},
        BotStoppedSpeakingFrame: {"id"},
        InterruptionFrame: {"id"},
        LLMFullResponseEndFrame: {"id"},
        TranscriptionFrame: {"id", "text"},
        TTSTextFrame: {"id", "text"},
        # ADK frames — AdkLLMFullResponseStartFrame before LLMFullResponseStartFrame
        AdkLLMFullResponseStartFrame: {"id", "invocation_id"},
        LLMFullResponseStartFrame: {"id"},
        AdkContextFrame: {"id", "invocation_id"},
        # AdkLLMTextFrame before LLMTextFrame (subclass before parent)
        AdkLLMTextFrame: {"id", "text", "invocation_id"},
        LLMTextFrame: {"id", "text"},
        InputTransportMessageFrame: {"id", "message"},
    }

    # Maps frame type → optional (service_type, FrameEndpoint) filter.
    # None filter = log from any source.
    FRAME_FILTERS: Dict[Type[Frame], Optional[Tuple[Type, FrameEndpoint]]] = {
        UserStartedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
        UserStoppedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
        BotStartedSpeakingFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
        BotStoppedSpeakingFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
        TranscriptionFrame: (STTService, FrameEndpoint.SOURCE),
        InterruptionFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
        # AdkLLMTextFrame before LLMTextFrame so isinstance() hits the subclass first
        AdkLLMTextFrame: (LLMService, FrameEndpoint.SOURCE),
        AdkLLMFullResponseStartFrame: (LLMService, FrameEndpoint.SOURCE),
        LLMTextFrame: (LLMService, FrameEndpoint.SOURCE),
        LLMFullResponseStartFrame: (LLMService, FrameEndpoint.SOURCE),
        LLMFullResponseEndFrame: (LLMService, FrameEndpoint.SOURCE),
        AdkContextFrame: None,
        FunctionCallsStartedFrame: (LLMService, FrameEndpoint.SOURCE),
        FunctionCallInProgressFrame: (LLMService, FrameEndpoint.SOURCE),
        FunctionCallResultFrame: (LLMService, FrameEndpoint.SOURCE),
        InputTransportMessageFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
        RTVIServerMessageFrame: (LLMService, FrameEndpoint.SOURCE),
        TTSTextFrame: (TTSService, FrameEndpoint.SOURCE),
        EndTaskFrame: (LLMService, FrameEndpoint.SOURCE),
    }

    def _format_value(self, value) -> str:
        if value is None:
            return "None"
        if isinstance(value, Event):
            return self._format_event(value)
        if isinstance(value, list):
            if not value:
                return "[]"
            if isinstance(value[0], Event):
                return f"[{', '.join(self._format_event(e) for e in value)}]"
            if len(value) > 3:
                return f"[{len(value)} items]"
            return str(value)
        if isinstance(value, str):
            if len(value) > 200:
                return f"'{value[:197]}...'"
            return f"'{value}'"
        if isinstance(value, (bytes, bytearray)):
            return f"{len(value)}B"
        if isinstance(value, dict):
            if "type" in value:
                return f"{{{value['type']}}}"
            return f"{{{len(value)} keys}}"
        result = str(value)
        return result[:197] + "..." if len(result) > 200 else result

    def _format_event(self, event: Event) -> str:
        parts = []
        if event.partial:
            parts.append("partial")
        if event.content and event.content.parts:
            texts, fn_calls, fn_resps = [], [], []
            for p in event.content.parts:
                if getattr(p, "text", None):
                    texts.append(p.text)
                if getattr(p, "function_call", None):
                    fn_calls.append(p.function_call.name)
                if getattr(p, "function_response", None):
                    fn_resps.append(p.function_response.name)
            if texts:
                t = "".join(texts)
                parts.append(f"'{t[:57]}...'" if len(t) > 60 else f"'{t}'")
            if fn_calls:
                parts.append(f"fn={fn_calls}")
            if fn_resps:
                parts.append(f"resp={fn_resps}")
        return f"Event({', '.join(parts)})" if parts else "Event()"

    def _should_log(self, frame: Frame, src, dst) -> bool:
        for frame_type, filter_cfg in self.FRAME_FILTERS.items():
            if isinstance(frame, frame_type):
                if filter_cfg is None:
                    return True
                service_type, endpoint = filter_cfg
                if endpoint == FrameEndpoint.SOURCE:
                    return isinstance(src, service_type)
                return isinstance(dst, service_type)
        return False

    def _frame_details(self, frame: Frame) -> str:
        allowlist = None
        for frame_type, allowed in self.FRAME_FIELD_ALLOWLIST.items():
            if isinstance(frame, frame_type):
                allowlist = allowed
                break

        parts = []
        if is_dataclass(frame):
            for f in fields(frame):
                if f.name in self.EXCLUDE_FIELDS:
                    continue
                if allowlist is not None and f.name not in allowlist:
                    continue
                val = getattr(frame, f.name)
                if val is None:
                    continue
                parts.append(f"{f.name}={self._format_value(val)}")
        return " ".join(parts)

    async def on_push_frame(self, data: FramePushed) -> None:
        if not self._should_log(data.frame, data.source, data.destination):
            return
        details = self._frame_details(data.frame)
        if details:
            logger.info(f"{data.frame.__class__.__name__} {details}")
        else:
            logger.info(data.frame.__class__.__name__)
