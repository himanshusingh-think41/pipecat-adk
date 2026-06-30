"""Concise debug observer for voice-agent pipelines."""

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
    TTSTextFrame,
    TranscriptionFrame,
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

from pipecat_adk.frames import VqlContextFrame, VqlLLMFullResponseStartFrame, VqlLLMTextFrame


class FrameEndpoint(Enum):
    SOURCE = auto()
    DESTINATION = auto()


class AdkDebugObserver(BaseObserver):
    EXCLUDE_FIELDS: Set[str] = {"audio", "image", "images", "name", "pts", "metadata"}

    FRAME_FIELD_ALLOWLIST: Dict[Type[Frame], Set[str]] = {
        UserStartedSpeakingFrame: {"id"},
        UserStoppedSpeakingFrame: {"id"},
        BotStartedSpeakingFrame: {"id"},
        BotStoppedSpeakingFrame: {"id"},
        InterruptionFrame: {"id"},
        LLMFullResponseEndFrame: {"id"},
        TranscriptionFrame: {"id", "text"},
        TTSTextFrame: {"id", "text"},
        VqlLLMFullResponseStartFrame: {"id", "turn_id"},
        LLMFullResponseStartFrame: {"id"},
        VqlContextFrame: {"id", "turn_id", "text"},
        VqlLLMTextFrame: {"id", "text", "turn_id"},
        LLMTextFrame: {"id", "text"},
        InputTransportMessageFrame: {"id", "message"},
    }

    FRAME_FILTERS: Dict[Type[Frame], Optional[Tuple[Type, FrameEndpoint]]] = {
        UserStartedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
        UserStoppedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
        BotStartedSpeakingFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
        BotStoppedSpeakingFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
        TranscriptionFrame: (STTService, FrameEndpoint.SOURCE),
        InterruptionFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
        VqlLLMTextFrame: (LLMService, FrameEndpoint.SOURCE),
        VqlLLMFullResponseStartFrame: (LLMService, FrameEndpoint.SOURCE),
        LLMTextFrame: (LLMService, FrameEndpoint.SOURCE),
        LLMFullResponseStartFrame: (LLMService, FrameEndpoint.SOURCE),
        LLMFullResponseEndFrame: (LLMService, FrameEndpoint.SOURCE),
        VqlContextFrame: None,
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
            for part in event.content.parts:
                if getattr(part, "text", None):
                    texts.append(part.text)
                if getattr(part, "function_call", None):
                    fn_calls.append(part.function_call.name)
                if getattr(part, "function_response", None):
                    fn_resps.append(part.function_response.name)
            if texts:
                text = "".join(texts)
                parts.append(f"'{text[:57]}...'" if len(text) > 60 else f"'{text}'")
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
            for field in fields(frame):
                if field.name in self.EXCLUDE_FIELDS:
                    continue
                if allowlist is not None and field.name not in allowlist:
                    continue
                value = getattr(frame, field.name)
                if value is None:
                    continue
                parts.append(f"{field.name}={self._format_value(value)}")
        return " ".join(parts)

    async def on_push_frame(self, data: FramePushed) -> None:
        if not self._should_log(data.frame, data.source, data.destination):
            return
        details = self._frame_details(data.frame)
        if details:
            logger.info(f"{data.frame.__class__.__name__} {details}")
        else:
            logger.info(data.frame.__class__.__name__)
