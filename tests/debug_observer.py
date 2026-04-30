"""Concise frame observer for pipecat-adk pipeline debugging.

Logs each significant frame exactly once (at the point it leaves its origin
service) at INFO level. Format: FrameName field=value ...  (src -> dst)

Uses source-based filtering (frame_type → (source_class, FrameEndpoint)) so
the same frame is not logged redundantly at every pipeline hop.
"""

from dataclasses import fields, is_dataclass
from typing import Dict, Optional, Set, Tuple, Type

from google.adk.events import Event
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.debug_log_observer import FrameEndpoint
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

from pipecat_adk.frames import AdkContextFrame


# Fields whose values should never appear in logs (binary data, internals).
_EXCLUDE_FIELDS: Set[str] = {"audio", "image", "images", "name", "pts", "metadata"}

# Per-frame allowlist: only these fields are logged for the given frame type.
# Subclasses must come BEFORE their parent class in dict ordering so that the
# first isinstance() match wins.
_FIELD_ALLOWLIST: Dict[Type[Frame], Set[str]] = {
    StartFrame: {"id"},
    UserStartedSpeakingFrame: {"id"},
    UserStoppedSpeakingFrame: {"id"},
    BotStartedSpeakingFrame: {"id"},
    BotStoppedSpeakingFrame: {"id"},
    StartInterruptionFrame: {"id"},
    InterruptionFrame: {"id"},
    LLMFullResponseStartFrame: {"id"},
    LLMFullResponseEndFrame: {"id"},
    TranscriptionFrame: {"id", "text"},
    TTSTextFrame: {"id", "text"},
    FunctionCallsStartedFrame: {"id"},
    FunctionCallInProgressFrame: {"id", "function_name"},
    FunctionCallResultFrame: {"id", "function_name"},
    # ADK-specific frames — include invocation_id for log correlation.
    AdkContextFrame: {"id", "invocation_id"},
    # LLMTextFrame after AdkContextFrame so the dict entry matches first.
    LLMTextFrame: {"id", "text"},
}

# Default set of frame types to observe, each with an optional source filter.
# None means "log regardless of source". (source_type, FrameEndpoint.SOURCE)
# means "only log when the frame exits that source type".
_DEFAULT_FRAME_FILTERS: Dict[Type[Frame], Optional[Tuple[Type, FrameEndpoint]]] = {
    StartFrame: None,
    # VAD / speaking events
    UserStartedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
    UserStoppedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
    BotStartedSpeakingFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
    BotStoppedSpeakingFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
    # Interruptions
    StartInterruptionFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
    InterruptionFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
    # STT
    TranscriptionFrame: (STTService, FrameEndpoint.SOURCE),
    # LLM
    LLMTextFrame: (LLMService, FrameEndpoint.SOURCE),
    LLMFullResponseStartFrame: (LLMService, FrameEndpoint.SOURCE),
    LLMFullResponseEndFrame: (LLMService, FrameEndpoint.SOURCE),
    FunctionCallsStartedFrame: (LLMService, FrameEndpoint.SOURCE),
    FunctionCallInProgressFrame: (LLMService, FrameEndpoint.SOURCE),
    FunctionCallResultFrame: (LLMService, FrameEndpoint.SOURCE),
    # ADK frames
    AdkContextFrame: None,  # log from any source (injected at various points)
    # TTS
    TTSTextFrame: (TTSService, FrameEndpoint.SOURCE),
}


class AdkDebugLogObserver(BaseObserver):
    """Concise INFO-level observer for pipecat-adk pipeline debugging.

    Each frame type is logged exactly once — when it exits the designated
    source processor. ADK Event objects are formatted compactly.

    Usage::

        observers=[RTVIObserver(rtvi), AdkDebugLogObserver()]

    To log additional or different frame types::

        from pipecat.observers.loggers.debug_log_observer import FrameEndpoint
        observers=[AdkDebugLogObserver(frame_filters={
            MyCustomFrame: (MyService, FrameEndpoint.SOURCE),
            SomeOtherFrame: None,  # log from any source
        })]
    """

    def __init__(
        self,
        frame_filters: Optional[Dict[Type[Frame], Optional[Tuple[Type, FrameEndpoint]]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_filters = frame_filters if frame_filters is not None else _DEFAULT_FRAME_FILTERS

    # ── Filtering ────────────────────────────────────────────────────────────

    def _should_log(self, frame: Frame, src, dst) -> bool:
        for frame_type, filter_cfg in self.frame_filters.items():
            if isinstance(frame, frame_type):
                if filter_cfg is None:
                    return True
                service_type, endpoint = filter_cfg
                if endpoint == FrameEndpoint.SOURCE:
                    return isinstance(src, service_type)
                if endpoint == FrameEndpoint.DESTINATION:
                    return isinstance(dst, service_type)
        return False

    # ── Formatting ───────────────────────────────────────────────────────────

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
            return f"'{value[:197]}...'" if len(value) > 200 else f"'{value}'"
        if isinstance(value, (bytes, bytearray)):
            return f"{len(value)}B"
        if isinstance(value, dict):
            return f"{{{value['type']}}}" if "type" in value else f"{{{len(value)} keys}}"
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

    def _extract_details(self, frame: Frame) -> str:
        allowlist: Optional[Set[str]] = None
        for frame_type, allowed in _FIELD_ALLOWLIST.items():
            if isinstance(frame, frame_type):
                allowlist = allowed
                break

        if not is_dataclass(frame):
            return ""

        parts = []
        for field in fields(frame):
            if field.name in _EXCLUDE_FIELDS:
                continue
            if allowlist is not None and field.name not in allowlist:
                continue
            value = getattr(frame, field.name)
            if value is None:
                continue
            parts.append(f"{field.name}={self._format_value(value)}")
        return " ".join(parts)

    # ── Observer hook ────────────────────────────────────────────────────────

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        src = data.source
        dst = data.destination

        if not self._should_log(frame, src, dst):
            return

        name = frame.__class__.__name__
        details = self._extract_details(frame)
        src_name = src.name if src else "?"
        dst_name = dst.name if dst else "?"

        if details:
            logger.info(f"{name} {details}  ({src_name} -> {dst_name})")
        else:
            logger.info(f"{name}  ({src_name} -> {dst_name})")
