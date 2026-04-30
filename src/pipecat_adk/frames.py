"""Frame definitions for the ADK-Pipecat bridge."""

from dataclasses import dataclass

from pipecat.frames.frames import SystemFrame


@dataclass
class AdkContextFrame(SystemFrame):
    """Carries invocation_id from user aggregator to LLM service.

    Pushed by AdkUserContextAggregator after persisting the user event.
    AdkBasedLLMService handles it by calling runner.run_async(invocation_id=...).
    """

    invocation_id: str
