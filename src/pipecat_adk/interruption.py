"""ADK plugin for deterministic interruption handling.

Replaces the old fuzzy-match InterruptionHandlerPlugin with an exact
positional approach: find [HEARD] marker events, locate the immediately
preceding model event, replace its text with the heard text.

No fuzzy matching. No difflib. The heard text is sourced directly from
the TTSTextFrame frames that actually reached the assistant aggregator,
so the correction is always deterministic.
"""

import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai.types import Content, Part
from loguru import logger

_HEARD_PATTERN = re.compile(
    r'\[HEARD\]\s*Agent was interrupted\.\s*Candidate only heard:\s*"(.*?)"',
    re.DOTALL,
)


class AdkInterruptionPlugin(BasePlugin):
    """Before-model plugin that truncates interrupted agent responses.

    When AdkAssistantContextAggregator detects an interruption it writes
    a [HEARD] event to the ADK session. This plugin finds those markers
    in the LLM request, locates the preceding model event, and replaces
    its full text with only the heard portion — giving the LLM an accurate
    view of what the candidate actually received.

    Example session history before this plugin runs::

        model:  "Have you worked with Java? What frameworks have you used?"
        user:   '<system>[HEARD] Agent was interrupted. Candidate only heard: "Have you worked with Java?"</system>'
        user:   "<candidate>Yes, I have used Java...</candidate>"

    After this plugin runs, the model event becomes::

        model:  "Have you worked with Java?"

    And the [HEARD] event is removed entirely from the request.
    """

    def __init__(self) -> None:
        super().__init__(name="adk_interruption_handler")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        if not llm_request or not hasattr(llm_request, "contents"):
            return None

        contents = llm_request.contents
        if not contents:
            return None

        modified = self._process_heard_markers(contents)
        if modified:
            logger.debug("AdkInterruptionPlugin: truncated interrupted model response(s)")

        return None

    def _process_heard_markers(self, contents: list[Content]) -> bool:
        modified = False
        new_contents: list[Content] = []

        for content in contents:
            heard_text = self._extract_heard_text(content)
            if heard_text is not None:
                # Find the most recent model event in new_contents
                model_idx = self._find_previous_model_event(new_contents)
                if model_idx is not None:
                    # Replace model event text with only what was heard
                    new_contents[model_idx] = Content(
                        role=new_contents[model_idx].role,
                        parts=[Part(text=heard_text)],
                    )
                    logger.debug(
                        f"Truncated model response to heard text: '{heard_text[:80]}'"
                    )
                else:
                    logger.warning(
                        "AdkInterruptionPlugin: [HEARD] marker found but no preceding "
                        "model event in request — leaving history unchanged."
                    )
                # Either way, drop the [HEARD] event from the request
                modified = True
            else:
                new_contents.append(content)

        contents[:] = new_contents
        return modified

    def _extract_heard_text(self, content: Content) -> Optional[str]:
        """Return the heard text from a [HEARD] event, or None if not a marker."""
        if not content or not content.parts:
            return None
        full_text = "".join(p.text for p in content.parts if p.text)
        match = _HEARD_PATTERN.search(full_text)
        return match.group(1) if match else None

    def _find_previous_model_event(self, contents: list[Content]) -> Optional[int]:
        """Return the index of the last model/assistant event, or None."""
        for i in range(len(contents) - 1, -1, -1):
            if contents[i].role in ("model", "assistant"):
                return i
        return None
