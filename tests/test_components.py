"""Isolated unit tests for pipecat-adk components.

Tests AdkAssistantContextAggregator in isolation using pipecat's run_test()
utility (source → processor → sink, no transports). Feeds frames directly
and checks ADK session events.
"""

import re
import unittest

from google.adk.sessions import InMemorySessionService

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSTextFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.base_text_aggregator import AggregationType

from pipecat_adk import AdkAssistantContextAggregator
from pipecat_adk.types import SessionParams


_SESSION_PARAMS = SessionParams(
    app_name="agents",
    user_id="test_user",
    session_id="test_session",
)


def _tts_text(text: str) -> TTSTextFrame:
    return TTSTextFrame(text=text, aggregated_by=AggregationType.SENTENCE)


class TestAdkAssistantContextAggregator(unittest.IsolatedAsyncioTestCase):
    """Tests for AdkAssistantContextAggregator's [HEARD] event logic.

    Feeds TTSTextFrame / BotStoppedSpeakingFrame / InterruptionFrame directly
    into the aggregator and checks the ADK session for [HEARD] events.

    SleepFrame(sleep=0.05) between frames lets the event loop flush the async
    ADK session write triggered by InterruptionFrame before we read results.
    """

    async def _make_aggregator(
        self,
    ) -> tuple[AdkAssistantContextAggregator, InMemorySessionService]:
        session_service = InMemorySessionService()
        await session_service.create_session(**_SESSION_PARAMS.model_dump())
        aggregator = AdkAssistantContextAggregator(
            session_service=session_service,
            session_params=_SESSION_PARAMS,
        )
        return aggregator, session_service

    async def _heard_events(self, session_service: InMemorySessionService) -> list:
        session = await session_service.get_session(**_SESSION_PARAMS.model_dump())
        return [
            e
            for e in (session.events if session else [])
            if (
                hasattr(e, "content")
                and e.content
                and e.content.parts
                and any("[HEARD]" in (getattr(p, "text", "") or "") for p in e.content.parts)
            )
        ]

    def _extract_heard_text(self, event) -> str:
        for part in event.content.parts:
            text = getattr(part, "text", "") or ""
            match = re.search(r'heard: "([^"]*)"', text)
            if match:
                return match.group(1)
        return ""

    async def test_interruption_with_no_tts_produces_no_heard_event(self):
        """InterruptionFrame with empty buffer writes no [HEARD] event."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                InterruptionFrame(),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 0, f"Expected no [HEARD] events; got {heard}")

    async def test_interruption_after_tts_text_writes_heard_event(self):
        """InterruptionFrame after TTSTextFrame writes a [HEARD] event with the spoken text."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _tts_text("Hello world"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 1, f"Expected 1 [HEARD] event; got {heard}")
        self.assertIn("Hello world", self._extract_heard_text(heard[0]))

    async def test_bot_stopped_speaking_clears_buffer_no_heard_event(self):
        """BotStoppedSpeakingFrame on a clean turn clears the buffer; no [HEARD] on interruption.

        Simulates a complete bot turn followed by an interruption (e.g. user speaks
        immediately after the bot finishes). The [HEARD] buffer was cleared when the
        bot stopped speaking cleanly, so no [HEARD] is written.
        """
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _tts_text("Hello world"),
                SleepFrame(sleep=0.05),
                BotStoppedSpeakingFrame(),  # clean turn end — clears buffer
                SleepFrame(sleep=0.05),
                InterruptionFrame(),        # next turn: buffer is empty
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(
            len(heard),
            0,
            f"Expected no [HEARD] after clean turn; got {heard}",
        )

    async def test_multiple_tts_frames_accumulated_in_heard_event(self):
        """Multiple TTSTextFrames before interruption are all joined in [HEARD]."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _tts_text("Hello"),
                _tts_text(" world"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 1, f"Expected 1 [HEARD] event; got {heard}")

        heard_text = self._extract_heard_text(heard[0])
        self.assertIn("Hello", heard_text)
        self.assertIn("world", heard_text)

    async def test_partial_heard_text_on_mid_turn_interruption(self):
        """Text from completed sentences appears in [HEARD]; unheard sentences do not.

        Simulates a two-sentence bot turn: sentence 1 finishes (its TTSTextFrame
        arrives), sentence 2 is still playing when the user interrupts. Only
        sentence 1's text should appear in [HEARD].
        """
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _tts_text("Done."),          # sentence 1: fully played
                SleepFrame(sleep=0.05),
                InterruptionFrame(),         # sentence 2 never finishes → no TTSTextFrame for it
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 1, f"Expected 1 [HEARD] event; got {heard}")
        self.assertIn("Done", self._extract_heard_text(heard[0]))


if __name__ == "__main__":
    unittest.main()
