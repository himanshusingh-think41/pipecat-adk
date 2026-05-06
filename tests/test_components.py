"""Isolated unit tests for pipecat-adk components.

Tests AdkAssistantContextAggregator in isolation using pipecat's run_test()
utility (source → processor → sink, no transports). Feeds frames directly
and checks ADK session events.

Frame sequence for each test reflects the per-invocation design:
  1. AdkLLMFullResponseStartFrame(invocation_id="inv1")  — creates map entry
  2. TTSTextFrame(text=..., context_id="inv1")            — accumulates heard text
  3. SleepFrame(sleep=0.05)                               — ensures low-priority frames
                                                            are processed before the
                                                            SystemFrame InterruptionFrame
  4. InterruptionFrame (optional)                         — marks was_interrupted
  5. TTSStoppedFrame(context_id="inv1")                   — triggers [HEARD] if interrupted
  6. SleepFrame(sleep=0.05)                               — lets async session write complete

Note: InterruptionFrame is a SystemFrame and is processed with high priority in
pipecat's FrameProcessorQueue. AdkLLMFullResponseStartFrame and TTSTextFrame are
ControlFrame/DataFrame and are low-priority. The SleepFrame before InterruptionFrame
ensures the pipeline flushes the low-priority frames before the interrupt is queued.
"""

import re
import unittest

from google.adk.sessions import InMemorySessionService

from pipecat.frames.frames import (
    InterruptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.base_text_aggregator import AggregationType

from pipecat_adk import AdkAssistantContextAggregator, AdkUserContextAggregator
from pipecat_adk.frames import AdkLLMFullResponseStartFrame
from pipecat_adk.types import SessionParams


_SESSION_PARAMS = SessionParams(
    app_name="agents",
    user_id="test_user",
    session_id="test_session",
)


def _start(invocation_id: str = "inv1") -> AdkLLMFullResponseStartFrame:
    return AdkLLMFullResponseStartFrame(invocation_id=invocation_id)


def _tts_text(text: str, context_id: str = "inv1") -> TTSTextFrame:
    return TTSTextFrame(text=text, context_id=context_id, aggregated_by=AggregationType.SENTENCE)


def _stopped(context_id: str = "inv1") -> TTSStoppedFrame:
    return TTSStoppedFrame(context_id=context_id)


class TestAdkAssistantContextAggregator(unittest.IsolatedAsyncioTestCase):
    """Tests for AdkAssistantContextAggregator's [HEARD] event logic.

    SleepFrame(sleep=0.05) before InterruptionFrame lets the pipeline flush the
    preceding low-priority frames before the high-priority SystemFrame fires.
    SleepFrame after TTSStoppedFrame lets the async session write complete.
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

    def _extract_invocation_id(self, event) -> str:
        for part in event.content.parts:
            text = getattr(part, "text", "") or ""
            match = re.search(r'invocation_id="([^"]*)"', text)
            if match:
                return match.group(1)
        return ""

    async def test_clean_turn_no_heard_event(self):
        """CT-01: Clean turn (no interruption) writes no [HEARD] event."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                _tts_text("Sentence one.", "inv1"),
                _tts_text("Sentence two.", "inv1"),
                _stopped("inv1"),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 0, f"Expected no [HEARD] events on clean turn; got {heard}")

    async def test_interruption_with_no_tts_produces_no_heard_event(self):
        """CT-03: Interrupt before any sentence completes — no [HEARD] event."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                _stopped("inv1"),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 0, f"Expected no [HEARD] events; got {heard}")

    async def test_interruption_after_tts_text_writes_heard_event(self):
        """CT-02: InterruptionFrame after TTSTextFrame → [HEARD] event with spoken text."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                _tts_text("Hello world", "inv1"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                _stopped("inv1"),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 1, f"Expected 1 [HEARD] event; got {heard}")
        self.assertIn("Hello world", self._extract_heard_text(heard[0]))
        self.assertEqual(self._extract_invocation_id(heard[0]), "inv1")

    async def test_clean_turn_followed_by_spurious_interrupt_no_heard(self):
        """CT-08: Late VAD after clean turn — no [HEARD] event.

        TTSStoppedFrame (unit-test proxy for BotStoppedSpeakingFrame) pops the
        invocation entry. A subsequent InterruptionFrame finds _invocations empty
        and writes nothing.
        """
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                _tts_text("Hello world", "inv1"),
                _stopped("inv1"),    # clean turn end — pops entry, map empty
                SleepFrame(sleep=0.05),
                InterruptionFrame(), # late VAD — _invocations is empty, no-op
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
        """CT-04: Multiple TTSTextFrames before interruption all appear in [HEARD]."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                _tts_text("Hello", "inv1"),
                _tts_text(" world", "inv1"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                _stopped("inv1"),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 1, f"Expected 1 [HEARD] event; got {heard}")

        heard_text = self._extract_heard_text(heard[0])
        self.assertIn("Hello", heard_text)
        self.assertIn("world", heard_text)

    async def test_partial_heard_text_on_mid_turn_interruption(self):
        """CT-02: Only fully-played sentences appear in [HEARD].

        Simulates a two-sentence turn: sentence 1 fully plays (TTSTextFrame arrives),
        sentence 2 is still playing when the user interrupts (no TTSTextFrame for it).
        Only sentence 1 appears in [HEARD].
        """
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                _tts_text("Done.", "inv1"),     # sentence 1: fully played
                SleepFrame(sleep=0.05),
                InterruptionFrame(),             # sentence 2 never finishes
                _stopped("inv1"),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 1, f"Expected 1 [HEARD] event; got {heard}")
        self.assertIn("Done", self._extract_heard_text(heard[0]))

    async def test_tts_text_without_known_context_id_is_ignored(self):
        """TTSTextFrame with an unregistered context_id is silently ignored."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                # TTSTextFrame with a different context_id — not in _invocations
                _tts_text("Some text", "unknown-uuid"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                _stopped("inv1"),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        # inv1 was interrupted but had no accumulated text → no [HEARD]
        self.assertEqual(len(heard), 0, f"Expected no [HEARD] for unknown context; got {heard}")

    async def test_two_concurrent_invocations_independent_heard_events(self):
        """CT-10: Concurrent invocations produce independent [HEARD] events."""
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                _start("inv2"),
                _tts_text("From inv1.", "inv1"),
                _tts_text("From inv2.", "inv2"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                _stopped("inv1"),
                _stopped("inv2"),
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 2, f"Expected 2 [HEARD] events; got {heard}")

        inv_ids = {self._extract_invocation_id(e) for e in heard}
        self.assertEqual(inv_ids, {"inv1", "inv2"})

        texts = {self._extract_heard_text(e) for e in heard}
        self.assertIn("From inv1.", texts)
        self.assertIn("From inv2.", texts)

    async def test_rapid_re_interruption(self):
        """CT-09: Two sequential interruptions — [HEARD] for first (has text), none for second.

        First InterruptionFrame fires with "First sentence." accumulated → writes [HEARD].
        Second invocation starts, no text accumulates, second InterruptionFrame writes nothing.
        """
        aggregator, session_service = await self._make_aggregator()

        await run_test(
            aggregator,
            frames_to_send=[
                _start("inv1"),
                _tts_text("First sentence.", "inv1"),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),             # interrupt inv1 — writes [HEARD] with "First sentence."
                SleepFrame(sleep=0.05),
                _start("inv2"),                  # new invocation after interrupt
                SleepFrame(sleep=0.05),
                InterruptionFrame(),             # interrupt inv2 before any text — writes nothing
                SleepFrame(sleep=0.05),
            ],
        )

        heard = await self._heard_events(session_service)
        self.assertEqual(len(heard), 1, f"Expected 1 [HEARD] event; got {heard}")
        self.assertEqual(self._extract_invocation_id(heard[0]), "inv1")
        self.assertIn("First sentence.", self._extract_heard_text(heard[0]))


    async def test_empty_aggregation_writes_no_user_event(self):
        """CT-11: push_aggregation() with no accumulated text is a no-op.

        AdkUserContextAggregator.push_aggregation() must return "" and write
        nothing to the ADK session when the aggregation buffer is empty.
        """
        session_service = InMemorySessionService()
        await session_service.create_session(**_SESSION_PARAMS.model_dump())
        aggregator = AdkUserContextAggregator(
            session_service=session_service,
            session_params=_SESSION_PARAMS,
        )

        result = await aggregator.push_aggregation()

        self.assertEqual(result, "")
        session = await session_service.get_session(**_SESSION_PARAMS.model_dump())
        self.assertEqual(len(session.events), 0, "Expected no events on empty push_aggregation")


if __name__ == "__main__":
    unittest.main()
