"""Isolated unit tests for Vql pipecat-adk aggregators.

Tests VqlAssistantContextAggregator and VqlUserContextAggregator in isolation
using pipecat's run_test() utility (source → processor → sink, no transports).
Feeds frames directly and checks that:
  - VqlAssistantContextAggregator pushes VqlTurnCompletedFrame upstream with
    correct turn_id, text, and interrupted flag.
  - VqlUserContextAggregator handles empty push_aggregation correctly.

Frame sequence for assistant clean-turn tests:
  1. VqlLLMFullResponseStartFrame(turn_id=...)  — initialises parent aggregator state
  2. TTSTextFrame(text=..., context_id=turn_id) — accumulates spoken text
  3. TTSStoppedFrame(context_id=turn_id)        — triggers turn completion
  4. SleepFrame(sleep=0.05)                     — lets async processing complete

Frame sequence for assistant interrupted-turn tests:
  1. VqlLLMFullResponseStartFrame(turn_id=...)  — initialises parent aggregator state
  2. TTSTextFrame(text=..., context_id=turn_id) — accumulates spoken text
  3. SleepFrame(sleep=0.05)                     — flushes low-priority frames before SystemFrame
  4. VqlInterruptionFrame(turn_id=...)          — triggers interrupted turn completion
  5. SleepFrame(sleep=0.05)                     — lets async processing complete

Note: VqlInterruptionFrame extends InterruptionFrame (a SystemFrame) and is
processed at high priority. SleepFrame before it ensures the pipeline flushes
preceding low-priority TTSTextFrame before the interrupt fires.
"""

import unittest

from pipecat.frames.frames import TTSStoppedFrame, TTSTextFrame
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.base_text_aggregator import AggregationType

from pipecat_adk import VqlAssistantContextAggregator, VqlUserContextAggregator
from pipecat_adk.frames import (
    VqlInterruptionFrame,
    VqlLLMFullResponseStartFrame,
    VqlTurnCompletedFrame,
)


def _tts_text(text: str, context_id: str = "tid1") -> TTSTextFrame:
    return TTSTextFrame(text=text, context_id=context_id, aggregated_by=AggregationType.SENTENCE)


def _stopped(context_id: str = "tid1") -> TTSStoppedFrame:
    return TTSStoppedFrame(context_id=context_id)


class TestVqlAssistantContextAggregator(unittest.IsolatedAsyncioTestCase):
    """Tests for VqlAssistantContextAggregator's VqlTurnCompletedFrame routing.

    Verifies that VqlTurnCompletedFrame is pushed upstream on turn completion,
    carrying the correct turn_id, accumulated text, and interrupted flag.
    AdkLLMService is the upstream consumer; these tests verify the aggregator
    half of that contract in isolation.
    """

    async def test_clean_turn_pushes_turn_completed_not_interrupted(self):
        """CT-01: Clean turn pushes VqlTurnCompletedFrame(interrupted=False) upstream."""
        turn_id = "tid-clean-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("Hello world", turn_id),
                _stopped(turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].turn_id, turn_id)
        self.assertFalse(completed[0].interrupted)
        self.assertIn("Hello world", completed[0].text)

    async def test_interrupted_turn_pushes_turn_completed_interrupted(self):
        """CT-02: VqlInterruptionFrame pushes VqlTurnCompletedFrame(interrupted=True) upstream."""
        turn_id = "tid-interrupted-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("Hello world", turn_id),
                SleepFrame(sleep=0.05),
                VqlInterruptionFrame(turn_id=turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].turn_id, turn_id)
        self.assertTrue(completed[0].interrupted)
        self.assertIn("Hello world", completed[0].text)

    async def test_interruption_before_any_tts_produces_no_frame(self):
        """CT-03: Interrupt before any TTS → no VqlTurnCompletedFrame (empty text)."""
        turn_id = "tid-empty-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                SleepFrame(sleep=0.05),
                VqlInterruptionFrame(turn_id=turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 0, "No frame expected when no TTS was spoken")

    async def test_clean_stop_with_no_tts_produces_no_frame(self):
        """CT-04: TTSStoppedFrame with no prior TTS → no VqlTurnCompletedFrame."""
        turn_id = "tid-notts-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _stopped(turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 0, "No frame expected when no TTS was spoken")

    async def test_multiple_tts_frames_accumulated(self):
        """CT-05: Multiple TTSTextFrames before clean stop → all text in VqlTurnCompletedFrame."""
        turn_id = "tid-multi-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("Hello", turn_id),
                _tts_text(" world", turn_id),
                _stopped(turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 1)
        self.assertIn("Hello", completed[0].text)
        self.assertIn("world", completed[0].text)

    async def test_multiple_tts_frames_accumulated_on_interruption(self):
        """CT-06: Multiple TTSTextFrames then interruption → all text in VqlTurnCompletedFrame."""
        turn_id = "tid-multi-int-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("First sentence.", turn_id),
                _tts_text(" Second sentence.", turn_id),
                SleepFrame(sleep=0.05),
                VqlInterruptionFrame(turn_id=turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0].interrupted)
        self.assertIn("First sentence", completed[0].text)
        self.assertIn("Second sentence", completed[0].text)

    async def test_tts_stopped_without_context_id_does_not_trigger_frame(self):
        """CT-07: TTSStoppedFrame with no context_id is passed through without triggering frame."""
        turn_id = "tid-noctx-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("Hello", turn_id),
                TTSStoppedFrame(),  # no context_id
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 0)

    async def test_turn_id_propagates_from_tts_stopped_context_id(self):
        """CT-08: turn_id on TTSStoppedFrame.context_id propagates to VqlTurnCompletedFrame."""
        turn_id = "unique-turn-xyz-789"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("Some text", turn_id),
                _stopped(turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].turn_id, turn_id)

    async def test_turn_id_propagates_from_interruption_frame(self):
        """CT-09: turn_id on VqlInterruptionFrame propagates to VqlTurnCompletedFrame."""
        turn_id = "unique-turn-abc-123"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("Some text", turn_id),
                SleepFrame(sleep=0.05),
                VqlInterruptionFrame(turn_id=turn_id),
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].turn_id, turn_id)

    async def test_late_interruption_after_clean_stop_produces_no_additional_frame(self):
        """CT-10: VqlInterruptionFrame after clean stop (reset) produces no VqlTurnCompletedFrame.

        After TTSStoppedFrame clears the aggregation buffer, a late VqlInterruptionFrame
        finds no text to report and is silently swallowed.
        """
        turn_id = "tid-late-int-1"
        aggregator = VqlAssistantContextAggregator()

        _, upstream = await run_test(
            aggregator,
            frames_to_send=[
                VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id="test_inv"),
                _tts_text("Hello world", turn_id),
                _stopped(turn_id),    # clean turn — clears aggregation buffer
                SleepFrame(sleep=0.05),
                VqlInterruptionFrame(turn_id=turn_id),  # late VAD — no text left
                SleepFrame(sleep=0.05),
            ],
        )

        completed = [f for f in upstream if isinstance(f, VqlTurnCompletedFrame)]
        self.assertEqual(
            len(completed),
            1,
            "Only the clean-turn frame expected; late interrupt produces no additional frame",
        )
        self.assertFalse(completed[0].interrupted)


class TestVqlUserContextAggregator(unittest.IsolatedAsyncioTestCase):
    """Tests for VqlUserContextAggregator's push_aggregation behavior."""

    async def test_empty_push_aggregation_returns_empty_string(self):
        """CT-11: Empty buffer → push_aggregation() returns '' and pushes no frames."""
        aggregator = VqlUserContextAggregator()
        result = await aggregator.push_aggregation()
        self.assertEqual(result, "")

    async def test_empty_push_aggregation_leaves_prev_turn_id_none(self):
        """CT-12: Empty push_aggregation() does not set _prev_turn_id."""
        aggregator = VqlUserContextAggregator()
        await aggregator.push_aggregation()
        self.assertIsNone(aggregator._prev_turn_id)


if __name__ == "__main__":
    unittest.main()
