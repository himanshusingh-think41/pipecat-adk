"""
Test pipecat-adk integration using mock services.

This test validates the full pipeline flow using MockLLM, MockSTTService,
MockTTSService, and MockInputTransport/MockOutputTransport with RTVI.
"""

import re
import unittest

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.tools import ToolContext
from google.genai.types import Part
from pipecat_adk import AdkInterruptionPlugin

from tests.mocks import MockLLM, TestRunner, Turn


class TestWithMocks(unittest.IsolatedAsyncioTestCase):
    """Test full pipeline with mock services."""

    async def test_basic_interaction(self):
        """Test basic user-bot interaction with MockLLM."""
        # Create MockLLM that says "Hi, I am a bot"
        mock_llm = MockLLM.single("Hi, I am a bot")

        # Create ADK agent with MockLLM
        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        # Note: app name must be "agents" to match TestRunner's hardcoded session_params
        app = App(
            name="agents",
            root_agent=agent,
            plugins=[AdkInterruptionPlugin()],
        )

        # Create test runner with the app
        async with TestRunner(app=app) as runner:
            # User joins and speaks
            await runner.join()
            await runner.speak_and_wait_for_response("Hi, I am John", timeout=5.0)

            # Verify exact conversation transcript
            self.assertEqual(runner.transcript, [
                Turn("user", "Hi, I am John"),
                Turn("bot", "Hi, I am a bot"),
            ])

    async def test_interruption_handling(self):
        """Test that user can interrupt bot's response."""
        mock_llm = MockLLM.single(
            "Hello! I'm so glad you're interested in learning about our company. "
            "We have a very long history that spans over 50 years, and we've been "
            "pioneers in many different areas..."
        )

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[AdkInterruptionPlugin()],
        )

        async with TestRunner(app=app, tts_delay=0.05) as runner:
            await runner.join()
            await runner.speak("Tell me about your company")
            # interrupt_bot waits for bot-started-speaking, then injects speech.
            await runner.interrupt_bot("Wait, I have a question", timeout=5.0)

            # Wait for the interruption transcription to arrive.
            def _has_interruption_transcription(bot_output, delta_messages):
                return any(
                    msg.get("type") == "user-transcription"
                    and msg.get("data", {}).get("final")
                    and "question" in msg.get("data", {}).get("text", "").lower()
                    for msg in delta_messages
                )
            await runner.wait_for(_has_interruption_transcription, timeout=5.0)

            events = await runner.events()
            self.assertGreater(len(events), 0, "Should have events in session")

            has_user_message = any(
                hasattr(e, "content") and e.content and e.content.role == "user"
                for e in events
            )
            self.assertTrue(has_user_message, "Should have at least one user message in session")

    async def test_function_call_handling(self):
        """Test that function calls are properly handled in the pipeline."""
        # Define a mock tool as a Python function
        def get_weather(location: str) -> dict:
            """Get the current weather for a location."""
            return {"weather": "sunny", "temperature": "72 degrees"}

        # Bot will make a function call, then respond with the result
        mock_llm = MockLLM.from_parts([
            # First turn: function call
            [Part.from_function_call(
                name="get_weather",
                args={"location": "San Francisco"}
            )],
            # Second turn: respond after function execution
            "The weather in San Francisco is sunny and 72 degrees!",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful weather assistant.",
            tools=[get_weather],
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[AdkInterruptionPlugin()],
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response(
                "What's the weather in San Francisco?",
                timeout=5.0,
            )

            # Verify exact bot response
            self.assertEqual(
                runner.last_bot_message,
                "The weather in San Francisco is sunny and 72 degrees!"
            )

            # Verify session contains function call event (gray-box check for tool execution)
            events = await runner.events()
            has_function_call = any(
                e.content.parts and hasattr(e.content.parts[0], 'function_call')
                for e in events
                if hasattr(e, 'content') and e.content and e.content.parts
            )
            self.assertTrue(has_function_call, "Function call should be in session history")

    async def test_multiple_function_calls_in_turn(self):
        """Test handling multiple function calls in a single turn."""
        # Define mock tools as Python functions
        def set_temperature(degrees: float) -> dict:
            """Set the room temperature."""
            return {"status": "success", "temperature": degrees}

        def set_lights(brightness: int) -> dict:
            """Set the room lights."""
            return {"status": "success", "brightness": brightness}

        # Bot makes multiple function calls in one turn
        mock_llm = MockLLM.from_parts([
            # First turn: two function calls
            [
                Part.from_function_call(
                    name="set_temperature",
                    args={"degrees": 72}
                ),
                Part.from_function_call(
                    name="set_lights",
                    args={"brightness": 80}
                ),
            ],
            # Second turn: confirm actions
            "I've set the temperature to 72 degrees and the lights to 80% brightness.",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a smart home assistant.",
            tools=[set_temperature, set_lights],
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[AdkInterruptionPlugin()],
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response(
                "Make the room comfortable",
                timeout=5.0,
            )

            # Verify exact bot response
            self.assertEqual(
                runner.last_bot_message,
                "I've set the temperature to 72 degrees and the lights to 80% brightness."
            )

            # Verify session has both function calls (gray-box check for tool execution)
            events = await runner.events()
            function_calls = []
            for e in events:
                if (hasattr(e, 'content') and e.content and e.content.parts):
                    for part in e.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call.name)

            self.assertIn("set_temperature", function_calls)
            self.assertIn("set_lights", function_calls)

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation flow."""
        # Bot will respond across multiple turns
        mock_llm = MockLLM.conversation([
            "Hi! How can I help you today?",
            "That sounds interesting! Tell me more.",
            "Great, I'd be happy to assist with that.",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[AdkInterruptionPlugin()],
        )

        async with TestRunner(app=app) as runner:
            # First turn
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)

            # Second turn
            await runner.speak_and_wait_for_response("I need help with a project", timeout=5.0)

            # Third turn
            await runner.speak_and_wait_for_response("Can you assist me?", timeout=5.0)

            # Verify full conversation transcript
            self.assertEqual(runner.transcript, [
                Turn("user", "Hello"),
                Turn("bot", "Hi! How can I help you today?"),
                Turn("user", "I need help with a project"),
                Turn("bot", "That sounds interesting! Tell me more."),
                Turn("user", "Can you assist me?"),
                Turn("bot", "Great, I'd be happy to assist with that."),
            ])


class TestCriticalPaths(unittest.IsolatedAsyncioTestCase):
    """Tests that verify the library's core correctness guarantees."""

    def _make_app(self, mock_llm, *, tools=None) -> App:
        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
            tools=tools or [],
        )
        return App(name="agents", root_agent=agent, plugins=[AdkInterruptionPlugin()])

    async def test_rtvi_message_ordering(self):
        """bot-tts-text must arrive between bot-started-speaking and bot-stopped-speaking.

        The transcript builder in BotOutput relies on this ordering to attribute
        text to the correct speaking turn. If bot-tts-text arrives outside the
        started/stopped window, text would be lost or mis-attributed.
        """
        app = self._make_app(MockLLM.single("Hello world"))

        async with TestRunner(app=app) as runner:
            await runner.join()
            # "Hello there" is long enough (>1 PCM sample) for MockVADAnalyzer to
            # detect speech onset (start_secs=0.0001s at 16kHz ≈ 1.6 samples).
            # Single-word speech like "Hi" (1 sample) would fall below the threshold.
            await runner.speak_and_wait_for_response("Hello there", timeout=5.0)

        types = [m.get("type") for m in runner.messages]

        started_idx = types.index("bot-started-speaking")
        stopped_idx = types.index("bot-stopped-speaking")
        tts_text_indices = [i for i, t in enumerate(types) if t == "bot-tts-text"]

        self.assertGreater(len(tts_text_indices), 0, "bot-tts-text message missing")
        for idx in tts_text_indices:
            self.assertGreater(
                idx, started_idx,
                f"bot-tts-text at {idx} must come after bot-started-speaking at {started_idx}",
            )
            self.assertLess(
                idx, stopped_idx,
                f"bot-tts-text at {idx} must come before bot-stopped-speaking at {stopped_idx}",
            )

    async def test_heard_event_contains_text_after_interruption(self):
        """[HEARD] event is written for sentences that fully played before interruption.

        Pipecat appends TTSTextFrame AFTER all audio for each sentence. Only
        sentences whose audio played to completion (and whose TTSTextFrame
        passed through the output transport) populate _context_aggregation and
        thus appear in the [HEARD] event.

        This test uses a two-sentence response and interrupts during sentence 2.
        Sentence 1 ("Done.") plays fully — its TTSTextFrame reaches the aggregator.
        Sentence 2 is still playing when the interruption fires, so [HEARD]
        contains only sentence 1's text.
        """
        # Two-sentence response. TTSService detects the period after "Done." and
        # calls run_tts twice (once per sentence), producing two separate TTSTextFrames.
        app = self._make_app(MockLLM.single(
            "Done. Now this is a much longer second sentence that will still be playing when interrupted."
        ))

        async with TestRunner(app=app, tts_delay=0.05) as runner:
            await runner.join()
            await runner.speak("Tell me something")

            # Wait until the first sentence's TTSTextFrame has been emitted by TTS
            # (RTVI sends bot-tts-text when it observes TTSTextFrame leaving TTS service).
            # By this point the first sentence's audio has fully played and its
            # TTSTextFrame is in the output transport queue, about to reach the aggregator.
            def _first_sentence_emitted(_bot_output, delta):
                return any(m.get("type") == "bot-tts-text" for m in delta)
            await runner.wait_for(_first_sentence_emitted, timeout=5.0)

            # Interrupt during sentence 2. stay_silent lets the event loop deliver
            # TTSTextFrame1 to AdkAssistantContextAggregator before the interruption fires.
            await runner.stay_silent(iterations=5)
            await runner.speak("Stop please")

            def _has_final_transcription(_bot_output, delta):
                return any(
                    m.get("type") == "user-transcription" and m.get("data", {}).get("final")
                    for m in delta
                )
            await runner.wait_for(_has_final_transcription, timeout=5.0)

        events = await runner.events()
        heard_events = [
            e for e in events
            if (
                hasattr(e, "content") and e.content and e.content.parts
                and any("[HEARD]" in (getattr(p, "text", "") or "") for p in e.content.parts)
            )
        ]

        self.assertGreater(len(heard_events), 0, "Expected at least one [HEARD] event in session")

        for event in heard_events:
            for part in event.content.parts:
                text = getattr(part, "text", "") or ""
                if "[HEARD]" not in text:
                    continue
                match = re.search(r'heard: "([^"]*)"', text)
                self.assertIsNotNone(
                    match, f"[HEARD] event text has unexpected format: {text!r}"
                )
                heard_text = match.group(1)
                self.assertGreater(
                    len(heard_text), 0,
                    f"[HEARD] event has empty heard text — was sentence 1 played fully? {text!r}",
                )

    async def test_no_heard_event_on_clean_turn(self):
        """No [HEARD] event is written when the bot finishes speaking without interruption.

        On a clean turn, BotStoppedSpeakingFrame clears AdkAssistantContextAggregator's
        _spoken_text buffer, so no [HEARD] event is written even though TTSTextFrame
        had populated the buffer during playback.
        """
        app = self._make_app(MockLLM.single("I will answer your question completely."))

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("Tell me something", timeout=5.0)

        events = await runner.events()
        heard_events = [
            e for e in events
            if (
                hasattr(e, "content") and e.content and e.content.parts
                and any("[HEARD]" in (getattr(p, "text", "") or "") for p in e.content.parts)
            )
        ]

        self.assertEqual(
            len(heard_events), 0,
            f"Expected no [HEARD] events on uninterrupted turn; found: {heard_events}",
        )

    async def test_session_state_persists_across_turns(self):
        """Session state written by a tool in turn 1 is readable in turn 2.

        ADK's InMemorySessionService persists tool_context.state mutations
        across invocations. This test verifies pipecat-adk doesn't reset
        or re-create the session between user turns.
        """
        def count_visits(tool_context: ToolContext) -> dict:
            """Increment visit_count in session state and return the new value."""
            current = tool_context.state.get("visit_count", 0) + 1
            tool_context.state["visit_count"] = current
            return {"visit_number": current}

        # Each user turn triggers: function-call LLM response, then text LLM response.
        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(name="count_visits", args={})],
            "This is your first visit.",
            [Part.from_function_call(name="count_visits", args={})],
            "This is your second visit.",
        ])

        app = self._make_app(mock_llm, tools=[count_visits])

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)

            state = await runner.session_state()
            self.assertEqual(
                state.get("visit_count"), 1,
                f"visit_count should be 1 after turn 1; state={state}",
            )

            await runner.speak_and_wait_for_response("Hello again", timeout=5.0)

            state = await runner.session_state()
            self.assertEqual(
                state.get("visit_count"), 2,
                f"visit_count should be 2 after turn 2; state={state}",
            )


if __name__ == "__main__":
    unittest.main()
