"""Tests for AdkInterruptionPlugin.

Exercises before_model_callback with LlmRequest objects to verify
[HEARD] marker detection, positional model-event truncation, and edge cases.
"""

import unittest
from unittest.mock import MagicMock

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content, Part

from pipecat_adk.interruption import AdkInterruptionPlugin


def _make_request(contents: list[Content]) -> LlmRequest:
    req = MagicMock(spec=LlmRequest)
    req.contents = contents
    return req


def _user(text: str) -> Content:
    return Content(role="user", parts=[Part(text=text)])


def _model(text: str) -> Content:
    return Content(role="model", parts=[Part(text=text)])


def _heard(text: str) -> Content:
    return Content(
        role="user",
        parts=[Part(text=f'<system>[HEARD] Agent was interrupted. Candidate only heard: "{text}"</system>')],
    )


class TestAdkInterruptionPlugin(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.plugin = AdkInterruptionPlugin()
        self.ctx = MagicMock(spec=CallbackContext)

    async def test_plugin_name(self):
        self.assertEqual(self.plugin.name, "adk_interruption_handler")

    async def test_none_request_returns_none(self):
        result = await self.plugin.before_model_callback(
            callback_context=self.ctx, llm_request=None
        )
        self.assertIsNone(result)

    async def test_request_without_contents_returns_none(self):
        req = MagicMock(spec=LlmRequest)
        del req.contents
        result = await self.plugin.before_model_callback(
            callback_context=self.ctx, llm_request=req
        )
        self.assertIsNone(result)

    async def test_empty_contents_returns_none(self):
        req = _make_request([])
        result = await self.plugin.before_model_callback(
            callback_context=self.ctx, llm_request=req
        )
        self.assertIsNone(result)

    async def test_no_heard_marker_leaves_contents_unchanged(self):
        contents = [_user("Hello"), _model("Hi there"), _user("Thanks")]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)
        self.assertEqual(len(req.contents), 3)
        self.assertEqual(req.contents[1].parts[0].text, "Hi there")

    async def test_heard_marker_truncates_preceding_model_event(self):
        contents = [
            _user("What frameworks do you know?"),
            _model("Have you worked with Java? What frameworks have you used?"),
            _heard("Have you worked with Java?"),
            _user("Yes, I have used Java"),
        ]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)

        # [HEARD] is removed, model event is truncated, user reply remains
        self.assertEqual(len(req.contents), 3)
        self.assertEqual(req.contents[0].parts[0].text, "What frameworks do you know?")
        self.assertEqual(req.contents[1].parts[0].text, "Have you worked with Java?")
        self.assertEqual(req.contents[2].parts[0].text, "Yes, I have used Java")

    async def test_heard_marker_with_empty_heard_text(self):
        contents = [
            _model("Something long"),
            _heard(""),
        ]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)
        # Marker is removed, model event is replaced with empty string
        self.assertEqual(len(req.contents), 1)
        self.assertEqual(req.contents[0].parts[0].text, "")

    async def test_heard_marker_without_preceding_model_event(self):
        # [HEARD] with no model event before it — marker is dropped, history unchanged
        contents = [
            _user("Hello"),
            _heard("Hello"),
        ]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)
        # Marker dropped, user event stays
        self.assertEqual(len(req.contents), 1)
        self.assertEqual(req.contents[0].parts[0].text, "Hello")

    async def test_multiple_heard_markers_each_truncates_preceding_model_event(self):
        contents = [
            _model("First long response about topic A and topic B."),
            _heard("First long response about topic A"),
            _user("Got it"),
            _model("Second long response about topic C and topic D."),
            _heard("Second long response about topic C"),
            _user("Understood"),
        ]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)

        self.assertEqual(len(req.contents), 4)
        self.assertEqual(req.contents[0].parts[0].text, "First long response about topic A")
        self.assertEqual(req.contents[1].parts[0].text, "Got it")
        self.assertEqual(req.contents[2].parts[0].text, "Second long response about topic C")
        self.assertEqual(req.contents[3].parts[0].text, "Understood")

    async def test_heard_text_with_quotes_and_special_chars(self):
        heard = "Have you used Java? It's great!"
        contents = [
            _model(f"{heard} What about Python?"),
            _heard(heard),
        ]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)
        self.assertEqual(len(req.contents), 1)
        self.assertEqual(req.contents[0].parts[0].text, heard)

    async def test_heard_text_with_newlines(self):
        heard = "Line one.\nLine two."
        contents = [
            _model(f"{heard}\nLine three."),
            _heard(heard),
        ]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)
        self.assertEqual(len(req.contents), 1)
        self.assertEqual(req.contents[0].parts[0].text, heard)

    async def test_returns_none_always(self):
        contents = [_model("Something"), _heard("Something")]
        req = _make_request(contents)
        result = await self.plugin.before_model_callback(
            callback_context=self.ctx, llm_request=req
        )
        self.assertIsNone(result)

    async def test_model_event_with_multiple_parts_replaced_by_single_heard_part(self):
        contents = [
            Content(role="model", parts=[Part(text="Part A. "), Part(text="Part B.")]),
            _heard("Part A."),
        ]
        req = _make_request(contents)
        await self.plugin.before_model_callback(callback_context=self.ctx, llm_request=req)
        self.assertEqual(len(req.contents), 1)
        self.assertEqual(len(req.contents[0].parts), 1)
        self.assertEqual(req.contents[0].parts[0].text, "Part A.")


if __name__ == "__main__":
    unittest.main()
