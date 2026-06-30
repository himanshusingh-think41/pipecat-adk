from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.planners import BuiltInPlanner
from google.genai import types

from pipecat_adk import AdkInterruptionPlugin


SYSTEM_INSTRUCTION = """You are a helpful voice AI assistant.

Keep answers concise and natural for speech. Avoid markdown, bullet symbols, and
special characters that sound awkward when read aloud.

If the user interrupts you, acknowledge it naturally and continue from the new question."""


def build_app() -> App:
    root_agent = Agent(
        name="voice_agent",
        model="gemini-2.5-flash",
        instruction=SYSTEM_INSTRUCTION,
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    return App(
        name="voice_agent",
        root_agent=root_agent,
        plugins=[AdkInterruptionPlugin()],
    )
