"""Agent definition for the assistant bot.

This module defines the ADK App with the root agent and plugins.
"""

from google.adk.agents import Agent
from google.adk.apps.app import App, ResumabilityConfig
from pipecat_adk import AdkInterruptionPlugin


SYSTEM_INSTRUCTION = """You are a helpful AI assistant. Be concise and friendly.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.

If the user interrupts you, acknowledge it gracefully."""


# Create ADK agent
root_agent = Agent(
    name="helpful_assistant",
    model="gemini-2.5-flash",
    instruction=SYSTEM_INSTRUCTION,
)

# ResumabilityConfig is required by AdkBasedLLMService — it pre-persists user
# events and resumes via invocation_id, which ADK only allows on resumable apps.
app = App(
    name="assistant",
    root_agent=root_agent,
    plugins=[AdkInterruptionPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
)
