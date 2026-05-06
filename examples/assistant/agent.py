"""Agent definition for the assistant bot.

This module defines the ADK App with the root agent and plugins.
"""

from google.adk.agents import Agent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pipecat_adk import AdkInterruptionPlugin


SYSTEM_INSTRUCTION = """You are a helpful AI assistant. Be concise and friendly.

Your output will be converted to audio so don't include special characters in your answers.

Respond in 50 words or less. If the user interrupts you, acknowledge it gracefully."""


# Create ADK agent
root_agent = Agent(
    name="helpful_assistant",
    model="gemini-3-flash-preview",
    instruction=SYSTEM_INSTRUCTION,
    # Disable thinking: thinking_budget=0 eliminates the 1-3s reasoning delay
    # that gemini-2.5-flash adds before generating any response tokens.
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    ),
)

# ResumabilityConfig is required by AdkBasedLLMService — it pre-persists user
# events and resumes via invocation_id, which ADK only allows on resumable apps.
app = App(
    name="assistant",
    root_agent=root_agent,
    plugins=[AdkInterruptionPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
)
