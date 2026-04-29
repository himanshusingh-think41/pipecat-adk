# pipecat-adk

**Build powerful voice-enabled AI agents by combining Pipecat's real-time audio pipelines with Google ADK's agent framework.**

## The Problem

**Pipecat** excels at real-time voice applications. It handles audio streaming, VAD, STT, TTS, and transport protocols beautifully. But as your application grows more complex—managing conversation history, handling interruptions correctly, persisting sessions, calling tools—things get difficult. Pipecat's context management wasn't designed for sophisticated agent workflows.

**Google ADK** (Agent Development Kit) excels at building agents. It provides rich concepts for sessions, state management, tool definitions, multi-agent orchestration, evaluations, and much more. But ADK wasn't designed for real-time voice—it expects request/response patterns, not streaming audio.

**pipecat-adk** bridges these two worlds, letting you build voice applications with Pipecat's real-time capabilities while leveraging ADK's agent framework for everything else.

## Installation

```bash
pip install pipecat-adk

# Or install from source
pip install -e /path/to/pipecat-adk
```

## Getting Started

If you have an existing Pipecat application, here's what you need to change:

### Before (Standard Pipecat)

```python
from pipecat.services.google import GoogleLLMService
from pipecat.services.google.llm import GoogleLLMContext
from pipecat.pipeline.pipeline import Pipeline

llm = GoogleLLMService(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

context_aggregator = llm.create_context_aggregator(
    GoogleLLMContext(messages=[{"role": "system", "content": "You are helpful"}])
)

pipeline = Pipeline([
    transport.input(),
    stt_service,
    context_aggregator.user(),
    llm,
    tts_service,
    transport.output(),
    context_aggregator.assistant(),
])
```

### After (With pipecat-adk)

```python
from pipecat_adk import AdkBasedLLMService, AdkInterruptionPlugin, SessionParams, make_adk_aware_tts
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.pipeline.pipeline import Pipeline

# 1. Define your ADK agent
agent = Agent(
    name="helpful_assistant",  # Note: use underscores, not hyphens
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

# 2. Set up session management
session_service = InMemorySessionService()
session_params = SessionParams(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
)
await session_service.create_session(**session_params.model_dump())

# 3. Create the LLM service (manages the ADK App internally)
llm = AdkBasedLLMService(
    agent=agent,
    session_service=session_service,
    session_params=session_params,
    plugins=[AdkInterruptionPlugin()],
)

# 4. Wrap TTS for accurate interruption tracking
AdkAwareTTS = make_adk_aware_tts(GoogleTTSService)
tts = AdkAwareTTS(voice_id=...)

# 5. Create context aggregators — pipeline structure stays the same
context_aggregator = llm.create_context_aggregator()

pipeline = Pipeline([
    transport.input(),
    stt_service,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

The pipeline structure stays the same—you swap the LLM service, context aggregator, and TTS wrapper.

## Key Challenges Solved

### 1. Context Management

**The Problem**: Pipecat manages conversation history by accumulating messages in an `LLMContext`. This works for simple cases, but breaks down when you need sophisticated history management, multi-turn reasoning, or agent handoffs.

**Our Solution**: ADK manages the conversation history. When a user speaks, `AdkUserContextAggregator`:
1. Persists the user event directly to ADK's session store
2. Clears Pipecat's context (since ADK owns the history)
3. Pushes `AdkContextFrame(invocation_id=...)` to trigger the LLM service

The user's transcription is sent directly to ADK without modification.

**Tradeoff**: You can't use Pipecat's context inspection tools. All history lives in ADK sessions, which you access via `session_service.get_session()`.

### 2. Persistence and Replayability

**The Problem**: Pipecat's context is ephemeral—restart the server and you lose everything. Building features like conversation replay, analytics, or multi-device continuity requires custom persistence logic.

**Our Solution**: Use any ADK session service. ADK provides:
- `InMemorySessionService` for development
- `DatabaseSessionService` for production persistence
- Custom implementations for your specific needs

Every event is persisted automatically. You get full conversation history across restarts, audit trails for compliance, and session handoff between agents.

**Tradeoff**: You need to manage session IDs and ensure they're unique per conversation. You also need to handle session cleanup and expiration.

### 3. Interruption Handling

**The Problem**: When a user interrupts the AI mid-sentence, the LLM on the next turn sees the full planned response as if the user heard everything—leading to confusing conversations.

**Our Solution**: A deterministic, two-part mechanism:

1. When interrupted, `AdkAssistantContextAggregator` knows exactly what TTS text was spoken (it tracks text per audio context via `TTSTextFrame`). It writes a `[HEARD]` event to the ADK session containing only what was actually spoken.
2. Before the next LLM call, `AdkInterruptionPlugin` finds the `[HEARD]` marker in the request, locates the preceding model event, and replaces its full text with the heard portion. The marker is then removed from the request.

The LLM sees only what the user actually heard. The full response remains in ADK session history for auditing.

**Tradeoff**: The session history contains `[HEARD]` marker events. If you analyze raw session data you'll need to filter these. The `make_adk_aware_tts` factory must be used to wrap your TTS service—this signals when audio contexts complete cleanly (no `[HEARD]` needed), keeping the tracking buffer accurate.

### 4. State Management

**The Problem**: ADK tools and events often produce state changes that clients need to know about. Coordinating this state between the AI and your application is tedious.

**Our Solution**: Override `_on_state_delta()` in a subclass of `AdkBasedLLMService`. The bridge calls this for every ADK event that carries a `state_delta`, before any text frames from the same event—so the client receives state before the bot starts speaking.

```python
class MyLLMService(AdkBasedLLMService):
    async def _on_state_delta(self, state_delta: dict) -> None:
        # Forward state to your client via RTVI, WebSocket, etc.
        await self.push_frame(RTVIServerMessageFrame(
            data={"type": "state-sync", "state_delta": state_delta}
        ))
```

To inject events programmatically (e.g., a timeout or a form submission), override `process_frame` and call `_persist_and_run`:

```python
class MyLLMService(AdkBasedLLMService):
    async def process_frame(self, frame, direction):
        if isinstance(frame, UserIdleFrame):
            await self._persist_and_run(
                content=Content(role="user", parts=[
                    Part(text="<system>User has been idle for 30s.</system>")
                ])
            )
        else:
            await super().process_frame(frame, direction)
```

**Tradeoff**: State integration requires subclassing `AdkBasedLLMService`. This keeps the bridge lean but means simple state forwarding can't be done with configuration alone.

### 5. Function Call Lifecycle

**The Problem**: When an AI calls a tool, you often want to mute the microphone or show a loading indicator. Standard integrations don't always emit the right frames at the right time.

**Our Solution**: When ADK executes a function call, the bridge pushes frames both upstream and downstream:

1. `FunctionCallsStartedFrame` → enables `STTMuteFilter` to mute the mic
2. `FunctionCallInProgressFrame` → lets your UI show "thinking..."
3. ADK executes the function
4. `FunctionCallResultFrame` → lets your UI show results, unmutes mic

**Tradeoff**: Function calls are managed entirely by ADK. You define tools using ADK's `FunctionTool` or as plain Python functions, not Pipecat's function calling mechanism. The frames inform Pipecat of the lifecycle but don't let you intercept or modify the calls.

### 6. Custom Context Injection

**The Problem**: You need to inject dynamic context into conversations—current time, user preferences, system state, etc.

**Our Solution**: Override `_build_user_event()` in a subclass of `AdkUserContextAggregator`:

```python
from pipecat_adk.aggregators import AdkUserContextAggregator
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part

class MyUserAggregator(AdkUserContextAggregator):
    async def _build_user_event(self, text: str, session) -> Event:
        parts = [
            Part(text=f"<system>Current time: {datetime.now()}</system>"),
            Part(text=text),
        ]
        return Event(
            invocation_id=Event.new_id(),
            author="user",
            content=Content(role="user", parts=parts),
        )
```

Then pass your aggregator to `create_context_aggregator` — or build the `AdkContextAggregatorPair` manually and use it in the pipeline.

**Tradeoff**: This runs on every user message. Keep it lightweight—avoid slow database queries or API calls here, or at least await them with care.

## Pipecat Ecosystem Limitations

pipecat-adk takes a fundamentally different approach to context management: **ADK owns the conversation history**, not Pipecat. This architectural decision enables ADK's powerful session management, but means some Pipecat ecosystem components won't work as expected.

### Why This Matters

Standard Pipecat components expect to read/write messages via `OpenAILLMContext`. Since ADK manages conversation history in its own session store, our context frames only carry an `invocation_id` reference—not the actual messages.

### Incompatible Components

| Component | What It Does | Why It's Incompatible |
|-----------|--------------|----------------------|
| **Mem0 Memory Service** | Enhances context with retrieved memories | Expects to add messages to context before LLM |
| **LangChain Framework** | Routes to LangChain agents | Alternative agent framework—use ADK or LangChain, not both |
| **Strands Framework** | Routes to Strands agents | Alternative agent framework—use ADK or Strands, not both |
| **IVR Navigator** | Stores messages for IVR mode switching | Expects to read/store messages from context |
| **LLM Log Observer** | Logs conversation messages | Will show empty context (use ADK session inspection instead) |

### Compatible Components

These Pipecat components work normally with pipecat-adk:
- **STT services** (Google, Deepgram, etc.)
- **TTS services** (Google, ElevenLabs, Cartesia, etc.) — wrap with `make_adk_aware_tts`
- **VAD analyzers** (Silero, WebRTC)
- **Transports** (WebRTC, WebSocket)
- **STTMuteFilter** (receives function call lifecycle frames)
- **UserIdleProcessor** (receives lifecycle frames)

## Complete Example

See [`examples/assistant/`](examples/assistant/) for a complete working application:

- **`agent.py`**: Defines the ADK Agent and includes `AdkInterruptionPlugin`
- **`bot.py`**: Sets up the Pipecat pipeline with `AdkBasedLLMService`
- **`run.py`**: FastAPI server for WebRTC signaling

To run:

```bash
cd examples/assistant
pip install -r requirements.txt
pip install -e ../..  # Install pipecat-adk in development mode
export GEMINI_API_KEY=your_key
python run.py
```

Open http://localhost:7860 to interact with the voice assistant.

## Testing Your Application

pipecat-adk ships with a complete mock testing infrastructure so you can test your agents without real API calls. The test utilities live in `tests/mocks.py` and `tests/test_utils.py`—copy them to your project or add `tests/` to your path.

### Basic Test Structure

```python
import unittest
from google.adk.agents import Agent
from google.adk.apps import App
from pipecat_adk import AdkInterruptionPlugin
from mocks import MockLLM, TestRunner, Turn

class TestMyAgent(unittest.IsolatedAsyncioTestCase):
    async def test_greeting(self):
        mock_llm = MockLLM.single("Hello! How can I help?")

        agent = Agent(name="test_agent", model=mock_llm,
                      instruction="You are helpful.")
        app = App(name="agents", root_agent=agent,
                  plugins=[AdkInterruptionPlugin()])

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("Hi")

            assert runner.last_bot_message == "Hello! How can I help?"
```

The `app.name` must be `"agents"` — `TestRunner` uses a hardcoded session scoped to that name.

### MockLLM

```python
# Single response
mock_llm = MockLLM.single("Hello!")

# Multi-turn — one response per user message
mock_llm = MockLLM.conversation(["Hello!", "Sure, I can help.", "You're welcome."])

# Function calls — list of Parts per turn, or a string for text-only turns
from google.genai.types import Part
mock_llm = MockLLM.from_parts([
    [Part.from_function_call(name="get_weather", args={"city": "NYC"})],
    "The weather in NYC is sunny.",
])
```

The number of responses must match the number of conversation turns, or later turns will time out.

### Conversational API

```python
async with TestRunner(app=app) as runner:
    await runner.join()

    # Speak and wait for bot reply
    await runner.speak_and_wait_for_response("Hello")
    assert runner.last_bot_message == "Hello! How can I help you today?"

    # Multi-turn with full transcript assertion
    await runner.speak_and_wait_for_response("I need help")
    assert runner.transcript == [
        Turn("user", "Hello"),
        Turn("bot", "Hello! How can I help you today?"),
        Turn("user", "I need help"),
        Turn("bot", "Sure, I can help with that."),
    ]
```

### Testing Interruptions

```python
async with TestRunner(app=app, tts_delay=0.05) as runner:
    await runner.join()
    await runner.speak("Tell me a long story")
    await runner.interrupt_bot("Wait, stop")
    # Bot receives the interruption; verify session state as needed
```

Use `tts_delay` to slow TTS output so the bot is still speaking when `interrupt_bot` fires.

### Gray-Box Inspection

```python
async with TestRunner(app=app) as runner:
    await runner.join()
    await runner.speak_and_wait_for_response("Set my theme to dark")

    # Inspect ADK session state (what tools wrote)
    state = await runner.session_state()
    assert state.get("theme") == "dark"

    # Inspect raw ADK events
    events = await runner.events()
```

### Test Utilities

```python
from test_utils import simplify_events

events = await runner.events()
simplified = simplify_events(events)
# Returns: [("user", "Hello"), ("agent", "Hi there!"), ...]
```

## API Reference

### Core Classes

- **`AdkBasedLLMService(agent, session_service, session_params, plugins=[])`**: Main LLM service. Creates the ADK App and Runner internally. Override `_on_state_delta(state_delta)` to forward state to clients, and `process_frame` + `_persist_and_run(content, state_delta)` to inject system events.
- **`SessionParams(app_name, user_id, session_id)`**: Dataclass for session identification.
- **`AdkInterruptionPlugin`**: ADK plugin for deterministic interruption handling. Pass in `plugins=[AdkInterruptionPlugin()]` when creating the service.

### TTS Factory

- **`make_adk_aware_tts(base_class)`**: Wraps any `TTSService` subclass to push `AdkAudioContextCompletedFrame` when an audio context plays to completion. Required for accurate interruption tracking.

### Context Aggregators

Created via `llm.create_context_aggregator()`:
- **User aggregator** (`AdkUserContextAggregator`): Persists speech to ADK, triggers the LLM service. Override `_build_user_event(text, session)` to inject custom context parts.
- **Assistant aggregator** (`AdkAssistantContextAggregator`): Tracks spoken text per TTS audio context. Writes `[HEARD]` events on interruption.

## Requirements

- Python >= 3.12
- pipecat-ai >= 0.0.102, < 1.0.0
- google-adk >= 1.18.0
- google-genai >= 1.51.0

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
