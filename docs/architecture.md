# Architecture

`AdkBasedLLMService` (and its companion aggregators) is a Pipecat `LLMService` that replaces the standard `_process_context → LLM API → LLMTextFrame` flow with Google ADK's `runner.run_async`. The bridge has three responsibilities and only three:

1. **User speech → ADK session**: collect user turns, persist them as ADK events, trigger the runner
2. **ADK events → Pipecat frames**: convert streaming text and function call lifecycle to the frames Pipecat components downstream expect
3. **Interruption correction**: when the user interrupts, write an accurate `[HEARD]` event to the ADK session so the journal reflects what was actually heard

Everything else—domain frame handling, state forwarding to clients, section routing, guardrails—is application-layer concern and belongs in subclasses.

---

## Data Flow

```
User speaks
  → STT → TranscriptionFrame
  → AdkUserContextAggregator
      persist Event(invocation_id=X, author="user", ...) to session
      push AdkContextFrame(invocation_id=X)
  → AdkBasedLLMService
      AdkLLMFullResponseStartFrame(invocation_id=X)
      runner.run_async(invocation_id=X) → stream ADK events
        partial event → AdkLLMTextFrame(text, invocation_id=X)
        function call → FunctionCallsStartedFrame + FunctionCallInProgressFrame (both directions)
        function response → FunctionCallResultFrame (both directions)
        state_delta → _on_state_delta(state_delta)
      LLMFullResponseEndFrame
  → TTS (must be wrapped with AdkTTSMixin)
      AdkLLMFullResponseStartFrame triggers mixin to set _pending_adk_invocation_id=X
      mixin.create_context_id() returns X instead of a random UUID
      TTSTextFrame(context_id=X, text="sentence 1")
      TTSTextFrame(context_id=X, text="sentence 2")
  → transport.output()
  → AdkAssistantContextAggregator
      AdkLLMFullResponseStartFrame: register invocation X in _invocations dict
      TTSTextFrame(context_id=X): append text to _invocations[X]
      BotStoppedSpeakingFrame: clear _invocations (clean turn, no [HEARD] needed)
      InterruptionFrame: write [HEARD] event for each invocation with accumulated text
```

---

## Interruption Handling

### Why the Current Approach

Naively, you might buffer the agent's streaming response and only commit what was spoken. This fails because:
- Tool calls mid-response would be lost entirely
- ADK's session wouldn't reflect what actually happened
- Buffering in a streaming system creates subtle timing bugs

Instead, the bridge uses an "accountant's approach": commit everything immediately, then annotate what was heard.

### How [HEARD] Events Work

**Step 1: ADK commits the full response**

`runner.run_async` streams the complete agent response and persists it to the session immediately. This is the ground truth of what the agent *said*.

**Step 2: The aggregator tracks what was spoken**

`AdkAssistantContextAggregator` accumulates `TTSTextFrame` text keyed by `context_id`. `AdkTTSMixin` (which must be applied to the TTS service) overrides `create_context_id()` to return the current ADK `invocation_id` instead of a random UUID, so all sentences for a single agent response share the same `context_id = invocation_id`. When a `BotStoppedSpeakingFrame` arrives (clean turn, no interruption), the aggregator clears the tracking map — no correction needed.

**Step 3: On interruption, write a [HEARD] event**

When `InterruptionFrame` arrives, the aggregator's buffer contains exactly the text that reached the TTS pipeline before the interruption. For each context still in the buffer, it writes:

```
user: '<system>[HEARD] Agent was interrupted. Candidate only heard: "sentence 1"</system>'
```

This event goes to the same ADK session, immediately after the full model response.

**Step 4: AdkInterruptionPlugin rewrites the request**

Before the next LLM call, `AdkInterruptionPlugin.before_model_callback` scans `llm_request.contents` for `[HEARD]` markers. For each one found, it:
1. Locates the immediately preceding model event in the request
2. Replaces that event's text with the heard portion
3. Removes the `[HEARD]` event from the request entirely

The LLM sees only what the user actually heard. The full response and the `[HEARD]` marker remain in the ADK session history for auditing.

### Why This Is Deterministic

The heard text is sourced directly from `TTSTextFrame.text` frames that actually passed through the pipeline. There is no fuzzy matching, no ASR comparison, no difflib — the text is exact. The same audio context interrupted at the same point will always produce the same `[HEARD]` event and the same truncated model text in the request.

---

## Extension Points

### 1. `_build_user_event(text, session) → Event`

Called in `AdkUserContextAggregator.push_aggregation` every time the user finishes speaking, before the event is persisted. The default creates a plain `Content(role="user", parts=[Part(text=text)])` event.

Override to:
- Wrap speech in application-specific XML: `<candidate>{text}</candidate>`
- Append extra context parts (code diffs, timing messages, supervisor instructions)
- Add a `state_delta` to the same event (e.g. increment a turn counter)

```python
from pipecat_adk.aggregators import AdkUserContextAggregator
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part

class MyUserAggregator(AdkUserContextAggregator):
    async def _build_user_event(self, text: str, session) -> Event:
        return Event(
            invocation_id=Event.new_id(),
            author="user",
            content=Content(role="user", parts=[
                Part(text=f"<system>Turn {session.state.get('turn', 0) + 1}</system>"),
                Part(text=text),
            ]),
            actions=EventActions(state_delta={"turn": session.state.get("turn", 0) + 1}),
        )
```

**What the bridge does with the returned event:** persists it via `session_service.append_event`, then pushes `AdkContextFrame(invocation_id=event.invocation_id)` to trigger the LLM service.

---

### 2. `_on_state_delta(state_delta: dict)`

Called in `AdkBasedLLMService._push_frames_from_event` for every ADK event that carries a `state_delta`. The default is a no-op. Guaranteed to be called before any text frames from the same event, so clients receive state before the bot starts speaking the response.

Override to forward session state to your client:

```python
class MyLLMService(AdkBasedLLMService):
    async def _on_state_delta(self, state_delta: dict) -> None:
        await self.push_frame(RTVIServerMessageFrame(
            data={"type": "state-sync", "state_delta": state_delta}
        ))
```

---

### 3. `process_frame` + `_persist_and_run`

The bridge handles `AdkContextFrame` (which triggers `_run_adk`). All other application-domain frames should be handled by subclasses.

`_persist_and_run(content, state_delta)` is the standard way to inject a system event into ADK and immediately run the agent. It creates the event, persists it, and calls `_run_adk`.

```python
class MyLLMService(AdkBasedLLMService):
    async def process_frame(self, frame, direction):
        if isinstance(frame, UserIdleFrame):
            await self._persist_and_run(
                content=Content(role="user", parts=[
                    Part(text=f"<system>User silent for {frame.idle_duration}s.</system>")
                ])
            )
        elif isinstance(frame, MeetingStartedFrame):
            await self._persist_and_run(
                content=Content(role="user", parts=[
                    Part(text="<system>User has joined the session.</system>")
                ])
            )
        else:
            await super().process_frame(frame, direction)
```

---

## What Belongs in the Application Layer

The bridge has zero knowledge of application domain. The following are definitively not bridge concerns:

| Concern | Why it's application-specific |
|---------|-------------------------------|
| XML wrapping (e.g. `<candidate>`) | A prompt convention, not a Pipecat/ADK contract |
| Timing messages | Domain logic |
| RTVI state-sync format | Application transport format |
| Multi-agent section routing | Application agent topology |
| Idle escalation / silence warnings | Application behavior policy |
| Domain frame types (quiz, violations, etc.) | Application feature |
| Guardrail interruptions vs. real interruptions | Application policy |
| Session cleanup / expiration | Application infrastructure |

The bridge (`pipecat_adk/`) has no imports from application code. Application layers subclass bridge types and add all domain logic.

---

## Public API Contract

The stable extension surface:

| Symbol | Description |
|--------|-------------|
| `AdkBasedLLMService.__init__(app, session_service, session_params)` | Construct with a pre-built `App`; alternatively pass `agent + plugins` to have the service build the `App` internally |
| `AdkBasedLLMService._persist_and_run(content, state_delta?)` | Inject event and run agent |
| `AdkBasedLLMService._on_state_delta(state_delta)` | Override to push state to client |
| `AdkUserContextAggregator._build_user_event(text, session) → Event` | Override to customize user events |
| `AdkContextAggregatorPair.user()` / `.assistant()` | Access the two aggregators |
| `AdkTTSMixin` | Mixin for TTS services; overrides `create_context_id()` to return `invocation_id`, linking played audio to the ADK invocation for `[HEARD]` tracking |
| `AdkLLMFullResponseStartFrame(invocation_id)` | Signals start of a new ADK invocation; consumed by the assistant aggregator and `AdkTTSMixin` |
| `AdkLLMTextFrame(text, invocation_id)` | LLM text with provenance; `append_to_context=False` so parent aggregator never accumulates it |

Internal and not stable: `_run_adk`, `_push_frames_from_event`, `_write_heard_event`, `_context_aggregation`.
