# Architecture

`AdkLLMService` (and its companion Vql aggregators) is a Pipecat `LLMService` that replaces the standard `_process_context → LLM API → LLMTextFrame` flow with Google ADK's `runner.run_async`. The bridge has three responsibilities and only three:

1. **User speech → ADK session**: collect user turns, call the runner with `new_message`, receive the response as a stream
2. **ADK events → Pipecat frames**: convert streaming text and function call lifecycle to the frames Pipecat components downstream expect
3. **Interruption correction**: when the user interrupts, write an accurate `[HEARD]` event to the ADK session so the journal reflects what was actually heard

Everything else — domain frame handling, state forwarding to clients, section routing, guardrails — is application-layer concern and belongs in subclasses.

---

## Naming Convention

- **Vql prefix** (`VqlContextFrame`, `VqlUserContextAggregator`, etc.): Pipecat-layer abstractions that have no knowledge of ADK internals.
- **Adk prefix** (`AdkLLMService`, `AdkInterruptionPlugin`): Components that directly call Google ADK APIs.

This split keeps the boundary clean: Vql components can be unit-tested without ADK; Adk components own session writes.

---

## Data Flow

```
User speaks
  → STT → TranscriptionFrame
  → VqlUserContextAggregator
      mint turn_id (UUID hex)
      push VqlContextFrame(turn_id, text)
  → AdkLLMService
      push VqlLLMFullResponseStartFrame(turn_id)
      runner.run_async(new_message=Content(role="user", parts=[Part(text=text)]))
        first event → learn invocation_id, store turn_id → invocation_id in map
        partial event with text → push VqlLLMTextFrame(text, turn_id)
        function call → FunctionCallsStartedFrame + FunctionCallInProgressFrame (both directions)
        function response → FunctionCallResultFrame (both directions)
        state_delta → _on_state_delta(state_delta)
      push LLMFullResponseEndFrame
  → TTS (must be wrapped with VqlTTSMixin)
      VqlLLMFullResponseStartFrame triggers mixin to set _turn_context_id = frame.turn_id
      TTSTextFrame(context_id=turn_id, text="sentence 1")
      TTSTextFrame(context_id=turn_id, text="sentence 2")
      TTSStoppedFrame(context_id=turn_id)
  → transport.output()
  → VqlAssistantContextAggregator
      LLMFullResponseStartFrame → parent sets _llm_response_started
      TTSTextFrame → parent accumulates in _aggregation
      TTSStoppedFrame(context_id=turn_id):
        push VqlTurnCompletedFrame(turn_id, text, interrupted=False) UPSTREAM
      [or on interruption:]
      VqlInterruptionFrame(turn_id):
        push VqlTurnCompletedFrame(turn_id, text, interrupted=True) UPSTREAM
  → AdkLLMService (receives VqlTurnCompletedFrame upstream)
      pop invocation_id from _turn_invocation_map[turn_id]
      if interrupted and text: write [HEARD] event to ADK session
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

**Step 2: VqlTTSMixin pins turn_id as TTS context_id**

`VqlTTSMixin` intercepts `VqlLLMFullResponseStartFrame` and sets `_turn_context_id = frame.turn_id`. The TTS service then stamps this as `context_id` on `TTSTextFrame` and `TTSStoppedFrame`. No state needs to be stored in the assistant aggregator — the turn_id propagates through the frame itself.

**Step 3: VqlAssistantContextAggregator accumulates and signals**

The aggregator accumulates `TTSTextFrame` text (via parent's `_aggregation`). When the turn ends:
- **Clean turn**: `TTSStoppedFrame(context_id=turn_id)` → pushes `VqlTurnCompletedFrame(turn_id, text, interrupted=False)` upstream
- **Interrupted turn**: `VqlInterruptionFrame(turn_id)` → pushes `VqlTurnCompletedFrame(turn_id, text, interrupted=True)` upstream

**Step 4: AdkLLMService writes the [HEARD] event**

`AdkLLMService` receives `VqlTurnCompletedFrame` upstream. It looks up `invocation_id = _turn_invocation_map[turn_id]`. If `interrupted=True` and there is text, it writes:

```
user: '<system>[HEARD] Agent was interrupted. Candidate only heard: "sentence 1"</system>'
```

This event goes to the ADK session, immediately after the full model response.

**Step 5: AdkInterruptionPlugin rewrites the request**

Before the next LLM call, `AdkInterruptionPlugin.before_model_callback` scans `llm_request.contents` for `[HEARD]` markers. For each one found, it:
1. Locates the immediately preceding model event in the request
2. Replaces that event's text with the heard portion
3. Removes the `[HEARD]` event from the request entirely

The LLM sees only what the user actually heard. The full response and the `[HEARD]` marker remain in the ADK session history for auditing.

### Why This Is Deterministic

The heard text is sourced directly from `TTSTextFrame.text` frames that actually passed through the pipeline. There is no fuzzy matching, no ASR comparison, no difflib — the text is exact.

---

## turn_id vs invocation_id

These are two distinct identifiers that serve different layers:

| | `turn_id` | `invocation_id` |
|--|-----------|-----------------|
| Owner | Vql (pipecat) layer | ADK (generated by runner) |
| Created by | `VqlUserContextAggregator` | ADK's `runner.run_async` |
| Format | `uuid4().hex` | ADK internal format |
| Visible to | All Vql frames | Only `AdkLLMService` |
| Purpose | Correlate user turn through pipecat pipeline | Identify ADK invocation in session journal |

The mapping `turn_id → invocation_id` is maintained exclusively in `AdkLLMService._turn_invocation_map`. No other component knows both IDs.

---

## Extension Points

### 1. `_build_user_event(text) → Content`

Override on `AdkLLMService` to customize what gets sent to the runner as `new_message`.

```python
class MyLLMService(AdkLLMService):
    async def _build_user_event(self, text: str) -> Content:
        return Content(role="user", parts=[
            Part(text=f"<system>Current time: {datetime.now()}</system>"),
            Part(text=text),
        ])
```

---

### 2. `_on_state_delta(state_delta: dict)`

Called for every ADK event that carries a `state_delta`. Default is a no-op. Guaranteed to be called before any text frames from the same event.

```python
class MyLLMService(AdkLLMService):
    async def _on_state_delta(self, state_delta: dict) -> None:
        await self.push_frame(RTVIServerMessageFrame(
            data={"type": "state-sync", "state_delta": state_delta}
        ))
```

---

### 3. `process_frame` + `_persist_and_run`

Override `process_frame` to handle domain frames and call `_persist_and_run` to inject system events directly into ADK without going through the user aggregator.

```python
class MyLLMService(AdkLLMService):
    async def process_frame(self, frame, direction):
        if isinstance(frame, UserIdleFrame):
            await self._persist_and_run(
                content=Content(role="user", parts=[
                    Part(text=f"<system>User silent for {frame.idle_duration}s.</system>")
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
| Session cleanup / expiration | Application infrastructure |

The bridge (`pipecat_adk/`) has no imports from application code. Application layers subclass bridge types and add all domain logic.

---

## Public API Contract

The stable extension surface:

| Symbol | Description |
|--------|-------------|
| `AdkLLMService.__init__(app, session_service, session_params)` | Construct with a pre-built `App`; alternatively pass `agent + plugins` to have the service build the `App` internally |
| `AdkLLMService._build_user_event(text) → Content` | Override to customize the Content passed to runner as `new_message` |
| `AdkLLMService._persist_and_run(content, state_delta?)` | Inject event and run agent |
| `AdkLLMService._on_state_delta(state_delta)` | Override to push state to client |
| `VqlContextAggregatorPair.user()` / `.assistant()` | Access the two aggregators |
| `VqlTTSMixin` | Mixin for TTS services; intercepts `VqlLLMFullResponseStartFrame` to pin `turn_id` as TTS `context_id` for `[HEARD]` tracking |
| `VqlLLMFullResponseStartFrame(turn_id)` | Signals start of a new Vql turn; consumed by `VqlTTSMixin` |
| `VqlLLMTextFrame(text, turn_id)` | LLM text with turn provenance; `append_to_context=False` so parent aggregator never accumulates it |
| `VqlTurnCompletedFrame(turn_id, text, interrupted)` | Pushed upstream when a turn ends; consumed by `AdkLLMService` to decide whether to write `[HEARD]` |

Internal and not stable: `_run_adk`, `_push_frames_from_event`, `_write_heard_event`, `_turn_invocation_map`.
