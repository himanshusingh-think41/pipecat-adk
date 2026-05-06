# RFC: Invocation Provenance and [HEARD] Correctness in pipecat-adk

**Status:** Proposed  
**Audience:** Implementors with working knowledge of Pipecat frame pipelines, Google ADK session model, and asyncio  

---

## 1. Problem

### 1.1 The Impedance Mismatch

Pipecat owns real-time audio transport and stream processing. Google ADK owns agent reasoning and conversation state. The bridge between them must satisfy two invariants simultaneously:

1. **ADK session invariant**: The session journal must accurately reflect what the agent said and what the user heard. ADK's `before_model_callback` plugins use the journal as ground truth.

2. **Audio invariant**: A sentence is only "heard" if the transport finished playing its audio before the user interrupted. Sentences still buffered or mid-playback at interruption time were not heard.

The fundamental difficulty is that ADK commits model events to the session journal **immediately** as `runner.run_async` yields them. By the time the transport finishes playing audio and we know what was heard, the full response is already committed. We must amend the journal post-hoc via a `[HEARD]` annotation event.

### 1.2 The Provenance Gap

Pipecat has no concept of an LLM invocation identifier. A call to `runner.run_async` produces `LLMTextFrame`s that flow downstream to TTS, which produces `TTSTextFrame`s that the transport holds, plays, and then releases upstream. By the time a `TTSTextFrame` arrives at the assistant aggregator, there is no mechanism to determine which ADK invocation produced it. Without this link, a `[HEARD]` event cannot be anchored to the correct model event in the ADK journal.

### 1.3 What This RFC Solves

This RFC specifies:

- A provenance chain from ADK `invocation_id` → `LLMTextFrame` → TTS `context_id` → `TTSTextFrame`
- A per-invocation accumulation structure in the assistant aggregator
- The correct lifecycle event (`TTSStoppedFrame`) that triggers `[HEARD]` writes
- Conformance tests for all edge cases

---

## 2. System Overview

### 2.1 Pipeline Topology

```
AudioIn → STT
            ↓
    AdkUserContextAggregator      (persists user event to ADK; triggers service)
            ↓
    AdkBasedLLMService            (calls runner.run_async; emits LLM frames)
            ↓
    AdkAssistantContextAggregator (accumulates per-invocation TTS text; writes [HEARD])
            ↓
         TTS Service               (sentences; emits TTSTextFrame + TTSAudioRawFrame)
            ↓
      Output Transport             (plays audio; releases TTSTextFrame upstream after playback)
```

Frame directions:
- **Downstream** (left to right above): `LLMFullResponseStartFrame`, `AdkLLMTextFrame`, `LLMFullResponseEndFrame`
- **Upstream** (right to left): `TTSTextFrame`, `TTSStoppedFrame`, `BotStoppedSpeakingFrame`, `InterruptionFrame`

### 2.2 ADK Session Journal

The journal is append-only. `runner.run_async(invocation_id=X, new_message=None)` resumes from the pre-persisted event with that `invocation_id` and appends model events as they stream. A `[HEARD]` annotation is appended after the model event it references; `AdkInterruptionPlugin` (a `before_model_callback`) rewrites the LLM request view at read time to truncate the model event to only the heard text and removes the annotation.

The journal is never mutated; only the model's view of it is rewritten per-call.

---

## 3. New Frame Types

### 3.1 `AdkLLMFullResponseStartFrame`

Extends `LLMFullResponseStartFrame` with an `invocation_id` field.

```python
@dataclass
class AdkLLMFullResponseStartFrame(LLMFullResponseStartFrame):
    invocation_id: str = ""
```

**Semantics**: Pushed by `AdkBasedLLMService` at the start of each `runner.run_async` call, before any text frames. Carries the ADK invocation ID downstream to TTS.

**Constraint**: `invocation_id` MUST be non-empty. Implementations MUST raise if it is empty.

### 3.2 `AdkLLMTextFrame`

Extends `LLMTextFrame` with an `invocation_id` field and sets `append_to_context = False`.

```python
@dataclass
class AdkLLMTextFrame(LLMTextFrame):
    invocation_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.append_to_context = False  # TTS text, not LLM text, builds the assistant context
```

**Semantics**: Every text fragment emitted from an ADK event carries its invocation's ID. `append_to_context = False` prevents the parent `LLMAssistantAggregator._handle_text` from accumulating this frame into `_aggregation` — only `TTSTextFrame` (which reflects actually-played audio) should contribute to the assistant context.

**Constraint**: Implementations MUST set `append_to_context = False`. If the parent aggregator accumulates `LLMTextFrame` text, `_aggregation` would contain both LLM-generated and TTS-forwarded text, producing duplicate content.

---

## 4. TTS Service: Pinning `context_id` to `invocation_id`

### 4.1 Mechanism

The TTS base class generates a random `context_id` UUID in this block (pseudocode matching pipecat's implementation):

```python
elif isinstance(frame, LLMFullResponseStartFrame):
    self._llm_response_started = True
    self._turn_context_id = self.create_context_id()   # generates UUID
    await self.on_turn_context_created(self._turn_context_id)
    await self.push_frame(frame, direction)
```

Implementations MUST override this to substitute `invocation_id` when the frame is an `AdkLLMFullResponseStartFrame`:

```python
elif isinstance(frame, LLMFullResponseStartFrame):
    self._llm_response_started = True
    if isinstance(frame, AdkLLMFullResponseStartFrame) and frame.invocation_id:
        self._turn_context_id = frame.invocation_id
    else:
        self._turn_context_id = self.create_context_id()
    await self.on_turn_context_created(self._turn_context_id)
    await self.push_frame(frame, direction)
```

With `reuse_context_id_within_turn = True` (the default), this `_turn_context_id` is reused for every sentence in the turn, including across function-call boundaries within the same ADK invocation. All `TTSTextFrame` and `TTSAudioRawFrame` instances for this response carry `context_id = invocation_id`.

**Mixin pattern**: Provide `AdkTTSMixin` that subclasses can apply to any concrete TTS service (ElevenLabs, Google, Gemini, etc.) to inject this behaviour without duplicating logic.

### 4.2 `TTSSpeakFrame` Interaction

`TTSSpeakFrame` (used for hold messages, transition audio, error recovery) temporarily clears `_turn_context_id` and generates a fresh UUID. This UUID is never registered in the per-invocation map (§5.3), so `TTSStoppedFrame` for these contexts is silently ignored by the aggregator. No client-side `append_to_context = False` workaround is required.

---

## 5. `AdkAssistantContextAggregator`

### 5.1 Design Principles

1. **Do not use `_aggregation` (parent state).** The parent accumulates from any `TextFrame` with `append_to_context = True`. We bypass this entirely.
2. **Accumulate per invocation, keyed by `context_id` (= `invocation_id`).** A dict maps each active invocation to its accumulated TTS text and interrupt flag.
3. **Write `[HEARD]` on `TTSStoppedFrame`, not on `InterruptionFrame`.** This guarantees all `TTSTextFrame`s for the audio context have already arrived (they precede `TTSStoppedFrame` in the serialization queue).
4. **`push_aggregation` is a no-op.** ADK already owns the session; we do not write to `LLMContext`.

### 5.2 Data Structure

```python
@dataclass
class _InvocationAccumulation:
    texts: list[str] = field(default_factory=list)
    was_interrupted: bool = False

    def heard_text(self) -> str:
        return " ".join(self.texts).strip()
```

The aggregator maintains:

```python
self._invocations: dict[str, _InvocationAccumulation] = {}
# key: invocation_id (== context_id)
```

### 5.3 Frame Handling

#### 5.3.1 `AdkLLMFullResponseStartFrame` (downstream)

```python
self._invocations[frame.invocation_id] = _InvocationAccumulation()
```

An entry is created for every new ADK invocation. If an entry for this `invocation_id` already exists (re-entrant invocation), it MUST be overwritten and a warning logged.

#### 5.3.2 `TTSTextFrame` (upstream)

```python
async def _handle_text(self, frame: TextFrame) -> None:
    if isinstance(frame, TTSTextFrame):
        state = self._invocations.get(frame.context_id)
        if state is not None:
            if frame.text:
                state.texts.append(frame.text)
        # Unknown context_id (TTSSpeakFrame, stale) → silently ignored
        return  # do NOT call super(); do NOT populate _aggregation
    # For all other TextFrame subtypes (LLMTextFrame with append_to_context=False → super no-ops)
    await super()._handle_text(frame)
```

**Critical**: return early for all `TTSTextFrame` cases, including unknown `context_id`. Do not fall through to `super()`.

#### 5.3.3 `InterruptionFrame` (upstream)

```python
async def _handle_interruptions(self, frame: InterruptionFrame) -> None:
    for state in self._invocations.values():
        state.was_interrupted = True
    await super()._handle_interruptions(frame)  # calls push_aggregation (our no-op)
```

Marks ALL active invocations as interrupted. Does not write `[HEARD]` here — deferred to `TTSStoppedFrame`.

**Rationale for deferral**: At the time `InterruptionFrame` arrives, some `TTSTextFrame`s for fully-played sentences may still be in transit upstream through the pipeline. Writing `[HEARD]` at interrupt time would miss these. `TTSStoppedFrame` is the serialization barrier.

#### 5.3.4 `TTSStoppedFrame` (upstream)

```python
async def _handle_invocation_end(self, context_id: str) -> None:
    state = self._invocations.pop(context_id, None)
    if state is None:
        return  # TTSSpeakFrame context or already cleaned up — ignore
    if state.was_interrupted and state.heard_text():
        await self._write_heard_event(state.heard_text(), context_id)
    # Clean turn: ADK session already has the full response. No action.

# In process_frame:
elif isinstance(frame, TTSStoppedFrame) and frame.context_id:
    await self._handle_invocation_end(frame.context_id)
```

**Guarantee**: By the time `TTSStoppedFrame(context_id=X)` arrives upstream, every `TTSTextFrame(context_id=X)` has already been forwarded by the transport and processed by the aggregator. The serialization queue enforces this ordering.

**Clean turn (`was_interrupted = False`)**: `state.texts` contains the full spoken text, but ADK already has it — discard.

**Interrupted turn (`was_interrupted = True`, `heard_text()` non-empty)**: Write `[HEARD]` anchored to `context_id` (= `invocation_id`).

**Interrupted turn with no text**: Bot was interrupted before any sentence completed. No `[HEARD]` needed — ADK's session has the full uncommitted response, and `AdkInterruptionPlugin` will find no `[HEARD]` marker and leave the model event intact (which is incorrect; see §7.1 for the known limitation).

#### 5.3.5 `push_aggregation`

```python
async def push_aggregation(self) -> str:
    # _aggregation is always empty: AdkLLMTextFrame has append_to_context=False,
    # TTSTextFrames are consumed by _handle_text before super() is called.
    await self.reset()
    return ""
```

#### 5.3.6 Function call frame handlers

The following MUST be no-ops. ADK manages function call lifecycle in its session internally; allowing these frames into `LLMContext` produces malformed context entries.

```python
async def _handle_function_call_in_progress(self, frame): pass
async def _handle_function_call_result(self, frame): pass
async def _handle_function_call_cancel(self, frame): pass
```

---

## 6. `AdkUserContextAggregator`

### 6.1 Pre-Persist Pattern

The user event is persisted to ADK **before** `runner.run_async` is called. This ensures the user's speech is durably in the journal even if the pipeline is cancelled between persist and service execution.

```python
async def push_aggregation(self) -> str:
    if not self._aggregation:
        return ""
    aggregation = self.aggregation_string()
    await self.reset()
    # Do NOT call super() — ADK owns conversation state, not LLMContext
    session = await self.session_service.get_session(...)
    event = await self._build_user_event(aggregation, session)
    await self.session_service.append_event(session, event)
    await self.push_frame(AdkContextFrame(invocation_id=event.invocation_id))
    return aggregation
```

### 6.2 Extension Point: `_build_user_event`

Override to enrich the user event before it enters the journal:

```python
async def _build_user_event(self, text: str, session: Session) -> Event:
    """
    text: aggregated STT transcription
    session: current ADK session (read session.state for context enrichment)
    Returns: Event ready for session_service.append_event
    """
    return Event(
        invocation_id=Event.new_id(),
        author="user",
        content=Content(role="user", parts=[Part(text=text)]),
    )
```

---

## 7. `AdkBasedLLMService`

### 7.1 `_run_adk`

```python
async def _run_adk(self, invocation_id: str) -> None:
    await self.push_frame(AdkLLMFullResponseStartFrame(invocation_id=invocation_id))
    await self.push_frame(LLMFullResponseStartFrame())  # for TTS compatibility
    try:
        async for event in self.runner.run_async(
            user_id=..., session_id=..., invocation_id=invocation_id,
            new_message=None,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            await self._push_frames_from_event(event)
    except Exception as e:
        logger.exception(f"ADK error: {e}")
    finally:
        await self.push_frame(LLMFullResponseEndFrame())
```

**Note on two start frames**: `AdkLLMFullResponseStartFrame` carries `invocation_id` for the TTS mixin and the aggregator. `LLMFullResponseStartFrame` (the standard pipecat frame) is also pushed for compatibility with TTS services that are not using the ADK mixin.

**Alternatively**, a single `AdkLLMFullResponseStartFrame` suffices if all TTS services in the deployment use `AdkTTSMixin`, since `AdkLLMFullResponseStartFrame` extends `LLMFullResponseStartFrame` and TTS handles the isinstance check.

### 7.2 `_push_frames_from_event`

For each partial event with text parts:

```python
await self.push_frame(AdkLLMTextFrame(text=text, invocation_id=event.invocation_id))
```

This replaces plain `LLMTextFrame`. The `invocation_id` is available on every ADK event; it is the same value throughout a single `runner.run_async` call.

### 7.3 Extension Points

| Method | Signature | Purpose |
|--------|-----------|---------|
| `_on_state_delta` | `(state_delta: dict) → None` | Called for every ADK event with a `state_delta`. Override to forward state to clients (RTVI state-sync, etc.). |
| `_persist_and_run` | `(content: Content, state_delta: dict) → None` | Persist a system event and invoke the agent. Use from `process_frame` overrides to inject domain events (silence warnings, section transitions, etc.). |
| `_build_user_event` | see §6.2 | Enrich user events before they enter the journal. |

---

## 8. `AdkInterruptionPlugin`

### 8.1 `[HEARD]` Event Format

Written by `AdkAssistantContextAggregator._write_heard_event`:

```
<system>[HEARD] invocation_id="{invocation_id}" Candidate only heard: "{heard_text}"</system>
```

The `invocation_id` field anchors the annotation to a specific model event in the journal. This is a departure from positional lookup and eliminates the fragility described below.

### 8.2 `before_model_callback` Logic

For each `[HEARD]` event in `llm_request.contents`:

1. Extract `invocation_id` and `heard_text` from the annotation.
2. Find the model event in `contents` whose `invocation_id` matches. ADK preserves `invocation_id` on events when constructing the LLM request.
3. Replace that model event's text parts with `heard_text`.
4. Remove the `[HEARD]` event from `contents`.

**Fallback**: If no model event with the matching `invocation_id` is found (compaction, summarisation), log a warning and remove the `[HEARD]` event without modification. Do not attempt positional fallback.

---

## 9. Conformance Tests

Each test is expressed as a sequence of frames entering the `AdkAssistantContextAggregator` and the expected `[HEARD]` events written to the ADK session. Frame directions are noted; upstream frames originate from the transport.

---

### CT-01: Clean Turn — No `[HEARD]` Written

**Scenario**: Bot completes a two-sentence response without interruption.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="Sentence one.", context_id="inv1")
← TTSTextFrame(text="Sentence two.", context_id="inv1")
← TTSStoppedFrame(context_id="inv1")
```

**Expected**: zero `[HEARD]` events written to ADK session.

**Invariant**: `_invocations` is empty after `TTSStoppedFrame`.

---

### CT-02: Interrupted Turn — `[HEARD]` Contains Only Fully-Played Sentences

**Scenario**: Two-sentence response; user interrupts after sentence one plays, during sentence two.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="Sentence one.", context_id="inv1")   # sentence one fully played
↑ InterruptionFrame                                        # user interrupts mid-sentence-two
← TTSStoppedFrame(context_id="inv1")                       # audio context cleaned up
```

**Expected**: one `[HEARD]` event with `invocation_id="inv1"` and `heard_text="Sentence one."`.

**Invariant**: `_invocations` is empty after `TTSStoppedFrame`.

---

### CT-03: Interrupted Before Any Sentence Completes — No `[HEARD]`

**Scenario**: User interrupts during the very first sentence; no `TTSTextFrame` has been forwarded.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
↑ InterruptionFrame
← TTSStoppedFrame(context_id="inv1")
```

**Expected**: zero `[HEARD]` events. `state.heard_text()` is empty; the guard `state.heard_text()` short-circuits.

**Note**: The ADK session contains the full uncommitted model response. `AdkInterruptionPlugin` will find no `[HEARD]` marker and leave the model event intact. This is a known limitation (see §10.1).

---

### CT-04: Multi-Sentence, Interrupted at Sentence N

**Scenario**: Five-sentence response; three sentences fully play, user interrupts during sentence four.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="One.", context_id="inv1")
← TTSTextFrame(text="Two.", context_id="inv1")
← TTSTextFrame(text="Three.", context_id="inv1")
↑ InterruptionFrame
← TTSStoppedFrame(context_id="inv1")
```

**Expected**: `[HEARD]` with `heard_text="One. Two. Three."`.

---

### CT-05: Function Call Within Turn — `[HEARD]` Contains Only TTS Text

**Scenario**: Bot says "Let me check that.", runs a tool (silent), then says "The answer is 42." User interrupts during "The answer is 42."

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="Let me check that.", context_id="inv1")
  [no TTSTextFrame during tool execution]
← TTSTextFrame(text="The answer is", context_id="inv1")    # partial — interrupted mid-sentence
↑ InterruptionFrame                                         # interrupt fires before "42." plays
← TTSStoppedFrame(context_id="inv1")
```

**Expected**: `[HEARD]` with `heard_text="Let me check that."` only. "The answer is" was in the TTS buffer but did not complete playback, so its `TTSTextFrame` was NOT forwarded by the transport.

**Invariant**: The `TTSTextFrame` for a sentence is forwarded ONLY after that sentence's audio finishes playing. Sentences mid-playback at interrupt time produce no forwarded `TTSTextFrame`.

---

### CT-06: `TTSSpeakFrame` (System Audio) — No `[HEARD]`

**Scenario**: Application pushes a hold message via `TTSSpeakFrame`. User interrupts during playback.

**Frame sequence**:
```
  [TTSSpeakFrame generates TTSTextFrame with context_id="uuid-not-inv-id"]
← TTSTextFrame(text="One moment please.", context_id="uuid-not-inv-id")
↑ InterruptionFrame
← TTSStoppedFrame(context_id="uuid-not-inv-id")
```

**Expected**: zero `[HEARD]` events. `"uuid-not-inv-id"` is not in `_invocations`; `_handle_invocation_end` finds `state = None` and returns immediately.

**Conformance requirement**: Implementations MUST NOT require clients to set `append_to_context=False` on `TTSSpeakFrame` to achieve this behaviour. The `context_id` mismatch is the mechanism.

---

### CT-07: Stale `TTSTextFrame` from Previous Invocation

**Scenario**: Invocation "inv1" is interrupted. Before `TTSStoppedFrame("inv1")` arrives, invocation "inv2" starts. A stale `TTSTextFrame(context_id="inv1")` arrives during "inv2".

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="First.", context_id="inv1")
↑ InterruptionFrame
→ AdkLLMFullResponseStartFrame(invocation_id="inv2")      # new invocation starts
← TTSTextFrame(context_id="inv1", text="Stale.")           # late frame from inv1
← TTSStoppedFrame(context_id="inv1")
← TTSTextFrame(context_id="inv2", text="Hello.")
← TTSStoppedFrame(context_id="inv2")
```

**Expected**:
- `[HEARD]` for "inv1" with `heard_text="First."` (stale "Stale." is NOT included because "inv1" has already been popped when `TTSStoppedFrame("inv1")` fires)

Wait — re-examining: in this sequence, `TTSTextFrame(context_id="inv1", text="Stale.")` arrives BEFORE `TTSStoppedFrame(context_id="inv1")`. So "inv1" is still in `_invocations` at that point, and "Stale." would be accumulated.

**Correction**: The test should document that "Stale." IS included because `TTSStoppedFrame` has not yet arrived to pop the entry. The stale frame arriving before `TTSStoppedFrame` is legitimate — the transport forwarded it before cleanup.

**Revised expected**:
- `[HEARD]` for "inv1" with `heard_text="First. Stale."`
- Clean `[HEARD]` / no `[HEARD]` for "inv2" depending on whether "inv2" completes cleanly

**Invariant**: `TTSStoppedFrame` is the serialization barrier. Any `TTSTextFrame` arriving for a context before its `TTSStoppedFrame` is legitimate and MUST be accumulated.

---

### CT-08: Late VAD After Clean Turn — No `[HEARD]`

**Scenario**: Bot finishes cleanly. VAD fires `InterruptionFrame` ~200ms later due to background noise.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="All done.", context_id="inv1")
← TTSStoppedFrame(context_id="inv1")                       # inv1 popped, map is empty
↑ InterruptionFrame                                        # late VAD — fires AFTER cleanup
```

**Expected**: zero `[HEARD]` events. By the time `InterruptionFrame` arrives, `_invocations` is empty; the `for state in _invocations.values()` loop is a no-op.

**This is the key late-VAD protection**: `TTSStoppedFrame` arriving before the spurious `InterruptionFrame` means there is nothing to mark as interrupted.

---

### CT-09: Rapid Re-Interruption (Two Invocations)

**Scenario**: User interrupts "inv1", bot starts "inv2", user interrupts "inv2" before any sentence completes.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="First sentence.", context_id="inv1")
↑ InterruptionFrame                                        # user interrupts inv1
→ AdkLLMFullResponseStartFrame(invocation_id="inv2")
↑ InterruptionFrame                                        # user interrupts inv2 immediately
← TTSStoppedFrame(context_id="inv1")
← TTSStoppedFrame(context_id="inv2")
```

**Expected**:
- `[HEARD]` for "inv1" with `heard_text="First sentence."`
- No `[HEARD]` for "inv2" (no `TTSTextFrame` for "inv2" arrived)

---

### CT-10: Concurrent Invocations (Extension Point Scenario)

**Scenario**: While "inv1" is running (service mid-stream), application code calls `_persist_and_run` which starts "inv2". Both produce TTS audio.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
→ AdkLLMFullResponseStartFrame(invocation_id="inv2")
← TTSTextFrame(text="From inv1.", context_id="inv1")
← TTSTextFrame(text="From inv2.", context_id="inv2")
↑ InterruptionFrame
← TTSStoppedFrame(context_id="inv1")
← TTSStoppedFrame(context_id="inv2")
```

**Expected**:
- `[HEARD]` for "inv1" with `heard_text="From inv1."`
- `[HEARD]` for "inv2" with `heard_text="From inv2."`

**Invariant**: Per-invocation map entries are independent. Each is processed at its own `TTSStoppedFrame`.

---

### CT-11: Empty STT Transcription — No ADK Event, No Run

**Scenario**: STT produces a `TranscriptionFrame` with empty or whitespace-only text.

**Expected**: `AdkUserContextAggregator.push_aggregation` checks `len(self._aggregation) == 0` and returns `""` without persisting an event or pushing `AdkContextFrame`. No ADK invocation starts; `_invocations` remains unchanged.

---

### CT-12: Pipeline Cancel Mid-Turn

**Scenario**: `CancelFrame` arrives while "inv1" is active and has accumulated text.

**Frame sequence**:
```
→ AdkLLMFullResponseStartFrame(invocation_id="inv1")
← TTSTextFrame(text="Partial.", context_id="inv1")
→ CancelFrame
```

**Expected**: `_handle_end_or_cancel` calls `_trigger_assistant_turn_stopped` → `push_aggregation` (no-op). The map entry for "inv1" is NOT automatically cleaned up unless `TTSStoppedFrame` or explicit cancel handling clears it.

**Required implementation behaviour**: On `CancelFrame` or `EndFrame`, implementations MUST clear `_invocations`. Whether to write `[HEARD]` for in-flight entries on cancel is a policy decision; the minimum conformance requirement is that `_invocations` is empty after pipeline termination.

---

### CT-13: `[HEARD]` Anchor Survives ADK Event Reordering

**Scenario**: ADK session contains events from two turns. Plugin must find the correct model event by `invocation_id`, not by position.

**Journal before plugin runs**:
```
user:   "Tell me about Python."        (invocation_id="u1")
model:  "Python is a language..."      (invocation_id="inv1")  ← full response
user:   [HEARD] invocation_id="inv1" heard: "Python is a language"
user:   "What about Ruby?"             (invocation_id="u2")
model:  "Ruby is also great..."        (invocation_id="inv2")  ← full response
```

**Plugin expected behaviour**: Truncate model event `invocation_id="inv1"` to `"Python is a language"`. Leave `invocation_id="inv2"` intact. Remove the `[HEARD]` event.

**Conformance requirement**: Plugin MUST use `invocation_id` from the `[HEARD]` annotation to locate the target model event. Positional lookup (find the most recent model event before the marker) is NOT conformant.

---

## 10. Known Limitations

### 10.1 Interrupted Before First Sentence

When a user interrupts before any sentence finishes playing, `state.heard_text()` is empty and no `[HEARD]` event is written. The ADK session contains the full model response with no annotation. `AdkInterruptionPlugin` will not truncate it. The model believes the user heard everything.

**Mitigation**: The model's own reasoning about the user's response will often correct for this. For strict correctness, a sentinel `[HEARD: nothing]` event could be written, and the plugin would replace the model event with empty text. This is left to implementations that require it.

### 10.2 TTS Normalization Divergence

`AdkInterruptionPlugin` writes `heard_text` into the model event verbatim. If TTS normalization changed the text (e.g., expanded "Dr." to "Doctor", stripped markdown), the model event after truncation will contain TTS-normalized text, not the LLM's original text. This is intentional: the model receives what the user actually heard, not what the model generated.

### 10.3 `TTSStoppedFrame` Delivery on Interruption

The specification requires `TTSStoppedFrame` to be delivered upstream after interruption for cleanup. Implementations MUST verify that their specific TTS service and transport combination delivers `TTSStoppedFrame` in the interrupted path. If `TTSStoppedFrame` is not delivered, `_invocations` entries will leak. Implementations should add a cleanup path in `_handle_interruptions` as a safety net: after a short drain window, any remaining entries with `was_interrupted = True` and non-empty `texts` SHOULD have `[HEARD]` written and be cleared.

---

## 11. Summary of Implementation Checklist

- [ ] Define `AdkLLMFullResponseStartFrame(invocation_id: str)`
- [ ] Define `AdkLLMTextFrame(invocation_id: str, append_to_context=False)`
- [ ] Implement `AdkTTSMixin` overriding `LLMFullResponseStartFrame` handling to pin `_turn_context_id = invocation_id`
- [ ] Implement `_InvocationAccumulation` dataclass
- [ ] In `AdkAssistantContextAggregator`:
  - [ ] Add `_invocations: dict[str, _InvocationAccumulation]`
  - [ ] Handle `AdkLLMFullResponseStartFrame` in `process_frame` (create entry)
  - [ ] Handle `TTSStoppedFrame` in `process_frame` (pop entry, write `[HEARD]` if interrupted)
  - [ ] Override `_handle_text` to route `TTSTextFrame` to map; return early before `super()`
  - [ ] Override `_handle_interruptions` to mark all entries; call `super()` (no-op `push_aggregation`)
  - [ ] Override `push_aggregation` as no-op
  - [ ] Override `_handle_function_call_*` as no-ops
  - [ ] Clear `_invocations` on `CancelFrame`/`EndFrame`
- [ ] In `AdkBasedLLMService._run_adk`: push `AdkLLMFullResponseStartFrame` before `LLMFullResponseStartFrame`
- [ ] In `AdkBasedLLMService._push_frames_from_event`: push `AdkLLMTextFrame` instead of `LLMTextFrame`
- [ ] Update `AdkInterruptionPlugin` to use `invocation_id`-anchored lookup instead of positional
- [ ] Update `[HEARD]` event format to include `invocation_id`
- [ ] Write conformance tests CT-01 through CT-13
