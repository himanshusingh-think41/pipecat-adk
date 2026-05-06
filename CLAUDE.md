# CLAUDE.md

Development guide for pipecat-adk contributors.

## Why

[Pipecat](https://github.com/pipecat-ai/pipecat) owns real-time audio; [Google ADK](https://github.com/google/adk-python) owns agent logic. Neither was designed for the other. Pipecat's `LLMContext` accumulates messages in-process; ADK manages sessions as persistent event journals. The impedance mismatch means naive composition loses interruption correctness, tool-call auditability, and session persistence.

This library swaps Pipecat's LLM service and context aggregators to route through ADK's `runner.run_async`, so ADK owns all conversation state while Pipecat handles audio. See [docs/architecture.md](docs/architecture.md).

## Development Commands

```bash
uv sync
uv run python -m unittest discover -s tests -v                              # all tests
uv run python -m unittest tests.test_with_mocks -v                          # specific file
uv run python -m unittest tests.test_with_mocks.TestWithMocks.test_basic_interaction -v
```

## Source Map

| File | Key abstraction |
|------|----------------|
| `service.py` | `AdkBasedLLMService` — receives `AdkContextFrame`, calls `runner.run_async(invocation_id)`, converts ADK events to Pipecat frames |
| `aggregators.py` | `AdkUserContextAggregator` — persists user event to ADK, pushes `AdkContextFrame`; `AdkAssistantContextAggregator` — accumulates spoken text this turn, writes `[HEARD]` on interruption, clears on `BotStoppedSpeakingFrame` |
| `interruption.py` | `AdkInterruptionPlugin` — `before_model_callback` finds `[HEARD]` markers, truncates preceding model event, removes marker |
| `tts_mixin.py` | `AdkTTSMixin` — overrides `create_context_id()` to return the ADK `invocation_id`; required for `[HEARD]` tracking |
| `frames.py` | `AdkContextFrame(invocation_id)` |
| `types.py` | `SessionParams(app_name, user_id, session_id)` |

Extension points: `_build_user_event`, `_on_state_delta`, `_persist_and_run` — see [docs/architecture.md](docs/architecture.md).

## Design Decisions

### ADK owns context, not Pipecat

`AdkUserContextAggregator.push_aggregation` persists directly to ADK and skips `LLMContext` entirely (`super()` is not called). Consequence: Pipecat's ecosystem components that read `LLMContext` (`LLM Log Observer`, `Mem0`, `IVR Navigator`) won't see messages. Access history via `session_service.get_session()`.

### Accountant's approach to interruptions

ADK commits the full response immediately — audit trail preserved, tool calls survive. On interruption, `[HEARD]` events annotate what was actually spoken; `AdkInterruptionPlugin` rewrites the request at read-time. The alternative (buffer and only commit the spoken portion) loses tool calls mid-response and creates race conditions in streaming.

### [HEARD] is exact, not fuzzy

Heard text is sourced directly from `TTSTextFrame.text` frames that passed through the pipeline — no difflib, no ASR re-comparison. At turn end, `AdkAssistantContextAggregator` knows whether the turn was interrupted (`InterruptionFrame`) or clean (`BotStoppedSpeakingFrame`), and acts accordingly: write `[HEARD]` or just clear the buffer.

### AdkTTSMixin is required for [HEARD] tracking

`AdkAssistantContextAggregator` correlates `TTSTextFrame.context_id` with ADK `invocation_id` to know which spoken text belongs to which turn. Standard TTS services generate a random UUID for `context_id` each turn, so the aggregator can never match them. `AdkTTSMixin` overrides `create_context_id()` to return `invocation_id` instead. **Always apply `AdkTTSMixin` to your TTS service** — without it, `[HEARD]` events are never written and interruption handling silently degrades.

```python
class AdkGoogleTTSService(AdkTTSMixin, GoogleTTSService):
    pass
```

### Stale session double-reload in push_aggregation

`AdkUserContextAggregator.push_aggregation` loads the session twice: once to pass to `_build_user_event` (which may read `session.state`), and again immediately before `session_service.append_event`. The second reload is necessary because `runner.run_async` from a concurrent invocation may advance `_storage_update_marker` between the first load and the append, causing a stale-session rejection. The extra round-trip is cheap; the failure mode without it is silent data loss.

### Function call frames: both directions

`AdkBasedLLMService._handle_function_call` pushes `FunctionCallsStartedFrame` and `FunctionCallInProgressFrame` both `UPSTREAM` and `DOWNSTREAM`. Upstream: `STTMuteFilter` needs to mute mic during tool execution. Downstream: UI needs "thinking..." indicators.

### Pre-persisting user events and ResumabilityConfig

We persist the user event to ADK *before* calling `runner.run_async`, then resume via `invocation_id` — this enables `_build_user_event` to enrich events with contextual data before they enter the session journal, and requires `ResumabilityConfig(is_resumable=True)` which ADK enforces as a hard constraint. See [docs/invocation-id-and-resumability.md](docs/invocation-id-and-resumability.md).

### Function call frames blocked from LLMContext

`AdkAssistantContextAggregator` no-ops `_handle_function_call_in_progress/result/cancel`. ADK manages tool calls internally in its session; letting those frames into `LLMContext` produces malformed context entries.

## Testing

End-to-end flows with mock services — no real API calls, no network. See **[tests/CLAUDE.md](tests/CLAUDE.md)** for the full guide (MockLLM, TestRunner DSL, wait strategies, gray-box inspection).

**Quick orientation:**

```python
async with TestRunner(app=app) as runner:   # app = App(name="agents", ...)
    await runner.join()
    await runner.speak_and_wait_for_response("Hi")
    assert runner.transcript == [Turn("user", "Hi"), Turn("bot", "Hello!")]
```

| File | Coverage |
|------|----------|
| `test_with_mocks.py` | Integration flows: basic, interruption, function calls, multi-turn |
| `test_components.py` | Unit tests for `AdkAssistantContextAggregator` [HEARD] logic |
| `test_plugin.py` | `AdkInterruptionPlugin` edge cases (12 tests) |
| `test_utils.py` | `simplify_events()` |

`TestRunner` accepts `app=App(name="agents", ...)` — app name must be `"agents"` to match the hardcoded session params inside `TestRunner`.

## Gotchas

- ADK agent names require underscores: `name="my_agent"` not `name="my-agent"`
- `AdkBasedLLMService` accepts either `app=App(...)` (full control) or `agent + plugins` (service builds App internally). Either way, the App **must** have `ResumabilityConfig(is_resumable=True)`.
- `AdkTTSMixin` must be mixed into every TTS service used with this library, or `[HEARD]` events will never be written.
- Use `DatabaseSessionService` for production — `InMemorySessionService` loses all history on restart. Both require creating the session before starting the pipeline.

## Dependencies

See [`pyproject.toml`](pyproject.toml). Key: `pipecat-ai>=1.1.0,<2.0.0`, `google-adk>=1.18.0`.
