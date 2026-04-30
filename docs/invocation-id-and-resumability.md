# Invocation ID and Resumability

## Why we pre-persist user events

The standard ADK pattern passes `new_message` directly to `runner.run_async`, which appends it to the session and runs the agent in one call. We don't do this.

Instead, `AdkUserContextAggregator` builds and persists the user event *before* calling the runner, then hands off only the `invocation_id` via `AdkContextFrame`. `AdkBasedLLMService` resumes from that pre-persisted event.

The reason is the `_build_user_event` extension point. Subclasses override it to enrich the event before it enters the session:

```python
async def _build_user_event(self, text: str, session) -> Event:
    # Override to inject context: XML tags, code diffs, section info,
    # timing messages, state_delta, etc.
    return Event(
        invocation_id=Event.new_id(),
        author="user",
        content=Content(role="user", parts=[Part(text=text)]),
    )
```

If we passed `new_message` directly, ADK would persist it as-is with no enrichment opportunity. Pre-persisting gives callers full control over what ends up in the session journal.

The same pattern powers `_persist_and_run` in `AdkBasedLLMService`, which lets subclasses inject system events (silence warnings, domain triggers, environment context) and then invoke the agent — again without going through the user aggregator path.

## Why ResumabilityConfig is required

Passing `invocation_id` to `runner.run_async` is ADK's "resume" path. ADK's runner hard-enforces this:

```python
# runners.py (google-adk source)
if invocation_id:
    if not self.resumability_config or not self.resumability_config.is_resumable:
        raise ValueError(
            f'invocation_id: {invocation_id} is provided but the app is not resumable.'
        )
```

There is no workaround. `ResumabilityConfig(is_resumable=True)` is a hard prerequisite for our invocation flow.

`AdkBasedLLMService` sets this unconditionally when it constructs the `App` internally (service.py:96), so users never need to configure it. But it is a real dependency — if ADK changes or removes `ResumabilityConfig` (it is currently marked `@experimental`), our invocation flow breaks.

## Trade-offs and known limitations

**We cannot accept a pre-built `App`.** Because we need to control `ResumabilityConfig` and wire plugins, `AdkBasedLLMService` constructs the `App` itself from the user-supplied `agent`. This means users cannot set other `App`-level config like `events_compaction_config` or `context_cache_config` without us exposing those fields explicitly. Tracked in [issue #4](https://github.com/recruit41/pipecat-adk/issues/4).

**ResumabilityConfig is experimental.** If ADK stabilizes or removes it, we may need to adjust. The alternative — passing `new_message` to `run_async` and losing the enrichment extension point — would be a significant design regression.
