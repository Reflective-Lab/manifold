# SelectedChatBackend provider/model drift

Status: **bug report** with proposed fix. Seeking maintainer alignment.
Originating context: Quorum's `ManifoldAcquisitionOriginator` smoke test surfaced a provenance inconsistency.

## TL;DR

`SelectedChatBackend.provider()` and `SelectedChatBackend.model()` can return values that disagree with the actual backend handle in `SelectedChatBackend.backend`. The two come from independent selection paths inside `select_chat_backend_with_secret_provider`, and there is no contract that they resolve to the same provider/model.

Observed in the wild: Quorum's smoke recorded provenance `vendor=anthropic model=gemini-2.5-flash` after a real round-trip through Manifold. The provenance descriptor is split across two providers.

## Reproduction

Quorum's `ManifoldAcquisitionOriginator` (in `marquee-apps/quorum-sense/crates/quorum-app/src/lib.rs`) does the following at call time:

```rust
let selected = select_chat_backend(&originator_selection_config())?;
let response = selected.backend.chat(req).await?;
// Provenance:
let provenance = SignalProvenance {
    vendor: selected.provider().to_string(),
    model: response.model.unwrap_or_else(|| selected.model().to_string()),
    // ...
};
```

With multiple vendor keys present in env (Anthropic + Gemini + OpenAI), the smoke produced:

```
PASS: Manifold originated 5 questions via vendor=anthropic model=gemini-2.5-flash
```

`response.model` (returned by the actual backend that handled the request) reported `gemini-2.5-flash`. `selected.provider()` reported `anthropic`. The `.backend` handle that actually serviced the request was a Gemini backend, but the descriptor says Anthropic.

## Root cause

`select_chat_backend_with_secret_provider` (`src/llm/selection.rs:67-80`) runs two independent selection steps:

```rust
pub fn select_chat_backend_with_secret_provider(
    config: &ChatBackendSelectionConfig,
    secrets: &dyn SecretProvider,
) -> Result<SelectedChatBackend, LlmError> {
    let (model_registry, registry_config) = model_registry_for_config(config, secrets)?;

    // Step 1: pick a top candidate from the model registry by fitness.
    let selection =
        model_registry.select_with_details(&registry_config.criteria.to_agent_requirements())?;

    // Step 2: build a chat-backend registry from the candidate list,
    //         then re-select via converge_provider's ChatBackendRegistry::select.
    let chat_registry = chat_backend_registry_from_candidates(&selection.candidates, secrets)?;
    let resolved = chat_registry.select(&registry_config)?;

    Ok(SelectedChatBackend {
        backend: resolved.backend(),  // ← from step 2
        selection,                     // ← from step 1
    })
}
```

`selection.selected` (used by `.provider()` and `.model()`) is the top pick from step 1's model-registry fitness function.

`resolved.backend()` is the top pick from step 2's `ChatBackendRegistry::select`, which uses a different ranking (`ChatBackendDescriptor` capabilities filtered by `ChatBackendSelectionConfig` criteria).

When step 1 and step 2 disagree (different tie-breaking, different filter logic for capability flags vs `AgentRequirements`), the returned `SelectedChatBackend` carries a descriptor from step 1 and a backend from step 2.

## Why this matters

1. **Provenance is wrong.** Apps that attach `SelectedChatBackend.provider()` and `.model()` to fact provenance (Quorum, Atlas, atelier tutorials) record vendor/model pairs that the actual round-trip didn't use. Audit trails become misleading; replay against the recorded model name fails.
2. **Routing is opaque.** Operators reading the selection trace see step 1's pick and assume that's what was called. The actual backend is invisible without inspecting `response.model` after the fact.
3. **Cost / latency / quality numbers come from the wrong model.** The descriptor's `cost_class`, `typical_latency_ms`, `quality` belong to step 1's pick; the call's actual cost/latency/quality were determined by step 2's pick.

## Proposed fix

Three options, in increasing order of behavior change:

### Option A — make the two paths return the same pick (recommended)

`ChatBackendRegistry::select` should be deterministic on the top candidate returned by step 1, OR step 1 should be skipped and selection should happen entirely through `ChatBackendRegistry::select`.

The cleanest expression: step 1 produces the ranked candidate list; step 2 takes that list and instantiates the **top** candidate as the backend (no re-ranking). Skip `chat_registry.select(&registry_config)` and instead instantiate the top candidate directly:

```rust
let selection =
    model_registry.select_with_details(&registry_config.criteria.to_agent_requirements())?;
let (top_metadata, _fitness) = selection
    .candidates
    .first()
    .ok_or(LlmError::NoCandidate)?;
let registered = registered_chat_backend_for_model(top_metadata, secrets)?;
let backend = registered.backend();
Ok(SelectedChatBackend { backend, selection })
```

This guarantees `selection.selected` matches the actual `.backend`.

### Option B — derive `.provider()` / `.model()` from the actual resolved backend

Keep both selection paths, but report from `resolved.backend()`'s descriptor:

```rust
pub fn provider(&self) -> &str {
    self.resolved_descriptor.provider().as_str()
}

pub fn model(&self) -> &str {
    self.resolved_descriptor.model().as_str()
}
```

This requires storing the resolved descriptor on `SelectedChatBackend` alongside `selection`. Truthful but loses the step-1 fitness trace.

### Option C — accept the divergence, rename methods to clarify intent

Rename `provider()` / `model()` to `intended_provider()` / `intended_model()`, and add separate `resolved_provider()` / `resolved_model()` that report what actually was called. Callers explicitly pick which one to attach to provenance.

This is the most truthful but most disruptive: every caller has to update.

## Recommendation

**Option A.** The two-path selection is internal implementation detail that callers cannot reason about. Collapsing to a single deterministic pick (top model-registry candidate, instantiated directly) keeps `selection.selected.provider/model` and `backend` aligned by construction. The cost is losing the `ChatBackendRegistry::select` step's ability to apply additional `ChatBackendSelectionConfig`-level filtering — but if that filtering matters, it should happen at step 1 (in `to_agent_requirements`), not as a second re-ranking pass.

## Audit hook

Consider adding a debug assertion in `SelectedChatBackend::new` (or wherever the struct is constructed) that the descriptor provider/model match what the backend's own descriptor reports. This catches future regressions where the two paths drift again:

```rust
debug_assert_eq!(
    self.selection.selected.provider,
    resolved_descriptor.provider().as_str(),
    "SelectedChatBackend descriptor/backend provider drift — see kb/Architecture/SelectedChatBackend Provider Drift Bug.md"
);
```

## Test plan

A unit test in `manifold-adapters` that registers two backends with different fitness scores and asserts the returned `SelectedChatBackend`'s `.provider()` matches the actual backend's own descriptor (queryable via a debug accessor or via an end-to-end chat call that echoes the provider).

## See also

- `marquee-apps/quorum-sense/crates/quorum-app/src/lib.rs` — first user of the affected `.provider()` / `.model()` accessors via `ManifoldAcquisitionOriginator`.
- `marquee-apps/atlas-integration/Justfile` — `spike-1-smoke` recipe that first surfaced the drift.
- `marquee-apps/atlas-integration/kb/Architecture/Upstream Types Audit.md` — the broader audit that flagged Manifold's selection surface as canonical for typed Intent.
