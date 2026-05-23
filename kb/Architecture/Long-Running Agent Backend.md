---
source: llm
---
# Long-Running Agent Backend — Design Note (Future Work)

> Status: not implemented. Captured from the 2026-05-23 design conversation.
> Companion to `Baseten Integration.md` — together they cover the two
> next-axis backend types (custom-hosted inference, long-running agents).

## The question

Today Manifold has two backend traits: `ChatBackend` (synchronous chat
completion, seconds) and `WebSearchBackend` (synchronous search). Three
realistic third-category candidates have emerged:

- **Perplexity Sonar Deep Research** — same wire shape as chat, but runs
  for 30s–5min, performs multi-step internal search-and-synthesize, and
  returns a structured research report with citations.
- **OpenAI Assistants API** — stateful threads, server-managed `runs`
  with polling, attached files via `file_search`, optional
  `code_interpreter`.
- **Hypothetical Reflective workflow agent** — a fine-tune deployed to
  Baseten that orchestrates multi-step planning + tool use server-side,
  returning a final deliverable.

All three produce what we informally call a **"Ready" (baked) result**:
a finished, structured artifact arrived at via multiple internal steps,
intended to be consumed authoritatively. The question this doc answers:
how should that map to Converge's existing "ready" semantics, and what
types can we reuse?

## What Converge already calls "ready"

Converge's convergence loop produces a typed result through the
**promotion gate**. The relevant published types live in
`converge_pack::gate`:

| Type | Role |
|---|---|
| `ProblemSpec` | The authoritative request — `problem_id`, `tenant_scope`, `objective`, `constraints`, typed inputs, budgets, determinism, audit envelope. |
| `SolverReport` | The authoritative result — solver version, objective value, replay envelope. |
| `ReplayEnvelope` | Integrity proof — `input_hash`, `output_hash`, `seed`, `solver_version`. |
| `AuditEnvelope` | Provenance — input hash, correlation IDs, submitter. |
| `GateDecision` | Promote / reject / defer with reasons. |
| `StopReason` | Typed stop reason (one of four "honest exits"). |
| `Diagnostic` | Per-step structured findings. |

The **invariant** that matters here: a `SolverReport` is more than a
chat response. It carries a `ReplayEnvelope` (so the run can be
replayed and verified deterministically), an `AuditEnvelope` (so the
submitter and provenance are known), and a typed `StopReason` (so the
caller knows *why* the run ended, not just *that* it did).

Long-running external agents should produce results that satisfy the
same audit/provenance shape — otherwise their output can't be ingested
back into a convergence loop as evidence without losing the
guarantees Converge promises.

## Proposed surface

A new manifold trait, parallel to `ChatBackend` / `WebSearchBackend`:

```rust
pub trait AgentBackend {
    type SubmitFut<'a>: Future<Output = Result<AgentRunHandle, AgentError>> + Send + 'a
    where Self: 'a;
    type PollFut<'a>: Future<Output = Result<AgentStatus, AgentError>> + Send + 'a
    where Self: 'a;
    type CancelFut<'a>: Future<Output = Result<(), AgentError>> + Send + 'a
    where Self: 'a;

    /// Submit a long-running job. Returns immediately with a handle.
    fn submit(&self, request: AgentRequest) -> Self::SubmitFut<'_>;

    /// Poll for current status. Implementations may cache.
    fn poll(&self, handle: &AgentRunHandle) -> Self::PollFut<'_>;

    /// Cancel a run. May be a no-op for providers that don't support it.
    fn cancel(&self, handle: &AgentRunHandle) -> Self::CancelFut<'_>;
}

pub struct AgentRunHandle {
    pub provider: String,
    pub run_id: String,
    /// Provider-specific opaque metadata (e.g., OpenAI thread_id).
    pub opaque: serde_json::Value,
}

pub enum AgentStatus {
    Pending,
    InProgress {
        elapsed: Duration,
        progress_events: Vec<AgentProgressEvent>,
    },
    Completed { artifact: AgentArtifact },
    Failed { stop_reason: AgentStopReason, message: String },
    Cancelled,
}
```

The interesting design decision is `AgentArtifact` — the "ready baked"
shape:

```rust
pub struct AgentArtifact {
    /// Free-form rendered result. For Perplexity, the synthesized answer.
    /// For Assistants, the final assistant message. For research agents,
    /// the report body.
    pub content: String,

    /// Structured citations / sources used. Same shape as
    /// WebSearchResult so callers that already handle search results
    /// can render these without new code.
    pub citations: Vec<crate::search::WebSearchResult>,

    /// Per-step trace of what the agent did. Use these to surface
    /// progress in UI or to inspect provenance after the fact.
    pub steps: Vec<AgentStep>,

    /// Total tokens used by the agent across all internal steps.
    pub usage: Option<converge_provider::TokenUsage>,

    /// Provider-specific metadata (run id, thread id, cost breakdown).
    pub metadata: HashMap<String, String>,

    /// Replay envelope MIRRORING converge_pack::gate::ReplayEnvelope.
    /// We don't take a hard dep on converge_pack here (manifold's
    /// chat backends are foundation-light); instead we expose the
    /// same field shape so callers wrapping AgentArtifact into a
    /// SolverReport can do so cheaply.
    pub replay: AgentReplay,
}

pub struct AgentReplay {
    /// Hash of the AgentRequest (canonicalized).
    pub input_hash: String,
    /// Hash of the artifact content + citations + steps.
    pub output_hash: String,
    /// Seed if the provider exposes one (Perplexity, Assistants don't;
    /// only deterministic agents do).
    pub seed: Option<u64>,
    /// Provider identifier + version. For Perplexity:
    /// `"perplexity-sonar-deep-research"`. For Assistants:
    /// `"openai-assistants/asst_..."`.
    pub agent_version: String,
}
```

The deliberate shape: `AgentArtifact` carries enough provenance metadata
that wrapping one into a `converge_pack::gate::SolverReport` is a
mechanical lift, done by the caller (typically the converge runtime
when it ingests external evidence). We don't make manifold depend on
converge_pack just for this — the field shapes are *intentionally
mirrorable*, not *literally the same type*.

## What each provider teaches us

### OpenAI Assistants

What's well-modelled:
- Stateful **threads** (`thread_id`) that persist across runs.
- Server-side **runs** with status polling (`queued`, `in_progress`,
  `requires_action`, `completed`, etc.).
- Built-in tools: `file_search`, `code_interpreter`.
- **Tool-use-with-pause**: when the run needs a client tool call, it
  pauses with `requires_action` until the client submits results.

Implications for our design:
- `AgentRunHandle.opaque` should carry both `thread_id` and `run_id`.
- `AgentStatus::InProgress` may need a `RequiresAction` sub-variant that
  surfaces pending tool calls and accepts results. *Or* we keep this
  out of `AgentBackend` and handle tool-roundtrip in a provider-specific
  layer — Assistants' tool flow is meaningfully different from
  `ChatBackend`'s synchronous tool-call pattern.
- File attachments are an open API question. Probably belongs in
  `AgentRequest` as `attached_files: Vec<FileHandle>`.

### Perplexity Sonar Deep Research

What's well-modelled:
- Submit synchronous: same chat completion endpoint, different model
  (`sonar-deep-research`).
- Long-running: 30s–5min, no native poll/progress — caller blocks.
- Output: synthesized research report + citation URLs.

Implications:
- `AgentBackend::submit` for Perplexity is just a long async HTTP call;
  there is no poll API to query. Implementations can fake the poll
  interface by tracking the future internally and returning
  `InProgress { progress_events: [] }` until the underlying future
  resolves, then `Completed`.
- Their `citations` array maps cleanly to `Vec<WebSearchResult>`.
- No `seed`, no replay determinism — `AgentReplay.seed = None`,
  `agent_version` carries the model name.

### Hypothetical Reflective workflow agent on Baseten

What's well-modelled:
- We choose the wire shape — could be SSE streaming, could be polling,
  could be a job queue with a webhook.
- The artifact shape matches our governance use case directly: a
  structured deliverable with cited evidence.

Implications:
- `AgentBackend` is the right surface for our own deployments too —
  resist the urge to roll a custom trait per Reflective fine-tune.
- The fine-tune's output should *natively* emit the
  `AgentArtifact`-equivalent shape, so the wire format and the
  in-process type align without translation.

## What we can reuse, what we can't

| Existing type | Reusable for agents? |
|---|---|
| `converge_provider::ChatRequest` | **No**, too synchronous. Need `AgentRequest` with attached files / context / persistent thread reference. |
| `converge_provider::ChatResponse` | **No**, no notion of multi-step trace, no citations, no replay envelope. |
| `converge_provider::TokenUsage` | **Yes**, agents have aggregate usage too. |
| `manifold::WebSearchResult` | **Yes**, perfect for `AgentArtifact.citations`. |
| `manifold::ChatEvent` (new, streaming) | **Yes**, repurposable as `AgentProgressEvent` for providers that stream progress (Assistants when run is in_progress). |
| `converge_pack::gate::ReplayEnvelope` | **Shape yes, type no.** We mirror its fields in `AgentReplay` to keep manifold's dep tree small. The runtime layer that ingests agent artifacts as Converge evidence does the conversion. |
| `converge_pack::gate::SolverReport` | Same as above — the caller wraps `AgentArtifact` into a `SolverReport` at ingestion time. |
| `converge_pack::gate::StopReason` | **Shape yes, type no.** Manifold's `AgentStopReason` mirrors the same four categories ("completed", "deadline", "user cancel", "irrecoverable error") that Converge's honest-exits taxonomy uses. |

## Implementation order when we pick this up

1. Define `AgentBackend` trait, `AgentRequest`, `AgentArtifact`,
   `AgentStatus`, `AgentRunHandle`, `AgentReplay` in
   `crates/manifold/src/llm/agent.rs` (or `crates/manifold/src/agent.rs`
   if it grows beyond LLMs). Gate on a new `agent` feature.
2. Implement `PerplexityDeepResearchBackend` as the first concrete
   backend. The wire is just our existing Perplexity chat endpoint
   with `model = "sonar-deep-research"`. It's the simplest agent —
   submit-and-wait — so it validates the trait's shape without the
   complications of stateful threads.
3. Write a live integration test: submit a research query, wait for
   completion (with a sensible timeout), verify the artifact has
   non-empty content + at least one citation.
4. Once that's working and stable, implement
   `OpenAiAssistantsBackend` — this exercises the polling +
   pause-for-tool-result paths and will likely surface the need for
   a `RequiresAction` sub-state on `AgentStatus`.
5. Document for downstream consumers how to wrap `AgentArtifact` into
   `converge_pack::gate::SolverReport` when ingesting agent results
   into a convergence loop.

## Open questions to resolve at implementation time

- Do we want `submit` to be synchronous (block until done) for
  submit-and-wait providers like Perplexity, or always async with poll?
  Async-with-poll is more uniform but adds machinery for simple cases.
- Should `AgentBackend` extend `ChatBackend` (so an agent is also a
  chat backend with a chat() that returns immediately with a placeholder
  message)? Probably not — different semantics deserve different surface.
- How should we handle the `RequiresAction` flow in Assistants? Two
  options: (a) bake it into `AgentStatus` (every backend now sees a
  pause-for-tool variant), or (b) keep it Assistants-specific via a
  trait extension `AssistantsBackend: AgentBackend` with extra methods.
  Lean (b).
- For the Reflective fine-tune case, do we want a *streaming* agent
  interface (progress events as they happen) or just poll-based?
  Streaming is better UX for long runs but more code.
- Where does the conversion `AgentArtifact -> SolverReport` live? Not
  in manifold (we don't depend on converge_pack at this layer). Likely
  in a small bridge crate that the runtime owns, since the runtime is
  what calls promotion gates.

## See also

- `kb/Architecture/Baseten Integration.md` — sibling future-work doc
  for the infrastructure layer that would host a Reflective agent
  fine-tune.
- `~/dev/reflective/stack/bedrock-platform/converge/kb/Architecture/Core Ideas.md`
  — the "promotion gate is the only place a proposal becomes truth"
  principle that this doc's mirrored-shape strategy preserves.
- `~/dev/reflective/stack/bedrock-platform/converge/kb/Architecture/System Overview.md`
  — convergence loop / promotion gate / typed stop reasons.
