---
source: llm
---
# Baseten Integration — Design Note (Future Work)

> Status: not implemented. Decisions captured here from the 2026-05-23
> design conversation so the future-builder doesn't re-derive them.

## Why Baseten matters here

Baseten is **infrastructure**, not a provider of intelligence. It hosts
models you've chosen — stock open-weight (Qwen, DeepSeek, Llama) or your
own fine-tunes (e.g., a `reflective-strategist-v1` tuned on governance
data). The Manifold abstraction should reflect that: Baseten is *where*
a model runs, not *what* the model is.

## Two deployment shapes

1. **Model Library deployment.** Baseten provides a preconfigured
   endpoint for a stock open model. The URL pattern is
   `https://app.baseten.co/v1/...` and the response is OpenAI-compatible.
   The model name in the request matches the library entry
   (`qwen3-72b`, `deepseek-v3`, etc.).

2. **Custom deployment.** Your Truss container or fine-tune. URL is
   deployment-specific — either
   `https://model-{deployment_id}.api.baseten.co/predict` (raw) or
   `https://app.baseten.co/v1/{deployment_id}/v1/chat/completions`
   (if you used Baseten's vLLM template, which keeps it OpenAI-compat).
   Model name in the request is whatever you set during deployment.

In both cases the wire format is OpenAI-compatible, so the backend
itself looks like the DeepSeek/Kimi/Qwen ones we already have.

## The provider-naming decision

For **stock** Baseten deployments, register them under
`provider = "baseten"`. For **Reflective-branded fine-tunes**, register
them under `provider = "reflective"`. Same `BasetenBackend` serves both;
only the `ModelMetadata` registration changes.

Why split: the selector ranks by provider identity. Calling our own
fine-tune `provider = "reflective"` lets selection rules express "prefer
reflective models for governance work" without needing to know which
infrastructure hosts them. If we later move the model to a different
host, the provider identity stays the same.

## Pricing — the awkward part

Baseten bills per-second of GPU time, not per-token. To fit our
`pricing_*_usd_per_million` model:

```
$/M_input  ≈ ($/GPU-hr ÷ 3600) × seconds_per_input_token
$/M_output ≈ ($/GPU-hr ÷ 3600) × seconds_per_output_token
```

For a 70B model on H100 (~$4-6/hr) at ~30-50 tok/s output, expect
~$0.05-0.10 per million output tokens. Rough but useful at routing
granularity.

**Recommendation:** ship with **static estimates**
(`with_pricing(0.05, 0.10)` based on measured throughput). Refine to
**observed cost** (instrument the backend, surface real spend via
Baseten's billing API) only if routing decisions are visibly wrong.

The `openrouter_id` field stays `None` for Baseten-hosted fine-tunes —
they're not in OpenRouter's catalog, so Phase B's live-pricing
auto-update doesn't touch them. Manual maintenance of these numbers.

## The "prefer my fine-tune for my domain" question

If a Reflective fine-tune is meant to be authoritative for governance
tasks, the selector should pick it over Claude or GPT-4o even when those
are cheaper. The mechanism today:

- Set `quality: 0.93+` on the fine-tune (your judgment, encoded as a
  number).
- Set `with_business_acumen(true)` or `with_reasoning(true)` per its
  actual capabilities.
- The fuzzy `cheap+high-quality → strong preference` rule naturally
  favors a high-quality cheap (self-hosted) model.

This is a coarse mechanism. The **better** long-term answer is
per-domain quality scoring:

```rust
domain_quality: HashMap<String, f64>,  // "governance" -> 0.98, "code" -> 0.70
```

with `AgentRequirements` carrying a `domain: Option<String>` that the
fuzzy logic reads. Deferred. Add only when there's a concrete second
domain (likely once two or more Reflective-branded models exist).

## Implementation sketch (~30 min when picked up)

1. **`crates/manifold/src/llm/baseten.rs`** — full standalone backend
   following the OpenAI-compat template (mirror DeepSeek almost exactly).
   - Reads `BASETEN_API_KEY` from env.
   - Reads `BASETEN_BASE_URL` from env (required — every account differs).
   - Optionally reads `BASETEN_DEPLOYMENT_ID` if routing per-deployment.
   - Same `try_new` / `from_secret_provider` / `with_model` /
     `with_base_url` shape as the other backends.

2. **Cargo feature** `baseten = ["_http", "_chat"]` in
   `crates/manifold/Cargo.toml`.

3. **Wire-up:**
   - `crates/manifold/src/llm/mod.rs` — add the module + export.
   - `crates/manifold/src/llm/selection.rs` — add backend instantiation
     arm, `is_chat_provider_available` arm, `normalize_provider_name`
     alias, and add to `chat_provider_registry` supported list.

4. **Seed entry** in `ModelSelector::default()`:
   ```rust
   #[cfg(feature = "baseten")]
   ModelMetadata::new("baseten", "meta-llama/llama-3.1-70b-instruct",
                      CostClass::VeryLow, 2500, 0.84)
       .with_tool_use(true)
       .with_code(true)
       .with_context_tokens(128_000)
       .with_pricing(0.05, 0.10),
   ```
   Validates the wiring without requiring a Reflective fine-tune.

5. **Live tests** in `crates/manifold/tests/live_llm_endpoints.rs` —
   three tests following the existing pattern
   (happy-path-multiturn, invalid-key-auth-denied, invalid-model).

6. **Future fine-tune additions** — register under
   `provider = "reflective"` with one `ModelMetadata` entry per
   deployment. No backend changes needed.

## Open questions to resolve at implementation time

- Does the Baseten URL pattern for OpenAI-compat differ enough across
  accounts that a single `BASETEN_BASE_URL` env var is awkward? Might
  need per-deployment URL config.
- Does Baseten emit any deployment-cost telemetry in response headers
  that we should capture into `ChatResponse.metadata`?
- For Reflective-branded models, should we add a capability flag like
  `is_native_to_reflective: bool` so the selector can short-circuit
  to them for governance workflows? Or rely on `with_business_acumen`?
