---
source: llm
---
# KB Mutation Log

Append an entry on every kb/ change. Format mirrors converge.

| Date | File | Change | Author |
|------|------|--------|--------|
| 2026-05-06 | kb/Planning/MILESTONES.md | Record v1.0.0 release line targeting Converge 3.8.1 | mixed |
| 2026-05-06 | crates/manifold/{src/brave.rs,src/tavily.rs,src/search.rs,src/fetch.rs,src/feed.rs,src/embedding,src/reranker,src/vector,src/tools,tests/live_search_endpoints.rs}; Cargo.toml; README.md; kb/Planning/MILESTONES.md; kb/Architecture/Surface.md | Completed remaining Manifold adapter migration: search/fetch/feed, embedding/reranking/vector, and OpenAPI/GraphQL tool adapters moved from Converge staging | codex |
| 2026-05-06 | crates/manifold/{config,examples,tests,src/registry_loader.rs}; README.md; kb/Planning/MILESTONES.md; kb/Architecture/Surface.md | Forced LLM migration completed: model registry, live chat examples, and live LLM probes moved to Manifold; Converge staging no longer defines LLM adapters | codex |
| 2026-05-06 | README.md; Cargo.toml; crates/manifold/{Cargo.toml,src/lib.rs,src/llm/*,src/secret.rs,src/model_selection.rs}; kb/Architecture/Surface.md; kb/Planning/MILESTONES.md | First physical LLM provider migration from Converge staging: chat adapters, secret abstraction, and model selection now compile in Manifold behind `llm-all` against Converge 3.8.1 local patch | codex |
| 2026-05-23 | kb/Architecture/Baseten Integration.md; kb/Planning/MILESTONES.md | Captured Baseten backend design as future activity: two deployment shapes (Library vs custom), provider-naming split (baseten for stock, reflective for fine-tunes), per-second-GPU pricing approximation, deferred per-domain quality scoring, ~30 min implementation sketch | llm |
| 2026-05-23 | crates/manifold/src/llm/streaming.rs; crates/manifold/src/llm/openrouter.rs; crates/manifold/src/llm/mod.rs; crates/manifold/Cargo.toml; Cargo.toml; crates/manifold/tests/live_llm_endpoints.rs; crates/manifold/src/llm/perplexity.rs | StreamingChatBackend trait + ChatEvent / ChatStream types; OpenRouter SSE streaming impl (hand-rolled unfold-based parser, no async-stream dep); live streaming test passes (delta+finish+usage events); relaxed Perplexity JSON test (response_format omitted, rely on system prompt); added reqwest `stream` feature | llm |
| 2026-05-23 | kb/Architecture/Long-Running Agent Backend.md; kb/Planning/MILESTONES.md | Captured AgentBackend design as future activity: mirrors converge_pack::gate::SolverReport / ReplayEnvelope field shapes so artifacts ingest cleanly into convergence loops; first impl candidate is PerplexityDeepResearchBackend; OpenAI Assistants and Reflective fine-tunes follow; design discusses what we reuse vs mirror | llm |
| YYYY-MM-DD | _path_ | _summary_ | human/llm/mixed |
