---
source: llm
---
# KB Mutation Log

Append an entry on every kb/ change. Format mirrors converge.

| Date | File | Change | Author |
|------|------|--------|--------|
| 2026-05-06 | crates/manifold/{config,examples,tests,src/registry_loader.rs}; README.md; kb/Planning/MILESTONES.md; kb/Architecture/Surface.md | Forced LLM migration completed: model registry, live chat examples, and live LLM probes moved to Manifold; Converge staging no longer defines LLM adapters | codex |
| 2026-05-06 | README.md; Cargo.toml; crates/manifold/{Cargo.toml,src/lib.rs,src/llm/*,src/secret.rs,src/model_selection.rs}; kb/Architecture/Surface.md; kb/Planning/MILESTONES.md | First physical LLM provider migration from Converge staging: chat adapters, secret abstraction, and model selection now compile in Manifold behind `llm-all` against Converge 3.8.1 local patch | codex |
| YYYY-MM-DD | _path_ | _summary_ | human/llm/mixed |
