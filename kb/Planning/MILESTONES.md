---
source: mixed
---
# Milestones

> See `~/dev/work/EPIC.md` for the coarse-grained outcomes these milestones advance.

## Current: v1.0 — Converge 3.8.1 Adapter Foundation

**Target:** 2026-05 | **Tracks:** Converge 3.8.1

- [x] Object-storage, experience-store, and vector adapters live in Manifold.
- [x] LLM adapter family moved from Converge staging and compiling behind
      `llm-all`.
- [x] Remove LLM adapter definitions, model catalog, live chat examples, and
      live LLM endpoint probes from Converge staging.
- [ ] Move search, fetch, feed, embedding, and tool adapters.
- [ ] Add downstream proof that products register Manifold handles through
      `converge_provider::ChatBackendRegistry`.
- [ ] Remove local Converge `[patch.crates-io]` after Converge 3.8.1 is
      published.
- [ ] First clean `just release-check` run after all migrated families land.
- [ ] Tag v1.0.0.
