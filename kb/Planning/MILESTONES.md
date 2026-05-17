---
source: mixed
---
# Milestones

> See `~/dev/reflective/stack/bedrock-platform/EPIC.md` for the coarse-grained outcomes these milestones advance.

## Shipped: v1.1.1 — Converge 3.9.1 alignment — 2026-05-17

**Tracks:** Converge 3.9.1

- [x] Bump converge-core / converge-experience / converge-pack /
      converge-provider / converge-storage to 3.9.1.
- [x] First clean `just release-check` run including all five gates.
- [x] Publish to crates.io (v1.1.0 was the prior published version).
- [x] Tag v1.1.1.

## Shipped: v1.1.0 — Adapter family migration — 2026-05-07

- [x] Object-storage, experience-store, and vector adapters live in Manifold.
- [x] LLM adapter family moved from Converge staging and compiling behind `llm-all`.
- [x] Remove LLM adapter definitions, model catalog, live chat examples, and
      live LLM endpoint probes from Converge staging.
- [x] Move search, fetch, feed, embedding, reranking, vector, and OpenAPI/GraphQL
      tool adapters.

## Shipped: v1.0.0 — Adapter Foundation — 2026-05

- [x] Workspace package version is `1.0.0`.
- [x] Tag v1.0.0.

## Open: pull-driven

- [ ] Downstream proof that products register Manifold handles through
      `converge_provider::ChatBackendRegistry`.
