---
tags: [architecture, surface]
source: mixed
---
# Surface

`manifold` exposes one canonical crate (`manifold`) plus feature-gated adapter
families with adapter-qualified type names.

## Public surface

- `manifold` — generic adapter implementations for Converge contracts.
- `llm` — chat adapters such as `AnthropicBackend`, `OpenAiBackend`,
  `GeminiBackend`, `MistralBackend`, `OpenRouterBackend`, `KongBackend`,
  `StaikBackend`, `ArceeBackend`, `WriterBackend`, and `MinMaxBackend`.
- `registry_loader` — YAML model registry loader and schema for the LLM model
  catalog.
- `object_storage` — local, S3, and GCS object-store builders.
- `experience` — SurrealDB and LanceDB experience stores.
- `vector` — LanceDB vector recall adapter.

## Contract dependencies

- `converge-provider` — chat capability traits, selection DTOs, and
  host-supplied registry contract.
- `converge-storage` — object storage contracts.
- `converge-experience` — experience storage contracts.
- `converge-core` — current storage/experience support types during the 3.8.1
  extraction line.

During the 3.8.1 migration this repo uses a local `[patch.crates-io]` override
to compile against `~/dev/work/converge`. Remove it after Converge 3.8.1 is
published.

## Forbidden imports

Per [Extension Release Checklist §1](https://github.com/Reflective-Lab/converge/blob/main/kb/Standards/Extension%20Release%20Checklist.md):

- No imports of Converge runtime or transport crates.
- No imports of temporary foundation adapter staging crates.
- No re-exports of foundation types except those promised stable.
