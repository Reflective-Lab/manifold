---
tags: [architecture, surface]
source: mixed
---
# Surface

`manifold` exposes one canonical published crate (`manifold`)
plus optional adapter crates with adapter-qualified names.

## Public surface

- `manifold` — _one-line description of the public crate_

## Contract dependencies

- `converge-pack` — `Pack`, `ProposedPlan`, `ProblemSpec`
- `converge-model` — semantic types
- `converge-provider-api` — capability identity (when applicable)

## Forbidden imports

Per [Extension Release Checklist §1](https://github.com/Reflective-Lab/converge/blob/main/kb/Standards/Extension%20Release%20Checklist.md):

- No imports of `converge-core` internals.
- No imports of foundation `runtime`, `provider`, or transport crates.
- No re-exports of foundation types except those promised stable.
