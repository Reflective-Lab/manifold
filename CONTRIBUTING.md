# Contributing to manifold

`manifold` is a Converge extension for generic adapters where the vendor is
interchangeable behind a capability or storage contract.

## Development

```sh
just check
just check-all
just test
just lint
```

## Boundary

Use Manifold for generic adapters: object storage, vector recall,
experience-store backends, web fetch, search, feed retrieval, LLM chat,
embedding generation, and similar swappable capabilities.

Use `../embassy` when the external party identity is part of the semantic
contract.

## Adding an Adapter

When adding an adapter:

1. Start from an existing Converge contract or an extension-local contract that
   has clear reuse.
2. Keep concrete SDK details out of public semantic types.
3. Put heavy dependencies behind feature flags.
4. Add deterministic tests for config parsing and capability behavior.
5. Document operator-owned credentials and runtime requirements.

## No `unsafe`

The workspace forbids `unsafe`.

## License

By contributing, you agree your contributions are licensed under MIT.
