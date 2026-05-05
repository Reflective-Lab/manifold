# manifold Agent Guide

This is the canonical agent entrypoint for `manifold`.

`manifold` is a Converge extension for generic adapters: storage, vector,
experience, provider, fetch, feed, search, embedding, LLM, and tool backends
where the vendor should be swappable behind a capability.

## Start Here

1. Read `README.md`.
2. Read `/Users/kpernyer/dev/extensions/kb/Modules/Manifold.md`.
3. Read `/Users/kpernyer/dev/extensions/kb/Architecture/Port Provider Boundary.md`.
4. Use `just --list` for local commands.

## Commands

```bash
just check
just check-all
just test
just lint
just doc
```

## Boundaries

- Use Manifold for interchangeable generic adapters.
- Use Embassy when the source identity is part of the API.
- Products own runtime assembly, credentials, and provider selection.

## Rules

- Preserve `unsafe_code = "forbid"`.
- Adapter builders should implement Converge contracts without defining new
  truth semantics.
- Prefer feature flags for heavy backend dependencies.
- Update `README.md`, `CHANGELOG.md`, and the extensions KB when adapter
  families are added.
