# manifold

[![CI](https://github.com/Reflective-Lab/manifold/actions/workflows/ci.yml/badge.svg)](https://github.com/Reflective-Lab/manifold/actions/workflows/ci.yml)
[![Security](https://github.com/Reflective-Lab/manifold/actions/workflows/security.yml/badge.svg)](https://github.com/Reflective-Lab/manifold/actions/workflows/security.yml)
[![dependency status](https://deps.rs/repo/github/Reflective-Lab/manifold/status.svg)](https://deps.rs/repo/github/Reflective-Lab/manifold)
![MSRV](https://img.shields.io/badge/MSRV-1.94.0-blue)
<img alt="gitleaks badge" src="https://img.shields.io/badge/protected%20by-gitleaks-blue">
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Generic adapter implementations for Converge contracts.

`manifold` is a Converge extension for provider, storage, vector, experience,
fetch, feed, search, LLM, embedding, and tool adapters where the concrete
vendor should be hidden behind an interchangeable capability.

## Why It Exists

Converge owns contracts and authority. Manifold owns concrete adapter code.
This keeps the foundation small while still giving deployments reusable
implementations for common operational backends.

## What Manifold Owns

- Object-store adapter builders.
- Experience-store adapters.
- Vector recall adapters.
- Future generic provider and tool adapters.

## Boundary

| Layer | Responsibility |
|---|---|
| Converge | Storage, vector, provider, and experience contracts. |
| Manifold | Concrete generic adapters for those contracts. |
| Embassy | Source-specific ports where foreign-system identity is part of the API. |
| Products | Runtime assembly, credentials, tenancy, and provider selection. |

Use Manifold when two providers can plausibly be swapped behind the same
contract. Use `../embassy` when the API must name the external source.

## Repository Layout

```text
crates/manifold/
  src/object_storage/  Local, S3, and GCS object-store builders
  src/experience/      SurrealDB and LanceDB experience stores
  src/vector/          LanceDB vector recall adapter
  src/lib.rs           Public adapter surface and Converge re-exports
```

## Current Adapter Families

| Feature | Adapter |
|---|---|
| `object-local` | Local filesystem object store |
| `object-s3` | S3-compatible object store |
| `object-gcs` | Google Cloud Storage object store |
| `experience-surrealdb` | SurrealDB experience store |
| `experience-lancedb` | LanceDB vector-indexed experience store |
| `vector-lancedb` | LanceDB vector recall |

## Feature Flags

- Default: `object-local`.
- `object-all`: local, S3, and GCS object storage.
- `all-storage`: all current object, experience, and vector adapters.

## Usage

```rust
use manifold::object_storage::build_store;
use manifold::{StorageConfig, StorageUri};

let config = StorageConfig {
    uri: StorageUri::Local("./objects".into()),
    prefix: None,
    public: false,
    endpoint: None,
    region: None,
};

let store = build_store(&config)?;
```

## Development

```sh
just check
just check-all
just test
just lint
just doc
```

Converge platform dependencies resolve from crates.io.

## Project Files

- [AGENTS.md](AGENTS.md) - agent entrypoint and boundary rules.
- [CHANGELOG.md](CHANGELOG.md) - release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md) - contribution guide.
- [SECURITY.md](SECURITY.md) - vulnerability reporting and operator notes.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - community expectations.

## Status

Scaffolded on 2026-05-05 as the generic adapter home for extracted Converge
extension functionality.

## License

MIT - see [LICENSE](LICENSE).
