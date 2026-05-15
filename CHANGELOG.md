# Changelog

All notable changes to manifold will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning before 1.0 with the usual pre-1.0 compatibility caveats.

## [Unreleased]

## [1.1.1] - 2026-05-15

### Changed

- Bumped release metadata for the coordinated extension release. No public API
  changes.

## [1.1.0] - 2026-05-07

### Added

- HuggingFace `ObjectStore` (read-only) under `object_storage::huggingface`.
- Generic `HtmlExtractBackend` trait with a `scraper`-backed implementation
  (`extract` module).
- Codex-style LLM adapter wiring under `llm`.

### Changed

- Cargo package renamed from `manifold` to `converge-manifold-adapters`; Rust
  library name remains `manifold`.
- Switched the dotenv dev-dep from the unmaintained `dotenv` crate to
  `dotenvy 0.15`.
- `deny.toml` updated to keep the security gate green under the foundation
  baseline.

### Fixed

- Code reformatted with `cargo fmt`; the `Format` and `Lint` CI jobs are
  now green.

## [0.1.0] - 2026-05-05

### Added

- Workspace scaffold for generic Converge adapters.
- Object-store adapter builders for local, S3, and GCS backends.
- SurrealDB and LanceDB experience-store adapters.
- LanceDB vector recall adapter.
- Standard GitHub community health files.
- `AGENTS.md` and `Justfile` workflow entrypoints.
