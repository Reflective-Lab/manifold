# Changelog

All notable changes to manifold will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning before 1.0 with the usual pre-1.0 compatibility caveats.

## [Unreleased]

### Changed

- Cargo package renamed from `manifold` to `converge-manifold-adapters`; Rust
  library name remains `manifold`.

## [0.1.0] - 2026-05-05

### Added

- Workspace scaffold for generic Converge adapters.
- Object-store adapter builders for local, S3, and GCS backends.
- SurrealDB and LanceDB experience-store adapters.
- LanceDB vector recall adapter.
- Standard GitHub community health files.
- `AGENTS.md` and `Justfile` workflow entrypoints.
