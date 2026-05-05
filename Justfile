# manifold development commands
# Install: brew install just  |  cargo install just
# Usage:   just --list

set dotenv-load := true

# Show available recipes
default:
    @just --list

# Build the workspace
build:
    cargo build --workspace

# Build release artifacts
build-release:
    cargo build --workspace --release

# Check all workspace targets
check:
    cargo check --workspace --all-targets

# Check all storage adapter features
check-all:
    cargo check --workspace --all-targets --features manifold/all-storage

# Run tests
test:
    cargo test --workspace --all-targets

# Run tests with all features
test-all:
    cargo test --workspace --all-targets --all-features

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Format code
fmt:
    cargo fmt --all

# Run clippy
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run clippy with all features
clippy-all:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Formatting plus clippy
lint: fmt-check clippy

# Generate docs
doc:
    cargo doc --workspace --no-deps

# Generate and open docs
doc-open:
    cargo doc --workspace --no-deps --open

# Local release hygiene
security-audit:
    cargo audit --deny warnings
    cargo deny check

# Session opener
focus: status check test

# Git status and recent commits
status:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d .git ]; then
      git status --short --branch
      git log --oneline -5
    else
      echo "no local .git"
    fi

# Remove build artifacts
clean:
    cargo clean
