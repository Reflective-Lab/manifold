// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Refresh the OpenRouter model catalog cache.
//!
//! Fetches `https://openrouter.ai/api/v1/models` and writes the result to
//! `~/.cache/manifold/openrouter-catalog.json`. Used to keep model pricing
//! and capability metadata up to date.
//!
//! Run with:
//!   cargo run --example refresh_catalog --features _http

use std::process::ExitCode;

use manifold::model_catalog::ModelCatalog;

fn main() -> ExitCode {
    let path = match ModelCatalog::default_cache_path() {
        Some(p) => p,
        None => {
            eprintln!("error: HOME env var not set, cannot determine cache path");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Fetching OpenRouter model catalog...");
    let catalog = match ModelCatalog::refresh_from_network() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to fetch catalog: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = catalog.save(&path) {
        eprintln!("error: failed to save catalog: {e}");
        return ExitCode::FAILURE;
    }

    let prompt_priced = catalog
        .entries
        .values()
        .filter(|e| e.pricing.prompt > 0.0)
        .count();
    let with_tools = catalog.entries.values().filter(|e| e.supports_tools()).count();
    let with_vision = catalog.entries.values().filter(|e| e.supports_vision()).count();

    println!("Saved {} models to {}", catalog.entries.len(), path.display());
    println!("  {prompt_priced} with prompt pricing, {with_tools} with tool use, {with_vision} with vision");

    ExitCode::SUCCESS
}
