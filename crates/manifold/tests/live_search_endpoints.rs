// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::sync::OnceLock;

#[cfg(feature = "brave")]
use manifold::BraveSearchProvider;
#[cfg(feature = "perplexity")]
use manifold::PerplexitySearchProvider;
#[cfg(feature = "tavily")]
use manifold::TavilySearchProvider;
use manifold::{WebSearchBackend, WebSearchError, WebSearchRequest};

fn load_env() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = dotenvy::dotenv();
    });
}

fn env_is_set(key: &str) -> bool {
    load_env();
    std::env::var_os(key).is_some_and(|value| !value.is_empty())
}

fn require_env(key: &str) -> bool {
    if env_is_set(key) {
        true
    } else {
        eprintln!("skipping live test because {key} is not set");
        false
    }
}

fn assert_search_auth_denied(error: WebSearchError) {
    assert!(
        matches!(error, WebSearchError::Auth(_)),
        "expected auth-style search failure, got {error:?}"
    );
}

#[cfg(feature = "brave")]
#[test]
#[ignore = "requires live API access"]
fn live_brave_search_happy_path() {
    if !require_env("BRAVE_API_KEY") {
        return;
    }

    let response = BraveSearchProvider::from_env()
        .unwrap()
        .search_web(
            &WebSearchRequest::new("Rust borrow checker ownership")
                .with_max_results(3)
                .with_favicon(true),
        )
        .unwrap();

    assert_eq!(response.provider, "brave");
    assert!(!response.results.is_empty());
    assert!(
        response
            .results
            .iter()
            .any(|result| result.url.starts_with("http"))
    );
}

#[cfg(feature = "tavily")]
#[test]
#[ignore = "requires live API access"]
fn live_tavily_search_happy_path() {
    if !require_env("TAVILY_API_KEY") {
        return;
    }

    let response = TavilySearchProvider::from_env()
        .unwrap()
        .search_web(
            &WebSearchRequest::new("Rust async runtime ownership")
                .with_max_results(3)
                .with_answer(true)
                .with_raw_content(true),
        )
        .unwrap();

    assert_eq!(response.provider, "tavily");
    assert!(!response.results.is_empty());
    assert!(
        response
            .results
            .iter()
            .any(|result| result.url.starts_with("http"))
    );
}

#[cfg(feature = "brave")]
#[test]
#[ignore = "requires live API access"]
fn live_brave_invalid_key_is_auth_denied() {
    let error = BraveSearchProvider::new("converge-invalid-brave-key")
        .search_web(&WebSearchRequest::new("Rust borrow checker"))
        .unwrap_err();
    assert_search_auth_denied(error);
}

#[cfg(feature = "tavily")]
#[test]
#[ignore = "requires live API access"]
fn live_tavily_invalid_key_is_auth_denied() {
    let error = TavilySearchProvider::new("converge-invalid-tavily-key")
        .search_web(&WebSearchRequest::new("Rust async runtime"))
        .unwrap_err();
    assert_search_auth_denied(error);
}

// ── Perplexity (chat-with-citations exposed as a search backend) ───────────

#[cfg(feature = "perplexity")]
#[test]
#[ignore = "requires live API access"]
fn live_perplexity_search_happy_path() {
    if !require_env("PERPLEXITY_API_KEY") {
        return;
    }

    let response = PerplexitySearchProvider::from_env()
        .unwrap()
        .search_web(&WebSearchRequest::new("Rust borrow checker"))
        .unwrap();

    assert_eq!(response.provider, "perplexity");
    // Perplexity always produces an LLM answer alongside citations.
    assert!(
        response.answer.is_some(),
        "perplexity response should include an answer"
    );
}

#[cfg(feature = "perplexity")]
#[test]
#[ignore = "requires live API access"]
fn live_perplexity_invalid_key_is_auth_denied() {
    let error = PerplexitySearchProvider::new("converge-invalid-perplexity-key")
        .search_web(&WebSearchRequest::new("Rust ownership"))
        .unwrap_err();
    assert_search_auth_denied(error);
}
