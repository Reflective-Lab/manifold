// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Provider contract types for structured observations and call context.
//!
//! These types define the boundary between providers (adapters) and the
//! Converge core engine. Providers produce observations; the engine decides.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

/// Capabilities that providers can offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// LLM text completion
    LlmCompletion,
    /// Text/image embedding generation
    Embedding,
    /// Re-ranking search results
    Reranking,
    /// Vector similarity search
    VectorSearch,
    /// Web search with citations
    WebSearch,
    /// Graph pattern matching
    GraphSearch,
}

/// Data sovereignty region for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Region {
    /// United States
    US,
    /// European Union
    EU,
    /// China
    CN,
    /// Local (on-premise, no network)
    Local,
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::US => write!(f, "US"),
            Self::EU => write!(f, "EU"),
            Self::CN => write!(f, "CN"),
            Self::Local => write!(f, "Local"),
        }
    }
}

/// Metadata about a provider implementation.
///
/// This is static information that describes what a provider offers.
#[derive(Debug, Clone)]
pub struct ProviderMeta {
    /// Provider name (e.g., "anthropic", "openai")
    pub name: &'static str,
    /// Provider version
    pub version: &'static str,
    /// Capabilities this provider offers
    pub capabilities: &'static [Capability],
    /// Vendor identifier
    pub vendor: &'static str,
    /// Region (for data sovereignty)
    pub region: Region,
}

impl ProviderMeta {
    /// Create new provider metadata.
    #[must_use]
    pub const fn new(
        name: &'static str,
        version: &'static str,
        capabilities: &'static [Capability],
        vendor: &'static str,
        region: Region,
    ) -> Self {
        Self {
            name,
            version,
            capabilities,
            vendor,
            region,
        }
    }
}

/// Context passed to every provider call for tracing and budgets.
///
/// This provides correlation IDs, timeouts, and budget constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCallContext {
    /// Root intent ID for correlation (from converge-core)
    pub root_intent_id: Option<String>,
    /// Trace ID for distributed tracing
    pub trace_id: String,
    /// User/org identifier (for auditing)
    pub user_id: Option<String>,
    /// Maximum allowed latency in milliseconds (timeout)
    pub timeout_ms: u64,
    /// Maximum allowed cost in USD
    pub max_cost: Option<f64>,
    /// Maximum tokens (input + output)
    pub max_tokens: Option<u32>,
}

impl Default for ProviderCallContext {
    fn default() -> Self {
        Self {
            root_intent_id: None,
            trace_id: generate_trace_id(),
            user_id: None,
            timeout_ms: 30_000, // 30 seconds
            max_cost: None,
            max_tokens: None,
        }
    }
}

impl ProviderCallContext {
    /// Create a new call context with a specific trace ID.
    pub fn with_trace_id(trace_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            ..Default::default()
        }
    }

    /// Set the root intent ID.
    #[must_use]
    pub fn with_root_intent(mut self, root_intent_id: impl Into<String>) -> Self {
        self.root_intent_id = Some(root_intent_id.into());
        self
    }

    /// Set the user ID.
    #[must_use]
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set the timeout in milliseconds.
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set the maximum cost budget.
    #[must_use]
    pub fn with_max_cost(mut self, max_cost: f64) -> Self {
        self.max_cost = Some(max_cost);
        self
    }

    /// Set the maximum token budget.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// Token usage information from a provider call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of input tokens
    pub input_tokens: u32,
    /// Number of output tokens
    pub output_tokens: u32,
}

impl TokenUsage {
    /// Create new token usage.
    #[must_use]
    pub const fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    /// Total tokens used.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// Structured result from every provider call.
///
/// This is the core type that providers return. It includes:
/// - The actual content
/// - Provenance metadata for tracing
/// - Cost and latency information
///
/// # Type Parameter
///
/// `T` is the content type (typically `String` for LLM responses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderObservation<T> {
    /// Stable reference ID for this observation
    pub observation_id: String,
    /// Canonical hash of the request
    pub request_hash: String,
    /// Provider that produced this observation
    pub vendor: String,
    /// Model used
    pub model: String,
    /// Call latency in milliseconds
    pub latency_ms: u64,
    /// Estimated cost in USD (if known)
    pub cost_estimate: Option<f64>,
    /// Token usage (if applicable)
    pub tokens: Option<TokenUsage>,
    /// The actual content
    pub content: T,
    /// Raw response (optional, size-bounded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

impl<T> ProviderObservation<T> {
    /// Create a new observation.
    pub fn new(
        vendor: impl Into<String>,
        model: impl Into<String>,
        content: T,
        latency_ms: u64,
    ) -> Self {
        let observation_id = generate_observation_id();
        Self {
            observation_id,
            request_hash: String::new(),
            vendor: vendor.into(),
            model: model.into(),
            latency_ms,
            cost_estimate: None,
            tokens: None,
            content,
            raw_response: None,
        }
    }

    /// Set the request hash.
    #[must_use]
    pub fn with_request_hash(mut self, hash: impl Into<String>) -> Self {
        self.request_hash = hash.into();
        self
    }

    /// Set the cost estimate.
    #[must_use]
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost_estimate = Some(cost);
        self
    }

    /// Set the token usage.
    #[must_use]
    pub fn with_tokens(mut self, input: u32, output: u32) -> Self {
        self.tokens = Some(TokenUsage::new(input, output));
        self
    }

    /// Set the raw response (will be truncated if too long).
    #[must_use]
    pub fn with_raw_response(mut self, raw: impl Into<String>) -> Self {
        const MAX_RAW_SIZE: usize = 10_000;
        let raw = raw.into();
        if raw.len() > MAX_RAW_SIZE {
            self.raw_response = Some(format!("{}...[truncated]", &raw[..MAX_RAW_SIZE]));
        } else {
            self.raw_response = Some(raw);
        }
        self
    }

    /// Generate provenance string for Facts.
    ///
    /// This string can be attached to `ProposedFact` instances to trace
    /// where the data came from.
    pub fn provenance(&self) -> String {
        format!("{}:{}:{}", self.vendor, self.model, self.observation_id)
    }
}

/// A timer for measuring provider call latency.
///
/// Use this to accurately measure call duration.
pub struct CallTimer {
    start: Instant,
}

impl CallTimer {
    /// Start a new timer.
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time in milliseconds.
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

/// Compute a canonical hash for a request.
///
/// This creates a deterministic fingerprint of a request that can be used
/// for caching and provenance tracking.
///
/// Uses the first 8 bytes of SHA-256 as a `u64` (big-endian), formatted as
/// 16 lowercase hex digits. The result is stable across compiler versions and
/// process invocations.
#[must_use]
pub fn canonical_hash(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let result = hasher.finalize();
    format!(
        "hash:{:016x}",
        u64::from_be_bytes(result[..8].try_into().unwrap())
    )
}

/// Generate a unique observation ID.
fn generate_observation_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    format!("obs-{timestamp:x}-{count:x}")
}

/// Generate a trace ID.
fn generate_trace_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    format!("trace-{timestamp:x}-{count:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_meta() {
        static CAPS: &[Capability] = &[Capability::LlmCompletion];
        let meta = ProviderMeta::new("test", "1.0", CAPS, "test-vendor", Region::US);
        assert_eq!(meta.name, "test");
        assert_eq!(meta.region, Region::US);
    }

    #[test]
    fn test_call_context_default() {
        let ctx = ProviderCallContext::default();
        assert_eq!(ctx.timeout_ms, 30_000);
        assert!(ctx.trace_id.starts_with("trace-"));
    }

    #[test]
    fn test_call_context_builder() {
        let ctx = ProviderCallContext::default()
            .with_root_intent("intent-123")
            .with_user("user-456")
            .with_timeout_ms(5000)
            .with_max_cost(1.0)
            .with_max_tokens(1000);

        assert_eq!(ctx.root_intent_id, Some("intent-123".into()));
        assert_eq!(ctx.user_id, Some("user-456".into()));
        assert_eq!(ctx.timeout_ms, 5000);
        assert_eq!(ctx.max_cost, Some(1.0));
        assert_eq!(ctx.max_tokens, Some(1000));
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn test_observation_provenance() {
        let obs = ProviderObservation::new("anthropic", "claude-3", "content", 100);
        let prov = obs.provenance();
        assert!(prov.starts_with("anthropic:claude-3:obs-"));
    }

    #[test]
    fn test_observation_builder() {
        let obs = ProviderObservation::new("openai", "gpt-4", "response", 500)
            .with_request_hash("hash:abc123")
            .with_cost(0.05)
            .with_tokens(100, 50);

        assert_eq!(obs.request_hash, "hash:abc123");
        assert_eq!(obs.cost_estimate, Some(0.05));
        assert_eq!(obs.tokens.unwrap().total(), 150);
    }

    #[test]
    fn test_raw_response_truncation() {
        let long_response = "x".repeat(20_000);
        let obs = ProviderObservation::new("test", "model", "content", 100)
            .with_raw_response(long_response);

        let raw = obs.raw_response.unwrap();
        assert!(raw.ends_with("...[truncated]"));
        assert!(raw.len() < 15_000);
    }

    #[test]
    fn test_canonical_hash_deterministic() {
        let hash1 = canonical_hash("test input");
        let hash2 = canonical_hash("test input");
        assert_eq!(hash1, hash2);

        let hash3 = canonical_hash("different input");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_call_timer() {
        let timer = CallTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10);
    }

    #[test]
    fn test_observation_ids_unique() {
        let obs1 = ProviderObservation::new("test", "model", "a", 1);
        let obs2 = ProviderObservation::new("test", "model", "b", 2);
        assert_ne!(obs1.observation_id, obs2.observation_id);
    }

    // ========================================================================
    // Property tests
    // ========================================================================

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn canonical_hash_is_deterministic(data in "[a-zA-Z0-9 ]{0,256}") {
            let h1 = canonical_hash(&data);
            let h2 = canonical_hash(&data);
            prop_assert_eq!(h1, h2);
        }

        #[test]
        fn canonical_hash_starts_with_prefix(data in ".{0,100}") {
            let h = canonical_hash(&data);
            prop_assert!(h.starts_with("hash:"));
            prop_assert_eq!(h.len(), 5 + 16); // "hash:" + 16 hex chars
        }

        #[test]
        fn token_usage_total_is_sum(input in 0u32..1_000_000, output in 0u32..1_000_000) {
            let usage = TokenUsage::new(input, output);
            prop_assert_eq!(usage.total(), input + output);
        }

        #[test]
        fn raw_response_truncation_boundary(len in 1usize..30_000) {
            let raw = "x".repeat(len);
            let obs = ProviderObservation::new("v", "m", "c", 1)
                .with_raw_response(raw.clone());
            let stored = obs.raw_response.unwrap();

            if len <= 10_000 {
                prop_assert_eq!(stored, raw);
            } else {
                prop_assert!(stored.len() < 15_000);
                prop_assert!(stored.ends_with("...[truncated]"));
            }
        }

        #[test]
        fn observation_ids_monotonically_unique(count in 2usize..50) {
            let ids: Vec<String> = (0..count)
                .map(|_| ProviderObservation::new("v", "m", "c", 1).observation_id)
                .collect();

            // All unique
            let unique: std::collections::HashSet<&String> = ids.iter().collect();
            prop_assert_eq!(unique.len(), ids.len());
        }

        #[test]
        fn provenance_format(
            vendor in "[a-z]{1,10}",
            mdl in "[a-z0-9-]{1,20}",
        ) {
            let obs = ProviderObservation::new(&vendor, &mdl, "content", 100);
            let prov = obs.provenance();
            let expected_prefix = format!("{vendor}:{mdl}:obs-");
            prop_assert!(prov.starts_with(&expected_prefix));
        }
    }

    // ========================================================================
    // Negative tests
    // ========================================================================

    #[test]
    fn call_context_serde_roundtrip() {
        let ctx = ProviderCallContext::default()
            .with_root_intent("intent-1")
            .with_user("user-1")
            .with_timeout_ms(5000)
            .with_max_cost(0.5)
            .with_max_tokens(2000);

        let json = serde_json::to_string(&ctx).unwrap();
        let round: ProviderCallContext = serde_json::from_str(&json).unwrap();
        assert_eq!(round.root_intent_id, Some("intent-1".into()));
        assert_eq!(round.timeout_ms, 5000);
        assert_eq!(round.max_tokens, Some(2000));
    }

    #[test]
    fn observation_serde_roundtrip() {
        let obs = ProviderObservation::new("openai", "gpt-4", "hello world", 200)
            .with_cost(0.01)
            .with_tokens(50, 10)
            .with_request_hash("hash:abc");

        let json = serde_json::to_string(&obs).unwrap();
        let round: ProviderObservation<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(round.vendor, "openai");
        assert_eq!(round.content, "hello world");
        assert_eq!(round.tokens.unwrap().total(), 60);
    }

    #[test]
    fn region_display() {
        assert_eq!(Region::US.to_string(), "US");
        assert_eq!(Region::EU.to_string(), "EU");
        assert_eq!(Region::CN.to_string(), "CN");
        assert_eq!(Region::Local.to_string(), "Local");
    }

    #[test]
    fn empty_raw_response_not_serialized() {
        let obs = ProviderObservation::new("v", "m", "c", 1);
        let json = serde_json::to_string(&obs).unwrap();
        assert!(!json.contains("raw_response"));
    }
}
