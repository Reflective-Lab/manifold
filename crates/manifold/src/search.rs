// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Generic web search request/response types for search-capable providers.

use serde::de;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::num::{NonZeroU64, NonZeroUsize};

/// Error type for web search operations.
#[derive(Debug, thiserror::Error)]
pub enum WebSearchError {
    /// Network/HTTP failure.
    #[error("network error: {0}")]
    Network(String),
    /// Authentication failure.
    #[error("authentication error: {0}")]
    Auth(String),
    /// Rate limit exceeded.
    #[error("rate limit exceeded: {0}")]
    RateLimit(String),
    /// Response parsing failure.
    #[error("parse error: {0}")]
    Parse(String),
    /// Provider-specific API failure.
    #[error("api error: {0}")]
    Api(String),
}

/// Search topic hint for providers that support topic routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTopic {
    #[default]
    General,
    News,
    Finance,
}

/// Search depth hint for providers that expose relevance vs. latency tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchDepth {
    #[default]
    Basic,
    Advanced,
    Fast,
    UltraFast,
}

/// Optional response parts a search caller can request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchResponsePart {
    Answer,
    RawContent,
    Images,
    Favicon,
}

/// Set of optional response parts a search caller requests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SearchResponseParts(BTreeSet<SearchResponsePart>);

impl SearchResponseParts {
    /// Create an empty response-part set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return true when no optional response parts are requested.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return true when the response-part set contains `part`.
    #[must_use]
    pub fn contains(&self, part: SearchResponsePart) -> bool {
        self.0.contains(&part)
    }

    /// Enable or disable a response part.
    pub fn set(&mut self, part: SearchResponsePart, enabled: bool) {
        if enabled {
            self.0.insert(part);
        } else {
            self.0.remove(&part);
        }
    }
}

/// Provider-agnostic web search request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebSearchRequest {
    /// Query text.
    pub query: String,
    /// Maximum results to return.
    pub max_results: Option<u32>,
    /// Country bias.
    pub country: Option<String>,
    /// Language bias.
    pub language: Option<String>,
    /// Relative freshness or time range hint.
    pub time_range: Option<String>,
    /// Topic/category hint.
    pub topic: SearchTopic,
    /// Search depth / quality hint.
    pub search_depth: SearchDepth,
    /// Optional response parts to include when supported.
    #[serde(default, skip_serializing_if = "SearchResponseParts::is_empty")]
    pub response_parts: SearchResponseParts,
    /// Optional allowlist of domains.
    pub include_domains: Vec<String>,
    /// Optional denylist of domains.
    pub exclude_domains: Vec<String>,
}

impl WebSearchRequest {
    /// Create a new web search request.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Self::default()
        }
    }

    /// Set the maximum number of results.
    #[must_use]
    pub fn with_max_results(mut self, max_results: u32) -> Self {
        self.max_results = Some(max_results);
        self
    }

    /// Set the country bias.
    #[must_use]
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the language bias.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the time range or freshness hint.
    #[must_use]
    pub fn with_time_range(mut self, time_range: impl Into<String>) -> Self {
        self.time_range = Some(time_range.into());
        self
    }

    /// Set the topic/category.
    #[must_use]
    pub fn with_topic(mut self, topic: SearchTopic) -> Self {
        self.topic = topic;
        self
    }

    /// Set the depth/latency tradeoff.
    #[must_use]
    pub fn with_search_depth(mut self, search_depth: SearchDepth) -> Self {
        self.search_depth = search_depth;
        self
    }

    /// Include an answer summary if supported.
    #[must_use]
    pub fn with_answer(mut self, include: bool) -> Self {
        self.response_parts.set(SearchResponsePart::Answer, include);
        self
    }

    /// Include raw content if supported.
    #[must_use]
    pub fn with_raw_content(mut self, include: bool) -> Self {
        self.response_parts
            .set(SearchResponsePart::RawContent, include);
        self
    }

    /// Include image results if supported.
    #[must_use]
    pub fn with_images(mut self, include: bool) -> Self {
        self.response_parts.set(SearchResponsePart::Images, include);
        self
    }

    /// Include favicon URLs if supported.
    #[must_use]
    pub fn with_favicon(mut self, include: bool) -> Self {
        self.response_parts
            .set(SearchResponsePart::Favicon, include);
        self
    }

    /// Restrict search to the given domains.
    #[must_use]
    pub fn with_include_domains(mut self, domains: Vec<String>) -> Self {
        self.include_domains = domains;
        self
    }

    /// Exclude the given domains.
    #[must_use]
    pub fn with_exclude_domains(mut self, domains: Vec<String>) -> Self {
        self.exclude_domains = domains;
        self
    }
}

/// Generic image result metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchImage {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Generic text result metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub favicon: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content: Option<String>,
}

/// Generic web search response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchResponse {
    pub provider: String,
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    pub results: Vec<WebSearchResult>,
    pub images: Vec<WebSearchImage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_time: Option<f64>,
}

/// Executable contract for provider-local web search adapters.
pub trait WebSearchBackend: Send + Sync {
    /// Stable provider identifier.
    fn provider_name(&self) -> &'static str;

    /// Execute a search request.
    fn search_web(&self, request: &WebSearchRequest) -> Result<WebSearchResponse, WebSearchError>;
}

// ---------------------------------------------------------------------------
// Web fetch (URL → content)
// ---------------------------------------------------------------------------

pub(crate) const DEFAULT_WEB_FETCH_MAX_BYTES: usize = 1_048_576;
pub(crate) const MAX_WEB_FETCH_BYTES: usize = 8 * 1_048_576;
pub(crate) const DEFAULT_WEB_FETCH_TIMEOUT_MS: u64 = 30_000;
pub(crate) const MAX_WEB_FETCH_TIMEOUT_MS: u64 = 120_000;

/// Absolute public HTTP(S) URL accepted by web fetch.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct WebFetchUrl(String);

impl WebFetchUrl {
    pub fn new(value: impl Into<String>) -> Result<Self, WebFetchError> {
        let value = value.into();
        let url = reqwest::Url::parse(&value)
            .map_err(|error| WebFetchError::InvalidUrl(error.to_string()))?;
        validate_public_http_url(&url)?;
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn parse(&self) -> Result<reqwest::Url, WebFetchError> {
        reqwest::Url::parse(self.as_str())
            .map_err(|error| WebFetchError::InvalidUrl(error.to_string()))
    }
}

impl std::fmt::Display for WebFetchUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for WebFetchUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Positive byte limit for web fetch responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct WebFetchByteLimit(NonZeroUsize);

impl WebFetchByteLimit {
    pub fn new(value: usize) -> Result<Self, WebFetchError> {
        let value = NonZeroUsize::new(value).ok_or_else(|| {
            WebFetchError::InvalidLimit("max_bytes must be greater than zero".into())
        })?;
        if value.get() > MAX_WEB_FETCH_BYTES {
            return Err(WebFetchError::InvalidLimit(format!(
                "max_bytes must be <= {MAX_WEB_FETCH_BYTES}"
            )));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for WebFetchByteLimit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = usize::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Positive timeout in milliseconds for a single web fetch request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct WebFetchTimeoutMs(NonZeroU64);

impl WebFetchTimeoutMs {
    pub fn new(value: u64) -> Result<Self, WebFetchError> {
        let value = NonZeroU64::new(value).ok_or_else(|| {
            WebFetchError::InvalidLimit("timeout_ms must be greater than zero".into())
        })?;
        if value.get() > MAX_WEB_FETCH_TIMEOUT_MS {
            return Err(WebFetchError::InvalidLimit(format!(
                "timeout_ms must be <= {MAX_WEB_FETCH_TIMEOUT_MS}"
            )));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for WebFetchTimeoutMs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// HTTP method for a fetch request. GET is the historical default; POST
/// is needed for SOAP envelopes, JSON-API POST endpoints, and other
/// services where the request payload doesn't fit in the URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum WebFetchMethod {
    #[default]
    Get,
    Post,
}

/// Request to fetch a single URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFetchRequest {
    /// URL to fetch.
    pub url: WebFetchUrl,
    /// HTTP method (default: GET).
    #[serde(default)]
    pub method: WebFetchMethod,
    /// Request body. Only meaningful when `method` is `POST`. Stored as a
    /// `String` so SOAP envelopes and JSON payloads serialize cleanly;
    /// binary payloads should be base64-encoded by the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    /// Optional HTTP headers to include on the original origin.
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    /// Maximum response body size in bytes (default: 1 MiB).
    #[serde(default = "default_max_bytes")]
    pub max_bytes: WebFetchByteLimit,
    /// Request timeout in milliseconds (default: 30 000).
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: WebFetchTimeoutMs,
}

fn default_max_bytes() -> WebFetchByteLimit {
    WebFetchByteLimit::new(DEFAULT_WEB_FETCH_MAX_BYTES).expect("default byte limit is non-zero")
}

fn default_timeout_ms() -> WebFetchTimeoutMs {
    WebFetchTimeoutMs::new(DEFAULT_WEB_FETCH_TIMEOUT_MS).expect("default timeout is non-zero")
}

impl WebFetchRequest {
    pub fn new(url: impl Into<String>) -> Result<Self, WebFetchError> {
        Ok(Self {
            url: WebFetchUrl::new(url)?,
            method: WebFetchMethod::Get,
            body: None,
            headers: Vec::new(),
            max_bytes: default_max_bytes(),
            timeout_ms: default_timeout_ms(),
        })
    }

    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Result<Self, WebFetchError> {
        self.max_bytes = WebFetchByteLimit::new(max_bytes)?;
        Ok(self)
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Result<Self, WebFetchError> {
        self.timeout_ms = WebFetchTimeoutMs::new(timeout_ms)?;
        Ok(self)
    }

    /// Set the HTTP method explicitly. Default is `GET`. Used together
    /// with [`Self::with_body`] for POST requests (SOAP envelopes,
    /// JSON-API POST endpoints).
    #[must_use]
    pub fn with_method(mut self, method: WebFetchMethod) -> Self {
        self.method = method;
        self
    }

    /// Attach a request body and automatically switch to `POST`. Callers
    /// that need a non-POST method with a body (rare) should call
    /// [`Self::with_method`] after this. The body is serialized as
    /// UTF-8 text; binary payloads must be base64-encoded by the caller.
    #[must_use]
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self.method = WebFetchMethod::Post;
        self
    }
}

/// Response from a URL fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFetchResponse {
    /// The fetched URL (after redirects).
    pub url: String,
    /// HTTP status code.
    pub status: u16,
    /// Content-Type header value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Response body as text.
    pub body: String,
    /// Whether the body was truncated to `max_bytes`.
    pub truncated: bool,
}

/// Error type for web fetch operations.
#[derive(Debug, thiserror::Error)]
pub enum WebFetchError {
    #[error("network error: {0}")]
    Network(String),
    #[error("timeout after {0}ms")]
    Timeout(u64),
    #[error("response too large (>{0} bytes)")]
    TooLarge(usize),
    #[error("invalid limit: {0}")]
    InvalidLimit(String),
    #[error("invalid url: {0}")]
    InvalidUrl(String),
    #[error("http {0}: {1}")]
    Http(u16, String),
}

/// Executable contract for fetching a URL and returning its content.
pub trait WebFetchBackend: Send + Sync {
    fn provider_name(&self) -> &'static str;

    fn fetch(&self, request: &WebFetchRequest) -> Result<WebFetchResponse, WebFetchError>;
}

pub(crate) fn validate_public_http_url(url: &reqwest::Url) -> Result<(), WebFetchError> {
    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(WebFetchError::InvalidUrl(format!(
                "unsupported URL scheme '{scheme}'"
            )));
        }
    }

    if !url.username().is_empty() || url.password().is_some() {
        return Err(WebFetchError::InvalidUrl(
            "URLs with embedded credentials are not allowed".into(),
        ));
    }

    let host = url
        .host_str()
        .ok_or_else(|| WebFetchError::InvalidUrl("URL must include a host".into()))?;
    let normalized_host = host.trim_end_matches('.');
    if normalized_host.eq_ignore_ascii_case("localhost") || normalized_host.ends_with(".localhost")
    {
        return Err(WebFetchError::InvalidUrl(
            "localhost targets are not allowed".into(),
        ));
    }

    let host_for_ip = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host_for_ip.parse::<IpAddr>() {
        reject_non_public_ip(ip)?;
    }

    Ok(())
}

pub(crate) fn reject_non_public_ip(ip: IpAddr) -> Result<(), WebFetchError> {
    let blocked = match ip {
        IpAddr::V4(ip) => is_non_public_ipv4(ip),
        IpAddr::V6(ip) => {
            let first = ip.segments()[0];
            let second = ip.segments()[1];
            let mapped_private = ipv4_mapped(ip).is_some_and(is_non_public_ipv4);
            ip.is_loopback()
                || ip.is_multicast()
                || ip.is_unspecified()
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || (first == 0x2001 && second == 0x0db8)
                || mapped_private
        }
    };

    if blocked {
        Err(WebFetchError::InvalidUrl(format!(
            "non-public IP targets are not allowed: {ip}"
        )))
    } else {
        Ok(())
    }
}

fn is_non_public_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_multicast()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || octets[0] == 0
        || (octets[0] == 100 && (64..=127).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        || (octets[0] == 198 && (18..=19).contains(&octets[1]))
        || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
        || octets[0] >= 240
}

fn ipv4_mapped(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let octets = ip.octets();
    (octets[..10].iter().all(|byte| *byte == 0) && octets[10] == 0xff && octets[11] == 0xff)
        .then(|| Ipv4Addr::new(octets[12], octets[13], octets[14], octets[15]))
}
