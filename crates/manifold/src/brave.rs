// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! Brave Search API provider.
//!
//! This module provides integration with the Brave Search API for web search
//! capabilities. It can be used standalone or to augment LLM responses with
//! real-time web information.
//!
//! # Capabilities
//!
//! Brave Search API offers multiple search capabilities:
//! - **Web Search**: General web search with snippets and citations
//! - **News Search**: Current news articles
//! - **Image Search**: Image results with metadata
//! - **Video Search**: Video results from various platforms
//! - **AI Summarizer**: AI-powered answer generation with citations (Pro plan)
//! - **Local POI Search**: Places of interest with enriched data
//!
//! # Example
//!
//! ```ignore
//! use manifold::brave::{BraveSearchProvider, BraveSearchRequest, BraveCapability};
//!
//! let provider = BraveSearchProvider::from_env()?;
//!
//! // Check capabilities
//! if provider.supports(BraveCapability::WebSearch) {
//!     let results = provider.search(&BraveSearchRequest::new("rust programming"))?;
//!     println!("{}", BraveSearchProvider::format_for_llm(&results, 5));
//! }
//! ```

use std::fmt::Write;

use serde::{Deserialize, Serialize};

use crate::search::{
    SearchResponsePart, WebSearchBackend, WebSearchError, WebSearchImage, WebSearchRequest,
    WebSearchResponse, WebSearchResult,
};

/// Brave Search capabilities.
///
/// These map to the various Brave Search API endpoints and features.
/// Use these to check what capabilities are available with your API plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BraveCapability {
    /// Web search - general search results with snippets.
    /// Available on all plans.
    WebSearch,
    /// News search - current news articles.
    /// Available on all plans.
    NewsSearch,
    /// Image search - image results with metadata.
    /// Available on all plans.
    ImageSearch,
    /// Video search - video results from various platforms.
    /// Available on all plans.
    VideoSearch,
    /// Local POI search - places of interest with enriched data.
    /// Available on all plans.
    LocalSearch,
    /// AI Summarizer - AI-powered answer generation with citations.
    /// Requires Pro AI plan.
    AiSummarizer,
    /// AI Grounding - OpenAI-compatible endpoint for RAG.
    /// Requires Pro AI plan.
    AiGrounding,
    /// Search Goggles - custom result filtering and re-ranking.
    /// Available on all plans.
    Goggles,
}

impl BraveCapability {
    /// Returns all basic capabilities available on free/basic plans.
    #[must_use]
    pub const fn basic_capabilities() -> &'static [Self] {
        &[
            Self::WebSearch,
            Self::NewsSearch,
            Self::ImageSearch,
            Self::VideoSearch,
            Self::LocalSearch,
            Self::Goggles,
        ]
    }

    /// Returns all capabilities available on Pro AI plans.
    #[must_use]
    pub const fn pro_capabilities() -> &'static [Self] {
        &[
            Self::WebSearch,
            Self::NewsSearch,
            Self::ImageSearch,
            Self::VideoSearch,
            Self::LocalSearch,
            Self::Goggles,
            Self::AiSummarizer,
            Self::AiGrounding,
        ]
    }

    /// Human-readable description of this capability.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::WebSearch => "General web search with snippets and citations",
            Self::NewsSearch => "Current news articles from trusted sources",
            Self::ImageSearch => "Image search with metadata and licensing info",
            Self::VideoSearch => "Video results from various platforms",
            Self::LocalSearch => "Places of interest with reviews and contact info",
            Self::AiSummarizer => "AI-powered summaries with source citations",
            Self::AiGrounding => "OpenAI-compatible RAG endpoint for grounding",
            Self::Goggles => "Custom result filtering and domain re-ranking",
        }
    }
}

impl std::fmt::Display for BraveCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WebSearch => write!(f, "web_search"),
            Self::NewsSearch => write!(f, "news_search"),
            Self::ImageSearch => write!(f, "image_search"),
            Self::VideoSearch => write!(f, "video_search"),
            Self::LocalSearch => write!(f, "local_search"),
            Self::AiSummarizer => write!(f, "ai_summarizer"),
            Self::AiGrounding => write!(f, "ai_grounding"),
            Self::Goggles => write!(f, "goggles"),
        }
    }
}

/// Error type for Brave Search operations.
#[derive(Debug, thiserror::Error)]
pub enum BraveSearchError {
    /// Network/HTTP error.
    #[error("Network error: {0}")]
    Network(String),

    /// API authentication error.
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// API response parsing error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// General API error.
    #[error("API error: {0}")]
    Api(String),
}

impl From<BraveSearchError> for WebSearchError {
    fn from(value: BraveSearchError) -> Self {
        match value {
            BraveSearchError::Network(message) => Self::Network(message),
            BraveSearchError::Auth(message) => Self::Auth(message),
            BraveSearchError::RateLimit(message) => Self::RateLimit(message),
            BraveSearchError::Parse(message) => Self::Parse(message),
            BraveSearchError::Api(message) => Self::Api(message),
        }
    }
}

/// A single search result from Brave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraveSearchResult {
    /// Title of the search result.
    pub title: String,
    /// URL of the result.
    pub url: String,
    /// Description/snippet of the result.
    pub description: String,
    /// Age of the result (e.g., "2 hours ago").
    #[serde(default)]
    pub age: Option<String>,
    /// Family-friendly rating.
    #[serde(default)]
    pub family_friendly: bool,
}

/// Response from Brave Search API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraveSearchResponse {
    /// Web search results.
    pub results: Vec<BraveSearchResult>,
    /// Query that was searched.
    pub query: String,
    /// Total number of results (estimated).
    #[serde(default)]
    pub total_results: Option<u64>,
    /// Time taken for the search (in seconds).
    #[serde(default)]
    pub search_time: Option<f64>,
}

/// Search request options.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BraveSearchRequest {
    /// The search query.
    pub query: String,
    /// Number of results to return (default: 10, max: 20).
    pub count: Option<u32>,
    /// Offset for pagination.
    pub offset: Option<u32>,
    /// Country code for localized results (e.g., "US", "GB").
    pub country: Option<String>,
    /// Language code (e.g., "en", "de").
    pub language: Option<String>,
    /// Safe search mode ("off", "moderate", "strict").
    pub safesearch: Option<String>,
    /// Freshness filter ("pd" = past day, "pw" = past week, "pm" = past month).
    pub freshness: Option<String>,
}

impl BraveSearchRequest {
    /// Creates a new search request with the given query.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Default::default()
        }
    }

    /// Sets the number of results to return.
    #[must_use]
    pub fn with_count(mut self, count: u32) -> Self {
        self.count = Some(count.min(20)); // Brave max is 20
        self
    }

    /// Sets the offset for pagination.
    #[must_use]
    pub fn with_offset(mut self, offset: u32) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Sets the country for localized results.
    #[must_use]
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Sets the language for results.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the safe search mode.
    #[must_use]
    pub fn with_safesearch(mut self, mode: impl Into<String>) -> Self {
        self.safesearch = Some(mode.into());
        self
    }

    /// Sets the freshness filter.
    #[must_use]
    pub fn with_freshness(mut self, freshness: impl Into<String>) -> Self {
        self.freshness = Some(freshness.into());
        self
    }
}

/// Brave Search API provider.
///
/// # Example
///
/// ```ignore
/// use manifold::brave::{BraveSearchProvider, BraveSearchRequest};
///
/// let provider = BraveSearchProvider::from_env()?;
/// let request = BraveSearchRequest::new("rust programming language")
///     .with_count(5);
/// let results = provider.search(&request)?;
///
/// for result in results.results {
///     println!("{}: {}", result.title, result.url);
/// }
/// ```
pub struct BraveSearchProvider {
    api_key: crate::secret::SecretString,
    base_url: String,
    client: reqwest::blocking::Client,
}

impl std::fmt::Debug for BraveSearchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BraveSearchProvider")
            .field("api_key", &self.api_key) // SecretString redacts automatically
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl BraveSearchProvider {
    fn classify_http_error(status: u16, body: &str) -> BraveSearchError {
        let normalized = body.to_ascii_lowercase();

        if status == 429 || normalized.contains("rate limit") || normalized.contains("quota") {
            return BraveSearchError::RateLimit(body.trim().to_string());
        }

        if matches!(status, 401 | 403)
            || ((status == 400 || status == 422)
                && (normalized.contains("api key")
                    || normalized.contains("subscription")
                    || normalized.contains("unauthorized")
                    || normalized.contains("authentication")
                    || normalized.contains("token")))
        {
            return BraveSearchError::Auth(body.trim().to_string());
        }

        BraveSearchError::Api(format!("HTTP {status}: {}", body.trim()))
    }

    /// Creates a new Brave Search provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: crate::secret::SecretString::new(api_key),
            base_url: "https://api.search.brave.com/res/v1".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Creates a provider using the `BRAVE_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns error if the environment variable is not set.
    pub fn from_env() -> Result<Self, BraveSearchError> {
        let api_key = std::env::var("BRAVE_API_KEY").map_err(|_| {
            BraveSearchError::Auth("BRAVE_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(api_key))
    }

    /// Checks if Brave Search is available (API key is set).
    #[must_use]
    pub fn is_available() -> bool {
        std::env::var("BRAVE_API_KEY").is_ok()
    }

    /// Returns capabilities available on the basic plan.
    ///
    /// Note: This doesn't verify your actual plan - it returns
    /// the capabilities that should be available on basic plans.
    #[must_use]
    pub const fn basic_capabilities() -> &'static [BraveCapability] {
        BraveCapability::basic_capabilities()
    }

    /// Returns all capabilities (requires Pro AI plan for AI features).
    #[must_use]
    pub const fn all_capabilities() -> &'static [BraveCapability] {
        BraveCapability::pro_capabilities()
    }

    /// Checks if this provider supports a specific capability.
    ///
    /// Note: This returns `true` for basic capabilities. AI capabilities
    /// (summarizer, grounding) require a Pro plan which cannot be verified
    /// without making an API call.
    #[must_use]
    pub fn supports(&self, capability: BraveCapability) -> bool {
        // Basic capabilities are always supported if we have an API key
        matches!(
            capability,
            BraveCapability::WebSearch
                | BraveCapability::NewsSearch
                | BraveCapability::ImageSearch
                | BraveCapability::VideoSearch
                | BraveCapability::LocalSearch
                | BraveCapability::Goggles
        )
    }

    /// Uses a custom base URL (for testing or proxies).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Performs a web search.
    ///
    /// # Errors
    ///
    /// Returns error if the search fails due to network, auth, or API issues.
    pub fn search(
        &self,
        request: &BraveSearchRequest,
    ) -> Result<BraveSearchResponse, BraveSearchError> {
        let mut url = format!(
            "{}/web/search?q={}",
            self.base_url,
            urlencoding::encode(&request.query)
        );

        if let Some(count) = request.count {
            let _ = write!(url, "&count={count}");
        }
        if let Some(offset) = request.offset {
            let _ = write!(url, "&offset={offset}");
        }
        if let Some(ref country) = request.country {
            let _ = write!(url, "&country={country}");
        }
        if let Some(ref language) = request.language {
            let _ = write!(url, "&search_lang={language}");
        }
        if let Some(ref safesearch) = request.safesearch {
            let _ = write!(url, "&safesearch={safesearch}");
        }
        if let Some(ref freshness) = request.freshness {
            let _ = write!(url, "&freshness={freshness}");
        }

        let response = self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .header("X-Subscription-Token", self.api_key.expose())
            .send()
            .map_err(|e| BraveSearchError::Network(format!("Request failed: {e}")))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().unwrap_or_default();
            return Err(Self::classify_http_error(status.as_u16(), &error_text));
        }

        // Parse the response
        let api_response: BraveApiResponse = response
            .json()
            .map_err(|e| BraveSearchError::Parse(format!("Failed to parse response: {e}")))?;

        // Convert to our response format
        let total_results = api_response.web.as_ref().and_then(|w| w.total_results);
        let results = api_response
            .web
            .map(|web| web.results)
            .unwrap_or_default()
            .into_iter()
            .map(|r| BraveSearchResult {
                title: r.title,
                url: r.url,
                description: r.description.unwrap_or_default(),
                age: r.age,
                family_friendly: r.family_friendly.unwrap_or(true),
            })
            .collect();

        Ok(BraveSearchResponse {
            results,
            query: request.query.clone(),
            total_results,
            search_time: None,
        })
    }

    /// Performs a simple search with just a query string.
    ///
    /// # Errors
    ///
    /// Returns error if the search fails.
    pub fn search_simple(
        &self,
        query: impl Into<String>,
    ) -> Result<BraveSearchResponse, BraveSearchError> {
        self.search(&BraveSearchRequest::new(query))
    }

    /// Formats search results as markdown for LLM context.
    #[must_use]
    pub fn format_as_markdown(response: &BraveSearchResponse) -> String {
        let mut output = format!("## Web Search Results for: {}\n\n", response.query);

        for (i, result) in response.results.iter().enumerate() {
            let _ = write!(
                output,
                "### {}. {}\n**URL:** {}\n{}\n\n",
                i + 1,
                result.title,
                result.url,
                result.description
            );
        }

        if response.results.is_empty() {
            output.push_str("No results found.\n");
        }

        output
    }

    /// Formats search results as a concise context string for LLM prompts.
    #[must_use]
    pub fn format_for_llm(response: &BraveSearchResponse, max_results: usize) -> String {
        let results: Vec<_> = response.results.iter().take(max_results).collect();

        if results.is_empty() {
            return "No web search results found.".to_string();
        }

        let mut output = String::from("Web search results:\n");
        for result in results {
            let _ = writeln!(
                output,
                "- {} ({}): {}",
                result.title, result.url, result.description
            );
        }
        output
    }

    fn from_web_request(request: &WebSearchRequest) -> BraveSearchRequest {
        let mut brave = BraveSearchRequest::new(request.query.clone());

        if let Some(max_results) = request.max_results {
            brave = brave.with_count(max_results);
        }
        if let Some(country) = &request.country {
            brave = brave.with_country(country.clone());
        }
        if let Some(language) = &request.language {
            brave = brave.with_language(language.clone());
        }
        if let Some(time_range) = &request.time_range {
            let freshness = match time_range.as_str() {
                "day" | "d" => Some("pd"),
                "week" | "w" => Some("pw"),
                "month" | "m" => Some("pm"),
                _ => None,
            };
            if let Some(freshness) = freshness {
                brave = brave.with_freshness(freshness);
            }
        }

        let _ = request.response_parts.contains(SearchResponsePart::Answer);
        let _ = request
            .response_parts
            .contains(SearchResponsePart::RawContent);
        let _ = request.response_parts.contains(SearchResponsePart::Images);
        let _ = request.response_parts.contains(SearchResponsePart::Favicon);
        let _ = request.include_domains.len();
        let _ = request.exclude_domains.len();
        let _ = request.topic;
        let _ = request.search_depth;

        brave
    }

    fn into_web_response(response: BraveSearchResponse) -> WebSearchResponse {
        WebSearchResponse {
            provider: "brave".to_string(),
            query: response.query,
            answer: None,
            results: response
                .results
                .into_iter()
                .map(|result| WebSearchResult {
                    title: result.title,
                    url: result.url,
                    content: result.description,
                    score: None,
                    published_at: result.age,
                    favicon: None,
                    raw_content: None,
                })
                .collect(),
            images: Vec::<WebSearchImage>::new(),
            response_time: response.search_time,
        }
    }
}

impl WebSearchBackend for BraveSearchProvider {
    fn provider_name(&self) -> &'static str {
        "brave"
    }

    fn search_web(&self, request: &WebSearchRequest) -> Result<WebSearchResponse, WebSearchError> {
        let request = Self::from_web_request(request);
        let response = self.search(&request)?;
        Ok(Self::into_web_response(response))
    }
}

// Internal API response structures

#[derive(Debug, Deserialize)]
struct BraveApiResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Vec<BraveApiResult>,
    #[serde(rename = "totalResults")]
    total_results: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct BraveApiResult {
    title: String,
    url: String,
    description: Option<String>,
    age: Option<String>,
    family_friendly: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_search_request_builder() {
        let request = BraveSearchRequest::new("test query")
            .with_count(5)
            .with_country("US")
            .with_language("en");

        assert_eq!(request.query, "test query");
        assert_eq!(request.count, Some(5));
        assert_eq!(request.country, Some("US".to_string()));
        assert_eq!(request.language, Some("en".to_string()));
    }

    #[test]
    fn test_count_clamped_to_max() {
        let request = BraveSearchRequest::new("test").with_count(100);
        assert_eq!(request.count, Some(20)); // Max is 20
    }

    proptest! {
        #[test]
        fn count_is_always_clamped(count in any::<u32>()) {
            let request = BraveSearchRequest::new("test").with_count(count);
            prop_assert!(request.count.unwrap() <= 20);
        }
    }

    #[test]
    fn test_format_as_markdown() {
        let response = BraveSearchResponse {
            query: "rust programming".to_string(),
            results: vec![BraveSearchResult {
                title: "The Rust Programming Language".to_string(),
                url: "https://rust-lang.org".to_string(),
                description: "A language empowering everyone to build reliable software."
                    .to_string(),
                age: None,
                family_friendly: true,
            }],
            total_results: Some(1000),
            search_time: None,
        };

        let markdown = BraveSearchProvider::format_as_markdown(&response);
        assert!(markdown.contains("rust programming"));
        assert!(markdown.contains("The Rust Programming Language"));
        assert!(markdown.contains("https://rust-lang.org"));
    }

    #[test]
    fn auth_like_422_responses_are_classified_as_auth_errors() {
        let error = BraveSearchProvider::classify_http_error(422, "Invalid subscription token");
        assert!(matches!(error, BraveSearchError::Auth(_)));
    }
}
