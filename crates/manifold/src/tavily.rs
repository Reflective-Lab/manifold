// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Tavily Search API provider.

use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Deserializer, Serialize};

use crate::search::{
    SearchDepth, SearchResponsePart, SearchTopic, WebSearchBackend, WebSearchError, WebSearchImage,
    WebSearchRequest, WebSearchResponse, WebSearchResult,
};
use crate::secret::SecretString;

/// Tavily Search API provider.
pub struct TavilySearchProvider {
    api_key: SecretString,
    base_url: String,
    client: Client,
}

impl std::fmt::Debug for TavilySearchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TavilySearchProvider")
            .field("api_key", &self.api_key)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl TavilySearchProvider {
    fn classify_http_error(status: u16, body: &str) -> WebSearchError {
        let normalized = body.to_ascii_lowercase();

        if status == 429 || normalized.contains("rate limit") || normalized.contains("quota") {
            return WebSearchError::RateLimit(body.trim().to_string());
        }

        if matches!(status, 401 | 403)
            || ((status == 400 || status == 422)
                && (normalized.contains("api key")
                    || normalized.contains("unauthorized")
                    || normalized.contains("authentication")
                    || normalized.contains("token")
                    || normalized.contains("credential")))
        {
            return WebSearchError::Auth(body.trim().to_string());
        }

        WebSearchError::Api(format!("HTTP {status}: {}", body.trim()))
    }

    /// Create a new provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::new(api_key),
            base_url: "https://api.tavily.com".to_string(),
            client: Client::new(),
        }
    }

    /// Create a provider from `TAVILY_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns an auth error if the environment variable is not set.
    pub fn from_env() -> Result<Self, WebSearchError> {
        let api_key = std::env::var("TAVILY_API_KEY").map_err(|_| {
            WebSearchError::Auth("TAVILY_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(api_key))
    }

    /// Check whether Tavily is available in the current environment.
    #[must_use]
    pub fn is_available() -> bool {
        std::env::var("TAVILY_API_KEY").is_ok()
    }

    /// Use a custom base URL (for tests or proxies).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn build_headers(&self) -> Result<HeaderMap, WebSearchError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let auth = format!("Bearer {}", self.api_key.expose());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth)
                .map_err(|e| WebSearchError::Auth(format!("invalid Tavily API key: {e}")))?,
        );
        Ok(headers)
    }

    fn build_request(&self, request: &WebSearchRequest) -> TavilySearchRequest {
        TavilySearchRequest {
            query: request.query.clone(),
            search_depth: match request.search_depth {
                SearchDepth::Basic => "basic",
                SearchDepth::Advanced => "advanced",
                SearchDepth::Fast => "fast",
                SearchDepth::UltraFast => "ultra-fast",
            }
            .to_string(),
            max_results: request.max_results.unwrap_or(5).min(20),
            topic: match request.topic {
                SearchTopic::General => "general",
                SearchTopic::News => "news",
                SearchTopic::Finance => "finance",
            }
            .to_string(),
            time_range: request.time_range.clone(),
            include_answer: request.response_parts.contains(SearchResponsePart::Answer),
            include_raw_content: request
                .response_parts
                .contains(SearchResponsePart::RawContent),
            include_images: request.response_parts.contains(SearchResponsePart::Images),
            include_image_descriptions: request.response_parts.contains(SearchResponsePart::Images),
            include_favicon: request.response_parts.contains(SearchResponsePart::Favicon),
            include_domains: if request.include_domains.is_empty() {
                None
            } else {
                Some(request.include_domains.clone())
            },
            exclude_domains: if request.exclude_domains.is_empty() {
                None
            } else {
                Some(request.exclude_domains.clone())
            },
            country: request.country.clone(),
        }
    }

    fn parse_response(&self, response: TavilySearchResponse, query: &str) -> WebSearchResponse {
        WebSearchResponse {
            provider: "tavily".to_string(),
            query: query.to_string(),
            answer: response.answer,
            results: response
                .results
                .into_iter()
                .map(|result| WebSearchResult {
                    title: result.title,
                    url: result.url,
                    content: result.content,
                    score: result.score,
                    published_at: None,
                    favicon: result.favicon,
                    raw_content: result.raw_content,
                })
                .collect(),
            images: response
                .images
                .into_iter()
                .map(|image| WebSearchImage {
                    url: image.url,
                    description: image.description,
                })
                .collect(),
            response_time: response.response_time,
        }
    }
}

impl WebSearchBackend for TavilySearchProvider {
    fn provider_name(&self) -> &'static str {
        "tavily"
    }

    fn search_web(&self, request: &WebSearchRequest) -> Result<WebSearchResponse, WebSearchError> {
        let url = format!("{}/search", self.base_url);
        let headers = self.build_headers()?;
        let request_body = self.build_request(request);

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request_body)
            .send()
            .map_err(|e| WebSearchError::Network(format!("request failed: {e}")))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().unwrap_or_default();
            return Err(Self::classify_http_error(status.as_u16(), &error_text));
        }

        let response_body: TavilySearchResponse = response
            .json()
            .map_err(|e| WebSearchError::Parse(format!("failed to parse response: {e}")))?;

        Ok(self.parse_response(response_body, &request.query))
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Serialize)]
struct TavilySearchRequest {
    query: String,
    search_depth: String,
    max_results: u32,
    topic: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_range: Option<String>,
    include_answer: bool,
    include_raw_content: bool,
    include_images: bool,
    include_image_descriptions: bool,
    include_favicon: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    country: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TavilySearchResponse {
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    images: Vec<TavilyImage>,
    #[serde(default)]
    results: Vec<TavilyResult>,
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    response_time: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
    #[serde(default)]
    score: Option<f32>,
    #[serde(default)]
    raw_content: Option<String>,
    #[serde(default)]
    favicon: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TavilyImage {
    url: String,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum F64OrString {
    Number(f64),
    Text(String),
}

fn deserialize_optional_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<F64OrString>::deserialize(deserializer)?;
    Ok(match value {
        Some(F64OrString::Number(value)) => Some(value),
        Some(F64OrString::Text(value)) => value.parse::<f64>().ok(),
        None => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn build_request_maps_generic_fields() {
        let provider = TavilySearchProvider::new("test-key");
        let request = WebSearchRequest::new("rust async")
            .with_max_results(7)
            .with_topic(SearchTopic::News)
            .with_search_depth(SearchDepth::Advanced)
            .with_answer(true)
            .with_images(true)
            .with_favicon(true)
            .with_country("united states");

        let built = provider.build_request(&request);
        assert_eq!(built.query, "rust async");
        assert_eq!(built.max_results, 7);
        assert_eq!(built.topic, "news");
        assert_eq!(built.search_depth, "advanced");
        assert!(built.include_answer);
        assert!(built.include_images);
        assert!(built.include_image_descriptions);
        assert!(built.include_favicon);
        assert_eq!(built.country.as_deref(), Some("united states"));
    }

    #[test]
    fn parse_response_converts_generic_output() {
        let provider = TavilySearchProvider::new("test-key");
        let response = TavilySearchResponse {
            answer: Some("Answer".to_string()),
            images: vec![TavilyImage {
                url: "https://example.com/image.png".to_string(),
                description: Some("An image".to_string()),
            }],
            results: vec![TavilyResult {
                title: "Example".to_string(),
                url: "https://example.com".to_string(),
                content: "Snippet".to_string(),
                score: Some(0.9),
                raw_content: Some("Full content".to_string()),
                favicon: Some("https://example.com/favicon.ico".to_string()),
            }],
            response_time: Some(1.23),
        };

        let parsed = provider.parse_response(response, "query");
        assert_eq!(parsed.provider, "tavily");
        assert_eq!(parsed.answer.as_deref(), Some("Answer"));
        assert_eq!(parsed.images.len(), 1);
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(parsed.response_time, Some(1.23));
    }

    proptest! {
        #[test]
        fn build_request_clamps_results_and_preserves_flags(
            max_results in any::<u32>(),
            include_answer in any::<bool>(),
            include_images in any::<bool>(),
            include_favicon in any::<bool>(),
        ) {
            let provider = TavilySearchProvider::new("test-key");
            let request = WebSearchRequest::new("rust")
                .with_max_results(max_results)
                .with_answer(include_answer)
                .with_images(include_images)
                .with_favicon(include_favicon);

            let built = provider.build_request(&request);
            prop_assert!(built.max_results <= 20);
            prop_assert_eq!(built.include_answer, include_answer);
            prop_assert_eq!(built.include_images, include_images);
            prop_assert_eq!(built.include_image_descriptions, include_images);
            prop_assert_eq!(built.include_favicon, include_favicon);
        }
    }

    #[test]
    fn auth_like_400_responses_are_classified_as_auth_errors() {
        let error = TavilySearchProvider::classify_http_error(400, "API key is invalid");
        assert!(matches!(error, WebSearchError::Auth(_)));
    }
}
