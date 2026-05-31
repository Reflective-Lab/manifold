// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

/// Perplexity Search provider — uses Perplexity's chat completions endpoint with online models (sonar, sonar-pro).
/// Returns the LLM-generated answer plus citation URLs as search results.
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::search::{
    WebSearchBackend, WebSearchError, WebSearchImage, WebSearchRequest, WebSearchResponse,
    WebSearchResult,
};
use crate::secret::SecretString;

/// Perplexity Search API provider.
pub struct PerplexitySearchProvider {
    api_key: SecretString,
    base_url: String,
    model: String,
    client: Client,
}

impl std::fmt::Debug for PerplexitySearchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerplexitySearchProvider")
            .field("api_key", &self.api_key)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl PerplexitySearchProvider {
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
                    || normalized.contains("token")))
        {
            return WebSearchError::Auth(body.trim().to_string());
        }

        WebSearchError::Api(format!("HTTP {status}: {}", body.trim()))
    }

    /// Create a new Perplexity Search provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::new(api_key),
            base_url: "https://api.perplexity.ai".to_string(),
            model: "sonar".to_string(),
            client: Client::new(),
        }
    }

    /// Create a provider from `PERPLEXITY_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns an auth error if the environment variable is not set.
    pub fn from_env() -> Result<Self, WebSearchError> {
        let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|_| {
            WebSearchError::Auth("PERPLEXITY_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(api_key))
    }

    /// Check whether Perplexity is available in the current environment.
    #[must_use]
    pub fn is_available() -> bool {
        std::env::var("PERPLEXITY_API_KEY").is_ok()
    }

    /// Use a custom base URL (for tests or proxies).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Use a custom Perplexity online model (e.g. `sonar`, `sonar-pro`).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn build_headers(&self) -> Result<HeaderMap, WebSearchError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let auth = format!("Bearer {}", self.api_key.expose());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth)
                .map_err(|e| WebSearchError::Auth(format!("invalid Perplexity API key: {e}")))?,
        );
        Ok(headers)
    }
}

impl WebSearchBackend for PerplexitySearchProvider {
    fn provider_name(&self) -> &'static str {
        "perplexity"
    }

    fn search_web(&self, request: &WebSearchRequest) -> Result<WebSearchResponse, WebSearchError> {
        let url = format!("{}/chat/completions", self.base_url);
        let headers = self.build_headers()?;

        let body = PerplexityChatRequest {
            model: self.model.clone(),
            messages: vec![PerplexityRequestMessage {
                role: "user".to_string(),
                content: request.query.clone(),
            }],
            max_tokens: request
                .max_results
                .map(|n| u32::from(n).saturating_mul(256)),
        };

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .map_err(|e| WebSearchError::Network(format!("request failed: {e}")))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().unwrap_or_default();
            return Err(Self::classify_http_error(status.as_u16(), &error_text));
        }

        let parsed: PerplexityChatResponse = response
            .json()
            .map_err(|e| WebSearchError::Parse(format!("failed to parse response: {e}")))?;

        let answer = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.clone());

        let results = parsed
            .citations
            .unwrap_or_default()
            .into_iter()
            .map(|url| WebSearchResult {
                title: url.clone(),
                url,
                content: String::new(),
                score: None,
                published_at: None,
                favicon: None,
                raw_content: None,
            })
            .collect();

        Ok(WebSearchResponse {
            provider: "perplexity".to_string(),
            query: request.query.clone(),
            answer,
            results,
            images: Vec::<WebSearchImage>::new(),
            response_time: None,
        })
    }
}

#[derive(Debug, Serialize)]
struct PerplexityChatRequest {
    model: String,
    messages: Vec<PerplexityRequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct PerplexityRequestMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct PerplexityChatResponse {
    #[serde(default)]
    choices: Vec<PerplexityChoice>,
    #[serde(default)]
    citations: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct PerplexityChoice {
    message: PerplexityMessage,
}

#[derive(Debug, Deserialize)]
struct PerplexityMessage {
    #[serde(default)]
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_chain() {
        let provider = PerplexitySearchProvider::new("test-key")
            .with_model("sonar-pro")
            .with_base_url("https://example.test");

        assert_eq!(provider.model, "sonar-pro");
        assert_eq!(provider.base_url, "https://example.test");
    }

    #[test]
    fn test_classify_401_is_auth_error() {
        let error = PerplexitySearchProvider::classify_http_error(401, "Unauthorized");
        assert!(matches!(error, WebSearchError::Auth(_)));
    }

    #[test]
    fn test_classify_429_is_rate_limit() {
        let error = PerplexitySearchProvider::classify_http_error(429, "Too Many Requests");
        assert!(matches!(error, WebSearchError::RateLimit(_)));
    }
}
