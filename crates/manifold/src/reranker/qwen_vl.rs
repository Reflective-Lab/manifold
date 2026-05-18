// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! Qwen3-VL multimodal reranker provider.
//!
//! Qwen3-VL-Reranker computes fine-grained relevance scores for enhanced
//! retrieval accuracy. It supports multimodal reranking:
//! - Text query → text candidates
//! - Text query → image candidates
//! - Image query → text candidates
//! - Mixed modality queries and candidates
//!
//! # Two-Stage Retrieval
//!
//! ```text
//! Query → Embedding → Vector Store → Top 100 candidates
//!                                          ↓
//!                              Qwen3-VL-Reranker
//!                                          ↓
//!                                   Top 10 results
//! ```
//!
//! # Example
//!
//! ```ignore
//! use manifold::reranker::QwenVLReranker;
//! use converge_core::capability::{Reranking, RerankRequest};
//!
//! let reranker = QwenVLReranker::from_huggingface("hf_xxx")?;
//!
//! let response = reranker.rerank(&RerankRequest::text(
//!     "sustainable energy solutions",
//!     vec![
//!         "Solar panels reduce carbon emissions...".into(),
//!         "The stock market closed higher today...".into(),
//!         "Wind turbines generate clean electricity...".into(),
//!     ],
//! ))?;
//!
//! // Results sorted by relevance
//! assert!(response.ranked[0].score > response.ranked[1].score);
//! ```

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use converge_core::capability::{
    CapabilityError, CapabilityErrorKind, EmbedInput, Modality, RankedItem, RerankRequest,
    RerankResponse, Reranking,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default model name for Qwen3-VL reranker.
pub const DEFAULT_QWEN_VL_RERANKER_MODEL: &str = "Qwen/Qwen3-VL-Reranker";

/// Endpoint configuration for Qwen3-VL Reranker.
#[derive(Debug, Clone)]
pub enum QwenVLRerankerEndpoint {
    /// `HuggingFace` Inference API.
    HuggingFace { api_key: String, model: String },
    /// Alibaba Cloud `DashScope` API.
    AlibabaCloud { api_key: String, model: String },
    /// Local server (vLLM, etc.).
    Local { url: String, model: String },
}

impl QwenVLRerankerEndpoint {
    fn base_url(&self) -> &str {
        match self {
            Self::HuggingFace { .. } => "https://api-inference.huggingface.co",
            Self::AlibabaCloud { .. } => "https://dashscope.aliyuncs.com",
            Self::Local { url, .. } => url,
        }
    }

    fn model(&self) -> &str {
        match self {
            Self::HuggingFace { model, .. }
            | Self::AlibabaCloud { model, .. }
            | Self::Local { model, .. } => model,
        }
    }

    fn api_key(&self) -> Option<&str> {
        match self {
            Self::HuggingFace { api_key, .. } | Self::AlibabaCloud { api_key, .. } => Some(api_key),
            Self::Local { .. } => None,
        }
    }
}

/// Qwen3-VL multimodal reranker provider.
///
/// This provider implements fine-grained relevance scoring using
/// Qwen3-VL-Reranker. It supports multimodal queries and candidates.
pub struct QwenVLReranker {
    endpoint: QwenVLRerankerEndpoint,
    client: reqwest::blocking::Client,
}

impl QwenVLReranker {
    /// Creates a new reranker with custom endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`CapabilityError`] if the underlying TLS stack fails to
    /// initialise (e.g. missing system certificate store).
    pub fn new(endpoint: QwenVLRerankerEndpoint) -> Result<Self, CapabilityError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| CapabilityError::network(format!("failed to create HTTP client: {e}")))?;
        Ok(Self { endpoint, client })
    }

    /// Creates a reranker using `HuggingFace` Inference API.
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty.
    pub fn from_huggingface(api_key: impl Into<String>) -> Result<Self, CapabilityError> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(CapabilityError::auth("HuggingFace API key is required"));
        }
        Self::new(QwenVLRerankerEndpoint::HuggingFace {
            api_key,
            model: DEFAULT_QWEN_VL_RERANKER_MODEL.into(),
        })
    }

    /// Creates a reranker using `HuggingFace` with `HUGGINGFACE_API_KEY` env var.
    ///
    /// # Errors
    ///
    /// Returns error if the environment variable is not set.
    pub fn from_huggingface_env() -> Result<Self, CapabilityError> {
        let api_key = std::env::var("HUGGINGFACE_API_KEY").map_err(|_| {
            CapabilityError::auth("HUGGINGFACE_API_KEY environment variable not set")
        })?;
        Self::from_huggingface(api_key)
    }

    /// Creates a reranker using Alibaba Cloud `DashScope` API.
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty.
    pub fn from_alibaba_cloud(api_key: impl Into<String>) -> Result<Self, CapabilityError> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(CapabilityError::auth("Alibaba Cloud API key is required"));
        }
        Self::new(QwenVLRerankerEndpoint::AlibabaCloud {
            api_key,
            model: "qwen-vl-reranker-v1".into(),
        })
    }

    /// Creates a reranker using Alibaba Cloud with `DASHSCOPE_API_KEY` env var.
    ///
    /// # Errors
    ///
    /// Returns error if the environment variable is not set.
    pub fn from_alibaba_cloud_env() -> Result<Self, CapabilityError> {
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .map_err(|_| CapabilityError::auth("DASHSCOPE_API_KEY environment variable not set"))?;
        Self::from_alibaba_cloud(api_key)
    }

    /// Creates a reranker using a local server.
    ///
    /// # Errors
    ///
    /// Returns [`CapabilityError`] if the underlying HTTP client cannot be
    /// initialised.
    pub fn from_local(url: impl Into<String>) -> Result<Self, CapabilityError> {
        Self::new(QwenVLRerankerEndpoint::Local {
            url: url.into(),
            model: DEFAULT_QWEN_VL_RERANKER_MODEL.into(),
        })
    }

    /// Sets a custom model name.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        self.endpoint = match self.endpoint {
            QwenVLRerankerEndpoint::HuggingFace { api_key, .. } => {
                QwenVLRerankerEndpoint::HuggingFace { api_key, model }
            }
            QwenVLRerankerEndpoint::AlibabaCloud { api_key, .. } => {
                QwenVLRerankerEndpoint::AlibabaCloud { api_key, model }
            }
            QwenVLRerankerEndpoint::Local { url, .. } => {
                QwenVLRerankerEndpoint::Local { url, model }
            }
        };
        self
    }

    /// Converts an `EmbedInput` to string representation for API.
    #[allow(clippy::self_only_used_in_recursion)]
    fn input_to_content(&self, input: &EmbedInput) -> Result<RerankContent, CapabilityError> {
        match input {
            EmbedInput::Text(text) => Ok(RerankContent::Text { text: text.clone() }),

            EmbedInput::ImageBytes { data, mime_type } => {
                let base64_data = BASE64.encode(data);
                Ok(RerankContent::Image {
                    image: format!("data:{mime_type};base64,{base64_data}"),
                })
            }

            EmbedInput::ImagePath(path) => {
                let data = std::fs::read(path).map_err(|e| {
                    CapabilityError::invalid_input(format!(
                        "Failed to read image file {}: {}",
                        path.display(),
                        e
                    ))
                })?;
                let mime_type = guess_mime_type(path);
                let base64_data = BASE64.encode(&data);
                Ok(RerankContent::Image {
                    image: format!("data:{mime_type};base64,{base64_data}"),
                })
            }

            EmbedInput::VideoFrame { path, timestamp_ms } => {
                Err(CapabilityError::invalid_input(format!(
                    "Video frame extraction not implemented. Extract frame at {}ms from {} first",
                    timestamp_ms,
                    path.display()
                )))
            }

            EmbedInput::Mixed(inputs) => {
                // For mixed inputs, combine text and image
                let contents: Result<Vec<_>, _> =
                    inputs.iter().map(|i| self.input_to_content(i)).collect();
                Ok(RerankContent::Mixed {
                    contents: contents?,
                })
            }
        }
    }

    /// Calls the `HuggingFace` reranker API.
    fn call_huggingface(
        &self,
        query: &RerankContent,
        candidates: &[RerankContent],
        top_k: Option<usize>,
    ) -> Result<Vec<RankedItem>, CapabilityError> {
        let api_key = self
            .endpoint
            .api_key()
            .ok_or_else(|| CapabilityError::auth("API key required for HuggingFace"))?;

        let model = self.endpoint.model();
        let url = format!("{}/models/{}", self.endpoint.base_url(), model);

        let payload = HuggingFaceRerankRequest {
            query: query.clone(),
            candidates: candidates.to_vec(),
            top_k,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| CapabilityError::network(format!("HuggingFace request failed: {e}")))?;

        if response.status().is_success() {
            let results: Vec<HuggingFaceRerankResult> =
                response.json().map_err(|e| CapabilityError {
                    kind: CapabilityErrorKind::ProviderError,
                    message: format!("Failed to parse rerank response: {e}"),
                    retryable: false,
                })?;

            Ok(results
                .into_iter()
                .map(|r| RankedItem {
                    index: r.index,
                    score: r.score,
                })
                .collect())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(CapabilityError {
                kind: if status.as_u16() == 401 {
                    CapabilityErrorKind::Authentication
                } else if status.as_u16() == 429 {
                    CapabilityErrorKind::RateLimit
                } else {
                    CapabilityErrorKind::ProviderError
                },
                message: format!("HuggingFace returned {status}: {body}"),
                retryable: status.as_u16() == 429 || status.as_u16() >= 500,
            })
        }
    }

    /// Calls the Alibaba Cloud `DashScope` reranker API.
    fn call_alibaba_cloud(
        &self,
        query: &RerankContent,
        candidates: &[RerankContent],
        top_k: Option<usize>,
    ) -> Result<Vec<RankedItem>, CapabilityError> {
        let api_key = self
            .endpoint
            .api_key()
            .ok_or_else(|| CapabilityError::auth("API key required for Alibaba Cloud"))?;

        let url = format!(
            "{}/api/v1/services/rerank/multimodal-rerank/generation",
            self.endpoint.base_url()
        );

        // Convert to DashScope format
        let query_text = match query {
            RerankContent::Text { text } => text.clone(),
            _ => {
                return Err(CapabilityError::invalid_input(
                    "DashScope reranker currently only supports text queries",
                ));
            }
        };

        let documents: Vec<String> = candidates
            .iter()
            .filter_map(|c| match c {
                RerankContent::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();

        let payload = DashScopeRerankRequest {
            model: self.endpoint.model().to_string(),
            input: DashScopeRerankInput {
                query: query_text,
                documents,
            },
            parameters: DashScopeRerankParams {
                top_n: top_k,
                return_documents: false,
            },
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| CapabilityError::network(format!("DashScope request failed: {e}")))?;

        if response.status().is_success() {
            let result: DashScopeRerankResponse = response.json().map_err(|e| CapabilityError {
                kind: CapabilityErrorKind::ProviderError,
                message: format!("Failed to parse DashScope response: {e}"),
                retryable: false,
            })?;

            Ok(result
                .output
                .results
                .into_iter()
                .map(|r| RankedItem {
                    index: r.index,
                    score: r.relevance_score,
                })
                .collect())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(CapabilityError {
                kind: CapabilityErrorKind::ProviderError,
                message: format!("DashScope returned {status}: {body}"),
                retryable: status.as_u16() >= 500,
            })
        }
    }

    /// Calls a local Cohere-compatible reranker API.
    fn call_local(
        &self,
        query: &RerankContent,
        candidates: &[RerankContent],
        top_k: Option<usize>,
    ) -> Result<Vec<RankedItem>, CapabilityError> {
        let url = format!("{}/v1/rerank", self.endpoint.base_url());

        // Extract text for local API
        let query_text = match query {
            RerankContent::Text { text } => text.clone(),
            _ => {
                return Err(CapabilityError::invalid_input(
                    "Local reranker endpoint may only support text queries",
                ));
            }
        };

        let documents: Vec<String> = candidates
            .iter()
            .filter_map(|c| match c {
                RerankContent::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();

        if documents.is_empty() {
            return Err(CapabilityError::invalid_input(
                "No text candidates found for local reranker",
            ));
        }

        let payload = LocalRerankRequest {
            model: self.endpoint.model().to_string(),
            query: query_text,
            documents,
            top_n: top_k,
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| CapabilityError::network(format!("Local rerank request failed: {e}")))?;

        if response.status().is_success() {
            let result: LocalRerankResponse = response.json().map_err(|e| CapabilityError {
                kind: CapabilityErrorKind::ProviderError,
                message: format!("Failed to parse local rerank response: {e}"),
                retryable: false,
            })?;

            Ok(result
                .results
                .into_iter()
                .map(|r| RankedItem {
                    index: r.index,
                    score: r.relevance_score,
                })
                .collect())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(CapabilityError {
                kind: CapabilityErrorKind::ProviderError,
                message: format!("Local endpoint returned {status}: {body}"),
                retryable: status.as_u16() >= 500,
            })
        }
    }
}

impl Reranking for QwenVLReranker {
    fn name(&self) -> &'static str {
        "qwen-vl-reranker"
    }

    fn modalities(&self) -> Vec<Modality> {
        vec![Modality::Text, Modality::Image]
    }

    fn rerank(&self, request: &RerankRequest) -> Result<RerankResponse, CapabilityError> {
        if request.candidates.is_empty() {
            return Err(CapabilityError::invalid_input("No candidates provided"));
        }

        // Convert inputs
        let query = self.input_to_content(&request.query)?;
        let candidates: Result<Vec<_>, _> = request
            .candidates
            .iter()
            .map(|c| self.input_to_content(c))
            .collect();
        let candidates = candidates?;

        // Call appropriate endpoint
        let mut ranked = match &self.endpoint {
            QwenVLRerankerEndpoint::HuggingFace { .. } => {
                self.call_huggingface(&query, &candidates, request.top_k)?
            }
            QwenVLRerankerEndpoint::AlibabaCloud { .. } => {
                self.call_alibaba_cloud(&query, &candidates, request.top_k)?
            }
            QwenVLRerankerEndpoint::Local { .. } => {
                self.call_local(&query, &candidates, request.top_k)?
            }
        };

        // Sort by score descending
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply top_k if specified
        if let Some(k) = request.top_k {
            ranked.truncate(k);
        }

        // Apply min_score filter
        if let Some(min) = request.min_score {
            ranked.retain(|r| r.score >= min);
        }

        Ok(RerankResponse {
            ranked,
            model: self.endpoint.model().to_string(),
        })
    }
}

// =============================================================================
// API REQUEST/RESPONSE TYPES
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum RerankContent {
    Text { text: String },
    Image { image: String },
    Mixed { contents: Vec<RerankContent> },
}

#[derive(Debug, Serialize)]
struct HuggingFaceRerankRequest {
    query: RerankContent,
    candidates: Vec<RerankContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRerankResult {
    index: usize,
    score: f64,
}

#[derive(Debug, Serialize)]
struct DashScopeRerankRequest {
    model: String,
    input: DashScopeRerankInput,
    parameters: DashScopeRerankParams,
}

#[derive(Debug, Serialize)]
struct DashScopeRerankInput {
    query: String,
    documents: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DashScopeRerankParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
    return_documents: bool,
}

#[derive(Debug, Deserialize)]
struct DashScopeRerankResponse {
    output: DashScopeRerankOutput,
}

#[derive(Debug, Deserialize)]
struct DashScopeRerankOutput {
    results: Vec<DashScopeRerankResult>,
}

#[derive(Debug, Deserialize)]
struct DashScopeRerankResult {
    index: usize,
    relevance_score: f64,
}

#[derive(Debug, Serialize)]
struct LocalRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct LocalRerankResponse {
    results: Vec<LocalRerankResult>,
}

#[derive(Debug, Deserialize)]
struct LocalRerankResult {
    index: usize,
    relevance_score: f64,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn guess_mime_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        _ => "application/octet-stream",
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_configuration() {
        let hf = QwenVLReranker::from_huggingface("test-key").unwrap();
        assert_eq!(hf.endpoint.model(), DEFAULT_QWEN_VL_RERANKER_MODEL);
        assert_eq!(hf.endpoint.api_key(), Some("test-key"));

        let local = QwenVLReranker::from_local("http://localhost:8080").unwrap();
        assert!(local.endpoint.api_key().is_none());
    }

    #[test]
    fn modalities() {
        let reranker = QwenVLReranker::from_local("http://localhost:8080").unwrap();
        let modalities = reranker.modalities();
        assert!(modalities.contains(&Modality::Text));
        assert!(modalities.contains(&Modality::Image));
    }

    #[test]
    fn custom_model() {
        let reranker = QwenVLReranker::from_huggingface("key")
            .unwrap()
            .with_model("custom/model");
        assert_eq!(reranker.endpoint.model(), "custom/model");
    }

    #[test]
    fn text_content_conversion() {
        let reranker = QwenVLReranker::from_local("http://localhost:8080").unwrap();
        let content = reranker
            .input_to_content(&EmbedInput::text("Hello"))
            .unwrap();

        match content {
            RerankContent::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn empty_candidates_error() {
        let reranker = QwenVLReranker::from_local("http://localhost:8080").unwrap();
        let result = reranker.rerank(&RerankRequest::new(EmbedInput::text("query"), vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn requires_api_key() {
        let result = QwenVLReranker::from_huggingface("");
        assert!(result.is_err());

        let result = QwenVLReranker::from_alibaba_cloud("");
        assert!(result.is_err());
    }
}
