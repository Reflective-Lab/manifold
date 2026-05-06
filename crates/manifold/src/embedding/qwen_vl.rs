// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! Qwen3-VL multimodal embedding provider.
//!
//! Qwen3-VL-Embedding is a state-of-the-art multimodal embedding model that
//! generates semantically rich vector representations in a unified embedding
//! space for:
//! - Text
//! - Images
//! - Screenshots
//! - Video frames
//! - Mixed modality inputs
//!
//! # Key Features
//!
//! - **Unified embedding space**: Text and images share the same semantic space
//! - **Configurable dimensions**: Support for different embedding sizes
//! - **Task-specific instructions**: Customize embeddings for retrieval, classification, etc.
//! - **30+ languages**: Strong multilingual support
//!
//! # Architecture Note
//!
//! In Converge, Qwen3-VL is a **Tool-class component**, not an Suggestor:
//! - Produces candidates with scores, not decisions
//! - Output goes through validation before becoming facts
//! - Expands what agents can *see*, not what they can *decide*
//!
//! # Example
//!
//! ```ignore
//! use manifold::embedding::QwenVLEmbedding;
//! use converge_core::capability::{Embedding, EmbedRequest, EmbedInput};
//!
//! // Via HuggingFace Inference API
//! let embedder = QwenVLEmbedding::from_huggingface("hf_xxx")?;
//!
//! // Via local server (vLLM, text-embeddings-inference, etc.)
//! let embedder = QwenVLEmbedding::from_local("http://localhost:8080")?;
//!
//! // Embed multimodal content
//! let response = embedder.embed(&EmbedRequest::new(vec![
//!     EmbedInput::text("Product: Premium Headphones"),
//!     EmbedInput::image_path("/images/headphones.jpg"),
//! ]))?;
//! ```

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use converge_core::capability::{
    CapabilityError, CapabilityErrorKind, EmbedInput, EmbedRequest, EmbedResponse, EmbedUsage,
    Embedding, Modality,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default model name for Qwen3-VL embedding.
pub const DEFAULT_QWEN_VL_MODEL: &str = "Qwen/Qwen3-VL-Embedding";

/// Default embedding dimensions.
pub const DEFAULT_DIMENSIONS: usize = 1024;

/// Endpoint configuration for Qwen3-VL.
#[derive(Debug, Clone)]
pub enum QwenVLEndpoint {
    /// `HuggingFace` Inference API.
    HuggingFace { api_key: String, model: String },
    /// Alibaba Cloud `DashScope` API.
    AlibabaCloud { api_key: String, model: String },
    /// Local server (vLLM, text-embeddings-inference, etc.).
    Local { url: String, model: String },
}

impl QwenVLEndpoint {
    /// Returns the base URL for the endpoint.
    fn base_url(&self) -> &str {
        match self {
            Self::HuggingFace { .. } => "https://api-inference.huggingface.co",
            Self::AlibabaCloud { .. } => "https://dashscope.aliyuncs.com",
            Self::Local { url, .. } => url,
        }
    }

    /// Returns the model identifier.
    fn model(&self) -> &str {
        match self {
            Self::HuggingFace { model, .. }
            | Self::AlibabaCloud { model, .. }
            | Self::Local { model, .. } => model,
        }
    }

    /// Returns the API key if applicable.
    fn api_key(&self) -> Option<&str> {
        match self {
            Self::HuggingFace { api_key, .. } | Self::AlibabaCloud { api_key, .. } => Some(api_key),
            Self::Local { .. } => None,
        }
    }
}

/// Qwen3-VL multimodal embedding provider.
///
/// This provider implements state-of-the-art multimodal embeddings using
/// Qwen3-VL-Embedding. It supports text, images, and mixed inputs in a
/// unified semantic space.
pub struct QwenVLEmbedding {
    endpoint: QwenVLEndpoint,
    client: reqwest::blocking::Client,
    default_dimensions: usize,
}

impl QwenVLEmbedding {
    /// Creates a new provider with custom endpoint.
    ///
    /// # Panics
    ///
    /// Panics if the HTTP client cannot be created.
    #[must_use]
    pub fn new(endpoint: QwenVLEndpoint) -> Self {
        Self {
            endpoint,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            default_dimensions: DEFAULT_DIMENSIONS,
        }
    }

    /// Creates a provider using `HuggingFace` Inference API.
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty.
    pub fn from_huggingface(api_key: impl Into<String>) -> Result<Self, CapabilityError> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(CapabilityError::auth("HuggingFace API key is required"));
        }
        Ok(Self::new(QwenVLEndpoint::HuggingFace {
            api_key,
            model: DEFAULT_QWEN_VL_MODEL.into(),
        }))
    }

    /// Creates a provider using `HuggingFace` with `HUGGINGFACE_API_KEY` env var.
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

    /// Creates a provider using Alibaba Cloud `DashScope` API.
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty.
    pub fn from_alibaba_cloud(api_key: impl Into<String>) -> Result<Self, CapabilityError> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(CapabilityError::auth("Alibaba Cloud API key is required"));
        }
        Ok(Self::new(QwenVLEndpoint::AlibabaCloud {
            api_key,
            model: "qwen-vl-embedding-v1".into(),
        }))
    }

    /// Creates a provider using Alibaba Cloud with `DASHSCOPE_API_KEY` env var.
    ///
    /// # Errors
    ///
    /// Returns error if the environment variable is not set.
    pub fn from_alibaba_cloud_env() -> Result<Self, CapabilityError> {
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .map_err(|_| CapabilityError::auth("DASHSCOPE_API_KEY environment variable not set"))?;
        Self::from_alibaba_cloud(api_key)
    }

    /// Creates a provider using a local server.
    ///
    /// Compatible with:
    /// - vLLM with embedding support
    /// - text-embeddings-inference
    /// - Any OpenAI-compatible embedding API
    #[must_use]
    pub fn from_local(url: impl Into<String>) -> Self {
        Self::new(QwenVLEndpoint::Local {
            url: url.into(),
            model: DEFAULT_QWEN_VL_MODEL.into(),
        })
    }

    /// Sets the default embedding dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.default_dimensions = dimensions;
        self
    }

    /// Sets a custom model name.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        self.endpoint = match self.endpoint {
            QwenVLEndpoint::HuggingFace { api_key, .. } => {
                QwenVLEndpoint::HuggingFace { api_key, model }
            }
            QwenVLEndpoint::AlibabaCloud { api_key, .. } => {
                QwenVLEndpoint::AlibabaCloud { api_key, model }
            }
            QwenVLEndpoint::Local { url, .. } => QwenVLEndpoint::Local { url, model },
        };
        self
    }

    /// Converts an `EmbedInput` to the API request format.
    #[allow(clippy::self_only_used_in_recursion)]
    fn input_to_content(&self, input: &EmbedInput) -> Result<InputContent, CapabilityError> {
        match input {
            EmbedInput::Text(text) => Ok(InputContent::Text { text: text.clone() }),

            EmbedInput::ImageBytes { data, mime_type } => {
                let base64_data = BASE64.encode(data);
                Ok(InputContent::Image {
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
                Ok(InputContent::Image {
                    image: format!("data:{mime_type};base64,{base64_data}"),
                })
            }

            EmbedInput::VideoFrame { path, timestamp_ms } => {
                // For video frames, we'd need to extract the frame
                // For now, return an error suggesting to pre-extract frames
                Err(CapabilityError::invalid_input(format!(
                    "Video frame extraction not implemented. Extract frame at {}ms from {} and use ImagePath instead",
                    timestamp_ms,
                    path.display()
                )))
            }

            EmbedInput::Mixed(inputs) => {
                // Mixed inputs are handled by sending multiple content items
                // Convert each and collect
                let contents: Result<Vec<_>, _> =
                    inputs.iter().map(|i| self.input_to_content(i)).collect();
                Ok(InputContent::Mixed {
                    contents: contents?,
                })
            }
        }
    }

    /// Calls the `HuggingFace` Inference API.
    fn call_huggingface(
        &self,
        inputs: &[InputContent],
        task_instruction: Option<&str>,
        dimensions: usize,
    ) -> Result<Vec<Vec<f32>>, CapabilityError> {
        let api_key = self
            .endpoint
            .api_key()
            .ok_or_else(|| CapabilityError::auth("API key required for HuggingFace"))?;

        let model = self.endpoint.model();
        let url = format!(
            "{}/pipeline/feature-extraction/{}",
            self.endpoint.base_url(),
            model
        );

        // Build request payload
        let payload = HuggingFaceRequest {
            inputs: inputs.to_vec(),
            parameters: HuggingFaceParams {
                task_instruction: task_instruction.map(String::from),
                dimensions: Some(dimensions),
                normalize: true,
            },
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
            let embeddings: Vec<Vec<f32>> = response.json().map_err(|e| CapabilityError {
                kind: CapabilityErrorKind::ProviderError,
                message: format!("Failed to parse embeddings: {e}"),
                retryable: false,
            })?;
            Ok(embeddings)
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

    /// Calls the Alibaba Cloud `DashScope` API.
    fn call_alibaba_cloud(
        &self,
        inputs: &[InputContent],
        task_instruction: Option<&str>,
        _dimensions: usize,
    ) -> Result<Vec<Vec<f32>>, CapabilityError> {
        let api_key = self
            .endpoint
            .api_key()
            .ok_or_else(|| CapabilityError::auth("API key required for Alibaba Cloud"))?;

        let url = format!(
            "{}/api/v1/services/embeddings/multimodal-embedding/generation",
            self.endpoint.base_url()
        );

        // Build DashScope request format
        let contents: Vec<DashScopeContent> = inputs
            .iter()
            .map(|input| match input {
                InputContent::Text { text } => DashScopeContent {
                    text: Some(text.clone()),
                    image: None,
                },
                InputContent::Image { image } => DashScopeContent {
                    text: None,
                    image: Some(image.clone()),
                },
                InputContent::Mixed { contents } => {
                    // Flatten mixed content - take first text and first image
                    let text = contents.iter().find_map(|c| {
                        if let InputContent::Text { text } = c {
                            Some(text.clone())
                        } else {
                            None
                        }
                    });
                    let image = contents.iter().find_map(|c| {
                        if let InputContent::Image { image } = c {
                            Some(image.clone())
                        } else {
                            None
                        }
                    });
                    DashScopeContent { text, image }
                }
            })
            .collect();

        let payload = DashScopeRequest {
            model: self.endpoint.model().to_string(),
            input: DashScopeInput { contents },
            parameters: DashScopeParams {
                instruction: task_instruction.map(String::from),
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
            let result: DashScopeResponse = response.json().map_err(|e| CapabilityError {
                kind: CapabilityErrorKind::ProviderError,
                message: format!("Failed to parse DashScope response: {e}"),
                retryable: false,
            })?;
            Ok(result
                .output
                .embeddings
                .into_iter()
                .map(|e| e.embedding)
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

    /// Calls a local OpenAI-compatible API.
    fn call_local(
        &self,
        inputs: &[InputContent],
        _task_instruction: Option<&str>,
        _dimensions: usize,
    ) -> Result<Vec<Vec<f32>>, CapabilityError> {
        let url = format!("{}/v1/embeddings", self.endpoint.base_url());

        // Convert inputs to text (local servers may not support multimodal)
        let input_texts: Vec<String> = inputs
            .iter()
            .filter_map(|input| match input {
                InputContent::Text { text } => Some(text.clone()),
                InputContent::Image { .. } => {
                    tracing::warn!("Local endpoint may not support image embeddings");
                    None
                }
                InputContent::Mixed { contents } => contents.iter().find_map(|c| {
                    if let InputContent::Text { text } = c {
                        Some(text.clone())
                    } else {
                        None
                    }
                }),
            })
            .collect();

        if input_texts.is_empty() {
            return Err(CapabilityError::invalid_input(
                "No text inputs found for local embedding endpoint",
            ));
        }

        let payload = LocalEmbeddingRequest {
            model: self.endpoint.model().to_string(),
            input: input_texts,
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| {
                CapabilityError::network(format!("Local embedding request failed: {e}"))
            })?;

        if response.status().is_success() {
            let result: LocalEmbeddingResponse = response.json().map_err(|e| CapabilityError {
                kind: CapabilityErrorKind::ProviderError,
                message: format!("Failed to parse local embedding response: {e}"),
                retryable: false,
            })?;
            Ok(result.data.into_iter().map(|d| d.embedding).collect())
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

impl Embedding for QwenVLEmbedding {
    fn name(&self) -> &'static str {
        "qwen-vl-embedding"
    }

    fn modalities(&self) -> Vec<Modality> {
        vec![Modality::Text, Modality::Image, Modality::Video]
    }

    fn default_dimensions(&self) -> usize {
        self.default_dimensions
    }

    fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, CapabilityError> {
        if request.inputs.is_empty() {
            return Err(CapabilityError::invalid_input("No inputs provided"));
        }

        // Convert inputs to API format
        let contents: Result<Vec<InputContent>, _> = request
            .inputs
            .iter()
            .map(|input| self.input_to_content(input))
            .collect();
        let contents = contents?;

        let dimensions = request.dimensions.unwrap_or(self.default_dimensions);

        // Call appropriate endpoint
        let embeddings = match &self.endpoint {
            QwenVLEndpoint::HuggingFace { .. } => {
                self.call_huggingface(&contents, request.task_instruction.as_deref(), dimensions)?
            }
            QwenVLEndpoint::AlibabaCloud { .. } => {
                self.call_alibaba_cloud(&contents, request.task_instruction.as_deref(), dimensions)?
            }
            QwenVLEndpoint::Local { .. } => {
                self.call_local(&contents, request.task_instruction.as_deref(), dimensions)?
            }
        };

        // Normalize if requested
        let embeddings = if request.normalize {
            embeddings.into_iter().map(normalize_vector).collect()
        } else {
            embeddings
        };

        let actual_dimensions = embeddings.first().map(Vec::len).unwrap_or(0);

        Ok(EmbedResponse {
            embeddings,
            model: self.endpoint.model().to_string(),
            dimensions: actual_dimensions,
            usage: Some(EmbedUsage { total_tokens: 0 }),
        })
    }
}

// =============================================================================
// API REQUEST/RESPONSE TYPES
// =============================================================================

/// Content item for embedding requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum InputContent {
    Text { text: String },
    Image { image: String },
    Mixed { contents: Vec<InputContent> },
}

/// `HuggingFace` Inference API request.
#[derive(Debug, Serialize)]
struct HuggingFaceRequest {
    inputs: Vec<InputContent>,
    parameters: HuggingFaceParams,
}

#[derive(Debug, Serialize)]
struct HuggingFaceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    task_instruction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    normalize: bool,
}

/// Alibaba Cloud `DashScope` request.
#[derive(Debug, Serialize)]
struct DashScopeRequest {
    model: String,
    input: DashScopeInput,
    parameters: DashScopeParams,
}

#[derive(Debug, Serialize)]
struct DashScopeInput {
    contents: Vec<DashScopeContent>,
}

#[derive(Debug, Serialize)]
struct DashScopeContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
}

#[derive(Debug, Serialize)]
struct DashScopeParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    instruction: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DashScopeResponse {
    output: DashScopeOutput,
}

#[derive(Debug, Deserialize)]
struct DashScopeOutput {
    embeddings: Vec<DashScopeEmbedding>,
}

#[derive(Debug, Deserialize)]
struct DashScopeEmbedding {
    embedding: Vec<f32>,
}

/// Local OpenAI-compatible request.
#[derive(Debug, Serialize)]
struct LocalEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LocalEmbeddingResponse {
    data: Vec<LocalEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct LocalEmbeddingData {
    embedding: Vec<f32>,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Guesses MIME type from file extension.
fn guess_mime_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "application/octet-stream",
    }
}

/// Normalizes a vector to unit length.
fn normalize_vector(v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        v
    } else {
        v.into_iter().map(|x| x / norm).collect()
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
        let hf = QwenVLEmbedding::from_huggingface("test-key").unwrap();
        assert_eq!(hf.endpoint.model(), DEFAULT_QWEN_VL_MODEL);
        assert_eq!(hf.endpoint.api_key(), Some("test-key"));

        let local = QwenVLEmbedding::from_local("http://localhost:8080");
        assert!(local.endpoint.api_key().is_none());
    }

    #[test]
    fn modalities() {
        let embedder = QwenVLEmbedding::from_local("http://localhost:8080");
        let modalities = embedder.modalities();
        assert!(modalities.contains(&Modality::Text));
        assert!(modalities.contains(&Modality::Image));
        assert!(modalities.contains(&Modality::Video));
    }

    #[test]
    fn default_dimensions() {
        let embedder = QwenVLEmbedding::from_local("http://localhost:8080");
        assert_eq!(embedder.default_dimensions(), DEFAULT_DIMENSIONS);

        let embedder = embedder.with_dimensions(512);
        assert_eq!(embedder.default_dimensions(), 512);
    }

    #[test]
    fn custom_model() {
        let embedder = QwenVLEmbedding::from_huggingface("key")
            .unwrap()
            .with_model("custom/model");
        assert_eq!(embedder.endpoint.model(), "custom/model");
    }

    #[test]
    fn text_input_conversion() {
        let embedder = QwenVLEmbedding::from_local("http://localhost:8080");
        let content = embedder
            .input_to_content(&EmbedInput::text("Hello world"))
            .unwrap();

        match content {
            InputContent::Text { text } => assert_eq!(text, "Hello world"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn mime_type_guessing() {
        assert_eq!(guess_mime_type(Path::new("test.png")), "image/png");
        assert_eq!(guess_mime_type(Path::new("test.jpg")), "image/jpeg");
        assert_eq!(guess_mime_type(Path::new("test.jpeg")), "image/jpeg");
        assert_eq!(guess_mime_type(Path::new("test.webp")), "image/webp");
        assert_eq!(
            guess_mime_type(Path::new("test.unknown")),
            "application/octet-stream"
        );
    }

    #[test]
    fn vector_normalization() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(v);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn empty_input_error() {
        let embedder = QwenVLEmbedding::from_local("http://localhost:8080");
        let result = embedder.embed(&EmbedRequest::new(vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn requires_api_key() {
        let result = QwenVLEmbedding::from_huggingface("");
        assert!(result.is_err());

        let result = QwenVLEmbedding::from_alibaba_cloud("");
        assert!(result.is_err());
    }
}
