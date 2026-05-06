// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! Embedding provider implementations for Converge.
//!
//! This module provides embedding providers that generate vector representations
//! of text, images, and other modalities. These embeddings enable semantic
//! similarity search in vector stores.
//!
//! # Available Providers
//!
//! - [`QwenVLEmbedding`] - Qwen3-VL multimodal embeddings (text + images + video)
//! - `OpenAIEmbedding` - `OpenAI` text embeddings (future)
//! - Ollama embedding adapters can be added here when the local inference
//!   surface is ready.
//!
//! # Example
//!
//! ```ignore
//! use manifold::embedding::QwenVLEmbedding;
//! use converge_core::capability::{Embedding, EmbedRequest, EmbedInput};
//!
//! // Create provider with HuggingFace endpoint
//! let embedder = QwenVLEmbedding::from_huggingface("your-api-key")?;
//!
//! // Embed text and images together
//! let request = EmbedRequest::new(vec![
//!     EmbedInput::text("A product description"),
//!     EmbedInput::image_path("/path/to/product.png"),
//! ]).with_task("Represent this for retrieval");
//!
//! let response = embedder.embed(&request)?;
//! ```

mod qwen_vl;

pub use qwen_vl::{QwenVLEmbedding, QwenVLEndpoint};

// Re-export core types for convenience
pub use converge_core::capability::{
    CapabilityError, EmbedInput, EmbedRequest, EmbedResponse, EmbedUsage, Embedding, Modality,
};
