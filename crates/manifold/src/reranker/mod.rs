// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! Reranker provider implementations for Converge.
//!
//! Rerankers compute fine-grained relevance scores between a query and
//! candidate items. This is the second stage in two-stage retrieval:
//!
//! 1. **Embedding recall**: Fast but coarse (vector similarity)
//! 2. **Reranking**: Slow but precise (cross-attention scoring)
//!
//! # Available Providers
//!
//! - [`QwenVLReranker`] - Qwen3-VL multimodal reranking
//! - `CohereReranker` - Cohere rerank API (future)
//!
//! # Example
//!
//! ```ignore
//! use manifold::reranker::QwenVLReranker;
//! use converge_core::capability::{Reranking, RerankRequest, EmbedInput};
//!
//! let reranker = QwenVLReranker::from_huggingface("your-api-key")?;
//!
//! let request = RerankRequest::text(
//!     "What is machine learning?",
//!     vec![
//!         "Machine learning is a subset of AI...".into(),
//!         "The weather today is sunny...".into(),
//!         "Deep learning uses neural networks...".into(),
//!     ],
//! ).with_top_k(2);
//!
//! let response = reranker.rerank(&request)?;
//! // response.ranked contains items sorted by relevance
//! ```

mod qwen_vl;

pub use qwen_vl::{QwenVLReranker, QwenVLRerankerEndpoint};

// Re-export core types for convenience
pub use converge_core::capability::{
    CapabilityError, EmbedInput, Modality, RankedItem, RerankRequest, RerankResponse, Reranking,
};
