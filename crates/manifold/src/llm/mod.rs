// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Remote chat backend implementations for the canonical `ChatBackend` surface.
//!
//! This module provides async multi-turn provider adapters built on
//! `converge_provider::ChatBackend`. These are the canonical remote
//! LLM interfaces used by application and tool layers.
//!
//! # Architecture
//!
//! ```text
//! converge-provider
//!     │
//!     │  ChatBackend / DynChatBackend
//!     ▼
//! manifold::llm
//!     │
//!     ├── AnthropicBackend
//!     ├── OpenAIBackend
//!     └── ...               → RemoteTraceLink (audit-only)
//! ```
//!
//! # Canonical Surface
//!
//! The canonical remote surface is `ChatBackend` plus its dyn-safe wrapper
//! `DynChatBackend`. Local inference is assembled outside the foundation.

#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "arcee")]
mod arcee;
mod error_classification;
mod format_contract;
#[cfg(feature = "gemini")]
mod gemini;
#[cfg(feature = "kong")]
mod kong;
#[cfg(feature = "minmax")]
mod minmax;
#[cfg(feature = "mistral")]
mod mistral;
#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openrouter")]
mod openrouter;
mod resilient;
mod selection;
#[cfg(feature = "staik")]
mod staik;
#[cfg(feature = "writer")]
mod writer;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicBackend;
#[cfg(feature = "arcee")]
pub use arcee::ArceeBackend;
pub use converge_provider::{ChatBackendSelectionConfig, ChatBackendSelectionConfigError};
#[cfg(feature = "gemini")]
pub use gemini::GeminiBackend;
#[cfg(feature = "kong")]
pub use kong::KongBackend;
#[cfg(feature = "minmax")]
pub use minmax::MinMaxBackend;
#[cfg(feature = "mistral")]
pub use mistral::MistralBackend;
#[cfg(feature = "openai")]
pub use openai::OpenAiBackend;
#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterBackend;
pub use resilient::ResilientChatBackend;
pub use selection::{
    SelectedChatBackend, select_chat_backend, select_chat_backend_with_secret_provider,
    select_healthy_chat_backend, select_healthy_chat_backend_with_secret_provider,
};
#[cfg(feature = "staik")]
pub use staik::StaikBackend;
#[cfg(feature = "writer")]
pub use writer::WriterBackend;
