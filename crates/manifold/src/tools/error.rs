// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Error types for tool operations.

use thiserror::Error;

/// Error kinds for tool operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolErrorKind {
    NotFound,
    InvalidArguments,
    ExecutionFailed,
    ConnectionFailed,
    ServerError,
    Timeout,
    ValidationFailed,
    PermissionDenied,
    UnsupportedSource,
}

impl ToolErrorKind {
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::ConnectionFailed | Self::ServerError | Self::Timeout
        )
    }
}

/// Error type for tool operations.
#[derive(Debug, Error)]
#[error("{kind:?}: {message}")]
pub struct ToolError {
    pub kind: ToolErrorKind,
    pub message: String,
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl ToolError {
    #[must_use]
    pub fn new(kind: ToolErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    #[must_use]
    pub fn not_found(tool_name: impl Into<String>) -> Self {
        let name = tool_name.into();
        Self::new(ToolErrorKind::NotFound, format!("Tool not found: {name}"))
    }

    #[must_use]
    pub fn invalid_arguments(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::InvalidArguments, message)
    }

    #[must_use]
    pub fn connection_failed(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::ConnectionFailed, message)
    }

    #[must_use]
    pub fn validation_failed(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::ValidationFailed, message)
    }

    #[must_use]
    pub fn unsupported_source(source_type: impl Into<String>) -> Self {
        let source = source_type.into();
        Self::new(
            ToolErrorKind::UnsupportedSource,
            format!("Unsupported tool source: {source}"),
        )
    }

    #[must_use]
    pub fn is_retryable(&self) -> bool {
        self.kind.is_retryable()
    }
}
