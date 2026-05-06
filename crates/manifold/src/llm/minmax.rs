// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use crate::llm::OpenAiBackend;
use crate::secret::{EnvSecretProvider, SecretProvider};
use converge_core::backend::{BackendError, BackendResult};
use converge_provider::{BoxFuture, ChatBackend, ChatRequest, ChatResponse, LlmError};

/// MiniMax backend — OpenAI-compatible endpoint for MiniMax M2/abab models.
///
/// Configure with `MINMAX_API_KEY`. Default model: `MiniMax-Text-01`.
pub struct MinMaxBackend {
    inner: OpenAiBackend,
}

impl MinMaxBackend {
    pub fn from_env() -> BackendResult<Self> {
        Self::from_secret_provider(&EnvSecretProvider)
    }

    pub fn from_secret_provider(secrets: &dyn SecretProvider) -> BackendResult<Self> {
        let api_key =
            secrets
                .get_secret("MINMAX_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("MINMAX_API_KEY: {e}"),
                })?;
        Ok(Self {
            inner: OpenAiBackend::new(api_key.expose().to_string())
                .with_base_url("https://api.minimax.chat/v1")
                .with_model("MiniMax-Text-01"),
        })
    }

    #[must_use]
    pub fn with_model(self, model: impl Into<String>) -> Self {
        Self {
            inner: self.inner.with_model(model),
        }
    }
}

impl ChatBackend for MinMaxBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, LlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        self.inner.chat(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::StaticSecretProvider;

    #[test]
    fn constructs_from_secret_provider() {
        let backend =
            MinMaxBackend::from_secret_provider(&StaticSecretProvider::new("test-key")).unwrap();
        let _ = backend.with_model("abab7-chat");
    }

    #[test]
    fn missing_key_returns_unavailable() {
        use crate::secret::{SecretError, SecretProvider, SecretString};

        struct Empty;
        impl SecretProvider for Empty {
            fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
                Err(SecretError::NotFound(key.to_string()))
            }
        }

        let err = MinMaxBackend::from_secret_provider(&Empty).err().unwrap();
        assert!(matches!(
            err,
            converge_core::backend::BackendError::Unavailable { .. }
        ));
    }
}
