// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Secret management abstraction for provider API keys.
//!
//! This module defines `SecretProvider`, a trait for loading secrets
//! (API keys, tokens) from any backend. The default implementation
//! (`EnvSecretProvider`) reads from environment variables — suitable
//! for development, CI, and dedicated pilot environments.
//!
//! For production multi-tenant deployments, swap in a
//! `VaultSecretProvider` or `GcpSecretProvider` (not yet implemented).
//!
//! # Security properties
//!
//! - Secrets are never logged (no `Debug` on `SecretString`)
//! - With the `secure` feature, secrets are zeroed from memory on drop
//! - `SecretProvider` implementations must be `Send + Sync`

use thiserror::Error;

/// Error loading a secret.
#[derive(Debug, Error)]
pub enum SecretError {
    /// The requested secret was not found.
    #[error("secret not found: {0}")]
    NotFound(String),

    /// Access to the secret was denied.
    #[error("access denied: {0}")]
    AccessDenied(String),

    /// The secret backend is unavailable.
    #[error("backend unavailable: {0}")]
    Unavailable(String),
}

/// A string that holds a secret value.
///
/// - Never appears in `Debug` output
/// - With `secure` feature: zeroed from memory on drop via `zeroize`
#[derive(Clone)]
pub struct SecretString {
    inner: String,
}

impl SecretString {
    /// Wraps a string as a secret.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            inner: value.into(),
        }
    }

    /// Returns the secret value. Use sparingly — only at the point
    /// where the key is placed into an HTTP header or request body.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.inner
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl std::fmt::Display for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

#[cfg(feature = "secure")]
impl Drop for SecretString {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.inner.zeroize();
    }
}

/// Trait for loading secrets from a backend.
///
/// Implementations must be thread-safe. The default implementation
/// reads from environment variables.
pub trait SecretProvider: Send + Sync {
    /// Loads a secret by key name.
    ///
    /// # Errors
    ///
    /// Returns `SecretError` if the secret cannot be loaded.
    fn get_secret(&self, key: &str) -> Result<SecretString, SecretError>;

    /// Checks whether a secret exists without loading it.
    fn has_secret(&self, key: &str) -> bool {
        self.get_secret(key).is_ok()
    }
}

/// Loads secrets from environment variables.
///
/// This is the default for development, CI, and dedicated pilot
/// environments where each deployment has its own env.
#[derive(Debug, Default, Clone)]
pub struct EnvSecretProvider;

impl SecretProvider for EnvSecretProvider {
    fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
        std::env::var(key)
            .map(SecretString::new)
            .map_err(|_| SecretError::NotFound(key.to_string()))
    }
}

/// A static secret provider for testing.
///
/// Returns the same secret for any key. Never use in production.
#[derive(Clone)]
pub struct StaticSecretProvider {
    value: SecretString,
}

impl StaticSecretProvider {
    /// Creates a provider that always returns the given value.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: SecretString::new(value),
        }
    }
}

impl std::fmt::Debug for StaticSecretProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticSecretProvider")
            .field("value", &"[REDACTED]")
            .finish()
    }
}

impl SecretProvider for StaticSecretProvider {
    fn get_secret(&self, _key: &str) -> Result<SecretString, SecretError> {
        Ok(self.value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_string_debug_is_redacted() {
        let s = SecretString::new("super-secret-key");
        assert_eq!(format!("{s:?}"), "[REDACTED]");
    }

    #[test]
    fn secret_string_display_is_redacted() {
        let s = SecretString::new("super-secret-key");
        assert_eq!(format!("{s}"), "[REDACTED]");
    }

    #[test]
    fn secret_string_expose_returns_value() {
        let s = SecretString::new("my-key-123");
        assert_eq!(s.expose(), "my-key-123");
    }

    #[test]
    fn env_provider_returns_not_found_for_missing_var() {
        let provider = EnvSecretProvider;
        let result = provider.get_secret("CONVERGE_TEST_NONEXISTENT_KEY_12345");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), SecretError::NotFound(k) if k == "CONVERGE_TEST_NONEXISTENT_KEY_12345")
        );
    }

    #[test]
    fn static_provider_returns_value_for_any_key() {
        let provider = StaticSecretProvider::new("test-secret");
        let s1 = provider.get_secret("ANY_KEY").unwrap();
        let s2 = provider.get_secret("OTHER_KEY").unwrap();
        assert_eq!(s1.expose(), "test-secret");
        assert_eq!(s2.expose(), "test-secret");
    }

    #[test]
    fn static_provider_debug_is_redacted() {
        let provider = StaticSecretProvider::new("secret");
        let debug = format!("{provider:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn has_secret_delegates_to_get_secret() {
        let provider = StaticSecretProvider::new("val");
        assert!(provider.has_secret("anything"));

        let env_provider = EnvSecretProvider;
        assert!(!env_provider.has_secret("CONVERGE_TEST_NONEXISTENT_KEY_12345"));
    }
}
