// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Shared exponential-backoff retry helper for LLM HTTP backends.
//!
//! All backends share the same policy: up to `max_retries` retries with a
//! base delay of 100 ms doubled on each attempt (`100ms * 2^attempt`).
//! The caller supplies a closure that executes one HTTP attempt and returns a
//! [`RetryOutcome`] describing whether to succeed, retry, or fail immediately.

use std::future::Future;
use std::time::Duration;

use converge_provider::LlmError as ChatLlmError;

/// Outcome of a single HTTP attempt, returned by the closure passed to
/// [`retry_with_backoff`].
pub(super) enum RetryOutcome<T> {
    /// The attempt succeeded; carry `value` out of the retry loop.
    Success(T),
    /// A transient error occurred (429, 5xx, network, parse); save `error` and
    /// retry on the next iteration.
    Retry(ChatLlmError),
    /// A permanent error occurred (4xx other than 429); abort immediately
    /// without further retries.
    Fail(ChatLlmError),
}

/// Run `attempt` up to `max_retries + 1` times with exponential backoff.
///
/// On each retry (attempt > 0) the helper sleeps `100ms * 2^attempt` before
/// calling `attempt` again. Returns the first `Success` value, the first
/// `Fail` error, or the last `Retry` error after all attempts are exhausted.
pub(super) async fn retry_with_backoff<T, F, Fut>(
    max_retries: usize,
    mut attempt: F,
) -> Result<T, ChatLlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = RetryOutcome<T>>,
{
    let mut last_error: Option<ChatLlmError> = None;

    for n in 0..=max_retries {
        if n > 0 {
            tokio::time::sleep(Duration::from_millis(100 * 2_u64.pow(n as u32))).await;
        }

        match attempt().await {
            RetryOutcome::Success(value) => return Ok(value),
            RetryOutcome::Retry(e) => last_error = Some(e),
            RetryOutcome::Fail(e) => return Err(e),
        }
    }

    Err(last_error.unwrap_or_else(|| ChatLlmError::ProviderError {
        message: "unknown error".to_string(),
        code: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn success_on_first_attempt_returns_value() {
        let result = retry_with_backoff(3, || async { RetryOutcome::Success(42u32) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn fail_aborts_immediately_without_retrying() {
        let mut calls = 0u32;
        let result = retry_with_backoff(3, || {
            calls += 1;
            async {
                RetryOutcome::<u32>::Fail(ChatLlmError::AuthDenied {
                    message: "denied".into(),
                })
            }
        })
        .await;
        assert!(result.is_err());
        // Only one call should have been made — Fail aborts immediately.
        assert_eq!(calls, 1, "Fail must not trigger retries");
    }

    #[tokio::test]
    async fn retry_exhaustion_returns_last_error() {
        let max_retries = 2;
        let result = retry_with_backoff(max_retries, || async {
            RetryOutcome::<u32>::Retry(ChatLlmError::ProviderError {
                message: "transient".into(),
                code: Some("503".into()),
            })
        })
        .await;
        let err = result.unwrap_err();
        // After exhausting retries the last Retry error must be returned.
        assert!(
            matches!(err, ChatLlmError::ProviderError { ref message, .. } if message == "transient"),
            "expected last retry error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn success_after_transient_failures_returns_value() {
        let mut calls = 0u32;
        let result = retry_with_backoff(3, || {
            calls += 1;
            let c = calls;
            async move {
                if c < 3 {
                    RetryOutcome::Retry(ChatLlmError::ProviderError {
                        message: "transient".into(),
                        code: None,
                    })
                } else {
                    RetryOutcome::Success("ok")
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls, 3);
    }
}
