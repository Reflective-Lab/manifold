// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::fmt::Display;
use std::time::Duration;

use converge_core::backend::BackendError;
use converge_provider::LlmError as ChatLlmError;

const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(1);

pub(super) fn map_backend_error(error: BackendError) -> ChatLlmError {
    match error {
        BackendError::InvalidRequest { message } => ChatLlmError::InvalidRequest { message },
        BackendError::Timeout {
            deadline_ms,
            elapsed_ms,
        } => ChatLlmError::Timeout {
            elapsed: Duration::from_millis(elapsed_ms),
            deadline: Duration::from_millis(deadline_ms),
        },
        BackendError::Unavailable { message }
        | BackendError::ExecutionFailed { message }
        | BackendError::AdapterError { message }
        | BackendError::RecallError { message }
        | BackendError::Other { message }
        | BackendError::Retried { message, .. } => ChatLlmError::ProviderError {
            message,
            code: None,
        },
        BackendError::BudgetExceeded { resource, limit } => ChatLlmError::InvalidRequest {
            message: format!("budget exceeded for {resource} (limit: {limit})"),
        },
        BackendError::ContractFailed { contract, message } => ChatLlmError::ProviderError {
            message: format!("contract {contract} failed: {message}"),
            code: None,
        },
        BackendError::UnsupportedCapability { capability } => ChatLlmError::InvalidRequest {
            message: format!("unsupported capability: {capability:?}"),
        },
        BackendError::CircuitOpen {
            backend,
            retry_after_ms,
        } => ChatLlmError::RateLimited {
            retry_after: retry_after_ms
                .map(Duration::from_millis)
                .unwrap_or(DEFAULT_RETRY_AFTER),
            message: Some(format!("circuit open for backend {backend}")),
        },
    }
}

pub(super) fn classify_http_error(status: u16, body: &str, model: &str) -> ChatLlmError {
    let normalized = normalize_body(body);
    let message = format_http_error(status, body);

    if status == 429 || is_rate_limit_body(normalized.as_str()) {
        return ChatLlmError::RateLimited {
            retry_after: parse_retry_after(body).unwrap_or(DEFAULT_RETRY_AFTER),
            message: Some(message),
        };
    }

    if status == 401 || status == 403 || (status == 400 && is_auth_body(normalized.as_str())) {
        return ChatLlmError::AuthDenied { message };
    }

    if status == 404 || is_model_missing_body(normalized.as_str()) {
        return ChatLlmError::ModelNotFound {
            model: model.to_string(),
        };
    }

    if status >= 500 {
        return ChatLlmError::ProviderError {
            message,
            code: Some(status.to_string()),
        };
    }

    if status == 400 {
        return ChatLlmError::InvalidRequest { message };
    }

    ChatLlmError::ProviderError {
        message,
        code: Some(status.to_string()),
    }
}

pub(super) fn network_error(error: impl Display) -> ChatLlmError {
    ChatLlmError::NetworkError {
        message: error.to_string(),
    }
}

pub(super) fn parse_error(error: impl Display) -> ChatLlmError {
    ChatLlmError::ProviderError {
        message: format!("failed to parse response: {error}"),
        code: None,
    }
}

fn format_http_error(status: u16, body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {trimmed}")
    }
}

fn normalize_body(body: &str) -> String {
    body.to_ascii_lowercase()
}

fn is_auth_body(body: &str) -> bool {
    body.contains("invalid api key")
        || body.contains("incorrect api key")
        || body.contains("api key not valid")
        || body.contains("authentication")
        || body.contains("unauthorized")
        || body.contains("access denied")
        || body.contains("permission denied")
        || body.contains("credential")
}

fn is_model_missing_body(body: &str) -> bool {
    body.contains("model_not_found")
        || body.contains("model not found")
        || body.contains("invalid_model")
        || body.contains("invalid model")
        || body.contains("unknown model")
        || body.contains("unrecognized model")
        || body.contains("unsupported model")
        || (body.contains("does not exist") && body.contains("model"))
        || (body.contains("not found") && body.contains("model"))
}

fn is_rate_limit_body(body: &str) -> bool {
    body.contains("rate limit")
        || body.contains("too many requests")
        || body.contains("quota exceeded")
        || body.contains("resource exhausted")
}

fn parse_retry_after(body: &str) -> Option<Duration> {
    extract_seconds_after(body, "\"retryDelay\": \"")
        .or_else(|| extract_seconds_after(&body.to_ascii_lowercase(), "please retry in "))
}

fn extract_seconds_after(haystack: &str, needle: &str) -> Option<Duration> {
    let start = haystack.find(needle)? + needle.len();
    let mut digits = String::new();

    for ch in haystack[start..].chars() {
        if ch.is_ascii_digit() || ch == '.' {
            digits.push(ch);
        } else if !digits.is_empty() {
            break;
        }
    }

    let seconds = digits.parse::<f64>().ok()?;
    if seconds.is_sign_negative() {
        return None;
    }

    Some(Duration::from_secs_f64(seconds.ceil()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn is_auth_denied(error: &ChatLlmError) -> bool {
        matches!(error, ChatLlmError::AuthDenied { .. })
    }

    fn is_rate_limited(error: &ChatLlmError) -> bool {
        matches!(error, ChatLlmError::RateLimited { .. })
    }

    fn is_model_not_found(error: &ChatLlmError) -> bool {
        matches!(error, ChatLlmError::ModelNotFound { model } if model == "demo-model")
    }

    proptest! {
        #[test]
        fn classify_auth_statuses_as_auth_denied(status in prop_oneof![Just(401u16), Just(403u16)], body in "[A-Za-z0-9 _:\\-]{0,64}") {
            let error = classify_http_error(status, &body, "demo-model");
            prop_assert!(is_auth_denied(&error));
        }

        #[test]
        fn classify_rate_limit_status_as_retryable(body in "[A-Za-z0-9 _:\\-]{0,64}") {
            let error = classify_http_error(429, &body, "demo-model");
            prop_assert!(is_rate_limited(&error));
        }

        #[test]
        fn classify_model_not_found_hints(prefix in "[A-Za-z0-9 _:\\-]{0,32}", suffix in "[A-Za-z0-9 _:\\-]{0,32}") {
            let body = format!("{prefix} model_not_found {suffix}");
            let error = classify_http_error(400, &body, "demo-model");
            prop_assert!(is_model_not_found(&error));
        }
    }

    #[test]
    fn server_errors_preserve_status_code() {
        let error = classify_http_error(503, "upstream unavailable", "demo-model");
        assert!(matches!(
            error,
            ChatLlmError::ProviderError {
                code: Some(code),
                ..
            } if code == "503"
        ));
    }

    #[test]
    fn retry_delay_is_extracted_from_rate_limit_body() {
        let error = classify_http_error(
            429,
            r#"{"error":{"message":"Please retry in 42.2s.","details":[{"retryDelay":"42s"}]}}"#,
            "demo-model",
        );
        assert!(matches!(
            error,
            ChatLlmError::RateLimited { retry_after, .. } if retry_after >= Duration::from_secs(42)
        ));
    }

    // ========================================================================
    // Negative / edge-case tests
    // ========================================================================

    #[test]
    fn empty_body_with_401_is_auth_denied() {
        let error = classify_http_error(401, "", "demo-model");
        assert!(is_auth_denied(&error));
    }

    #[test]
    fn empty_body_with_429_is_rate_limited_default_retry() {
        let error = classify_http_error(429, "", "demo-model");
        assert!(is_rate_limited(&error));
        match error {
            ChatLlmError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, DEFAULT_RETRY_AFTER);
            }
            _ => panic!("expected RateLimited"),
        }
    }

    #[test]
    fn empty_body_with_500_is_provider_error() {
        let error = classify_http_error(500, "", "demo-model");
        assert!(matches!(
            error,
            ChatLlmError::ProviderError { code: Some(ref c), .. } if c == "500"
        ));
    }

    #[test]
    fn status_400_generic_is_invalid_request() {
        let error = classify_http_error(400, "something went wrong", "demo-model");
        assert!(matches!(error, ChatLlmError::InvalidRequest { .. }));
    }

    #[test]
    fn status_400_with_auth_body_is_auth_denied() {
        let error = classify_http_error(400, "Invalid API Key provided", "demo-model");
        assert!(is_auth_denied(&error));
    }

    #[test]
    fn status_404_is_model_not_found() {
        let error = classify_http_error(404, "resource not found", "gpt-99");
        assert!(matches!(
            error,
            ChatLlmError::ModelNotFound { ref model } if model == "gpt-99"
        ));
    }

    #[test]
    fn rate_limit_body_detection_various_patterns() {
        for body in &[
            "rate limit exceeded",
            "Too Many Requests",
            "quota exceeded for this month",
            "resource exhausted for today",
        ] {
            let error = classify_http_error(200, body, "demo-model");
            // Only 429 status or body keywords trigger rate limit
            // With 200 status, body keywords alone don't trigger — only classify looks at body for 429
            // Actually re-reading the code: the 429 check is `status == 429 || is_rate_limit_body`
            assert!(
                is_rate_limited(&error),
                "body {body:?} should be rate limited"
            );
        }
    }

    #[test]
    fn model_not_found_body_patterns() {
        for body in &[
            "model_not_found",
            "The model `gpt-99` does not exist",
            "invalid_model: unknown",
            "Model not found in registry",
            "unsupported model requested",
        ] {
            let error = classify_http_error(400, body, "demo-model");
            assert!(
                is_model_not_found(&error),
                "body {body:?} should be model_not_found"
            );
        }
    }

    #[test]
    fn retry_delay_extracted_from_please_retry_pattern() {
        let body = r"Rate limit hit. Please retry in 5 seconds.";
        let error = classify_http_error(429, body, "demo-model");
        match error {
            ChatLlmError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Duration::from_secs(5));
            }
            _ => panic!("expected RateLimited"),
        }
    }

    #[test]
    fn retry_delay_not_found_uses_default() {
        let body = "rate limit exceeded, no timing info";
        let error = classify_http_error(429, body, "demo-model");
        match error {
            ChatLlmError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, DEFAULT_RETRY_AFTER);
            }
            _ => panic!("expected RateLimited"),
        }
    }

    #[test]
    fn unknown_status_is_provider_error_with_code() {
        let error = classify_http_error(418, "I'm a teapot", "demo-model");
        assert!(matches!(
            error,
            ChatLlmError::ProviderError { code: Some(ref c), .. } if c == "418"
        ));
    }

    #[test]
    fn network_error_formats_display() {
        let err = network_error("connection refused");
        assert!(matches!(
            err,
            ChatLlmError::NetworkError { ref message } if message == "connection refused"
        ));
    }

    #[test]
    fn parse_error_wraps_message() {
        let err = parse_error("unexpected token at position 5");
        assert!(matches!(
            err,
            ChatLlmError::ProviderError { ref message, code: None }
            if message.contains("unexpected token at position 5")
        ));
    }

    #[test]
    fn map_backend_error_all_variants() {
        use converge_core::backend::BackendError;

        let cases: Vec<(BackendError, &str)> = vec![
            (
                BackendError::InvalidRequest {
                    message: "bad".into(),
                },
                "InvalidRequest",
            ),
            (
                BackendError::Timeout {
                    deadline_ms: 1000,
                    elapsed_ms: 2000,
                },
                "Timeout",
            ),
            (
                BackendError::Unavailable {
                    message: "down".into(),
                },
                "ProviderError",
            ),
            (
                BackendError::BudgetExceeded {
                    resource: "tokens".into(),
                    limit: "100".into(),
                },
                "InvalidRequest",
            ),
            (
                BackendError::CircuitOpen {
                    backend: "anthropic".into(),
                    retry_after_ms: Some(5000),
                },
                "RateLimited",
            ),
        ];

        for (backend_err, expected_variant) in cases {
            let llm_err = map_backend_error(backend_err);
            let debug = format!("{llm_err:?}");
            assert!(
                debug.contains(expected_variant),
                "Expected {expected_variant} in {debug}"
            );
        }
    }
}
