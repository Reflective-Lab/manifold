// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! HTTP fetch provider — fetches a single URL and returns its content.

use std::io::Read;
use std::net::ToSocketAddrs;
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::search::{
    WebFetchBackend, WebFetchError, WebFetchMethod, WebFetchRequest, WebFetchResponse,
    reject_non_public_ip, validate_public_http_url,
};

const MAX_REDIRECTS: usize = 10;

/// HTTP-based web fetch provider.
///
/// Uses `reqwest` under the hood (the same HTTP stack as the search providers).
pub struct HttpFetchProvider {
    client: Client,
    user_agent: String,
}

impl std::fmt::Debug for HttpFetchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpFetchProvider")
            .field("user_agent", &self.user_agent)
            .finish_non_exhaustive()
    }
}

impl HttpFetchProvider {
    /// Creates a new `HttpFetchProvider`.
    ///
    /// # Errors
    ///
    /// Returns [`WebFetchError::Network`] if the underlying TLS stack fails to
    /// initialise (e.g. missing system certificate store).
    pub fn new() -> Result<Self, WebFetchError> {
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| WebFetchError::Network(format!("failed to build reqwest client: {e}")))?;
        Ok(Self {
            client,
            user_agent: format!("converge/{}", env!("CARGO_PKG_VERSION")),
        })
    }

    #[must_use]
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    fn validate_request(&self, request: &WebFetchRequest) -> Result<reqwest::Url, WebFetchError> {
        let url = request.url.parse()?;
        self.validate_outbound_url(&url)?;
        Ok(url)
    }

    fn validate_outbound_url(&self, url: &reqwest::Url) -> Result<(), WebFetchError> {
        validate_public_http_url(url)?;

        let host = url
            .host_str()
            .ok_or_else(|| WebFetchError::InvalidUrl("URL must include a host".into()))?;
        if host.parse::<std::net::IpAddr>().is_ok() {
            return Ok(());
        }

        let port = url
            .port_or_known_default()
            .ok_or_else(|| WebFetchError::InvalidUrl("URL must include a valid port".into()))?;
        let mut resolved_any = false;
        for addr in (host, port)
            .to_socket_addrs()
            .map_err(|e| WebFetchError::InvalidUrl(format!("failed to resolve host: {e}")))?
        {
            resolved_any = true;
            reject_non_public_ip(addr.ip())?;
        }
        if !resolved_any {
            return Err(WebFetchError::InvalidUrl(
                "host resolved to no addresses".into(),
            ));
        }
        Ok(())
    }

    fn validate_redirect_url(
        &self,
        current_url: &reqwest::Url,
        location: &str,
    ) -> Result<reqwest::Url, WebFetchError> {
        let next_url = current_url
            .join(location)
            .map_err(|e| WebFetchError::InvalidUrl(e.to_string()))?;
        if current_url.scheme() == "https" && next_url.scheme() == "http" {
            return Err(WebFetchError::InvalidUrl(
                "https-to-http redirects are not allowed".into(),
            ));
        }
        self.validate_outbound_url(&next_url)?;
        Ok(next_url)
    }
}

impl WebFetchBackend for HttpFetchProvider {
    fn provider_name(&self) -> &'static str {
        "http"
    }

    fn fetch(&self, request: &WebFetchRequest) -> Result<WebFetchResponse, WebFetchError> {
        let mut url = self.validate_request(request)?;
        let original_url = url.clone();
        let max_bytes = request.max_bytes.get();
        let timeout_ms = request.timeout_ms.get();

        let mut headers = HeaderMap::new();
        for (name, value) in &request.headers {
            let name = HeaderName::try_from(name.as_str())
                .map_err(|e| WebFetchError::Network(format!("invalid header name: {e}")))?;
            let value = HeaderValue::from_str(value)
                .map_err(|e| WebFetchError::Network(format!("invalid header value: {e}")))?;
            headers.insert(name, value);
        }

        for redirect_count in 0..=MAX_REDIRECTS {
            let request_headers = caller_headers_for_url(&original_url, &url, &headers);
            let builder = match request.method {
                WebFetchMethod::Get => self.client.get(url.clone()),
                WebFetchMethod::Post => {
                    let mut b = self.client.post(url.clone());
                    if let Some(body) = request.body.as_ref() {
                        b = b.body(body.clone());
                    }
                    b
                }
            };
            let response = builder
                .timeout(Duration::from_millis(timeout_ms))
                .headers(request_headers)
                .header("User-Agent", &self.user_agent)
                .send()
                .map_err(|e| {
                    if e.is_timeout() {
                        WebFetchError::Timeout(timeout_ms)
                    } else {
                        WebFetchError::Network(e.to_string())
                    }
                })?;

            if response.status().is_redirection() {
                // Only GET follows redirects automatically. POST + redirect
                // is ambiguous (RFC 7231 §6.4: 301/302/303 typically
                // convert POST→GET; 307/308 must preserve method) — keep
                // the contract honest by surfacing the redirect status
                // back to the caller instead of guessing.
                if matches!(request.method, WebFetchMethod::Post) {
                    let status = response.status().as_u16();
                    let final_url = response.url().to_string();
                    let content_type = response
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .map(String::from);
                    let content_length = response.content_length();
                    let body = read_bounded_body(response, content_length, max_bytes)?;
                    return Ok(WebFetchResponse {
                        url: final_url,
                        status,
                        content_type,
                        body,
                        truncated: false,
                    });
                }
                if redirect_count == MAX_REDIRECTS {
                    return Err(WebFetchError::Network("too many redirects".into()));
                }
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        WebFetchError::InvalidUrl("redirect response missing Location".into())
                    })?;
                url = self.validate_redirect_url(&url, location)?;
                continue;
            }

            let status = response.status().as_u16();
            let final_url = response.url().to_string();
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(String::from);
            let content_length = response.content_length();

            let body = read_bounded_body(response, content_length, max_bytes)?;

            return Ok(WebFetchResponse {
                url: final_url,
                status,
                content_type,
                body,
                truncated: false,
            });
        }

        Err(WebFetchError::Network("too many redirects".into()))
    }
}

fn read_bounded_body(
    reader: impl Read,
    content_length: Option<u64>,
    max_bytes: usize,
) -> Result<String, WebFetchError> {
    if content_length.is_some_and(|len| len > max_bytes as u64) {
        return Err(WebFetchError::TooLarge(max_bytes));
    }

    let mut bytes = Vec::with_capacity(max_bytes.min(64 * 1024));
    let mut limited = reader.take(max_bytes as u64 + 1);
    limited
        .read_to_end(&mut bytes)
        .map_err(|e| WebFetchError::Network(e.to_string()))?;
    if bytes.len() > max_bytes {
        return Err(WebFetchError::TooLarge(max_bytes));
    }

    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn caller_headers_for_url(
    original_url: &reqwest::Url,
    current_url: &reqwest::Url,
    headers: &HeaderMap,
) -> HeaderMap {
    if same_origin(original_url, current_url) {
        headers.clone()
    } else {
        HeaderMap::new()
    }
}

fn same_origin(left: &reqwest::Url, right: &reqwest::Url) -> bool {
    left.scheme() == right.scheme()
        && left
            .host_str()
            .zip(right.host_str())
            .is_some_and(|(left, right)| left.eq_ignore_ascii_case(right))
        && left.port_or_known_default() == right.port_or_known_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn invalid_url_returns_error() {
        assert!(matches!(
            WebFetchRequest::new("not a url"),
            Err(WebFetchError::InvalidUrl(_))
        ));
    }

    #[test]
    fn unsafe_url_targets_are_rejected_before_fetch() {
        for url in [
            "file:///etc/passwd",
            "https://user:pass@example.com/",
            "http://localhost/",
            "http://localhost./",
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[2001:db8::1]/",
        ] {
            assert!(
                matches!(WebFetchRequest::new(url), Err(WebFetchError::InvalidUrl(_))),
                "{url} should be rejected"
            );
        }
    }

    #[test]
    fn invalid_limits_are_rejected_by_builder() {
        let zero_bytes = WebFetchRequest::new("https://example.com")
            .unwrap()
            .with_max_bytes(0);
        assert!(matches!(zero_bytes, Err(WebFetchError::InvalidLimit(_))));

        let huge_bytes = WebFetchRequest::new("https://example.com")
            .unwrap()
            .with_max_bytes(crate::search::MAX_WEB_FETCH_BYTES + 1);
        assert!(matches!(huge_bytes, Err(WebFetchError::InvalidLimit(_))));

        let zero_timeout = WebFetchRequest::new("https://example.com")
            .unwrap()
            .with_timeout_ms(0);
        assert!(matches!(zero_timeout, Err(WebFetchError::InvalidLimit(_))));
    }

    #[test]
    fn redirect_to_non_public_target_is_rejected() {
        let provider = HttpFetchProvider::new().unwrap();
        let current = reqwest::Url::parse("https://example.com/feed").unwrap();
        let result = provider.validate_redirect_url(&current, "http://127.0.0.1/admin");

        assert!(matches!(result, Err(WebFetchError::InvalidUrl(_))));
    }

    #[test]
    fn https_to_http_redirect_is_rejected() {
        let provider = HttpFetchProvider::new().unwrap();
        let current = reqwest::Url::parse("https://example.com/feed").unwrap();
        let result = provider.validate_redirect_url(&current, "http://example.com/feed");

        assert!(matches!(result, Err(WebFetchError::InvalidUrl(_))));
    }

    #[test]
    fn caller_headers_are_only_sent_to_original_origin() {
        let original = reqwest::Url::parse("https://example.com/feed").unwrap();
        let same_origin_url = reqwest::Url::parse("https://example.com/next").unwrap();
        let different_origin_url = reqwest::Url::parse("https://other.example/feed").unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );

        assert!(
            caller_headers_for_url(&original, &same_origin_url, &headers)
                .contains_key(reqwest::header::AUTHORIZATION)
        );
        assert!(caller_headers_for_url(&original, &different_origin_url, &headers).is_empty());
    }

    #[test]
    fn bounded_body_rejects_oversized_content_length() {
        let result = read_bounded_body(Cursor::new("small"), Some(10), 5);

        assert!(matches!(result, Err(WebFetchError::TooLarge(5))));
    }

    #[test]
    fn bounded_body_rejects_stream_past_limit() {
        let result = read_bounded_body(Cursor::new("abcdef"), None, 5);

        assert!(matches!(result, Err(WebFetchError::TooLarge(5))));
    }

    #[test]
    fn default_user_agent_contains_crate_version() {
        let provider = HttpFetchProvider::new().unwrap();
        assert!(provider.user_agent.starts_with("converge/"));
    }

    #[test]
    fn builder_overrides_user_agent() {
        let provider = HttpFetchProvider::new()
            .unwrap()
            .with_user_agent("test-agent/1.0");
        assert_eq!(provider.user_agent, "test-agent/1.0");
    }
}
