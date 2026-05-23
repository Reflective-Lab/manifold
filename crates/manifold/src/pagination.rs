// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Closure-driven pagination helper for any `WebFetchBackend`.
//!
//! Most paginated upstream APIs fit one of three patterns:
//!
//! - **Cursor / continuation token** (TED, GitHub, OpenSanctions): the
//!   response carries an opaque next-page token; the next request
//!   includes that token. The client doesn't know the total count
//!   ahead of time.
//! - **Offset / page-number** (USAspending list endpoints, many REST
//!   APIs): the client increments a `page` or `offset` parameter
//!   until the response is empty or the total count is reached.
//! - **RFC 5988 Link header** (GitHub, many WordPress APIs): the
//!   `Link: <...>; rel="next"` header tells the client the next URL.
//!
//! This module doesn't bake in any of those strategies. Instead it
//! provides a generic [`paginate`] function that the caller drives
//! with a closure: given the prior response, return the next request
//! (or `None` to stop). The strategy lives in the caller's closure so
//! each upstream can use whichever shape it actually exposes.
//!
//! ## Example: TED-style cursor pagination
//!
//! ```ignore
//! use manifold::{HttpFetchProvider, WebFetchBackend, WebFetchRequest};
//! use manifold::pagination::{paginate, PaginationConfig};
//!
//! let backend = HttpFetchProvider::new()?;
//! let initial = WebFetchRequest::new("https://api.example.com/search")?
//!     .with_body(r#"{"pageSize": 100}"#);
//!
//! let pages = paginate(
//!     &backend,
//!     initial,
//!     |prior| {
//!         let json: serde_json::Value = serde_json::from_str(&prior.body).ok()?;
//!         let token = json["iterationNextToken"].as_str()?;
//!         if token.is_empty() { return None; }
//!         let body = format!(r#"{{"pageSize": 100, "iterationNextToken": "{token}"}}"#);
//!         WebFetchRequest::new("https://api.example.com/search")
//!             .ok()
//!             .map(|r| r.with_body(body))
//!     },
//!     PaginationConfig::default().with_max_pages(50),
//! )?;
//! for page in pages {
//!     // Parse page.body into typed records.
//! }
//! ```

use std::thread;
use std::time::Duration;

use crate::search::{WebFetchBackend, WebFetchError, WebFetchRequest, WebFetchResponse};

/// Configuration for [`paginate`]. Defaults are conservative: 10
/// pages max with a 100 ms politeness pause between calls.
#[derive(Debug, Clone)]
pub struct PaginationConfig {
    /// Maximum number of pages to fetch. Stops at this count even if
    /// the advance closure would return another request — prevents
    /// runaway loops on misbehaving upstreams.
    pub max_pages: usize,
    /// Pause between successive page fetches. Set to `Duration::ZERO`
    /// to disable. Politeness floor that keeps consumers off
    /// rate-limit triggers when paging through large datasets.
    pub politeness_delay: Duration,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            max_pages: 10,
            politeness_delay: Duration::from_millis(100),
        }
    }
}

impl PaginationConfig {
    #[must_use]
    pub fn with_max_pages(mut self, max_pages: usize) -> Self {
        self.max_pages = max_pages;
        self
    }

    #[must_use]
    pub fn with_politeness_delay(mut self, delay: Duration) -> Self {
        self.politeness_delay = delay;
        self
    }
}

/// Error type for pagination loops. Either the underlying fetch
/// failed, the caller's advance closure produced a request that
/// itself failed to build, or the page cap was reached.
#[derive(Debug, thiserror::Error)]
pub enum PaginationError {
    #[error("fetch failed on page {page}: {source}")]
    Fetch {
        page: usize,
        #[source]
        source: WebFetchError,
    },
    #[error("advance closure failed on page {page}: {message}")]
    Advance { page: usize, message: String },
    #[error("hit max_pages={max} without exhausting the upstream")]
    MaxPagesReached { max: usize },
}

/// Fetch the initial request and then keep calling `next_request`
/// until it returns `None`. Returns every page's response in order.
///
/// The caller supplies the strategy: cursor token extraction,
/// offset increment, link-header parse, etc. The closure receives
/// the prior response and returns the next [`WebFetchRequest`]
/// (or `None` to stop, or `Some(Err(...))` if the response indicated
/// a malformed pagination token).
///
/// Stops when:
/// - `next_request` returns `None`.
/// - `config.max_pages` has been reached. (Errors with
///   [`PaginationError::MaxPagesReached`] rather than silently
///   truncating — silent truncation under a paging cap is exactly
///   the kind of thing that produces audit gaps.)
/// - Any fetch fails (the error is surfaced; prior pages are
///   discarded).
pub fn paginate<F>(
    backend: &impl WebFetchBackend,
    initial: WebFetchRequest,
    mut next_request: F,
    config: PaginationConfig,
) -> Result<Vec<WebFetchResponse>, PaginationError>
where
    F: FnMut(&WebFetchResponse) -> Option<Result<WebFetchRequest, String>>,
{
    if config.max_pages == 0 {
        return Err(PaginationError::MaxPagesReached { max: 0 });
    }
    let mut pages: Vec<WebFetchResponse> = Vec::new();
    let mut current = initial;
    loop {
        let page_index = pages.len();
        let response = backend
            .fetch(&current)
            .map_err(|source| PaginationError::Fetch {
                page: page_index,
                source,
            })?;
        pages.push(response);

        if pages.len() >= config.max_pages {
            // Decide if there's a next page — if there is, the cap is
            // a real audit signal. If `next_request` says we're done
            // anyway, return cleanly.
            let last = pages.last().expect("just pushed");
            if next_request(last).is_some() {
                return Err(PaginationError::MaxPagesReached {
                    max: config.max_pages,
                });
            }
            break;
        }

        let last = pages.last().expect("just pushed");
        match next_request(last) {
            None => break,
            Some(Err(message)) => {
                return Err(PaginationError::Advance {
                    page: page_index,
                    message,
                });
            }
            Some(Ok(next)) => {
                if !config.politeness_delay.is_zero() {
                    thread::sleep(config.politeness_delay);
                }
                current = next;
            }
        }
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{WebFetchBackend, WebFetchError, WebFetchRequest, WebFetchResponse};
    use std::sync::Mutex;

    /// Programmable backend: each call returns the next response in
    /// the supplied script. Lets pagination logic be exercised
    /// without network or wiremock. Uses `Mutex` to satisfy the
    /// `Sync` bound on `WebFetchBackend`.
    struct ScriptedBackend {
        responses: Mutex<Vec<WebFetchResponse>>,
        captured_urls: Mutex<Vec<String>>,
    }

    impl ScriptedBackend {
        fn new(responses: Vec<WebFetchResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
                captured_urls: Mutex::new(Vec::new()),
            }
        }

        fn url_count(&self) -> usize {
            self.captured_urls.lock().unwrap().len()
        }
    }

    impl WebFetchBackend for ScriptedBackend {
        fn provider_name(&self) -> &'static str {
            "scripted_test"
        }

        fn fetch(&self, request: &WebFetchRequest) -> Result<WebFetchResponse, WebFetchError> {
            self.captured_urls
                .lock()
                .unwrap()
                .push(request.url.as_str().to_string());
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err(WebFetchError::Network("script exhausted".into()))
            } else {
                Ok(responses.remove(0))
            }
        }
    }

    fn json_response(url: &str, body: &str) -> WebFetchResponse {
        WebFetchResponse {
            url: url.to_string(),
            status: 200,
            content_type: Some("application/json".into()),
            body: body.to_string(),
            truncated: false,
        }
    }

    #[test]
    fn stops_when_advance_returns_none() {
        // Intent: a typical small-result paginated response that fits
        // on one page must return one page and stop — no extra fetch.
        let backend = ScriptedBackend::new(vec![json_response("https://x", r#"{"items":[1,2],"next":null}"#)]);
        let initial = WebFetchRequest::new("https://x").unwrap();
        let pages = paginate(
            &backend,
            initial,
            |_| None,
            PaginationConfig::default(),
        )
        .unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(backend.url_count(), 1);
    }

    #[test]
    fn cursor_strategy_threads_token_through_three_pages() {
        // Intent: cursor-based pagination drives the advance closure
        // off the response body. Each page's token is the input to
        // the next request. Three pages then a None token = stop.
        let backend = ScriptedBackend::new(vec![
            json_response("https://x", r#"{"items":[1,2],"next":"AAA"}"#),
            json_response("https://x", r#"{"items":[3,4],"next":"BBB"}"#),
            json_response("https://x", r#"{"items":[5,6],"next":null}"#),
        ]);
        let initial = WebFetchRequest::new("https://x").unwrap();
        let pages = paginate(
            &backend,
            initial,
            |prior| {
                let parsed: serde_json::Value = serde_json::from_str(&prior.body).ok()?;
                let token = parsed["next"].as_str()?;
                if token.is_empty() {
                    return None;
                }
                let body = format!(r#"{{"cursor":"{token}"}}"#);
                Some(
                    WebFetchRequest::new("https://x")
                        .map(|r| r.with_body(body))
                        .map_err(|e| e.to_string()),
                )
            },
            PaginationConfig::default().with_politeness_delay(Duration::ZERO),
        )
        .unwrap();
        assert_eq!(pages.len(), 3);
    }

    #[test]
    fn max_pages_with_more_pages_pending_is_an_error() {
        // Intent: silent truncation under a paging cap creates audit
        // gaps where downstream code thinks it saw everything. Surface
        // the cap as a typed error so a caller that wants partial
        // results can opt in explicitly via a higher max_pages.
        let backend = ScriptedBackend::new(vec![
            json_response("https://x", r#"{"next":"AAA"}"#),
            json_response("https://x", r#"{"next":"BBB"}"#),
        ]);
        let initial = WebFetchRequest::new("https://x").unwrap();
        let err = paginate(
            &backend,
            initial,
            |_| {
                Some(
                    WebFetchRequest::new("https://x").map_err(|e| e.to_string()),
                )
            },
            PaginationConfig::default()
                .with_max_pages(2)
                .with_politeness_delay(Duration::ZERO),
        )
        .unwrap_err();
        assert!(matches!(err, PaginationError::MaxPagesReached { max: 2 }));
    }

    #[test]
    fn max_pages_reached_exactly_when_advance_says_stop_returns_ok() {
        // Intent: hitting max_pages on the same iteration that the
        // advance closure would have stopped anyway should NOT error.
        // The cap is only a real signal when more pages remain.
        let backend = ScriptedBackend::new(vec![json_response("https://x", "{}")]);
        let initial = WebFetchRequest::new("https://x").unwrap();
        let pages = paginate(
            &backend,
            initial,
            |_| None,
            PaginationConfig::default().with_max_pages(1),
        )
        .unwrap();
        assert_eq!(pages.len(), 1);
    }

    #[test]
    fn advance_error_surfaces_with_page_index() {
        // Intent: a malformed cursor token in the response body should
        // produce a typed error with the page index so the caller can
        // pinpoint which upstream page broke the pagination chain.
        let backend = ScriptedBackend::new(vec![
            json_response("https://x", "{}"),
            json_response("https://x", "{}"),
        ]);
        let initial = WebFetchRequest::new("https://x").unwrap();
        let mut call_count = 0;
        let err = paginate(
            &backend,
            initial,
            |_prior| {
                call_count += 1;
                if call_count == 1 {
                    Some(WebFetchRequest::new("https://x").map_err(|e| e.to_string()))
                } else {
                    Some(Err("malformed cursor on page 1".to_string()))
                }
            },
            PaginationConfig::default().with_politeness_delay(Duration::ZERO),
        )
        .unwrap_err();
        match err {
            PaginationError::Advance { page, message } => {
                assert_eq!(page, 1);
                assert!(message.contains("malformed cursor"));
            }
            other => panic!("expected Advance error, got {other:?}"),
        }
    }

    #[test]
    fn fetch_failure_on_second_page_surfaces_with_page_index() {
        // Intent: an underlying fetch error mid-pagination must be
        // attributable to the right page. Auditing partial paginated
        // pulls needs to know which page broke.
        let backend = ScriptedBackend::new(vec![json_response("https://x", "{}")]);
        let initial = WebFetchRequest::new("https://x").unwrap();
        let err = paginate(
            &backend,
            initial,
            |_| Some(WebFetchRequest::new("https://x").map_err(|e| e.to_string())),
            PaginationConfig::default().with_politeness_delay(Duration::ZERO),
        )
        .unwrap_err();
        match err {
            PaginationError::Fetch { page, .. } => assert_eq!(page, 1),
            other => panic!("expected Fetch error, got {other:?}"),
        }
    }
}
