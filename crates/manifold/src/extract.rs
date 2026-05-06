// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Generic HTML extraction backend.
//!
//! `WebFetchBackend` (in [`crate::search`]) gets HTML *bytes* over HTTP.
//! This module turns those bytes into structured data via CSS selectors —
//! the *parse* half of "scrape a web page". The pair forms the standard
//! Reflective-Lab pattern for any project that needs to read information
//! out of public HTML (financial filings, regulatory filings, news pages,
//! research papers, etc.).
//!
//! ## Boundary
//!
//! The trait is deliberately narrow: take HTML, take CSS selectors, return
//! per-match text + attributes. Domain extractors (e.g. "find Item 1A
//! risk-factor headings in a 10-K") layer on top by *picking the right
//! selectors* and interpreting the matches. They live in their own
//! domain crate, not here.

use std::collections::HashMap;

use scraper::{Html, Selector};

/// One match produced by an [`HtmlExtractBackend`].
///
/// Attributes preserve insertion order from the source HTML (HashMap is
/// fine for lookup; if ordering matters in your downstream domain logic,
/// re-extract via the underlying scraper crate directly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedNode {
    /// The CSS selector that produced this match (for traceability when
    /// callers pass multiple selectors per request).
    pub selector: String,
    /// Inner text of the matched node, with whitespace collapsed and
    /// leading/trailing whitespace trimmed.
    pub text: String,
    /// Attributes of the matched element.
    pub attributes: HashMap<String, String>,
}

/// Errors from an HTML extraction operation.
#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    /// A selector string failed to parse (invalid CSS syntax).
    #[error("invalid CSS selector {selector:?}: {reason}")]
    InvalidSelector { selector: String, reason: String },
}

/// Generic contract for "extract structured data from HTML".
///
/// Implemented by concrete backends (e.g. [`ScraperHtmlBackend`]). Domain
/// projects depend on the trait, not the impl, so they can swap backends
/// (an LLM-based extractor, an XPath-based one, a remote service) without
/// touching their own code.
pub trait HtmlExtractBackend: Send + Sync {
    fn provider_name(&self) -> &'static str;

    /// Run each `selector` against `html` and return all matches.
    ///
    /// The returned vector preserves the order of selectors and, within a
    /// selector, the document order of matches.
    fn extract(
        &self,
        html: &str,
        selectors: &[&str],
    ) -> Result<Vec<ExtractedNode>, ExtractError>;
}

/// `scraper`-backed HTML extractor.
///
/// Parses with `html5ever` (via the `scraper` crate) — the same parser
/// used by Servo. Tolerant of malformed HTML, which financial filings
/// often are.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScraperHtmlBackend;

impl ScraperHtmlBackend {
    pub const fn new() -> Self {
        Self
    }
}

impl HtmlExtractBackend for ScraperHtmlBackend {
    fn provider_name(&self) -> &'static str {
        "scraper"
    }

    fn extract(
        &self,
        html: &str,
        selectors: &[&str],
    ) -> Result<Vec<ExtractedNode>, ExtractError> {
        let document = Html::parse_document(html);
        let mut out = Vec::new();
        for selector_str in selectors {
            let selector = Selector::parse(selector_str).map_err(|e| {
                ExtractError::InvalidSelector {
                    selector: (*selector_str).to_string(),
                    reason: e.to_string(),
                }
            })?;
            for element in document.select(&selector) {
                let text = collapse_whitespace(&element.text().collect::<String>());
                let attributes = element
                    .value()
                    .attrs()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                out.push(ExtractedNode {
                    selector: (*selector_str).to_string(),
                    text,
                    attributes,
                });
            }
        }
        Ok(out)
    }
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
<html>
  <body>
    <div class="risk-factor"><span style="font-weight:bold">First risk heading.</span></div>
    <div class="risk-factor"><span style="font-weight:bold">Second risk heading.</span></div>
    <p>Body text not part of any heading.</p>
    <a href="https://example.com" class="cite">Cited source</a>
  </body>
</html>
"#;

    #[test]
    fn extracts_text_from_class_selector() {
        let backend = ScraperHtmlBackend::new();
        let nodes = backend
            .extract(SAMPLE, &["div.risk-factor span"])
            .expect("extract");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].text, "First risk heading.");
        assert_eq!(nodes[1].text, "Second risk heading.");
        assert_eq!(nodes[0].selector, "div.risk-factor span");
    }

    #[test]
    fn preserves_attributes() {
        let backend = ScraperHtmlBackend::new();
        let nodes = backend.extract(SAMPLE, &["a.cite"]).expect("extract");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].attributes.get("href").map(String::as_str), Some("https://example.com"));
        assert_eq!(nodes[0].attributes.get("class").map(String::as_str), Some("cite"));
    }

    #[test]
    fn collapses_whitespace_in_text() {
        let html = r#"<p>  multiple    spaces
                       and newlines  </p>"#;
        let backend = ScraperHtmlBackend::new();
        let nodes = backend.extract(html, &["p"]).expect("extract");
        assert_eq!(nodes[0].text, "multiple spaces and newlines");
    }

    #[test]
    fn multiple_selectors_run_in_order() {
        let backend = ScraperHtmlBackend::new();
        let nodes = backend
            .extract(SAMPLE, &["a.cite", "div.risk-factor span"])
            .expect("extract");
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].selector, "a.cite");
        assert_eq!(nodes[1].selector, "div.risk-factor span");
        assert_eq!(nodes[2].selector, "div.risk-factor span");
    }

    #[test]
    fn invalid_selector_returns_error() {
        let backend = ScraperHtmlBackend::new();
        let err = backend
            .extract(SAMPLE, &["not a >> valid <<<< selector"])
            .expect_err("should fail");
        assert!(matches!(err, ExtractError::InvalidSelector { .. }));
    }

    #[test]
    fn no_matches_returns_empty_vec() {
        let backend = ScraperHtmlBackend::new();
        let nodes = backend.extract(SAMPLE, &["div.nonexistent"]).expect("extract");
        assert!(nodes.is_empty());
    }

    #[test]
    fn provider_name_is_stable() {
        assert_eq!(ScraperHtmlBackend::new().provider_name(), "scraper");
    }
}
