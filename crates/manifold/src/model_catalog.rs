// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Live model catalog fetched from OpenRouter's `/api/v1/models` endpoint.
//!
//! OpenRouter publishes a unified registry of ~300 models across providers
//! (Anthropic, OpenAI, Google, DeepSeek, etc.) with current pricing,
//! context windows, and supported parameters. This module fetches that
//! registry and caches it locally.
//!
//! # Usage
//!
//! ```ignore
//! use std::time::Duration;
//! use manifold::model_catalog::ModelCatalog;
//!
//! // Refresh if cache is older than 7 days, otherwise use cached copy.
//! let cache_path = ModelCatalog::default_cache_path().unwrap();
//! let catalog = ModelCatalog::load_or_refresh(&cache_path, Duration::from_secs(7 * 24 * 3600))?;
//!
//! // Look up live pricing for a specific model:
//! if let Some((prompt, completion)) = catalog.pricing_per_million("anthropic/claude-sonnet-4") {
//!     println!("Prompt: ${prompt}/M tokens, Completion: ${completion}/M tokens");
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Error type for catalog operations.
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    /// Network or HTTP failure when fetching from OpenRouter.
    #[error("network error: {0}")]
    Network(String),
    /// Failed to parse the catalog JSON.
    #[error("parse error: {0}")]
    Parse(String),
    /// Filesystem error reading/writing the cache.
    #[error("io error: {0}")]
    Io(String),
}

/// Pricing block from OpenRouter (USD per token, not per million).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogPricing {
    /// USD per prompt token.
    #[serde(default, deserialize_with = "deserialize_price")]
    pub prompt: f64,
    /// USD per completion token.
    #[serde(default, deserialize_with = "deserialize_price")]
    pub completion: f64,
    /// USD per input image (when supported).
    #[serde(default, deserialize_with = "deserialize_price_opt")]
    pub image: Option<f64>,
    /// USD per request (when applicable).
    #[serde(default, deserialize_with = "deserialize_price_opt")]
    pub request: Option<f64>,
}

/// Architecture metadata for a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogArchitecture {
    /// Input modalities, e.g. `["text", "image"]`.
    #[serde(default)]
    pub input_modalities: Vec<String>,
    /// Output modalities, e.g. `["text"]`.
    #[serde(default)]
    pub output_modalities: Vec<String>,
    /// Tokenizer family (e.g. `"Claude"`, `"GPT"`).
    #[serde(default)]
    pub tokenizer: String,
    /// Optional instruct type tag.
    #[serde(default)]
    pub instruct_type: Option<String>,
}

/// One model entry in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// OpenRouter ID, e.g. `"anthropic/claude-sonnet-4"`.
    pub id: String,
    /// Human-readable name.
    #[serde(default)]
    pub name: String,
    /// Max context length in tokens.
    #[serde(default)]
    pub context_length: usize,
    /// Current pricing.
    #[serde(default)]
    pub pricing: CatalogPricing,
    /// Architecture metadata (modalities, tokenizer).
    #[serde(default)]
    pub architecture: Option<CatalogArchitecture>,
    /// Request parameters the model accepts (`"tools"`, `"response_format"`, etc.).
    #[serde(default)]
    pub supported_parameters: Vec<String>,
}

impl CatalogEntry {
    /// True if this model accepts `tools` in the request (function calling).
    #[must_use]
    pub fn supports_tools(&self) -> bool {
        self.supported_parameters
            .iter()
            .any(|p| p == "tools" || p == "tool_choice")
    }

    /// True if the model accepts `response_format` (JSON mode / structured output).
    #[must_use]
    pub fn supports_structured_output(&self) -> bool {
        self.supported_parameters
            .iter()
            .any(|p| p == "response_format" || p == "structured_outputs")
    }

    /// True if input modalities include images.
    #[must_use]
    pub fn supports_vision(&self) -> bool {
        self.architecture
            .as_ref()
            .is_some_and(|a| a.input_modalities.iter().any(|m| m == "image"))
    }

    /// True if this model supports a reasoning parameter (CoT / thinking).
    #[must_use]
    pub fn supports_reasoning(&self) -> bool {
        self.supported_parameters
            .iter()
            .any(|p| p == "reasoning" || p == "include_reasoning")
    }
}

/// Full catalog fetched from OpenRouter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalog {
    /// Unix timestamp (seconds) when this catalog was last fetched from the network.
    pub fetched_at: u64,
    /// Entries keyed by OpenRouter ID.
    pub entries: HashMap<String, CatalogEntry>,
}

impl ModelCatalog {
    /// Default cache location: `~/.cache/manifold/openrouter-catalog.json`.
    ///
    /// Returns `None` if `HOME` is unset.
    #[must_use]
    pub fn default_cache_path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|h| {
            let mut p = PathBuf::from(h);
            p.push(".cache");
            p.push("manifold");
            p.push("openrouter-catalog.json");
            p
        })
    }

    /// Load from disk if the cache is fresher than `ttl`; otherwise refresh from network
    /// and write the result back to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if both disk read and network fetch fail.
    pub fn load_or_refresh(cache_path: &Path, ttl: Duration) -> Result<Self, CatalogError> {
        if let Ok(content) = std::fs::read_to_string(cache_path)
            && let Ok(cached) = serde_json::from_str::<Self>(&content)
        {
            let now = now_secs();
            if now.saturating_sub(cached.fetched_at) < ttl.as_secs() {
                return Ok(cached);
            }
        }
        let fresh = Self::refresh_from_network()?;
        let _ = fresh.save(cache_path);
        Ok(fresh)
    }

    /// Load from disk only — no network. Errors if the cache is missing or unparseable.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::Io` if the cache file doesn't exist or is unreadable.
    /// Returns `CatalogError::Parse` if the file is malformed.
    pub fn load_from_disk(cache_path: &Path) -> Result<Self, CatalogError> {
        let content =
            std::fs::read_to_string(cache_path).map_err(|e| CatalogError::Io(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| CatalogError::Parse(e.to_string()))
    }

    /// Fetch a fresh catalog from `https://openrouter.ai/api/v1/models`.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::Network` on HTTP failure or non-2xx status.
    /// Returns `CatalogError::Parse` if the response body can't be deserialized.
    pub fn refresh_from_network() -> Result<Self, CatalogError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| CatalogError::Network(e.to_string()))?;

        let response = client
            .get("https://openrouter.ai/api/v1/models")
            .send()
            .map_err(|e| CatalogError::Network(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            return Err(CatalogError::Network(format!(
                "HTTP {}",
                status.as_u16()
            )));
        }

        let body: OpenRouterModelsResponse = response
            .json()
            .map_err(|e| CatalogError::Parse(e.to_string()))?;

        let entries: HashMap<String, CatalogEntry> = body
            .data
            .into_iter()
            .map(|entry| (entry.id.clone(), entry))
            .collect();

        Ok(Self {
            fetched_at: now_secs(),
            entries,
        })
    }

    /// Write the catalog to disk, creating parent directories as needed.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::Io` on filesystem failure.
    pub fn save(&self, cache_path: &Path) -> Result<(), CatalogError> {
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CatalogError::Io(e.to_string()))?;
        }
        let json =
            serde_json::to_string_pretty(self).map_err(|e| CatalogError::Parse(e.to_string()))?;
        std::fs::write(cache_path, json).map_err(|e| CatalogError::Io(e.to_string()))
    }

    /// Look up an entry by OpenRouter ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.get(id)
    }

    /// Get pricing in USD per **million** tokens: `(prompt, completion)`.
    ///
    /// OpenRouter publishes prices per token; this multiplies by 1e6 for the
    /// more legible "per million" unit common in industry quotes.
    #[must_use]
    pub fn pricing_per_million(&self, id: &str) -> Option<(f64, f64)> {
        let entry = self.get(id)?;
        Some((
            entry.pricing.prompt * 1_000_000.0,
            entry.pricing.completion * 1_000_000.0,
        ))
    }

    /// How long ago this catalog was fetched.
    #[must_use]
    pub fn age(&self) -> Duration {
        Duration::from_secs(now_secs().saturating_sub(self.fetched_at))
    }
}

// ============================================================================
// Internal: OpenRouter API response shape
// ============================================================================

#[derive(Debug, Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<CatalogEntry>,
}

// OpenRouter sometimes returns prices as strings (e.g., "0.00000125"), sometimes
// as numbers. Accept both.
fn deserialize_price<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let v: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;
    match v {
        serde_json::Value::Null => Ok(0.0),
        serde_json::Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => {
            if s.is_empty() {
                return Ok(0.0);
            }
            s.parse().map_err(D::Error::custom)
        }
        _ => Ok(0.0),
    }
}

fn deserialize_price_opt<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let v: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;
    match v {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => Ok(n.as_f64()),
        serde_json::Value::String(s) => {
            if s.is_empty() {
                return Ok(None);
            }
            s.parse().map(Some).map_err(D::Error::custom)
        }
        _ => Ok(None),
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_string_pricing() {
        let json = r#"{
            "data": [
                {
                    "id": "test/model",
                    "name": "Test Model",
                    "context_length": 128000,
                    "pricing": {
                        "prompt": "0.00000125",
                        "completion": "0.0000025",
                        "image": null,
                        "request": null
                    },
                    "supported_parameters": ["tools", "response_format"]
                }
            ]
        }"#;
        let response: OpenRouterModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 1);
        let entry = &response.data[0];
        assert!((entry.pricing.prompt - 0.000_001_25).abs() < 1e-12);
        assert!((entry.pricing.completion - 0.000_002_5).abs() < 1e-12);
        assert!(entry.supports_tools());
        assert!(entry.supports_structured_output());
    }

    #[test]
    fn pricing_per_million_converts() {
        let entry = CatalogEntry {
            id: "x/y".to_string(),
            name: String::new(),
            context_length: 0,
            pricing: CatalogPricing {
                prompt: 0.000_001_25,
                completion: 0.000_002_5,
                image: None,
                request: None,
            },
            architecture: None,
            supported_parameters: vec![],
        };
        let mut entries = HashMap::new();
        entries.insert(entry.id.clone(), entry);
        let catalog = ModelCatalog {
            fetched_at: now_secs(),
            entries,
        };
        let (prompt, completion) = catalog.pricing_per_million("x/y").unwrap();
        assert!((prompt - 1.25).abs() < 1e-6);
        assert!((completion - 2.5).abs() < 1e-6);
    }
}
