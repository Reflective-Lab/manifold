// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! YAML-based model registry loader.
//!
//! Loads model metadata from `config/models.yaml` and provides
//! a registry that can be used for model selection.
//!
//! # Example
//!
//! ```ignore
//! use manifold::registry_loader::{load_registry, RegistryConfig};
//!
//! // Load from default path
//! let registry = load_registry()?;
//!
//! // Check available providers
//! for provider in registry.providers() {
//!     println!("{}: {} (key: {})",
//!         provider.id,
//!         provider.api_url,
//!         if provider.is_available() { "set" } else { "missing" }
//!     );
//! }
//! ```

use crate::model_selection::{ModelMetadata, ModelSelector};
use converge_provider::selection::{ComplianceLevel, CostClass, DataSovereignty};
use schemars::JsonSchema;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Error type for registry loading.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Failed to read the YAML file.
    #[error("Failed to read registry file: {0}")]
    IoError(#[from] std::io::Error),

    /// Failed to parse the YAML.
    #[error("Failed to parse registry YAML: {0}")]
    ParseError(#[from] serde_yaml::Error),

    /// Validation error in the registry.
    #[error("Registry validation failed: {0}")]
    ValidationError(String),
}

// =============================================================================
// YAML SCHEMA TYPES (Type-safe with serde enums)
// =============================================================================

/// Root of the YAML file.
///
/// This is the schema for `config/models.yaml`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegistryYaml {
    /// All providers.
    pub providers: HashMap<String, ProviderYaml>,
}

/// Provider type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderTypeYaml {
    /// Direct API access to model provider (default).
    #[default]
    Direct,
    /// Routes to multiple underlying providers (adds latency overhead).
    Aggregator,
}

/// A provider in the YAML.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProviderYaml {
    /// Environment variable for API key.
    pub env_key: String,
    /// Optional secondary environment variable (e.g., Baidu secret key).
    #[serde(default)]
    pub env_key_secondary: Option<String>,
    /// URL to get an API key.
    pub key_url: String,
    /// API endpoint URL.
    pub api_url: String,
    /// ISO country code (2 letters) or "LOCAL".
    pub country: String,
    /// Region (US, EU, CN, LOCAL, etc.).
    pub region: RegionYaml,
    /// Compliance certifications.
    #[serde(default)]
    pub compliance: Vec<ComplianceYaml>,
    /// Provider type (direct or aggregator).
    #[serde(default)]
    pub provider_type: ProviderTypeYaml,
    /// Models provided.
    pub models: HashMap<String, ModelYaml>,
}

/// Region enum - type-safe parsing.
///
/// Represents the data residency region for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
pub enum RegionYaml {
    /// United States
    US,
    /// European Union
    EU,
    /// European Economic Area
    EEA,
    /// Switzerland
    CH,
    /// China
    CN,
    /// Japan
    JP,
    /// United Kingdom
    UK,
    /// Local/on-premises (any jurisdiction)
    LOCAL,
}

impl RegionYaml {
    /// Converts to string for storage.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::US => "US",
            Self::EU => "EU",
            Self::EEA => "EEA",
            Self::CH => "CH",
            Self::CN => "CN",
            Self::JP => "JP",
            Self::UK => "UK",
            Self::LOCAL => "LOCAL",
        }
    }
}

/// Compliance certification enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
pub enum ComplianceYaml {
    /// General Data Protection Regulation (EU)
    GDPR,
    /// Service Organization Control 2
    SOC2,
    /// Health Insurance Portability and Accountability Act
    HIPAA,
}

impl From<ComplianceYaml> for ComplianceLevel {
    fn from(c: ComplianceYaml) -> Self {
        match c {
            ComplianceYaml::GDPR => ComplianceLevel::GDPR,
            ComplianceYaml::SOC2 => ComplianceLevel::SOC2,
            ComplianceYaml::HIPAA => ComplianceLevel::HIPAA,
        }
    }
}

/// Cost class for model pricing tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
pub enum CostClassYaml {
    /// Very low cost (e.g., Haiku, GPT-3.5, local models)
    VeryLow,
    /// Low cost (e.g., Sonnet, GPT-4o)
    Low,
    /// Medium cost (e.g., GPT-4 Turbo)
    Medium,
    /// High cost (e.g., Opus, o1-mini)
    High,
    /// Very high cost (e.g., o1-preview)
    VeryHigh,
}

impl From<CostClassYaml> for CostClass {
    fn from(c: CostClassYaml) -> Self {
        match c {
            CostClassYaml::VeryLow => CostClass::VeryLow,
            CostClassYaml::Low => CostClass::Low,
            CostClassYaml::Medium => CostClass::Medium,
            CostClassYaml::High => CostClass::High,
            CostClassYaml::VeryHigh => CostClass::VeryHigh,
        }
    }
}

/// Model capability flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityYaml {
    /// Function/tool calling support
    ToolUse,
    /// Image understanding
    Vision,
    /// JSON mode / schema enforcement
    StructuredOutput,
    /// Code generation/understanding
    Code,
    /// Multi-step logical reasoning
    Reasoning,
    /// Good performance across languages
    Multilingual,
    /// Real-time web information retrieval
    WebSearch,
    /// Audio input/output support
    Audio,
    /// Image generation support
    ImageGeneration,
    /// Streaming responses
    Streaming,
    /// Logprobs support
    Logprobs,
    /// Deterministic seed support
    Seed,
    /// Tool choice (e.g., required/none/auto)
    ToolChoice,
    /// Parallel tool call support
    ParallelToolCalls,
    /// Prompt caching support
    PromptCaching,
    /// Built-in file search retrieval
    FileSearch,
    /// Built-in code interpreter / sandbox execution
    CodeInterpreter,
    /// Built-in browser automation / computer use
    ComputerUse,
    /// Tool-level web search (native search tool)
    ToolSearch,
    /// Model Context Protocol tool support
    Mcp,
    /// Hosted shell tool support
    HostedShell,
    /// Apply-patch tool support
    ApplyPatch,
    /// Native context compaction support
    NativeCompaction,
    /// Reasoning effort controls (e.g., low/medium/high)
    ReasoningEffort,
    /// Strong content generation / business writing
    ContentGeneration,
    /// Business acumen (financial, strategic, market analysis)
    BusinessAcumen,
}

impl CapabilityYaml {
    /// Stable `snake_case` string representation used in API responses.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ToolUse => "tool_use",
            Self::Vision => "vision",
            Self::StructuredOutput => "structured_output",
            Self::Code => "code",
            Self::Reasoning => "reasoning",
            Self::Multilingual => "multilingual",
            Self::WebSearch => "web_search",
            Self::Audio => "audio",
            Self::ImageGeneration => "image_generation",
            Self::Streaming => "streaming",
            Self::Logprobs => "logprobs",
            Self::Seed => "seed",
            Self::ToolChoice => "tool_choice",
            Self::ParallelToolCalls => "parallel_tool_calls",
            Self::PromptCaching => "prompt_caching",
            Self::FileSearch => "file_search",
            Self::CodeInterpreter => "code_interpreter",
            Self::ComputerUse => "computer_use",
            Self::ToolSearch => "tool_search",
            Self::Mcp => "mcp",
            Self::HostedShell => "hosted_shell",
            Self::ApplyPatch => "apply_patch",
            Self::NativeCompaction => "native_compaction",
            Self::ReasoningEffort => "reasoning_effort",
            Self::ContentGeneration => "content_generation",
            Self::BusinessAcumen => "business_acumen",
        }
    }
}

/// Supported reasoning effort level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffortYaml {
    /// Disable explicit chain-of-thought style effort controls.
    None,
    /// Minimal extra reasoning.
    Minimal,
    /// Low extra reasoning.
    Low,
    /// Medium extra reasoning.
    Medium,
    /// High extra reasoning.
    High,
    /// Extra-high reasoning.
    Xhigh,
}

impl ReasoningEffortYaml {
    /// Stable `snake_case` string representation used in API responses.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }
}

/// Model type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelTypeYaml {
    /// LLM for chat/completion (default)
    #[default]
    Llm,
    /// Vector embedding model
    Embedding,
    /// Cross-encoder reranking model
    Reranker,
    /// OCR / Document AI model
    Ocr,
}

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureYaml {
    /// Traditional transformer (all parameters active).
    #[default]
    Dense,
    /// Mixture of Experts (only subset active per forward pass).
    Moe,
    /// Hybrid architecture (e.g., Jamba's Mamba-Transformer).
    Hybrid,
}

/// Input modality type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModalityYaml {
    /// Text input/output.
    Text,
    /// Image input.
    Image,
    /// Video input.
    Video,
    /// Audio input.
    Audio,
}

/// Agentic capabilities configuration.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgenticYaml {
    /// Maximum number of parallel agents this model can orchestrate.
    #[serde(default)]
    pub max_parallel_agents: Option<u32>,
    /// Whether the model supports agent orchestration/swarm.
    #[serde(default)]
    pub supports_orchestration: bool,
}

/// Pricing information (USD per million tokens).
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PricingYaml {
    /// Input price per million tokens (USD).
    #[serde(default)]
    pub input_per_m: Option<f64>,
    /// Output price per million tokens (USD).
    #[serde(default)]
    pub output_per_m: Option<f64>,
}

/// Rate limit information (provider- or model-specific).
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RateLimitsYaml {
    /// Requests per minute.
    #[serde(default)]
    pub requests_per_min: Option<u32>,
    /// Tokens per minute.
    #[serde(default)]
    pub tokens_per_min: Option<u32>,
    /// Requests per day.
    #[serde(default)]
    pub requests_per_day: Option<u32>,
    /// Maximum concurrent requests.
    #[serde(default)]
    pub concurrent_requests: Option<u32>,
}

/// A model entry in the registry.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelYaml {
    /// Cost class - validated at parse time.
    pub cost_class: CostClassYaml,
    /// Typical latency in milliseconds (must be > 0).
    pub typical_latency_ms: u32,
    /// Quality score (must be 0.0-1.0).
    pub quality: f64,
    /// Context window size in tokens.
    #[serde(default = "default_context_tokens")]
    pub context_tokens: usize,
    /// Capabilities list - validated at parse time.
    #[serde(default)]
    pub capabilities: Vec<CapabilityYaml>,
    /// Model type - validated at parse time.
    #[serde(default, rename = "type")]
    pub model_type: ModelTypeYaml,
    /// Embedding dimensions (required for embedding models).
    #[serde(default)]
    pub dimensions: Option<usize>,

    // === ENRICHED SCHEMA ===
    /// Model architecture (dense, moe, hybrid).
    #[serde(default)]
    pub architecture: ArchitectureYaml,
    /// Total parameters in billions.
    #[serde(default)]
    pub total_params_b: Option<f64>,
    /// Active parameters per forward pass in billions (for `MoE` models).
    #[serde(default)]
    pub active_params_b: Option<f64>,
    /// Maximum output tokens.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    /// Whether the model is native multimodal (trained on mixed modalities).
    #[serde(default)]
    pub native_multimodal: bool,
    /// Supported input modalities.
    #[serde(default)]
    pub modalities: Vec<ModalityYaml>,
    /// Agentic/swarm capabilities.
    #[serde(default)]
    pub agentic: Option<AgenticYaml>,
    /// Whether the model supports extended thinking/reasoning mode.
    #[serde(default)]
    pub thinking_mode: bool,
    /// Supported reasoning effort levels (e.g., [low, medium, high]).
    #[serde(default)]
    pub reasoning_effort_levels: Vec<ReasoningEffortYaml>,
    /// Whether the model supports native context compaction.
    #[serde(default)]
    pub native_compaction: bool,
    /// Model ID of the thinking variant (if this is the base model).
    #[serde(default)]
    pub thinking_variant: Option<String>,
    /// Pricing information.
    #[serde(default)]
    pub pricing: Option<PricingYaml>,
    /// Model publisher or organization (e.g., `OpenAI`, Anthropic).
    #[serde(default)]
    pub publisher: Option<String>,
    /// Model family name (e.g., Claude, GPT, Llama).
    #[serde(default)]
    pub family: Option<String>,
    /// Release date (ISO-8601 format recommended).
    #[serde(default)]
    pub release_date: Option<String>,
    /// Training data cutoff date (ISO-8601 format recommended).
    #[serde(default)]
    pub training_cutoff: Option<String>,
    /// Whether model weights are openly available.
    #[serde(default)]
    pub open_weights: bool,
    /// License identifier or URL.
    #[serde(default)]
    pub license: Option<String>,
    /// Whether the model is deprecated.
    #[serde(default)]
    pub deprecated: bool,
    /// Whether the model is in beta/preview.
    #[serde(default)]
    pub beta: bool,
    /// Benchmark scores (keyed by benchmark name).
    #[serde(default)]
    pub benchmarks: HashMap<String, f64>,
    /// Free-form tags for routing or promotion.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Rate limit information (if published).
    #[serde(default)]
    pub rate_limits: Option<RateLimitsYaml>,
    /// Free-form notes.
    #[serde(default)]
    pub notes: Option<String>,
}

fn default_context_tokens() -> usize {
    8192
}

/// Generates JSON Schema for the model registry.
///
/// This can be used for:
/// - IDE autocompletion in YAML files
/// - Pre-runtime validation
/// - Documentation generation
///
/// # Example
///
/// ```
/// use manifold::registry_loader::generate_schema;
///
/// let schema = generate_schema();
/// println!("{}", serde_json::to_string_pretty(&schema).unwrap());
/// ```
#[must_use]
pub fn generate_schema() -> schemars::schema::RootSchema {
    schemars::schema_for!(RegistryYaml)
}

// =============================================================================
// LOADED REGISTRY
// =============================================================================

/// Provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    /// Direct API access to model provider.
    Direct,
    /// Routes to multiple underlying providers (adds latency overhead).
    Aggregator,
}

/// A loaded provider with its models.
#[derive(Debug, Clone)]
pub struct LoadedProvider {
    /// Provider ID (e.g., "anthropic").
    pub id: String,
    /// Environment variable name for API key.
    pub env_key: String,
    /// Optional secondary env key.
    pub env_key_secondary: Option<String>,
    /// URL to get an API key.
    pub key_url: String,
    /// API endpoint URL.
    pub api_url: String,
    /// ISO country code.
    pub country: String,
    /// Region.
    pub region: String,
    /// Compliance certifications.
    pub compliance: Vec<ComplianceLevel>,
    /// Provider type (direct or aggregator).
    pub provider_type: ProviderType,
    /// Models available.
    pub models: Vec<LoadedModel>,
}

impl LoadedProvider {
    /// Checks if this provider is available (env key is set).
    #[must_use]
    pub fn is_available(&self) -> bool {
        let primary_ok = std::env::var(&self.env_key).is_ok();
        let secondary_ok = self
            .env_key_secondary
            .as_ref()
            .map(|k| std::env::var(k).is_ok())
            .unwrap_or(true);
        primary_ok && secondary_ok
    }

    /// Returns the API key from environment (if available).
    #[must_use]
    pub fn api_key(&self) -> Option<String> {
        std::env::var(&self.env_key).ok()
    }

    /// Returns the secondary API key from environment (if available).
    #[must_use]
    pub fn secondary_api_key(&self) -> Option<String> {
        self.env_key_secondary
            .as_ref()
            .and_then(|k| std::env::var(k).ok())
    }
}

/// Model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Traditional transformer (all parameters active).
    Dense,
    /// Mixture of Experts (only subset active per forward pass).
    Moe,
    /// Hybrid architecture (e.g., Jamba's Mamba-Transformer).
    Hybrid,
}

/// Input modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Text input/output.
    Text,
    /// Image input.
    Image,
    /// Video input.
    Video,
    /// Audio input.
    Audio,
}

/// Agentic capabilities.
#[derive(Debug, Clone, Default)]
pub struct AgenticCapabilities {
    /// Maximum number of parallel agents this model can orchestrate.
    pub max_parallel_agents: Option<u32>,
    /// Whether the model supports agent orchestration/swarm.
    pub supports_orchestration: bool,
}

/// Pricing information (USD per million tokens).
#[derive(Debug, Clone, Default)]
pub struct Pricing {
    /// Input price per million tokens (USD).
    pub input_per_m: Option<f64>,
    /// Output price per million tokens (USD).
    pub output_per_m: Option<f64>,
}

/// Rate limit information (provider- or model-specific).
#[derive(Debug, Clone, Default)]
pub struct RateLimits {
    /// Requests per minute.
    pub requests_per_min: Option<u32>,
    /// Tokens per minute.
    pub tokens_per_min: Option<u32>,
    /// Requests per day.
    pub requests_per_day: Option<u32>,
    /// Maximum concurrent requests.
    pub concurrent_requests: Option<u32>,
}

/// Reasoning effort level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningEffort {
    /// Disable explicit chain-of-thought style effort controls.
    None,
    /// Minimal extra reasoning.
    Minimal,
    /// Low extra reasoning.
    Low,
    /// Medium extra reasoning.
    Medium,
    /// High extra reasoning.
    High,
    /// Extra-high reasoning.
    Xhigh,
}

impl ReasoningEffort {
    /// Stable `snake_case` string representation used in API responses.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }
}

/// A loaded model.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct LoadedModel {
    /// Model ID.
    pub id: String,
    /// Cost class.
    pub cost_class: CostClass,
    /// Typical latency in ms.
    pub typical_latency_ms: u32,
    /// Quality score.
    pub quality: f64,
    /// Context tokens.
    pub context_tokens: usize,
    /// Model type (llm, embedding, reranker).
    pub model_type: ModelType,
    /// Embedding dimensions (for embedding models).
    pub dimensions: Option<usize>,
    /// Full capability list (`snake_case` enum values from YAML).
    pub capabilities: Vec<CapabilityYaml>,
    // Capabilities
    /// Tool use support.
    pub supports_tool_use: bool,
    /// Vision support.
    pub supports_vision: bool,
    /// Structured output support.
    pub supports_structured_output: bool,
    /// Code support.
    pub supports_code: bool,
    /// Reasoning support.
    pub supports_reasoning: bool,
    /// Multilingual support.
    pub supports_multilingual: bool,
    /// Web search support.
    pub supports_web_search: bool,
    /// Content generation / business writing support.
    pub supports_content_generation: bool,
    /// Business acumen (financial, strategic, market analysis).
    pub supports_business_acumen: bool,

    // === ENRICHED FIELDS ===
    /// Model architecture (dense, moe, hybrid).
    pub architecture: Architecture,
    /// Total parameters in billions.
    pub total_params_b: Option<f64>,
    /// Active parameters per forward pass in billions (for `MoE` models).
    pub active_params_b: Option<f64>,
    /// Maximum output tokens.
    pub max_output_tokens: Option<usize>,
    /// Whether the model is native multimodal (trained on mixed modalities).
    pub native_multimodal: bool,
    /// Supported input modalities.
    pub modalities: Vec<Modality>,
    /// Agentic/swarm capabilities.
    pub agentic: Option<AgenticCapabilities>,
    /// Whether the model supports extended thinking/reasoning mode.
    pub thinking_mode: bool,
    /// Supported reasoning effort levels.
    pub reasoning_effort_levels: Vec<ReasoningEffort>,
    /// Whether the model supports native context compaction.
    pub native_compaction: bool,
    /// Model ID of the thinking variant (if this is the base model).
    pub thinking_variant: Option<String>,
    /// Pricing information.
    pub pricing: Option<Pricing>,
    /// Model publisher or organization (e.g., `OpenAI`, Anthropic).
    pub publisher: Option<String>,
    /// Model family name (e.g., Claude, GPT, Llama).
    pub family: Option<String>,
    /// Release date (ISO-8601 format recommended).
    pub release_date: Option<String>,
    /// Training data cutoff date (ISO-8601 format recommended).
    pub training_cutoff: Option<String>,
    /// Whether model weights are openly available.
    pub open_weights: bool,
    /// License identifier or URL.
    pub license: Option<String>,
    /// Whether the model is deprecated.
    pub deprecated: bool,
    /// Whether the model is in beta/preview.
    pub beta: bool,
    /// Benchmark scores (keyed by benchmark name).
    pub benchmarks: HashMap<String, f64>,
    /// Free-form tags for routing or promotion.
    pub tags: Vec<String>,
    /// Rate limit information (if published).
    pub rate_limits: Option<RateLimits>,
    /// Free-form notes.
    pub notes: Option<String>,
}

/// Model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// LLM for chat/completion.
    Llm,
    /// Embedding model.
    Embedding,
    /// Reranker model.
    Reranker,
    /// OCR / Document AI model.
    Ocr,
}

/// The loaded model registry.
#[derive(Debug, Clone)]
pub struct LoadedRegistry {
    /// All providers.
    providers: Vec<LoadedProvider>,
}

impl LoadedRegistry {
    /// Returns all providers.
    #[must_use]
    pub fn providers(&self) -> &[LoadedProvider] {
        &self.providers
    }

    /// Returns available providers (with API keys set).
    #[must_use]
    pub fn available_providers(&self) -> Vec<&LoadedProvider> {
        self.providers.iter().filter(|p| p.is_available()).collect()
    }

    /// Finds a provider by ID.
    #[must_use]
    pub fn get_provider(&self, id: &str) -> Option<&LoadedProvider> {
        self.providers.iter().find(|p| p.id == id)
    }

    /// Returns all LLM models.
    #[must_use]
    pub fn llm_models(&self) -> Vec<(&LoadedProvider, &LoadedModel)> {
        self.providers
            .iter()
            .flat_map(|p| {
                p.models
                    .iter()
                    .filter(|m| m.model_type == ModelType::Llm)
                    .map(move |m| (p, m))
            })
            .collect()
    }

    /// Returns all embedding models.
    #[must_use]
    pub fn embedding_models(&self) -> Vec<(&LoadedProvider, &LoadedModel)> {
        self.providers
            .iter()
            .flat_map(|p| {
                p.models
                    .iter()
                    .filter(|m| m.model_type == ModelType::Embedding)
                    .map(move |m| (p, m))
            })
            .collect()
    }

    /// Returns all reranker models.
    #[must_use]
    pub fn reranker_models(&self) -> Vec<(&LoadedProvider, &LoadedModel)> {
        self.providers
            .iter()
            .flat_map(|p| {
                p.models
                    .iter()
                    .filter(|m| m.model_type == ModelType::Reranker)
                    .map(move |m| (p, m))
            })
            .collect()
    }

    /// Converts to a `ModelSelector` for use with the selection system.
    #[must_use]
    pub fn to_model_selector(&self) -> ModelSelector {
        let mut selector = ModelSelector::empty();

        for provider in &self.providers {
            for model in &provider.models {
                if model.model_type != ModelType::Llm {
                    continue; // ModelSelector is for LLMs only
                }

                let data_sovereignty = match provider.region.as_str() {
                    "EU" | "EEA" => DataSovereignty::EU,
                    "CH" => DataSovereignty::Switzerland,
                    "CN" => DataSovereignty::China,
                    "US" => DataSovereignty::US,
                    "LOCAL" => DataSovereignty::OnPremises,
                    _ => DataSovereignty::Any,
                };

                let compliance = provider
                    .compliance
                    .first()
                    .copied()
                    .unwrap_or(ComplianceLevel::None);

                let metadata = ModelMetadata::new(
                    &provider.id,
                    &model.id,
                    model.cost_class,
                    model.typical_latency_ms,
                    model.quality,
                )
                .with_reasoning(model.supports_reasoning)
                .with_web_search(model.supports_web_search)
                .with_data_sovereignty(data_sovereignty)
                .with_compliance(compliance)
                .with_multilingual(model.supports_multilingual)
                .with_context_tokens(model.context_tokens)
                .with_tool_use(model.supports_tool_use)
                .with_vision(model.supports_vision)
                .with_structured_output(model.supports_structured_output)
                .with_code(model.supports_code)
                .with_content_generation(model.supports_content_generation)
                .with_business_acumen(model.supports_business_acumen)
                .with_location(&provider.country, &provider.region);

                selector = selector.with_model(metadata);
            }
        }

        selector
    }

    /// Prints a summary of all providers.
    pub fn print_summary(&self) {
        println!("Model Registry Summary");
        println!("======================\n");

        for provider in &self.providers {
            let status = if provider.is_available() {
                "✓ available"
            } else {
                "✗ no key"
            };

            println!(
                "{} ({}) - {} models [{}]",
                provider.id,
                provider.region,
                provider.models.len(),
                status
            );
            println!("  Key URL: {}", provider.key_url);
            println!("  API URL: {}", provider.api_url);
            println!();
        }
    }
}

// =============================================================================
// LOADING FUNCTIONS
// =============================================================================

/// Default path for the model registry relative to crate root.
pub const DEFAULT_REGISTRY_PATH: &str = "crates/manifold/config/models.yaml";

/// Loads the registry from the default path.
///
/// Tries these paths in order:
/// 1. `crates/manifold/config/models.yaml` (when run from workspace root)
/// 2. `config/models.yaml` (when run from the manifold crate directory)
/// 3. `CONVERGE_MODELS_PATH` environment variable
///
/// # Errors
///
/// Returns error if the file cannot be read or parsed.
pub fn load_registry() -> Result<LoadedRegistry, RegistryError> {
    // Check environment variable first
    if let Ok(path) = std::env::var("CONVERGE_MODELS_PATH") {
        return load_registry_from_path(&path);
    }

    // Try workspace-relative path
    if std::path::Path::new(DEFAULT_REGISTRY_PATH).exists() {
        return load_registry_from_path(DEFAULT_REGISTRY_PATH);
    }

    // Try crate-relative path
    let crate_path = "config/models.yaml";
    if std::path::Path::new(crate_path).exists() {
        return load_registry_from_path(crate_path);
    }

    // Fall back to compiled-in default
    load_registry_from_str(include_str!("../config/models.yaml"))
}

/// Loads the registry from a specific path.
///
/// # Errors
///
/// Returns error if the file cannot be read or parsed.
pub fn load_registry_from_path(path: impl AsRef<Path>) -> Result<LoadedRegistry, RegistryError> {
    let content = std::fs::read_to_string(path)?;
    load_registry_from_str(&content)
}

/// Loads the registry from a YAML string.
///
/// # Errors
///
/// Returns error if the YAML cannot be parsed or validation fails.
pub fn load_registry_from_str(yaml: &str) -> Result<LoadedRegistry, RegistryError> {
    let registry_yaml: RegistryYaml = serde_yaml::from_str(yaml)?;

    let mut providers = Vec::new();
    let mut errors = Vec::new();

    for (provider_id, provider_yaml) in registry_yaml.providers {
        // Validate provider
        if let Err(e) = validate_provider(&provider_id, &provider_yaml) {
            errors.push(e);
            continue;
        }

        let compliance = provider_yaml
            .compliance
            .iter()
            .map(|c| ComplianceLevel::from(*c))
            .collect();

        let mut models = Vec::new();

        for (model_id, model_yaml) in provider_yaml.models {
            // Validate model
            if let Err(e) = validate_model(&provider_id, &model_id, &model_yaml) {
                errors.push(e);
                continue;
            }

            let capabilities: std::collections::HashSet<_> =
                model_yaml.capabilities.iter().copied().collect();

            // Map modalities
            let modalities: Vec<Modality> = model_yaml
                .modalities
                .iter()
                .map(|m| match m {
                    ModalityYaml::Text => Modality::Text,
                    ModalityYaml::Image => Modality::Image,
                    ModalityYaml::Video => Modality::Video,
                    ModalityYaml::Audio => Modality::Audio,
                })
                .collect();

            // Map reasoning effort levels
            let reasoning_effort_levels = model_yaml
                .reasoning_effort_levels
                .iter()
                .copied()
                .map(ReasoningEffort::from)
                .collect();

            // Map agentic capabilities
            let agentic = model_yaml.agentic.as_ref().map(|a| AgenticCapabilities {
                max_parallel_agents: a.max_parallel_agents,
                supports_orchestration: a.supports_orchestration,
            });

            // Map pricing
            let pricing = model_yaml.pricing.as_ref().map(|p| Pricing {
                input_per_m: p.input_per_m,
                output_per_m: p.output_per_m,
            });

            // Map rate limits
            let rate_limits = model_yaml.rate_limits.as_ref().map(|r| RateLimits {
                requests_per_min: r.requests_per_min,
                tokens_per_min: r.tokens_per_min,
                requests_per_day: r.requests_per_day,
                concurrent_requests: r.concurrent_requests,
            });

            let model = LoadedModel {
                id: model_id,
                cost_class: model_yaml.cost_class.into(),
                typical_latency_ms: model_yaml.typical_latency_ms,
                quality: model_yaml.quality,
                context_tokens: model_yaml.context_tokens,
                model_type: model_yaml.model_type.into(),
                dimensions: model_yaml.dimensions,
                capabilities: model_yaml.capabilities.clone(),
                supports_tool_use: capabilities.contains(&CapabilityYaml::ToolUse),
                supports_vision: capabilities.contains(&CapabilityYaml::Vision),
                supports_structured_output: capabilities
                    .contains(&CapabilityYaml::StructuredOutput),
                supports_code: capabilities.contains(&CapabilityYaml::Code),
                supports_reasoning: capabilities.contains(&CapabilityYaml::Reasoning),
                supports_multilingual: capabilities.contains(&CapabilityYaml::Multilingual),
                supports_web_search: capabilities.contains(&CapabilityYaml::WebSearch),
                supports_content_generation: capabilities
                    .contains(&CapabilityYaml::ContentGeneration),
                supports_business_acumen: capabilities.contains(&CapabilityYaml::BusinessAcumen),
                // Enriched fields
                architecture: model_yaml.architecture.into(),
                total_params_b: model_yaml.total_params_b,
                active_params_b: model_yaml.active_params_b,
                max_output_tokens: model_yaml.max_output_tokens,
                native_multimodal: model_yaml.native_multimodal,
                modalities,
                agentic,
                thinking_mode: model_yaml.thinking_mode,
                reasoning_effort_levels,
                native_compaction: model_yaml.native_compaction,
                thinking_variant: model_yaml.thinking_variant.clone(),
                pricing,
                publisher: model_yaml.publisher.clone(),
                family: model_yaml.family.clone(),
                release_date: model_yaml.release_date.clone(),
                training_cutoff: model_yaml.training_cutoff.clone(),
                open_weights: model_yaml.open_weights,
                license: model_yaml.license.clone(),
                deprecated: model_yaml.deprecated,
                beta: model_yaml.beta,
                benchmarks: model_yaml.benchmarks.clone(),
                tags: model_yaml.tags.clone(),
                rate_limits,
                notes: model_yaml.notes.clone(),
            };

            models.push(model);
        }

        // Sort models by id for consistent ordering
        models.sort_by(|a, b| a.id.cmp(&b.id));

        let provider = LoadedProvider {
            id: provider_id,
            env_key: provider_yaml.env_key,
            env_key_secondary: provider_yaml.env_key_secondary,
            key_url: provider_yaml.key_url,
            api_url: provider_yaml.api_url,
            country: provider_yaml.country,
            region: provider_yaml.region.as_str().to_string(),
            compliance,
            provider_type: provider_yaml.provider_type.into(),
            models,
        };

        providers.push(provider);
    }

    // Fail if there were any validation errors
    if !errors.is_empty() {
        return Err(RegistryError::ValidationError(errors.join("; ")));
    }

    // Sort providers alphabetically for consistent ordering
    providers.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(LoadedRegistry { providers })
}

/// Validates a provider entry.
fn validate_provider(id: &str, provider: &ProviderYaml) -> Result<(), String> {
    // Validate env_key is not empty
    if provider.env_key.is_empty() {
        return Err(format!("Provider '{id}': env_key cannot be empty"));
    }

    // Validate URLs are valid
    if !provider.key_url.starts_with("http://") && !provider.key_url.starts_with("https://") {
        return Err(format!(
            "Provider '{id}': key_url must be a valid URL, got '{}'",
            provider.key_url
        ));
    }

    if !provider.api_url.starts_with("http://") && !provider.api_url.starts_with("https://") {
        return Err(format!(
            "Provider '{id}': api_url must be a valid URL, got '{}'",
            provider.api_url
        ));
    }

    // Validate country code (2 letters or LOCAL)
    if provider.country != "LOCAL" && provider.country.len() != 2 {
        return Err(format!(
            "Provider '{id}': country must be 2-letter ISO code or 'LOCAL', got '{}'",
            provider.country
        ));
    }

    // Validate has at least one model
    if provider.models.is_empty() {
        return Err(format!("Provider '{id}': must have at least one model"));
    }

    Ok(())
}

/// Validates a model entry.
fn validate_model(provider_id: &str, model_id: &str, model: &ModelYaml) -> Result<(), String> {
    // Validate quality is in range
    if !(0.0..=1.0).contains(&model.quality) {
        return Err(format!(
            "Model '{provider_id}/{model_id}': quality must be 0.0-1.0, got {}",
            model.quality
        ));
    }

    // Validate latency is reasonable
    if model.typical_latency_ms == 0 {
        return Err(format!(
            "Model '{provider_id}/{model_id}': typical_latency_ms must be > 0"
        ));
    }

    // Validate context_tokens is reasonable
    if model.context_tokens == 0 {
        return Err(format!(
            "Model '{provider_id}/{model_id}': context_tokens must be > 0"
        ));
    }

    // Validate embedding models have dimensions
    if model.model_type == ModelTypeYaml::Embedding && model.dimensions.is_none() {
        return Err(format!(
            "Model '{provider_id}/{model_id}': embedding models must specify dimensions"
        ));
    }

    Ok(())
}

impl From<ModelTypeYaml> for ModelType {
    fn from(t: ModelTypeYaml) -> Self {
        match t {
            ModelTypeYaml::Llm => ModelType::Llm,
            ModelTypeYaml::Embedding => ModelType::Embedding,
            ModelTypeYaml::Reranker => ModelType::Reranker,
            ModelTypeYaml::Ocr => ModelType::Ocr,
        }
    }
}

impl From<ArchitectureYaml> for Architecture {
    fn from(a: ArchitectureYaml) -> Self {
        match a {
            ArchitectureYaml::Dense => Architecture::Dense,
            ArchitectureYaml::Moe => Architecture::Moe,
            ArchitectureYaml::Hybrid => Architecture::Hybrid,
        }
    }
}

impl From<ReasoningEffortYaml> for ReasoningEffort {
    fn from(effort: ReasoningEffortYaml) -> Self {
        match effort {
            ReasoningEffortYaml::None => Self::None,
            ReasoningEffortYaml::Minimal => Self::Minimal,
            ReasoningEffortYaml::Low => Self::Low,
            ReasoningEffortYaml::Medium => Self::Medium,
            ReasoningEffortYaml::High => Self::High,
            ReasoningEffortYaml::Xhigh => Self::Xhigh,
        }
    }
}

impl From<ProviderTypeYaml> for ProviderType {
    fn from(p: ProviderTypeYaml) -> Self {
        match p {
            ProviderTypeYaml::Direct => ProviderType::Direct,
            ProviderTypeYaml::Aggregator => ProviderType::Aggregator,
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_YAML: &str = r"
providers:
  test-provider:
    env_key: TEST_API_KEY
    key_url: https://test.com/keys
    api_url: https://api.test.com/v1
    country: US
    region: US
    models:
      test-model:
        cost_class: Low
        typical_latency_ms: 2000
        quality: 0.85
        context_tokens: 128000
        capabilities: [tool_use, reasoning, code]

      test-embedding:
        cost_class: VeryLow
        typical_latency_ms: 100
        quality: 0.80
        context_tokens: 8192
        capabilities: []
        type: embedding
        dimensions: 1024
";

    const INVALID_COST_CLASS_YAML: &str = r"
providers:
  bad-provider:
    env_key: TEST_KEY
    key_url: https://test.com/keys
    api_url: https://api.test.com/v1
    country: US
    region: US
    models:
      bad-model:
        cost_class: SuperLow
        typical_latency_ms: 100
        quality: 0.5
";

    const INVALID_CAPABILITY_YAML: &str = r"
providers:
  bad-provider:
    env_key: TEST_KEY
    key_url: https://test.com/keys
    api_url: https://api.test.com/v1
    country: US
    region: US
    models:
      bad-model:
        cost_class: Low
        typical_latency_ms: 100
        quality: 0.5
        capabilities: [tool_use, telepathy]
";

    const INVALID_QUALITY_YAML: &str = r"
providers:
  bad-provider:
    env_key: TEST_KEY
    key_url: https://test.com/keys
    api_url: https://api.test.com/v1
    country: US
    region: US
    models:
      bad-model:
        cost_class: Low
        typical_latency_ms: 100
        quality: 1.5
";

    const MISSING_DIMENSIONS_YAML: &str = r"
providers:
  bad-provider:
    env_key: TEST_KEY
    key_url: https://test.com/keys
    api_url: https://api.test.com/v1
    country: US
    region: US
    models:
      bad-embedding:
        cost_class: Low
        typical_latency_ms: 100
        quality: 0.5
        type: embedding
";

    const UNKNOWN_FIELD_YAML: &str = r"
providers:
  bad-provider:
    env_key: TEST_KEY
    key_url: https://test.com/keys
    api_url: https://api.test.com/v1
    country: US
    region: US
    unknown_field: oops
    models:
      model:
        cost_class: Low
        typical_latency_ms: 100
        quality: 0.5
";

    #[test]
    fn parse_yaml() {
        let registry = load_registry_from_str(TEST_YAML).unwrap();
        assert_eq!(registry.providers.len(), 1);

        let provider = &registry.providers[0];
        assert_eq!(provider.id, "test-provider");
        assert_eq!(provider.key_url, "https://test.com/keys");
        assert_eq!(provider.api_url, "https://api.test.com/v1");
        assert_eq!(provider.models.len(), 2);
    }

    #[test]
    fn parse_model_capabilities() {
        let registry = load_registry_from_str(TEST_YAML).unwrap();
        let provider = &registry.providers[0];

        let llm = provider
            .models
            .iter()
            .find(|m| m.id == "test-model")
            .unwrap();
        assert!(llm.supports_tool_use);
        assert!(llm.supports_reasoning);
        assert!(llm.supports_code);
        assert!(!llm.supports_vision);
        assert_eq!(llm.model_type, ModelType::Llm);
    }

    #[test]
    fn parse_embedding_model() {
        let registry = load_registry_from_str(TEST_YAML).unwrap();
        let provider = &registry.providers[0];

        let embedding = provider
            .models
            .iter()
            .find(|m| m.id == "test-embedding")
            .unwrap();
        assert_eq!(embedding.model_type, ModelType::Embedding);
        assert_eq!(embedding.dimensions, Some(1024));
    }

    #[test]
    fn filter_by_model_type() {
        let registry = load_registry_from_str(TEST_YAML).unwrap();

        let llms = registry.llm_models();
        assert_eq!(llms.len(), 1);
        assert_eq!(llms[0].1.id, "test-model");

        let embeddings = registry.embedding_models();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].1.id, "test-embedding");
    }

    #[test]
    fn to_model_selector() {
        let registry = load_registry_from_str(TEST_YAML).unwrap();
        let selector = registry.to_model_selector();

        // Should have 1 LLM model (embedding is excluded)
        let reqs = converge_core::model_selection::AgentRequirements::balanced();
        let satisfying = selector.list_satisfying(&reqs);
        assert_eq!(satisfying.len(), 1);
    }

    #[test]
    fn provider_availability() {
        let registry = load_registry_from_str(TEST_YAML).unwrap();
        let provider = &registry.providers[0];

        // Should not be available (TEST_API_KEY not set by default)
        // Note: We don't test setting env vars as it requires unsafe in Rust 2024
        let _ = provider.is_available(); // Just verify method works
    }

    #[test]
    fn load_real_registry() {
        // This tests the compiled-in registry via include_str!
        let registry = load_registry().unwrap();

        // Should have multiple providers
        assert!(
            registry.providers.len() >= 10,
            "Expected at least 10 providers"
        );

        // Check some known providers exist
        let provider_ids: Vec<_> = registry.providers.iter().map(|p| p.id.as_str()).collect();
        assert!(provider_ids.contains(&"anthropic"), "Missing anthropic");
        assert!(provider_ids.contains(&"openai"), "Missing openai");
        assert!(provider_ids.contains(&"mistral"), "Missing mistral");
        assert!(provider_ids.contains(&"ollama"), "Missing ollama");

        // Check anthropic has correct URLs
        let anthropic = registry.get_provider("anthropic").unwrap();
        assert_eq!(
            anthropic.key_url,
            "https://console.anthropic.com/settings/keys"
        );
        assert_eq!(anthropic.api_url, "https://api.anthropic.com/v1");
        assert_eq!(anthropic.env_key, "ANTHROPIC_API_KEY");

        // Check ollama is marked as LOCAL
        let ollama = registry.get_provider("ollama").unwrap();
        assert_eq!(ollama.region, "LOCAL");

        // Check we have LLM models
        let llms = registry.llm_models();
        assert!(llms.len() >= 30, "Expected at least 30 LLM models");

        // Check we have embedding models
        let embeddings = registry.embedding_models();
        assert!(
            embeddings.len() >= 3,
            "Expected at least 3 embedding models"
        );

        println!(
            "Loaded {} providers with {} LLM models and {} embedding models",
            registry.providers.len(),
            llms.len(),
            embeddings.len()
        );
    }

    // =========================================================================
    // TYPE-SAFE VALIDATION TESTS
    // =========================================================================

    #[test]
    fn rejects_invalid_cost_class() {
        let result = load_registry_from_str(INVALID_COST_CLASS_YAML);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("SuperLow") || err.contains("unknown variant"),
            "Expected error about invalid cost class, got: {err}"
        );
    }

    #[test]
    fn rejects_invalid_capability() {
        let result = load_registry_from_str(INVALID_CAPABILITY_YAML);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("telepathy") || err.contains("unknown variant"),
            "Expected error about invalid capability, got: {err}"
        );
    }

    #[test]
    fn rejects_invalid_quality() {
        let result = load_registry_from_str(INVALID_QUALITY_YAML);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("quality") && err.contains("1.5"),
            "Expected error about quality out of range, got: {err}"
        );
    }

    #[test]
    fn rejects_embedding_without_dimensions() {
        let result = load_registry_from_str(MISSING_DIMENSIONS_YAML);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dimensions"),
            "Expected error about missing dimensions, got: {err}"
        );
    }

    #[test]
    fn rejects_unknown_fields() {
        let result = load_registry_from_str(UNKNOWN_FIELD_YAML);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unknown_field") || err.contains("unknown field"),
            "Expected error about unknown field, got: {err}"
        );
    }

    #[test]
    fn rejects_invalid_region() {
        let yaml = r"
providers:
  bad:
    env_key: KEY
    key_url: https://test.com
    api_url: https://api.test.com
    country: US
    region: INVALID
    models:
      m:
        cost_class: Low
        typical_latency_ms: 100
        quality: 0.5
";
        let result = load_registry_from_str(yaml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("INVALID") || err.contains("unknown variant"),
            "Expected error about invalid region, got: {err}"
        );
    }

    #[test]
    fn rejects_invalid_url() {
        let yaml = r"
providers:
  bad:
    env_key: KEY
    key_url: not-a-url
    api_url: https://api.test.com
    country: US
    region: US
    models:
      m:
        cost_class: Low
        typical_latency_ms: 100
        quality: 0.5
";
        let result = load_registry_from_str(yaml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("key_url") && err.contains("URL"),
            "Expected error about invalid URL, got: {err}"
        );
    }

    #[test]
    fn rejects_zero_latency() {
        let yaml = r"
providers:
  bad:
    env_key: KEY
    key_url: https://test.com
    api_url: https://api.test.com
    country: US
    region: US
    models:
      m:
        cost_class: Low
        typical_latency_ms: 0
        quality: 0.5
";
        let result = load_registry_from_str(yaml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("latency") && err.contains('0'),
            "Expected error about zero latency, got: {err}"
        );
    }

    #[test]
    fn rejects_empty_provider() {
        let yaml = r"
providers:
  empty:
    env_key: KEY
    key_url: https://test.com
    api_url: https://api.test.com
    country: US
    region: US
    models: {}
";
        let result = load_registry_from_str(yaml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("at least one model"),
            "Expected error about empty models, got: {err}"
        );
    }
}
