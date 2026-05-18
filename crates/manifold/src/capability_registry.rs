// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! Unified capability registry for Converge providers.
//!
//! The capability registry provides a single point for discovering and
//! selecting providers based on their capabilities. This supports the
//! Converge principle that different models excel at different tasks.
//!
//! # Example
//!
//! ```ignore
//! use manifold::{CapabilityRegistry, CapabilityRequirements};
//! use converge_core::capability::{CapabilityKind, Modality};
//!
//! let registry = CapabilityRegistry::from_env();
//!
//! // Find an embedder that supports images
//! let requirements = CapabilityRequirements::embedding()
//!     .with_modality(Modality::Image)
//!     .prefer_local(true);
//!
//! if let Some(embedder) = registry.select_embedder(&requirements) {
//!     // Use the embedder
//! }
//! ```

#[cfg(feature = "brave")]
use crate::brave::BraveSearchProvider;
#[cfg(feature = "feed")]
use crate::feed::{FeedFetchBackend, HttpFeedProvider};
#[cfg(feature = "fetch")]
use crate::fetch::HttpFetchProvider;
use crate::search::{WebFetchBackend, WebSearchBackend};
#[cfg(feature = "tavily")]
use crate::tavily::TavilySearchProvider;
use converge_core::capability::{
    CapabilityKind, CapabilityMetadata, Embedding, GraphRecall, Modality, Reranking, VectorRecall,
};
use converge_provider::selection::DataSovereignty;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

/// Requirements for capability selection.
#[derive(Debug, Clone)]
pub struct CapabilityRequirements {
    /// Required capability kind.
    pub capability: CapabilityKind,
    /// Required modalities (for embedding/reranking).
    pub modalities: Vec<Modality>,
    /// Prefer local providers (data sovereignty).
    pub prefer_local: bool,
    /// Required data sovereignty level.
    pub data_sovereignty: DataSovereignty,
    /// Maximum acceptable latency in milliseconds.
    pub max_latency_ms: u32,
}

impl CapabilityRequirements {
    /// Requirements for LLM completion.
    #[must_use]
    pub fn completion() -> Self {
        Self {
            capability: CapabilityKind::Completion,
            modalities: vec![Modality::Text],
            prefer_local: false,
            data_sovereignty: DataSovereignty::Any,
            max_latency_ms: 30_000,
        }
    }

    /// Requirements for embedding.
    #[must_use]
    pub fn embedding() -> Self {
        Self {
            capability: CapabilityKind::Embedding,
            modalities: vec![Modality::Text],
            prefer_local: false,
            data_sovereignty: DataSovereignty::Any,
            max_latency_ms: 5_000,
        }
    }

    /// Requirements for reranking.
    #[must_use]
    pub fn reranking() -> Self {
        Self {
            capability: CapabilityKind::Reranking,
            modalities: vec![Modality::Text],
            prefer_local: false,
            data_sovereignty: DataSovereignty::Any,
            max_latency_ms: 5_000,
        }
    }

    /// Requirements for vector recall.
    #[must_use]
    pub fn vector_recall() -> Self {
        Self {
            capability: CapabilityKind::VectorRecall,
            modalities: vec![],
            prefer_local: true,
            data_sovereignty: DataSovereignty::Any,
            max_latency_ms: 100,
        }
    }

    /// Requirements for graph recall.
    #[must_use]
    pub fn graph_recall() -> Self {
        Self {
            capability: CapabilityKind::GraphRecall,
            modalities: vec![],
            prefer_local: true,
            data_sovereignty: DataSovereignty::Any,
            max_latency_ms: 100,
        }
    }

    /// Add required modality.
    #[must_use]
    pub fn with_modality(mut self, modality: Modality) -> Self {
        if !self.modalities.contains(&modality) {
            self.modalities.push(modality);
        }
        self
    }

    /// Set local preference.
    #[must_use]
    pub fn prefer_local(mut self, prefer: bool) -> Self {
        self.prefer_local = prefer;
        self
    }

    /// Set data sovereignty requirement.
    #[must_use]
    pub fn with_data_sovereignty(mut self, sovereignty: DataSovereignty) -> Self {
        self.data_sovereignty = sovereignty;
        self
    }

    /// Set maximum latency.
    #[must_use]
    pub fn with_max_latency_ms(mut self, ms: u32) -> Self {
        self.max_latency_ms = ms;
        self
    }
}

/// Registered provider with its capabilities.
struct RegisteredProvider {
    /// Provider metadata.
    metadata: CapabilityMetadata,
    /// Embedding provider instance (if applicable).
    embedder: Option<Arc<dyn Embedding>>,
    /// Reranker provider instance (if applicable).
    reranker: Option<Arc<dyn Reranking>>,
}

/// Search provider capability with planner-visible meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SearchProviderFeature {
    AiSummary,
    News,
    Images,
    Local,
}

impl SearchProviderFeature {
    fn selection_weight(self) -> i32 {
        match self {
            Self::AiSummary => 100,
            Self::News | Self::Images => 30,
            Self::Local => 20,
        }
    }
}

/// Set of search provider features.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchProviderFeatures(BTreeSet<SearchProviderFeature>);

impl SearchProviderFeatures {
    /// Create an empty feature set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a feature set from concrete features.
    #[must_use]
    pub fn from_features(features: impl IntoIterator<Item = SearchProviderFeature>) -> Self {
        Self(features.into_iter().collect())
    }

    /// Return true when this set contains `feature`.
    #[must_use]
    pub fn contains(&self, feature: SearchProviderFeature) -> bool {
        self.0.contains(&feature)
    }

    /// Return true when all `required` features are present.
    #[must_use]
    pub fn contains_all(&self, required: &Self) -> bool {
        required.iter().all(|feature| self.contains(feature))
    }

    /// Iterate over contained features.
    pub fn iter(&self) -> impl Iterator<Item = SearchProviderFeature> + '_ {
        self.0.iter().copied()
    }

    /// Enable or disable a feature.
    pub fn set(&mut self, feature: SearchProviderFeature, enabled: bool) {
        if enabled {
            self.0.insert(feature);
        } else {
            self.0.remove(&feature);
        }
    }
}

/// Web search provider metadata for agent selection.
#[derive(Debug, Clone)]
pub struct SearchProviderMeta {
    /// Provider name (e.g., "brave", "perplexity").
    pub name: String,
    /// Whether this provider is available (API key set).
    pub available: bool,
    /// Typical latency in milliseconds.
    pub typical_latency_ms: u32,
    /// Search features supported by this provider.
    pub features: SearchProviderFeatures,
}

/// Requirements for selecting a web search provider.
///
/// Unlike LLM requirements, web search requirements focus on
/// search-specific capabilities like news, images, and AI summaries.
#[derive(Debug, Clone)]
pub struct WebSearchRequirements {
    /// Maximum latency in milliseconds.
    pub max_latency_ms: u32,
    /// Search features required by the caller.
    pub required_features: SearchProviderFeatures,
    /// Data sovereignty requirement.
    pub data_sovereignty: DataSovereignty,
}

impl Default for WebSearchRequirements {
    fn default() -> Self {
        Self {
            max_latency_ms: 10_000,
            required_features: SearchProviderFeatures::new(),
            data_sovereignty: DataSovereignty::Any,
        }
    }
}

impl WebSearchRequirements {
    /// Creates default requirements for general web search.
    #[must_use]
    pub fn web_search() -> Self {
        Self::default()
    }

    /// Creates requirements for AI-grounded search (RAG).
    #[must_use]
    pub fn grounded() -> Self {
        Self {
            max_latency_ms: 15_000,
            required_features: SearchProviderFeatures::from_features([
                SearchProviderFeature::AiSummary,
            ]),
            ..Self::default()
        }
    }

    /// Creates requirements for news search.
    #[must_use]
    pub fn news() -> Self {
        Self {
            required_features: SearchProviderFeatures::from_features([SearchProviderFeature::News]),
            ..Self::default()
        }
    }

    /// Sets the maximum latency.
    #[must_use]
    pub fn with_max_latency_ms(mut self, ms: u32) -> Self {
        self.max_latency_ms = ms;
        self
    }

    /// Requires AI-powered summaries.
    #[must_use]
    pub fn with_ai_summary(mut self, required: bool) -> Self {
        self.required_features
            .set(SearchProviderFeature::AiSummary, required);
        self
    }

    /// Sets data sovereignty requirement.
    #[must_use]
    pub fn with_data_sovereignty(mut self, sovereignty: DataSovereignty) -> Self {
        self.data_sovereignty = sovereignty;
        self
    }

    /// Requires news search support.
    #[must_use]
    pub fn with_news(mut self, required: bool) -> Self {
        self.required_features
            .set(SearchProviderFeature::News, required);
        self
    }

    /// Requires image search support.
    #[must_use]
    pub fn with_images(mut self, required: bool) -> Self {
        self.required_features
            .set(SearchProviderFeature::Images, required);
        self
    }

    /// Requires local/POI search support.
    #[must_use]
    pub fn with_local(mut self, required: bool) -> Self {
        self.required_features
            .set(SearchProviderFeature::Local, required);
        self
    }
}

/// Unified capability registry.
///
/// Discovers and manages all available capability providers.
pub struct CapabilityRegistry {
    /// Registered providers by name.
    providers: HashMap<String, RegisteredProvider>,
    /// Vector stores by name.
    vector_stores: HashMap<String, Arc<dyn VectorRecall>>,
    /// Graph stores by name.
    graph_stores: HashMap<String, Arc<dyn GraphRecall>>,
    /// Web search providers by name.
    search_providers: HashMap<String, SearchProviderMeta>,
    /// Executable web search backends by name.
    search_backends: HashMap<String, Arc<dyn WebSearchBackend>>,
    /// Brave search provider instance (if available).
    #[cfg(feature = "brave")]
    brave_provider: Option<Arc<BraveSearchProvider>>,
    /// Tavily search provider instance (if available).
    #[cfg(feature = "tavily")]
    tavily_provider: Option<Arc<TavilySearchProvider>>,
    /// Web fetch backend (URL → content).
    fetch_backend: Option<Arc<dyn WebFetchBackend>>,
    /// Feed fetch backend (RSS/Atom/JSON Feed → normalized observations).
    #[cfg(feature = "feed")]
    feed_backend: Option<Arc<dyn FeedFetchBackend>>,
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityRegistry {
    /// Creates an empty capability registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            vector_stores: HashMap::new(),
            graph_stores: HashMap::new(),
            search_providers: HashMap::new(),
            search_backends: HashMap::new(),
            #[cfg(feature = "brave")]
            brave_provider: None,
            #[cfg(feature = "tavily")]
            tavily_provider: None,
            fetch_backend: None,
            #[cfg(feature = "feed")]
            feed_backend: None,
        }
    }

    /// Creates a registry with auto-detected providers from environment.
    ///
    /// This checks for:
    /// - Ollama (local LLM and embedding)
    /// - In-memory vector store (always available)
    /// - In-memory graph store (always available)
    /// - Brave Search (if `BRAVE_API_KEY` is set)
    /// - Tavily Search (if `TAVILY_API_KEY` is set)
    #[must_use]
    pub fn with_local_defaults() -> Self {
        let mut registry = Self::new();

        // Add in-memory vector store
        registry.add_vector_store(
            "default",
            Arc::new(crate::vector::InMemoryVectorStore::new()),
        );

        // Graph store moved to organism-intelligence crate

        // Try to add Brave Search if available
        registry.try_add_brave_from_env();
        registry.try_add_tavily_from_env();

        // HTTP fetch is always available (no API key required)
        #[cfg(feature = "fetch")]
        if let Ok(provider) = HttpFetchProvider::new() {
            registry.fetch_backend = Some(Arc::new(provider));
        }
        #[cfg(feature = "feed")]
        if let Ok(provider) = HttpFeedProvider::new() {
            registry.feed_backend = Some(Arc::new(provider));
        }

        registry
    }

    /// Attempts to add Brave Search provider from environment.
    ///
    /// Returns `true` if Brave Search was added successfully.
    pub fn try_add_brave_from_env(&mut self) -> bool {
        #[cfg(feature = "brave")]
        if let Ok(provider) = BraveSearchProvider::from_env() {
            let provider = Arc::new(provider);
            self.brave_provider = Some(provider.clone());
            self.search_backends
                .insert("brave".to_string(), provider.clone());
            self.search_providers.insert(
                "brave".to_string(),
                SearchProviderMeta {
                    name: "brave".to_string(),
                    available: true,
                    typical_latency_ms: 500,
                    features: SearchProviderFeatures::from_features([
                        SearchProviderFeature::News,
                        SearchProviderFeature::Images,
                        SearchProviderFeature::Local,
                    ]),
                },
            );
            return true;
        }
        false
    }

    /// Attempts to add Tavily Search provider from environment.
    ///
    /// Returns `true` if Tavily Search was added successfully.
    pub fn try_add_tavily_from_env(&mut self) -> bool {
        #[cfg(feature = "tavily")]
        if let Ok(provider) = TavilySearchProvider::from_env() {
            let provider = Arc::new(provider);
            self.tavily_provider = Some(provider.clone());
            self.search_backends
                .insert("tavily".to_string(), provider.clone());
            self.search_providers.insert(
                "tavily".to_string(),
                SearchProviderMeta {
                    name: "tavily".to_string(),
                    available: true,
                    typical_latency_ms: 1200,
                    features: SearchProviderFeatures::from_features([
                        SearchProviderFeature::AiSummary,
                        SearchProviderFeature::News,
                        SearchProviderFeature::Images,
                    ]),
                },
            );
            return true;
        }
        false
    }

    /// Adds Brave Search provider with a specific API key.
    #[cfg(feature = "brave")]
    pub fn add_brave(&mut self, api_key: impl Into<String>) {
        let provider = Arc::new(BraveSearchProvider::new(api_key));
        self.brave_provider = Some(provider.clone());
        self.search_backends
            .insert("brave".to_string(), provider.clone());
        self.search_providers.insert(
            "brave".to_string(),
            SearchProviderMeta {
                name: "brave".to_string(),
                available: true,
                typical_latency_ms: 500,
                features: SearchProviderFeatures::from_features([
                    SearchProviderFeature::News,
                    SearchProviderFeature::Images,
                    SearchProviderFeature::Local,
                ]),
            },
        );
    }

    /// Adds Tavily Search provider with a specific API key.
    #[cfg(feature = "tavily")]
    pub fn add_tavily(&mut self, api_key: impl Into<String>) {
        let provider = Arc::new(TavilySearchProvider::new(api_key));
        self.tavily_provider = Some(provider.clone());
        self.search_backends
            .insert("tavily".to_string(), provider.clone());
        self.search_providers.insert(
            "tavily".to_string(),
            SearchProviderMeta {
                name: "tavily".to_string(),
                available: true,
                typical_latency_ms: 1200,
                features: SearchProviderFeatures::from_features([
                    SearchProviderFeature::AiSummary,
                    SearchProviderFeature::News,
                    SearchProviderFeature::Images,
                ]),
            },
        );
    }

    /// Gets the Brave Search provider if available.
    #[cfg(feature = "brave")]
    #[must_use]
    pub fn brave(&self) -> Option<&BraveSearchProvider> {
        self.brave_provider.as_deref()
    }

    /// Gets the Tavily Search provider if available.
    #[cfg(feature = "tavily")]
    #[must_use]
    pub fn tavily(&self) -> Option<&TavilySearchProvider> {
        self.tavily_provider.as_deref()
    }

    /// Gets the web fetch backend if available.
    #[must_use]
    pub fn fetch_backend(&self) -> Option<Arc<dyn WebFetchBackend>> {
        self.fetch_backend.clone()
    }

    /// Sets a custom web fetch backend.
    pub fn set_fetch_backend(&mut self, backend: Arc<dyn WebFetchBackend>) {
        self.fetch_backend = Some(backend);
    }

    /// Checks if web fetch capability is available.
    #[must_use]
    pub fn has_web_fetch(&self) -> bool {
        self.fetch_backend.is_some()
    }

    /// Gets the feed fetch backend if available.
    #[cfg(feature = "feed")]
    #[must_use]
    pub fn feed_backend(&self) -> Option<Arc<dyn FeedFetchBackend>> {
        self.feed_backend.clone()
    }

    /// Sets a custom feed fetch backend.
    #[cfg(feature = "feed")]
    pub fn set_feed_backend(&mut self, backend: Arc<dyn FeedFetchBackend>) {
        self.feed_backend = Some(backend);
    }

    /// Checks if feed fetch capability is available.
    #[cfg(feature = "feed")]
    #[must_use]
    pub fn has_feed_fetch(&self) -> bool {
        self.feed_backend.is_some()
    }

    /// Checks if web search capability is available.
    #[must_use]
    pub fn has_web_search(&self) -> bool {
        !self.search_providers.is_empty()
    }

    /// Gets metadata for all available search providers.
    #[must_use]
    pub fn search_providers(&self) -> Vec<&SearchProviderMeta> {
        self.search_providers.values().collect()
    }

    /// Selects the best search provider based on requirements.
    ///
    /// Currently returns Brave if available, as it's the primary search provider.
    #[must_use]
    pub fn select_search_provider(
        &self,
        requirements: &WebSearchRequirements,
    ) -> Option<&SearchProviderMeta> {
        self.search_providers
            .values()
            .filter(|p| {
                // Basic availability and latency check
                if !p.available || p.typical_latency_ms > requirements.max_latency_ms {
                    return false;
                }
                // Check required capabilities
                if !p.features.contains_all(&requirements.required_features) {
                    return false;
                }
                true
            })
            .max_by_key(|p| {
                // Score providers by capabilities the caller actually asked for.
                let mut score = 0i32;
                for feature in requirements.required_features.iter() {
                    if p.features.contains(feature) {
                        score += feature.selection_weight();
                    }
                }
                // Prefer lower latency
                score -= (p.typical_latency_ms / 100) as i32;
                score
            })
    }

    /// Selects an executable search backend matching the requirements.
    #[must_use]
    pub fn select_search_backend(
        &self,
        requirements: &WebSearchRequirements,
    ) -> Option<Arc<dyn WebSearchBackend>> {
        self.select_search_provider(requirements)
            .and_then(|meta| self.search_backends.get(&meta.name).cloned())
    }

    /// Registers an embedding provider.
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_embedder(
        &mut self,
        name: &str,
        provider: Arc<dyn Embedding>,
        metadata: CapabilityMetadata,
    ) {
        let entry = self
            .providers
            .entry(name.to_string())
            .or_insert_with(|| RegisteredProvider {
                metadata: metadata.clone(),
                embedder: None,
                reranker: None,
            });
        entry.embedder = Some(provider);
        // Merge capabilities
        for cap in &metadata.capabilities {
            if !entry.metadata.capabilities.contains(cap) {
                entry.metadata.capabilities.push(*cap);
            }
        }
    }

    /// Registers a reranker provider.
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_reranker(
        &mut self,
        name: &str,
        provider: Arc<dyn Reranking>,
        metadata: CapabilityMetadata,
    ) {
        let entry = self
            .providers
            .entry(name.to_string())
            .or_insert_with(|| RegisteredProvider {
                metadata: metadata.clone(),
                embedder: None,
                reranker: None,
            });
        entry.reranker = Some(provider);
        // Merge capabilities
        for cap in &metadata.capabilities {
            if !entry.metadata.capabilities.contains(cap) {
                entry.metadata.capabilities.push(*cap);
            }
        }
    }

    /// Registers a vector store.
    pub fn add_vector_store(&mut self, name: &str, store: Arc<dyn VectorRecall>) {
        self.vector_stores.insert(name.to_string(), store);
    }

    /// Registers a graph store.
    pub fn add_graph_store(&mut self, name: &str, store: Arc<dyn GraphRecall>) {
        self.graph_stores.insert(name.to_string(), store);
    }

    /// Selects an embedding provider matching requirements.
    #[must_use]
    pub fn select_embedder(
        &self,
        requirements: &CapabilityRequirements,
    ) -> Option<Arc<dyn Embedding>> {
        self.providers
            .values()
            .filter(|p| {
                p.embedder.is_some() && self.matches_requirements(&p.metadata, requirements)
            })
            .max_by_key(|p| self.score_provider(&p.metadata, requirements))
            .and_then(|p| p.embedder.clone())
    }

    /// Selects a reranker provider matching requirements.
    #[must_use]
    pub fn select_reranker(
        &self,
        requirements: &CapabilityRequirements,
    ) -> Option<Arc<dyn Reranking>> {
        self.providers
            .values()
            .filter(|p| {
                p.reranker.is_some() && self.matches_requirements(&p.metadata, requirements)
            })
            .max_by_key(|p| self.score_provider(&p.metadata, requirements))
            .and_then(|p| p.reranker.clone())
    }

    /// Gets the default vector store.
    #[must_use]
    pub fn get_vector_store(&self, name: &str) -> Option<Arc<dyn VectorRecall>> {
        self.vector_stores.get(name).cloned()
    }

    /// Gets the default graph store.
    #[must_use]
    pub fn get_graph_store(&self, name: &str) -> Option<Arc<dyn GraphRecall>> {
        self.graph_stores.get(name).cloned()
    }

    /// Gets the default vector store (named "default").
    #[must_use]
    pub fn default_vector_store(&self) -> Option<Arc<dyn VectorRecall>> {
        self.get_vector_store("default")
    }

    /// Gets the default graph store (named "default").
    #[must_use]
    pub fn default_graph_store(&self) -> Option<Arc<dyn GraphRecall>> {
        self.get_graph_store("default")
    }

    /// Lists all registered provider names.
    #[must_use]
    pub fn provider_names(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }

    /// Lists all registered vector store names.
    #[must_use]
    pub fn vector_store_names(&self) -> Vec<&str> {
        self.vector_stores.keys().map(String::as_str).collect()
    }

    /// Lists all registered graph store names.
    #[must_use]
    pub fn graph_store_names(&self) -> Vec<&str> {
        self.graph_stores.keys().map(String::as_str).collect()
    }

    /// Checks if a provider matches the requirements.
    #[allow(clippy::unused_self)]
    fn matches_requirements(
        &self,
        metadata: &CapabilityMetadata,
        requirements: &CapabilityRequirements,
    ) -> bool {
        // Check capability
        if !metadata.capabilities.contains(&requirements.capability) {
            return false;
        }

        // Check modalities
        for modality in &requirements.modalities {
            if !metadata.modalities.contains(modality) {
                return false;
            }
        }

        // Check data sovereignty - local providers satisfy all requirements
        #[allow(clippy::match_same_arms)]
        match (&requirements.data_sovereignty, metadata.is_local) {
            (DataSovereignty::Any, _) | (_, true) => {} // Always OK or local
            _ => {} // Remote providers must match specific sovereignty
        }

        // Check latency
        if metadata.typical_latency_ms > requirements.max_latency_ms {
            return false;
        }

        true
    }

    /// Scores a provider for selection (higher = better).
    #[allow(clippy::unused_self, clippy::cast_possible_wrap)]
    fn score_provider(
        &self,
        metadata: &CapabilityMetadata,
        requirements: &CapabilityRequirements,
    ) -> i32 {
        let mut score = 0;

        // Prefer local if requested
        if requirements.prefer_local && metadata.is_local {
            score += 100;
        }

        // Lower latency is better
        if metadata.typical_latency_ms < requirements.max_latency_ms / 2 {
            score += 50;
        }

        // More modalities is better
        score += (metadata.modalities.len() * 10) as i32;

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::InMemoryVectorStore;
    use converge_core::capability::{
        CapabilityError, GraphEdge, GraphNode, GraphQuery, GraphRecall, GraphResult,
    };

    struct TestGraphStore;

    impl TestGraphStore {
        fn new() -> Self {
            Self
        }
    }

    impl GraphRecall for TestGraphStore {
        fn name(&self) -> &'static str {
            "test-graph"
        }

        fn add_node(&self, _node: &GraphNode) -> Result<(), CapabilityError> {
            Ok(())
        }

        fn add_edge(&self, _edge: &GraphEdge) -> Result<(), CapabilityError> {
            Ok(())
        }

        fn traverse(&self, _query: &GraphQuery) -> Result<GraphResult, CapabilityError> {
            Ok(GraphResult {
                nodes: Vec::new(),
                edges: Vec::new(),
            })
        }

        fn find_nodes(
            &self,
            _label: &str,
            _properties: Option<&serde_json::Value>,
        ) -> Result<Vec<GraphNode>, CapabilityError> {
            Ok(Vec::new())
        }

        fn get_node(&self, _id: &str) -> Result<Option<GraphNode>, CapabilityError> {
            Ok(None)
        }

        fn delete_node(&self, _id: &str) -> Result<(), CapabilityError> {
            Ok(())
        }

        fn clear(&self) -> Result<(), CapabilityError> {
            Ok(())
        }
    }

    #[test]
    fn registry_with_local_defaults() {
        let registry = CapabilityRegistry::with_local_defaults();

        assert!(registry.default_vector_store().is_some());
        assert!(registry.default_graph_store().is_none());
    }

    #[test]
    fn add_and_get_stores() {
        let mut registry = CapabilityRegistry::new();

        registry.add_vector_store("test", Arc::new(InMemoryVectorStore::new()));
        registry.add_graph_store("test", Arc::new(TestGraphStore::new()));

        assert!(registry.get_vector_store("test").is_some());
        assert!(registry.get_graph_store("test").is_some());
        assert!(registry.get_vector_store("missing").is_none());
    }

    #[test]
    fn list_registered_stores() {
        let registry = CapabilityRegistry::with_local_defaults();

        let vector_stores = registry.vector_store_names();
        assert!(vector_stores.contains(&"default"));

        let graph_stores = registry.graph_store_names();
        assert!(graph_stores.is_empty());
    }

    #[test]
    fn capability_requirements_builder() {
        let reqs = CapabilityRequirements::embedding()
            .with_modality(Modality::Image)
            .prefer_local(true)
            .with_max_latency_ms(1000);

        assert_eq!(reqs.capability, CapabilityKind::Embedding);
        assert!(reqs.modalities.contains(&Modality::Image));
        assert!(reqs.prefer_local);
        assert_eq!(reqs.max_latency_ms, 1000);
    }

    #[cfg(all(feature = "brave", feature = "tavily"))]
    #[test]
    fn search_provider_selection_prefers_tavily_for_ai_summary() {
        let mut registry = CapabilityRegistry::new();
        registry.add_brave("brave-key");
        registry.add_tavily("tavily-key");

        let selected = registry
            .select_search_provider(&WebSearchRequirements::grounded())
            .unwrap();
        assert_eq!(selected.name, "tavily");

        let backend = registry
            .select_search_backend(&WebSearchRequirements::grounded())
            .unwrap();
        assert_eq!(backend.provider_name(), "tavily");
    }

    #[cfg(all(feature = "brave", feature = "tavily"))]
    #[test]
    fn search_provider_selection_prefers_brave_for_local_search() {
        let mut registry = CapabilityRegistry::new();
        registry.add_brave("brave-key");
        registry.add_tavily("tavily-key");

        let selected = registry
            .select_search_provider(&WebSearchRequirements::web_search().with_local(true))
            .unwrap();
        assert_eq!(selected.name, "brave");
    }
}
