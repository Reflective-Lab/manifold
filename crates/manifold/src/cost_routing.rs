// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Cost-aware routing.
//!
//! Hard budget constraint layered on top of the fuzzy fitness scoring:
//! given an estimated prompt token count and a max output token count, only
//! consider models whose total request cost stays under a USD ceiling.
//!
//! Cost data comes from `ModelMetadata.pricing_*_usd_per_million`, populated
//! by the OpenRouter catalog (see [`crate::model_catalog`]) or set explicitly
//! via `ModelMetadata::with_pricing`.
//!
//! Models with no pricing data are admitted by default (`BudgetMode::Lenient`)
//! so this layer doesn't silently exclude direct backends that aren't in
//! the OpenRouter registry (Kong-routed, Staik, Qwen-max, etc.). Use
//! [`BudgetMode::Strict`] when you need to guarantee every selected model
//! was checked against the budget.
//!
//! # Example
//!
//! ```ignore
//! use manifold::cost_routing::{BudgetMode, CostBudget};
//! use manifold::model_selection::ProviderRegistry;
//! use converge_provider::selection::AgentRequirements;
//!
//! let registry = ProviderRegistry::from_env();
//! let reqs = AgentRequirements::balanced();
//! let budget = CostBudget::new(800, 256, 0.01); // 800 prompt + 256 output, $0.01 cap
//! let result = registry.select_within_budget(&reqs, &budget)?;
//! ```

use crate::model_selection::{ModelMetadata, ProviderRegistry, RejectionReason, SelectionResult};
use converge_provider::selection::AgentRequirements;
use converge_provider::LlmError;

/// How to handle models that have no pricing data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetMode {
    /// Admit models with no pricing data (default). Useful when some backends
    /// don't appear in the live catalog yet but you still want them selectable.
    Lenient,
    /// Reject models with no pricing data. Use when you need a hard
    /// guarantee that every candidate was cost-checked.
    Strict,
}

impl Default for BudgetMode {
    fn default() -> Self {
        Self::Lenient
    }
}

/// A per-request cost ceiling expressed in USD.
#[derive(Debug, Clone, Copy)]
pub struct CostBudget {
    /// Estimated prompt token count for the request.
    pub prompt_tokens_estimate: u32,
    /// Upper bound on completion tokens.
    pub max_completion_tokens: u32,
    /// Maximum total USD cost permitted.
    pub max_cost_usd: f64,
    /// Behavior for models without pricing data.
    pub mode: BudgetMode,
}

impl CostBudget {
    /// Construct a budget in `Lenient` mode.
    #[must_use]
    pub fn new(prompt_tokens_estimate: u32, max_completion_tokens: u32, max_cost_usd: f64) -> Self {
        Self {
            prompt_tokens_estimate,
            max_completion_tokens,
            max_cost_usd,
            mode: BudgetMode::Lenient,
        }
    }

    /// Switch to strict mode (rejects unpriced models).
    #[must_use]
    pub fn strict(mut self) -> Self {
        self.mode = BudgetMode::Strict;
        self
    }

    /// Estimate request cost in USD for `model`. Returns `None` when the
    /// model's prompt or completion price is unknown.
    #[must_use]
    pub fn estimate_cost(&self, model: &ModelMetadata) -> Option<f64> {
        let prompt = model.pricing_prompt_usd_per_million?;
        let completion = model.pricing_completion_usd_per_million?;
        Some(
            f64::from(self.prompt_tokens_estimate) * prompt / 1_000_000.0
                + f64::from(self.max_completion_tokens) * completion / 1_000_000.0,
        )
    }

    /// True if `model` is admitted under this budget.
    #[must_use]
    pub fn admits(&self, model: &ModelMetadata) -> bool {
        match self.estimate_cost(model) {
            Some(cost) => cost <= self.max_cost_usd,
            None => matches!(self.mode, BudgetMode::Lenient),
        }
    }

    /// Classify a model against this budget. Returns `None` if admitted,
    /// otherwise the matching `RejectionReason`.
    #[must_use]
    pub fn rejection_reason(&self, model: &ModelMetadata) -> Option<RejectionReason> {
        match self.estimate_cost(model) {
            Some(cost) if cost > self.max_cost_usd => Some(RejectionReason::OverBudget {
                estimated_cost_usd: cost,
                max_cost_usd: self.max_cost_usd,
            }),
            Some(_) => None,
            None => {
                if matches!(self.mode, BudgetMode::Strict) {
                    Some(RejectionReason::UnpricedUnderStrictBudget)
                } else {
                    None
                }
            }
        }
    }
}

impl ProviderRegistry {
    /// List all models that satisfy the requirements AND fit the budget.
    /// Ordering matches the underlying registry (not fitness-sorted).
    pub fn list_within_budget(
        &self,
        requirements: &AgentRequirements,
        budget: &CostBudget,
    ) -> Vec<&ModelMetadata> {
        self.list_available(requirements)
            .into_iter()
            .filter(|m| budget.admits(m))
            .collect()
    }

    /// Select the best-fitness model that satisfies the requirements AND
    /// fits the budget. Returns a full [`SelectionResult`] with budget-
    /// rejected candidates surfaced in the `rejected` list, alongside the
    /// existing capability/cost-class rejections.
    ///
    /// # Errors
    ///
    /// Returns `LlmError::ProviderError` when no model satisfies both
    /// requirements and budget.
    pub fn select_within_budget(
        &self,
        requirements: &AgentRequirements,
        budget: &CostBudget,
    ) -> Result<SelectionResult, LlmError> {
        let mut full = self.select_with_details(requirements)?;

        // Walk the already-ranked candidates; move budget-rejected ones into
        // the `rejected` list. The remaining candidates are still in fitness
        // order so candidates[0] is the best in-budget pick.
        let mut in_budget: Vec<_> = Vec::with_capacity(full.candidates.len());
        for (candidate, fitness) in full.candidates {
            if let Some(reason) = budget.rejection_reason(&candidate) {
                full.rejected.push((candidate, reason));
            } else {
                in_budget.push((candidate, fitness));
            }
        }

        if in_budget.is_empty() {
            return Err(LlmError::ProviderError {
                message: format!(
                    "No model satisfies requirements within budget ${:.6} (prompt={}, max_output={})",
                    budget.max_cost_usd,
                    budget.prompt_tokens_estimate,
                    budget.max_completion_tokens
                ),
                code: None,
            });
        }

        let (selected, fitness) = in_budget[0].clone();
        full.candidates = in_budget;
        full.selected = selected;
        full.fitness = fitness;
        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_provider::selection::CostClass;

    fn meta_priced(
        provider: &str,
        model: &str,
        cost_class: CostClass,
        latency: u32,
        quality: f64,
        prompt_usd: f64,
        completion_usd: f64,
    ) -> ModelMetadata {
        ModelMetadata::new(provider, model, cost_class, latency, quality)
            .with_reasoning(true)
            .with_pricing(prompt_usd, completion_usd)
    }

    fn meta_unpriced(
        provider: &str,
        model: &str,
        cost_class: CostClass,
        latency: u32,
        quality: f64,
    ) -> ModelMetadata {
        ModelMetadata::new(provider, model, cost_class, latency, quality).with_reasoning(true)
    }

    #[test]
    fn estimate_cost_priced_model() {
        // gpt-4o-style: $2.50/M prompt, $10/M completion
        let model = meta_priced("openai", "gpt-4o", CostClass::Low, 2500, 0.92, 2.5, 10.0);
        let budget = CostBudget::new(1000, 500, 100.0);
        // 1000 * 2.50/1M + 500 * 10.0/1M = 0.0025 + 0.005 = 0.0075
        let cost = budget.estimate_cost(&model).unwrap();
        assert!((cost - 0.0075).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn estimate_cost_unpriced_model_is_none() {
        let model = meta_unpriced("local", "self-hosted", CostClass::Free, 800, 0.78);
        let budget = CostBudget::new(1000, 500, 100.0);
        assert!(budget.estimate_cost(&model).is_none());
    }

    #[test]
    fn admits_priced_under_budget() {
        let model = meta_priced("openai", "gpt-4o", CostClass::Low, 2500, 0.92, 2.5, 10.0);
        let budget = CostBudget::new(1000, 500, 0.01); // 0.0075 < 0.01
        assert!(budget.admits(&model));
    }

    #[test]
    fn rejects_priced_over_budget() {
        let model = meta_priced("openai", "o1", CostClass::VeryHigh, 15_000, 0.96, 15.0, 60.0);
        // 1000 * 15/1M + 500 * 60/1M = 0.015 + 0.03 = 0.045
        let budget = CostBudget::new(1000, 500, 0.01);
        assert!(!budget.admits(&model));
        let reason = budget.rejection_reason(&model).unwrap();
        assert!(matches!(reason, RejectionReason::OverBudget { .. }));
    }

    #[test]
    fn lenient_admits_unpriced_model() {
        let model = meta_unpriced("local", "self-hosted", CostClass::Free, 800, 0.78);
        let budget = CostBudget::new(1000, 500, 0.001);
        assert!(budget.admits(&model));
        assert!(budget.rejection_reason(&model).is_none());
    }

    #[test]
    fn strict_rejects_unpriced_model() {
        let model = meta_unpriced("local", "self-hosted", CostClass::Free, 800, 0.78);
        let budget = CostBudget::new(1000, 500, 0.001).strict();
        assert!(!budget.admits(&model));
        let reason = budget.rejection_reason(&model).unwrap();
        assert!(matches!(reason, RejectionReason::UnpricedUnderStrictBudget));
    }

    // Integration test for `ProviderRegistry::select_within_budget` lives in
    // tests/cost_routing_integration.rs — it needs the real registry, which
    // requires API key env vars and is awkward to mock from a unit test.

    #[test]
    fn budget_rejection_reason_message_is_formatted() {
        let model = meta_priced("openai", "o1", CostClass::VeryHigh, 15_000, 0.96, 15.0, 60.0);
        let budget = CostBudget::new(1000, 500, 0.01);
        let reason = budget.rejection_reason(&model).unwrap();
        let msg = format!("{reason}");
        assert!(msg.contains("exceeds budget"), "msg={msg}");
    }
}
