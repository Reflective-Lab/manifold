// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Fuzzy-logic model selection scoring.
//!
//! Replaces the previous crisp/bucketed fitness scoring with a Mamdani fuzzy
//! inference system that maps three crisp inputs (cost, latency, quality)
//! through linguistic terms (cheap/moderate/expensive, fast/medium/slow,
//! low/medium/high) and rules to a preference output (weak/moderate/strong).
//!
//! # Why fuzzy
//!
//! The old crisp scoring (5 cost buckets × discrete latency × raw quality,
//! summed with fixed weights) had hard jumps at bucket boundaries and gave
//! a single numeric score with no explanation. Fuzzy gives smooth membership
//! transitions, encodes domain knowledge as linguistic rules, and produces
//! a trace of which rules fired (`activated_rules`) — useful for explaining
//! governed selection decisions to enterprise customers.
//!
//! Hard constraints (capability requirements, sovereignty, compliance) stay
//! crisp — those are binary by nature.

use converge_fuzzy_inference::{
    FuzzyConsequent, FuzzyExpression, FuzzyInferenceInput, FuzzyInferenceOutput, FuzzyRule,
    FuzzySet, LinguisticVariable, MembershipFunction, infer,
};
use converge_provider::selection::CostClass;
use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::model_selection::ModelMetadata;

const COST_VAR: &str = "cost";
const LATENCY_VAR: &str = "latency";
const QUALITY_VAR: &str = "quality";
const PREFERENCE_VAR: &str = "preference";

/// Compute a fuzzy fitness output for a single model.
///
/// Returns `None` if the inference fails (e.g., invalid configuration —
/// should not happen in practice because the variable/rule set is statically
/// defined).
#[must_use]
pub fn fuzzy_fitness(metadata: &ModelMetadata) -> Option<FuzzyInferenceOutput> {
    let inputs = BTreeMap::from([
        (COST_VAR.to_string(), cost_proxy_usd_per_million(metadata)),
        (
            LATENCY_VAR.to_string(),
            f64::from(metadata.typical_latency_ms),
        ),
        (QUALITY_VAR.to_string(), metadata.quality),
    ]);

    let input = FuzzyInferenceInput {
        inputs,
        variables: selection_variables().clone(),
        rules: selection_rules().clone(),
    };

    infer(&input).ok()
}

/// Scalar preference score in [0.0, 1.0] derived from the fuzzy output.
///
/// Uses a weighted-average defuzzification over the three preference levels:
/// `weak` weighted 0.0, `moderate` weighted 0.5, `strong` weighted 1.0.
/// Returns 0.0 if no rules fired or inference failed.
#[must_use]
pub fn preference_score(metadata: &ModelMetadata) -> f64 {
    let Some(output) = fuzzy_fitness(metadata) else {
        return 0.0;
    };
    weighted_preference(&output)
}

fn weighted_preference(output: &FuzzyInferenceOutput) -> f64 {
    let weak = output
        .memberships
        .get(&format!("{PREFERENCE_VAR}.weak"))
        .map(|m| m.value())
        .unwrap_or(0.0);
    let moderate = output
        .memberships
        .get(&format!("{PREFERENCE_VAR}.moderate"))
        .map(|m| m.value())
        .unwrap_or(0.0);
    let strong = output
        .memberships
        .get(&format!("{PREFERENCE_VAR}.strong"))
        .map(|m| m.value())
        .unwrap_or(0.0);

    let denom = weak + moderate + strong;
    if denom < f64::EPSILON {
        return 0.0;
    }
    (0.0 * weak + 0.5 * moderate + 1.0 * strong) / denom
}

/// Bundle of fitness signals derived from a single fuzzy inference pass.
///
/// Used to populate the legacy `FitnessBreakdown` shape while also exposing
/// the rule trace for explainability. Cheaper than calling four individual
/// membership helpers (which would each run inference).
#[derive(Debug, Clone, Default)]
pub struct FitnessSummary {
    /// Weighted-average preference score in [0.0, 1.0].
    pub preference: f64,
    /// Membership in cost.cheap (0..1) — populates legacy `cost_score`.
    pub cost_cheap: f64,
    /// Membership in latency.fast (0..1) — populates legacy `latency_score`.
    pub latency_fast: f64,
    /// Membership in quality.high (0..1) — populates legacy `quality_score`.
    pub quality_high: f64,
    /// IDs of the rules that fired, in evaluation order. Empty when no rule fires.
    pub activated_rule_ids: Vec<String>,
}

/// Compute the full fitness summary in a single fuzzy inference pass.
#[must_use]
pub fn fitness_summary(metadata: &ModelMetadata) -> FitnessSummary {
    let Some(output) = fuzzy_fitness(metadata) else {
        return FitnessSummary::default();
    };
    FitnessSummary {
        preference: weighted_preference(&output),
        cost_cheap: input_membership(&output, COST_VAR, "cheap"),
        latency_fast: input_membership(&output, LATENCY_VAR, "fast"),
        quality_high: input_membership(&output, QUALITY_VAR, "high"),
        activated_rule_ids: output
            .activated_rules
            .iter()
            .map(|r| r.id.clone())
            .collect(),
    }
}

fn input_membership(output: &FuzzyInferenceOutput, variable: &str, set: &str) -> f64 {
    output
        .input_memberships
        .get(variable)
        .and_then(|sets| sets.get(set))
        .map(|m| m.value())
        .unwrap_or(0.0)
}

/// Best-effort cost proxy in USD per million prompt tokens.
///
/// Prefers live catalog pricing when available. Falls back to a
/// representative value per cost class.
fn cost_proxy_usd_per_million(metadata: &ModelMetadata) -> f64 {
    if let Some(prompt) = metadata.pricing_prompt_usd_per_million {
        return prompt;
    }
    match metadata.cost_class {
        CostClass::Free => 0.05,
        CostClass::VeryLow => 0.5,
        CostClass::Low => 1.5,
        CostClass::Medium => 5.0,
        CostClass::High => 15.0,
        CostClass::VeryHigh => 50.0,
    }
}

// ============================================================================
// Linguistic-variable and rule definitions (computed once, cached forever)
// ============================================================================

fn selection_variables() -> &'static Vec<LinguisticVariable> {
    static VARS: OnceLock<Vec<LinguisticVariable>> = OnceLock::new();
    VARS.get_or_init(build_variables)
}

fn selection_rules() -> &'static Vec<FuzzyRule> {
    static RULES: OnceLock<Vec<FuzzyRule>> = OnceLock::new();
    RULES.get_or_init(build_rules)
}

fn build_variables() -> Vec<LinguisticVariable> {
    vec![
        LinguisticVariable {
            name: COST_VAR.to_string(),
            sets: vec![
                FuzzySet {
                    name: "cheap".to_string(),
                    function: MembershipFunction::LeftShoulder {
                        start: 0.5,
                        end: 2.0,
                    },
                },
                FuzzySet {
                    name: "moderate".to_string(),
                    function: MembershipFunction::Triangular {
                        min: 1.0,
                        peak: 4.0,
                        max: 12.0,
                    },
                },
                FuzzySet {
                    name: "expensive".to_string(),
                    function: MembershipFunction::RightShoulder {
                        start: 8.0,
                        end: 25.0,
                    },
                },
            ],
        },
        LinguisticVariable {
            name: LATENCY_VAR.to_string(),
            sets: vec![
                FuzzySet {
                    name: "fast".to_string(),
                    function: MembershipFunction::LeftShoulder {
                        start: 1500.0,
                        end: 3500.0,
                    },
                },
                FuzzySet {
                    name: "medium".to_string(),
                    function: MembershipFunction::Triangular {
                        min: 2000.0,
                        peak: 4500.0,
                        max: 8000.0,
                    },
                },
                FuzzySet {
                    name: "slow".to_string(),
                    function: MembershipFunction::RightShoulder {
                        start: 6000.0,
                        end: 12_000.0,
                    },
                },
            ],
        },
        LinguisticVariable {
            name: QUALITY_VAR.to_string(),
            sets: vec![
                FuzzySet {
                    name: "low".to_string(),
                    function: MembershipFunction::LeftShoulder {
                        start: 0.65,
                        end: 0.80,
                    },
                },
                FuzzySet {
                    name: "medium".to_string(),
                    function: MembershipFunction::Triangular {
                        min: 0.70,
                        peak: 0.85,
                        max: 0.93,
                    },
                },
                FuzzySet {
                    name: "high".to_string(),
                    function: MembershipFunction::RightShoulder {
                        start: 0.88,
                        end: 0.96,
                    },
                },
            ],
        },
        LinguisticVariable {
            name: PREFERENCE_VAR.to_string(),
            sets: vec![
                FuzzySet {
                    name: "weak".to_string(),
                    function: MembershipFunction::Triangular {
                        min: 0.0,
                        peak: 0.2,
                        max: 0.5,
                    },
                },
                FuzzySet {
                    name: "moderate".to_string(),
                    function: MembershipFunction::Triangular {
                        min: 0.3,
                        peak: 0.5,
                        max: 0.7,
                    },
                },
                FuzzySet {
                    name: "strong".to_string(),
                    function: MembershipFunction::Triangular {
                        min: 0.5,
                        peak: 0.8,
                        max: 1.0,
                    },
                },
            ],
        },
    ]
}

fn is_(variable: &str, set: &str) -> FuzzyExpression {
    FuzzyExpression::Is {
        variable: variable.to_string(),
        set: set.to_string(),
    }
}

fn and_(terms: Vec<FuzzyExpression>) -> FuzzyExpression {
    FuzzyExpression::And { terms }
}

fn rule(id: &str, when: FuzzyExpression, then_set: &str) -> FuzzyRule {
    FuzzyRule {
        id: Some(id.to_string()),
        when,
        then: FuzzyConsequent {
            variable: PREFERENCE_VAR.to_string(),
            set: then_set.to_string(),
        },
        weight: None,
    }
}

fn build_rules() -> Vec<FuzzyRule> {
    vec![
        // Sweet spot: cheap or moderate cost with high quality → strong preference.
        rule(
            "cheap+high-quality",
            and_(vec![is_(COST_VAR, "cheap"), is_(QUALITY_VAR, "high")]),
            "strong",
        ),
        rule(
            "moderate+high-quality",
            and_(vec![is_(COST_VAR, "moderate"), is_(QUALITY_VAR, "high")]),
            "strong",
        ),
        rule(
            "cheap+medium-quality",
            and_(vec![is_(COST_VAR, "cheap"), is_(QUALITY_VAR, "medium")]),
            "strong",
        ),
        // Acceptable: moderate cost + medium quality, or expensive + high quality.
        rule(
            "moderate+medium-quality",
            and_(vec![is_(COST_VAR, "moderate"), is_(QUALITY_VAR, "medium")]),
            "moderate",
        ),
        rule(
            "expensive+high-quality",
            and_(vec![is_(COST_VAR, "expensive"), is_(QUALITY_VAR, "high")]),
            "moderate",
        ),
        rule(
            "cheap+low-quality",
            and_(vec![is_(COST_VAR, "cheap"), is_(QUALITY_VAR, "low")]),
            "moderate",
        ),
        // Discouraged: expensive with mid/low quality.
        rule(
            "expensive+medium-quality",
            and_(vec![is_(COST_VAR, "expensive"), is_(QUALITY_VAR, "medium")]),
            "weak",
        ),
        rule(
            "expensive+low-quality",
            and_(vec![is_(COST_VAR, "expensive"), is_(QUALITY_VAR, "low")]),
            "weak",
        ),
        // Latency effects: slow latency penalizes, fast+high-quality boosts.
        rule("slow-penalty", is_(LATENCY_VAR, "slow"), "weak"),
        rule(
            "fast+high-quality-boost",
            and_(vec![is_(LATENCY_VAR, "fast"), is_(QUALITY_VAR, "high")]),
            "strong",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_provider::selection::{ComplianceLevel, DataSovereignty};

    fn meta(
        provider: &str,
        model: &str,
        cost: CostClass,
        latency: u32,
        quality: f64,
    ) -> ModelMetadata {
        ModelMetadata::new(provider, model, cost, latency, quality)
    }

    #[test]
    fn cheap_and_high_quality_scores_higher_than_expensive() {
        let cheap_good = meta("a", "cheap-good", CostClass::VeryLow, 1500, 0.92);
        let expensive_good = meta("b", "expensive-good", CostClass::VeryHigh, 12_000, 0.95);
        let s_cheap = preference_score(&cheap_good);
        let s_expensive = preference_score(&expensive_good);
        assert!(
            s_cheap > s_expensive,
            "cheap-good ({s_cheap:.3}) should beat expensive-good ({s_expensive:.3})"
        );
    }

    #[test]
    fn slow_latency_lowers_preference() {
        let fast = meta("a", "fast", CostClass::Low, 1500, 0.90);
        let slow = meta("b", "slow", CostClass::Low, 11_000, 0.90);
        assert!(
            preference_score(&fast) > preference_score(&slow),
            "fast model should beat slow model"
        );
    }

    #[test]
    fn live_catalog_pricing_overrides_cost_class() {
        let mut m = meta("test", "model-with-price", CostClass::High, 2000, 0.90);
        m.pricing_prompt_usd_per_million = Some(0.5); // cheap, despite High cost_class
        let summary = fitness_summary(&m);
        assert!(
            summary.cost_cheap > 0.9,
            "expected cheap membership > 0.9, got {}",
            summary.cost_cheap
        );
    }

    #[test]
    fn preference_score_in_unit_range() {
        let m = meta("test", "any", CostClass::Medium, 3000, 0.80);
        let score = preference_score(&m);
        assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
    }

    #[test]
    fn unused_metadata_fields_dont_affect_scoring() {
        // Sanity: changing capability flags does not change the fuzzy score
        // (those are hard-constraint filters handled elsewhere).
        let mut m1 = meta("a", "x", CostClass::Low, 2000, 0.85);
        let m2 = m1.clone();
        m1 = m1
            .with_tool_use(true)
            .with_vision(true)
            .with_data_sovereignty(DataSovereignty::EU)
            .with_compliance(ComplianceLevel::GDPR);
        assert!((preference_score(&m1) - preference_score(&m2)).abs() < 1e-9);
    }
}
