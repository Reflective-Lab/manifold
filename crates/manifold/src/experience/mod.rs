// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Runtime persistence adapters for Converge experience-store contracts.

#[cfg(feature = "experience-lancedb")]
mod lancedb_store;
#[cfg(feature = "experience-surrealdb")]
mod surrealdb_store;

#[cfg(feature = "experience-lancedb")]
pub use lancedb_store::{LanceDbConfig, LanceDbExperienceStore, SimilarEvent, VectorEvent};
#[cfg(feature = "experience-surrealdb")]
pub use surrealdb_store::{SurrealDbConfig, SurrealDbExperienceStore};
