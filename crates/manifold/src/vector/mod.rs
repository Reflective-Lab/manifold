// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Vector-store capability adapters.

#[cfg(feature = "vector-lancedb")]
mod lancedb;

#[cfg(feature = "vector-lancedb")]
pub use lancedb::LanceStore;

pub use converge_core::capability::{
    CapabilityError, VectorMatch, VectorQuery, VectorRecall, VectorRecord,
};
