// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Generic adapter implementations for Converge contracts.
//!
//! Manifold owns interchangeable operational adapters. It imports Converge
//! contracts and external SDKs; Converge does not import Manifold.

pub mod experience;
pub mod llm;
pub mod object_storage;
#[cfg(feature = "registry")]
pub mod registry_loader;
pub mod secret;
pub mod vector;

pub mod model_selection;

pub use converge_storage::{
    GetResult, ObjectPath, ObjectStore, PutResult, StorageConfig, StorageError, StorageUri,
};
pub use llm::*;
pub use secret::{EnvSecretProvider, SecretError, SecretProvider, SecretString};
