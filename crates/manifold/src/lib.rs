// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Generic adapter implementations for Converge contracts.
//!
//! Manifold owns interchangeable operational adapters. It imports Converge
//! contracts and external SDKs; Converge does not import Manifold.

pub mod experience;
pub mod object_storage;
pub mod vector;

pub use converge_storage::{
    GetResult, ObjectPath, ObjectStore, PutResult, StorageConfig, StorageError, StorageUri,
};
