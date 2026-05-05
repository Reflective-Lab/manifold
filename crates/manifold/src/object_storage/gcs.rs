// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::gcp::GoogleCloudStorageBuilder;

use crate::StorageError;

pub fn build(bucket: &str) -> Result<Arc<dyn ObjectStore>, StorageError> {
    let store = GoogleCloudStorageBuilder::from_env()
        .with_bucket_name(bucket)
        .build()
        .map_err(|e| StorageError::Config(format!("GCS setup failed: {e}")))?;

    tracing::info!(bucket, "connected to Google Cloud Storage");
    Ok(Arc::new(store))
}

pub fn build_public(bucket: &str) -> Result<Arc<dyn ObjectStore>, StorageError> {
    let store = GoogleCloudStorageBuilder::new()
        .with_bucket_name(bucket)
        .with_skip_signature(true)
        .build()
        .map_err(|e| StorageError::Config(format!("GCS public-access setup failed: {e}")))?;

    tracing::info!(
        bucket,
        "connected to Google Cloud Storage (public, unsigned)"
    );
    Ok(Arc::new(store))
}
