// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;

use crate::StorageError;

pub fn build(
    bucket: &str,
    endpoint: Option<&str>,
    region: Option<&str>,
) -> Result<Arc<dyn ObjectStore>, StorageError> {
    let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);

    if let Some(endpoint) = endpoint {
        builder = builder.with_endpoint(endpoint).with_allow_http(true);
    }

    if let Some(region) = region {
        builder = builder.with_region(region);
    }

    let store = builder
        .build()
        .map_err(|e| StorageError::Config(format!("S3 setup failed: {e}")))?;

    tracing::info!(bucket, endpoint, region, "connected to S3-compatible store");
    Ok(Arc::new(store))
}
