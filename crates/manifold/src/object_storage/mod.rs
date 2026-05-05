// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Object-store adapter builders for [`converge_storage`] contracts.

use std::sync::Arc;

use converge_storage::{ObjectStore, StorageConfig, StorageError, StorageUri};

#[cfg(feature = "object-gcs")]
mod gcs;
#[cfg(feature = "object-local")]
mod local;
#[cfg(feature = "object-s3")]
mod s3;

#[cfg(any(feature = "object-s3", test))]
fn resolve_s3_options<'a>(
    config: &'a StorageConfig,
    endpoint: Option<&'a String>,
    region: Option<&'a String>,
) -> (Option<&'a str>, Option<&'a str>) {
    (
        config.endpoint.as_deref().or(endpoint.map(String::as_str)),
        config.region.as_deref().or(region.map(String::as_str)),
    )
}

/// Build an [`ObjectStore`] from a [`StorageConfig`].
///
/// Returns a type-erased `Arc<dyn ObjectStore>` suitable for use across async
/// boundaries. The concrete backend is selected by the URI scheme and enabled
/// Manifold feature set.
pub fn build_store(config: &StorageConfig) -> Result<Arc<dyn ObjectStore>, StorageError> {
    match &config.uri {
        #[cfg(feature = "object-local")]
        StorageUri::Local(path) => local::build(path),

        #[cfg(feature = "object-s3")]
        StorageUri::S3 {
            bucket,
            endpoint,
            region,
        } => {
            let (endpoint, region) = resolve_s3_options(config, endpoint.as_ref(), region.as_ref());
            s3::build(bucket, endpoint, region)
        }

        #[cfg(feature = "object-gcs")]
        StorageUri::Gcs { bucket } if config.public => gcs::build_public(bucket),

        #[cfg(feature = "object-gcs")]
        StorageUri::Gcs { bucket } => gcs::build(bucket),

        #[allow(unreachable_patterns)]
        other => Err(StorageError::UnsupportedScheme(other.scheme().to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_s3_options_override_uri_values() {
        let config = StorageConfig {
            uri: StorageUri::S3 {
                bucket: "bucket".to_string(),
                endpoint: Some("http://uri-endpoint:9000".to_string()),
                region: Some("uri-region".to_string()),
            },
            prefix: None,
            public: false,
            endpoint: Some("http://config-endpoint:9000".to_string()),
            region: Some("config-region".to_string()),
        };

        let StorageUri::S3 {
            endpoint, region, ..
        } = &config.uri
        else {
            unreachable!();
        };

        let (resolved_endpoint, resolved_region) =
            resolve_s3_options(&config, endpoint.as_ref(), region.as_ref());

        assert_eq!(resolved_endpoint, Some("http://config-endpoint:9000"));
        assert_eq!(resolved_region, Some("config-region"));
    }

    #[test]
    fn uri_s3_options_are_used_as_fallback() {
        let config = StorageConfig {
            uri: StorageUri::S3 {
                bucket: "bucket".to_string(),
                endpoint: Some("http://uri-endpoint:9000".to_string()),
                region: Some("uri-region".to_string()),
            },
            prefix: None,
            public: false,
            endpoint: None,
            region: None,
        };

        let StorageUri::S3 {
            endpoint, region, ..
        } = &config.uri
        else {
            unreachable!();
        };

        let (resolved_endpoint, resolved_region) =
            resolve_s3_options(&config, endpoint.as_ref(), region.as_ref());

        assert_eq!(resolved_endpoint, Some("http://uri-endpoint:9000"));
        assert_eq!(resolved_region, Some("uri-region"));
    }
}
