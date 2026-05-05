// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::path::Path;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

use crate::StorageError;

pub fn build(path: &Path) -> Result<Arc<dyn ObjectStore>, StorageError> {
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|e| {
            StorageError::Config(format!("failed to create local storage dir: {e}"))
        })?;
        tracing::info!(path = %path.display(), "created local storage directory");
    }

    let store = LocalFileSystem::new_with_prefix(path)
        .map_err(|e| StorageError::Config(format!("local filesystem setup failed: {e}")))?;

    Ok(Arc::new(store))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_creates_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c");
        assert!(!nested.exists());

        let store = build(&nested).unwrap();
        assert!(nested.exists());
        assert!(Arc::strong_count(&store) == 1);
    }

    #[test]
    fn build_succeeds_on_existing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = build(dir.path()).unwrap();
        assert!(Arc::strong_count(&store) == 1);
    }
}
