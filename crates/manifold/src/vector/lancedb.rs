// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! `LanceDB` embedded vector store implementation.
//!
//! `LanceDB` is an embedded vector database built on Apache Arrow, providing
//! fast vector similarity search without requiring a separate server process.
//!
//! # Example
//!
//! ```ignore
//! use converge_provider::vector::LanceStore;
//! use converge_core::capability::{VectorRecall, VectorRecord, VectorQuery};
//!
//! let store = LanceStore::new("/tmp/vectors", "embeddings", 1024)?;
//!
//! store.upsert(&VectorRecord {
//!     id: "doc-1".into(),
//!     vector: vec![0.1; 1024],
//!     payload: serde_json::json!({"title": "Hello"}),
//! })?;
//!
//! let matches = store.query(&VectorQuery::new(vec![0.1; 1024], 10))?;
//! ```

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use converge_core::capability::{
    CapabilityError, VectorMatch, VectorQuery, VectorRecall, VectorRecord,
};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::{Arc, RwLock};
use tracing::debug;

/// `LanceDB` embedded vector store.
///
/// This store uses `LanceDB` for fast vector similarity search with an
/// embedded database (no separate server required).
///
/// # Thread Safety
///
/// This store is thread-safe and can be shared across threads.
///
/// # Storage
///
/// Data is stored on disk at the specified path. The store creates
/// a single table for all vectors.
pub struct LanceStore {
    /// `LanceDB` connection.
    db: lancedb::Connection,
    /// Cached table reference.
    table: RwLock<Option<lancedb::Table>>,
    /// Table name.
    table_name: String,
    /// Vector dimensions (must match all inserted vectors).
    vector_dim: usize,
    /// Tokio runtime handle for async operations.
    runtime: tokio::runtime::Handle,
    /// Owned runtime (kept alive when we create our own).
    _owned_runtime: Option<tokio::runtime::Runtime>,
}

impl LanceStore {
    /// Creates a new `LanceDB` vector store.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path for the database files
    /// * `table_name` - Name of the table to store vectors
    /// * `vector_dim` - Dimensionality of vectors (e.g., 1024 for BGE-large)
    ///
    /// # Errors
    ///
    /// Returns error if the database cannot be opened.
    pub fn new(
        path: impl Into<String>,
        table_name: impl Into<String>,
        vector_dim: usize,
    ) -> Result<Self, CapabilityError> {
        let path = path.into();
        let table_name = table_name.into();

        // Get or create a tokio runtime handle
        let (runtime, owned_runtime) = if let Ok(handle) = tokio::runtime::Handle::try_current() {
            (handle, None)
        } else {
            // No runtime exists, create one and keep it alive
            let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
            let handle = rt.handle().clone();
            (handle, Some(rt))
        };

        // Connect to LanceDB
        let db = runtime
            .block_on(lancedb::connect(&path).execute())
            .map_err(|e| CapabilityError::store(format!("Failed to connect to LanceDB: {e}")))?;

        debug!(path = %path, table = %table_name, dim = vector_dim, "Connected to LanceDB");

        Ok(Self {
            db,
            table: RwLock::new(None),
            table_name,
            vector_dim,
            runtime,
            _owned_runtime: owned_runtime,
        })
    }

    /// Creates a new store with a provided runtime handle.
    ///
    /// Use this when you have an existing tokio runtime.
    pub fn with_runtime(
        path: impl Into<String>,
        table_name: impl Into<String>,
        vector_dim: usize,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, CapabilityError> {
        let path = path.into();
        let table_name = table_name.into();

        let db = runtime
            .block_on(lancedb::connect(&path).execute())
            .map_err(|e| CapabilityError::store(format!("Failed to connect to LanceDB: {e}")))?;

        Ok(Self {
            db,
            table: RwLock::new(None),
            table_name,
            vector_dim,
            runtime,
            _owned_runtime: None,
        })
    }

    /// Returns the vector dimensionality.
    #[must_use]
    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    /// Returns the table name.
    #[must_use]
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Gets or creates the table.
    fn get_or_create_table(&self) -> Result<lancedb::Table, CapabilityError> {
        // Check if we have a cached table
        {
            let guard = self.table.read().expect("Lock poisoned");
            if let Some(ref table) = *guard {
                return Ok(table.clone());
            }
        }

        // Need to create or open the table
        let mut guard = self.table.write().expect("Lock poisoned");

        // Double-check after acquiring write lock
        if let Some(ref table) = *guard {
            return Ok(table.clone());
        }

        // Check if table exists
        let table_names: Vec<String> = self
            .runtime
            .block_on(self.db.table_names().execute())
            .map_err(|e| CapabilityError::store(format!("Failed to list tables: {e}")))?;

        let table = if table_names.contains(&self.table_name) {
            // Open existing table
            self.runtime
                .block_on(self.db.open_table(&self.table_name).execute())
                .map_err(|e| CapabilityError::store(format!("Failed to open table: {e}")))?
        } else {
            // Create new table with schema
            let schema = self.create_arrow_schema();
            let empty_batch = self.create_empty_batch(&schema)?;
            let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], Arc::clone(&schema));

            self.runtime
                .block_on(
                    self.db
                        .create_table(&self.table_name, Box::new(batches))
                        .execute(),
                )
                .map_err(|e| CapabilityError::store(format!("Failed to create table: {e}")))?
        };

        *guard = Some(table.clone());
        Ok(table)
    }

    /// Creates the Arrow schema for the table.
    fn create_arrow_schema(&self) -> Arc<ArrowSchema> {
        let vector_field = Field::new("item", DataType::Float32, true);
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(vector_field), self.vector_dim as i32),
                false,
            ),
            Field::new("payload", DataType::Utf8, true),
        ]))
    }

    /// Creates an empty batch for table initialization.
    fn create_empty_batch(
        &self,
        schema: &Arc<ArrowSchema>,
    ) -> Result<RecordBatch, CapabilityError> {
        let ids: Vec<&str> = vec![];
        let id_array = StringArray::from(ids.clone());

        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let empty_values = Float32Array::from(Vec::<f32>::new());
        let vector_array = FixedSizeListArray::try_new(
            vector_field,
            self.vector_dim as i32,
            Arc::new(empty_values),
            None,
        )
        .map_err(|e| CapabilityError::store(format!("Failed to create vector array: {e}")))?;

        let payload_array = StringArray::from(ids);

        RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(payload_array),
            ],
        )
        .map_err(|e| CapabilityError::store(format!("Failed to create batch: {e}")))
    }

    /// Converts records to a `RecordBatch`.
    fn records_to_batch(&self, records: &[VectorRecord]) -> Result<RecordBatch, CapabilityError> {
        if records.is_empty() {
            return Err(CapabilityError::invalid_input("No records to insert"));
        }

        // Validate vector dimensions
        for record in records {
            if record.vector.len() != self.vector_dim {
                return Err(CapabilityError::invalid_input(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.vector_dim,
                    record.vector.len()
                )));
            }
        }

        let schema = self.create_arrow_schema();

        // Build arrays
        let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        let id_array = StringArray::from(ids);

        // Flatten all vectors into a single array
        let all_values: Vec<f32> = records
            .iter()
            .flat_map(|r| r.vector.iter().copied())
            .collect();
        let values_array = Float32Array::from(all_values);

        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = FixedSizeListArray::try_new(
            vector_field,
            self.vector_dim as i32,
            Arc::new(values_array),
            None,
        )
        .map_err(|e| CapabilityError::store(format!("Failed to create vector array: {e}")))?;

        // Serialize payloads to JSON strings
        let payloads: Vec<String> = records
            .iter()
            .map(|r| serde_json::to_string(&r.payload).unwrap_or_else(|_| "{}".to_string()))
            .collect();
        let payload_array = StringArray::from(
            payloads
                .iter()
                .map(std::string::String::as_str)
                .collect::<Vec<_>>(),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(payload_array),
            ],
        )
        .map_err(|e| CapabilityError::store(format!("Failed to create batch: {e}")))
    }
}

impl VectorRecall for LanceStore {
    fn name(&self) -> &'static str {
        "lancedb"
    }

    fn upsert(&self, record: &VectorRecord) -> Result<(), CapabilityError> {
        self.upsert_batch(std::slice::from_ref(record))
    }

    fn upsert_batch(&self, records: &[VectorRecord]) -> Result<(), CapabilityError> {
        if records.is_empty() {
            return Ok(());
        }

        let table = self.get_or_create_table()?;
        let batch = self.records_to_batch(records)?;

        // For upsert, we use merge_insert
        let schema = batch.schema();

        self.runtime
            .block_on(async {
                let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
                let mut op = table.merge_insert(&["id"]);
                op.when_matched_update_all(None);
                op.when_not_matched_insert_all();
                op.execute(Box::new(batches)).await
            })
            .map_err(|e| CapabilityError::store(format!("Failed to upsert: {e}")))?;

        debug!(count = records.len(), "Upserted records to LanceDB");
        Ok(())
    }

    fn query(&self, query: &VectorQuery) -> Result<Vec<VectorMatch>, CapabilityError> {
        if query.vector.len() != self.vector_dim {
            return Err(CapabilityError::invalid_input(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.vector_dim,
                query.vector.len()
            )));
        }

        let table = self.get_or_create_table()?;

        let results: Vec<RecordBatch> = self.runtime.block_on(async {
            table
                .query()
                .nearest_to(query.vector.as_slice())
                .map_err(|e| CapabilityError::store(format!("Query setup failed: {e}")))?
                .limit(query.top_k)
                .execute()
                .await
                .map_err(|e| CapabilityError::store(format!("Query failed: {e}")))?
                .try_collect::<Vec<RecordBatch>>()
                .await
                .map_err(|e| CapabilityError::store(format!("Failed to collect results: {e}")))
        })?;

        let mut matches = Vec::new();

        for batch in results {
            let id_col: &StringArray = batch
                .column_by_name("id")
                .and_then(|c: &ArrayRef| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| CapabilityError::store("Missing id column"))?;

            let score_col: &Float32Array = batch
                .column_by_name("_distance")
                .and_then(|c: &ArrayRef| c.as_any().downcast_ref::<Float32Array>())
                .ok_or_else(|| CapabilityError::store("Missing _distance column"))?;

            let payload_col: Option<&StringArray> = batch
                .column_by_name("payload")
                .and_then(|c: &ArrayRef| c.as_any().downcast_ref::<StringArray>());

            for i in 0..batch.num_rows() {
                let id = id_col.value(i).to_string();
                // LanceDB returns distance, convert to similarity (1 - distance for L2, or use directly for cosine)
                let distance = f64::from(score_col.value(i));
                // For cosine distance, similarity = 1 - distance
                let score = 1.0 - distance;

                // Apply min_score filter
                if let Some(min_score) = query.min_score
                    && score < min_score
                {
                    continue;
                }

                let payload = payload_col
                    .map(|col: &StringArray| col.value(i))
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Null);

                matches.push(VectorMatch { id, score, payload });
            }
        }

        // Sort by score descending (highest similarity first)
        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(query.top_k);

        debug!(count = matches.len(), "Query returned matches");
        Ok(matches)
    }

    fn delete(&self, id: &str) -> Result<(), CapabilityError> {
        let table = self.get_or_create_table()?;

        self.runtime
            .block_on(table.delete(&format!("id = '{id}'")))
            .map_err(|e| CapabilityError::store(format!("Failed to delete: {e}")))?;

        debug!(id = %id, "Deleted record from LanceDB");
        Ok(())
    }

    fn clear(&self) -> Result<(), CapabilityError> {
        // Drop and recreate the table
        let table_names: Vec<String> = self
            .runtime
            .block_on(self.db.table_names().execute())
            .map_err(|e| CapabilityError::store(format!("Failed to list tables: {e}")))?;

        if table_names.contains(&self.table_name) {
            self.runtime
                .block_on(self.db.drop_table(&self.table_name, &[]))
                .map_err(|e| CapabilityError::store(format!("Failed to drop table: {e}")))?;
        }

        // Clear cached table reference
        {
            let mut guard = self.table.write().expect("Lock poisoned");
            *guard = None;
        }

        debug!(table = %self.table_name, "Cleared LanceDB table");
        Ok(())
    }

    fn count(&self) -> Result<usize, CapabilityError> {
        let table = self.get_or_create_table()?;

        let count = self
            .runtime
            .block_on(table.count_rows(None))
            .map_err(|e| CapabilityError::store(format!("Failed to count rows: {e}")))?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn create_test_store() -> (LanceStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceStore::new(temp_dir.path().to_str().unwrap(), "test_vectors", 3).unwrap();
        (store, temp_dir)
    }

    #[test]
    fn test_store_creation() {
        let (store, _dir) = create_test_store();
        assert_eq!(store.name(), "lancedb");
        assert_eq!(store.vector_dim(), 3);
        assert_eq!(store.table_name(), "test_vectors");
    }

    #[test]
    fn test_upsert_and_count() {
        let (store, _dir) = create_test_store();

        store
            .upsert(&VectorRecord {
                id: "doc-1".into(),
                vector: vec![1.0, 0.0, 0.0],
                payload: json!({"title": "Document 1"}),
            })
            .unwrap();

        assert_eq!(store.count().unwrap(), 1);

        store
            .upsert(&VectorRecord {
                id: "doc-2".into(),
                vector: vec![0.0, 1.0, 0.0],
                payload: json!({"title": "Document 2"}),
            })
            .unwrap();

        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn test_upsert_overwrites() {
        let (store, _dir) = create_test_store();

        store
            .upsert(&VectorRecord {
                id: "doc-1".into(),
                vector: vec![1.0, 0.0, 0.0],
                payload: json!({"version": 1}),
            })
            .unwrap();

        store
            .upsert(&VectorRecord {
                id: "doc-1".into(),
                vector: vec![0.0, 1.0, 0.0],
                payload: json!({"version": 2}),
            })
            .unwrap();

        // Should still be 1 record (upsert, not insert)
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn test_query() {
        let (store, _dir) = create_test_store();

        store
            .upsert_batch(&[
                VectorRecord {
                    id: "doc-1".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    payload: json!({"title": "Doc 1"}),
                },
                VectorRecord {
                    id: "doc-2".into(),
                    vector: vec![0.9, 0.1, 0.0],
                    payload: json!({"title": "Doc 2"}),
                },
                VectorRecord {
                    id: "doc-3".into(),
                    vector: vec![0.0, 1.0, 0.0],
                    payload: json!({"title": "Doc 3"}),
                },
            ])
            .unwrap();

        let matches = store
            .query(&VectorQuery::new(vec![1.0, 0.0, 0.0], 2))
            .unwrap();

        assert_eq!(matches.len(), 2);
        // First match should be doc-1 (exact match)
        assert_eq!(matches[0].id, "doc-1");
        // Second should be doc-2 (close match)
        assert_eq!(matches[1].id, "doc-2");
    }

    #[test]
    fn test_query_with_min_score() {
        let (store, _dir) = create_test_store();

        store
            .upsert_batch(&[
                VectorRecord {
                    id: "close".into(),
                    vector: vec![0.95, 0.05, 0.0],
                    payload: json!({}),
                },
                VectorRecord {
                    id: "far".into(),
                    vector: vec![0.0, 0.0, 1.0],
                    payload: json!({}),
                },
            ])
            .unwrap();

        let matches = store
            .query(&VectorQuery::new(vec![1.0, 0.0, 0.0], 10).with_min_score(0.5))
            .unwrap();

        // Only the close match should pass the threshold
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "close");
    }

    #[test]
    fn test_delete() {
        let (store, _dir) = create_test_store();

        store
            .upsert_batch(&[
                VectorRecord {
                    id: "doc-1".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    payload: json!({}),
                },
                VectorRecord {
                    id: "doc-2".into(),
                    vector: vec![0.0, 1.0, 0.0],
                    payload: json!({}),
                },
            ])
            .unwrap();

        assert_eq!(store.count().unwrap(), 2);

        store.delete("doc-1").unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn test_clear() {
        let (store, _dir) = create_test_store();

        store
            .upsert_batch(&[
                VectorRecord {
                    id: "doc-1".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    payload: json!({}),
                },
                VectorRecord {
                    id: "doc-2".into(),
                    vector: vec![0.0, 1.0, 0.0],
                    payload: json!({}),
                },
            ])
            .unwrap();

        assert_eq!(store.count().unwrap(), 2);

        store.clear().unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let (store, _dir) = create_test_store();

        let result = store.upsert(&VectorRecord {
            id: "bad".into(),
            vector: vec![1.0, 0.0], // Wrong dimension (2 instead of 3)
            payload: json!({}),
        });

        assert!(result.is_err());
    }
}
