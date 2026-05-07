// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! LanceDB-backed experience store for vector-indexed event retrieval.
//!
//! This store extends the standard append/query model with vector similarity
//! search over experience events. Events are stored with an embedding vector
//! that enables "find similar events" queries.
//!
//! The embedding vector is caller-provided — this store does not compute
//! embeddings itself (that belongs in converge-llm or a dedicated embedding
//! service).

use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::runtime::Runtime;

use converge_core::{
    ArtifactId, ArtifactKind, CorrelationId, EventId, EventQuery, ExperienceEvent,
    ExperienceEventEnvelope, ExperienceStore, ExperienceStoreError, ExperienceStoreResult,
    LifecycleEvent, ReplayTrace, TenantId, TraceLinkId,
};

/// Configuration for LanceDB connection.
#[derive(Debug, Clone)]
pub struct LanceDbConfig {
    /// Path to the LanceDB database directory (local) or URI (remote).
    pub uri: String,
    /// Embedding vector dimension. Must match the model used to generate vectors.
    pub embedding_dim: usize,
}

impl LanceDbConfig {
    /// Create a new config.
    #[must_use]
    pub fn new(uri: impl Into<String>, embedding_dim: usize) -> Self {
        Self {
            uri: uri.into(),
            embedding_dim,
        }
    }
}

/// A vector-indexed experience event for similarity search.
#[derive(Debug, Clone)]
pub struct VectorEvent {
    /// The experience event envelope.
    pub envelope: ExperienceEventEnvelope,
    /// Embedding vector for this event (dimension must match config).
    pub vector: Vec<f32>,
}

/// Result of a vector similarity search.
#[derive(Debug, Clone)]
pub struct SimilarEvent {
    /// The matching event envelope.
    pub envelope: ExperienceEventEnvelope,
    /// Cosine distance from the query vector (lower = more similar).
    pub distance: f32,
}

const TABLE_NAME: &str = "experience_events";

/// LanceDB implementation of the experience store with vector search.
pub struct LanceDbExperienceStore {
    db: lancedb::Connection,
    config: LanceDbConfig,
    runtime: Runtime,
}

impl LanceDbExperienceStore {
    /// Connect to LanceDB and initialize the events table.
    pub fn connect(config: LanceDbConfig) -> ExperienceStoreResult<Self> {
        let runtime = Runtime::new().map_err(|err| ExperienceStoreError::StorageError {
            message: format!("Failed to create runtime: {err}"),
        })?;

        let db = runtime
            .block_on(connect(&config.uri).execute())
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to connect to LanceDB: {err}"),
            })?;

        let store = Self {
            db,
            config,
            runtime,
        };
        store.runtime.block_on(store.ensure_table())?;
        Ok(store)
    }

    /// Create the events table if it doesn't exist.
    async fn ensure_table(&self) -> ExperienceStoreResult<()> {
        let tables = self.db.table_names().execute().await.map_err(|err| {
            ExperienceStoreError::StorageError {
                message: format!("Failed to list tables: {err}"),
            }
        })?;

        if !tables.contains(&TABLE_NAME.to_string()) {
            let schema = Arc::new(self.table_schema());
            self.db
                .create_empty_table(TABLE_NAME, schema)
                .execute()
                .await
                .map_err(|err| ExperienceStoreError::StorageError {
                    message: format!("Failed to create table: {err}"),
                })?;
        }
        Ok(())
    }

    fn table_schema(&self) -> Schema {
        Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("occurred_at", DataType::Utf8, false),
            Field::new("tenant_id", DataType::Utf8, true),
            Field::new("correlation_id", DataType::Utf8, true),
            Field::new("kind", DataType::Utf8, false),
            Field::new("event_json", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.config.embedding_dim as i32,
                ),
                false,
            ),
        ])
    }

    fn make_vector_array(&self, vector: &[f32]) -> ExperienceStoreResult<FixedSizeListArray> {
        let values = Float32Array::from(vector.to_vec());
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        FixedSizeListArray::try_new(
            field,
            self.config.embedding_dim as i32,
            Arc::new(values),
            None,
        )
        .map_err(|err| ExperienceStoreError::StorageError {
            message: format!("Failed to create vector array: {err}"),
        })
    }

    /// Append a vector-annotated event.
    pub fn append_vector_event(&self, event: VectorEvent) -> ExperienceStoreResult<()> {
        converge_experience::validate_envelope(&event.envelope)?;

        if event.vector.len() != self.config.embedding_dim {
            return Err(ExperienceStoreError::InvalidQuery {
                message: format!(
                    "vector dimension mismatch: expected {}, got {}",
                    self.config.embedding_dim,
                    event.vector.len()
                ),
            });
        }

        // Validate vector values are finite (no NaN/Inf)
        if event.vector.iter().any(|v| !v.is_finite()) {
            return Err(ExperienceStoreError::InvalidQuery {
                message: "vector contains NaN or infinite values".to_string(),
            });
        }

        self.runtime.block_on(self.insert_vector_event(&event))
    }

    async fn insert_vector_event(&self, event: &VectorEvent) -> ExperienceStoreResult<()> {
        let table = self
            .db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to open table: {err}"),
            })?;

        let event_json = serde_json::to_string(&event.envelope.event).map_err(|err| {
            ExperienceStoreError::StorageError {
                message: format!("Failed to serialize event: {err}"),
            }
        })?;

        let kind_str = format!("{:?}", event.envelope.event.kind());
        let vector_array = self.make_vector_array(&event.vector)?;

        let schema = Arc::new(self.table_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![event.envelope.event_id.as_str()])),
                Arc::new(StringArray::from(vec![event.envelope.occurred_at.as_str()])),
                Arc::new(StringArray::from(vec![event.envelope.tenant_id.as_deref()])),
                Arc::new(StringArray::from(vec![
                    event.envelope.correlation_id.as_deref(),
                ])),
                Arc::new(StringArray::from(vec![kind_str.as_str()])),
                Arc::new(StringArray::from(vec![event_json.as_str()])),
                Arc::new(vector_array),
            ],
        )
        .map_err(|err| ExperienceStoreError::StorageError {
            message: format!("Failed to create record batch: {err}"),
        })?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table
            .add(batches)
            .execute()
            .await
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to insert event: {err}"),
            })?;

        Ok(())
    }

    /// Search for events similar to the given query vector.
    ///
    /// Returns events ordered by cosine distance (most similar first).
    pub fn search_similar(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> ExperienceStoreResult<Vec<SimilarEvent>> {
        if query_vector.len() != self.config.embedding_dim {
            return Err(ExperienceStoreError::InvalidQuery {
                message: format!(
                    "query vector dimension mismatch: expected {}, got {}",
                    self.config.embedding_dim,
                    query_vector.len()
                ),
            });
        }

        if query_vector.iter().any(|v| !v.is_finite()) {
            return Err(ExperienceStoreError::InvalidQuery {
                message: "query vector contains NaN or infinite values".to_string(),
            });
        }

        self.runtime
            .block_on(self.search_similar_async(query_vector, limit))
    }

    async fn search_similar_async(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> ExperienceStoreResult<Vec<SimilarEvent>> {
        use tokio_stream::StreamExt;

        let table = self
            .db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to open table: {err}"),
            })?;

        let mut stream = table
            .vector_search(query_vector)
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to build vector search: {err}"),
            })?
            .limit(limit)
            .execute()
            .await
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to execute vector search: {err}"),
            })?;

        let mut events = Vec::new();
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to read result batch: {err}"),
            })?;

            let event_ids: &StringArray = batch
                .column_by_name("event_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| ExperienceStoreError::StorageError {
                    message: "Missing event_id column".to_string(),
                })?;

            let occurred_ats: &StringArray = batch
                .column_by_name("occurred_at")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| ExperienceStoreError::StorageError {
                    message: "Missing occurred_at column".to_string(),
                })?;

            let tenant_ids: Option<&StringArray> = batch
                .column_by_name("tenant_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let correlation_ids: Option<&StringArray> = batch
                .column_by_name("correlation_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let event_jsons: &StringArray = batch
                .column_by_name("event_json")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| ExperienceStoreError::StorageError {
                    message: "Missing event_json column".to_string(),
                })?;

            let distances: Option<&Float32Array> = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for row in 0..batch.num_rows() {
                let event: ExperienceEvent =
                    serde_json::from_str(event_jsons.value(row)).map_err(|err| {
                        ExperienceStoreError::StorageError {
                            message: format!("Failed to deserialize event: {err}"),
                        }
                    })?;

                let tenant_id = tenant_ids.and_then(|a| {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row).to_string())
                    }
                });
                let correlation_id = correlation_ids.and_then(|a| {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row).to_string())
                    }
                });

                let envelope = ExperienceEventEnvelope {
                    event_id: EventId::new(event_ids.value(row).to_string()),
                    occurred_at: occurred_ats.value(row).to_string().into(),
                    tenant_id: tenant_id.map(TenantId::new),
                    correlation_id: correlation_id.map(CorrelationId::new),
                    event,
                };

                let distance = distances.map_or(0.0, |d| d.value(row));

                events.push(SimilarEvent { envelope, distance });
            }
        }

        Ok(events)
    }
}

/// ExperienceStore implementation (standard append/query without vectors).
///
/// For vector operations, use [`LanceDbExperienceStore::append_vector_event`]
/// and [`LanceDbExperienceStore::search_similar`] directly.
impl ExperienceStore for LanceDbExperienceStore {
    fn append_event(&self, event: ExperienceEventEnvelope) -> ExperienceStoreResult<()> {
        // Store with a zero vector when no embedding is provided.
        let zero_vector = vec![0.0f32; self.config.embedding_dim];
        self.append_vector_event(VectorEvent {
            envelope: event,
            vector: zero_vector,
        })
    }

    fn query_events(
        &self,
        _query: &EventQuery,
    ) -> ExperienceStoreResult<Vec<ExperienceEventEnvelope>> {
        // LanceDB is optimized for vector search, not filtered queries.
        Err(ExperienceStoreError::InvalidQuery {
            message: "LanceDB store supports vector search via search_similar(), not filtered queries. Use SurrealDB or InMemory for EventQuery.".to_string(),
        })
    }

    fn write_artifact_state_transition(
        &self,
        artifact_id: &ArtifactId,
        artifact_kind: ArtifactKind,
        event: LifecycleEvent,
    ) -> ExperienceStoreResult<()> {
        let payload = ExperienceEvent::ArtifactStateTransitioned {
            artifact_id: artifact_id.clone(),
            artifact_kind,
            event,
        };
        let envelope = ExperienceEventEnvelope::new(
            format!(
                "evt-{:x}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ),
            payload,
        );
        self.append_event(envelope)
    }

    fn get_trace_link(
        &self,
        _trace_link_id: &TraceLinkId,
    ) -> ExperienceStoreResult<Option<ReplayTrace>> {
        // Trace links are not indexed in LanceDB.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_core::DecisionStep;

    fn test_config() -> LanceDbConfig {
        let dir = tempfile::tempdir().expect("temp dir");
        // Leak the tempdir so it persists for the test duration.
        let path = dir.path().to_str().expect("path").to_string();
        std::mem::forget(dir);
        LanceDbConfig::new(path, 4)
    }

    fn make_event(id: &str) -> ExperienceEventEnvelope {
        ExperienceEventEnvelope::new(
            id,
            ExperienceEvent::OutcomeRecorded {
                chain_id: "chain-1".into(),
                step: DecisionStep::Planning,
                passed: true,
                stop_reason: None,
                latency_ms: None,
                tokens: None,
                cost_microdollars: None,
                backend: None,
                metadata: Default::default(),
            },
        )
    }

    #[test]
    fn connect_and_append() {
        let config = test_config();
        let store = LanceDbExperienceStore::connect(config).expect("connect");
        let event = VectorEvent {
            envelope: make_event("evt-1"),
            vector: vec![1.0, 0.0, 0.0, 0.0],
        };
        store.append_vector_event(event).expect("append");
    }

    #[test]
    fn rejects_wrong_dimension() {
        let config = test_config();
        let store = LanceDbExperienceStore::connect(config).expect("connect");
        let event = VectorEvent {
            envelope: make_event("evt-1"),
            vector: vec![1.0, 0.0], // dim=2, expected 4
        };
        let err = store.append_vector_event(event).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn rejects_nan_vector() {
        let config = test_config();
        let store = LanceDbExperienceStore::connect(config).expect("connect");
        let event = VectorEvent {
            envelope: make_event("evt-1"),
            vector: vec![1.0, f32::NAN, 0.0, 0.0],
        };
        let err = store.append_vector_event(event).unwrap_err();
        assert!(err.to_string().contains("NaN or infinite"));
    }

    #[test]
    fn rejects_infinite_vector() {
        let config = test_config();
        let store = LanceDbExperienceStore::connect(config).expect("connect");
        let event = VectorEvent {
            envelope: make_event("evt-1"),
            vector: vec![1.0, f32::INFINITY, 0.0, 0.0],
        };
        let err = store.append_vector_event(event).unwrap_err();
        assert!(err.to_string().contains("NaN or infinite"));
    }

    #[test]
    fn search_returns_similar_events() {
        let config = test_config();
        let store = LanceDbExperienceStore::connect(config).expect("connect");

        // Insert 3 events with known vectors
        for (id, vec) in [
            ("evt-north", vec![1.0, 0.0, 0.0, 0.0]),
            ("evt-east", vec![0.0, 1.0, 0.0, 0.0]),
            ("evt-south", vec![-1.0, 0.0, 0.0, 0.0]),
        ] {
            store
                .append_vector_event(VectorEvent {
                    envelope: make_event(id),
                    vector: vec,
                })
                .expect("append");
        }

        // Search near "north" — should return evt-north first
        let results = store
            .search_similar(&[0.9, 0.1, 0.0, 0.0], 2)
            .expect("search");
        assert!(!results.is_empty());
        assert_eq!(results[0].envelope.event_id, "evt-north");
    }
}
