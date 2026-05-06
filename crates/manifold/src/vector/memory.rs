// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! In-memory vector store for testing and small workloads.

use converge_core::capability::{
    CapabilityError, VectorMatch, VectorQuery, VectorRecall, VectorRecord,
};
use std::collections::HashMap;
use std::sync::RwLock;

use super::cosine_similarity;

/// In-memory vector store.
///
/// This is a simple vector store that keeps all vectors in memory.
/// Suitable for:
/// - Testing and development
/// - Small workloads (< 100k vectors)
/// - Ephemeral vector caches
///
/// For production workloads, use a persistent vector-store adapter from
/// `manifold` or another extension.
///
/// # Thread Safety
///
/// This store is thread-safe and can be shared across threads.
pub struct InMemoryVectorStore {
    records: RwLock<HashMap<String, VectorRecord>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryVectorStore {
    /// Creates a new empty in-memory vector store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a store with pre-loaded records.
    #[must_use]
    pub fn with_records(records: Vec<VectorRecord>) -> Self {
        let store = Self::new();
        for record in records {
            let _ = store.upsert(&record);
        }
        store
    }

    /// Returns all records in the store.
    ///
    /// Useful for debugging and testing.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub fn all_records(&self) -> Vec<VectorRecord> {
        self.records
            .read()
            .expect("Lock poisoned")
            .values()
            .cloned()
            .collect()
    }
}

impl VectorRecall for InMemoryVectorStore {
    fn name(&self) -> &'static str {
        "in-memory"
    }

    fn upsert(&self, record: &VectorRecord) -> Result<(), CapabilityError> {
        let mut records = self.records.write().expect("Lock poisoned");
        records.insert(record.id.clone(), record.clone());
        Ok(())
    }

    fn query(&self, query: &VectorQuery) -> Result<Vec<VectorMatch>, CapabilityError> {
        let records = self.records.read().expect("Lock poisoned");

        // Compute similarity for all records
        let mut matches: Vec<VectorMatch> = records
            .values()
            .map(|record| {
                let score = f64::from(cosine_similarity(&query.vector, &record.vector));
                VectorMatch {
                    id: record.id.clone(),
                    score,
                    payload: record.payload.clone(),
                }
            })
            .filter(|m| {
                // Apply minimum score filter
                query.min_score.is_none_or(|min| m.score >= min)
            })
            .collect();

        // Sort by score descending
        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top_k
        matches.truncate(query.top_k);

        Ok(matches)
    }

    fn delete(&self, id: &str) -> Result<(), CapabilityError> {
        let mut records = self.records.write().expect("Lock poisoned");
        records.remove(id);
        Ok(())
    }

    fn clear(&self) -> Result<(), CapabilityError> {
        let mut records = self.records.write().expect("Lock poisoned");
        records.clear();
        Ok(())
    }

    fn count(&self) -> Result<usize, CapabilityError> {
        let records = self.records.read().expect("Lock poisoned");
        Ok(records.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn upsert_and_query() {
        let store = InMemoryVectorStore::new();

        // Insert records
        store
            .upsert(&VectorRecord {
                id: "doc-1".into(),
                vector: vec![1.0, 0.0, 0.0],
                payload: json!({"title": "Document 1"}),
            })
            .unwrap();

        store
            .upsert(&VectorRecord {
                id: "doc-2".into(),
                vector: vec![0.9, 0.1, 0.0],
                payload: json!({"title": "Document 2"}),
            })
            .unwrap();

        store
            .upsert(&VectorRecord {
                id: "doc-3".into(),
                vector: vec![0.0, 1.0, 0.0],
                payload: json!({"title": "Document 3"}),
            })
            .unwrap();

        assert_eq!(store.count().unwrap(), 3);

        // Query for similar vectors
        let matches = store
            .query(&VectorQuery::new(vec![1.0, 0.0, 0.0], 2))
            .unwrap();

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].id, "doc-1"); // Exact match
        assert_eq!(matches[1].id, "doc-2"); // Close match
    }

    #[test]
    fn query_with_min_score() {
        let store = InMemoryVectorStore::new();

        store
            .upsert(&VectorRecord {
                id: "close".into(),
                vector: vec![0.95, 0.05, 0.0],
                payload: json!({}),
            })
            .unwrap();

        store
            .upsert(&VectorRecord {
                id: "far".into(),
                vector: vec![0.0, 0.0, 1.0],
                payload: json!({}),
            })
            .unwrap();

        let matches = store
            .query(&VectorQuery::new(vec![1.0, 0.0, 0.0], 10).with_min_score(0.5))
            .unwrap();

        // Only the close match should be returned
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "close");
    }

    #[test]
    fn upsert_overwrites() {
        let store = InMemoryVectorStore::new();

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

        assert_eq!(store.count().unwrap(), 1);

        let records = store.all_records();
        assert_eq!(records[0].payload["version"], 2);
    }

    #[test]
    fn delete_and_clear() {
        let store = InMemoryVectorStore::new();

        store
            .upsert(&VectorRecord {
                id: "doc-1".into(),
                vector: vec![1.0, 0.0, 0.0],
                payload: json!({}),
            })
            .unwrap();

        store
            .upsert(&VectorRecord {
                id: "doc-2".into(),
                vector: vec![0.0, 1.0, 0.0],
                payload: json!({}),
            })
            .unwrap();

        assert_eq!(store.count().unwrap(), 2);

        store.delete("doc-1").unwrap();
        assert_eq!(store.count().unwrap(), 1);

        store.clear().unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }
}
