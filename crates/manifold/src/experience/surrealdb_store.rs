// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! SurrealDB-backed experience store implementation.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::remote::ws::{Client, Ws};
use surrealdb::opt::auth::Root;
use tokio::runtime::Runtime;

use converge_core::{
    ArtifactId, ArtifactKind, CorrelationId, EventId, EventQuery, ExperienceEvent,
    ExperienceEventEnvelope, ExperienceEventKind, ExperienceStore, ExperienceStoreError,
    ExperienceStoreResult, LifecycleEvent, ReplayTrace, TenantId, TraceLinkId,
};

/// Configuration for SurrealDB connection.
#[derive(Debug, Clone)]
pub struct SurrealDbConfig {
    pub url: String,
    pub namespace: String,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl SurrealDbConfig {
    /// Create a new config with required connection fields.
    #[must_use]
    pub fn new(
        url: impl Into<String>,
        namespace: impl Into<String>,
        database: impl Into<String>,
    ) -> Self {
        Self {
            url: url.into(),
            namespace: namespace.into(),
            database: database.into(),
            username: None,
            password: None,
        }
    }

    /// Add root credentials (optional).
    #[must_use]
    pub fn with_root_auth(
        mut self,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }
}

/// SurrealDB implementation of the experience store.
pub struct SurrealDbExperienceStore {
    db: Surreal<Client>,
    runtime: Runtime,
}

impl SurrealDbExperienceStore {
    /// Connect to SurrealDB and initialize schema.
    pub fn connect(config: SurrealDbConfig) -> ExperienceStoreResult<Self> {
        let runtime = Runtime::new().map_err(|err| ExperienceStoreError::StorageError {
            message: format!("Failed to create runtime: {err}"),
        })?;

        let db = runtime
            .block_on(async { Surreal::new::<Ws>(&config.url).await })
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to connect to SurrealDB: {err}"),
            })?;

        let store = Self { db, runtime };
        store.runtime.block_on(async {
            if let (Some(username), Some(password)) = (config.username, config.password) {
                store
                    .db
                    .signin(Root { username, password })
                    .await
                    .map_err(|err| ExperienceStoreError::StorageError {
                        message: format!("Failed to sign in: {err}"),
                    })?;
            }

            store
                .db
                .use_ns(&config.namespace)
                .use_db(&config.database)
                .await
                .map_err(|err| ExperienceStoreError::StorageError {
                    message: format!("Failed to select namespace/database: {err}"),
                })?;

            store.init_schema().await
        })?;

        Ok(store)
    }

    async fn init_schema(&self) -> ExperienceStoreResult<()> {
        let schema = r"
DEFINE TABLE event SCHEMALESS;
DEFINE INDEX event_tenant_ts ON TABLE event COLUMNS tenant_id, occurred_at;
DEFINE INDEX event_kind_tenant ON TABLE event COLUMNS kind, tenant_id;
DEFINE INDEX event_correlation ON TABLE event COLUMNS correlation_id;
DEFINE INDEX event_chain ON TABLE event COLUMNS chain_id;
DEFINE INDEX event_trace_link ON TABLE event COLUMNS trace_link_id;
DEFINE INDEX event_artifact ON TABLE event COLUMNS artifact_id, state;
DEFINE TABLE trace_link SCHEMALESS;
DEFINE INDEX trace_link_id_idx ON TABLE trace_link COLUMNS trace_link_id;
";

        self.db
            .query(schema)
            .await
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to initialize schema: {err}"),
            })?;
        Ok(())
    }

    async fn insert_event(&self, event: &ExperienceEventEnvelope) -> ExperienceStoreResult<()> {
        let record = EventRecord::from_envelope(event)?;
        let record = to_json_value("event record", record)?;
        let _: Option<serde_json::Value> = self
            .db
            .create(("event", event.event_id.as_str()))
            .content(record)
            .await
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to insert event: {err}"),
            })?;

        if let ExperienceEvent::ReplayTraceRecorded {
            trace_link_id,
            trace_link,
        } = &event.event
        {
            let record = ReplayTraceRecord::from_trace_link(trace_link_id, trace_link)?;
            let record = to_json_value("trace link record", record)?;
            let _: Option<serde_json::Value> = self
                .db
                .create(("trace_link", trace_link_id.as_str()))
                .content(record)
                .await
                .map_err(|err| ExperienceStoreError::StorageError {
                    message: format!("Failed to insert trace link: {err}"),
                })?;
        }

        Ok(())
    }

    async fn select_trace_link(
        &self,
        trace_link_id: &str,
    ) -> ExperienceStoreResult<Option<ReplayTrace>> {
        let mut response = self
            .db
            .query("SELECT trace_link FROM trace_link WHERE trace_link_id = $id LIMIT 1")
            .bind(("id", trace_link_id.to_string()))
            .await
            .map_err(|err| ExperienceStoreError::StorageError {
                message: format!("Failed to query trace link: {err}"),
            })?;

        let record: Option<serde_json::Value> =
            response
                .take(0)
                .map_err(|err| ExperienceStoreError::StorageError {
                    message: format!("Failed to decode trace link: {err}"),
                })?;
        record
            .map(|row| {
                let row: ReplayTraceRecord = from_json_value("trace link record", row)?;
                from_json_value("trace link", row.trace_link)
            })
            .transpose()
    }

    async fn select_events(
        &self,
        query: &EventQuery,
    ) -> ExperienceStoreResult<Vec<ExperienceEventEnvelope>> {
        let mut filters = Vec::new();
        let mut bindings: Vec<(&str, serde_json::Value)> = Vec::new();

        if let Some(ref tenant_id) = query.tenant_id {
            filters.push("tenant_id = $tenant_id");
            bindings.push((
                "tenant_id",
                serde_json::Value::String(tenant_id.to_string()),
            ));
        }

        if let Some(ref correlation_id) = query.correlation_id {
            filters.push("correlation_id = $correlation_id");
            bindings.push((
                "correlation_id",
                serde_json::Value::String(correlation_id.to_string()),
            ));
        }

        if let Some(ref chain_id) = query.chain_id {
            filters.push("chain_id = $chain_id");
            bindings.push(("chain_id", serde_json::Value::String(chain_id.to_string())));
        }

        if let Some(ref range) = query.time_range {
            if let Some(ref start) = range.start {
                filters.push("occurred_at >= $start");
                bindings.push(("start", serde_json::Value::String(start.to_string())));
            }
            if let Some(ref end) = range.end {
                filters.push("occurred_at <= $end");
                bindings.push(("end", serde_json::Value::String(end.to_string())));
            }
        }

        if !query.kinds.is_empty() {
            let kinds = query
                .kinds
                .iter()
                .map(|kind| serde_json::Value::String(kind_to_str(*kind).to_string()))
                .collect::<Vec<_>>();
            filters.push("kind INSIDE $kinds");
            bindings.push(("kinds", serde_json::Value::Array(kinds)));
        }

        let mut sql = String::from(
            "SELECT event_id, occurred_at, tenant_id, correlation_id, kind, chain_id, event, \
            trace_link_id, artifact_id, state FROM event",
        );
        if !filters.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&filters.join(" AND "));
        }
        sql.push_str(" ORDER BY occurred_at ASC");
        if query.limit.is_some() {
            sql.push_str(" LIMIT $limit");
        }

        let mut query_builder = self.db.query(sql);
        for (key, value) in bindings {
            query_builder = query_builder.bind((key, value));
        }
        if let Some(limit) = query.limit {
            query_builder = query_builder.bind(("limit", limit as i64));
        }

        let mut response =
            query_builder
                .await
                .map_err(|err| ExperienceStoreError::StorageError {
                    message: format!("Failed to query events: {err}"),
                })?;

        let records: Vec<serde_json::Value> =
            response
                .take(0)
                .map_err(|err| ExperienceStoreError::StorageError {
                    message: format!("Failed to decode events: {err}"),
                })?;

        records
            .into_iter()
            .map(|record| {
                let record: EventRecord = from_json_value("event record", record)?;
                record.into_envelope()
            })
            .collect()
    }
}

impl ExperienceStore for SurrealDbExperienceStore {
    fn append_event(&self, event: ExperienceEventEnvelope) -> ExperienceStoreResult<()> {
        converge_experience::validate_envelope(&event)?;
        self.runtime.block_on(self.insert_event(&event))
    }

    fn query_events(
        &self,
        query: &EventQuery,
    ) -> ExperienceStoreResult<Vec<ExperienceEventEnvelope>> {
        self.runtime.block_on(self.select_events(query))
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
        let envelope = ExperienceEventEnvelope::new(format!("evt-{}", uuid_stub()), payload);
        self.append_event(envelope)
    }

    fn get_trace_link(
        &self,
        trace_link_id: &TraceLinkId,
    ) -> ExperienceStoreResult<Option<ReplayTrace>> {
        self.runtime
            .block_on(self.select_trace_link(trace_link_id.as_str()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EventRecord {
    event_id: String,
    occurred_at: String,
    tenant_id: Option<String>,
    correlation_id: Option<String>,
    kind: String,
    chain_id: Option<String>,
    trace_link_id: Option<String>,
    artifact_id: Option<String>,
    state: Option<String>,
    event: serde_json::Value,
}

impl EventRecord {
    fn from_envelope(envelope: &ExperienceEventEnvelope) -> ExperienceStoreResult<Self> {
        let (trace_link_id, artifact_id, state) = event_index_fields(&envelope.event);
        Ok(Self {
            event_id: envelope.event_id.as_str().to_string(),
            occurred_at: envelope.occurred_at.as_str().to_string(),
            tenant_id: envelope.tenant_id.as_ref().map(ToString::to_string),
            correlation_id: envelope.correlation_id.as_ref().map(ToString::to_string),
            kind: kind_to_str(envelope.event.kind()).to_string(),
            chain_id: event_chain_id(&envelope.event).map(ToString::to_string),
            trace_link_id,
            artifact_id,
            state,
            event: to_json_value("experience event", &envelope.event)?,
        })
    }

    fn into_envelope(self) -> ExperienceStoreResult<ExperienceEventEnvelope> {
        Ok(ExperienceEventEnvelope {
            event_id: EventId::new(self.event_id),
            occurred_at: self.occurred_at.into(),
            tenant_id: self.tenant_id.map(TenantId::new),
            correlation_id: self.correlation_id.map(CorrelationId::new),
            event: from_json_value("experience event", self.event)?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplayTraceRecord {
    trace_link_id: Option<String>,
    trace_link: serde_json::Value,
}

impl ReplayTraceRecord {
    fn from_trace_link(
        trace_link_id: &TraceLinkId,
        trace_link: &ReplayTrace,
    ) -> ExperienceStoreResult<Self> {
        Ok(Self {
            trace_link_id: Some(trace_link_id.as_str().to_string()),
            trace_link: to_json_value("trace link", trace_link)?,
        })
    }
}

fn to_json_value(
    label: &'static str,
    value: impl Serialize,
) -> ExperienceStoreResult<serde_json::Value> {
    serde_json::to_value(value).map_err(|err| ExperienceStoreError::StorageError {
        message: format!("Failed to encode {label}: {err}"),
    })
}

fn from_json_value<T: DeserializeOwned>(
    label: &'static str,
    value: serde_json::Value,
) -> ExperienceStoreResult<T> {
    serde_json::from_value(value).map_err(|err| ExperienceStoreError::StorageError {
        message: format!("Failed to decode {label}: {err}"),
    })
}

fn kind_to_str(kind: ExperienceEventKind) -> &'static str {
    match kind {
        ExperienceEventKind::ProposalCreated => "proposal_created",
        ExperienceEventKind::ProposalValidated => "proposal_validated",
        ExperienceEventKind::FactPromoted => "fact_promoted",
        ExperienceEventKind::RecallExecuted => "recall_executed",
        ExperienceEventKind::ReplayTraceRecorded => "trace_link_recorded",
        ExperienceEventKind::ReplayabilityDowngraded => "replayability_downgraded",
        ExperienceEventKind::ArtifactStateTransitioned => "artifact_state_transitioned",
        ExperienceEventKind::ArtifactRollbackRecorded => "artifact_rollback_recorded",
        ExperienceEventKind::BackendInvoked => "backend_invoked",
        ExperienceEventKind::OutcomeRecorded => "outcome_recorded",
        ExperienceEventKind::BudgetExceeded => "budget_exceeded",
        ExperienceEventKind::PolicySnapshotCaptured => "policy_snapshot_captured",
        ExperienceEventKind::HypothesisResolved => "hypothesis_resolved",
        ExperienceEventKind::GateDecisionRecorded => "gate_decision_recorded",
    }
}

fn event_chain_id(event: &ExperienceEvent) -> Option<&str> {
    match event {
        ExperienceEvent::ProposalCreated { chain_id, .. } => Some(chain_id.as_str()),
        ExperienceEvent::ProposalValidated { chain_id, .. } => Some(chain_id.as_str()),
        ExperienceEvent::OutcomeRecorded { chain_id, .. } => Some(chain_id.as_str()),
        ExperienceEvent::BudgetExceeded { chain_id, .. } => Some(chain_id.as_str()),
        _ => None,
    }
}

fn event_index_fields(event: &ExperienceEvent) -> (Option<String>, Option<String>, Option<String>) {
    match event {
        ExperienceEvent::ReplayTraceRecorded { trace_link_id, .. } => {
            (Some(trace_link_id.to_string()), None, None)
        }
        ExperienceEvent::ReplayabilityDowngraded { trace_link_id, .. } => {
            (Some(trace_link_id.to_string()), None, None)
        }
        ExperienceEvent::BackendInvoked { trace_link_id, .. } => {
            (Some(trace_link_id.to_string()), None, None)
        }
        ExperienceEvent::RecallExecuted { trace_link_id, .. } => {
            (trace_link_id.as_ref().map(ToString::to_string), None, None)
        }
        ExperienceEvent::ArtifactStateTransitioned {
            artifact_id, event, ..
        } => (
            None,
            Some(artifact_id.to_string()),
            Some(event.to_state.to_string()),
        ),
        ExperienceEvent::ArtifactRollbackRecorded { rollback } => (
            None,
            Some(rollback.artifact_id.clone()),
            Some("rolled_back".to_string()),
        ),
        _ => (None, None, None),
    }
}

fn uuid_stub() -> String {
    // Minimal stub to avoid bringing in a UUID dependency in core flows.
    // Production should inject stable IDs at the caller.
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_core::BudgetResource;

    #[test]
    fn kind_to_string_is_stable() {
        assert_eq!(
            kind_to_str(ExperienceEventKind::ProposalCreated),
            "proposal_created"
        );
        assert_eq!(
            kind_to_str(ExperienceEventKind::BudgetExceeded),
            "budget_exceeded"
        );
    }

    #[test]
    fn event_record_maps_envelope() {
        let event = ExperienceEvent::BudgetExceeded {
            chain_id: "chain-1".into(),
            resource: BudgetResource::Tokens,
            limit: "10".to_string(),
            observed: None,
        };
        let envelope = ExperienceEventEnvelope::new("evt-1", event);
        let record = EventRecord::from_envelope(&envelope).expect("record");
        assert_eq!(record.event_id, "evt-1");
        assert_eq!(record.kind, "budget_exceeded");
        assert_eq!(record.chain_id.as_deref(), Some("chain-1"));
    }
}
