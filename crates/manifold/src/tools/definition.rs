// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Core tool definition types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Describes a tool's interface and capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for input parameters.
    pub input_schema: InputSchema,
    /// Where this tool came from.
    #[serde(default)]
    pub source: ToolSource,
    /// Optional annotations/metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub annotations: HashMap<String, String>,
}

impl ToolDefinition {
    /// Creates a new tool definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: InputSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            source: ToolSource::Inline,
            annotations: HashMap::new(),
        }
    }

    /// Sets the tool source.
    #[must_use]
    pub fn with_source(mut self, source: ToolSource) -> Self {
        self.source = source;
        self
    }

    /// Adds an annotation.
    #[must_use]
    pub fn with_annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }

    /// Returns true if this is an MCP tool.
    #[must_use]
    pub fn is_mcp(&self) -> bool {
        matches!(self.source, ToolSource::Mcp { .. })
    }

    /// Returns true if this is an `OpenAPI` tool.
    #[must_use]
    pub fn is_openapi(&self) -> bool {
        matches!(self.source, ToolSource::OpenApi { .. })
    }

    /// Returns true if this is a GraphQL tool.
    #[must_use]
    pub fn is_graphql(&self) -> bool {
        matches!(self.source, ToolSource::GraphQl { .. })
    }
}

/// Where a tool definition originated from.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSource {
    /// Tool from an MCP server.
    Mcp {
        server_name: String,
        server_uri: String,
    },
    /// Tool from an `OpenAPI` specification.
    OpenApi {
        spec_path: String,
        operation_id: String,
        method: String,
        path: String,
    },
    /// Tool from a GraphQL schema.
    GraphQl {
        endpoint: String,
        operation_name: String,
        operation_type: GraphQlOperationType,
    },
    /// Tool defined inline in code.
    #[default]
    Inline,
}

/// Type of GraphQL operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GraphQlOperationType {
    #[default]
    Query,
    Mutation,
    Subscription,
}

/// JSON Schema for tool input parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InputSchema {
    #[serde(flatten)]
    pub schema: serde_json::Value,
}

impl InputSchema {
    #[must_use]
    pub fn from_json_schema(schema: serde_json::Value) -> Self {
        Self { schema }
    }

    #[must_use]
    pub fn empty() -> Self {
        Self {
            schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    #[must_use]
    pub fn required_properties(&self) -> Vec<String> {
        self.schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn properties(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.schema.get("properties").and_then(|v| v.as_object())
    }
}

/// A request to invoke a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    #[must_use]
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            call_id: generate_call_id(),
            tool_name: tool_name.into(),
            arguments,
        }
    }
}

/// The result of a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub content: ToolResultContent,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl ToolResult {
    #[must_use]
    pub fn text(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            content: ToolResultContent::Text(content.into()),
            is_error: false,
            metadata: HashMap::new(),
        }
    }

    #[must_use]
    pub fn error(call_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            content: ToolResultContent::Text(message.into()),
            is_error: true,
            metadata: HashMap::new(),
        }
    }

    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match &self.content {
            ToolResultContent::Text(s) => Some(s),
            ToolResultContent::Json(v) => v.as_str(),
        }
    }
}

/// Content types for tool results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Json(serde_json::Value),
}

fn generate_call_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("call-{timestamp:x}-{count:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_definition_creation() {
        let tool = ToolDefinition::new("test_tool", "A test tool", InputSchema::empty());
        assert_eq!(tool.name, "test_tool");
        assert!(matches!(tool.source, ToolSource::Inline));
    }

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::new("test_tool", json!({"input": "hello"}));
        assert_eq!(call.tool_name, "test_tool");
        assert!(call.call_id.starts_with("call-"));
    }

    #[test]
    fn test_tool_result() {
        let result = ToolResult::text("call-123", "Success");
        assert!(!result.is_error);
        assert_eq!(result.as_text(), Some("Success"));
    }

    // ========================================================================
    // ToolDefinition builder and source detection
    // ========================================================================

    #[test]
    fn tool_definition_with_source_and_annotations() {
        let tool = ToolDefinition::new("get_user", "Fetch a user", InputSchema::empty())
            .with_source(ToolSource::Mcp {
                server_name: "user-svc".into(),
                server_uri: "stdio://user-svc".into(),
            })
            .with_annotation("category", "data")
            .with_annotation("auth", "required");

        assert!(tool.is_mcp());
        assert!(!tool.is_openapi());
        assert!(!tool.is_graphql());
        assert_eq!(tool.annotations.len(), 2);
        assert_eq!(tool.annotations["category"], "data");
    }

    #[test]
    fn tool_source_openapi_detection() {
        let tool = ToolDefinition::new("list_items", "List items", InputSchema::empty())
            .with_source(ToolSource::OpenApi {
                spec_path: "openapi.yaml".into(),
                operation_id: "listItems".into(),
                method: "GET".into(),
                path: "/items".into(),
            });

        assert!(tool.is_openapi());
        assert!(!tool.is_mcp());
        assert!(!tool.is_graphql());
    }

    #[test]
    fn tool_source_graphql_detection() {
        let tool = ToolDefinition::new("query_users", "Query users", InputSchema::empty())
            .with_source(ToolSource::GraphQl {
                endpoint: "https://api.example.com/graphql".into(),
                operation_name: "GetUsers".into(),
                operation_type: GraphQlOperationType::Query,
            });

        assert!(tool.is_graphql());
        assert!(!tool.is_mcp());
        assert!(!tool.is_openapi());
    }

    #[test]
    fn inline_source_is_default() {
        let tool = ToolDefinition::new("test", "test", InputSchema::empty());
        assert!(matches!(tool.source, ToolSource::Inline));
        assert!(!tool.is_mcp());
        assert!(!tool.is_openapi());
        assert!(!tool.is_graphql());
    }

    // ========================================================================
    // InputSchema
    // ========================================================================

    #[test]
    fn input_schema_empty_has_no_required() {
        let schema = InputSchema::empty();
        assert!(schema.required_properties().is_empty());
        assert!(schema.properties().unwrap().is_empty());
    }

    #[test]
    fn input_schema_from_json_extracts_required() {
        let schema = InputSchema::from_json_schema(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }));
        assert_eq!(schema.required_properties(), vec!["name"]);
        assert_eq!(schema.properties().unwrap().len(), 2);
    }

    #[test]
    fn input_schema_no_required_field_returns_empty() {
        let schema = InputSchema::from_json_schema(json!({
            "type": "object",
            "properties": {"x": {"type": "string"}}
        }));
        assert!(schema.required_properties().is_empty());
    }

    // ========================================================================
    // ToolResult
    // ========================================================================

    #[test]
    fn tool_result_error_flag() {
        let result = ToolResult::error("call-1", "something failed");
        assert!(result.is_error);
        assert_eq!(result.as_text(), Some("something failed"));
    }

    #[test]
    fn tool_result_json_content() {
        let result = ToolResult {
            call_id: "call-1".into(),
            content: ToolResultContent::Json(json!({"status": "ok"})),
            is_error: false,
            metadata: HashMap::new(),
        };
        // as_text returns None for json objects (only works for json strings)
        assert!(result.as_text().is_none());
    }

    #[test]
    fn tool_call_ids_are_unique() {
        let c1 = ToolCall::new("tool", json!({}));
        let c2 = ToolCall::new("tool", json!({}));
        assert_ne!(c1.call_id, c2.call_id);
    }

    // ========================================================================
    // Serde round-trips
    // ========================================================================

    #[test]
    fn tool_definition_serde_roundtrip() {
        let tool = ToolDefinition::new("my_tool", "A tool", InputSchema::empty())
            .with_annotation("version", "1.0");

        let json_str = serde_json::to_string(&tool).unwrap();
        let round: ToolDefinition = serde_json::from_str(&json_str).unwrap();
        assert_eq!(round.name, "my_tool");
        assert_eq!(round.annotations["version"], "1.0");
    }

    #[test]
    fn tool_source_serde_roundtrip() {
        let sources = vec![
            ToolSource::Inline,
            ToolSource::Mcp {
                server_name: "s".into(),
                server_uri: "u".into(),
            },
            ToolSource::OpenApi {
                spec_path: "p".into(),
                operation_id: "o".into(),
                method: "GET".into(),
                path: "/x".into(),
            },
            ToolSource::GraphQl {
                endpoint: "e".into(),
                operation_name: "Q".into(),
                operation_type: GraphQlOperationType::Mutation,
            },
        ];

        for source in &sources {
            let json_str = serde_json::to_string(source).unwrap();
            let round: ToolSource = serde_json::from_str(&json_str).unwrap();
            assert_eq!(format!("{round:?}"), format!("{source:?}"));
        }
    }
}
