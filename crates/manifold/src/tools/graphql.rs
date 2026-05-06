// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! GraphQL to `ToolDefinition` converter.

use super::{GraphQlOperationType, InputSchema, ToolDefinition, ToolError, ToolSource};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Converts GraphQL schemas to tool definitions.
#[derive(Debug)]
pub struct GraphQlConverter {
    endpoint: String,
    include_queries: bool,
    include_mutations: bool,
    include_subscriptions: bool,
    name_prefix: Option<String>,
    field_filter: Option<Vec<String>>,
}

impl GraphQlConverter {
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            include_queries: true,
            include_mutations: true,
            include_subscriptions: false,
            name_prefix: None,
            field_filter: None,
        }
    }

    #[must_use]
    pub fn include_queries(mut self, include: bool) -> Self {
        self.include_queries = include;
        self
    }

    #[must_use]
    pub fn include_mutations(mut self, include: bool) -> Self {
        self.include_mutations = include;
        self
    }

    #[must_use]
    pub fn include_subscriptions(mut self, include: bool) -> Self {
        self.include_subscriptions = include;
        self
    }

    #[must_use]
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = Some(prefix.into());
        self
    }

    #[must_use]
    pub fn with_field_filter(mut self, fields: Vec<String>) -> Self {
        self.field_filter = Some(fields);
        self
    }

    pub fn from_introspection(&self, json: &str) -> Result<Vec<ToolDefinition>, ToolError> {
        let introspection: GraphQlIntrospectionResult = serde_json::from_str(json)
            .map_err(|e| ToolError::validation_failed(format!("Invalid introspection: {e}")))?;
        self.convert_schema(&introspection)
    }

    fn convert_schema(
        &self,
        introspection: &GraphQlIntrospectionResult,
    ) -> Result<Vec<ToolDefinition>, ToolError> {
        let schema = introspection
            .data
            .as_ref()
            .and_then(|d| d.__schema.as_ref())
            .ok_or_else(|| ToolError::validation_failed("Missing __schema"))?;

        let type_map: HashMap<String, &GraphQlType> = schema
            .types
            .iter()
            .filter_map(|t| t.name.as_ref().map(|name| (name.clone(), t)))
            .collect();

        let mut tools = Vec::new();

        if self.include_queries
            && let Some(query_type) = schema
                .query_type
                .as_ref()
                .and_then(|qt| qt.name.as_ref())
                .and_then(|name| type_map.get(name))
        {
            tools.extend(self.convert_type_fields(
                query_type,
                GraphQlOperationType::Query,
                &type_map,
            ));
        }

        if self.include_mutations
            && let Some(mutation_type) = schema
                .mutation_type
                .as_ref()
                .and_then(|mt| mt.name.as_ref())
                .and_then(|name| type_map.get(name))
        {
            tools.extend(self.convert_type_fields(
                mutation_type,
                GraphQlOperationType::Mutation,
                &type_map,
            ));
        }

        Ok(tools)
    }

    fn convert_type_fields(
        &self,
        gql_type: &GraphQlType,
        operation_type: GraphQlOperationType,
        type_map: &HashMap<String, &GraphQlType>,
    ) -> Vec<ToolDefinition> {
        let Some(fields) = &gql_type.fields else {
            return Vec::new();
        };

        fields
            .iter()
            .filter(|field| {
                if let Some(ref filter) = self.field_filter {
                    field
                        .name
                        .as_ref()
                        .map(|n| filter.iter().any(|f| n.contains(f)))
                        .unwrap_or(false)
                } else {
                    true
                }
            })
            .filter_map(|field| {
                let name = field.name.as_ref()?;
                if name.starts_with("__") {
                    return None;
                }

                let tool_name = if let Some(ref prefix) = self.name_prefix {
                    format!("{prefix}_{name}")
                } else {
                    name.clone()
                };

                let description = field
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("{operation_type:?} {name}"));

                let input_schema = self.build_input_schema(field, type_map);

                Some(
                    ToolDefinition::new(tool_name, description, input_schema).with_source(
                        ToolSource::GraphQl {
                            endpoint: self.endpoint.clone(),
                            operation_name: name.clone(),
                            operation_type,
                        },
                    ),
                )
            })
            .collect()
    }

    fn build_input_schema(
        &self,
        field: &GraphQlField,
        type_map: &HashMap<String, &GraphQlType>,
    ) -> InputSchema {
        let args = match &field.args {
            Some(a) if !a.is_empty() => a,
            _ => return InputSchema::empty(),
        };

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for arg in args {
            if let Some(name) = &arg.name {
                let (prop, is_required) = self.input_value_to_property(arg, type_map);
                properties.insert(name.clone(), prop);
                if is_required {
                    required.push(name.clone());
                }
            }
        }

        InputSchema::from_json_schema(serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        }))
    }

    fn input_value_to_property(
        &self,
        input: &GraphQlInputValue,
        type_map: &HashMap<String, &GraphQlType>,
    ) -> (serde_json::Value, bool) {
        if let Some(ref type_ref) = input.type_ref {
            self.type_ref_to_schema(type_ref, type_map)
        } else {
            (serde_json::json!({"type": "string"}), false)
        }
    }

    #[allow(clippy::only_used_in_recursion, clippy::self_only_used_in_recursion)]
    fn type_ref_to_schema(
        &self,
        type_ref: &GraphQlTypeRef,
        type_map: &HashMap<String, &GraphQlType>,
    ) -> (serde_json::Value, bool) {
        match type_ref.kind.as_deref() {
            Some("NON_NULL") => {
                if let Some(ref inner) = type_ref.of_type {
                    let (schema, _) = self.type_ref_to_schema(inner, type_map);
                    (schema, true)
                } else {
                    (serde_json::json!({"type": "string"}), true)
                }
            }
            Some("LIST") => {
                if let Some(ref inner) = type_ref.of_type {
                    let (items_schema, _) = self.type_ref_to_schema(inner, type_map);
                    (
                        serde_json::json!({"type": "array", "items": items_schema}),
                        false,
                    )
                } else {
                    (serde_json::json!({"type": "array"}), false)
                }
            }
            Some("SCALAR") => {
                let json_type = match type_ref.name.as_deref() {
                    Some("Int") => "integer",
                    Some("Float") => "number",
                    Some("Boolean") => "boolean",
                    _ => "string",
                };
                (serde_json::json!({"type": json_type}), false)
            }
            _ => (serde_json::json!({"type": "string"}), false),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphQlIntrospectionResult {
    pub data: Option<GraphQlIntrospectionData>,
}

#[derive(Debug, Deserialize, Serialize)]
#[allow(clippy::pub_underscore_fields)]
pub struct GraphQlIntrospectionData {
    #[serde(rename = "__schema")]
    pub __schema: Option<GraphQlSchema>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphQlSchema {
    #[serde(rename = "queryType")]
    pub query_type: Option<GraphQlTypeRef>,
    #[serde(rename = "mutationType")]
    pub mutation_type: Option<GraphQlTypeRef>,
    #[serde(default)]
    pub types: Vec<GraphQlType>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphQlType {
    pub kind: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub fields: Option<Vec<GraphQlField>>,
    #[serde(rename = "inputFields")]
    pub input_fields: Option<Vec<GraphQlInputValue>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphQlField {
    pub name: Option<String>,
    pub description: Option<String>,
    pub args: Option<Vec<GraphQlInputValue>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphQlInputValue {
    pub name: Option<String>,
    pub description: Option<String>,
    #[serde(rename = "type")]
    pub type_ref: Option<GraphQlTypeRef>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphQlTypeRef {
    pub kind: Option<String>,
    pub name: Option<String>,
    #[serde(rename = "ofType")]
    pub of_type: Option<Box<GraphQlTypeRef>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_introspection() {
        let json = r#"{
            "data": {
                "__schema": {
                    "queryType": { "name": "Query" },
                    "mutationType": null,
                    "types": [{
                        "kind": "OBJECT",
                        "name": "Query",
                        "fields": [{
                            "name": "user",
                            "description": "Get user",
                            "args": [{"name": "id", "type": {"kind": "SCALAR", "name": "ID"}}]
                        }]
                    }]
                }
            }
        }"#;

        let converter = GraphQlConverter::new("https://api.example.com/graphql");
        let tools = converter.from_introspection(json).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "user");
    }
}
