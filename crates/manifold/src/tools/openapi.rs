// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! `OpenAPI` to `ToolDefinition` converter.

use super::{InputSchema, ToolDefinition, ToolError, ToolSource};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Converts `OpenAPI` specifications to tool definitions.
#[derive(Debug, Default)]
pub struct OpenApiConverter {
    base_url: Option<String>,
    tag_filter: Option<Vec<String>>,
    name_prefix: Option<String>,
}

impl OpenApiConverter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    #[must_use]
    pub fn with_tag_filter(mut self, tags: Vec<String>) -> Self {
        self.tag_filter = Some(tags);
        self
    }

    #[must_use]
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = Some(prefix.into());
        self
    }

    pub fn from_yaml(&self, yaml: &str) -> Result<Vec<ToolDefinition>, ToolError> {
        let spec: OpenApiSpec = serde_yaml::from_str(yaml)
            .map_err(|e| ToolError::validation_failed(format!("Invalid OpenAPI YAML: {e}")))?;
        self.convert_spec(&spec)
    }

    pub fn from_json(&self, json: &str) -> Result<Vec<ToolDefinition>, ToolError> {
        let spec: OpenApiSpec = serde_json::from_str(json)
            .map_err(|e| ToolError::validation_failed(format!("Invalid OpenAPI JSON: {e}")))?;
        self.convert_spec(&spec)
    }

    #[allow(clippy::unnecessary_wraps)]
    fn convert_spec(&self, spec: &OpenApiSpec) -> Result<Vec<ToolDefinition>, ToolError> {
        let base_url = self
            .base_url
            .clone()
            .or_else(|| spec.servers.first().map(|s| s.url.clone()))
            .unwrap_or_default();

        let spec_path = spec
            .info
            .as_ref()
            .and_then(|i| i.title.as_ref())
            .cloned()
            .unwrap_or_else(|| "openapi".to_string());

        let mut tools = Vec::new();

        for (path, path_item) in &spec.paths {
            let methods = [
                ("get", &path_item.get),
                ("post", &path_item.post),
                ("put", &path_item.put),
                ("patch", &path_item.patch),
                ("delete", &path_item.delete),
            ];

            for (method, operation) in methods {
                if let Some(op) = operation {
                    if let Some(ref filter) = self.tag_filter {
                        let op_tags = op.tags.as_deref().unwrap_or(&[]);
                        if !filter.iter().any(|f| op_tags.contains(f)) {
                            continue;
                        }
                    }

                    let operation_id = op
                        .operation_id
                        .clone()
                        .unwrap_or_else(|| format!("{}_{}", method, path.replace('/', "_")));

                    let name = if let Some(ref prefix) = self.name_prefix {
                        format!("{prefix}_{operation_id}")
                    } else {
                        operation_id.clone()
                    };

                    let description = op
                        .summary
                        .clone()
                        .or_else(|| op.description.clone())
                        .unwrap_or_else(|| format!("{} {}", method.to_uppercase(), path));

                    let input_schema = self.build_input_schema(op, path);

                    let mut tool = ToolDefinition::new(name, description, input_schema)
                        .with_source(ToolSource::OpenApi {
                            spec_path: spec_path.clone(),
                            operation_id,
                            method: method.to_uppercase(),
                            path: path.clone(),
                        });

                    if !base_url.is_empty() {
                        tool = tool.with_annotation("base_url", &base_url);
                    }

                    tools.push(tool);
                }
            }
        }

        Ok(tools)
    }

    fn build_input_schema(&self, operation: &OpenApiOperation, path: &str) -> InputSchema {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        if let Some(params) = &operation.parameters {
            for param in params {
                let prop = param
                    .schema
                    .clone()
                    .unwrap_or(serde_json::json!({"type": "string"}));
                properties.insert(param.name.clone(), prop);
                if param.required.unwrap_or(false) {
                    required.push(param.name.clone());
                }
            }
        }

        for segment in path.split('/') {
            if segment.starts_with('{') && segment.ends_with('}') {
                let param_name = &segment[1..segment.len() - 1];
                if !properties.contains_key(param_name) {
                    properties.insert(
                        param_name.to_string(),
                        serde_json::json!({"type": "string"}),
                    );
                    required.push(param_name.to_string());
                }
            }
        }

        if let Some(body) = &operation.request_body
            && let Some(content) = &body.content
            && let Some(schema) = content
                .get("application/json")
                .and_then(|c| c.schema.as_ref())
        {
            if let Some(body_props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (key, value) in body_props {
                    properties.insert(key.clone(), value.clone());
                }
            }
            if let Some(body_required) = schema.get("required").and_then(|r| r.as_array()) {
                for req in body_required {
                    if let Some(s) = req.as_str()
                        && !required.contains(&s.to_string())
                    {
                        required.push(s.to_string());
                    }
                }
            }
        }

        InputSchema::from_json_schema(serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        }))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenApiSpec {
    #[serde(default)]
    pub openapi: String,
    #[serde(default)]
    pub info: Option<OpenApiInfo>,
    #[serde(default)]
    pub servers: Vec<OpenApiServer>,
    #[serde(default)]
    pub paths: HashMap<String, OpenApiPathItem>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenApiInfo {
    pub title: Option<String>,
    pub version: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenApiServer {
    pub url: String,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct OpenApiPathItem {
    #[serde(default)]
    pub get: Option<OpenApiOperation>,
    #[serde(default)]
    pub post: Option<OpenApiOperation>,
    #[serde(default)]
    pub put: Option<OpenApiOperation>,
    #[serde(default)]
    pub patch: Option<OpenApiOperation>,
    #[serde(default)]
    pub delete: Option<OpenApiOperation>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct OpenApiOperation {
    #[serde(rename = "operationId")]
    pub operation_id: Option<String>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub parameters: Option<Vec<OpenApiParameter>>,
    #[serde(rename = "requestBody")]
    pub request_body: Option<OpenApiRequestBody>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenApiParameter {
    pub name: String,
    #[serde(rename = "in")]
    pub location: String,
    pub required: Option<bool>,
    pub schema: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenApiRequestBody {
    pub content: Option<HashMap<String, OpenApiMediaType>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenApiMediaType {
    pub schema: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_simple_spec() {
        let yaml = r#"
openapi: "3.0.0"
paths:
  /pets:
    get:
      operationId: listPets
      summary: List pets
"#;
        let converter = OpenApiConverter::new();
        let tools = converter.from_yaml(yaml).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "listPets");
    }

    #[test]
    fn test_with_prefix() {
        let yaml = r#"
openapi: "3.0.0"
paths:
  /test:
    get:
      operationId: getTest
"#;
        let converter = OpenApiConverter::new().with_name_prefix("api");
        let tools = converter.from_yaml(yaml).unwrap();
        assert!(tools[0].name.starts_with("api_"));
    }
}
