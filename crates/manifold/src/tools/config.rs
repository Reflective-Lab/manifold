// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! YAML-based tool configuration loader.

use super::{GraphQlConverter, OpenApiConverter, ToolDefinition, ToolError, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Error type for tools configuration loading.
#[derive(Debug, thiserror::Error)]
pub enum ToolsConfigError {
    #[error("Failed to read config: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Failed to parse YAML: {0}")]
    ParseError(#[from] serde_yaml::Error),
    #[error("Validation failed: {0}")]
    ValidationError(String),
    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),
}

/// Root of the tools YAML configuration.
#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolsConfig {
    #[serde(default)]
    pub openapi_specs: HashMap<String, OpenApiConfig>,
    #[serde(default)]
    pub graphql_endpoints: HashMap<String, GraphQlConfig>,
    #[serde(default)]
    pub inline_tools: Vec<InlineToolConfig>,
}

/// `OpenAPI` specification configuration.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenApiConfig {
    pub path: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub name_prefix: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl OpenApiConfig {
    #[must_use]
    pub fn to_converter(&self) -> OpenApiConverter {
        let mut converter = OpenApiConverter::new();
        if let Some(ref base_url) = self.base_url {
            converter = converter.with_base_url(base_url);
        }
        if let Some(ref prefix) = self.name_prefix {
            converter = converter.with_name_prefix(prefix);
        }
        if !self.tags.is_empty() {
            converter = converter.with_tag_filter(self.tags.clone());
        }
        converter
    }

    pub fn load_tools(&self, base_path: &Path) -> Result<Vec<ToolDefinition>, ToolsConfigError> {
        let spec_path = base_path.join(&self.path);
        let content = std::fs::read_to_string(&spec_path)?;
        let converter = self.to_converter();
        converter
            .from_yaml(&content)
            .or_else(|_| converter.from_json(&content))
            .map_err(ToolsConfigError::from)
    }
}

/// GraphQL endpoint configuration.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GraphQlConfig {
    pub endpoint: String,
    #[serde(default)]
    pub auth_header: Option<String>,
    #[serde(default = "default_enabled")]
    pub include_queries: bool,
    #[serde(default)]
    pub include_mutations: bool,
    #[serde(default)]
    pub name_prefix: Option<String>,
    #[serde(default)]
    pub field_filter: Vec<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl GraphQlConfig {
    #[must_use]
    pub fn to_converter(&self) -> GraphQlConverter {
        let mut converter = GraphQlConverter::new(&self.endpoint)
            .include_queries(self.include_queries)
            .include_mutations(self.include_mutations);
        if let Some(ref prefix) = self.name_prefix {
            converter = converter.with_name_prefix(prefix);
        }
        if !self.field_filter.is_empty() {
            converter = converter.with_field_filter(self.field_filter.clone());
        }
        converter
    }
}

/// Inline tool definition.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InlineToolConfig {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl InlineToolConfig {
    #[must_use]
    pub fn to_tool_definition(&self) -> ToolDefinition {
        use super::InputSchema;
        ToolDefinition::new(
            &self.name,
            &self.description,
            if self.input_schema.is_null() {
                InputSchema::empty()
            } else {
                InputSchema::from_json_schema(self.input_schema.clone())
            },
        )
    }
}

fn default_enabled() -> bool {
    true
}

pub fn load_tools_config(path: impl AsRef<Path>) -> Result<ToolsConfig, ToolsConfigError> {
    let content = std::fs::read_to_string(path)?;
    let config: ToolsConfig = serde_yaml::from_str(&content)?;
    Ok(config)
}

pub fn parse_tools_config(yaml: &str) -> Result<ToolsConfig, ToolsConfigError> {
    let config: ToolsConfig = serde_yaml::from_str(yaml)?;
    Ok(config)
}

pub fn build_registry_from_config(
    config: &ToolsConfig,
    base_path: &Path,
) -> Result<ToolRegistry, ToolsConfigError> {
    let mut registry = ToolRegistry::new();

    for (name, openapi_config) in &config.openapi_specs {
        if openapi_config.enabled {
            match openapi_config.load_tools(base_path) {
                Ok(tools) => registry.register_all(tools),
                Err(e) => tracing::warn!("Failed to load OpenAPI '{}': {}", name, e),
            }
        }
    }

    for tool_config in &config.inline_tools {
        if tool_config.enabled {
            registry.register(tool_config.to_tool_definition());
        }
    }

    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let yaml = r"
inline_tools:
  - name: echo
    description: Echo input
";
        let config = parse_tools_config(yaml).unwrap();
        assert_eq!(config.inline_tools.len(), 1);
    }
}
