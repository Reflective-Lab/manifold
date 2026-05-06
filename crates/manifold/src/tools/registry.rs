// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Tool registry for unified tool discovery and invocation.

use super::{ToolCall, ToolDefinition, ToolError, ToolResult, ToolSource};
use std::collections::HashMap;

/// Registry for tool discovery and invocation.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: ToolDefinition) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn register_all(&mut self, tools: impl IntoIterator<Item = ToolDefinition>) {
        for tool in tools {
            self.register(tool);
        }
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    #[must_use]
    pub fn list_tools(&self) -> Vec<&ToolDefinition> {
        self.tools.values().collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    #[must_use]
    pub fn tools_by_source(&self, filter: SourceFilter) -> Vec<&ToolDefinition> {
        self.tools
            .values()
            .filter(|t| filter.matches(&t.source))
            .collect()
    }

    pub fn call_tool(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let tool = self
            .get(&call.tool_name)
            .ok_or_else(|| ToolError::not_found(&call.tool_name))?;

        match &tool.source {
            ToolSource::Inline => Err(ToolError::unsupported_source("inline")),
            ToolSource::Mcp { .. } => Err(ToolError::unsupported_source("mcp (use McpClient)")),
            ToolSource::OpenApi { .. } => Err(ToolError::unsupported_source("openapi")),
            ToolSource::GraphQl { .. } => Err(ToolError::unsupported_source("graphql")),
        }
    }

    #[must_use]
    pub fn to_llm_tools(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema.schema
                    }
                })
            })
            .collect()
    }

    #[must_use]
    pub fn to_anthropic_tools(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema.schema
                })
            })
            .collect()
    }
}

/// Filter for selecting tools by source type.
#[derive(Debug, Clone, Copy, Default)]
pub enum SourceFilter {
    #[default]
    All,
    Mcp,
    OpenApi,
    GraphQl,
    Inline,
}

impl SourceFilter {
    #[must_use]
    pub fn matches(&self, source: &ToolSource) -> bool {
        match self {
            Self::All => true,
            Self::Mcp => matches!(source, ToolSource::Mcp { .. }),
            Self::OpenApi => matches!(source, ToolSource::OpenApi { .. }),
            Self::GraphQl => matches!(source, ToolSource::GraphQl { .. }),
            Self::Inline => matches!(source, ToolSource::Inline),
        }
    }
}

/// Trait for tool execution handlers.
pub trait ToolHandler: std::fmt::Debug + Send + Sync {
    fn can_handle(&self, tool: &ToolDefinition) -> bool;
    fn execute(&self, tool: &ToolDefinition, call: &ToolCall) -> Result<ToolResult, ToolError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::InputSchema;

    #[test]
    fn test_registry_operations() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(ToolDefinition::new("test", "Test", InputSchema::empty()));
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("test"));
    }
}
