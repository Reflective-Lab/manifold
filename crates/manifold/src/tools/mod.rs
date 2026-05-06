// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Tool abstractions for `OpenAPI` and GraphQL integration.
//!
//! This module provides a unified interface for tool discovery, definition,
//! and execution across multiple sources:
//!
//! - **`OpenAPI`**: Convert `OpenAPI` specs to tool definitions
//! - **GraphQL**: Introspect GraphQL schemas for tool discovery
//!
//! # Core Types
//!
//! - [`ToolDefinition`]: Describes a tool's interface (name, schema, source)
//! - [`ToolCall`]: A request to invoke a tool
//! - [`ToolResult`]: The outcome of a tool invocation
//! - [`ToolSource`]: Where the tool came from (`OpenAPI`, GraphQL, inline)

mod definition;
mod error;
mod registry;

// Integration modules
pub mod config;
pub mod graphql;
pub mod openapi;

// Re-exports
pub use definition::{
    GraphQlOperationType, InputSchema, ToolCall, ToolDefinition, ToolResult, ToolResultContent,
    ToolSource,
};
pub use error::{ToolError, ToolErrorKind};
pub use registry::{SourceFilter, ToolHandler, ToolRegistry};

// Convenience re-exports from submodules
pub use config::{
    GraphQlConfig, InlineToolConfig, OpenApiConfig, ToolsConfig, ToolsConfigError,
    build_registry_from_config, load_tools_config, parse_tools_config,
};
pub use graphql::GraphQlConverter;
pub use openapi::OpenApiConverter;

/// Tool format for tool definitions injected into prompts.
#[derive(Debug, Clone, Copy, Default)]
pub enum ToolFormat {
    #[default]
    Anthropic,
    OpenAi,
    Generic,
}
