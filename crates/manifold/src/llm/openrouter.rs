// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::collections::HashMap;

use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use super::error_classification::{
    classify_http_error, map_backend_error, network_error, parse_error,
};
use super::format_contract::finalize_chat_response;
use super::retry::{RetryOutcome, retry_with_backoff};
use crate::secret::{EnvSecretProvider, SecretProvider, SecretString};
use converge_core::backend::{BackendError, BackendResult};
use converge_provider::{
    BoxFuture, ChatBackend, ChatRequest, ChatResponse, ChatRole, FinishReason as ChatFinishReason,
    LlmError as ChatLlmError, ResponseFormat, TokenUsage as ChatTokenUsage,
    ToolCall as ChatToolCall,
};

/// OpenRouter backend — routes to any model through `openrouter.ai/api`.
///
/// OpenRouter uses the OpenAI chat completions format. Model names follow
/// the `provider/model` convention (e.g. `anthropic/claude-sonnet-4`).
pub struct OpenRouterBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
    site_url: String,
    site_name: String,
}

impl OpenRouterBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction.
    /// Production code should prefer [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "OPENROUTER_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "anthropic/claude-sonnet-4".to_string(),
            base_url: "https://openrouter.ai/api".to_string(),
            client: Client::new(),
            temperature: 0.0,
            max_retries: 3,
            site_url: String::new(),
            site_name: String::new(),
        })
    }

    pub fn from_env() -> BackendResult<Self> {
        Self::from_secret_provider(&EnvSecretProvider)
    }

    pub fn from_secret_provider(secrets: &dyn SecretProvider) -> BackendResult<Self> {
        let api_key =
            secrets
                .get_secret("OPENROUTER_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("OPENROUTER_API_KEY: {e}"),
                })?;

        let model = secrets
            .get_secret("OPENROUTER_MODEL")
            .map(|s| s.expose().to_string())
            .unwrap_or_else(|_| "anthropic/claude-sonnet-4".to_string());
        let base_url = secrets
            .get_secret("OPENROUTER_BASE_URL")
            .map(|s| s.expose().to_string())
            .unwrap_or_else(|_| "https://openrouter.ai/api".to_string());
        let site_url = secrets
            .get_secret("OPENROUTER_SITE_URL")
            .map(|s| s.expose().to_string())
            .unwrap_or_default();
        let site_name = secrets
            .get_secret("OPENROUTER_SITE_NAME")
            .map(|s| s.expose().to_string())
            .unwrap_or_default();

        Ok(Self {
            api_key,
            model,
            base_url,
            client: Client::new(),
            temperature: 0.0,
            max_retries: 3,
            site_url,
            site_name,
        })
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    #[must_use]
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    #[must_use]
    pub fn with_site_url(mut self, url: impl Into<String>) -> Self {
        self.site_url = url.into();
        self
    }

    #[must_use]
    pub fn with_site_name(mut self, name: impl Into<String>) -> Self {
        self.site_name = name.into();
        self
    }

    fn build_headers(&self) -> BackendResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let auth_value = format!("Bearer {}", self.api_key.expose());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value).map_err(|e| BackendError::InvalidRequest {
                message: format!("Invalid API key: {e}"),
            })?,
        );

        if !self.site_url.is_empty() {
            if let Ok(val) = HeaderValue::from_str(&self.site_url) {
                headers.insert("HTTP-Referer", val);
            }
        }
        if !self.site_name.is_empty() {
            if let Ok(val) = HeaderValue::from_str(&self.site_name) {
                headers.insert("X-Title", val);
            }
        }

        Ok(headers)
    }

    fn build_request(&self, req: &ChatRequest) -> OpenRouterRequest {
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let temperature = req.temperature.unwrap_or(self.temperature);
        let max_tokens = req.max_tokens.map(|t| t as usize).unwrap_or(4096);

        let mut messages = Vec::new();

        // Append format instruction to system prompt for all structured formats.
        // JSON also gets native response_format, but OpenAI-compatible APIs require
        // the word "json" in the messages when using json_object mode.
        let system_content = if let Some(instruction) = req.response_format.system_instruction() {
            let base = req.system.clone().unwrap_or_default();
            Some(format!("{base}\n\n{instruction}"))
        } else {
            req.system.clone()
        };

        if let Some(system) = &system_content {
            messages.push(OpenRouterMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        for msg in &req.messages {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
            };
            let tool_calls = if msg.tool_calls.is_empty() {
                None
            } else {
                Some(
                    msg.tool_calls
                        .iter()
                        .map(|tool_call| OpenRouterResponseToolCall {
                            id: tool_call.id.clone(),
                            function: OpenRouterResponseFunction {
                                name: tool_call.name.clone(),
                                arguments: tool_call.arguments.clone(),
                            },
                        })
                        .collect(),
                )
            };
            let content = if msg.content.is_empty() && tool_calls.is_some() {
                None
            } else {
                Some(msg.content.clone())
            };
            messages.push(OpenRouterMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            });
        }

        let tools: Option<Vec<OpenRouterTool>> = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| OpenRouterTool {
                        r#type: "function".to_string(),
                        function: OpenRouterFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        // Only JSON has native API-level enforcement; other structured formats
        // are handled via the system prompt instruction above.
        let response_format = match req.response_format {
            ResponseFormat::Json => Some(serde_json::json!({"type": "json_object"})),
            _ => None,
        };

        let stop = if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences.clone())
        };

        OpenRouterRequest {
            model,
            messages,
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            tools,
            response_format,
            stop,
        }
    }

    async fn chat_async(&self, req: ChatRequest) -> Result<ChatResponse, ChatLlmError> {
        let openrouter_req = self.build_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let response = self.execute_with_retries(&model, &openrouter_req).await?;

        let choice = response.choices.first();

        let content = choice
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let tool_calls = choice
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|calls| {
                calls
                    .iter()
                    .map(|tc| ChatToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let finish_reason = choice.and_then(|c| match c.finish_reason.as_deref() {
            Some("stop") => Some(ChatFinishReason::Stop),
            Some("length") => Some(ChatFinishReason::Length),
            Some("tool_calls") => Some(ChatFinishReason::ToolCalls),
            Some("content_filter") => Some(ChatFinishReason::ContentFilter),
            _ => None,
        });

        let metadata = extract_openrouter_metadata(&response);

        finalize_chat_response(
            &req,
            ChatResponse {
                content,
                tool_calls,
                usage: response.usage.map(|u| ChatTokenUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                }),
                model: Some(response.model),
                finish_reason,
                metadata,
            },
        )
    }

    async fn execute_with_retries(
        &self,
        model: &str,
        request: &OpenRouterRequest,
    ) -> Result<OpenRouterResponse, ChatLlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let headers = self.build_headers().map_err(map_backend_error)?;

        retry_with_backoff(self.max_retries, || {
            let client = &self.client;
            let url = &url;
            let headers = headers.clone();
            let request = request;
            async move {
                match client.post(url).headers(headers).json(request).send().await {
                    Ok(response) => {
                        let status = response.status();
                        if status.is_success() {
                            match response.json::<OpenRouterResponse>().await {
                                Ok(parsed) => RetryOutcome::Success(parsed),
                                Err(e) => RetryOutcome::Retry(parse_error(e)),
                            }
                        } else if status.as_u16() == 429 || status.as_u16() >= 500 {
                            let body = response.text().await.unwrap_or_default();
                            RetryOutcome::Retry(classify_http_error(status.as_u16(), &body, model))
                        } else {
                            let body = response.text().await.unwrap_or_default();
                            RetryOutcome::Fail(classify_http_error(status.as_u16(), &body, model))
                        }
                    }
                    Err(e) => RetryOutcome::Retry(network_error(e)),
                }
            }
        })
        .await
    }
}

fn extract_openrouter_metadata(response: &OpenRouterResponse) -> HashMap<String, String> {
    let mut meta = HashMap::new();

    if let Some(provider) = &response.provider {
        meta.insert("provider".to_string(), provider.clone());
    }

    if let Some(usage) = &response.usage {
        if let Some(cost) = usage.cost {
            meta.insert("cost".to_string(), cost.to_string());
        }
        if let Some(byok) = usage.is_byok {
            meta.insert("is_byok".to_string(), byok.to_string());
        }
        if let Some(details) = &usage.cost_details {
            if let Some(v) = details.upstream_inference_cost {
                meta.insert("cost.upstream".to_string(), v.to_string());
            }
            if let Some(v) = details.upstream_inference_prompt_cost {
                meta.insert("cost.prompt".to_string(), v.to_string());
            }
            if let Some(v) = details.upstream_inference_completions_cost {
                meta.insert("cost.completion".to_string(), v.to_string());
            }
        }
    }

    meta
}

impl ChatBackend for OpenRouterBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// OpenRouter API Types (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct OpenRouterMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenRouterTool {
    r#type: String,
    function: OpenRouterFunction,
}

#[derive(Debug, Serialize)]
struct OpenRouterFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    model: String,
    choices: Vec<OpenRouterChoice>,
    usage: Option<OpenRouterUsage>,
    #[serde(default)]
    provider: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterResponseToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterResponseToolCall {
    id: String,
    function: OpenRouterResponseFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    #[serde(default)]
    cost: Option<f64>,
    #[serde(default)]
    is_byok: Option<bool>,
    #[serde(default)]
    cost_details: Option<OpenRouterCostDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterCostDetails {
    #[serde(default)]
    upstream_inference_cost: Option<f64>,
    #[serde(default)]
    upstream_inference_prompt_cost: Option<f64>,
    #[serde(default)]
    upstream_inference_completions_cost: Option<f64>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use converge_core::traits::{
        ChatMessage, ChatRequest, ChatRole, ResponseFormat, ToolDefinition,
    };
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_openrouter_backend_creation() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap()
            .with_model("openai/gpt-4o")
            .with_temperature(0.5);

        assert_eq!(backend.model, "openai/gpt-4o");
        assert_eq!(backend.temperature, 0.5);
        assert_eq!(backend.api_key.expose(), "test-key");
        assert_eq!(backend.base_url, "https://openrouter.ai/api");
    }

    #[test]
    fn test_default_model_is_claude() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
        assert_eq!(backend.model, "anthropic/claude-sonnet-4");
    }

    #[test]
    fn test_build_request_basic() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: Vec::new(),
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let openrouter_req = backend.build_request(&req);

        assert_eq!(openrouter_req.model, "anthropic/claude-sonnet-4");
        assert_eq!(openrouter_req.messages.len(), 1);
        assert_eq!(openrouter_req.messages[0].role, "user");
        assert!(openrouter_req.tools.is_none());
    }

    #[test]
    fn test_build_request_with_tools() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "What's the weather?".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            }],
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let openrouter_req = backend.build_request(&req);
        let tools = openrouter_req.tools.unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_site_headers_set() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap()
            .with_site_url("https://converge.zone")
            .with_site_name("Converge");

        let headers = backend.build_headers().unwrap();

        assert_eq!(
            headers.get("HTTP-Referer").unwrap().to_str().unwrap(),
            "https://converge.zone"
        );
        assert_eq!(
            headers.get("X-Title").unwrap().to_str().unwrap(),
            "Converge"
        );
    }

    #[test]
    fn test_site_headers_omitted_when_empty() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
        let headers = backend.build_headers().unwrap();

        assert!(headers.get("HTTP-Referer").is_none());
        assert!(headers.get("X-Title").is_none());
    }

    #[test]
    fn test_chat_with_mock_server() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .and(header("HTTP-Referer", "https://converge.zone"))
                .and(header("X-Title", "Converge"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "gen-test",
                    "model": "anthropic/claude-sonnet-4",
                    "provider": "Anthropic",
                    "choices": [{
                        "message": {
                            "content": "Hello from OpenRouter!",
                            "tool_calls": null
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.000_075,
                        "is_byok": false,
                        "cost_details": {
                            "upstream_inference_cost": 0.000_075,
                            "upstream_inference_prompt_cost": 0.000_03,
                            "upstream_inference_completions_cost": 0.000_045
                        }
                    }
                })))
                .mount(&server)
                .await;
        });

        let backend = OpenRouterBackend::try_new("test-key").unwrap()
            .with_base_url(server.uri())
            .with_site_url("https://converge.zone")
            .with_site_name("Converge");

        let response = runtime
            .block_on(backend.chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "Hello".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }],
                system: None,
                tools: Vec::new(),
                response_format: ResponseFormat::default(),
                max_tokens: None,
                temperature: None,
                stop_sequences: Vec::new(),
                model: None,
            }))
            .unwrap();

        assert_eq!(response.content, "Hello from OpenRouter!");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::Stop));
        assert_eq!(response.usage.as_ref().unwrap().total_tokens, 15);

        // Verify cost/provider metadata was captured from response body
        assert_eq!(response.metadata.get("provider").unwrap(), "Anthropic");
        assert_eq!(response.metadata.get("cost").unwrap(), "0.000075");
        assert_eq!(response.metadata.get("is_byok").unwrap(), "false");
        assert_eq!(response.metadata.get("cost.upstream").unwrap(), "0.000075");
        assert_eq!(response.metadata.get("cost.prompt").unwrap(), "0.00003");
        assert_eq!(
            response.metadata.get("cost.completion").unwrap(),
            "0.000045"
        );

        drop(server);
        drop(runtime);
    }
}
