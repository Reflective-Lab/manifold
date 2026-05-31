// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::sync::OnceLock;
use std::time::Duration;

use converge_core::traits::CapabilityError;
#[cfg(any(
    feature = "anthropic",
    feature = "deepseek",
    feature = "gemini",
    feature = "kimi",
    feature = "kong",
    feature = "minmax",
    feature = "mistral",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "qwen",
    feature = "staik",
))]
use converge_provider::ChatBackend;
use converge_provider::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, DynChatBackend, LlmError, ResponseFormat,
    SelectionCriteria, ToolDefinition,
};
#[cfg(feature = "anthropic")]
use manifold::AnthropicBackend;
#[cfg(feature = "deepseek")]
use manifold::DeepSeekBackend;
#[cfg(feature = "gemini")]
use manifold::GeminiBackend;
#[cfg(feature = "kimi")]
use manifold::KimiBackend;
#[cfg(feature = "minmax")]
use manifold::MinMaxBackend;
#[cfg(feature = "mistral")]
use manifold::MistralBackend;
#[cfg(feature = "openai")]
use manifold::OpenAiBackend;
#[cfg(feature = "perplexity")]
use manifold::PerplexityBackend;
#[cfg(feature = "qwen")]
use manifold::QwenBackend;
#[cfg(feature = "staik")]
use manifold::StaikBackend;
use manifold::{ChatBackendSelectionConfig, select_chat_backend};

#[derive(Clone, Copy)]
struct ConversationTuning {
    system_prompt: &'static str,
    max_tokens: u32,
    temperature: f32,
}

fn tuning_for(provider: &str) -> ConversationTuning {
    match provider {
        "openai" => ConversationTuning {
            system_prompt: "You are a concise collaboration assistant. Keep stable facts identical across turns. Prefer one short line. No preamble.",
            max_tokens: 48,
            temperature: 0.1,
        },
        "anthropic" => ConversationTuning {
            system_prompt: "You are a careful collaboration assistant. Preserve exact prior facts unless changed. Reply with one short line and no preamble.",
            max_tokens: 64,
            temperature: 0.1,
        },
        "gemini" => ConversationTuning {
            system_prompt: "You are a compact long-context assistant. Retain stable facts across turns. Reply with one short line only and no commentary.",
            max_tokens: 56,
            temperature: 0.0,
        },
        "mistral" => ConversationTuning {
            system_prompt: "You are a concise assistant. Preserve facts exactly across turns. Reply in one short line with no extra words.",
            max_tokens: 56,
            temperature: 0.1,
        },
        _ => ConversationTuning {
            system_prompt: "You are a concise assistant. Reply with one short line.",
            max_tokens: 56,
            temperature: 0.1,
        },
    }
}

fn load_env() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = dotenvy::dotenv();
    });
}

fn env_is_set(key: &str) -> bool {
    load_env();
    std::env::var_os(key).is_some_and(|value| !value.is_empty())
}

fn require_env(key: &str) -> bool {
    if env_is_set(key) {
        true
    } else {
        eprintln!("skipping live test because {key} is not set");
        false
    }
}

fn conversation_request(
    messages: Vec<ChatMessage>,
    provider: &str,
    max_tokens: Option<u32>,
) -> ChatRequest {
    let tuning = tuning_for(provider);
    ChatRequest {
        messages,
        system: Some(tuning.system_prompt.to_string()),
        tools: Vec::new(),
        response_format: ResponseFormat::Text,
        max_tokens: max_tokens.or(Some(tuning.max_tokens)),
        temperature: Some(tuning.temperature),
        stop_sequences: Vec::new(),
        model: None,
    }
}

// ── Format-integrity helpers ───────────────────────────────────────────────
// These exercise wire-format paths that selection-based tests don't cover:
// `response_format: Json` translation, and the tool_call request/response
// roundtrip. Each helper is provider-agnostic; the per-backend tests below
// just construct the backend and pass it through.

fn json_probe_request() -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "Return a JSON object with one key `status` set to \"ok\". \
                      No prose, no markdown fences."
                .to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        system: Some(
            "You output only valid JSON. No prose, no code fences, no commentary.".to_string(),
        ),
        tools: Vec::new(),
        response_format: ResponseFormat::Json,
        max_tokens: Some(64),
        temperature: Some(0.0),
        stop_sequences: Vec::new(),
        model: None,
    }
}

fn assert_json_status_ok(response: &ChatResponse, provider: &str) {
    let trimmed = response.content.trim();
    // Tolerate ```json … ``` fences some models stubbornly emit.
    let json_str = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(trimmed)
        .trim();
    let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|e| {
        panic!("[{provider}] response was not valid JSON.\nraw=`{trimmed}`\nerror={e}")
    });
    let status = parsed.get("status").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        status.eq_ignore_ascii_case("ok"),
        "[{provider}] expected status=\"ok\", got `{trimmed}` (parsed={parsed})",
    );
}

/// Lenient JSON check for providers whose `response_format` semantics are
/// genuinely different from OpenAI's `json_object`. Currently used for
/// Perplexity, where strict json_schema enforcement is awkward and we rely
/// on the system prompt to constrain the output. Asserts the response is
/// valid JSON but not the specific shape.
fn assert_response_is_valid_json(response: &ChatResponse, provider: &str) {
    let trimmed = response.content.trim();
    let json_str = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(trimmed)
        .trim();
    serde_json::from_str::<serde_json::Value>(json_str).unwrap_or_else(|e| {
        panic!("[{provider}] response was not valid JSON.\nraw=`{trimmed}`\nerror={e}")
    });
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get current weather conditions for a city.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }),
    }
}

fn tool_call_first_request(tool: ToolDefinition) -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "What's the weather in Stockholm right now? Use the get_weather tool."
                .to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        system: Some("You help by calling tools when appropriate.".to_string()),
        tools: vec![tool],
        response_format: ResponseFormat::Text,
        max_tokens: Some(256),
        temperature: Some(0.0),
        stop_sequences: Vec::new(),
        model: None,
    }
}

fn tool_call_followup_request(
    tool: ToolDefinition,
    first_content: String,
    first_tool_calls: Vec<converge_provider::ToolCall>,
    tool_call_id: String,
) -> ChatRequest {
    ChatRequest {
        messages: vec![
            ChatMessage {
                role: ChatRole::User,
                content: "What's the weather in Stockholm right now? Use the get_weather tool."
                    .to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: first_content,
                tool_calls: first_tool_calls,
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::Tool,
                content: r#"{"temperature_c": 12, "conditions": "partly cloudy"}"#.to_string(),
                tool_calls: Vec::new(),
                tool_call_id: Some(tool_call_id),
            },
        ],
        system: Some("You help by calling tools when appropriate.".to_string()),
        tools: vec![tool],
        response_format: ResponseFormat::Text,
        max_tokens: Some(256),
        temperature: Some(0.0),
        stop_sequences: Vec::new(),
        model: None,
    }
}

fn assert_response_references_tool_output(response: &ChatResponse, provider: &str) {
    let lower = response.content.to_ascii_lowercase();
    assert!(
        lower.contains("partly cloudy")
            || lower.contains("cloudy")
            || lower.contains("12")
            || lower.contains("twelve"),
        "[{provider}] final response should reference tool output \
         (temperature_c=12, conditions=\"partly cloudy\"), got: `{}`",
        response.content,
    );
}

fn negative_request() -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "Reply with the word test.".to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        system: Some("Be concise.".to_string()),
        tools: Vec::new(),
        response_format: ResponseFormat::Text,
        max_tokens: Some(24),
        temperature: Some(0.0),
        stop_sequences: Vec::new(),
        model: None,
    }
}

fn selection_config_for(provider: &str) -> ChatBackendSelectionConfig {
    let mut config = ChatBackendSelectionConfig::default().with_provider_override(provider);
    // Providers without a CostClass::VeryLow / Free model in the registry
    // can't satisfy the default `interactive()` criteria (which require
    // CostTier::Minimal). Use `analysis()` instead — it's still strict
    // enough to exercise the live API meaningfully.
    if matches!(provider, "gemini" | "mistral" | "minmax" | "kimi") {
        config = config.with_criteria(SelectionCriteria::analysis());
    }
    config
}

async fn execute_live_chat(
    backend: &dyn DynChatBackend,
    provider: &str,
    stage: &str,
    request: ChatRequest,
) -> Result<ChatResponse, LlmError> {
    let mut last_error = None;

    for attempt in 0..3 {
        match backend.chat(request.clone()).await {
            Ok(response) => return Ok(response),
            Err(error) if error.is_retryable() && attempt < 2 => {
                let sleep_for = error
                    .retry_after()
                    .unwrap_or_else(|| Duration::from_secs(2_u64.pow(attempt as u32)));
                last_error = Some(error);
                tokio::time::sleep(sleep_for).await;
            }
            Err(error) => {
                eprintln!("{stage} live call for {provider} failed: {error}");
                return Err(error);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| LlmError::ProviderError {
        message: format!("{stage} live call for {provider} exhausted retries"),
        code: None,
    }))
}

async fn run_multiturn_chat_probe(provider: &str, env_key: &str) -> Result<(), LlmError> {
    if !require_env(env_key) {
        return Ok(());
    }

    let selected = select_chat_backend(&selection_config_for(provider)).map_err(|error| {
        eprintln!("failed to select {provider} backend: {error}");
        error
    })?;

    assert_eq!(selected.provider(), provider);
    assert!(!selected.model().is_empty());

    let token = format!("{}-TOKEN-20260413", provider.to_ascii_uppercase());
    let mut messages = vec![ChatMessage {
        role: ChatRole::User,
        content: format!(
            "Remember the tracking token {token} for the next turn. Reply with exactly STORED."
        ),
        tool_calls: Vec::new(),
        tool_call_id: None,
    }];

    let first = execute_live_chat(
        selected.backend.as_ref(),
        provider,
        "first",
        conversation_request(messages.clone(), provider, Some(24)),
    )
    .await?;
    assert!(!first.content.trim().is_empty());

    messages.push(ChatMessage {
        role: ChatRole::Assistant,
        content: first.content,
        tool_calls: first.tool_calls,
        tool_call_id: None,
    });
    messages.push(ChatMessage {
        role: ChatRole::User,
        content:
            "What tracking token did I ask you to remember earlier? Reply as TOKEN=<value> ACK."
                .to_string(),
        tool_calls: Vec::new(),
        tool_call_id: None,
    });

    let second = execute_live_chat(
        selected.backend.as_ref(),
        provider,
        "second",
        conversation_request(messages, provider, None),
    )
    .await?;

    let normalized = second.content.to_ascii_uppercase();
    assert!(
        normalized.contains(&token),
        "final {provider} response did not preserve prior-turn token: {:?}",
        second.content
    );
    assert!(
        normalized.contains("ACK"),
        "final {provider} response did not include ACK sentinel: {:?}",
        second.content
    );

    Ok(())
}

fn assert_auth_denied(error: LlmError) {
    assert!(
        matches!(error, LlmError::AuthDenied { .. }),
        "expected AuthDenied, got {error:?}"
    );
}

fn assert_model_not_found(error: LlmError) {
    assert!(
        matches!(error, LlmError::ModelNotFound { .. }),
        "expected ModelNotFound, got {error:?}"
    );
}

#[cfg(feature = "openai")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_openai_happy_path_multiturn() {
    run_multiturn_chat_probe("openai", "OPENAI_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "anthropic")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_anthropic_happy_path_multiturn() {
    run_multiturn_chat_probe("anthropic", "ANTHROPIC_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "gemini")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_gemini_happy_path_multiturn() {
    if let Err(error) = run_multiturn_chat_probe("gemini", "GEMINI_API_KEY").await {
        if error.is_retryable() {
            eprintln!(
                "skipping gemini happy path because the live provider is temporarily unavailable: {error}"
            );
            return;
        }
        panic!("gemini happy path failed with non-retryable error: {error}");
    }
}

#[cfg(feature = "mistral")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_mistral_happy_path_multiturn() {
    run_multiturn_chat_probe("mistral", "MISTRAL_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "kong")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_kong_happy_path_multiturn() {
    run_multiturn_chat_probe("kong", "KONG_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "openai")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_openai_invalid_key_is_auth_denied() {
    let backend = OpenAiBackend::try_new("converge-invalid-openai-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "anthropic")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_anthropic_invalid_key_is_auth_denied() {
    let backend = AnthropicBackend::try_new("converge-invalid-anthropic-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "gemini")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_gemini_invalid_key_is_auth_denied() {
    let backend = GeminiBackend::try_new("converge-invalid-gemini-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "mistral")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_mistral_invalid_key_is_auth_denied() {
    let backend = MistralBackend::try_new("converge-invalid-mistral-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "openai")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_openai_invalid_model_is_model_not_found() {
    if !require_env("OPENAI_API_KEY") {
        return;
    }

    let backend = OpenAiBackend::from_env()
        .unwrap()
        .with_model("gpt-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

#[cfg(feature = "anthropic")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_anthropic_invalid_model_is_model_not_found() {
    if !require_env("ANTHROPIC_API_KEY") {
        return;
    }

    let backend = AnthropicBackend::from_env()
        .unwrap()
        .with_model("claude-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

#[cfg(feature = "gemini")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_gemini_invalid_model_is_model_not_found() {
    if !require_env("GEMINI_API_KEY") {
        return;
    }

    let backend = GeminiBackend::from_env()
        .unwrap()
        .with_model("gemini-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

#[cfg(feature = "mistral")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_mistral_invalid_model_is_model_not_found() {
    if !require_env("MISTRAL_API_KEY") {
        return;
    }

    let backend = MistralBackend::from_env()
        .unwrap()
        .with_model("mistral-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

// ── Perplexity ─────────────────────────────────────────────────────────────

#[cfg(feature = "perplexity")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_perplexity_happy_path_multiturn() {
    if let Err(error) = run_multiturn_chat_probe("perplexity", "PERPLEXITY_API_KEY").await {
        if error.is_retryable() {
            eprintln!(
                "skipping perplexity happy path because the live provider is temporarily unavailable: {error}"
            );
            return;
        }
        panic!("perplexity happy path failed with non-retryable error: {error}");
    }
}

#[cfg(feature = "perplexity")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_perplexity_invalid_key_is_auth_denied() {
    let backend = PerplexityBackend::try_new("converge-invalid-perplexity-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "perplexity")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_perplexity_invalid_model_is_model_not_found() {
    if !require_env("PERPLEXITY_API_KEY") {
        return;
    }

    let backend = PerplexityBackend::from_env()
        .unwrap()
        .with_model("pplx-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

// ── DeepSeek ───────────────────────────────────────────────────────────────

#[cfg(feature = "deepseek")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_deepseek_happy_path_multiturn() {
    run_multiturn_chat_probe("deepseek", "DEEPSEEK_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "deepseek")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_deepseek_invalid_key_is_auth_denied() {
    let backend = DeepSeekBackend::try_new("converge-invalid-deepseek-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "deepseek")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_deepseek_invalid_model_is_model_not_found() {
    if !require_env("DEEPSEEK_API_KEY") {
        return;
    }

    let backend = DeepSeekBackend::from_env()
        .unwrap()
        .with_model("deepseek-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

// ── Kimi (Moonshot) ────────────────────────────────────────────────────────

#[cfg(feature = "kimi")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_kimi_happy_path_multiturn() {
    run_multiturn_chat_probe("kimi", "KIMI_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "kimi")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_kimi_invalid_key_is_auth_denied() {
    let backend = KimiBackend::try_new("converge-invalid-kimi-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "kimi")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_kimi_invalid_model_is_model_not_found() {
    if !require_env("KIMI_API_KEY") {
        return;
    }

    let backend = KimiBackend::from_env()
        .unwrap()
        .with_model("moonshot-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

// ── Qwen (DashScope) ───────────────────────────────────────────────────────

#[cfg(feature = "qwen")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_qwen_happy_path_multiturn() {
    run_multiturn_chat_probe("qwen", "QWEN_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "qwen")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_qwen_invalid_key_is_auth_denied() {
    let backend = QwenBackend::try_new("converge-invalid-qwen-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "qwen")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_qwen_invalid_model_is_model_not_found() {
    if !require_env("QWEN_API_KEY") {
        return;
    }

    let backend = QwenBackend::from_env()
        .unwrap()
        .with_model("qwen-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

// ── MiniMax ────────────────────────────────────────────────────────────────

#[cfg(feature = "minmax")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_minmax_happy_path_multiturn() {
    // MiniMax backend reads MINIMAX_API_KEY (note the `I`) to match the
    // company's branding; the cargo feature name is `minmax` for legacy reasons.
    run_multiturn_chat_probe("minmax", "MINIMAX_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "minmax")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_minmax_invalid_key_is_auth_denied() {
    let backend = MinMaxBackend::try_new("converge-invalid-minmax-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "minmax")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_minmax_invalid_model_is_model_not_found() {
    if !require_env("MINIMAX_API_KEY") {
        return;
    }

    let backend = MinMaxBackend::from_env()
        .unwrap()
        .with_model("minimax-converge-not-real")
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_model_not_found(error);
}

// ── Staik (Swedish EU/SE-hosted OpenAI-compatible) ─────────────────────────

#[cfg(feature = "staik")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_staik_happy_path_multiturn() {
    run_multiturn_chat_probe("staik", "STAIK_API_KEY")
        .await
        .unwrap();
}

#[cfg(feature = "staik")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_staik_invalid_key_is_auth_denied() {
    let backend = StaikBackend::try_new("converge-invalid-staik-key")
        .unwrap()
        .with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

// ── Format-integrity tests ─────────────────────────────────────────────────
// These call backends DIRECTLY (no selection layer) to exercise the wire
// format for `response_format: Json` and tool_call roundtrips. Failures
// here usually indicate a request-serialization or response-parsing bug
// specific to the provider's API shape, not a model behavior issue.

#[cfg(feature = "anthropic")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_anthropic_json_format_integrity() {
    if !require_env("ANTHROPIC_API_KEY") {
        return;
    }
    let backend = AnthropicBackend::from_env().unwrap().with_max_retries(0);
    let response = ChatBackend::chat(&backend, json_probe_request())
        .await
        .unwrap();
    assert_json_status_ok(&response, "anthropic");
}

#[cfg(feature = "mistral")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_mistral_json_format_integrity() {
    if !require_env("MISTRAL_API_KEY") {
        return;
    }
    let backend = MistralBackend::from_env().unwrap().with_max_retries(0);
    let response = ChatBackend::chat(&backend, json_probe_request())
        .await
        .unwrap();
    assert_json_status_ok(&response, "mistral");
}

#[cfg(feature = "openrouter")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_openrouter_json_format_integrity() {
    if !require_env("OPENROUTER_API_KEY") {
        return;
    }
    let backend = manifold::OpenRouterBackend::from_env()
        .unwrap()
        .with_max_retries(0);
    let response = ChatBackend::chat(&backend, json_probe_request())
        .await
        .unwrap();
    assert_json_status_ok(&response, "openrouter");
}

#[cfg(feature = "perplexity")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_perplexity_json_format_integrity() {
    if !require_env("PERPLEXITY_API_KEY") {
        return;
    }
    let backend = PerplexityBackend::from_env().unwrap().with_max_retries(0);
    let response = ChatBackend::chat(&backend, json_probe_request())
        .await
        .unwrap();
    // Perplexity's JSON mode is best-effort (see backend doc comment). Assert
    // we got valid JSON back — content shape is the system prompt's job.
    assert_response_is_valid_json(&response, "perplexity");
}

#[cfg(feature = "staik")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_staik_json_format_integrity() {
    if !require_env("STAIK_API_KEY") {
        return;
    }
    let backend = StaikBackend::from_env().unwrap().with_max_retries(0);
    let response = ChatBackend::chat(&backend, json_probe_request())
        .await
        .unwrap();
    assert_json_status_ok(&response, "staik");
}

// Tool-call roundtrip: anthropic, mistral, openrouter. Perplexity and Staik
// have model-dependent tool support; skip until we know which model each
// will select.

#[cfg(feature = "anthropic")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_anthropic_tool_call_roundtrip() {
    if !require_env("ANTHROPIC_API_KEY") {
        return;
    }
    let backend = AnthropicBackend::from_env().unwrap().with_max_retries(0);
    let tool = weather_tool();
    let first = ChatBackend::chat(&backend, tool_call_first_request(tool.clone()))
        .await
        .unwrap();
    assert!(
        !first.tool_calls.is_empty(),
        "[anthropic] expected a tool_call from the first turn, got content: `{}`",
        first.content,
    );
    let call_id = first.tool_calls[0].id.clone();
    let call_name = first.tool_calls[0].name.clone();
    assert_eq!(call_name, "get_weather", "[anthropic] wrong tool name");
    let followup = tool_call_followup_request(tool, first.content, first.tool_calls, call_id);
    let second = ChatBackend::chat(&backend, followup).await.unwrap();
    assert_response_references_tool_output(&second, "anthropic");
}

#[cfg(feature = "mistral")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_mistral_tool_call_roundtrip() {
    if !require_env("MISTRAL_API_KEY") {
        return;
    }
    let backend = MistralBackend::from_env().unwrap().with_max_retries(0);
    let tool = weather_tool();
    let first = ChatBackend::chat(&backend, tool_call_first_request(tool.clone()))
        .await
        .unwrap();
    assert!(
        !first.tool_calls.is_empty(),
        "[mistral] expected a tool_call from the first turn, got content: `{}`",
        first.content,
    );
    let call_id = first.tool_calls[0].id.clone();
    let call_name = first.tool_calls[0].name.clone();
    assert_eq!(call_name, "get_weather", "[mistral] wrong tool name");
    let followup = tool_call_followup_request(tool, first.content, first.tool_calls, call_id);
    let second = ChatBackend::chat(&backend, followup).await.unwrap();
    assert_response_references_tool_output(&second, "mistral");
}

#[cfg(feature = "openrouter")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_openrouter_tool_call_roundtrip() {
    if !require_env("OPENROUTER_API_KEY") {
        return;
    }
    let backend = manifold::OpenRouterBackend::from_env()
        .unwrap()
        .with_max_retries(0);
    let tool = weather_tool();
    let first = ChatBackend::chat(&backend, tool_call_first_request(tool.clone()))
        .await
        .unwrap();
    assert!(
        !first.tool_calls.is_empty(),
        "[openrouter] expected a tool_call from the first turn, got content: `{}`",
        first.content,
    );
    let call_id = first.tool_calls[0].id.clone();
    let call_name = first.tool_calls[0].name.clone();
    assert_eq!(call_name, "get_weather", "[openrouter] wrong tool name");
    let followup = tool_call_followup_request(tool, first.content, first.tool_calls, call_id);
    let second = ChatBackend::chat(&backend, followup).await.unwrap();
    assert_response_references_tool_output(&second, "openrouter");
}

// ── Streaming tests ────────────────────────────────────────────────────────

#[cfg(all(feature = "openrouter", feature = "streaming"))]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_openrouter_streaming_emits_tokens_and_finish() {
    use futures::StreamExt;
    use manifold::{ChatEvent, OpenRouterBackend, StreamingChatBackend};

    if !require_env("OPENROUTER_API_KEY") {
        return;
    }
    let backend = OpenRouterBackend::from_env().unwrap().with_max_retries(0);
    let req = ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "Count from one to five, one number per line. Nothing else.".to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        system: Some("Reply concisely, no preamble.".to_string()),
        tools: Vec::new(),
        response_format: ResponseFormat::Text,
        max_tokens: Some(64),
        temperature: Some(0.0),
        stop_sequences: Vec::new(),
        model: None,
    };
    let mut stream = backend.chat_stream(req).await.unwrap();
    let mut delta_count = 0;
    let mut text = String::new();
    let mut finish_seen = false;
    let mut usage_seen = false;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ChatEvent::TextDelta(s) => {
                delta_count += 1;
                text.push_str(&s);
            }
            ChatEvent::Finish(_) => finish_seen = true,
            ChatEvent::Usage(_) => usage_seen = true,
            _ => {}
        }
    }
    assert!(
        delta_count > 1,
        "expected multiple text deltas, got {delta_count}"
    );
    assert!(!text.is_empty(), "expected non-empty accumulated text");
    assert!(finish_seen, "expected a Finish event");
    // Usage may or may not arrive depending on stream_options support;
    // not a hard requirement.
    eprintln!(
        "[streaming] {delta_count} deltas, {} chars, finish={finish_seen}, usage={usage_seen}",
        text.len()
    );
}

// NOTE: no `live_staik_invalid_model_is_model_not_found` test.
// The Staik gateway transparently routes unknown model names to a default
// upstream model (observed: requests with `staik-converge-not-real` resolve
// to `qwen3.6:35b-a3b`) and returns a successful response rather than a
// ModelNotFound error. The behavior is intentional gateway routing; the
// test scaffolding here can't trigger ModelNotFound from a Staik client.
