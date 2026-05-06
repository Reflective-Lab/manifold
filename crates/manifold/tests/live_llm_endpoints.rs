// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::sync::OnceLock;
use std::time::Duration;

use converge_core::traits::CapabilityError;
use converge_provider::{
    ChatBackend, ChatMessage, ChatRequest, ChatResponse, ChatRole, DynChatBackend, LlmError,
    ResponseFormat, SelectionCriteria,
};
#[cfg(feature = "anthropic")]
use manifold::AnthropicBackend;
#[cfg(feature = "gemini")]
use manifold::GeminiBackend;
#[cfg(feature = "mistral")]
use manifold::MistralBackend;
#[cfg(feature = "openai")]
use manifold::OpenAiBackend;
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
        let _ = dotenv::dotenv();
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
    if matches!(provider, "gemini" | "mistral") {
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
    let backend = OpenAiBackend::new("converge-invalid-openai-key").with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "anthropic")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_anthropic_invalid_key_is_auth_denied() {
    let backend = AnthropicBackend::new("converge-invalid-anthropic-key").with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "gemini")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_gemini_invalid_key_is_auth_denied() {
    let backend = GeminiBackend::new("converge-invalid-gemini-key").with_max_retries(0);
    let error = ChatBackend::chat(&backend, negative_request())
        .await
        .unwrap_err();
    assert_auth_denied(error);
}

#[cfg(feature = "mistral")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires live API access"]
async fn live_mistral_invalid_key_is_auth_denied() {
    let backend = MistralBackend::new("converge-invalid-mistral-key").with_max_retries(0);
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
