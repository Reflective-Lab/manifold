// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::error::Error;

use converge_provider::{
    ChatMessage, ChatRequest, ChatRole, RequiredCapabilities, ResponseFormat, SelectionCriteria,
};
use manifold::{ChatBackendSelectionConfig, SelectedChatBackend, select_chat_backend};

struct ConversationTuning {
    system_prompt: String,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Clone)]
struct CompanyState {
    name: String,
    city: String,
    product: String,
}

fn tuning_for(provider: &str) -> ConversationTuning {
    match provider {
        "openai" => ConversationTuning {
            system_prompt: "You are a concise collaboration assistant. Use the minimum words needed. Keep stable facts identical across turns. Prefer short labeled lines over prose. No preamble.".to_string(),
            max_tokens: 140,
            temperature: 0.2,
        },
        "anthropic" => ConversationTuning {
            system_prompt: "You are a careful collaboration assistant. Be concise, but preserve exact prior facts unless the user changes them. Use plain text with one item per line. No prefatory filler.".to_string(),
            max_tokens: 180,
            temperature: 0.2,
        },
        "gemini" => ConversationTuning {
            system_prompt: "You are a concise long-context assistant. Maintain a compact internal summary of stable facts. When the user asks for an exact format, output every required field in that exact order and never omit unchanged fields. Use short labeled lines only. No preamble, no commentary.".to_string(),
            max_tokens: 160,
            temperature: 0.0,
        },
        _ => ConversationTuning {
            system_prompt: "You are a concise assistant. Use short labeled lines and no preamble.".to_string(),
            max_tokens: 160,
            temperature: 0.2,
        },
    }
}

fn scenario_turns() -> Vec<&'static str> {
    vec![
        "Invent a tiny fictional company and output exactly three lines in this format: name: <value>, city: <value>, product: <value>.",
        "Change only the city to Lisbon. Keep the name and product exactly the same. Output the same three lines only.",
        "In one short sentence, restate the final company using the updated city and the unchanged name and product.",
    ]
}

fn parse_company_state(response: &str) -> Result<CompanyState, String> {
    let mut name = None;
    let mut city = None;
    let mut product = None;
    let lines: Vec<&str> = response
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    if lines.len() != 3 {
        return Err(format!(
            "expected exactly 3 non-empty lines, got {}: {:?}",
            lines.len(),
            lines
        ));
    }

    for line in lines {
        let Some((key, value)) = line.split_once(':') else {
            return Err(format!("expected key: value line, got {line:?}"));
        };
        let value = value.trim();
        if value.is_empty() {
            return Err(format!("empty value for {key:?}"));
        }

        match key.trim() {
            "name" => name = Some(value.to_string()),
            "city" => city = Some(value.to_string()),
            "product" => product = Some(value.to_string()),
            other => return Err(format!("unexpected key {other:?}")),
        }
    }

    Ok(CompanyState {
        name: name.ok_or_else(|| "missing name".to_string())?,
        city: city.ok_or_else(|| "missing city".to_string())?,
        product: product.ok_or_else(|| "missing product".to_string())?,
    })
}

fn validate_turn(
    turn_index: usize,
    response: &str,
    prior: Option<&CompanyState>,
) -> Result<Option<CompanyState>, String> {
    match turn_index {
        0 => parse_company_state(response).map(Some),
        1 => {
            let previous = prior.ok_or_else(|| "missing prior company state".to_string())?;
            let current = parse_company_state(response)?;
            if current.name != previous.name {
                return Err(format!(
                    "name changed unexpectedly: {:?} -> {:?}",
                    previous.name, current.name
                ));
            }
            if current.product != previous.product {
                return Err(format!(
                    "product changed unexpectedly: {:?} -> {:?}",
                    previous.product, current.product
                ));
            }
            if current.city != "Lisbon" {
                return Err(format!("expected city Lisbon, got {:?}", current.city));
            }
            Ok(Some(current))
        }
        2 => {
            let previous = prior.ok_or_else(|| "missing prior company state".to_string())?;
            let normalized = response.trim();
            for required in [&previous.name, &previous.city, &previous.product] {
                if !normalized.contains(required) {
                    return Err(format!(
                        "final sentence missing required fact {:?}: {:?}",
                        required, normalized
                    ));
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn has_explicit_selection_env() -> bool {
    [
        "CONVERGE_LLM_PROFILE",
        "CONVERGE_LLM_JURISDICTION",
        "CONVERGE_LLM_LATENCY",
        "CONVERGE_LLM_COST",
        "CONVERGE_LLM_COMPLEXITY",
        "CONVERGE_LLM_COMPLIANCE",
        "CONVERGE_LLM_TOOL_USE",
        "CONVERGE_LLM_VISION",
        "CONVERGE_LLM_STRUCTURED_OUTPUT",
        "CONVERGE_LLM_CODE",
        "CONVERGE_LLM_MULTILINGUAL",
        "CONVERGE_LLM_WEB_SEARCH",
        "CONVERGE_LLM_CONTEXT_TOKENS",
        "CONVERGE_LLM_USER_COUNTRY",
        "CONVERGE_LLM_USER_REGION",
    ]
    .into_iter()
    .any(|key| std::env::var_os(key).is_some())
}

fn selection_config_from_args() -> Result<ChatBackendSelectionConfig, Box<dyn Error>> {
    let mut config = ChatBackendSelectionConfig::from_env()?;
    if !has_explicit_selection_env() {
        config.criteria = SelectionCriteria::interactive()
            .with_capabilities(RequiredCapabilities::none().with_min_context(8_000));
    }

    if let Some(provider) = std::env::args().nth(1) {
        config = config.with_provider_override(provider);
    }

    Ok(config)
}

async fn run_live_chat(selected: SelectedChatBackend) -> Result<(), Box<dyn Error>> {
    let tuning = tuning_for(selected.provider());
    let mut messages = Vec::new();
    let mut company_state = None;

    println!(
        "provider: {}\nmodel: {}\nfitness: {:.3}\n",
        selected.provider(),
        selected.model(),
        selected.selection.fitness.total
    );
    println!("system: {}\n", tuning.system_prompt);

    for (idx, turn) in scenario_turns().into_iter().enumerate() {
        println!("user {}: {}\n", idx + 1, turn);
        messages.push(ChatMessage {
            role: ChatRole::User,
            content: turn.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });

        let response = selected
            .backend
            .chat(ChatRequest {
                messages: messages.clone(),
                system: Some(tuning.system_prompt.clone()),
                tools: Vec::new(),
                response_format: ResponseFormat::Text,
                max_tokens: Some(tuning.max_tokens),
                temperature: Some(tuning.temperature),
                stop_sequences: Vec::new(),
                model: None,
            })
            .await?;

        println!("assistant {}:\n{}\n", idx + 1, response.content.trim());
        company_state = validate_turn(idx, &response.content, company_state.as_ref())
            .map_err(|error| format!("validation failed on turn {}: {}", idx + 1, error))?
            .or(company_state);

        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: response.content,
            tool_calls: response.tool_calls,
            tool_call_id: None,
        });
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();
    let config = selection_config_from_args()?;
    let selected = select_chat_backend(&config)?;
    run_live_chat(selected).await
}
