// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::collections::HashSet;

use converge_provider::{ChatRequest, ChatResponse, ChatRole, LlmError, ResponseFormat};

const REQUEST_LANGUAGE_KEY: &str = "request.language";
const RESPONSE_LANGUAGE_KEY: &str = "response.language";

pub(super) fn finalize_chat_response(
    request: &ChatRequest,
    mut response: ChatResponse,
) -> Result<ChatResponse, LlmError> {
    let requested_format = request.response_format;
    if response.tool_calls.is_empty() {
        response.content = normalize_content(&response.content, requested_format)?;
    }
    annotate_language_metadata(request, requested_format, &mut response);
    Ok(response)
}

fn annotate_language_metadata(
    request: &ChatRequest,
    requested_format: ResponseFormat,
    response: &mut ChatResponse,
) {
    if !response.metadata.contains_key(REQUEST_LANGUAGE_KEY) {
        if let Some(language) = detect_request_language(request) {
            response
                .metadata
                .insert(REQUEST_LANGUAGE_KEY.to_string(), language);
        }
    }

    if matches!(
        requested_format,
        ResponseFormat::Text | ResponseFormat::Markdown
    ) && !response.metadata.contains_key(RESPONSE_LANGUAGE_KEY)
    {
        if let Some(language) = detect_language(&response.content) {
            response
                .metadata
                .insert(RESPONSE_LANGUAGE_KEY.to_string(), language);
        }
    }
}

fn detect_request_language(request: &ChatRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == ChatRole::User && !message.content.trim().is_empty())
        .and_then(|message| detect_language(&message.content))
        .or_else(|| request.system.as_deref().and_then(detect_language))
        .or_else(|| {
            request
                .messages
                .iter()
                .rev()
                .find(|message| !message.content.trim().is_empty())
                .and_then(|message| detect_language(&message.content))
        })
}

fn detect_language(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut has_latin = false;
    let mut has_cyrillic = false;
    let mut has_han = false;
    let mut has_arabic = false;
    let mut has_devanagari = false;

    for ch in trimmed.chars() {
        if is_hiragana_or_katakana(ch) {
            return Some("ja".to_string());
        }
        if is_hangul(ch) {
            return Some("ko".to_string());
        }
        if is_han(ch) {
            has_han = true;
        }
        if is_cyrillic(ch) {
            has_cyrillic = true;
        }
        if is_arabic(ch) {
            has_arabic = true;
        }
        if is_devanagari(ch) {
            has_devanagari = true;
        }
        if ch.is_ascii_alphabetic()
            || matches!(
                ch,
                'À'..='Ö' | 'Ø'..='ö' | 'ø'..='ÿ' | 'Ā'..='ſ' | 'ƀ'..='ɏ'
            )
        {
            has_latin = true;
        }
    }

    if has_arabic {
        return Some("ar".to_string());
    }
    if has_devanagari {
        return Some("hi".to_string());
    }
    if has_han {
        return Some("zh".to_string());
    }
    if has_cyrillic {
        return Some("und-Cyrl".to_string());
    }
    if has_latin {
        return Some(detect_latin_language(trimmed));
    }

    Some("und".to_string())
}

fn detect_latin_language(content: &str) -> String {
    let lower = content.to_lowercase();
    let tokens: HashSet<&str> = lower
        .split(|ch: char| !ch.is_alphabetic())
        .filter(|token| !token.is_empty())
        .collect();

    if lower.contains(['å', 'ä', 'ö'])
        || count_hits(&tokens, &["och", "att", "är", "för", "med", "som"]) >= 2
    {
        return "sv".to_string();
    }

    if count_hits(
        &tokens,
        &[
            "the",
            "and",
            "with",
            "from",
            "that",
            "this",
            "return",
            "please",
            "reply",
            "summarize",
            "exactly",
            "keep",
            "only",
        ],
    ) >= 2
    {
        return "en".to_string();
    }

    "und-Latn".to_string()
}

fn count_hits(tokens: &HashSet<&str>, candidates: &[&str]) -> usize {
    candidates
        .iter()
        .filter(|candidate| tokens.contains(**candidate))
        .count()
}

fn is_hiragana_or_katakana(ch: char) -> bool {
    matches!(ch, '\u{3040}'..='\u{30ff}' | '\u{31f0}'..='\u{31ff}')
}

fn is_hangul(ch: char) -> bool {
    matches!(
        ch,
        '\u{1100}'..='\u{11ff}' | '\u{3130}'..='\u{318f}' | '\u{ac00}'..='\u{d7af}'
    )
}

fn is_han(ch: char) -> bool {
    matches!(
        ch,
        '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}'
    )
}

fn is_cyrillic(ch: char) -> bool {
    matches!(ch, '\u{0400}'..='\u{04ff}' | '\u{0500}'..='\u{052f}')
}

fn is_arabic(ch: char) -> bool {
    matches!(
        ch,
        '\u{0600}'..='\u{06ff}' | '\u{0750}'..='\u{077f}' | '\u{08a0}'..='\u{08ff}'
    )
}

fn is_devanagari(ch: char) -> bool {
    matches!(ch, '\u{0900}'..='\u{097f}')
}

fn normalize_content(content: &str, requested_format: ResponseFormat) -> Result<String, LlmError> {
    match requested_format {
        ResponseFormat::Text | ResponseFormat::Markdown => Ok(content.to_string()),
        ResponseFormat::Json => validate_json(content),
        ResponseFormat::Yaml => validate_yaml(content),
        ResponseFormat::Toml => validate_toml(content),
    }
}

fn validate_json(content: &str) -> Result<String, LlmError> {
    let normalized = normalized_candidate(content);
    let value = serde_json::from_str::<serde_json::Value>(normalized).map_err(|error| {
        format_mismatch(
            ResponseFormat::Json,
            format!("expected JSON object or array: {error}"),
            normalized,
        )
    })?;

    if value.is_object() || value.is_array() {
        return Ok(normalized.to_string());
    }

    Err(format_mismatch(
        ResponseFormat::Json,
        "expected JSON object or array".to_string(),
        normalized,
    ))
}

fn validate_yaml(content: &str) -> Result<String, LlmError> {
    let normalized = normalized_candidate(content);
    let value = serde_yaml::from_str::<serde_yaml::Value>(normalized).map_err(|error| {
        format_mismatch(
            ResponseFormat::Yaml,
            format!("expected YAML mapping or sequence: {error}"),
            normalized,
        )
    })?;

    if matches!(
        value,
        serde_yaml::Value::Mapping(_) | serde_yaml::Value::Sequence(_)
    ) {
        return Ok(normalized.to_string());
    }

    Err(format_mismatch(
        ResponseFormat::Yaml,
        "expected YAML mapping or sequence".to_string(),
        normalized,
    ))
}

fn validate_toml(content: &str) -> Result<String, LlmError> {
    let normalized = normalized_candidate(content);
    let value = toml::from_str::<toml::Value>(normalized).map_err(|error| {
        format_mismatch(
            ResponseFormat::Toml,
            format!("expected TOML document: {error}"),
            normalized,
        )
    })?;

    if matches!(value, toml::Value::Table(_) | toml::Value::Array(_)) {
        return Ok(normalized.to_string());
    }

    Err(format_mismatch(
        ResponseFormat::Toml,
        "expected TOML table or array".to_string(),
        normalized,
    ))
}

fn normalized_candidate(content: &str) -> &str {
    strip_code_fences(content).trim()
}

fn strip_code_fences(content: &str) -> &str {
    let trimmed = content.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(after_tag) = rest.find('\n') {
            let inner = &rest[after_tag + 1..];
            if let Some(end) = inner.rfind("```") {
                return inner[..end].trim();
            }
        }
    }
    trimmed
}

fn format_mismatch(expected: ResponseFormat, detail: String, content: &str) -> LlmError {
    let preview = preview(content);
    LlmError::ResponseFormatMismatch {
        expected,
        message: if preview.is_empty() {
            detail
        } else {
            format!("{detail}; response preview: {preview}")
        },
    }
}

fn preview(content: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 120;

    let trimmed = content.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut preview = String::new();
    for ch in trimmed.chars().take(MAX_PREVIEW_CHARS) {
        preview.push(ch);
    }
    if trimmed.chars().count() > MAX_PREVIEW_CHARS {
        preview.push_str("...");
    }
    preview
}

#[cfg(test)]
mod tests {
    use converge_core::traits::{
        ChatMessage, ChatRequest, ChatResponse, ChatRole, LlmError, ResponseFormat, ToolCall,
    };

    use super::finalize_chat_response;

    fn request(content: &str, response_format: ResponseFormat) -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: content.to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: Vec::new(),
            response_format,
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        }
    }

    fn response(content: &str) -> ChatResponse {
        ChatResponse {
            content: content.to_string(),
            tool_calls: Vec::new(),
            usage: None,
            model: None,
            finish_reason: None,
            metadata: Default::default(),
        }
    }

    #[test]
    fn strips_json_code_fences() {
        let response = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response("```json\n{\"facts\":[\"a\"]}\n```"),
        )
        .unwrap();

        assert_eq!(response.content, "{\"facts\":[\"a\"]}");
    }

    #[test]
    fn rejects_chatty_json_wrapper() {
        let error = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response("Here is the JSON you asked for:\n{\"facts\":[\"a\"]}"),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ResponseFormatMismatch {
                expected: ResponseFormat::Json,
                ..
            }
        ));
    }

    #[test]
    fn rejects_yaml_scalar() {
        let error = finalize_chat_response(
            &request("Return YAML.", ResponseFormat::Yaml),
            response("plain text reply"),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ResponseFormatMismatch {
                expected: ResponseFormat::Yaml,
                ..
            }
        ));
    }

    #[test]
    fn skips_validation_for_tool_calls() {
        let mut response = response("not json");
        response.tool_calls = vec![ToolCall {
            id: "call-1".to_string(),
            name: "lookup".to_string(),
            arguments: "{}".to_string(),
        }];

        let finalized =
            finalize_chat_response(&request("Return JSON.", ResponseFormat::Json), response)
                .unwrap();
        assert_eq!(finalized.content, "not json");
    }

    #[test]
    fn text_passthrough_unchanged() {
        let response = finalize_chat_response(
            &request("Hello world!", ResponseFormat::Text),
            response("Hello world!"),
        )
        .unwrap();

        assert_eq!(response.content, "Hello world!");
    }

    #[test]
    fn markdown_passthrough_unchanged() {
        let md = "# Header\n\n- bullet 1\n- bullet 2";
        let response = finalize_chat_response(
            &request("Write markdown.", ResponseFormat::Markdown),
            response(md),
        )
        .unwrap();
        assert_eq!(response.content, md);
    }

    #[test]
    fn valid_json_object_accepted() {
        let response = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response(r#"{"key": "value"}"#),
        )
        .unwrap();

        assert_eq!(response.content, r#"{"key": "value"}"#);
    }

    #[test]
    fn valid_json_array_accepted() {
        let response = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response(r"[1, 2, 3]"),
        )
        .unwrap();

        assert_eq!(response.content, "[1, 2, 3]");
    }

    #[test]
    fn json_scalar_rejected() {
        let err = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response(r#""just a string""#),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            LlmError::ResponseFormatMismatch {
                expected: ResponseFormat::Json,
                ..
            }
        ));
    }

    #[test]
    fn json_number_rejected() {
        let err = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response("42"),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            LlmError::ResponseFormatMismatch {
                expected: ResponseFormat::Json,
                ..
            }
        ));
    }

    #[test]
    fn empty_content_rejected_as_json() {
        let err =
            finalize_chat_response(&request("Return JSON.", ResponseFormat::Json), response(""))
                .unwrap_err();

        assert!(matches!(
            err,
            LlmError::ResponseFormatMismatch {
                expected: ResponseFormat::Json,
                ..
            }
        ));
    }

    #[test]
    fn strips_yaml_code_fences() {
        let yaml_fenced = "```yaml\nkey: value\nlist:\n  - a\n  - b\n```";
        let response = finalize_chat_response(
            &request("Return YAML.", ResponseFormat::Yaml),
            response(yaml_fenced),
        )
        .unwrap();

        assert!(response.content.contains("key: value"));
        assert!(!response.content.contains("```"));
    }

    #[test]
    fn valid_yaml_mapping_accepted() {
        let response = finalize_chat_response(
            &request("Return YAML.", ResponseFormat::Yaml),
            response("key: value\nother: 42"),
        )
        .unwrap();

        assert!(response.content.contains("key: value"));
    }

    #[test]
    fn valid_yaml_sequence_accepted() {
        let response = finalize_chat_response(
            &request("Return YAML.", ResponseFormat::Yaml),
            response("- item1\n- item2"),
        )
        .unwrap();

        assert!(response.content.contains("- item1"));
    }

    #[test]
    fn valid_toml_accepted() {
        let toml_content = "[section]\nkey = \"value\"\ncount = 42";
        let response = finalize_chat_response(
            &request("Return TOML.", ResponseFormat::Toml),
            response(toml_content),
        )
        .unwrap();

        assert!(response.content.contains("key = \"value\""));
    }

    #[test]
    fn toml_scalar_rejected() {
        let err = finalize_chat_response(
            &request("Return TOML.", ResponseFormat::Toml),
            response("just text"),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            LlmError::ResponseFormatMismatch {
                expected: ResponseFormat::Toml,
                ..
            }
        ));
    }

    #[test]
    fn strips_toml_code_fences() {
        let toml_fenced = "```toml\n[section]\nkey = \"value\"\n```";
        let response = finalize_chat_response(
            &request("Return TOML.", ResponseFormat::Toml),
            response(toml_fenced),
        )
        .unwrap();

        assert!(response.content.contains("key = \"value\""));
        assert!(!response.content.contains("```"));
    }

    #[test]
    fn whitespace_only_rejected_as_json() {
        let err = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response("   \n  \t  "),
        )
        .unwrap_err();

        assert!(matches!(err, LlmError::ResponseFormatMismatch { .. }));
    }

    #[test]
    fn json_with_leading_trailing_whitespace_accepted() {
        let response = finalize_chat_response(
            &request("Return JSON.", ResponseFormat::Json),
            response("  \n{\"key\": \"val\"}\n  "),
        )
        .unwrap();

        assert_eq!(response.content, r#"{"key": "val"}"#);
    }

    #[test]
    fn annotates_request_and_response_languages_for_text() {
        let response = finalize_chat_response(
            &request(
                "Summarize the aircraft in four bullet points.",
                ResponseFormat::Text,
            ),
            response("这是中文回复。"),
        )
        .unwrap();

        assert_eq!(
            response.metadata.get("request.language"),
            Some(&"en".to_string())
        );
        assert_eq!(
            response.metadata.get("response.language"),
            Some(&"zh".to_string())
        );
    }

    #[test]
    fn skips_response_language_for_structured_formats() {
        let response = finalize_chat_response(
            &request("Return JSON only.", ResponseFormat::Json),
            response(r#"{"status":"ok"}"#),
        )
        .unwrap();

        assert_eq!(
            response.metadata.get("request.language"),
            Some(&"en".to_string())
        );
        assert_eq!(response.metadata.get("response.language"), None);
    }
}
