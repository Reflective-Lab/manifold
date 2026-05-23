// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Live streaming demo: real OpenRouter call, tokens printed as they arrive.
//!
//! Run with:
//!   cargo run --example streaming_demo --features "openrouter streaming"
//!
//! Requires `OPENROUTER_API_KEY` in the env (Keychain-backed via direnv).

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use futures::StreamExt;
use manifold::{ChatEvent, OpenRouterBackend, StreamingChatBackend};

use converge_provider::{ChatMessage, ChatRequest, ChatRole, ResponseFormat};

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    if std::env::var_os("OPENROUTER_API_KEY").is_none() {
        eprintln!("error: OPENROUTER_API_KEY is not set");
        return ExitCode::FAILURE;
    }

    let backend = match OpenRouterBackend::from_env() {
        Ok(b) => b.with_max_retries(0),
        Err(e) => {
            eprintln!("error: backend construction failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let prompt =
        "Write a 5-line haiku-style verse about an LLM streaming tokens to a terminal. \
         Output the verse only, one line per stanza-line. No preamble.";
    let request = ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        system: Some(
            "You are a poet. Write evocatively. Reply with only the verse, no preamble."
                .to_string(),
        ),
        tools: Vec::new(),
        response_format: ResponseFormat::Text,
        max_tokens: Some(200),
        temperature: Some(0.7),
        stop_sequences: Vec::new(),
        model: None,
    };

    println!("\n=== OpenRouter streaming demo ===");
    println!("model: anthropic/claude-sonnet-4 (OpenRouter backend default)");
    println!("prompt: {prompt}\n");
    println!("--- live tokens (each `▌` = one delta arriving) ---");
    let _ = std::io::stdout().flush();

    let start = Instant::now();
    let mut stream = match backend.chat_stream(request).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("\nerror: opening stream failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut deltas = 0usize;
    let mut first_byte_at: Option<u128> = None;
    let mut total_chars = 0usize;
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<converge_provider::TokenUsage> = None;

    while let Some(event) = stream.next().await {
        match event {
            Ok(ChatEvent::TextDelta(s)) => {
                if first_byte_at.is_none() {
                    first_byte_at = Some(start.elapsed().as_millis());
                }
                deltas += 1;
                total_chars += s.len();
                // Print a marker between deltas, then the content. Flush so
                // the terminal renders each chunk as it arrives.
                print!("\x1b[2m▌\x1b[0m{s}");
                let _ = std::io::stdout().flush();
            }
            Ok(ChatEvent::Finish(reason)) => {
                finish_reason = Some(format!("{reason:?}"));
            }
            Ok(ChatEvent::Usage(u)) => {
                usage = Some(u);
            }
            Ok(other) => {
                eprintln!("\n[other event: {other:?}]");
            }
            Err(e) => {
                eprintln!("\nerror: stream event failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let total_ms = start.elapsed().as_millis();
    println!("\n");
    println!("--- end of stream ---");
    println!("first delta at: {:>6} ms", first_byte_at.unwrap_or(0));
    println!("total elapsed:  {total_ms:>6} ms");
    println!("delta count:    {deltas:>6}");
    println!("total chars:    {total_chars:>6}");
    if let Some(r) = finish_reason {
        println!("finish reason:  {r}");
    }
    if let Some(u) = usage {
        println!(
            "usage:          prompt={} completion={} total={} tokens",
            u.prompt_tokens, u.completion_tokens, u.total_tokens
        );
        if u.completion_tokens > 0 && total_ms > 0 {
            let tps = (u.completion_tokens as f64) * 1000.0 / (total_ms as f64);
            println!("throughput:     {tps:>6.1} completion tokens/sec");
        }
    }

    ExitCode::SUCCESS
}
