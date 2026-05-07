// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Matrix test: model × format × prompt through OpenRouter.
//!
//! Tests format compliance, token efficiency, latency, and estimated cost
//! across models and output formats. Results inform model selection and
//! format guidance in the Converge documentation.
//!
//! Usage:
//!   OPENROUTER_API_KEY=sk-or-v1-... cargo run --features openrouter --example format_matrix

use std::time::{Duration, Instant};

use converge_provider::{
    ChatBackend, ChatMessage, ChatRequest, ChatRole, ResponseFormat, TokenUsage,
};
use manifold::OpenRouterBackend;

const MODELS: &[Model] = &[
    Model {
        id: "anthropic/claude-sonnet-4",
        input_cost_per_m: 3.00,
        output_cost_per_m: 15.00,
    },
    Model {
        id: "anthropic/claude-haiku-4.5",
        input_cost_per_m: 0.80,
        output_cost_per_m: 4.00,
    },
    Model {
        id: "openai/gpt-4o",
        input_cost_per_m: 2.50,
        output_cost_per_m: 10.00,
    },
    Model {
        id: "openai/gpt-4o-mini",
        input_cost_per_m: 0.15,
        output_cost_per_m: 0.60,
    },
    Model {
        id: "google/gemini-2.5-flash",
        input_cost_per_m: 0.15,
        output_cost_per_m: 0.60,
    },
    Model {
        id: "meta-llama/llama-3.1-70b-instruct",
        input_cost_per_m: 0.52,
        output_cost_per_m: 0.75,
    },
    Model {
        id: "mistralai/mistral-large",
        input_cost_per_m: 2.00,
        output_cost_per_m: 6.00,
    },
];

const FORMATS: &[ResponseFormat] = &[
    ResponseFormat::Text,
    ResponseFormat::Markdown,
    ResponseFormat::Json,
    ResponseFormat::Yaml,
    ResponseFormat::Toml,
];

const MAX_PARALLEL: usize = 3;

struct Model {
    id: &'static str,
    input_cost_per_m: f64,  // $ per 1M input tokens
    output_cost_per_m: f64, // $ per 1M output tokens
}

impl Model {
    fn short_name(&self) -> &str {
        self.id.split('/').next_back().unwrap_or(self.id)
    }

    fn estimate_cost(&self, usage: &TokenUsage) -> f64 {
        let input = f64::from(usage.prompt_tokens) * self.input_cost_per_m / 1_000_000.0;
        let output = f64::from(usage.completion_tokens) * self.output_cost_per_m / 1_000_000.0;
        input + output
    }
}

fn format_name(f: ResponseFormat) -> &'static str {
    match f {
        ResponseFormat::Text => "text",
        ResponseFormat::Markdown => "md",
        ResponseFormat::Json => "json",
        ResponseFormat::Yaml => "yaml",
        ResponseFormat::Toml => "toml",
    }
}

// ============================================================================
// Test prompts — ranging from simple to complex nested structures
// ============================================================================

struct TestPrompt {
    name: &'static str,
    user_message: &'static str,
}

const PROMPTS: &[TestPrompt] = &[
    // Simple flat record
    TestPrompt {
        name: "vendor_profile",
        user_message: "Return a vendor profile with these fields: name (string), region (string), certifications (list of strings), annual_cost_usd (number), risk_score (number between 0 and 1). Use vendor name 'Acme AI' with region 'EU', certifications SOC2 and ISO27001, cost 45000, risk 0.3.",
    },
    // Nested record with arrays of objects
    TestPrompt {
        name: "decision_record",
        user_message: "Return a governance decision record with: decision_id (string), outcome (approve/reject/escalate), confidence (number 0-1), evidence (list of objects with source and finding fields), rationale (string). Use decision_id 'DEC-001', outcome 'approve', confidence 0.87, two pieces of evidence, and a brief rationale.",
    },
    // Multi-entity comparison (wider output)
    TestPrompt {
        name: "vendor_comparison",
        user_message: "Compare three AI vendors in a structured format. For each vendor include: name, region, strengths (list), weaknesses (list), compliance_score (0-1), monthly_cost_usd. Vendors: 1) Acme AI (EU, strong on privacy, weak on scale, compliance 0.92, $3800/mo), 2) Beta ML (US, strong on performance and tooling, weak on EU compliance, compliance 0.71, $5200/mo), 3) Gamma LLM (EU, strong on cost and multilingual, weak on documentation, compliance 0.88, $2100/mo).",
    },
    // Deeply nested with mixed types
    TestPrompt {
        name: "audit_trail",
        user_message: "Return an audit trail entry with: audit_id (string), timestamp (ISO 8601), actor (object with id, role, department), action (string), target (object with type, id, name), policy_checks (list of objects with policy_id, result pass/fail, and detail), outcome (string), metadata (object with ip_address, session_id, duration_ms). Use realistic sample data for a vendor approval action.",
    },
];

// ============================================================================
// Result types
// ============================================================================

struct CellResult {
    model_idx: usize,
    format: ResponseFormat,
    prompt: String,
    pass: bool,
    parse_error: Option<String>,
    latency: Duration,
    usage: Option<TokenUsage>,
    raw_content: String,
}

impl CellResult {
    fn output_tokens(&self) -> u32 {
        self.usage.as_ref().map_or(0, |u| u.completion_tokens)
    }

    fn input_tokens(&self) -> u32 {
        self.usage.as_ref().map_or(0, |u| u.prompt_tokens)
    }
}

// ============================================================================
// Validation
// ============================================================================

fn validate_format(content: &str, format: ResponseFormat) -> Result<(), String> {
    let trimmed = strip_code_fences(content);

    match format {
        ResponseFormat::Text => Ok(()),
        ResponseFormat::Markdown => {
            if trimmed.contains('#')
                || trimmed.contains("- ")
                || trimmed.contains("* ")
                || trimmed.contains("```")
                || trimmed.contains('|')
            {
                Ok(())
            } else {
                Err("No Markdown structure detected".to_string())
            }
        }
        ResponseFormat::Json => serde_json::from_str::<serde_json::Value>(trimmed)
            .map(|_| ())
            .map_err(|e| format!("Invalid JSON: {e}")),
        ResponseFormat::Yaml => serde_yaml::from_str::<serde_yaml::Value>(trimmed)
            .map(|_| ())
            .map_err(|e| format!("Invalid YAML: {e}")),
        ResponseFormat::Toml => toml::from_str::<toml::Value>(trimmed)
            .map(|_| ())
            .map_err(|e| format!("Invalid TOML: {e}")),
    }
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

// ============================================================================
// Runner
// ============================================================================

async fn run_cell(
    backend: &OpenRouterBackend,
    model_idx: usize,
    format: ResponseFormat,
    prompt: &TestPrompt,
) -> CellResult {
    let req = ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.user_message.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        system: None,
        tools: Vec::new(),
        response_format: format,
        max_tokens: Some(2048),
        temperature: Some(0.0),
        stop_sequences: Vec::new(),
        model: Some(MODELS[model_idx].id.to_string()),
    };

    let start = Instant::now();
    let result = backend.chat(req).await;
    let latency = start.elapsed();

    match result {
        Ok(response) => {
            let validation = validate_format(&response.content, format);
            CellResult {
                model_idx,
                format,
                prompt: prompt.name.to_string(),
                pass: validation.is_ok(),
                parse_error: validation.err(),
                latency,
                usage: response.usage,
                raw_content: response.content,
            }
        }
        Err(e) => CellResult {
            model_idx,
            format,
            prompt: prompt.name.to_string(),
            pass: false,
            parse_error: Some(format!("API error: {e}")),
            latency,
            usage: None,
            raw_content: String::new(),
        },
    }
}

// ============================================================================
// Output
// ============================================================================

fn print_results(results: &[CellResult]) {
    // ── Compliance matrix ──
    for prompt in PROMPTS {
        println!("\n## {}\n", prompt.name);

        print!("{:<30}", "Model");
        for f in FORMATS {
            print!("  {:>6}", format_name(*f));
        }
        println!("   {:>5}/{:<5}", "pass", "total");
        println!("{}", "-".repeat(30 + FORMATS.len() * 8 + 14));

        for (i, model) in MODELS.iter().enumerate() {
            print!("{:<30}", model.short_name());
            let mut pass_count = 0;
            let total = FORMATS.len();

            for f in FORMATS {
                let cell = results
                    .iter()
                    .find(|r| r.model_idx == i && r.format == *f && r.prompt == prompt.name);
                match cell {
                    Some(c) if c.pass => {
                        print!("    \x1b[32m OK\x1b[0m");
                        pass_count += 1;
                    }
                    Some(_) => print!("  \x1b[31mFAIL\x1b[0m"),
                    None => print!("     -"),
                }
            }
            println!("   {:>5}/{:<5}", pass_count, total);
        }
    }

    // ── Token efficiency by format (averaged across models and prompts) ──
    println!("\n## Token efficiency by format\n");
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10}",
        "Format", "Avg In", "Avg Out", "Avg Total", "vs JSON"
    );
    println!("{}", "-".repeat(52));

    let json_avg = avg_tokens(results, ResponseFormat::Json);

    for f in FORMATS {
        let cells: Vec<&CellResult> = results
            .iter()
            .filter(|r| r.format == *f && r.pass)
            .collect();
        if cells.is_empty() {
            continue;
        }
        let avg_in =
            cells.iter().map(|c| c.input_tokens()).sum::<u32>() as f64 / cells.len() as f64;
        let avg_out =
            cells.iter().map(|c| c.output_tokens()).sum::<u32>() as f64 / cells.len() as f64;
        let avg_total = avg_in + avg_out;
        let vs_json = if json_avg > 0.0 {
            format!("{:+.0}%", (avg_total - json_avg) / json_avg * 100.0)
        } else {
            "-".to_string()
        };
        println!(
            "{:<8} {:>10.0} {:>10.0} {:>10.0} {:>10}",
            format_name(*f),
            avg_in,
            avg_out,
            avg_total,
            vs_json,
        );
    }

    // ── Latency by model ──
    println!("\n## Average latency by model\n");
    println!("{:<30} {:>10} {:>10} {:>10}", "Model", "Avg", "Min", "Max");
    println!("{}", "-".repeat(62));

    for (i, model) in MODELS.iter().enumerate() {
        let cells: Vec<&CellResult> = results
            .iter()
            .filter(|r| r.model_idx == i && r.pass)
            .collect();
        if cells.is_empty() {
            println!("{:<30}       (no successful calls)", model.short_name());
            continue;
        }
        let avg = cells.iter().map(|c| c.latency).sum::<Duration>() / cells.len() as u32;
        let min = cells.iter().map(|c| c.latency).min().unwrap();
        let max = cells.iter().map(|c| c.latency).max().unwrap();
        println!(
            "{:<30} {:>8.1}s {:>8.1}s {:>8.1}s",
            model.short_name(),
            avg.as_secs_f64(),
            min.as_secs_f64(),
            max.as_secs_f64(),
        );
    }

    // ── Cost estimate per 1000 calls ──
    println!("\n## Estimated cost per 1000 calls (by model × format, using prompt averages)\n");
    print!("{:<30}", "Model");
    for f in FORMATS {
        print!("  {:>8}", format_name(*f));
    }
    println!();
    println!("{}", "-".repeat(30 + FORMATS.len() * 10));

    for (i, model) in MODELS.iter().enumerate() {
        print!("{:<30}", model.short_name());
        for f in FORMATS {
            let cells: Vec<&CellResult> = results
                .iter()
                .filter(|r| r.model_idx == i && r.format == *f && r.pass)
                .collect();
            if cells.is_empty() {
                print!("        -");
                continue;
            }
            let avg_cost: f64 = cells
                .iter()
                .filter_map(|c| c.usage.as_ref().map(|u| model.estimate_cost(u)))
                .sum::<f64>()
                / cells.len() as f64;
            print!("  ${:>6.2}", avg_cost * 1000.0);
        }
        println!();
    }

    // ── Failures ──
    let failures: Vec<&CellResult> = results.iter().filter(|r| !r.pass).collect();
    if !failures.is_empty() {
        println!("\n## Failures\n");
        for f in &failures {
            let model = &MODELS[f.model_idx];
            println!(
                "  {} × {} × {}",
                model.short_name(),
                format_name(f.format),
                f.prompt,
            );
            if let Some(err) = &f.parse_error {
                println!("    Error: {err}");
            }
            if !f.raw_content.is_empty() {
                let preview: String = f.raw_content.chars().take(150).collect();
                println!("    Output: {preview}...");
            }
            println!();
        }
    }

    // ── Summary ──
    let total = results.len();
    let passed = results.iter().filter(|r| r.pass).count();
    let total_tokens: u32 = results
        .iter()
        .filter_map(|r| r.usage.as_ref())
        .map(|t| t.total_tokens)
        .sum();
    let total_cost: f64 = results
        .iter()
        .filter_map(|r| {
            r.usage
                .as_ref()
                .map(|u| MODELS[r.model_idx].estimate_cost(u))
        })
        .sum();
    println!(
        "\n## Summary: {passed}/{total} passed | {total_tokens} tokens | ${total_cost:.4} total cost\n"
    );
}

fn avg_tokens(results: &[CellResult], format: ResponseFormat) -> f64 {
    let cells: Vec<&CellResult> = results
        .iter()
        .filter(|r| r.format == format && r.pass)
        .collect();
    if cells.is_empty() {
        return 0.0;
    }
    cells
        .iter()
        .map(|c| (c.input_tokens() + c.output_tokens()) as f64)
        .sum::<f64>()
        / cells.len() as f64
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let backend = std::sync::Arc::new(OpenRouterBackend::from_env()?);
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL));

    let total_cells = MODELS.len() * FORMATS.len() * PROMPTS.len();
    println!("# Format Matrix Test");
    println!();
    println!(
        "Models: {}  Formats: {}  Prompts: {}  Total: {} cells",
        MODELS.len(),
        FORMATS.len(),
        PROMPTS.len(),
        total_cells,
    );

    let mut handles = Vec::new();

    for (model_idx, _model) in MODELS.iter().enumerate() {
        for format in FORMATS {
            for prompt in PROMPTS {
                let sem = semaphore.clone();
                let backend = backend.clone();
                let format = *format;
                let prompt_name = prompt.name;
                let prompt_msg = prompt.user_message;

                handles.push(tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();

                    eprintln!(
                        "  Running: {} × {} × {}",
                        MODELS[model_idx].short_name(),
                        format_name(format),
                        prompt_name,
                    );

                    let prompt = TestPrompt {
                        name: prompt_name,
                        user_message: prompt_msg,
                    };

                    run_cell(&backend, model_idx, format, &prompt).await
                }));
            }
        }
    }

    let mut results = Vec::with_capacity(total_cells);
    for handle in handles {
        results.push(handle.await?);
    }

    results.sort_by(|a, b| {
        a.prompt
            .cmp(&b.prompt)
            .then(a.model_idx.cmp(&b.model_idx))
            .then(format_name(a.format).cmp(format_name(b.format)))
    });

    print_results(&results);

    Ok(())
}
