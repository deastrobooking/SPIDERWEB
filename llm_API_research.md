I'll provide you with a comprehensive guide to making LLM (Large Language Model) calls from Rust web frameworks like Actix and Axum. Here's how to integrate various LLM APIs into your Rust applications.

## Setup and Dependencies

First, add these dependencies to your `Cargo.toml`:

```toml
[dependencies]
# Web framework (choose one)
actix-web = "4.4"
# or
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json"] }
# For OpenAI API
openai = "0.1"  # or use reqwest directly
# For Anthropic Claude
claude-rs = "0.1"  # if available, or use reqwest
```

## Common LLM Client Trait

Let's create a generic LLM client trait that works with both frameworks:

```rust
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMResponse {
    pub text: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[async_trait::async_trait]
pub trait LLMClient {
    async fn generate(&self, request: &LLMRequest) -> Result<LLMResponse, Box<dyn Error>>;
}
```

## OpenAI Client Implementation

```rust
use reqwest::Client;
use serde_json::json;

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl LLMClient for OpenAIClient {
    async fn generate(&self, request: &LLMRequest) -> Result<LLMResponse, Box<dyn Error>> {
        let url = format!("{}/chat/completions", self.base_url);
        
        let body = json!({
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "max_tokens": request.max_tokens.unwrap_or(1000),
            "temperature": request.temperature.unwrap_or(0.7)
        });

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("API error: {}", error_text).into());
        }

        let json_response: serde_json::Value = response.json().await?;
        
        let text = json_response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let usage = if let Some(usage_obj) = json_response["usage"].as_object() {
            Some(Usage {
                prompt_tokens: usage_obj["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: usage_obj["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: usage_obj["total_tokens"].as_u64().unwrap_or(0) as u32,
            })
        } else {
            None
        };

        Ok(LLMResponse { text, usage })
    }
}
```

## Anthropic Claude Client

```rust
pub struct ClaudeClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl ClaudeClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.anthropic.com/v1".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl LLMClient for ClaudeClient {
    async fn generate(&self, request: &LLMRequest) -> Result<LLMResponse, Box<dyn Error>> {
        let url = format!("{}/messages", self.base_url);
        
        let body = json!({
            "model": "claude-3-sonnet-20240229",
            "max_tokens": request.max_tokens.unwrap_or(1000),
            "temperature": request.temperature.unwrap_or(0.7),
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        });

        let response = self.client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("API error: {}", error_text).into());
        }

        let json_response: serde_json::Value = response.json().await?;
        
        let text = json_response["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let usage = if let Some(usage_obj) = json_response["usage"].as_object() {
            Some(Usage {
                prompt_tokens: usage_obj["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: usage_obj["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: 0, // Claude doesn't provide total
            })
        } else {
            None
        };

        Ok(LLMResponse { text, usage })
    }
}
```

## Actix Web Implementation

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result};
use std::sync::Arc;

// Application state
pub struct AppState {
    llm_client: Arc<dyn LLMClient + Send + Sync>,
}

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub usage: Option<Usage>,
    pub error: Option<String>,
}

async fn generate_text(
    data: web::Data<AppState>,
    request: web::Json<GenerateRequest>,
) -> Result<HttpResponse> {
    let llm_request = LLMRequest {
        prompt: request.prompt.clone(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
    };

    match data.llm_client.generate(&llm_request).await {
        Ok(response) => Ok(HttpResponse::Ok().json(GenerateResponse {
            text: response.text,
            usage: response.usage,
            error: None,
        })),
        Err(e) => Ok(HttpResponse::InternalServerError().json(GenerateResponse {
            text: String::new(),
            usage: None,
            error: Some(e.to_string()),
        })),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize LLM client
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set");
    
    let llm_client: Arc<dyn LLMClient + Send + Sync> = 
        Arc::new(OpenAIClient::new(api_key));
    
    // Or use Claude:
    // let llm_client: Arc<dyn LLMClient + Send + Sync> = 
    //     Arc::new(ClaudeClient::new(api_key));

    let app_state = web::Data::new(AppState { llm_client });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/generate", web::post().to(generate_text))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

## Axum Implementation

```rust
use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

async fn generate_text(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let llm_request = LLMRequest {
        prompt: request.prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
    };

    match state.llm_client.generate(&llm_request).await {
        Ok(response) => Json(GenerateResponse {
            text: response.text,
            usage: response.usage,
            error: None,
        }),
        Err(e) => Json(GenerateResponse {
            text: String::new(),
            usage: None,
            error: Some(e.to_string()),
        }),
    }
}

#[tokio::main]
async fn main() {
    // Initialize LLM client
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set");
    
    let llm_client: Arc<dyn LLMClient + Send + Sync> = 
        Arc::new(OpenAIClient::new(api_key));

    let app_state = Arc::new(AppState { llm_client });

    let app = Router::new()
        .route("/generate", post(generate_text))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    
    println!("Server running on http://0.0.0.0:3000");
    
    axum::serve(listener, app).await.unwrap();
}
```

## Advanced Features

### 1. Streaming Responses

```rust
use futures::StreamExt;
use tokio_stream::Stream;

pub struct StreamingLLMClient {
    // Similar to regular client but with streaming support
}

impl StreamingLLMClient {
    pub async fn generate_stream(
        &self,
        request: &LLMRequest,
    ) -> Result<impl Stream<Item = Result<String, Box<dyn Error>>>, Box<dyn Error>> {
        // Implementation for streaming responses
        // This would use Server-Sent Events (SSE) for web frameworks
        unimplemented!()
    }
}
```

### 2. Rate Limiting and Retries

```rust
use std::time::Duration;
use tokio::time::sleep;

impl OpenAIClient {
    pub async fn generate_with_retry(
        &self,
        request: &LLMRequest,
        max_retries: u32,
    ) -> Result<LLMResponse, Box<dyn Error>> {
        for attempt in 0..=max_retries {
            match self.generate(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if attempt == max_retries {
                        return Err(e);
                    }
                    // Exponential backoff
                    sleep(Duration::from_secs(2u64.pow(attempt))).await;
                }
            }
        }
        unreachable!()
    }
}
```

### 3. Configuration Management

```rust
use config::Config;

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout: Duration,
}

impl LLMConfig {
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let settings = Config::builder()
            .add_source(config::Environment::default())
            .build()?;

        Ok(Self {
            api_key: settings.get_string("LLM_API_KEY")?,
            base_url: settings.get_string("LLM_BASE_URL").unwrap_or_default(),
            model: settings.get_string("LLM_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string()),
            max_tokens: settings.get_int("LLM_MAX_TOKENS").unwrap_or(1000) as u32,
            temperature: settings.get_float("LLM_TEMPERATURE").unwrap_or(0.7) as f32,
            timeout: Duration::from_secs(settings.get_int("LLM_TIMEOUT").unwrap_or(30) as u64),
        })
    }
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockLLMClient;

    #[async_trait::async_trait]
    impl LLMClient for MockLLMClient {
        async fn generate(&self, request: &LLMRequest) -> Result<LLMResponse, Box<dyn Error>> {
            Ok(LLMResponse {
                text: format!("Mock response to: {}", request.prompt),
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                }),
            })
        }
    }

    #[tokio::test]
    async fn test_llm_generate() {
        let client = MockLLMClient;
        let request = LLMRequest {
            prompt: "Hello".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.5),
        };

        let response = client.generate(&request).await.unwrap();
        assert!(response.text.contains("Mock response"));
    }
}
```

## Key Considerations

1. **Error Handling**: Always handle API errors gracefully
2. **Timeouts**: Set appropriate timeouts for LLM API calls
3. **Rate Limiting**: Implement rate limiting to avoid hitting API limits
4. **Security**: Keep API keys secure using environment variables
5. **Async**: Use async/await properly for non-blocking operations
6. **State Management**: Share LLM clients efficiently across requests
7. **Logging**: Implement proper logging for debugging and monitoring

This is a solid, idiomatic foundation. Below are the fixes and “2025-ready” upgrades I’d make so this guide drops straight into production projects (Actix or Axum) and handles OpenAI **Responses API** + Anthropic **Messages API**, streaming, JSON/structured outputs, retries, and observability.

---

# Quick fixes (in your current code)

* **Missing deps/imports**: you use `async_trait`, `CorsLayer`, `tokio_stream::Stream`, etc.
* **Model/endpoint drift**: `gpt-3.5-turbo` & `/v1/chat/completions` still work, but OpenAI’s *Responses API* is the current path forward (Assistants sunsets Aug 2026; Chat Completions continues but new features land in Responses). ([OpenAI Developer Community][1])
* **Anthropic models**: default to **Claude 3.7 Sonnet** and `/v1/messages`; keep `anthropic-version: 2023-06-01`. Streaming is supported. ([Google Cloud Documentation][2])

### Minimal dependency patch (Cargo.toml)

```toml
[dependencies]
# choose one web framework
actix-web = "4.5"    # or latest 4.x
# or
axum = "0.8"         # 0.8.x
tokio = { version = "1.48", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# HTTP + streaming
reqwest = { version = "0.12", default-features = false, features = ["json", "stream", "rustls-tls"] }
tokio-stream = "0.1"
reqwest-eventsource = "0.6"      # SSE client for streaming

# ergonomics & config
async-trait = "0.1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
config = "0.14"
dotenvy = "0.15"

# CORS for Axum
tower-http = { version = "0.6", features = ["cors", "trace", "compression-full"] }
```

(Reqwest 0.12.x and Tokio 1.4x are current; Axum 0.8.x is the 2025 line.) ([crates.io][3])

---

# Trait + error type

Prefer a concrete error enum for clean handling and testability:

```rust
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub model: Option<String>, // allow override per-call
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMResponse {
    pub text: String,
    pub usage: Option<Usage>,
    pub raw: serde_json::Value, // keep raw for tool-use, debugging
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api: {0}")]
    Api(String),
    #[error("parse: {0}")]
    Parse(String),
}

#[async_trait::async_trait]
pub trait LLMClient: Send + Sync {
    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LlmError>;
}
```

---

# OpenAI: prefer the **Responses API** (with a Chat Completions fallback)

**Why**: New features (built-in tools, structured outputs, agents) land in Responses. If you must keep `/chat/completions`, retain it behind a feature flag. ([OpenAI Platform][4])

```rust
use reqwest::Client;
use serde_json::json;

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String, // allow Azure/vLLM override
    default_model: String,
}

impl OpenAIClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(4))
                .timeout(std::time::Duration::from_secs(60))
                .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
                .build()
                .expect("build client"),
            api_key,
            base_url: "https://api.openai.com/v1".into(),
            default_model: "gpt-5".into(), // pick your default
        }
    }
}

#[async_trait::async_trait]
impl LLMClient for OpenAIClient {
    async fn generate(&self, r: &LLMRequest) -> Result<LLMResponse, LlmError> {
        let url = format!("{}/responses", self.base_url);
        let model = r.model.clone().unwrap_or_else(|| self.default_model.clone());

        // Use Responses API with JSON output enabled (optional)
        let body = json!({
            "model": model,
            "input": r.prompt,
            "max_output_tokens": r.max_tokens.unwrap_or(1000),
            "temperature": r.temperature.unwrap_or(0.7),
            // Choose **one** of the following depending on your use case:
            // "response_format": { "type": "json_object" },            // JSON mode
            // or for strict schemas:
            // "response_format": {
            //   "type": "json_schema",
            //   "json_schema": { "name": "MySchema", "schema": { "type": "object", "properties": { "answer": { "type": "string" } }, "required": ["answer"], "additionalProperties": false } }
            // }
        });

        let resp = self.client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(LlmError::Api(resp.text().await?));
        }

        let v: serde_json::Value = resp.json().await?;
        // Responses API shape: consolidate any `output_text` helpers if present
        let text = v.pointer("/output_text")
            .and_then(|x| x.as_str())
            .or_else(|| v.pointer("/output/0/content/0/text").and_then(|x| x.as_str()))
            .unwrap_or("")
            .to_string();

        let usage = v.get("usage").and_then(|u| {
            Some(Usage {
                prompt_tokens: u.get("input_tokens")?.as_u64().unwrap_or(0) as u32,
                completion_tokens: u.get("output_tokens")?.as_u64().unwrap_or(0) as u32,
                total_tokens: u.get("total_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32,
            })
        });

        Ok(LLMResponse { text, usage, raw: v })
    }
}
```

* **Structured outputs**: prefer JSON Schema (fail-closed) over “best-effort JSON mode” for production. ([OpenAI Platform][5])
* **Assistants → Responses** migration background (for context). ([OpenAI Developer Community][1])

> If you need to keep the *Chat Completions* version, your original implementation is fine—just swap the model default and keep an eye on roadmap notes. ([Stack Overflow][6])

---

# Anthropic (Claude) client + streaming

**Default model**: `claude-3-7-sonnet-latest`. Endpoint: `POST https://api.anthropic.com/v1/messages`, headers `x-api-key` + `anthropic-version: 2023-06-01`. ([Google Cloud Documentation][2])

```rust
use reqwest::Client;
use serde_json::json;

pub struct ClaudeClient {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl ClaudeClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.anthropic.com/v1".into(),
            default_model: "claude-3-7-sonnet-latest".into(),
        }
    }
}

#[async_trait::async_trait]
impl LLMClient for ClaudeClient {
    async fn generate(&self, r: &LLMRequest) -> Result<LLMResponse, LlmError> {
        let url = format!("{}/messages", self.base_url);
        let model = r.model.clone().unwrap_or_else(|| self.default_model.clone());

        let body = json!({
            "model": model,
            "max_tokens": r.max_tokens.unwrap_or(1000),
            "temperature": r.temperature.unwrap_or(0.7),
            "messages": [{ "role": "user", "content": r.prompt }],
        });

        let resp = self.client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(LlmError::Api(resp.text().await?));
        }

        let v: serde_json::Value = resp.json().await?;
        let text = v.pointer("/content/0/text")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();

        let usage = v.get("usage").map(|u| Usage {
            prompt_tokens: u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            completion_tokens: u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            total_tokens: 0, // not provided
        });

        Ok(LLMResponse { text, usage, raw: v })
    }
}
```

### Streaming (SSE) with `reqwest-eventsource`

```rust
use futures::{StreamExt, TryStreamExt};
use reqwest_eventsource::{Event, EventSource};

pub async fn anthropic_stream(client: &ClaudeClient, r: &LLMRequest) -> Result<(), LlmError> {
    let url = format!("{}/messages", client.base_url);
    let body = serde_json::json!({
        "model": client.default_model,
        "max_tokens": r.max_tokens.unwrap_or(1000),
        "messages": [{ "role": "user", "content": r.prompt }],
        "stream": true
    });

    let builder = client.client
        .post(&url)
        .header("x-api-key", &client.api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body);

    let mut es = EventSource::new(builder).map_err(|e| LlmError::Http(e.into()))?;
    while let Some(event) = es.next().await {
        match event {
            Ok(Event::Message(msg)) => {
                if msg.data == "[DONE]" { break; }
                // Each event is a JSON line; extract delta text fields
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg.data) {
                    if let Some(t) = v.pointer("/delta/text").and_then(|x| x.as_str()) {
                        // stream t to client (SSE/WebSocket/etc.)
                    }
                }
            }
            Ok(Event::Open) => {}
            Err(e) => { es.close(); return Err(LlmError::Http(e.into())); }
        }
    }
    Ok(())
}
```

(Anthropic 3.7 Sonnet and streaming notes.) ([Google Cloud Documentation][2])
(Using `reqwest-eventsource` for SSE.) ([Docs.rs][7])

---

# Actix / Axum streaming endpoints (SSE)

* **Actix**: `HttpResponse::Ok().insert_header(("Content-Type","text/event-stream"))` and write lines `data: ...\n\n`.
* **Axum**: return `Sse<impl Stream<Item = Result<Event, Infallible>>>` from `axum::response::sse`.

(Framework versions & docs for 2025 lines.) ([actix.rs][8])

---

# JSON/Structured outputs (OpenAI)

Prefer **Structured Outputs** with a JSON Schema for fail-closed parsing:

* Set `response_format: { "type": "json_schema", "json_schema": { ... } }`.
* The model guarantees the schema or returns an error, which you map to `LlmError::Parse`. ([OpenAI Platform][5])

---

# Robust retries + rate limits

* Keep your exponential backoff; also retry **429**, **408**, **5xx**, and network timeouts with jitter.
* For Axum, consider `tower-governor` for per-route rate limiting and `tower-http`’s `TraceLayer` for visibility. ([github.com][9])

---

# Config, multi-provider, and observability

* Configurable `base_url` lets you point to **Azure OpenAI** or a self-hosted **vLLM**/**Ollama** without code changes. (Azure Responses API doc.) ([Microsoft Learn][10])
* Add `tracing_subscriber::fmt().with_env_filter(...)` and log request IDs + provider/model.
* Keep keys in env (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), load via `dotenvy`.

---

# Testing

* Mock HTTP with `httpmock` or `wiremock-rs` and assert on your typed `LLMResponse`.
* Your `MockLLMClient` is great—keep it for pure unit tests.

---

## Drop-in patches you can paste

**1) Add `model` to your request and thread it through handlers**

```rust
#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub model: Option<String>,
}
```

**2) Axum handler stays the same, but forwards `model`**
*(Same for Actix—no framework-specific changes required.)*

**3) Replace OpenAI Chat-only code with the Responses version above**
(Keep your Chat version behind a feature flag if you need both.)

---

# Why these updates?

* **Responses API** is where OpenAI ships new capabilities (built-in tools, structured outputs, agents); Chat continues, but you’ll miss newer features. ([OpenAI Platform][4])
* **Claude 3.7 Sonnet** is Anthropic’s current top general model; your code now defaults sensibly and supports streaming out-of-the-box. ([Google Cloud Documentation][2])
* **Crate versions** track the current 2025 ecosystem (reqwest 0.12.x, Tokio 1.4x, Axum 0.8.x, tower-http 0.6.x). ([crates.io][3])

If you want, I can add:

* a **tool/function-calling loop** (OpenAI Responses + Anthropic tool_use/tool_result interop),
* **SSE server handlers** for both Actix and Axum,
* a tiny **provider switch** (`OPENAI|ANTHROPIC|AZURE|OLLAMA`) with a single `dyn LLMClient` behind it.

[1]: https://community.openai.com/t/migration-guide-for-assistants-api-to-responses-api-is-now-available/1354626?utm_source=chatgpt.com "Migration Guide for Assistants API to Responses ..."
[2]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude?utm_source=chatgpt.com "Anthropic's Claude models | Generative AI on Vertex AI"
[3]: https://crates.io/crates/reqwest/versions?utm_source=chatgpt.com "reqwest"
[4]: https://platform.openai.com/docs/guides/migrate-to-responses?utm_source=chatgpt.com "Migrate to the Responses API"
[5]: https://platform.openai.com/docs/guides/structured-outputs?utm_source=chatgpt.com "Structured model outputs - OpenAI API"
[6]: https://stackoverflow.com/questions/75041247/whats-the-correct-url-to-test-openai-api?utm_source=chatgpt.com "curl - What's the correct URL to test OpenAI API?"
[7]: https://docs.rs/reqwest-eventsource/?utm_source=chatgpt.com "reqwest_eventsource - Rust"
[8]: https://actix.rs/?utm_source=chatgpt.com "Actix Web"
[9]: https://github.com/benwis/tower-governor?utm_source=chatgpt.com "benwis/tower-governor: Rate Limiting middleware for ..."
[10]: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses?utm_source=chatgpt.com "Azure OpenAI Responses API"
