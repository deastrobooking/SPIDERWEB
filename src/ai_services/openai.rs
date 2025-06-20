// OpenAI integration for model distillation and knowledge extraction
// The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
// Do not change this unless explicitly requested by the user

use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;

/// OpenAI service client for model distillation and knowledge extraction
#[derive(Clone)]
pub struct OpenAIService {
    client: Client,
    api_key: Option<String>,
    base_url: String,
}

impl OpenAIService {
    pub fn new() -> Self {
        let api_key = env::var("OPENAI_API_KEY").ok();
        
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key: Some(api_key),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    fn get_api_key(&self) -> Result<&String> {
        self.api_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."))
    }

    /// Generate synthetic training data using GPT models
    pub async fn generate_synthetic_data(
        &self,
        prompt: &str,
        count: usize,
        model: Option<&str>,
    ) -> Result<Vec<String>> {
        let api_key = self.get_api_key()?;
        let model = model.unwrap_or("gpt-4o");
        
        let request_body = json!({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a synthetic data generator. Generate diverse, high-quality training examples. Return each example as a separate JSON object on its own line."
                },
                {
                    "role": "user",
                    "content": format!("Generate {} diverse examples of: {}. Format each example as a JSON object with relevant fields.", count, prompt)
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.8
        });

        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                let examples: Vec<String> = content
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| line.trim().to_string())
                    .collect();
                    
                return Ok(examples);
            }
        }
        
        Err(anyhow::anyhow!("Failed to generate synthetic data"))
    }

    /// Extract knowledge from existing models using GPT analysis
    pub async fn extract_knowledge(
        &self,
        model_description: &str,
        training_data_sample: &[String],
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let prompt = format!(
            "Analyze this machine learning model and training data to extract key knowledge patterns:\n\
             Model: {}\n\
             Sample Data: {:?}\n\
             \n\
             Provide analysis in JSON format with keys: concepts, patterns, relationships, optimization_hints",
            model_description,
            training_data_sample.iter().take(10).collect::<Vec<_>>()
        );

        let request_body = json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert machine learning analyst. Extract knowledge patterns from models and data. Respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3,
            "response_format": { "type": "json_object" }
        });

        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                let knowledge: HashMap<String, Value> = serde_json::from_str(content)
                    .unwrap_or_else(|_| {
                        let mut map = HashMap::new();
                        map.insert("raw_analysis".to_string(), json!(content));
                        map
                    });
                return Ok(knowledge);
            }
        }
        
        Err(anyhow::anyhow!("Failed to extract knowledge"))
    }

    /// Perform model distillation using GPT as teacher
    pub async fn distill_model_knowledge(
        &self,
        teacher_outputs: &[f32],
        input_descriptions: &[String],
        target_model_type: &str,
    ) -> Result<Vec<(String, f32)>> {
        let api_key = self.get_api_key()?;
        
        let examples: Vec<String> = teacher_outputs
            .iter()
            .zip(input_descriptions.iter())
            .take(10) // Limit examples to avoid token limits
            .map(|(output, input)| format!("Input: {} -> Output: {}", input, output))
            .collect();

        let prompt = format!(
            "You are helping distill knowledge from a teacher model to a student model.\n\
             Target student model type: {}\n\
             Teacher model examples:\n{}\n\
             \n\
             Generate training guidance as JSON with format: {{\"guidance\": [{{\"rule\": \"...\", \"confidence\": 0.95}}]}}",
            target_model_type,
            examples.join("\n")
        );

        let request_body = json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in knowledge distillation for machine learning models. Respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "response_format": { "type": "json_object" }
        });

        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                if let Ok(parsed) = serde_json::from_str::<Value>(content) {
                    if let Some(guidance) = parsed["guidance"].as_array() {
                        let rules: Vec<(String, f32)> = guidance
                            .iter()
                            .filter_map(|item| {
                                if let (Some(rule), Some(conf)) = (
                                    item["rule"].as_str(),
                                    item["confidence"].as_f64(),
                                ) {
                                    Some((rule.to_string(), conf as f32))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        return Ok(rules);
                    }
                }
            }
        }
        
        Err(anyhow::anyhow!("Failed to distill model knowledge"))
    }

    /// Generate embeddings using OpenAI's embedding models
    pub async fn generate_embeddings(
        &self,
        texts: &[String],
        model: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let api_key = self.get_api_key()?;
        let model = model.unwrap_or("text-embedding-3-small");
        
        let request_body = json!({
            "model": model,
            "input": texts
        });

        let response = self
            .client
            .post(&format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(data) = response_json["data"].as_array() {
            let embeddings: Vec<Vec<f32>> = data
                .iter()
                .filter_map(|item| {
                    item["embedding"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect()
                        })
                })
                .collect();
            return Ok(embeddings);
        }
        
        Err(anyhow::anyhow!("Failed to generate embeddings"))
    }

    /// Optimize model hyperparameters using GPT analysis
    pub async fn optimize_hyperparameters(
        &self,
        model_type: &str,
        current_params: &HashMap<String, Value>,
        performance_metrics: &HashMap<String, f32>,
        training_history: &[HashMap<String, f32>],
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let prompt = format!(
            "Analyze this machine learning model's performance and suggest hyperparameter optimizations:\n\
             Model Type: {}\n\
             Current Parameters: {:?}\n\
             Performance Metrics: {:?}\n\
             Training History (last 5 epochs): {:?}\n\
             \n\
             Provide optimized hyperparameters in JSON format with explanations.",
            model_type,
            current_params,
            performance_metrics,
            training_history.iter().rev().take(5).collect::<Vec<_>>()
        );

        let request_body = json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in machine learning hyperparameter optimization. Provide specific, actionable recommendations in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1200,
            "temperature": 0.1,
            "response_format": { "type": "json_object" }
        });

        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                let optimized_params: HashMap<String, Value> = serde_json::from_str(content)
                    .unwrap_or_else(|_| {
                        let mut map = HashMap::new();
                        map.insert("error".to_string(), json!("Failed to parse optimization suggestions"));
                        map.insert("raw_response".to_string(), json!(content));
                        map
                    });
                return Ok(optimized_params);
            }
        }
        
        Err(anyhow::anyhow!("Failed to optimize hyperparameters"))
    }
}

// Legacy compatibility
pub use OpenAIService as OpenAIDistiller;