// Anthropic integration for reasoning and knowledge extraction
// The newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
// Do not change this unless explicitly requested by the user

use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;

/// Anthropic service client for advanced reasoning and model analysis
#[derive(Clone)]
pub struct AnthropicService {
    client: Client,
    api_key: Option<String>,
    base_url: String,
}

impl AnthropicService {
    pub fn new() -> Self {
        let api_key = env::var("ANTHROPIC_API_KEY").ok();
        
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.anthropic.com/v1".to_string(),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key: Some(api_key),
            base_url: "https://api.anthropic.com/v1".to_string(),
        }
    }

    fn get_api_key(&self) -> Result<&String> {
        self.api_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."))
    }

    /// Advanced reasoning about model architectures and optimization strategies
    pub async fn reason_about_model(
        &self,
        model_description: &str,
        training_data_characteristics: &HashMap<String, Value>,
        performance_requirements: &HashMap<String, f32>,
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let prompt = format!(
            "As an expert in machine learning theory and practice, analyze this model setup and provide deep reasoning:\n\n\
             Model Description: {}\n\
             Training Data Characteristics: {:?}\n\
             Performance Requirements: {:?}\n\n\
             Please provide a comprehensive analysis covering:\n\
             1. Architectural suitability for the task\n\
             2. Potential bottlenecks and optimization opportunities\n\
             3. Training strategy recommendations\n\
             4. Risk assessment and mitigation strategies\n\
             5. Alternative approaches to consider\n\n\
             Format your response as JSON with these sections.",
            model_description,
            training_data_characteristics,
            performance_requirements
        );

        let request_body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3
        });

        let response = self
            .client
            .post(&format!("{}/messages", self.base_url))
            .header("x-api-key", api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(content) = response_json["content"].as_array() {
            if let Some(text) = content[0]["text"].as_str() {
                // Try to parse as JSON, fallback to structured text analysis
                let analysis = if let Ok(parsed) = serde_json::from_str::<HashMap<String, Value>>(text) {
                    parsed
                } else {
                    let mut analysis = HashMap::new();
                    analysis.insert("reasoning".to_string(), json!(text));
                    analysis.insert("structured".to_string(), json!(false));
                    analysis
                };
                return Ok(analysis);
            }
        }
        
        Err(anyhow::anyhow!("Failed to get reasoning analysis"))
    }

    /// Analyze model interpretability and explainability
    pub async fn analyze_model_interpretability(
        &self,
        model_type: &str,
        model_predictions: &[(String, f32, Vec<f32>)], // (input, prediction, features)
        domain_context: &str,
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let sample_predictions: Vec<String> = model_predictions
            .iter()
            .take(5)
            .map(|(input, pred, features)| {
                format!("Input: '{}' -> Prediction: {:.3} (Feature importance: {:?})", 
                       input, pred, &features[..std::cmp::min(features.len(), 5)])
            })
            .collect();

        let prompt = format!(
            "Analyze the interpretability of this machine learning model:\n\n\
             Model Type: {}\n\
             Domain Context: {}\n\
             Sample Predictions:\n{}\n\n\
             Provide analysis on:\n\
             1. Model transparency and explainability\n\
             2. Feature importance patterns\n\
             3. Bias detection and fairness considerations\n\
             4. Recommendations for improving interpretability\n\
             5. Stakeholder communication strategies\n\n\
             Respond in JSON format with clear sections.",
            model_type,
            domain_context,
            sample_predictions.join("\n")
        );

        let request_body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1800,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        });

        let response = self
            .client
            .post(&format!("{}/messages", self.base_url))
            .header("x-api-key", api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(content) = response_json["content"].as_array() {
            if let Some(text) = content[0]["text"].as_str() {
                let analysis = if let Ok(parsed) = serde_json::from_str::<HashMap<String, Value>>(text) {
                    parsed
                } else {
                    let mut analysis = HashMap::new();
                    analysis.insert("interpretability_analysis".to_string(), json!(text));
                    analysis
                };
                return Ok(analysis);
            }
        }
        
        Err(anyhow::anyhow!("Failed to analyze model interpretability"))
    }

    /// Generate advanced training strategies based on complex reasoning
    pub async fn generate_training_strategy(
        &self,
        problem_domain: &str,
        dataset_characteristics: &HashMap<String, Value>,
        constraints: &HashMap<String, Value>,
        success_metrics: &[String],
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let prompt = format!(
            "Design a comprehensive machine learning training strategy:\n\n\
             Problem Domain: {}\n\
             Dataset Characteristics: {:?}\n\
             Constraints: {:?}\n\
             Success Metrics: {:?}\n\n\
             Please reason through and provide:\n\
             1. Optimal model architecture selection rationale\n\
             2. Training methodology and curriculum design\n\
             3. Data augmentation and preprocessing strategies\n\
             4. Regularization and generalization techniques\n\
             5. Evaluation and validation frameworks\n\
             6. Risk mitigation and fallback plans\n\n\
             Provide detailed reasoning for each recommendation in JSON format.",
            problem_domain,
            dataset_characteristics,
            constraints,
            success_metrics
        );

        let request_body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2500,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1
        });

        let response = self
            .client
            .post(&format!("{}/messages", self.base_url))
            .header("x-api-key", api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(content) = response_json["content"].as_array() {
            if let Some(text) = content[0]["text"].as_str() {
                let strategy = if let Ok(parsed) = serde_json::from_str::<HashMap<String, Value>>(text) {
                    parsed
                } else {
                    let mut strategy = HashMap::new();
                    strategy.insert("training_strategy".to_string(), json!(text));
                    strategy
                };
                return Ok(strategy);
            }
        }
        
        Err(anyhow::anyhow!("Failed to generate training strategy"))
    }

    /// Advanced model debugging and troubleshooting
    pub async fn debug_model_issues(
        &self,
        problem_description: &str,
        training_metrics: &HashMap<String, Vec<f32>>,
        model_config: &HashMap<String, Value>,
        error_patterns: &[String],
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let metrics_summary: HashMap<String, String> = training_metrics
            .iter()
            .map(|(key, values)| {
                let avg = values.iter().sum::<f32>() / values.len() as f32;
                let trend = if values.len() > 1 {
                    let last = values[values.len() - 1];
                    let first = values[0];
                    if last > first { "increasing" } else { "decreasing" }
                } else { "stable" };
                (key.clone(), format!("avg: {:.4}, trend: {}", avg, trend))
            })
            .collect();

        let prompt = format!(
            "Debug this machine learning model issue:\n\n\
             Problem Description: {}\n\
             Training Metrics Summary: {:?}\n\
             Model Configuration: {:?}\n\
             Error Patterns: {:?}\n\n\
             Please provide:\n\
             1. Root cause analysis with reasoning\n\
             2. Specific diagnostic steps to confirm hypothesis\n\
             3. Prioritized solution recommendations\n\
             4. Prevention strategies for similar issues\n\
             5. Alternative approaches if primary solutions fail\n\n\
             Format as JSON with clear action items.",
            problem_description,
            metrics_summary,
            model_config,
            error_patterns
        );

        let request_body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        });

        let response = self
            .client
            .post(&format!("{}/messages", self.base_url))
            .header("x-api-key", api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        if let Some(content) = response_json["content"].as_array() {
            if let Some(text) = content[0]["text"].as_str() {
                let debug_analysis = if let Ok(parsed) = serde_json::from_str::<HashMap<String, Value>>(text) {
                    parsed
                } else {
                    let mut analysis = HashMap::new();
                    analysis.insert("debug_analysis".to_string(), json!(text));
                    analysis
                };
                return Ok(debug_analysis);
            }
        }
        
        Err(anyhow::anyhow!("Failed to debug model issues"))
    }
}

// Legacy compatibility
pub use AnthropicService as AnthropicEnhancer;