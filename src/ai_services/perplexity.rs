// Perplexity AI integration for search-enhanced model training and real-time knowledge access

use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;

/// Perplexity service client for search-enhanced AI capabilities
#[derive(Clone)]
pub struct PerplexityService {
    client: Client,
    api_key: Option<String>,
    base_url: String,
}

impl PerplexityService {
    pub fn new() -> Self {
        let api_key = env::var("PERPLEXITY_API_KEY").ok();
        
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.perplexity.ai".to_string(),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key: Some(api_key),
            base_url: "https://api.perplexity.ai".to_string(),
        }
    }

    fn get_api_key(&self) -> Result<&String> {
        self.api_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Perplexity API key not configured. Please set PERPLEXITY_API_KEY environment variable."))
    }

    /// Search-enhanced model training recommendations
    pub async fn enhance_training_with_search(
        &self,
        model_domain: &str,
        training_objectives: &[String],
        current_challenges: &[String],
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let search_query = format!(
            "Latest research and best practices for {} machine learning models in 2024-2025. Focus on: {}. Address challenges: {}",
            model_domain,
            training_objectives.join(", "),
            current_challenges.join(", ")
        );

        let request_body = json!({
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert ML researcher with access to the latest publications and industry practices. Provide current, evidence-based recommendations."
                },
                {
                    "role": "user",
                    "content": search_query
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.2,
            "search_recency_filter": "month",
            "return_citations": true
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
            return Err(anyhow::anyhow!("Perplexity API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        let mut enhancement_data = HashMap::new();
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                enhancement_data.insert("search_enhanced_recommendations".to_string(), json!(content));
            }
        }

        if let Some(citations) = response_json["citations"].as_array() {
            enhancement_data.insert("research_citations".to_string(), json!(citations));
        }

        enhancement_data.insert("search_method".to_string(), json!("perplexity_real_time"));
        enhancement_data.insert("knowledge_cutoff".to_string(), json!("real_time_search"));

        Ok(enhancement_data)
    }

    /// Get latest industry trends and benchmarks
    pub async fn get_industry_benchmarks(
        &self,
        model_type: &str,
        performance_domain: &str,
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let query = format!(
            "Current state-of-the-art performance benchmarks for {} models in {} domain. Include latest papers from 2024-2025, performance metrics, and breakthrough techniques.",
            model_type, performance_domain
        );

        let request_body = json!({
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a ML benchmarking expert. Provide specific, measurable performance data with citations."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.1,
            "search_recency_filter": "month",
            "return_citations": true,
            "return_related_questions": true
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
            return Err(anyhow::anyhow!("Perplexity API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        let mut benchmarks = HashMap::new();
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                benchmarks.insert("benchmark_analysis".to_string(), json!(content));
            }
        }

        if let Some(citations) = response_json["citations"].as_array() {
            benchmarks.insert("benchmark_sources".to_string(), json!(citations));
        }

        if let Some(related) = response_json["related_questions"].as_array() {
            benchmarks.insert("related_research_questions".to_string(), json!(related));
        }

        Ok(benchmarks)
    }

    /// Research-backed data augmentation strategies
    pub async fn suggest_data_augmentation(
        &self,
        data_type: &str,
        domain: &str,
        current_dataset_size: usize,
        target_performance: &HashMap<String, f32>,
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let query = format!(
            "Latest data augmentation techniques for {} data in {} domain. Dataset size: {} samples. Target performance: {:?}. Include recent papers and proven methods from 2024-2025.",
            data_type, domain, current_dataset_size, target_performance
        );

        let request_body = json!({
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data augmentation specialist. Provide specific, implementable techniques with research backing."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            "max_tokens": 1800,
            "temperature": 0.3,
            "search_recency_filter": "month",
            "return_citations": true
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
            return Err(anyhow::anyhow!("Perplexity API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        let mut augmentation_strategies = HashMap::new();
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                augmentation_strategies.insert("augmentation_recommendations".to_string(), json!(content));
            }
        }

        if let Some(citations) = response_json["citations"].as_array() {
            augmentation_strategies.insert("research_sources".to_string(), json!(citations));
        }

        augmentation_strategies.insert("search_enhanced".to_string(), json!(true));
        augmentation_strategies.insert("knowledge_freshness".to_string(), json!("real_time"));

        Ok(augmentation_strategies)
    }

    /// Get competitive analysis and market trends
    pub async fn analyze_competitive_landscape(
        &self,
        application_domain: &str,
        model_capabilities: &[String],
    ) -> Result<HashMap<String, Value>> {
        let api_key = self.get_api_key()?;
        
        let query = format!(
            "Competitive analysis of AI models in {} domain with capabilities: {}. Include latest market leaders, performance comparisons, and emerging trends in 2024-2025.",
            application_domain,
            model_capabilities.join(", ")
        );

        let request_body = json!({
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a market intelligence analyst specializing in AI/ML competitive landscapes. Provide current, actionable insights."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.2,
            "search_recency_filter": "week",
            "return_citations": true,
            "return_related_questions": true
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
            return Err(anyhow::anyhow!("Perplexity API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        let mut competitive_analysis = HashMap::new();
        
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(content) = choices[0]["message"]["content"].as_str() {
                competitive_analysis.insert("competitive_landscape".to_string(), json!(content));
            }
        }

        if let Some(citations) = response_json["citations"].as_array() {
            competitive_analysis.insert("market_sources".to_string(), json!(citations));
        }

        if let Some(related) = response_json["related_questions"].as_array() {
            competitive_analysis.insert("strategic_questions".to_string(), json!(related));
        }

        competitive_analysis.insert("analysis_type".to_string(), json!("real_time_competitive"));

        Ok(competitive_analysis)
    }
}