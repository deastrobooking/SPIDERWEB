// Model registry and management API

use actix_web::{web, HttpResponse, Result as ActixResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::api::{ApiState, auth::AuthToken};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub framework: String,
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub model_type: ModelType,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub metrics: Option<ModelMetrics>,
    pub status: ModelStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ModelType {
    Classification,
    Regression,
    LanguageModel,
    ImageClassification,
    ObjectDetection,
    TextGeneration,
    Custom,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ModelStatus {
    Training,
    Ready,
    Failed,
    Deprecated,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: Option<f64>,
    pub loss: Option<f64>,
    pub f1_score: Option<f64>,
    pub inference_latency_ms: Option<f64>,
    pub model_size_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
    pub total: usize,
    pub page: usize,
    pub page_size: usize,
}

pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        
        // Add some demo models
        registry.add_demo_models();
        registry
    }

    fn add_demo_models(&mut self) {
        let sentiment_model = ModelInfo {
            id: "sentiment-v1".to_string(),
            name: "Sentiment Analysis".to_string(),
            description: "BERT-based sentiment classification model".to_string(),
            framework: "PyTorch".to_string(),
            version: "1.0.0".to_string(),
            created_at: Utc::now(),
            model_type: ModelType::Classification,
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                }
            }),
            output_schema: serde_json::json!({
                "type": "object", 
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"}
                }
            }),
            metrics: Some(ModelMetrics {
                accuracy: Some(0.94),
                loss: Some(0.15),
                f1_score: Some(0.93),
                inference_latency_ms: Some(45.0),
                model_size_mb: Some(440.0),
            }),
            status: ModelStatus::Ready,
        };

        let image_classifier = ModelInfo {
            id: "resnet50-imagenet".to_string(),
            name: "ImageNet Classifier".to_string(),
            description: "ResNet-50 trained on ImageNet dataset".to_string(),
            framework: "TensorFlow".to_string(),
            version: "2.1.0".to_string(),
            created_at: Utc::now(),
            model_type: ModelType::ImageClassification,
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "image": {"type": "string", "format": "base64"}
                }
            }),
            output_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "class": {"type": "string"},
                                "probability": {"type": "number"}
                            }
                        }
                    }
                }
            }),
            metrics: Some(ModelMetrics {
                accuracy: Some(0.76),
                loss: Some(0.89),
                f1_score: Some(0.75),
                inference_latency_ms: Some(120.0),
                model_size_mb: Some(98.0),
            }),
            status: ModelStatus::Ready,
        };

        self.models.insert(sentiment_model.id.clone(), sentiment_model);
        self.models.insert(image_classifier.id.clone(), image_classifier);
    }

    pub fn list_models(&self, page: usize, page_size: usize) -> ModelListResponse {
        let models: Vec<ModelInfo> = self.models.values()
            .skip(page * page_size)
            .take(page_size)
            .cloned()
            .collect();

        ModelListResponse {
            total: self.models.len(),
            models,
            page,
            page_size,
        }
    }

    pub fn get_model(&self, id: &str) -> Option<&ModelInfo> {
        self.models.get(id)
    }
}

#[actix_web::get("/models")]
pub async fn list_models(
    query: web::Query<ListModelsQuery>,
    state: web::Data<ApiState>,
    _auth: AuthToken,
) -> ActixResult<HttpResponse> {
    let registry = state.model_registry.read().await;
    let response = registry.list_models(query.page.unwrap_or(0), query.page_size.unwrap_or(10));
    Ok(HttpResponse::Ok().json(response))
}

#[derive(Debug, Deserialize)]
pub struct ListModelsQuery {
    pub page: Option<usize>,
    pub page_size: Option<usize>,
    pub model_type: Option<String>,
}

#[actix_web::get("/models/{model_id}")]
pub async fn get_model_info(
    model_id: web::Path<String>,
    state: web::Data<ApiState>,
    _auth: AuthToken,
) -> ActixResult<HttpResponse> {
    let registry = state.model_registry.read().await;
    
    match registry.get_model(&model_id) {
        Some(model) => Ok(HttpResponse::Ok().json(model)),
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Model not found"
        })))
    }
}

#[actix_web::post("/models")]
pub async fn upload_model(
    _request: web::Json<UploadModelRequest>,
    _state: web::Data<ApiState>,
    _auth: AuthToken,
) -> ActixResult<HttpResponse> {
    // In production, this would handle model file upload and registration
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "message": "Model upload endpoint - implementation pending",
        "note": "Would handle model file upload, validation, and registration"
    })))
}

#[actix_web::delete("/models/{model_id}")]
pub async fn delete_model(
    _model_id: web::Path<String>,
    _state: web::Data<ApiState>,
    _auth: AuthToken,
) -> ActixResult<HttpResponse> {
    // In production, this would remove the model from registry and storage
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "message": "Model deletion endpoint - implementation pending"
    })))
}

#[derive(Debug, Deserialize)]
pub struct UploadModelRequest {
    pub name: String,
    pub description: String,
    pub model_type: ModelType,
    pub framework: String,
}