// Inference API endpoints

use actix_web::{web, HttpResponse, Result as ActixResult};
use serde::{Deserialize, Serialize};
use crate::api::{ApiState, auth::AuthToken};

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub model_id: String,
    pub input: serde_json::Value,
    pub options: Option<PredictionOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionOptions {
    pub batch_size: Option<usize>,
    pub return_probabilities: Option<bool>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionResponse {
    pub predictions: serde_json::Value,
    pub confidence: Option<Vec<f32>>,
    pub latency_ms: u64,
    pub model_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchPredictionRequest {
    pub model_id: String,
    pub inputs: Vec<serde_json::Value>,
    pub options: Option<PredictionOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchPredictionResponse {
    pub results: Vec<PredictionResponse>,
    pub total_latency_ms: u64,
    pub processed_count: usize,
}

#[actix_web::post("/predict")]
pub async fn predict(
    request: web::Json<PredictionRequest>,
    _state: web::Data<ApiState>,
    _auth: AuthToken,
) -> ActixResult<HttpResponse> {
    let start_time = std::time::Instant::now();
    
    // Mock prediction for demonstration
    let mock_prediction = serde_json::json!({
        "class": "positive",
        "score": 0.85
    });
    
    let response = PredictionResponse {
        predictions: mock_prediction,
        confidence: Some(vec![0.85, 0.15]),
        latency_ms: start_time.elapsed().as_millis() as u64,
        model_version: "1.0.0".to_string(),
    };
    
    Ok(HttpResponse::Ok().json(response))
}

#[actix_web::post("/batch_predict")]
pub async fn batch_predict(
    request: web::Json<BatchPredictionRequest>,
    _state: web::Data<ApiState>,
    _auth: AuthToken,
) -> ActixResult<HttpResponse> {
    let start_time = std::time::Instant::now();
    
    let results: Vec<PredictionResponse> = request.inputs.iter().enumerate().map(|(i, _input)| {
        PredictionResponse {
            predictions: serde_json::json!({
                "class": if i % 2 == 0 { "positive" } else { "negative" },
                "score": 0.75 + (i as f64 * 0.05) % 0.25
            }),
            confidence: Some(vec![0.75, 0.25]),
            latency_ms: 5,
            model_version: "1.0.0".to_string(),
        }
    }).collect();
    
    let response = BatchPredictionResponse {
        processed_count: results.len(),
        results,
        total_latency_ms: start_time.elapsed().as_millis() as u64,
    };
    
    Ok(HttpResponse::Ok().json(response))
}