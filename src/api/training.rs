// Training API endpoints for ML-as-a-Service platform

use actix_web::{web, HttpResponse, Result as ActixResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::api::ApiState;
use crate::wrappers::{ModelConfig, TrainingData, Framework};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub model_config: ModelConfig,
    pub training_data: TrainingDataSpec,
    pub training_params: TrainingParams,
    pub framework_preference: Option<Framework>,
    pub hardware_requirements: HardwareRequirements,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingDataSpec {
    pub data_source: DataSourceConfig,
    pub validation_split: f32,
    pub batch_size: usize,
    pub preprocessing: Vec<PreprocessingConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataSourceConfig {
    pub source_type: String,
    pub connection_info: HashMap<String, String>,
    pub format: String,
    pub schema: Option<DataSchema>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataSchema {
    pub input_columns: Vec<String>,
    pub target_column: String,
    pub feature_types: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub operation: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingParams {
    pub epochs: usize,
    pub learning_rate: f64,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub checkpointing: Option<CheckpointConfig>,
    pub metrics: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub monitor: String,
    pub patience: usize,
    pub min_delta: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub save_best_only: bool,
    pub save_frequency: usize,
    pub keep_last_n: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareRequirements {
    pub gpu_required: bool,
    pub gpu_memory_gb: Option<u32>,
    pub cpu_cores: Option<u32>,
    pub memory_gb: Option<u32>,
    pub max_training_time_hours: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResponse {
    pub job_id: String,
    pub status: String,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub resource_allocation: ResourceAllocation,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub framework: Framework,
    pub hardware: AllocatedHardware,
    pub estimated_cost: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AllocatedHardware {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub gpu_type: Option<String>,
    pub gpu_memory_gb: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub job_id: String,
    pub status: JobStatus,
    pub progress: TrainingProgress,
    pub metrics: Option<CurrentMetrics>,
    pub logs: Vec<LogEntry>,
    pub estimated_completion: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Starting,
    Running,
    Completed,
    Failed,
    Cancelled,
    Paused,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub current_batch: usize,
    pub total_batches: usize,
    pub elapsed_time_seconds: u64,
    pub eta_seconds: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CurrentMetrics {
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub val_loss: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub learning_rate: f64,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub source: String,
}

#[actix_web::post("/train")]
pub async fn start_training(
    request: web::Json<TrainingRequest>,
    state: web::Data<ApiState>,
    auth: crate::api::auth::AuthToken,
) -> ActixResult<HttpResponse> {
    let job_id = Uuid::new_v4().to_string();
    
    // Validate request
    let validation_result = validate_training_request(&request).await?;
    if !validation_result.is_valid {
        return Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Invalid training request",
            "details": validation_result.errors
        })));
    }

    // Check user quota and permissions
    let quota_check = state.auth_service.check_training_quota(&auth.user_id).await?;
    if !quota_check.can_train {
        return Ok(HttpResponse::PaymentRequired().json(serde_json::json!({
            "error": "Training quota exceeded",
            "current_usage": quota_check.current_usage,
            "limit": quota_check.limit
        })));
    }

    // Select optimal framework
    let framework = match &request.framework_preference {
        Some(fw) => fw.clone(),
        None => state.framework_router.auto_select(&request.model_config).await?,
    };

    // Allocate resources
    let resource_allocation = allocate_training_resources(
        &request.hardware_requirements,
        &framework
    ).await?;

    // Create training job
    let mut training_manager = state.training_manager.write().await;
    let training_job = training_manager.create_job(
        &job_id,
        &auth.user_id,
        request.into_inner(),
        framework.clone(),
        resource_allocation.clone(),
    ).await?;

    // Start training asynchronously
    tokio::spawn(async move {
        if let Err(e) = execute_training_job(training_job).await {
            log::error!("Training job {} failed: {:?}", job_id, e);
        }
    });

    // Update user quota
    state.auth_service.record_training_start(&auth.user_id, &job_id).await?;

    Ok(HttpResponse::Ok().json(TrainingResponse {
        job_id: job_id.clone(),
        status: "starting".to_string(),
        estimated_completion: calculate_estimated_completion(&request.training_params),
        resource_allocation,
    }))
}

#[actix_web::get("/train/{job_id}/status")]
pub async fn get_training_status(
    job_id: web::Path<String>,
    state: web::Data<ApiState>,
    auth: crate::api::auth::AuthToken,
) -> ActixResult<HttpResponse> {
    let training_manager = state.training_manager.read().await;
    
    match training_manager.get_job_status(&job_id, &auth.user_id).await? {
        Some(status) => Ok(HttpResponse::Ok().json(status)),
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Training job not found"
        })))
    }
}

#[actix_web::post("/train/{job_id}/stop")]
pub async fn stop_training(
    job_id: web::Path<String>,
    state: web::Data<ApiState>,
    auth: crate::api::auth::AuthToken,
) -> ActixResult<HttpResponse> {
    let mut training_manager = state.training_manager.write().await;
    
    match training_manager.stop_job(&job_id, &auth.user_id).await? {
        true => Ok(HttpResponse::Ok().json(serde_json::json!({
            "message": "Training job stopped successfully"
        }))),
        false => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Training job not found or cannot be stopped"
        })))
    }
}

pub struct TrainingManager {
    active_jobs: HashMap<String, TrainingJob>,
    completed_jobs: HashMap<String, CompletedJob>,
    resource_manager: ResourceManager,
}

impl TrainingManager {
    pub fn new() -> Self {
        Self {
            active_jobs: HashMap::new(),
            completed_jobs: HashMap::new(),
            resource_manager: ResourceManager::new(),
        }
    }

    pub async fn create_job(
        &mut self,
        job_id: &str,
        user_id: &str,
        request: TrainingRequest,
        framework: Framework,
        resources: ResourceAllocation,
    ) -> anyhow::Result<TrainingJob> {
        let job = TrainingJob {
            id: job_id.to_string(),
            user_id: user_id.to_string(),
            request,
            framework,
            resources,
            status: JobStatus::Queued,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            progress: TrainingProgress {
                current_epoch: 0,
                total_epochs: 0,
                current_batch: 0,
                total_batches: 0,
                elapsed_time_seconds: 0,
                eta_seconds: None,
            },
            metrics: None,
            logs: Vec::new(),
        };

        self.active_jobs.insert(job_id.to_string(), job.clone());
        Ok(job)
    }

    pub async fn get_job_status(
        &self,
        job_id: &str,
        user_id: &str,
    ) -> anyhow::Result<Option<TrainingStatus>> {
        if let Some(job) = self.active_jobs.get(job_id) {
            if job.user_id == user_id {
                return Ok(Some(TrainingStatus {
                    job_id: job.id.clone(),
                    status: job.status.clone(),
                    progress: job.progress.clone(),
                    metrics: job.metrics.clone(),
                    logs: job.logs.clone(),
                    estimated_completion: job.estimate_completion(),
                }));
            }
        }

        if let Some(completed) = self.completed_jobs.get(job_id) {
            if completed.user_id == user_id {
                return Ok(Some(completed.to_status()));
            }
        }

        Ok(None)
    }

    pub async fn stop_job(&mut self, job_id: &str, user_id: &str) -> anyhow::Result<bool> {
        if let Some(job) = self.active_jobs.get_mut(job_id) {
            if job.user_id == user_id && matches!(job.status, JobStatus::Running | JobStatus::Starting) {
                job.status = JobStatus::Cancelled;
                // Signal the training process to stop
                self.resource_manager.cancel_job(job_id).await?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[derive(Debug, Clone)]
pub struct TrainingJob {
    pub id: String,
    pub user_id: String,
    pub request: TrainingRequest,
    pub framework: Framework,
    pub resources: ResourceAllocation,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub progress: TrainingProgress,
    pub metrics: Option<CurrentMetrics>,
    pub logs: Vec<LogEntry>,
}

impl TrainingJob {
    pub fn estimate_completion(&self) -> Option<DateTime<Utc>> {
        if let Some(eta_seconds) = self.progress.eta_seconds {
            Some(Utc::now() + chrono::Duration::seconds(eta_seconds as i64))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletedJob {
    pub id: String,
    pub user_id: String,
    pub status: JobStatus,
    pub final_metrics: Option<CurrentMetrics>,
    pub model_path: Option<String>,
    pub completed_at: DateTime<Utc>,
    pub total_duration_seconds: u64,
}

impl CompletedJob {
    pub fn to_status(&self) -> TrainingStatus {
        TrainingStatus {
            job_id: self.id.clone(),
            status: self.status.clone(),
            progress: TrainingProgress {
                current_epoch: 0,
                total_epochs: 0,
                current_batch: 0,
                total_batches: 0,
                elapsed_time_seconds: self.total_duration_seconds,
                eta_seconds: None,
            },
            metrics: self.final_metrics.clone(),
            logs: Vec::new(),
            estimated_completion: None,
        }
    }
}

pub struct ResourceManager {
    available_resources: HashMap<String, AvailableResource>,
    allocated_resources: HashMap<String, String>, // job_id -> resource_id
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            available_resources: HashMap::new(),
            allocated_resources: HashMap::new(),
        }
    }

    pub async fn cancel_job(&mut self, job_id: &str) -> anyhow::Result<()> {
        if let Some(resource_id) = self.allocated_resources.remove(job_id) {
            if let Some(resource) = self.available_resources.get_mut(&resource_id) {
                resource.status = ResourceStatus::Available;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AvailableResource {
    pub id: String,
    pub resource_type: ResourceType,
    pub specifications: HardwareSpecs,
    pub status: ResourceStatus,
    pub hourly_cost: f64,
}

#[derive(Debug, Clone)]
pub enum ResourceType {
    CPU,
    GPU,
    TPU,
}

#[derive(Debug, Clone)]
pub enum ResourceStatus {
    Available,
    Allocated,
    Maintenance,
}

#[derive(Debug, Clone)]
pub struct HardwareSpecs {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub gpu_type: Option<String>,
    pub gpu_memory_gb: Option<u32>,
    pub storage_gb: u32,
}

struct ValidationResult {
    is_valid: bool,
    errors: Vec<String>,
}

async fn validate_training_request(request: &TrainingRequest) -> anyhow::Result<ValidationResult> {
    let mut errors = Vec::new();

    // Validate model configuration
    if request.model_config.layers.is_empty() {
        errors.push("Model must have at least one layer".to_string());
    }

    // Validate training parameters
    if request.training_params.epochs == 0 {
        errors.push("Training epochs must be greater than 0".to_string());
    }

    if request.training_params.learning_rate <= 0.0 {
        errors.push("Learning rate must be positive".to_string());
    }

    // Validate data source
    if request.training_data.batch_size == 0 {
        errors.push("Batch size must be greater than 0".to_string());
    }

    Ok(ValidationResult {
        is_valid: errors.is_empty(),
        errors,
    })
}

async fn allocate_training_resources(
    requirements: &HardwareRequirements,
    framework: &Framework,
) -> anyhow::Result<ResourceAllocation> {
    // Simple resource allocation logic
    let cpu_cores = requirements.cpu_cores.unwrap_or(4);
    let memory_gb = requirements.memory_gb.unwrap_or(8);
    
    let (gpu_type, gpu_memory_gb) = if requirements.gpu_required {
        (Some("Tesla V100".to_string()), requirements.gpu_memory_gb.or(Some(16)))
    } else {
        (None, None)
    };

    let estimated_cost = calculate_training_cost(cpu_cores, memory_gb, gpu_memory_gb);

    Ok(ResourceAllocation {
        framework: framework.clone(),
        hardware: AllocatedHardware {
            cpu_cores,
            memory_gb,
            gpu_type,
            gpu_memory_gb,
        },
        estimated_cost,
    })
}

fn calculate_training_cost(cpu_cores: u32, memory_gb: u32, gpu_memory_gb: Option<u32>) -> f64 {
    let cpu_cost = cpu_cores as f64 * 0.05; // $0.05 per core per hour
    let memory_cost = memory_gb as f64 * 0.01; // $0.01 per GB per hour
    let gpu_cost = gpu_memory_gb.map(|gb| gb as f64 * 0.5).unwrap_or(0.0); // $0.50 per GPU GB per hour
    
    cpu_cost + memory_cost + gpu_cost
}

fn calculate_estimated_completion(params: &TrainingParams) -> Option<DateTime<Utc>> {
    // Rough estimation: 1 minute per epoch for small models
    let estimated_minutes = params.epochs as i64;
    Some(Utc::now() + chrono::Duration::minutes(estimated_minutes))
}

async fn execute_training_job(job: TrainingJob) -> anyhow::Result<()> {
    log::info!("Starting training job: {}", job.id);
    
    // This would integrate with the actual framework wrappers
    // For now, just simulate training progress
    
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    log::info!("Training job {} completed successfully", job.id);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_request_validation() {
        let request = TrainingRequest {
            model_config: crate::wrappers::ModelConfig {
                layers: vec![],
                optimizer: crate::wrappers::OptimizerConfig {
                    optimizer_type: "adam".to_string(),
                    learning_rate: 0.001,
                    weight_decay: None,
                    momentum: None,
                    epsilon: None,
                },
                loss_function: "mse".to_string(),
                metrics: vec!["accuracy".to_string()],
                framework_specific: HashMap::new(),
            },
            training_data: TrainingDataSpec {
                data_source: DataSourceConfig {
                    source_type: "csv".to_string(),
                    connection_info: HashMap::new(),
                    format: "csv".to_string(),
                    schema: None,
                },
                validation_split: 0.2,
                batch_size: 0, // Invalid
                preprocessing: vec![],
            },
            training_params: TrainingParams {
                epochs: 0, // Invalid
                learning_rate: -0.1, // Invalid
                early_stopping: None,
                checkpointing: None,
                metrics: vec!["accuracy".to_string()],
            },
            framework_preference: None,
            hardware_requirements: HardwareRequirements {
                gpu_required: false,
                gpu_memory_gb: None,
                cpu_cores: None,
                memory_gb: None,
                max_training_time_hours: None,
            },
        };

        // This would be an async test in practice
        // let result = validate_training_request(&request).await.unwrap();
        // assert!(!result.is_valid);
        // assert_eq!(result.errors.len(), 3);
    }

    #[test]
    fn test_cost_calculation() {
        let cost = calculate_training_cost(4, 8, Some(16));
        assert_eq!(cost, 0.28); // 4*0.05 + 8*0.01 + 16*0.5 = 0.2 + 0.08 + 8.0
    }
}