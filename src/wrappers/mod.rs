// Framework wrapper implementations for multi-framework ML platform

pub mod tensorflow;
pub mod pytorch;
pub mod keras;
pub mod cuda;
pub mod metal;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Framework {
    TensorFlow,
    PyTorch,
    Keras,
    Native,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub layers: Vec<LayerConfig>,
    pub optimizer: OptimizerConfig,
    pub loss_function: String,
    pub metrics: Vec<String>,
    pub framework_specific: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layer_type: String,
    pub units: Option<usize>,
    pub activation: Option<String>,
    pub dropout_rate: Option<f32>,
    pub kernel_size: Option<Vec<usize>>,
    pub stride: Option<Vec<usize>>,
    pub padding: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f64,
    pub weight_decay: Option<f64>,
    pub momentum: Option<f64>,
    pub epsilon: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub num_samples: usize,
    pub data_source: DataSource,
    pub preprocessing: Vec<PreprocessingStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    Local { path: String },
    Remote { url: String, auth: Option<String> },
    Database { connection_string: String, query: String },
    Stream { endpoint: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStep {
    pub step_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub val_loss: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub epoch: usize,
    pub training_time: f64,
    pub memory_usage: Option<u64>,
}

#[async_trait]
pub trait ModelWrapper: Send + Sync {
    async fn create_model(&self, config: &ModelConfig) -> Result<Box<dyn TrainableModel>>;
    async fn load_model(&self, path: &str) -> Result<Box<dyn TrainableModel>>;
    async fn get_supported_operations(&self) -> Vec<String>;
    fn framework_name(&self) -> &'static str;
    fn supports_gpu(&self) -> bool;
}

#[async_trait]
pub trait TrainableModel: Send + Sync {
    async fn train(&mut self, data: &TrainingData, epochs: usize) -> Result<Vec<TrainingMetrics>>;
    async fn predict(&self, input: &[f32]) -> Result<Vec<f32>>;
    async fn evaluate(&self, test_data: &TrainingData) -> Result<TrainingMetrics>;
    async fn save(&self, path: &str) -> Result<()>;
    fn model_size(&self) -> usize;
    fn parameter_count(&self) -> usize;
}

pub struct FrameworkRouter {
    wrappers: HashMap<Framework, Box<dyn ModelWrapper>>,
    auto_selector: AutoFrameworkSelector,
}

impl FrameworkRouter {
    pub fn new() -> Self {
        let mut wrappers: HashMap<Framework, Box<dyn ModelWrapper>> = HashMap::new();
        
        // Initialize framework wrappers
        wrappers.insert(Framework::TensorFlow, Box::new(tensorflow::TensorFlowWrapper::new()));
        wrappers.insert(Framework::PyTorch, Box::new(pytorch::PyTorchWrapper::new()));
        wrappers.insert(Framework::Keras, Box::new(keras::KerasWrapper::new()));
        
        Self {
            wrappers,
            auto_selector: AutoFrameworkSelector::new(),
        }
    }

    pub async fn create_model(&self, 
        framework: Framework, 
        config: &ModelConfig
    ) -> Result<Box<dyn TrainableModel>> {
        let selected_framework = match framework {
            Framework::Auto => self.auto_selector.select_best_framework(config).await?,
            other => other,
        };

        let wrapper = self.wrappers.get(&selected_framework)
            .ok_or_else(|| anyhow::anyhow!("Framework {:?} not available", selected_framework))?;

        wrapper.create_model(config).await
    }

    pub async fn auto_select(&self, config: &ModelConfig) -> Result<Framework> {
        self.auto_selector.select_best_framework(config).await
    }

    pub fn get_available_frameworks(&self) -> Vec<Framework> {
        self.wrappers.keys().cloned().collect()
    }
}

pub struct AutoFrameworkSelector;

impl AutoFrameworkSelector {
    pub fn new() -> Self {
        Self
    }

    pub async fn select_best_framework(&self, config: &ModelConfig) -> Result<Framework> {
        // Simple selection logic based on model characteristics
        let has_convolution = config.layers.iter()
            .any(|layer| layer.layer_type.contains("conv"));
        
        let has_recurrent = config.layers.iter()
            .any(|layer| layer.layer_type.contains("lstm") || layer.layer_type.contains("rnn"));
        
        if has_convolution {
            Ok(Framework::TensorFlow)
        } else if has_recurrent {
            Ok(Framework::PyTorch)
        } else {
            Ok(Framework::Keras)
        }
    }
}

// Hardware acceleration detection
pub fn detect_available_accelerators() -> Vec<AcceleratorType> {
    let mut accelerators = Vec::new();
    
    // Check for CUDA
    if cuda::CudaAccelerator::is_available() {
        accelerators.push(AcceleratorType::CUDA);
    }
    
    // Check for Metal (macOS)
    #[cfg(target_os = "macos")]
    if metal::MetalAccelerator::is_available() {
        accelerators.push(AcceleratorType::Metal);
    }
    
    // Always have CPU
    accelerators.push(AcceleratorType::CPU);
    
    accelerators
}

#[derive(Debug, Clone, PartialEq)]
pub enum AcceleratorType {
    CPU,
    CUDA,
    Metal,
    OpenCL,
}