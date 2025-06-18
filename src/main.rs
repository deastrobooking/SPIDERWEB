// Main application for ML-as-a-Service platform

use anyhow::Result;
use tokio;
use tracing::{info, error};
use tracing_subscriber;

mod tensor;
mod nn;
mod optim;
mod loss;
mod data;
mod metrics;
mod utils;
mod autograd;
mod backend;
mod transforms;
mod vision;
mod nlp;
mod distributed;
mod api;
mod wrappers;
mod ai_services;
mod llm;
mod deployment;

use api::create_api_server;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("Starting Rust ML Framework API Server");

    // Initialize framework components
    initialize_framework().await?;

    // Start the API server
    if let Err(e) = create_api_server().await {
        error!("Failed to start API server: {}", e);
        return Err(e.into());
    }

    Ok(())
}

async fn initialize_framework() -> Result<()> {
    info!("Initializing ML framework components");

    // Initialize BLAS/LAPACK backends
    backend::initialize_compute_backends()?;

    // Check available hardware accelerators
    let accelerators = wrappers::detect_available_accelerators();
    info!("Available accelerators: {:?}", accelerators);

    // Initialize distributed training if multiple GPUs available
    if accelerators.iter().any(|a| matches!(a, wrappers::AcceleratorType::CUDA)) {
        backend::initialize_cuda_context()?;
        info!("CUDA distributed training initialized");
    }

    // Initialize AI service integrations
    ai_services::initialize_external_services().await?;
    info!("External AI services initialized");

    // Start global LLM training pool manager
    llm::start_global_pool_manager().await?;
    info!("Global LLM training pool started");

    info!("Framework initialization complete");
    Ok(())
}

// Framework initialization function for library use
pub fn init() -> Result<()> {
    backend::initialize_compute_backends()
}

// Set random seed for reproducibility
pub fn set_seed(seed: u64) {
    utils::set_global_seed(seed);
}

// Public API exports
pub use tensor::Tensor;
pub use nn::{Module, Sequential, Linear, Conv2d, ReLU, Sigmoid, Dropout, BatchNorm2d};
pub use optim::{Optimizer, SGD, Adam, AdamW, RMSprop, Adagrad};
pub use loss::{Loss, MSELoss, CrossEntropyLoss, BCELoss};
pub use data::{Dataset, DataLoader, TensorDataset};
pub use metrics::{accuracy, precision, recall, f1_score};
pub use vision::{resnet18, resnet50, vgg16, mobilenet_v2};
pub use nlp::{Tokenizer, BertModel, GPTModel};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_framework_initialization() {
        let result = initialize_framework().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_basic_tensor_operations() {
        init().unwrap();
        
        let a = Tensor::ones(&[2, 3]);
        let b = Tensor::zeros(&[2, 3]);
        let c = &a + &b;
        
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.mean(), 1.0);
    }

    #[test]
    fn test_neural_network_creation() {
        init().unwrap();
        
        let model = Sequential::new()
            .add(Linear::new(10, 5))
            .add(ReLU::new())
            .add(Linear::new(5, 1));
        
        let input = Tensor::randn(&[1, 10]);
        let output = model.forward(&input).unwrap();
        
        assert_eq!(output.shape(), &[1, 1]);
    }

    #[test]
    fn test_optimizer_creation() {
        init().unwrap();
        
        let model = Sequential::new()
            .add(Linear::new(10, 1));
        
        let params: Vec<*mut Tensor> = model.parameters_mut()
            .into_iter()
            .map(|p| p as *mut Tensor)
            .collect();
        
        let optimizer = Adam::new(params, 0.001);
        assert!(optimizer.learning_rate() > 0.0);
    }
}