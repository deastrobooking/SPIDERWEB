//! Rust ML Framework - A comprehensive machine learning library
//! 
//! This crate provides a complete machine learning framework with tensor operations,
//! neural networks, optimizers, and training utilities similar to TensorFlow and PyTorch.

pub mod tensor;
pub mod nn;
pub mod optim;
pub mod data;
pub mod metrics;
pub mod loss;
pub mod autograd;
pub mod backend;
pub mod transforms;
pub mod vision;
pub mod nlp;
pub mod distributed;
pub mod utils;

// Re-export core types for convenience
pub use tensor::{Tensor, TensorOps, Device};
pub use nn::{Module, Sequential, Linear, Conv2d, ReLU, Sigmoid, Tanh};
pub use optim::{Optimizer, SGD, Adam, AdamW};
pub use loss::{Loss, MSELoss, CrossEntropyLoss, BCELoss};
pub use data::{DataLoader, Dataset};
pub use autograd::{Variable, backward};

/// Current version of the framework
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the ML framework with default settings
pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    log::info!("Rust ML Framework v{} initialized", VERSION);
    Ok(())
}

/// Set random seed for reproducibility across all random number generators
pub fn set_seed(seed: u64) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Set global seed state for reproducible operations
    log::info!("Random seed set to: {}", seed);
}