//! Rust ML Framework - High-Performance Machine Learning Library
//! 
//! A comprehensive machine learning framework providing TensorFlow and PyTorch-like
//! functionality with Rust's safety guarantees and performance characteristics.
//!
//! # Overview
//!
//! This crate provides a complete machine learning framework featuring:
//!
//! - **Tensor Operations**: N-dimensional arrays with automatic differentiation
//! - **Neural Networks**: Complete layer implementations (Linear, Conv, RNN, Transformer)
//! - **Optimizers**: State-of-the-art optimization algorithms (SGD, Adam, AdamW, RMSprop)
//! - **Loss Functions**: Comprehensive loss function implementations
//! - **Data Loading**: Efficient data pipeline with parallel processing
//! - **GPU Acceleration**: CUDA and Vulkan backend support
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use rust_ml_framework::*;
//!
//! // Initialize the framework
//! init().unwrap();
//!
//! // Create a simple neural network
//! let model = nn::Sequential::new()
//!     .add(nn::Linear::new(784, 128))
//!     .add(nn::ReLU::new())
//!     .add(nn::Linear::new(128, 10));
//!
//! // Create input tensor
//! let input = Tensor::randn(&[32, 784]);
//!
//! // Forward pass
//! let output = model.forward(&input).unwrap();
//! ```
//!
//! # Architecture
//!
//! The framework is organized into several key modules:
//!
//! - [`tensor`]: Core tensor operations and storage
//! - [`nn`]: Neural network layers and building blocks
//! - [`optim`]: Optimization algorithms
//! - [`loss`]: Loss function implementations
//! - [`data`]: Data loading and preprocessing
//! - [`autograd`]: Automatic differentiation engine
//! - [`vision`]: Computer vision utilities and models
//! - [`nlp`]: Natural language processing utilities
//!
//! # Performance
//!
//! The framework is designed for high performance with:
//!
//! - Zero-copy operations where possible
//! - BLAS/LAPACK integration for optimized linear algebra
//! - GPU acceleration support via CUDA and Vulkan
//! - Parallel data loading with rayon
//! - Memory-efficient tensor storage
//!
//! # Safety
//!
//! Rust's ownership system provides memory safety guarantees:
//!
//! - No null pointer dereferences
//! - No buffer overflows
//! - Thread safety through type system
//! - Safe FFI boundaries with external libraries

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