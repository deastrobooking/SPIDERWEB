//! Neural network layers and modules
//! 
//! This module provides neural network building blocks similar to PyTorch's nn module
//! and TensorFlow's layers API.

pub mod linear;
pub mod conv;
pub mod activation;
pub mod normalization;
pub mod pooling;
pub mod dropout;
pub mod rnn;
pub mod transformer;
pub mod loss;

use crate::tensor::Tensor;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

// Re-export commonly used types
pub use linear::Linear;
pub use conv::{Conv1d, Conv2d, Conv3d};
pub use activation::{ReLU, Sigmoid, Tanh, Softmax, GELU, Swish};
pub use normalization::{BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm};
pub use pooling::{MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, AdaptiveAvgPool2d};
pub use dropout::Dropout;
pub use rnn::{RNN, LSTM, GRU};
pub use transformer::{MultiHeadAttention, TransformerEncoder, TransformerDecoder};

/// Base trait for all neural network modules
pub trait Module: Send + Sync {
    /// Forward pass through the module
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    
    /// Get module parameters
    fn parameters(&self) -> Vec<&Tensor>;
    
    /// Get mutable module parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    
    /// Set module to training mode
    fn train(&mut self);
    
    /// Set module to evaluation mode
    fn eval(&mut self);
    
    /// Check if module is in training mode
    fn training(&self) -> bool;
    
    /// Get module name
    fn name(&self) -> &str;
    
    /// Clone the module
    fn clone_module(&self) -> Box<dyn Module>;
}

/// Sequential container for modules
#[derive(Debug, Clone)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    /// Create new sequential container
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }
    
    /// Add a module to the sequence
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }
    
    /// Add a module by boxed reference
    pub fn add_boxed(mut self, module: Box<dyn Module>) -> Self {
        self.modules.push(module);
        self
    }
    
    /// Get number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone_tensor();
        for module in &self.modules {
            output = module.forward(&output)?;
        }
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for module in &mut self.modules {
            params.extend(module.parameters_mut());
        }
        params
    }
    
    fn train(&mut self) {
        self.training = true;
        for module in &mut self.modules {
            module.train();
        }
    }
    
    fn eval(&mut self) {
        self.training = false;
        for module in &mut self.modules {
            module.eval();
        }
    }
    
    fn training(&self) -> bool {
        self.training
    }
    
    fn name(&self) -> &str {
        "Sequential"
    }
    
    fn clone_module(&self) -> Box<dyn Module> {
        let mut new_seq = Sequential::new();
        for module in &self.modules {
            new_seq.modules.push(module.clone_module());
        }
        new_seq.training = self.training;
        Box::new(new_seq)
    }
}

/// Parameter initialization strategies
#[derive(Debug, Clone)]
pub enum InitType {
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// Kaiming/He uniform initialization
    KaimingUniform,
    /// Kaiming/He normal initialization
    KaimingNormal,
    /// Zero initialization
    Zero,
    /// Constant initialization
    Constant(f32),
    /// Uniform random initialization
    Uniform(f32, f32),
    /// Normal random initialization
    Normal(f32, f32),
}

/// Initialize parameters according to specified strategy
pub fn init_parameter(param: &mut Tensor, init_type: InitType, fan_in: usize, fan_out: usize) {
    match init_type {
        InitType::XavierUniform => {
            let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
            *param = Tensor::rand(param.shape()) * (2.0 * limit) - limit;
        }
        InitType::XavierNormal => {
            let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
            *param = Tensor::randn(param.shape()) * std;
        }
        InitType::KaimingUniform => {
            let limit = (6.0 / fan_in as f32).sqrt();
            *param = Tensor::rand(param.shape()) * (2.0 * limit) - limit;
        }
        InitType::KaimingNormal => {
            let std = (2.0 / fan_in as f32).sqrt();
            *param = Tensor::randn(param.shape()) * std;
        }
        InitType::Zero => {
            *param = Tensor::zeros(param.shape());
        }
        InitType::Constant(value) => {
            *param = Tensor::full(param.shape(), value);
        }
        InitType::Uniform(low, high) => {
            *param = Tensor::rand(param.shape()) * (high - low) + low;
        }
        InitType::Normal(mean, std) => {
            *param = Tensor::randn(param.shape()) * std + mean;
        }
    }
}

/// Utility function to create a simple MLP
pub fn mlp(input_size: usize, hidden_sizes: &[usize], output_size: usize, activation: &str) -> Sequential {
    let mut model = Sequential::new();
    
    let mut prev_size = input_size;
    for &hidden_size in hidden_sizes {
        model = model.add(Linear::new(prev_size, hidden_size));
        model = match activation {
            "relu" => model.add(ReLU::new()),
            "sigmoid" => model.add(Sigmoid::new()),
            "tanh" => model.add(Tanh::new()),
            "gelu" => model.add(GELU::new()),
            _ => model.add(ReLU::new()),
        };
        prev_size = hidden_size;
    }
    
    model = model.add(Linear::new(prev_size, output_size));
    model
}

/// Utility function to create a simple CNN
pub fn simple_cnn(input_channels: usize, num_classes: usize) -> Sequential {
    Sequential::new()
        .add(Conv2d::new(input_channels, 32, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(32, 64, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(64, 128, 3, 1, 1))
        .add(ReLU::new())
        .add(AdaptiveAvgPool2d::new(1))
        .add(Linear::new(128, num_classes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_creation() {
        let model = Sequential::new()
            .add(Linear::new(10, 5))
            .add(ReLU::new())
            .add(Linear::new(5, 1));
        
        assert_eq!(model.len(), 3);
        assert!(model.training());
    }

    #[test]
    fn test_mlp_creation() {
        let model = mlp(784, &[256, 128], 10, "relu");
        assert_eq!(model.len(), 5); // 2 linear + 2 relu + 1 final linear
    }
}