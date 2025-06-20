//! Neural network module system for the Rust ML framework
//! 
//! Provides a PyTorch-like nn.Module interface for building neural networks
//! with automatic parameter management and gradient computation.

use crate::tensor::Tensor;
use crate::autograd::{TensorId, AutogradEngine};
use std::collections::HashMap;
use std::sync::Arc;
use ndarray::ArrayD;

/// Core trait for all neural network modules
pub trait Module: Send + Sync {
    /// Forward pass through the module
    fn forward(&self, input: &Tensor) -> Tensor;
    
    /// Get all trainable parameters
    fn parameters(&self) -> Vec<&Parameter>;
    
    /// Get all trainable parameters (mutable)
    fn parameters_mut(&mut self) -> Vec<&mut Parameter>;
    
    /// Zero out all gradients
    fn zero_grad(&mut self);
    
    /// Set module to training mode
    fn train(&mut self);
    
    /// Set module to evaluation mode
    fn eval(&mut self);
    
    /// Check if module is in training mode
    fn training(&self) -> bool;
    
    /// Get module name for debugging
    fn name(&self) -> &str;
}

/// Trainable parameter with gradient tracking
#[derive(Debug, Clone)]
pub struct Parameter {
    pub tensor: Tensor,
    pub requires_grad: bool,
    pub name: String,
}

impl Parameter {
    pub fn new(tensor: Tensor, requires_grad: bool, name: String) -> Self {
        Self {
            tensor,
            requires_grad,
            name,
        }
    }
    
    pub fn zero_grad(&mut self) {
        if let Some(ref mut grad) = self.tensor.grad {
            grad.fill(0.0);
        }
    }
    
    pub fn has_grad(&self) -> bool {
        self.tensor.grad.is_some()
    }
}

/// Linear (fully connected) layer
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        use ndarray_rand::RandomExt;
        use rand::distributions::StandardNormal;
        
        // Xavier/Glorot initialization
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight_data = ArrayD::random([out_features, in_features].as_slice(), StandardNormal)
            .mapv(|x: f32| x * std);
        
        let weight_tensor = Tensor::new(weight_data, true);
        let weight = Parameter::new(weight_tensor, true, "weight".to_string());
        
        let bias_param = if bias {
            let bias_data = ArrayD::zeros([out_features].as_slice());
            let bias_tensor = Tensor::new(bias_data, true);
            Some(Parameter::new(bias_tensor, true, "bias".to_string()))
        } else {
            None
        };
        
        Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Matrix multiplication: output = input @ weight.T
        let output = input.matmul(&self.weight.tensor.transpose());
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            output.add(&bias.tensor)
        } else {
            output
        }
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }
    
    fn train(&mut self) {
        self.training = true;
    }
    
    fn eval(&mut self) {
        self.training = false;
    }
    
    fn training(&self) -> bool {
        self.training
    }
    
    fn name(&self) -> &str {
        "Linear"
    }
}

/// Sequential container for chaining modules
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }
    
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }
    
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, mut input: &Tensor) -> Tensor {
        let mut current = input.clone();
        
        for module in &self.modules {
            current = module.forward(&current);
        }
        
        current
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut params = Vec::new();
        for module in &mut self.modules {
            params.extend(module.parameters_mut());
        }
        params
    }
    
    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
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
}

/// ReLU activation function
pub struct ReLU {
    training: bool,
}

impl ReLU {
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
    
    fn zero_grad(&mut self) {
        // No parameters, nothing to zero
    }
    
    fn train(&mut self) {
        self.training = true;
    }
    
    fn eval(&mut self) {
        self.training = false;
    }
    
    fn training(&self) -> bool {
        self.training
    }
    
    fn name(&self) -> &str {
        "ReLU"
    }
}

/// Dropout layer for regularization
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p <= 1.0, "Dropout probability must be between 0 and 1");
        Self { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training && self.p > 0.0 {
            input.dropout(self.p)
        } else {
            input.clone()
        }
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
    
    fn zero_grad(&mut self) {
        // No parameters, nothing to zero
    }
    
    fn train(&mut self) {
        self.training = true;
    }
    
    fn eval(&mut self) {
        self.training = false;
    }
    
    fn training(&self) -> bool {
        self.training
    }
    
    fn name(&self) -> &str {
        "Dropout"
    }
}

/// Batch normalization layer
pub struct BatchNorm1d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    weight: Parameter,
    bias: Parameter,
    running_mean: ArrayD<f32>,
    running_var: ArrayD<f32>,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        let weight_data = ArrayD::ones([num_features].as_slice());
        let weight_tensor = Tensor::new(weight_data, true);
        let weight = Parameter::new(weight_tensor, true, "weight".to_string());
        
        let bias_data = ArrayD::zeros([num_features].as_slice());
        let bias_tensor = Tensor::new(bias_data, true);
        let bias = Parameter::new(bias_tensor, true, "bias".to_string());
        
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            weight,
            bias,
            running_mean: ArrayD::zeros([num_features].as_slice()),
            running_var: ArrayD::ones([num_features].as_slice()),
            training: true,
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training {
            // Training mode: compute batch statistics
            input.batch_norm(&self.weight.tensor, &self.bias.tensor, self.eps)
        } else {
            // Evaluation mode: use running statistics
            input.batch_norm_eval(&self.weight.tensor, &self.bias.tensor, 
                                 &self.running_mean, &self.running_var, self.eps)
        }
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight, &mut self.bias]
    }
    
    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }
    
    fn train(&mut self) {
        self.training = true;
    }
    
    fn eval(&mut self) {
        self.training = false;
    }
    
    fn training(&self) -> bool {
        self.training
    }
    
    fn name(&self) -> &str {
        "BatchNorm1d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_linear_layer() {
        let linear = Linear::new(784, 128, true);
        let input_data = ArrayD::ones([1, 784].as_slice());
        let input = Tensor::new(input_data, false);
        
        let output = linear.forward(&input);
        assert_eq!(output.shape(), &[1, 128]);
    }

    #[test]
    fn test_sequential() {
        let model = Sequential::new()
            .add(Linear::new(784, 128, true))
            .add(ReLU::new())
            .add(Linear::new(128, 10, true));
        
        let input_data = ArrayD::ones([1, 784].as_slice());
        let input = Tensor::new(input_data, false);
        
        let output = model.forward(&input);
        assert_eq!(output.shape(), &[1, 10]);
    }

    #[test]
    fn test_parameters() {
        let mut linear = Linear::new(10, 5, true);
        let params = linear.parameters();
        assert_eq!(params.len(), 2); // weight + bias
        
        linear.zero_grad();
        // Verify gradients are zeroed
    }
}