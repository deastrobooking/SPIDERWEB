//! Linear/Dense layer implementation

use crate::tensor::Tensor;
use crate::nn::{Module, InitType, init_parameter};
use anyhow::Result;

/// Linear/Dense layer (fully connected layer)
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weight = Tensor::zeros(&[out_features, in_features]);
        let mut bias = Tensor::zeros(&[out_features]);
        
        // Initialize weights with Kaiming uniform initialization
        init_parameter(&mut weight, InitType::KaimingUniform, in_features, out_features);
        init_parameter(&mut bias, InitType::Zero, in_features, out_features);
        
        Self {
            weight: weight.requires_grad(true),
            bias: Some(bias.requires_grad(true)),
            in_features,
            out_features,
            training: true,
        }
    }
    
    /// Create a linear layer without bias
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        let mut weight = Tensor::zeros(&[out_features, in_features]);
        init_parameter(&mut weight, InitType::KaimingUniform, in_features, out_features);
        
        Self {
            weight: weight.requires_grad(true),
            bias: None,
            in_features,
            out_features,
            training: true,
        }
    }
    
    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }
    
    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }
    
    /// Get weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    
    /// Get bias tensor
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input shape: (batch_size, in_features) or (..., in_features)
        // Weight shape: (out_features, in_features)
        // Output shape: (batch_size, out_features) or (..., out_features)
        
        let output = input.matmul(&self.weight.t())?;
        
        if let Some(ref bias) = self.bias {
            // Broadcast bias across batch dimension
            Ok(&output + bias)
        } else {
            Ok(output)
        }
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
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
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let layer = Linear::new(10, 5);
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert_eq!(layer.weight().shape(), &[5, 10]);
        assert!(layer.bias().is_some());
    }

    #[test]
    fn test_linear_forward() {
        let layer = Linear::new(3, 2);
        let input = Tensor::ones(&[4, 3]); // batch_size=4, features=3
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 2]);
    }
}