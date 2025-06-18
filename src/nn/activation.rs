//! Activation function implementations

use crate::tensor::Tensor;
use crate::nn::Module;
use anyhow::Result;

/// ReLU activation function
#[derive(Debug, Clone)]
pub struct ReLU {
    training: bool,
}

impl ReLU {
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // ReLU(x) = max(0, x)
        let zero = Tensor::zeros(input.shape());
        let output_data = input.data().iter()
            .zip(zero.data().iter())
            .map(|(x, z)| x.max(*z))
            .collect::<Vec<f32>>();
        
        Tensor::from_vec(output_data, input.shape())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
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
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Sigmoid activation function
#[derive(Debug, Clone)]
pub struct Sigmoid {
    training: bool,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Sigmoid(x) = 1 / (1 + exp(-x))
        let neg_input = -input;
        let exp_neg = neg_input.exp();
        let one = Tensor::ones(input.shape());
        let denominator = &one + &exp_neg;
        Ok(&one / &denominator)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
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
        "Sigmoid"
    }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Tanh activation function
#[derive(Debug, Clone)]
pub struct Tanh {
    training: bool,
}

impl Tanh {
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let two_x = input * &Tensor::full(input.shape(), 2.0);
        let exp_2x = two_x.exp();
        let one = Tensor::ones(input.shape());
        let numerator = &exp_2x - &one;
        let denominator = &exp_2x + &one;
        Ok(&numerator / &denominator)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
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
        "Tanh"
    }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Softmax activation function
#[derive(Debug, Clone)]
pub struct Softmax {
    dim: usize,
    training: bool,
}

impl Softmax {
    pub fn new(dim: usize) -> Self {
        Self { dim, training: true }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Softmax(x_i) = exp(x_i) / sum(exp(x_j))
        // For numerical stability, subtract max before exp
        let max_val = input.max();
        let shifted = input - &Tensor::full(input.shape(), max_val);
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum_axis(self.dim)?;
        
        // Broadcast sum back to original shape
        let output_data = exp_vals.data().iter()
            .zip(sum_exp.data().iter().cycle())
            .map(|(exp_val, sum_val)| exp_val / sum_val)
            .collect::<Vec<f32>>();
            
        Tensor::from_vec(output_data, input.shape())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
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
        "Softmax"
    }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// GELU activation function (Gaussian Error Linear Unit)
#[derive(Debug, Clone)]
pub struct GELU {
    training: bool,
}

impl GELU {
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = input.pow(3.0);
        let term = input + &(&x_cubed * 0.044715);
        let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
        let tanh_input = &term * sqrt_2_pi;
        
        // Compute tanh manually
        let exp_2x = (&tanh_input * 2.0).exp();
        let one = Tensor::ones(input.shape());
        let tanh_val = (&exp_2x - &one) / (&exp_2x + &one);
        
        let result = input * 0.5 * (&one + &tanh_val);
        Ok(result)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
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
        "GELU"
    }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Swish activation function (SiLU)
#[derive(Debug, Clone)]
pub struct Swish {
    training: bool,
}

impl Swish {
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Module for Swish {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Swish(x) = x * sigmoid(x)
        let sigmoid = input.clone_tensor();
        let neg_input = -&sigmoid;
        let exp_neg = neg_input.exp();
        let one = Tensor::ones(input.shape());
        let sigmoid_output = &one / (&one + &exp_neg);
        
        Ok(input * &sigmoid_output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
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
        "Swish"
    }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let relu = ReLU::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let output = relu.forward(&input).unwrap();
        let expected = vec![0.0, 0.0, 1.0, 2.0];
        assert_eq!(output.to_vec(), expected);
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid::new();
        let input = Tensor::zeros(&[2, 2]);
        let output = sigmoid.forward(&input).unwrap();
        // sigmoid(0) = 0.5
        assert!((output.to_vec()[0] - 0.5).abs() < 1e-6);
    }
}