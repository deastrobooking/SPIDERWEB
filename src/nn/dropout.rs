//! Dropout layer implementation

use crate::tensor::Tensor;
use crate::nn::Module;
use anyhow::Result;
use rand::Rng;

/// Dropout layer for regularization
#[derive(Debug, Clone)]
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self {
            p: p.max(0.0).min(1.0), // Clamp between 0 and 1
            training: true,
        }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone_tensor());
        }
        
        if self.p == 1.0 {
            return Ok(Tensor::zeros(input.shape()));
        }
        
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);
        
        let mask_data = input.data().iter()
            .map(|_| {
                if rng.gen::<f32>() < self.p {
                    0.0
                } else {
                    scale
                }
            })
            .collect::<Vec<f32>>();
        
        let mask = Tensor::from_vec(mask_data, input.shape())?;
        Ok(input * &mask)
    }
    
    fn parameters(&self) -> Vec<&Tensor> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { Vec::new() }
    
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
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_eval_mode() {
        let mut dropout = Dropout::new(0.5);
        dropout.eval();
        
        let input = Tensor::ones(&[4, 4]);
        let output = dropout.forward(&input).unwrap();
        
        // In eval mode, output should be identical to input
        assert_eq!(output.data(), input.data());
    }

    #[test]
    fn test_dropout_zero_probability() {
        let dropout = Dropout::new(0.0);
        let input = Tensor::ones(&[4, 4]);
        let output = dropout.forward(&input).unwrap();
        
        // With p=0, no dropout should occur
        assert_eq!(output.data(), input.data());
    }
}