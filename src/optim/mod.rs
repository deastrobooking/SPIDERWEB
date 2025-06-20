//! Optimization algorithms for the Rust ML framework
//! 
//! Provides optimizers similar to PyTorch's optim module with state-of-the-art
//! algorithms including SGD, Adam, AdamW, RMSprop, and more.

use crate::nn::module::Parameter;
use std::collections::HashMap;
use ndarray::ArrayD;

/// Core optimizer trait
pub trait Optimizer: Send + Sync {
    /// Perform a single optimization step
    fn step(&mut self, parameters: &mut [&mut Parameter]);
    
    /// Zero gradients for all parameters
    fn zero_grad(&mut self, parameters: &mut [&mut Parameter]) {
        for param in parameters {
            param.zero_grad();
        }
    }
    
    /// Get current learning rate
    fn get_lr(&self) -> f32;
    
    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
    
    /// Get optimizer name
    fn name(&self) -> &str;
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    momentum_buffers: HashMap<String, ArrayD<f32>>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            momentum_buffers: HashMap::new(),
        }
    }
    
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Parameter]) {
        for param in parameters {
            if !param.requires_grad || param.tensor.grad.is_none() {
                continue;
            }
            
            let grad = param.tensor.grad.as_ref().unwrap();
            let mut d_p = grad.clone();
            
            // Apply weight decay
            if self.weight_decay != 0.0 {
                d_p = d_p + &param.tensor.data * self.weight_decay;
            }
            
            // Apply momentum
            if self.momentum != 0.0 {
                let buf = self.momentum_buffers
                    .entry(param.name.clone())
                    .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
                
                *buf = buf.clone() * self.momentum + &d_p * (1.0 - self.dampening);
                
                if self.nesterov {
                    d_p = d_p + buf.clone() * self.momentum;
                } else {
                    d_p = buf.clone();
                }
            }
            
            // Update parameters
            param.tensor.data = &param.tensor.data - &d_p * self.lr;
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    fn name(&self) -> &str {
        "SGD"
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    step_count: usize,
    exp_avg: HashMap<String, ArrayD<f32>>,
    exp_avg_sq: HashMap<String, ArrayD<f32>>,
    max_exp_avg_sq: HashMap<String, ArrayD<f32>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
        }
    }
    
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
    
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut Parameter]) {
        self.step_count += 1;
        
        for param in parameters {
            if !param.requires_grad || param.tensor.grad.is_none() {
                continue;
            }
            
            let grad = param.tensor.grad.as_ref().unwrap();
            let mut grad_data = grad.clone();
            
            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad_data = grad_data + &param.tensor.data * self.weight_decay;
            }
            
            // Get or initialize moment estimates
            let exp_avg = self.exp_avg
                .entry(param.name.clone())
                .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
            
            let exp_avg_sq = self.exp_avg_sq
                .entry(param.name.clone())
                .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
            
            // Update biased first moment estimate
            *exp_avg = exp_avg.clone() * self.beta1 + &grad_data * (1.0 - self.beta1);
            
            // Update biased second raw moment estimate
            *exp_avg_sq = exp_avg_sq.clone() * self.beta2 + 
                         grad_data.mapv(|x| x * x) * (1.0 - self.beta2);
            
            let denom = if self.amsgrad {
                let max_exp_avg_sq = self.max_exp_avg_sq
                    .entry(param.name.clone())
                    .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
                
                // Maintains the maximum of all 2nd moment running avg. till now
                for (max_val, curr_val) in max_exp_avg_sq.iter_mut().zip(exp_avg_sq.iter()) {
                    *max_val = max_val.max(*curr_val);
                }
                
                max_exp_avg_sq.mapv(|x| x.sqrt() + self.eps)
            } else {
                exp_avg_sq.mapv(|x| x.sqrt() + self.eps)
            };
            
            // Bias correction
            let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
            let step_size = self.lr * (bias_correction2.sqrt() / bias_correction1);
            
            // Update parameters
            let update = exp_avg.clone() / denom * step_size;
            param.tensor.data = &param.tensor.data - &update;
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    fn name(&self) -> &str {
        "Adam"
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    step_count: usize,
    exp_avg: HashMap<String, ArrayD<f32>>,
    exp_avg_sq: HashMap<String, ArrayD<f32>>,
    max_exp_avg_sq: HashMap<String, ArrayD<f32>>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            amsgrad: false,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
        }
    }
    
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameters: &mut [&mut Parameter]) {
        self.step_count += 1;
        
        for param in parameters {
            if !param.requires_grad || param.tensor.grad.is_none() {
                continue;
            }
            
            let grad = param.tensor.grad.as_ref().unwrap();
            
            // Get or initialize moment estimates
            let exp_avg = self.exp_avg
                .entry(param.name.clone())
                .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
            
            let exp_avg_sq = self.exp_avg_sq
                .entry(param.name.clone())
                .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
            
            // Update biased first moment estimate
            *exp_avg = exp_avg.clone() * self.beta1 + grad * (1.0 - self.beta1);
            
            // Update biased second raw moment estimate
            *exp_avg_sq = exp_avg_sq.clone() * self.beta2 + 
                         grad.mapv(|x| x * x) * (1.0 - self.beta2);
            
            let denom = exp_avg_sq.mapv(|x| x.sqrt() + self.eps);
            
            // Bias correction
            let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
            let step_size = self.lr * (bias_correction2.sqrt() / bias_correction1);
            
            // Apply decoupled weight decay
            param.tensor.data = &param.tensor.data * (1.0 - self.lr * self.weight_decay);
            
            // Update parameters
            let update = exp_avg.clone() / denom * step_size;
            param.tensor.data = &param.tensor.data - &update;
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    fn name(&self) -> &str {
        "AdamW"
    }
}

/// RMSprop optimizer
pub struct RMSprop {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    square_avg: HashMap<String, ArrayD<f32>>,
    momentum_buffer: HashMap<String, ArrayD<f32>>,
    grad_avg: HashMap<String, ArrayD<f32>>,
}

impl RMSprop {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }
    
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
    
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, parameters: &mut [&mut Parameter]) {
        for param in parameters {
            if !param.requires_grad || param.tensor.grad.is_none() {
                continue;
            }
            
            let grad = param.tensor.grad.as_ref().unwrap();
            let mut grad_data = grad.clone();
            
            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad_data = grad_data + &param.tensor.data * self.weight_decay;
            }
            
            // Get or initialize state
            let square_avg = self.square_avg
                .entry(param.name.clone())
                .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
            
            // Update square average
            *square_avg = square_avg.clone() * self.alpha + 
                         grad_data.mapv(|x| x * x) * (1.0 - self.alpha);
            
            let avg = if self.centered {
                let grad_avg = self.grad_avg
                    .entry(param.name.clone())
                    .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
                
                *grad_avg = grad_avg.clone() * self.alpha + &grad_data * (1.0 - self.alpha);
                
                // Centered RMSprop
                (square_avg.clone() - grad_avg.mapv(|x| x * x)).mapv(|x| (x + self.eps).sqrt())
            } else {
                square_avg.mapv(|x| (x + self.eps).sqrt())
            };
            
            let update = if self.momentum > 0.0 {
                let buf = self.momentum_buffer
                    .entry(param.name.clone())
                    .or_insert_with(|| ArrayD::zeros(param.tensor.data.raw_dim()));
                
                *buf = buf.clone() * self.momentum + &grad_data / &avg;
                buf.clone()
            } else {
                grad_data / avg
            };
            
            // Update parameters
            param.tensor.data = &param.tensor.data - &update * self.lr;
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    fn name(&self) -> &str {
        "RMSprop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::ArrayD;

    #[test]
    fn test_sgd_optimizer() {
        let mut sgd = SGD::new(0.01);
        
        // Create test parameter
        let data = ArrayD::ones([2, 2].as_slice());
        let grad = ArrayD::ones([2, 2].as_slice()) * 0.1;
        let mut tensor = Tensor::new(data.clone(), true);
        tensor.grad = Some(grad);
        
        let mut param = Parameter::new(tensor, true, "test".to_string());
        let mut params = vec![&mut param];
        
        let initial_data = param.tensor.data.clone();
        sgd.step(&mut params);
        
        // Check that parameters were updated
        assert_ne!(param.tensor.data, initial_data);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut adam = Adam::new(0.001);
        
        // Create test parameter
        let data = ArrayD::ones([2, 2].as_slice());
        let grad = ArrayD::ones([2, 2].as_slice()) * 0.1;
        let mut tensor = Tensor::new(data.clone(), true);
        tensor.grad = Some(grad);
        
        let mut param = Parameter::new(tensor, true, "test".to_string());
        let mut params = vec![&mut param];
        
        let initial_data = param.tensor.data.clone();
        adam.step(&mut params);
        
        // Check that parameters were updated
        assert_ne!(param.tensor.data, initial_data);
    }
}