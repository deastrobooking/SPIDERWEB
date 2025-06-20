//! Optimization algorithms for training neural networks
//! 
//! This module provides optimizers similar to PyTorch's optim module
//! and TensorFlow's optimizers.

pub mod sgd;
pub mod adam;
pub mod adamw;
pub mod rmsprop;
pub mod adagrad;

use crate::tensor::Tensor;
use anyhow::Result;
use std::collections::HashMap;

// Re-export optimizers
pub use sgd::SGD;
pub use adam::Adam;
pub use adamw::AdamW;
pub use rmsprop::RMSprop;
pub use adagrad::Adagrad;

/// Base trait for all optimizers
pub trait Optimizer: Send + Sync {
    /// Perform one optimization step
    fn step(&mut self) -> Result<()>;
    
    /// Zero out all gradients
    fn zero_grad(&mut self);
    
    /// Get learning rate
    fn learning_rate(&self) -> f32;
    
    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);
    
    /// Add parameter group
    fn add_param_group(&mut self, params: Vec<*mut Tensor>);
    
    /// Get optimizer state
    fn state_dict(&self) -> HashMap<String, f32>;
    
    /// Load optimizer state
    fn load_state_dict(&mut self, state: HashMap<String, f32>);
}

/// Learning rate scheduler trait
pub trait LRScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer);
    fn get_lr(&self) -> f32;
}

/// Step learning rate scheduler
pub struct StepLR {
    step_size: usize,
    gamma: f32,
    current_step: usize,
    base_lr: f32,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f32, base_lr: f32) -> Self {
        Self {
            step_size,
            gamma,
            current_step: 0,
            base_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        if self.current_step % self.step_size == 0 {
            let new_lr = optimizer.learning_rate() * self.gamma;
            optimizer.set_learning_rate(new_lr);
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.base_lr * self.gamma.powi((self.current_step / self.step_size) as i32)
    }
}

/// Exponential learning rate scheduler
pub struct ExponentialLR {
    gamma: f32,
    base_lr: f32,
}

impl ExponentialLR {
    pub fn new(gamma: f32, base_lr: f32) -> Self {
        Self { gamma, base_lr }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        let new_lr = optimizer.learning_rate() * self.gamma;
        optimizer.set_learning_rate(new_lr);
    }
    
    fn get_lr(&self) -> f32 {
        self.base_lr * self.gamma
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f32,
    current_step: usize,
    base_lr: f32,
}

impl CosineAnnealingLR {
    pub fn new(t_max: usize, eta_min: f32, base_lr: f32) -> Self {
        Self {
            t_max,
            eta_min,
            current_step: 0,
            base_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.eta_min + (self.base_lr - self.eta_min) * 
            (1.0 + (std::f32::consts::PI * self.current_step as f32 / self.t_max as f32).cos()) / 2.0;
        optimizer.set_learning_rate(lr);
    }
    
    fn get_lr(&self) -> f32 {
        self.eta_min + (self.base_lr - self.eta_min) * 
            (1.0 + (std::f32::consts::PI * self.current_step as f32 / self.t_max as f32).cos()) / 2.0
    }
}