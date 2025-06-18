//! Stochastic Gradient Descent optimizer

use crate::tensor::Tensor;
use crate::optim::Optimizer;
use anyhow::Result;
use std::collections::HashMap;

/// SGD optimizer with momentum support
pub struct SGD {
    params: Vec<*mut Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
    velocity: HashMap<usize, Tensor>,
}

impl SGD {
    /// Create new SGD optimizer
    pub fn new(params: Vec<*mut Tensor>, lr: f32) -> Self {
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }
    
    /// Create SGD with momentum
    pub fn with_momentum(params: Vec<*mut Tensor>, lr: f32, momentum: f32) -> Self {
        Self {
            params,
            lr,
            momentum,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }
    
    /// Create SGD with weight decay
    pub fn with_weight_decay(params: Vec<*mut Tensor>, lr: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }
    
    /// Create SGD with Nesterov momentum
    pub fn with_nesterov(params: Vec<*mut Tensor>, lr: f32, momentum: f32) -> Self {
        Self {
            params,
            lr,
            momentum,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: true,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        for (i, param_ptr) in self.params.iter().enumerate() {
            unsafe {
                let param = &mut **param_ptr;
                
                if let Some(grad) = param.grad() {
                    let mut d_p = grad.clone_tensor();
                    
                    // Add weight decay
                    if self.weight_decay != 0.0 {
                        d_p = &d_p + &(param * self.weight_decay);
                    }
                    
                    // Apply momentum
                    if self.momentum != 0.0 {
                        let buf = self.velocity.entry(i).or_insert_with(|| {
                            Tensor::zeros(param.shape())
                        });
                        
                        *buf = &(*buf * self.momentum) + &(&d_p * (1.0 - self.dampening));
                        
                        if self.nesterov {
                            d_p = &d_p + &(buf * self.momentum);
                        } else {
                            d_p = buf.clone_tensor();
                        }
                    }
                    
                    // Update parameters
                    *param = param - &(&d_p * self.lr);
                }
            }
        }
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        for param_ptr in &self.params {
            unsafe {
                let param = &mut **param_ptr;
                param.zero_grad();
            }
        }
    }
    
    fn learning_rate(&self) -> f32 {
        self.lr
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    fn add_param_group(&mut self, params: Vec<*mut Tensor>) {
        self.params.extend(params);
    }
    
    fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("lr".to_string(), self.lr);
        state.insert("momentum".to_string(), self.momentum);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("dampening".to_string(), self.dampening);
        state.insert("nesterov".to_string(), if self.nesterov { 1.0 } else { 0.0 });
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("lr") {
            self.lr = lr;
        }
        if let Some(&momentum) = state.get("momentum") {
            self.momentum = momentum;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&dampening) = state.get("dampening") {
            self.dampening = dampening;
        }
        if let Some(&nesterov) = state.get("nesterov") {
            self.nesterov = nesterov != 0.0;
        }
    }
}

unsafe impl Send for SGD {}
unsafe impl Sync for SGD {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_sgd_creation() {
        let mut param = Tensor::ones(&[2, 2]).requires_grad(true);
        let sgd = SGD::new(vec![&mut param as *mut Tensor], 0.01);
        assert_eq!(sgd.learning_rate(), 0.01);
    }
}