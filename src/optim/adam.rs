//! Adam optimizer implementation

use crate::tensor::Tensor;
use crate::optim::Optimizer;
use anyhow::Result;
use std::collections::HashMap;

/// Adam optimizer with adaptive learning rates
pub struct Adam {
    params: Vec<*mut Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    step_count: usize,
    momentum: HashMap<usize, Tensor>,
    velocity: HashMap<usize, Tensor>,
    max_velocity: HashMap<usize, Tensor>,
}

impl Adam {
    /// Create new Adam optimizer
    pub fn new(params: Vec<*mut Tensor>, lr: f32) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            max_velocity: HashMap::new(),
        }
    }
    
    /// Create Adam with custom parameters
    pub fn with_params(params: Vec<*mut Tensor>, lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            max_velocity: HashMap::new(),
        }
    }
    
    /// Create Adam with weight decay (AdamW-like behavior)
    pub fn with_weight_decay(params: Vec<*mut Tensor>, lr: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            amsgrad: false,
            step_count: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            max_velocity: HashMap::new(),
        }
    }
    
    /// Create AMSGrad variant
    pub fn amsgrad(params: Vec<*mut Tensor>, lr: f32) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: true,
            step_count: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            max_velocity: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        
        for (i, param_ptr) in self.params.iter().enumerate() {
            unsafe {
                let param = &mut **param_ptr;
                
                if let Some(grad) = param.grad() {
                    let mut grad_data = grad.clone_tensor();
                    
                    // Add weight decay
                    if self.weight_decay != 0.0 {
                        grad_data = &grad_data + &(param * self.weight_decay);
                    }
                    
                    // Initialize momentum and velocity if needed
                    let m = self.momentum.entry(i).or_insert_with(|| {
                        Tensor::zeros(param.shape())
                    });
                    let v = self.velocity.entry(i).or_insert_with(|| {
                        Tensor::zeros(param.shape())
                    });
                    
                    // Update biased first moment estimate
                    *m = &(*m * self.beta1) + &(&grad_data * (1.0 - self.beta1));
                    
                    // Update biased second raw moment estimate
                    let grad_squared = &grad_data * &grad_data;
                    *v = &(*v * self.beta2) + &(&grad_squared * (1.0 - self.beta2));
                    
                    // Compute bias-corrected first moment estimate
                    let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
                    let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
                    
                    let m_hat = m * (1.0 / bias_correction1);
                    let v_hat = v * (1.0 / bias_correction2);
                    
                    let denominator = if self.amsgrad {
                        let v_max = self.max_velocity.entry(i).or_insert_with(|| {
                            Tensor::zeros(param.shape())
                        });
                        
                        // Element-wise maximum
                        let v_max_data = v_hat.data().iter()
                            .zip(v_max.data().iter())
                            .map(|(v_hat_val, v_max_val)| v_hat_val.max(*v_max_val))
                            .collect::<Vec<f32>>();
                        *v_max = Tensor::from_vec(v_max_data, param.shape())?;
                        
                        v_max.sqrt() + self.eps
                    } else {
                        v_hat.sqrt() + self.eps
                    };
                    
                    // Update parameters
                    let update = &m_hat / &denominator;
                    *param = param - &(&update * self.lr);
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
        state.insert("beta1".to_string(), self.beta1);
        state.insert("beta2".to_string(), self.beta2);
        state.insert("eps".to_string(), self.eps);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("step_count".to_string(), self.step_count as f32);
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("lr") {
            self.lr = lr;
        }
        if let Some(&beta1) = state.get("beta1") {
            self.beta1 = beta1;
        }
        if let Some(&beta2) = state.get("beta2") {
            self.beta2 = beta2;
        }
        if let Some(&eps) = state.get("eps") {
            self.eps = eps;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&step_count) = state.get("step_count") {
            self.step_count = step_count as usize;
        }
    }
}

unsafe impl Send for Adam {}
unsafe impl Sync for Adam {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_adam_creation() {
        let mut param = Tensor::ones(&[2, 2]).requires_grad(true);
        let adam = Adam::new(vec![&mut param as *mut Tensor], 0.001);
        assert_eq!(adam.learning_rate(), 0.001);
        assert_eq!(adam.beta1, 0.9);
        assert_eq!(adam.beta2, 0.999);
    }
}