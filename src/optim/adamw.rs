//! AdamW optimizer implementation (Adam with decoupled weight decay)

use crate::tensor::Tensor;
use crate::optim::Optimizer;
use anyhow::Result;
use std::collections::HashMap;

/// AdamW optimizer with decoupled weight decay
pub struct AdamW {
    params: Vec<*mut Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    momentum: HashMap<usize, Tensor>,
    velocity: HashMap<usize, Tensor>,
}

impl AdamW {
    /// Create new AdamW optimizer
    pub fn new(params: Vec<*mut Tensor>, lr: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step_count: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        }
    }
    
    /// Create AdamW with custom parameters
    pub fn with_params(params: Vec<*mut Tensor>, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        
        for (i, param_ptr) in self.params.iter().enumerate() {
            unsafe {
                let param = &mut **param_ptr;
                
                if let Some(grad) = param.grad() {
                    // Decoupled weight decay (applied directly to parameters)
                    if self.weight_decay != 0.0 {
                        *param = param * (1.0 - self.lr * self.weight_decay);
                    }
                    
                    // Initialize momentum and velocity if needed
                    let m = self.momentum.entry(i).or_insert_with(|| {
                        Tensor::zeros(param.shape())
                    });
                    let v = self.velocity.entry(i).or_insert_with(|| {
                        Tensor::zeros(param.shape())
                    });
                    
                    // Update biased first moment estimate
                    *m = &(*m * self.beta1) + &(grad * (1.0 - self.beta1));
                    
                    // Update biased second raw moment estimate
                    let grad_squared = grad * grad;
                    *v = &(*v * self.beta2) + &(&grad_squared * (1.0 - self.beta2));
                    
                    // Compute bias-corrected first moment estimate
                    let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
                    let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
                    
                    let m_hat = m * (1.0 / bias_correction1);
                    let v_hat = v * (1.0 / bias_correction2);
                    
                    // Update parameters
                    let denominator = v_hat.sqrt() + self.eps;
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

unsafe impl Send for AdamW {}
unsafe impl Sync for AdamW {}