//! Adagrad optimizer implementation

use crate::tensor::Tensor;
use crate::optim::Optimizer;
use anyhow::Result;
use std::collections::HashMap;

/// Adagrad optimizer
pub struct Adagrad {
    params: Vec<*mut Tensor>,
    lr: f32,
    lr_decay: f32,
    weight_decay: f32,
    eps: f32,
    sum_squares: HashMap<usize, Tensor>,
    step_count: usize,
}

impl Adagrad {
    pub fn new(params: Vec<*mut Tensor>, lr: f32) -> Self {
        Self {
            params,
            lr,
            lr_decay: 0.0,
            weight_decay: 0.0,
            eps: 1e-10,
            sum_squares: HashMap::new(),
            step_count: 0,
        }
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let clr = self.lr / (1.0 + (self.step_count as f32 - 1.0) * self.lr_decay);
        
        for (i, param_ptr) in self.params.iter().enumerate() {
            unsafe {
                let param = &mut **param_ptr;
                if let Some(grad) = param.grad() {
                    let mut grad_data = grad.clone_tensor();
                    
                    if self.weight_decay != 0.0 {
                        grad_data = &grad_data + &(param * self.weight_decay);
                    }
                    
                    let sum_sq = self.sum_squares.entry(i).or_insert_with(|| {
                        Tensor::zeros(param.shape())
                    });
                    
                    *sum_sq = sum_sq + &(&grad_data * &grad_data);
                    let std = sum_sq.sqrt() + self.eps;
                    *param = param - &(&(&grad_data / &std) * clr);
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
    
    fn learning_rate(&self) -> f32 { self.lr }
    fn set_learning_rate(&mut self, lr: f32) { self.lr = lr; }
    fn add_param_group(&mut self, params: Vec<*mut Tensor>) { self.params.extend(params); }
    
    fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("lr".to_string(), self.lr);
        state.insert("lr_decay".to_string(), self.lr_decay);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("eps".to_string(), self.eps);
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("lr") { self.lr = lr; }
        if let Some(&lr_decay) = state.get("lr_decay") { self.lr_decay = lr_decay; }
        if let Some(&weight_decay) = state.get("weight_decay") { self.weight_decay = weight_decay; }
        if let Some(&eps) = state.get("eps") { self.eps = eps; }
    }
}

unsafe impl Send for Adagrad {}
unsafe impl Sync for Adagrad {}