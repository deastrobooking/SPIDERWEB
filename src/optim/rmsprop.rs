//! RMSprop optimizer implementation

use crate::tensor::Tensor;
use crate::optim::Optimizer;
use anyhow::Result;
use std::collections::HashMap;

/// RMSprop optimizer
pub struct RMSprop {
    params: Vec<*mut Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    square_avg: HashMap<usize, Tensor>,
    momentum_buffer: HashMap<usize, Tensor>,
    grad_avg: HashMap<usize, Tensor>,
}

impl RMSprop {
    pub fn new(params: Vec<*mut Tensor>, lr: f32) -> Self {
        Self {
            params,
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
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> Result<()> {
        for (i, param_ptr) in self.params.iter().enumerate() {
            unsafe {
                let param = &mut **param_ptr;
                if let Some(grad) = param.grad() {
                    let mut grad_data = grad.clone_tensor();
                    
                    if self.weight_decay != 0.0 {
                        grad_data = &grad_data + &(param * self.weight_decay);
                    }
                    
                    let square_avg = self.square_avg.entry(i).or_insert_with(|| {
                        Tensor::zeros(param.shape())
                    });
                    
                    *square_avg = &(*square_avg * self.alpha) + &(&(&grad_data * &grad_data) * (1.0 - self.alpha));
                    
                    let avg = if self.centered {
                        let grad_avg = self.grad_avg.entry(i).or_insert_with(|| {
                            Tensor::zeros(param.shape())
                        });
                        *grad_avg = &(*grad_avg * self.alpha) + &(&grad_data * (1.0 - self.alpha));
                        
                        let centered_var = square_avg - &(grad_avg * grad_avg);
                        (centered_var + self.eps).sqrt()
                    } else {
                        (square_avg + self.eps).sqrt()
                    };
                    
                    if self.momentum > 0.0 {
                        let buf = self.momentum_buffer.entry(i).or_insert_with(|| {
                            Tensor::zeros(param.shape())
                        });
                        *buf = &(*buf * self.momentum) + &(&grad_data / &avg);
                        *param = param - &(buf * self.lr);
                    } else {
                        *param = param - &(&(&grad_data / &avg) * self.lr);
                    }
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
        state.insert("alpha".to_string(), self.alpha);
        state.insert("eps".to_string(), self.eps);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("momentum".to_string(), self.momentum);
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("lr") { self.lr = lr; }
        if let Some(&alpha) = state.get("alpha") { self.alpha = alpha; }
        if let Some(&eps) = state.get("eps") { self.eps = eps; }
        if let Some(&weight_decay) = state.get("weight_decay") { self.weight_decay = weight_decay; }
        if let Some(&momentum) = state.get("momentum") { self.momentum = momentum; }
    }
}

unsafe impl Send for RMSprop {}
unsafe impl Sync for RMSprop {}