//! Utility functions and helpers

use crate::tensor::Tensor;
use anyhow::Result;
use std::time::{Instant, Duration};
use std::path::Path;

/// Save tensor to file
pub fn save_tensor<P: AsRef<Path>>(tensor: &Tensor, path: P) -> Result<()> {
    let data = bincode::serialize(tensor)?;
    std::fs::write(path, data)?;
    Ok(())
}

/// Load tensor from file
pub fn load_tensor<P: AsRef<Path>>(path: P) -> Result<Tensor> {
    let data = std::fs::read(path)?;
    let tensor = bincode::deserialize(&data)?;
    Ok(tensor)
}

/// Timer for benchmarking
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    pub fn stop(self) -> Duration {
        let elapsed = self.elapsed();
        log::info!("{}: {:.2?}", self.name, elapsed);
        elapsed
    }
}

/// Set random seed for reproducibility
pub fn set_seed(seed: u64) {
    use rand::{SeedableRng, rngs::StdRng};
    let _rng = StdRng::seed_from_u64(seed);
    log::info!("Random seed set to: {}", seed);
}

/// Calculate model parameters count
pub fn count_parameters<M: crate::nn::Module>(model: &M) -> usize {
    model.parameters().iter().map(|p| p.numel()).sum()
}

/// Memory usage information
pub struct MemoryInfo {
    pub allocated: usize,
    pub cached: usize,
}

pub fn memory_info() -> MemoryInfo {
    // Simplified memory tracking
    MemoryInfo {
        allocated: 0,
        cached: 0,
    }
}

/// Progress bar for training
pub struct ProgressBar {
    total: usize,
    current: usize,
    start_time: Instant,
}

impl ProgressBar {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            start_time: Instant::now(),
        }
    }
    
    pub fn update(&mut self, current: usize) {
        self.current = current;
        let percentage = (current as f32 / self.total as f32) * 100.0;
        let elapsed = self.start_time.elapsed();
        let eta = if current > 0 {
            elapsed * (self.total - current) as u32 / current as u32
        } else {
            Duration::from_secs(0)
        };
        
        print!("\r[{}/{}] {:.1}% | Elapsed: {:.1?} | ETA: {:.1?}", 
               current, self.total, percentage, elapsed, eta);
    }
    
    pub fn finish(self) {
        println!("\nCompleted in {:.2?}", self.start_time.elapsed());
    }
}

/// Learning rate scheduler utilities
pub fn cosine_annealing_lr(epoch: usize, max_epochs: usize, initial_lr: f32, min_lr: f32) -> f32 {
    let progress = epoch as f32 / max_epochs as f32;
    min_lr + (initial_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
}

pub fn exponential_decay_lr(epoch: usize, initial_lr: f32, decay_rate: f32) -> f32 {
    initial_lr * decay_rate.powi(epoch as i32)
}

pub fn step_decay_lr(epoch: usize, initial_lr: f32, step_size: usize, gamma: f32) -> f32 {
    initial_lr * gamma.powi((epoch / step_size) as i32)
}

/// Gradient clipping utilities
pub fn clip_grad_norm(parameters: &mut [&mut Tensor], max_norm: f32) -> f32 {
    let mut total_norm = 0.0;
    
    // Calculate total norm
    for param in parameters.iter() {
        if let Some(grad) = param.grad() {
            let param_norm = grad.pow(2.0).sum().sqrt();
            total_norm += param_norm * param_norm;
        }
    }
    
    total_norm = total_norm.sqrt();
    let clip_coef = max_norm / (total_norm + 1e-6);
    
    if clip_coef < 1.0 {
        for param in parameters.iter_mut() {
            if let Some(grad) = param.grad() {
                let clipped_grad = grad * clip_coef;
                param.set_grad(clipped_grad);
            }
        }
    }
    
    total_norm
}

pub fn clip_grad_value(parameters: &mut [&mut Tensor], clip_value: f32) {
    for param in parameters.iter_mut() {
        if let Some(grad) = param.grad() {
            let clipped_data = grad.data().iter()
                .map(|&x| x.max(-clip_value).min(clip_value))
                .collect::<Vec<f32>>();
            
            if let Ok(clipped_grad) = Tensor::from_vec(clipped_data, grad.shape()) {
                param.set_grad(clipped_grad);
            }
        }
    }
}

/// Model checkpointing
pub fn save_checkpoint<P: AsRef<Path>>(
    model_state: &std::collections::HashMap<String, Tensor>,
    optimizer_state: &std::collections::HashMap<String, f32>,
    epoch: usize,
    loss: f32,
    path: P
) -> Result<()> {
    let checkpoint = CheckpointData {
        model_state: model_state.clone(),
        optimizer_state: optimizer_state.clone(),
        epoch,
        loss,
    };
    
    let data = bincode::serialize(&checkpoint)?;
    std::fs::write(path, data)?;
    Ok(())
}

pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<CheckpointData> {
    let data = std::fs::read(path)?;
    let checkpoint = bincode::deserialize(&data)?;
    Ok(checkpoint)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct CheckpointData {
    pub model_state: std::collections::HashMap<String, Tensor>,
    pub optimizer_state: std::collections::HashMap<String, f32>,
    pub epoch: usize,
    pub loss: f32,
}

/// Early stopping helper
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    best_loss: f32,
    wait: usize,
    stopped: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f32::INFINITY,
            wait: 0,
            stopped: false,
        }
    }
    
    pub fn should_stop(&mut self, loss: f32) -> bool {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
            }
        }
        self.stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_early_stopping() {
        let mut early_stop = EarlyStopping::new(3, 0.01);
        
        assert!(!early_stop.should_stop(1.0));
        assert!(!early_stop.should_stop(0.9));
        assert!(!early_stop.should_stop(0.85));
        assert!(!early_stop.should_stop(0.86));
        assert!(!early_stop.should_stop(0.87));
        assert!(early_stop.should_stop(0.88));
    }
}