//! Loss functions for training neural networks

use crate::tensor::Tensor;
use anyhow::Result;

/// Base trait for loss functions
pub trait Loss: Send + Sync {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
}

/// Mean Squared Error Loss
#[derive(Debug, Clone)]
pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions - targets;
        let squared = &diff * &diff;
        
        match self.reduction {
            Reduction::Mean => Ok(Tensor::full(&[1], squared.mean())),
            Reduction::Sum => Ok(Tensor::full(&[1], squared.sum())),
            Reduction::None => Ok(squared),
        }
    }
    
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions - targets;
        let grad = &diff * 2.0;
        
        match self.reduction {
            Reduction::Mean => Ok(&grad / (predictions.numel() as f32)),
            Reduction::Sum => Ok(grad),
            Reduction::None => Ok(grad),
        }
    }
}

/// Cross Entropy Loss
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Apply log softmax to predictions
        let log_softmax = log_softmax(predictions, 1)?;
        
        // Compute negative log likelihood
        let nll = nll_loss(&log_softmax, targets)?;
        
        match self.reduction {
            Reduction::Mean => Ok(Tensor::full(&[1], nll.mean())),
            Reduction::Sum => Ok(Tensor::full(&[1], nll.sum())),
            Reduction::None => Ok(nll),
        }
    }
    
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Gradient of cross entropy is softmax(predictions) - targets (one-hot)
        let softmax_pred = softmax(predictions, 1)?;
        let targets_one_hot = one_hot(targets, predictions.shape()[1])?;
        
        let grad = &softmax_pred - &targets_one_hot;
        
        match self.reduction {
            Reduction::Mean => Ok(&grad / (predictions.shape()[0] as f32)),
            Reduction::Sum => Ok(grad),
            Reduction::None => Ok(grad),
        }
    }
}

/// Binary Cross Entropy Loss
#[derive(Debug, Clone)]
pub struct BCELoss {
    reduction: Reduction,
}

impl BCELoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Loss for BCELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // BCE = -[y*log(p) + (1-y)*log(1-p)]
        let eps = 1e-8;
        let pred_clipped = clip(predictions, eps, 1.0 - eps)?;
        
        let log_pred = pred_clipped.log();
        let log_one_minus_pred = (&Tensor::ones(predictions.shape()) - &pred_clipped).log();
        
        let loss = &(&(-targets) * &log_pred) - &(&(&Tensor::ones(targets.shape()) - targets) * &log_one_minus_pred);
        
        match self.reduction {
            Reduction::Mean => Ok(Tensor::full(&[1], loss.mean())),
            Reduction::Sum => Ok(Tensor::full(&[1], loss.sum())),
            Reduction::None => Ok(loss),
        }
    }
    
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let eps = 1e-8;
        let pred_clipped = clip(predictions, eps, 1.0 - eps)?;
        
        let grad_data = predictions.data().iter()
            .zip(targets.data().iter())
            .map(|(p, t)| {
                let p_clipped = p.max(eps).min(1.0 - eps);
                (p_clipped - t) / (p_clipped * (1.0 - p_clipped))
            })
            .collect::<Vec<f32>>();
            
        let grad = Tensor::from_vec(grad_data, predictions.shape())?;
        
        match self.reduction {
            Reduction::Mean => Ok(&grad / (predictions.numel() as f32)),
            Reduction::Sum => Ok(grad),
            Reduction::None => Ok(grad),
        }
    }
}

/// Huber Loss (Smooth L1 Loss)
#[derive(Debug, Clone)]
pub struct HuberLoss {
    delta: f32,
    reduction: Reduction,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        Self {
            delta,
            reduction: Reduction::Mean,
        }
    }
}

impl Loss for HuberLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions - targets;
        let abs_diff = diff.abs();
        
        let loss_data = abs_diff.data().iter()
            .map(|&x| {
                if x <= self.delta {
                    0.5 * x * x
                } else {
                    self.delta * (x - 0.5 * self.delta)
                }
            })
            .collect::<Vec<f32>>();
            
        let loss = Tensor::from_vec(loss_data, predictions.shape())?;
        
        match self.reduction {
            Reduction::Mean => Ok(Tensor::full(&[1], loss.mean())),
            Reduction::Sum => Ok(Tensor::full(&[1], loss.sum())),
            Reduction::None => Ok(loss),
        }
    }
    
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions - targets;
        
        let grad_data = diff.data().iter()
            .map(|&x| {
                if x.abs() <= self.delta {
                    x
                } else {
                    self.delta * x.signum()
                }
            })
            .collect::<Vec<f32>>();
            
        let grad = Tensor::from_vec(grad_data, predictions.shape())?;
        
        match self.reduction {
            Reduction::Mean => Ok(&grad / (predictions.numel() as f32)),
            Reduction::Sum => Ok(grad),
            Reduction::None => Ok(grad),
        }
    }
}

/// Reduction strategies for loss functions
#[derive(Debug, Clone, Copy)]
pub enum Reduction {
    None,   // No reduction
    Mean,   // Mean of all elements
    Sum,    // Sum of all elements
}

// Helper functions
fn log_softmax(input: &Tensor, dim: usize) -> Result<Tensor> {
    let max_vals = input.sum_axis(dim)?; // Simplified - should be max along axis
    let shifted = input - &max_vals;
    let exp_vals = shifted.exp();
    let sum_exp = exp_vals.sum_axis(dim)?;
    let log_sum_exp = sum_exp.log();
    Ok(&shifted - &log_sum_exp)
}

fn softmax(input: &Tensor, dim: usize) -> Result<Tensor> {
    let max_vals = input.sum_axis(dim)?; // Simplified - should be max along axis
    let shifted = input - &max_vals;
    let exp_vals = shifted.exp();
    let sum_exp = exp_vals.sum_axis(dim)?;
    Ok(&exp_vals / &sum_exp)
}

fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Simplified negative log likelihood implementation
    let batch_size = log_probs.shape()[0];
    let mut loss_data = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let target_class = targets.data()[i] as usize;
        let log_prob = log_probs.data()[i * log_probs.shape()[1] + target_class];
        loss_data.push(-log_prob);
    }
    
    Tensor::from_vec(loss_data, &[batch_size])
}

fn one_hot(indices: &Tensor, num_classes: usize) -> Result<Tensor> {
    let batch_size = indices.numel();
    let mut one_hot_data = vec![0.0; batch_size * num_classes];
    
    for i in 0..batch_size {
        let class_idx = indices.data()[i] as usize;
        if class_idx < num_classes {
            one_hot_data[i * num_classes + class_idx] = 1.0;
        }
    }
    
    Tensor::from_vec(one_hot_data, &[batch_size, num_classes])
}

fn clip(input: &Tensor, min_val: f32, max_val: f32) -> Result<Tensor> {
    let clipped_data = input.data().iter()
        .map(|&x| x.max(min_val).min(max_val))
        .collect::<Vec<f32>>();
    
    Tensor::from_vec(clipped_data, input.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let mse = MSELoss::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let target = Tensor::from_vec(vec![1.5, 2.5, 2.5], &[3]).unwrap();
        
        let loss = mse.forward(&pred, &target).unwrap();
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_bce_loss() {
        let bce = BCELoss::new();
        let pred = Tensor::from_vec(vec![0.8, 0.2, 0.9], &[3]).unwrap();
        let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], &[3]).unwrap();
        
        let loss = bce.forward(&pred, &target).unwrap();
        assert!(loss.data()[0] > 0.0);
    }
}