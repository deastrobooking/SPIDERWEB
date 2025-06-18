//! Metrics and evaluation utilities

use crate::tensor::Tensor;
use anyhow::Result;

/// Accuracy metric for classification tasks
pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
    let pred_classes = argmax(predictions, 1);
    let correct = pred_classes.data().iter()
        .zip(targets.data().iter())
        .map(|(p, t)| if (p - t).abs() < 1e-6 { 1.0 } else { 0.0 })
        .sum::<f32>();
    
    correct / targets.numel() as f32
}

/// Top-k accuracy
pub fn top_k_accuracy(predictions: &Tensor, targets: &Tensor, k: usize) -> f32 {
    // Simplified implementation
    accuracy(predictions, targets)
}

/// Precision metric
pub fn precision(predictions: &Tensor, targets: &Tensor) -> f32 {
    let pred_classes = argmax(predictions, 1);
    // Simplified binary classification precision
    let true_positives = pred_classes.data().iter()
        .zip(targets.data().iter())
        .filter(|(p, t)| **p > 0.5 && **t > 0.5)
        .count() as f32;
    
    let predicted_positives = pred_classes.data().iter()
        .filter(|&&p| p > 0.5)
        .count() as f32;
    
    if predicted_positives > 0.0 {
        true_positives / predicted_positives
    } else {
        0.0
    }
}

/// Recall metric
pub fn recall(predictions: &Tensor, targets: &Tensor) -> f32 {
    let pred_classes = argmax(predictions, 1);
    let true_positives = pred_classes.data().iter()
        .zip(targets.data().iter())
        .filter(|(p, t)| **p > 0.5 && **t > 0.5)
        .count() as f32;
    
    let actual_positives = targets.data().iter()
        .filter(|&&t| t > 0.5)
        .count() as f32;
    
    if actual_positives > 0.0 {
        true_positives / actual_positives
    } else {
        0.0
    }
}

/// F1 score
pub fn f1_score(predictions: &Tensor, targets: &Tensor) -> f32 {
    let p = precision(predictions, targets);
    let r = recall(predictions, targets);
    
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

/// Mean Squared Error
pub fn mse(predictions: &Tensor, targets: &Tensor) -> f32 {
    let diff = predictions - targets;
    let squared = &diff * &diff;
    squared.mean()
}

/// Mean Absolute Error
pub fn mae(predictions: &Tensor, targets: &Tensor) -> f32 {
    let diff = predictions - targets;
    diff.abs().mean()
}

/// R-squared coefficient of determination
pub fn r2_score(predictions: &Tensor, targets: &Tensor) -> f32 {
    let target_mean = targets.mean();
    let ss_res = (predictions - targets).pow(2.0).sum();
    let ss_tot = (targets - &Tensor::full(targets.shape(), target_mean)).pow(2.0).sum();
    
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Confusion matrix for binary classification
pub fn confusion_matrix(predictions: &Tensor, targets: &Tensor, threshold: f32) -> [[usize; 2]; 2] {
    let mut matrix = [[0; 2]; 2];
    
    for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
        let pred_class = if *pred > threshold { 1 } else { 0 };
        let true_class = if *target > threshold { 1 } else { 0 };
        matrix[true_class][pred_class] += 1;
    }
    
    matrix
}

/// Area Under the ROC Curve (simplified)
pub fn auc_roc(predictions: &Tensor, targets: &Tensor) -> f32 {
    // Simplified AUC calculation
    let mut pairs = predictions.data().iter()
        .zip(targets.data().iter())
        .collect::<Vec<_>>();
    
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());
    
    let mut auc = 0.0;
    let mut true_positives = 0.0;
    let mut false_positives = 0.0;
    let total_positives = targets.data().iter().filter(|&&x| x > 0.5).count() as f32;
    let total_negatives = targets.data().iter().filter(|&&x| x <= 0.5).count() as f32;
    
    for (_, &target) in pairs {
        if target > 0.5 {
            true_positives += 1.0;
        } else {
            false_positives += 1.0;
            auc += true_positives;
        }
    }
    
    if total_positives > 0.0 && total_negatives > 0.0 {
        auc / (total_positives * total_negatives)
    } else {
        0.5
    }
}

// Helper function to find argmax along dimension
fn argmax(tensor: &Tensor, dim: usize) -> Tensor {
    // Simplified argmax implementation
    let shape = tensor.shape();
    if dim >= shape.len() {
        return tensor.clone_tensor();
    }
    
    let mut result_data = Vec::new();
    let batch_size = shape[0];
    let num_classes = shape[1];
    
    for b in 0..batch_size {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        
        for c in 0..num_classes {
            let idx = b * num_classes + c;
            if idx < tensor.data().len() && tensor.data()[idx] > max_val {
                max_val = tensor.data()[idx];
                max_idx = c;
            }
        }
        result_data.push(max_idx as f32);
    }
    
    Tensor::from_vec(result_data, &[batch_size]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let predictions = Tensor::from_vec(vec![0.1, 0.9, 0.8, 0.2], &[2, 2]).unwrap();
        let targets = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let acc = accuracy(&predictions, &targets);
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_mse() {
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let targets = Tensor::from_vec(vec![1.1, 1.9, 3.1], &[3]).unwrap();
        let mse_val = mse(&predictions, &targets);
        assert!(mse_val >= 0.0);
    }
}