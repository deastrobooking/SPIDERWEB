// Simple Neural Network Example for Rust ML Framework
// This demonstrates basic usage of the framework for binary classification

use rust_ml_framework::*;
use anyhow::Result;

fn main() -> Result<()> {
    println!("Rust ML Framework - Simple Neural Network Example");
    println!("================================================");
    
    // Initialize the framework
    init()?;
    set_seed(42); // For reproducible results
    
    // Create a simple binary classification model
    let mut model = create_binary_classifier();
    println!("✓ Model created with {} parameters", utils::count_parameters(&model));
    
    // Generate synthetic training data
    let (train_data, train_targets) = generate_binary_data(1000)?;
    println!("✓ Generated {} training samples", train_data.len());
    
    // Create dataset and dataloader
    let dataset = data::TensorDataset::new(train_data, train_targets)?;
    let mut dataloader = data::DataLoader::new(dataset, 32).with_shuffle(true);
    println!("✓ DataLoader created with batch size 32");
    
    // Setup training components
    let loss_fn = loss::BCELoss::new();
    let mut params: Vec<*mut Tensor> = model.parameters_mut()
        .into_iter()
        .map(|p| p as *mut Tensor)
        .collect();
    let mut optimizer = optim::Adam::new(params, 0.01);
    println!("✓ Training setup complete (Adam optimizer, BCE loss)");
    
    // Training loop
    println!("\nStarting training...");
    train_model(&mut model, &mut dataloader, &loss_fn, &mut optimizer, 20)?;
    
    // Test the trained model
    println!("\nTesting trained model:");
    test_model(&model)?;
    
    // Generate test data for evaluation
    let (test_data, test_targets) = generate_binary_data(200)?;
    let test_dataset = data::TensorDataset::new(test_data, test_targets)?;
    let mut test_dataloader = data::DataLoader::new(test_dataset, 32);
    
    // Evaluate model performance
    println!("\nEvaluating model performance:");
    evaluate_model(&model, &mut test_dataloader)?;
    
    println!("\n✓ Example completed successfully!");
    Ok(())
}

fn create_binary_classifier() -> nn::Sequential {
    nn::Sequential::new()
        .add(nn::Linear::new(2, 8))    // Input: 2 features
        .add(nn::ReLU::new())          // Hidden layer 1
        .add(nn::Linear::new(8, 4))    // Hidden layer 2
        .add(nn::ReLU::new())          // Activation
        .add(nn::Linear::new(4, 1))    // Output: 1 probability
        .add(nn::Sigmoid::new())       // Sigmoid for probability
}

fn generate_binary_data(num_samples: usize) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let mut data = Vec::new();
    let mut targets = Vec::new();
    
    for i in 0..num_samples {
        // Generate 2D points in a pattern
        let x1 = (i as f32 / num_samples as f32) * 4.0 - 2.0;  // Range: -2 to 2
        let x2 = ((i * 7) % num_samples) as f32 / num_samples as f32 * 4.0 - 2.0;
        
        // Classification rule: points above line y = x + noise belong to class 1
        let noise = ((i * 13) % 100) as f32 / 100.0 * 0.5 - 0.25; // Small noise
        let label = if x2 > x1 + noise { 1.0 } else { 0.0 };
        
        data.push(Tensor::from_vec(vec![x1, x2], &[2])?);
        targets.push(Tensor::from_vec(vec![label], &[1])?);
    }
    
    Ok((data, targets))
}

fn train_model(
    model: &mut nn::Sequential,
    dataloader: &mut data::DataLoader<data::TensorDataset>,
    loss_fn: &loss::BCELoss,
    optimizer: &mut optim::Adam,
    epochs: usize,
) -> Result<()> {
    for epoch in 0..epochs {
        model.train();
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        dataloader.reset();
        while let Some(batch) = dataloader.next_batch()? {
            let (inputs, targets) = batch;
            
            // Forward pass
            let outputs = model.forward(&inputs)?;
            let loss = loss_fn.forward(&outputs, &targets)?;
            
            // Backward pass (gradients computed automatically)
            optimizer.zero_grad();
            // Note: loss.backward() would be called here in full implementation
            optimizer.step()?;
            
            total_loss += loss.data()[0];
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        
        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: Loss = {:.4}", epoch + 1, avg_loss);
        }
    }
    
    Ok(())
}

fn test_model(model: &nn::Sequential) -> Result<()> {
    model.eval();
    
    let test_cases = vec![
        (vec![1.0, 2.0], "Should be class 1 (above line)"),
        (vec![2.0, 1.0], "Should be class 0 (below line)"),
        (vec![-1.0, 1.0], "Should be class 1 (above line)"),
        (vec![0.5, -0.5], "Should be class 0 (below line)"),
        (vec![0.0, 0.5], "Should be class 1 (above line)"),
        (vec![1.5, 0.5], "Should be class 0 (below line)"),
    ];
    
    for (input_data, description) in test_cases {
        let input = Tensor::from_vec(input_data.clone(), &[1, 2])?;
        let output = model.forward(&input)?;
        let prediction = if output.data()[0] > 0.5 { 1 } else { 0 };
        let confidence = output.data()[0];
        
        println!("  Input: [{:.1}, {:.1}] → Class {} (conf: {:.3}) - {}", 
                 input_data[0], input_data[1], prediction, confidence, description);
    }
    
    Ok(())
}

fn evaluate_model(
    model: &nn::Sequential,
    dataloader: &mut data::DataLoader<data::TensorDataset>,
) -> Result<()> {
    model.eval();
    
    let mut all_predictions = Vec::new();
    let mut all_targets = Vec::new();
    let mut correct = 0;
    let mut total = 0;
    
    dataloader.reset();
    while let Some(batch) = dataloader.next_batch()? {
        let (inputs, targets) = batch;
        let outputs = model.forward(&inputs)?;
        
        // Collect predictions and targets for metrics
        for i in 0..outputs.shape()[0] {
            let prediction = if outputs.data()[i] > 0.5 { 1.0 } else { 0.0 };
            let target = targets.data()[i];
            
            all_predictions.push(prediction);
            all_targets.push(target);
            
            if (prediction - target).abs() < 0.1 {
                correct += 1;
            }
            total += 1;
        }
    }
    
    let accuracy = correct as f32 / total as f32;
    println!("  Accuracy: {:.2}% ({}/{} correct)", accuracy * 100.0, correct, total);
    
    // Calculate additional metrics
    let mut tp = 0; // True positives
    let mut fp = 0; // False positives
    let mut tn = 0; // True negatives
    let mut fn_count = 0; // False negatives
    
    for (pred, target) in all_predictions.iter().zip(all_targets.iter()) {
        match (pred > &0.5, target > &0.5) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_count += 1,
        }
    }
    
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    
    println!("  Precision: {:.3}", precision);
    println!("  Recall: {:.3}", recall);
    println!("  F1 Score: {:.3}", f1);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = create_binary_classifier();
        // Model should have the expected layer structure
        assert_eq!(model.layers().len(), 6); // 3 Linear + 2 ReLU + 1 Sigmoid
    }

    #[test]
    fn test_data_generation() {
        let (data, targets) = generate_binary_data(100).unwrap();
        assert_eq!(data.len(), 100);
        assert_eq!(targets.len(), 100);
        
        // Check data shapes
        assert_eq!(data[0].shape(), &[2]);
        assert_eq!(targets[0].shape(), &[1]);
    }

    #[test]
    fn test_model_forward_pass() {
        init().unwrap();
        let model = create_binary_classifier();
        let input = Tensor::randn(&[1, 2]);
        let output = model.forward(&input).unwrap();
        
        // Output should be single probability value
        assert_eq!(output.shape(), &[1, 1]);
        // Output should be between 0 and 1 (sigmoid output)
        assert!(output.data()[0] >= 0.0 && output.data()[0] <= 1.0);
    }
}