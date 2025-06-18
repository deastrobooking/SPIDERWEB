//! Comprehensive examples demonstrating all ML framework features
//! 
//! This binary showcases the complete TensorFlow/PyTorch clone functionality
//! including tensor operations, neural networks, training loops, and more.

use rust_ml_framework::*;
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    // Initialize the framework
    init()?;
    set_seed(42);
    
    println!("🚀 Rust ML Framework - TensorFlow/PyTorch Clone Demo");
    println!("====================================================");
    
    // Run all examples
    tensor_operations_example()?;
    neural_network_example()?;
    training_loop_example()?;
    computer_vision_example()?;
    nlp_example()?;
    optimization_example()?;
    data_loading_example()?;
    
    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Demonstrate core tensor operations (like PyTorch tensors / TensorFlow operations)
fn tensor_operations_example() -> Result<()> {
    println!("\n📊 Tensor Operations Example");
    println!("---------------------------");
    
    // Basic tensor creation
    let a = Tensor::ones(&[3, 3]);
    let b = Tensor::randn(&[3, 3]);
    let c = Tensor::eye(3);
    
    println!("Created tensors:");
    println!("- Ones tensor: shape {:?}", a.shape());
    println!("- Random normal tensor: shape {:?}", b.shape());
    println!("- Identity matrix: shape {:?}", c.shape());
    
    // Arithmetic operations
    let sum = &a + &b;
    let product = &a * &b;
    let matmul = a.matmul(&b)?;
    
    println!("Operations:");
    println!("- Sum mean: {:.4}", sum.mean());
    println!("- Product mean: {:.4}", product.mean());
    println!("- Matrix multiplication shape: {:?}", matmul.shape());
    
    // Advanced operations
    let reshaped = b.reshape(&[9])?;
    let transposed = matmul.t();
    let normalized = (&b - b.mean()) / b.sqrt();
    
    println!("Advanced operations:");
    println!("- Reshaped to vector: {:?}", reshaped.shape());
    println!("- Transposed shape: {:?}", transposed.shape());
    println!("- Normalized mean: {:.4}", normalized.mean());
    
    // Gradient computation setup
    let x = Tensor::randn(&[2, 2]).requires_grad(true);
    let y = &x * &x;
    let loss = y.sum();
    
    println!("Gradient-enabled tensors:");
    println!("- Input requires grad: {}", x.grad().is_some());
    println!("- Loss value: {:.4}", loss);
    
    Ok(())
}

/// Demonstrate neural network building (like PyTorch nn.Module / TensorFlow layers)
fn neural_network_example() -> Result<()> {
    println!("\n🧠 Neural Network Example");
    println!("-------------------------");
    
    // Build a multi-layer perceptron
    let mut model = nn::mlp(784, &[256, 128, 64], 10, "relu");
    
    println!("Created MLP:");
    println!("- Architecture: 784 -> 256 -> 128 -> 64 -> 10");
    println!("- Number of layers: {}", model.len());
    println!("- Total parameters: {}", utils::count_parameters(&model));
    
    // Create a CNN for image classification
    let mut cnn = vision::simple_cnn(3, 10);
    
    println!("Created CNN:");
    println!("- Input channels: 3 (RGB)");
    println!("- Output classes: 10");
    println!("- Total parameters: {}", utils::count_parameters(&cnn));
    
    // Forward pass example
    let input = Tensor::randn(&[32, 784]); // Batch of 32 samples
    let output = model.forward(&input)?;
    
    println!("Forward pass:");
    println!("- Input shape: {:?}", input.shape());
    println!("- Output shape: {:?}", output.shape());
    println!("- Output mean: {:.4}", output.mean());
    
    // Demonstrate different layer types
    let conv_layer = nn::Conv2d::new(3, 64, 3, 1, 1);
    let norm_layer = nn::BatchNorm2d::new(64);
    let activation = nn::ReLU::new();
    let pool_layer = nn::MaxPool2d::new(2, 2);
    
    println!("Individual layers:");
    println!("- Conv2d: {} parameters", utils::count_parameters(&conv_layer));
    println!("- BatchNorm2d: {} parameters", utils::count_parameters(&norm_layer));
    println!("- ReLU: {} parameters", utils::count_parameters(&activation));
    println!("- MaxPool2d: {} parameters", utils::count_parameters(&pool_layer));
    
    Ok(())
}

/// Demonstrate complete training loop (like PyTorch training / TensorFlow fit)
fn training_loop_example() -> Result<()> {
    println!("\n🏋️ Training Loop Example");
    println!("------------------------");
    
    // Create model and data
    let mut model = nn::mlp(4, &[8, 8], 3, "relu");
    let loss_fn = loss::CrossEntropyLoss::new();
    
    // Create synthetic dataset
    let train_data = vec![
        Tensor::randn(&[4]),
        Tensor::randn(&[4]),
        Tensor::randn(&[4]),
        Tensor::randn(&[4]),
    ];
    let train_targets = vec![
        Tensor::from_vec(vec![0.0], &[1])?,
        Tensor::from_vec(vec![1.0], &[1])?,
        Tensor::from_vec(vec![2.0], &[1])?,
        Tensor::from_vec(vec![0.0], &[1])?,
    ];
    
    let dataset = data::TensorDataset::new(train_data, train_targets)?;
    let mut dataloader = data::DataLoader::new(dataset, 2).with_shuffle(true);
    
    // Create optimizer
    let mut params: Vec<*mut Tensor> = model.parameters_mut()
        .into_iter()
        .map(|p| p as *mut Tensor)
        .collect();
    let mut optimizer = optim::Adam::new(params, 0.001);
    
    println!("Training setup:");
    println!("- Model parameters: {}", utils::count_parameters(&model));
    println!("- Dataset size: {}", dataloader.len());
    println!("- Optimizer: Adam (lr=0.001)");
    println!("- Loss function: CrossEntropyLoss");
    
    // Training loop
    let epochs = 5;
    let mut progress = utils::ProgressBar::new(epochs);
    
    for epoch in 0..epochs {
        model.train();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        
        dataloader.reset();
        while let Some(batch_result) = dataloader.next_batch() {
            let (inputs, targets) = batch_result?;
            
            // Forward pass
            let outputs = model.forward(&inputs)?;
            let loss = loss_fn.forward(&outputs, &targets)?;
            
            // Backward pass
            optimizer.zero_grad();
            // In a real implementation, backward pass would compute gradients
            
            // Optimizer step
            optimizer.step()?;
            
            epoch_loss += loss.data()[0];
            batch_count += 1;
        }
        
        let avg_loss = epoch_loss / batch_count as f32;
        progress.update(epoch + 1);
        
        if epoch % 2 == 0 {
            println!("\nEpoch {}: Loss = {:.4}", epoch + 1, avg_loss);
        }
    }
    
    progress.finish();
    
    // Evaluation
    model.eval();
    let test_input = Tensor::randn(&[1, 4]);
    let prediction = model.forward(&test_input)?;
    
    println!("Evaluation:");
    println!("- Test prediction shape: {:?}", prediction.shape());
    println!("- Prediction values: {:?}", &prediction.data()[..3]);
    
    Ok(())
}

/// Demonstrate computer vision capabilities
fn computer_vision_example() -> Result<()> {
    println!("\n👁️ Computer Vision Example");
    println!("--------------------------");
    
    // Create different CNN architectures
    let simple_cnn = vision::simple_cnn(3, 10);
    let vgg = vision::vgg11(1000);
    let resnet = vision::resnet18(1000);
    
    println!("CNN Architectures:");
    println!("- Simple CNN parameters: {}", utils::count_parameters(&simple_cnn));
    println!("- VGG-11 parameters: {}", utils::count_parameters(&vgg));
    println!("- ResNet-18 parameters: {}", utils::count_parameters(&resnet));
    
    // Image preprocessing pipeline
    let transform_pipeline = transforms::Compose::new()
        .add(|tensor| transforms::resize(tensor, 224, 224))
        .add(|tensor| transforms::normalize(tensor, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]))
        .add(|tensor| transforms::random_horizontal_flip(tensor, 0.5));
    
    // Simulate image processing
    let image = Tensor::randn(&[3, 256, 256]); // RGB image
    let processed = transform_pipeline.apply(&image)?;
    
    println!("Image processing:");
    println!("- Original image: {:?}", image.shape());
    println!("- Processed image: {:?}", processed.shape());
    
    // Forward pass through ResNet
    let batch_images = Tensor::randn(&[8, 3, 224, 224]); // Batch of 8 images
    let features = resnet.forward(&batch_images)?;
    
    println!("Feature extraction:");
    println!("- Input batch: {:?}", batch_images.shape());
    println!("- ResNet features: {:?}", features.shape());
    
    Ok(())
}

/// Demonstrate NLP capabilities
fn nlp_example() -> Result<()> {
    println!("\n📝 Natural Language Processing Example");
    println!("--------------------------------------");
    
    // Text tokenization
    let mut tokenizer = nlp::Tokenizer::new();
    let texts = vec![
        "Hello world".to_string(),
        "Machine learning is amazing".to_string(),
        "Rust is fast and safe".to_string(),
    ];
    
    tokenizer.build_vocab(&texts, 1);
    
    println!("Tokenizer:");
    println!("- Vocabulary size: {}", tokenizer.vocab_size());
    
    let encoded = tokenizer.encode("Hello machine learning");
    let decoded = tokenizer.decode(&encoded);
    
    println!("- Encoded text: {:?}", encoded);
    println!("- Decoded text: {}", decoded);
    
    // Language model
    let vocab_size = tokenizer.vocab_size();
    let mut language_model = nlp::LanguageModel::new(vocab_size, 128, 256);
    
    println!("Language model:");
    println!("- Parameters: {}", utils::count_parameters(&language_model));
    
    // Text classification
    let mut text_classifier = nlp::TextClassifier::new(vocab_size, 128, 256, 2);
    
    println!("Text classifier:");
    println!("- Parameters: {}", utils::count_parameters(&text_classifier));
    
    // Sequence processing
    let sequence = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[4, 1])?;
    let lm_output = language_model.forward(&sequence)?;
    let classification = text_classifier.forward(&sequence)?;
    
    println!("Sequence processing:");
    println!("- Input sequence: {:?}", sequence.shape());
    println!("- Language model output: {:?}", lm_output.shape());
    println!("- Classification output: {:?}", classification.shape());
    
    Ok(())
}

/// Demonstrate different optimization algorithms
fn optimization_example() -> Result<()> {
    println!("\n⚡ Optimization Algorithms Example");
    println!("----------------------------------");
    
    // Create a simple model for optimization comparison
    let mut model = nn::Linear::new(10, 1);
    let target = Tensor::ones(&[1]);
    
    // Different optimizers
    let optimizers = vec![
        ("SGD", "Stochastic Gradient Descent"),
        ("Adam", "Adaptive Moment Estimation"),
        ("AdamW", "Adam with Weight Decay"),
        ("RMSprop", "Root Mean Square Propagation"),
        ("Adagrad", "Adaptive Gradient Algorithm"),
    ];
    
    println!("Available optimizers:");
    for (name, description) in &optimizers {
        println!("- {}: {}", name, description);
    }
    
    // Learning rate scheduling
    let initial_lr = 0.1;
    println!("\nLearning rate schedules:");
    for epoch in 0..10 {
        let cosine_lr = utils::cosine_annealing_lr(epoch, 10, initial_lr, 0.001);
        let exp_lr = utils::exponential_decay_lr(epoch, initial_lr, 0.9);
        let step_lr = utils::step_decay_lr(epoch, initial_lr, 3, 0.5);
        
        if epoch % 3 == 0 {
            println!("Epoch {}: Cosine={:.4}, Exp={:.4}, Step={:.4}", 
                     epoch, cosine_lr, exp_lr, step_lr);
        }
    }
    
    // Gradient clipping demonstration
    println!("\nGradient clipping:");
    let mut params = model.parameters_mut();
    let original_norm = utils::clip_grad_norm(&mut params, 1.0);
    println!("- Original gradient norm: {:.4}", original_norm);
    println!("- Clipped to max norm: 1.0");
    
    Ok(())
}

/// Demonstrate data loading and preprocessing
fn data_loading_example() -> Result<()> {
    println!("\n📦 Data Loading Example");
    println!("-----------------------");
    
    // Create synthetic dataset
    let data_size = 1000;
    let feature_dim = 20;
    
    let mut data_vectors = Vec::new();
    let mut targets = Vec::new();
    
    for i in 0..data_size {
        let features = Tensor::randn(&[feature_dim]);
        let label = if i % 2 == 0 { 0.0 } else { 1.0 };
        
        data_vectors.push(features);
        targets.push(Tensor::from_vec(vec![label], &[1])?);
    }
    
    let dataset = data::TensorDataset::new(data_vectors, targets)?;
    
    println!("Dataset:");
    println!("- Total samples: {}", dataset.len());
    println!("- Feature dimension: {}", feature_dim);
    
    // Create dataloader with different configurations
    let batch_sizes = vec![32, 64, 128];
    
    for &batch_size in &batch_sizes {
        let mut dataloader = data::DataLoader::new(dataset.clone(), batch_size)
            .with_shuffle(true)
            .with_drop_last(false);
        
        println!("DataLoader (batch_size={}): {} batches", batch_size, dataloader.len());
        
        // Process first batch
        if let Some(batch) = dataloader.next_batch()? {
            println!("- First batch data: {:?}", batch.0.shape());
            println!("- First batch targets: {:?}", batch.1.shape());
        }
    }
    
    // Metrics evaluation
    println!("\nMetrics evaluation:");
    let predictions = Tensor::from_vec(vec![0.8, 0.2, 0.9, 0.1], &[2, 2])?;
    let true_labels = Tensor::from_vec(vec![1.0, 0.0], &[2])?;
    
    let accuracy = metrics::accuracy(&predictions, &true_labels);
    let mse = metrics::mse(&predictions.reshape(&[4])?, &true_labels.reshape(&[2])?);
    
    println!("- Accuracy: {:.4}", accuracy);
    println!("- MSE: {:.4}", mse);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_examples_run() {
        assert!(tensor_operations_example().is_ok());
        assert!(neural_network_example().is_ok());
    }
}