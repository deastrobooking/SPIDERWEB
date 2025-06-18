# Examples

This document provides practical examples for common machine learning tasks using the Rust ML Framework.

## Basic Tensor Operations

```rust
use rust_ml_framework::*;

fn tensor_basics() -> anyhow::Result<()> {
    // Create tensors
    let a = Tensor::randn(&[3, 4]);
    let b = Tensor::ones(&[4, 2]);
    
    // Matrix multiplication
    let c = a.matmul(&b)?;
    println!("Result shape: {:?}", c.shape()); // [3, 2]
    
    // Element-wise operations
    let x = Tensor::randn(&[2, 3]);
    let y = Tensor::randn(&[2, 3]);
    let sum = &x + &y;
    let product = &x * &y;
    
    // Reductions
    println!("Sum: {:.4}", sum.sum());
    println!("Mean: {:.4}", x.mean());
    
    Ok(())
}
```

## Simple Neural Network

```rust
use rust_ml_framework::*;

fn simple_network() -> anyhow::Result<()> {
    // Create a simple feedforward network
    let mut model = nn::Sequential::new()
        .add(nn::Linear::new(784, 128))
        .add(nn::ReLU::new())
        .add(nn::Dropout::new(0.2))
        .add(nn::Linear::new(128, 64))
        .add(nn::ReLU::new())
        .add(nn::Linear::new(64, 10));
    
    // Sample input (batch of 32 MNIST-like images)
    let input = Tensor::randn(&[32, 784]);
    
    // Forward pass
    let output = model.forward(&input)?;
    println!("Output shape: {:?}", output.shape()); // [32, 10]
    
    Ok(())
}
```

## Training Loop

```rust
use rust_ml_framework::*;

fn training_example() -> anyhow::Result<()> {
    // Model and data setup
    let mut model = nn::mlp(784, &[256, 128], 10, "relu");
    let loss_fn = loss::CrossEntropyLoss::new();
    
    // Create synthetic dataset
    let data_size = 1000;
    let mut train_data = Vec::new();
    let mut train_targets = Vec::new();
    
    for i in 0..data_size {
        train_data.push(Tensor::randn(&[784]));
        train_targets.push(Tensor::from_vec(vec![(i % 10) as f32], &[1])?);
    }
    
    let dataset = data::TensorDataset::new(train_data, train_targets)?;
    let mut dataloader = data::DataLoader::new(dataset, 32).with_shuffle(true);
    
    // Optimizer
    let mut params: Vec<*mut Tensor> = model.parameters_mut()
        .into_iter()
        .map(|p| p as *mut Tensor)
        .collect();
    let mut optimizer = optim::Adam::new(params, 0.001);
    
    // Training loop
    let epochs = 10;
    for epoch in 0..epochs {
        model.train();
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        dataloader.reset();
        while let Some(batch) = dataloader.next_batch()? {
            let (inputs, targets) = batch;
            
            // Forward pass
            let outputs = model.forward(&inputs)?;
            let loss = loss_fn.forward(&outputs, &targets)?;
            
            // Backward pass
            optimizer.zero_grad();
            // In full implementation: loss.backward();
            optimizer.step()?;
            
            total_loss += loss.data()[0];
            batch_count += 1;
        }
        
        let avg_loss = total_loss / batch_count as f32;
        println!("Epoch {}: Loss = {:.4}", epoch + 1, avg_loss);
    }
    
    Ok(())
}
```

## Convolutional Neural Network

```rust
use rust_ml_framework::*;

fn cnn_example() -> anyhow::Result<()> {
    // Create CNN for image classification
    let mut model = nn::Sequential::new()
        .add(nn::Conv2d::new(3, 32, 3, 1, 1))
        .add(nn::BatchNorm2d::new(32))
        .add(nn::ReLU::new())
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Conv2d::new(32, 64, 3, 1, 1))
        .add(nn::BatchNorm2d::new(64))
        .add(nn::ReLU::new())
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Conv2d::new(64, 128, 3, 1, 1))
        .add(nn::ReLU::new())
        .add(nn::AdaptiveAvgPool2d::new(1))
        .add(nn::Linear::new(128, 10));
    
    // Input: batch of RGB images
    let input = Tensor::randn(&[16, 3, 32, 32]);
    let output = model.forward(&input)?;
    
    println!("CNN output shape: {:?}", output.shape()); // [16, 10]
    
    Ok(())
}
```

## ResNet Architecture

```rust
use rust_ml_framework::*;

fn resnet_example() -> anyhow::Result<()> {
    // Use pre-built ResNet-18
    let model = vision::resnet18(1000);
    
    // ImageNet-sized input
    let input = Tensor::randn(&[8, 3, 224, 224]);
    let output = model.forward(&input)?;
    
    println!("ResNet-18 output: {:?}", output.shape()); // [8, 1000]
    
    // Custom ResNet block
    let resnet_block = vision::ResNetBlock::new(64, 128, 2);
    let block_input = Tensor::randn(&[4, 64, 32, 32]);
    let block_output = resnet_block.forward(&block_input)?;
    
    println!("ResNet block output: {:?}", block_output.shape()); // [4, 128, 16, 16]
    
    Ok(())
}
```

## Text Processing

```rust
use rust_ml_framework::*;

fn nlp_example() -> anyhow::Result<()> {
    // Create tokenizer
    let mut tokenizer = nlp::Tokenizer::new();
    let texts = vec![
        "machine learning is amazing".to_string(),
        "rust provides memory safety".to_string(),
        "neural networks learn patterns".to_string(),
    ];
    
    tokenizer.build_vocab(&texts, 1);
    
    // Tokenize text
    let text = "machine learning with rust";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);
    
    println!("Original: {}", text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: {}", decoded);
    
    // Text classification model
    let mut classifier = nlp::TextClassifier::new(
        tokenizer.vocab_size(), // vocab_size
        128,                    // embedding_dim
        256,                    // hidden_size
        2                       // num_classes
    );
    
    // Process sequence
    let sequence = Tensor::from_vec(
        tokens.into_iter().map(|t| t as f32).collect(),
        &[text.split_whitespace().count(), 1]
    )?;
    
    let classification = classifier.forward(&sequence)?;
    println!("Classification output: {:?}", classification.shape());
    
    Ok(())
}
```

## Image Data Pipeline

```rust
use rust_ml_framework::*;

fn image_pipeline_example() -> anyhow::Result<()> {
    // Create image transforms
    let transform = transforms::Compose::new()
        .add(|t| transforms::resize(t, 224, 224))
        .add(|t| transforms::normalize(
            t, 
            &[0.485, 0.456, 0.406], 
            &[0.229, 0.224, 0.225]
        ))
        .add(|t| transforms::random_horizontal_flip(t, 0.5));
    
    // Sample image processing
    let image = Tensor::randn(&[3, 256, 256]); // RGB image
    let processed = transform.apply(&image)?;
    
    println!("Original image: {:?}", image.shape());
    println!("Processed image: {:?}", processed.shape());
    
    // Image dataset (would load from folder in practice)
    let sample_images = vec![
        Tensor::randn(&[3, 224, 224]),
        Tensor::randn(&[3, 224, 224]),
        Tensor::randn(&[3, 224, 224]),
    ];
    let sample_labels = vec![
        Tensor::from_vec(vec![0.0], &[1])?,
        Tensor::from_vec(vec![1.0], &[1])?,
        Tensor::from_vec(vec![2.0], &[1])?,
    ];
    
    let dataset = data::TensorDataset::new(sample_images, sample_labels)?;
    let mut dataloader = data::DataLoader::new(dataset, 2);
    
    if let Some(batch) = dataloader.next_batch()? {
        println!("Batch images: {:?}", batch.0.shape());
        println!("Batch labels: {:?}", batch.1.shape());
    }
    
    Ok(())
}
```

## Custom Model Implementation

```rust
use rust_ml_framework::*;

struct CustomCNN {
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    bn1: nn::BatchNorm2d,
    bn2: nn::BatchNorm2d,
    fc1: nn::Linear,
    fc2: nn::Linear,
    relu: nn::ReLU,
    pool: nn::MaxPool2d,
    dropout: nn::Dropout,
    training: bool,
}

impl CustomCNN {
    fn new(num_classes: usize) -> Self {
        Self {
            conv1: nn::Conv2d::new(3, 32, 3, 1, 1),
            conv2: nn::Conv2d::new(32, 64, 3, 1, 1),
            bn1: nn::BatchNorm2d::new(32),
            bn2: nn::BatchNorm2d::new(64),
            fc1: nn::Linear::new(64 * 8 * 8, 128),
            fc2: nn::Linear::new(128, num_classes),
            relu: nn::ReLU::new(),
            pool: nn::MaxPool2d::new(2, 2),
            dropout: nn::Dropout::new(0.5),
            training: true,
        }
    }
}

impl nn::Module for CustomCNN {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        // First conv block
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.pool.forward(&x)?;
        
        // Second conv block
        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.pool.forward(&x)?;
        
        // Flatten
        let batch_size = x.shape()[0];
        let x = x.reshape(&[batch_size, 64 * 8 * 8])?;
        
        // Fully connected layers
        let x = self.fc1.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.dropout.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        
        Ok(x)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        params.extend(self.fc1.parameters_mut());
        params.extend(self.fc2.parameters_mut());
        params
    }
    
    fn train(&mut self) {
        self.training = true;
        self.bn1.train();
        self.bn2.train();
        self.dropout.train();
    }
    
    fn eval(&mut self) {
        self.training = false;
        self.bn1.eval();
        self.bn2.eval();
        self.dropout.eval();
    }
    
    fn training(&self) -> bool {
        self.training
    }
    
    fn name(&self) -> &str {
        "CustomCNN"
    }
    
    fn clone_module(&self) -> Box<dyn nn::Module> {
        Box::new(CustomCNN::new(10))
    }
}

fn custom_model_example() -> anyhow::Result<()> {
    let mut model = CustomCNN::new(10);
    
    let input = Tensor::randn(&[4, 3, 32, 32]);
    let output = model.forward(&input)?;
    
    println!("Custom CNN output: {:?}", output.shape()); // [4, 10]
    
    Ok(())
}
```

## Evaluation and Metrics

```rust
use rust_ml_framework::*;

fn evaluation_example() -> anyhow::Result<()> {
    // Sample predictions and targets
    let predictions = Tensor::from_vec(
        vec![0.9, 0.1, 0.2, 0.8, 0.7, 0.3, 0.1, 0.9],
        &[4, 2]
    )?;
    let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0], &[4])?;
    
    // Classification metrics
    let accuracy = metrics::accuracy(&predictions, &targets);
    let precision = metrics::precision(&predictions, &targets);
    let recall = metrics::recall(&predictions, &targets);
    let f1 = metrics::f1_score(&predictions, &targets);
    
    println!("Classification Metrics:");
    println!("Accuracy: {:.4}", accuracy);
    println!("Precision: {:.4}", precision);
    println!("Recall: {:.4}", recall);
    println!("F1 Score: {:.4}", f1);
    
    // Regression metrics
    let pred_reg = Tensor::from_vec(vec![2.5, 0.0, 2.1, 1.8], &[4])?;
    let target_reg = Tensor::from_vec(vec![3.0, -0.5, 2.0, 1.7], &[4])?;
    
    let mse = metrics::mse(&pred_reg, &target_reg);
    let mae = metrics::mae(&pred_reg, &target_reg);
    let r2 = metrics::r2_score(&pred_reg, &target_reg);
    
    println!("\nRegression Metrics:");
    println!("MSE: {:.4}", mse);
    println!("MAE: {:.4}", mae);
    println!("R² Score: {:.4}", r2);
    
    Ok(())
}
```

## Optimizer Comparison

```rust
use rust_ml_framework::*;

fn optimizer_comparison() -> anyhow::Result<()> {
    let model = nn::Linear::new(10, 1);
    
    // Different optimizers for the same model
    let param_ptrs: Vec<*mut Tensor> = vec![];
    
    let sgd = optim::SGD::new(param_ptrs.clone(), 0.01);
    let adam = optim::Adam::new(param_ptrs.clone(), 0.001);
    let adamw = optim::AdamW::new(param_ptrs.clone(), 0.001, 0.01);
    let rmsprop = optim::RMSprop::new(param_ptrs.clone(), 0.01);
    
    println!("Optimizer Learning Rates:");
    println!("SGD: {}", sgd.learning_rate());
    println!("Adam: {}", adam.learning_rate());
    println!("AdamW: {}", adamw.learning_rate());
    println!("RMSprop: {}", rmsprop.learning_rate());
    
    // Learning rate scheduling
    let initial_lr = 0.1;
    println!("\nLearning Rate Schedules:");
    for epoch in 0..10 {
        let cosine_lr = utils::cosine_annealing_lr(epoch, 10, initial_lr, 0.001);
        let exp_lr = utils::exponential_decay_lr(epoch, initial_lr, 0.9);
        let step_lr = utils::step_decay_lr(epoch, initial_lr, 3, 0.5);
        
        if epoch % 3 == 0 {
            println!("Epoch {}: Cosine={:.4}, Exp={:.4}, Step={:.4}", 
                     epoch, cosine_lr, exp_lr, step_lr);
        }
    }
    
    Ok(())
}
```

## Advanced Training Features

```rust
use rust_ml_framework::*;

fn advanced_training() -> anyhow::Result<()> {
    // Model setup
    let mut model = vision::simple_cnn(3, 10);
    
    // Early stopping
    let mut early_stopping = utils::EarlyStopping::new(5, 0.001);
    
    // Training with advanced features
    let epochs = 100;
    let mut best_loss = f32::INFINITY;
    
    for epoch in 0..epochs {
        // Simulate training
        let current_loss = 1.0 / (epoch as f32 + 1.0); // Decreasing loss
        
        // Check early stopping
        if early_stopping.should_stop(current_loss) {
            println!("Early stopping at epoch {}", epoch);
            break;
        }
        
        // Save best model
        if current_loss < best_loss {
            best_loss = current_loss;
            println!("New best model at epoch {} with loss {:.4}", epoch, current_loss);
            
            // In practice: save model checkpoint
            // utils::save_checkpoint(&model_state, &optimizer_state, epoch, current_loss, "best_model.pt")?;
        }
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, current_loss);
        }
    }
    
    // Gradient clipping example
    let mut params = model.parameters_mut();
    let grad_norm = utils::clip_grad_norm(&mut params, 1.0);
    println!("Gradient norm: {:.4}", grad_norm);
    
    Ok(())
}
```

## Running Examples

To run these examples:

```bash
# Add to src/bin/my_examples.rs
cargo run --bin my_examples

# Or run the comprehensive examples
cargo run --bin examples
```

Each example demonstrates different aspects of the framework and can be combined to create more complex applications.