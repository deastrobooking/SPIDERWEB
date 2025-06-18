// Computer Vision Demo - Image Classification with CNN
// Demonstrates convolutional neural networks using the Rust ML Framework

use rust_ml_framework::*;
use anyhow::Result;

fn main() -> Result<()> {
    println!("Rust ML Framework - Computer Vision Demo");
    println!("=======================================");
    
    init()?;
    set_seed(42);
    
    // Create a CNN for image classification (CIFAR-10 style)
    let mut model = create_cnn_classifier();
    println!("✓ CNN model created with {} parameters", utils::count_parameters(&model));
    
    // Generate synthetic image data (32x32x3 RGB images)
    let (train_images, train_labels) = generate_image_data(1000)?;
    println!("✓ Generated {} training images (32x32x3)", train_images.len());
    
    // Create data pipeline
    let dataset = data::TensorDataset::new(train_images, train_labels)?;
    let mut dataloader = data::DataLoader::new(dataset, 16).with_shuffle(true);
    println!("✓ Image DataLoader created with batch size 16");
    
    // Setup training
    let loss_fn = loss::CrossEntropyLoss::new();
    let mut params: Vec<*mut Tensor> = model.parameters_mut()
        .into_iter()
        .map(|p| p as *mut Tensor)
        .collect();
    let mut optimizer = optim::Adam::new(params, 0.001);
    println!("✓ Training setup complete (CrossEntropy loss, Adam optimizer)");
    
    // Training loop for image classification
    println!("\nTraining CNN on image data...");
    train_cnn(&mut model, &mut dataloader, &loss_fn, &mut optimizer, 10)?;
    
    // Test image classification
    println!("\nTesting image classification:");
    test_image_classification(&model)?;
    
    // Demonstrate pre-trained architectures
    println!("\nDemonstrating pre-trained architectures:");
    demonstrate_pretrained_models()?;
    
    println!("\n✓ Computer vision demo completed!");
    Ok(())
}

fn create_cnn_classifier() -> nn::Sequential {
    nn::Sequential::new()
        // First convolutional block
        .add(nn::Conv2d::new(3, 32, 3, 1, 1))  // 3 input channels (RGB), 32 filters, 3x3 kernel
        .add(nn::BatchNorm2d::new(32))
        .add(nn::ReLU::new())
        .add(nn::MaxPool2d::new(2, 2))         // Downsample to 16x16
        
        // Second convolutional block
        .add(nn::Conv2d::new(32, 64, 3, 1, 1))
        .add(nn::BatchNorm2d::new(64))
        .add(nn::ReLU::new())
        .add(nn::MaxPool2d::new(2, 2))         // Downsample to 8x8
        
        // Third convolutional block
        .add(nn::Conv2d::new(64, 128, 3, 1, 1))
        .add(nn::BatchNorm2d::new(128))
        .add(nn::ReLU::new())
        .add(nn::MaxPool2d::new(2, 2))         // Downsample to 4x4
        
        // Global average pooling and classifier
        .add(nn::AdaptiveAvgPool2d::new(1))    // Global average pooling to 1x1
        .add(nn::Flatten::new())               // Flatten to vector
        .add(nn::Dropout::new(0.5))            // Regularization
        .add(nn::Linear::new(128, 10))         // 10 classes for classification
}

fn generate_image_data(num_samples: usize) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    
    for i in 0..num_samples {
        // Generate random 32x32x3 RGB image
        let mut image_data = Vec::with_capacity(32 * 32 * 3);
        
        // Create patterns for different classes
        let class = i % 10;
        let base_color = class as f32 / 10.0;
        
        for y in 0..32 {
            for x in 0..32 {
                for c in 0..3 {
                    // Add some structure based on class
                    let pattern = match class {
                        0..=2 => (x + y) as f32 / 64.0,      // Diagonal pattern
                        3..=5 => (x * y) as f32 / 1024.0,    // Curved pattern
                        6..=7 => ((x - 16).pow(2) + (y - 16).pow(2)) as f32 / 512.0, // Circular
                        _ => (x as f32).sin() * (y as f32).cos(), // Sinusoidal
                    };
                    
                    let noise = ((i * 17 + x * 7 + y * 13 + c * 3) % 100) as f32 / 100.0;
                    let value = (base_color + pattern * 0.5 + noise * 0.2).clamp(0.0, 1.0);
                    image_data.push(value);
                }
            }
        }
        
        images.push(Tensor::from_vec(image_data, &[3, 32, 32])?);
        labels.push(Tensor::from_vec(vec![class as f32], &[1])?);
    }
    
    Ok((images, labels))
}

fn train_cnn(
    model: &mut nn::Sequential,
    dataloader: &mut data::DataLoader<data::TensorDataset>,
    loss_fn: &loss::CrossEntropyLoss,
    optimizer: &mut optim::Adam,
    epochs: usize,
) -> Result<()> {
    for epoch in 0..epochs {
        model.train();
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        dataloader.reset();
        while let Some(batch) = dataloader.next_batch()? {
            let (images, labels) = batch;
            
            // Forward pass through CNN
            let logits = model.forward(&images)?;
            let loss = loss_fn.forward(&logits, &labels)?;
            
            // Backward pass
            optimizer.zero_grad();
            optimizer.step()?;
            
            total_loss += loss.data()[0];
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("  Epoch {}: Loss = {:.4}", epoch + 1, avg_loss);
    }
    
    Ok(())
}

fn test_image_classification(model: &nn::Sequential) -> Result<()> {
    model.eval();
    
    // Test with a few sample images
    for class in 0..5 {
        // Generate test image for each class
        let mut image_data = Vec::with_capacity(32 * 32 * 3);
        let base_color = class as f32 / 10.0;
        
        for y in 0..32 {
            for x in 0..32 {
                for c in 0..3 {
                    let pattern = match class {
                        0..=2 => (x + y) as f32 / 64.0,
                        3..=5 => (x * y) as f32 / 1024.0,
                        _ => 0.5,
                    };
                    let value = (base_color + pattern * 0.5).clamp(0.0, 1.0);
                    image_data.push(value);
                }
            }
        }
        
        let image = Tensor::from_vec(image_data, &[1, 3, 32, 32])?; // Add batch dimension
        let logits = model.forward(&image)?;
        
        // Find predicted class (argmax)
        let mut max_idx = 0;
        let mut max_val = logits.data()[0];
        for i in 1..10 {
            if logits.data()[i] > max_val {
                max_val = logits.data()[i];
                max_idx = i;
            }
        }
        
        println!("  Class {} → Predicted: {} (confidence: {:.3})", 
                 class, max_idx, max_val);
    }
    
    Ok(())
}

fn demonstrate_pretrained_models() -> Result<()> {
    // Show available pre-trained architectures
    println!("  Available architectures:");
    
    // ResNet variants
    let resnet18 = vision::resnet18(false)?; // Not pre-trained yet
    println!("    - ResNet-18: {} parameters", utils::count_parameters(&resnet18));
    
    let resnet50 = vision::resnet50(false)?;
    println!("    - ResNet-50: {} parameters", utils::count_parameters(&resnet50));
    
    // VGG variants
    let vgg16 = vision::vgg16(false)?;
    println!("    - VGG-16: {} parameters", utils::count_parameters(&vgg16));
    
    // MobileNet for efficiency
    let mobilenet = vision::mobilenet_v2(false)?;
    println!("    - MobileNet-V2: {} parameters", utils::count_parameters(&mobilenet));
    
    // Demonstrate transfer learning setup
    println!("  Transfer learning example:");
    let mut pretrained_model = vision::resnet18(true)?; // Would load pre-trained weights
    
    // Replace classifier for new task (e.g., 5 classes instead of 1000)
    pretrained_model.replace_classifier(5)?;
    println!("    - Modified ResNet-18 for 5-class classification");
    
    // Freeze early layers for transfer learning
    pretrained_model.freeze_backbone()?;
    println!("    - Backbone frozen, only training classifier");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnn_creation() {
        let model = create_cnn_classifier();
        // Should have conv layers, batch norm, pooling, and classifier
        assert!(model.layers().len() > 10);
    }

    #[test]
    fn test_image_data_generation() {
        let (images, labels) = generate_image_data(10).unwrap();
        assert_eq!(images.len(), 10);
        assert_eq!(labels.len(), 10);
        
        // Check image shape (3 channels, 32x32)
        assert_eq!(images[0].shape(), &[3, 32, 32]);
        assert_eq!(labels[0].shape(), &[1]);
    }

    #[test]
    fn test_cnn_forward_pass() {
        init().unwrap();
        let model = create_cnn_classifier();
        let batch = Tensor::randn(&[2, 3, 32, 32]); // Batch of 2 images
        let output = model.forward(&batch).unwrap();
        
        // Should output logits for 10 classes
        assert_eq!(output.shape(), &[2, 10]);
    }
}