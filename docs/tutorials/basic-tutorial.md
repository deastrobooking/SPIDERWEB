# Basic Tutorial: Your First Neural Network

This tutorial will guide you through creating and training your first neural network using the Rust ML Framework.

## Prerequisites

- Basic understanding of machine learning concepts
- Rust programming knowledge
- Framework installed (see [Getting Started](../getting-started.md))

## Step 1: Project Setup

Create a new Rust project:

```bash
cargo new my_ml_project
cd my_ml_project
```

Add the framework to your `Cargo.toml`:

```toml
[dependencies]
rust-ml-framework = { path = "../rust-ml-framework" }
anyhow = "1.0"
```

## Step 2: Basic Tensor Operations

Let's start with tensor fundamentals:

```rust
use rust_ml_framework::*;
use anyhow::Result;

fn main() -> Result<()> {
    // Initialize the framework
    init()?;
    
    // Create tensors
    let a = Tensor::ones(&[3, 3]);
    let b = Tensor::randn(&[3, 3]);
    
    println!("Tensor A:\n{}", a);
    println!("Tensor B shape: {:?}", b.shape());
    
    // Basic operations
    let sum = &a + &b;
    let product = a.matmul(&b)?;
    
    println!("Sum mean: {:.4}", sum.mean());
    println!("Product shape: {:?}", product.shape());
    
    Ok(())
}
```

## Step 3: Building Your First Model

Create a simple feedforward network for binary classification:

```rust
use rust_ml_framework::*;

fn create_model() -> nn::Sequential {
    nn::Sequential::new()
        .add(nn::Linear::new(2, 4))    // Input layer: 2 features -> 4 hidden
        .add(nn::ReLU::new())          // Activation function
        .add(nn::Linear::new(4, 4))    // Hidden layer: 4 -> 4
        .add(nn::ReLU::new())          // Activation function
        .add(nn::Linear::new(4, 1))    // Output layer: 4 -> 1 output
        .add(nn::Sigmoid::new())       // Sigmoid for binary classification
}

fn main() -> Result<()> {
    init()?;
    
    let model = create_model();
    println!("Model created with {} parameters", 
             utils::count_parameters(&model));
    
    // Test forward pass
    let input = Tensor::randn(&[1, 2]);  // Single sample with 2 features
    let output = model.forward(&input)?;
    
    println!("Input: {:?}", input.data());
    println!("Output: {:.4}", output.data()[0]);
    
    Ok(())
}
```

## Next Steps

- Try the [Computer Vision Tutorial](computer-vision-tutorial.md) for image classification
- Explore [NLP Tutorial](nlp-tutorial.md) for text processing
- Read about [Advanced Training](../performance.md) techniques
- Check out more [Examples](../examples.md)