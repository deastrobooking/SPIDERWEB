# Getting Started

This guide will help you set up and start using the Rust ML Framework.

## Installation

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))
- BLAS/LAPACK libraries for optimized linear algebra
- Optional: CUDA toolkit for GPU acceleration

### Building the Framework

1. Clone the repository:
```bash
git clone <repository-url>
cd rust-ml-framework
```

2. Build the framework:
```bash
cargo build --release
```

3. Run examples:
```bash
cargo run --bin examples
```

### Feature Flags

The framework supports optional features:

```toml
[dependencies]
rust-ml-framework = { version = "0.1", features = ["gpu", "python"] }
```

Available features:
- `gpu`: Enable CUDA/OpenCL GPU acceleration
- `python`: Python bindings for interoperability
- `distributed`: Multi-node training support

## Your First Model

Here's a simple neural network example:

```rust
use rust_ml_framework::*;

fn main() -> anyhow::Result<()> {
    // Initialize framework
    init()?;
    set_seed(42);
    
    // Create a simple MLP
    let mut model = nn::Sequential::new()
        .add(nn::Linear::new(784, 256))
        .add(nn::ReLU::new())
        .add(nn::Dropout::new(0.2))
        .add(nn::Linear::new(256, 10));
    
    // Create sample data
    let input = Tensor::randn(&[32, 784]);
    let target = Tensor::randint(0, 10, &[32]);
    
    // Forward pass
    let output = model.forward(&input)?;
    
    // Compute loss
    let loss_fn = loss::CrossEntropyLoss::new();
    let loss = loss_fn.forward(&output, &target)?;
    
    println!("Loss: {:.4}", loss.data()[0]);
    
    Ok(())
}
```

## Core Concepts

### Tensors

Tensors are the fundamental data structure:

```rust
// Create tensors
let a = Tensor::ones(&[3, 3]);
let b = Tensor::randn(&[3, 3]);
let c = Tensor::zeros(&[3, 3]);

// Operations
let sum = &a + &b;
let product = a.matmul(&b)?;
let reshaped = b.reshape(&[9])?;

// Gradient computation
let x = Tensor::randn(&[2, 2]).requires_grad(true);
```

### Neural Networks

Build models with the Module trait:

```rust
// Sequential model
let model = nn::Sequential::new()
    .add(nn::Conv2d::new(3, 64, 3, 1, 1))
    .add(nn::BatchNorm2d::new(64))
    .add(nn::ReLU::new())
    .add(nn::MaxPool2d::new(2, 2));

// Custom models implement Module trait
struct MyModel {
    linear: nn::Linear,
    activation: nn::ReLU,
}

impl nn::Module for MyModel {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.linear.forward(input)?;
        self.activation.forward(&x)
    }
    // ... other required methods
}
```

### Training Loop

Standard training pattern:

```rust
// Setup
let mut model = create_model();
let mut optimizer = optim::Adam::new(model.parameters_mut(), 0.001);
let loss_fn = loss::CrossEntropyLoss::new();

// Training loop
for epoch in 0..epochs {
    for batch in dataloader {
        let (inputs, targets) = batch?;
        
        // Forward pass
        let outputs = model.forward(&inputs)?;
        let loss = loss_fn.forward(&outputs, &targets)?;
        
        // Backward pass
        optimizer.zero_grad();
        // loss.backward(); // Compute gradients
        optimizer.step()?;
    }
}
```

## Next Steps

1. Read [Core Concepts](core-concepts.md) for deeper understanding
2. Follow [Tutorials](tutorials/) for hands-on learning
3. Explore [Examples](examples.md) for specific use cases
4. Check [API Reference](api-reference.md) for detailed documentation