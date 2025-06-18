# Core Concepts

This document explains the fundamental concepts of the Rust ML Framework.

## Tensors

Tensors are multi-dimensional arrays that form the foundation of all computations.

### Creation

```rust
// Basic creation
let zeros = Tensor::zeros(&[3, 4]);
let ones = Tensor::ones(&[2, 2]);
let random = Tensor::randn(&[5, 5]);
let identity = Tensor::eye(3);

// From data
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_vec(data, &[2, 2])?;

// With gradients
let x = Tensor::randn(&[2, 2]).requires_grad(true);
```

### Operations

```rust
// Arithmetic
let a = Tensor::ones(&[3, 3]);
let b = Tensor::randn(&[3, 3]);
let sum = &a + &b;
let product = &a * &b;

// Linear algebra
let matmul = a.matmul(&b)?;
let transposed = a.t();

// Reductions
let mean = a.mean();
let sum_all = a.sum();
let sum_axis = a.sum_axis(0)?;

// Shape manipulation
let reshaped = a.reshape(&[9])?;
let squeezed = a.squeeze();
let unsqueezed = a.unsqueeze(0)?;
```

### Device Management

```rust
// Move tensors between devices
let tensor = Tensor::ones(&[3, 3]);
let gpu_tensor = tensor.to_device(Device::CUDA(0));
let cpu_tensor = gpu_tensor.to_device(Device::CPU);
```

## Neural Networks

The framework uses a modular approach to building neural networks.

### Module Trait

All neural network components implement the `Module` trait:

```rust
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
    fn training(&self) -> bool;
}
```

### Built-in Layers

#### Linear Layers
```rust
let linear = nn::Linear::new(784, 256);
let no_bias = nn::Linear::new_no_bias(256, 128);
```

#### Convolutional Layers
```rust
let conv2d = nn::Conv2d::new(3, 64, 3, 1, 1); // in_ch, out_ch, kernel, stride, padding
let conv1d = nn::Conv1d::new(1, 32, 5, 1, 2);
```

#### Activation Functions
```rust
let relu = nn::ReLU::new();
let sigmoid = nn::Sigmoid::new();
let tanh = nn::Tanh::new();
let gelu = nn::GELU::new();
let softmax = nn::Softmax::new(1); // dimension
```

#### Normalization
```rust
let batch_norm = nn::BatchNorm2d::new(64);
let layer_norm = nn::LayerNorm::new(vec![512]);
let group_norm = nn::GroupNorm::new(4, 64);
```

#### Pooling
```rust
let max_pool = nn::MaxPool2d::new(2, 2);
let avg_pool = nn::AvgPool2d::new(2, 2);
let adaptive_pool = nn::AdaptiveAvgPool2d::new(1);
```

### Sequential Models

```rust
let model = nn::Sequential::new()
    .add(nn::Linear::new(784, 256))
    .add(nn::ReLU::new())
    .add(nn::Dropout::new(0.5))
    .add(nn::Linear::new(256, 10));
```

### Custom Models

```rust
struct ConvNet {
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    fc: nn::Linear,
    relu: nn::ReLU,
    pool: nn::MaxPool2d,
}

impl ConvNet {
    fn new() -> Self {
        Self {
            conv1: nn::Conv2d::new(1, 32, 3, 1, 1),
            conv2: nn::Conv2d::new(32, 64, 3, 1, 1),
            fc: nn::Linear::new(64 * 7 * 7, 10),
            relu: nn::ReLU::new(),
            pool: nn::MaxPool2d::new(2, 2),
        }
    }
}

impl nn::Module for ConvNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.relu.forward(&x)?;
        let x = self.pool.forward(&x)?;
        
        let x = self.conv2.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.pool.forward(&x)?;
        
        let x = x.reshape(&[x.shape()[0], -1])?; // Flatten
        self.fc.forward(&x)
    }
    
    // Implementation of other required methods...
}
```

## Automatic Differentiation

The framework provides automatic gradient computation through the autograd system.

### Gradient Computation

```rust
// Enable gradients
let x = Tensor::randn(&[2, 2]).requires_grad(true);
let y = &x * &x;
let loss = y.sum();

// Backward pass computes gradients
// loss.backward();

// Access gradients
if let Some(grad) = x.grad() {
    println!("Gradient: {:?}", grad.shape());
}
```

### Gradient Context

```rust
// Disable gradients for inference
no_grad!({
    let output = model.forward(&input)?;
    // No gradients computed
});

// Enable gradients
enable_grad!({
    let output = model.forward(&input)?;
    // Gradients computed
});
```

## Optimization

The framework provides various optimization algorithms.

### Optimizers

```rust
// SGD with momentum
let sgd = optim::SGD::with_momentum(params, 0.01, 0.9);

// Adam
let adam = optim::Adam::new(params, 0.001);

// AdamW with weight decay
let adamw = optim::AdamW::new(params, 0.001, 0.01);

// RMSprop
let rmsprop = optim::RMSprop::new(params, 0.01);
```

### Learning Rate Scheduling

```rust
// Step scheduler
let scheduler = optim::StepLR::new(50, 0.1, 0.01);

// Cosine annealing
let scheduler = optim::CosineAnnealingLR::new(100, 0.001, 0.01);

// Apply scheduler
scheduler.step(&mut optimizer);
```

## Loss Functions

Various loss functions for different tasks.

### Classification Losses

```rust
let cross_entropy = loss::CrossEntropyLoss::new();
let bce = loss::BCELoss::new();
```

### Regression Losses

```rust
let mse = loss::MSELoss::new();
let huber = loss::HuberLoss::new(1.0);
```

### Usage

```rust
let loss_fn = loss::CrossEntropyLoss::new();
let loss = loss_fn.forward(&predictions, &targets)?;
let gradients = loss_fn.backward(&predictions, &targets)?;
```

## Data Loading

Efficient data loading and preprocessing.

### Datasets

```rust
// Tensor dataset
let dataset = data::TensorDataset::new(data_vectors, targets)?;

// CSV dataset
let dataset = data::CSVDataset::from_file("data.csv", 0)?; // target in column 0

// Image dataset
let dataset = data::ImageDataset::from_folder("images/")?;
```

### DataLoader

```rust
let dataloader = data::DataLoader::new(dataset, 32) // batch_size
    .with_shuffle(true)
    .with_drop_last(false);

for batch_result in dataloader {
    let (inputs, targets) = batch_result?;
    // Training code
}
```

### Transforms

```rust
let transform = transforms::Compose::new()
    .add(|t| transforms::resize(t, 224, 224))
    .add(|t| transforms::normalize(t, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]))
    .add(|t| transforms::random_horizontal_flip(t, 0.5));

let processed = transform.apply(&image)?;
```

## Performance Considerations

### Memory Management

```rust
// Monitor memory usage
let memory = utils::memory_info();
println!("Allocated: {} bytes", memory.allocated);

// Use in-place operations when possible
let mut tensor = Tensor::zeros(&[1000, 1000]);
// Prefer in-place ops to reduce memory allocation
```

### Parallelization

The framework automatically parallelizes operations using:
- BLAS/LAPACK for linear algebra
- Rayon for data parallelism
- SIMD for vectorized operations

### GPU Acceleration

```rust
// Check GPU availability
if backend::GpuBackend::is_available() {
    let gpu_backend = backend::GpuBackend::new(0);
    // Use GPU for computations
}
```

## Error Handling

The framework uses `anyhow::Result` for error handling:

```rust
use anyhow::Result;

fn train_model() -> Result<()> {
    let model = nn::Linear::new(10, 1);
    let input = Tensor::randn(&[5, 10]);
    let output = model.forward(&input)?; // ? propagates errors
    Ok(())
}
```

Common error types:
- Shape mismatches in operations
- Device compatibility issues
- Out of memory errors
- Invalid parameter values