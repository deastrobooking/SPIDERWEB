# API Reference

Complete API documentation for the Rust ML Framework.

## Tensor Module

### Tensor

Core tensor type for multi-dimensional arrays.

```rust
pub struct Tensor {
    // Internal implementation
}
```

#### Creation Methods

```rust
impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self>
    pub fn zeros(shape: &[usize]) -> Self
    pub fn ones(shape: &[usize]) -> Self
    pub fn full(shape: &[usize], value: f32) -> Self
    pub fn rand(shape: &[usize]) -> Self
    pub fn randn(shape: &[usize]) -> Self
    pub fn eye(n: usize) -> Self
    pub fn arange(start: f32, end: f32, step: f32) -> Self
}
```

#### Properties

```rust
impl Tensor {
    pub fn shape(&self) -> &[usize]
    pub fn ndim(&self) -> usize
    pub fn numel(&self) -> usize
    pub fn device(&self) -> Device
    pub fn data(&self) -> &[f32]
    pub fn to_vec(&self) -> Vec<f32>
}
```

#### Operations

```rust
impl Tensor {
    // Shape manipulation
    pub fn reshape(&self, shape: &[usize]) -> Result<Self>
    pub fn t(&self) -> Self
    pub fn permute(&self, dims: &[usize]) -> Result<Self>
    pub fn squeeze(&self) -> Self
    pub fn unsqueeze(&self, dim: usize) -> Result<Self>
    
    // Mathematical operations
    pub fn matmul(&self, other: &Self) -> Result<Self>
    pub fn sum(&self) -> f32
    pub fn mean(&self) -> f32
    pub fn sum_axis(&self, axis: usize) -> Result<Self>
    pub fn mean_axis(&self, axis: usize) -> Result<Self>
    
    // Element-wise functions
    pub fn abs(&self) -> Self
    pub fn sqrt(&self) -> Self
    pub fn exp(&self) -> Self
    pub fn log(&self) -> Self
    pub fn sin(&self) -> Self
    pub fn cos(&self) -> Self
    pub fn pow(&self, exponent: f32) -> Self
    
    // Gradient support
    pub fn requires_grad(self, requires_grad: bool) -> Self
    pub fn grad(&self) -> Option<&Tensor>
    pub fn zero_grad(&mut self)
}
```

### Device

```rust
#[derive(Debug, Clone, Copy)]
pub enum Device {
    CPU,
    CUDA(usize),
    Metal(usize),
}
```

## Neural Network Module (nn)

### Module Trait

Base trait for all neural network components.

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
    fn training(&self) -> bool;
    fn name(&self) -> &str;
    fn clone_module(&self) -> Box<dyn Module>;
}
```

### Linear Layer

```rust
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self
    pub fn in_features(&self) -> usize
    pub fn out_features(&self) -> usize
    pub fn weight(&self) -> &Tensor
    pub fn bias(&self) -> Option<&Tensor>
}
```

### Convolutional Layers

```rust
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize
    ) -> Self
}
```

### Activation Functions

```rust
pub struct ReLU;
pub struct Sigmoid;
pub struct Tanh;
pub struct GELU;
pub struct Swish;

pub struct Softmax {
    dim: usize,
}

impl Softmax {
    pub fn new(dim: usize) -> Self
}
```

### Normalization Layers

```rust
pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
    momentum: f32,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self
}

pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self
}
```

### Sequential Container

```rust
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self
    pub fn add<M: Module + 'static>(self, module: M) -> Self
    pub fn len(&self) -> usize
}
```

### Utility Functions

```rust
pub fn mlp(
    input_size: usize,
    hidden_sizes: &[usize],
    output_size: usize,
    activation: &str
) -> Sequential

pub fn simple_cnn(
    input_channels: usize,
    num_classes: usize
) -> Sequential
```

## Optimization Module (optim)

### Optimizer Trait

```rust
pub trait Optimizer: Send + Sync {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);
    fn add_param_group(&mut self, params: Vec<*mut Tensor>);
    fn state_dict(&self) -> HashMap<String, f32>;
    fn load_state_dict(&mut self, state: HashMap<String, f32>);
}
```

### SGD Optimizer

```rust
pub struct SGD {
    params: Vec<*mut Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
}

impl SGD {
    pub fn new(params: Vec<*mut Tensor>, lr: f32) -> Self
    pub fn with_momentum(params: Vec<*mut Tensor>, lr: f32, momentum: f32) -> Self
    pub fn with_weight_decay(params: Vec<*mut Tensor>, lr: f32, weight_decay: f32) -> Self
}
```

### Adam Optimizer

```rust
pub struct Adam {
    params: Vec<*mut Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

impl Adam {
    pub fn new(params: Vec<*mut Tensor>, lr: f32) -> Self
    pub fn with_params(
        params: Vec<*mut Tensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32
    ) -> Self
}
```

### Learning Rate Schedulers

```rust
pub struct StepLR {
    step_size: usize,
    gamma: f32,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f32, base_lr: f32) -> Self
}

pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f32,
}

impl CosineAnnealingLR {
    pub fn new(t_max: usize, eta_min: f32, base_lr: f32) -> Self
}
```

## Loss Functions Module

### Loss Trait

```rust
pub trait Loss: Send + Sync {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
}
```

### Loss Functions

```rust
pub struct MSELoss {
    reduction: Reduction,
}

pub struct CrossEntropyLoss {
    reduction: Reduction,
}

pub struct BCELoss {
    reduction: Reduction,
}

pub struct HuberLoss {
    delta: f32,
    reduction: Reduction,
}

#[derive(Debug, Clone, Copy)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}
```

## Data Module

### Dataset Trait

```rust
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get_item(&self, index: usize) -> Result<(Tensor, Tensor)>;
    fn is_empty(&self) -> bool;
}
```

### DataLoader

```rust
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self
    pub fn with_shuffle(self, shuffle: bool) -> Self
    pub fn with_drop_last(self, drop_last: bool) -> Self
    pub fn reset(&mut self)
    pub fn len(&self) -> usize
    pub fn next_batch(&mut self) -> Result<Option<(Tensor, Tensor)>>
}
```

### Built-in Datasets

```rust
pub struct TensorDataset {
    data: Vec<Tensor>,
    targets: Vec<Tensor>,
}

impl TensorDataset {
    pub fn new(data: Vec<Tensor>, targets: Vec<Tensor>) -> Result<Self>
    pub fn from_arrays(data_array: Tensor, targets_array: Tensor) -> Result<Self>
}

pub struct CSVDataset {
    data: Vec<Vec<f32>>,
    targets: Vec<f32>,
    feature_dim: usize,
}

impl CSVDataset {
    pub fn from_file<P: AsRef<Path>>(path: P, target_column: usize) -> Result<Self>
    pub fn feature_dim(&self) -> usize
}
```

## Metrics Module

### Classification Metrics

```rust
pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f32
pub fn precision(predictions: &Tensor, targets: &Tensor) -> f32
pub fn recall(predictions: &Tensor, targets: &Tensor) -> f32
pub fn f1_score(predictions: &Tensor, targets: &Tensor) -> f32
pub fn auc_roc(predictions: &Tensor, targets: &Tensor) -> f32
```

### Regression Metrics

```rust
pub fn mse(predictions: &Tensor, targets: &Tensor) -> f32
pub fn mae(predictions: &Tensor, targets: &Tensor) -> f32
pub fn r2_score(predictions: &Tensor, targets: &Tensor) -> f32
```

## Vision Module

### Pre-built Architectures

```rust
pub fn resnet18(num_classes: usize) -> Sequential
pub fn vgg11(num_classes: usize) -> Sequential
pub fn simple_cnn(input_channels: usize, num_classes: usize) -> Sequential
```

### ResNet Block

```rust
pub struct ResNetBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    relu: ReLU,
    downsample: Option<Sequential>,
}

impl ResNetBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self
}
```

## NLP Module

### Tokenizer

```rust
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    vocab_size: usize,
}

impl Tokenizer {
    pub fn new() -> Self
    pub fn build_vocab(&mut self, texts: &[String], min_freq: usize)
    pub fn encode(&self, text: &str) -> Vec<usize>
    pub fn decode(&self, token_ids: &[usize]) -> String
    pub fn vocab_size(&self) -> usize
}
```

### Embedding Layer

```rust
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self
    pub fn from_pretrained(embeddings: Tensor, freeze: bool) -> Self
}
```

## Transforms Module

### Image Transforms

```rust
pub fn normalize(tensor: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor>
pub fn resize(tensor: &Tensor, target_height: usize, target_width: usize) -> Result<Tensor>
pub fn random_horizontal_flip(tensor: &Tensor, probability: f32) -> Result<Tensor>
pub fn center_crop(tensor: &Tensor, crop_height: usize, crop_width: usize) -> Result<Tensor>
pub fn to_grayscale(tensor: &Tensor) -> Result<Tensor>
```

### Transform Composition

```rust
pub struct Compose {
    transforms: Vec<Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>>,
}

impl Compose {
    pub fn new() -> Self
    pub fn add<F>(self, transform: F) -> Self 
    where F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static
    pub fn apply(&self, tensor: &Tensor) -> Result<Tensor>
}
```

## Utilities Module

### Training Utilities

```rust
pub fn save_tensor<P: AsRef<Path>>(tensor: &Tensor, path: P) -> Result<()>
pub fn load_tensor<P: AsRef<Path>>(path: P) -> Result<Tensor>
pub fn set_seed(seed: u64)
pub fn count_parameters<M: Module>(model: &M) -> usize
```

### Progress Tracking

```rust
pub struct ProgressBar {
    total: usize,
    current: usize,
}

impl ProgressBar {
    pub fn new(total: usize) -> Self
    pub fn update(&mut self, current: usize)
    pub fn finish(self)
}
```

### Early Stopping

```rust
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32) -> Self
    pub fn should_stop(&mut self, loss: f32) -> bool
}
```

### Gradient Utilities

```rust
pub fn clip_grad_norm(parameters: &mut [&mut Tensor], max_norm: f32) -> f32
pub fn clip_grad_value(parameters: &mut [&mut Tensor], clip_value: f32)
```

### Learning Rate Scheduling

```rust
pub fn cosine_annealing_lr(epoch: usize, max_epochs: usize, initial_lr: f32, min_lr: f32) -> f32
pub fn exponential_decay_lr(epoch: usize, initial_lr: f32, decay_rate: f32) -> f32
pub fn step_decay_lr(epoch: usize, initial_lr: f32, step_size: usize, gamma: f32) -> f32
```