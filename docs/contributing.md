# Contributing to Rust ML Framework

We welcome contributions to make this the best machine learning framework in Rust! This guide will help you get started.

## Getting Started

### Development Environment Setup

1. Install Rust 1.70 or later:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update
```

2. Clone the repository:
```bash
git clone <repository-url>
cd rust-ml-framework
```

3. Install dependencies:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev liblapack-dev pkg-config

# macOS
brew install openblas lapack

# Windows
# Use vcpkg or pre-built binaries
```

4. Build and test:
```bash
cargo build
cargo test
```

## Development Workflow

### Before Starting Work

1. Check existing issues and discussions
2. Create an issue to discuss major changes
3. Fork the repository and create a feature branch
4. Ensure your development environment is working

### Making Changes

1. Write tests for new functionality
2. Ensure all tests pass: `cargo test`
3. Run clippy for code quality: `cargo clippy`
4. Format code: `cargo fmt`
5. Update documentation as needed

### Submitting Changes

1. Push your feature branch to your fork
2. Create a pull request with:
   - Clear description of changes
   - Reference to related issues
   - Test results and benchmarks
   - Documentation updates

## Code Standards

### Rust Style Guide

Follow the official Rust style guide and use `rustfmt`:

```bash
cargo fmt
```

### Code Quality

Use `clippy` to catch common mistakes:

```bash
cargo clippy -- -D warnings
```

### Documentation

All public APIs must have documentation:

```rust
/// Calculate the mean squared error between predictions and targets.
/// 
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// 
/// # Returns
/// Scalar tensor containing the MSE loss
/// 
/// # Example
/// ```
/// let mse = metrics::mse(&predictions, &targets);
/// ```
pub fn mse(predictions: &Tensor, targets: &Tensor) -> f32 {
    // Implementation
}
```

## Testing Guidelines

### Unit Tests

Write comprehensive unit tests for all functionality:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Tensor::ones(&[2, 3]);
        let b = Tensor::ones(&[3, 4]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
    }
}
```

### Integration Tests

Create integration tests in the `tests/` directory:

```rust
// tests/integration_test.rs
use rust_ml_framework::*;

#[test]
fn test_training_pipeline() {
    let model = nn::mlp(784, &[128, 64], 10, "relu");
    let input = Tensor::randn(&[32, 784]);
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), &[32, 10]);
}
```

### Benchmarks

Add benchmarks for performance-critical code:

```rust
// benches/tensor_ops.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_ml_framework::*;

fn benchmark_matmul(c: &mut Criterion) {
    let a = Tensor::randn(&[1000, 1000]);
    let b = Tensor::randn(&[1000, 1000]);
    
    c.bench_function("matmul_1000x1000", |bench| {
        bench.iter(|| {
            black_box(a.matmul(&b).unwrap())
        })
    });
}

criterion_group!(benches, benchmark_matmul);
criterion_main!(benches);
```

## Types of Contributions

### Bug Fixes

1. Reproduce the bug with a minimal test case
2. Fix the issue
3. Add regression tests
4. Update documentation if needed

### New Features

1. Discuss the feature in an issue first
2. Implement with comprehensive tests
3. Add documentation and examples
4. Consider performance implications
5. Update the changelog

### Performance Improvements

1. Add benchmarks to measure current performance
2. Implement optimizations
3. Verify improvements with benchmarks
4. Ensure correctness is maintained
5. Document performance characteristics

### Documentation

1. API documentation for all public functions
2. Code examples that compile and run
3. Tutorials for common use cases
4. Performance guides and best practices

## Architecture Guidelines

### Module Organization

```
src/
├── tensor.rs              # Core tensor operations
├── nn/                    # Neural network layers
│   ├── mod.rs            # Module trait and utilities
│   ├── linear.rs         # Linear layers
│   ├── conv.rs           # Convolutional layers
│   └── ...
├── optim/                # Optimizers
├── loss/                 # Loss functions
├── data/                 # Data loading
└── ...
```

### Design Principles

1. **Performance**: Zero-cost abstractions when possible
2. **Safety**: Leverage Rust's type system for correctness
3. **Ergonomics**: APIs should be intuitive and easy to use
4. **Modularity**: Components should be composable
5. **Compatibility**: Similar APIs to PyTorch/TensorFlow when appropriate

### Error Handling

Use `anyhow::Result` for error propagation:

```rust
use anyhow::{Result, anyhow};

pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
    if input.shape().len() != 2 {
        return Err(anyhow!("Expected 2D input, got {}D", input.shape().len()));
    }
    // Implementation
}
```

## Specific Contribution Areas

### High Priority

1. **Automatic Differentiation**: Complete gradient computation
2. **GPU Support**: CUDA kernel implementations
3. **Convolution Optimization**: Fast convolution algorithms
4. **Pre-trained Models**: Model zoo with common architectures

### Medium Priority

1. **Distributed Training**: Multi-GPU and multi-node support
2. **Model Serving**: Inference server implementation
3. **Python Bindings**: PyO3-based Python interface
4. **Advanced Optimizers**: LAMB, lookahead, etc.

### Documentation Needed

1. **Tutorials**: Step-by-step guides for common tasks
2. **Examples**: Real-world use cases and benchmarks
3. **API Docs**: Complete function documentation
4. **Performance Guide**: Optimization strategies

## Review Process

### Code Review Checklist

- [ ] Code follows Rust style guidelines
- [ ] Tests cover new functionality
- [ ] Documentation is complete and accurate
- [ ] Performance impact is considered
- [ ] Breaking changes are justified and documented
- [ ] Examples compile and run correctly

### Performance Review

- [ ] Benchmarks show expected performance
- [ ] Memory usage is reasonable
- [ ] No performance regressions
- [ ] SIMD/GPU optimizations where appropriate

### Security Review

- [ ] No unsafe code without justification
- [ ] Input validation for public APIs
- [ ] No buffer overflows or memory leaks
- [ ] Dependencies are secure and up-to-date

## Communication

### Issues

Use clear, descriptive titles and provide:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Rust version, etc.)
- Minimal code example

### Pull Requests

- Reference related issues
- Describe changes and motivation
- Include test results
- Update documentation
- Keep changes focused and atomic

### Discussions

- Use GitHub Discussions for questions and ideas
- Be respectful and constructive
- Help others learn and contribute
- Share knowledge and best practices

## Recognition

Contributors are recognized through:
- GitHub contributors list
- Changelog acknowledgments
- Community highlights
- Maintainer nominations for significant contributions

## Getting Help

### Development Questions

- Check existing documentation and examples
- Search closed issues and discussions
- Ask specific questions with context
- Provide minimal reproducible examples

### Technical Support

- Use GitHub Issues for bugs
- Use GitHub Discussions for questions
- Tag maintainers for urgent issues
- Be patient and respectful

## Maintenance

### Release Process

1. Update version numbers
2. Update changelog
3. Run full test suite
4. Create release tags
5. Publish to crates.io
6. Update documentation

### Backwards Compatibility

- Follow semantic versioning
- Deprecate before removing features
- Provide migration guides
- Maintain compatibility when possible

## Community Guidelines

### Code of Conduct

- Be welcoming and inclusive
- Respect different viewpoints
- Focus on technical merit
- Help create a positive environment
- Report inappropriate behavior

### Best Practices

- Start small and build up
- Ask questions early and often
- Share knowledge and learnings
- Celebrate others' contributions
- Learn from feedback and mistakes

Thank you for contributing to the Rust ML Framework! Together we can build the future of machine learning in Rust.