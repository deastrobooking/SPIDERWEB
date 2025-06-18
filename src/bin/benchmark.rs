//! Benchmarking utilities for the ML framework

use rust_ml_framework::*;
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Rust ML Framework Performance Benchmarks");
    println!("============================================");
    
    benchmark_tensor_ops()?;
    benchmark_neural_networks()?;
    benchmark_training_loops()?;
    
    Ok(())
}

fn benchmark_tensor_ops() -> Result<()> {
    println!("\n📊 Tensor Operations Benchmark");
    println!("------------------------------");
    
    let sizes = vec![100, 500, 1000, 2000];
    
    for size in sizes {
        println!("\nMatrix size: {}x{}", size, size);
        
        // Matrix multiplication benchmark
        let a = Tensor::randn(&[size, size]);
        let b = Tensor::randn(&[size, size]);
        
        let start = Instant::now();
        let _result = a.matmul(&b)?;
        let duration = start.elapsed();
        
        println!("- Matrix multiplication: {:?}", duration);
        
        // Element-wise operations
        let start = Instant::now();
        let _result = &a + &b;
        let duration = start.elapsed();
        
        println!("- Element-wise addition: {:?}", duration);
        
        let start = Instant::now();
        let _result = a.exp();
        let duration = start.elapsed();
        
        println!("- Exponential function: {:?}", duration);
        
        // Memory usage
        let memory = utils::memory_info();
        println!("- Memory allocated: {} bytes", memory.allocated);
    }
    
    Ok(())
}

fn benchmark_neural_networks() -> Result<()> {
    println!("\n🧠 Neural Network Benchmark");
    println!("---------------------------");
    
    let batch_sizes = vec![1, 32, 128, 512];
    let input_size = 784;
    let hidden_sizes = vec![256, 128, 64];
    let output_size = 10;
    
    for batch_size in batch_sizes {
        println!("\nBatch size: {}", batch_size);
        
        let model = nn::mlp(input_size, &hidden_sizes, output_size, "relu");
        let input = Tensor::randn(&[batch_size, input_size]);
        
        // Forward pass benchmark
        let start = Instant::now();
        let _output = model.forward(&input)?;
        let forward_time = start.elapsed();
        
        println!("- Forward pass: {:?}", forward_time);
        
        // Throughput calculation
        let samples_per_second = batch_size as f64 / forward_time.as_secs_f64();
        println!("- Throughput: {:.0} samples/sec", samples_per_second);
    }
    
    // CNN benchmark
    println!("\nCNN Benchmark:");
    let cnn = vision::resnet18(1000);
    let input = Tensor::randn(&[32, 3, 224, 224]);
    
    let start = Instant::now();
    let _output = cnn.forward(&input)?;
    let duration = start.elapsed();
    
    println!("- ResNet-18 forward pass (32 images): {:?}", duration);
    
    Ok(())
}

fn benchmark_training_loops() -> Result<()> {
    println!("\n🏋️ Training Loop Benchmark");
    println!("--------------------------");
    
    let model = nn::mlp(100, &[50, 25], 10, "relu");
    let batch_size = 32;
    let input = Tensor::randn(&[batch_size, 100]);
    let target = Tensor::randn(&[batch_size, 10]);
    
    let loss_fn = loss::MSELoss::new();
    
    // Benchmark different optimizers
    let optimizers = vec!["SGD", "Adam", "AdamW"];
    
    for optimizer_name in optimizers {
        println!("\nOptimizer: {}", optimizer_name);
        
        let start = Instant::now();
        
        // Simulate training steps
        for _ in 0..100 {
            let output = model.forward(&input)?;
            let _loss = loss_fn.forward(&output, &target)?;
            // Gradient computation and optimizer step would happen here
        }
        
        let duration = start.elapsed();
        let steps_per_second = 100.0 / duration.as_secs_f64();
        
        println!("- 100 training steps: {:?}", duration);
        println!("- Steps per second: {:.1}", steps_per_second);
    }
    
    Ok(())
}