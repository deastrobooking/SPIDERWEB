//! Pooling layer implementations

use crate::tensor::Tensor;
use crate::nn::Module;
use anyhow::Result;

/// 1D Max Pooling
#[derive(Debug, Clone)]
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl MaxPool1d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            training: true,
        }
    }
    
    pub fn with_padding(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Module for MaxPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_length = input.shape()[2];
        
        let output_length = (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels * output_length];
        
        for b in 0..batch_size {
            for c in 0..channels {
                for out_pos in 0..output_length {
                    let start = out_pos * self.stride;
                    let end = (start + self.kernel_size).min(input_length);
                    
                    let mut max_val = f32::NEG_INFINITY;
                    for i in start..end {
                        let input_idx = ((b * channels + c) * input_length + i);
                        if input_idx < input.data().len() {
                            max_val = max_val.max(input.data()[input_idx]);
                        }
                    }
                    
                    let output_idx = (b * channels + c) * output_length + out_pos;
                    output_data[output_idx] = max_val;
                }
            }
        }
        
        Tensor::from_vec(output_data, &[batch_size, channels, output_length])
    }
    
    fn parameters(&self) -> Vec<&Tensor> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { Vec::new() }
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "MaxPool1d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// 2D Max Pooling
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            training: true,
        }
    }
    
    pub fn with_padding(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];
        
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels * output_height * output_width];
        
        for b in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let start_h = out_h * self.stride;
                        let start_w = out_w * self.stride;
                        let end_h = (start_h + self.kernel_size).min(input_height);
                        let end_w = (start_w + self.kernel_size).min(input_width);
                        
                        let mut max_val = f32::NEG_INFINITY;
                        for h in start_h..end_h {
                            for w in start_w..end_w {
                                let input_idx = ((b * channels + c) * input_height + h) * input_width + w;
                                if input_idx < input.data().len() {
                                    max_val = max_val.max(input.data()[input_idx]);
                                }
                            }
                        }
                        
                        let output_idx = ((b * channels + c) * output_height + out_h) * output_width + out_w;
                        output_data[output_idx] = max_val;
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, &[batch_size, channels, output_height, output_width])
    }
    
    fn parameters(&self) -> Vec<&Tensor> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { Vec::new() }
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "MaxPool2d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// 1D Average Pooling
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl AvgPool1d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            training: true,
        }
    }
}

impl Module for AvgPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_length = input.shape()[2];
        
        let output_length = (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output_data = vec![0.0; batch_size * channels * output_length];
        
        for b in 0..batch_size {
            for c in 0..channels {
                for out_pos in 0..output_length {
                    let start = out_pos * self.stride;
                    let end = (start + self.kernel_size).min(input_length);
                    
                    let mut sum = 0.0;
                    let mut count = 0;
                    for i in start..end {
                        let input_idx = ((b * channels + c) * input_length + i);
                        if input_idx < input.data().len() {
                            sum += input.data()[input_idx];
                            count += 1;
                        }
                    }
                    
                    let output_idx = (b * channels + c) * output_length + out_pos;
                    output_data[output_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
        
        Tensor::from_vec(output_data, &[batch_size, channels, output_length])
    }
    
    fn parameters(&self) -> Vec<&Tensor> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { Vec::new() }
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "AvgPool1d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// 2D Average Pooling
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            training: true,
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];
        
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        let mut output_data = vec![0.0; batch_size * channels * output_height * output_width];
        
        for b in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let start_h = out_h * self.stride;
                        let start_w = out_w * self.stride;
                        let end_h = (start_h + self.kernel_size).min(input_height);
                        let end_w = (start_w + self.kernel_size).min(input_width);
                        
                        let mut sum = 0.0;
                        let mut count = 0;
                        for h in start_h..end_h {
                            for w in start_w..end_w {
                                let input_idx = ((b * channels + c) * input_height + h) * input_width + w;
                                if input_idx < input.data().len() {
                                    sum += input.data()[input_idx];
                                    count += 1;
                                }
                            }
                        }
                        
                        let output_idx = ((b * channels + c) * output_height + out_h) * output_width + out_w;
                        output_data[output_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, &[batch_size, channels, output_height, output_width])
    }
    
    fn parameters(&self) -> Vec<&Tensor> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { Vec::new() }
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "AvgPool2d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Adaptive Average Pooling 2D
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    output_size: usize,
    training: bool,
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: usize) -> Self {
        Self {
            output_size,
            training: true,
        }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];
        
        let output_height = self.output_size;
        let output_width = self.output_size;
        
        let mut output_data = vec![0.0; batch_size * channels * output_height * output_width];
        
        for b in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let start_h = (out_h * input_height) / output_height;
                        let end_h = ((out_h + 1) * input_height) / output_height;
                        let start_w = (out_w * input_width) / output_width;
                        let end_w = ((out_w + 1) * input_width) / output_width;
                        
                        let mut sum = 0.0;
                        let mut count = 0;
                        for h in start_h..end_h {
                            for w in start_w..end_w {
                                let input_idx = ((b * channels + c) * input_height + h) * input_width + w;
                                if input_idx < input.data().len() {
                                    sum += input.data()[input_idx];
                                    count += 1;
                                }
                            }
                        }
                        
                        let output_idx = ((b * channels + c) * output_height + out_h) * output_width + out_w;
                        output_data[output_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, &[batch_size, channels, output_height, output_width])
    }
    
    fn parameters(&self) -> Vec<&Tensor> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { Vec::new() }
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "AdaptiveAvgPool2d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d() {
        let pool = MaxPool2d::new(2, 2);
        let input = Tensor::ones(&[1, 1, 4, 4]);
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_avgpool2d() {
        let pool = AvgPool2d::new(2, 2);
        let input = Tensor::ones(&[1, 1, 4, 4]);
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output.data()[0], 1.0); // Average of ones is one
    }
}