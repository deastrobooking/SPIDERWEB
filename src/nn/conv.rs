//! Convolutional layer implementations

use crate::tensor::Tensor;
use crate::nn::{Module, InitType, init_parameter};
use anyhow::{Result, anyhow};

/// 1D Convolutional layer
#[derive(Debug, Clone)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl Conv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut weight = Tensor::zeros(&[out_channels, in_channels, kernel_size]);
        let mut bias = Tensor::zeros(&[out_channels]);
        
        init_parameter(&mut weight, InitType::KaimingUniform, in_channels * kernel_size, out_channels);
        init_parameter(&mut bias, InitType::Zero, in_channels, out_channels);
        
        Self {
            weight: weight.requires_grad(true),
            bias: Some(bias.requires_grad(true)),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified 1D convolution - would need optimized implementation
        // Input shape: (batch_size, in_channels, length)
        // Output shape: (batch_size, out_channels, output_length)
        
        let batch_size = input.shape()[0];
        let input_length = input.shape()[2];
        let output_length = (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        let output = Tensor::zeros(&[batch_size, self.out_channels, output_length]);
        
        // This is a placeholder implementation - real conv would use optimized algorithms
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "Conv1d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// 2D Convolutional layer
#[derive(Debug, Clone)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut weight = Tensor::zeros(&[out_channels, in_channels, kernel_size, kernel_size]);
        let mut bias = Tensor::zeros(&[out_channels]);
        
        init_parameter(&mut weight, InitType::KaimingUniform, 
                      in_channels * kernel_size * kernel_size, out_channels);
        init_parameter(&mut bias, InitType::Zero, in_channels, out_channels);
        
        Self {
            weight: weight.requires_grad(true),
            bias: Some(bias.requires_grad(true)),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input shape: (batch_size, in_channels, height, width)
        // Output shape: (batch_size, out_channels, out_height, out_width)
        
        let batch_size = input.shape()[0];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];
        
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        let output = Tensor::zeros(&[batch_size, self.out_channels, output_height, output_width]);
        
        // Placeholder implementation - real conv2d would use optimized algorithms like im2col + GEMM
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "Conv2d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// 3D Convolutional layer
#[derive(Debug, Clone)]
pub struct Conv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl Conv3d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut weight = Tensor::zeros(&[out_channels, in_channels, kernel_size, kernel_size, kernel_size]);
        let mut bias = Tensor::zeros(&[out_channels]);
        
        init_parameter(&mut weight, InitType::KaimingUniform, 
                      in_channels * kernel_size * kernel_size * kernel_size, out_channels);
        init_parameter(&mut bias, InitType::Zero, in_channels, out_channels);
        
        Self {
            weight: weight.requires_grad(true),
            bias: Some(bias.requires_grad(true)),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Module for Conv3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input shape: (batch_size, in_channels, depth, height, width)
        // Output shape: (batch_size, out_channels, out_depth, out_height, out_width)
        
        let batch_size = input.shape()[0];
        let input_depth = input.shape()[2];
        let input_height = input.shape()[3];
        let input_width = input.shape()[4];
        
        let output_depth = (input_depth + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        let output = Tensor::zeros(&[batch_size, self.out_channels, output_depth, output_height, output_width]);
        
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "Conv3d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2d::new(3, 64, 3, 1, 1);
        assert_eq!(conv.weight.shape(), &[64, 3, 3, 3]);
        assert_eq!(conv.bias.as_ref().unwrap().shape(), &[64]);
    }
}