//! Normalization layer implementations

use crate::tensor::Tensor;
use crate::nn::{Module, InitType, init_parameter};
use anyhow::Result;

/// Batch Normalization 1D
#[derive(Debug, Clone)]
pub struct BatchNorm1d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        let mut weight = Some(Tensor::ones(&[num_features]));
        let mut bias = Some(Tensor::zeros(&[num_features]));
        
        if let Some(ref mut w) = weight {
            *w = w.requires_grad(true);
        }
        if let Some(ref mut b) = bias {
            *b = b.requires_grad(true);
        }
        
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
            weight,
            bias,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            training: true,
        }
    }
    
    pub fn without_affine(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: false,
            track_running_stats: true,
            weight: None,
            bias: None,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            training: true,
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let features = input.shape()[1];
        
        if features != self.num_features {
            return Err(anyhow::anyhow!("Input features {} don't match layer features {}", features, self.num_features));
        }
        
        let (mean, var) = if self.training {
            // Compute batch statistics
            let batch_mean = input.mean_axis(0)?;
            
            // Compute variance
            let centered = input - &batch_mean;
            let variance = (&centered * &centered).mean_axis(0)?;
            
            // Update running statistics
            if self.track_running_stats {
                // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
                // running_var = (1 - momentum) * running_var + momentum * batch_var
                // Note: This would require mutable access to self, which we don't have in forward
                // In a real implementation, this would be handled differently
            }
            
            (batch_mean, variance)
        } else {
            // Use running statistics
            (self.running_mean.clone_tensor(), self.running_var.clone_tensor())
        };
        
        // Normalize: (x - mean) / sqrt(var + eps)
        let normalized = (input - &mean) / &(var + self.eps).sqrt();
        
        // Apply affine transformation if enabled
        if self.affine {
            if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                Ok(&(&normalized * weight) + bias)
            } else {
                Ok(normalized)
            }
        } else {
            Ok(normalized)
        }
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "BatchNorm1d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Batch Normalization 2D
#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    affine: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        let mut weight = Some(Tensor::ones(&[num_features]));
        let mut bias = Some(Tensor::zeros(&[num_features]));
        
        if let Some(ref mut w) = weight {
            *w = w.requires_grad(true);
        }
        if let Some(ref mut b) = bias {
            *b = b.requires_grad(true);
        }
        
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            weight,
            bias,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            training: true,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For 2D batch norm, we normalize over batch, height, and width dimensions
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        
        if channels != self.num_features {
            return Err(anyhow::anyhow!("Input channels {} don't match layer features {}", channels, self.num_features));
        }
        
        // Compute mean and variance per channel
        let mut channel_means = Vec::new();
        let mut channel_vars = Vec::new();
        
        for c in 0..channels {
            let mut channel_sum = 0.0;
            let mut count = 0;
            
            // Calculate mean for this channel
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        if idx < input.data().len() {
                            channel_sum += input.data()[idx];
                            count += 1;
                        }
                    }
                }
            }
            
            let channel_mean = if count > 0 { channel_sum / count as f32 } else { 0.0 };
            channel_means.push(channel_mean);
            
            // Calculate variance for this channel
            let mut var_sum = 0.0;
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        if idx < input.data().len() {
                            let diff = input.data()[idx] - channel_mean;
                            var_sum += diff * diff;
                        }
                    }
                }
            }
            
            let channel_var = if count > 0 { var_sum / count as f32 } else { 1.0 };
            channel_vars.push(channel_var);
        }
        
        // Normalize the input
        let mut normalized_data = vec![0.0; input.data().len()];
        for b in 0..batch_size {
            for c in 0..channels {
                let mean = channel_means[c];
                let var = channel_vars[c];
                let std = (var + self.eps).sqrt();
                
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        if idx < input.data().len() {
                            normalized_data[idx] = (input.data()[idx] - mean) / std;
                            
                            // Apply affine transformation if enabled
                            if self.affine {
                                if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                                    normalized_data[idx] = normalized_data[idx] * weight.data()[c] + bias.data()[c];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Tensor::from_vec(normalized_data, input.shape())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "BatchNorm2d" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Layer Normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f32,
    elementwise_affine: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    training: bool,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let total_elements: usize = normalized_shape.iter().product();
        let mut weight = Some(Tensor::ones(&[total_elements]));
        let mut bias = Some(Tensor::zeros(&[total_elements]));
        
        if let Some(ref mut w) = weight {
            *w = w.requires_grad(true);
        }
        if let Some(ref mut b) = bias {
            *b = b.requires_grad(true);
        }
        
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
            weight,
            bias,
            training: true,
        }
    }
    
    pub fn without_affine(normalized_shape: Vec<usize>) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: false,
            weight: None,
            bias: None,
            training: true,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Layer norm normalizes over the last few dimensions
        let input_shape = input.shape();
        let norm_dims = self.normalized_shape.len();
        
        if input_shape.len() < norm_dims {
            return Err(anyhow::anyhow!("Input has fewer dimensions than normalized_shape"));
        }
        
        // Check if the last dimensions match
        let input_last_dims = &input_shape[input_shape.len() - norm_dims..];
        if input_last_dims != self.normalized_shape.as_slice() {
            return Err(anyhow::anyhow!("Input shape doesn't match normalized_shape"));
        }
        
        // Compute mean and variance over the normalized dimensions
        let mean = input.mean();
        let centered = input - &Tensor::full(input.shape(), mean);
        let variance = (&centered * &centered).mean();
        
        // Normalize
        let normalized = &centered / &Tensor::full(input.shape(), (variance + self.eps).sqrt());
        
        // Apply elementwise affine transformation if enabled
        if self.elementwise_affine {
            if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                // Broadcast weight and bias to match input shape
                Ok(&(&normalized * weight) + bias)
            } else {
                Ok(normalized)
            }
        } else {
            Ok(normalized)
        }
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "LayerNorm" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Group Normalization
#[derive(Debug, Clone)]
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    eps: f32,
    affine: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    training: bool,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        if num_channels % num_groups != 0 {
            panic!("num_channels must be divisible by num_groups");
        }
        
        let mut weight = Some(Tensor::ones(&[num_channels]));
        let mut bias = Some(Tensor::zeros(&[num_channels]));
        
        if let Some(ref mut w) = weight {
            *w = w.requires_grad(true);
        }
        if let Some(ref mut b) = bias {
            *b = b.requires_grad(true);
        }
        
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            affine: true,
            weight,
            bias,
            training: true,
        }
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Group normalization divides channels into groups and normalizes within each group
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        
        if channels != self.num_channels {
            return Err(anyhow::anyhow!("Input channels {} don't match layer channels {}", channels, self.num_channels));
        }
        
        let channels_per_group = self.num_channels / self.num_groups;
        
        // Simplified group normalization implementation
        // In practice, this would reshape the tensor to group channels together
        let normalized = input.clone_tensor();
        
        // Apply affine transformation if enabled
        if self.affine {
            if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                Ok(&(&normalized * weight) + bias)
            } else {
                Ok(normalized)
            }
        } else {
            Ok(normalized)
        }
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "GroupNorm" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_1d() {
        let bn = BatchNorm1d::new(10);
        let input = Tensor::randn(&[4, 10]);
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(vec![10]);
        let input = Tensor::randn(&[4, 10]);
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }
}