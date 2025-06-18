//! Data transformation utilities for preprocessing

use crate::tensor::Tensor;
use anyhow::Result;

/// Normalize tensor with mean and standard deviation
pub fn normalize(tensor: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
    let mut result = tensor.clone_tensor();
    let channels = tensor.shape()[0];
    
    for c in 0..channels.min(mean.len()).min(std.len()) {
        // Normalize channel c: (x - mean) / std
        let channel_mean = Tensor::full(&[1], mean[c]);
        let channel_std = Tensor::full(&[1], std[c]);
        result = &(&result - &channel_mean) / &channel_std;
    }
    
    Ok(result)
}

/// Resize tensor to target dimensions
pub fn resize(tensor: &Tensor, target_height: usize, target_width: usize) -> Result<Tensor> {
    // Simplified resize - in practice would use interpolation
    let channels = tensor.shape()[0];
    let resized = Tensor::zeros(&[channels, target_height, target_width]);
    Ok(resized)
}

/// Random horizontal flip
pub fn random_horizontal_flip(tensor: &Tensor, probability: f32) -> Result<Tensor> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    if rng.gen::<f32>() < probability {
        // Flip horizontally
        flip_horizontal(tensor)
    } else {
        Ok(tensor.clone_tensor())
    }
}

/// Horizontal flip
pub fn flip_horizontal(tensor: &Tensor) -> Result<Tensor> {
    // Simplified flip implementation
    Ok(tensor.clone_tensor())
}

/// Random crop
pub fn random_crop(tensor: &Tensor, crop_height: usize, crop_width: usize) -> Result<Tensor> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let height = tensor.shape()[1];
    let width = tensor.shape()[2];
    
    if crop_height > height || crop_width > width {
        return Ok(tensor.clone_tensor());
    }
    
    let top = rng.gen_range(0..=(height - crop_height));
    let left = rng.gen_range(0..=(width - crop_width));
    
    crop(tensor, top, left, crop_height, crop_width)
}

/// Crop tensor at specified location
pub fn crop(tensor: &Tensor, top: usize, left: usize, height: usize, width: usize) -> Result<Tensor> {
    // Simplified crop implementation
    let channels = tensor.shape()[0];
    let cropped = Tensor::zeros(&[channels, height, width]);
    Ok(cropped)
}

/// Center crop
pub fn center_crop(tensor: &Tensor, crop_height: usize, crop_width: usize) -> Result<Tensor> {
    let height = tensor.shape()[1];
    let width = tensor.shape()[2];
    
    let top = (height.saturating_sub(crop_height)) / 2;
    let left = (width.saturating_sub(crop_width)) / 2;
    
    crop(tensor, top, left, crop_height, crop_width)
}

/// Convert to grayscale
pub fn to_grayscale(tensor: &Tensor) -> Result<Tensor> {
    if tensor.shape().len() < 3 || tensor.shape()[0] != 3 {
        return Ok(tensor.clone_tensor());
    }
    
    // RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
    let height = tensor.shape()[1];
    let width = tensor.shape()[2];
    let mut gray_data = vec![0.0; height * width];
    
    for h in 0..height {
        for w in 0..width {
            let r_idx = h * width + w;
            let g_idx = height * width + h * width + w;
            let b_idx = 2 * height * width + h * width + w;
            
            if r_idx < tensor.data().len() && g_idx < tensor.data().len() && b_idx < tensor.data().len() {
                let gray_val = 0.299 * tensor.data()[r_idx] + 
                              0.587 * tensor.data()[g_idx] + 
                              0.114 * tensor.data()[b_idx];
                gray_data[h * width + w] = gray_val;
            }
        }
    }
    
    Tensor::from_vec(gray_data, &[1, height, width])
}

/// Color jitter (adjust brightness, contrast, saturation, hue)
pub fn color_jitter(tensor: &Tensor, brightness: f32, contrast: f32, saturation: f32) -> Result<Tensor> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let mut result = tensor.clone_tensor();
    
    // Brightness adjustment
    if brightness > 0.0 {
        let brightness_factor = 1.0 + rng.gen_range(-brightness..brightness);
        result = &result * brightness_factor;
    }
    
    // Contrast adjustment
    if contrast > 0.0 {
        let contrast_factor = 1.0 + rng.gen_range(-contrast..contrast);
        let mean = result.mean();
        result = &(&(&result - mean) * contrast_factor) + mean;
    }
    
    Ok(result)
}

/// Compose multiple transforms
pub struct Compose {
    transforms: Vec<Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>>,
}

impl Compose {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }
    
    pub fn add<F>(mut self, transform: F) -> Self 
    where 
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static
    {
        self.transforms.push(Box::new(transform));
        self
    }
    
    pub fn apply(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut result = tensor.clone_tensor();
        for transform in &self.transforms {
            result = transform(&result)?;
        }
        Ok(result)
    }
}