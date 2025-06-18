//! Data loading and preprocessing utilities
//! 
//! This module provides dataset handling and data loading functionality
//! similar to PyTorch's DataLoader and TensorFlow's tf.data API.

use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Base trait for datasets
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get_item(&self, index: usize) -> Result<(Tensor, Tensor)>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// DataLoader for batched data loading
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: Vec<usize>,
    current_batch: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        let dataset_len = dataset.len();
        let indices: Vec<usize> = (0..dataset_len).collect();
        
        Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            indices,
            current_batch: 0,
        }
    }
    
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        if shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
        self
    }
    
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
    
    pub fn reset(&mut self) {
        self.current_batch = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }
    
    pub fn len(&self) -> usize {
        let dataset_len = self.dataset.len();
        if self.drop_last {
            dataset_len / self.batch_size
        } else {
            (dataset_len + self.batch_size - 1) / self.batch_size
        }
    }
    
    pub fn next_batch(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        let start_idx = self.current_batch * self.batch_size;
        let dataset_len = self.dataset.len();
        
        if start_idx >= dataset_len {
            return Ok(None);
        }
        
        let end_idx = (start_idx + self.batch_size).min(dataset_len);
        let actual_batch_size = end_idx - start_idx;
        
        if self.drop_last && actual_batch_size < self.batch_size {
            return Ok(None);
        }
        
        let mut batch_data = Vec::new();
        let mut batch_targets = Vec::new();
        
        for i in start_idx..end_idx {
            let idx = self.indices[i];
            let (data, target) = self.dataset.get_item(idx)?;
            batch_data.push(data);
            batch_targets.push(target);
        }
        
        // Stack tensors into batches
        let stacked_data = stack_tensors(&batch_data)?;
        let stacked_targets = stack_tensors(&batch_targets)?;
        
        self.current_batch += 1;
        Ok(Some((stacked_data, stacked_targets)))
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = Result<(Tensor, Tensor)>;
    
    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// In-memory tensor dataset
pub struct TensorDataset {
    data: Vec<Tensor>,
    targets: Vec<Tensor>,
}

impl TensorDataset {
    pub fn new(data: Vec<Tensor>, targets: Vec<Tensor>) -> Result<Self> {
        if data.len() != targets.len() {
            return Err(anyhow!("Data and targets must have the same length"));
        }
        Ok(Self { data, targets })
    }
    
    pub fn from_arrays(data_array: Tensor, targets_array: Tensor) -> Result<Self> {
        let batch_size = data_array.shape()[0];
        if targets_array.shape()[0] != batch_size {
            return Err(anyhow!("Data and targets must have the same batch size"));
        }
        
        let mut data = Vec::new();
        let mut targets = Vec::new();
        
        for i in 0..batch_size {
            // Extract individual samples - simplified implementation
            let sample_data = data_array.clone_tensor();
            let sample_target = targets_array.clone_tensor();
            data.push(sample_data);
            targets.push(sample_target);
        }
        
        Ok(Self { data, targets })
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get_item(&self, index: usize) -> Result<(Tensor, Tensor)> {
        if index >= self.len() {
            return Err(anyhow!("Index out of bounds"));
        }
        Ok((self.data[index].clone_tensor(), self.targets[index].clone_tensor()))
    }
}

/// CSV dataset loader
pub struct CSVDataset {
    data: Vec<Vec<f32>>,
    targets: Vec<f32>,
    feature_dim: usize,
}

impl CSVDataset {
    pub fn from_file<P: AsRef<Path>>(path: P, target_column: usize) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut data = Vec::new();
        let mut targets = Vec::new();
        let mut feature_dim = 0;
        
        for result in reader.records() {
            let record = result?;
            let mut row = Vec::new();
            let mut target = 0.0;
            
            for (i, field) in record.iter().enumerate() {
                let value: f32 = field.parse()
                    .map_err(|_| anyhow!("Failed to parse field as float: {}", field))?;
                
                if i == target_column {
                    target = value;
                } else {
                    row.push(value);
                }
            }
            
            if feature_dim == 0 {
                feature_dim = row.len();
            } else if row.len() != feature_dim {
                return Err(anyhow!("Inconsistent number of features"));
            }
            
            data.push(row);
            targets.push(target);
        }
        
        Ok(Self {
            data,
            targets,
            feature_dim,
        })
    }
    
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }
}

impl Dataset for CSVDataset {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get_item(&self, index: usize) -> Result<(Tensor, Tensor)> {
        if index >= self.len() {
            return Err(anyhow!("Index out of bounds"));
        }
        
        let data_tensor = Tensor::from_vec(self.data[index].clone(), &[self.feature_dim])?;
        let target_tensor = Tensor::from_vec(vec![self.targets[index]], &[1])?;
        
        Ok((data_tensor, target_tensor))
    }
}

/// Image dataset for computer vision tasks
pub struct ImageDataset {
    image_paths: Vec<String>,
    labels: Vec<usize>,
    transform: Option<Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>>,
}

impl ImageDataset {
    pub fn from_folder<P: AsRef<Path>>(folder: P) -> Result<Self> {
        let mut image_paths = Vec::new();
        let mut labels = Vec::new();
        let mut class_map = HashMap::new();
        let mut next_class_id = 0;
        
        for entry in std::fs::read_dir(folder)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let class_name = path.file_name().unwrap().to_string_lossy().to_string();
                let class_id = *class_map.entry(class_name).or_insert_with(|| {
                    let id = next_class_id;
                    next_class_id += 1;
                    id
                });
                
                for image_entry in std::fs::read_dir(&path)? {
                    let image_entry = image_entry?;
                    let image_path = image_entry.path();
                    
                    if let Some(ext) = image_path.extension() {
                        let ext = ext.to_string_lossy().to_lowercase();
                        if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp") {
                            image_paths.push(image_path.to_string_lossy().to_string());
                            labels.push(class_id);
                        }
                    }
                }
            }
        }
        
        Ok(Self {
            image_paths,
            labels,
            transform: None,
        })
    }
    
    pub fn with_transform<F>(mut self, transform: F) -> Self 
    where 
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static
    {
        self.transform = Some(Box::new(transform));
        self
    }
    
    pub fn num_classes(&self) -> usize {
        self.labels.iter().max().map(|&x| x + 1).unwrap_or(0)
    }
}

impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get_item(&self, index: usize) -> Result<(Tensor, Tensor)> {
        if index >= self.len() {
            return Err(anyhow!("Index out of bounds"));
        }
        
        // Load image - simplified implementation
        let image_tensor = load_image(&self.image_paths[index])?;
        let transformed_image = if let Some(ref transform) = self.transform {
            transform(&image_tensor)?
        } else {
            image_tensor
        };
        
        let label_tensor = Tensor::from_vec(vec![self.labels[index] as f32], &[1])?;
        
        Ok((transformed_image, label_tensor))
    }
}

// Helper functions
fn stack_tensors(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(anyhow!("Cannot stack empty tensor list"));
    }
    
    let first_shape = tensors[0].shape();
    let batch_size = tensors.len();
    let mut stacked_shape = vec![batch_size];
    stacked_shape.extend_from_slice(first_shape);
    
    let mut stacked_data = Vec::new();
    for tensor in tensors {
        if tensor.shape() != first_shape {
            return Err(anyhow!("All tensors must have the same shape"));
        }
        stacked_data.extend_from_slice(tensor.data());
    }
    
    Tensor::from_vec(stacked_data, &stacked_shape)
}

fn load_image(path: &str) -> Result<Tensor> {
    // Simplified image loading - would use image crate in real implementation
    // For now, return a dummy tensor representing a 3-channel RGB image
    let dummy_image = Tensor::randn(&[3, 224, 224]);
    Ok(dummy_image)
}

/// Data preprocessing transforms
pub mod transforms {
    use super::*;
    
    pub fn normalize(mean: &[f32], std: &[f32]) -> impl Fn(&Tensor) -> Result<Tensor> {
        let mean = mean.to_vec();
        let std = std.to_vec();
        move |tensor: &Tensor| {
            let mut normalized = tensor.clone_tensor();
            // Apply normalization: (x - mean) / std
            for (i, (&m, &s)) in mean.iter().zip(std.iter()).enumerate() {
                // Simplified normalization - would need proper channel-wise operations
            }
            Ok(normalized)
        }
    }
    
    pub fn resize(height: usize, width: usize) -> impl Fn(&Tensor) -> Result<Tensor> {
        move |tensor: &Tensor| {
            // Simplified resize - would use proper image interpolation
            let resized = Tensor::zeros(&[tensor.shape()[0], height, width]);
            Ok(resized)
        }
    }
    
    pub fn to_tensor() -> impl Fn(&Tensor) -> Result<Tensor> {
        |tensor: &Tensor| Ok(tensor.clone_tensor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dataset() {
        let data = vec![
            Tensor::ones(&[2]),
            Tensor::zeros(&[2]),
        ];
        let targets = vec![
            Tensor::from_vec(vec![1.0], &[1]).unwrap(),
            Tensor::from_vec(vec![0.0], &[1]).unwrap(),
        ];
        
        let dataset = TensorDataset::new(data, targets).unwrap();
        assert_eq!(dataset.len(), 2);
        
        let (data, target) = dataset.get_item(0).unwrap();
        assert_eq!(data.shape(), &[2]);
        assert_eq!(target.shape(), &[1]);
    }

    #[test]
    fn test_dataloader() {
        let data = vec![
            Tensor::ones(&[2]),
            Tensor::zeros(&[2]),
            Tensor::full(&[2], 0.5),
            Tensor::full(&[2], -1.0),
        ];
        let targets = vec![
            Tensor::from_vec(vec![1.0], &[1]).unwrap(),
            Tensor::from_vec(vec![0.0], &[1]).unwrap(),
            Tensor::from_vec(vec![1.0], &[1]).unwrap(),
            Tensor::from_vec(vec![0.0], &[1]).unwrap(),
        ];
        
        let dataset = TensorDataset::new(data, targets).unwrap();
        let mut dataloader = DataLoader::new(dataset, 2);
        
        assert_eq!(dataloader.len(), 2);
        
        let batch = dataloader.next_batch().unwrap().unwrap();
        assert_eq!(batch.0.shape()[0], 2); // batch size
    }
}