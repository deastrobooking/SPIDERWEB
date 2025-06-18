//! Core tensor operations and data structures
//! 
//! This module provides the fundamental Tensor type and operations that form
//! the backbone of the ML framework, similar to torch.Tensor in PyTorch
//! and tf.Tensor in TensorFlow.

use ndarray::{Array, ArrayD, Dimension, IxDyn, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::{Distribution, Uniform, StandardNormal};
use serde::{Serialize, Deserialize};
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt;
use anyhow::{Result, anyhow};

/// Device types for tensor computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    CPU,
    CUDA(usize),  // GPU device index
    Metal(usize), // Metal GPU device index
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}

/// Data types supported by tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U8,
    Bool,
}

/// Core tensor structure that holds n-dimensional arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    data: ArrayD<f32>,
    device: Device,
    dtype: DType,
    requires_grad: bool,
    grad: Option<Box<Tensor>>,
}

impl Tensor {
    /// Create a new tensor from a multi-dimensional array
    pub fn new(data: ArrayD<f32>) -> Self {
        Self {
            data,
            device: Device::CPU,
            dtype: DType::F32,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor from a vector with specified shape
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| anyhow!("Shape error: {}", e))?;
        Ok(Self::new(array))
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self::new(data)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        let data = ArrayD::ones(IxDyn(shape));
        Self::new(data)
    }

    /// Create a tensor filled with a specific value
    pub fn full(shape: &[usize], value: f32) -> Self {
        let data = ArrayD::from_elem(IxDyn(shape), value);
        Self::new(data)
    }

    /// Create a tensor with random values from uniform distribution [0, 1)
    pub fn rand(shape: &[usize]) -> Self {
        let data = ArrayD::random(IxDyn(shape), Uniform::new(0.0, 1.0));
        Self::new(data)
    }

    /// Create a tensor with random values from standard normal distribution
    pub fn randn(shape: &[usize]) -> Self {
        let data = ArrayD::random(IxDyn(shape), StandardNormal);
        Self::new(data)
    }

    /// Create an identity matrix
    pub fn eye(n: usize) -> Self {
        let mut data = ArrayD::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            data[[i, i]] = 1.0;
        }
        Self::new(data)
    }

    /// Create a tensor from a range of values
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let n = ((end - start) / step).ceil() as usize;
        let data: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        let array = Array::from_vec(data).into_dyn();
        Self::new(array)
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Move tensor to device
    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Enable gradient computation for this tensor
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Get gradient if available
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_deref()
    }

    /// Set gradient
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(Box::new(grad));
    }

    /// Zero out gradients
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Reshape tensor
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let reshaped = self.data.clone().into_shape(IxDyn(shape))
            .map_err(|e| anyhow!("Reshape error: {}", e))?;
        let mut result = Self::new(reshaped);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// Transpose tensor (swap last two dimensions)
    pub fn t(&self) -> Self {
        let ndim = self.ndim();
        if ndim < 2 {
            return self.clone();
        }
        
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(ndim - 2, ndim - 1);
        
        let transposed = self.data.permuted_axes(axes);
        let mut result = Self::new(transposed);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Permute tensor dimensions
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.ndim() {
            return Err(anyhow!("Number of dimensions in permute must match tensor dimensions"));
        }
        
        let permuted = self.data.permuted_axes(dims.to_vec());
        let mut result = Self::new(permuted);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// Squeeze tensor (remove dimensions of size 1)
    pub fn squeeze(&self) -> Self {
        let mut new_shape = Vec::new();
        for &dim in self.shape() {
            if dim != 1 {
                new_shape.push(dim);
            }
        }
        
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        
        self.reshape(&new_shape).unwrap()
    }

    /// Unsqueeze tensor (add dimension of size 1 at specified position)
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_shape = self.shape().to_vec();
        if dim > new_shape.len() {
            return Err(anyhow!("Dimension out of range"));
        }
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    /// Get tensor data as slice
    pub fn data(&self) -> &[f32] {
        self.data.as_slice().unwrap()
    }

    /// Convert to vector
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.data.mean().unwrap()
    }

    /// Sum along specified axis
    pub fn sum_axis(&self, axis: usize) -> Result<Self> {
        if axis >= self.ndim() {
            return Err(anyhow!("Axis {} out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        let summed = self.data.sum_axis(Axis(axis));
        let mut result = Self::new(summed);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// Mean along specified axis
    pub fn mean_axis(&self, axis: usize) -> Result<Self> {
        if axis >= self.ndim() {
            return Err(anyhow!("Axis {} out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        let mean = self.data.mean_axis(Axis(axis)).unwrap();
        let mut result = Self::new(mean);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// Maximum value
    pub fn max(&self) -> f32 {
        self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Minimum value
    pub fn min(&self) -> f32 {
        self.data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Self {
        let abs_data = self.data.mapv(|x| x.abs());
        let mut result = Self::new(abs_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Self {
        let sqrt_data = self.data.mapv(|x| x.sqrt());
        let mut result = Self::new(sqrt_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Self {
        let exp_data = self.data.mapv(|x| x.exp());
        let mut result = Self::new(exp_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> Self {
        let log_data = self.data.mapv(|x| x.ln());
        let mut result = Self::new(log_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Element-wise sine
    pub fn sin(&self) -> Self {
        let sin_data = self.data.mapv(|x| x.sin());
        let mut result = Self::new(sin_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Element-wise cosine
    pub fn cos(&self) -> Self {
        let cos_data = self.data.mapv(|x| x.cos());
        let mut result = Self::new(cos_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Element-wise power
    pub fn pow(&self, exponent: f32) -> Self {
        let pow_data = self.data.mapv(|x| x.powf(exponent));
        let mut result = Self::new(pow_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        use ndarray_linalg::Dot;
        
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(anyhow!("Both tensors must be at least 2D for matrix multiplication"));
        }
        
        let result_data = self.data.dot(&other.data);
        let mut result = Self::new(result_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }

    /// Clone tensor
    pub fn clone_tensor(&self) -> Self {
        let mut result = Self::new(self.data.clone());
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }
}

/// Trait for tensor operations
pub trait TensorOps {
    fn add(&self, other: &Self) -> Result<Self>;
    fn sub(&self, other: &Self) -> Result<Self>;
    fn mul(&self, other: &Self) -> Result<Self>;
    fn div(&self, other: &Self) -> Result<Self>;
}

impl TensorOps for Tensor {
    fn add(&self, other: &Self) -> Result<Self> {
        let result_data = (&self.data + &other.data);
        let mut result = Self::new(result_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        let result_data = (&self.data - &other.data);
        let mut result = Self::new(result_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        let result_data = (&self.data * &other.data);
        let mut result = Self::new(result_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }

    fn div(&self, other: &Self) -> Result<Self> {
        let result_data = (&self.data / &other.data);
        let mut result = Self::new(result_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }
}

// Implement standard operators
impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        self.add(other).unwrap()
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other).unwrap()
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other).unwrap()
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: Self) -> Self::Output {
        self.div(other).unwrap()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let neg_data = self.data.mapv(|x| -x);
        let mut result = Tensor::new(neg_data);
        result.device = self.device;
        result.dtype = self.dtype;
        result.requires_grad = self.requires_grad;
        result
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, device={:?}, dtype={:?})", 
               self.shape(), self.device, self.dtype)?;
        if self.numel() <= 10 {
            write!(f, "\n{}", self.data)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_operations() {
        let a = Tensor::ones(&[2, 2]);
        let b = Tensor::full(&[2, 2], 2.0);
        let c = &a + &b;
        
        assert_eq!(c.sum(), 12.0); // 4 elements * 3.0 each
    }
}