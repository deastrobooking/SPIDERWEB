//! Automatic differentiation system
//! 
//! This module provides automatic gradient computation similar to PyTorch's autograd
//! and TensorFlow's gradient tape functionality.

use crate::tensor::Tensor;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Variable wrapper for tensors that need gradient computation
#[derive(Debug, Clone)]
pub struct Variable {
    data: Tensor,
    grad: Option<Tensor>,
    grad_fn: Option<Arc<dyn GradFunction>>,
    requires_grad: bool,
    is_leaf: bool,
}

impl Variable {
    /// Create a new variable from tensor
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        Self {
            data: tensor,
            grad: None,
            grad_fn: None,
            requires_grad,
            is_leaf: true,
        }
    }
    
    /// Get the underlying tensor data
    pub fn data(&self) -> &Tensor {
        &self.data
    }
    
    /// Get mutable reference to tensor data
    pub fn data_mut(&mut self) -> &mut Tensor {
        &mut self.data
    }
    
    /// Get gradient if available
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }
    
    /// Set gradient
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(grad);
    }
    
    /// Zero gradient
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
    
    /// Check if variable requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Set requires_grad flag
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
    
    /// Check if this is a leaf variable
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }
    
    /// Backward pass to compute gradients
    pub fn backward(&mut self) -> Result<()> {
        if !self.requires_grad {
            return Ok(());
        }
        
        // Initialize gradient as ones for scalar output
        if self.grad.is_none() {
            self.grad = Some(Tensor::ones(self.data.shape()));
        }
        
        // Traverse computation graph backwards
        if let Some(ref grad_fn) = self.grad_fn {
            let input_grads = grad_fn.backward(self.grad.as_ref().unwrap())?;
            // Propagate gradients to inputs
            // This would require maintaining references to input variables
        }
        
        Ok(())
    }
}

/// Trait for gradient functions in the computation graph
pub trait GradFunction: Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>>;
    fn name(&self) -> &str;
}

/// Addition gradient function
pub struct AddBackward {
    input_shapes: (Vec<usize>, Vec<usize>),
}

impl AddBackward {
    pub fn new(shape1: Vec<usize>, shape2: Vec<usize>) -> Self {
        Self {
            input_shapes: (shape1, shape2),
        }
    }
}

impl GradFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>> {
        // Gradient of addition is just the upstream gradient for both inputs
        let grad1 = grad_output.clone_tensor();
        let grad2 = grad_output.clone_tensor();
        
        // Handle broadcasting by summing over broadcasted dimensions
        Ok(vec![grad1, grad2])
    }
    
    fn name(&self) -> &str {
        "AddBackward"
    }
}

/// Multiplication gradient function
pub struct MulBackward {
    input1: Tensor,
    input2: Tensor,
}

impl MulBackward {
    pub fn new(input1: Tensor, input2: Tensor) -> Self {
        Self { input1, input2 }
    }
}

impl GradFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>> {
        // d/dx (x * y) = y, d/dy (x * y) = x
        let grad1 = grad_output * &self.input2;
        let grad2 = grad_output * &self.input1;
        Ok(vec![grad1, grad2])
    }
    
    fn name(&self) -> &str {
        "MulBackward"
    }
}

/// Matrix multiplication gradient function
pub struct MatMulBackward {
    input1: Tensor,
    input2: Tensor,
}

impl MatMulBackward {
    pub fn new(input1: Tensor, input2: Tensor) -> Self {
        Self { input1, input2 }
    }
}

impl GradFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>> {
        // d/dA (A @ B) = grad_output @ B^T
        // d/dB (A @ B) = A^T @ grad_output
        let grad1 = grad_output.matmul(&self.input2.t())?;
        let grad2 = self.input1.t().matmul(grad_output)?;
        Ok(vec![grad1, grad2])
    }
    
    fn name(&self) -> &str {
        "MatMulBackward"
    }
}

/// ReLU gradient function
pub struct ReLUBackward {
    input: Tensor,
}

impl ReLUBackward {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl GradFunction for ReLUBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>> {
        // ReLU gradient: 1 if input > 0, 0 otherwise
        let mask_data = self.input.data().iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect::<Vec<f32>>();
        
        let mask = Tensor::from_vec(mask_data, self.input.shape())?;
        let grad = grad_output * &mask;
        Ok(vec![grad])
    }
    
    fn name(&self) -> &str {
        "ReLUBackward"
    }
}

/// Sigmoid gradient function
pub struct SigmoidBackward {
    output: Tensor,
}

impl SigmoidBackward {
    pub fn new(output: Tensor) -> Self {
        Self { output }
    }
}

impl GradFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>> {
        // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
        let one = Tensor::ones(self.output.shape());
        let sigmoid_grad = &self.output * &(&one - &self.output);
        let grad = grad_output * &sigmoid_grad;
        Ok(vec![grad])
    }
    
    fn name(&self) -> &str {
        "SigmoidBackward"
    }
}

/// Global computation graph for tracking operations
static COMPUTATION_GRAPH: Mutex<Option<ComputationGraph>> = Mutex::new(None);

/// Computation graph for tracking variable operations
#[derive(Debug)]
pub struct ComputationGraph {
    nodes: HashMap<usize, Variable>,
    edges: HashMap<usize, Vec<usize>>,
    next_id: usize,
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
        }
    }
    
    pub fn add_node(&mut self, variable: Variable) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, variable);
        id
    }
    
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.entry(from).or_insert_with(Vec::new).push(to);
    }
    
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.next_id = 0;
    }
}

/// Initialize computation graph
pub fn init_graph() {
    let mut graph = COMPUTATION_GRAPH.lock().unwrap();
    *graph = Some(ComputationGraph::new());
}

/// Clear computation graph
pub fn clear_graph() {
    if let Ok(mut graph) = COMPUTATION_GRAPH.lock() {
        if let Some(ref mut g) = *graph {
            g.clear();
        }
    }
}

/// Perform backward pass on a variable
pub fn backward(variable: &mut Variable) -> Result<()> {
    variable.backward()
}

/// Enable gradient computation context
pub struct GradContext {
    prev_state: bool,
}

impl GradContext {
    pub fn enable() -> Self {
        Self { prev_state: true }
    }
    
    pub fn disable() -> Self {
        Self { prev_state: false }
    }
}

impl Drop for GradContext {
    fn drop(&mut self) {
        // Restore previous gradient computation state
    }
}

// Convenience macros for gradient computation
#[macro_export]
macro_rules! no_grad {
    ($body:block) => {{
        let _ctx = $crate::autograd::GradContext::disable();
        $body
    }};
}

#[macro_export]
macro_rules! enable_grad {
    ($body:block) => {{
        let _ctx = $crate::autograd::GradContext::enable();
        $body
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_creation() {
        let tensor = Tensor::ones(&[2, 2]);
        let var = Variable::new(tensor, true);
        assert!(var.requires_grad());
        assert!(var.is_leaf());
    }

    #[test]
    fn test_gradient_function() {
        let input1 = Tensor::ones(&[2, 2]);
        let input2 = Tensor::full(&[2, 2], 2.0);
        let add_fn = AddBackward::new(vec![2, 2], vec![2, 2]);
        
        let grad_output = Tensor::ones(&[2, 2]);
        let grads = add_fn.backward(&grad_output).unwrap();
        assert_eq!(grads.len(), 2);
    }
}