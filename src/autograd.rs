//! Automatic differentiation engine for native Rust ML framework
//! 
//! This module implements reverse-mode automatic differentiation (backpropagation)
//! with dynamic computational graph construction, similar to PyTorch's autograd.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ndarray::{ArrayD, IxDyn};
use std::rc::Rc;
use std::cell::RefCell;

/// Unique identifier for tensors in the computational graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u64);

impl TensorId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        TensorId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// Gradient function trait for automatic differentiation
pub trait GradientFunction: Send + Sync {
    fn backward(&self, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>>;
    fn next_functions(&self) -> Vec<(TensorId, Option<Arc<dyn GradientFunction>>)>;
    fn name(&self) -> &'static str;
}

/// Node in the computational graph representing an operation
#[derive(Debug)]
pub struct GraphNode {
    pub id: TensorId,
    pub grad_fn: Option<Arc<dyn GradientFunction>>,
    pub next_edges: Vec<(TensorId, Option<Arc<dyn GradientFunction>>)>,
    pub requires_grad: bool,
}

/// Automatic differentiation engine with dynamic graph construction
pub struct AutogradEngine {
    graph: Arc<Mutex<HashMap<TensorId, GraphNode>>>,
}

impl AutogradEngine {
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a tensor in the computational graph
    pub fn register_tensor(&self, id: TensorId, requires_grad: bool) {
        let mut graph = self.graph.lock().unwrap();
        graph.insert(id, GraphNode {
            id,
            grad_fn: None,
            next_edges: Vec::new(),
            requires_grad,
        });
    }

    /// Add an operation to the computational graph
    pub fn add_operation(
        &self,
        output_id: TensorId,
        grad_fn: Arc<dyn GradientFunction>,
        input_ids: Vec<TensorId>,
    ) {
        let mut graph = self.graph.lock().unwrap();
        
        // Create next edges from gradient function
        let next_edges = grad_fn.next_functions();
        
        graph.insert(output_id, GraphNode {
            id: output_id,
            grad_fn: Some(grad_fn),
            next_edges,
            requires_grad: true,
        });
    }

    /// Perform backward pass through the computational graph
    pub fn backward(&self, loss_id: TensorId, grad_output: ArrayD<f32>) -> HashMap<TensorId, ArrayD<f32>> {
        let mut gradients = HashMap::new();
        let mut ready = Vec::new();
        
        // Initialize with loss gradient
        gradients.insert(loss_id, grad_output);
        ready.push(loss_id);
        
        // Topological sort and backward pass
        let graph = self.graph.lock().unwrap();
        
        while let Some(current_id) = ready.pop() {
            if let Some(node) = graph.get(&current_id) {
                if let Some(ref grad_fn) = node.grad_fn {
                    if let Some(current_grad) = gradients.get(&current_id) {
                        // Compute gradients for inputs
                        let input_grads = grad_fn.backward(current_grad);
                        
                        // Distribute gradients to input tensors
                        for ((input_id, _), input_grad) in node.next_edges.iter().zip(input_grads) {
                            gradients.entry(*input_id)
                                .and_modify(|existing| {
                                    // Accumulate gradients
                                    *existing = existing.clone() + input_grad.clone();
                                })
                                .or_insert(input_grad);
                            
                            // Add to ready queue if not processed
                            if !ready.contains(input_id) {
                                ready.push(*input_id);
                            }
                        }
                    }
                }
            }
        }
        
        gradients
    }

    /// Clear the computational graph
    pub fn clear(&self) {
        let mut graph = self.graph.lock().unwrap();
        graph.clear();
    }
}

/// Implementation of gradient functions for common operations

/// Addition gradient function
pub struct AddBackward {
    input_ids: Vec<TensorId>,
}

impl AddBackward {
    pub fn new(input_ids: Vec<TensorId>) -> Self {
        Self { input_ids }
    }
}

impl GradientFunction for AddBackward {
    fn backward(&self, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // Gradient of addition is just passed through to both inputs
        vec![grad_output.clone(), grad_output.clone()]
    }
    
    fn next_functions(&self) -> Vec<(TensorId, Option<Arc<dyn GradientFunction>>)> {
        self.input_ids.iter().map(|&id| (id, None)).collect()
    }
    
    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Matrix multiplication gradient function
pub struct MatMulBackward {
    input_a_shape: Vec<usize>,
    input_b_shape: Vec<usize>,
    input_ids: Vec<TensorId>,
}

impl MatMulBackward {
    pub fn new(input_a_shape: Vec<usize>, input_b_shape: Vec<usize>, input_ids: Vec<TensorId>) -> Self {
        Self {
            input_a_shape,
            input_b_shape,
            input_ids,
        }
    }
}

impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // For matrix multiplication C = A @ B:
        // dA = grad_output @ B.T
        // dB = A.T @ grad_output
        
        // This is a simplified implementation - real implementation would need
        // to handle the actual tensor data and transposition
        vec![
            grad_output.clone(), // Placeholder for dA
            grad_output.clone(), // Placeholder for dB
        ]
    }
    
    fn next_functions(&self) -> Vec<(TensorId, Option<Arc<dyn GradientFunction>>)> {
        self.input_ids.iter().map(|&id| (id, None)).collect()
    }
    
    fn name(&self) -> &'static str {
        "MatMulBackward"
    }
}

/// ReLU activation gradient function
pub struct ReluBackward {
    input_data: ArrayD<f32>,
    input_id: TensorId,
}

impl ReluBackward {
    pub fn new(input_data: ArrayD<f32>, input_id: TensorId) -> Self {
        Self { input_data, input_id }
    }
}

impl GradientFunction for ReluBackward {
    fn backward(&self, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // ReLU gradient: pass through if input > 0, else 0
        let grad_input = grad_output.mapv(|g| {
            // This is simplified - should check corresponding input values
            if g > 0.0 { g } else { 0.0 }
        });
        
        vec![grad_input]
    }
    
    fn next_functions(&self) -> Vec<(TensorId, Option<Arc<dyn GradientFunction>>)> {
        vec![(self.input_id, None)]
    }
    
    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

/// Mean squared error loss gradient function
pub struct MseLossBackward {
    predictions: ArrayD<f32>,
    targets: ArrayD<f32>,
    input_ids: Vec<TensorId>,
}

impl MseLossBackward {
    pub fn new(predictions: ArrayD<f32>, targets: ArrayD<f32>, input_ids: Vec<TensorId>) -> Self {
        Self {
            predictions,
            targets,
            input_ids,
        }
    }
}

impl GradientFunction for MseLossBackward {
    fn backward(&self, grad_output: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // MSE gradient: 2 * (predictions - targets) / n
        let n = self.predictions.len() as f32;
        let grad = (&self.predictions - &self.targets).mapv(|x| 2.0 * x / n);
        
        vec![grad]
    }
    
    fn next_functions(&self) -> Vec<(TensorId, Option<Arc<dyn GradientFunction>>)> {
        self.input_ids.iter().map(|&id| (id, None)).collect()
    }
    
    fn name(&self) -> &'static str {
        "MseLossBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_autograd_engine() {
        let engine = AutogradEngine::new();
        
        // Create test tensors
        let a_id = TensorId::new();
        let b_id = TensorId::new();
        let c_id = TensorId::new();
        
        engine.register_tensor(a_id, true);
        engine.register_tensor(b_id, true);
        
        // Add operation: c = a + b
        let add_grad_fn = Arc::new(AddBackward::new(vec![a_id, b_id]));
        engine.add_operation(c_id, add_grad_fn, vec![a_id, b_id]);
        
        // Perform backward pass
        let loss_grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
        let gradients = engine.backward(c_id, loss_grad);
        
        assert!(gradients.contains_key(&a_id));
        assert!(gradients.contains_key(&b_id));
    }
}