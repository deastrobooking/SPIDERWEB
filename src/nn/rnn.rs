//! Recurrent Neural Network layer implementations

use crate::tensor::Tensor;
use crate::nn::{Module, InitType, init_parameter};
use anyhow::Result;

/// Basic RNN layer
#[derive(Debug, Clone)]
pub struct RNN {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    training: bool,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut weight_ih = Tensor::zeros(&[hidden_size, input_size]);
        let mut weight_hh = Tensor::zeros(&[hidden_size, hidden_size]);
        let mut bias_ih = Tensor::zeros(&[hidden_size]);
        let mut bias_hh = Tensor::zeros(&[hidden_size]);
        
        init_parameter(&mut weight_ih, InitType::XavierUniform, input_size, hidden_size);
        init_parameter(&mut weight_hh, InitType::XavierUniform, hidden_size, hidden_size);
        init_parameter(&mut bias_ih, InitType::Zero, input_size, hidden_size);
        init_parameter(&mut bias_hh, InitType::Zero, hidden_size, hidden_size);
        
        Self {
            input_size,
            hidden_size,
            weight_ih: weight_ih.requires_grad(true),
            weight_hh: weight_hh.requires_grad(true),
            bias_ih: Some(bias_ih.requires_grad(true)),
            bias_hh: Some(bias_hh.requires_grad(true)),
            training: true,
        }
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];
        
        let mut hidden = Tensor::zeros(&[batch_size, self.hidden_size]);
        let mut outputs = Vec::new();
        
        for t in 0..seq_len {
            // Extract input at time t
            let x_t = input.clone_tensor(); // Simplified - would need proper slicing
            
            // h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
            let linear_ih = x_t.matmul(&self.weight_ih.t())?;
            let linear_hh = hidden.matmul(&self.weight_hh.t())?;
            
            let mut h_new = &linear_ih + &linear_hh;
            if let (Some(ref b_ih), Some(ref b_hh)) = (&self.bias_ih, &self.bias_hh) {
                h_new = &(&h_new + b_ih) + b_hh;
            }
            
            // Apply tanh activation
            hidden = h_new; // Simplified - would apply tanh here
            outputs.push(hidden.clone_tensor());
        }
        
        // Stack outputs along time dimension
        Ok(outputs[0].clone_tensor()) // Simplified return
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref mut bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "RNN" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// LSTM layer
#[derive(Debug, Clone)]
pub struct LSTM {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    training: bool,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // LSTM has 4 gates: input, forget, cell, output
        let gate_size = 4 * hidden_size;
        
        let mut weight_ih = Tensor::zeros(&[gate_size, input_size]);
        let mut weight_hh = Tensor::zeros(&[gate_size, hidden_size]);
        let mut bias_ih = Tensor::zeros(&[gate_size]);
        let mut bias_hh = Tensor::zeros(&[gate_size]);
        
        init_parameter(&mut weight_ih, InitType::XavierUniform, input_size, gate_size);
        init_parameter(&mut weight_hh, InitType::XavierUniform, hidden_size, gate_size);
        init_parameter(&mut bias_ih, InitType::Zero, input_size, gate_size);
        init_parameter(&mut bias_hh, InitType::Zero, hidden_size, gate_size);
        
        Self {
            input_size,
            hidden_size,
            weight_ih: weight_ih.requires_grad(true),
            weight_hh: weight_hh.requires_grad(true),
            bias_ih: Some(bias_ih.requires_grad(true)),
            bias_hh: Some(bias_hh.requires_grad(true)),
            training: true,
        }
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified LSTM implementation
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];
        
        let mut hidden = Tensor::zeros(&[batch_size, self.hidden_size]);
        let mut cell = Tensor::zeros(&[batch_size, self.hidden_size]);
        let mut outputs = Vec::new();
        
        for _t in 0..seq_len {
            // LSTM computation would go here
            // For now, just pass through the hidden state
            outputs.push(hidden.clone_tensor());
        }
        
        Ok(outputs[0].clone_tensor())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref mut bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "LSTM" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// GRU layer
#[derive(Debug, Clone)]
pub struct GRU {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    training: bool,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // GRU has 3 gates: reset, update, new
        let gate_size = 3 * hidden_size;
        
        let mut weight_ih = Tensor::zeros(&[gate_size, input_size]);
        let mut weight_hh = Tensor::zeros(&[gate_size, hidden_size]);
        let mut bias_ih = Tensor::zeros(&[gate_size]);
        let mut bias_hh = Tensor::zeros(&[gate_size]);
        
        init_parameter(&mut weight_ih, InitType::XavierUniform, input_size, gate_size);
        init_parameter(&mut weight_hh, InitType::XavierUniform, hidden_size, gate_size);
        init_parameter(&mut bias_ih, InitType::Zero, input_size, gate_size);
        init_parameter(&mut bias_hh, InitType::Zero, hidden_size, gate_size);
        
        Self {
            input_size,
            hidden_size,
            weight_ih: weight_ih.requires_grad(true),
            weight_hh: weight_hh.requires_grad(true),
            bias_ih: Some(bias_ih.requires_grad(true)),
            bias_hh: Some(bias_hh.requires_grad(true)),
            training: true,
        }
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified GRU implementation
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];
        
        let mut hidden = Tensor::zeros(&[batch_size, self.hidden_size]);
        let mut outputs = Vec::new();
        
        for _t in 0..seq_len {
            // GRU computation would go here
            outputs.push(hidden.clone_tensor());
        }
        
        Ok(outputs[0].clone_tensor())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref mut bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "GRU" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}