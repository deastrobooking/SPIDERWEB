//! Natural Language Processing utilities and models

use crate::tensor::Tensor;
use crate::nn::{Module, Linear, LSTM, Embedding};
use anyhow::Result;
use std::collections::HashMap;

/// Text tokenizer for converting text to token IDs
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    inverse_vocab: HashMap<usize, String>,
    vocab_size: usize,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();
        
        // Add special tokens
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        vocab.insert("<sos>".to_string(), 2);
        vocab.insert("<eos>".to_string(), 3);
        
        inverse_vocab.insert(0, "<pad>".to_string());
        inverse_vocab.insert(1, "<unk>".to_string());
        inverse_vocab.insert(2, "<sos>".to_string());
        inverse_vocab.insert(3, "<eos>".to_string());
        
        Self {
            vocab,
            inverse_vocab,
            vocab_size: 4,
        }
    }
    
    pub fn build_vocab(&mut self, texts: &[String], min_freq: usize) {
        let mut word_counts = HashMap::new();
        
        for text in texts {
            for word in text.split_whitespace() {
                *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }
        
        for (word, count) in word_counts {
            if count >= min_freq && !self.vocab.contains_key(&word) {
                let idx = self.vocab_size;
                self.vocab.insert(word.clone(), idx);
                self.inverse_vocab.insert(idx, word);
                self.vocab_size += 1;
            }
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                self.vocab.get(&word.to_lowercase())
                    .copied()
                    .unwrap_or(1) // <unk> token
            })
            .collect()
    }
    
    pub fn decode(&self, token_ids: &[usize]) -> String {
        token_ids.iter()
            .filter_map(|&id| self.inverse_vocab.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Word embeddings layer
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    training: bool,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = Tensor::randn(&[num_embeddings, embedding_dim]).requires_grad(true);
        
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            training: true,
        }
    }
    
    pub fn from_pretrained(embeddings: Tensor, freeze: bool) -> Self {
        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];
        
        let weight = if freeze {
            embeddings
        } else {
            embeddings.requires_grad(true)
        };
        
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            training: true,
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified embedding lookup
        let input_shape = input.shape();
        let seq_len = input_shape[0];
        let batch_size = if input_shape.len() > 1 { input_shape[1] } else { 1 };
        
        let mut output_data = Vec::new();
        
        for i in 0..seq_len {
            for b in 0..batch_size {
                let token_id = input.data()[i * batch_size + b] as usize;
                if token_id < self.num_embeddings {
                    for d in 0..self.embedding_dim {
                        let embedding_idx = token_id * self.embedding_dim + d;
                        if embedding_idx < self.weight.data().len() {
                            output_data.push(self.weight.data()[embedding_idx]);
                        } else {
                            output_data.push(0.0);
                        }
                    }
                } else {
                    // Out of vocabulary - use zero embedding
                    for _ in 0..self.embedding_dim {
                        output_data.push(0.0);
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, &[seq_len, batch_size, self.embedding_dim])
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
    
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "Embedding" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// LSTM-based language model
pub struct LanguageModel {
    embedding: Embedding,
    lstm: LSTM,
    output_layer: Linear,
    vocab_size: usize,
    hidden_size: usize,
}

impl LanguageModel {
    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_size: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embedding_dim),
            lstm: LSTM::new(embedding_dim, hidden_size),
            output_layer: Linear::new(hidden_size, vocab_size),
            vocab_size,
            hidden_size,
        }
    }
}

impl Module for LanguageModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let embedded = self.embedding.forward(input)?;
        let lstm_output = self.lstm.forward(&embedded)?;
        self.output_layer.forward(&lstm_output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        params.extend(self.lstm.parameters());
        params.extend(self.output_layer.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters_mut());
        params.extend(self.lstm.parameters_mut());
        params.extend(self.output_layer.parameters_mut());
        params
    }
    
    fn train(&mut self) {
        self.embedding.train();
        self.lstm.train();
        self.output_layer.train();
    }
    
    fn eval(&mut self) {
        self.embedding.eval();
        self.lstm.eval();
        self.output_layer.eval();
    }
    
    fn training(&self) -> bool {
        self.embedding.training()
    }
    
    fn name(&self) -> &str { "LanguageModel" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(LanguageModel::new(self.vocab_size, 128, self.hidden_size))
    }
}

/// Text classification model
pub struct TextClassifier {
    embedding: Embedding,
    lstm: LSTM,
    classifier: Linear,
}

impl TextClassifier {
    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_size: usize, num_classes: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embedding_dim),
            lstm: LSTM::new(embedding_dim, hidden_size),
            classifier: Linear::new(hidden_size, num_classes),
        }
    }
}

impl Module for TextClassifier {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let embedded = self.embedding.forward(input)?;
        let lstm_output = self.lstm.forward(&embedded)?;
        // Use last hidden state for classification
        self.classifier.forward(&lstm_output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        params.extend(self.lstm.parameters());
        params.extend(self.classifier.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters_mut());
        params.extend(self.lstm.parameters_mut());
        params.extend(self.classifier.parameters_mut());
        params
    }
    
    fn train(&mut self) {
        self.embedding.train();
        self.lstm.train();
        self.classifier.train();
    }
    
    fn eval(&mut self) {
        self.embedding.eval();
        self.lstm.eval();
        self.classifier.eval();
    }
    
    fn training(&self) -> bool {
        self.embedding.training()
    }
    
    fn name(&self) -> &str { "TextClassifier" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(TextClassifier::new(1000, 128, 256, 2))
    }
}

/// BLEU score for machine translation evaluation
pub fn bleu_score(predictions: &[String], references: &[String], n: usize) -> f32 {
    if predictions.len() != references.len() || predictions.is_empty() {
        return 0.0;
    }
    
    let mut total_score = 0.0;
    
    for (pred, ref_text) in predictions.iter().zip(references.iter()) {
        let pred_ngrams = get_ngrams(pred, n);
        let ref_ngrams = get_ngrams(ref_text, n);
        
        let mut matches = 0;
        for ngram in &pred_ngrams {
            if ref_ngrams.contains(ngram) {
                matches += 1;
            }
        }
        
        let score = if pred_ngrams.is_empty() {
            0.0
        } else {
            matches as f32 / pred_ngrams.len() as f32
        };
        
        total_score += score;
    }
    
    total_score / predictions.len() as f32
}

fn get_ngrams(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return vec![text.to_string()];
    }
    
    let mut ngrams = Vec::new();
    for i in 0..=words.len() - n {
        let ngram = words[i..i + n].join(" ");
        ngrams.push(ngram);
    }
    ngrams
}