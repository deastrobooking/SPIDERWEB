// NLP Demo - Text Classification with Transformers
// Demonstrates natural language processing using the Rust ML Framework

use rust_ml_framework::*;
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("Rust ML Framework - NLP Demo");
    println!("============================");
    
    init()?;
    set_seed(42);
    
    // Create tokenizer and vocabulary
    let mut tokenizer = create_tokenizer()?;
    println!("✓ Tokenizer created with {} vocabulary entries", tokenizer.vocab_size());
    
    // Create transformer model for text classification
    let mut model = create_transformer_classifier(tokenizer.vocab_size(), 512, 8, 6)?;
    println!("✓ Transformer model created with {} parameters", utils::count_parameters(&model));
    
    // Generate text classification dataset
    let (texts, labels) = generate_text_data(1000);
    let (train_tokens, train_labels) = tokenize_dataset(&mut tokenizer, &texts, &labels)?;
    println!("✓ Tokenized {} text samples", train_tokens.len());
    
    // Create data pipeline
    let dataset = data::TensorDataset::new(train_tokens, train_labels)?;
    let mut dataloader = data::DataLoader::new(dataset, 8).with_shuffle(true);
    println!("✓ Text DataLoader created with batch size 8");
    
    // Setup training
    let loss_fn = loss::CrossEntropyLoss::new();
    let mut params: Vec<*mut Tensor> = model.parameters_mut()
        .into_iter()
        .map(|p| p as *mut Tensor)
        .collect();
    let mut optimizer = optim::AdamW::new(params, 0.0001, 0.01);
    println!("✓ Training setup complete (AdamW optimizer, CrossEntropy loss)");
    
    // Training loop
    println!("\nTraining transformer on text classification...");
    train_transformer(&mut model, &mut dataloader, &loss_fn, &mut optimizer, 5)?;
    
    // Test text classification
    println!("\nTesting text classification:");
    test_text_classification(&model, &mut tokenizer)?;
    
    // Demonstrate NLP utilities
    println!("\nDemonstrating NLP utilities:");
    demonstrate_nlp_features(&mut tokenizer)?;
    
    println!("\n✓ NLP demo completed!");
    Ok(())
}

fn create_tokenizer() -> Result<nlp::Tokenizer> {
    let mut tokenizer = nlp::Tokenizer::new();
    
    // Add special tokens
    tokenizer.add_special_tokens(&["[PAD]", "[UNK]", "[CLS]", "[SEP]"]);
    
    // Common vocabulary for sentiment analysis
    let vocab = vec![
        // Positive words
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "best",
        "perfect", "outstanding", "brilliant", "awesome", "superb", "magnificent",
        
        // Negative words  
        "bad", "terrible", "awful", "horrible", "worst", "hate", "disgusting", "pathetic",
        "useless", "disappointing", "boring", "annoying", "frustrating", "poor",
        
        // Neutral/common words
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "this", "that", "these", "those", "is", "are", "was", "were", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "can", "may", "might",
        "movie", "film", "book", "story", "character", "plot", "acting", "music", "sound",
        "very", "really", "quite", "pretty", "somewhat", "rather", "extremely", "highly",
    ];
    
    for word in vocab {
        tokenizer.add_token(word);
    }
    
    Ok(tokenizer)
}

fn create_transformer_classifier(
    vocab_size: usize, 
    d_model: usize, 
    nhead: usize, 
    num_layers: usize
) -> Result<nn::Sequential> {
    Ok(nn::Sequential::new()
        // Embedding layer
        .add(nn::Embedding::new(vocab_size, d_model))
        .add(nn::PositionalEncoding::new(d_model, 512)) // Max sequence length 512
        
        // Transformer encoder layers
        .add(nn::TransformerEncoder::new(d_model, nhead, num_layers))
        
        // Classification head
        .add(nn::GlobalAvgPool1d::new())  // Pool over sequence dimension
        .add(nn::Dropout::new(0.1))
        .add(nn::Linear::new(d_model, 256))
        .add(nn::ReLU::new())
        .add(nn::Dropout::new(0.1))
        .add(nn::Linear::new(256, 3))     // 3 classes: positive, negative, neutral
    )
}

fn generate_text_data(num_samples: usize) -> (Vec<String>, Vec<i32>) {
    let positive_templates = vec![
        "This movie is really good and entertaining",
        "I love this film, it's excellent",
        "Amazing story with great acting",
        "Fantastic movie, highly recommended",
        "Outstanding performance, wonderful plot",
        "Brilliant acting and superb music",
        "Perfect film with excellent characters",
        "The best movie I have seen",
    ];
    
    let negative_templates = vec![
        "This movie is terrible and boring",
        "I hate this film, it's awful",
        "Horrible story with bad acting", 
        "Worst movie ever, very disappointing",
        "Poor performance, useless plot",
        "Pathetic acting and annoying music",
        "Disgusting film with terrible characters",
        "The most frustrating movie to watch",
    ];
    
    let neutral_templates = vec![
        "This movie is okay, somewhat interesting",
        "The film is average with decent acting",
        "Pretty good story but could be better",
        "Quite watchable with reasonable plot",
        "Rather standard movie with okay music",
        "Somewhat entertaining but not great",
        "The film is fine, nothing special",
        "Moderately good with average characters",
    ];
    
    let mut texts = Vec::new();
    let mut labels = Vec::new();
    
    for i in 0..num_samples {
        let label = i % 3;
        let text = match label {
            0 => positive_templates[i % positive_templates.len()].to_string(),
            1 => negative_templates[i % negative_templates.len()].to_string(), 
            2 => neutral_templates[i % neutral_templates.len()].to_string(),
            _ => unreachable!(),
        };
        
        texts.push(text);
        labels.push(label as i32);
    }
    
    (texts, labels)
}

fn tokenize_dataset(
    tokenizer: &mut nlp::Tokenizer,
    texts: &[String], 
    labels: &[i32]
) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let max_length = 32;
    let mut tokenized_texts = Vec::new();
    let mut tensor_labels = Vec::new();
    
    for (text, &label) in texts.iter().zip(labels.iter()) {
        // Tokenize text
        let tokens = tokenizer.encode(text, max_length)?;
        tokenized_texts.push(Tensor::from_vec(
            tokens.into_iter().map(|t| t as f32).collect(),
            &[max_length]
        )?);
        
        tensor_labels.push(Tensor::from_vec(vec![label as f32], &[1])?);
    }
    
    Ok((tokenized_texts, tensor_labels))
}

fn train_transformer(
    model: &mut nn::Sequential,
    dataloader: &mut data::DataLoader<data::TensorDataset>,
    loss_fn: &loss::CrossEntropyLoss,
    optimizer: &mut optim::AdamW,
    epochs: usize,
) -> Result<()> {
    for epoch in 0..epochs {
        model.train();
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        dataloader.reset();
        while let Some(batch) = dataloader.next_batch()? {
            let (input_ids, labels) = batch;
            
            // Forward pass through transformer
            let logits = model.forward(&input_ids)?;
            let loss = loss_fn.forward(&logits, &labels)?;
            
            // Backward pass
            optimizer.zero_grad();
            optimizer.step()?;
            
            total_loss += loss.data()[0];
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("  Epoch {}: Loss = {:.4}", epoch + 1, avg_loss);
    }
    
    Ok(())
}

fn test_text_classification(model: &nn::Sequential, tokenizer: &mut nlp::Tokenizer) -> Result<()> {
    model.eval();
    
    let test_texts = vec![
        ("This movie is absolutely fantastic!", "Positive"),
        ("I hate this boring film", "Negative"), 
        ("The movie is okay, nothing special", "Neutral"),
        ("Amazing story with excellent acting", "Positive"),
        ("Terrible plot and bad music", "Negative"),
    ];
    
    for (text, expected) in test_texts {
        let tokens = tokenizer.encode(text, 32)?;
        let input = Tensor::from_vec(
            tokens.into_iter().map(|t| t as f32).collect(),
            &[1, 32] // Add batch dimension
        )?;
        
        let logits = model.forward(&input)?;
        
        // Get predicted class
        let mut max_idx = 0;
        let mut max_val = logits.data()[0];
        for i in 1..3 {
            if logits.data()[i] > max_val {
                max_val = logits.data()[i];
                max_idx = i;
            }
        }
        
        let predicted = match max_idx {
            0 => "Positive",
            1 => "Negative", 
            2 => "Neutral",
            _ => "Unknown",
        };
        
        println!("  \"{}\" → {} (expected: {}, confidence: {:.3})", 
                 text, predicted, expected, max_val);
    }
    
    Ok(())
}

fn demonstrate_nlp_features(tokenizer: &mut nlp::Tokenizer) -> Result<()> {
    println!("  Text preprocessing utilities:");
    
    // Text cleaning
    let dirty_text = "  This is A MESSY text!!! With   spaces...  ";
    let clean_text = nlp::preprocess::clean_text(dirty_text);
    println!("    - Text cleaning: \"{}\" → \"{}\"", dirty_text, clean_text);
    
    // Tokenization
    let text = "The quick brown fox jumps over the lazy dog";
    let tokens = tokenizer.tokenize(text)?;
    println!("    - Tokenization: \"{}\" → {:?}", text, tokens);
    
    // N-grams
    let bigrams = nlp::ngrams::extract_ngrams(&tokens, 2);
    println!("    - Bigrams: {:?}", bigrams);
    
    // Word embeddings
    let embedding_dim = 100;
    let word_embeddings = nlp::embeddings::WordEmbeddings::new(tokenizer.vocab_size(), embedding_dim);
    println!("    - Word embeddings: {} words × {} dimensions", 
             tokenizer.vocab_size(), embedding_dim);
    
    // Named entity recognition (placeholder)
    let entities = nlp::ner::extract_entities(text)?;
    println!("    - Named entities: {:?}", entities);
    
    // Sentiment analysis components
    println!("  Advanced NLP models:");
    
    // BERT-style model
    let bert_config = nlp::BertConfig::new(tokenizer.vocab_size(), 768, 12, 12);
    let bert_model = nlp::BertModel::new(bert_config)?;
    println!("    - BERT model: {} parameters", utils::count_parameters(&bert_model));
    
    // GPT-style model  
    let gpt_config = nlp::GPTConfig::new(tokenizer.vocab_size(), 768, 12, 12);
    let gpt_model = nlp::GPTModel::new(gpt_config)?;
    println!("    - GPT model: {} parameters", utils::count_parameters(&gpt_model));
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = create_tokenizer().unwrap();
        assert!(tokenizer.vocab_size() > 50);
    }

    #[test] 
    fn test_transformer_creation() {
        let model = create_transformer_classifier(1000, 256, 4, 2).unwrap();
        // Should have embedding, positional encoding, transformer, and classifier
        assert!(model.layers().len() >= 6);
    }

    #[test]
    fn test_text_data_generation() {
        let (texts, labels) = generate_text_data(30);
        assert_eq!(texts.len(), 30);
        assert_eq!(labels.len(), 30);
        
        // Should have 3 different classes
        let unique_labels: std::collections::HashSet<_> = labels.into_iter().collect();
        assert_eq!(unique_labels.len(), 3);
    }

    #[test]
    fn test_tokenization() {
        let mut tokenizer = create_tokenizer().unwrap();
        let text = "This is a test";
        let tokens = tokenizer.encode(text, 10).unwrap();
        assert_eq!(tokens.len(), 10); // Should pad/truncate to max_length
    }
}