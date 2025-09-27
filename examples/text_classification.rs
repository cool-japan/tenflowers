use tenflowers_core::{Tensor, Device};
use tenflowers_neural::{
    Sequential, Dense, LSTM, Adam, Model, Optimizer, Embedding,
    categorical_cross_entropy, ReLU, Dropout
};
use tenflowers_dataset::{TensorDataset, Dataset};
use scirs2_core::random::Random;
use std::collections::HashMap;

/// Simple vocabulary for text processing
struct SimpleVocab {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    vocab_size: usize,
}

impl SimpleVocab {
    fn new() -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            vocab_size: 0,
        };
        
        // Add special tokens
        vocab.add_word("<UNK>".to_string());
        vocab.add_word("<PAD>".to_string());
        
        // Add common words for sentiment analysis
        let words = vec![
            "good", "bad", "great", "terrible", "amazing", "awful",
            "love", "hate", "like", "dislike", "fantastic", "horrible",
            "excellent", "poor", "wonderful", "disappointing",
            "the", "a", "an", "is", "was", "are", "were", "this", "that",
            "movie", "film", "book", "product", "service", "experience",
            "very", "really", "quite", "extremely", "not", "never", "always"
        ];
        
        for word in words {
            vocab.add_word(word.to_string());
        }
        
        vocab
    }
    
    fn add_word(&mut self, word: String) {
        if !self.word_to_id.contains_key(&word) {
            let id = self.vocab_size;
            self.word_to_id.insert(word.clone(), id);
            self.id_to_word.insert(id, word);
            self.vocab_size += 1;
        }
    }
    
    fn word_to_id(&self, word: &str) -> usize {
        *self.word_to_id.get(word).unwrap_or(&0) // UNK token
    }
    
    fn encode_text(&self, text: &str, max_length: usize) -> Vec<usize> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ids = Vec::new();
        
        for word in words.iter().take(max_length) {
            ids.push(self.word_to_id(word));
        }
        
        // Pad with PAD token (id=1)
        while ids.len() < max_length {
            ids.push(1);
        }
        
        ids
    }
}

/// Generate synthetic text classification data
fn generate_synthetic_text_data(samples: usize, vocab: &SimpleVocab) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = rand::thread_rng();
    let max_length = 20;
    
    // Positive and negative word patterns
    let positive_patterns = vec![
        "this movie is great",
        "amazing product very good",
        "love this book excellent",
        "fantastic service wonderful experience",
        "really like this quite good",
    ];
    
    let negative_patterns = vec![
        "this movie is terrible",
        "awful product very bad",
        "hate this book horrible",
        "disappointing service poor experience",
        "really dislike this quite bad",
    ];
    
    let mut sequences = Vec::new();
    let mut labels = Vec::new();
    
    for i in 0..samples {
        let is_positive = i % 2 == 0;
        
        let base_text = if is_positive {
            &positive_patterns[rng.gen_range(0..positive_patterns.len())]
        } else {
            &negative_patterns[rng.gen_range(0..negative_patterns.len())]
        };
        
        // Add some noise words
        let noise_words = vec!["the", "a", "is", "was", "this", "that"];
        let mut text = base_text.to_string();
        for _ in 0..rng.gen_range(0..3) {
            text.push_str(" ");
            text.push_str(noise_words[rng.gen_range(0..noise_words.len())]);
        }
        
        let sequence = vocab.encode_text(&text, max_length);
        sequences.extend(sequence);
        
        // Binary classification: 0 = negative, 1 = positive
        if is_positive {
            labels.extend(vec![0.0, 1.0]);
        } else {
            labels.extend(vec![1.0, 0.0]);
        }
    }
    
    let x_tensor = Tensor::from_vec(
        sequences.iter().map(|&x| x as f32).collect(),
        &[samples, max_length]
    );
    let y_tensor = Tensor::from_vec(labels, &[samples, 2]);
    
    (x_tensor, y_tensor)
}

/// Create RNN model for text classification
fn create_text_model(vocab_size: usize, embedding_dim: usize, hidden_dim: usize) -> Sequential<f32> {
    Sequential::new(vec![
        // Embedding layer
        Box::new(Embedding::new(vocab_size, embedding_dim)),
        
        // LSTM layer
        Box::new(LSTM::new(embedding_dim, hidden_dim, 1, true, false, 0.0, true)),
        
        // Dense layers
        Box::new(Dense::new(hidden_dim, 64, true).with_activation("relu".to_string())),
        Box::new(Dropout::new(0.3)),
        Box::new(Dense::new(64, 32, true).with_activation("relu".to_string())),
        Box::new(Dense::new(32, 2, true).with_activation("softmax".to_string())), // Binary classification
    ])
}

/// Training loop for text model
fn train_text_model<M: Model<f32>>(
    model: &mut M,
    optimizer: &mut Adam<f32>,
    train_dataset: &TensorDataset<f32>,
    epochs: usize,
    batch_size: usize,
) -> tenflowers_core::Result<()> {
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;
        let dataset_len = train_dataset.len();
        
        model.set_training(true);
        
        // Training loop
        for batch_start in (0..dataset_len).step_by(batch_size) {
            batch_count += 1;
            
            // Get sample from dataset
            let (sample_x, sample_y) = train_dataset.get(batch_start)?;
            
            // Forward pass
            let predictions = model.forward(&sample_x)?;
            let loss = categorical_cross_entropy(&predictions, &sample_y)?;
            
            // Get scalar loss value
            let loss_value: f32 = *loss.get(&[]).unwrap_or(&0.0);
            epoch_loss += loss_value;
            
            // Backward pass and optimization
            model.zero_grad();
            optimizer.step(model)?;
            
            if batch_count % 15 == 0 {
                println!("Epoch {}, Batch {}, Loss: {:.4}", epoch + 1, batch_count, loss_value);
            }
        }
        
        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f32 } else { 0.0 };
        println!("Epoch {} completed. Average Loss: {:.4}", epoch + 1, avg_loss);
    }
    
    Ok(())
}

/// Calculate accuracy for text classification
fn evaluate_text_accuracy<M: Model<f32>>(
    model: &mut M,
    dataset: &TensorDataset<f32>,
    num_samples: usize,
) -> tenflowers_core::Result<f32> {
    model.set_training(false);
    
    let mut correct = 0;
    let mut total = 0;
    let dataset_len = dataset.len().min(num_samples);
    
    for i in 0..dataset_len {
        let (sample_x, sample_y) = dataset.get(i)?;
        let predictions = model.forward(&sample_x)?;
        
        // Find predicted class (argmax)
        let pred_data = predictions.data();
        let pred_class = if pred_data.len() >= 2 {
            if pred_data[0] > pred_data[1] { 0 } else { 1 }
        } else { 0 };
        
        // Find true class
        let true_data = sample_y.data();
        let true_class = if true_data.len() >= 2 {
            if true_data[0] > true_data[1] { 0 } else { 1 }
        } else { 0 };
        
        if pred_class == true_class {
            correct += 1;
        }
        total += 1;
    }
    
    Ok(correct as f32 / total as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS Text Classification Example");
    println!("======================================");

    // Create vocabulary
    println!("Creating vocabulary...");
    let vocab = SimpleVocab::new();
    println!("Vocabulary size: {}", vocab.vocab_size);

    // Generate synthetic data
    println!("Generating synthetic text data...");
    let (x_train, y_train) = generate_synthetic_text_data(800, &vocab);
    let (x_val, y_val) = generate_synthetic_text_data(200, &vocab);

    println!("Training data shape: {:?}", x_train.shape());
    println!("Training labels shape: {:?}", y_train.shape());

    // Create text model
    println!("Creating LSTM model...");
    let embedding_dim = 32;
    let hidden_dim = 64;
    let mut model = create_text_model(vocab.vocab_size, embedding_dim, hidden_dim);

    // Create optimizer
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Create datasets
    let train_dataset = TensorDataset::new(x_train, y_train);
    let val_dataset = TensorDataset::new(x_val, y_val);

    println!("Starting training...");
    
    // Training configuration
    let epochs = 8;
    let batch_size = 16;
    
    // Train the model
    train_text_model(&mut model, &mut optimizer, &train_dataset, epochs, batch_size)?;

    println!("Training completed!");
    
    // Evaluate accuracy
    println!("Evaluating model...");
    
    let train_accuracy = evaluate_text_accuracy(&mut model, &train_dataset, 100)?;
    println!("Training Accuracy (100 samples): {:.2}%", train_accuracy * 100.0);
    
    let val_accuracy = evaluate_text_accuracy(&mut model, &val_dataset, 100)?;
    println!("Validation Accuracy (100 samples): {:.2}%", val_accuracy * 100.0);
    
    // Test on some example texts
    println!("\nTesting on example texts:");
    let test_texts = vec![
        "this movie is amazing and wonderful",
        "terrible product very disappointing experience",
        "love this book excellent writing",
        "hate this awful service",
    ];
    
    model.set_training(false);
    for (i, text) in test_texts.iter().enumerate() {
        let sequence = vocab.encode_text(text, 20);
        let input_tensor = Tensor::from_vec(
            sequence.iter().map(|&x| x as f32).collect(),
            &[1, 20]
        );
        
        let prediction = model.forward(&input_tensor)?;
        let pred_data = prediction.data();
        let sentiment = if pred_data[0] > pred_data[1] { "Negative" } else { "Positive" };
        let confidence = pred_data[1] * 100.0;
        
        println!("Text {}: \"{}\" -> {} ({:.1}% confidence)", 
                 i + 1, text, sentiment, confidence);
    }
    
    println!("\nText Classification example completed successfully!");
    
    Ok(())
}