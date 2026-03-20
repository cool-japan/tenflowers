//! Hierarchical Attention mechanism
//!
//! Implements a two-level attention mechanism for document classification:
//! 1. Word-level attention: attends over words within each sentence
//! 2. Sentence-level attention: attends over sentence representations
//!
//! Based on "Hierarchical Attention Networks for Document Classification"
//! <https://aclanthology.org/N16-1174/>
//!
//! Input tensor layout:
//!   [num_sentences, words_per_sentence, hidden_size]
//!
//! The forward pass:
//!   - For each sentence, compute word-level attention to get a sentence vector
//!   - Over all sentence vectors, compute sentence-level attention to get a document vector
//!
//! Output: [1, hidden_size] (document representation) or [num_sentences, hidden_size] via
//!         `forward_word_level` for per-sentence representations.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::Random;
use tenflowers_core::{Result, Tensor, TensorError};

/// Hierarchical Attention mechanism
///
/// Two-level (word + sentence) attention for document-level representation.
///
/// Weight matrices:
///   - Word level:  `w_word` [hidden_size, attention_size], `u_word` \[attention_size\]
///   - Sentence level: `w_sent` [hidden_size, attention_size], `u_sent` \[attention_size\]
///
/// Biases (optional): `b_word`, `b_sent` \[attention_size\]
#[derive(Debug)]
pub struct HierarchicalAttention<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    hidden_size: usize,
    attention_size: usize,

    // Word-level attention parameters
    w_word: Tensor<T>,
    b_word: Tensor<T>,
    u_word: Tensor<T>,

    // Sentence-level attention parameters
    w_sent: Tensor<T>,
    b_sent: Tensor<T>,
    u_sent: Tensor<T>,

    // Cached attention weights (for introspection after forward)
    last_word_weights: Option<Tensor<T>>,
    last_sent_weights: Option<Tensor<T>>,

    training: bool,
}

impl<T> Clone for HierarchicalAttention<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            attention_size: self.attention_size,
            w_word: self.w_word.clone(),
            b_word: self.b_word.clone(),
            u_word: self.u_word.clone(),
            w_sent: self.w_sent.clone(),
            b_sent: self.b_sent.clone(),
            u_sent: self.u_sent.clone(),
            last_word_weights: self.last_word_weights.clone(),
            last_sent_weights: self.last_sent_weights.clone(),
            training: self.training,
        }
    }
}

impl<T> HierarchicalAttention<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new Hierarchical Attention mechanism.
    ///
    /// # Arguments
    /// * `hidden_size` - Dimensionality of word / sentence hidden states.
    ///
    /// Attention inner dimension defaults to `hidden_size`.
    pub fn new(hidden_size: usize) -> Result<Self> {
        Self::with_attention_size(hidden_size, hidden_size)
    }

    /// Create with an explicit attention projection size.
    pub fn with_attention_size(hidden_size: usize, attention_size: usize) -> Result<Self> {
        if hidden_size == 0 {
            return Err(TensorError::invalid_argument(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if attention_size == 0 {
            return Err(TensorError::invalid_argument(
                "attention_size must be > 0".to_string(),
            ));
        }

        let scale = T::from(1.0 / (hidden_size as f64).sqrt())
            .ok_or_else(|| TensorError::invalid_argument("Failed to convert scale".to_string()))?;
        let att_scale = T::from(1.0 / (attention_size as f64).sqrt()).ok_or_else(|| {
            TensorError::invalid_argument("Failed to convert attention scale".to_string())
        })?;

        Ok(Self {
            hidden_size,
            attention_size,
            w_word: Self::init_weight(&[hidden_size, attention_size], scale)?,
            b_word: Tensor::zeros(&[attention_size]),
            u_word: Self::init_weight(&[attention_size], att_scale)?,
            w_sent: Self::init_weight(&[hidden_size, attention_size], scale)?,
            b_sent: Tensor::zeros(&[attention_size]),
            u_sent: Self::init_weight(&[attention_size], att_scale)?,
            last_word_weights: None,
            last_sent_weights: None,
            training: true,
        })
    }

    // ------------------------------------------------------------------ helpers

    fn init_weight(shape: &[usize], scale: T) -> Result<Tensor<T>> {
        let mut rng = Random::seed(42);
        let total = shape.iter().product::<usize>();
        let values: Vec<T> = (0..total)
            .map(|_| {
                let rv = rng.gen_range(-1.0..1.0);
                let t = T::from(rv).ok_or_else(|| {
                    TensorError::invalid_argument("Failed to convert random value".to_string())
                });
                // We use ok_or_else above but need to return T, not Result<T>.
                // Because Float guarantees finite conversion for f64 range values,
                // this is safe for all practical T (f32/f64).
                match t {
                    Ok(v) => v * scale,
                    Err(_) => T::zero(),
                }
            })
            .collect();
        Tensor::from_data(values, shape)
    }

    /// Numerically stable softmax over the first axis of a 2-D tensor
    /// `scores` : [seq_len, batch_size]  (softmax along dim-0 for each column)
    fn softmax_dim0(scores: &Tensor<T>) -> Result<Tensor<T>> {
        let shape = scores.shape().dims();
        let rows = shape[0];
        let cols = shape[1];

        let data = scores.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access scores data".to_string())
        })?;

        let mut out = vec![T::zero(); rows * cols];

        for c in 0..cols {
            // max for stability
            let mut mx = T::neg_infinity();
            for r in 0..rows {
                let v = data[r * cols + c];
                if v > mx {
                    mx = v;
                }
            }
            // exp + sum
            let mut sum = T::zero();
            for r in 0..rows {
                let idx = r * cols + c;
                let e = (data[idx] - mx).exp();
                out[idx] = e;
                sum = sum + e;
            }
            // guard against zero sum (e.g. rows == 0)
            if sum == T::zero() {
                let uniform = T::one() / T::from(rows).unwrap_or(T::one());
                for r in 0..rows {
                    out[r * cols + c] = uniform;
                }
            } else {
                for r in 0..rows {
                    out[r * cols + c] = out[r * cols + c] / sum;
                }
            }
        }

        Tensor::from_data(out, &[rows, cols])
    }

    /// Softmax over a 1-D tensor.
    fn softmax_1d(scores: &[T]) -> Vec<T> {
        let len = scores.len();
        if len == 0 {
            return Vec::new();
        }
        let mut mx = T::neg_infinity();
        for &v in scores {
            if v > mx {
                mx = v;
            }
        }
        let mut out = Vec::with_capacity(len);
        let mut sum = T::zero();
        for &v in scores {
            let e = (v - mx).exp();
            out.push(e);
            sum = sum + e;
        }
        if sum == T::zero() {
            let u = T::one() / T::from(len).unwrap_or(T::one());
            return vec![u; len];
        }
        for v in &mut out {
            *v = *v / sum;
        }
        out
    }

    // ------------------------------------------------------------------ core

    /// Word-level attention for a single sentence.
    ///
    /// `words` : [num_words, hidden_size]
    ///
    /// Returns `(sentence_vec [hidden_size], word_weights [num_words])`.
    fn attend_words(&self, words: &[T], num_words: usize) -> Result<(Vec<T>, Vec<T>)> {
        let h = self.hidden_size;
        let a = self.attention_size;

        let w_data = self
            .w_word
            .as_slice()
            .ok_or_else(|| TensorError::device_error_simple("Cannot access w_word".to_string()))?;
        let b_data = self
            .b_word
            .as_slice()
            .ok_or_else(|| TensorError::device_error_simple("Cannot access b_word".to_string()))?;
        let u_data = self
            .u_word
            .as_slice()
            .ok_or_else(|| TensorError::device_error_simple("Cannot access u_word".to_string()))?;

        // For each word: projected = tanh(W * word + b), score = u^T * projected
        let mut scores = Vec::with_capacity(num_words);
        // Keep projected for debugging if needed; we only need scores + original words
        for w_idx in 0..num_words {
            let word_start = w_idx * h;
            // proj_j = sum_k words[w_idx, k] * W[k, j] + b[j]
            let mut score = T::zero();
            for j in 0..a {
                let mut proj_j = b_data[j];
                for k in 0..h {
                    // W is [hidden_size, attention_size] row-major => W[k][j] = w_data[k*a + j]
                    proj_j = proj_j + words[word_start + k] * w_data[k * a + j];
                }
                // tanh
                proj_j = proj_j.tanh();
                // accumulate u^T * proj
                score = score + u_data[j] * proj_j;
            }
            scores.push(score);
        }

        // softmax over words
        let weights = Self::softmax_1d(&scores);

        // weighted sum of word vectors
        let mut sent_vec = vec![T::zero(); h];
        for w_idx in 0..num_words {
            let alpha = weights[w_idx];
            let word_start = w_idx * h;
            for k in 0..h {
                sent_vec[k] = sent_vec[k] + alpha * words[word_start + k];
            }
        }

        Ok((sent_vec, weights))
    }

    /// Sentence-level attention over sentence vectors.
    ///
    /// `sentence_vecs` : flat [num_sentences * hidden_size]
    ///
    /// Returns `(doc_vec [hidden_size], sent_weights [num_sentences])`.
    fn attend_sentences(
        &self,
        sentence_vecs: &[T],
        num_sentences: usize,
    ) -> Result<(Vec<T>, Vec<T>)> {
        let h = self.hidden_size;
        let a = self.attention_size;

        let w_data = self
            .w_sent
            .as_slice()
            .ok_or_else(|| TensorError::device_error_simple("Cannot access w_sent".to_string()))?;
        let b_data = self
            .b_sent
            .as_slice()
            .ok_or_else(|| TensorError::device_error_simple("Cannot access b_sent".to_string()))?;
        let u_data = self
            .u_sent
            .as_slice()
            .ok_or_else(|| TensorError::device_error_simple("Cannot access u_sent".to_string()))?;

        let mut scores = Vec::with_capacity(num_sentences);
        for s_idx in 0..num_sentences {
            let sv_start = s_idx * h;
            let mut score = T::zero();
            for j in 0..a {
                let mut proj_j = b_data[j];
                for k in 0..h {
                    proj_j = proj_j + sentence_vecs[sv_start + k] * w_data[k * a + j];
                }
                proj_j = proj_j.tanh();
                score = score + u_data[j] * proj_j;
            }
            scores.push(score);
        }

        let weights = Self::softmax_1d(&scores);

        let mut doc_vec = vec![T::zero(); h];
        for s_idx in 0..num_sentences {
            let alpha = weights[s_idx];
            let sv_start = s_idx * h;
            for k in 0..h {
                doc_vec[k] = doc_vec[k] + alpha * sentence_vecs[sv_start + k];
            }
        }

        Ok((doc_vec, weights))
    }

    /// Full hierarchical forward pass.
    ///
    /// # Arguments
    /// * `input` - Tensor of shape `[num_sentences, words_per_sentence, hidden_size]`.
    ///
    /// # Returns
    /// Document vector of shape `[1, hidden_size]`.
    pub fn forward_hierarchical(&mut self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let dims = input.shape().dims();
        if dims.len() != 3 {
            return Err(TensorError::invalid_argument(format!(
                "HierarchicalAttention expects 3-D input [num_sentences, words_per_sentence, hidden_size], got {} dims",
                dims.len()
            )));
        }
        let num_sentences = dims[0];
        let words_per_sentence = dims[1];
        let h = dims[2];

        if h != self.hidden_size {
            return Err(TensorError::invalid_argument(format!(
                "Expected hidden_size {}, got {}",
                self.hidden_size, h
            )));
        }

        let data = input.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access input data".to_string())
        })?;

        // Word-level attention for each sentence
        let mut sentence_vecs: Vec<T> = Vec::with_capacity(num_sentences * h);
        let mut all_word_weights: Vec<T> = Vec::with_capacity(num_sentences * words_per_sentence);

        for s in 0..num_sentences {
            let sent_start = s * words_per_sentence * h;
            let sent_end = sent_start + words_per_sentence * h;
            let (sv, ww) = self.attend_words(&data[sent_start..sent_end], words_per_sentence)?;
            sentence_vecs.extend_from_slice(&sv);
            all_word_weights.extend_from_slice(&ww);
        }

        // Cache word-level attention weights [num_sentences, words_per_sentence]
        self.last_word_weights = Some(Tensor::from_data(
            all_word_weights,
            &[num_sentences, words_per_sentence],
        )?);

        // Sentence-level attention
        let (doc_vec, sent_weights) = self.attend_sentences(&sentence_vecs, num_sentences)?;

        // Cache sentence-level attention weights [num_sentences]
        self.last_sent_weights = Some(Tensor::from_data(sent_weights, &[num_sentences])?);

        Tensor::from_data(doc_vec, &[1, h])
    }

    /// Word-level only: compute per-sentence representations.
    ///
    /// Input: `[num_sentences, words_per_sentence, hidden_size]`
    /// Output: `[num_sentences, hidden_size]`
    pub fn forward_word_level(&mut self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let dims = input.shape().dims();
        if dims.len() != 3 {
            return Err(TensorError::invalid_argument(format!(
                "Expected 3-D input, got {} dims",
                dims.len()
            )));
        }
        let num_sentences = dims[0];
        let words_per_sentence = dims[1];
        let h = dims[2];

        if h != self.hidden_size {
            return Err(TensorError::invalid_argument(format!(
                "Expected hidden_size {}, got {}",
                self.hidden_size, h
            )));
        }

        let data = input.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access input data".to_string())
        })?;

        let mut sentence_vecs: Vec<T> = Vec::with_capacity(num_sentences * h);
        let mut all_word_weights: Vec<T> = Vec::with_capacity(num_sentences * words_per_sentence);

        for s in 0..num_sentences {
            let start = s * words_per_sentence * h;
            let end = start + words_per_sentence * h;
            let (sv, ww) = self.attend_words(&data[start..end], words_per_sentence)?;
            sentence_vecs.extend_from_slice(&sv);
            all_word_weights.extend_from_slice(&ww);
        }

        self.last_word_weights = Some(Tensor::from_data(
            all_word_weights,
            &[num_sentences, words_per_sentence],
        )?);

        Tensor::from_data(sentence_vecs, &[num_sentences, h])
    }

    /// Return the last computed sentence-level attention weights.
    ///
    /// Shape: `[num_sentences]` after a `forward_hierarchical` call.
    pub fn get_attention_weights(&self) -> Option<&Tensor<T>> {
        self.last_sent_weights.as_ref()
    }

    /// Return the last computed word-level attention weights.
    ///
    /// Shape: `[num_sentences, words_per_sentence]` after any forward call.
    pub fn get_word_attention_weights(&self) -> Option<&Tensor<T>> {
        self.last_word_weights.as_ref()
    }
}

// -------------------------------------------------------------------------- Layer trait

impl<T> Layer<T> for HierarchicalAttention<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Forward pass through the Layer trait.
    ///
    /// Accepts either:
    ///   - 3-D `[num_sentences, words_per_sentence, hidden_size]` -> full hierarchical
    ///   - 2-D `[seq_len, hidden_size]` -> treated as a single sentence, word-level only
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let dims = input.shape().dims();

        match dims.len() {
            3 => {
                // Full hierarchical path.  We need &mut self for caching, so
                // we clone, run, and return the tensor (weights are not cached
                // through the immutable Layer trait — use forward_hierarchical
                // directly for introspection).
                let mut clone = self.clone();
                clone.forward_hierarchical(input)
            }
            2 => {
                // Single sentence: [seq_len, hidden_size]
                let seq_len = dims[0];
                let h = dims[1];
                if h != self.hidden_size {
                    return Err(TensorError::invalid_argument(format!(
                        "Expected hidden_size {}, got {}",
                        self.hidden_size, h
                    )));
                }
                let data = input.as_slice().ok_or_else(|| {
                    TensorError::device_error_simple("Cannot access input data".to_string())
                })?;
                let (sent_vec, _weights) = self.attend_words(data, seq_len)?;
                Tensor::from_data(sent_vec, &[1, h])
            }
            _ => Err(TensorError::invalid_argument(format!(
                "HierarchicalAttention expects 2-D or 3-D input, got {} dims",
                dims.len()
            ))),
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![
            &self.w_word,
            &self.b_word,
            &self.u_word,
            &self.w_sent,
            &self.b_sent,
            &self.u_sent,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![
            &mut self.w_word,
            &mut self.b_word,
            &mut self.u_word,
            &mut self.w_sent,
            &mut self.b_sent,
            &mut self.u_sent,
        ]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

// -------------------------------------------------------------------------- Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input_3d(num_sentences: usize, words: usize, hidden: usize) -> Tensor<f32> {
        let total = num_sentences * words * hidden;
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        Tensor::from_data(data, &[num_sentences, words, hidden]).expect("tensor creation")
    }

    #[test]
    fn test_new_valid() {
        let att = HierarchicalAttention::<f32>::new(16);
        assert!(att.is_ok());
    }

    #[test]
    fn test_new_zero_hidden() {
        let att = HierarchicalAttention::<f32>::new(0);
        assert!(att.is_err());
    }

    #[test]
    fn test_with_attention_size() {
        let att = HierarchicalAttention::<f32>::with_attention_size(32, 16);
        assert!(att.is_ok());
        let att = att.expect("should succeed");
        assert_eq!(att.hidden_size, 32);
        assert_eq!(att.attention_size, 16);
    }

    #[test]
    fn test_forward_hierarchical_shape() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(3, 5, 8);
        let out = att.forward_hierarchical(&input).expect("forward");
        assert_eq!(out.shape().dims(), &[1, 8]);
    }

    #[test]
    fn test_word_level_shape() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(4, 6, 8);
        let out = att.forward_word_level(&input).expect("forward_word_level");
        assert_eq!(out.shape().dims(), &[4, 8]);
    }

    #[test]
    fn test_attention_weights_cached() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(3, 5, 8);
        let _ = att.forward_hierarchical(&input).expect("forward");

        let sw = att.get_attention_weights().expect("sent weights");
        assert_eq!(sw.shape().dims(), &[3]);

        let ww = att.get_word_attention_weights().expect("word weights");
        assert_eq!(ww.shape().dims(), &[3, 5]);
    }

    #[test]
    fn test_word_weights_sum_to_one() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(2, 4, 8);
        let _ = att.forward_hierarchical(&input).expect("forward");

        let ww = att.get_word_attention_weights().expect("word weights");
        let ww_data = ww.as_slice().expect("slice");
        // Each sentence's word weights should sum to ~1.0
        for s in 0..2 {
            let sum: f32 = (0..4).map(|w| ww_data[s * 4 + w]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "word weights sum = {}", sum);
        }
    }

    #[test]
    fn test_sent_weights_sum_to_one() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(5, 3, 8);
        let _ = att.forward_hierarchical(&input).expect("forward");

        let sw = att.get_attention_weights().expect("sent weights");
        let sw_data = sw.as_slice().expect("slice");
        let sum: f32 = sw_data.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sent weights sum = {}", sum);
    }

    #[test]
    fn test_layer_trait_forward_3d() {
        let att = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(2, 4, 8);
        let out = att.forward(&input).expect("Layer forward");
        assert_eq!(out.shape().dims(), &[1, 8]);
    }

    #[test]
    fn test_layer_trait_forward_2d() {
        let att = HierarchicalAttention::<f32>::new(8).expect("new");
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::from_data(data, &[3, 8]).expect("tensor");
        let out = att.forward(&input).expect("Layer forward 2d");
        assert_eq!(out.shape().dims(), &[1, 8]);
    }

    #[test]
    fn test_parameters_count() {
        let att = HierarchicalAttention::<f32>::new(8).expect("new");
        assert_eq!(att.parameters().len(), 6);
    }

    #[test]
    fn test_parameters_mut_count() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        assert_eq!(att.parameters_mut().len(), 6);
    }

    #[test]
    fn test_set_training() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        assert!(att.training);
        att.set_training(false);
        assert!(!att.training);
        att.set_training(true);
        assert!(att.training);
    }

    #[test]
    fn test_clone_box() {
        let att = HierarchicalAttention::<f32>::new(8).expect("new");
        let boxed: Box<dyn Layer<f32>> = att.clone_box();
        let input = make_input_3d(2, 3, 8);
        let out = boxed.forward(&input).expect("cloned forward");
        assert_eq!(out.shape().dims(), &[1, 8]);
    }

    #[test]
    fn test_wrong_hidden_size() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(2, 3, 16); // hidden=16, expects 8
        let res = att.forward_hierarchical(&input);
        assert!(res.is_err());
    }

    #[test]
    fn test_wrong_dims() {
        let mut att = HierarchicalAttention::<f32>::new(8).expect("new");
        let data: Vec<f32> = vec![0.0; 8];
        let input = Tensor::from_data(data, &[8]).expect("1d tensor");
        let res = att.forward_hierarchical(&input);
        assert!(res.is_err());
    }

    #[test]
    fn test_single_sentence_single_word() {
        let mut att = HierarchicalAttention::<f32>::new(4).expect("new");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data.clone(), &[1, 1, 4]).expect("tensor");
        let out = att.forward_hierarchical(&input).expect("forward");
        // With a single word and single sentence, the output should equal the input word
        let out_data = out.as_slice().expect("slice");
        for i in 0..4 {
            assert!(
                (out_data[i] - data[i]).abs() < 1e-5,
                "elem {} mismatch: {} vs {}",
                i,
                out_data[i],
                data[i]
            );
        }
    }

    #[test]
    fn test_f64_support() {
        let mut att = HierarchicalAttention::<f64>::new(8).expect("new");
        let total = 2 * 3 * 8;
        let data: Vec<f64> = (0..total).map(|i| i as f64 * 0.01).collect();
        let input = Tensor::from_data(data, &[2, 3, 8]).expect("tensor");
        let out = att.forward_hierarchical(&input).expect("forward");
        assert_eq!(out.shape().dims(), &[1, 8]);
    }

    #[test]
    fn test_deterministic_output() {
        // Two identical runs should produce identical results
        let mut att1 = HierarchicalAttention::<f32>::new(8).expect("new");
        let mut att2 = HierarchicalAttention::<f32>::new(8).expect("new");
        let input = make_input_3d(2, 4, 8);

        let out1 = att1.forward_hierarchical(&input).expect("forward1");
        let out2 = att2.forward_hierarchical(&input).expect("forward2");

        let d1 = out1.as_slice().expect("s1");
        let d2 = out2.as_slice().expect("s2");
        for i in 0..d1.len() {
            assert!(
                (d1[i] - d2[i]).abs() < 1e-6,
                "non-deterministic at {}: {} vs {}",
                i,
                d1[i],
                d2[i]
            );
        }
    }
}
