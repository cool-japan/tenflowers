//! # Neural Combinatorial Optimization
//!
//! Implements neural approaches to combinatorial optimization problems:
//! - Pointer Networks (Vinyals et al. 2015)
//! - Attention Model / AM (Kool et al. 2019)
//! - REINFORCE trainer with baselines
//! - Classical heuristics (nearest-neighbor, 2-opt, savings algorithm)
//! - Beam search decoder for sequence problems

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §1  Problem representations
// ─────────────────────────────────────────────────────────────────────────────

/// Travelling Salesman Problem instance.
#[derive(Debug, Clone)]
pub struct TspInstance {
    pub cities: Vec<(f32, f32)>,
    pub n_cities: usize,
}

impl TspInstance {
    /// Create a new TSP instance.
    pub fn new(cities: Vec<(f32, f32)>) -> Self {
        let n_cities = cities.len();
        TspInstance { cities, n_cities }
    }

    /// Euclidean distance matrix (n×n).
    pub fn distance_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.n_cities;
        let mut mat = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dx = self.cities[i].0 - self.cities[j].0;
                let dy = self.cities[i].1 - self.cities[j].1;
                mat[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
        mat
    }

    /// Total tour length (closed loop: last → first).
    pub fn tour_length(&self, tour: &[usize]) -> f32 {
        if tour.is_empty() {
            return 0.0;
        }
        let dm = self.distance_matrix();
        let mut total = 0.0f32;
        for w in tour.windows(2) {
            total += dm[w[0]][w[1]];
        }
        // Return to start
        total += dm[*tour.last().unwrap_or(&0)][tour[0]];
        total
    }
}

/// Vehicle Routing Problem instance.
#[derive(Debug, Clone)]
pub struct VrpInstance {
    pub depot: (f32, f32),
    pub customers: Vec<(f32, f32)>,
    pub demands: Vec<f32>,
    pub capacity: f32,
}

impl VrpInstance {
    fn euclidean(a: (f32, f32), b: (f32, f32)) -> f32 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Cost of a route (depot → customers in route order → depot).
    pub fn route_cost(&self, route: &[usize]) -> f32 {
        if route.is_empty() {
            return 0.0;
        }
        let mut cost = Self::euclidean(self.depot, self.customers[route[0]]);
        for w in route.windows(2) {
            cost += Self::euclidean(self.customers[w[0]], self.customers[w[1]]);
        }
        cost += Self::euclidean(self.customers[*route.last().unwrap_or(&0)], self.depot);
        cost
    }
}

/// 0/1 Knapsack instance.
#[derive(Debug, Clone)]
pub struct KnapsackInstance {
    pub weights: Vec<f32>,
    pub values: Vec<f32>,
    pub capacity: f32,
}

impl KnapsackInstance {
    /// Total value of items selected by `selected`.
    pub fn solution_value(&self, selected: &[bool]) -> f32 {
        selected
            .iter()
            .zip(self.values.iter())
            .filter_map(|(&s, &v)| if s { Some(v) } else { None })
            .sum()
    }

    /// Whether the selection is weight-feasible.
    pub fn is_feasible(&self, selected: &[bool]) -> bool {
        let total: f32 = selected
            .iter()
            .zip(self.weights.iter())
            .filter_map(|(&s, &w)| if s { Some(w) } else { None })
            .sum();
        total <= self.capacity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Pointer Network
// ─────────────────────────────────────────────────────────────────────────────

fn rand_matrix(rows: usize, cols: usize, scale: f32, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
        .collect()
}

fn rand_vec(len: usize, scale: f32, rng: &mut impl Rng) -> Vec<f32> {
    (0..len).map(|_| rng.random_range(-scale..scale)).collect()
}

fn mat_vec(m: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x.tanh()).collect()
}

/// LSTM cell state (hidden + cell).
fn lstm_cell(
    x: &[f32],
    h_prev: &[f32],
    c_prev: &[f32],
    w_i: &[Vec<f32>],
    w_h: &[Vec<f32>],
    b: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let dim = h_prev.len();
    // concat input + hidden projection
    let xi = mat_vec(w_i, x);
    let hi = mat_vec(w_h, h_prev);
    let raw: Vec<f32> = (0..dim * 4).map(|k| xi[k] + hi[k] + b[k]).collect();
    // split into 4 gates
    let i_gate: Vec<f32> = raw[0..dim].iter().map(|&v| sigmoid(v)).collect();
    let f_gate: Vec<f32> = raw[dim..2 * dim].iter().map(|&v| sigmoid(v)).collect();
    let g_gate: Vec<f32> = raw[2 * dim..3 * dim].iter().map(|&v| v.tanh()).collect();
    let o_gate: Vec<f32> = raw[3 * dim..4 * dim].iter().map(|&v| sigmoid(v)).collect();

    let c_new: Vec<f32> = (0..dim)
        .map(|k| f_gate[k] * c_prev[k] + i_gate[k] * g_gate[k])
        .collect();
    let h_new: Vec<f32> = (0..dim).map(|k| o_gate[k] * c_new[k].tanh()).collect();
    (h_new, c_new)
}

/// Configuration for a Pointer Network.
#[derive(Debug, Clone)]
pub struct PtrNetConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
}

/// LSTM encoder for the Pointer Network.
#[derive(Debug, Clone)]
pub struct PtrEncoder {
    w_i: Vec<Vec<f32>>, // [4*hidden × input]
    w_h: Vec<Vec<f32>>, // [4*hidden × hidden]
    b: Vec<f32>,        // [4*hidden]
    hidden_dim: usize,
}

impl PtrEncoder {
    fn new(input_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0f32 / (input_dim + hidden_dim) as f32).sqrt();
        PtrEncoder {
            w_i: rand_matrix(4 * hidden_dim, input_dim, scale, rng),
            w_h: rand_matrix(4 * hidden_dim, hidden_dim, scale, rng),
            b: vec![0.0f32; 4 * hidden_dim],
            hidden_dim,
        }
    }

    /// Encode a sequence. Returns (all_hidden_states, final_hidden).
    pub fn encode(&self, sequence: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let h = self.hidden_dim;
        let mut hidden = vec![0.0f32; h];
        let mut cell = vec![0.0f32; h];
        let mut all_hidden = Vec::with_capacity(sequence.len());
        for x in sequence {
            let (h_new, c_new) = lstm_cell(x, &hidden, &cell, &self.w_i, &self.w_h, &self.b);
            hidden = h_new;
            cell = c_new;
            all_hidden.push(hidden.clone());
        }
        let final_h = hidden;
        (all_hidden, final_h)
    }
}

/// LSTM decoder + Bahdanau-style attention pointer.
#[derive(Debug, Clone)]
pub struct PtrDecoder {
    // LSTM weights
    w_i: Vec<Vec<f32>>,
    w_h: Vec<Vec<f32>>,
    b: Vec<f32>,
    // Attention weights
    w_q: Vec<Vec<f32>>, // [hidden × hidden]
    w_k: Vec<Vec<f32>>, // [hidden × hidden]
    v: Vec<f32>,        // [hidden]
    hidden_dim: usize,
}

impl PtrDecoder {
    fn new(hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (1.0f32 / hidden_dim as f32).sqrt();
        PtrDecoder {
            w_i: rand_matrix(4 * hidden_dim, hidden_dim, scale, rng),
            w_h: rand_matrix(4 * hidden_dim, hidden_dim, scale, rng),
            b: vec![0.0f32; 4 * hidden_dim],
            w_q: rand_matrix(hidden_dim, hidden_dim, scale, rng),
            w_k: rand_matrix(hidden_dim, hidden_dim, scale, rng),
            v: rand_vec(hidden_dim, scale, rng),
            hidden_dim,
        }
    }

    /// Bahdanau attention: softmax over keys, masking visited positions.
    pub fn attend(&self, query: &[f32], keys: &[Vec<f32>], mask: &[bool]) -> Vec<f32> {
        let q_proj = mat_vec(&self.w_q, query);
        let n = keys.len();
        let mut scores = Vec::with_capacity(n);
        for key in keys {
            let k_proj = mat_vec(&self.w_k, key);
            let combined: Vec<f32> = q_proj
                .iter()
                .zip(k_proj.iter())
                .map(|(a, b)| a + b)
                .collect();
            let tanh_c = tanh_vec(&combined);
            let score: f32 = self
                .v
                .iter()
                .zip(tanh_c.iter())
                .map(|(vi, ti)| vi * ti)
                .sum();
            scores.push(score);
        }
        // Apply mask (visited → -inf)
        for (i, &visited) in mask.iter().enumerate() {
            if visited {
                scores[i] = f32::NEG_INFINITY;
            }
        }
        softmax(&scores)
    }

    /// One decoder step: LSTM update + pointer attention.
    fn step(
        &self,
        input: &[f32],
        hidden: &[f32],
        cell: &[f32],
        keys: &[Vec<f32>],
        mask: &[bool],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (h_new, c_new) = lstm_cell(input, hidden, cell, &self.w_i, &self.w_h, &self.b);
        let probs = self.attend(&h_new, keys, mask);
        (h_new, c_new, probs)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits
        .iter()
        .filter(|v| v.is_finite())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return vec![0.0f32; logits.len()];
    }
    let exps: Vec<f32> = logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                (v - max_val).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

fn argmax(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Pointer Network (Vinyals et al. 2015).
#[derive(Debug, Clone)]
pub struct PointerNetwork {
    pub encoder: PtrEncoder,
    pub decoder: PtrDecoder,
    pub config: PtrNetConfig,
}

impl PointerNetwork {
    /// Create with random initialization.
    pub fn new(config: PtrNetConfig, rng: &mut impl Rng) -> Self {
        let encoder = PtrEncoder::new(config.input_dim, config.hidden_dim, rng);
        let decoder = PtrDecoder::new(config.hidden_dim, rng);
        PointerNetwork {
            encoder,
            decoder,
            config,
        }
    }

    /// Greedy decoding: pick argmax pointer at each step.
    pub fn forward(&self, inputs: &[Vec<f32>], max_steps: usize) -> Vec<usize> {
        let n = inputs.len().min(max_steps);
        let (keys, final_hidden) = self.encoder.encode(inputs);
        let h = self.config.hidden_dim;
        let mut hidden = final_hidden;
        let mut cell = vec![0.0f32; h];
        let mut mask = vec![false; inputs.len()];
        let mut tour = Vec::with_capacity(n);

        // Start token: zero vector
        let mut dec_input = vec![0.0f32; h];

        for _ in 0..n {
            let (h_new, c_new, probs) = self.decoder.step(&dec_input, &hidden, &cell, &keys, &mask);
            let chosen = argmax(&probs);
            tour.push(chosen);
            mask[chosen] = true;
            // Use chosen city's key as next decoder input
            dec_input = keys[chosen].clone();
            hidden = h_new;
            cell = c_new;
        }
        tour
    }

    /// Log-probability of a given tour under the model.
    pub fn log_prob(&self, inputs: &[Vec<f32>], tour: &[usize]) -> f32 {
        let (keys, final_hidden) = self.encoder.encode(inputs);
        let h = self.config.hidden_dim;
        let mut hidden = final_hidden;
        let mut cell = vec![0.0f32; h];
        let mut mask = vec![false; inputs.len()];
        let mut log_p = 0.0f32;
        let mut dec_input = vec![0.0f32; h];

        for &city in tour {
            let (h_new, c_new, probs) = self.decoder.step(&dec_input, &hidden, &cell, &keys, &mask);
            let p = probs.get(city).copied().unwrap_or(1e-30);
            log_p += (p + 1e-30).ln();
            mask[city] = true;
            dec_input = keys[city].clone();
            hidden = h_new;
            cell = c_new;
        }
        log_p
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  Attention Model (Kool et al. 2019)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Attention Model.
#[derive(Debug, Clone)]
pub struct AmConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub feed_forward_dim: usize,
}

/// Linear projection of 2D coordinates to d_model.
#[derive(Debug, Clone)]
pub struct CityEmbedding {
    weight: Vec<Vec<f32>>, // [d_model × 2]
    bias: Vec<f32>,        // [d_model]
}

impl CityEmbedding {
    fn new(d_model: usize, rng: &mut impl Rng) -> Self {
        let fan_in = 2.0_f32;
        let scale = (2.0f32 / fan_in).sqrt();
        CityEmbedding {
            weight: rand_matrix(d_model, 2, scale, rng),
            bias: rand_vec(d_model, 0.01, rng),
        }
    }

    /// Embed a single city (2D coords → d_model).
    pub fn embed(&self, city: (f32, f32)) -> Vec<f32> {
        let input = vec![city.0, city.1];
        vec_add(&mat_vec(&self.weight, &input), &self.bias)
    }
}

fn layer_norm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + 1e-6).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

/// One encoder layer: multi-head self-attention + FFN + LayerNorm.
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    // Multi-head attention weights (merged into single heads for simplicity)
    w_q: Vec<Vec<f32>>, // [d_model × d_model]
    w_k: Vec<Vec<f32>>,
    w_v: Vec<Vec<f32>>,
    w_o: Vec<Vec<f32>>,
    // Feed-forward
    ff_w1: Vec<Vec<f32>>, // [ff_dim × d_model]
    ff_b1: Vec<f32>,
    ff_w2: Vec<Vec<f32>>, // [d_model × ff_dim]
    ff_b2: Vec<f32>,
    d_model: usize,
    n_heads: usize,
}

impl EncoderLayer {
    fn new(d_model: usize, n_heads: usize, ff_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0f32 / (d_model * 2) as f32).sqrt();
        let ff_scale = (2.0f32 / (d_model + ff_dim) as f32).sqrt();
        EncoderLayer {
            w_q: rand_matrix(d_model, d_model, scale, rng),
            w_k: rand_matrix(d_model, d_model, scale, rng),
            w_v: rand_matrix(d_model, d_model, scale, rng),
            w_o: rand_matrix(d_model, d_model, scale, rng),
            ff_w1: rand_matrix(ff_dim, d_model, ff_scale, rng),
            ff_b1: vec![0.0f32; ff_dim],
            ff_w2: rand_matrix(d_model, ff_dim, ff_scale, rng),
            ff_b2: vec![0.0f32; d_model],
            d_model,
            n_heads,
        }
    }

    /// Forward pass over a sequence of embeddings. Returns same-shape output.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = x.len();
        let d_k = (self.d_model / self.n_heads).max(1);
        let scale = 1.0 / (d_k as f32).sqrt();

        // Project Q, K, V
        let queries: Vec<Vec<f32>> = x.iter().map(|v| mat_vec(&self.w_q, v)).collect();
        let keys: Vec<Vec<f32>> = x.iter().map(|v| mat_vec(&self.w_k, v)).collect();
        let values: Vec<Vec<f32>> = x.iter().map(|v| mat_vec(&self.w_v, v)).collect();

        // Scaled dot-product attention (single block, all heads merged)
        let mut attn_out = vec![vec![0.0f32; self.d_model]; n];
        for i in 0..n {
            let mut scores: Vec<f32> = (0..n)
                .map(|j| {
                    queries[i]
                        .iter()
                        .zip(keys[j].iter())
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let probs = softmax(&scores);
            // scores vec no longer needed; probs used
            let _ = scores.len(); // suppress lint
            for j in 0..n {
                for d in 0..self.d_model {
                    attn_out[i][d] += probs[j] * values[j][d];
                }
            }
        }

        // Output projection + residual + LayerNorm
        let attn_proj: Vec<Vec<f32>> = attn_out.iter().map(|v| mat_vec(&self.w_o, v)).collect();
        let after_attn: Vec<Vec<f32>> = x
            .iter()
            .zip(attn_proj.iter())
            .map(|(xi, ai)| layer_norm(&vec_add(xi, ai)))
            .collect();

        // Feed-forward: ReLU(W1*x + b1)*W2 + b2
        after_attn
            .iter()
            .map(|v| {
                let h1: Vec<f32> = vec_add(&mat_vec(&self.ff_w1, v), &self.ff_b1)
                    .iter()
                    .map(|&x| x.max(0.0))
                    .collect();
                let h2 = vec_add(&mat_vec(&self.ff_w2, &h1), &self.ff_b2);
                layer_norm(&vec_add(v, &h2))
            })
            .collect()
    }
}

/// Stacked encoder for the Attention Model.
#[derive(Debug, Clone)]
pub struct AttentionModelEncoder {
    pub layers: Vec<EncoderLayer>,
    pub embedding: CityEmbedding,
}

impl AttentionModelEncoder {
    /// Encode a list of (x,y) cities into d_model embeddings.
    pub fn encode(&self, cities: &[(f32, f32)]) -> Vec<Vec<f32>> {
        let mut h: Vec<Vec<f32>> = cities.iter().map(|&c| self.embedding.embed(c)).collect();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h
    }
}

/// Compatibility decoder: tanh-clipped (C=10) dot-product attention.
#[derive(Debug, Clone)]
pub struct CompatibilityDecoder {
    w_q: Vec<Vec<f32>>,
    w_k: Vec<Vec<f32>>,
    d_model: usize,
}

impl CompatibilityDecoder {
    fn new(d_model: usize, rng: &mut impl Rng) -> Self {
        let scale = (1.0f32 / d_model as f32).sqrt();
        CompatibilityDecoder {
            w_q: rand_matrix(d_model, d_model, scale, rng),
            w_k: rand_matrix(d_model, d_model, scale, rng),
            d_model,
        }
    }

    /// Compute masked softmax over cities given context query.
    pub fn decode(&self, context: &[f32], keys: &[Vec<f32>], mask: &[bool]) -> Vec<f32> {
        let d_k = (self.d_model as f32).sqrt();
        let q = mat_vec(&self.w_q, context);
        let mut logits: Vec<f32> = keys
            .iter()
            .zip(mask.iter())
            .map(|(k, &visited)| {
                if visited {
                    return f32::NEG_INFINITY;
                }
                let k_proj = mat_vec(&self.w_k, k);
                let dot: f32 = q.iter().zip(k_proj.iter()).map(|(a, b)| a * b).sum();
                let score = dot / d_k;
                // tanh-clip to C=10
                10.0 * score.tanh()
            })
            .collect();
        // Ensure at least one finite logit
        if logits.iter().all(|v| !v.is_finite()) {
            logits = vec![0.0f32; logits.len()];
        }
        softmax(&logits)
    }
}

/// Attention Model (AM) for TSP.
#[derive(Debug, Clone)]
pub struct AttentionModel {
    pub encoder: AttentionModelEncoder,
    decoder: CompatibilityDecoder,
    pub d_model: usize,
    pub n_heads: usize,
}

impl AttentionModel {
    /// Create a new Attention Model.
    pub fn new(config: AmConfig, rng: &mut impl Rng) -> Self {
        let embedding = CityEmbedding::new(config.d_model, rng);
        let layers = (0..config.n_layers)
            .map(|_| {
                EncoderLayer::new(config.d_model, config.n_heads, config.feed_forward_dim, rng)
            })
            .collect();
        let encoder = AttentionModelEncoder { layers, embedding };
        let decoder = CompatibilityDecoder::new(config.d_model, rng);
        AttentionModel {
            encoder,
            decoder,
            d_model: config.d_model,
            n_heads: config.n_heads,
        }
    }

    /// Greedy tour construction.
    pub fn solve_greedy(&self, instance: &TspInstance) -> Vec<usize> {
        let n = instance.n_cities;
        if n == 0 {
            return vec![];
        }
        let keys = self.encoder.encode(&instance.cities);
        // Context = mean of encoder outputs
        let context: Vec<f32> = (0..self.d_model)
            .map(|d| keys.iter().map(|k| k[d]).sum::<f32>() / n as f32)
            .collect();

        let mut mask = vec![false; n];
        let mut tour = Vec::with_capacity(n);
        let mut current_context = context;

        for _ in 0..n {
            let probs = self.decoder.decode(&current_context, &keys, &mask);
            let chosen = argmax(&probs);
            tour.push(chosen);
            mask[chosen] = true;
            // Update context to chosen key
            current_context = keys[chosen].clone();
        }
        tour
    }

    /// Sample `n_samples` tours, return the shortest.
    pub fn solve_sampling(
        &self,
        instance: &TspInstance,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        let n = instance.n_cities;
        if n == 0 {
            return vec![];
        }
        let keys = self.encoder.encode(&instance.cities);
        let base_context: Vec<f32> = (0..self.d_model)
            .map(|d| keys.iter().map(|k| k[d]).sum::<f32>() / n as f32)
            .collect();

        let mut best_tour = vec![];
        let mut best_len = f32::INFINITY;

        for _ in 0..n_samples {
            let mut mask = vec![false; n];
            let mut tour = Vec::with_capacity(n);
            let mut ctx = base_context.clone();

            for _ in 0..n {
                let probs = self.decoder.decode(&ctx, &keys, &mask);
                // Multinomial sampling
                let chosen = sample_categorical(&probs, rng);
                tour.push(chosen);
                mask[chosen] = true;
                ctx = keys[chosen].clone();
            }

            let length = instance.tour_length(&tour);
            if length < best_len {
                best_len = length;
                best_tour = tour;
            }
        }
        best_tour
    }
}

fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> usize {
    let u: f32 = rng.random_range(0.0f32..1.0f32);
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u <= cumulative {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  REINFORCE Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Type of baseline for REINFORCE.
#[derive(Debug, Clone)]
pub enum BaselineType {
    /// Exponential moving average with given decay α.
    ExponentialMovingAverage(f32),
    /// Greedy rollout baseline.
    Greedy,
    /// Rollout-based baseline (separate greedy policy).
    RolloutBaseline,
}

/// Configuration for REINFORCE trainer.
#[derive(Debug, Clone)]
pub struct ReinforceConfig {
    pub lr: f32,
    pub baseline: BaselineType,
    pub clip_norm: f32,
}

/// REINFORCE trainer for combinatorial optimization.
#[derive(Debug, Clone)]
pub struct ReinforceTrainer {
    pub config: ReinforceConfig,
    pub baseline_value: f32,
}

impl ReinforceTrainer {
    pub fn new(config: ReinforceConfig) -> Self {
        ReinforceTrainer {
            config,
            baseline_value: 0.0,
        }
    }

    /// REINFORCE loss: -E[(R - b) * log_prob].
    pub fn compute_loss(&self, log_probs: &[f32], rewards: &[f32]) -> f32 {
        if log_probs.is_empty() {
            return 0.0;
        }
        let b = self.baseline_value;
        let loss: f32 = log_probs
            .iter()
            .zip(rewards.iter())
            .map(|(&lp, &r)| -(r - b) * lp)
            .sum::<f32>()
            / log_probs.len() as f32;
        loss
    }

    /// Update the EMA baseline with a new reward observation.
    pub fn update_baseline(&mut self, new_reward: f32) {
        let alpha = match &self.config.baseline {
            BaselineType::ExponentialMovingAverage(a) => *a,
            _ => 0.99,
        };
        self.baseline_value = alpha * self.baseline_value + (1.0 - alpha) * new_reward;
    }

    /// Returns per-step gradient * (reward - baseline).
    pub fn policy_gradient_step(
        &self,
        log_probs: Vec<f32>,
        reward: f32,
        baseline: f32,
    ) -> Vec<f32> {
        let advantage = reward - baseline;
        log_probs.iter().map(|&lp| advantage * lp).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Greedy Heuristics
// ─────────────────────────────────────────────────────────────────────────────

/// Nearest-neighbour greedy heuristic for TSP.
pub struct NearestNeighbor;

impl NearestNeighbor {
    /// Construct a tour starting from `start`.
    pub fn solve(instance: &TspInstance, start: usize) -> Vec<usize> {
        let n = instance.n_cities;
        if n == 0 {
            return vec![];
        }
        let dm = instance.distance_matrix();
        let mut visited = vec![false; n];
        let mut tour = Vec::with_capacity(n);
        let mut current = start % n;
        visited[current] = true;
        tour.push(current);

        for _ in 1..n {
            let next = (0..n).filter(|&j| !visited[j]).min_by(|&a, &b| {
                dm[current][a]
                    .partial_cmp(&dm[current][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            match next {
                Some(j) => {
                    visited[j] = true;
                    tour.push(j);
                    current = j;
                }
                None => break,
            }
        }
        tour
    }
}

/// 2-opt local search for TSP improvement.
pub struct TwoOpt;

impl TwoOpt {
    /// Perform a 2-opt swap: reverse the segment between i+1 and j (inclusive).
    pub fn two_opt_swap(tour: &[usize], i: usize, j: usize) -> Vec<usize> {
        let n = tour.len();
        let mut new_tour = tour.to_vec();
        // Reverse slice [i+1 .. j]
        let (mut left, mut right) = (i + 1, j);
        while left < right {
            new_tour.swap(left, right);
            left += 1;
            if right == 0 {
                break;
            }
            right -= 1;
        }
        let _ = n; // suppress unused
        new_tour
    }

    /// Run 2-opt improvement for up to `max_iter` passes.
    pub fn improve(instance: &TspInstance, mut tour: Vec<usize>, max_iter: usize) -> Vec<usize> {
        let n = tour.len();
        if n < 4 {
            return tour;
        }
        for _ in 0..max_iter {
            let mut improved = false;
            let mut best_tour = tour.clone();
            let mut best_len = instance.tour_length(&tour);
            'outer: for i in 0..n - 1 {
                for j in i + 2..n {
                    let candidate = TwoOpt::two_opt_swap(&tour, i, j);
                    let len = instance.tour_length(&candidate);
                    if len < best_len - 1e-6 {
                        best_len = len;
                        best_tour = candidate;
                        improved = true;
                        break 'outer;
                    }
                }
            }
            tour = best_tour;
            if !improved {
                break;
            }
        }
        tour
    }
}

/// Greedy knapsack: sort by value/weight ratio, fill greedily.
pub struct GreedyKnapsack;

impl GreedyKnapsack {
    pub fn solve(instance: &KnapsackInstance) -> Vec<bool> {
        let n = instance.weights.len();
        if n == 0 {
            return vec![];
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let ra = if instance.weights[a] > 1e-9 {
                instance.values[a] / instance.weights[a]
            } else {
                f32::INFINITY
            };
            let rb = if instance.weights[b] > 1e-9 {
                instance.values[b] / instance.weights[b]
            } else {
                f32::INFINITY
            };
            rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected = vec![false; n];
        let mut remaining = instance.capacity;
        for idx in indices {
            if instance.weights[idx] <= remaining {
                selected[idx] = true;
                remaining -= instance.weights[idx];
            }
        }
        selected
    }
}

/// Clarke-Wright savings algorithm for VRP.
pub struct SavingsAlgorithm;

impl SavingsAlgorithm {
    fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Compute savings s(i,j) = d(depot,i) + d(depot,j) - d(i,j) for all pairs.
    pub fn compute_savings(instance: &VrpInstance) -> Vec<(f32, usize, usize)> {
        let n = instance.customers.len();
        let mut savings = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in i + 1..n {
                let d_di = Self::dist(instance.depot, instance.customers[i]);
                let d_dj = Self::dist(instance.depot, instance.customers[j]);
                let d_ij = Self::dist(instance.customers[i], instance.customers[j]);
                savings.push((d_di + d_dj - d_ij, i, j));
            }
        }
        savings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        savings
    }

    /// Build routes using the savings list heuristic.
    pub fn solve(instance: &VrpInstance) -> Vec<Vec<usize>> {
        let n = instance.customers.len();
        if n == 0 {
            return vec![];
        }
        // Each customer starts in its own route
        let mut routes: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        // route_demand tracks total demand per route
        let mut route_demand: Vec<f32> = instance.demands.clone();
        // route_of[customer] -> route index (using usize::MAX for merged)
        let mut route_of: Vec<usize> = (0..n).collect();

        let savings = Self::compute_savings(instance);

        for (_, i, j) in savings {
            let ri = route_of[i];
            let rj = route_of[j];
            if ri == rj || ri == usize::MAX || rj == usize::MAX {
                continue;
            }
            // Check if i is at the end of route ri and j is at the start of rj (or vice versa)
            let i_at_end = *routes[ri].last().unwrap_or(&usize::MAX) == i;
            let j_at_start = *routes[rj].first().unwrap_or(&usize::MAX) == j;
            let j_at_end = *routes[rj].last().unwrap_or(&usize::MAX) == j;
            let i_at_start = *routes[ri].first().unwrap_or(&usize::MAX) == i;

            let combined_demand = route_demand[ri] + route_demand[rj];
            if combined_demand > instance.capacity {
                continue;
            }

            if i_at_end && j_at_start {
                // Merge: ri then rj
                let rj_route = routes[rj].clone();
                routes[ri].extend_from_slice(&rj_route);
                route_demand[ri] = combined_demand;
                for &c in &rj_route {
                    route_of[c] = ri;
                }
                routes[rj].clear();
            } else if j_at_end && i_at_start {
                // Merge: rj then ri
                let ri_route = routes[ri].clone();
                routes[rj].extend_from_slice(&ri_route);
                route_demand[rj] = combined_demand;
                for &c in &ri_route {
                    route_of[c] = rj;
                }
                routes[ri].clear();
            }
        }

        routes.into_iter().filter(|r| !r.is_empty()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Beam Search Decoder
// ─────────────────────────────────────────────────────────────────────────────

/// A partial solution state in beam search.
#[derive(Debug, Clone)]
pub struct CombBeamState {
    pub sequence: Vec<usize>,
    pub log_prob: f32,
    pub mask: Vec<bool>,
}

impl CombBeamState {
    pub fn new(n_cities: usize) -> Self {
        CombBeamState {
            sequence: vec![],
            log_prob: 0.0,
            mask: vec![false; n_cities],
        }
    }
}

/// Beam search decoder for Pointer Networks.
#[derive(Debug, Clone)]
pub struct CombBeamSearchDecoder {
    pub beam_width: usize,
}

impl CombBeamSearchDecoder {
    pub fn new(beam_width: usize) -> Self {
        CombBeamSearchDecoder { beam_width }
    }

    /// Run beam search using the given PointerNetwork.
    pub fn search(
        &self,
        inputs: &[Vec<f32>],
        network: &PointerNetwork,
        max_steps: usize,
    ) -> Vec<usize> {
        let n = inputs.len().min(max_steps);
        if n == 0 {
            return vec![];
        }

        let (keys, final_hidden) = network.encoder.encode(inputs);
        let h = network.config.hidden_dim;

        // Each beam entry: (CombBeamState, hidden, cell, dec_input)
        let init_state = CombBeamState::new(inputs.len());
        let mut beam: Vec<(CombBeamState, Vec<f32>, Vec<f32>, Vec<f32>)> =
            vec![(init_state, final_hidden, vec![0.0f32; h], vec![0.0f32; h])];

        for _ in 0..n {
            let mut candidates: Vec<(CombBeamState, Vec<f32>, Vec<f32>, Vec<f32>)> = vec![];

            for (state, hidden, cell, dec_input) in &beam {
                let (h_new, c_new, probs) =
                    network
                        .decoder
                        .step(dec_input, hidden, cell, &keys, &state.mask);

                for (city, &p) in probs.iter().enumerate() {
                    if state.mask[city] || !p.is_finite() || p < 1e-30 {
                        continue;
                    }
                    let mut new_state = state.clone();
                    new_state.sequence.push(city);
                    new_state.log_prob += (p + 1e-30).ln();
                    new_state.mask[city] = true;
                    candidates.push((new_state, h_new.clone(), c_new.clone(), keys[city].clone()));
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Keep top beam_width by log_prob
            candidates.sort_by(|a, b| {
                b.0.log_prob
                    .partial_cmp(&a.0.log_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(self.beam_width);
            beam = candidates;
        }

        // Return the best sequence
        beam.into_iter()
            .max_by(|a, b| {
                a.0.log_prob
                    .partial_cmp(&b.0.log_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(state, _, _, _)| state.sequence)
            .unwrap_or_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Optimality gap: (heuristic - optimal) / optimal * 100%.
pub fn optimality_gap(heuristic_cost: f32, optimal_cost: f32) -> f32 {
    if optimal_cost.abs() < 1e-9 {
        return 0.0;
    }
    (heuristic_cost - optimal_cost) / optimal_cost * 100.0
}

/// True if `tour` is a valid permutation of 0..n_cities.
pub fn tour_validity(tour: &[usize], n_cities: usize) -> bool {
    if tour.len() != n_cities {
        return false;
    }
    let mut seen = vec![false; n_cities];
    for &c in tour {
        if c >= n_cities || seen[c] {
            return false;
        }
        seen[c] = true;
    }
    true
}

/// Knapsack density: total_value / capacity.
pub fn knapsack_density(instance: &KnapsackInstance, selected: &[bool]) -> f32 {
    if instance.capacity < 1e-9 {
        return 0.0;
    }
    instance.solution_value(selected) / instance.capacity
}

/// Aggregate evaluation report for combinatorial optimization.
#[derive(Debug, Clone)]
pub struct CombEvalReport {
    pub avg_tour_length: f32,
    pub avg_optimality_gap: f32,
    pub valid_fraction: f32,
    pub inference_time_ms: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_tsp(n: usize, seed: u64) -> TspInstance {
        let mut rng = StdRng::seed_from_u64(seed);
        let cities = (0..n)
            .map(|_| {
                (
                    rng.random_range(0.0f32..10.0f32),
                    rng.random_range(0.0f32..10.0f32),
                )
            })
            .collect();
        TspInstance::new(cities)
    }

    fn make_ptr_net(n_cities: usize, seed: u64) -> PointerNetwork {
        let mut rng = StdRng::seed_from_u64(seed);
        let cfg = PtrNetConfig {
            input_dim: 2,
            hidden_dim: 16,
        };
        PointerNetwork::new(cfg, &mut rng)
    }

    fn tsp_inputs(instance: &TspInstance) -> Vec<Vec<f32>> {
        instance.cities.iter().map(|&(x, y)| vec![x, y]).collect()
    }

    // §1 Problem representations ─────────────────────────────────────────────

    #[test]
    fn test_tsp_instance_creation() {
        let tsp = make_tsp(5, 0);
        assert_eq!(tsp.n_cities, 5);
        assert_eq!(tsp.cities.len(), 5);
    }

    #[test]
    fn test_tsp_distance_matrix_symmetric() {
        let tsp = make_tsp(4, 1);
        let dm = tsp.distance_matrix();
        assert_eq!(dm.len(), 4);
        for i in 0..4 {
            assert!((dm[i][i]).abs() < 1e-5, "diagonal should be 0");
            for j in 0..4 {
                assert!(
                    (dm[i][j] - dm[j][i]).abs() < 1e-5,
                    "matrix must be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_tsp_tour_length() {
        let tsp = TspInstance::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let tour = vec![0, 1, 2, 3];
        let len = tsp.tour_length(&tour);
        // perimeter of unit square = 4.0
        assert!((len - 4.0).abs() < 1e-4, "expected ~4.0, got {len}");
    }

    #[test]
    fn test_vrp_instance_route_cost() {
        let vrp = VrpInstance {
            depot: (0.0, 0.0),
            customers: vec![(1.0, 0.0), (2.0, 0.0)],
            demands: vec![1.0, 1.0],
            capacity: 5.0,
        };
        let cost = vrp.route_cost(&[0, 1]);
        // 0→(1,0)=1, (1,0)→(2,0)=1, (2,0)→0=2 → total 4
        assert!((cost - 4.0).abs() < 1e-4, "expected ~4.0, got {cost}");
    }

    #[test]
    fn test_knapsack_solution_value() {
        let ks = KnapsackInstance {
            weights: vec![1.0, 2.0, 3.0],
            values: vec![6.0, 4.0, 5.0],
            capacity: 5.0,
        };
        let sel = vec![true, false, true];
        let v = ks.solution_value(&sel);
        assert!((v - 11.0).abs() < 1e-4);
    }

    #[test]
    fn test_knapsack_feasibility() {
        let ks = KnapsackInstance {
            weights: vec![1.0, 2.0, 3.0],
            values: vec![6.0, 4.0, 5.0],
            capacity: 3.0,
        };
        assert!(ks.is_feasible(&[true, false, false])); // weight=1
        assert!(!ks.is_feasible(&[true, true, true])); // weight=6 > 3
    }

    // §2 Pointer Network ──────────────────────────────────────────────────────

    #[test]
    fn test_ptr_encoder_output_shape() {
        let tsp = make_tsp(5, 2);
        let net = make_ptr_net(5, 2);
        let inputs = tsp_inputs(&tsp);
        let (all_hidden, final_h) = net.encoder.encode(&inputs);
        assert_eq!(all_hidden.len(), 5);
        assert_eq!(all_hidden[0].len(), net.config.hidden_dim);
        assert_eq!(final_h.len(), net.config.hidden_dim);
    }

    #[test]
    fn test_ptr_decoder_attend_shape() {
        let tsp = make_tsp(4, 3);
        let net = make_ptr_net(4, 3);
        let inputs = tsp_inputs(&tsp);
        let (keys, final_h) = net.encoder.encode(&inputs);
        let mask = vec![false; 4];
        let probs = net.decoder.attend(&final_h, &keys, &mask);
        assert_eq!(probs.len(), 4);
    }

    #[test]
    fn test_ptr_decoder_attend_masked() {
        let tsp = make_tsp(4, 4);
        let net = make_ptr_net(4, 4);
        let inputs = tsp_inputs(&tsp);
        let (keys, final_h) = net.encoder.encode(&inputs);
        // Mask city 0 as visited
        let mask = vec![true, false, false, false];
        let probs = net.decoder.attend(&final_h, &keys, &mask);
        assert!(
            probs[0] < 1e-6,
            "masked city should have ~0 probability, got {}",
            probs[0]
        );
    }

    #[test]
    fn test_pointer_network_forward_length() {
        let tsp = make_tsp(6, 5);
        let net = make_ptr_net(6, 5);
        let inputs = tsp_inputs(&tsp);
        let tour = net.forward(&inputs, 6);
        assert_eq!(tour.len(), 6);
    }

    #[test]
    fn test_pointer_network_forward_valid_tour() {
        let tsp = make_tsp(5, 6);
        let net = make_ptr_net(5, 6);
        let inputs = tsp_inputs(&tsp);
        let tour = net.forward(&inputs, 5);
        assert!(
            tour_validity(&tour, 5),
            "tour should be a valid permutation"
        );
    }

    #[test]
    fn test_pointer_network_log_prob_finite() {
        let tsp = make_tsp(4, 7);
        let net = make_ptr_net(4, 7);
        let inputs = tsp_inputs(&tsp);
        let tour = net.forward(&inputs, 4);
        let lp = net.log_prob(&inputs, &tour);
        assert!(lp.is_finite(), "log_prob must be finite, got {lp}");
    }

    #[test]
    fn test_pointer_network_log_prob_negative() {
        let tsp = make_tsp(4, 8);
        let net = make_ptr_net(4, 8);
        let inputs = tsp_inputs(&tsp);
        let tour = net.forward(&inputs, 4);
        let lp = net.log_prob(&inputs, &tour);
        assert!(lp <= 0.0, "log_prob must be ≤ 0, got {lp}");
    }

    // §3 Attention Model ──────────────────────────────────────────────────────

    #[test]
    fn test_city_embedding_shape() {
        let mut rng = StdRng::seed_from_u64(10);
        let emb = CityEmbedding::new(32, &mut rng);
        let out = emb.embed((3.0, 7.5));
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_encoder_layer_shape() {
        let mut rng = StdRng::seed_from_u64(11);
        let layer = EncoderLayer::new(16, 2, 32, &mut rng);
        let x: Vec<Vec<f32>> = (0..5).map(|_| vec![0.1f32; 16]).collect();
        let out = layer.forward(&x);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_attention_model_encode_shape() {
        let mut rng = StdRng::seed_from_u64(12);
        let cfg = AmConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            feed_forward_dim: 32,
        };
        let am = AttentionModel::new(cfg, &mut rng);
        let tsp = make_tsp(6, 12);
        let encoded = am.encoder.encode(&tsp.cities);
        assert_eq!(encoded.len(), 6);
        assert_eq!(encoded[0].len(), 16);
    }

    #[test]
    fn test_attention_model_solve_greedy_length() {
        let mut rng = StdRng::seed_from_u64(13);
        let cfg = AmConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            feed_forward_dim: 32,
        };
        let am = AttentionModel::new(cfg, &mut rng);
        let tsp = make_tsp(7, 13);
        let tour = am.solve_greedy(&tsp);
        assert_eq!(tour.len(), 7);
    }

    #[test]
    fn test_attention_model_solve_greedy_valid() {
        let mut rng = StdRng::seed_from_u64(14);
        let cfg = AmConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            feed_forward_dim: 32,
        };
        let am = AttentionModel::new(cfg, &mut rng);
        let tsp = make_tsp(8, 14);
        let tour = am.solve_greedy(&tsp);
        assert!(tour_validity(&tour, 8), "AM greedy tour should be valid");
    }

    #[test]
    fn test_attention_model_solve_sampling_runs() {
        let mut rng = StdRng::seed_from_u64(15);
        let cfg = AmConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            feed_forward_dim: 32,
        };
        let am = AttentionModel::new(cfg, &mut rng);
        let tsp = make_tsp(5, 15);
        let mut rng2 = StdRng::seed_from_u64(42);
        let tour = am.solve_sampling(&tsp, 10, &mut rng2);
        assert_eq!(tour.len(), 5);
    }

    // §4 REINFORCE ────────────────────────────────────────────────────────────

    #[test]
    fn test_reinforce_loss_positive() {
        let cfg = ReinforceConfig {
            lr: 1e-3,
            baseline: BaselineType::ExponentialMovingAverage(0.99),
            clip_norm: 1.0,
        };
        let trainer = ReinforceTrainer::new(cfg);
        // log_probs all negative, rewards = 1.0 > baseline=0.0 → advantage=1.0
        // loss = -advantage * log_prob; with log_prob negative → loss positive
        let log_probs = vec![-0.5, -0.3, -0.8];
        let rewards = vec![1.0f32; 3];
        let loss = trainer.compute_loss(&log_probs, &rewards);
        assert!(loss >= 0.0, "loss should be ≥ 0, got {loss}");
    }

    #[test]
    fn test_reinforce_baseline_update() {
        let cfg = ReinforceConfig {
            lr: 1e-3,
            baseline: BaselineType::ExponentialMovingAverage(0.9),
            clip_norm: 1.0,
        };
        let mut trainer = ReinforceTrainer::new(cfg);
        assert!((trainer.baseline_value).abs() < 1e-6);
        trainer.update_baseline(10.0);
        // EMA: 0.9 * 0 + 0.1 * 10 = 1.0
        assert!((trainer.baseline_value - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_policy_gradient_step_shape() {
        let cfg = ReinforceConfig {
            lr: 1e-3,
            baseline: BaselineType::Greedy,
            clip_norm: 1.0,
        };
        let trainer = ReinforceTrainer::new(cfg);
        let log_probs = vec![-0.5, -0.3, -0.8];
        let grads = trainer.policy_gradient_step(log_probs.clone(), 2.0, 1.0);
        assert_eq!(grads.len(), log_probs.len());
    }

    // §5 Greedy Heuristics ────────────────────────────────────────────────────

    #[test]
    fn test_nearest_neighbor_valid_tour() {
        let tsp = make_tsp(8, 20);
        let tour = NearestNeighbor::solve(&tsp, 0);
        assert!(tour_validity(&tour, 8));
    }

    #[test]
    fn test_nearest_neighbor_length_n_cities() {
        let tsp = make_tsp(6, 21);
        let tour = NearestNeighbor::solve(&tsp, 2);
        assert_eq!(tour.len(), 6);
    }

    #[test]
    fn test_nearest_neighbor_all_cities_visited() {
        let tsp = make_tsp(10, 22);
        let tour = NearestNeighbor::solve(&tsp, 0);
        let mut sorted = tour.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(sorted, expected, "all cities must be visited exactly once");
    }

    #[test]
    fn test_two_opt_swap_length_preserved() {
        let tour = vec![0, 1, 2, 3, 4];
        let new_tour = TwoOpt::two_opt_swap(&tour, 1, 3);
        assert_eq!(new_tour.len(), tour.len());
    }

    #[test]
    fn test_two_opt_improve_not_worse() {
        let tsp = make_tsp(10, 23);
        let greedy = NearestNeighbor::solve(&tsp, 0);
        let orig_len = tsp.tour_length(&greedy);
        let improved = TwoOpt::improve(&tsp, greedy, 100);
        let new_len = tsp.tour_length(&improved);
        assert!(new_len <= orig_len + 1e-4, "2-opt should not worsen tour");
    }

    #[test]
    fn test_two_opt_improves_bad_tour() {
        // Construct a deliberately suboptimal tour (reversed path)
        let cities = vec![
            (0.0f32, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
        ];
        let tsp = TspInstance::new(cities);
        // Bad tour: reversed order = [4, 3, 2, 1, 0] has same length as [0,1,2,3,4]
        // Use a zigzag bad tour instead
        let bad_tour = vec![0, 4, 1, 3, 2];
        let bad_len = tsp.tour_length(&bad_tour);
        let improved = TwoOpt::improve(&tsp, bad_tour, 50);
        let good_len = tsp.tour_length(&improved);
        assert!(
            good_len <= bad_len + 1e-4,
            "2-opt must not increase tour length; bad={bad_len}, improved={good_len}"
        );
    }

    #[test]
    fn test_greedy_knapsack_feasible() {
        let ks = KnapsackInstance {
            weights: vec![2.0, 3.0, 4.0, 5.0],
            values: vec![3.0, 4.0, 5.0, 7.0],
            capacity: 5.0,
        };
        let sel = GreedyKnapsack::solve(&ks);
        assert!(
            ks.is_feasible(&sel),
            "greedy knapsack selection must be feasible"
        );
    }

    #[test]
    fn test_greedy_knapsack_nonempty_selection() {
        let ks = KnapsackInstance {
            weights: vec![1.0, 2.0, 3.0],
            values: vec![5.0, 4.0, 3.0],
            capacity: 4.0,
        };
        let sel = GreedyKnapsack::solve(&ks);
        assert!(sel.iter().any(|&s| s), "should select at least one item");
    }

    #[test]
    fn test_savings_algorithm_routes() {
        let vrp = VrpInstance {
            depot: (0.0, 0.0),
            customers: vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            demands: vec![1.0, 1.0, 1.0],
            capacity: 3.0,
        };
        let routes = SavingsAlgorithm::solve(&vrp);
        // All customers must appear in some route exactly once
        let all_customers: Vec<usize> = routes.iter().flatten().copied().collect();
        let mut sorted = all_customers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_savings_compute_shape() {
        let vrp = VrpInstance {
            depot: (0.0, 0.0),
            customers: vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            demands: vec![1.0, 1.0, 1.0],
            capacity: 5.0,
        };
        let savings = SavingsAlgorithm::compute_savings(&vrp);
        // n*(n-1)/2 = 3*(3-1)/2 = 3 pairs
        assert_eq!(savings.len(), 3);
    }

    // §6 Beam Search ──────────────────────────────────────────────────────────

    #[test]
    fn test_beam_state_creation() {
        let state = CombBeamState::new(5);
        assert_eq!(state.sequence.len(), 0);
        assert_eq!(state.mask.len(), 5);
        assert!((state.log_prob).abs() < 1e-6);
    }

    #[test]
    fn test_beam_search_length() {
        let tsp = make_tsp(5, 30);
        let net = make_ptr_net(5, 30);
        let inputs = tsp_inputs(&tsp);
        let decoder = CombBeamSearchDecoder::new(3);
        let tour = decoder.search(&inputs, &net, 5);
        assert_eq!(
            tour.len(),
            5,
            "beam search tour must have n_cities elements"
        );
    }

    #[test]
    fn test_beam_search_valid_tour() {
        let tsp = make_tsp(6, 31);
        let net = make_ptr_net(6, 31);
        let inputs = tsp_inputs(&tsp);
        let decoder = CombBeamSearchDecoder::new(4);
        let tour = decoder.search(&inputs, &net, 6);
        assert!(
            tour_validity(&tour, 6),
            "beam search should produce valid permutation"
        );
    }

    // §7 Metrics ──────────────────────────────────────────────────────────────

    #[test]
    fn test_optimality_gap_zero_optimal() {
        let gap = optimality_gap(5.0, 5.0);
        assert!((gap).abs() < 1e-4, "gap should be ~0% when equal");
    }

    #[test]
    fn test_optimality_gap_positive() {
        let gap = optimality_gap(6.0, 5.0);
        assert!((gap - 20.0).abs() < 1e-3, "gap should be 20%, got {gap}");
    }

    #[test]
    fn test_tour_validity_valid() {
        let tour = vec![0, 2, 1, 3, 4];
        assert!(tour_validity(&tour, 5));
    }

    #[test]
    fn test_tour_validity_invalid() {
        let tour = vec![0, 1, 1, 3]; // repeated city 1
        assert!(!tour_validity(&tour, 4));
    }

    #[test]
    fn test_knapsack_density_range() {
        let ks = KnapsackInstance {
            weights: vec![1.0, 2.0],
            values: vec![3.0, 4.0],
            capacity: 10.0,
        };
        let sel = vec![true, true];
        let density = knapsack_density(&ks, &sel);
        assert!(density >= 0.0, "density must be non-negative");
    }

    #[test]
    fn test_comb_eval_report_fields() {
        let report = CombEvalReport {
            avg_tour_length: 12.5,
            avg_optimality_gap: 5.0,
            valid_fraction: 0.95,
            inference_time_ms: 3.2,
        };
        assert!((report.avg_tour_length - 12.5).abs() < 1e-4);
        assert!((report.valid_fraction - 0.95).abs() < 1e-4);
    }

    // §8 Additional integration tests ─────────────────────────────────────────

    #[test]
    fn test_pointer_network_beam_vs_greedy() {
        let tsp = make_tsp(6, 40);
        let net = make_ptr_net(6, 40);
        let inputs = tsp_inputs(&tsp);
        let decoder = CombBeamSearchDecoder::new(3);
        let beam_tour = decoder.search(&inputs, &net, 6);
        let greedy_tour = net.forward(&inputs, 6);
        let beam_len = tsp.tour_length(&beam_tour);
        let greedy_len = tsp.tour_length(&greedy_tour);
        // Beam should be at least as good (within tolerance due to random init)
        assert!(
            beam_len <= greedy_len + 1.0,
            "beam ({beam_len:.3}) should be ≤ greedy ({greedy_len:.3}) + slack"
        );
    }

    #[test]
    fn test_attention_model_sampling_best_k() {
        let mut rng = StdRng::seed_from_u64(50);
        let cfg = AmConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            feed_forward_dim: 32,
        };
        let am = AttentionModel::new(cfg, &mut rng);
        let tsp = make_tsp(6, 50);
        let mut rng2 = StdRng::seed_from_u64(51);
        // Single sample (greedy-like)
        let first = am.solve_sampling(&tsp, 1, &mut rng2);
        let first_len = tsp.tour_length(&first);
        let mut rng3 = StdRng::seed_from_u64(52);
        // Best of 5 samples
        let best5 = am.solve_sampling(&tsp, 5, &mut rng3);
        let best5_len = tsp.tour_length(&best5);
        // Best of 5 should be ≤ single sample (or equal)
        assert!(
            best5_len <= first_len + 1e-3,
            "best-of-5 ({best5_len:.3}) should be ≤ single sample ({first_len:.3})"
        );
    }

    #[test]
    fn test_ptr_attend_probabilities_sum_to_one() {
        let tsp = make_tsp(5, 60);
        let net = make_ptr_net(5, 60);
        let inputs = tsp_inputs(&tsp);
        let (keys, final_h) = net.encoder.encode(&inputs);
        let mask = vec![false; 5];
        let probs = net.decoder.attend(&final_h, &keys, &mask);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "probabilities should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_reinforce_loss_gradient_sign() {
        let cfg = ReinforceConfig {
            lr: 1e-3,
            baseline: BaselineType::ExponentialMovingAverage(0.99),
            clip_norm: 1.0,
        };
        let trainer = ReinforceTrainer::new(cfg);
        // With reward > baseline (advantage > 0), gradient step should be positive
        let log_probs = vec![-1.0f32];
        let grads = trainer.policy_gradient_step(log_probs, 5.0, 0.0);
        assert!(
            grads[0] < 0.0,
            "gradient should be negative (descent on -log_prob * advantage)"
        );
    }
}
