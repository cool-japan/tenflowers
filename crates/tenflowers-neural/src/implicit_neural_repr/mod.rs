//! # Implicit Neural Representations (INRs)
//!
//! Coordinate-based neural networks that parameterize continuous signals.
//!
//! ## Components
//!
//! | Struct | Description |
//! |--------|-------------|
//! | [`InrSirenLayer`] | Sinusoidal activation layer (Sitzmann et al. 2020) |
//! | [`InrSirenNetwork`] | Full SIREN MLP |
//! | [`InrFourierFeatureNetwork`] | Random Fourier Features (Tancik et al. 2020) |
//! | [`InrHashGridEncoding`] | Multi-resolution hash encoding (Müller et al. 2022) |
//! | [`InrNeuralSdf`] | Neural Signed Distance Function |
//! | [`InrOccupancyNetwork`] | Occupancy field (Mescheder et al. 2019) |
//! | [`InrImageFitting`] | Image fitting with INR |
//! | [`InrSuperResolution`] | Super-resolution via INR |
//! | [`InrMetaSdf`] | Meta-learning for SDFs |
//! | [`InrMetrics`] | Quality metrics (PSNR, SSIM, Chamfer, Hausdorff, IoU) |
//!
//! All randomness uses `scirs2_core::random` — never `rand`.
//! No `unwrap()` anywhere; all fallible paths return `Result`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller: generate one N(0,1) sample (f64).
#[inline]
fn inr_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Dot product of two slices.
#[inline]
fn inr_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Sigmoid function.
#[inline]
fn inr_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Clamp value to range.
#[inline]
fn inr_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  InrSirenLayer — SIREN sinusoidal activation layer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single SIREN layer.
#[derive(Debug, Clone)]
pub struct InrSirenLayerConfig {
    pub fan_in: usize,
    pub fan_out: usize,
    /// Frequency parameter omega_0 (typically 30.0 for the first layer).
    pub omega_0: f64,
    /// Whether this is the first layer (uses different init).
    pub is_first: bool,
}

/// SIREN layer: `forward(x) = sin(omega_0 * (Wx + b))`.
///
/// Initialization follows Sitzmann et al. 2020:
/// - First layer: W ~ U(-1/fan_in, 1/fan_in)
/// - Hidden layers: W ~ U(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
#[derive(Debug, Clone)]
pub struct InrSirenLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub omega_0: f64,
    pub fan_in: usize,
    pub fan_out: usize,
}

impl InrSirenLayer {
    /// Create a new SIREN layer with proper initialization.
    pub fn new(config: &InrSirenLayerConfig, rng: &mut impl Rng) -> Self {
        let limit = if config.is_first {
            1.0 / config.fan_in as f64
        } else {
            (6.0_f64 / config.fan_in as f64).sqrt() / config.omega_0
        };
        let weights: Vec<Vec<f64>> = (0..config.fan_out)
            .map(|_| {
                (0..config.fan_in)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
                    .collect()
            })
            .collect();
        let biases: Vec<f64> = (0..config.fan_out)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
            .collect();
        Self {
            weights,
            biases,
            omega_0: config.omega_0,
            fan_in: config.fan_in,
            fan_out: config.fan_out,
        }
    }

    /// Forward pass: sin(omega_0 * (Wx + b)).
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(row, &bias)| {
                let pre = inr_dot(row, x) + bias;
                (self.omega_0 * pre).sin()
            })
            .collect()
    }

    /// Linear pre-activation (without sin): omega_0 * (Wx + b).
    pub fn pre_activation(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(row, &bias)| self.omega_0 * (inr_dot(row, x) + bias))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  InrSirenNetwork — Full SIREN MLP
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a SIREN network.
#[derive(Debug, Clone)]
pub struct InrSirenConfig {
    /// Input dimensionality (1=audio, 2=image, 3=volume/SDF).
    pub input_dim: usize,
    /// Hidden layer widths.
    pub hidden_dims: Vec<usize>,
    /// Output dimensionality.
    pub output_dim: usize,
    /// omega_0 for the first layer (default: 30.0).
    pub omega_0_first: f64,
    /// omega_0 for hidden layers (default: 30.0).
    pub omega_0_hidden: f64,
}

impl Default for InrSirenConfig {
    fn default() -> Self {
        Self {
            input_dim: 2,
            hidden_dims: vec![256, 256, 256],
            output_dim: 1,
            omega_0_first: 30.0,
            omega_0_hidden: 30.0,
        }
    }
}

/// Full SIREN MLP: stack of SirenLayers + final linear output layer.
#[derive(Debug, Clone)]
pub struct InrSirenNetwork {
    pub layers: Vec<InrSirenLayer>,
    /// Final linear layer weights (no sin activation).
    pub out_weights: Vec<Vec<f64>>,
    pub out_biases: Vec<f64>,
    pub config: InrSirenConfig,
}

impl InrSirenNetwork {
    /// Construct a new SIREN network.
    pub fn new(config: InrSirenConfig, seed: u64) -> Result<Self> {
        if config.hidden_dims.is_empty() {
            return Err(TensorError::compute_error_simple(
                "SIREN requires at least one hidden layer".to_string(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::new();

        // First layer
        let first_cfg = InrSirenLayerConfig {
            fan_in: config.input_dim,
            fan_out: config.hidden_dims[0],
            omega_0: config.omega_0_first,
            is_first: true,
        };
        layers.push(InrSirenLayer::new(&first_cfg, &mut rng));

        // Hidden layers
        for i in 1..config.hidden_dims.len() {
            let cfg = InrSirenLayerConfig {
                fan_in: config.hidden_dims[i - 1],
                fan_out: config.hidden_dims[i],
                omega_0: config.omega_0_hidden,
                is_first: false,
            };
            layers.push(InrSirenLayer::new(&cfg, &mut rng));
        }

        // Final linear layer (no sin)
        let last_hidden = *config
            .hidden_dims
            .last()
            .ok_or_else(|| TensorError::compute_error_simple("Empty hidden dims".to_string()))?;
        let limit = (6.0_f64 / last_hidden as f64).sqrt() / config.omega_0_hidden;
        let out_weights: Vec<Vec<f64>> = (0..config.output_dim)
            .map(|_| {
                (0..last_hidden)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
                    .collect()
            })
            .collect();
        let out_biases = vec![0.0; config.output_dim];

        Ok(Self {
            layers,
            out_weights,
            out_biases,
            config,
        })
    }

    /// Forward pass: coordinates -> signal values.
    pub fn forward(&self, coordinates: &[f64]) -> Vec<f64> {
        let mut h = coordinates.to_vec();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        // Linear output
        self.out_weights
            .iter()
            .zip(self.out_biases.iter())
            .map(|(row, &bias)| inr_dot(row, &h) + bias)
            .collect()
    }

    /// Compute gradient of the output w.r.t. input via finite differences.
    /// Returns gradient vector of shape [output_dim x input_dim].
    pub fn gradient_fd(&self, coordinates: &[f64], eps: f64) -> Vec<Vec<f64>> {
        let n_in = coordinates.len();
        let base = self.forward(coordinates);
        let n_out = base.len();
        let mut grad = vec![vec![0.0; n_in]; n_out];
        for j in 0..n_in {
            let mut x_plus = coordinates.to_vec();
            let mut x_minus = coordinates.to_vec();
            x_plus[j] += eps;
            x_minus[j] -= eps;
            let f_plus = self.forward(&x_plus);
            let f_minus = self.forward(&x_minus);
            for i in 0..n_out {
                grad[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * eps);
            }
        }
        grad
    }

    /// Eikonal loss: (||grad f(x)|| - 1)^2 averaged over given points.
    pub fn eikonal_loss(&self, points: &[Vec<f64>], eps: f64) -> f64 {
        if points.is_empty() {
            return 0.0;
        }
        let sum: f64 = points
            .iter()
            .map(|pt| {
                let grad = self.gradient_fd(pt, eps);
                // For SDF output (index 0), compute ||grad f||
                let norm_sq: f64 = grad[0].iter().map(|g| g * g).sum();
                let norm = norm_sq.sqrt();
                (norm - 1.0) * (norm - 1.0)
            })
            .sum();
        sum / points.len() as f64
    }

    /// SGD update for all parameters.
    pub fn sgd_update(
        &mut self,
        grads_layers: &[(Vec<Vec<f64>>, Vec<f64>)],
        grads_out: &(Vec<Vec<f64>>, Vec<f64>),
        lr: f64,
    ) {
        for (layer, (gw, gb)) in self.layers.iter_mut().zip(grads_layers.iter()) {
            for (row, grow) in layer.weights.iter_mut().zip(gw.iter()) {
                for (w, g) in row.iter_mut().zip(grow.iter()) {
                    *w -= lr * g;
                }
            }
            for (b, g) in layer.biases.iter_mut().zip(gb.iter()) {
                *b -= lr * g;
            }
        }
        let (ref gw_out, ref gb_out) = grads_out;
        for (row, grow) in self.out_weights.iter_mut().zip(gw_out.iter()) {
            for (w, g) in row.iter_mut().zip(grow.iter()) {
                *w -= lr * g;
            }
        }
        for (b, g) in self.out_biases.iter_mut().zip(gb_out.iter()) {
            *b -= lr * g;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  InrFourierFeatureNetwork — Random Fourier Features (Tancik et al. 2020)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Fourier Feature encoding.
#[derive(Debug, Clone)]
pub struct InrFourierConfig {
    pub input_dim: usize,
    /// Number of Fourier basis functions.
    pub n_frequencies: usize,
    /// Gaussian bandwidth sigma.
    pub sigma: f64,
    /// Hidden layer widths for the MLP on top.
    pub hidden_dims: Vec<usize>,
    /// Output dimensionality.
    pub output_dim: usize,
}

impl Default for InrFourierConfig {
    fn default() -> Self {
        Self {
            input_dim: 2,
            n_frequencies: 256,
            sigma: 10.0,
            hidden_dims: vec![256, 256],
            output_dim: 3,
        }
    }
}

/// Random Fourier Feature network: gamma(x) = [cos(2*pi*B*x), sin(2*pi*B*x)]
/// followed by an MLP.
#[derive(Debug, Clone)]
pub struct InrFourierFeatureNetwork {
    /// B matrix: [n_frequencies x input_dim], sampled ~ N(0, sigma^2).
    pub b_matrix: Vec<Vec<f64>>,
    /// MLP weights\[layer\]\[out\]\[in\].
    pub mlp_weights: Vec<Vec<Vec<f64>>>,
    /// MLP biases\[layer\]\[out\].
    pub mlp_biases: Vec<Vec<f64>>,
    pub config: InrFourierConfig,
}

impl InrFourierFeatureNetwork {
    /// Construct a new Fourier Feature network.
    pub fn new(config: InrFourierConfig, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);

        // B matrix ~ N(0, sigma^2)
        let b_matrix: Vec<Vec<f64>> = (0..config.n_frequencies)
            .map(|_| {
                (0..config.input_dim)
                    .map(|_| inr_normal(&mut rng) * config.sigma)
                    .collect()
            })
            .collect();

        // Encoded dim = 2 * n_frequencies (cos + sin)
        let encoded_dim = 2 * config.n_frequencies;

        // Build MLP
        let mut dims = vec![encoded_dim];
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.output_dim);

        let mut mlp_weights = Vec::new();
        let mut mlp_biases = Vec::new();
        for i in 0..dims.len() - 1 {
            let fan_in = dims[i];
            let fan_out = dims[i + 1];
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..fan_out)
                .map(|_| {
                    (0..fan_in)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
                        .collect()
                })
                .collect();
            mlp_weights.push(w);
            mlp_biases.push(vec![0.0; fan_out]);
        }

        Ok(Self {
            b_matrix,
            mlp_weights,
            mlp_biases,
            config,
        })
    }

    /// Encode coordinates using Fourier features: gamma(x) = [cos(2*pi*Bx), sin(2*pi*Bx)].
    pub fn encode(&self, coordinates: &[f64]) -> Vec<f64> {
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut encoded = Vec::with_capacity(2 * self.config.n_frequencies);
        for row in &self.b_matrix {
            let proj = inr_dot(row, coordinates);
            encoded.push((two_pi * proj).cos());
            encoded.push((two_pi * proj).sin());
        }
        encoded
    }

    /// Forward pass: encode then MLP.
    pub fn forward(&self, coordinates: &[f64]) -> Vec<f64> {
        let mut h = self.encode(coordinates);
        let n_layers = self.mlp_weights.len();
        for (l, (w, b)) in self
            .mlp_weights
            .iter()
            .zip(self.mlp_biases.iter())
            .enumerate()
        {
            let pre: Vec<f64> = w
                .iter()
                .zip(b.iter())
                .map(|(row, &bias)| inr_dot(row, &h) + bias)
                .collect();
            h = if l < n_layers - 1 {
                pre.into_iter().map(|v| v.max(0.0)).collect() // ReLU
            } else {
                pre // Linear output
            };
        }
        h
    }

    /// SGD update for MLP parameters.
    pub fn sgd_update_mlp(&mut self, grads_w: &[Vec<Vec<f64>>], grads_b: &[Vec<f64>], lr: f64) {
        for (l, (gw, gb)) in grads_w.iter().zip(grads_b.iter()).enumerate() {
            if l < self.mlp_weights.len() {
                for (row, grow) in self.mlp_weights[l].iter_mut().zip(gw.iter()) {
                    for (w, g) in row.iter_mut().zip(grow.iter()) {
                        *w -= lr * g;
                    }
                }
                for (b, g) in self.mlp_biases[l].iter_mut().zip(gb.iter()) {
                    *b -= lr * g;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  InrHashGridEncoding — Multi-resolution hash encoding (Instant NGP)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for hash grid encoding.
#[derive(Debug, Clone)]
pub struct InrHashGridConfig {
    /// Number of resolution levels L.
    pub n_levels: usize,
    /// Features per level F.
    pub n_features_per_level: usize,
    /// log2 of hash table size T per level.
    pub log2_hashmap_size: usize,
    /// Base (coarsest) resolution.
    pub base_resolution: usize,
    /// Per-level growth factor b.
    pub per_level_scale: f64,
    /// Input spatial dimensionality (2 or 3).
    pub input_dim: usize,
}

impl Default for InrHashGridConfig {
    fn default() -> Self {
        Self {
            n_levels: 16,
            n_features_per_level: 2,
            log2_hashmap_size: 19,
            base_resolution: 16,
            per_level_scale: 1.38,
            input_dim: 3,
        }
    }
}

/// Multi-resolution hash grid encoding (Mueller et al. 2022).
///
/// Each level l has resolution `N_l = base_resolution * b^l` and a hash table
/// of size T = 2^log2_hashmap_size. Vertices are hashed with:
/// `h(x) = (x1 XOR x2*pi2 XOR x3*pi3) mod T`
/// and features are trilinearly interpolated.
#[derive(Debug, Clone)]
pub struct InrHashGridEncoding {
    /// Hash tables: \[level\]\[entry\]\[feature\].
    pub tables: Vec<Vec<Vec<f64>>>,
    /// Resolutions per level.
    pub resolutions: Vec<usize>,
    pub config: InrHashGridConfig,
}

/// Large primes for spatial hashing (Teschner et al.).
const HASH_PRIME_1: u64 = 1;
const HASH_PRIME_2: u64 = 2_654_435_761;
const HASH_PRIME_3: u64 = 805_459_861;

impl InrHashGridEncoding {
    /// Construct with randomly initialized feature tables.
    pub fn new(config: InrHashGridConfig, seed: u64) -> Result<Self> {
        if config.n_levels == 0 || config.n_features_per_level == 0 {
            return Err(TensorError::compute_error_simple(
                "Hash grid needs n_levels > 0 and n_features_per_level > 0".to_string(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let table_size = 1usize << config.log2_hashmap_size;

        let mut resolutions = Vec::with_capacity(config.n_levels);
        let mut tables = Vec::with_capacity(config.n_levels);

        for l in 0..config.n_levels {
            let res = (config.base_resolution as f64 * config.per_level_scale.powi(l as i32)).ceil()
                as usize;
            resolutions.push(res);

            // Initialize features with small uniform noise
            let entries: Vec<Vec<f64>> = (0..table_size)
                .map(|_| {
                    (0..config.n_features_per_level)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * 1e-4)
                        .collect()
                })
                .collect();
            tables.push(entries);
        }

        Ok(Self {
            tables,
            resolutions,
            config,
        })
    }

    /// Hash function: (x1 XOR x2*pi2 XOR x3*pi3) mod T.
    fn hash_vertex(&self, indices: &[usize]) -> usize {
        let table_size = 1usize << self.config.log2_hashmap_size;
        let primes = [HASH_PRIME_1, HASH_PRIME_2, HASH_PRIME_3];
        let mut h: u64 = 0;
        for (i, &idx) in indices.iter().enumerate() {
            let p = if i < primes.len() {
                primes[i]
            } else {
                primes[i % primes.len()]
            };
            h ^= (idx as u64).wrapping_mul(p);
        }
        (h as usize) % table_size
    }

    /// Encode a single coordinate (values in [0, 1]^d).
    pub fn encode(&self, coordinates: &[f64]) -> Vec<f64> {
        let d = self.config.input_dim.min(coordinates.len());
        let total_features = self.config.n_levels * self.config.n_features_per_level;
        let mut result = Vec::with_capacity(total_features);

        for (l, res) in self.resolutions.iter().enumerate() {
            let res_f = *res as f64;

            // Scaled coordinates
            let scaled: Vec<f64> = (0..d)
                .map(|i| inr_clamp(coordinates[i], 0.0, 1.0) * res_f)
                .collect();

            // Floor indices and interpolation weights
            let floor_idx: Vec<usize> = scaled.iter().map(|&s| s.floor() as usize).collect();
            let weights: Vec<f64> = scaled.iter().map(|&s| s - s.floor()).collect();

            // Enumerate 2^d vertices of the grid cell
            let n_vertices = 1usize << d;
            let mut interpolated = vec![0.0; self.config.n_features_per_level];

            for v in 0..n_vertices {
                let mut vertex = Vec::with_capacity(d);
                let mut w = 1.0;
                for dim in 0..d {
                    let bit = (v >> dim) & 1;
                    vertex.push(floor_idx[dim] + bit);
                    w *= if bit == 1 {
                        weights[dim]
                    } else {
                        1.0 - weights[dim]
                    };
                }
                let hash_idx = self.hash_vertex(&vertex);
                let features = &self.tables[l][hash_idx];
                for (f, &feat) in interpolated.iter_mut().zip(features.iter()) {
                    *f += w * feat;
                }
            }

            result.extend_from_slice(&interpolated);
        }

        result
    }

    /// Total encoded feature dimension.
    pub fn output_dim(&self) -> usize {
        self.config.n_levels * self.config.n_features_per_level
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  InrNeuralSdf — Neural Signed Distance Function
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Neural SDF.
#[derive(Debug, Clone)]
pub struct InrNeuralSdfConfig {
    pub siren_config: InrSirenConfig,
    /// Epsilon for finite-difference gradient estimation.
    pub gradient_eps: f64,
    /// Maximum steps for sphere tracing.
    pub max_trace_steps: usize,
    /// Convergence threshold for sphere tracing.
    pub trace_threshold: f64,
    /// Maximum trace distance.
    pub max_trace_dist: f64,
}

impl Default for InrNeuralSdfConfig {
    fn default() -> Self {
        Self {
            siren_config: InrSirenConfig {
                input_dim: 3,
                hidden_dims: vec![256, 256, 256],
                output_dim: 1,
                omega_0_first: 30.0,
                omega_0_hidden: 30.0,
            },
            gradient_eps: 1e-4,
            max_trace_steps: 128,
            trace_threshold: 1e-4,
            max_trace_dist: 10.0,
        }
    }
}

/// Result of sphere tracing.
#[derive(Debug, Clone)]
pub struct InrSphereTraceResult {
    /// Whether a surface was hit.
    pub hit: bool,
    /// Hit point (or final point if no hit).
    pub point: Vec<f64>,
    /// SDF value at the final point.
    pub sdf_value: f64,
    /// Number of steps taken.
    pub steps: usize,
    /// Total distance traveled.
    pub distance: f64,
}

/// Neural Signed Distance Function.
///
/// Uses a SIREN network to represent f: R^3 -> R where f(x) is the signed
/// distance to the nearest surface. Supports Eikonal regularization
/// (||grad f|| = 1) and sphere tracing for rendering.
#[derive(Debug, Clone)]
pub struct InrNeuralSdf {
    pub network: InrSirenNetwork,
    pub config: InrNeuralSdfConfig,
}

impl InrNeuralSdf {
    /// Create a new Neural SDF.
    pub fn new(config: InrNeuralSdfConfig, seed: u64) -> Result<Self> {
        let network = InrSirenNetwork::new(config.siren_config.clone(), seed)?;
        Ok(Self { network, config })
    }

    /// Compute the signed distance value at a point.
    pub fn compute_sdf(&self, point: &[f64]) -> f64 {
        let out = self.network.forward(point);
        out[0]
    }

    /// Compute the SDF gradient at a point via central finite differences.
    pub fn sdf_gradient(&self, point: &[f64]) -> Vec<f64> {
        let eps = self.config.gradient_eps;
        let n = point.len();
        let mut grad = Vec::with_capacity(n);
        for i in 0..n {
            let mut p_plus = point.to_vec();
            let mut p_minus = point.to_vec();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let f_plus = self.compute_sdf(&p_plus);
            let f_minus = self.compute_sdf(&p_minus);
            grad.push((f_plus - f_minus) / (2.0 * eps));
        }
        grad
    }

    /// Eikonal loss over a set of points: average (||grad f|| - 1)^2.
    pub fn eikonal_loss(&self, points: &[Vec<f64>]) -> f64 {
        self.network.eikonal_loss(points, self.config.gradient_eps)
    }

    /// Sphere tracing (ray marching): march along a ray until |f(p)| < eps.
    pub fn sphere_trace(&self, ray_origin: &[f64], ray_direction: &[f64]) -> InrSphereTraceResult {
        let max_steps = self.config.max_trace_steps;
        let threshold = self.config.trace_threshold;
        let max_dist = self.config.max_trace_dist;

        // Normalize direction
        let dir_norm: f64 = ray_direction.iter().map(|d| d * d).sum::<f64>().sqrt();
        let dir: Vec<f64> = if dir_norm > 1e-12 {
            ray_direction.iter().map(|d| d / dir_norm).collect()
        } else {
            ray_direction.to_vec()
        };

        let mut t = 0.0;

        for step in 0..max_steps {
            let point: Vec<f64> = ray_origin
                .iter()
                .zip(dir.iter())
                .map(|(&o, &d)| o + t * d)
                .collect();
            let sdf = self.compute_sdf(&point);

            if sdf.abs() < threshold {
                return InrSphereTraceResult {
                    hit: true,
                    point,
                    sdf_value: sdf,
                    steps: step + 1,
                    distance: t,
                };
            }

            // March forward by |sdf|
            t += sdf.abs();
            if t > max_dist {
                break;
            }
        }

        let final_point: Vec<f64> = ray_origin
            .iter()
            .zip(dir.iter())
            .map(|(&o, &d)| o + t * d)
            .collect();
        let final_sdf = self.compute_sdf(&final_point);
        InrSphereTraceResult {
            hit: false,
            point: final_point,
            sdf_value: final_sdf,
            steps: max_steps,
            distance: t,
        }
    }

    /// Train the SDF on point samples with SDF values.
    /// Returns final MSE + eikonal loss.
    pub fn train(
        &mut self,
        points: &[Vec<f64>],
        sdf_values: &[f64],
        eikonal_points: &[Vec<f64>],
        epochs: usize,
        lr: f64,
        eikonal_weight: f64,
    ) -> Result<f64> {
        if points.len() != sdf_values.len() {
            return Err(TensorError::compute_error_simple(
                "Points and SDF values must have same length".to_string(),
            ));
        }
        let eps = self.config.gradient_eps;
        let mut final_loss = 0.0;

        for _epoch in 0..epochs {
            // Compute data loss: MSE
            let data_loss = self.compute_data_loss(points, sdf_values);

            // Compute eikonal loss
            let eik_loss = self.network.eikonal_loss(eikonal_points, eps);

            final_loss = data_loss + eikonal_weight * eik_loss;

            // Numerical gradient for output weights (simplified SGD)
            let delta = 1e-5;
            for i in 0..self.network.out_weights.len() {
                for j in 0..self.network.out_weights[i].len() {
                    self.network.out_weights[i][j] += delta;
                    let loss_plus = self.compute_data_loss(points, sdf_values);
                    self.network.out_weights[i][j] -= 2.0 * delta;
                    let loss_minus = self.compute_data_loss(points, sdf_values);
                    self.network.out_weights[i][j] += delta; // restore
                    let grad = (loss_plus - loss_minus) / (2.0 * delta);
                    self.network.out_weights[i][j] -= lr * grad;
                }
            }
            for i in 0..self.network.out_biases.len() {
                self.network.out_biases[i] += delta;
                let loss_plus = self.compute_data_loss(points, sdf_values);
                self.network.out_biases[i] -= 2.0 * delta;
                let loss_minus = self.compute_data_loss(points, sdf_values);
                self.network.out_biases[i] += delta;
                let grad = (loss_plus - loss_minus) / (2.0 * delta);
                self.network.out_biases[i] -= lr * grad;
            }
        }

        Ok(final_loss)
    }

    /// Internal: compute MSE data loss.
    fn compute_data_loss(&self, points: &[Vec<f64>], sdf_values: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (pt, &target) in points.iter().zip(sdf_values.iter()) {
            let pred = self.compute_sdf(pt);
            loss += (pred - target) * (pred - target);
        }
        loss / points.len().max(1) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  InrOccupancyNetwork — Occupancy field (Mescheder et al. 2019)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Occupancy Network.
#[derive(Debug, Clone)]
pub struct InrOccupancyConfig {
    /// Point dimensionality (usually 3).
    pub point_dim: usize,
    /// Latent code dimension z.
    pub latent_dim: usize,
    /// Encoder hidden dims.
    pub encoder_hidden: Vec<usize>,
    /// Decoder hidden dims.
    pub decoder_hidden: Vec<usize>,
}

impl Default for InrOccupancyConfig {
    fn default() -> Self {
        Self {
            point_dim: 3,
            latent_dim: 128,
            encoder_hidden: vec![256, 256],
            decoder_hidden: vec![256, 256],
        }
    }
}

/// Simple MLP for occupancy network sub-modules.
#[derive(Debug, Clone)]
struct InrMlp {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    n_layers: usize,
}

impl InrMlp {
    fn new(dims: &[usize], rng: &mut impl Rng) -> Self {
        let n_layers = dims.len().saturating_sub(1);
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let fan_in = dims[l];
            let fan_out = dims[l + 1];
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..fan_out)
                .map(|_| {
                    (0..fan_in)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
                        .collect()
                })
                .collect();
            weights.push(w);
            biases.push(vec![0.0; fan_out]);
        }
        Self {
            weights,
            biases,
            n_layers,
        }
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut h = x.to_vec();
        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let pre: Vec<f64> = w
                .iter()
                .zip(b.iter())
                .map(|(row, &bias)| inr_dot(row, &h) + bias)
                .collect();
            h = if l < self.n_layers - 1 {
                pre.into_iter().map(|v| v.max(0.0)).collect()
            } else {
                pre
            };
        }
        h
    }
}

/// Occupancy Network: f(x, z) -> \[0,1\] occupancy probability.
///
/// - Encoder: point cloud -> mean-pooled latent z
/// - Decoder: (point || z) -> sigmoid -> occupancy
#[derive(Debug, Clone)]
pub struct InrOccupancyNetwork {
    encoder: InrMlp,
    decoder: InrMlp,
    pub config: InrOccupancyConfig,
}

impl InrOccupancyNetwork {
    /// Create a new occupancy network.
    pub fn new(config: InrOccupancyConfig, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);

        // Encoder: point_dim -> hidden -> latent_dim
        let mut enc_dims = vec![config.point_dim];
        enc_dims.extend_from_slice(&config.encoder_hidden);
        enc_dims.push(config.latent_dim);
        let encoder = InrMlp::new(&enc_dims, &mut rng);

        // Decoder: (point_dim + latent_dim) -> hidden -> 1
        let mut dec_dims = vec![config.point_dim + config.latent_dim];
        dec_dims.extend_from_slice(&config.decoder_hidden);
        dec_dims.push(1);
        let decoder = InrMlp::new(&dec_dims, &mut rng);

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    /// Encode a point cloud into a latent code z by mean pooling.
    pub fn encode(&self, point_cloud: &[Vec<f64>]) -> Result<Vec<f64>> {
        if point_cloud.is_empty() {
            return Err(TensorError::compute_error_simple(
                "Empty point cloud".to_string(),
            ));
        }
        let n = point_cloud.len() as f64;
        let mut z_sum = vec![0.0; self.config.latent_dim];
        for pt in point_cloud {
            let z_i = self.encoder.forward(pt);
            for (s, &z) in z_sum.iter_mut().zip(z_i.iter()) {
                *s += z;
            }
        }
        for s in &mut z_sum {
            *s /= n;
        }
        Ok(z_sum)
    }

    /// Decode: given a query point and latent z, return occupancy probability.
    pub fn decode(&self, point: &[f64], z: &[f64]) -> f64 {
        let mut input = point.to_vec();
        input.extend_from_slice(z);
        let logit = self.decoder.forward(&input);
        inr_sigmoid(logit[0])
    }

    /// Binary cross-entropy loss for occupancy.
    pub fn bce_loss(&self, query_points: &[Vec<f64>], occupancy_labels: &[f64], z: &[f64]) -> f64 {
        if query_points.is_empty() {
            return 0.0;
        }
        let mut loss = 0.0;
        for (pt, &label) in query_points.iter().zip(occupancy_labels.iter()) {
            let p = self.decode(pt, z);
            let p_clamped = inr_clamp(p, 1e-7, 1.0 - 1e-7);
            loss -= label * p_clamped.ln() + (1.0 - label) * (1.0 - p_clamped).ln();
        }
        loss / query_points.len() as f64
    }

    /// Simple marching cubes: extract iso-surface at threshold 0.5.
    /// Returns a list of vertices that are near the 0.5 iso-surface.
    pub fn marching_cubes_simple(
        &self,
        z: &[f64],
        grid_res: usize,
        bounds_min: &[f64; 3],
        bounds_max: &[f64; 3],
    ) -> Vec<[f64; 3]> {
        let mut surface_pts = Vec::new();
        let threshold = 0.5;

        for ix in 0..grid_res {
            for iy in 0..grid_res {
                for iz in 0..grid_res {
                    let x = bounds_min[0]
                        + (ix as f64 + 0.5) / grid_res as f64 * (bounds_max[0] - bounds_min[0]);
                    let y = bounds_min[1]
                        + (iy as f64 + 0.5) / grid_res as f64 * (bounds_max[1] - bounds_min[1]);
                    let z_coord = bounds_min[2]
                        + (iz as f64 + 0.5) / grid_res as f64 * (bounds_max[2] - bounds_min[2]);

                    let occ = self.decode(&[x, y, z_coord], z);
                    if (occ - threshold).abs() < 0.1 {
                        surface_pts.push([x, y, z_coord]);
                    }
                }
            }
        }
        surface_pts
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  InrImageFitting — Image fitting with INR
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for image fitting.
#[derive(Debug, Clone)]
pub struct InrImageFitConfig {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Number of channels (e.g. 3 for RGB).
    pub channels: usize,
    /// Use SIREN (true) or Fourier Features (false).
    pub use_siren: bool,
    /// Hidden dims for the network.
    pub hidden_dims: Vec<usize>,
    /// Omega_0 for SIREN.
    pub omega_0: f64,
    /// Sigma for Fourier features.
    pub fourier_sigma: f64,
    /// Number of Fourier frequencies.
    pub n_frequencies: usize,
}

impl Default for InrImageFitConfig {
    fn default() -> Self {
        Self {
            width: 32,
            height: 32,
            channels: 3,
            use_siren: true,
            hidden_dims: vec![256, 256, 256],
            omega_0: 30.0,
            fourier_sigma: 10.0,
            n_frequencies: 256,
        }
    }
}

/// Image fitting with implicit neural representations.
///
/// Maps (x, y) coordinates -> (r, g, b) values.
#[derive(Debug, Clone)]
pub enum InrImageFitting {
    Siren(InrSirenNetwork),
    Fourier(InrFourierFeatureNetwork),
}

impl InrImageFitting {
    /// Create a new image fitter.
    pub fn new(config: &InrImageFitConfig, seed: u64) -> Result<Self> {
        if config.use_siren {
            let siren_cfg = InrSirenConfig {
                input_dim: 2,
                hidden_dims: config.hidden_dims.clone(),
                output_dim: config.channels,
                omega_0_first: config.omega_0,
                omega_0_hidden: config.omega_0,
            };
            let net = InrSirenNetwork::new(siren_cfg, seed)?;
            Ok(InrImageFitting::Siren(net))
        } else {
            let fourier_cfg = InrFourierConfig {
                input_dim: 2,
                n_frequencies: config.n_frequencies,
                sigma: config.fourier_sigma,
                hidden_dims: config.hidden_dims.clone(),
                output_dim: config.channels,
            };
            let net = InrFourierFeatureNetwork::new(fourier_cfg, seed)?;
            Ok(InrImageFitting::Fourier(net))
        }
    }

    /// Forward: (x, y) -> (r, g, b).
    pub fn predict(&self, x: f64, y: f64) -> Vec<f64> {
        match self {
            InrImageFitting::Siren(net) => net.forward(&[x, y]),
            InrImageFitting::Fourier(net) => net.forward(&[x, y]),
        }
    }

    /// Generate normalized coordinates for an image grid.
    pub fn make_coords(width: usize, height: usize) -> Vec<[f64; 2]> {
        let mut coords = Vec::with_capacity(width * height);
        for iy in 0..height {
            for ix in 0..width {
                let x = (ix as f64 + 0.5) / width as f64;
                let y = (iy as f64 + 0.5) / height as f64;
                coords.push([x, y]);
            }
        }
        coords
    }

    /// Fit to image pixels. Returns final MSE.
    ///
    /// `pixels`: flat array of [R, G, B, R, G, B, ...] in row-major order, values in \[0,1\].
    /// `coords`: normalized (x, y) for each pixel.
    pub fn fit_image(
        &mut self,
        pixels: &[f64],
        coords: &[[f64; 2]],
        epochs: usize,
        lr: f64,
    ) -> Result<f64> {
        let n_channels = match self {
            InrImageFitting::Siren(net) => net.config.output_dim,
            InrImageFitting::Fourier(net) => net.config.output_dim,
        };
        if pixels.len() != coords.len() * n_channels {
            return Err(TensorError::compute_error_simple(format!(
                "Pixel count {} != coords {} * channels {}",
                pixels.len(),
                coords.len(),
                n_channels
            )));
        }

        let delta = 1e-5;
        let mut final_mse = 0.0;

        for _epoch in 0..epochs {
            // Compute MSE
            final_mse = self.compute_mse_internal(coords, pixels, n_channels);

            // Numerical gradient SGD on output layer weights only (tractable)
            match self {
                InrImageFitting::Siren(ref mut net) => {
                    for i in 0..net.out_weights.len() {
                        for j in 0..net.out_weights[i].len() {
                            net.out_weights[i][j] += delta;
                            let loss_p = Self::compute_mse_siren(net, coords, pixels, n_channels);
                            net.out_weights[i][j] -= 2.0 * delta;
                            let loss_m = Self::compute_mse_siren(net, coords, pixels, n_channels);
                            net.out_weights[i][j] += delta;
                            let grad = (loss_p - loss_m) / (2.0 * delta);
                            net.out_weights[i][j] -= lr * grad;
                        }
                    }
                    for i in 0..net.out_biases.len() {
                        net.out_biases[i] += delta;
                        let loss_p = Self::compute_mse_siren(net, coords, pixels, n_channels);
                        net.out_biases[i] -= 2.0 * delta;
                        let loss_m = Self::compute_mse_siren(net, coords, pixels, n_channels);
                        net.out_biases[i] += delta;
                        let grad = (loss_p - loss_m) / (2.0 * delta);
                        net.out_biases[i] -= lr * grad;
                    }
                }
                InrImageFitting::Fourier(ref mut net) => {
                    let last = net.mlp_weights.len() - 1;
                    for i in 0..net.mlp_weights[last].len() {
                        for j in 0..net.mlp_weights[last][i].len() {
                            net.mlp_weights[last][i][j] += delta;
                            let loss_p = Self::compute_mse_fourier(net, coords, pixels, n_channels);
                            net.mlp_weights[last][i][j] -= 2.0 * delta;
                            let loss_m = Self::compute_mse_fourier(net, coords, pixels, n_channels);
                            net.mlp_weights[last][i][j] += delta;
                            let grad = (loss_p - loss_m) / (2.0 * delta);
                            net.mlp_weights[last][i][j] -= lr * grad;
                        }
                    }
                    for i in 0..net.mlp_biases[last].len() {
                        net.mlp_biases[last][i] += delta;
                        let loss_p = Self::compute_mse_fourier(net, coords, pixels, n_channels);
                        net.mlp_biases[last][i] -= 2.0 * delta;
                        let loss_m = Self::compute_mse_fourier(net, coords, pixels, n_channels);
                        net.mlp_biases[last][i] += delta;
                        let grad = (loss_p - loss_m) / (2.0 * delta);
                        net.mlp_biases[last][i] -= lr * grad;
                    }
                }
            }
        }

        Ok(final_mse)
    }

    fn compute_mse_internal(&self, coords: &[[f64; 2]], pixels: &[f64], n_ch: usize) -> f64 {
        match self {
            InrImageFitting::Siren(net) => Self::compute_mse_siren(net, coords, pixels, n_ch),
            InrImageFitting::Fourier(net) => Self::compute_mse_fourier(net, coords, pixels, n_ch),
        }
    }

    fn compute_mse_siren(
        net: &InrSirenNetwork,
        coords: &[[f64; 2]],
        pixels: &[f64],
        n_ch: usize,
    ) -> f64 {
        let mut mse = 0.0;
        for (i, coord) in coords.iter().enumerate() {
            let pred = net.forward(&[coord[0], coord[1]]);
            for c in 0..n_ch {
                let d = pred[c] - pixels[i * n_ch + c];
                mse += d * d;
            }
        }
        mse / (coords.len() * n_ch) as f64
    }

    fn compute_mse_fourier(
        net: &InrFourierFeatureNetwork,
        coords: &[[f64; 2]],
        pixels: &[f64],
        n_ch: usize,
    ) -> f64 {
        let mut mse = 0.0;
        for (i, coord) in coords.iter().enumerate() {
            let pred = net.forward(&[coord[0], coord[1]]);
            for c in 0..n_ch {
                let d = pred[c] - pixels[i * n_ch + c];
                mse += d * d;
            }
        }
        mse / (coords.len() * n_ch) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  InrSuperResolution — Super-resolution via INR
// ─────────────────────────────────────────────────────────────────────────────

/// Super-resolution via implicit neural representations.
///
/// Train on low-res coordinates -> pixel values, then query at high-res
/// coordinates for arbitrary upscaling.
#[derive(Debug, Clone)]
pub struct InrSuperResolution {
    pub fitter: InrImageFitting,
    pub scale_factor: f64,
}

impl InrSuperResolution {
    /// Create a super-resolution model.
    pub fn new(config: &InrImageFitConfig, scale_factor: f64, seed: u64) -> Result<Self> {
        let fitter = InrImageFitting::new(config, seed)?;
        Ok(Self {
            fitter,
            scale_factor,
        })
    }

    /// Train on low-resolution image.
    pub fn train_low_res(
        &mut self,
        pixels: &[f64],
        width: usize,
        height: usize,
        epochs: usize,
        lr: f64,
    ) -> Result<f64> {
        let coords = InrImageFitting::make_coords(width, height);
        self.fitter.fit_image(pixels, &coords, epochs, lr)
    }

    /// Upscale: generate high-res coordinates and query the INR.
    pub fn upscale(&self, width: usize, height: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
        let hr_w = (width as f64 * self.scale_factor).ceil() as usize;
        let hr_h = (height as f64 * self.scale_factor).ceil() as usize;
        let coords = InrImageFitting::make_coords(hr_w, hr_h);
        let n_channels = match &self.fitter {
            InrImageFitting::Siren(net) => net.config.output_dim,
            InrImageFitting::Fourier(net) => net.config.output_dim,
        };
        let mut pixels = Vec::with_capacity(coords.len() * n_channels);
        for coord in &coords {
            let pred = self.fitter.predict(coord[0], coord[1]);
            pixels.extend_from_slice(&pred);
        }
        (coords, pixels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  InrMetaSdf — Meta-learning for SDFs (MAML-style)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for MetaSDF.
#[derive(Debug, Clone)]
pub struct InrMetaSdfConfig {
    pub sdf_config: InrNeuralSdfConfig,
    /// Number of inner-loop gradient steps.
    pub inner_steps: usize,
    /// Inner-loop learning rate.
    pub inner_lr: f64,
    /// Outer (meta) learning rate.
    pub meta_lr: f64,
}

impl Default for InrMetaSdfConfig {
    fn default() -> Self {
        Self {
            sdf_config: InrNeuralSdfConfig::default(),
            inner_steps: 5,
            inner_lr: 0.01,
            meta_lr: 0.001,
        }
    }
}

/// A single SDF task: point samples + SDF values.
#[derive(Debug, Clone)]
pub struct InrSdfTask {
    pub points: Vec<Vec<f64>>,
    pub sdf_values: Vec<f64>,
}

/// Meta-learning for SDFs (MAML-style).
///
/// Learns a good initialization for fast SDF fitting.
#[derive(Debug, Clone)]
pub struct InrMetaSdf {
    pub base_sdf: InrNeuralSdf,
    pub config: InrMetaSdfConfig,
}

impl InrMetaSdf {
    /// Create a new MetaSDF.
    pub fn new(config: InrMetaSdfConfig, seed: u64) -> Result<Self> {
        let base_sdf = InrNeuralSdf::new(config.sdf_config.clone(), seed)?;
        Ok(Self { base_sdf, config })
    }

    /// Meta-train on a set of SDF tasks.
    /// Returns average post-adaptation loss.
    pub fn meta_train(&mut self, tasks: &[InrSdfTask], epochs: usize) -> Result<f64> {
        if tasks.is_empty() {
            return Err(TensorError::compute_error_simple(
                "No tasks for meta-training".to_string(),
            ));
        }

        let mut avg_loss = 0.0;

        for _epoch in 0..epochs {
            let mut total_loss = 0.0;

            for task in tasks {
                // Clone the base network for inner-loop adaptation
                let mut adapted = self.base_sdf.clone();

                // Inner loop: few gradient steps
                for _step in 0..self.config.inner_steps {
                    let delta = 1e-5;
                    for i in 0..adapted.network.out_weights.len() {
                        for j in 0..adapted.network.out_weights[i].len() {
                            adapted.network.out_weights[i][j] += delta;
                            let l_p = adapted.compute_data_loss(&task.points, &task.sdf_values);
                            adapted.network.out_weights[i][j] -= 2.0 * delta;
                            let l_m = adapted.compute_data_loss(&task.points, &task.sdf_values);
                            adapted.network.out_weights[i][j] += delta;
                            let grad = (l_p - l_m) / (2.0 * delta);
                            adapted.network.out_weights[i][j] -= self.config.inner_lr * grad;
                        }
                    }
                }

                let task_loss = adapted.compute_data_loss(&task.points, &task.sdf_values);
                total_loss += task_loss;
            }

            avg_loss = total_loss / tasks.len() as f64;

            // Outer loop: update base network output weights
            let delta = 1e-5;
            for i in 0..self.base_sdf.network.out_weights.len() {
                for j in 0..self.base_sdf.network.out_weights[i].len() {
                    self.base_sdf.network.out_weights[i][j] += delta;
                    let mut l_p = 0.0;
                    for task in tasks {
                        l_p += self
                            .base_sdf
                            .compute_data_loss(&task.points, &task.sdf_values);
                    }
                    self.base_sdf.network.out_weights[i][j] -= 2.0 * delta;
                    let mut l_m = 0.0;
                    for task in tasks {
                        l_m += self
                            .base_sdf
                            .compute_data_loss(&task.points, &task.sdf_values);
                    }
                    self.base_sdf.network.out_weights[i][j] += delta;
                    let grad = (l_p - l_m) / (2.0 * delta * tasks.len() as f64);
                    self.base_sdf.network.out_weights[i][j] -= self.config.meta_lr * grad;
                }
            }
        }

        Ok(avg_loss)
    }

    /// Adapt the meta-initialized SDF to a new shape.
    pub fn adapt(&self, task: &InrSdfTask, n_steps: usize) -> Result<InrNeuralSdf> {
        let mut adapted = self.base_sdf.clone();
        let delta = 1e-5;

        for _step in 0..n_steps {
            for i in 0..adapted.network.out_weights.len() {
                for j in 0..adapted.network.out_weights[i].len() {
                    adapted.network.out_weights[i][j] += delta;
                    let l_p = adapted.compute_data_loss(&task.points, &task.sdf_values);
                    adapted.network.out_weights[i][j] -= 2.0 * delta;
                    let l_m = adapted.compute_data_loss(&task.points, &task.sdf_values);
                    adapted.network.out_weights[i][j] += delta;
                    let grad = (l_p - l_m) / (2.0 * delta);
                    adapted.network.out_weights[i][j] -= self.config.inner_lr * grad;
                }
            }
        }

        Ok(adapted)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  InrMetrics — Quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// INR quality metrics.
#[derive(Debug, Clone)]
pub struct InrMetrics;

impl InrMetrics {
    /// Peak Signal-to-Noise Ratio (dB).
    /// `max_val` is the maximum pixel value (e.g., 1.0 for normalized).
    pub fn psnr(predicted: &[f64], target: &[f64], max_val: f64) -> Result<f64> {
        if predicted.len() != target.len() || predicted.is_empty() {
            return Err(TensorError::compute_error_simple(
                "PSNR: arrays must be non-empty and same length".to_string(),
            ));
        }
        let mse: f64 = predicted
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f64>()
            / predicted.len() as f64;
        if mse < 1e-15 {
            return Ok(100.0); // Essentially identical
        }
        Ok(10.0 * (max_val * max_val / mse).log10())
    }

    /// Structural Similarity Index (SSIM) for 2D images.
    ///
    /// Simplified per-pixel-block version using a sliding window.
    pub fn ssim(
        predicted: &[f64],
        target: &[f64],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<f64> {
        let n = width * height * channels;
        if predicted.len() != n || target.len() != n {
            return Err(TensorError::compute_error_simple(
                "SSIM: dimension mismatch".to_string(),
            ));
        }

        let c1 = 0.01_f64 * 0.01;
        let c2 = 0.03_f64 * 0.03;
        let window = 7usize;
        let half = window / 2;

        let mut ssim_sum = 0.0;
        let mut count = 0.0;

        for ch in 0..channels {
            for y in half..height.saturating_sub(half) {
                for x in half..width.saturating_sub(half) {
                    let mut mu_p = 0.0;
                    let mut mu_t = 0.0;
                    let mut w_count = 0.0;

                    for dy in 0..window {
                        for dx in 0..window {
                            let py = y + dy - half;
                            let px = x + dx - half;
                            if py < height && px < width {
                                let idx = (py * width + px) * channels + ch;
                                mu_p += predicted[idx];
                                mu_t += target[idx];
                                w_count += 1.0;
                            }
                        }
                    }
                    if w_count < 1.0 {
                        continue;
                    }
                    mu_p /= w_count;
                    mu_t /= w_count;

                    let mut var_p = 0.0;
                    let mut var_t = 0.0;
                    let mut cov = 0.0;
                    for dy in 0..window {
                        for dx in 0..window {
                            let py = y + dy - half;
                            let px = x + dx - half;
                            if py < height && px < width {
                                let idx = (py * width + px) * channels + ch;
                                let p = predicted[idx];
                                let t = target[idx];
                                var_p += (p - mu_p) * (p - mu_p);
                                var_t += (t - mu_t) * (t - mu_t);
                                cov += (p - mu_p) * (t - mu_t);
                            }
                        }
                    }
                    var_p /= w_count;
                    var_t /= w_count;
                    cov /= w_count;

                    let numerator = (2.0 * mu_p * mu_t + c1) * (2.0 * cov + c2);
                    let denominator = (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2);
                    ssim_sum += numerator / denominator;
                    count += 1.0;
                }
            }
        }

        if count < 1.0 {
            return Ok(1.0);
        }
        Ok(ssim_sum / count)
    }

    /// Chamfer distance between two 3D point sets.
    pub fn chamfer_distance(set_a: &[Vec<f64>], set_b: &[Vec<f64>]) -> Result<f64> {
        if set_a.is_empty() || set_b.is_empty() {
            return Err(TensorError::compute_error_simple(
                "Chamfer: point sets must be non-empty".to_string(),
            ));
        }

        // A -> B
        let mut sum_a = 0.0;
        for a in set_a {
            let min_d = set_b
                .iter()
                .map(|b| {
                    a.iter()
                        .zip(b.iter())
                        .map(|(ai, bi)| (ai - bi) * (ai - bi))
                        .sum::<f64>()
                })
                .fold(f64::MAX, f64::min);
            sum_a += min_d;
        }

        // B -> A
        let mut sum_b = 0.0;
        for b in set_b {
            let min_d = set_a
                .iter()
                .map(|a| {
                    a.iter()
                        .zip(b.iter())
                        .map(|(ai, bi)| (ai - bi) * (ai - bi))
                        .sum::<f64>()
                })
                .fold(f64::MAX, f64::min);
            sum_b += min_d;
        }

        Ok(sum_a / set_a.len() as f64 + sum_b / set_b.len() as f64)
    }

    /// Hausdorff distance between two 3D point sets.
    pub fn hausdorff_distance(set_a: &[Vec<f64>], set_b: &[Vec<f64>]) -> Result<f64> {
        if set_a.is_empty() || set_b.is_empty() {
            return Err(TensorError::compute_error_simple(
                "Hausdorff: point sets must be non-empty".to_string(),
            ));
        }

        let directed_ab = set_a
            .iter()
            .map(|a| {
                set_b
                    .iter()
                    .map(|b| {
                        a.iter()
                            .zip(b.iter())
                            .map(|(ai, bi)| (ai - bi) * (ai - bi))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .fold(f64::MAX, f64::min)
            })
            .fold(0.0_f64, f64::max);

        let directed_ba = set_b
            .iter()
            .map(|b| {
                set_a
                    .iter()
                    .map(|a| {
                        a.iter()
                            .zip(b.iter())
                            .map(|(ai, bi)| (ai - bi) * (ai - bi))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .fold(f64::MAX, f64::min)
            })
            .fold(0.0_f64, f64::max);

        Ok(directed_ab.max(directed_ba))
    }

    /// Intersection over Union (IoU) for binary occupancy predictions.
    pub fn iou(predicted: &[bool], target: &[bool]) -> Result<f64> {
        if predicted.len() != target.len() || predicted.is_empty() {
            return Err(TensorError::compute_error_simple(
                "IoU: arrays must be non-empty and same length".to_string(),
            ));
        }
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (p, t) in predicted.iter().zip(target.iter()) {
            if *p && *t {
                intersection += 1;
            }
            if *p || *t {
                union += 1;
            }
        }
        if union == 0 {
            return Ok(1.0); // Both empty
        }
        Ok(intersection as f64 / union as f64)
    }
}

/// Evaluation report for INR quality.
#[derive(Debug, Clone)]
pub struct InrReport {
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub chamfer: Option<f64>,
    pub hausdorff: Option<f64>,
    pub iou: Option<f64>,
}

impl InrReport {
    pub fn new() -> Self {
        Self {
            psnr: None,
            ssim: None,
            chamfer: None,
            hausdorff: None,
            iou: None,
        }
    }
}

impl Default for InrReport {
    fn default() -> Self {
        Self::new()
    }
}
