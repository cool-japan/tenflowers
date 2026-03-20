//! Operator Learning & Neural Scientific Computing — Track OL.
//! FNO, DeepONet, Symbolic Regression, Hamiltonian/Lagrangian NNs, Score Matching.

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

/// Xavier/Glorot uniform init for a `[fan_in × fan_out]` weight matrix.
pub(crate) fn xavier_init_f64(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    let n = fan_in * fan_out;
    (0..n)
        .map(|_| {
            let u: f64 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

/// Xavier/Glorot uniform init returning f32 weights.
pub(crate) fn xavier_init_f32(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let n = fan_in * fan_out;
    (0..n)
        .map(|_| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

/// GeLU activation: `x * Φ(x)` approximated via tanh.
#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// ReLU activation.
#[inline]
pub(crate) fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Dense layer forward: y = W·x + b  (f32).
pub(crate) fn dense_forward_f32(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let in_dim = input.len();
    let mut out = bias.to_vec();
    for j in 0..out_dim {
        let row_offset = j * in_dim;
        let mut acc = 0.0_f32;
        for i in 0..in_dim {
            acc += weights[row_offset + i] * input[i];
        }
        out[j] += acc;
    }
    out
}

/// Dense layer forward: y = W·x + b  (f64).
pub(crate) fn dense_forward_f64(
    input: &[f64],
    weights: &[f64],
    bias: &[f64],
    out_dim: usize,
) -> Vec<f64> {
    let in_dim = input.len();
    let mut out = bias.to_vec();
    for j in 0..out_dim {
        let row_offset = j * in_dim;
        let mut acc = 0.0_f64;
        for i in 0..in_dim {
            acc += weights[row_offset + i] * input[i];
        }
        out[j] += acc;
    }
    out
}

/// Box-Muller: one N(0,1) sample.
#[inline]
pub(crate) fn sample_normal_bm(rng: &mut StdRng) -> f32 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos()) as f32
}

/// Compute the (unnormalised) 1-D DFT of a real-valued signal.
pub(crate) fn dft_real(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mut re = vec![0.0_f32; n];
    let mut im = vec![0.0_f32; n];
    let two_pi_over_n = std::f64::consts::TAU / n as f64;
    for k in 0..n {
        let mut r = 0.0_f64;
        let mut i = 0.0_f64;
        for t in 0..n {
            let angle = two_pi_over_n * (k * t) as f64;
            r += x[t] as f64 * angle.cos();
            i -= x[t] as f64 * angle.sin();
        }
        re[k] = r as f32;
        im[k] = i as f32;
    }
    (re, im)
}

/// Compute the 1-D IDFT (inverse, unnormalised × 1/N).
pub(crate) fn idft_real(re: &[f32], im: &[f32]) -> Vec<f32> {
    let n = re.len();
    let two_pi_over_n = std::f64::consts::TAU / n as f64;
    let inv_n = 1.0 / n as f64;
    (0..n)
        .map(|t| {
            let mut sum = 0.0_f64;
            for k in 0..n {
                let angle = two_pi_over_n * (k * t) as f64;
                sum += re[k] as f64 * angle.cos() - im[k] as f64 * angle.sin();
            }
            (sum * inv_n) as f32
        })
        .collect()
}

/// 1-D spectral convolution with learnable complex Fourier weights.
#[derive(Debug, Clone)]
pub struct SpectralConv1d {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Number of Fourier modes to retain.
    pub modes: usize,
    /// Real part of complex weight tensor.
    pub w_re: Vec<f32>,
    /// Imaginary part of complex weight tensor.
    pub w_im: Vec<f32>,
}

impl SpectralConv1d {
    /// Construct with Xavier-initialised complex weights.
    pub fn new(in_channels: usize, out_channels: usize, modes: usize, seed: u64) -> Result<Self> {
        if modes == 0 {
            return Err(TensorError::invalid_argument_op(
                "SpectralConv1d::new",
                "modes must be > 0",
            ));
        }
        let fan = in_channels * modes;
        let w_re = xavier_init_f32(fan, out_channels, seed);
        let w_im = xavier_init_f32(fan, out_channels, seed.wrapping_add(1));
        Ok(Self {
            in_channels,
            out_channels,
            modes,
            w_re,
            w_im,
        })
    }

    /// Forward pass.
    pub fn forward(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let ic = self.in_channels;
        let oc = self.out_channels;
        let modes = self.modes;

        if input.len() != ic * n {
            return Err(TensorError::invalid_argument_op(
                "SpectralConv1d::forward",
                &format!(
                    "input length {} != in_channels({}) * n({})",
                    input.len(),
                    ic,
                    n
                ),
            ));
        }
        if modes > n / 2 + 1 {
            return Err(TensorError::invalid_argument_op(
                "SpectralConv1d::forward",
                "modes must be <= n/2+1",
            ));
        }

        let mut in_re = vec![vec![0.0_f32; n]; ic];
        let mut in_im = vec![vec![0.0_f32; n]; ic];
        for c in 0..ic {
            let x_c: Vec<f32> = (0..n).map(|i| input[c * n + i]).collect();
            let (r, im) = dft_real(&x_c);
            in_re[c] = r;
            in_im[c] = im;
        }

        let mut out_re = vec![vec![0.0_f32; n]; oc];
        let mut out_im = vec![vec![0.0_f32; n]; oc];

        for o in 0..oc {
            for m in 0..modes {
                let mut acc_re = 0.0_f32;
                let mut acc_im = 0.0_f32;
                for c in 0..ic {
                    let wi = (o * ic + c) * modes + m;
                    let wr = self.w_re[wi];
                    let wimg = self.w_im[wi];
                    let xr = in_re[c][m];
                    let xi = in_im[c][m];
                    acc_re += wr * xr - wimg * xi;
                    acc_im += wr * xi + wimg * xr;
                }
                out_re[o][m] = acc_re;
                out_im[o][m] = acc_im;
            }
        }

        let mut result = vec![0.0_f32; oc * n];
        for o in 0..oc {
            let spatial = idft_real(&out_re[o], &out_im[o]);
            for i in 0..n {
                result[o * n + i] = spatial[i];
            }
        }
        Ok(result)
    }
}

/// 2-D spectral convolution (separable DFT per row then per column).
#[derive(Debug, Clone)]
pub struct SpectralConv2d {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Fourier modes along x dimension.
    pub modes_x: usize,
    /// Fourier modes along y dimension.
    pub modes_y: usize,
    /// Real part of complex weight tensor.
    pub w_re: Vec<f32>,
    /// Imaginary part of complex weight tensor.
    pub w_im: Vec<f32>,
}

impl SpectralConv2d {
    /// Construct with Xavier-initialised complex weights.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        modes_x: usize,
        modes_y: usize,
        seed: u64,
    ) -> Result<Self> {
        if modes_x == 0 || modes_y == 0 {
            return Err(TensorError::invalid_argument_op(
                "SpectralConv2d::new",
                "modes_x and modes_y must be > 0",
            ));
        }
        let fan = in_channels * modes_x * modes_y;
        let w_re = xavier_init_f32(fan, out_channels, seed);
        let w_im = xavier_init_f32(fan, out_channels, seed.wrapping_add(1));
        Ok(Self {
            in_channels,
            out_channels,
            modes_x,
            modes_y,
            w_re,
            w_im,
        })
    }

    /// Forward pass — separable 2-D DFT via row/column 1-D DFT.
    pub fn forward(&self, input: &[f32], nx: usize, ny: usize) -> Result<Vec<f32>> {
        let ic = self.in_channels;
        let oc = self.out_channels;
        let mx = self.modes_x;
        let my = self.modes_y;

        if input.len() != ic * nx * ny {
            return Err(TensorError::invalid_argument_op(
                "SpectralConv2d::forward",
                "input length mismatch",
            ));
        }

        let spectrum_size = ic * nx * ny;
        let mut sp_re = vec![0.0_f32; spectrum_size];
        let mut sp_im = vec![0.0_f32; spectrum_size];

        for c in 0..ic {
            for row in 0..nx {
                let x_row: Vec<f32> = (0..ny)
                    .map(|col| input[c * nx * ny + row * ny + col])
                    .collect();
                let (rr, ri) = dft_real(&x_row);
                for col in 0..ny {
                    sp_re[c * nx * ny + row * ny + col] = rr[col];
                    sp_im[c * nx * ny + row * ny + col] = ri[col];
                }
            }
            for col in 0..ny {
                let x_col_re: Vec<f32> = (0..nx)
                    .map(|row| sp_re[c * nx * ny + row * ny + col])
                    .collect();
                let x_col_im: Vec<f32> = (0..nx)
                    .map(|row| sp_im[c * nx * ny + row * ny + col])
                    .collect();
                let n = nx;
                let two_pi_over_n = std::f64::consts::TAU / n as f64;
                for k in 0..n {
                    let mut r = 0.0_f64;
                    let mut im = 0.0_f64;
                    for t in 0..n {
                        let angle = two_pi_over_n * (k * t) as f64;
                        let cos_a = angle.cos() as f32;
                        let sin_a = angle.sin() as f32;
                        r += (x_col_re[t] * cos_a + x_col_im[t] * sin_a) as f64;
                        im += (x_col_im[t] * cos_a - x_col_re[t] * sin_a) as f64;
                    }
                    sp_re[c * nx * ny + k * ny + col] = r as f32;
                    sp_im[c * nx * ny + k * ny + col] = im as f32;
                }
            }
        }

        let out_sp_size = oc * nx * ny;
        let mut out_re = vec![0.0_f32; out_sp_size];
        let mut out_im = vec![0.0_f32; out_sp_size];

        let capped_mx = mx.min(nx);
        let capped_my = my.min(ny);

        for o in 0..oc {
            for kx in 0..capped_mx {
                for ky in 0..capped_my {
                    let mut acc_re = 0.0_f32;
                    let mut acc_im = 0.0_f32;
                    for c in 0..ic {
                        let wi = ((o * ic + c) * mx + kx) * my + ky;
                        let wr = self.w_re[wi];
                        let wimg = self.w_im[wi];
                        let xr = sp_re[c * nx * ny + kx * ny + ky];
                        let xi = sp_im[c * nx * ny + kx * ny + ky];
                        acc_re += wr * xr - wimg * xi;
                        acc_im += wr * xi + wimg * xr;
                    }
                    out_re[o * nx * ny + kx * ny + ky] = acc_re;
                    out_im[o * nx * ny + kx * ny + ky] = acc_im;
                }
            }
        }

        let mut result = vec![0.0_f32; oc * nx * ny];
        for o in 0..oc {
            for col in 0..ny {
                let re_col: Vec<f32> = (0..nx)
                    .map(|row| out_re[o * nx * ny + row * ny + col])
                    .collect();
                let im_col: Vec<f32> = (0..nx)
                    .map(|row| out_im[o * nx * ny + row * ny + col])
                    .collect();
                let two_pi_over_n = std::f64::consts::TAU / nx as f64;
                let inv_n = 1.0 / nx as f64;
                for t in 0..nx {
                    let mut r = 0.0_f64;
                    for k in 0..nx {
                        let angle = two_pi_over_n * (k * t) as f64;
                        r += re_col[k] as f64 * angle.cos() - im_col[k] as f64 * angle.sin();
                    }
                    out_re[o * nx * ny + t * ny + col] = (r * inv_n) as f32;
                }
            }
            for row in 0..nx {
                let re_row: Vec<f32> = (0..ny)
                    .map(|col| out_re[o * nx * ny + row * ny + col])
                    .collect();
                let im_row: Vec<f32> = (0..ny)
                    .map(|col| out_im[o * nx * ny + row * ny + col])
                    .collect();
                let spatial = idft_real(&re_row, &im_row);
                for col in 0..ny {
                    result[o * nx * ny + row * ny + col] = spatial[col];
                }
            }
        }
        Ok(result)
    }
}

/// One spectral layer of an FNO: `SpectralConv1d` + bypass linear + GeLU.
#[derive(Debug, Clone)]
pub struct FnoLayer {
    /// Spectral convolution component.
    pub spectral: SpectralConv1d,
    /// Bypass (residual) linear weight matrix.
    pub bypass_w: Vec<f32>,
    /// Bypass bias vector.
    pub bypass_b: Vec<f32>,
    /// Number of channels.
    pub channels: usize,
}

impl FnoLayer {
    /// Construct an FNO layer with `channels` width and `modes` spectral modes.
    pub fn new(channels: usize, modes: usize, seed: u64) -> Result<Self> {
        let spectral = SpectralConv1d::new(channels, channels, modes, seed)?;
        let bypass_w = xavier_init_f32(channels, channels, seed.wrapping_add(10));
        let bypass_b = vec![0.0_f32; channels];
        Ok(Self {
            spectral,
            bypass_w,
            bypass_b,
            channels,
        })
    }

    /// Forward: `output = GeLU(spectral(x, n) + bypass(x))`.
    pub fn forward(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let c = self.channels;
        let spectral_out = self.spectral.forward(input, n)?;

        let mut bypass_out = vec![0.0_f32; c * n];
        for pos in 0..n {
            let x_slice: Vec<f32> = (0..c).map(|ch| input[ch * n + pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.bypass_w, &self.bypass_b, c);
            for ch in 0..c {
                bypass_out[ch * n + pos] = y[ch];
            }
        }

        let mut out = vec![0.0_f32; c * n];
        for i in 0..c * n {
            out[i] = gelu(spectral_out[i] + bypass_out[i]);
        }
        Ok(out)
    }
}

/// Configuration for `FnoModel`.
#[derive(Debug, Clone)]
pub struct FnoConfig {
    /// Input dimension (function space dimension).
    pub input_dim: usize,
    /// Number of channels in the FNO.
    pub channels: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Number of FNO blocks.
    pub num_blocks: usize,
    /// Number of Fourier modes to retain.
    pub modes: usize,
}

impl FnoConfig {
    /// Create a default FnoConfig.
    pub fn new(
        input_dim: usize,
        channels: usize,
        output_dim: usize,
        num_blocks: usize,
        modes: usize,
    ) -> Self {
        Self {
            input_dim,
            channels,
            output_dim,
            num_blocks,
            modes,
        }
    }
}

/// One FnoBlock = FnoLayer (already includes bypass + GeLU).
pub type FnoBlock = FnoLayer;

/// Full Fourier Neural Operator model.
#[derive(Debug, Clone)]
pub struct FnoModel {
    /// Configuration for this FNO model.
    pub config: FnoConfig,
    /// Lifting layer weight matrix (input_dim → channels).
    pub lift_w: Vec<f32>,
    /// Lifting layer bias.
    pub lift_b: Vec<f32>,
    /// Stacked FNO blocks.
    pub blocks: Vec<FnoBlock>,
    /// Projection layer weight matrix (channels → output_dim).
    pub proj_w: Vec<f32>,
    /// Projection layer bias.
    pub proj_b: Vec<f32>,
}

impl FnoModel {
    /// Construct an FnoModel.
    pub fn new(config: FnoConfig, seed: u64) -> Result<Self> {
        let lift_w = xavier_init_f32(config.input_dim, config.channels, seed);
        let lift_b = vec![0.0_f32; config.channels];

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for i in 0..config.num_blocks {
            blocks.push(FnoLayer::new(
                config.channels,
                config.modes,
                seed.wrapping_add(100 + i as u64),
            )?);
        }

        let proj_w =
            xavier_init_f32(config.channels, config.output_dim, seed.wrapping_add(200));
        let proj_b = vec![0.0_f32; config.output_dim];

        Ok(Self {
            config,
            lift_w,
            lift_b,
            blocks,
            proj_w,
            proj_b,
        })
    }

    /// Forward pass.
    pub fn forward(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let id = self.config.input_dim;
        let ch = self.config.channels;
        let od = self.config.output_dim;

        if input.len() != id * n {
            return Err(TensorError::invalid_argument_op(
                "FnoModel::forward",
                &format!(
                    "input length {} != input_dim({}) * n({})",
                    input.len(),
                    id,
                    n
                ),
            ));
        }

        let mut lifted = vec![0.0_f32; ch * n];
        for pos in 0..n {
            let x_slice: Vec<f32> = (0..id).map(|d| input[d * n + pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.lift_w, &self.lift_b, ch);
            for c in 0..ch {
                lifted[c * n + pos] = y[c];
            }
        }

        let mut h = lifted;
        for block in &self.blocks {
            h = block.forward(&h, n)?;
        }

        let mut out = vec![0.0_f32; od * n];
        for pos in 0..n {
            let x_slice: Vec<f32> = (0..ch).map(|c| h[c * n + pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.proj_w, &self.proj_b, od);
            for d in 0..od {
                out[d * n + pos] = y[d];
            }
        }
        Ok(out)
    }
}

/// A compact MLP with configurable hidden layers and ReLU activations.
#[derive(Debug, Clone)]
pub struct Mlp {
    /// Weight matrices for each layer.
    pub weights: Vec<Vec<f32>>,
    /// Bias vectors for each layer.
    pub biases: Vec<Vec<f32>>,
    /// Output dimensions for each layer.
    pub dims: Vec<usize>,
    /// Input dimension.
    pub input_dim: usize,
}

impl Mlp {
    /// Create an MLP with `layer_dims = [in, h1, h2, ..., out]`.
    pub fn new(layer_dims: &[usize], seed: u64) -> Result<Self> {
        if layer_dims.len() < 2 {
            return Err(TensorError::invalid_argument_op(
                "Mlp::new",
                "layer_dims must have at least 2 elements (in, out)",
            ));
        }
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut dims = Vec::new();
        let input_dim = layer_dims[0];

        for i in 0..layer_dims.len() - 1 {
            let fan_in = layer_dims[i];
            let fan_out = layer_dims[i + 1];
            weights.push(xavier_init_f32(
                fan_in,
                fan_out,
                seed.wrapping_add(i as u64 * 7),
            ));
            biases.push(vec![0.0_f32; fan_out]);
            dims.push(fan_out);
        }
        Ok(Self {
            weights,
            biases,
            dims,
            input_dim,
        })
    }

    /// Forward pass with ReLU on hidden layers (no activation on final layer).
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "Mlp::forward",
                &format!("input len {} != input_dim {}", input.len(), self.input_dim),
            ));
        }
        let mut h = input.to_vec();
        let n_layers = self.dims.len();
        for (l, (&out_dim, (w, b))) in self
            .dims
            .iter()
            .zip(self.weights.iter().zip(self.biases.iter()))
            .enumerate()
        {
            h = dense_forward_f32(&h, w, b, out_dim);
            if l < n_layers - 1 {
                for v in h.iter_mut() {
                    *v = relu(*v);
                }
            }
        }
        Ok(h)
    }

    /// Output dimension.
    pub fn output_dim(&self) -> usize {
        *self.dims.last().unwrap_or(&self.input_dim)
    }
}

/// Branch network: encodes input function values at fixed sensor points.
#[derive(Debug, Clone)]
pub struct BranchNet {
    /// Inner MLP.
    pub mlp: Mlp,
    /// Number of sensor points.
    pub n_sensors: usize,
    /// Latent dimension `p`.
    pub p: usize,
}

impl BranchNet {
    /// Construct with `n_sensors` input, `hidden_dims` hidden layers, `p` output.
    pub fn new(n_sensors: usize, hidden_dims: &[usize], p: usize, seed: u64) -> Result<Self> {
        let mut dims = vec![n_sensors];
        dims.extend_from_slice(hidden_dims);
        dims.push(p);
        let mlp = Mlp::new(&dims, seed)?;
        Ok(Self { mlp, n_sensors, p })
    }

    /// Encode sensor values to latent `b ∈ ℝ^p`.
    pub fn encode(&self, sensor_values: &[f32]) -> Result<Vec<f32>> {
        if sensor_values.len() != self.n_sensors {
            return Err(TensorError::invalid_argument_op(
                "BranchNet::encode",
                &format!(
                    "expected {} sensor values, got {}",
                    self.n_sensors,
                    sensor_values.len()
                ),
            ));
        }
        self.mlp.forward(sensor_values)
    }
}

/// Trunk network: encodes query locations (coordinates) → basis functions.
#[derive(Debug, Clone)]
pub struct TrunkNet {
    /// Inner MLP.
    pub mlp: Mlp,
    /// Dimension of query location.
    pub loc_dim: usize,
    /// Latent dimension `p`.
    pub p: usize,
}

impl TrunkNet {
    /// Construct with `loc_dim` input, `hidden_dims` hidden, `p` output.
    pub fn new(loc_dim: usize, hidden_dims: &[usize], p: usize, seed: u64) -> Result<Self> {
        let mut dims = vec![loc_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(p);
        let mlp = Mlp::new(&dims, seed)?;
        Ok(Self { mlp, loc_dim, p })
    }

    /// Encode query location to basis vector.
    pub fn encode(&self, location: &[f32]) -> Result<Vec<f32>> {
        if location.len() != self.loc_dim {
            return Err(TensorError::invalid_argument_op(
                "TrunkNet::encode",
                &format!("expected loc_dim={}, got {}", self.loc_dim, location.len()),
            ));
        }
        self.mlp.forward(location)
    }
}

/// DeepONet — output = Σ_i b_i · t_i + bias.
#[derive(Debug, Clone)]
pub struct DeepONet {
    /// Branch network for encoding input functions.
    pub branch: BranchNet,
    /// Trunk network for encoding query locations.
    pub trunk: TrunkNet,
    /// Scalar output bias.
    pub output_bias: f32,
}

impl DeepONet {
    /// Create a DeepONet.
    pub fn new(branch: BranchNet, trunk: TrunkNet) -> Result<Self> {
        if branch.p != trunk.p {
            return Err(TensorError::invalid_argument_op(
                "DeepONet::new",
                &format!("branch.p ({}) != trunk.p ({})", branch.p, trunk.p),
            ));
        }
        Ok(Self {
            branch,
            trunk,
            output_bias: 0.0,
        })
    }

    /// Evaluate operator at one query location.
    pub fn forward(&self, sensor_values: &[f32], query_loc: &[f32]) -> Result<f32> {
        let b = self.branch.encode(sensor_values)?;
        let t = self.trunk.encode(query_loc)?;
        let dot: f32 = b.iter().zip(t.iter()).map(|(bi, ti)| bi * ti).sum();
        Ok(dot + self.output_bias)
    }

    /// Evaluate at multiple query locations (batch of queries, same sensor values).
    pub fn forward_batch(&self, sensor_values: &[f32], queries: &[Vec<f32>]) -> Result<Vec<f32>> {
        let b = self.branch.encode(sensor_values)?;
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            let t = self.trunk.encode(q)?;
            let dot: f32 = b.iter().zip(t.iter()).map(|(bi, ti)| bi * ti).sum();
            out.push(dot + self.output_bias);
        }
        Ok(out)
    }
}

/// MSE-based trainer for DeepONet.
#[derive(Debug, Clone)]
pub struct DeepONetTrainer {
    /// Learning rate.
    pub lr: f32,
}

impl DeepONetTrainer {
    /// Create a trainer.
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Compute MSE loss over a batch of (sensor_values, queries, targets).
    pub fn mse_loss(
        &self,
        model: &DeepONet,
        sensors_batch: &[Vec<f32>],
        queries_batch: &[Vec<Vec<f32>>],
        targets_batch: &[Vec<f32>],
    ) -> Result<f32> {
        if sensors_batch.len() != queries_batch.len()
            || sensors_batch.len() != targets_batch.len()
        {
            return Err(TensorError::invalid_argument_op(
                "DeepONetTrainer::mse_loss",
                "batch sizes do not match",
            ));
        }
        let mut total_loss = 0.0_f32;
        let mut n_total = 0_usize;
        for ((sensors, queries), targets) in sensors_batch
            .iter()
            .zip(queries_batch.iter())
            .zip(targets_batch.iter())
        {
            for (q, &tgt) in queries.iter().zip(targets.iter()) {
                let pred = model.forward(sensors, q)?;
                total_loss += (pred - tgt) * (pred - tgt);
                n_total += 1;
            }
        }
        if n_total == 0 {
            return Err(TensorError::invalid_argument_op(
                "DeepONetTrainer::mse_loss",
                "empty batch",
            ));
        }
        Ok(total_loss / n_total as f32)
    }
}

/// DeepONet variant with random sensor placement support.
#[derive(Debug, Clone)]
pub struct ContinuousSensorDeepONet {
    /// Underlying DeepONet model.
    pub model: DeepONet,
    /// Number of sensor points.
    pub n_sensors: usize,
    /// Domain bounds `[lo, hi]`.
    pub domain: [f32; 2],
}

impl ContinuousSensorDeepONet {
    /// Create with random sensor support.
    pub fn new(model: DeepONet, n_sensors: usize, domain_lo: f32, domain_hi: f32) -> Self {
        Self {
            model,
            n_sensors,
            domain: [domain_lo, domain_hi],
        }
    }

    /// Sample `n_sensors` sensor locations uniformly in `[domain_lo, domain_hi]`.
    pub fn sample_sensors(&self, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let lo = self.domain[0];
        let hi = self.domain[1];
        (0..self.n_sensors)
            .map(|_| {
                let u: f32 = rng.random();
                lo + u * (hi - lo)
            })
            .collect()
    }

    /// Evaluate at query locations given arbitrary sensor locations and values.
    pub fn forward(&self, sensor_values: &[f32], query_locs: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.model.forward_batch(sensor_values, query_locs)
    }
}

/// A node in a symbolic expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicNode {
    /// Constant value.
    Constant(f64),
    /// Variable index into the input slice.
    Variable(usize),
    /// Addition of two sub-expressions.
    Add,
    /// Multiplication.
    Mul,
    /// Subtraction (left − right).
    Sub,
    /// Division (left / right); returns 0 on division by zero.
    Div,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Exponential.
    Exp,
    /// Natural log (abs value, returns -inf for 0).
    Log,
    /// Power (left ^ right).
    Pow,
}

/// A symbolic expression tree.
#[derive(Debug, Clone)]
pub struct SymbolicExpr {
    /// Nodes stored as (operator, left_child_idx, right_child_idx).
    pub nodes: Vec<(SymbolicNode, usize, usize)>,
    /// Index of the root node.
    pub root: usize,
}

impl SymbolicExpr {
    /// Create a constant expression.
    pub fn constant(v: f64) -> Self {
        Self {
            nodes: vec![(SymbolicNode::Constant(v), 0, 0)],
            root: 0,
        }
    }

    /// Create a variable expression.
    pub fn variable(idx: usize) -> Self {
        Self {
            nodes: vec![(SymbolicNode::Variable(idx), 0, 0)],
            root: 0,
        }
    }

    /// Build a binary expression from op, left, and right sub-trees.
    pub fn binary(op: SymbolicNode, lhs: SymbolicExpr, rhs: SymbolicExpr) -> Self {
        let mut nodes = lhs.nodes.clone();
        let lhs_root = lhs.root;
        let rhs_offset = nodes.len();
        let rhs_len = rhs.nodes.len();
        let rhs_root_idx = rhs.root;
        for (n, l, r) in rhs.nodes {
            let l2 = if l < rhs_len { l + rhs_offset } else { l };
            let r2 = if r < rhs_len { r + rhs_offset } else { r };
            nodes.push((n, l2, r2));
        }
        let rhs_root = rhs_root_idx + rhs_offset;
        let root = nodes.len();
        nodes.push((op, lhs_root, rhs_root));
        Self { nodes, root }
    }

    /// Build a unary expression from op and child sub-tree.
    pub fn unary(op: SymbolicNode, child: SymbolicExpr) -> Self {
        let mut nodes = child.nodes.clone();
        let child_root = child.root;
        let root = nodes.len();
        nodes.push((op, child_root, 0));
        Self { nodes, root }
    }

    /// Evaluate the expression given variable values.
    pub fn evaluate(&self, vars: &[f64]) -> f64 {
        self.eval_node(self.root, vars)
    }

    fn eval_node(&self, idx: usize, vars: &[f64]) -> f64 {
        match &self.nodes[idx] {
            (SymbolicNode::Constant(v), _, _) => *v,
            (SymbolicNode::Variable(i), _, _) => vars.get(*i).copied().unwrap_or(0.0),
            (SymbolicNode::Add, l, r) => {
                self.eval_node(*l, vars) + self.eval_node(*r, vars)
            }
            (SymbolicNode::Mul, l, r) => {
                self.eval_node(*l, vars) * self.eval_node(*r, vars)
            }
            (SymbolicNode::Sub, l, r) => {
                self.eval_node(*l, vars) - self.eval_node(*r, vars)
            }
            (SymbolicNode::Div, l, r) => {
                let denom = self.eval_node(*r, vars);
                if denom.abs() < 1e-15 {
                    0.0
                } else {
                    self.eval_node(*l, vars) / denom
                }
            }
            (SymbolicNode::Sin, l, _) => self.eval_node(*l, vars).sin(),
            (SymbolicNode::Cos, l, _) => self.eval_node(*l, vars).cos(),
            (SymbolicNode::Exp, l, _) => self.eval_node(*l, vars).exp().min(1e30),
            (SymbolicNode::Log, l, _) => {
                let v = self.eval_node(*l, vars).abs();
                if v < 1e-15 {
                    f64::NEG_INFINITY
                } else {
                    v.ln()
                }
            }
            (SymbolicNode::Pow, l, r) => {
                let base = self.eval_node(*l, vars);
                let exp = self.eval_node(*r, vars);
                base.powf(exp)
            }
        }
    }

    /// Symbolic differentiation with respect to variable `var_idx`.
    pub fn derivative(&self, var_idx: usize) -> SymbolicExpr {
        self.diff_node(self.root, var_idx)
    }

    fn diff_node(&self, idx: usize, var: usize) -> SymbolicExpr {
        match &self.nodes[idx] {
            (SymbolicNode::Constant(_), _, _) => SymbolicExpr::constant(0.0),
            (SymbolicNode::Variable(i), _, _) => {
                if *i == var {
                    SymbolicExpr::constant(1.0)
                } else {
                    SymbolicExpr::constant(0.0)
                }
            }
            (SymbolicNode::Add, l, r) => SymbolicExpr::binary(
                SymbolicNode::Add,
                self.diff_node(*l, var),
                self.diff_node(*r, var),
            ),
            (SymbolicNode::Sub, l, r) => SymbolicExpr::binary(
                SymbolicNode::Sub,
                self.diff_node(*l, var),
                self.diff_node(*r, var),
            ),
            (SymbolicNode::Mul, l, r) => {
                let df = self.diff_node(*l, var);
                let dg = self.diff_node(*r, var);
                let f = self.subtree(*l);
                let g = self.subtree(*r);
                let t1 = SymbolicExpr::binary(SymbolicNode::Mul, df, g);
                let t2 = SymbolicExpr::binary(SymbolicNode::Mul, f, dg);
                SymbolicExpr::binary(SymbolicNode::Add, t1, t2)
            }
            (SymbolicNode::Div, l, r) => {
                let df = self.diff_node(*l, var);
                let dg = self.diff_node(*r, var);
                let f = self.subtree(*l);
                let g = self.subtree(*r);
                let g2 = self.subtree(*r);
                let num_l = SymbolicExpr::binary(SymbolicNode::Mul, df, g);
                let num_r = SymbolicExpr::binary(SymbolicNode::Mul, f, dg);
                let num = SymbolicExpr::binary(SymbolicNode::Sub, num_l, num_r);
                let den = SymbolicExpr::binary(SymbolicNode::Mul, g2, self.subtree(*r));
                SymbolicExpr::binary(SymbolicNode::Div, num, den)
            }
            (SymbolicNode::Sin, l, _) => {
                let cos_child = SymbolicExpr::unary(SymbolicNode::Cos, self.subtree(*l));
                SymbolicExpr::binary(SymbolicNode::Mul, cos_child, self.diff_node(*l, var))
            }
            (SymbolicNode::Cos, l, _) => {
                let neg_sin = SymbolicExpr::binary(
                    SymbolicNode::Mul,
                    SymbolicExpr::constant(-1.0),
                    SymbolicExpr::unary(SymbolicNode::Sin, self.subtree(*l)),
                );
                SymbolicExpr::binary(SymbolicNode::Mul, neg_sin, self.diff_node(*l, var))
            }
            (SymbolicNode::Exp, l, _) => {
                let exp_child = SymbolicExpr::unary(SymbolicNode::Exp, self.subtree(*l));
                SymbolicExpr::binary(SymbolicNode::Mul, exp_child, self.diff_node(*l, var))
            }
            (SymbolicNode::Log, l, _) => {
                let df = self.diff_node(*l, var);
                let f = self.subtree(*l);
                SymbolicExpr::binary(SymbolicNode::Div, df, f)
            }
            (SymbolicNode::Pow, l, r) => {
                let exp_val = self.eval_node(*r, &[]);
                let base = self.subtree(*l);
                let new_exp = SymbolicExpr::constant(exp_val - 1.0);
                let coeff = SymbolicExpr::constant(exp_val);
                let power_term = SymbolicExpr::binary(SymbolicNode::Pow, base, new_exp);
                let df = self.diff_node(*l, var);
                SymbolicExpr::binary(
                    SymbolicNode::Mul,
                    SymbolicExpr::binary(SymbolicNode::Mul, coeff, power_term),
                    df,
                )
            }
        }
    }

    /// Extract a subtree rooted at `idx` as a new `SymbolicExpr`.
    pub fn subtree(&self, idx: usize) -> SymbolicExpr {
        let mut new_nodes = Vec::new();
        self.collect_subtree(idx, &mut new_nodes);
        let root = new_nodes.len() - 1;
        SymbolicExpr {
            nodes: new_nodes,
            root,
        }
    }

    pub(crate) fn collect_subtree(
        &self,
        idx: usize,
        out: &mut Vec<(SymbolicNode, usize, usize)>,
    ) -> usize {
        match &self.nodes[idx] {
            (node @ SymbolicNode::Constant(_), _, _)
            | (node @ SymbolicNode::Variable(_), _, _) => {
                let new_idx = out.len();
                out.push((node.clone(), 0, 0));
                new_idx
            }
            (node @ SymbolicNode::Sin, l, _)
            | (node @ SymbolicNode::Cos, l, _)
            | (node @ SymbolicNode::Exp, l, _)
            | (node @ SymbolicNode::Log, l, _) => {
                let l_new = self.collect_subtree(*l, out);
                let new_idx = out.len();
                out.push((node.clone(), l_new, 0));
                new_idx
            }
            (node, l, r) => {
                let l_new = self.collect_subtree(*l, out);
                let r_new = self.collect_subtree(*r, out);
                let new_idx = out.len();
                out.push((node.clone(), l_new, r_new));
                new_idx
            }
        }
    }

    /// Basic algebraic simplification (one pass).
    pub fn simplify(&self) -> SymbolicExpr {
        self.simplify_node(self.root)
    }

    fn simplify_node(&self, idx: usize) -> SymbolicExpr {
        match &self.nodes[idx] {
            (SymbolicNode::Constant(_), _, _) | (SymbolicNode::Variable(_), _, _) => {
                self.subtree(idx)
            }
            (SymbolicNode::Add, l, r) => {
                let ls = self.simplify_node(*l);
                let rs = self.simplify_node(*r);
                if let (SymbolicNode::Constant(v), _, _) = &ls.nodes[ls.root] {
                    if *v == 0.0 {
                        return rs;
                    }
                }
                if let (SymbolicNode::Constant(v), _, _) = &rs.nodes[rs.root] {
                    if *v == 0.0 {
                        return ls;
                    }
                }
                SymbolicExpr::binary(SymbolicNode::Add, ls, rs)
            }
            (SymbolicNode::Mul, l, r) => {
                let ls = self.simplify_node(*l);
                let rs = self.simplify_node(*r);
                if let (SymbolicNode::Constant(v), _, _) = &ls.nodes[ls.root] {
                    if *v == 0.0 {
                        return SymbolicExpr::constant(0.0);
                    }
                    if *v == 1.0 {
                        return rs;
                    }
                }
                if let (SymbolicNode::Constant(v), _, _) = &rs.nodes[rs.root] {
                    if *v == 0.0 {
                        return SymbolicExpr::constant(0.0);
                    }
                    if *v == 1.0 {
                        return ls;
                    }
                }
                SymbolicExpr::binary(SymbolicNode::Mul, ls, rs)
            }
            (SymbolicNode::Sub, l, r) => {
                let ls = self.simplify_node(*l);
                let rs = self.simplify_node(*r);
                if let (SymbolicNode::Constant(v), _, _) = &rs.nodes[rs.root] {
                    if *v == 0.0 {
                        return ls;
                    }
                }
                SymbolicExpr::binary(SymbolicNode::Sub, ls, rs)
            }
            (SymbolicNode::Div, l, r) => {
                let ls = self.simplify_node(*l);
                let rs = self.simplify_node(*r);
                if let (SymbolicNode::Constant(v), _, _) = &rs.nodes[rs.root] {
                    if *v == 1.0 {
                        return ls;
                    }
                }
                SymbolicExpr::binary(SymbolicNode::Div, ls, rs)
            }
            (op @ SymbolicNode::Sin, l, _)
            | (op @ SymbolicNode::Cos, l, _)
            | (op @ SymbolicNode::Exp, l, _)
            | (op @ SymbolicNode::Log, l, _) => {
                SymbolicExpr::unary(op.clone(), self.simplify_node(*l))
            }
            (op, l, r) => {
                let ls = self.simplify_node(*l);
                let rs = self.simplify_node(*r);
                SymbolicExpr::binary(op.clone(), ls, rs)
            }
        }
    }
}
