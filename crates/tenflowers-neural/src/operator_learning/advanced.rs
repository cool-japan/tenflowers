//! Advanced Neural Operator algorithms: WNO, GNO, PINO, U-Net Neural Operator.

use super::{dense_forward_f32, gelu, relu, xavier_init_f32, Mlp};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Haar DWT helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Haar DWT forward transform on a signal of length `n` (must be even).
/// Returns (approximation coefficients, detail coefficients) each of length n/2.
pub(crate) fn haar_dwt(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let half = n / 2;
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let mut approx = Vec::with_capacity(half);
    let mut detail = Vec::with_capacity(half);
    for i in 0..half {
        let a = x[2 * i];
        let b = x[2 * i + 1];
        approx.push((a + b) * inv_sqrt2);
        detail.push((a - b) * inv_sqrt2);
    }
    (approx, detail)
}

/// Haar IDWT: reconstruct signal of length 2*half from (approx, detail).
pub(crate) fn haar_idwt(approx: &[f32], detail: &[f32]) -> Vec<f32> {
    let half = approx.len();
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let mut out = vec![0.0_f32; 2 * half];
    for i in 0..half {
        let a = approx[i];
        let d = detail[i];
        out[2 * i] = (a + d) * inv_sqrt2;
        out[2 * i + 1] = (a - d) * inv_sqrt2;
    }
    out
}

/// Multi-level Haar DWT: decomposes signal into `levels` of approximation + detail.
/// Returns (approx, Vec<detail>) where approx has length n/(2^levels) and each detail[j] has length n/(2^(j+1)).
pub(crate) fn haar_dwt_multilevel(x: &[f32], levels: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut approx = x.to_vec();
    let mut details = Vec::with_capacity(levels);
    for _ in 0..levels {
        if approx.len() < 2 {
            break;
        }
        let (a, d) = haar_dwt(&approx);
        details.push(d);
        approx = a;
    }
    (approx, details)
}

/// Multi-level Haar IDWT: reconstruct signal from (approx, details).
pub(crate) fn haar_idwt_multilevel(approx: &[f32], details: &[Vec<f32>]) -> Vec<f32> {
    let mut signal = approx.to_vec();
    for d in details.iter().rev() {
        signal = haar_idwt(&signal, d);
    }
    signal
}

// ─────────────────────────────────────────────────────────────────────────────
// Wavelet Neural Operator (WNO)
// ─────────────────────────────────────────────────────────────────────────────

/// A single Wavelet Neural Operator layer.
///
/// Applies Haar DWT to the input, truncates to `num_modes` approximation coefficients,
/// applies a learnable linear map in the wavelet domain, then reconstructs via IDWT
/// and adds a bypass (pointwise linear) connection.
#[derive(Debug, Clone)]
pub struct WnoLayer {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Number of wavelet decomposition levels.
    pub levels: usize,
    /// Number of approximation modes to retain (truncation).
    pub num_modes: usize,
    /// Weight matrix for wavelet-domain mixing: shape [out_channels × (in_channels * num_modes)].
    pub wavelet_w: Vec<f32>,
    /// Bias in wavelet domain.
    pub wavelet_b: Vec<f32>,
    /// Bypass linear weight [out_channels × in_channels].
    pub bypass_w: Vec<f32>,
    /// Bypass bias.
    pub bypass_b: Vec<f32>,
}

impl WnoLayer {
    /// Construct a WnoLayer.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        levels: usize,
        num_modes: usize,
        seed: u64,
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(TensorError::invalid_argument_op(
                "WnoLayer::new",
                "channels must be > 0",
            ));
        }
        if levels == 0 {
            return Err(TensorError::invalid_argument_op(
                "WnoLayer::new",
                "levels must be > 0",
            ));
        }
        if num_modes == 0 {
            return Err(TensorError::invalid_argument_op(
                "WnoLayer::new",
                "num_modes must be > 0",
            ));
        }
        let wavelet_fan_in = in_channels * num_modes;
        let wavelet_w = xavier_init_f32(wavelet_fan_in, out_channels, seed);
        let wavelet_b = vec![0.0_f32; out_channels * num_modes];
        let bypass_w = xavier_init_f32(in_channels, out_channels, seed.wrapping_add(1));
        let bypass_b = vec![0.0_f32; out_channels];
        Ok(Self {
            in_channels,
            out_channels,
            levels,
            num_modes,
            wavelet_w,
            wavelet_b,
            bypass_w,
            bypass_b,
        })
    }

    /// Forward pass.
    ///
    /// `input` has shape `[in_channels × n]` (channels first).
    /// Returns tensor of shape `[out_channels × n]`.
    pub fn forward(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let ic = self.in_channels;
        let oc = self.out_channels;

        if input.len() != ic * n {
            return Err(TensorError::invalid_argument_op(
                "WnoLayer::forward",
                &format!(
                    "input length {} != in_channels({}) * n({})",
                    input.len(),
                    ic,
                    n
                ),
            ));
        }

        // Decompose each channel via multi-level Haar DWT.
        let mut channel_approxs: Vec<Vec<f32>> = Vec::with_capacity(ic);
        let mut channel_details: Vec<Vec<Vec<f32>>> = Vec::with_capacity(ic);
        for c in 0..ic {
            let x_c: Vec<f32> = (0..n).map(|i| input[c * n + i]).collect();
            let (approx, details) = haar_dwt_multilevel(&x_c, self.levels);
            channel_approxs.push(approx);
            channel_details.push(details);
        }

        // Truncate approximation to num_modes.
        let approx_len = channel_approxs[0].len();
        let modes = self.num_modes.min(approx_len);

        // Build wavelet-domain feature vector: concatenation of truncated approxs across channels.
        // For each output channel, compute weighted sum over input channel modes.
        let mut wavelet_out_approx = vec![vec![0.0_f32; modes]; oc];
        for o in 0..oc {
            for m in 0..modes {
                let mut acc = 0.0_f32;
                for c in 0..ic {
                    let wi = o * (ic * modes) + c * modes + m;
                    let w = self.wavelet_w.get(wi).copied().unwrap_or(0.0);
                    acc += w * channel_approxs[c].get(m).copied().unwrap_or(0.0);
                }
                wavelet_out_approx[o][m] = acc + self.wavelet_b.get(o * modes + m).copied().unwrap_or(0.0);
            }
        }

        // Reconstruct via IDWT using first input channel's detail structure as template.
        // Pad approximation back to original approx_len then IDWT with details from channel 0.
        let mut wavelet_spatial = vec![vec![0.0_f32; n]; oc];
        for o in 0..oc {
            // Pad or truncate to approx_len.
            let mut padded = wavelet_out_approx[o].clone();
            padded.resize(approx_len, 0.0_f32);
            // Use channel 0's details for reconstruction (shared structural template).
            let reconstructed = haar_idwt_multilevel(&padded, &channel_details[0]);
            // Resize to n.
            let rlen = reconstructed.len().min(n);
            wavelet_spatial[o][..rlen].copy_from_slice(&reconstructed[..rlen]);
        }

        // Bypass: pointwise linear per position.
        let mut bypass_out = vec![0.0_f32; oc * n];
        for pos in 0..n {
            let x_slice: Vec<f32> = (0..ic).map(|c| input[c * n + pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.bypass_w, &self.bypass_b, oc);
            for o in 0..oc {
                bypass_out[o * n + pos] = y[o];
            }
        }

        // Sum and apply GeLU.
        let mut out = vec![0.0_f32; oc * n];
        for o in 0..oc {
            for i in 0..n {
                out[o * n + i] = gelu(wavelet_spatial[o][i] + bypass_out[o * n + i]);
            }
        }
        Ok(out)
    }
}

/// Wavelet Neural Operator model: stacked WnoLayers with encoder/decoder.
#[derive(Debug, Clone)]
pub struct WnoModel {
    /// Input dimension.
    pub input_dim: usize,
    /// Number of hidden channels.
    pub channels: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Lifting weight (input_dim → channels).
    pub lift_w: Vec<f32>,
    /// Lifting bias.
    pub lift_b: Vec<f32>,
    /// Stacked WNO layers.
    pub layers: Vec<WnoLayer>,
    /// Projection weight (channels → output_dim).
    pub proj_w: Vec<f32>,
    /// Projection bias.
    pub proj_b: Vec<f32>,
}

impl WnoModel {
    /// Construct a WnoModel.
    pub fn new(
        input_dim: usize,
        channels: usize,
        output_dim: usize,
        num_layers: usize,
        levels: usize,
        num_modes: usize,
        seed: u64,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(TensorError::invalid_argument_op(
                "WnoModel::new",
                "num_layers must be > 0",
            ));
        }
        let lift_w = xavier_init_f32(input_dim, channels, seed);
        let lift_b = vec![0.0_f32; channels];
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(WnoLayer::new(
                channels,
                channels,
                levels,
                num_modes,
                seed.wrapping_add(100 + i as u64 * 13),
            )?);
        }
        let proj_w = xavier_init_f32(channels, output_dim, seed.wrapping_add(999));
        let proj_b = vec![0.0_f32; output_dim];
        Ok(Self {
            input_dim,
            channels,
            output_dim,
            lift_w,
            lift_b,
            layers,
            proj_w,
            proj_b,
        })
    }

    /// Forward pass. `input` shape: `[input_dim × n]`. Returns `[output_dim × n]`.
    pub fn forward(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let id = self.input_dim;
        let ch = self.channels;
        let od = self.output_dim;

        if input.len() != id * n {
            return Err(TensorError::invalid_argument_op(
                "WnoModel::forward",
                &format!(
                    "input length {} != input_dim({}) * n({})",
                    input.len(),
                    id,
                    n
                ),
            ));
        }

        // Lift.
        let mut h = vec![0.0_f32; ch * n];
        for pos in 0..n {
            let x_slice: Vec<f32> = (0..id).map(|d| input[d * n + pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.lift_w, &self.lift_b, ch);
            for c in 0..ch {
                h[c * n + pos] = y[c];
            }
        }

        // WNO layers.
        for layer in &self.layers {
            h = layer.forward(&h, n)?;
        }

        // Project.
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

// ─────────────────────────────────────────────────────────────────────────────
// Graph Neural Operator (GNO)
// ─────────────────────────────────────────────────────────────────────────────

/// Parameterised kernel MLP K(x, y) → R^(out_dim) for the GNO integral operator.
///
/// Maps concatenation [x; y] of two spatial coordinates to a kernel matrix row.
#[derive(Debug, Clone)]
pub struct GnoKernel {
    /// MLP implementing K(x, y).
    pub mlp: Mlp,
    /// Spatial dimension of input coordinates.
    pub coord_dim: usize,
    /// Output feature dimension (number of output channels per input channel).
    pub out_dim: usize,
}

impl GnoKernel {
    /// Construct a GnoKernel.
    pub fn new(
        coord_dim: usize,
        out_dim: usize,
        hidden_dims: &[usize],
        seed: u64,
    ) -> Result<Self> {
        let input_dim = 2 * coord_dim; // [x; y] concatenation
        let mut dims = vec![input_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(out_dim);
        let mlp = Mlp::new(&dims, seed)?;
        Ok(Self {
            mlp,
            coord_dim,
            out_dim,
        })
    }

    /// Evaluate K(x, y) where x and y are `coord_dim`-dimensional vectors.
    pub fn evaluate(&self, x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.coord_dim || y.len() != self.coord_dim {
            return Err(TensorError::invalid_argument_op(
                "GnoKernel::evaluate",
                "coordinate dimension mismatch",
            ));
        }
        let mut input = x.to_vec();
        input.extend_from_slice(y);
        self.mlp.forward(&input)
    }
}

/// GNO layer implementing the integral operator via message passing.
///
/// (Ku)(x_i) ≈ Σ_j K(x_i, x_j) · u(x_j)  (sum over neighbours or all nodes)
#[derive(Debug, Clone)]
pub struct GnoLayer {
    /// Kernel MLP K(x, y).
    pub kernel: GnoKernel,
    /// Number of input feature channels.
    pub in_features: usize,
    /// Number of output feature channels.
    pub out_features: usize,
    /// Linear weight for residual connection: [out_features × in_features].
    pub residual_w: Vec<f32>,
    /// Residual bias.
    pub residual_b: Vec<f32>,
}

impl GnoLayer {
    /// Construct a GnoLayer.
    pub fn new(
        coord_dim: usize,
        in_features: usize,
        out_features: usize,
        kernel_hidden: &[usize],
        seed: u64,
    ) -> Result<Self> {
        // Kernel maps [x; y] → R^out_features (one row of the kernel matrix per output channel).
        let kernel = GnoKernel::new(coord_dim, out_features, kernel_hidden, seed)?;
        let residual_w = xavier_init_f32(in_features, out_features, seed.wrapping_add(77));
        let residual_b = vec![0.0_f32; out_features];
        Ok(Self {
            kernel,
            in_features,
            out_features,
            residual_w,
            residual_b,
        })
    }

    /// Forward pass.
    ///
    /// `coords`: [n × coord_dim] flattened row-major.
    /// `features`: [n × in_features] flattened row-major.
    /// Returns `[n × out_features]` flattened row-major.
    pub fn forward(&self, coords: &[f32], features: &[f32], n: usize) -> Result<Vec<f32>> {
        let cd = self.kernel.coord_dim;
        let inf = self.in_features;
        let outf = self.out_features;

        if coords.len() != n * cd {
            return Err(TensorError::invalid_argument_op(
                "GnoLayer::forward",
                &format!(
                    "coords length {} != n({}) * coord_dim({})",
                    coords.len(),
                    n,
                    cd
                ),
            ));
        }
        if features.len() != n * inf {
            return Err(TensorError::invalid_argument_op(
                "GnoLayer::forward",
                &format!(
                    "features length {} != n({}) * in_features({})",
                    features.len(),
                    n,
                    inf
                ),
            ));
        }

        let mut out = vec![0.0_f32; n * outf];

        for i in 0..n {
            let xi: Vec<f32> = (0..cd).map(|d| coords[i * cd + d]).collect();

            // Integral approximation: Σ_j K(x_i, x_j) · u(x_j).
            // K produces a vector of length out_features; we treat it as a scalar weight per channel.
            let mut integral = vec![0.0_f32; outf];
            for j in 0..n {
                let xj: Vec<f32> = (0..cd).map(|d| coords[j * cd + d]).collect();
                let k_ij = self.kernel.evaluate(&xi, &xj)?;
                // k_ij is length out_features; multiply each by the corresponding input feature
                // summed over in_features (simplified: use first in_features channels).
                let scale_factor = if inf > 0 {
                    features[j * inf..(j + 1) * inf].iter().sum::<f32>() / inf as f32
                } else {
                    1.0
                };
                for o in 0..outf {
                    integral[o] += k_ij[o] * scale_factor;
                }
            }
            // Normalize by n.
            let inv_n = 1.0 / n.max(1) as f32;
            for o in 0..outf {
                integral[o] *= inv_n;
            }

            // Residual: W · u(x_i).
            let u_i: Vec<f32> = (0..inf).map(|f| features[i * inf + f]).collect();
            let res = dense_forward_f32(&u_i, &self.residual_w, &self.residual_b, outf);

            // Combine + GeLU.
            for o in 0..outf {
                out[i * outf + o] = gelu(integral[o] + res[o]);
            }
        }
        Ok(out)
    }
}

/// Graph Neural Operator model: encoder-process-decoder for mesh-based PDE solving.
#[derive(Debug, Clone)]
pub struct GnoModel {
    /// Encoder MLP: maps input features to hidden features.
    pub encoder: Mlp,
    /// GNO processing layers.
    pub gno_layers: Vec<GnoLayer>,
    /// Decoder MLP: maps hidden features to output.
    pub decoder: Mlp,
    /// Spatial coordinate dimension.
    pub coord_dim: usize,
    /// Input feature dimension.
    pub input_features: usize,
    /// Hidden feature dimension.
    pub hidden_features: usize,
    /// Output feature dimension.
    pub output_features: usize,
}

impl GnoModel {
    /// Construct a GnoModel.
    pub fn new(
        coord_dim: usize,
        input_features: usize,
        hidden_features: usize,
        output_features: usize,
        num_layers: usize,
        kernel_hidden: &[usize],
        seed: u64,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(TensorError::invalid_argument_op(
                "GnoModel::new",
                "num_layers must be > 0",
            ));
        }
        let encoder = Mlp::new(&[input_features, hidden_features, hidden_features], seed)?;
        let mut gno_layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            gno_layers.push(GnoLayer::new(
                coord_dim,
                hidden_features,
                hidden_features,
                kernel_hidden,
                seed.wrapping_add(200 + i as u64 * 17),
            )?);
        }
        let decoder = Mlp::new(
            &[hidden_features, hidden_features, output_features],
            seed.wrapping_add(900),
        )?;
        Ok(Self {
            encoder,
            gno_layers,
            decoder,
            coord_dim,
            input_features,
            hidden_features,
            output_features,
        })
    }

    /// Forward pass.
    ///
    /// `coords`: [n × coord_dim] row-major.
    /// `features`: [n × input_features] row-major.
    /// Returns `[n × output_features]` row-major.
    pub fn forward(&self, coords: &[f32], features: &[f32], n: usize) -> Result<Vec<f32>> {
        let inf = self.input_features;
        let hf = self.hidden_features;
        let outf = self.output_features;

        if features.len() != n * inf {
            return Err(TensorError::invalid_argument_op(
                "GnoModel::forward",
                "feature dimension mismatch",
            ));
        }

        // Encode each node.
        let mut h = vec![0.0_f32; n * hf];
        for i in 0..n {
            let f_i: Vec<f32> = (0..inf).map(|d| features[i * inf + d]).collect();
            let enc = self.encoder.forward(&f_i)?;
            for d in 0..hf {
                h[i * hf + d] = enc[d];
            }
        }

        // GNO layers (row-major features [n × hf]).
        for layer in &self.gno_layers {
            let new_h = layer.forward(coords, &h, n)?;
            h = new_h;
        }

        // Decode each node.
        let mut out = vec![0.0_f32; n * outf];
        for i in 0..n {
            let h_i: Vec<f32> = (0..hf).map(|d| h[i * hf + d]).collect();
            let dec = self.decoder.forward(&h_i)?;
            for d in 0..outf {
                out[i * outf + d] = dec[d];
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Physics-Informed Neural Operator (PINO)
// ─────────────────────────────────────────────────────────────────────────────

/// Type of PDE for the PINO residual computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdeType {
    /// Heat equation: ∂u/∂t = α ∇²u.
    Heat,
    /// Burgers equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x².
    Burgers,
    /// Wave equation: ∂²u/∂t² = c² ∇²u.
    Wave,
}

/// Combined data + PDE-residual loss for physics-informed neural operator training.
#[derive(Debug, Clone)]
pub struct PinoLoss {
    /// PDE type to enforce.
    pub pde_type: PdeType,
    /// PDE parameter (diffusivity α for heat, viscosity ν for Burgers, wave speed c for wave).
    pub pde_param: f32,
    /// Weight for the PDE residual term relative to data loss.
    pub pde_weight: f32,
    /// Finite-difference step size for spatial derivatives.
    pub dx: f32,
    /// Finite-difference step size for temporal derivatives.
    pub dt: f32,
}

impl PinoLoss {
    /// Create a PinoLoss.
    pub fn new(pde_type: PdeType, pde_param: f32, pde_weight: f32, dx: f32, dt: f32) -> Self {
        Self {
            pde_type,
            pde_param,
            pde_weight,
            dx,
            dt,
        }
    }

    /// Compute data MSE loss.
    fn data_loss(&self, predicted: &[f32], target: &[f32]) -> Result<f32> {
        if predicted.len() != target.len() {
            return Err(TensorError::invalid_argument_op(
                "PinoLoss::data_loss",
                "predicted and target length mismatch",
            ));
        }
        if predicted.is_empty() {
            return Ok(0.0);
        }
        let sum: f32 = predicted
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        Ok(sum / predicted.len() as f32)
    }

    /// Compute PDE residual on a 1D spatial field `u` at time step `t`.
    /// `u_prev` and `u_next` are fields at t-1 and t+1 respectively.
    fn pde_residual(
        &self,
        u_prev: &[f32],
        u_curr: &[f32],
        u_next: &[f32],
    ) -> Result<f32> {
        let n = u_curr.len();
        if u_prev.len() != n || u_next.len() != n || n < 3 {
            return Err(TensorError::invalid_argument_op(
                "PinoLoss::pde_residual",
                "field length mismatch or too small",
            ));
        }
        let dx = self.dx;
        let dt = self.dt;
        let mut res_sum = 0.0_f32;

        match self.pde_type {
            PdeType::Heat => {
                let alpha = self.pde_param;
                // ∂u/∂t ≈ (u_next - u_prev) / (2dt)
                // ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
                for i in 1..n - 1 {
                    let dudt = (u_next[i] - u_prev[i]) / (2.0 * dt);
                    let d2udx2 = (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]) / (dx * dx);
                    let res = dudt - alpha * d2udx2;
                    res_sum += res * res;
                }
            }
            PdeType::Burgers => {
                let nu = self.pde_param;
                // ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
                for i in 1..n - 1 {
                    let dudt = (u_next[i] - u_prev[i]) / (2.0 * dt);
                    let dudx = (u_curr[i + 1] - u_curr[i - 1]) / (2.0 * dx);
                    let d2udx2 =
                        (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]) / (dx * dx);
                    let res = dudt + u_curr[i] * dudx - nu * d2udx2;
                    res_sum += res * res;
                }
            }
            PdeType::Wave => {
                let c = self.pde_param;
                // ∂²u/∂t² ≈ (u_next - 2u_curr + u_prev) / dt²
                // c² ∂²u/∂x²
                for i in 1..n - 1 {
                    let d2udt2 =
                        (u_next[i] - 2.0 * u_curr[i] + u_prev[i]) / (dt * dt);
                    let d2udx2 =
                        (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]) / (dx * dx);
                    let res = d2udt2 - c * c * d2udx2;
                    res_sum += res * res;
                }
            }
        }

        let interior_pts = (n - 2).max(1) as f32;
        Ok(res_sum / interior_pts)
    }

    /// Compute total PINO loss = data_loss + pde_weight * pde_residual.
    ///
    /// `fields` is a sequence of time slices [t_steps × n_x] (flattened row-major).
    /// `predicted` and `target` are flat arrays of same length.
    pub fn compute(
        &self,
        predicted: &[f32],
        target: &[f32],
        fields: &[Vec<f32>],
    ) -> Result<f32> {
        let dl = self.data_loss(predicted, target)?;

        if fields.len() < 3 {
            return Ok(dl);
        }

        let mut pde_total = 0.0_f32;
        let t_steps = fields.len();
        for t in 1..t_steps - 1 {
            let res = self.pde_residual(&fields[t - 1], &fields[t], &fields[t + 1])?;
            pde_total += res;
        }
        let pde_avg = pde_total / (t_steps - 2).max(1) as f32;

        Ok(dl + self.pde_weight * pde_avg)
    }
}

/// Trainer for physics-informed neural operator, wrapping an FNO-style model.
#[derive(Debug, Clone)]
pub struct PinoTrainer {
    /// Loss function combining data and PDE residual.
    pub loss_fn: PinoLoss,
    /// Learning rate for gradient updates.
    pub lr: f32,
    /// Number of training epochs.
    pub epochs: usize,
}

impl PinoTrainer {
    /// Create a PinoTrainer.
    pub fn new(loss_fn: PinoLoss, lr: f32, epochs: usize) -> Self {
        Self {
            loss_fn,
            lr,
            epochs,
        }
    }

    /// Evaluate the combined PINO loss on a single example.
    pub fn evaluate_loss(
        &self,
        predicted: &[f32],
        target: &[f32],
        fields: &[Vec<f32>],
    ) -> Result<f32> {
        self.loss_fn.compute(predicted, target, fields)
    }

    /// Simulate one "pseudo-training" step: compute loss and return it.
    /// (Actual weight updates require autograd; this returns the scalar loss.)
    pub fn train_step(
        &self,
        predicted: &[f32],
        target: &[f32],
        fields: &[Vec<f32>],
    ) -> Result<f32> {
        self.evaluate_loss(predicted, target, fields)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// U-Net Neural Operator (UNO)
// ─────────────────────────────────────────────────────────────────────────────

/// Encoder block for the U-Net Neural Operator.
///
/// Performs strided (factor-2) downsampling with a learnable convolutional-style
/// linear projection applied at every other position.
#[derive(Debug, Clone)]
pub struct UnoEncoder {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Linear weight for each strided position: [out_channels × in_channels].
    pub weight: Vec<f32>,
    /// Bias.
    pub bias: Vec<f32>,
}

impl UnoEncoder {
    /// Create a UnoEncoder.
    pub fn new(in_channels: usize, out_channels: usize, seed: u64) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(TensorError::invalid_argument_op(
                "UnoEncoder::new",
                "channels must be > 0",
            ));
        }
        let weight = xavier_init_f32(in_channels, out_channels, seed);
        let bias = vec![0.0_f32; out_channels];
        Ok(Self {
            in_channels,
            out_channels,
            weight,
            bias,
        })
    }

    /// Forward: downsample input `[in_channels × n]` → `[out_channels × n/2]` (drops last if odd).
    pub fn forward(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let ic = self.in_channels;
        let oc = self.out_channels;
        if input.len() != ic * n {
            return Err(TensorError::invalid_argument_op(
                "UnoEncoder::forward",
                "input length mismatch",
            ));
        }
        let n_out = n / 2;
        let mut out = vec![0.0_f32; oc * n_out];
        for pos in 0..n_out {
            let src_pos = pos * 2; // stride-2 downsampling
            let x_slice: Vec<f32> = (0..ic).map(|c| input[c * n + src_pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.weight, &self.bias, oc);
            for o in 0..oc {
                out[o * n_out + pos] = relu(y[o]);
            }
        }
        Ok(out)
    }
}

/// Decoder block for the U-Net Neural Operator.
///
/// Upsamples by factor 2 via nearest-neighbour interpolation followed by a linear projection.
#[derive(Debug, Clone)]
pub struct UnoDecoder {
    /// Number of input channels (may include skip connection channels).
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Linear weight: [out_channels × in_channels].
    pub weight: Vec<f32>,
    /// Bias.
    pub bias: Vec<f32>,
}

impl UnoDecoder {
    /// Create a UnoDecoder.
    pub fn new(in_channels: usize, out_channels: usize, seed: u64) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(TensorError::invalid_argument_op(
                "UnoDecoder::new",
                "channels must be > 0",
            ));
        }
        let weight = xavier_init_f32(in_channels, out_channels, seed);
        let bias = vec![0.0_f32; out_channels];
        Ok(Self {
            in_channels,
            out_channels,
            weight,
            bias,
        })
    }

    /// Forward: upsample `[in_channels × n_in]` → `[out_channels × n_out]` (n_out = 2*n_in).
    pub fn forward(&self, input: &[f32], n_in: usize) -> Result<Vec<f32>> {
        let ic = self.in_channels;
        let oc = self.out_channels;
        if input.len() != ic * n_in {
            return Err(TensorError::invalid_argument_op(
                "UnoDecoder::forward",
                "input length mismatch",
            ));
        }
        let n_out = 2 * n_in;
        let mut out = vec![0.0_f32; oc * n_out];
        for pos in 0..n_out {
            // Nearest-neighbour upsampling.
            let src_pos = pos / 2;
            let x_slice: Vec<f32> = (0..ic).map(|c| input[c * n_in + src_pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.weight, &self.bias, oc);
            for o in 0..oc {
                out[o * n_out + pos] = gelu(y[o]);
            }
        }
        Ok(out)
    }
}

/// Skip connection block for multi-scale feature fusion in the U-Net Neural Operator.
#[derive(Debug, Clone)]
pub struct UnoSkipConnection {
    /// Number of encoder feature channels.
    pub enc_channels: usize,
    /// Number of decoder feature channels.
    pub dec_channels: usize,
    /// Number of output fused channels.
    pub out_channels: usize,
    /// Linear weight: [out_channels × (enc_channels + dec_channels)].
    pub weight: Vec<f32>,
    /// Bias.
    pub bias: Vec<f32>,
}

impl UnoSkipConnection {
    /// Create a UnoSkipConnection.
    pub fn new(
        enc_channels: usize,
        dec_channels: usize,
        out_channels: usize,
        seed: u64,
    ) -> Result<Self> {
        let in_ch = enc_channels + dec_channels;
        if in_ch == 0 || out_channels == 0 {
            return Err(TensorError::invalid_argument_op(
                "UnoSkipConnection::new",
                "channels must be > 0",
            ));
        }
        let weight = xavier_init_f32(in_ch, out_channels, seed);
        let bias = vec![0.0_f32; out_channels];
        Ok(Self {
            enc_channels,
            dec_channels,
            out_channels,
            weight,
            bias,
        })
    }

    /// Fuse encoder features `[enc_channels × n]` and decoder features `[dec_channels × n]`.
    /// Returns `[out_channels × n]`.
    pub fn forward(&self, enc: &[f32], dec: &[f32], n: usize) -> Result<Vec<f32>> {
        let ec = self.enc_channels;
        let dc = self.dec_channels;
        let oc = self.out_channels;
        if enc.len() != ec * n || dec.len() != dc * n {
            return Err(TensorError::invalid_argument_op(
                "UnoSkipConnection::forward",
                "input channel dimension mismatch",
            ));
        }
        let in_ch = ec + dc;
        let mut out = vec![0.0_f32; oc * n];
        for pos in 0..n {
            let mut concat = Vec::with_capacity(in_ch);
            for c in 0..ec {
                concat.push(enc[c * n + pos]);
            }
            for c in 0..dc {
                concat.push(dec[c * n + pos]);
            }
            let y = dense_forward_f32(&concat, &self.weight, &self.bias, oc);
            for o in 0..oc {
                out[o * n + pos] = relu(y[o]);
            }
        }
        Ok(out)
    }
}

/// Full U-Net Neural Operator: multi-resolution encoder-decoder with skip connections.
///
/// Architecture:
/// - Encoder: 3 downsampling stages.
/// - Bottleneck: linear layer.
/// - Decoder: 3 upsampling stages with skip connections.
#[derive(Debug, Clone)]
pub struct UnoModel {
    /// Input dimension.
    pub input_dim: usize,
    /// Base number of channels (doubled at each encoder level).
    pub base_channels: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Lifting layer: input_dim → base_channels.
    pub lift_w: Vec<f32>,
    /// Lifting bias.
    pub lift_b: Vec<f32>,
    /// Encoder stages.
    pub encoders: Vec<UnoEncoder>,
    /// Bottleneck MLP.
    pub bottleneck: Mlp,
    /// Decoder stages.
    pub decoders: Vec<UnoDecoder>,
    /// Skip connection fusions.
    pub skips: Vec<UnoSkipConnection>,
    /// Projection weight: base_channels → output_dim.
    pub proj_w: Vec<f32>,
    /// Projection bias.
    pub proj_b: Vec<f32>,
}

impl UnoModel {
    /// Construct a UnoModel with 3 encoder/decoder levels.
    pub fn new(
        input_dim: usize,
        base_channels: usize,
        output_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if base_channels == 0 {
            return Err(TensorError::invalid_argument_op(
                "UnoModel::new",
                "base_channels must be > 0",
            ));
        }
        let lift_w = xavier_init_f32(input_dim, base_channels, seed);
        let lift_b = vec![0.0_f32; base_channels];

        // 3 encoder stages: base → 2*base → 4*base → 8*base
        let mut encoders = Vec::new();
        let enc_channels = [
            (base_channels, 2 * base_channels),
            (2 * base_channels, 4 * base_channels),
            (4 * base_channels, 8 * base_channels),
        ];
        for (i, (ic, oc)) in enc_channels.iter().enumerate() {
            encoders.push(UnoEncoder::new(*ic, *oc, seed.wrapping_add(100 + i as u64 * 11))?);
        }

        let bottleneck = Mlp::new(
            &[8 * base_channels, 8 * base_channels, 8 * base_channels],
            seed.wrapping_add(500),
        )?;

        // 3 decoder stages: each upsamples by 2x then skip-fuses.
        // Decoder receives only the previous h_dec (not pre-concat with skip).
        let dec_configs = [
            (8 * base_channels, 4 * base_channels),
            (4 * base_channels, 2 * base_channels),
            (2 * base_channels, base_channels),
        ];
        let mut decoders = Vec::new();
        for (i, (ic, oc)) in dec_configs.iter().enumerate() {
            decoders.push(UnoDecoder::new(*ic, *oc, seed.wrapping_add(600 + i as u64 * 11))?);
        }

        // Skip connections: fuse encoder skip (enc_ch) with decoder upsampled (dec_ch) → out_ch.
        // skip_features[2-i] has channels: 4bc, 2bc, bc for i=0,1,2.
        let skip_configs = [
            (4 * base_channels, 4 * base_channels, 4 * base_channels),
            (2 * base_channels, 2 * base_channels, 2 * base_channels),
            (base_channels, base_channels, base_channels),
        ];
        let mut skips = Vec::new();
        for (i, (ec, dc, oc)) in skip_configs.iter().enumerate() {
            skips.push(UnoSkipConnection::new(
                *ec,
                *dc,
                *oc,
                seed.wrapping_add(700 + i as u64 * 11),
            )?);
        }

        let proj_w = xavier_init_f32(base_channels, output_dim, seed.wrapping_add(999));
        let proj_b = vec![0.0_f32; output_dim];

        Ok(Self {
            input_dim,
            base_channels,
            output_dim,
            lift_w,
            lift_b,
            encoders,
            bottleneck,
            decoders,
            skips,
            proj_w,
            proj_b,
        })
    }

    /// Forward pass. `input`: `[input_dim × n]`. Returns `[output_dim × n]`.
    ///
    /// Note: `n` must be divisible by 8 for 3 levels of downsampling.
    pub fn forward(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let id = self.input_dim;
        let bc = self.base_channels;
        let od = self.output_dim;

        if input.len() != id * n {
            return Err(TensorError::invalid_argument_op(
                "UnoModel::forward",
                &format!(
                    "input length {} != input_dim({}) * n({})",
                    input.len(),
                    id,
                    n
                ),
            ));
        }

        // Lift input_dim → base_channels.
        let mut h = vec![0.0_f32; bc * n];
        for pos in 0..n {
            let x_slice: Vec<f32> = (0..id).map(|d| input[d * n + pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.lift_w, &self.lift_b, bc);
            for c in 0..bc {
                h[c * n + pos] = y[c];
            }
        }

        // Encoder: collect skip features.
        let mut skip_features: Vec<(Vec<f32>, usize)> = Vec::new();
        let mut current_n = n;
        let mut h_enc = h.clone();
        for enc in &self.encoders {
            let enc_out = enc.forward(&h_enc, current_n)?;
            skip_features.push((h_enc.clone(), current_n));
            h_enc = enc_out;
            current_n /= 2;
        }

        // Bottleneck: apply MLP pointwise.
        let bottleneck_ch = 8 * bc;
        let mut h_bot = vec![0.0_f32; bottleneck_ch * current_n];
        for pos in 0..current_n {
            let x_slice: Vec<f32> = (0..bottleneck_ch).map(|c| h_enc[c * current_n + pos]).collect();
            let y = self.bottleneck.forward(&x_slice)?;
            for c in 0..bottleneck_ch {
                h_bot[c * current_n + pos] = y[c];
            }
        }

        // Decoder: upsample with skip connections.
        let mut h_dec = h_bot;
        for (i, (dec, skip_layer)) in self.decoders.iter().zip(self.skips.iter()).enumerate() {
            let (skip_h, skip_n) = &skip_features[2 - i]; // reverse order
            // Upsample h_dec.
            let dec_out = dec.forward(&h_dec, current_n)?;
            current_n *= 2;
            // Fuse with skip.
            let fused = skip_layer.forward(skip_h, &dec_out, *skip_n)?;
            h_dec = fused;
        }

        // Project to output_dim.
        let mut out = vec![0.0_f32; od * n];
        let final_ch = bc;
        for pos in 0..n {
            let x_slice: Vec<f32> = (0..final_ch).map(|c| h_dec[c * n + pos]).collect();
            let y = dense_forward_f32(&x_slice, &self.proj_w, &self.proj_b, od);
            for d in 0..od {
                out[d * n + pos] = y[d];
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for neural operator models.
#[derive(Debug, Clone)]
pub struct NeuralOperatorMetrics {
    /// Relative L2 error: ||u_pred - u_true||₂ / ||u_true||₂.
    pub relative_l2: f32,
    /// Mean absolute error.
    pub mae: f32,
    /// Maximum absolute error.
    pub max_error: f32,
    /// R² coefficient of determination.
    pub r_squared: f32,
}

impl NeuralOperatorMetrics {
    /// Compute metrics from predicted and true fields.
    pub fn compute(predicted: &[f32], target: &[f32]) -> Result<Self> {
        if predicted.len() != target.len() {
            return Err(TensorError::invalid_argument_op(
                "NeuralOperatorMetrics::compute",
                "predicted and target length mismatch",
            ));
        }
        if predicted.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "NeuralOperatorMetrics::compute",
                "empty arrays",
            ));
        }
        let n = predicted.len() as f32;

        let target_mean = target.iter().sum::<f32>() / n;

        let ss_res: f32 = predicted
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        let ss_tot: f32 = target.iter().map(|t| (t - target_mean).powi(2)).sum();
        let norm_target: f32 = target.iter().map(|t| t * t).sum::<f32>().sqrt();

        let relative_l2 = if norm_target > 1e-10 {
            ss_res.sqrt() / norm_target
        } else {
            ss_res.sqrt()
        };

        let mae = predicted
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f32>()
            / n;

        let max_error = predicted
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0_f32, f32::max);

        let r_squared = if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            if ss_res < 1e-10 { 1.0 } else { 0.0 }
        };

        Ok(Self {
            relative_l2,
            mae,
            max_error,
            r_squared,
        })
    }
}
