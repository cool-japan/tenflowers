//! # Data Augmentation Library (Track B Round 48)
//!
//! A comprehensive, production-grade data augmentation framework providing
//! geometric, color/value, mixing, and policy-based augmentations.
//!
//! ## Algorithms Implemented
//!
//! - **DaHorizontalFlip / DaVerticalFlip** — Axis flipping for 2D images.
//! - **DaRandomCrop** — Padded random crop for images and sequences.
//! - **DaRandomRotation** — Nearest-neighbor rotation via affine transform.
//! - **DaShear** — Horizontal shear transformation.
//! - **DaBrightness / DaContrast** — Value-space jitter.
//! - **DaGaussianNoise** — Additive white Gaussian noise.
//! - **DaGaussianBlur** — 1D/2D Gaussian kernel convolution.
//! - **DaRandomErasing** — Random rectangular region erasure (Zhong et al. 2020).
//! - **DaMixup** — Convex interpolation of samples (Zhang et al. 2018).
//! - **DaCutmix** — Cut-and-paste region mixing (Yun et al. 2019).
//! - **DaAutoAugment** — AutoAugment policy search (Cubuk et al. 2019).
//! - **DaRandAugment** — Simplified N+M augmentation (Cubuk et al. 2020).
//! - **DaTrivialAugment** — Single uniformly-random op (Müller & Hutter 2021).
//! - **DaAugmentationPipeline** — Sequential/independent composition.
//! - **DaDiffAugment** — Differentiable augmentation for GAN training (Zhao et al. 2020).
//! - **DaMetrics** — Diversity, strength, and coverage metrics.
//!
//! ## References
//!
//! - Cubuk et al. (2019) "AutoAugment: Learning Augmentation Strategies from Data"
//! - Cubuk et al. (2020) "RandAugment: Practical automated data augmentation"
//! - Müller & Hutter (2021) "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation"
//! - Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"
//! - Yun et al. (2019) "CutMix: Training Strategy that Makes use of Sample Mixing"
//! - Zhong et al. (2020) "Random Erasing Data Augmentation"
//! - Zhao et al. (2020) "Differentiable Augmentation for Data-Efficient GAN Training"

#[cfg(test)]
mod tests;

use std::fmt;

// ─── §0  Error type ──────────────────────────────────────────────────────────

/// Errors produced by the data-augmentation module.
#[derive(Debug, Clone)]
pub enum DaError {
    /// Invalid input data or shape.
    InvalidInput(String),
    /// Configuration error.
    ConfigError(String),
    /// Numerical computation error.
    NumericalError(String),
}

impl fmt::Display for DaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaError::InvalidInput(msg) => write!(f, "InvalidInput: {}", msg),
            DaError::ConfigError(msg) => write!(f, "ConfigError: {}", msg),
            DaError::NumericalError(msg) => write!(f, "NumericalError: {}", msg),
        }
    }
}

impl std::error::Error for DaError {}

// ─── §0b  Seeded RNG ─────────────────────────────────────────────────────────

/// Xorshift64-based fast uniform U(0,1) random number.
///
/// Uses a splitmix64 step to advance the state, then maps to `(0, 1)`.
/// The state is advanced in-place.
pub fn da_rand01(seed: &mut u64) -> f64 {
    // splitmix64 state advance.
    *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // Map to (0, 1) — avoid exact 0.
    let v = (z >> 11) as f64 / (1u64 << 53) as f64;
    if v == 0.0 {
        f64::MIN_POSITIVE
    } else {
        v
    }
}

/// Box-Muller transform: two U(0,1) samples → N(0,1).
pub fn da_randn(seed: &mut u64) -> f64 {
    let u1 = da_rand01(seed).max(1e-15);
    let u2 = da_rand01(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Uniform integer in `[lo, hi)`.
pub fn da_randint(seed: &mut u64, lo: i64, hi: i64) -> i64 {
    if hi <= lo {
        return lo;
    }
    let range = (hi - lo) as u64;
    // Use da_rand01 internally for consistent state advancement.
    let raw_f = da_rand01(seed);
    let raw = (raw_f * range as f64) as u64;
    lo + (raw.min(range - 1)) as i64
}

// ─── §1  DaSample ────────────────────────────────────────────────────────────

/// A single data sample with flat data array, shape descriptor, and optional labels.
///
/// The data is stored in row-major (C-order) flat layout. For an image of
/// `shape = [H, W]` index `(r, c)` maps to `r*W + c`. For `[C, H, W]` it is
/// `c_idx*H*W + r*W + col`.
#[derive(Debug, Clone)]
pub struct DaSample {
    /// Flat data array in row-major order.
    pub data: Vec<f64>,
    /// Shape of the sample, e.g. `[H, W]` or `[C, H, W]` or `[T, F]`.
    pub shape: Vec<usize>,
    /// Hard integer label (class index).
    pub label: Option<usize>,
    /// Soft / mixed label vector (e.g. from Mixup/CutMix).
    pub soft_label: Option<Vec<f64>>,
}

impl DaSample {
    /// Create a new sample, validating that `data.len() == product(shape)`.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, DaError> {
        if shape.is_empty() {
            return Err(DaError::InvalidInput("shape must be non-empty".to_string()));
        }
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(DaError::InvalidInput(format!(
                "data length {} does not match shape product {}",
                data.len(),
                expected
            )));
        }
        Ok(Self {
            data,
            shape,
            label: None,
            soft_label: None,
        })
    }

    /// Read the value at multi-dimensional `indices`.
    pub fn get(&self, indices: &[usize]) -> Result<f64, DaError> {
        let flat = self.flat_index(indices)?;
        Ok(self.data[flat])
    }

    /// Write `val` at multi-dimensional `indices`.
    pub fn set(&mut self, indices: &[usize], val: f64) -> Result<(), DaError> {
        let flat = self.flat_index(indices)?;
        self.data[flat] = val;
        Ok(())
    }

    /// Total number of elements.
    #[inline]
    pub fn flat_len(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Clone this sample (convenience alias for `Clone::clone`).
    pub fn clone_sample(&self) -> DaSample {
        self.clone()
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn flat_index(&self, indices: &[usize]) -> Result<usize, DaError> {
        if indices.len() != self.shape.len() {
            return Err(DaError::InvalidInput(format!(
                "index length {} != ndim {}",
                indices.len(),
                self.shape.len()
            )));
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (i, &idx) in indices.iter().enumerate().rev() {
            if idx >= self.shape[i] {
                return Err(DaError::InvalidInput(format!(
                    "index {} out of bounds for dim {} (size {})",
                    idx, i, self.shape[i]
                )));
            }
            flat += idx * stride;
            stride *= self.shape[i];
        }
        Ok(flat)
    }

    /// Return `(H, W)` for 2D shapes, or `(1, L)` for 1D shapes.
    fn hw(&self) -> (usize, usize) {
        match self.shape.len() {
            1 => (1, self.shape[0]),
            2 => (self.shape[0], self.shape[1]),
            _ => {
                // Last two dims for CHW etc.
                let ndim = self.shape.len();
                (self.shape[ndim - 2], self.shape[ndim - 1])
            }
        }
    }
}

// ─── §2  DaAugment trait ─────────────────────────────────────────────────────

/// Trait implemented by all augmentation operations.
pub trait DaAugment {
    /// Apply the augmentation stochastically, using `seed` for randomness.
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError>;

    /// A human-readable name for logging and policy dispatch.
    fn name(&self) -> &str;

    /// Base probability with which this augmentation is applied inside a policy.
    fn probability(&self) -> f64;
}

// ─── §3  Geometric transformations ───────────────────────────────────────────

/// Flip each row of the spatial dimensions (mirror along the W axis).
///
/// For `shape = [H, W]` pixel `(r, c)` maps to `(r, W-1-c)`.
/// Channels (leading dims) are preserved unchanged.
#[derive(Debug, Clone)]
pub struct DaHorizontalFlip {
    /// Probability in `[0, 1]` of applying the flip.
    pub p: f64,
}

impl DaAugment for DaHorizontalFlip {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let (h, w) = sample.hw();
        let channels = sample.flat_len() / (h * w);
        let mut out = sample.clone_sample();
        for c in 0..channels {
            for row in 0..h {
                for col in 0..(w / 2) {
                    let a = c * h * w + row * w + col;
                    let b = c * h * w + row * w + (w - 1 - col);
                    out.data.swap(a, b);
                }
            }
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "HorizontalFlip"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

/// Flip along the H axis (top ↔ bottom).
#[derive(Debug, Clone)]
pub struct DaVerticalFlip {
    /// Probability in `[0, 1]` of applying the flip.
    pub p: f64,
}

impl DaAugment for DaVerticalFlip {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let (h, w) = sample.hw();
        let channels = sample.flat_len() / (h * w);
        let mut out = sample.clone_sample();
        for c in 0..channels {
            for row in 0..(h / 2) {
                for col in 0..w {
                    let a = c * h * w + row * w + col;
                    let b = c * h * w + (h - 1 - row) * w + col;
                    out.data.swap(a, b);
                }
            }
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "VerticalFlip"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

/// Pad then take a random crop of size `(crop_h × crop_w)`.
///
/// For 1D inputs (`shape = [L]`), the input is treated as `(1, L)` and a
/// window of length `crop_w` is sampled.
#[derive(Debug, Clone)]
pub struct DaRandomCrop {
    /// Probability in `[0, 1]` of applying the crop.
    pub p: f64,
    /// Height of the output crop.
    pub crop_h: usize,
    /// Width of the output crop.
    pub crop_w: usize,
    /// Number of pixels to pad on each side before cropping.
    pub pad: usize,
}

impl DaAugment for DaRandomCrop {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            // Return a centered crop of the same shape (no change to meta).
            return self.center_crop(sample);
        }
        self.random_crop_impl(sample, seed)
    }

    fn name(&self) -> &str {
        "RandomCrop"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

impl DaRandomCrop {
    fn center_crop(&self, sample: &DaSample) -> Result<DaSample, DaError> {
        let (h, w) = sample.hw();
        let channels = sample.flat_len() / (h * w);
        let ph = h + 2 * self.pad;
        let pw = w + 2 * self.pad;
        if self.crop_h > ph || self.crop_w > pw {
            return Err(DaError::ConfigError(format!(
                "crop ({} × {}) larger than padded size ({} × {})",
                self.crop_h, self.crop_w, ph, pw
            )));
        }
        let start_r = (ph.saturating_sub(self.crop_h)) / 2;
        let start_c = (pw.saturating_sub(self.crop_w)) / 2;
        self.do_crop(sample, channels, h, w, ph, pw, start_r, start_c)
    }

    fn random_crop_impl(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        let (h, w) = sample.hw();
        let channels = sample.flat_len() / (h * w);
        let ph = h + 2 * self.pad;
        let pw = w + 2 * self.pad;
        if self.crop_h > ph || self.crop_w > pw {
            return Err(DaError::ConfigError(format!(
                "crop ({} × {}) larger than padded size ({} × {})",
                self.crop_h, self.crop_w, ph, pw
            )));
        }
        let max_r = ph - self.crop_h;
        let max_c = pw - self.crop_w;
        let start_r = da_randint(seed, 0, (max_r + 1) as i64) as usize;
        let start_c = da_randint(seed, 0, (max_c + 1) as i64) as usize;
        self.do_crop(sample, channels, h, w, ph, pw, start_r, start_c)
    }

    fn do_crop(
        &self,
        sample: &DaSample,
        channels: usize,
        h: usize,
        w: usize,
        ph: usize,
        pw: usize,
        start_r: usize,
        start_c: usize,
    ) -> Result<DaSample, DaError> {
        let total = channels * self.crop_h * self.crop_w;
        let mut out_data = vec![0.0f64; total];
        for c in 0..channels {
            for dr in 0..self.crop_h {
                for dc in 0..self.crop_w {
                    let pr = start_r + dr;
                    let pc = start_c + dc;
                    // Source pixel (accounting for padding → original coords)
                    let val = if pr < self.pad
                        || pc < self.pad
                        || pr >= h + self.pad
                        || pc >= w + self.pad
                    {
                        0.0 // zero-padding
                    } else {
                        let sr = pr - self.pad;
                        let sc = pc - self.pad;
                        sample.data[c * h * w + sr * w + sc]
                    };
                    out_data[c * self.crop_h * self.crop_w + dr * self.crop_w + dc] = val;
                }
            }
        }
        // Adjust shape: replace last two dims.
        let mut new_shape = sample.shape.clone();
        let ndim = new_shape.len();
        if ndim >= 2 {
            new_shape[ndim - 2] = self.crop_h;
            new_shape[ndim - 1] = self.crop_w;
        } else {
            new_shape[0] = self.crop_w;
        }
        let _ = ph; // suppress unused warning
        let _ = pw;
        let mut out = DaSample::new(out_data, new_shape)?;
        out.label = sample.label;
        out.soft_label = sample.soft_label.clone();
        Ok(out)
    }
}

/// Rotate the spatial plane by a random angle in `[-max_degrees, max_degrees]`.
///
/// Uses nearest-neighbour resampling; out-of-bounds pixels are filled with 0.
#[derive(Debug, Clone)]
pub struct DaRandomRotation {
    /// Probability of applying the rotation.
    pub p: f64,
    /// Maximum rotation in degrees.
    pub max_degrees: f64,
}

impl DaAugment for DaRandomRotation {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let angle_deg = (da_rand01(seed) * 2.0 - 1.0) * self.max_degrees;
        let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let (h, w) = sample.hw();
        let channels = sample.flat_len() / (h * w);
        let cx = (w as f64 - 1.0) / 2.0;
        let cy = (h as f64 - 1.0) / 2.0;

        let mut out_data = vec![0.0f64; sample.flat_len()];
        for c in 0..channels {
            for row in 0..h {
                for col in 0..w {
                    // Compute source coord by inverse rotation.
                    let dy = row as f64 - cy;
                    let dx = col as f64 - cx;
                    let src_x = cos_a * dx + sin_a * dy + cx;
                    let src_y = -sin_a * dx + cos_a * dy + cy;
                    let sr = src_y.round() as i64;
                    let sc = src_x.round() as i64;
                    let val = if sr >= 0 && sr < h as i64 && sc >= 0 && sc < w as i64 {
                        sample.data[c * h * w + sr as usize * w + sc as usize]
                    } else {
                        0.0
                    };
                    out_data[c * h * w + row * w + col] = val;
                }
            }
        }
        let mut out = DaSample::new(out_data, sample.shape.clone())?;
        out.label = sample.label;
        out.soft_label = sample.soft_label.clone();
        Ok(out)
    }

    fn name(&self) -> &str {
        "RandomRotation"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

/// Horizontal shear: `x' = x + s * y`, `y' = y` where `s` is random in
/// `[-max_shear, max_shear]`.
#[derive(Debug, Clone)]
pub struct DaShear {
    /// Probability of applying the shear.
    pub p: f64,
    /// Maximum shear coefficient.
    pub max_shear: f64,
}

impl DaAugment for DaShear {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let s = (da_rand01(seed) * 2.0 - 1.0) * self.max_shear;
        let (h, w) = sample.hw();
        let channels = sample.flat_len() / (h * w);
        let cx = (w as f64 - 1.0) / 2.0;
        let cy = (h as f64 - 1.0) / 2.0;

        let mut out_data = vec![0.0f64; sample.flat_len()];
        for c in 0..channels {
            for row in 0..h {
                for col in 0..w {
                    // Inverse shear: src_x = x' - s * y', src_y = y'
                    let dy = row as f64 - cy;
                    let dx = col as f64 - cx;
                    let src_x = dx - s * dy + cx;
                    let src_y = dy + cy;
                    let sr = src_y.round() as i64;
                    let sc = src_x.round() as i64;
                    let val = if sr >= 0 && sr < h as i64 && sc >= 0 && sc < w as i64 {
                        sample.data[c * h * w + sr as usize * w + sc as usize]
                    } else {
                        0.0
                    };
                    out_data[c * h * w + row * w + col] = val;
                }
            }
        }
        let mut out = DaSample::new(out_data, sample.shape.clone())?;
        out.label = sample.label;
        out.soft_label = sample.soft_label.clone();
        Ok(out)
    }

    fn name(&self) -> &str {
        "Shear"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

// ─── §4  Color / value augmentations ─────────────────────────────────────────

/// Multiply all values by a random factor in `factor_range`.
#[derive(Debug, Clone)]
pub struct DaBrightness {
    /// Probability of application.
    pub p: f64,
    /// `(min_factor, max_factor)` — e.g. `(0.5, 1.5)`.
    pub factor_range: (f64, f64),
}

impl DaAugment for DaBrightness {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let (lo, hi) = self.factor_range;
        let factor = lo + da_rand01(seed) * (hi - lo);
        let mut out = sample.clone_sample();
        for v in &mut out.data {
            *v *= factor;
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "Brightness"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

/// Contrast jitter: `x' = (x - mean) * factor + mean`.
#[derive(Debug, Clone)]
pub struct DaContrast {
    /// Probability of application.
    pub p: f64,
    /// `(min_factor, max_factor)` — e.g. `(0.5, 1.5)`.
    pub factor_range: (f64, f64),
}

impl DaAugment for DaContrast {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let (lo, hi) = self.factor_range;
        let factor = lo + da_rand01(seed) * (hi - lo);
        let n = sample.data.len() as f64;
        let mean = sample.data.iter().sum::<f64>() / n.max(1.0);
        let mut out = sample.clone_sample();
        for v in &mut out.data {
            *v = (*v - mean) * factor + mean;
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "Contrast"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

/// Add i.i.d. Gaussian noise `N(0, std^2)` to every element.
#[derive(Debug, Clone)]
pub struct DaGaussianNoise {
    /// Probability of application.
    pub p: f64,
    /// `(min_std, max_std)` — e.g. `(0.0, 0.1)`.
    pub std_range: (f64, f64),
}

impl DaAugment for DaGaussianNoise {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let (lo, hi) = self.std_range;
        let std = lo + da_rand01(seed) * (hi - lo);
        let mut out = sample.clone_sample();
        for v in &mut out.data {
            *v += std * da_randn(seed);
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "GaussianNoise"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

/// Gaussian blur applied independently per channel.
///
/// The 1D kernel `k[i] = exp(-i^2 / (2 * sigma^2))` (truncated, normalised)
/// is convolved along both spatial axes for 2D data.
#[derive(Debug, Clone)]
pub struct DaGaussianBlur {
    /// Probability of application.
    pub p: f64,
    /// `(min_sigma, max_sigma)`.
    pub sigma_range: (f64, f64),
    /// Length of the 1D kernel (should be odd).
    pub kernel_size: usize,
}

impl DaAugment for DaGaussianBlur {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let (lo, hi) = self.sigma_range;
        let sigma = lo + da_rand01(seed) * (hi - lo);
        if sigma < 1e-9 {
            return Ok(sample.clone_sample());
        }
        let ks = if self.kernel_size % 2 == 0 {
            self.kernel_size + 1
        } else {
            self.kernel_size
        };
        let half = (ks / 2) as i64;
        // Build normalised 1D Gaussian kernel.
        let raw_k: Vec<f64> = (0..ks)
            .map(|i| {
                let d = (i as i64 - half) as f64;
                (-(d * d) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let ksum: f64 = raw_k.iter().sum();
        let kernel: Vec<f64> = raw_k.iter().map(|v| v / ksum).collect();

        let (h, w) = sample.hw();
        let channels = sample.flat_len() / (h * w);
        let mut tmp = sample.data.clone();
        let mut out_data = vec![0.0f64; sample.flat_len()];

        // Convolve along W (columns) first.
        for c in 0..channels {
            for row in 0..h {
                for col in 0..w {
                    let mut acc = 0.0f64;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sc = col as i64 + ki as i64 - half;
                        let sc_clamped = sc.clamp(0, w as i64 - 1) as usize;
                        acc += kv * sample.data[c * h * w + row * w + sc_clamped];
                    }
                    tmp[c * h * w + row * w + col] = acc;
                }
            }
        }
        // Convolve along H (rows).
        for c in 0..channels {
            for row in 0..h {
                for col in 0..w {
                    let mut acc = 0.0f64;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sr = row as i64 + ki as i64 - half;
                        let sr_clamped = sr.clamp(0, h as i64 - 1) as usize;
                        acc += kv * tmp[c * h * w + sr_clamped * w + col];
                    }
                    out_data[c * h * w + row * w + col] = acc;
                }
            }
        }
        let mut out = DaSample::new(out_data, sample.shape.clone())?;
        out.label = sample.label;
        out.soft_label = sample.soft_label.clone();
        Ok(out)
    }

    fn name(&self) -> &str {
        "GaussianBlur"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

/// Randomly erase a rectangular region (Zhong et al. 2020).
///
/// The erased area is a fraction of total area in `scale`, with aspect ratio
/// drawn from `ratio` = `(min_r, max_r)`. Erased pixels are set to 0.
#[derive(Debug, Clone)]
pub struct DaRandomErasing {
    /// Probability of application.
    pub p: f64,
    /// `(min_area_frac, max_area_frac)` — e.g. `(0.02, 0.33)`.
    pub scale: (f64, f64),
    /// `(min_aspect_ratio, max_aspect_ratio)` — e.g. `(0.3, 3.3)`.
    pub ratio: (f64, f64),
}

impl DaAugment for DaRandomErasing {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= self.p {
            return Ok(sample.clone_sample());
        }
        let (h, w) = sample.hw();
        let total = (h * w) as f64;
        let mut out = sample.clone_sample();

        // Try up to 10 times to find a valid rectangle.
        for _ in 0..10 {
            let (s_lo, s_hi) = self.scale;
            let area = (s_lo + da_rand01(seed) * (s_hi - s_lo)) * total;
            let (r_lo, r_hi) = self.ratio;
            let asp = r_lo + da_rand01(seed) * (r_hi - r_lo);
            let rect_h = (area / asp).sqrt();
            let rect_w = (area * asp).sqrt();
            let rh = rect_h.round() as usize;
            let rw = rect_w.round() as usize;
            if rh == 0 || rw == 0 || rh > h || rw > w {
                continue;
            }
            let r0 = da_randint(seed, 0, (h - rh + 1) as i64) as usize;
            let c0 = da_randint(seed, 0, (w - rw + 1) as i64) as usize;
            let channels = out.flat_len() / (h * w);
            for c in 0..channels {
                for dr in 0..rh {
                    for dc in 0..rw {
                        let idx = c * h * w + (r0 + dr) * w + (c0 + dc);
                        out.data[idx] = 0.0;
                    }
                }
            }
            break;
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "RandomErasing"
    }
    fn probability(&self) -> f64 {
        self.p
    }
}

// ─── §5  DaMixup & DaCutmix ──────────────────────────────────────────────────

/// Convex combination of two samples (Zhang et al. 2018).
///
/// `x_mix = λ x1 + (1-λ) x2` where `λ ~ Beta(alpha, alpha)`.
#[derive(Debug, Clone)]
pub struct DaMixup {
    /// Concentration parameter of the Beta distribution.
    pub alpha: f64,
}

impl DaMixup {
    /// Create with given `alpha` (use `alpha = 0.4` for standard training).
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Sample `λ ~ Beta(alpha, alpha)` using a Gamma-ratio method.
    ///
    /// We approximate Beta(α, α) via two Gamma(α, 1) samples from the
    /// Marsaglia-Tsang GD algorithm (rejection sampling).
    pub fn beta_sample(alpha: f64, seed: &mut u64) -> f64 {
        if alpha <= 0.0 {
            return 0.5;
        }
        if (alpha - 1.0).abs() < 1e-9 {
            return da_rand01(seed);
        }
        let g1 = gamma_sample(alpha, seed);
        let g2 = gamma_sample(alpha, seed);
        let total = g1 + g2;
        if total < 1e-15 {
            0.5
        } else {
            g1 / total
        }
    }

    /// Mix two samples.  Both must have identical shapes.
    pub fn mix(&self, s1: &DaSample, s2: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if s1.shape != s2.shape {
            return Err(DaError::InvalidInput(format!(
                "shape mismatch: {:?} vs {:?}",
                s1.shape, s2.shape
            )));
        }
        let lam = Self::beta_sample(self.alpha, seed);
        let mixed: Vec<f64> = s1
            .data
            .iter()
            .zip(s2.data.iter())
            .map(|(a, b)| lam * a + (1.0 - lam) * b)
            .collect();
        let mut out = DaSample::new(mixed, s1.shape.clone())?;
        // Soft labels.
        let n_classes = s1
            .soft_label
            .as_ref()
            .map(|v| v.len())
            .or_else(|| s2.soft_label.as_ref().map(|v| v.len()))
            .unwrap_or(0);
        if n_classes > 0 {
            let y1 = one_hot_or_soft(&s1.soft_label, s1.label, n_classes);
            let y2 = one_hot_or_soft(&s2.soft_label, s2.label, n_classes);
            let soft: Vec<f64> = y1
                .iter()
                .zip(y2.iter())
                .map(|(a, b)| lam * a + (1.0 - lam) * b)
                .collect();
            out.soft_label = Some(soft);
        }
        Ok(out)
    }
}

/// CutMix: paste a rectangular region of `s2` into `s1` (Yun et al. 2019).
#[derive(Debug, Clone)]
pub struct DaCutmix {
    /// Concentration parameter for Beta area-ratio sampling.
    pub alpha: f64,
}

impl DaCutmix {
    /// Create with given `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Mix two samples by cutting a random region from `s2` into `s1`.
    pub fn mix(&self, s1: &DaSample, s2: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if s1.shape != s2.shape {
            return Err(DaError::InvalidInput(format!(
                "shape mismatch: {:?} vs {:?}",
                s1.shape, s2.shape
            )));
        }
        let (h, w) = s1.hw();
        let channels = s1.flat_len() / (h * w);
        let lam = DaMixup::beta_sample(self.alpha, seed);
        // Box dimensions based on lambda.
        let cut_ratio = (1.0 - lam).sqrt();
        let cut_h = ((h as f64 * cut_ratio).round() as usize).clamp(1, h);
        let cut_w = ((w as f64 * cut_ratio).round() as usize).clamp(1, w);
        let r0 = da_randint(seed, 0, (h - cut_h + 1).max(1) as i64) as usize;
        let c0 = da_randint(seed, 0, (w - cut_w + 1).max(1) as i64) as usize;
        let r1 = (r0 + cut_h).min(h);
        let c1 = (c0 + cut_w).min(w);
        let actual_area = (r1 - r0) * (c1 - c0);
        let actual_lam = 1.0 - (actual_area as f64 / (h * w) as f64);

        let mut out_data = s1.data.clone();
        for c in 0..channels {
            for row in r0..r1 {
                for col in c0..c1 {
                    let idx = c * h * w + row * w + col;
                    out_data[idx] = s2.data[idx];
                }
            }
        }
        let mut out = DaSample::new(out_data, s1.shape.clone())?;
        let n_classes = s1
            .soft_label
            .as_ref()
            .map(|v| v.len())
            .or_else(|| s2.soft_label.as_ref().map(|v| v.len()))
            .unwrap_or(0);
        if n_classes > 0 {
            let y1 = one_hot_or_soft(&s1.soft_label, s1.label, n_classes);
            let y2 = one_hot_or_soft(&s2.soft_label, s2.label, n_classes);
            let soft: Vec<f64> = y1
                .iter()
                .zip(y2.iter())
                .map(|(a, b)| actual_lam * a + (1.0 - actual_lam) * b)
                .collect();
            out.soft_label = Some(soft);
        }
        Ok(out)
    }
}

/// Batch-level Mixup/CutMix policy for training loops.
#[derive(Debug, Clone)]
pub struct DaMixupBatch {
    /// Underlying Mixup operator.
    pub mixup: DaMixup,
    /// Optional CutMix operator.
    pub cutmix: Option<DaCutmix>,
    /// Probability of using CutMix vs Mixup when both are enabled.
    pub cutmix_prob: f64,
}

impl DaMixupBatch {
    /// Create; set `use_cutmix = true` to enable CutMix as well.
    pub fn new(alpha: f64, use_cutmix: bool) -> Self {
        Self {
            mixup: DaMixup::new(alpha),
            cutmix: if use_cutmix {
                Some(DaCutmix::new(alpha))
            } else {
                None
            },
            cutmix_prob: 0.5,
        }
    }

    /// Augment a batch: for each sample, mix with a randomly chosen partner.
    pub fn augment_batch(
        &self,
        batch: &[DaSample],
        seed: &mut u64,
    ) -> Result<Vec<DaSample>, DaError> {
        let n = batch.len();
        if n == 0 {
            return Ok(vec![]);
        }
        // Random permutation.
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = da_randint(seed, 0, (i + 1) as i64) as usize;
            perm.swap(i, j);
        }
        let mut out = Vec::with_capacity(n);
        for (i, s1) in batch.iter().enumerate() {
            let s2 = &batch[perm[i]];
            let use_cut = self.cutmix.is_some() && da_rand01(seed) < self.cutmix_prob;
            let mixed = if use_cut {
                self.cutmix.as_ref().expect("cutmix checked Some above").mix(s1, s2, seed)?
            } else {
                self.mixup.mix(s1, s2, seed)?
            };
            out.push(mixed);
        }
        Ok(out)
    }
}

// ─── §6  DaAutoAugment ───────────────────────────────────────────────────────

/// A single AutoAugment operation (name + probability + magnitude).
#[derive(Debug, Clone)]
pub struct DaOperation {
    /// Name of the operation (e.g. "Rotate", "ShearX", "Brightness").
    pub name: String,
    /// Probability `p ∈ [0, 1]` of applying this operation.
    pub prob: f64,
    /// Magnitude parameter (interpretation depends on the operation).
    pub magnitude: f64,
}

/// Two operations forming one AutoAugment sub-policy.
#[derive(Debug, Clone)]
pub struct DaSubpolicy {
    /// First operation in the sub-policy.
    pub op1: DaOperation,
    /// Second operation in the sub-policy.
    pub op2: DaOperation,
}

/// AutoAugment policy (Cubuk et al. 2019).
///
/// Contains a list of sub-policies; at inference/training time one sub-policy
/// is selected uniformly at random and its two operations are applied
/// stochastically.
#[derive(Debug, Clone)]
pub struct DaAutoAugment {
    /// All sub-policies.
    pub subpolicies: Vec<DaSubpolicy>,
}

impl DaAutoAugment {
    /// The 25-subpolicy ImageNet policy from Cubuk et al. 2019, Table 3.
    pub fn imagenet_policy() -> Self {
        let ops: &[(&str, f64, f64, &str, f64, f64)] = &[
            ("Posterize", 0.4, 8.0, "Rotate", 0.6, 9.0),
            ("Solarize", 0.6, 5.0, "AutoContrast", 0.6, 5.0),
            ("Equalize", 0.8, 8.0, "Equalize", 0.6, 3.0),
            ("Posterize", 0.6, 7.0, "Posterize", 0.6, 6.0),
            ("Equalize", 0.4, 7.0, "Solarize", 0.2, 4.0),
            ("Equalize", 0.4, 4.0, "Rotate", 0.8, 8.0),
            ("Solarize", 0.6, 3.0, "Equalize", 0.6, 7.0),
            ("Posterize", 0.8, 5.0, "Equalize", 1.0, 2.0),
            ("Rotate", 0.2, 3.0, "Solarize", 0.6, 8.0),
            ("Equalize", 0.6, 8.0, "Posterize", 0.4, 6.0),
            ("Rotate", 0.8, 8.0, "Color", 1.0, 2.0),
            ("Rotate", 0.9, 9.0, "Equalize", 1.0, 2.0),
            ("ShearY", 0.2, 7.0, "Posterize", 0.3, 7.0),
            ("Color", 0.4, 3.0, "Brightness", 0.6, 7.0),
            ("Sharpness", 0.3, 9.0, "Brightness", 0.7, 9.0),
            ("Equalize", 0.6, 5.0, "Equalize", 0.5, 1.0),
            ("Contrast", 0.6, 7.0, "Sharpness", 0.6, 5.0),
            ("Color", 0.7, 7.0, "TranslateX", 0.5, 8.0),
            ("Equalize", 0.3, 7.0, "AutoContrast", 0.4, 8.0),
            ("TranslateY", 0.4, 3.0, "Sharpness", 0.2, 6.0),
            ("Brightness", 0.9, 6.0, "Color", 0.2, 8.0),
            ("Solarize", 0.5, 2.0, "Invert", 0.0, 3.0),
            ("Equalize", 0.2, 0.0, "AutoContrast", 0.6, 0.0),
            ("Equalize", 0.2, 8.0, "Equalize", 0.6, 4.0),
            ("Color", 0.9, 9.0, "Equalize", 0.6, 4.0),
        ];
        let subpolicies = ops
            .iter()
            .map(|&(n1, p1, m1, n2, p2, m2)| DaSubpolicy {
                op1: DaOperation {
                    name: n1.to_string(),
                    prob: p1,
                    magnitude: m1,
                },
                op2: DaOperation {
                    name: n2.to_string(),
                    prob: p2,
                    magnitude: m2,
                },
            })
            .collect();
        Self { subpolicies }
    }

    /// Select a random sub-policy and apply its two operations.
    pub fn apply_policy(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        if self.subpolicies.is_empty() {
            return Ok(sample.clone_sample());
        }
        let idx = da_randint(seed, 0, self.subpolicies.len() as i64) as usize;
        let sp = &self.subpolicies[idx];
        let s1 = self.apply_operation(sample, &sp.op1, seed)?;
        self.apply_operation(&s1, &sp.op2, seed)
    }

    /// Dispatch a named operation.
    pub fn apply_operation(
        &self,
        sample: &DaSample,
        op: &DaOperation,
        seed: &mut u64,
    ) -> Result<DaSample, DaError> {
        if da_rand01(seed) >= op.prob {
            return Ok(sample.clone_sample());
        }
        let m = op.magnitude;
        match op.name.as_str() {
            "Rotate" => DaRandomRotation {
                p: 1.0,
                max_degrees: m,
            }
            .apply(sample, seed),
            "ShearX" | "ShearY" | "Shear" => DaShear {
                p: 1.0,
                max_shear: m / 30.0,
            }
            .apply(sample, seed),
            "TranslateX" | "TranslateY" => {
                // Implement as a random crop with slight offset.
                let (h, w) = sample.hw();
                let shift = (m / 30.0 * w.min(h) as f64).round() as usize;
                let pad = shift + 1;
                DaRandomCrop {
                    p: 1.0,
                    crop_h: h,
                    crop_w: w,
                    pad,
                }
                .apply(sample, seed)
            }
            "AutoContrast" => {
                let min_v = sample.data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_v = sample
                    .data
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let range = (max_v - min_v).max(1e-9);
                let mut out = sample.clone_sample();
                for v in &mut out.data {
                    *v = (*v - min_v) / range;
                }
                Ok(out)
            }
            "Invert" => {
                let max_v = sample
                    .data
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut out = sample.clone_sample();
                for v in &mut out.data {
                    *v = max_v - *v;
                }
                Ok(out)
            }
            "Equalize" => {
                // Approximate histogram equalization on flattened data.
                let n = sample.data.len();
                let mut sorted = sample.data.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mut out = sample.clone_sample();
                for v in &mut out.data {
                    // rank / (n-1)
                    let pos = sorted.partition_point(|x| *x <= *v);
                    *v = pos.min(n - 1) as f64 / (n - 1).max(1) as f64;
                }
                Ok(out)
            }
            "Solarize" => {
                // Invert pixels above threshold (m / 10 of data range).
                let max_v = sample
                    .data
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let threshold = max_v * (1.0 - m / 10.0).clamp(0.0, 1.0);
                let mut out = sample.clone_sample();
                for v in &mut out.data {
                    if *v > threshold {
                        *v = max_v - *v;
                    }
                }
                Ok(out)
            }
            "Posterize" => {
                // Reduce to `bits` levels by right-shifting.
                let bits = (m.round() as u32).clamp(1, 8);
                let levels = (1u32 << bits) as f64;
                let min_v = sample.data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_v = sample
                    .data
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let range = (max_v - min_v).max(1e-9);
                let mut out = sample.clone_sample();
                for v in &mut out.data {
                    let norm = (*v - min_v) / range;
                    let q = (norm * levels).floor() / levels;
                    *v = q * range + min_v;
                }
                Ok(out)
            }
            "Brightness" | "Color" => {
                let factor = 1.0 + (m / 30.0 - 0.5) * 0.8;
                DaBrightness {
                    p: 1.0,
                    factor_range: (factor, factor),
                }
                .apply(sample, seed)
            }
            "Contrast" => {
                let factor = 1.0 + (m / 30.0 - 0.5) * 0.8;
                DaContrast {
                    p: 1.0,
                    factor_range: (factor, factor),
                }
                .apply(sample, seed)
            }
            "Sharpness" => {
                // Simple unsharp mask: x' = x + alpha*(x - blur(x))
                let alpha = m / 30.0;
                let blurred = DaGaussianBlur {
                    p: 1.0,
                    sigma_range: (1.0, 1.0),
                    kernel_size: 3,
                }
                .apply(sample, seed)?;
                let mut out = sample.clone_sample();
                for (i, v) in out.data.iter_mut().enumerate() {
                    *v += alpha * (*v - blurred.data[i]);
                }
                Ok(out)
            }
            _ => {
                // Unknown op: identity.
                Ok(sample.clone_sample())
            }
        }
    }
}

// ─── §7  DaRandAugment ───────────────────────────────────────────────────────

/// RandAugment (Cubuk et al. 2020): apply `n` uniformly-sampled operations,
/// each at the given `magnitude`.
#[derive(Debug, Clone)]
pub struct DaRandAugment {
    /// Number of operations to apply.
    pub n: usize,
    /// Magnitude in `[0, 30]`.
    pub magnitude: f64,
}

impl DaRandAugment {
    /// Create with given `n` and `magnitude`.
    pub fn new(n: usize, magnitude: f64) -> Self {
        Self { n, magnitude }
    }

    /// All available operation names.
    pub fn operations() -> Vec<String> {
        vec![
            "Identity",
            "AutoContrast",
            "Equalize",
            "Rotate",
            "Solarize",
            "Color",
            "Posterize",
            "Contrast",
            "Brightness",
            "Sharpness",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Pick `n` random operations and apply them sequentially.
    pub fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        let ops = Self::operations();
        let policy = DaAutoAugment {
            subpolicies: vec![],
        };
        let mut cur = sample.clone_sample();
        let n_ops = ops.len() as i64;
        for _ in 0..self.n {
            let idx = da_randint(seed, 0, n_ops) as usize;
            let op = DaOperation {
                name: ops[idx].clone(),
                prob: 1.0,
                magnitude: self.magnitude,
            };
            cur = policy.apply_operation(&cur, &op, seed)?;
        }
        Ok(cur)
    }
}

// ─── §8  DaTrivialAugment ────────────────────────────────────────────────────

/// TrivialAugment (Müller & Hutter 2021): sample one operation uniformly,
/// apply at a uniformly-sampled magnitude in `[0, 1]` (normalised to `[0, 30]`).
#[derive(Debug, Clone)]
pub struct DaTrivialAugment {
    /// Number of magnitude bins (unused in TrivialAugment; kept for API parity).
    pub num_magnitude_bins: usize,
}

impl DaTrivialAugment {
    /// Create with default 31 bins (standard TrivialAugment setting).
    pub fn new() -> Self {
        Self {
            num_magnitude_bins: 31,
        }
    }

    /// Apply one randomly-chosen operation at a uniformly-sampled magnitude.
    pub fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        let ops = DaRandAugment::operations();
        let idx = da_randint(seed, 0, ops.len() as i64) as usize;
        let magnitude = da_rand01(seed) * 30.0;
        let op = DaOperation {
            name: ops[idx].clone(),
            prob: 1.0,
            magnitude,
        };
        let policy = DaAutoAugment {
            subpolicies: vec![],
        };
        policy.apply_operation(sample, &op, seed)
    }
}

impl Default for DaTrivialAugment {
    fn default() -> Self {
        Self::new()
    }
}

// ─── §9  DaAugmentationPipeline ──────────────────────────────────────────────

/// Compose a sequence of augmentations, each applied independently with its
/// own probability.
pub struct DaAugmentationPipeline {
    /// Augmentation operators in application order.
    pub augmentations: Vec<Box<dyn DaAugment>>,
    /// Per-augmentation probabilities.
    pub p_each: Vec<f64>,
}

impl DaAugmentationPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            augmentations: vec![],
            p_each: vec![],
        }
    }

    /// Append an augmentation with the given probability.
    pub fn add<A: DaAugment + 'static>(&mut self, aug: A, prob: f64) {
        self.p_each.push(prob);
        self.augmentations.push(Box::new(aug));
    }

    /// Apply each augmentation independently (each with its probability).
    pub fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        let mut cur = sample.clone_sample();
        for (aug, &p) in self.augmentations.iter().zip(self.p_each.iter()) {
            if da_rand01(seed) < p {
                cur = aug.apply(&cur, seed)?;
            }
        }
        Ok(cur)
    }

    /// Build a pipeline that wraps a RandAugment policy as a single step.
    pub fn from_randaugment(n: usize, magnitude: f64) -> Self {
        let mut pipe = Self::new();
        pipe.add(
            DaRandAugmentWrapper {
                inner: DaRandAugment::new(n, magnitude),
            },
            1.0,
        );
        pipe
    }

    /// Build a pipeline that wraps the ImageNet AutoAugment policy.
    pub fn from_autoaugment() -> Self {
        let mut pipe = Self::new();
        pipe.add(
            DaAutoAugmentWrapper {
                inner: DaAutoAugment::imagenet_policy(),
            },
            1.0,
        );
        pipe
    }

    /// Apply the pipeline to every sample in a batch.
    pub fn augment_batch(
        &self,
        batch: &[DaSample],
        seed: &mut u64,
    ) -> Result<Vec<DaSample>, DaError> {
        let mut out = Vec::with_capacity(batch.len());
        for s in batch {
            out.push(self.apply(s, seed)?);
        }
        Ok(out)
    }
}

impl Default for DaAugmentationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// Thin wrappers to implement DaAugment for policy objects.
struct DaRandAugmentWrapper {
    inner: DaRandAugment,
}
impl DaAugment for DaRandAugmentWrapper {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        self.inner.apply(sample, seed)
    }
    fn name(&self) -> &str {
        "RandAugment"
    }
    fn probability(&self) -> f64 {
        1.0
    }
}

struct DaAutoAugmentWrapper {
    inner: DaAutoAugment,
}
impl DaAugment for DaAutoAugmentWrapper {
    fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        self.inner.apply_policy(sample, seed)
    }
    fn name(&self) -> &str {
        "AutoAugment"
    }
    fn probability(&self) -> f64 {
        1.0
    }
}

// ─── §10  DaDiffAugment ──────────────────────────────────────────────────────

/// Differentiable Augmentation policy selector.
#[derive(Debug, Clone)]
pub enum DaDiffPolicy {
    /// Brightness + saturation + contrast only.
    Color,
    /// Random integer pixel translation.
    Translation,
    /// Random 50%-area rectangular cutout.
    Cutout,
    /// All three: Color → Translation → Cutout.
    ColorTranslationCutout,
}

/// Differentiable Augmentation for GAN training (Zhao et al. 2020).
#[derive(Debug, Clone)]
pub struct DaDiffAugment {
    /// Which differentiable policy to apply.
    pub policy: DaDiffPolicy,
}

impl DaDiffAugment {
    /// Create with the given policy.
    pub fn new(policy: DaDiffPolicy) -> Self {
        Self { policy }
    }

    /// Apply the selected policy.
    pub fn apply(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        match &self.policy {
            DaDiffPolicy::Color => self.color_augment(sample, seed),
            DaDiffPolicy::Translation => self.translation_augment(sample, seed),
            DaDiffPolicy::Cutout => self.cutout_augment(sample, seed),
            DaDiffPolicy::ColorTranslationCutout => {
                let s1 = self.color_augment(sample, seed)?;
                let s2 = self.translation_augment(&s1, seed)?;
                self.cutout_augment(&s2, seed)
            }
        }
    }

    /// Color jitter: brightness → saturation → contrast.
    pub fn color_augment(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        // Brightness: random factor in [0.5, 1.5].
        let b_factor = 0.5 + da_rand01(seed);
        let mut out = sample.clone_sample();
        for v in &mut out.data {
            *v *= b_factor;
        }
        // Saturation: uniform mix with mean (approximate for multi-channel).
        let (h, w) = sample.hw();
        let channels = out.flat_len() / (h * w);
        if channels > 1 {
            let s_factor = 0.5 + da_rand01(seed);
            for pixel in 0..(h * w) {
                let mean_c: f64 = (0..channels)
                    .map(|c| out.data[c * h * w + pixel])
                    .sum::<f64>()
                    / channels as f64;
                for c in 0..channels {
                    let v = &mut out.data[c * h * w + pixel];
                    *v = mean_c + s_factor * (*v - mean_c);
                }
            }
        }
        // Contrast: (x - mean) * factor + mean.
        let c_factor = 0.5 + da_rand01(seed);
        let mean: f64 = out.data.iter().sum::<f64>() / out.data.len().max(1) as f64;
        for v in &mut out.data {
            *v = (*v - mean) * c_factor + mean;
        }
        Ok(out)
    }

    /// Integer-shift translation (pad with 0, then random crop).
    pub fn translation_augment(
        &self,
        sample: &DaSample,
        seed: &mut u64,
    ) -> Result<DaSample, DaError> {
        let (h, w) = sample.hw();
        let max_shift = ((h.min(w) as f64) * 0.125).ceil() as usize + 1;
        DaRandomCrop {
            p: 1.0,
            crop_h: h,
            crop_w: w,
            pad: max_shift,
        }
        .apply(sample, seed)
    }

    /// Random cutout covering ~50% area.
    pub fn cutout_augment(&self, sample: &DaSample, seed: &mut u64) -> Result<DaSample, DaError> {
        DaRandomErasing {
            p: 1.0,
            scale: (0.45, 0.55),
            ratio: (0.9, 1.1),
        }
        .apply(sample, seed)
    }
}

// ─── §11  DaMetrics ──────────────────────────────────────────────────────────

/// Metrics for analysing the quality and diversity of data augmentation.
pub struct DaMetrics;

impl DaMetrics {
    /// Mean pairwise L2 distance between all augmented versions and the original.
    pub fn augmentation_diversity(original: &DaSample, augmented: &[DaSample]) -> f64 {
        if augmented.is_empty() {
            return 0.0;
        }
        let sum: f64 = augmented
            .iter()
            .map(|a| l2_dist(&original.data, &a.data))
            .sum();
        sum / augmented.len() as f64
    }

    /// Fraction of samples where the hard label is preserved.
    ///
    /// For non-mixing augmentations this should be 1.0.
    pub fn label_preserving_ratio(original_labels: &[usize], augmented_labels: &[usize]) -> f64 {
        if original_labels.is_empty() {
            return 1.0;
        }
        let matches = original_labels
            .iter()
            .zip(augmented_labels.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / original_labels.len() as f64
    }

    /// Normalised L2 distance: `||x - x'||_2 / ||x||_2`.
    ///
    /// Returns 0 when the original has zero norm.
    pub fn augmentation_strength(original: &DaSample, augmented: &DaSample) -> f64 {
        let norm_orig = l2_norm(&original.data);
        if norm_orig < 1e-15 {
            return 0.0;
        }
        l2_dist(&original.data, &augmented.data) / norm_orig
    }

    /// Entropy of the sample distribution in feature space using a histogram.
    ///
    /// Values are binned per-dimension; returns the mean per-dimension entropy
    /// in `[0, 1]` (normalised by `log2(n_bins)`).
    pub fn coverage_of_space(samples: &[DaSample], n_bins: usize) -> f64 {
        if samples.is_empty() || n_bins == 0 {
            return 0.0;
        }
        let n_bins = n_bins.max(2);
        let dim = samples[0].flat_len();
        if dim == 0 {
            return 0.0;
        }
        // Compute per-dimension entropy and average.
        let mut total_entropy = 0.0f64;
        for d in 0..dim {
            // Collect all values for this dimension.
            let vals: Vec<f64> = samples.iter().map(|s| s.data[d]).collect();
            let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max_v - min_v).max(1e-15);
            let mut counts = vec![0usize; n_bins];
            for v in &vals {
                let bin = (((*v - min_v) / range) * (n_bins as f64 - 1e-9)).floor() as usize;
                let bin = bin.min(n_bins - 1);
                counts[bin] += 1;
            }
            let n_samp = vals.len() as f64;
            let entropy: f64 = counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / n_samp;
                    -p * p.log2()
                })
                .sum();
            let max_entropy = (n_bins as f64).log2().max(1e-9);
            total_entropy += (entropy / max_entropy).clamp(0.0, 1.0);
        }
        total_entropy / dim as f64
    }
}

// ─── §12  Internal helpers ────────────────────────────────────────────────────

/// Compute L2 norm of a vector.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute L2 distance between two equal-length vectors.
fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Return the soft label vector, falling back to a one-hot encoding from the
/// hard label if no soft label is available.
fn one_hot_or_soft(soft: &Option<Vec<f64>>, label: Option<usize>, n_classes: usize) -> Vec<f64> {
    if let Some(sl) = soft {
        if sl.len() == n_classes {
            return sl.clone();
        }
    }
    let mut oh = vec![0.0f64; n_classes];
    if let Some(l) = label {
        if l < n_classes {
            oh[l] = 1.0;
        }
    }
    oh
}

/// Gamma(shape, 1) sampler via Marsaglia-Tsang GD algorithm.
///
/// Valid for `shape > 1/3`. For `shape <= 1/3` we fall back to a squeeze.
fn gamma_sample(shape: f64, seed: &mut u64) -> f64 {
    if shape <= 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        // Use the relation: Gamma(α) = Gamma(α+1) * U^(1/α)
        let g = gamma_sample(shape + 1.0, seed);
        let u = da_rand01(seed).max(1e-15);
        return g * u.powf(1.0 / shape);
    }
    // Marsaglia-Tsang GD (2000).
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = da_randn(seed);
        let v_raw = 1.0 + c * x;
        if v_raw <= 0.0 {
            continue;
        }
        let v = v_raw * v_raw * v_raw;
        let u = da_rand01(seed).max(1e-15);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * (x2 * x2) {
            return d * v;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}
