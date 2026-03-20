//! Satellite & Remote Sensing Machine Learning Components.
//!
//! Sections:
//! 1. SpectralData  — multispectral/hyperspectral image representation & indices
//! 2. ChangeDetection — pixel-wise and neural change detection
//! 3. SarProcessor   — SAR speckle filtering, polarimetric decomposition, classification
//! 4. LandCoverClassification — pixel & object-based LULC
//! 5. SuperResolution — satellite image SR with residual blocks & pan-sharpening
//! 6. GeoAiUtils     — haversine, UTM projection, Web Mercator tile index
//! 7. SatelliteMetrics — OA, kappa, F1, PSNR, SSIM

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ── Small shared math helpers ─────────────────────────────────────────────────

#[inline]
fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-88.0, 88.0)).exp())
}

fn softmax_vec(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(f32::EPSILON);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Dense layer: weights layout [out_dim × in_dim] (row-major)
fn dense_linear(weights: &[Vec<f32>], bias: &[f32], input: &[f32]) -> Vec<f32> {
    weights
        .iter()
        .enumerate()
        .map(|(o, row)| {
            let b = bias.get(o).copied().unwrap_or(0.0);
            row.iter()
                .zip(input.iter())
                .fold(b, |acc, (w, x)| acc + w * x)
        })
        .collect()
}

fn dense_relu(weights: &[Vec<f32>], bias: &[f32], input: &[f32]) -> Vec<f32> {
    dense_linear(weights, bias, input)
        .into_iter()
        .map(relu_f32)
        .collect()
}

/// Xavier-uniform initialiser (fan_in × fan_out matrix → vec of rows)
fn xavier_rows(fan_in: usize, fan_out: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    (0..fan_out)
        .map(|_| {
            (0..fan_in)
                .map(|_| {
                    let u: f32 = rng.random();
                    u * 2.0 * limit - limit
                })
                .collect()
        })
        .collect()
}

fn zeros_vec(n: usize) -> Vec<f32> {
    vec![0.0_f32; n]
}

// ═════════════════════════════════════════════════════════════════════════════
// §1  SpectralData
// ═════════════════════════════════════════════════════════════════════════════

/// Canonical satellite spectral bands.
#[derive(Clone, Debug, PartialEq)]
pub enum SpectralBand {
    Blue,    // ~0.49 µm
    Green,   // ~0.56 µm
    Red,     // ~0.665 µm
    Nir,     // ~0.842 µm
    Swir1,   // ~1.61 µm
    Swir2,   // ~2.19 µm
    Thermal, // ~10.9 µm
}

/// Multispectral image with B bands, H rows, W columns.
/// Storage: `bands[b][h * width + w]`
#[derive(Clone, Debug)]
pub struct MultispectralImage {
    pub bands: Vec<Vec<f32>>,
    pub n_bands: usize,
    pub height: usize,
    pub width: usize,
}

impl MultispectralImage {
    /// Construct from a flat vector of band images, each of length `height * width`.
    pub fn new(bands: Vec<Vec<f32>>, height: usize, width: usize) -> Self {
        let n_bands = bands.len();
        Self {
            bands,
            n_bands,
            height,
            width,
        }
    }

    /// Spectrum at pixel (h, w): length == n_bands.
    pub fn pixel_spectrum(&self, h: usize, w: usize) -> Vec<f32> {
        let idx = h * self.width + w;
        self.bands
            .iter()
            .map(|b| b.get(idx).copied().unwrap_or(0.0))
            .collect()
    }

    /// Flat pixel data for band `b`.
    pub fn band(&self, b: usize) -> &[f32] {
        if b < self.n_bands {
            &self.bands[b]
        } else {
            &[]
        }
    }

    /// (n_bands, height, width)
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n_bands, self.height, self.width)
    }
}

/// Common remote-sensing spectral indices.
pub struct SpectralIndices;

impl SpectralIndices {
    const EPS: f32 = 1e-7;

    pub fn ndvi(red: f32, nir: f32) -> f32 {
        (nir - red) / (nir + red + Self::EPS)
    }

    pub fn ndwi(green: f32, nir: f32) -> f32 {
        (green - nir) / (green + nir + Self::EPS)
    }

    pub fn ndbi(swir1: f32, nir: f32) -> f32 {
        (swir1 - nir) / (swir1 + nir + Self::EPS)
    }

    /// Enhanced Vegetation Index.
    pub fn evi(blue: f32, red: f32, nir: f32) -> f32 {
        let denom = nir + 6.0 * red - 7.5 * blue + 1.0;
        if denom.abs() < Self::EPS {
            0.0
        } else {
            2.5 * (nir - red) / denom
        }
    }

    /// Compute per-pixel NDVI map from a `MultispectralImage`.
    pub fn compute_ndvi_map(
        img: &MultispectralImage,
        red_band: usize,
        nir_band: usize,
    ) -> Vec<f32> {
        let n = img.height * img.width;
        let red = img.band(red_band);
        let nir = img.band(nir_band);
        (0..n)
            .map(|i| {
                let r = red.get(i).copied().unwrap_or(0.0);
                let ni = nir.get(i).copied().unwrap_or(0.0);
                Self::ndvi(r, ni)
            })
            .collect()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §2  ChangeDetection
// ═════════════════════════════════════════════════════════════════════════════

/// Per-pixel change detection result.
#[derive(Clone, Debug)]
pub struct ChangeMap {
    pub changed: Vec<bool>,
    pub height: usize,
    pub width: usize,
    pub change_score: Vec<f32>,
}

/// Image differencing change detector.
pub struct DifferenceCD;

impl DifferenceCD {
    /// Detect changes using L2 per-pixel difference and Otsu thresholding.
    pub fn compute(img1: &MultispectralImage, img2: &MultispectralImage) -> ChangeMap {
        let n = img1.height * img1.width;
        let mut scores: Vec<f32> = (0..n)
            .map(|i| {
                let mut d2 = 0.0_f32;
                for b in 0..img1.n_bands.min(img2.n_bands) {
                    let v1 = img1.bands[b].get(i).copied().unwrap_or(0.0);
                    let v2 = img2.bands[b].get(i).copied().unwrap_or(0.0);
                    let diff = v2 - v1;
                    d2 += diff * diff;
                }
                d2.sqrt()
            })
            .collect();

        let threshold = Self::threshold_otsu(&scores);
        let changed = scores.iter().map(|&s| s > threshold).collect();

        ChangeMap {
            changed,
            height: img1.height,
            width: img1.width,
            change_score: scores,
        }
    }

    /// Otsu's method: maximise inter-class variance over 256 histogram bins.
    pub fn threshold_otsu(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if (max - min).abs() < f32::EPSILON {
            return min;
        }
        let bins = 256_usize;
        let mut hist = vec![0_u64; bins];
        for &v in values {
            let idx = (((v - min) / (max - min)) * (bins as f32 - 1.0)) as usize;
            hist[idx.min(bins - 1)] += 1;
        }
        let n = values.len() as f64;
        // total mean
        let mut total_sum = 0.0_f64;
        for (i, &h) in hist.iter().enumerate() {
            total_sum += i as f64 * h as f64;
        }
        let mut best_var = 0.0_f64;
        let mut best_t = 0_usize;
        let mut w0 = 0_u64;
        let mut sum_fg = 0.0_f64;
        for t in 0..bins {
            w0 += hist[t];
            sum_fg += t as f64 * hist[t] as f64;
            let w1 = n as u64 - w0;
            if w0 == 0 || w1 == 0 {
                continue;
            }
            let mu0 = sum_fg / w0 as f64;
            let mu1 = (total_sum - sum_fg) / w1 as f64;
            let var = w0 as f64 * w1 as f64 * (mu0 - mu1).powi(2);
            if var > best_var {
                best_var = var;
                best_t = t;
            }
        }
        min + (best_t as f32 / (bins as f32 - 1.0)) * (max - min)
    }
}

/// Change Vector Analysis.
pub struct CvaChangeDetection;

impl CvaChangeDetection {
    /// L2 magnitude of the change vector between two spectra.
    pub fn magnitude(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| {
                let d = b - a;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Direction (angle in degrees) of the 2D projection of the change vector.
    pub fn direction(v1: &[f32], v2: &[f32]) -> f32 {
        let dx = v2.first().copied().unwrap_or(0.0) - v1.first().copied().unwrap_or(0.0);
        let dy = v2.get(1).copied().unwrap_or(0.0) - v1.get(1).copied().unwrap_or(0.0);
        dy.atan2(dx).to_degrees()
    }

    /// Detect changes: pixels whose magnitude exceeds `threshold` are marked changed.
    pub fn detect(
        img1: &MultispectralImage,
        img2: &MultispectralImage,
        threshold: f32,
    ) -> ChangeMap {
        let n = img1.height * img1.width;
        let scores: Vec<f32> = (0..n)
            .map(|i| {
                let sp1 = img1.pixel_spectrum(i / img1.width, i % img1.width);
                let sp2 = img2.pixel_spectrum(i / img2.width, i % img2.width);
                Self::magnitude(&sp1, &sp2)
            })
            .collect();
        let changed = scores.iter().map(|&s| s > threshold).collect();
        ChangeMap {
            changed,
            height: img1.height,
            width: img1.width,
            change_score: scores,
        }
    }
}

/// Small MLP-based change detector.
pub struct NeuralChangeDetector {
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

impl NeuralChangeDetector {
    /// Build a 2-layer MLP; input dim = 2 * n_bands.
    pub fn new(n_bands: usize, hidden: usize, rng: &mut impl Rng) -> Self {
        let input_dim = 2 * n_bands;
        let l1_w = xavier_rows(input_dim, hidden, rng);
        let l1_b = zeros_vec(hidden);
        let l2_w = xavier_rows(hidden, 1, rng);
        let l2_b = zeros_vec(1);
        Self {
            layers: vec![(l1_w, l1_b), (l2_w, l2_b)],
        }
    }

    /// Forward pass returning change probability in [0, 1].
    pub fn predict(&self, spectrum1: &[f32], spectrum2: &[f32]) -> f32 {
        let mut input: Vec<f32> = spectrum1.iter().chain(spectrum2.iter()).cloned().collect();
        for (idx, (w, b)) in self.layers.iter().enumerate() {
            if idx < self.layers.len() - 1 {
                input = dense_relu(w, b, &input);
            } else {
                let out = dense_linear(w, b, &input);
                input = vec![sigmoid_f32(out.first().copied().unwrap_or(0.0))];
            }
        }
        input.first().copied().unwrap_or(0.0)
    }

    /// Produce a ChangeMap using a 0.5 threshold on the predicted probability.
    pub fn detect_map(&self, img1: &MultispectralImage, img2: &MultispectralImage) -> ChangeMap {
        let n = img1.height * img1.width;
        let scores: Vec<f32> = (0..n)
            .map(|i| {
                let sp1 = img1.pixel_spectrum(i / img1.width, i % img1.width);
                let sp2 = img2.pixel_spectrum(i / img2.width, i % img2.width);
                self.predict(&sp1, &sp2)
            })
            .collect();
        let changed = scores.iter().map(|&s| s > 0.5).collect();
        ChangeMap {
            changed,
            height: img1.height,
            width: img1.width,
            change_score: scores,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §3  SarProcessor
// ═════════════════════════════════════════════════════════════════════════════

/// Single-polarisation or single-look complex SAR image.
#[derive(Clone, Debug)]
pub struct SarImage {
    pub amplitude: Vec<f32>,
    pub phase: Option<Vec<f32>>,
    pub height: usize,
    pub width: usize,
}

impl SarImage {
    pub fn new(amplitude: Vec<f32>, phase: Option<Vec<f32>>, height: usize, width: usize) -> Self {
        Self {
            amplitude,
            phase,
            height,
            width,
        }
    }
}

/// SAR speckle filtering methods.
pub struct SpeckleFilter;

impl SpeckleFilter {
    /// Lee filter: local-statistics adaptive filter.
    pub fn lee_filter(image: &SarImage, window_size: usize) -> SarImage {
        let h = image.height;
        let w = image.width;
        let amp = &image.amplitude;
        let half = window_size / 2;
        let n = h * w;

        // Estimate global noise variance from the whole image
        let global_mean: f32 = amp.iter().sum::<f32>() / n.max(1) as f32;
        let global_var: f32 =
            amp.iter().map(|&v| (v - global_mean).powi(2)).sum::<f32>() / n.max(1) as f32;
        let noise_var = global_var / (global_mean * global_mean).max(f32::EPSILON);

        let mut out = vec![0.0_f32; n];

        for row in 0..h {
            for col in 0..w {
                let r_lo = row.saturating_sub(half);
                let r_hi = (row + half + 1).min(h);
                let c_lo = col.saturating_sub(half);
                let c_hi = (col + half + 1).min(w);

                let mut sum = 0.0_f32;
                let mut sq_sum = 0.0_f32;
                let mut cnt = 0_usize;
                for ri in r_lo..r_hi {
                    for ci in c_lo..c_hi {
                        let v = amp[ri * w + ci];
                        sum += v;
                        sq_sum += v * v;
                        cnt += 1;
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let local_mean = sum / cnt as f32;
                let local_var = (sq_sum / cnt as f32 - local_mean * local_mean).max(0.0);
                let sigma_v_sq = local_var / (local_mean * local_mean).max(f32::EPSILON);
                let weight = (sigma_v_sq - noise_var) / (sigma_v_sq + 1.0).max(f32::EPSILON);
                let weight = weight.clamp(0.0, 1.0);

                let center = amp[row * w + col];
                out[row * w + col] = local_mean + weight * (center - local_mean);
            }
        }
        SarImage::new(out, image.phase.clone(), h, w)
    }

    /// Median filter using a sliding window.
    pub fn median_filter(image: &SarImage, window_size: usize) -> SarImage {
        let h = image.height;
        let w = image.width;
        let amp = &image.amplitude;
        let half = window_size / 2;
        let n = h * w;
        let mut out = vec![0.0_f32; n];

        for row in 0..h {
            for col in 0..w {
                let r_lo = row.saturating_sub(half);
                let r_hi = (row + half + 1).min(h);
                let c_lo = col.saturating_sub(half);
                let c_hi = (col + half + 1).min(w);

                let mut vals: Vec<f32> = Vec::with_capacity(window_size * window_size);
                for ri in r_lo..r_hi {
                    for ci in c_lo..c_hi {
                        vals.push(amp[ri * w + ci]);
                    }
                }
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = vals.len() / 2;
                out[row * w + col] = vals.get(mid).copied().unwrap_or(0.0);
            }
        }
        SarImage::new(out, image.phase.clone(), h, w)
    }
}

/// Polarimetric decomposition utilities.
pub struct PolarimetricDecomposition;

impl PolarimetricDecomposition {
    /// Pauli decomposition: returns [|HH+VV|²/2, |HH-VV|²/2, |HV|²] as 3 channels.
    pub fn pauli_decomposition(hh: &[f32], hv: &[f32], vv: &[f32]) -> Vec<Vec<f32>> {
        let n = hh.len().min(hv.len()).min(vv.len());
        let mut alpha = Vec::with_capacity(n);
        let mut beta = Vec::with_capacity(n);
        let mut gamma = Vec::with_capacity(n);
        for i in 0..n {
            let a = (hh[i] + vv[i]) / std::f32::consts::SQRT_2;
            let b = (hh[i] - vv[i]) / std::f32::consts::SQRT_2;
            alpha.push(a * a * 0.5);
            beta.push(b * b * 0.5);
            gamma.push(hv[i] * hv[i]);
        }
        vec![alpha, beta, gamma]
    }

    /// H/A/α decomposition via simplified power iteration on each pixel's 3×3 coherency matrix.
    /// Input: slice of n pixel 3×3 matrices (row-major).
    /// Returns: (entropy, anisotropy, alpha_deg)
    pub fn entropy_anisotropy_alpha(coherency_matrix: &[Vec<Vec<f32>>]) -> (f32, f32, f32) {
        if coherency_matrix.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        // Average over pixels
        let n = coherency_matrix.len();
        let size = 3_usize;
        let mut mean_t = vec![vec![0.0_f32; size]; size];
        for cm in coherency_matrix {
            for r in 0..size.min(cm.len()) {
                for c in 0..size.min(cm[r].len()) {
                    mean_t[r][c] += cm[r][c] / n as f32;
                }
            }
        }

        // Approximate 3 eigenvalues via a simplified power-iteration approach:
        // use diagonal elements as proxies (exact for diagonal T), then normalise.
        let d0 = mean_t[0][0].max(0.0);
        let d1 = mean_t[1][1].max(0.0);
        let d2 = mean_t[2][2].max(0.0);
        let sum = (d0 + d1 + d2).max(f32::EPSILON);
        let p0 = d0 / sum;
        let p1 = d1 / sum;
        let p2 = d2 / sum;

        // Shannon entropy H = -Σ p_i log3(p_i)
        let log3_inv = 1.0 / (3.0_f32).ln();
        let h_entropy = [p0, p1, p2].iter().fold(0.0_f32, |acc, &p| {
            if p < f32::EPSILON {
                acc
            } else {
                acc - p * p.ln() * log3_inv
            }
        });

        // Anisotropy A = (λ1 - λ2) / (λ1 + λ2) using the two smallest eigenvalues
        let mut eigs = [d0, d1, d2];
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let aniso = if (eigs[1] + eigs[2]).abs() < f32::EPSILON {
            0.0
        } else {
            (eigs[1] - eigs[2]) / (eigs[1] + eigs[2])
        };

        // Mean scattering angle α (degrees): dominant mechanism
        // Approximate: 0° surface, 45° dipole, 90° volume → weighted by p_i
        let alpha_deg = p0 * 0.0 + p1 * 45.0 + p2 * 90.0;

        (h_entropy.clamp(0.0, 1.0), aniso.clamp(0.0, 1.0), alpha_deg)
    }
}

/// Linear SAR feature classifier (multi-class).
pub struct SarClassifier {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl SarClassifier {
    pub fn new(feature_dim: usize, n_classes: usize, rng: &mut impl Rng) -> Self {
        let weights = xavier_rows(feature_dim, n_classes, rng);
        let bias = zeros_vec(n_classes);
        Self { weights, bias }
    }

    /// Softmax class probabilities.
    pub fn classify(&self, features: &[f32]) -> Vec<f32> {
        let mut logits = dense_linear(&self.weights, &self.bias, features);
        softmax_vec(&mut logits);
        logits
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §4  LandCoverClassification
// ═════════════════════════════════════════════════════════════════════════════

/// LULC class taxonomy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LandCoverClass {
    Water,
    Forest,
    Grassland,
    Cropland,
    Urban,
    Barren,
    Snow,
    Wetland,
}

impl LandCoverClass {
    fn all_classes() -> [LandCoverClass; 8] {
        [
            LandCoverClass::Water,
            LandCoverClass::Forest,
            LandCoverClass::Grassland,
            LandCoverClass::Cropland,
            LandCoverClass::Urban,
            LandCoverClass::Barren,
            LandCoverClass::Snow,
            LandCoverClass::Wetland,
        ]
    }

    fn index(&self) -> usize {
        match self {
            LandCoverClass::Water => 0,
            LandCoverClass::Forest => 1,
            LandCoverClass::Grassland => 2,
            LandCoverClass::Cropland => 3,
            LandCoverClass::Urban => 4,
            LandCoverClass::Barren => 5,
            LandCoverClass::Snow => 6,
            LandCoverClass::Wetland => 7,
        }
    }

    fn from_index(i: usize) -> LandCoverClass {
        match i {
            0 => LandCoverClass::Water,
            1 => LandCoverClass::Forest,
            2 => LandCoverClass::Grassland,
            3 => LandCoverClass::Cropland,
            4 => LandCoverClass::Urban,
            5 => LandCoverClass::Barren,
            6 => LandCoverClass::Snow,
            _ => LandCoverClass::Wetland,
        }
    }
}

/// Dense map of LULC classes per pixel.
#[derive(Clone, Debug)]
pub struct LandCoverMap {
    pub classes: Vec<LandCoverClass>,
    pub height: usize,
    pub width: usize,
}

/// 2-layer MLP pixel classifier.
pub struct PixelClassifier {
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    pub n_classes: usize,
}

impl PixelClassifier {
    pub fn new(input_dim: usize, hidden_dim: usize, n_classes: usize, rng: &mut impl Rng) -> Self {
        let l1_w = xavier_rows(input_dim, hidden_dim, rng);
        let l1_b = zeros_vec(hidden_dim);
        let l2_w = xavier_rows(hidden_dim, n_classes, rng);
        let l2_b = zeros_vec(n_classes);
        Self {
            layers: vec![(l1_w, l1_b), (l2_w, l2_b)],
            n_classes,
        }
    }

    fn forward(&self, spectrum: &[f32]) -> Vec<f32> {
        let mut x = spectrum.to_vec();
        for (idx, (w, b)) in self.layers.iter().enumerate() {
            if idx < self.layers.len() - 1 {
                x = dense_relu(w, b, &x);
            } else {
                x = dense_linear(w, b, &x);
                softmax_vec(&mut x);
            }
        }
        x
    }

    pub fn predict(&self, spectrum: &[f32]) -> LandCoverClass {
        let probs = self.forward(spectrum);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        LandCoverClass::from_index(best % self.n_classes)
    }

    pub fn classify_image(&self, img: &MultispectralImage) -> LandCoverMap {
        let n = img.height * img.width;
        let classes = (0..n)
            .map(|i| {
                let sp = img.pixel_spectrum(i / img.width, i % img.width);
                self.predict(&sp)
            })
            .collect();
        LandCoverMap {
            classes,
            height: img.height,
            width: img.width,
        }
    }
}

/// Object-based image analysis utilities.
pub struct ObjectBasedSegmentation;

impl ObjectBasedSegmentation {
    /// Simple SLIC-style segmentation: k-means in spectral+spatial space.
    /// Returns segment id per pixel (0..n_segments).
    pub fn simple_slic(img: &MultispectralImage, n_segments: usize) -> Vec<usize> {
        let h = img.height;
        let w = img.width;
        let n_pixels = h * w;
        let n_seg = n_segments.max(1);

        // Initialise cluster centres on a grid
        let mut centers: Vec<Vec<f32>> = Vec::with_capacity(n_seg);
        let step = (n_pixels as f64 / n_seg as f64).sqrt() as usize + 1;
        let mut seen = 0_usize;
        'outer: for ri in (0..h).step_by(step.max(1)) {
            for ci in (0..w).step_by(step.max(1)) {
                let mut feat: Vec<f32> = img.pixel_spectrum(ri, ci);
                // Append normalised spatial features
                feat.push(ri as f32 / h.max(1) as f32);
                feat.push(ci as f32 / w.max(1) as f32);
                centers.push(feat);
                seen += 1;
                if seen >= n_seg {
                    break 'outer;
                }
            }
        }
        // Pad if fewer grid points than n_seg
        while centers.len() < n_seg {
            centers.push(
                centers
                    .last()
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; img.n_bands + 2]),
            );
        }

        let feat_dim = img.n_bands + 2;
        let mut labels = vec![0_usize; n_pixels];

        // k-means iterations
        for _iter in 0..10 {
            // Assignment
            for i in 0..n_pixels {
                let ri = i / w;
                let ci = i % w;
                let mut feat: Vec<f32> = img.pixel_spectrum(ri, ci);
                feat.push(ri as f32 / h.max(1) as f32);
                feat.push(ci as f32 / w.max(1) as f32);

                let best = centers
                    .iter()
                    .enumerate()
                    .map(|(k, c)| {
                        let d: f32 = feat
                            .iter()
                            .zip(c.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (k, d)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, _)| k)
                    .unwrap_or(0);
                labels[i] = best;
            }

            // Update centres
            let mut sums = vec![vec![0.0_f32; feat_dim]; n_seg];
            let mut counts = vec![0_usize; n_seg];
            for i in 0..n_pixels {
                let ri = i / w;
                let ci = i % w;
                let mut feat: Vec<f32> = img.pixel_spectrum(ri, ci);
                feat.push(ri as f32 / h.max(1) as f32);
                feat.push(ci as f32 / w.max(1) as f32);
                let k = labels[i];
                for d in 0..feat_dim {
                    sums[k][d] += feat.get(d).copied().unwrap_or(0.0);
                }
                counts[k] += 1;
            }
            for k in 0..n_seg {
                if counts[k] > 0 {
                    for d in 0..feat_dim {
                        centers[k][d] = sums[k][d] / counts[k] as f32;
                    }
                }
            }
        }

        labels
    }

    /// Compute mean spectrum per segment.
    /// Returns a Vec of length n_segments, each entry being a mean spectrum `Vec<f32>`.
    pub fn segment_statistics(img: &MultispectralImage, segments: &[usize]) -> Vec<Vec<f32>> {
        let n_pixels = img.height * img.width;
        let n_seg = segments.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
        if n_seg == 0 {
            return Vec::new();
        }
        let mut sums = vec![vec![0.0_f32; img.n_bands]; n_seg];
        let mut counts = vec![0_usize; n_seg];
        for i in 0..n_pixels.min(segments.len()) {
            let sp = img.pixel_spectrum(i / img.width, i % img.width);
            let k = segments[i];
            if k < n_seg {
                for (b, v) in sp.iter().enumerate() {
                    sums[k][b] += v;
                }
                counts[k] += 1;
            }
        }
        sums.iter_mut()
            .zip(counts.iter())
            .map(|(s, &c)| {
                if c > 0 {
                    s.iter().map(|&v| v / c as f32).collect()
                } else {
                    vec![0.0; img.n_bands]
                }
            })
            .collect()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §5  SuperResolution
// ═════════════════════════════════════════════════════════════════════════════

/// Configuration for the satellite super-resolution model.
#[derive(Clone, Debug)]
pub struct SrConfig {
    pub scale_factor: usize,
    pub n_features: usize,
    pub n_residual_blocks: usize,
}

/// A single residual block: Conv → ReLU → Conv + skip.
/// Weights: [n_features × n_features] each, stored as row-major Vec<`Vec<f32>>`.
pub struct SatResidualBlock {
    pub conv1: Vec<Vec<f32>>,
    pub conv2: Vec<Vec<f32>>,
    pub bias1: Vec<f32>,
    pub bias2: Vec<f32>,
}

impl SatResidualBlock {
    /// Pointwise convolution (1×1 kernel) over `spatial` positions each of `channels` features.
    pub fn forward(&self, x: &[f32], spatial: usize, channels: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(spatial * channels);
        for s in 0..spatial {
            let slice = &x[s * channels..(s * channels + channels).min(x.len())];
            let h = dense_relu(&self.conv1, &self.bias1, slice);
            let mut y = dense_linear(&self.conv2, &self.bias2, &h);
            // Residual connection
            for (c, v) in y.iter_mut().enumerate() {
                *v += slice.get(c).copied().unwrap_or(0.0);
            }
            out.extend(y);
        }
        out
    }
}

/// Satellite super-resolution model.
pub struct SatelliteSrModel {
    pub initial_conv: Vec<Vec<f32>>,
    pub residual_blocks: Vec<SatResidualBlock>,
    pub upsample_conv: Vec<Vec<f32>>,
    pub config: SrConfig,
}

impl SatelliteSrModel {
    pub fn new(config: SrConfig, rng: &mut impl Rng) -> Self {
        let nf = config.n_features;
        let initial_conv = xavier_rows(1, nf, rng); // simplified 1-channel input
        let residual_blocks = (0..config.n_residual_blocks)
            .map(|_| SatResidualBlock {
                conv1: xavier_rows(nf, nf, rng),
                conv2: xavier_rows(nf, nf, rng),
                bias1: zeros_vec(nf),
                bias2: zeros_vec(nf),
            })
            .collect();
        let upsample_conv = xavier_rows(nf, 1, rng);
        Self {
            initial_conv,
            residual_blocks,
            upsample_conv,
            config,
        }
    }

    /// Bilinear upsample by `scale_factor`, then apply a pointwise convolution.
    pub fn upsample(&self, img: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
        let sf = self.config.scale_factor.max(1);
        let out_h = h * sf;
        let out_w = w * sf;
        let mut upsampled = vec![0.0_f32; out_h * out_w * channels];

        for oh in 0..out_h {
            for ow in 0..out_w {
                // Source pixel (bilinear nearest)
                let src_h = (oh as f32 / sf as f32).min((h as f32) - 1.0);
                let src_w = (ow as f32 / sf as f32).min((w as f32) - 1.0);
                let r0 = src_h as usize;
                let c0 = src_w as usize;
                let r1 = (r0 + 1).min(h - 1);
                let c1 = (c0 + 1).min(w - 1);
                let dr = src_h - r0 as f32;
                let dc = src_w - c0 as f32;

                for ch in 0..channels {
                    let v00 = img
                        .get((r0 * w + c0) * channels + ch)
                        .copied()
                        .unwrap_or(0.0);
                    let v01 = img
                        .get((r0 * w + c1) * channels + ch)
                        .copied()
                        .unwrap_or(0.0);
                    let v10 = img
                        .get((r1 * w + c0) * channels + ch)
                        .copied()
                        .unwrap_or(0.0);
                    let v11 = img
                        .get((r1 * w + c1) * channels + ch)
                        .copied()
                        .unwrap_or(0.0);
                    let interp = (1.0 - dr) * ((1.0 - dc) * v00 + dc * v01)
                        + dr * ((1.0 - dc) * v10 + dc * v11);
                    upsampled[(oh * out_w + ow) * channels + ch] = interp;
                }
            }
        }
        upsampled
    }
}

/// Pan-sharpening algorithms.
pub struct PanSharpening;

impl PanSharpening {
    /// Brovey transform: ms_i * pan / mean(ms).
    pub fn brovey(pan: &[f32], ms_bands: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = pan.len();
        let nb = ms_bands.len();
        if nb == 0 || n == 0 {
            return Vec::new();
        }
        (0..n).fold(vec![Vec::with_capacity(n); nb], |mut acc, i| {
            let ms_sum: f32 = ms_bands
                .iter()
                .map(|b| b.get(i).copied().unwrap_or(0.0))
                .sum();
            let denom = (ms_sum / nb as f32).max(f32::EPSILON);
            let pan_val = pan.get(i).copied().unwrap_or(0.0);
            for (k, band) in acc.iter_mut().enumerate() {
                let ms_val = ms_bands[k].get(i).copied().unwrap_or(0.0);
                band.push(ms_val * pan_val / denom);
            }
            acc
        })
    }

    /// Gram-Schmidt spectral sharpening.
    /// Projects pan onto the space spanned by mean(ms_bands), subtracts, adds residual.
    pub fn gram_schmidt(pan: &[f32], ms_bands: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = pan.len();
        let nb = ms_bands.len();
        if nb == 0 || n == 0 {
            return Vec::new();
        }

        // First GS vector: mean of all MS bands (synthetic PAN estimate)
        let synth_pan: Vec<f32> = (0..n)
            .map(|i| {
                ms_bands
                    .iter()
                    .map(|b| b.get(i).copied().unwrap_or(0.0))
                    .sum::<f32>()
                    / nb as f32
            })
            .collect();

        // Compute projection coefficient: <pan, synth> / <synth, synth>
        let dot_num: f32 = pan.iter().zip(synth_pan.iter()).map(|(a, b)| a * b).sum();
        let dot_den: f32 = synth_pan
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .max(f32::EPSILON);
        let proj = dot_num / dot_den;

        // Residual between true pan and projected synthetic pan
        let residual: Vec<f32> = pan
            .iter()
            .zip(synth_pan.iter())
            .map(|(p, s)| p - proj * s)
            .collect();

        // Sharpen each MS band by adding the residual (scaled by mean intensity ratio)
        ms_bands
            .iter()
            .map(|band| {
                let band_mean: f32 = band.iter().sum::<f32>() / n.max(1) as f32;
                let synth_mean: f32 = synth_pan.iter().sum::<f32>() / n.max(1) as f32;
                let scale = if synth_mean.abs() > f32::EPSILON {
                    band_mean / synth_mean
                } else {
                    1.0
                };
                band.iter()
                    .zip(residual.iter())
                    .map(|(ms, r)| ms + scale * r)
                    .collect()
            })
            .collect()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §6  GeoAiUtils
// ═════════════════════════════════════════════════════════════════════════════

/// Axis-aligned bounding box in geographic coordinates.
#[derive(Clone, Debug)]
pub struct GeoBbox {
    pub min_lon: f32,
    pub min_lat: f32,
    pub max_lon: f32,
    pub max_lat: f32,
}

/// A geographic point.
#[derive(Clone, Copy, Debug)]
pub struct GeoPoint {
    pub lon: f32,
    pub lat: f32,
}

/// Haversine great-circle distance in kilometres.
pub fn haversine_distance(p1: GeoPoint, p2: GeoPoint) -> f32 {
    const R: f32 = 6371.0; // Earth radius km
    let d_lat = (p2.lat - p1.lat).to_radians();
    let d_lon = (p2.lon - p1.lon).to_radians();
    let lat1 = p1.lat.to_radians();
    let lat2 = p2.lat.to_radians();
    let a = (d_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    R * c
}

/// Equirectangular UTM-like projection.
pub struct UtmProjection;

impl UtmProjection {
    const R_EARTH: f32 = 6_371_000.0; // metres

    /// Convert (lat, lon) to Cartesian offsets from a reference point (metres).
    pub fn lat_lon_to_meters(lat: f32, lon: f32, ref_lat: f32, ref_lon: f32) -> (f32, f32) {
        let d_lat = (lat - ref_lat).to_radians();
        let d_lon = (lon - ref_lon).to_radians();
        let x = d_lon * ref_lat.to_radians().cos() * Self::R_EARTH;
        let y = d_lat * Self::R_EARTH;
        (x, y)
    }
}

/// Web Mercator tile index utilities.
pub struct TileIndex;

impl TileIndex {
    /// Convert (lat, lon) at zoom level to (tile_x, tile_y).
    pub fn deg_to_tile(lat: f32, lon: f32, zoom: u8) -> (u32, u32) {
        let n = (1u32 << zoom) as f64;
        let x = ((lon as f64 + 180.0) / 360.0 * n) as u32;
        let lat_rad = (lat as f64).to_radians();
        let y = ((1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n) as u32;
        (x, y)
    }

    /// Return the geographic bounding box of a Web Mercator tile.
    pub fn tile_to_bbox(x: u32, y: u32, zoom: u8) -> GeoBbox {
        let n = (1u32 << zoom) as f64;
        let min_lon = x as f64 / n * 360.0 - 180.0;
        let max_lon = (x as f64 + 1.0) / n * 360.0 - 180.0;

        let to_lat = |tile_y: f64| -> f64 {
            let mercator = std::f64::consts::PI * (1.0 - 2.0 * tile_y / n);
            mercator.sinh().atan().to_degrees()
        };
        let max_lat = to_lat(y as f64);
        let min_lat = to_lat(y as f64 + 1.0);

        GeoBbox {
            min_lon: min_lon as f32,
            min_lat: min_lat as f32,
            max_lon: max_lon as f32,
            max_lat: max_lat as f32,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §7  SatelliteMetrics
// ═════════════════════════════════════════════════════════════════════════════

/// Overall accuracy.
pub fn overall_accuracy(predicted: &[LandCoverClass], target: &[LandCoverClass]) -> f32 {
    if predicted.is_empty() {
        return 0.0;
    }
    let correct = predicted
        .iter()
        .zip(target.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f32 / predicted.len() as f32
}

/// Cohen's kappa coefficient.
pub fn kappa_coefficient(predicted: &[LandCoverClass], target: &[LandCoverClass]) -> f32 {
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }
    let n_classes = 8;
    let p_o = overall_accuracy(predicted, target);

    // Build confusion matrix
    let mut conf = vec![vec![0_usize; n_classes]; n_classes];
    for (p, t) in predicted.iter().zip(target.iter()) {
        conf[t.index()][p.index()] += 1;
    }

    // Expected accuracy p_e = Σ (row_sum * col_sum) / n²
    let n_f = n as f64;
    let p_e: f64 = (0..n_classes)
        .map(|k| {
            let row: usize = conf[k].iter().sum();
            let col: usize = conf.iter().map(|r| r[k]).sum();
            (row as f64 * col as f64) / (n_f * n_f)
        })
        .sum();

    let denom = 1.0 - p_e;
    if denom.abs() < 1e-9 {
        1.0
    } else {
        ((p_o as f64 - p_e) / denom) as f32
    }
}

/// F1 score for binary change detection.
pub fn change_detection_f1(change_map: &ChangeMap, target: &[bool]) -> f32 {
    let n = change_map.changed.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut fn_ = 0_usize;
    for i in 0..n {
        match (change_map.changed[i], target[i]) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            _ => {}
        }
    }
    let denom = 2 * tp + fp + fn_;
    if denom == 0 {
        1.0
    } else {
        2.0 * tp as f32 / denom as f32
    }
}

/// Peak signal-to-noise ratio (dB).
pub fn psnr(original: &[f32], reconstructed: &[f32]) -> f32 {
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / n as f32;
    if mse < f32::EPSILON {
        return f32::INFINITY;
    }
    let max_val = original.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    10.0 * (max_val * max_val / mse).log10()
}

/// Simplified SSIM: luminance × contrast × structure over the whole image.
pub fn ssim(img1: &[f32], img2: &[f32], _window_size: usize) -> f32 {
    let n = img1.len().min(img2.len());
    if n == 0 {
        return 0.0;
    }
    let mean1: f32 = img1.iter().sum::<f32>() / n as f32;
    let mean2: f32 = img2.iter().sum::<f32>() / n as f32;

    let mut var1 = 0.0_f32;
    let mut var2 = 0.0_f32;
    let mut cov = 0.0_f32;
    for i in 0..n {
        let d1 = img1[i] - mean1;
        let d2 = img2[i] - mean2;
        var1 += d1 * d1;
        var2 += d2 * d2;
        cov += d1 * d2;
    }
    let nf = n as f32;
    var1 /= nf;
    var2 /= nf;
    cov /= nf;

    let c1 = (0.01_f32 * 255.0).powi(2);
    let c2 = (0.03_f32 * 255.0).powi(2);

    let num = (2.0 * mean1 * mean2 + c1) * (2.0 * cov + c2);
    let den = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);
    if den.abs() < f32::EPSILON {
        1.0
    } else {
        num / den
    }
}

/// Summary evaluation report.
#[derive(Clone, Debug)]
pub struct SatelliteEvalReport {
    pub oa: f32,
    pub kappa: f32,
    pub f1_change: f32,
    pub psnr: f32,
    pub ssim: f32,
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper to build a small synthetic image ──────────────────────────────

    fn make_image(n_bands: usize, h: usize, w: usize, seed: u64) -> MultispectralImage {
        let mut rng = StdRng::seed_from_u64(seed);
        let bands = (0..n_bands)
            .map(|_| (0..h * w).map(|_| rng.random::<f32>()).collect())
            .collect();
        MultispectralImage::new(bands, h, w)
    }

    // ── §1 SpectralData ──────────────────────────────────────────────────────

    #[test]
    fn test_multispectral_image_creation() {
        let img = make_image(4, 8, 8, 1);
        assert_eq!(img.n_bands, 4);
    }

    #[test]
    fn test_multispectral_shape() {
        let img = make_image(6, 10, 12, 2);
        assert_eq!(img.shape(), (6, 10, 12));
    }

    #[test]
    fn test_pixel_spectrum_length() {
        let img = make_image(5, 4, 4, 3);
        assert_eq!(img.pixel_spectrum(1, 2).len(), 5);
    }

    #[test]
    fn test_ndvi_range() {
        for &(r, n) in &[(0.1_f32, 0.9_f32), (0.5, 0.3), (0.8, 0.2)] {
            let v = SpectralIndices::ndvi(r, n);
            assert!((-1.0..=1.0).contains(&v), "NDVI={v} out of range");
        }
    }

    #[test]
    fn test_ndvi_healthy_vegetation() {
        let v = SpectralIndices::ndvi(0.1, 0.9);
        assert!(v > 0.5, "High NIR should give positive NDVI, got {v}");
    }

    #[test]
    fn test_ndwi_water() {
        let v = SpectralIndices::ndwi(0.9, 0.1);
        assert!(
            v > 0.0,
            "High green over low NIR should give positive NDWI, got {v}"
        );
    }

    #[test]
    fn test_ndbi_built() {
        let v = SpectralIndices::ndbi(0.9, 0.1);
        assert!(
            v > 0.0,
            "High SWIR1 over low NIR should give positive NDBI, got {v}"
        );
    }

    #[test]
    fn test_evi_finite() {
        let v = SpectralIndices::evi(0.1, 0.2, 0.8);
        assert!(v.is_finite(), "EVI should be finite");
    }

    #[test]
    fn test_compute_ndvi_map_length() {
        let img = make_image(4, 6, 6, 10);
        let map = SpectralIndices::compute_ndvi_map(&img, 2, 3);
        assert_eq!(map.len(), 36);
    }

    // ── §2 ChangeDetection ──────────────────────────────────────────────────

    #[test]
    fn test_difference_cd_change_map_size() {
        let img1 = make_image(4, 8, 8, 11);
        let img2 = make_image(4, 8, 8, 12);
        let cm = DifferenceCD::compute(&img1, &img2);
        assert_eq!(cm.changed.len(), 64);
        assert_eq!(cm.change_score.len(), 64);
    }

    #[test]
    fn test_otsu_threshold_in_range() {
        let vals: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let t = DifferenceCD::threshold_otsu(&vals);
        assert!((0.0..=1.0).contains(&t), "Otsu threshold out of range: {t}");
    }

    #[test]
    fn test_cva_magnitude_zero_same() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let m = CvaChangeDetection::magnitude(&v, &v);
        assert!(m < 1e-5, "Magnitude of same vector should be ~0, got {m}");
    }

    #[test]
    fn test_cva_direction_range() {
        let v1 = vec![0.0_f32, 0.0];
        let v2 = vec![1.0_f32, 1.0];
        let d = CvaChangeDetection::direction(&v1, &v2);
        assert!(d.is_finite());
    }

    #[test]
    fn test_cva_detect_size() {
        let img1 = make_image(3, 5, 5, 20);
        let img2 = make_image(3, 5, 5, 21);
        let cm = CvaChangeDetection::detect(&img1, &img2, 0.5);
        assert_eq!(cm.changed.len(), 25);
    }

    #[test]
    fn test_neural_change_detector_predict_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let det = NeuralChangeDetector::new(4, 16, &mut rng);
        let s1 = vec![0.1_f32, 0.2, 0.3, 0.4];
        let s2 = vec![0.5_f32, 0.6, 0.7, 0.8];
        let p = det.predict(&s1, &s2);
        assert!(
            (0.0..=1.0).contains(&p),
            "Probability should be in [0,1], got {p}"
        );
    }

    #[test]
    fn test_neural_change_detector_map_size() {
        let mut rng = StdRng::seed_from_u64(43);
        let det = NeuralChangeDetector::new(3, 8, &mut rng);
        let img1 = make_image(3, 4, 4, 30);
        let img2 = make_image(3, 4, 4, 31);
        let cm = det.detect_map(&img1, &img2);
        assert_eq!(cm.changed.len(), 16);
    }

    // ── §3 SarProcessor ─────────────────────────────────────────────────────

    #[test]
    fn test_sar_image_creation() {
        let amp = vec![1.0_f32; 100];
        let sar = SarImage::new(amp, None, 10, 10);
        assert_eq!(sar.height, 10);
        assert_eq!(sar.width, 10);
    }

    #[test]
    fn test_lee_filter_shape() {
        let mut rng = StdRng::seed_from_u64(50);
        let amp: Vec<f32> = (0..100).map(|_| rng.random::<f32>()).collect();
        let sar = SarImage::new(amp, None, 10, 10);
        let filtered = SpeckleFilter::lee_filter(&sar, 3);
        assert_eq!(filtered.amplitude.len(), 100);
    }

    #[test]
    fn test_median_filter_shape() {
        let mut rng = StdRng::seed_from_u64(51);
        let amp: Vec<f32> = (0..100).map(|_| rng.random::<f32>()).collect();
        let sar = SarImage::new(amp, None, 10, 10);
        let filtered = SpeckleFilter::median_filter(&sar, 3);
        assert_eq!(filtered.amplitude.len(), 100);
    }

    #[test]
    fn test_pauli_decomposition_shape() {
        let hh = vec![1.0_f32; 16];
        let hv = vec![0.5_f32; 16];
        let vv = vec![0.8_f32; 16];
        let result = PolarimetricDecomposition::pauli_decomposition(&hh, &hv, &vv);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 16);
    }

    #[test]
    fn test_pauli_decomposition_nonneg() {
        let hh = vec![0.7_f32; 8];
        let hv = vec![0.3_f32; 8];
        let vv = vec![0.5_f32; 8];
        let result = PolarimetricDecomposition::pauli_decomposition(&hh, &hv, &vv);
        for ch in &result {
            for &v in ch {
                assert!(v >= 0.0, "Pauli channels should be non-negative");
            }
        }
    }

    #[test]
    fn test_sar_classifier_softmax_sums_to_one() {
        let mut rng = StdRng::seed_from_u64(60);
        let clf = SarClassifier::new(8, 4, &mut rng);
        let feat: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let probs = clf.classify(&feat);
        assert_eq!(probs.len(), 4);
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "Softmax should sum to 1, got {s}");
    }

    // ── §4 LandCoverClassification ───────────────────────────────────────────

    #[test]
    fn test_land_cover_class_variants() {
        let classes = LandCoverClass::all_classes();
        assert_eq!(classes.len(), 8);
    }

    #[test]
    fn test_pixel_classifier_predict_valid_class() {
        let mut rng = StdRng::seed_from_u64(70);
        let clf = PixelClassifier::new(6, 16, 8, &mut rng);
        let spectrum = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let cls = clf.predict(&spectrum);
        // Just ensure it's a valid LandCoverClass (round-trip via index)
        let idx = cls.index();
        assert!(idx < 8);
    }

    #[test]
    fn test_classify_image_size() {
        let mut rng = StdRng::seed_from_u64(71);
        let clf = PixelClassifier::new(4, 8, 8, &mut rng);
        let img = make_image(4, 6, 6, 72);
        let map = clf.classify_image(&img);
        assert_eq!(map.classes.len(), 36);
    }

    #[test]
    fn test_simple_slic_length() {
        let img = make_image(4, 8, 8, 80);
        let segments = ObjectBasedSegmentation::simple_slic(&img, 4);
        assert_eq!(segments.len(), 64);
    }

    #[test]
    fn test_segment_statistics_shape() {
        let img = make_image(4, 4, 4, 81);
        let n_seg = 4;
        let segments = ObjectBasedSegmentation::simple_slic(&img, n_seg);
        let stats = ObjectBasedSegmentation::segment_statistics(&img, &segments);
        // Number of stats entries = max segment id + 1 (≤ n_seg initially)
        assert!(!stats.is_empty());
        for s in &stats {
            assert_eq!(s.len(), 4, "Each segment stat should have 4 band values");
        }
    }

    // ── §5 SuperResolution ───────────────────────────────────────────────────

    #[test]
    fn test_sr_model_upsample_size() {
        let mut rng = StdRng::seed_from_u64(90);
        let config = SrConfig {
            scale_factor: 2,
            n_features: 8,
            n_residual_blocks: 2,
        };
        let model = SatelliteSrModel::new(config, &mut rng);
        let img = vec![0.5_f32; 4 * 4]; // 4×4 single-channel
        let out = model.upsample(&img, 1, 4, 4);
        assert_eq!(out.len(), 8 * 8, "Upsampled output should be 8×8");
    }

    #[test]
    fn test_brovey_sharpening_n_bands() {
        let pan = vec![0.8_f32; 16];
        let ms = vec![vec![0.3_f32; 16], vec![0.4_f32; 16], vec![0.5_f32; 16]];
        let sharp = PanSharpening::brovey(&pan, &ms);
        assert_eq!(sharp.len(), 3);
        assert_eq!(sharp[0].len(), 16);
    }

    #[test]
    fn test_gram_schmidt_n_bands() {
        let pan = vec![0.7_f32; 16];
        let ms = vec![vec![0.2_f32; 16], vec![0.5_f32; 16], vec![0.3_f32; 16]];
        let sharp = PanSharpening::gram_schmidt(&pan, &ms);
        assert_eq!(sharp.len(), 3);
        assert_eq!(sharp[0].len(), 16);
    }

    // ── §6 GeoAiUtils ────────────────────────────────────────────────────────

    #[test]
    fn test_haversine_same_point_zero() {
        let p = GeoPoint {
            lon: 25.0,
            lat: 59.4,
        };
        let d = haversine_distance(p, p);
        assert!(d < 1e-3, "Same-point distance should be ~0, got {d}");
    }

    #[test]
    fn test_haversine_known_distance() {
        // London (51.5°N, 0°) to Paris (48.85°N, 2.35°E) ≈ 341 km
        let london = GeoPoint {
            lon: 0.0,
            lat: 51.5,
        };
        let paris = GeoPoint {
            lon: 2.35,
            lat: 48.85,
        };
        let d = haversine_distance(london, paris);
        assert!(
            (d - 341.0).abs() < 30.0,
            "London–Paris distance should be ≈341 km, got {d}"
        );
    }

    #[test]
    fn test_lat_lon_to_meters_origin() {
        let (x, y) = UtmProjection::lat_lon_to_meters(0.0, 0.0, 0.0, 0.0);
        assert!(
            x.abs() < 1e-3 && y.abs() < 1e-3,
            "Origin should map to (0,0)"
        );
    }

    #[test]
    fn test_deg_to_tile_zoom0() {
        let (x, y) = TileIndex::deg_to_tile(0.0, 0.0, 0);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_tile_to_bbox_valid() {
        let bbox = TileIndex::tile_to_bbox(0, 0, 1);
        assert!(bbox.min_lon < bbox.max_lon);
        assert!(bbox.min_lat < bbox.max_lat);
    }

    // ── §7 SatelliteMetrics ──────────────────────────────────────────────────

    #[test]
    fn test_overall_accuracy_perfect() {
        let cls = vec![
            LandCoverClass::Water,
            LandCoverClass::Forest,
            LandCoverClass::Urban,
        ];
        let oa = overall_accuracy(&cls, &cls);
        assert!(
            (oa - 1.0).abs() < 1e-6,
            "Perfect prediction OA should be 1.0"
        );
    }

    #[test]
    fn test_kappa_perfect() {
        let cls = vec![
            LandCoverClass::Water,
            LandCoverClass::Forest,
            LandCoverClass::Grassland,
            LandCoverClass::Urban,
        ];
        let k = kappa_coefficient(&cls, &cls);
        assert!(
            (k - 1.0).abs() < 1e-4,
            "Perfect kappa should be 1.0, got {k}"
        );
    }

    #[test]
    fn test_kappa_random() {
        // Worst-case random: each pixel predicted with a different class
        let pred = vec![
            LandCoverClass::Forest,
            LandCoverClass::Water,
            LandCoverClass::Urban,
            LandCoverClass::Grassland,
        ];
        let target = vec![
            LandCoverClass::Water,
            LandCoverClass::Forest,
            LandCoverClass::Grassland,
            LandCoverClass::Urban,
        ];
        let k = kappa_coefficient(&pred, &target);
        // All wrong → kappa should be ≤ 0
        assert!(k <= 0.1, "Random-like kappa should be ≤ 0.1, got {k}");
    }

    #[test]
    fn test_change_f1_perfect() {
        let cm = ChangeMap {
            changed: vec![true, false, true, false],
            height: 2,
            width: 2,
            change_score: vec![1.0, 0.0, 1.0, 0.0],
        };
        let target = vec![true, false, true, false];
        let f1 = change_detection_f1(&cm, &target);
        assert!(
            (f1 - 1.0).abs() < 1e-6,
            "Perfect F1 should be 1.0, got {f1}"
        );
    }

    #[test]
    fn test_psnr_perfect_infinite() {
        let img = vec![0.5_f32; 100];
        let p = psnr(&img, &img);
        assert!(
            p.is_infinite(),
            "Same image PSNR should be infinity, got {p}"
        );
    }

    #[test]
    fn test_psnr_different_finite() {
        let orig: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let rec: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0 + 0.1).min(1.0)).collect();
        let p = psnr(&orig, &rec);
        assert!(
            p.is_finite() && p > 0.0,
            "PSNR of different images should be finite positive, got {p}"
        );
    }

    #[test]
    fn test_ssim_same_image() {
        let img: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let s = ssim(&img, &img, 7);
        assert!(
            (s - 1.0).abs() < 1e-4,
            "SSIM of same image should be ~1.0, got {s}"
        );
    }

    #[test]
    fn test_satellite_eval_report_fields() {
        let report = SatelliteEvalReport {
            oa: 0.9,
            kappa: 0.85,
            f1_change: 0.8,
            psnr: 30.0,
            ssim: 0.95,
        };
        assert!((report.oa - 0.9).abs() < 1e-6);
        assert!((report.kappa - 0.85).abs() < 1e-6);
        assert!((report.f1_change - 0.8).abs() < 1e-6);
        assert!((report.psnr - 30.0).abs() < 1e-6);
        assert!((report.ssim - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_anisotropy_alpha_finite() {
        let cm = vec![
            vec![
                vec![0.6_f32, 0.1, 0.05],
                vec![0.1, 0.3, 0.05],
                vec![0.05, 0.05, 0.1],
            ],
            vec![
                vec![0.5_f32, 0.0, 0.0],
                vec![0.0, 0.4, 0.0],
                vec![0.0, 0.0, 0.1],
            ],
        ];
        let (h, a, alpha) = PolarimetricDecomposition::entropy_anisotropy_alpha(&cm);
        assert!(h.is_finite() && (0.0..=1.0).contains(&h), "H={h}");
        assert!(a.is_finite() && (0.0..=1.0).contains(&a), "A={a}");
        assert!(alpha.is_finite() && alpha >= 0.0, "alpha={alpha}");
    }
}
