//! Image & point cloud synthesis: Perlin noise, Poisson disk, 3D shapes, label noise, augmentation.

use super::math::sample_normal;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

/// Perlin noise generator (hash-based gradient noise).
pub struct PerlinNoise {
    perm: [u8; 512],
}

impl PerlinNoise {
    /// Create a new instance with the given seed.
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut p: Vec<u8> = (0..=255_u8).collect();
        for i in (1..256_usize).rev() {
            let j = rng.random::<u64>() as usize % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0_u8; 512];
        perm[..256].copy_from_slice(&p[..256]);
        perm[256..512].copy_from_slice(&p[..256]);
        Self { perm }
    }

    fn grad2(hash: u8, x: f64, y: f64) -> f64 {
        match hash & 3 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            _ => -x - y,
        }
    }

    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    /// Evaluate Perlin noise at `(x, y)`.
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        let xi = x.floor() as i64 & 255;
        let yi = y.floor() as i64 & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = Self::fade(xf);
        let v = Self::fade(yf);
        let p = &self.perm;
        let aa = p[(p[xi as usize] as i64 + yi) as usize & 511];
        let ab = p[(p[xi as usize] as i64 + yi + 1) as usize & 511];
        let ba = p[(p[(xi + 1) as usize & 255] as i64 + yi) as usize & 511];
        let bb = p[(p[(xi + 1) as usize & 255] as i64 + yi + 1) as usize & 511];
        let x1 = Self::lerp(Self::grad2(aa, xf, yf), Self::grad2(ba, xf - 1.0, yf), u);
        let x2 = Self::lerp(
            Self::grad2(ab, xf, yf - 1.0),
            Self::grad2(bb, xf - 1.0, yf - 1.0),
            u,
        );
        Self::lerp(x1, x2, v)
    }

    /// Fractional Brownian Motion via layered Perlin octaves.
    pub fn fbm(&self, x: f64, y: f64, octaves: usize, lacunarity: f64, gain: f64) -> f64 {
        let mut val = 0.0_f64;
        let mut amp = 1.0_f64;
        let mut freq = 1.0_f64;
        let mut max_val = 0.0_f64;
        for _ in 0..octaves {
            val += self.noise2d(x * freq, y * freq) * amp;
            max_val += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        if max_val > 0.0 {
            val / max_val
        } else {
            0.0
        }
    }
}

/// Poisson disk sampling in 2D (Bridson's algorithm).
pub struct PoissonDiskSampling;

impl PoissonDiskSampling {
    /// Generate Poisson-disk samples within `[0, width] × [0, height]`.
    pub fn sample(width: f64, height: f64, min_dist: f64, seed: u64) -> Vec<[f64; 2]> {
        let mut rng = StdRng::seed_from_u64(seed);
        let cell_size = min_dist / std::f64::consts::SQRT_2;
        let grid_w = ((width / cell_size).ceil() as usize).max(1);
        let grid_h = ((height / cell_size).ceil() as usize).max(1);
        let mut grid: Vec<Option<[f64; 2]>> = vec![None; grid_w * grid_h];
        let mut samples: Vec<[f64; 2]> = Vec::new();
        let mut active: Vec<[f64; 2]> = Vec::new();
        let first = [width * rng.random::<f64>(), height * rng.random::<f64>()];
        samples.push(first);
        active.push(first);
        let gx = (first[0] / cell_size) as usize;
        let gy = (first[1] / cell_size) as usize;
        grid[gy * grid_w + gx] = Some(first);
        let k = 30_usize;
        while !active.is_empty() {
            let idx = rng.random::<u64>() as usize % active.len();
            let base = active[idx];
            let mut found = false;
            for _ in 0..k {
                let angle = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let r = min_dist * (1.0 + rng.random::<f64>());
                let candidate = [base[0] + r * angle.cos(), base[1] + r * angle.sin()];
                if candidate[0] < 0.0
                    || candidate[0] >= width
                    || candidate[1] < 0.0
                    || candidate[1] >= height
                {
                    continue;
                }
                let cgx = (candidate[0] / cell_size) as usize;
                let cgy = (candidate[1] / cell_size) as usize;
                let mut ok = true;
                let lo_x = cgx.saturating_sub(2);
                let hi_x = (cgx + 2).min(grid_w - 1);
                let lo_y = cgy.saturating_sub(2);
                let hi_y = (cgy + 2).min(grid_h - 1);
                'check: for ny in lo_y..=hi_y {
                    for nx in lo_x..=hi_x {
                        if let Some(pt) = grid[ny * grid_w + nx] {
                            let dx = pt[0] - candidate[0];
                            let dy = pt[1] - candidate[1];
                            if (dx * dx + dy * dy).sqrt() < min_dist {
                                ok = false;
                                break 'check;
                            }
                        }
                    }
                }
                if ok {
                    samples.push(candidate);
                    active.push(candidate);
                    grid[cgy * grid_w + cgx] = Some(candidate);
                    found = true;
                    break;
                }
            }
            if !found {
                active.remove(idx);
            }
        }
        samples
    }
}

/// Point cloud generator for basic 3D shapes.
pub struct SyntheticShapeGenerator;

impl SyntheticShapeGenerator {
    /// Uniformly sample `n` points on a sphere of radius `r`.
    pub fn sphere(n: usize, r: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let u = rng.random::<f64>();
                let v = rng.random::<f64>();
                let theta = 2.0 * std::f64::consts::PI * u;
                let phi = (2.0 * v - 1.0).acos();
                [
                    r * phi.sin() * theta.cos(),
                    r * phi.sin() * theta.sin(),
                    r * phi.cos(),
                ]
            })
            .collect()
    }

    /// Sample `n` points on a cylinder surface with radius `r` and height `h`.
    pub fn cylinder(n: usize, r: f64, h: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let theta = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                let z = rng.random::<f64>() * h - h / 2.0;
                [r * theta.cos(), r * theta.sin(), z]
            })
            .collect()
    }

    /// Sample `n` points on a torus with major radius `big_r` and tube radius `r`.
    pub fn torus(n: usize, big_r: f64, r: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                let theta = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                [
                    (big_r + r * theta.cos()) * phi.cos(),
                    (big_r + r * theta.cos()) * phi.sin(),
                    r * theta.sin(),
                ]
            })
            .collect()
    }
}

/// Label noise generator for classification datasets.
pub struct NoisyLabelGenerator;

impl NoisyLabelGenerator {
    /// Symmetric label noise: flip each label with probability `noise_rate`.
    pub fn flip_labels(y: &[usize], n_classes: usize, noise_rate: f64, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        y.iter()
            .map(|&label| {
                if rng.random::<f64>() < noise_rate && n_classes > 1 {
                    let offset = 1 + rng.random::<u64>() as usize % (n_classes - 1);
                    (label + offset) % n_classes
                } else {
                    label
                }
            })
            .collect()
    }

    /// Asymmetric (transition-matrix) label noise.
    pub fn asymmetric_noise(y: &[usize], transition_matrix: &[Vec<f64>], seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        y.iter()
            .map(|&label| {
                if label >= transition_matrix.len() {
                    return label;
                }
                let row = &transition_matrix[label];
                let u: f64 = rng.random();
                let mut cum = 0.0_f64;
                let mut chosen = label;
                for (j, &p) in row.iter().enumerate() {
                    cum += p;
                    if u <= cum {
                        chosen = j;
                        break;
                    }
                }
                chosen
            })
            .collect()
    }
}

/// Trait for single-sample augmentation transforms.
pub trait Augment: Send + Sync {
    /// Apply the augmentation to `x` using the given RNG.
    fn augment(&self, x: &[f64], rng: &mut StdRng) -> Vec<f64>;
}

/// Gaussian noise augmentation.
pub struct GaussianNoise {
    sigma: f64,
}

impl GaussianNoise {
    /// Create with given standard deviation.
    pub fn new(sigma: f64) -> Self {
        Self { sigma }
    }
}

impl Augment for GaussianNoise {
    fn augment(&self, x: &[f64], rng: &mut StdRng) -> Vec<f64> {
        x.iter()
            .map(|&xi| xi + self.sigma * sample_normal(rng))
            .collect()
    }
}

/// Feature dropout augmentation.
pub struct FeatureDropout {
    rate: f64,
}

impl FeatureDropout {
    /// Create with given dropout rate.
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }
}

impl Augment for FeatureDropout {
    fn augment(&self, x: &[f64], rng: &mut StdRng) -> Vec<f64> {
        x.iter()
            .map(|&xi| {
                if rng.random::<f64>() < self.rate {
                    0.0
                } else {
                    xi
                }
            })
            .collect()
    }
}

/// Composable augmentation pipeline.
pub struct DataAugmentationPipeline {
    steps: Vec<Box<dyn Augment>>,
}

impl DataAugmentationPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a transform step.
    pub fn add_step(&mut self, transform: Box<dyn Augment>) {
        self.steps.push(transform);
    }

    /// Apply all steps in sequence.
    pub fn apply(&self, x: &[f64], rng: &mut StdRng) -> Vec<f64> {
        self.steps
            .iter()
            .fold(x.to_vec(), |acc, step| step.augment(&acc, rng))
    }
}

impl Default for DataAugmentationPipeline {
    fn default() -> Self {
        Self::new()
    }
}
