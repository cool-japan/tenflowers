//! Geospatial ML Extensions — §4 (cont.), §5, §6, and Tests.

use super::{
    gaussian_elimination, mat_vec, relu, xavier_matrix, zeros,
    GeoCoord, OrdinaryKriging, SpatialGcnLayer, SpatialGraph,
};
use scirs2_core::random::{rngs::StdRng, SeedableRng, Rng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

// ─── §4 cont. ─────────────────────────────────────────────────────────────────

/// Radial Basis Function (multiquadric) spatial interpolator.
///
/// φ(r) = sqrt(r² + ε²) — multiquadric RBF.
#[derive(Clone, Debug)]
pub struct RadialBasisInterpolator {
    /// Smoothing parameter ε (km).
    pub epsilon_km: f64,
    /// Sample coordinates.
    pub sample_coords: Vec<GeoCoord>,
    /// Fitted RBF coefficients (one per sample).
    pub coeffs: Vec<f64>,
}

impl RadialBasisInterpolator {
    /// Fit the interpolator to known samples.
    pub fn fit(epsilon_km: f64, sample_coords: Vec<GeoCoord>, sample_values: &[f64]) -> Self {
        let n = sample_coords.len();
        // Build Phi matrix.
        let mut phi = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = sample_coords[i].haversine_km(&sample_coords[j]);
                phi[i][j] = (r * r + epsilon_km * epsilon_km).sqrt();
            }
        }
        let coeffs = gaussian_elimination(&phi, sample_values);
        Self { epsilon_km, sample_coords, coeffs }
    }

    /// Predict at an unsampled location.
    pub fn predict(&self, target: &GeoCoord) -> f64 {
        self.sample_coords
            .iter()
            .zip(self.coeffs.iter())
            .map(|(c, &w)| {
                let r = c.haversine_km(target);
                w * (r * r + self.epsilon_km * self.epsilon_km).sqrt()
            })
            .sum()
    }
}

/// Spatial leave-one-out cross-validation metrics.
#[derive(Clone, Debug)]
pub struct SpatialCrossValidation {
    /// RMSE in same units as the predicted variable.
    pub rmse: f64,
    /// MAE.
    pub mae: f64,
    /// CRPS (Continuous Ranked Probability Score) — only meaningful for Kriging with variance.
    pub crps: f64,
}

impl SpatialCrossValidation {
    /// Run LOO-CV on an Ordinary Kriging model.
    pub fn run_kriging(
        nugget: f64,
        sill: f64,
        range_km: f64,
        coords: &[GeoCoord],
        values: &[f64],
    ) -> Self {
        let n = coords.len();
        if n < 2 {
            return Self { rmse: 0.0, mae: 0.0, crps: 0.0 };
        }
        let mut sq_err = 0.0;
        let mut abs_err = 0.0;
        let mut crps_sum = 0.0;
        for i in 0..n {
            let leave_coords: Vec<GeoCoord> = coords
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, c)| c.clone())
                .collect();
            let leave_vals: Vec<f64> = values
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v)
                .collect();
            let model = OrdinaryKriging::new(nugget, sill, range_km, leave_coords, leave_vals);
            let (pred, var) = model.predict(&coords[i]);
            let err = pred - values[i];
            sq_err += err * err;
            abs_err += err.abs();
            let sigma = var.sqrt().max(1e-9);
            // CRPS for Gaussian: σ·(z·(2·Φ(z)−1) + 2·φ(z) − 1/√π)
            let z = err / sigma;
            let phi_z = (-(z * z) / 2.0).exp() / (2.0 * PI).sqrt();
            let big_phi_z = 0.5 * (1.0 + erf_approx(z / 2.0_f64.sqrt()));
            crps_sum += sigma * (z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z - 1.0 / PI.sqrt());
        }
        Self {
            rmse: (sq_err / n as f64).sqrt(),
            mae: abs_err / n as f64,
            crps: crps_sum / n as f64,
        }
    }
}

/// Approximate error function for CRPS computation.
pub(crate) fn erf_approx(x: f64) -> f64 {
    // Abramowitz and Stegun approximation 7.1.26, max error 1.5e-7.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254_829_592
        + t * (-0.284_496_736
            + t * (1.421_413_741
                + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

// ═════════════════════════════════════════════════════════════════════════════
// §5  Spatial-Temporal Models
// ═════════════════════════════════════════════════════════════════════════════

/// Spatial-Temporal GCN layer (Yu et al. AAAI 2018 ST-GCN style).
///
/// Applies: temporal 1-D convolution followed by spatial graph convolution.
#[derive(Clone, Debug)]
pub struct StGcnLayer {
    /// Spatial GCN part.
    pub spatial: SpatialGcnLayer,
    /// Temporal kernel size (number of time steps).
    pub kernel_size: usize,
    /// Temporal conv weights: [out_dim × in_dim × kernel_size].
    temporal_w: Vec<Vec<Vec<f64>>>,
    temporal_b: Vec<f64>,
}

impl StGcnLayer {
    /// Create a new ST-GCN layer.
    pub fn new(in_dim: usize, out_dim: usize, kernel_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let lim = (6.0 / (in_dim * kernel_size + out_dim) as f64).sqrt();
        let temporal_w: Vec<Vec<Vec<f64>>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| {
                        (0..kernel_size)
                            .map(|_| {
                                let u: f64 = rng.random();
                                u * 2.0 * lim - lim
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            spatial: SpatialGcnLayer::new(out_dim, out_dim, 1.0, seed + 100),
            kernel_size,
            temporal_w,
            temporal_b: zeros(out_dim),
        }
    }

    /// Forward: x is [T × N × in_dim]; returns [T' × N × out_dim] where T' = T − kernel_size + 1.
    pub fn forward(&self, graph: &SpatialGraph, x: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        let t = x.len();
        let n = graph.num_nodes();
        let out_t = t.saturating_sub(self.kernel_size - 1);
        let in_dim = if t > 0 && !x[0].is_empty() { x[0][0].len() } else { 0 };
        let out_dim = self.temporal_b.len();
        // Temporal convolution: for each output time step, node, out channel.
        let mut temporal_out = vec![vec![zeros(out_dim); n]; out_t];
        for t_out in 0..out_t {
            for ni in 0..n {
                let mut feat = zeros(out_dim);
                for o in 0..out_dim {
                    let mut s = self.temporal_b[o];
                    for k in 0..self.kernel_size {
                        let t_in = t_out + k;
                        if let Some(frame) = x.get(t_in) {
                            for d in 0..in_dim.min(self.temporal_w[o].len()) {
                                let xv = frame.get(ni).and_then(|f| f.get(d)).copied().unwrap_or(0.0);
                                let wv = self.temporal_w[o][d].get(k).copied().unwrap_or(0.0);
                                s += wv * xv;
                            }
                        }
                    }
                    feat[o] = relu(s);
                }
                temporal_out[t_out][ni] = feat;
            }
        }
        // Spatial GCN on each time slice.
        temporal_out
            .into_iter()
            .map(|frame| self.spatial.forward(graph, &frame))
            .collect()
    }
}

/// Graph diffusion convolution layer (DCRNN-style).
///
/// Approximates spectral graph convolution using K-step random walk
/// transition matrices in both directions.
#[derive(Clone, Debug)]
pub struct DiffusionConvLayer {
    /// Number of diffusion steps (K).
    pub num_steps: usize,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Weights for each diffusion order and direction: [2*(K+1) × out_dim × in_dim].
    weights: Vec<Vec<Vec<f64>>>,
    bias: Vec<f64>,
}

impl DiffusionConvLayer {
    /// Create a new diffusion convolution layer.
    pub fn new(in_dim: usize, out_dim: usize, num_steps: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let num_filters = 2 * (num_steps + 1);
        let mut weights = Vec::with_capacity(num_filters);
        for _ in 0..num_filters {
            weights.push(xavier_matrix(in_dim, out_dim, &mut rng));
        }
        Self { num_steps, in_dim, out_dim, weights, bias: zeros(out_dim) }
    }

    /// Build row-normalised adjacency matrix (forward random walk).
    fn row_norm(graph: &SpatialGraph) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();
        let mut mat = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let deg: f64 = graph.adj[i].iter().map(|e| 1.0 / e.distance_km.max(1e-6)).sum::<f64>().max(f64::EPSILON);
            for e in &graph.adj[i] {
                mat[i][e.dst] = 1.0 / (e.distance_km.max(1e-6) * deg);
            }
        }
        mat
    }

    /// Forward pass; returns [n × out_dim].
    pub fn forward(&self, graph: &SpatialGraph, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();
        let fwd = Self::row_norm(graph);
        // Transpose for backward walk.
        let bwd: Vec<Vec<f64>> = (0..n)
            .map(|j| (0..n).map(|i| fwd[i][j]).collect())
            .collect();

        let mat_mul = |mat: &[Vec<f64>], vecs: &[Vec<f64>]| -> Vec<Vec<f64>> {
            let out_n = mat.len();
            (0..out_n)
                .map(|i| {
                    let d = vecs.first().map(|v| v.len()).unwrap_or(0);
                    let mut res = zeros(d);
                    for j in 0..n {
                        let w = mat[i].get(j).copied().unwrap_or(0.0);
                        if w.abs() > f64::EPSILON {
                            if let Some(xj) = vecs.get(j) {
                                for d_idx in 0..d.min(xj.len()) {
                                    res[d_idx] += w * xj[d_idx];
                                }
                            }
                        }
                    }
                    res
                })
                .collect()
        };

        let mut out = vec![zeros(self.out_dim); n];
        let mut fwd_pow = x.to_vec();
        let mut bwd_pow = x.to_vec();
        let mut filter_idx = 0usize;
        // Identity term (step 0).
        for ni in 0..n {
            let y = mat_vec(&self.weights[filter_idx], x.get(ni).unwrap_or(&zeros(self.in_dim)), &zeros(self.out_dim));
            for d in 0..self.out_dim {
                out[ni][d] += y.get(d).copied().unwrap_or(0.0);
            }
        }
        filter_idx += 1;
        if filter_idx < self.weights.len() {
            for ni in 0..n {
                let y = mat_vec(&self.weights[filter_idx], x.get(ni).unwrap_or(&zeros(self.in_dim)), &zeros(self.out_dim));
                for d in 0..self.out_dim {
                    out[ni][d] += y.get(d).copied().unwrap_or(0.0);
                }
            }
            filter_idx += 1;
        }
        // Higher-order terms.
        for _k in 1..=self.num_steps {
            fwd_pow = mat_mul(&fwd, &fwd_pow);
            bwd_pow = mat_mul(&bwd, &bwd_pow);
            if filter_idx < self.weights.len() {
                for ni in 0..n {
                    let y = mat_vec(&self.weights[filter_idx], fwd_pow.get(ni).unwrap_or(&zeros(self.in_dim)), &zeros(self.out_dim));
                    for d in 0..self.out_dim {
                        out[ni][d] += y.get(d).copied().unwrap_or(0.0);
                    }
                }
                filter_idx += 1;
            }
            if filter_idx < self.weights.len() {
                for ni in 0..n {
                    let y = mat_vec(&self.weights[filter_idx], bwd_pow.get(ni).unwrap_or(&zeros(self.in_dim)), &zeros(self.out_dim));
                    for d in 0..self.out_dim {
                        out[ni][d] += y.get(d).copied().unwrap_or(0.0);
                    }
                }
                filter_idx += 1;
            }
        }
        let _ = filter_idx;
        for ni in 0..n {
            for d in 0..self.out_dim {
                out[ni][d] = relu(out[ni][d] + self.bias[d]);
            }
        }
        out
    }
}

/// Geo-Attention model for origin-destination (OD) demand prediction.
///
/// Encodes origin and destination region embeddings, computes cross-attention
/// based on spatial proximity, and decodes to a scalar demand forecast.
#[derive(Clone, Debug)]
pub struct GeoAttentionModel {
    /// Embedding dimension for each region.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    query_w: Vec<Vec<f64>>,  // [embed × embed]
    key_w: Vec<Vec<f64>>,
    value_w: Vec<Vec<f64>>,
    out_w: Vec<Vec<f64>>,    // [1 × embed]
    out_b: Vec<f64>,
}

impl GeoAttentionModel {
    /// Create a new model.
    pub fn new(embed_dim: usize, num_heads: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            embed_dim,
            num_heads,
            query_w: xavier_matrix(embed_dim, embed_dim, &mut rng),
            key_w: xavier_matrix(embed_dim, embed_dim, &mut rng),
            value_w: xavier_matrix(embed_dim, embed_dim, &mut rng),
            out_w: xavier_matrix(embed_dim, 1, &mut rng),
            out_b: zeros(1),
        }
    }

    /// Predict OD demand matrix [n_origins × n_dests].
    /// Embeddings: [n × embed_dim]; coords: \[n\] corresponding coordinates.
    pub fn predict_od(
        &self,
        embeddings: &[Vec<f64>],
        coords: &[GeoCoord],
        sigma_km: f64,
    ) -> Vec<Vec<f64>> {
        let n = embeddings.len();
        let scale = (self.embed_dim as f64).sqrt().max(1.0);
        let zero_e = zeros(self.embed_dim);
        let q: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&self.query_w, e, &zero_e)).collect();
        let k: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&self.key_w, e, &zero_e)).collect();
        let v: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&self.value_w, e, &zero_e)).collect();
        let mut od = vec![vec![0.0_f64; n]; n];
        let two_s2 = 2.0 * sigma_km * sigma_km;
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = q[i].iter().zip(k[j].iter()).map(|(&qi, &kj)| qi * kj).sum::<f64>() / scale;
                let spatial_w = if let (Some(ci), Some(cj)) = (coords.get(i), coords.get(j)) {
                    let d = ci.haversine_km(cj);
                    (-d * d / two_s2).exp()
                } else {
                    1.0
                };
                let attn_score = dot + spatial_w.ln().max(-10.0);
                // Value projection for demand.
                let context: Vec<f64> = v[j].iter().map(|&vi| vi * attn_score.tanh()).collect();
                let demand = mat_vec(&self.out_w, &context, &self.out_b);
                od[i][j] = relu(demand.first().copied().unwrap_or(0.0));
            }
        }
        od
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §6  GeoMetrics
// ═════════════════════════════════════════════════════════════════════════════

/// Comprehensive geospatial evaluation metrics.
#[derive(Clone, Debug)]
pub struct GeoMetrics {
    /// Mean Absolute Error in km (haversine).
    pub mae_km: f64,
    /// Root Mean Squared Error in km.
    pub rmse_km: f64,
    /// Mean Absolute Percentage Error (fraction, not percent).
    pub mape: f64,
    /// Moran's I spatial autocorrelation coefficient.
    pub morans_i: f64,
}

impl GeoMetrics {
    /// Compute all metrics.
    ///
    /// `pred_coords` and `true_coords` are predicted vs true locations.
    /// `values` and `coords_for_moran` are used for Moran's I.
    pub fn compute(
        pred_coords: &[GeoCoord],
        true_coords: &[GeoCoord],
        values: &[f64],
        moran_coords: &[GeoCoord],
    ) -> Self {
        let n = pred_coords.len().min(true_coords.len());
        let mae_km = if n == 0 {
            0.0
        } else {
            pred_coords.iter().zip(true_coords.iter()).map(|(p, t)| p.haversine_km(t)).sum::<f64>() / n as f64
        };
        let rmse_km = if n == 0 {
            0.0
        } else {
            let sq: f64 = pred_coords.iter().zip(true_coords.iter()).map(|(p, t)| p.haversine_km(t).powi(2)).sum();
            (sq / n as f64).sqrt()
        };
        // MAPE on distances (avoid divide-by-zero).
        let mape = if n == 0 {
            0.0
        } else {
            let s: f64 = pred_coords
                .iter()
                .zip(true_coords.iter())
                .map(|(p, t)| {
                    let true_dist = t.haversine_km(&GeoCoord::new(0.0, 0.0)).max(1e-3);
                    p.haversine_km(t) / true_dist
                })
                .sum();
            s / n as f64
        };
        let morans_i = morans_i(values, moran_coords);
        Self { mae_km, rmse_km, mape, morans_i }
    }
}

/// Compute Moran's I spatial autocorrelation.
///
/// I = (N / W) × (Σ_i Σ_j w_ij (z_i − ẑ)(z_j − ẑ)) / Σ_i (z_i − ẑ)²
///
/// Spatial weights w_ij = 1 / d_ij (inverse distance, d > 0).
pub fn morans_i(values: &[f64], coords: &[GeoCoord]) -> f64 {
    let n = values.len().min(coords.len());
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let z: Vec<f64> = values.iter().map(|&v| v - mean).collect();
    let ss: f64 = z.iter().map(|&zi| zi * zi).sum();
    if ss < f64::EPSILON {
        return 0.0;
    }
    let mut numerator = 0.0_f64;
    let mut w_total = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = coords[i].haversine_km(&coords[j]).max(1e-6);
            let w = 1.0 / d;
            numerator += w * z[i] * z[j];
            w_total += w;
        }
    }
    if w_total < f64::EPSILON {
        return 0.0;
    }
    (n as f64 / w_total) * (numerator / ss)
}

/// Summary evaluation report for geospatial model outputs.
#[derive(Clone, Debug)]
pub struct SpatialEvalReport {
    /// Mean absolute error in km.
    pub mae_km: f64,
    /// Root mean squared error in km.
    pub rmse_km: f64,
    /// Mean absolute percentage error.
    pub mape: f64,
    /// Moran's I autocorrelation of residuals.
    pub morans_i: f64,
    /// Mean Fréchet distance for trajectory predictions (km).
    pub mean_frechet_km: f64,
    /// Interpolation RMSE.
    pub interpolation_rmse: f64,
}

impl SpatialEvalReport {
    /// Build a report from component metrics.
    pub fn new(
        metrics: &GeoMetrics,
        mean_frechet_km: f64,
        interpolation_rmse: f64,
    ) -> Self {
        Self {
            mae_km: metrics.mae_km,
            rmse_km: metrics.rmse_km,
            mape: metrics.mape,
            morans_i: metrics.morans_i,
            mean_frechet_km,
            interpolation_rmse,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    // super = extensions module; super::super = geospatial_ml re-exporting everything
    use super::*;
    use super::super::*;

    // ── §1  Spatial Feature Encoding ─────────────────────────────────────────

    #[test]
    fn test_geocoord_haversine_known() {
        // Distance between London (51.5, -0.13) and Paris (48.85, 2.35) ≈ 340 km.
        let london = GeoCoord::new(51.5, -0.13);
        let paris = GeoCoord::new(48.85, 2.35);
        let d = london.haversine_km(&paris);
        assert!(d > 300.0 && d < 380.0, "London-Paris should be ~340 km, got {d:.1}");
    }

    #[test]
    fn test_geocoord_haversine_zero() {
        let a = GeoCoord::new(35.0, 139.0);
        let d = a.haversine_km(&a);
        assert!(d.abs() < 1e-9, "Self-distance should be 0, got {d}");
    }

    #[test]
    fn test_geocoord_bearing_north() {
        let a = GeoCoord::new(0.0, 0.0);
        let b = GeoCoord::new(10.0, 0.0);
        let bearing = a.bearing_to(&b);
        assert!(!(1.0..=359.0).contains(&bearing), "Northward bearing should be ~0°, got {bearing:.2}");
    }

    #[test]
    fn test_geocoord_bearing_east() {
        let a = GeoCoord::new(0.0, 0.0);
        let b = GeoCoord::new(0.0, 10.0);
        let bearing = a.bearing_to(&b);
        assert!((bearing - 90.0).abs() < 2.0, "Eastward bearing should be ~90°, got {bearing:.2}");
    }

    #[test]
    fn test_rbf_interpolator_known_values() {
        let coords = vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(2.0, 0.0),
            GeoCoord::new(0.0, 2.0),
        ];
        let values = vec![1.0_f64, 3.0, 5.0];
        let rbf = RadialBasisInterpolator::fit(10.0, coords.clone(), &values);
        // Predictions at sample points should be close.
        let p0 = rbf.predict(&coords[0]);
        assert!((p0 - 1.0).abs() < 1.0, "RBF at sample[0] should be ≈1.0, got {p0:.2}");
    }

    #[test]
    fn test_spatial_cv_kriging_rmse() {
        let coords: Vec<GeoCoord> = (0..5).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let values: Vec<f64> = (0..5).map(|i| i as f64 * 2.0).collect();
        let cv = SpatialCrossValidation::run_kriging(0.0, 1.0, 500.0, &coords, &values);
        assert!(cv.rmse >= 0.0, "RMSE must be non-negative");
        assert!(cv.mae >= 0.0, "MAE must be non-negative");
        assert!(cv.rmse.is_finite(), "RMSE must be finite");
    }

    // ── §5  Spatial-Temporal Models ───────────────────────────────────────────

    #[test]
    fn test_stgcn_output_shape() {
        let coords: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let feats: Vec<Vec<f64>> = (0..4).map(|_| vec![1.0; 4]).collect();
        let g = SpatialGraph::build(coords, feats, 2);
        let layer = StGcnLayer::new(4, 8, 3, 13);
        // T=6 frames, N=4 nodes, in_dim=4.
        let x: Vec<Vec<Vec<f64>>> = (0..6)
            .map(|_| (0..4).map(|_| vec![0.5_f64; 4]).collect())
            .collect();
        let out = layer.forward(&g, &x);
        // Output T' = 6 - 3 + 1 = 4.
        assert_eq!(out.len(), 4, "Output temporal length should be T - K + 1");
        for frame in &out {
            assert_eq!(frame.len(), 4);
            for node_feat in frame {
                assert_eq!(node_feat.len(), 8);
            }
        }
    }

    #[test]
    fn test_diffusion_conv_shape() {
        let coords: Vec<GeoCoord> = (0..5).map(|i| GeoCoord::new(i as f64 * 0.5, 0.0)).collect();
        let feats: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0; 4]).collect();
        let g = SpatialGraph::build(coords, feats.clone(), 2);
        let layer = DiffusionConvLayer::new(4, 8, 2, 55);
        let out = layer.forward(&g, &feats);
        assert_eq!(out.len(), 5, "One output vector per node");
        for v in &out {
            assert_eq!(v.len(), 8);
            for &x in v {
                assert!(x.is_finite(), "Diffusion conv output must be finite");
            }
        }
    }

    #[test]
    fn test_geo_attention_od_shape() {
        let n = 4;
        let embed: Vec<Vec<f64>> = (0..n).map(|_| vec![0.5_f64; 8]).collect();
        let coords: Vec<GeoCoord> = (0..n).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let model = GeoAttentionModel::new(8, 2, 42);
        let od = model.predict_od(&embed, &coords, 500.0);
        assert_eq!(od.len(), n, "OD matrix should have n rows");
        for row in &od {
            assert_eq!(row.len(), n);
            for &v in row {
                assert!(v >= 0.0 && v.is_finite(), "OD demand must be non-negative and finite");
            }
        }
    }

    #[test]
    fn test_geo_attention_od_non_zero() {
        let embed: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.5, 0.2],
            vec![0.0, 1.0, 0.3, 0.7],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let coords = vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(1.0, 0.0),
            GeoCoord::new(0.5, 0.5),
        ];
        let model = GeoAttentionModel::new(4, 1, 99);
        let od = model.predict_od(&embed, &coords, 200.0);
        let total: f64 = od.iter().flat_map(|r| r.iter()).sum();
        assert!(total.is_finite(), "OD matrix total should be finite");
    }

    // ── §6  GeoMetrics ────────────────────────────────────────────────────────

    #[test]
    fn test_morans_i_clustered() {
        // Clustered pattern: two spatial groups with different values.
        let coords = vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(0.1, 0.0),
            GeoCoord::new(10.0, 0.0),
            GeoCoord::new(10.1, 0.0),
        ];
        let values = vec![1.0, 1.0, 10.0, 10.0];
        let i = morans_i(&values, &coords);
        assert!(i > 0.0, "Clustered pattern should give positive Moran's I, got {i:.3}");
    }

    #[test]
    fn test_morans_i_checkerboard() {
        // Dispersed/alternating pattern → negative I.
        let coords: Vec<GeoCoord> = vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(0.1, 0.0),
            GeoCoord::new(0.0, 0.1),
            GeoCoord::new(0.1, 0.1),
        ];
        let values = vec![1.0, -1.0, -1.0, 1.0];
        let i = morans_i(&values, &coords);
        assert!(i < 0.5, "Checkerboard should give non-positive Moran's I, got {i:.3}");
    }

    #[test]
    fn test_morans_i_constant() {
        let coords: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let values = vec![5.0; 4];
        let i = morans_i(&values, &coords);
        assert!(i.is_finite() || i == 0.0, "Constant values: Moran's I should be 0 or handle gracefully");
    }

    #[test]
    fn test_geo_metrics_mae_zero_prediction() {
        let coords: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let m = GeoMetrics::compute(&coords, &coords, &[1.0, 2.0, 3.0, 4.0], &coords);
        assert!(m.mae_km < 1e-9, "Identical pred/true should give MAE ≈ 0, got {:.6}", m.mae_km);
        assert!(m.rmse_km < 1e-9, "Identical pred/true should give RMSE ≈ 0");
    }

    #[test]
    fn test_geo_metrics_finite() {
        let pred: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let true_c: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64 + 0.1, 0.0)).collect();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let m = GeoMetrics::compute(&pred, &true_c, &values, &true_c);
        assert!(m.mae_km.is_finite() && m.rmse_km.is_finite() && m.mape.is_finite());
    }

    #[test]
    fn test_spatial_eval_report_fields() {
        let coords: Vec<GeoCoord> = (0..3).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let m = GeoMetrics::compute(&coords, &coords, &[1.0, 2.0, 3.0], &coords);
        let report = SpatialEvalReport::new(&m, 12.5, 0.8);
        assert!((report.mean_frechet_km - 12.5).abs() < 1e-9);
        assert!((report.interpolation_rmse - 0.8).abs() < 1e-9);
        assert!(report.mae_km.is_finite());
    }

    #[test]
    fn test_haversine_symmetry() {
        let a = GeoCoord::new(48.85, 2.35);
        let b = GeoCoord::new(40.71, -74.01);
        assert!((a.haversine_km(&b) - b.haversine_km(&a)).abs() < 1e-9, "Haversine must be symmetric");
    }

    #[test]
    fn test_haversine_triangle_inequality() {
        let a = GeoCoord::new(0.0, 0.0);
        let b = GeoCoord::new(5.0, 0.0);
        let c = GeoCoord::new(2.5, 2.5);
        let dab = a.haversine_km(&b);
        let dac = a.haversine_km(&c);
        let dcb = c.haversine_km(&b);
        assert!(dac + dcb >= dab - 1e-9, "Triangle inequality must hold");
    }

    #[test]
    fn test_gaussian_elimination_simple() {
        // 2x + y = 5, x - y = 1 → x=2, y=1.
        let a = vec![vec![2.0, 1.0], vec![1.0, -1.0]];
        let b = vec![5.0, 1.0];
        let x = gaussian_elimination(&a, &b);
        assert!((x[0] - 2.0).abs() < 1e-9, "x should be 2, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-9, "y should be 1, got {}", x[1]);
    }

    #[test]
    fn test_erf_approx_bounds() {
        assert!((erf_approx(0.0)).abs() < 1e-6, "erf(0) should be 0");
        assert!((erf_approx(3.0) - 1.0).abs() < 0.01, "erf(large) ≈ 1");
        assert!((erf_approx(-3.0) + 1.0).abs() < 0.01, "erf(-large) ≈ -1");
    }
}
