//! Geospatial Machine Learning Components.
//!
//! Sections:
//! 1. SpatialFeatureEncoding — coordinate encoding, H3-style grid, quadkey, sinusoidal, tokenizer
//! 2. SpatialGraphNetworks   — KNN graph, spatial GCN, spatial attention, urban computing GNN
//! 3. TrajectoryAnalysis     — GPS trajectory, LSTM encoder, frequency map, DBSCAN cluster, HMM Viterbi
//! 4. SpatialInterpolation   — Ordinary Kriging, IDW, RBF, spatial cross-validation
//! 5. SpatialTemporalModels  — ST-GCN, diffusion convolution, geo attention O-D demand
//! 6. GeoMetrics             — haversine MAE/RMSE, MAPE, Moran's I, SpatialEvalReport

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use std::f64::consts::PI;

// ── Small shared math helpers ─────────────────────────────────────────────────

pub(crate) const EARTH_RADIUS_KM: f64 = 6371.0;

#[inline]
pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp())
}

#[inline]
pub(crate) fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

pub(crate) fn softmax_inplace(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0_f64;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        s += *x;
    }
    let inv = 1.0 / s.max(f64::EPSILON);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Dense matrix-vector product: W [out×in], x [in] → y [out].
pub(crate) fn mat_vec(w: &[Vec<f64>], x: &[f64], bias: &[f64]) -> Vec<f64> {
    w.iter()
        .enumerate()
        .map(|(o, row)| {
            let b = bias.get(o).copied().unwrap_or(0.0);
            row.iter().zip(x.iter()).fold(b, |a, (&wi, &xi)| a + wi * xi)
        })
        .collect()
}

/// Xavier uniform initialiser.
pub(crate) fn xavier_matrix(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let lim = (6.0 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_out)
        .map(|_| {
            (0..fan_in)
                .map(|_| {
                    let u: f64 = rng.random();
                    u * 2.0 * lim - lim
                })
                .collect()
        })
        .collect()
}

pub(crate) fn zeros(n: usize) -> Vec<f64> {
    vec![0.0; n]
}

// ═════════════════════════════════════════════════════════════════════════════
// §1  Spatial Feature Encoding
// ═════════════════════════════════════════════════════════════════════════════

/// A WGS-84 geographic coordinate (latitude, longitude in decimal degrees).
#[derive(Clone, Debug, PartialEq)]
pub struct GeoCoord {
    /// Latitude in degrees (–90 to +90).
    pub lat: f64,
    /// Longitude in degrees (–180 to +180).
    pub lon: f64,
}

impl GeoCoord {
    /// Create a new coordinate.
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Haversine great-circle distance to another point in kilometres.
    pub fn haversine_km(&self, other: &GeoCoord) -> f64 {
        haversine_km(self.lat, self.lon, other.lat, other.lon)
    }

    /// Bearing (azimuth) in degrees [0, 360) from self to other.
    pub fn bearing_to(&self, other: &GeoCoord) -> f64 {
        let lat1 = self.lat.to_radians();
        let lat2 = other.lat.to_radians();
        let dlon = (other.lon - self.lon).to_radians();
        let y = dlon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
        let angle = y.atan2(x).to_degrees();
        (angle + 360.0) % 360.0
    }
}

/// Haversine distance in km between two lat/lon pairs.
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let lat1r = lat1.to_radians();
    let lat2r = lat2.to_radians();
    let a = (dlat / 2.0).sin().powi(2) + lat1r.cos() * lat2r.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * a.sqrt().asin()
}

/// Approximate H3-style hexagonal grid cell ID via sinusoidal bucketing.
///
/// Encodes (lat, lon) into a 64-bit cell identifier at a resolution level
/// (0 = coarsest, 15 = finest). No dependency on the h3 crate; this is an
/// approximate spatial index that preserves locality.
#[derive(Clone, Debug)]
pub struct H3GridEncoder {
    /// Resolution level (0–15). Higher = finer grid.
    pub resolution: u8,
}

impl H3GridEncoder {
    /// Create an encoder at the given resolution.
    pub fn new(resolution: u8) -> Self {
        Self {
            resolution: resolution.min(15),
        }
    }

    /// Encode a coordinate into an approximate hexagonal cell ID.
    pub fn encode(&self, coord: &GeoCoord) -> u64 {
        // Scale factor: each resolution halves the cell size approximately.
        let scale = (1u64 << self.resolution) as f64;
        // Sinusoidal projection: x accounts for longitude compression by latitude.
        let x = (coord.lon + 180.0) / 360.0 * scale * coord.lat.to_radians().cos();
        let y = (coord.lat + 90.0) / 180.0 * scale;
        // Hex grid: offset every other row by half a cell.
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let ix_hex = if iy % 2 == 0 { ix } else { ix };
        // Cantor pairing to produce a unique 64-bit ID.
        let res_bits = (self.resolution as u64) << 56;
        let combined = (((ix_hex + 1_000_000) as u64) << 28) ^ ((iy + 1_000_000) as u64);
        res_bits | (combined & 0x00FFFFFFFFFFFFFF)
    }

    /// Decode a cell ID to an approximate centre coordinate.
    pub fn decode(&self, cell_id: u64) -> GeoCoord {
        let scale = (1u64 << self.resolution) as f64;
        let inner = cell_id & 0x00FFFFFFFFFFFFFF;
        let iy_raw = (inner & 0x0FFFFFFF) as i64 - 1_000_000;
        let ix_raw = ((inner >> 28) & 0x0FFFFFFF) as i64 - 1_000_000;
        let lat = (iy_raw as f64 + 0.5) / scale * 180.0 - 90.0;
        let lat_cos = lat.to_radians().cos().max(1e-8);
        let lon = (ix_raw as f64 + 0.5) / (scale * lat_cos) * 360.0 - 180.0;
        GeoCoord::new(lat.clamp(-90.0, 90.0), lon.clamp(-180.0, 180.0))
    }
}

/// Bing Maps quadkey encoder (Web Mercator tile system).
///
/// Each character in the quadkey string ('0'..'3') represents one level of the tile
/// quadtree. Level 1 = 4 tiles; level N = 4^N tiles.
#[derive(Clone, Debug)]
pub struct QuadkeyEncoder {
    /// Tile zoom level (1–23).
    pub level: u8,
}

impl QuadkeyEncoder {
    /// Create an encoder at the given zoom level.
    pub fn new(level: u8) -> Self {
        Self {
            level: level.clamp(1, 23),
        }
    }

    /// Encode lat/lon to a quadkey string of length `level`.
    pub fn encode(&self, lat: f64, lon: f64) -> String {
        let n = 1u32 << self.level;
        // Mercator projection to pixel tile coords.
        let sin_lat = lat.to_radians().sin().clamp(-0.9999, 0.9999);
        let pixel_x = ((lon + 180.0) / 360.0 * n as f64).min(n as f64 - 1e-10) as u32;
        let pixel_y = ((0.5 - ((1.0 + sin_lat) / (1.0 - sin_lat)).ln() / (4.0 * PI)) * n as f64)
            .clamp(0.0, n as f64 - 1e-10) as u32;
        let mut qk = String::with_capacity(self.level as usize);
        for i in (0..self.level).rev() {
            let mask = 1u32 << i;
            let digit = match (pixel_x & mask != 0, pixel_y & mask != 0) {
                (false, false) => '0',
                (true, false) => '1',
                (false, true) => '2',
                (true, true) => '3',
            };
            qk.push(digit);
        }
        qk
    }

    /// Decode a quadkey string to the bounding box (min_lat, min_lon, max_lat, max_lon).
    pub fn decode(&self, quadkey: &str) -> (f64, f64, f64, f64) {
        let n = 1u32 << self.level;
        let mut x = 0u32;
        let mut y = 0u32;
        for (i, ch) in quadkey.chars().enumerate() {
            let mask = 1u32 << (self.level as usize - 1 - i);
            match ch {
                '1' | '3' => x |= mask,
                _ => {}
            }
            match ch {
                '2' | '3' => y |= mask,
                _ => {}
            }
        }
        // Inverse Web Mercator: lat = atan(sinh(π × (1 − 2·pixel_y/n))) in degrees.
        let to_lat = |pixel_y: u32| -> f64 {
            let t = 1.0 - 2.0 * pixel_y as f64 / n as f64;
            (PI * t).sinh().atan().to_degrees()
        };
        let to_lon = |pixel_x: u32| -> f64 { pixel_x as f64 / n as f64 * 360.0 - 180.0 };
        (to_lat(y + 1), to_lon(x), to_lat(y), to_lon(x + 1))
    }
}

/// Positional encoding for geographic coordinates using sinusoidal frequency bands.
///
/// Similar to NeRF Fourier features: maps (lat, lon) to a high-dimensional
/// vector using sin/cos at multiple frequency scales.
#[derive(Clone, Debug)]
pub struct SpatialSinusoidalEncoding {
    /// Number of frequency bands.
    pub num_frequencies: usize,
    /// Base for frequency progression (e.g. 2.0 → octave spacing).
    pub freq_base: f64,
}

impl SpatialSinusoidalEncoding {
    /// Create with default settings (8 frequencies, base 2).
    pub fn new(num_frequencies: usize, freq_base: f64) -> Self {
        Self { num_frequencies, freq_base }
    }

    /// Output dimension: 2 (raw lat/lon) + 4 * num_frequencies (sin/cos for each).
    pub fn output_dim(&self) -> usize {
        2 + 4 * self.num_frequencies
    }

    /// Encode a coordinate to a feature vector.
    pub fn encode(&self, coord: &GeoCoord) -> Vec<f64> {
        let mut feat = Vec::with_capacity(self.output_dim());
        // Normalise to [−1, 1].
        let lat_n = coord.lat / 90.0;
        let lon_n = coord.lon / 180.0;
        feat.push(lat_n);
        feat.push(lon_n);
        for k in 0..self.num_frequencies {
            let freq = self.freq_base.powi(k as i32) * PI;
            feat.push((freq * lat_n).sin());
            feat.push((freq * lat_n).cos());
            feat.push((freq * lon_n).sin());
            feat.push((freq * lon_n).cos());
        }
        feat
    }

    /// Batch encode a slice of coordinates.
    pub fn encode_batch(&self, coords: &[GeoCoord]) -> Vec<Vec<f64>> {
        coords.iter().map(|c| self.encode(c)).collect()
    }
}

/// Discretise continuous coordinates into a fixed vocabulary of spatial tokens.
///
/// The world is divided into a `lat_bins × lon_bins` regular grid; each cell
/// gets a unique integer token ID.
#[derive(Clone, Debug)]
pub struct GeoTokenizer {
    /// Number of latitude bins.
    pub lat_bins: usize,
    /// Number of longitude bins.
    pub lon_bins: usize,
}

impl GeoTokenizer {
    /// Create a new tokenizer with the given bin counts.
    pub fn new(lat_bins: usize, lon_bins: usize) -> Self {
        Self { lat_bins, lon_bins }
    }

    /// Total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.lat_bins * self.lon_bins
    }

    /// Convert a coordinate to a token ID.
    pub fn encode(&self, coord: &GeoCoord) -> usize {
        let lat_idx = ((coord.lat + 90.0) / 180.0 * self.lat_bins as f64)
            .floor()
            .clamp(0.0, (self.lat_bins - 1) as f64) as usize;
        let lon_idx = ((coord.lon + 180.0) / 360.0 * self.lon_bins as f64)
            .floor()
            .clamp(0.0, (self.lon_bins - 1) as f64) as usize;
        lat_idx * self.lon_bins + lon_idx
    }

    /// Decode a token ID back to the centre coordinate of its grid cell.
    pub fn decode(&self, token: usize) -> GeoCoord {
        let lat_idx = token / self.lon_bins;
        let lon_idx = token % self.lon_bins;
        let lat = (lat_idx as f64 + 0.5) / self.lat_bins as f64 * 180.0 - 90.0;
        let lon = (lon_idx as f64 + 0.5) / self.lon_bins as f64 * 360.0 - 180.0;
        GeoCoord::new(lat, lon)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §2  Spatial Graph Networks
// ═════════════════════════════════════════════════════════════════════════════

/// An edge in the spatial graph with distance and bearing features.
#[derive(Clone, Debug)]
pub struct SpatialEdge {
    /// Destination node index.
    pub dst: usize,
    /// Great-circle distance in km.
    pub distance_km: f64,
    /// Bearing in degrees [0, 360).
    pub bearing_deg: f64,
}

/// K-nearest-neighbour graph built on geographic coordinates.
///
/// Edge features are distance (km) and bearing (degrees).
#[derive(Clone, Debug)]
pub struct SpatialGraph {
    /// Node coordinates.
    pub coords: Vec<GeoCoord>,
    /// Node feature vectors (optional, one per node).
    pub node_features: Vec<Vec<f64>>,
    /// Adjacency: adj\[i\] = list of edges from node i.
    pub adj: Vec<Vec<SpatialEdge>>,
    /// Number of neighbours k.
    pub k: usize,
}

impl SpatialGraph {
    /// Build a k-NN spatial graph from coordinates and optional node features.
    pub fn build(coords: Vec<GeoCoord>, node_features: Vec<Vec<f64>>, k: usize) -> Self {
        let n = coords.len();
        let k = k.min(n.saturating_sub(1));
        let mut adj: Vec<Vec<SpatialEdge>> = vec![Vec::new(); n];
        for i in 0..n {
            // Compute distances to all other nodes.
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, coords[i].haversine_km(&coords[j])))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (j, dist) in dists.into_iter().take(k) {
                let bearing = coords[i].bearing_to(&coords[j]);
                adj[i].push(SpatialEdge { dst: j, distance_km: dist, bearing_deg: bearing });
            }
        }
        Self { coords, node_features, adj, k }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }
}

/// Spatial GCN layer using inverse-distance-weighted aggregation.
///
/// For each node i, aggregates neighbours weighted by 1/d_ij (softmax-normalised),
/// then applies a linear transform + ReLU.
#[derive(Clone, Debug)]
pub struct SpatialGcnLayer {
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    weights: Vec<Vec<f64>>, // [out_dim × in_dim]
    bias: Vec<f64>,
    /// Temperature for softmax over inverse distances.
    pub temperature: f64,
}

impl SpatialGcnLayer {
    /// Create a new layer with Xavier-initialised weights.
    pub fn new(in_dim: usize, out_dim: usize, temperature: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let weights = xavier_matrix(in_dim, out_dim, &mut rng);
        Self { in_dim, out_dim, weights, bias: zeros(out_dim), temperature }
    }

    /// Forward pass: returns new node features [n × out_dim].
    pub fn forward(&self, graph: &SpatialGraph, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();
        let mut out = vec![zeros(self.out_dim); n];
        for i in 0..n {
            let edges = &graph.adj[i];
            // Compute IDW weights.
            let inv_dists: Vec<f64> = edges
                .iter()
                .map(|e| 1.0 / (e.distance_km.max(1e-6) * self.temperature))
                .collect();
            // Softmax-normalise.
            let max_id = inv_dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = inv_dists.iter().map(|&v| (v - max_id).exp()).collect();
            let sum_exp = exp_vals.iter().sum::<f64>().max(f64::EPSILON);
            // Aggregate.
            let mut agg = if x.get(i).map(|r| r.len()).unwrap_or(0) >= self.in_dim {
                x[i][..self.in_dim].to_vec()
            } else {
                zeros(self.in_dim)
            };
            for (edge, &ew) in edges.iter().zip(exp_vals.iter()) {
                let w = ew / sum_exp;
                let nb = x.get(edge.dst).and_then(|r| if r.len() >= self.in_dim { Some(&r[..self.in_dim]) } else { None });
                if let Some(nb_feat) = nb {
                    for d in 0..self.in_dim {
                        agg[d] += w * nb_feat[d];
                    }
                }
            }
            // Linear + ReLU.
            let h = mat_vec(&self.weights, &agg, &self.bias);
            out[i] = h.into_iter().map(relu).collect();
        }
        out
    }
}

/// Spatial attention layer: attention score = Gaussian RBF of distance.
///
/// For each node i, computes attention over neighbours using exp(−d²/2σ²),
/// then applies a value projection.
#[derive(Clone, Debug)]
pub struct SpatialAttentionLayer {
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    /// Bandwidth σ (km) for the Gaussian RBF kernel.
    pub sigma_km: f64,
    value_weights: Vec<Vec<f64>>, // [out_dim × in_dim]
    value_bias: Vec<f64>,
}

impl SpatialAttentionLayer {
    /// Create a new spatial attention layer.
    pub fn new(in_dim: usize, out_dim: usize, sigma_km: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let value_weights = xavier_matrix(in_dim, out_dim, &mut rng);
        Self {
            in_dim,
            out_dim,
            sigma_km,
            value_weights,
            value_bias: zeros(out_dim),
        }
    }

    /// Forward pass. Returns updated node embeddings [n × out_dim].
    pub fn forward(&self, graph: &SpatialGraph, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();
        let two_sigma2 = 2.0 * self.sigma_km * self.sigma_km;
        let mut out = vec![zeros(self.out_dim); n];
        for i in 0..n {
            let edges = &graph.adj[i];
            // RBF attention weights.
            let attn_raw: Vec<f64> = edges
                .iter()
                .map(|e| (-e.distance_km.powi(2) / two_sigma2).exp())
                .collect();
            // Softmax.
            let max_a = attn_raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = attn_raw.iter().map(|&a| (a - max_a).exp()).collect();
            let sum_e = exps.iter().sum::<f64>().max(f64::EPSILON);
            let mut agg = zeros(self.in_dim);
            for (edge, &ew) in edges.iter().zip(exps.iter()) {
                let w = ew / sum_e;
                if let Some(nb) = x.get(edge.dst) {
                    for d in 0..self.in_dim.min(nb.len()) {
                        agg[d] += w * nb[d];
                    }
                }
            }
            out[i] = mat_vec(&self.value_weights, &agg, &self.value_bias)
                .into_iter()
                .map(relu)
                .collect();
        }
        out
    }
}

/// Point-of-Interest feature types for urban region embedding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PoiCategory {
    /// Food and dining.
    Food,
    /// Transport hubs.
    Transport,
    /// Shopping and retail.
    Retail,
    /// Entertainment and leisure.
    Entertainment,
    /// Healthcare.
    Healthcare,
    /// Education.
    Education,
    /// Office and business.
    Office,
    /// Residential.
    Residential,
}

/// Urban region node with aggregated POI feature histogram.
#[derive(Clone, Debug)]
pub struct UrbanRegion {
    /// Region centroid coordinate.
    pub coord: GeoCoord,
    /// POI category counts (one per PoiCategory variant).
    pub poi_counts: Vec<f64>,
    /// Additional scalar features (population density, etc.).
    pub extra_features: Vec<f64>,
}

impl UrbanRegion {
    /// Construct with 8 POI categories.
    pub fn new(coord: GeoCoord, poi_counts: Vec<f64>, extra_features: Vec<f64>) -> Self {
        Self { coord, poi_counts, extra_features }
    }

    /// Concatenated feature vector: poi_counts ++ extra_features.
    pub fn feature_vec(&self) -> Vec<f64> {
        let mut v = self.poi_counts.clone();
        v.extend_from_slice(&self.extra_features);
        v
    }
}

/// Graph neural network for urban region embedding (POI-aware).
///
/// Two-layer spatial GCN over a k-NN region graph.
#[derive(Clone, Debug)]
pub struct UrbanComputingGnn {
    layer1: SpatialGcnLayer,
    layer2: SpatialGcnLayer,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl UrbanComputingGnn {
    /// Create a two-layer urban GNN.
    pub fn new(in_dim: usize, hidden_dim: usize, embed_dim: usize, seed: u64) -> Self {
        Self {
            layer1: SpatialGcnLayer::new(in_dim, hidden_dim, 1.0, seed),
            layer2: SpatialGcnLayer::new(hidden_dim, embed_dim, 1.0, seed + 1),
            embed_dim,
        }
    }

    /// Embed urban regions given a spatial graph.
    pub fn embed(&self, graph: &SpatialGraph, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let h = self.layer1.forward(graph, x);
        self.layer2.forward(graph, &h)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §3  Trajectory Analysis
// ═════════════════════════════════════════════════════════════════════════════

/// A GPS trajectory observation.
#[derive(Clone, Debug)]
pub struct TrajectoryPoint {
    /// Coordinate.
    pub coord: GeoCoord,
    /// Unix timestamp (seconds).
    pub timestamp: f64,
}

impl TrajectoryPoint {
    /// Construct a new point.
    pub fn new(lat: f64, lon: f64, timestamp: f64) -> Self {
        Self { coord: GeoCoord::new(lat, lon), timestamp }
    }
}

/// A GPS trajectory: ordered sequence of (lat, lon, timestamp) observations.
#[derive(Clone, Debug)]
pub struct GeoTrajectory {
    /// Ordered list of trajectory points.
    pub points: Vec<TrajectoryPoint>,
}

impl GeoTrajectory {
    /// Create from a list of points.
    pub fn new(points: Vec<TrajectoryPoint>) -> Self {
        Self { points }
    }

    /// Total length of the trajectory in km.
    pub fn length_km(&self) -> f64 {
        self.points
            .windows(2)
            .map(|w| w[0].coord.haversine_km(&w[1].coord))
            .sum()
    }

    /// Duration in seconds.
    pub fn duration_s(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let first = self.points.first().map(|p| p.timestamp).unwrap_or(0.0);
        let last = self.points.last().map(|p| p.timestamp).unwrap_or(0.0);
        (last - first).max(0.0)
    }

    /// Average speed in km/h.
    pub fn avg_speed_kmh(&self) -> f64 {
        let dur_h = self.duration_s() / 3600.0;
        if dur_h < 1e-9 {
            return 0.0;
        }
        self.length_km() / dur_h
    }
}

/// LSTM-based encoder for GPS trajectories.
///
/// Encodes a variable-length trajectory into a fixed-size embedding using a
/// 1-layer LSTM over (normalised_lat, normalised_lon, speed_kmh) inputs.
#[derive(Clone, Debug)]
pub struct TrajectoryEncoder {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Input feature dimension (3: lat, lon, speed).
    pub input_dim: usize,
    // LSTM weights — gates order: [i, f, g, o] each [hidden × (input + hidden)].
    wh: Vec<Vec<f64>>, // [4*hidden × (input+hidden)]
    bh: Vec<f64>,       // [4*hidden]
}

impl TrajectoryEncoder {
    /// Create a new trajectory encoder.
    pub fn new(hidden_dim: usize, seed: u64) -> Self {
        let input_dim = 3;
        let concat_dim = input_dim + hidden_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let wh = xavier_matrix(concat_dim, 4 * hidden_dim, &mut rng);
        let bh = zeros(4 * hidden_dim);
        Self { hidden_dim, input_dim, wh, bh }
    }

    fn lstm_step(&self, h: &[f64], c: &[f64], x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut concat = x.to_vec();
        concat.extend_from_slice(h);
        let gates = mat_vec(&self.wh, &concat, &self.bh);
        let hd = self.hidden_dim;
        let i_gate: Vec<f64> = gates[0..hd].iter().map(|&v| sigmoid(v)).collect();
        let f_gate: Vec<f64> = gates[hd..2 * hd].iter().map(|&v| sigmoid(v)).collect();
        let g_gate: Vec<f64> = gates[2 * hd..3 * hd].iter().map(|&v| tanh_act(v)).collect();
        let o_gate: Vec<f64> = gates[3 * hd..4 * hd].iter().map(|&v| sigmoid(v)).collect();
        let mut new_c = vec![0.0; hd];
        let mut new_h = vec![0.0; hd];
        for d in 0..hd {
            new_c[d] = f_gate[d] * c[d] + i_gate[d] * g_gate[d];
            new_h[d] = o_gate[d] * new_c[d].tanh();
        }
        (new_h, new_c)
    }

    /// Encode a trajectory; returns the final hidden state.
    pub fn encode(&self, traj: &GeoTrajectory) -> Vec<f64> {
        let n = traj.points.len();
        if n == 0 {
            return zeros(self.hidden_dim);
        }
        let mut h = zeros(self.hidden_dim);
        let mut c = zeros(self.hidden_dim);
        // Compute step-wise speeds.
        let speeds: Vec<f64> = {
            let mut s = vec![0.0f64; n];
            for i in 1..n {
                let d = traj.points[i - 1].coord.haversine_km(&traj.points[i].coord);
                let dt = (traj.points[i].timestamp - traj.points[i - 1].timestamp).max(1e-9);
                s[i] = d / (dt / 3600.0); // km/h
            }
            s
        };
        for (pt, spd) in traj.points.iter().zip(speeds.iter()) {
            let x = [pt.coord.lat / 90.0, pt.coord.lon / 180.0, spd / 200.0];
            let (nh, nc) = self.lstm_step(&h, &c, &x);
            h = nh;
            c = nc;
        }
        h
    }
}

/// Rasterise a collection of trajectories into a 2D density heatmap.
///
/// Each cell accumulates the number of trajectory points that fall within it.
#[derive(Clone, Debug)]
pub struct FrequencyMapEncoding {
    /// Grid resolution in latitude direction.
    pub lat_bins: usize,
    /// Grid resolution in longitude direction.
    pub lon_bins: usize,
    /// Bounding box: (min_lat, max_lat, min_lon, max_lon).
    pub bbox: (f64, f64, f64, f64),
}

impl FrequencyMapEncoding {
    /// Create a new frequency map.
    pub fn new(lat_bins: usize, lon_bins: usize, bbox: (f64, f64, f64, f64)) -> Self {
        Self { lat_bins, lon_bins, bbox }
    }

    /// Rasterise trajectories to a flat heatmap grid of size lat_bins × lon_bins.
    pub fn rasterize(&self, trajectories: &[GeoTrajectory]) -> Vec<f64> {
        let mut grid = vec![0.0_f64; self.lat_bins * self.lon_bins];
        let (min_lat, max_lat, min_lon, max_lon) = self.bbox;
        let lat_range = (max_lat - min_lat).max(f64::EPSILON);
        let lon_range = (max_lon - min_lon).max(f64::EPSILON);
        for traj in trajectories {
            for pt in &traj.points {
                let lat_idx = ((pt.coord.lat - min_lat) / lat_range * self.lat_bins as f64)
                    .floor()
                    .clamp(0.0, (self.lat_bins - 1) as f64) as usize;
                let lon_idx = ((pt.coord.lon - min_lon) / lon_range * self.lon_bins as f64)
                    .floor()
                    .clamp(0.0, (self.lon_bins - 1) as f64) as usize;
                grid[lat_idx * self.lon_bins + lon_idx] += 1.0;
            }
        }
        grid
    }
}

/// Approximate Fréchet distance between two trajectories using dynamic programming.
///
/// O(n·m) DP on the "leash" length matrix; uses haversine distances (km).
pub fn frechet_distance_km(a: &GeoTrajectory, b: &GeoTrajectory) -> f64 {
    let n = a.points.len();
    let m = b.points.len();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let d = |i: usize, j: usize| -> f64 {
        a.points[i].coord.haversine_km(&b.points[j].coord)
    };
    let mut ca = vec![vec![f64::INFINITY; m]; n];
    ca[0][0] = d(0, 0);
    for j in 1..m {
        ca[0][j] = ca[0][j - 1].max(d(0, j));
    }
    for i in 1..n {
        ca[i][0] = ca[i - 1][0].max(d(i, 0));
    }
    for i in 1..n {
        for j in 1..m {
            let prev = ca[i - 1][j].min(ca[i - 1][j - 1]).min(ca[i][j - 1]);
            ca[i][j] = prev.max(d(i, j));
        }
    }
    ca[n - 1][m - 1]
}

/// DBSCAN clustering of trajectories using Fréchet distance.
///
/// Each trajectory is assigned a cluster label (−1 = noise, ≥ 0 = cluster).
#[derive(Clone, Debug)]
pub struct TrajectoryClusterer {
    /// Epsilon neighbourhood radius in km.
    pub eps_km: f64,
    /// Minimum trajectories to form a core.
    pub min_trajectories: usize,
}

impl TrajectoryClusterer {
    /// Create a new clusterer.
    pub fn new(eps_km: f64, min_trajectories: usize) -> Self {
        Self { eps_km, min_trajectories }
    }

    /// Cluster trajectories; returns cluster labels (−1 = noise).
    pub fn cluster(&self, trajectories: &[GeoTrajectory]) -> Vec<i32> {
        let n = trajectories.len();
        let mut labels = vec![-1i32; n];
        let mut visited = vec![false; n];
        let mut cluster_id = 0i32;

        for i in 0..n {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            let mut neighbours: Vec<usize> = (0..n)
                .filter(|&j| j != i && frechet_distance_km(&trajectories[i], &trajectories[j]) <= self.eps_km)
                .collect();
            if neighbours.len() + 1 < self.min_trajectories {
                // Noise for now; may be updated.
                continue;
            }
            labels[i] = cluster_id;
            let mut qi = 0;
            while qi < neighbours.len() {
                let j = neighbours[qi];
                if !visited[j] {
                    visited[j] = true;
                    let nn: Vec<usize> = (0..n)
                        .filter(|&k| k != j && frechet_distance_km(&trajectories[j], &trajectories[k]) <= self.eps_km)
                        .collect();
                    if nn.len() + 1 >= self.min_trajectories {
                        for &k in &nn {
                            if !neighbours.contains(&k) {
                                neighbours.push(k);
                            }
                        }
                    }
                }
                if labels[j] < 0 {
                    labels[j] = cluster_id;
                }
                qi += 1;
            }
            cluster_id += 1;
        }
        labels
    }
}

/// Movement state from the HMM Viterbi decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MovementState {
    /// Device is stationary.
    Stop,
    /// Device is in motion.
    Move,
}

/// Detect stops vs. moves in a trajectory using speed threshold + HMM Viterbi.
#[derive(Clone, Debug)]
pub struct MovementPatternDetector {
    /// Speed threshold (km/h) below which a point is considered stationary.
    pub stop_threshold_kmh: f64,
    /// HMM: probability of staying in Stop state.
    pub p_stay_stop: f64,
    /// HMM: probability of staying in Move state.
    pub p_stay_move: f64,
    /// Emission probability: P(obs=stop | state=Stop).
    pub p_emit_stop_in_stop: f64,
    /// Emission probability: P(obs=move | state=Move).
    pub p_emit_move_in_move: f64,
}

impl MovementPatternDetector {
    /// Create a detector with sensible defaults.
    pub fn new(stop_threshold_kmh: f64) -> Self {
        Self {
            stop_threshold_kmh,
            p_stay_stop: 0.9,
            p_stay_move: 0.85,
            p_emit_stop_in_stop: 0.9,
            p_emit_move_in_move: 0.9,
        }
    }

    /// Detect movement states for each trajectory point via Viterbi.
    pub fn detect(&self, traj: &GeoTrajectory) -> Vec<MovementState> {
        let n = traj.points.len();
        if n == 0 {
            return Vec::new();
        }
        // Observation: true = "looks stopped" based on speed.
        let obs_stop: Vec<bool> = {
            let mut s = vec![true; n];
            for i in 1..n {
                let d = traj.points[i - 1].coord.haversine_km(&traj.points[i].coord);
                let dt = (traj.points[i].timestamp - traj.points[i - 1].timestamp).max(1e-9);
                let speed = d / (dt / 3600.0);
                s[i] = speed < self.stop_threshold_kmh;
            }
            s
        };
        // Viterbi — 2 states: 0=Stop, 1=Move.
        let log_trans = [
            [self.p_stay_stop.ln(), (1.0 - self.p_stay_stop).ln()],
            [(1.0 - self.p_stay_move).ln(), self.p_stay_move.ln()],
        ];
        let emit = |state: usize, obs_is_stop: bool| -> f64 {
            if state == 0 {
                if obs_is_stop { self.p_emit_stop_in_stop } else { 1.0 - self.p_emit_stop_in_stop }
            } else {
                if obs_is_stop { 1.0 - self.p_emit_move_in_move } else { self.p_emit_move_in_move }
            }
            .ln()
        };
        let mut viterbi = vec![[0.0_f64; 2]; n];
        let mut back = vec![[0usize; 2]; n];
        for s in 0..2 {
            viterbi[0][s] = 0.5_f64.ln() + emit(s, obs_stop[0]);
        }
        for t in 1..n {
            for s in 0..2 {
                let e = emit(s, obs_stop[t]);
                let (best_score, best_prev) = (0..2)
                    .map(|prev| (viterbi[t - 1][prev] + log_trans[prev][s] + e, prev))
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((f64::NEG_INFINITY, 0));
                viterbi[t][s] = best_score;
                back[t][s] = best_prev;
            }
        }
        // Backtrack.
        let mut states = vec![0usize; n];
        states[n - 1] = if viterbi[n - 1][0] >= viterbi[n - 1][1] { 0 } else { 1 };
        for t in (0..n - 1).rev() {
            states[t] = back[t + 1][states[t + 1]];
        }
        states
            .into_iter()
            .map(|s| if s == 0 { MovementState::Stop } else { MovementState::Move })
            .collect()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §4  Spatial Interpolation
// ═════════════════════════════════════════════════════════════════════════════

/// Ordinary Kriging with a spherical variogram model.
///
/// Variogram: γ(h) = nugget + sill * (1.5 * h/range − 0.5 * (h/range)³)  for h ≤ range
///                  = nugget + sill                                          for h > range
#[derive(Clone, Debug)]
pub struct OrdinaryKriging {
    /// Nugget effect (variance at distance 0).
    pub nugget: f64,
    /// Sill (partial sill above nugget).
    pub sill: f64,
    /// Range parameter in km.
    pub range_km: f64,
    /// Known sample locations.
    pub sample_coords: Vec<GeoCoord>,
    /// Known sample values.
    pub sample_values: Vec<f64>,
}

impl OrdinaryKriging {
    /// Construct with known samples.
    pub fn new(
        nugget: f64,
        sill: f64,
        range_km: f64,
        sample_coords: Vec<GeoCoord>,
        sample_values: Vec<f64>,
    ) -> Self {
        Self { nugget, sill, range_km, sample_coords, sample_values }
    }

    fn variogram(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        let hr = h / self.range_km;
        if hr >= 1.0 {
            self.nugget + self.sill
        } else {
            self.nugget + self.sill * (1.5 * hr - 0.5 * hr.powi(3))
        }
    }

    /// Predict value at an unsampled location using Ordinary Kriging.
    /// Returns (prediction, kriging_variance).
    pub fn predict(&self, target: &GeoCoord) -> (f64, f64) {
        let n = self.sample_coords.len();
        if n == 0 {
            return (0.0, self.nugget + self.sill);
        }
        // Build (n+1) × (n+1) kriging system with Lagrange multiplier.
        let size = n + 1;
        let mut a = vec![vec![0.0_f64; size]; size];
        for i in 0..n {
            for j in 0..n {
                let h = self.sample_coords[i].haversine_km(&self.sample_coords[j]);
                a[i][j] = self.variogram(h);
            }
            a[i][n] = 1.0;
            a[n][i] = 1.0;
        }
        // RHS: variogram to target.
        let mut b = vec![0.0_f64; size];
        for i in 0..n {
            let h = self.sample_coords[i].haversine_km(target);
            b[i] = self.variogram(h);
        }
        b[n] = 1.0;
        // Solve via Gaussian elimination.
        let weights = gaussian_elimination(&a, &b);
        let pred: f64 = weights.iter().take(n).zip(self.sample_values.iter()).map(|(&w, &z)| w * z).sum();
        let variance: f64 = weights
            .iter()
            .zip(b.iter())
            .take(n)
            .map(|(&w, &bv)| w * bv)
            .sum::<f64>()
            + weights.get(n).copied().unwrap_or(0.0);
        (pred, variance.max(0.0))
    }
}

/// Gaussian elimination to solve Ax = b; returns x.
pub(crate) fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();
    for col in 0..n {
        // Partial pivot.
        let pivot_row = (col..n)
            .max_by(|&i, &j| {
                aug[i][col].abs().partial_cmp(&aug[j][col].abs()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for j in col..=n {
            aug[col][j] /= pivot;
        }
        for i in 0..n {
            if i == col {
                continue;
            }
            let factor = aug[i][col];
            for j in col..=n {
                let sub = factor * aug[col][j];
                aug[i][j] -= sub;
            }
        }
    }
    aug.iter().map(|row| row.get(n).copied().unwrap_or(0.0)).collect()
}

/// Inverse Distance Weighting spatial interpolator.
#[derive(Clone, Debug)]
pub struct IdwInterpolator {
    /// Power parameter (typically 1 or 2).
    pub power: f64,
    /// Number of nearest neighbours to use.
    pub k: usize,
    /// Sample coordinates.
    pub sample_coords: Vec<GeoCoord>,
    /// Sample values.
    pub sample_values: Vec<f64>,
}

impl IdwInterpolator {
    /// Construct with known samples.
    pub fn new(power: f64, k: usize, sample_coords: Vec<GeoCoord>, sample_values: Vec<f64>) -> Self {
        Self { power, k, sample_coords, sample_values }
    }

    /// Predict value at an unsampled location.
    pub fn predict(&self, target: &GeoCoord) -> f64 {
        let n = self.sample_coords.len();
        if n == 0 {
            return 0.0;
        }
        let mut dists: Vec<(usize, f64)> = self
            .sample_coords
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.haversine_km(target)))
            .collect();
        // Check for exact hit.
        if let Some(&(i, _d)) = dists.iter().find(|&&(_, d)| d < 1e-9) {
            return self.sample_values[i];
        }
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.k.min(n);
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &(i, d) in dists.iter().take(k) {
            let w = 1.0 / d.powf(self.power);
            num += w * self.sample_values[i];
            den += w;
        }
        if den < f64::EPSILON { 0.0 } else { num / den }
    }
}
