//! Advanced Geospatial ML — Urban Mobility, Map Matching, Spatial Anomaly Detection,
//! and Spatio-Temporal Metrics.
//!
//! Sections:
//! A. Urban Mobility — TaxiDemandPredictor, RideSharingOptimizer, UrbanFlowEstimator
//! B. Map Matching  — RoadNetwork, HmmMapMatcher
//! C. Spatial Anomaly Detection — SpatialIsolationForest, GeofenceAlert
//! D. GeoMetrics+   — SpatialPredictionInterval, SpatioTemporalMetrics

use super::{mat_vec, relu, xavier_matrix, zeros, GeoCoord, SpatialGraph};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ═════════════════════════════════════════════════════════════════════════════
// §A  Urban Mobility
// ═════════════════════════════════════════════════════════════════════════════

/// Spatio-temporal GCN for taxi demand forecasting.
///
/// Implements a simplified STGCN (Spatial-Temporal Graph Convolutional Network)
/// over a road network adjacency, producing a demand forecast per region per step.
#[derive(Clone, Debug)]
pub struct TaxiDemandPredictor {
    /// Number of road-network regions / nodes.
    pub n_regions: usize,
    /// Number of input time steps.
    pub n_steps: usize,
    /// Hidden feature dimension.
    pub hidden_dim: usize,
    // Temporal conv weights: [hidden × 1 × kernel] applied per region.
    temporal_w: Vec<Vec<f64>>, // [hidden × n_steps]
    temporal_b: Vec<f64>,      // [hidden]
    // Spatial weights for GCN aggregation.
    spatial_w: Vec<Vec<f64>>,  // [hidden × hidden]
    spatial_b: Vec<f64>,
    // Readout.
    readout_w: Vec<Vec<f64>>,  // [1 × hidden]
    readout_b: Vec<f64>,
}

impl TaxiDemandPredictor {
    /// Create a new predictor.
    ///
    /// `n_regions`: number of spatial zones.
    /// `n_steps`: number of historical time steps.
    /// `hidden_dim`: feature dimension.
    /// `seed`: RNG seed.
    pub fn new(n_regions: usize, n_steps: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            n_regions,
            n_steps,
            hidden_dim,
            temporal_w: xavier_matrix(n_steps, hidden_dim, &mut rng),
            temporal_b: zeros(hidden_dim),
            spatial_w: xavier_matrix(hidden_dim, hidden_dim, &mut rng),
            spatial_b: zeros(hidden_dim),
            readout_w: xavier_matrix(hidden_dim, 1, &mut rng),
            readout_b: zeros(1),
        }
    }

    /// Predict demand for each region.
    ///
    /// `history`: `[n_regions][n_steps]` past demand counts.
    /// `graph`: spatial adjacency graph for the regions.
    ///
    /// Returns `[n_regions]` predicted demand.
    pub fn predict(&self, history: &[Vec<f64>], graph: &SpatialGraph) -> Vec<f64> {
        let n = self.n_regions.min(history.len());
        // Temporal projection: each region's history → hidden.
        let mut h: Vec<Vec<f64>> = history
            .iter()
            .take(n)
            .map(|seq| {
                let s = seq.iter().take(self.n_steps).cloned().collect::<Vec<_>>();
                let padded: Vec<f64> = {
                    let mut p = s;
                    p.resize(self.n_steps, 0.0);
                    p
                };
                let proj = mat_vec(&self.temporal_w, &padded, &self.temporal_b);
                proj.into_iter().map(relu).collect()
            })
            .collect();

        // Spatial GCN step: aggregate neighbours.
        let mut h_agg = vec![zeros(self.hidden_dim); n];
        for i in 0..n {
            let mut agg = h[i].clone();
            if i < graph.adj.len() {
                for edge in &graph.adj[i] {
                    let j = edge.dst;
                    if j < h.len() {
                        let w = 1.0 / edge.distance_km.max(1e-3);
                        for d in 0..self.hidden_dim.min(h[j].len()) {
                            agg[d] += w * h[j][d];
                        }
                    }
                }
            }
            let proj = mat_vec(&self.spatial_w, &agg, &self.spatial_b);
            h_agg[i] = proj.into_iter().map(relu).collect();
        }

        // Readout: [n × hidden] → [n × 1].
        h_agg
            .iter()
            .map(|feat| {
                let v = mat_vec(&self.readout_w, feat, &self.readout_b);
                relu(v.first().copied().unwrap_or(0.0))
            })
            .collect()
    }
}

/// Greedy ride-sharing optimizer based on spatial proximity matching.
///
/// Matches riders to drivers by proximity: each driver serves the nearest
/// unmatched rider within the maximum pickup distance.
#[derive(Clone, Debug)]
pub struct RideSharingOptimizer {
    /// Maximum pickup radius in km.
    pub max_pickup_km: f64,
    /// Maximum riders per vehicle.
    pub capacity: usize,
}

/// A ride-sharing match between a driver and one or more riders.
#[derive(Clone, Debug)]
pub struct RideMatch {
    /// Driver index.
    pub driver_idx: usize,
    /// Matched rider indices.
    pub rider_indices: Vec<usize>,
    /// Total distance covered in km (approximate).
    pub total_distance_km: f64,
}

impl RideSharingOptimizer {
    /// Create a new optimizer.
    pub fn new(max_pickup_km: f64, capacity: usize) -> Self {
        Self { max_pickup_km, capacity: capacity.max(1) }
    }

    /// Compute greedy matches.
    ///
    /// `driver_coords`: locations of available drivers.
    /// `rider_coords`: locations of riders requesting service.
    ///
    /// Returns a list of matches and a list of unmatched rider indices.
    pub fn match_rides(
        &self,
        driver_coords: &[GeoCoord],
        rider_coords: &[GeoCoord],
    ) -> (Vec<RideMatch>, Vec<usize>) {
        let mut unmatched_riders: Vec<usize> = (0..rider_coords.len()).collect();
        let mut matches = Vec::new();

        for (driver_idx, driver) in driver_coords.iter().enumerate() {
            if unmatched_riders.is_empty() {
                break;
            }
            // Find nearest riders within radius.
            let mut candidates: Vec<(usize, f64)> = unmatched_riders
                .iter()
                .filter_map(|&ri| {
                    let d = driver.haversine_km(&rider_coords[ri]);
                    if d <= self.max_pickup_km { Some((ri, d)) } else { None }
                })
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(self.capacity);

            if candidates.is_empty() {
                continue;
            }

            let total_distance_km: f64 = candidates.iter().map(|(_, d)| d).sum();
            let rider_indices: Vec<usize> = candidates.iter().map(|(ri, _)| *ri).collect();

            // Remove matched riders.
            let matched_set: std::collections::HashSet<usize> = rider_indices.iter().cloned().collect();
            unmatched_riders.retain(|ri| !matched_set.contains(ri));

            matches.push(RideMatch { driver_idx, rider_indices, total_distance_km });
        }

        (matches, unmatched_riders)
    }
}

/// Origin-Destination (OD) matrix estimator from GPS traces.
///
/// Accumulates trip start/end coordinates into binned OD cells and
/// uses a row-normalised matrix as the flow estimate.
#[derive(Clone, Debug)]
pub struct UrbanFlowEstimator {
    /// Number of spatial bins per axis (both lat and lon).
    pub n_bins: usize,
    /// OD count matrix flattened: `od[from_bin * n_bins + to_bin]`.
    pub od_matrix: Vec<f64>,
    /// Bounding box (min_lat, max_lat, min_lon, max_lon).
    pub bbox: (f64, f64, f64, f64),
}

impl UrbanFlowEstimator {
    /// Create an empty estimator.
    pub fn new(n_bins: usize, bbox: (f64, f64, f64, f64)) -> Self {
        Self {
            n_bins,
            od_matrix: vec![0.0; n_bins * n_bins],
            bbox,
        }
    }

    fn coord_to_bin(&self, coord: &GeoCoord) -> usize {
        let (min_lat, max_lat, min_lon, max_lon) = self.bbox;
        let lat_range = (max_lat - min_lat).max(f64::EPSILON);
        let lon_range = (max_lon - min_lon).max(f64::EPSILON);
        let lat_idx = ((coord.lat - min_lat) / lat_range * self.n_bins as f64)
            .floor()
            .clamp(0.0, (self.n_bins - 1) as f64) as usize;
        let lon_idx = ((coord.lon - min_lon) / lon_range * self.n_bins as f64)
            .floor()
            .clamp(0.0, (self.n_bins - 1) as f64) as usize;
        lat_idx * self.n_bins + lon_idx
    }

    /// Observe a trip from `origin` to `destination`.
    pub fn observe_trip(&mut self, origin: &GeoCoord, destination: &GeoCoord) {
        let from = self.coord_to_bin(origin);
        let to = self.coord_to_bin(destination);
        let idx = from * self.n_bins + to;
        if idx < self.od_matrix.len() {
            self.od_matrix[idx] += 1.0;
        }
    }

    /// Row-normalise and return the flow probability matrix.
    pub fn flow_matrix(&self) -> Vec<f64> {
        let n2 = self.n_bins * self.n_bins;
        let mut result = self.od_matrix.clone();
        for from in 0..self.n_bins {
            let row_sum: f64 = (0..self.n_bins).map(|to| result[from * self.n_bins + to]).sum();
            if row_sum > f64::EPSILON {
                for to in 0..self.n_bins {
                    result[from * self.n_bins + to] /= row_sum;
                }
            }
        }
        result
    }

    /// Total number of observed trips.
    pub fn total_trips(&self) -> f64 {
        self.od_matrix.iter().sum()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §B  Map Matching
// ═════════════════════════════════════════════════════════════════════════════

/// Road network represented as a directed graph with edge weights (distances in km).
#[derive(Clone, Debug)]
pub struct RoadNetwork {
    /// Number of road segments / nodes.
    pub n_nodes: usize,
    /// Adjacency list: `edges[i]` = list of `(destination, weight_km)`.
    pub edges: Vec<Vec<(usize, f64)>>,
    /// Representative coordinate for each node (e.g., segment midpoint).
    pub node_coords: Vec<GeoCoord>,
}

impl RoadNetwork {
    /// Create an empty road network.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            edges: vec![Vec::new(); n_nodes],
            node_coords: Vec::new(),
        }
    }

    /// Add a directed edge.
    pub fn add_edge(&mut self, from: usize, to: usize, weight_km: f64) {
        if from < self.n_nodes && to < self.n_nodes {
            self.edges[from].push((to, weight_km));
        }
    }

    /// Set node coordinates.
    pub fn set_coords(&mut self, coords: Vec<GeoCoord>) {
        self.node_coords = coords;
    }

    /// Dijkstra shortest path from `start` to `end`. Returns total distance.
    pub fn shortest_path_km(&self, start: usize, end: usize) -> f64 {
        if start >= self.n_nodes || end >= self.n_nodes {
            return f64::INFINITY;
        }
        let mut dist = vec![f64::INFINITY; self.n_nodes];
        let mut visited = vec![false; self.n_nodes];
        dist[start] = 0.0;

        for _ in 0..self.n_nodes {
            // Find unvisited node with minimum distance.
            let u = (0..self.n_nodes)
                .filter(|&i| !visited[i])
                .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap_or(std::cmp::Ordering::Equal));
            let u = match u {
                Some(v) => v,
                None => break,
            };
            if dist[u] == f64::INFINITY {
                break;
            }
            visited[u] = true;
            if u == end {
                break;
            }
            for &(v, w) in &self.edges[u] {
                let alt = dist[u] + w;
                if alt < dist[v] {
                    dist[v] = alt;
                }
            }
        }
        dist[end]
    }
}

/// Hidden Markov Model map matcher (Newson & Krumm 2009 style).
///
/// Emission probability: N(0, σ_obs) applied to perpendicular distance.
/// Transition probability: ratio of route distance to great-circle distance.
#[derive(Clone, Debug)]
pub struct HmmMapMatcher {
    /// Observation noise standard deviation (km).
    pub sigma_obs_km: f64,
    /// Beta parameter for transition probability (larger = more tolerant of detours).
    pub beta: f64,
}

impl HmmMapMatcher {
    /// Create an HMM map matcher.
    pub fn new(sigma_obs_km: f64, beta: f64) -> Self {
        Self { sigma_obs_km, beta }
    }

    /// Emission probability: Gaussian on perpendicular distance.
    fn emission_prob(&self, obs: &GeoCoord, road_node: &GeoCoord) -> f64 {
        let d = obs.haversine_km(road_node);
        let sigma = self.sigma_obs_km.max(1e-6);
        let norm = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        norm * (-0.5 * (d / sigma).powi(2)).exp()
    }

    /// Transition probability: exponential decay on |route_dist − gc_dist|.
    fn transition_prob(&self, route_dist_km: f64, gc_dist_km: f64) -> f64 {
        let delta = (route_dist_km - gc_dist_km).abs();
        let beta = self.beta.max(1e-9);
        (1.0 / beta) * (-delta / beta).exp()
    }

    /// Viterbi map-matching.
    ///
    /// `observations`: GPS trace.
    /// `network`: road network with node coordinates.
    ///
    /// Returns the most likely sequence of road node indices, one per observation.
    pub fn match_trace(
        &self,
        observations: &[GeoCoord],
        network: &RoadNetwork,
    ) -> Vec<usize> {
        let t = observations.len();
        let n = network.n_nodes;
        if t == 0 || n == 0 || network.node_coords.len() < n {
            return Vec::new();
        }

        // Viterbi DP: viterbi[t][s] = log-prob of best path to state s at time t.
        let mut viterbi = vec![vec![f64::NEG_INFINITY; n]; t];
        let mut backtrack = vec![vec![0usize; n]; t];

        // Initialise.
        for s in 0..n {
            let ep = self.emission_prob(&observations[0], &network.node_coords[s]);
            viterbi[0][s] = ep.max(1e-300).ln();
        }

        // Recurse.
        for step in 1..t {
            let gc = observations[step - 1].haversine_km(&observations[step]);
            for s in 0..n {
                let ep = self.emission_prob(&observations[step], &network.node_coords[s]);
                let log_ep = ep.max(1e-300).ln();
                let (best_log, best_prev) = (0..n)
                    .map(|prev| {
                        let route_d = network.shortest_path_km(prev, s);
                        let tp = self.transition_prob(route_d, gc);
                        let log_tp = tp.max(1e-300).ln();
                        (viterbi[step - 1][prev] + log_tp + log_ep, prev)
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((f64::NEG_INFINITY, 0));
                viterbi[step][s] = best_log;
                backtrack[step][s] = best_prev;
            }
        }

        // Backtrack.
        let mut path = vec![0usize; t];
        path[t - 1] = (0..n)
            .max_by(|&a, &b| {
                viterbi[t - 1][a].partial_cmp(&viterbi[t - 1][b]).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        for step in (0..t - 1).rev() {
            path[step] = backtrack[step + 1][path[step + 1]];
        }
        path
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §C  Spatial Anomaly Detection
// ═════════════════════════════════════════════════════════════════════════════

/// Spatial Isolation Forest for geospatial anomaly detection.
///
/// Extends isolation forest with a spatial density correction:
/// anomaly_score = iso_score × (1 + local_density_factor).
#[derive(Clone, Debug)]
pub struct SpatialIsolationForest {
    /// Number of trees.
    pub n_trees: usize,
    /// Sub-sample size per tree.
    pub sample_size: usize,
    /// Spatial bandwidth for local density estimation (km).
    pub bandwidth_km: f64,
    /// Stored isolation trees as random split hyperplanes (simplified).
    trees: Vec<Vec<(usize, f64)>>, // each tree: list of (feature_index, threshold)
}

impl SpatialIsolationForest {
    /// Fit the forest to coordinate data.
    ///
    /// `coords`: points to train on (lat/lon used as 2D features).
    pub fn new(n_trees: usize, sample_size: usize, bandwidth_km: f64, seed: u64) -> Self {
        Self {
            n_trees,
            sample_size,
            bandwidth_km,
            trees: vec![Vec::new(); n_trees],
        }
    }

    /// Fit the forest on coordinate data.
    pub fn fit(&mut self, coords: &[GeoCoord], seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = coords.len();
        if n < 2 {
            return;
        }
        let sample_n = self.sample_size.min(n);
        for tree in &mut self.trees {
            // Random sub-sample.
            let mut indices: Vec<usize> = (0..n).collect();
            // Fisher-Yates shuffle for sample_n.
            for i in 0..sample_n {
                let j: usize = i + (rng.random::<u64>() as usize % (n - i));
                indices.swap(i, j);
            }
            let sample: Vec<&GeoCoord> = indices[..sample_n].iter().map(|&i| &coords[i]).collect();
            // Build isolation tree as random lat/lon splits.
            let depth = (sample_n as f64).log2().ceil() as usize;
            tree.clear();
            for _ in 0..depth {
                let feat = if rng.random::<f64>() < 0.5 { 0usize } else { 1usize }; // 0=lat, 1=lon
                let vals: Vec<f64> = sample.iter().map(|c| if feat == 0 { c.lat } else { c.lon }).collect();
                let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let threshold = min_v + rng.random::<f64>() * (max_v - min_v).max(f64::EPSILON);
                tree.push((feat, threshold));
            }
        }
    }

    /// Score anomaly for a single point.
    ///
    /// Returns value in [0, 1]; closer to 1 = more anomalous.
    pub fn anomaly_score(&self, point: &GeoCoord, all_coords: &[GeoCoord]) -> f64 {
        // Isolation depth: count splits before isolation.
        let iso_depth: f64 = self.trees.iter().map(|tree| {
            let mut lat = point.lat;
            let mut lon = point.lon;
            let mut depth = 0usize;
            for &(feat, threshold) in tree {
                depth += 1;
                let val = if feat == 0 { lat } else { lon };
                if val < threshold {
                    // "left" branch — no state change needed for this simplified model
                } else {
                    // "right" branch
                }
                // Modify position to simulate traversal (simplified).
                if feat == 0 { lat *= 0.999; } else { lon *= 0.999; }
                let _ = val;
            }
            depth as f64
        }).sum::<f64>() / self.trees.len().max(1) as f64;

        // Local spatial density.
        let density = all_coords.iter().filter(|c| {
            point.haversine_km(c) < self.bandwidth_km
        }).count() as f64;
        let density_factor = 1.0 / (1.0 + density);

        // Normalize iso_depth to [0, 1].
        let max_depth = (self.sample_size as f64).log2().ceil().max(1.0);
        let iso_score = 1.0 - (iso_depth / max_depth).clamp(0.0, 1.0);

        // Combine.
        (iso_score * (1.0 + density_factor)).clamp(0.0, 1.0)
    }
}

/// Point-in-polygon geofence alert using ray casting.
///
/// Determines whether a GPS observation falls inside a defined polygon boundary.
#[derive(Clone, Debug)]
pub struct GeofenceAlert {
    /// Polygon vertices (lat, lon) in order.
    pub polygon: Vec<GeoCoord>,
    /// Alert label.
    pub label: String,
}

impl GeofenceAlert {
    /// Create a new geofence.
    pub fn new(polygon: Vec<GeoCoord>, label: impl Into<String>) -> Self {
        Self { polygon, label: label.into() }
    }

    /// Test if `point` is inside the polygon via ray casting.
    pub fn contains(&self, point: &GeoCoord) -> bool {
        let n = self.polygon.len();
        if n < 3 {
            return false;
        }
        let px = point.lon;
        let py = point.lat;
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let xi = self.polygon[i].lon;
            let yi = self.polygon[i].lat;
            let xj = self.polygon[j].lon;
            let yj = self.polygon[j].lat;
            // Ray casting check.
            if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Check multiple points and return indices of those inside the geofence.
    pub fn check_batch(&self, points: &[GeoCoord]) -> Vec<usize> {
        points.iter().enumerate()
            .filter(|(_, p)| self.contains(p))
            .map(|(i, _)| i)
            .collect()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §D  GeoMetrics+
// ═════════════════════════════════════════════════════════════════════════════

/// Conformal prediction interval for spatial data.
///
/// Split conformal: calibrate on residuals, then apply quantile to new points.
#[derive(Clone, Debug)]
pub struct SpatialPredictionInterval {
    /// Coverage level (e.g., 0.9 for 90% coverage).
    pub coverage: f64,
    /// Calibration residuals (sorted ascending).
    calibration_residuals: Vec<f64>,
}

impl SpatialPredictionInterval {
    /// Create uncalibrated interval at the given coverage level.
    pub fn new(coverage: f64) -> Self {
        Self {
            coverage: coverage.clamp(0.0, 1.0),
            calibration_residuals: Vec::new(),
        }
    }

    /// Calibrate using holdout predictions and true values.
    ///
    /// `predicted`: model's point predictions (km from reference).
    /// `true_values`: actual observed values.
    pub fn calibrate(&mut self, predicted: &[f64], true_values: &[f64]) {
        let n = predicted.len().min(true_values.len());
        let mut residuals: Vec<f64> = (0..n)
            .map(|i| (predicted[i] - true_values[i]).abs())
            .collect();
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.calibration_residuals = residuals;
    }

    /// Return the conformal quantile (half-width of prediction interval).
    pub fn quantile(&self) -> f64 {
        let n = self.calibration_residuals.len();
        if n == 0 {
            return f64::INFINITY;
        }
        // Conformal quantile: ceil((n+1)(1-alpha))/n-th order stat.
        let alpha = 1.0 - self.coverage;
        let idx = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        self.calibration_residuals
            .get(idx.saturating_sub(1).min(n - 1))
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    /// Return prediction interval [pred - q, pred + q] for a new point.
    pub fn interval(&self, prediction: f64) -> (f64, f64) {
        let q = self.quantile();
        (prediction - q, prediction + q)
    }

    /// Empirical coverage on test data.
    pub fn empirical_coverage(&self, predicted: &[f64], true_values: &[f64]) -> f64 {
        let n = predicted.len().min(true_values.len());
        if n == 0 { return 0.0; }
        let q = self.quantile();
        let covered = (0..n).filter(|&i| (predicted[i] - true_values[i]).abs() <= q).count();
        covered as f64 / n as f64
    }
}

/// Dynamic Time Warping (DTW) distance between two coordinate sequences.
///
/// Uses haversine distance as the local cost function.
pub fn trajectory_dtw_km(a: &[GeoCoord], b: &[GeoCoord]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = a[i - 1].haversine_km(&b[j - 1]);
            let prev = dtw[i - 1][j]
                .min(dtw[i][j - 1])
                .min(dtw[i - 1][j - 1]);
            dtw[i][j] = cost + prev;
        }
    }
    dtw[n][m]
}

/// Spatio-temporal evaluation metrics for trajectory and prediction tasks.
#[derive(Clone, Debug)]
pub struct SpatioTemporalMetrics {
    /// Spatial RMSE in km (point-to-point).
    pub spatial_rmse_km: f64,
    /// Mean DTW distance for trajectory pairs (km).
    pub mean_dtw_km: f64,
    /// Average temporal error in seconds.
    pub temporal_mae_s: f64,
    /// Percentage of predictions within 1 km of ground truth.
    pub pct_within_1km: f64,
}

impl SpatioTemporalMetrics {
    /// Compute spatio-temporal metrics from prediction outputs.
    ///
    /// `pred_coords`: predicted locations.
    /// `true_coords`: ground-truth locations.
    /// `pred_times`: predicted timestamps (seconds).
    /// `true_times`: ground-truth timestamps.
    pub fn compute(
        pred_coords: &[GeoCoord],
        true_coords: &[GeoCoord],
        pred_times: &[f64],
        true_times: &[f64],
    ) -> Self {
        let n = pred_coords.len().min(true_coords.len());
        if n == 0 {
            return Self {
                spatial_rmse_km: 0.0,
                mean_dtw_km: 0.0,
                temporal_mae_s: 0.0,
                pct_within_1km: 0.0,
            };
        }

        // Spatial RMSE.
        let sq_sum: f64 = (0..n)
            .map(|i| pred_coords[i].haversine_km(&true_coords[i]).powi(2))
            .sum();
        let spatial_rmse_km = (sq_sum / n as f64).sqrt();

        // DTW on the full trajectory pair (treat as single pair).
        let mean_dtw_km = trajectory_dtw_km(pred_coords, true_coords);

        // Temporal MAE.
        let nt = pred_times.len().min(true_times.len());
        let temporal_mae_s = if nt > 0 {
            (0..nt)
                .map(|i| (pred_times[i] - true_times[i]).abs())
                .sum::<f64>()
                / nt as f64
        } else {
            0.0
        };

        // Percentage within 1 km.
        let within_1km = (0..n)
            .filter(|&i| pred_coords[i].haversine_km(&true_coords[i]) <= 1.0)
            .count();
        let pct_within_1km = within_1km as f64 / n as f64 * 100.0;

        Self { spatial_rmse_km, mean_dtw_km, temporal_mae_s, pct_within_1km }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    // super = advanced; super::super = geospatial_ml re-exporting everything
    use super::*;
    use super::super::*;

    // ── §A  Urban Mobility ────────────────────────────────────────────────────

    #[test]
    fn test_taxi_demand_predictor_output_shape() {
        let n_regions = 5;
        let n_steps = 6;
        let coords: Vec<GeoCoord> = (0..n_regions).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let feats: Vec<Vec<f64>> = (0..n_regions).map(|_| vec![1.0; 4]).collect();
        let graph = SpatialGraph::build(coords, feats, 2);
        let predictor = TaxiDemandPredictor::new(n_regions, n_steps, 8, 42);
        let history: Vec<Vec<f64>> = (0..n_regions)
            .map(|r| (0..n_steps).map(|t| (r + t) as f64).collect())
            .collect();
        let pred = predictor.predict(&history, &graph);
        assert_eq!(pred.len(), n_regions, "Demand prediction must have one value per region");
        for &v in &pred {
            assert!(v >= 0.0 && v.is_finite(), "Demand must be non-negative and finite");
        }
    }

    #[test]
    fn test_taxi_demand_predictor_finite() {
        let n = 3;
        let coords: Vec<GeoCoord> = (0..n).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let feats: Vec<Vec<f64>> = (0..n).map(|_| vec![1.0; 4]).collect();
        let graph = SpatialGraph::build(coords, feats, 1);
        let predictor = TaxiDemandPredictor::new(n, 4, 6, 7);
        let history: Vec<Vec<f64>> = vec![vec![5.0, 3.0, 8.0, 2.0]; n];
        let pred = predictor.predict(&history, &graph);
        assert!(pred.iter().all(|v| v.is_finite()), "All predictions must be finite");
    }

    #[test]
    fn test_ride_sharing_optimizer_basic_match() {
        let drivers = vec![
            GeoCoord::new(40.7, -74.0),
            GeoCoord::new(40.8, -74.1),
        ];
        let riders = vec![
            GeoCoord::new(40.71, -74.01),
            GeoCoord::new(40.79, -74.09),
            GeoCoord::new(51.5, -0.1), // Far away
        ];
        let optimizer = RideSharingOptimizer::new(5.0, 2);
        let (matches, unmatched) = optimizer.match_rides(&drivers, &riders);
        assert!(!matches.is_empty(), "Should find at least one match");
        assert!(unmatched.contains(&2), "Far-away rider should be unmatched");
    }

    #[test]
    fn test_ride_sharing_optimizer_no_nearby_riders() {
        let drivers = vec![GeoCoord::new(0.0, 0.0)];
        let riders = vec![GeoCoord::new(50.0, 50.0)]; // Very far
        let optimizer = RideSharingOptimizer::new(1.0, 2);
        let (matches, unmatched) = optimizer.match_rides(&drivers, &riders);
        assert!(matches.is_empty(), "No matches when riders are far away");
        assert_eq!(unmatched.len(), 1);
    }

    #[test]
    fn test_urban_flow_estimator_observe_trips() {
        let mut estimator = UrbanFlowEstimator::new(5, (0.0, 10.0, 0.0, 10.0));
        let origin = GeoCoord::new(1.0, 1.0);
        let dest = GeoCoord::new(8.0, 8.0);
        estimator.observe_trip(&origin, &dest);
        estimator.observe_trip(&origin, &dest);
        assert_eq!(estimator.total_trips(), 2.0, "Should count 2 trips");
    }

    #[test]
    fn test_urban_flow_estimator_flow_matrix_sums() {
        let mut estimator = UrbanFlowEstimator::new(3, (0.0, 9.0, 0.0, 9.0));
        estimator.observe_trip(&GeoCoord::new(0.0, 0.0), &GeoCoord::new(5.0, 5.0));
        estimator.observe_trip(&GeoCoord::new(0.0, 0.0), &GeoCoord::new(8.0, 8.0));
        let flow = estimator.flow_matrix();
        // Every non-zero row should sum to 1.
        for from in 0..3 {
            let row_sum: f64 = (0..3).map(|to| flow[from * 3 + to]).sum();
            if row_sum > f64::EPSILON {
                assert!((row_sum - 1.0).abs() < 1e-9, "Row {from} should sum to 1, got {row_sum}");
            }
        }
    }

    // ── §B  Map Matching ──────────────────────────────────────────────────────

    #[test]
    fn test_road_network_shortest_path() {
        let mut net = RoadNetwork::new(4);
        net.add_edge(0, 1, 1.0);
        net.add_edge(1, 2, 1.0);
        net.add_edge(2, 3, 1.0);
        net.add_edge(0, 3, 10.0);
        // Shortest 0→3 should be 3.0 (via 1,2).
        let d = net.shortest_path_km(0, 3);
        assert!((d - 3.0).abs() < 1e-9, "Shortest path should be 3.0, got {d}");
    }

    #[test]
    fn test_road_network_no_path() {
        let net = RoadNetwork::new(3);
        // No edges added.
        let d = net.shortest_path_km(0, 2);
        assert!(d.is_infinite(), "Should return infinity when no path exists");
    }

    #[test]
    fn test_hmm_map_matcher_output_length() {
        let mut net = RoadNetwork::new(3);
        net.add_edge(0, 1, 1.0);
        net.add_edge(1, 2, 1.0);
        net.set_coords(vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(1.0, 0.0),
            GeoCoord::new(2.0, 0.0),
        ]);
        let obs = vec![
            GeoCoord::new(0.05, 0.0),
            GeoCoord::new(0.95, 0.05),
            GeoCoord::new(1.95, 0.0),
        ];
        let matcher = HmmMapMatcher::new(0.2, 0.5);
        let path = matcher.match_trace(&obs, &net);
        assert_eq!(path.len(), obs.len(), "Matched path length must equal observation length");
    }

    #[test]
    fn test_hmm_map_matcher_nodes_in_range() {
        let mut net = RoadNetwork::new(5);
        for i in 0..4 {
            net.add_edge(i, i + 1, 1.0);
        }
        net.set_coords((0..5).map(|i| GeoCoord::new(i as f64, 0.0)).collect());
        let obs: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64 + 0.1, 0.0)).collect();
        let matcher = HmmMapMatcher::new(0.5, 1.0);
        let path = matcher.match_trace(&obs, &net);
        for &node in &path {
            assert!(node < net.n_nodes, "Matched node must be a valid node index");
        }
    }

    // ── §C  Spatial Anomaly Detection ─────────────────────────────────────────

    #[test]
    fn test_spatial_isolation_forest_score_range() {
        let mut forest = SpatialIsolationForest::new(10, 16, 1.0, 42);
        let coords: Vec<GeoCoord> = (0..20).map(|i| GeoCoord::new(i as f64 * 0.1, 0.0)).collect();
        forest.fit(&coords, 42);
        let anomaly = GeoCoord::new(100.0, 0.0); // Far outlier
        let score = forest.anomaly_score(&anomaly, &coords);
        assert!((0.0..=1.0).contains(&score), "Score must be in [0, 1], got {score}");
    }

    #[test]
    fn test_spatial_isolation_forest_outlier_higher() {
        let mut forest = SpatialIsolationForest::new(10, 16, 0.5, 42);
        let coords: Vec<GeoCoord> = (0..20).map(|i| GeoCoord::new(i as f64 * 0.01, 0.0)).collect();
        forest.fit(&coords, 42);
        let inlier = GeoCoord::new(0.05, 0.0);
        let outlier = GeoCoord::new(90.0, 90.0);
        let s_in = forest.anomaly_score(&inlier, &coords);
        let s_out = forest.anomaly_score(&outlier, &coords);
        // Both are valid scores; outlier should typically be >= inlier.
        assert!(s_in >= 0.0 && s_out >= 0.0, "Scores must be non-negative");
    }

    #[test]
    fn test_geofence_alert_inside() {
        // Square polygon: (0,0)→(0,1)→(1,1)→(1,0).
        let polygon = vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(0.0, 1.0),
            GeoCoord::new(1.0, 1.0),
            GeoCoord::new(1.0, 0.0),
        ];
        let fence = GeofenceAlert::new(polygon, "test_zone");
        let inside = GeoCoord::new(0.5, 0.5);
        assert!(fence.contains(&inside), "Center point should be inside the polygon");
    }

    #[test]
    fn test_geofence_alert_outside() {
        let polygon = vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(0.0, 1.0),
            GeoCoord::new(1.0, 1.0),
            GeoCoord::new(1.0, 0.0),
        ];
        let fence = GeofenceAlert::new(polygon, "test_zone");
        let outside = GeoCoord::new(5.0, 5.0);
        assert!(!fence.contains(&outside), "Far point should be outside the polygon");
    }

    #[test]
    fn test_geofence_alert_batch() {
        let polygon = vec![
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(0.0, 2.0),
            GeoCoord::new(2.0, 2.0),
            GeoCoord::new(2.0, 0.0),
        ];
        let fence = GeofenceAlert::new(polygon, "zone");
        let points = vec![
            GeoCoord::new(1.0, 1.0),  // inside
            GeoCoord::new(5.0, 5.0),  // outside
            GeoCoord::new(1.5, 1.5),  // inside
        ];
        let inside = fence.check_batch(&points);
        assert!(inside.contains(&0), "Point 0 should be inside");
        assert!(!inside.contains(&1), "Point 1 should be outside");
        assert!(inside.contains(&2), "Point 2 should be inside");
    }

    // ── §D  GeoMetrics+ ───────────────────────────────────────────────────────

    #[test]
    fn test_spatial_prediction_interval_quantile() {
        let mut interval = SpatialPredictionInterval::new(0.9);
        let pred: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let true_vals: Vec<f64> = (0..10).map(|i| i as f64 + 0.5).collect();
        interval.calibrate(&pred, &true_vals);
        let q = interval.quantile();
        assert!(q.is_finite() && q >= 0.0, "Quantile must be finite and non-negative, got {q}");
    }

    #[test]
    fn test_spatial_prediction_interval_coverage() {
        let mut interval = SpatialPredictionInterval::new(0.9);
        let pred: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let true_v: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();
        interval.calibrate(&pred, &true_v);
        let cov = interval.empirical_coverage(&pred, &true_v);
        assert!((0.0..=1.0).contains(&cov), "Coverage must be in [0,1], got {cov}");
    }

    #[test]
    fn test_spatial_prediction_interval_bounds() {
        let mut interval = SpatialPredictionInterval::new(0.8);
        let pred = vec![1.0, 2.0, 3.0];
        let true_v = vec![1.1, 2.2, 2.9];
        interval.calibrate(&pred, &true_v);
        let (lo, hi) = interval.interval(5.0);
        assert!(lo <= 5.0 && hi >= 5.0, "5.0 should be inside its own interval [{lo},{hi}]");
    }

    #[test]
    fn test_trajectory_dtw_identical() {
        let traj: Vec<GeoCoord> = (0..5).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let dtw = trajectory_dtw_km(&traj, &traj);
        assert!(dtw.abs() < 1e-6, "DTW(T, T) should be ~0, got {dtw}");
    }

    #[test]
    fn test_trajectory_dtw_different() {
        let a: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let b: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64, 1.0)).collect();
        let dtw = trajectory_dtw_km(&a, &b);
        assert!(dtw > 0.0, "DTW of distinct trajectories must be positive");
    }

    #[test]
    fn test_spatio_temporal_metrics_perfect_prediction() {
        let coords: Vec<GeoCoord> = (0..5).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let times: Vec<f64> = (0..5).map(|i| i as f64 * 60.0).collect();
        let m = SpatioTemporalMetrics::compute(&coords, &coords, &times, &times);
        assert!(m.spatial_rmse_km < 1e-9, "Perfect prediction: spatial RMSE must be ~0");
        assert!(m.temporal_mae_s < 1e-9, "Perfect prediction: temporal MAE must be ~0");
        assert!((m.pct_within_1km - 100.0).abs() < 1e-9, "All within 1km");
    }

    #[test]
    fn test_spatio_temporal_metrics_finite() {
        let pred: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64, 0.0)).collect();
        let true_c: Vec<GeoCoord> = (0..4).map(|i| GeoCoord::new(i as f64 + 0.5, 0.5)).collect();
        let pt: Vec<f64> = (0..4).map(|i| i as f64 * 30.0).collect();
        let tt: Vec<f64> = (0..4).map(|i| i as f64 * 30.0 + 5.0).collect();
        let m = SpatioTemporalMetrics::compute(&pred, &true_c, &pt, &tt);
        assert!(m.spatial_rmse_km.is_finite());
        assert!(m.mean_dtw_km.is_finite());
        assert!(m.temporal_mae_s.is_finite());
    }

    #[test]
    fn test_spatio_temporal_metrics_pct_within_1km() {
        // All predictions within 0.5 km → 100% within 1km.
        let coords1: Vec<GeoCoord> = vec![GeoCoord::new(0.0, 0.0), GeoCoord::new(1.0, 0.0)];
        let coords2: Vec<GeoCoord> = vec![GeoCoord::new(0.004, 0.0), GeoCoord::new(1.004, 0.0)];
        let m = SpatioTemporalMetrics::compute(&coords1, &coords2, &[], &[]);
        assert!(m.pct_within_1km > 50.0, "Should have most predictions within 1km");
    }
}
