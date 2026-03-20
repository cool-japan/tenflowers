//! # Multi-Object Tracking (MOT)
//!
//! This module implements state-of-the-art multi-object tracking algorithms
//! including SORT, DeepSORT, and ByteTrack with Kalman filter state estimation,
//! Hungarian assignment, and appearance-based re-identification.
//!
//! ## Key Algorithms
//!
//! - **SORT** (Bewley et al. 2016): Simple Online and Realtime Tracking using
//!   Kalman filters and Hungarian assignment with IoU cost.
//! - **DeepSORT** (Wojke et al. 2017): SORT augmented with deep appearance
//!   features for robust re-identification across occlusions.
//! - **ByteTrack** (Zhang et al. 2022): Two-stage association using both
//!   high- and low-confidence detections for complete tracking.
//!
//! ## References
//!
//! - Bewley et al. (2016) "Simple Online and Realtime Tracking"
//! - Wojke et al. (2017) "Simple Online and Realtime Tracking with a Deep Association Metric"
//! - Zhang et al. (2022) "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
//! - Dendorfer et al. (2021) "MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking"

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error Type
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for Multi-Object Tracking operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MotError {
    /// Detection data is invalid (e.g., negative dimensions, NaN values).
    InvalidDetection(String),
    /// A numerical computation failed (e.g., singular matrix, NaN result).
    NumericalError(String),
    /// The assignment algorithm encountered an error.
    AssignmentError(String),
}

impl fmt::Display for MotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MotError::InvalidDetection(msg) => write!(f, "InvalidDetection: {}", msg),
            MotError::NumericalError(msg) => write!(f, "NumericalError: {}", msg),
            MotError::AssignmentError(msg) => write!(f, "AssignmentError: {}", msg),
        }
    }
}

impl std::error::Error for MotError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal RNG (no external rand crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG returning uniform f64 in [0, 1).
#[inline]
fn mot_rand01(seed: &mut u64) -> f64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 MotBoundingBox
// ─────────────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in (x, y, width, height) format.
///
/// All coordinates use the top-left corner convention with width/height
/// extending rightward and downward.
#[derive(Debug, Clone, PartialEq)]
pub struct MotBoundingBox {
    /// Left edge x-coordinate.
    pub x: f64,
    /// Top edge y-coordinate.
    pub y: f64,
    /// Width of the box.
    pub w: f64,
    /// Height of the box.
    pub h: f64,
}

impl MotBoundingBox {
    /// Create a new bounding box from (x, y, width, height).
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }

    /// Compute the Intersection over Union (IoU) between two bounding boxes.
    ///
    /// Returns a value in [0, 1] where 1 means perfect overlap and 0 means
    /// no overlap.
    pub fn iou(&self, other: &MotBoundingBox) -> f64 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area_self = self.w * self.h;
        let area_other = other.w * other.h;
        let union = area_self + area_other - intersection;

        if union <= 0.0 {
            return 0.0;
        }

        (intersection / union).clamp(0.0, 1.0)
    }

    /// Return the center point (cx, cy) of the bounding box.
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// Convert to [x1, y1, x2, y2] corner format.
    pub fn to_xyxy(&self) -> [f64; 4] {
        [self.x, self.y, self.x + self.w, self.y + self.h]
    }

    /// Create a bounding box from [x1, y1, x2, y2] corner format.
    pub fn from_xyxy(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            x: x1,
            y: y1,
            w: (x2 - x1).abs(),
            h: (y2 - y1).abs(),
        }
    }

    /// Convert to SORT state representation [cx, cy, s, r].
    ///
    /// - `cx`, `cy`: Center coordinates
    /// - `s`: Area (scale)
    /// - `r`: Aspect ratio (w / h)
    pub fn to_state(&self) -> [f64; 4] {
        let (cx, cy) = self.center();
        let s = self.w * self.h;
        let r = if self.h > 0.0 { self.w / self.h } else { 1.0 };
        [cx, cy, s, r]
    }

    /// Create a bounding box from SORT state representation [cx, cy, s, r].
    ///
    /// - `cx`, `cy`: Center coordinates
    /// - `s`: Area (scale)
    /// - `r`: Aspect ratio (w / h)
    pub fn from_state(cx: f64, cy: f64, s: f64, r: f64) -> Self {
        let s_safe = s.max(0.0);
        let r_safe = r.max(1e-6);
        // s = w * h, r = w / h → h = sqrt(s / r), w = sqrt(s * r)
        let w = (s_safe * r_safe).sqrt();
        let h = (s_safe / r_safe).sqrt();
        Self {
            x: cx - w / 2.0,
            y: cy - h / 2.0,
            w,
            h,
        }
    }

    /// Return the area of the bounding box.
    pub fn area(&self) -> f64 {
        self.w * self.h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 MotKalmanFilter
// ─────────────────────────────────────────────────────────────────────────────

/// SORT-style constant-velocity Kalman filter for bounding box tracking.
///
/// State vector: `[cx, cy, s, r, v_cx, v_cy, v_s]`
/// - `cx`, `cy`: Center position
/// - `s`: Scale (area)
/// - `r`: Aspect ratio (treated as constant, no velocity)
/// - `v_cx`, `v_cy`, `v_s`: Velocities for cx, cy, s
///
/// Measurement vector: `[cx, cy, s, r]` (4-dimensional)
#[derive(Debug, Clone)]
pub struct MotKalmanFilter {
    /// State mean estimate (7-dimensional).
    pub state: [f64; 7],
    /// State covariance (7×7).
    pub covariance: [[f64; 7]; 7],
    /// State transition matrix (7×7).
    pub f_mat: [[f64; 7]; 7],
    /// Observation/measurement matrix (4×7), maps state → measurement.
    pub h_mat: [[f64; 4]; 7],
    /// Process noise covariance (7×7).
    pub q_mat: [[f64; 7]; 7],
    /// Measurement noise covariance (4×4).
    pub r_mat: [[f64; 4]; 4],
}

impl MotKalmanFilter {
    /// Create a new Kalman filter initialized from an observed bounding box.
    pub fn new(initial_bbox: &MotBoundingBox) -> Self {
        let state_arr = initial_bbox.to_state();
        // State: [cx, cy, s, r, v_cx, v_cy, v_s] with zero velocities
        let state: [f64; 7] = [
            state_arr[0],
            state_arr[1],
            state_arr[2],
            state_arr[3],
            0.0,
            0.0,
            0.0,
        ];

        // State transition: F = I + dt*(velocity block), dt=1
        // x_new = x + v*dt, v_new = v
        let mut f_mat = [[0.0f64; 7]; 7];
        for i in 0..7 {
            f_mat[i][i] = 1.0;
        }
        // cx' = cx + v_cx; position[0] += velocity[4]
        f_mat[0][4] = 1.0;
        // cy' = cy + v_cy; position[1] += velocity[5]
        f_mat[1][5] = 1.0;
        // s' = s + v_s;    position[2] += velocity[6]
        f_mat[2][6] = 1.0;

        // Measurement matrix H: maps [cx,cy,s,r,...] → [cx,cy,s,r]
        // H[row][col]: row=measurement_dim, col=state_dim
        let mut h_mat = [[0.0f64; 4]; 7];
        h_mat[0][0] = 1.0; // cx
        h_mat[1][1] = 1.0; // cy
        h_mat[2][2] = 1.0; // s
        h_mat[3][3] = 1.0; // r

        // Process noise Q (diagonal)
        let mut q_mat = [[0.0f64; 7]; 7];
        q_mat[0][0] = 1.0; // cx process noise
        q_mat[1][1] = 1.0; // cy process noise
        q_mat[2][2] = 10.0; // s process noise
        q_mat[3][3] = 1e-2; // r process noise
        q_mat[4][4] = 0.01; // v_cx process noise
        q_mat[5][5] = 0.01; // v_cy process noise
        q_mat[6][6] = 0.1; // v_s process noise

        // Measurement noise R (diagonal)
        let mut r_mat = [[0.0f64; 4]; 4];
        r_mat[0][0] = 1.0; // cx measurement noise
        r_mat[1][1] = 1.0; // cy measurement noise
        r_mat[2][2] = 10.0; // s measurement noise
        r_mat[3][3] = 1e-2; // r measurement noise

        // Initial covariance P (large uncertainty in velocities)
        let mut covariance = [[0.0f64; 7]; 7];
        covariance[0][0] = 2.0; // cx
        covariance[1][1] = 2.0; // cy
        covariance[2][2] = 10.0; // s
        covariance[3][3] = 2.0; // r
        covariance[4][4] = 25.0; // v_cx (high uncertainty)
        covariance[5][5] = 25.0; // v_cy (high uncertainty)
        covariance[6][6] = 100.0; // v_s (high uncertainty)

        Self {
            state,
            covariance,
            f_mat,
            h_mat,
            q_mat,
            r_mat,
        }
    }

    /// Predict the next state: x = F*x, P = F*P*F^T + Q.
    ///
    /// Returns the predicted measurement [cx, cy, s, r].
    pub fn predict(&mut self) -> [f64; 4] {
        // x = F * x
        let mut new_state = [0.0f64; 7];
        for i in 0..7 {
            for j in 0..7 {
                new_state[i] += self.f_mat[i][j] * self.state[j];
            }
        }
        self.state = new_state;

        // P = F * P * F^T + Q
        // Step 1: tmp = F * P
        let mut tmp = [[0.0f64; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                for k in 0..7 {
                    tmp[i][j] += self.f_mat[i][k] * self.covariance[k][j];
                }
            }
        }
        // Step 2: FPFt = tmp * F^T
        let mut fpft = [[0.0f64; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                for k in 0..7 {
                    fpft[i][j] += tmp[i][k] * self.f_mat[j][k]; // F^T[k][j] = F[j][k]
                }
            }
        }
        // P = FPFt + Q
        for i in 0..7 {
            for j in 0..7 {
                self.covariance[i][j] = fpft[i][j] + self.q_mat[i][j];
            }
        }

        // Return [cx, cy, s, r]
        [self.state[0], self.state[1], self.state[2], self.state[3]]
    }

    /// Perform a Kalman update with a new measurement bounding box.
    ///
    /// K = P H^T (H P H^T + R)^{-1}
    /// x += K (z - H x)
    /// P = (I - K H) P
    pub fn update(&mut self, measurement: &MotBoundingBox) {
        let z = measurement.to_state();

        // Compute innovation: y = z - H*x
        let mut y = [0.0f64; 4];
        for i in 0..4 {
            // H[state_dim][meas_dim] — h_mat[col][row] due to our layout
            let hx_i: f64 = (0..7).map(|j| self.h_mat[j][i] * self.state[j]).sum();
            y[i] = z[i] - hx_i;
        }

        // S = H P H^T + R (4×4 innovation covariance)
        // Step 1: PH^T = P * H^T (7×4)
        let mut ph_t = [[0.0f64; 4]; 7];
        for i in 0..7 {
            for j in 0..4 {
                // H^T[state_dim][meas_dim]: (H^T)_{i,j} = H_{j,i} = h_mat[i][j]
                ph_t[i][j] = (0..7)
                    .map(|k| self.covariance[i][k] * self.h_mat[k][j])
                    .sum();
            }
        }

        // Step 2: S = H * PH^T + R (4×4)
        let mut s_mat = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                // H[meas_dim][state_dim]: H_{i,k} = h_mat[k][i]
                let h_pht_ij: f64 = (0..7).map(|k| self.h_mat[k][i] * ph_t[k][j]).sum();
                s_mat[i][j] = h_pht_ij + self.r_mat[i][j];
            }
        }

        // Step 3: S^{-1} via cofactor expansion (4×4 matrix)
        let s_inv = match mat4_inverse(&s_mat) {
            Some(inv) => inv,
            None => return, // singular — skip update
        };

        // Step 4: K = PH^T * S^{-1} (7×4)
        let mut k_mat = [[0.0f64; 4]; 7];
        for i in 0..7 {
            for j in 0..4 {
                k_mat[i][j] = (0..4).map(|k| ph_t[i][k] * s_inv[k][j]).sum();
            }
        }

        // Step 5: x = x + K * y
        for i in 0..7 {
            self.state[i] += (0..4).map(|j| k_mat[i][j] * y[j]).sum::<f64>();
        }

        // Step 6: P = (I - K*H) * P
        // KH (7×7)
        let mut kh = [[0.0f64; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                // K is 7×4, H is h_mat which is 7×4 (col=meas, row=state_dim)
                // (KH)_{i,j} = sum_k K_{i,k} * H_{k,j}
                kh[i][j] = (0..4).map(|k| k_mat[i][k] * self.h_mat[j][k]).sum();
            }
        }
        // I - KH
        let mut i_minus_kh = [[0.0f64; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                i_minus_kh[i][j] = if i == j { 1.0 } else { 0.0 } - kh[i][j];
            }
        }
        // P = (I - KH) * P
        let old_p = self.covariance;
        for i in 0..7 {
            for j in 0..7 {
                self.covariance[i][j] = (0..7).map(|k| i_minus_kh[i][k] * old_p[k][j]).sum();
            }
        }
    }

    /// Convert the current state to a MotBoundingBox.
    pub fn get_bbox(&self) -> MotBoundingBox {
        let cx = self.state[0];
        let cy = self.state[1];
        let s = self.state[2].max(0.0);
        let r = self.state[3].max(1e-6);
        MotBoundingBox::from_state(cx, cy, s, r)
    }
}

/// Compute the inverse of a 4×4 matrix via cofactor/adjugate method.
///
/// Returns `None` if the matrix is singular (|det| < 1e-10).
fn mat4_inverse(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    // Compute 3×3 minors for all 16 cofactors
    let cofactor = |r: usize, c: usize| -> f64 {
        let rows: Vec<usize> = (0..4).filter(|&i| i != r).collect();
        let cols: Vec<usize> = (0..4).filter(|&j| j != c).collect();
        let det3 = m[rows[0]][cols[0]]
            * (m[rows[1]][cols[1]] * m[rows[2]][cols[2]]
                - m[rows[1]][cols[2]] * m[rows[2]][cols[1]])
            - m[rows[0]][cols[1]]
                * (m[rows[1]][cols[0]] * m[rows[2]][cols[2]]
                    - m[rows[1]][cols[2]] * m[rows[2]][cols[0]])
            + m[rows[0]][cols[2]]
                * (m[rows[1]][cols[0]] * m[rows[2]][cols[1]]
                    - m[rows[1]][cols[1]] * m[rows[2]][cols[0]]);
        let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
        sign * det3
    };

    // Compute determinant via first row expansion
    let det = (0..4).map(|j| m[0][j] * cofactor(0, j)).sum::<f64>();

    if det.abs() < 1e-10 {
        return None;
    }

    let inv_det = 1.0 / det;
    let mut inv = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            // Adjugate is transpose of cofactor matrix
            inv[i][j] = cofactor(j, i) * inv_det;
        }
    }
    Some(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 MotTrack
// ─────────────────────────────────────────────────────────────────────────────

/// State of an individual object track.
#[derive(Debug, Clone, PartialEq)]
pub enum MotTrackState {
    /// Track exists for fewer than `min_hits` frames; not yet output.
    Tentative,
    /// Track has been confirmed and is actively output.
    Confirmed,
    /// Track is marked for deletion (too long since last update).
    Deleted,
}

/// Single-object track combining a Kalman filter with lifecycle management.
#[derive(Debug, Clone)]
pub struct MotTrack {
    /// Unique track identifier.
    pub track_id: usize,
    /// Underlying Kalman filter.
    pub kf: MotKalmanFilter,
    /// Number of frames in which this track received a matched detection.
    pub hits: usize,
    /// Total number of frames since this track was created.
    pub age: usize,
    /// Frames since the last successful association.
    pub time_since_update: usize,
    /// Lifecycle state of the track.
    pub state: MotTrackState,
    /// History of bounding boxes (one per associated detection).
    pub history: Vec<MotBoundingBox>,
}

impl MotTrack {
    /// Create a new track initialized with the given bounding box.
    pub fn new(track_id: usize, bbox: &MotBoundingBox) -> Self {
        Self {
            track_id,
            kf: MotKalmanFilter::new(bbox),
            hits: 1,
            age: 1,
            time_since_update: 0,
            state: MotTrackState::Tentative,
            history: vec![bbox.clone()],
        }
    }

    /// Advance the Kalman filter prediction by one frame.
    pub fn predict(&mut self) {
        self.kf.predict();
        self.age += 1;
        self.time_since_update += 1;
    }

    /// Update the track with a new associated detection.
    pub fn update(&mut self, bbox: &MotBoundingBox) {
        self.kf.update(bbox);
        self.hits += 1;
        self.time_since_update = 0;
        self.history.push(bbox.clone());
    }

    /// Return true if the track is in the Confirmed state.
    pub fn is_confirmed(&self) -> bool {
        self.state == MotTrackState::Confirmed
    }

    /// Return true if the track is marked for deletion.
    pub fn is_deleted(&self) -> bool {
        self.state == MotTrackState::Deleted
    }

    /// Return the current predicted bounding box from the Kalman filter state.
    pub fn predicted_bbox(&self) -> MotBoundingBox {
        self.kf.get_bbox()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 MotHungarian
// ─────────────────────────────────────────────────────────────────────────────

/// Hungarian algorithm (Kuhn-Munkres) for optimal bipartite assignment.
///
/// Minimizes the total cost of assigning rows to columns using the
/// augmenting path method, which runs in O(n³) time.
pub struct MotHungarian;

impl MotHungarian {
    /// Solve the assignment problem for the given cost matrix.
    ///
    /// Returns `assignment[i] = Some(j)` if row `i` is assigned to column `j`,
    /// or `None` if row `i` is unassigned (when there are more rows than columns).
    ///
    /// Uses the augmenting path / shortest path method (Jonker-Volgenant style).
    pub fn solve(cost_matrix: &[Vec<f64>]) -> Vec<Option<usize>> {
        let n_rows = cost_matrix.len();
        if n_rows == 0 {
            return vec![];
        }
        let n_cols = cost_matrix[0].len();

        if n_cols == 0 {
            return vec![None; n_rows];
        }

        // Pad to square matrix if needed
        let n = n_rows.max(n_cols);
        let big = f64::MAX / 2.0;

        // Build padded square cost matrix
        let mut c = vec![vec![big; n]; n];
        for i in 0..n_rows {
            for j in 0..n_cols {
                c[i][j] = cost_matrix[i][j];
            }
        }

        // Hungarian algorithm with potential functions (Jonker-Volgenant approach)
        // u[i]: row potentials, v[j]: column potentials
        let mut u = vec![0.0f64; n + 1];
        let mut v = vec![0.0f64; n + 1];
        // p[j]: row assigned to column j (1-indexed, 0 = unassigned)
        let mut p = vec![0usize; n + 1];
        // way[j]: previous column in the augmenting path
        let mut way = vec![0usize; n + 1];

        for i in 1..=n {
            p[0] = i;
            let mut j0 = 0usize;
            let mut minval = vec![f64::MAX; n + 1];
            let mut used = vec![false; n + 1];

            loop {
                used[j0] = true;
                let i0 = p[j0];
                let mut delta = f64::MAX;
                let mut j1 = 0usize;

                for j in 1..=n {
                    if !used[j] {
                        let val = c[i0 - 1][j - 1] - u[i0] - v[j];
                        if val < minval[j] {
                            minval[j] = val;
                            way[j] = j0;
                        }
                        if minval[j] < delta {
                            delta = minval[j];
                            j1 = j;
                        }
                    }
                }

                for j in 0..=n {
                    if used[j] {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minval[j] -= delta;
                    }
                }

                j0 = j1;

                if p[j0] == 0 {
                    break;
                }
            }

            loop {
                let j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
                if j0 == 0 {
                    break;
                }
            }
        }

        // Extract assignment (only for original rows and columns)
        let mut assignment = vec![None; n_rows];
        for j in 1..=n {
            let row_idx = p[j];
            if row_idx > 0 && row_idx <= n_rows && j <= n_cols {
                assignment[row_idx - 1] = Some(j - 1);
            }
        }

        assignment
    }

    /// Build an IoU cost matrix for Hungarian assignment.
    ///
    /// `cost[i][j] = 1.0 - IoU(tracks[i], detections[j])`
    pub fn iou_cost_matrix(
        tracks: &[MotBoundingBox],
        detections: &[MotBoundingBox],
    ) -> Vec<Vec<f64>> {
        tracks
            .iter()
            .map(|t| detections.iter().map(|d| 1.0 - t.iou(d)).collect())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 MotSort
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the SORT tracker.
#[derive(Debug, Clone)]
pub struct MotSortConfig {
    /// Maximum number of frames to keep an unmatched track alive.
    pub max_age: usize,
    /// Minimum number of hits before a track is confirmed.
    pub min_hits: usize,
    /// Minimum IoU for a detection-track association.
    pub iou_threshold: f64,
}

impl Default for MotSortConfig {
    fn default() -> Self {
        Self {
            max_age: 3,
            min_hits: 3,
            iou_threshold: 0.3,
        }
    }
}

/// SORT: Simple Online and Realtime Tracker (Bewley et al. 2016).
///
/// Combines Kalman filtering with Hungarian assignment using IoU cost.
/// Each detection is represented as a bounding box; the tracker maintains
/// a set of tracks and updates them every frame.
pub struct MotSort {
    /// Tracker configuration.
    pub config: MotSortConfig,
    /// Current set of tracks (both active and tentative).
    pub tracks: Vec<MotTrack>,
    /// Counter for assigning unique track IDs.
    pub next_id: usize,
    /// Frame counter.
    pub frame: usize,
}

impl MotSort {
    /// Create a new SORT tracker with the given configuration.
    pub fn new(config: MotSortConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
            frame: 0,
        }
    }

    /// Update the tracker with a new set of detections.
    ///
    /// Returns a list of `(track_id, bbox)` pairs for all confirmed, active tracks.
    pub fn update(&mut self, detections: &[MotBoundingBox]) -> Vec<(usize, MotBoundingBox)> {
        self.frame += 1;

        // Step 1: Predict all tracks
        for track in &mut self.tracks {
            track.predict();
        }

        // Step 2: Collect predicted bboxes for association
        let predicted_bboxes: Vec<MotBoundingBox> =
            self.tracks.iter().map(|t| t.predicted_bbox()).collect();

        // Step 3: Associate detections to tracks
        let (matched, unmatched_tracks, unmatched_dets) =
            Self::associate(&predicted_bboxes, detections, self.config.iou_threshold);

        // Step 4: Update matched tracks
        for (ti, di) in &matched {
            self.tracks[*ti].update(&detections[*di]);
        }

        // Step 5: Mark unmatched tracks; update their state
        for ti in &unmatched_tracks {
            // time_since_update was already incremented in predict()
            if self.tracks[*ti].time_since_update > self.config.max_age {
                self.tracks[*ti].state = MotTrackState::Deleted;
            }
        }

        // Step 6: Create new tracks for unmatched detections
        for di in &unmatched_dets {
            let id = self.next_id;
            self.next_id += 1;
            self.tracks.push(MotTrack::new(id, &detections[*di]));
        }

        // Step 7: Update track states (Tentative → Confirmed)
        for track in &mut self.tracks {
            if (track.hits >= self.config.min_hits || self.frame <= self.config.min_hits)
                && track.state == MotTrackState::Tentative {
                    track.state = MotTrackState::Confirmed;
                }
        }

        // Step 8: Remove deleted tracks
        self.tracks.retain(|t| t.state != MotTrackState::Deleted);

        // Step 9: Collect outputs
        self.tracks
            .iter()
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .map(|t| (t.track_id, t.predicted_bbox()))
            .collect()
    }

    /// Associate detections to tracks using Hungarian assignment with IoU cost.
    ///
    /// Returns `(matched_pairs, unmatched_track_idxs, unmatched_det_idxs)`.
    pub fn associate(
        tracks: &[MotBoundingBox],
        detections: &[MotBoundingBox],
        threshold: f64,
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if tracks.is_empty() {
            let unmatched_dets = (0..detections.len()).collect();
            return (vec![], vec![], unmatched_dets);
        }
        if detections.is_empty() {
            let unmatched_tracks = (0..tracks.len()).collect();
            return (vec![], unmatched_tracks, vec![]);
        }

        let cost = MotHungarian::iou_cost_matrix(tracks, detections);
        let assignment = MotHungarian::solve(&cost);

        let mut matched = Vec::new();
        let mut unmatched_tracks = Vec::new();
        let mut matched_det_set = vec![false; detections.len()];

        for (ti, opt_di) in assignment.iter().enumerate() {
            if ti >= tracks.len() {
                break;
            }
            match opt_di {
                Some(di) if cost[ti][*di] <= 1.0 - threshold => {
                    matched.push((ti, *di));
                    matched_det_set[*di] = true;
                }
                _ => {
                    unmatched_tracks.push(ti);
                }
            }
        }

        let unmatched_dets = (0..detections.len())
            .filter(|&di| !matched_det_set[di])
            .collect();

        (matched, unmatched_tracks, unmatched_dets)
    }

    /// Return the number of non-deleted tracks.
    pub fn n_active_tracks(&self) -> usize {
        self.tracks.iter().filter(|t| !t.is_deleted()).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6 MotAppearanceFeature
// ─────────────────────────────────────────────────────────────────────────────

/// Appearance feature extractor for Re-ID (used in DeepSORT).
///
/// Implements a two-layer MLP: `patch → hidden → embedding`.
/// Outputs are L2-normalized to the unit hypersphere.
pub struct MotAppearanceFeature {
    /// Weight matrix for the first layer [patch_dim × hidden_dim].
    pub w1: Vec<Vec<f64>>,
    /// Bias for the first layer \[hidden_dim\].
    pub b1: Vec<f64>,
    /// Weight matrix for the second layer [hidden_dim × embed_dim].
    pub w2: Vec<Vec<f64>>,
    /// Bias for the second layer \[embed_dim\].
    pub b2: Vec<f64>,
    /// Input dimensionality.
    pub patch_dim: usize,
    /// Hidden layer dimensionality.
    pub hidden_dim: usize,
    /// Output embedding dimensionality.
    pub embed_dim: usize,
}

impl MotAppearanceFeature {
    /// Create a new appearance feature extractor with Xavier-initialized weights.
    pub fn new(patch_dim: usize, hidden_dim: usize, embed_dim: usize) -> Self {
        let mut seed: u64 = 0xDEADBEEF_CAFEBABE;

        let xavier1 = (6.0 / (patch_dim + hidden_dim) as f64).sqrt();
        let w1: Vec<Vec<f64>> = (0..patch_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| (mot_rand01(&mut seed) * 2.0 - 1.0) * xavier1)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0f64; hidden_dim];

        let xavier2 = (6.0 / (hidden_dim + embed_dim) as f64).sqrt();
        let w2: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| (mot_rand01(&mut seed) * 2.0 - 1.0) * xavier2)
                    .collect()
            })
            .collect();
        let b2 = vec![0.0f64; embed_dim];

        Self {
            w1,
            b1,
            w2,
            b2,
            patch_dim,
            hidden_dim,
            embed_dim,
        }
    }

    /// Extract an L2-normalized embedding from an image patch.
    ///
    /// Forward pass: `z = L2_norm(ReLU(W2 @ ReLU(W1 @ patch + b1)) + b2)`
    pub fn extract(&self, patch: &[f64]) -> Vec<f64> {
        // Layer 1: hidden = ReLU(W1^T * patch + b1)
        let mut hidden = vec![0.0f64; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let dot: f64 = (0..self.patch_dim.min(patch.len()))
                .map(|i| self.w1[i][h] * patch[i])
                .sum();
            hidden[h] = (dot + self.b1[h]).max(0.0); // ReLU
        }

        // Layer 2: out = ReLU(W2^T * hidden + b2)
        let mut out = vec![0.0f64; self.embed_dim];
        for e in 0..self.embed_dim {
            let dot: f64 = (0..self.hidden_dim)
                .map(|h| self.w2[h][e] * hidden[h])
                .sum();
            out[e] = (dot + self.b2[e]).max(0.0); // ReLU
        }

        // L2 normalize
        let norm: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            out.iter_mut().for_each(|x| *x /= norm);
        } else {
            // If all zeros, return uniform unit vector
            let uniform_val = 1.0 / (self.embed_dim as f64).sqrt();
            out.iter_mut().for_each(|x| *x = uniform_val);
        }

        out
    }

    /// Compute the cosine distance between two embedding vectors.
    ///
    /// `cosine_distance(a, b) = 1 - cosine_similarity(a, b)`
    /// Result is in [0, 2] for L2-normalized vectors.
    pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < 1e-12 || norm_b < 1e-12 {
            return 1.0; // treat as orthogonal
        }

        let cos_sim = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
        1.0 - cos_sim
    }

    /// Compute the Euclidean distance between two embedding vectors.
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7 MotDeepSort
// ─────────────────────────────────────────────────────────────────────────────

/// A DeepSORT track that combines motion tracking with an appearance buffer.
pub struct MotDeepSortTrack {
    /// The underlying SORT track with Kalman filter.
    pub track: MotTrack,
    /// Circular buffer of recent appearance embeddings.
    pub appearance_buffer: Vec<Vec<f64>>,
    /// Maximum number of embeddings to retain.
    pub buffer_size: usize,
}

impl MotDeepSortTrack {
    /// Create a new DeepSORT track.
    pub fn new(track: MotTrack, buffer_size: usize) -> Self {
        Self {
            track,
            appearance_buffer: Vec::new(),
            buffer_size,
        }
    }

    /// Add a new appearance embedding to the circular buffer.
    pub fn add_appearance(&mut self, feature: Vec<f64>) {
        if self.appearance_buffer.len() >= self.buffer_size {
            self.appearance_buffer.remove(0);
        }
        self.appearance_buffer.push(feature);
    }

    /// Compute the mean appearance embedding from the buffer.
    pub fn mean_appearance(&self) -> Option<Vec<f64>> {
        if self.appearance_buffer.is_empty() {
            return None;
        }
        let dim = self.appearance_buffer[0].len();
        let n = self.appearance_buffer.len() as f64;
        let mut mean = vec![0.0f64; dim];
        for feat in &self.appearance_buffer {
            for (m, f) in mean.iter_mut().zip(feat.iter()) {
                *m += f / n;
            }
        }
        Some(mean)
    }
}

/// DeepSORT: Deep Appearance Metric SORT (Wojke et al. 2017).
///
/// Extends SORT with deep appearance features for robust re-identification
/// across occlusions and similar-looking objects.
pub struct MotDeepSort {
    /// The inner SORT tracker.
    pub sort: MotSort,
    /// Appearance feature extractor.
    pub appearance: MotAppearanceFeature,
    /// Per-track appearance history (parallel to `sort.tracks`).
    pub appearance_tracks: Vec<MotDeepSortTrack>,
    /// Weight for motion cost vs. appearance cost (0=pure appearance, 1=pure motion).
    pub lambda: f64,
    /// Maximum cosine distance for a valid appearance match.
    pub appearance_threshold: f64,
}

impl MotDeepSort {
    /// Create a new DeepSORT tracker.
    pub fn new(config: MotSortConfig, embed_dim: usize) -> Self {
        let patch_dim = 64;
        let hidden_dim = 128;
        Self {
            sort: MotSort::new(config),
            appearance: MotAppearanceFeature::new(patch_dim, hidden_dim, embed_dim),
            appearance_tracks: Vec::new(),
            lambda: 0.5,
            appearance_threshold: 0.7,
        }
    }

    /// Update the DeepSORT tracker with detections and their appearance patches.
    ///
    /// Returns `(track_id, bbox)` pairs for all confirmed, active tracks.
    pub fn update(
        &mut self,
        detections: &[(MotBoundingBox, Vec<f64>)],
    ) -> Vec<(usize, MotBoundingBox)> {
        let bboxes: Vec<MotBoundingBox> = detections.iter().map(|(b, _)| b.clone()).collect();
        let det_features: Vec<Vec<f64>> = detections
            .iter()
            .map(|(_, patch)| self.appearance.extract(patch))
            .collect();

        self.sort.frame += 1;

        // Predict all tracks
        for track in &mut self.sort.tracks {
            track.predict();
        }

        let predicted_bboxes: Vec<MotBoundingBox> = self
            .sort
            .tracks
            .iter()
            .map(|t| t.predicted_bbox())
            .collect();

        // Build combined cost matrix
        let n_tracks = predicted_bboxes.len();
        let n_dets = bboxes.len();

        let matched;
        let unmatched_tracks;
        let unmatched_dets;

        if n_tracks == 0 {
            matched = vec![];
            unmatched_tracks = vec![];
            unmatched_dets = (0..n_dets).collect();
        } else if n_dets == 0 {
            matched = vec![];
            unmatched_tracks = (0..n_tracks).collect();
            unmatched_dets = vec![];
        } else {
            // Collect track appearance features
            let track_features: Vec<Vec<f64>> = self
                .appearance_tracks
                .iter()
                .filter_map(|t| t.mean_appearance())
                .collect();

            let iou_cost = MotHungarian::iou_cost_matrix(&predicted_bboxes, &bboxes);

            // Compute combined cost
            let combined_cost: Vec<Vec<f64>> = if track_features.len() == n_tracks {
                let app_cost = self.appearance_cost_matrix(&track_features, &det_features);
                (0..n_tracks)
                    .map(|ti| {
                        (0..n_dets)
                            .map(|di| {
                                self.lambda * iou_cost[ti][di]
                                    + (1.0 - self.lambda) * app_cost[ti][di]
                            })
                            .collect()
                    })
                    .collect()
            } else {
                iou_cost
            };

            let assignment = MotHungarian::solve(&combined_cost);
            let threshold = self.sort.config.iou_threshold;

            let mut _matched = Vec::new();
            let mut _unmatched_tracks = Vec::new();
            let mut matched_det_set = vec![false; n_dets];

            for (ti, opt_di) in assignment.iter().enumerate() {
                if ti >= n_tracks {
                    break;
                }
                match opt_di {
                    Some(di) if combined_cost[ti][*di] <= 1.0 - threshold => {
                        _matched.push((ti, *di));
                        matched_det_set[*di] = true;
                    }
                    _ => {
                        _unmatched_tracks.push(ti);
                    }
                }
            }

            let _unmatched_dets: Vec<usize> =
                (0..n_dets).filter(|&di| !matched_det_set[di]).collect();

            matched = _matched;
            unmatched_tracks = _unmatched_tracks;
            unmatched_dets = _unmatched_dets;
        }

        // Update matched tracks
        for (ti, di) in &matched {
            self.sort.tracks[*ti].update(&bboxes[*di]);
            if *ti < self.appearance_tracks.len() {
                self.appearance_tracks[*ti].add_appearance(det_features[*di].clone());
            }
        }

        // Handle unmatched tracks
        for ti in &unmatched_tracks {
            if self.sort.tracks[*ti].time_since_update > self.sort.config.max_age {
                self.sort.tracks[*ti].state = MotTrackState::Deleted;
            }
        }

        // Create new tracks for unmatched detections
        for di in &unmatched_dets {
            let id = self.sort.next_id;
            self.sort.next_id += 1;
            let new_track = MotTrack::new(id, &bboxes[*di]);
            let mut ds_track = MotDeepSortTrack::new(new_track.clone(), 100);
            ds_track.add_appearance(det_features[*di].clone());
            self.sort.tracks.push(new_track);
            self.appearance_tracks.push(ds_track);
        }

        // Update track states
        for track in &mut self.sort.tracks {
            if (track.hits >= self.sort.config.min_hits
                || self.sort.frame <= self.sort.config.min_hits)
                && track.state == MotTrackState::Tentative {
                    track.state = MotTrackState::Confirmed;
                }
        }

        // Remove deleted tracks
        let deleted_ids: Vec<usize> = self
            .sort
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.state == MotTrackState::Deleted)
            .map(|(i, _)| i)
            .collect();

        // Remove in reverse order to preserve indices
        for &i in deleted_ids.iter().rev() {
            self.sort.tracks.remove(i);
            if i < self.appearance_tracks.len() {
                self.appearance_tracks.remove(i);
            }
        }

        // Collect outputs
        self.sort
            .tracks
            .iter()
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .map(|t| (t.track_id, t.predicted_bbox()))
            .collect()
    }

    /// Compute appearance cost matrix (cosine distance).
    ///
    /// `cost[i][j] = cosine_distance(track_features[i], det_features[j])`
    pub fn appearance_cost_matrix(
        &self,
        track_features: &[Vec<f64>],
        det_features: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        track_features
            .iter()
            .map(|tf| {
                det_features
                    .iter()
                    .map(|df| MotAppearanceFeature::cosine_distance(tf, df))
                    .collect()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8 MotByteTrack
// ─────────────────────────────────────────────────────────────────────────────

/// ByteTrack: Multi-Object Tracking by Associating Every Detection Box
/// (Zhang et al. 2022).
///
/// Performs two-stage association:
/// 1. Associate high-confidence detections with existing tracks.
/// 2. Associate low-confidence detections with remaining unmatched tracks.
///    This recovers occluded or partially visible objects.
pub struct MotByteTrack {
    /// Inner SORT tracker used for state management.
    pub inner: MotSort,
    /// Lower confidence threshold for including detections in stage 2.
    pub low_thresh: f64,
    /// Upper confidence threshold for stage 1 (high-confidence) detections.
    pub high_thresh: f64,
    /// IoU threshold for the second association stage.
    pub second_iou_thresh: f64,
}

impl MotByteTrack {
    /// Create a new ByteTrack tracker.
    pub fn new(high_thresh: f64, low_thresh: f64) -> Self {
        let config = MotSortConfig {
            max_age: 30,
            min_hits: 3,
            iou_threshold: 0.3,
        };
        Self {
            inner: MotSort::new(config),
            low_thresh,
            high_thresh,
            second_iou_thresh: 0.5,
        }
    }

    /// Update the ByteTrack tracker with detections and confidence scores.
    ///
    /// Returns `(track_id, bbox, confidence)` for all confirmed tracks.
    pub fn update(
        &mut self,
        detections: &[(MotBoundingBox, f64)],
    ) -> Vec<(usize, MotBoundingBox, f64)> {
        self.inner.frame += 1;

        // Split detections by confidence
        let (high_dets, low_dets) = Self::split_by_confidence(detections, self.high_thresh);

        // Predict all existing tracks
        for track in &mut self.inner.tracks {
            track.predict();
        }

        let predicted_bboxes: Vec<MotBoundingBox> = self
            .inner
            .tracks
            .iter()
            .map(|t| t.predicted_bbox())
            .collect();

        // Convert reference slices to owned vecs for associate()
        let high_dets_owned: Vec<MotBoundingBox> = high_dets.iter().map(|b| (*b).clone()).collect();
        let low_dets_owned: Vec<MotBoundingBox> = low_dets.iter().map(|b| (*b).clone()).collect();

        // ── Stage 1: Associate high-confidence detections ──────────────────
        let (matched1, unmatched_tracks1, unmatched_high_dets) = MotSort::associate(
            &predicted_bboxes,
            &high_dets_owned,
            self.inner.config.iou_threshold,
        );

        // Get original confidence for high-conf detections (by position)
        let high_conf_scores: Vec<f64> = {
            let mut scores = Vec::new();
            for (_, score) in detections.iter().filter(|(_, s)| *s >= self.high_thresh) {
                scores.push(*score);
            }
            scores
        };

        // Update matched tracks (stage 1)
        for (ti, di) in &matched1 {
            self.inner.tracks[*ti].update(&high_dets_owned[*di]);
        }

        // ── Stage 2: Associate low-confidence detections with remaining tracks ──
        let unmatched_track_bboxes: Vec<MotBoundingBox> = unmatched_tracks1
            .iter()
            .map(|&ti| predicted_bboxes[ti].clone())
            .collect();

        let (matched2, still_unmatched, _) = MotSort::associate(
            &unmatched_track_bboxes,
            &low_dets_owned,
            self.second_iou_thresh,
        );

        // Update matched tracks (stage 2) — only update Kalman, not count as confirmed hit
        for (rel_ti, di) in &matched2 {
            let actual_ti = unmatched_tracks1[*rel_ti];
            self.inner.tracks[actual_ti].update(&low_dets_owned[*di]);
        }

        // Mark remaining unmatched tracks (from both stages)
        let actually_unmatched: Vec<usize> = still_unmatched
            .iter()
            .map(|&rel_ti| unmatched_tracks1[rel_ti])
            .collect();

        for ti in &actually_unmatched {
            if self.inner.tracks[*ti].time_since_update > self.inner.config.max_age {
                self.inner.tracks[*ti].state = MotTrackState::Deleted;
            }
        }

        // ── Stage 3: Initialize new tracks from unmatched high-conf detections ──
        for di in &unmatched_high_dets {
            let id = self.inner.next_id;
            self.inner.next_id += 1;
            self.inner
                .tracks
                .push(MotTrack::new(id, &high_dets_owned[*di]));
        }

        // Update track confirmation state
        for track in &mut self.inner.tracks {
            if (track.hits >= self.inner.config.min_hits
                || self.inner.frame <= self.inner.config.min_hits)
                && track.state == MotTrackState::Tentative {
                    track.state = MotTrackState::Confirmed;
                }
        }

        // Remove deleted tracks
        self.inner.tracks.retain(|t| !t.is_deleted());

        // Collect outputs with confidence scores
        let mut results = Vec::new();
        for track in &self.inner.tracks {
            if track.is_confirmed() && track.time_since_update == 0 {
                // Find best matching confidence (use high_thresh as default)
                let conf = high_conf_scores
                    .first()
                    .copied()
                    .unwrap_or(self.high_thresh);
                results.push((track.track_id, track.predicted_bbox(), conf));
            }
        }

        results
    }

    /// Split detections into high- and low-confidence groups.
    ///
    /// Returns `(high_confidence_bboxes, low_confidence_bboxes)`.
    pub fn split_by_confidence(
        detections: &[(MotBoundingBox, f64)],
        threshold: f64,
    ) -> (Vec<&MotBoundingBox>, Vec<&MotBoundingBox>) {
        let mut high = Vec::new();
        let mut low = Vec::new();
        for (bbox, score) in detections {
            if *score >= threshold {
                high.push(bbox);
            } else {
                low.push(bbox);
            }
        }
        (high, low)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9 MotEloRating
// ─────────────────────────────────────────────────────────────────────────────

/// ELO-based rating system for comparing multi-object tracker configurations.
///
/// Each tracker configuration maintains an ELO rating that is updated based
/// on head-to-head comparisons using standard ELO update rules.
pub struct MotEloRating {
    /// List of `(tracker_name, elo_score)` pairs.
    pub ratings: Vec<(String, f64)>,
    /// ELO K-factor (controls how much each match affects ratings).
    pub k_factor: f64,
}

impl MotEloRating {
    /// Create a new ELO rating system.
    pub fn new(k_factor: f64) -> Self {
        Self {
            ratings: Vec::new(),
            k_factor,
        }
    }

    /// Register a new tracker with an initial ELO score.
    pub fn add_tracker(&mut self, name: &str, initial_elo: f64) {
        self.ratings.push((name.to_string(), initial_elo));
    }

    /// Compute the expected score for player A against player B.
    ///
    /// `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`
    pub fn expected_score(ra: f64, rb: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rb - ra) / 400.0))
    }

    /// Update ELO ratings after a match where `winner_idx` beats `loser_idx`.
    pub fn update_ratings(&mut self, winner_idx: usize, loser_idx: usize) {
        if winner_idx >= self.ratings.len() || loser_idx >= self.ratings.len() {
            return;
        }
        let ra = self.ratings[winner_idx].1;
        let rb = self.ratings[loser_idx].1;

        let e_winner = Self::expected_score(ra, rb);
        let e_loser = Self::expected_score(rb, ra);

        self.ratings[winner_idx].1 += self.k_factor * (1.0 - e_winner);
        self.ratings[loser_idx].1 += self.k_factor * (0.0 - e_loser);
    }

    /// Return the sorted ranking (highest ELO first).
    pub fn ranking(&self) -> Vec<(String, f64)> {
        let mut sorted = self.ratings.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10 MotMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Standard multi-object tracking evaluation metrics.
pub struct MotMetrics;

impl MotMetrics {
    /// Compute MOTA (Multiple Object Tracking Accuracy).
    ///
    /// `MOTA = 1 - (FP + FN + IDSW) / GT`
    ///
    /// Values > 1 indicate fewer errors than ground truth objects (rare but possible).
    pub fn mota(
        n_false_positives: usize,
        n_missed_detections: usize,
        n_id_switches: usize,
        n_gt_objects: usize,
    ) -> f64 {
        if n_gt_objects == 0 {
            return if n_false_positives == 0 && n_id_switches == 0 {
                1.0
            } else {
                0.0
            };
        }
        let errors = (n_false_positives + n_missed_detections + n_id_switches) as f64;
        1.0 - errors / n_gt_objects as f64
    }

    /// Compute IDF1 (Identity F1 Score).
    ///
    /// `IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)`
    pub fn idf1(
        id_true_positives: usize,
        id_false_positives: usize,
        id_false_negatives: usize,
    ) -> f64 {
        let numerator = 2 * id_true_positives;
        let denominator = 2 * id_true_positives + id_false_positives + id_false_negatives;
        if denominator == 0 {
            return 1.0;
        }
        numerator as f64 / denominator as f64
    }

    /// Compute HOTA (Higher Order Tracking Accuracy).
    ///
    /// `HOTA = sqrt(DetA * AssA)`
    ///
    /// Both inputs should be in [0, 1].
    pub fn hota(detection_accuracy: f64, association_accuracy: f64) -> f64 {
        (detection_accuracy.max(0.0) * association_accuracy.max(0.0)).sqrt()
    }

    /// Compute the completeness (quality) of a single track.
    ///
    /// `quality = hits / age`
    pub fn track_quality(track: &MotTrack) -> f64 {
        if track.age == 0 {
            return 0.0;
        }
        (track.hits as f64 / track.age as f64).clamp(0.0, 1.0)
    }

    /// Count the number of identity switches in a predicted trajectory.
    ///
    /// `pred_trajectory`: sequence of `(frame_idx, track_id)` pairs for a
    /// single ground-truth object. An identity switch occurs whenever the
    /// assigned track_id changes between consecutive frames.
    pub fn count_id_switches(pred_trajectory: &[(usize, usize)]) -> usize {
        if pred_trajectory.len() < 2 {
            return 0;
        }
        let mut count = 0;
        // Sort by frame to ensure temporal order
        let mut sorted = pred_trajectory.to_vec();
        sorted.sort_by_key(|(f, _)| *f);

        for window in sorted.windows(2) {
            if window[0].1 != window[1].1 {
                count += 1;
            }
        }
        count
    }

    /// Compute the average IoU between predicted and ground-truth bounding boxes.
    ///
    /// Matches by frame index; unmatched frames contribute 0 IoU.
    pub fn average_iou_over_time(
        predicted: &[(usize, MotBoundingBox)],
        gt: &[(usize, MotBoundingBox)],
    ) -> f64 {
        if gt.is_empty() {
            return 0.0;
        }

        // Build a map from frame → predicted bbox
        let mut pred_map: std::collections::HashMap<usize, &MotBoundingBox> =
            std::collections::HashMap::new();
        for (frame, bbox) in predicted {
            pred_map.insert(*frame, bbox);
        }

        let total_iou: f64 = gt
            .iter()
            .map(|(frame, gt_box)| {
                pred_map
                    .get(frame)
                    .map(|pred_box| gt_box.iou(pred_box))
                    .unwrap_or(0.0)
            })
            .sum();

        total_iou / gt.len() as f64
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
