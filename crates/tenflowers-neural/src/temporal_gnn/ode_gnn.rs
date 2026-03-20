//! ODE-based GNN for continuous-time dynamic graphs.
//!
//! GraphOdeFunc, OdeGnn, InterpNodeFeatures, EventGraph.

use tenflowers_core::{Result, TensorError};

use super::types::{linear, matvec, rand_mat, tanh_act, vecadd, zero_vec, OdeSolver, TemporalEdge};
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ─────────────────────────────────────────────────────────────────────────────
// GraphOdeFunc
// ─────────────────────────────────────────────────────────────────────────────

/// Defines the ODE dynamics dx/dt = f(x, A, t) for a graph.
#[derive(Debug, Clone)]
pub struct GraphOdeFunc {
    /// Node feature / hidden dimension.
    pub hidden_dim: usize,
    /// Number of nodes.
    pub n_nodes: usize,
    // GNN layer weights.
    w_agg: Vec<Vec<f64>>,
    b_agg: Vec<f64>,
    w_self: Vec<Vec<f64>>,
    b_self: Vec<f64>,
}

impl GraphOdeFunc {
    /// Create a new GraphOdeFunc.
    pub fn new(hidden_dim: usize, n_nodes: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let s = (2.0 / hidden_dim as f64).sqrt();
        Self {
            hidden_dim,
            n_nodes,
            w_agg: rand_mat(hidden_dim, hidden_dim, s, &mut rng),
            b_agg: zero_vec(hidden_dim),
            w_self: rand_mat(hidden_dim, hidden_dim, s, &mut rng),
            b_self: zero_vec(hidden_dim),
        }
    }

    /// Evaluate dx/dt given current state `x` and adjacency `adj`.
    ///
    /// `x`: `[N × hidden_dim]` flattened as `[N][hidden_dim]`.
    /// Returns `dx`: same shape.
    pub fn eval(&self, x: &[Vec<f64>], adj: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = x.len();
        if adj.len() != n {
            return Err(TensorError::invalid_argument(format!(
                "GraphOdeFunc: adj row count {} != n_nodes {}",
                adj.len(),
                n
            )));
        }
        let mut dx: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            // Aggregate neighbour states.
            let mut agg = vec![0.0f64; self.hidden_dim];
            for j in 0..n {
                let a = adj[i][j];
                if a.abs() < 1e-12 {
                    continue;
                }
                for (ak, &xk) in agg.iter_mut().zip(x[j].iter()) {
                    *ak += a * xk;
                }
            }
            let agg_t = linear(&self.w_agg, &self.b_agg, &agg);
            let self_t = linear(&self.w_self, &self.b_self, &x[i]);
            let sum = vecadd(&agg_t, &self_t);
            let dxi: Vec<f64> = sum.iter().map(|&v| tanh_act(v)).collect();
            dx.push(dxi);
        }
        Ok(dx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OdeGnn
// ─────────────────────────────────────────────────────────────────────────────

/// ODE-based GNN for continuous-time dynamic graphs.
///
/// Evolves node embeddings from time `t0` to `t1` by numerically integrating
/// the graph ODE `dx/dt = f(x, A, t)`.
#[derive(Debug, Clone)]
pub struct OdeGnn {
    pub func: GraphOdeFunc,
    pub solver: OdeSolver,
    /// Number of integration steps.
    pub n_steps: usize,
    pub hidden_dim: usize,
}

impl OdeGnn {
    /// Create a new ODE-GNN.
    pub fn new(
        hidden_dim: usize,
        n_nodes: usize,
        n_steps: usize,
        solver: OdeSolver,
        seed: u64,
    ) -> Self {
        Self {
            func: GraphOdeFunc::new(hidden_dim, n_nodes, seed),
            solver,
            n_steps,
            hidden_dim,
        }
    }

    /// Integrate node states from t0 to t1.
    ///
    /// `x0`: initial node states `[N][hidden_dim]`.
    /// `adj`: `[N][N]` adjacency matrix.
    /// Returns final node states.
    pub fn integrate(
        &self,
        x0: &[Vec<f64>],
        adj: &[Vec<f64>],
        t0: f64,
        t1: f64,
    ) -> Result<Vec<Vec<f64>>> {
        if t1 <= t0 {
            return Err(TensorError::invalid_argument(format!(
                "OdeGnn::integrate — t1 ({t1}) must be > t0 ({t0})"
            )));
        }
        let dt = (t1 - t0) / self.n_steps as f64;
        let mut x = x0.to_vec();
        match self.solver {
            OdeSolver::Euler => {
                for step in 0..self.n_steps {
                    let _t = t0 + step as f64 * dt;
                    let dx = self.func.eval(&x, adj)?;
                    for (xi, dxi) in x.iter_mut().zip(dx.iter()) {
                        for (xij, &dxij) in xi.iter_mut().zip(dxi.iter()) {
                            *xij += dt * dxij;
                        }
                    }
                }
            }
            OdeSolver::Rk4 => {
                for step in 0..self.n_steps {
                    let _t = t0 + step as f64 * dt;
                    let k1 = self.func.eval(&x, adj)?;
                    let x2: Vec<Vec<f64>> = x
                        .iter()
                        .zip(k1.iter())
                        .map(|(xi, k1i)| {
                            xi.iter()
                                .zip(k1i.iter())
                                .map(|(&v, &k)| v + 0.5 * dt * k)
                                .collect()
                        })
                        .collect();
                    let k2 = self.func.eval(&x2, adj)?;
                    let x3: Vec<Vec<f64>> = x
                        .iter()
                        .zip(k2.iter())
                        .map(|(xi, k2i)| {
                            xi.iter()
                                .zip(k2i.iter())
                                .map(|(&v, &k)| v + 0.5 * dt * k)
                                .collect()
                        })
                        .collect();
                    let k3 = self.func.eval(&x3, adj)?;
                    let x4: Vec<Vec<f64>> = x
                        .iter()
                        .zip(k3.iter())
                        .map(|(xi, k3i)| {
                            xi.iter()
                                .zip(k3i.iter())
                                .map(|(&v, &k)| v + dt * k)
                                .collect()
                        })
                        .collect();
                    let k4 = self.func.eval(&x4, adj)?;
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] +=
                                dt / 6.0 * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
                        }
                    }
                }
            }
        }
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InterpNodeFeatures — linear interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Stores (time, feature) snapshots and allows querying at arbitrary times via
/// linear interpolation.
#[derive(Debug, Clone)]
pub struct InterpNodeFeatures {
    /// Sorted snapshots: (time, features).
    pub snapshots: Vec<(f64, Vec<f64>)>,
}

impl InterpNodeFeatures {
    /// Create from unsorted (time, feature) pairs.
    pub fn new(mut snapshots: Vec<(f64, Vec<f64>)>) -> Result<Self> {
        if snapshots.is_empty() {
            return Err(TensorError::invalid_argument(
                "InterpNodeFeatures: need at least one snapshot".to_string(),
            ));
        }
        snapshots.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Self { snapshots })
    }

    /// Interpolate features at query time `t`.
    pub fn query(&self, t: f64) -> Vec<f64> {
        let snaps = &self.snapshots;
        if t <= snaps[0].0 {
            return snaps[0].1.clone();
        }
        if t >= snaps[snaps.len() - 1].0 {
            return snaps[snaps.len() - 1].1.clone();
        }
        // Binary search for the interval.
        let pos = snaps.partition_point(|s| s.0 <= t);
        let lo = pos.saturating_sub(1);
        let hi = pos.min(snaps.len() - 1);
        if lo == hi {
            return snaps[lo].1.clone();
        }
        let t0 = snaps[lo].0;
        let t1 = snaps[hi].0;
        let alpha = if (t1 - t0).abs() < 1e-15 {
            0.0
        } else {
            (t - t0) / (t1 - t0)
        };
        snaps[lo]
            .1
            .iter()
            .zip(snaps[hi].1.iter())
            .map(|(&a, &b)| a + alpha * (b - a))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EventGraph — streaming edge events with replay buffer
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates streaming temporal edges and supports replay.
#[derive(Debug, Clone)]
pub struct EventGraph {
    /// All accumulated events.
    pub events: Vec<TemporalEdge>,
    /// Maximum buffer size (oldest events are dropped when exceeded).
    pub max_buffer: usize,
}

impl EventGraph {
    /// Create a new EventGraph with given buffer limit.
    pub fn new(max_buffer: usize) -> Result<Self> {
        if max_buffer == 0 {
            return Err(TensorError::invalid_argument(
                "EventGraph: max_buffer must be > 0".to_string(),
            ));
        }
        Ok(Self {
            events: Vec::new(),
            max_buffer,
        })
    }

    /// Add a new edge event.
    pub fn add_event(&mut self, edge: TemporalEdge) {
        if self.events.len() >= self.max_buffer {
            self.events.remove(0);
        }
        self.events.push(edge);
    }

    /// Replay events in chronological order in time range `[t_start, t_end]`.
    pub fn replay(&self, t_start: f64, t_end: f64) -> Vec<&TemporalEdge> {
        let mut events: Vec<&TemporalEdge> = self
            .events
            .iter()
            .filter(|e| e.time >= t_start && e.time <= t_end)
            .collect();
        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        events
    }
}
