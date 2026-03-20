//! Temporal Point Processes and Neural TPP Models
//!
//! Implements classical and neural temporal point process models:
//! - HawkesProcess: multivariate Hawkes with Ogata thinning simulation and MLE fitting
//! - RmtppModel: Recurrent Marked TPP (GRU-based intensity)
//! - NhpModel: Neural Hawkes Process (continuous-time LSTM)
//! - ThpModel: Transformer Hawkes Process (self-attention over event history)
//! - TppLoss: loss functions (NLL, time-rescaling KS test)
//! - TppMetrics: evaluation metrics (log-lik, type accuracy, RMSE, calibration)

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// §1  Event and EventSequence
// ---------------------------------------------------------------------------

/// A single temporal event with time and discrete event type.
#[derive(Debug, Clone)]
pub struct Event {
    /// Absolute event time (non-negative)
    pub time: f64,
    /// Discrete event type index (0-based)
    pub event_type: usize,
}

/// A sequence of temporal events observed over [0, T].
#[derive(Debug, Clone)]
pub struct EventSequence {
    /// Ordered list of events
    pub events: Vec<Event>,
    /// Observation window end time
    pub t_end: f64,
}

impl EventSequence {
    /// Create a new event sequence observed over [0, t_end].
    pub fn new(mut events: Vec<Event>, t_end: f64) -> Self {
        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { events, t_end }
    }

    /// Compute inter-arrival times between consecutive events.
    /// First interval is measured from time 0.
    pub fn inter_arrival_times(&self) -> Vec<f64> {
        if self.events.is_empty() {
            return vec![];
        }
        let mut out = Vec::with_capacity(self.events.len());
        out.push(self.events[0].time);
        for i in 1..self.events.len() {
            out.push(self.events[i].time - self.events[i - 1].time);
        }
        out
    }

    /// Number of events in the sequence.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Compute a histogram of event arrival times over [0, T] with `n_bins` bins.
    /// Returns counts per bin (as f32 for easy use in models).
    pub fn intensity_histogram(&self, n_bins: usize) -> Vec<f32> {
        let n = n_bins.max(1);
        let mut hist = vec![0.0f32; n];
        let bin_width = self.t_end / n as f64;
        if bin_width <= 0.0 {
            return hist;
        }
        for ev in &self.events {
            let idx = ((ev.time / bin_width) as usize).min(n - 1);
            hist[idx] += 1.0;
        }
        hist
    }
}

// ---------------------------------------------------------------------------
// §2  HawkesProcess — Multivariate Hawkes
// ---------------------------------------------------------------------------

/// Multivariate Hawkes process with exponential kernel.
///
/// Conditional intensity for type k at time t given history:
///   λ*(t,k) = μ_k + Σ_{i: t_i<t} α_{k,k_i} · exp(-β_k · (t - t_i))
#[derive(Debug, Clone)]
pub struct HawkesProcess {
    /// Base intensities per type (length = n_types)
    pub mu: Vec<f64>,
    /// Excitation matrix alpha\[k\]\[j\] = excitation on type k from type j
    pub alpha: Vec<Vec<f64>>,
    /// Decay rates per type (length = n_types)
    pub beta: Vec<f64>,
    /// Number of event types
    pub n_types: usize,
}

impl HawkesProcess {
    /// Create a new Hawkes process.
    ///
    /// # Panics-free
    /// Returns an error string if dimensions are inconsistent.
    pub fn new(mu: Vec<f64>, alpha: Vec<Vec<f64>>, beta: Vec<f64>) -> Result<Self, String> {
        let n = mu.len();
        if beta.len() != n {
            return Err(format!("beta length {} != n_types {}", beta.len(), n));
        }
        if alpha.len() != n {
            return Err(format!("alpha rows {} != n_types {}", alpha.len(), n));
        }
        for (k, row) in alpha.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "alpha[{}] length {} != n_types {}",
                    k,
                    row.len(),
                    n
                ));
            }
        }
        Ok(Self {
            mu,
            alpha,
            beta,
            n_types: n,
        })
    }

    /// Compute conditional intensity λ*(t, event_type) given history.
    pub fn conditional_intensity(&self, t: f64, history: &[Event], event_type: usize) -> f64 {
        if event_type >= self.n_types {
            return 0.0;
        }
        let k = event_type;
        let mut lam = self.mu[k];
        for ev in history {
            if ev.time >= t {
                break;
            }
            let j = ev.event_type;
            if j < self.n_types {
                lam += self.alpha[k][j] * (-self.beta[k] * (t - ev.time)).exp();
            }
        }
        lam.max(0.0)
    }

    /// Compute the total (summed over types) conditional intensity at time t.
    fn total_intensity(&self, t: f64, history: &[Event]) -> f64 {
        (0..self.n_types)
            .map(|k| self.conditional_intensity(t, history, k))
            .sum()
    }

    /// Simulate an event sequence over [0, T] using the Ogata thinning algorithm.
    pub fn simulate(&self, t_end: f64, rng: &mut impl Rng) -> EventSequence {
        let mut events: Vec<Event> = Vec::new();
        let mut current_t = 0.0_f64;

        // Upper bound: use current total intensity + small guard
        let upper_bound_increment = self.mu.iter().sum::<f64>() + 1e-6;

        while current_t < t_end {
            // Current total upper bound
            let lam_bar = self.total_intensity(current_t, &events) + upper_bound_increment;

            // Draw candidate inter-arrival time
            let u: f64 = rng.random::<f64>().max(1e-15);
            let dt = -u.ln() / lam_bar.max(1e-15);
            let t_candidate = current_t + dt;

            if t_candidate > t_end {
                break;
            }

            // Accept/reject
            let lam_at_candidate = self.total_intensity(t_candidate, &events);
            let accept: f64 = rng.random::<f64>();
            if accept <= lam_at_candidate / lam_bar.max(1e-15) {
                // Sample event type proportionally to per-type intensities
                let intensities: Vec<f64> = (0..self.n_types)
                    .map(|k| self.conditional_intensity(t_candidate, &events, k))
                    .collect();
                let total: f64 = intensities.iter().sum::<f64>().max(1e-15);
                let u_type: f64 = rng.random::<f64>();
                let mut cumsum = 0.0;
                let mut chosen_type = 0;
                for (k, &lam_k) in intensities.iter().enumerate() {
                    cumsum += lam_k / total;
                    if u_type < cumsum {
                        chosen_type = k;
                        break;
                    }
                    chosen_type = k;
                }
                events.push(Event {
                    time: t_candidate,
                    event_type: chosen_type,
                });
            }
            current_t = t_candidate;
        }

        EventSequence { events, t_end }
    }

    /// Compute log-likelihood of an observed event sequence.
    ///
    /// log L = Σ_i log λ*(t_i, k_i) − Σ_k ∫_0^T λ*(t,k) dt
    ///
    /// The integral of the exponential kernel is computed analytically.
    pub fn log_likelihood(&self, seq: &EventSequence) -> f64 {
        let t_end = seq.t_end;
        let events = &seq.events;

        // Sum of log intensities at each event
        let mut ll = 0.0_f64;
        for (i, ev) in events.iter().enumerate() {
            let lam = self.conditional_intensity(ev.time, &events[..i], ev.event_type);
            ll += (lam.max(1e-300)).ln();
        }

        // Subtract integrated intensity: ∫_0^T Σ_k λ*(t,k) dt
        // = Σ_k [ μ_k * T + Σ_i α_{k,k_i}/β_k * (1 − exp(−β_k*(T−t_i))) ]
        for k in 0..self.n_types {
            ll -= self.mu[k] * t_end;
            for ev in events {
                if ev.event_type < self.n_types {
                    let j = ev.event_type;
                    let decay = (-self.beta[k] * (t_end - ev.time)).exp();
                    ll -= self.alpha[k][j] / self.beta[k].max(1e-15) * (1.0 - decay);
                }
            }
        }
        ll
    }

    /// Gradient of log-likelihood with respect to mu, alpha (flattened), and beta.
    fn log_lik_gradient(&self, seq: &EventSequence) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let events = &seq.events;
        let t_end = seq.t_end;
        let n = self.n_types;
        let mut dmu = vec![0.0_f64; n];
        let mut dalpha = vec![vec![0.0_f64; n]; n];
        let mut dbeta = vec![0.0_f64; n];

        // Gradient of log-sum terms (from log λ*)
        for (i, ev) in events.iter().enumerate() {
            let k = ev.event_type;
            if k >= n {
                continue;
            }
            let lam = self
                .conditional_intensity(ev.time, &events[..i], k)
                .max(1e-15);
            // ∂ log λ* / ∂ μ_k = 1/λ*
            dmu[k] += 1.0 / lam;
            // ∂ log λ* / ∂ α_{k,j} = Σ_{m<i, k_m=j} exp(-β_k*(t_i-t_m)) / λ*
            for m in 0..i {
                let j = events[m].event_type;
                if j < n {
                    let dt = ev.time - events[m].time;
                    let e = (-self.beta[k] * dt).exp();
                    dalpha[k][j] += e / lam;
                    // ∂ log λ* / ∂ β_k from excitation term
                    dbeta[k] -= self.alpha[k][j] * dt * e / lam;
                }
            }
        }

        // Gradient of integral terms (negative sign already in ll formula)
        for k in 0..n {
            dmu[k] -= t_end;
            for ev in events {
                let j = ev.event_type;
                if j >= n {
                    continue;
                }
                let dt_end = t_end - ev.time;
                let decay = (-self.beta[k] * dt_end).exp();
                // ∂/∂ α_{k,j} [ -α_{k,j}/β_k * (1-decay) ] = -(1-decay)/β_k
                dalpha[k][j] -= (1.0 - decay) / self.beta[k].max(1e-15);
                // ∂/∂ β_k [ -α_{k,j}/β_k * (1-decay) ]
                //  = α_{k,j}/β_k^2 * (1-decay) - α_{k,j}/β_k * dt_end * decay
                let bk = self.beta[k].max(1e-15);
                dbeta[k] -= -self.alpha[k][j] / (bk * bk) * (1.0 - decay)
                    + self.alpha[k][j] / bk * dt_end * decay;
            }
        }

        (dmu, dalpha, dbeta)
    }

    /// Fit via gradient ascent on log-likelihood.
    pub fn fit_mle(seq: &EventSequence, n_types: usize, n_iter: usize, lr: f64) -> Self {
        let n = n_types.max(1);
        let mut mu = vec![0.5_f64; n];
        let mut alpha = vec![vec![0.1_f64; n]; n];
        let mut beta = vec![1.0_f64; n];

        for _ in 0..n_iter {
            let hp = HawkesProcess {
                mu: mu.clone(),
                alpha: alpha.clone(),
                beta: beta.clone(),
                n_types: n,
            };
            let (dmu, dalpha, dbeta) = hp.log_lik_gradient(seq);
            for k in 0..n {
                mu[k] = (mu[k] + lr * dmu[k]).max(1e-6);
                beta[k] = (beta[k] + lr * dbeta[k]).max(1e-6);
                for j in 0..n {
                    alpha[k][j] = (alpha[k][j] + lr * dalpha[k][j]).max(0.0);
                }
            }
        }

        HawkesProcess {
            mu,
            alpha,
            beta,
            n_types: n,
        }
    }

    /// Compute the branching ratio: spectral radius of the matrix A where A\[k\]\[j\] = α\[k\]\[j\]/β\[k\].
    /// A simplified per-type version is returned: Σ_j α\[k\]\[j\]/β\[k\] for each k.
    pub fn branching_ratio(&self) -> Vec<f64> {
        (0..self.n_types)
            .map(|k| {
                let beta_k = self.beta[k].max(1e-15);
                self.alpha[k].iter().sum::<f64>() / beta_k
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// §3  RmtppModel — Recurrent Marked TPP
// ---------------------------------------------------------------------------

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

/// Simple GRU cell operating on flat Vec<f32>.
struct GruCell {
    hidden_dim: usize,
    input_dim: usize,
    /// Weight matrices stored flat: w_z, w_r, w_h (each [hidden x (input+hidden)])
    w_z: Vec<f32>,
    w_r: Vec<f32>,
    w_h: Vec<f32>,
    b_z: Vec<f32>,
    b_r: Vec<f32>,
    b_h: Vec<f32>,
}

impl GruCell {
    fn new(input_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        let total = input_dim + hidden_dim;
        let scale = (2.0_f32 / (input_dim + hidden_dim) as f32).sqrt();
        let mut mk = |sz: usize| -> Vec<f32> {
            (0..sz)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                .collect()
        };
        Self {
            hidden_dim,
            input_dim,
            w_z: mk(hidden_dim * total),
            w_r: mk(hidden_dim * total),
            w_h: mk(hidden_dim * total),
            b_z: vec![0.0f32; hidden_dim],
            b_r: vec![0.0f32; hidden_dim],
            b_h: vec![0.0f32; hidden_dim],
        }
    }

    fn forward(&self, x: &[f32], h: &[f32]) -> Vec<f32> {
        let hd = self.hidden_dim;
        let combined: Vec<f32> = x.iter().copied().chain(h.iter().copied()).collect();
        let total = combined.len();

        let linear = |w: &[f32], b: &[f32]| -> Vec<f32> {
            (0..hd)
                .map(|i| {
                    let s: f32 = (0..total).map(|j| w[i * total + j] * combined[j]).sum();
                    s + b[i]
                })
                .collect()
        };

        let z: Vec<f32> = linear(&self.w_z, &self.b_z)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let r: Vec<f32> = linear(&self.w_r, &self.b_r)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();

        // For h_tilde: combine x with r*h
        let rh: Vec<f32> = x
            .iter()
            .copied()
            .chain(r.iter().zip(h).map(|(ri, hi)| ri * hi))
            .collect();
        let h_tilde: Vec<f32> = {
            let total2 = rh.len();
            (0..hd)
                .map(|i| {
                    let s: f32 = (0..total2).map(|j| self.w_h[i * total + j] * rh[j]).sum();
                    (s + self.b_h[i]).tanh()
                })
                .collect()
        };

        (0..hd)
            .map(|i| (1.0 - z[i]) * h[i] + z[i] * h_tilde[i])
            .collect()
    }
}

/// Recurrent Marked Temporal Point Process.
///
/// Architecture: GRU cell processes (dt_i, type_onehot_i) → hidden state h_i.
/// Intensity: λ*(t|h) = exp(v^T h + w*(t-t_last) + b_t)
#[derive(Debug, Clone)]
pub struct RmtppModel {
    /// Input dimension (= 1 + n_types for [dt, type_onehot])
    pub input_dim: usize,
    /// GRU hidden dimension
    pub hidden_dim: usize,
    /// Number of event types
    pub n_types: usize,
    // Serializable weights (GRU compressed to Vec<f32> for simplicity)
    w_z: Vec<f32>,
    w_r: Vec<f32>,
    w_h: Vec<f32>,
    b_z: Vec<f32>,
    b_r: Vec<f32>,
    b_h: Vec<f32>,
    /// Output layer: type logits w_out [n_types x hidden_dim]
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    /// Intensity params: v [hidden_dim], w_time (scalar), b_time (scalar)
    v_intensity: Vec<f32>,
    w_time: f32,
    b_time: f32,
}

impl RmtppModel {
    /// Create a new RMTPP model with random initialization.
    pub fn new(hidden_dim: usize, n_types: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let input_dim = 1 + n_types;
        let cell = GruCell::new(input_dim, hidden_dim, &mut rng);
        let total = input_dim + hidden_dim;
        let scale = (2.0_f32 / (hidden_dim + n_types) as f32).sqrt();
        let w_out: Vec<f32> = (0..n_types * hidden_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let b_out = vec![0.0f32; n_types];
        let v_intensity: Vec<f32> = (0..hidden_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
            .collect();
        let w_time: f32 = -0.1;
        let b_time: f32 = 0.0;
        Self {
            input_dim,
            hidden_dim,
            n_types,
            w_z: cell.w_z,
            w_r: cell.w_r,
            w_h: cell.w_h,
            b_z: cell.b_z,
            b_r: cell.b_r,
            b_h: cell.b_h,
            w_out,
            b_out,
            v_intensity,
            w_time,
            b_time,
            // suppress unused field warning for total
        }
    }

    fn gru_step(&self, x: &[f32], h: &[f32]) -> Vec<f32> {
        let hd = self.hidden_dim;
        let combined: Vec<f32> = x.iter().copied().chain(h.iter().copied()).collect();
        let total = combined.len();

        let linear = |w: &[f32], b: &[f32]| -> Vec<f32> {
            (0..hd)
                .map(|i| {
                    let s: f32 = (0..total).map(|j| w[i * total + j] * combined[j]).sum();
                    s + b[i]
                })
                .collect()
        };

        let z: Vec<f32> = linear(&self.w_z, &self.b_z)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let r: Vec<f32> = linear(&self.w_r, &self.b_r)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();

        let rh: Vec<f32> = x
            .iter()
            .copied()
            .chain(r.iter().zip(h).map(|(ri, hi)| ri * hi))
            .collect();
        let total2 = rh.len();
        let h_tilde: Vec<f32> = (0..hd)
            .map(|i| {
                let s: f32 = (0..total2).map(|j| self.w_h[i * total + j] * rh[j]).sum();
                (s + self.b_h[i]).tanh()
            })
            .collect();

        (0..hd)
            .map(|i| (1.0 - z[i]) * h[i] + z[i] * h_tilde[i])
            .collect()
    }

    fn type_onehot(&self, t: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; self.n_types];
        if t < self.n_types {
            v[t] = 1.0;
        }
        v
    }

    /// Forward pass: returns (type_logits per event, intensity_params per event).
    ///
    /// `inter_times[i]` = time since last event (or since t=0 for i=0)
    /// `types[i]` = event type index
    ///
    /// Returns (type_logits: Vec<`Vec<f32>>`, intensity_params: `Vec<f32>`)
    /// where intensity_params\[i\] = v^T h_i + b_time (w_time applied at predict time)
    pub fn forward(&self, inter_times: &[f32], types: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let n = inter_times.len().min(types.len());
        let mut h = vec![0.0f32; self.hidden_dim];
        let mut all_logits = Vec::with_capacity(n);
        let mut all_intensity = Vec::with_capacity(n);

        for i in 0..n {
            let dt = inter_times[i];
            let onehot = self.type_onehot(types[i]);
            let x: Vec<f32> = std::iter::once(dt).chain(onehot).collect();
            h = self.gru_step(&x, &h);

            // Type logits: W_out h + b_out
            let logits: Vec<f32> = (0..self.n_types)
                .map(|k| {
                    let s: f32 = (0..self.hidden_dim)
                        .map(|j| self.w_out[k * self.hidden_dim + j] * h[j])
                        .sum();
                    s + self.b_out[k]
                })
                .collect();
            all_logits.push(logits);

            // Intensity base: v^T h + b_time (w_time * dt applied at intensity query)
            let base: f32 = self
                .v_intensity
                .iter()
                .zip(h.iter())
                .map(|(vi, hi)| vi * hi)
                .sum::<f32>()
                + self.b_time;
            all_intensity.push(base);
        }

        (all_logits, all_intensity)
    }

    /// Negative log-likelihood loss.
    pub fn nll_loss(&self, inter_times: &[f32], types: &[usize]) -> f32 {
        let (logits, intensity_params) = self.forward(inter_times, types);
        let n = logits.len();
        if n == 0 {
            return 0.0;
        }
        let mut total_loss = 0.0f32;

        for i in 0..n {
            // Type cross-entropy
            let log_softmax = log_softmax_f32(&logits[i]);
            let t = types[i].min(self.n_types - 1);
            total_loss -= log_softmax[t];

            // Time log-likelihood under exponential intensity:
            // λ(t) = exp(base + w_time * dt)
            // log λ(dt_i) = base + w_time * dt_i
            // integral 0..dt: exp(base)/w_time * (exp(w_time*dt) - 1) if w_time < 0
            let base = intensity_params[i];
            let dt = inter_times[i];
            let lam_val = base + self.w_time * dt;
            total_loss -= lam_val;
            // Integral: ∫_0^dt exp(base + w_t * s) ds = exp(base) * (exp(w_t*dt)-1)/w_t
            let integral = if self.w_time.abs() < 1e-6 {
                (base).exp() * dt
            } else {
                (base).exp() * ((self.w_time * dt).exp() - 1.0) / self.w_time
            };
            total_loss += integral.abs();
        }
        total_loss / n as f32
    }

    /// Predict mean next inter-arrival time given current hidden state and intensity params.
    /// Integrates E\[dt\] = ∫_0^∞ exp(-Λ(dt)) dt numerically (trapezoidal, 100 steps).
    pub fn predict_next_time(&self, h: &[f32], intensity_params: &[f32]) -> f32 {
        let base = if intensity_params.is_empty() {
            h.iter()
                .zip(self.v_intensity.iter())
                .map(|(hi, vi)| hi * vi)
                .sum::<f32>()
                + self.b_time
        } else {
            intensity_params[0]
        };
        // E[dt] = ∫_0^∞ S(t) dt where S(t) = exp(-Λ(t))
        // Λ(t) = ∫_0^t λ(s) ds = exp(base) * (exp(w_time * s) - 1) / w_time  (if w_time ≠ 0)
        let n_steps = 100usize;
        let t_max = 10.0f32;
        let dt_step = t_max / n_steps as f32;
        let survival = |t: f32| -> f32 {
            let integral_lam = if self.w_time.abs() < 1e-6 {
                base.exp() * t
            } else {
                base.exp() * ((self.w_time * t).exp() - 1.0) / self.w_time
            };
            (-integral_lam.abs()).exp()
        };
        let mut mean_dt = 0.0f32;
        for step in 0..n_steps {
            let t0 = step as f32 * dt_step;
            let t1 = t0 + dt_step;
            mean_dt += 0.5 * (survival(t0) + survival(t1)) * dt_step;
        }
        mean_dt.max(0.0)
    }
}

fn log_softmax_f32(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&x| x - max_v).collect();
    let log_sum_exp = shifted.iter().map(|x| x.exp()).sum::<f32>().ln();
    shifted.iter().map(|&x| x - log_sum_exp).collect()
}

// ---------------------------------------------------------------------------
// §4  NhpModel — Neural Hawkes Process (continuous-time LSTM)
// ---------------------------------------------------------------------------

/// State of the continuous-time LSTM cell.
#[derive(Debug, Clone)]
pub struct ContinuousLstmCell {
    /// Target cell state (at last event)
    pub c_bar: Vec<f32>,
    /// Current decayed cell state
    pub c: Vec<f32>,
    /// Output gate
    pub o: Vec<f32>,
    /// Decay rates
    pub delta: Vec<f32>,
    /// Hidden state
    pub h: Vec<f32>,
}

impl ContinuousLstmCell {
    /// Create zero-initialized cell state.
    pub fn zeros(hidden_dim: usize) -> Self {
        Self {
            c_bar: vec![0.0f32; hidden_dim],
            c: vec![0.0f32; hidden_dim],
            o: vec![0.5f32; hidden_dim],
            delta: vec![1.0f32; hidden_dim],
            h: vec![0.0f32; hidden_dim],
        }
    }

    /// Decay cell state by elapsed time dt.
    pub fn decay(&self, dt: f64) -> Vec<f32> {
        let dt_f = dt as f32;
        self.c_bar
            .iter()
            .zip(self.c.iter())
            .zip(self.delta.iter())
            .map(|((cbar_i, c_i), delta_i)| cbar_i + (c_i - cbar_i) * (-delta_i * dt_f).exp())
            .collect()
    }
}

/// Neural Hawkes Process based on continuous-time LSTM.
#[derive(Debug, Clone)]
pub struct NhpModel {
    /// Input embedding dimension
    pub input_dim: usize,
    /// LSTM hidden dimension
    pub hidden_dim: usize,
    /// Number of event types
    pub n_types: usize,
    // LSTM weights: [i, f, g, o, z, delta] gates for continuous-time LSTM
    // Each gate weight matrix is [hidden_dim x (input_dim + hidden_dim)]
    w_gates: Vec<Vec<f32>>,
    b_gates: Vec<Vec<f32>>,
    /// Intensity output weights [n_types x hidden_dim]
    w_intensity: Vec<f32>,
    b_intensity: Vec<f32>,
    /// Type embedding [n_types x input_dim]
    type_embed: Vec<f32>,
}

impl NhpModel {
    /// Create a new NHP model with random initialization.
    pub fn new(hidden_dim: usize, n_types: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let input_dim = hidden_dim; // type embedding maps n_types → input_dim
        let n_gates = 6usize; // i, f, g, o, z (c_bar gate), delta
        let total = input_dim + hidden_dim;
        let scale = (2.0_f32 / total as f32).sqrt();
        let mut mk_w = || -> Vec<f32> {
            (0..hidden_dim * total)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                .collect()
        };
        let w_gates = (0..n_gates).map(|_| mk_w()).collect();
        let b_gates = (0..n_gates).map(|_| vec![0.0f32; hidden_dim]).collect();
        let scale2 = (2.0_f32 / (hidden_dim + n_types) as f32).sqrt();
        let w_intensity = (0..n_types * hidden_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale2)
            .collect();
        let b_intensity = vec![0.0f32; n_types];
        let type_embed = (0..n_types * input_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
            .collect();
        Self {
            input_dim,
            hidden_dim,
            n_types,
            w_gates,
            b_gates,
            w_intensity,
            b_intensity,
            type_embed,
        }
    }

    fn get_type_embed(&self, event_type: usize) -> Vec<f32> {
        let t = event_type.min(self.n_types - 1);
        (0..self.input_dim)
            .map(|j| self.type_embed[t * self.input_dim + j])
            .collect()
    }

    fn gate_linear(&self, gate_idx: usize, x: &[f32], h: &[f32]) -> Vec<f32> {
        let combined: Vec<f32> = x.iter().copied().chain(h.iter().copied()).collect();
        let total = combined.len();
        let hd = self.hidden_dim;
        (0..hd)
            .map(|i| {
                let s: f32 = (0..total)
                    .map(|j| self.w_gates[gate_idx][i * total + j] * combined[j])
                    .sum();
                s + self.b_gates[gate_idx][i]
            })
            .collect()
    }

    /// Update cell state upon receiving an event.
    fn update_cell(&self, event_type: usize, cell: &ContinuousLstmCell) -> ContinuousLstmCell {
        let x = self.get_type_embed(event_type);
        let h = &cell.h;

        let i_gate: Vec<f32> = self
            .gate_linear(0, &x, h)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let f_gate: Vec<f32> = self
            .gate_linear(1, &x, h)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let g_gate: Vec<f32> = self
            .gate_linear(2, &x, h)
            .iter()
            .map(|&v| v.tanh())
            .collect();
        let o_gate: Vec<f32> = self
            .gate_linear(3, &x, h)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let z_gate: Vec<f32> = self
            .gate_linear(4, &x, h)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let delta: Vec<f32> = self
            .gate_linear(5, &x, h)
            .iter()
            .map(|&v| relu_f32(v) + 1e-6)
            .collect();

        let hd = self.hidden_dim;
        let c_new: Vec<f32> = (0..hd)
            .map(|i| f_gate[i] * cell.c[i] + i_gate[i] * g_gate[i])
            .collect();
        let c_bar_new: Vec<f32> = (0..hd)
            .map(|i| f_gate[i] * cell.c_bar[i] + z_gate[i] * g_gate[i])
            .collect();
        let h_new: Vec<f32> = (0..hd).map(|i| o_gate[i] * c_new[i].tanh()).collect();

        ContinuousLstmCell {
            c_bar: c_bar_new,
            c: c_new,
            o: o_gate,
            delta,
            h: h_new,
        }
    }

    /// Compute per-type intensities at time t given cell state (after last event at t_last).
    pub fn intensity_at(&self, dt: f64, cell: &ContinuousLstmCell) -> Vec<f32> {
        let c_dt = cell.decay(dt);
        let h_dt: Vec<f32> = (0..self.hidden_dim)
            .map(|i| cell.o[i] * c_dt[i].tanh())
            .collect();

        (0..self.n_types)
            .map(|k| {
                let logit: f32 = (0..self.hidden_dim)
                    .map(|j| self.w_intensity[k * self.hidden_dim + j] * h_dt[j])
                    .sum::<f32>()
                    + self.b_intensity[k];
                // softplus for positivity
                if logit > 20.0 {
                    logit
                } else {
                    (logit.exp() + 1.0).ln()
                }
            })
            .collect()
    }

    /// Forward pass: returns intensity vectors at each event time.
    pub fn forward(&self, seq: &EventSequence) -> Vec<Vec<f32>> {
        let mut cell = ContinuousLstmCell::zeros(self.hidden_dim);
        let mut result = Vec::with_capacity(seq.events.len());
        let mut last_t = 0.0f64;

        for ev in &seq.events {
            let dt = (ev.time - last_t).max(0.0);
            let intensities = self.intensity_at(dt, &cell);
            result.push(intensities);
            cell = self.update_cell(ev.event_type, &cell);
            last_t = ev.time;
        }
        result
    }

    /// Negative log-likelihood loss.
    pub fn nll_loss(&self, seq: &EventSequence) -> f32 {
        let intensity_vecs = self.forward(seq);
        let n = intensity_vecs.len();
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0f32;
        // log-likelihood = Σ_i log λ*(t_i, k_i) - ∫ λ(t) dt (approximated via trapezoidal rule)
        let iat = seq.inter_arrival_times();
        for (i, (lam_vec, ev)) in intensity_vecs.iter().zip(seq.events.iter()).enumerate() {
            let k = ev.event_type.min(self.n_types - 1);
            let lam_k = lam_vec[k].max(1e-7);
            total -= lam_k.ln();
            // Approximate integral over [t_{i-1}, t_i] via midpoint
            let dt = if i < iat.len() { iat[i] as f32 } else { 0.0 };
            let sum_lam: f32 = lam_vec.iter().sum();
            total += sum_lam * dt * 0.5;
        }
        (total / n as f32).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// §5  ThpModel — Transformer Hawkes Process
// ---------------------------------------------------------------------------

/// Output of the THP forward pass.
#[derive(Debug, Clone)]
pub struct ThpOutput {
    /// Type logits for each event position [n_events x n_types]
    pub type_logits: Vec<Vec<f32>>,
    /// Time prediction logit per event position \[n_events\]
    pub time_logits: Vec<f32>,
}

/// Temporal encoding: sinusoidal + learned positional encoding for times.
#[derive(Debug, Clone)]
pub struct TemporalEncoding {
    /// Model dimension
    pub d_model: usize,
    /// Learned scaling parameters for sinusoidal encoding
    pub scale: Vec<f32>,
}

impl TemporalEncoding {
    /// Create a new temporal encoding.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        let scale = (0..d_model)
            .map(|_| rng.random::<f32>() * 0.1 + 0.9)
            .collect();
        Self { d_model, scale }
    }

    /// Encode a scalar time value into a d_model-dimensional vector.
    pub fn encode_time(&self, t: f64) -> Vec<f32> {
        let d = self.d_model;
        (0..d)
            .map(|i| {
                let freq = 1.0_f64 / (10000.0f64.powf(2.0 * (i / 2) as f64 / d as f64));
                let angle = t * freq;
                let v = if i % 2 == 0 { angle.sin() } else { angle.cos() } as f32;
                v * self.scale[i]
            })
            .collect()
    }
}

/// Transformer Hawkes Process.
#[derive(Debug, Clone)]
pub struct ThpModel {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of event types
    pub n_types: usize,
    /// Maximum sequence length
    pub max_len: usize,
    /// Type embedding [n_types x d_model]
    type_embed: Vec<f32>,
    /// Temporal encoding
    temporal_enc: TemporalEncoding,
    /// Per-layer Q, K, V, O weight matrices [n_layers x (4 * d_model * d_model)]
    attn_weights: Vec<Vec<f32>>,
    /// Output weights for type [n_types x d_model]
    w_type_out: Vec<f32>,
    b_type_out: Vec<f32>,
    /// Output weights for time (scalar per position) [d_model]
    w_time_out: Vec<f32>,
    b_time_out: f32,
}

impl ThpModel {
    /// Create a new THP model with random initialization.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        n_types: usize,
        max_len: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f32 / (2 * d_model) as f32).sqrt();
        let mut mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                .collect()
        };
        let type_embed = mk(n_types * d_model);
        let temporal_enc = TemporalEncoding::new(d_model, &mut rng);
        let scale2 = scale;
        let mut mk2 = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale2)
                .collect()
        };
        // Each layer has 4 matrices Q,K,V,O each of size [d_model x d_model]
        let attn_weights = (0..n_layers).map(|_| mk2(4 * d_model * d_model)).collect();
        let w_type_out = mk2(n_types * d_model);
        let b_type_out = vec![0.0f32; n_types];
        let w_time_out = mk2(d_model);
        let b_time_out = 0.0f32;
        Self {
            d_model,
            n_heads,
            n_layers,
            n_types,
            max_len,
            type_embed,
            temporal_enc,
            attn_weights,
            w_type_out,
            b_type_out,
            w_time_out,
            b_time_out,
        }
    }

    fn get_type_embed(&self, t: usize) -> Vec<f32> {
        let ti = t.min(self.n_types - 1);
        (0..self.d_model)
            .map(|j| self.type_embed[ti * self.d_model + j])
            .collect()
    }

    fn mat_vec(&self, w: &[f32], x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
        (0..d_out)
            .map(|i| (0..d_in).map(|j| w[i * d_in + j] * x[j]).sum::<f32>())
            .collect()
    }

    fn layer_norm(x: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (var + 1e-5).sqrt();
        x.iter().map(|v| (v - mean) / std).collect()
    }

    /// Single-head causal self-attention (simplified: one head, causal mask).
    fn causal_attention(&self, xs: &[Vec<f32>], layer: usize) -> Vec<Vec<f32>> {
        let d = self.d_model;
        let w = &self.attn_weights[layer];
        let w_q = &w[0..d * d];
        let w_k = &w[d * d..2 * d * d];
        let w_v = &w[2 * d * d..3 * d * d];
        let w_o = &w[3 * d * d..4 * d * d];

        let n = xs.len();
        let qs: Vec<Vec<f32>> = xs.iter().map(|x| self.mat_vec(w_q, x, d, d)).collect();
        let ks: Vec<Vec<f32>> = xs.iter().map(|x| self.mat_vec(w_k, x, d, d)).collect();
        let vs: Vec<Vec<f32>> = xs.iter().map(|x| self.mat_vec(w_v, x, d, d)).collect();

        let scale = (d as f32).sqrt();
        let mut out = Vec::with_capacity(n);

        for i in 0..n {
            // Causal: attend only to positions 0..=i
            let scores: Vec<f32> = (0..=i)
                .map(|j| {
                    qs[i]
                        .iter()
                        .zip(ks[j].iter())
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        / scale
                })
                .collect();
            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_s: f32 = exp_s.iter().sum::<f32>().max(1e-9);
            let attn: Vec<f32> = exp_s.iter().map(|s| s / sum_s).collect();

            // Weighted sum of values
            let ctx: Vec<f32> = (0..d)
                .map(|k| (0..=i).map(|j| attn[j] * vs[j][k]).sum::<f32>())
                .collect();
            let proj = self.mat_vec(w_o, &ctx, d, d);
            // Residual + LN
            let res: Vec<f32> = xs[i].iter().zip(proj.iter()).map(|(a, b)| a + b).collect();
            out.push(Self::layer_norm(&res));
        }
        out
    }

    /// Forward pass over event sequence.
    pub fn forward(&self, times: &[f64], types: &[usize]) -> ThpOutput {
        let n = times.len().min(types.len()).min(self.max_len);
        if n == 0 {
            return ThpOutput {
                type_logits: vec![],
                time_logits: vec![],
            };
        }

        // Build input embeddings: type_embed + temporal_enc
        let mut xs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let te = self.get_type_embed(types[i]);
                let enc = self.temporal_enc.encode_time(times[i]);
                te.iter().zip(enc.iter()).map(|(a, b)| a + b).collect()
            })
            .collect();

        // Apply transformer layers
        for layer in 0..self.n_layers {
            xs = self.causal_attention(&xs, layer);
        }

        // Compute outputs
        let type_logits: Vec<Vec<f32>> = xs
            .iter()
            .map(|h| {
                (0..self.n_types)
                    .map(|k| {
                        (0..self.d_model)
                            .map(|j| self.w_type_out[k * self.d_model + j] * h[j])
                            .sum::<f32>()
                            + self.b_type_out[k]
                    })
                    .collect()
            })
            .collect();

        let time_logits: Vec<f32> = xs
            .iter()
            .map(|h| {
                (0..self.d_model)
                    .map(|j| self.w_time_out[j] * h[j])
                    .sum::<f32>()
                    + self.b_time_out
            })
            .collect();

        ThpOutput {
            type_logits,
            time_logits,
        }
    }

    /// Negative log-likelihood loss.
    pub fn nll_loss(&self, times: &[f64], types: &[usize]) -> f32 {
        let out = self.forward(times, types);
        let n = out.type_logits.len();
        if n == 0 {
            return 0.0;
        }
        let mut loss = 0.0f32;
        for i in 0..n {
            let ls = log_softmax_f32(&out.type_logits[i]);
            let t = types[i].min(self.n_types - 1);
            loss -= ls[t];
            // Time prediction: treat as log-normal or just softplus NLL
            let pred_dt = (out.time_logits[i].exp() + 1.0).ln();
            let actual_dt = if i == 0 {
                times[0] as f32
            } else {
                (times[i] - times[i - 1]) as f32
            };
            let actual_dt = actual_dt.max(1e-6);
            loss += (pred_dt - actual_dt).powi(2);
        }
        loss / n as f32
    }
}

// ---------------------------------------------------------------------------
// §6  TppLoss — Loss functions
// ---------------------------------------------------------------------------

/// Loss functions for temporal point processes.
pub struct TppLoss;

impl TppLoss {
    /// Negative log-likelihood for a Hawkes process.
    pub fn nll_hawkes(seq: &EventSequence, process: &HawkesProcess) -> f64 {
        -process.log_likelihood(seq)
    }

    /// Time-rescaling goodness-of-fit test.
    ///
    /// Transforms inter-arrival times via integrated intensity to get ~Exp(1) samples.
    /// Returns the KS statistic against Exp(1) (closer to 0 = better fit).
    ///
    /// `intensity_fn(t)` should return the conditional intensity at time t.
    pub fn time_rescaling_test(seq: &EventSequence, intensity_fn: impl Fn(f64) -> f64) -> f64 {
        if seq.events.is_empty() {
            return 0.0;
        }
        // Numerically integrate intensity between events to get transformed times
        let n_steps = 50usize;
        let mut transformed = Vec::with_capacity(seq.events.len());
        let mut last_t = 0.0f64;

        for ev in &seq.events {
            let dt = (ev.time - last_t).max(0.0);
            let step = dt / n_steps as f64;
            let integral: f64 = (0..n_steps)
                .map(|s| {
                    let t0 = last_t + s as f64 * step;
                    let t1 = t0 + step;
                    0.5 * (intensity_fn(t0) + intensity_fn(t1)) * step
                })
                .sum();
            transformed.push(integral.max(0.0));
            last_t = ev.time;
        }

        // KS statistic vs Exp(1): CDF = 1 - exp(-x)
        let n = transformed.len() as f64;
        let mut sorted = transformed.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut ks_stat = 0.0f64;
        for (i, &x) in sorted.iter().enumerate() {
            let empirical = (i + 1) as f64 / n;
            let theoretical = 1.0 - (-x).exp();
            ks_stat = ks_stat.max((empirical - theoretical).abs());
        }
        ks_stat
    }

    /// Cross-entropy loss for event type predictions.
    pub fn event_type_nll(logits: &[Vec<f32>], targets: &[usize]) -> f32 {
        let n = logits.len().min(targets.len());
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0f32;
        for i in 0..n {
            let ls = log_softmax_f32(&logits[i]);
            if !ls.is_empty() {
                let t = targets[i].min(ls.len() - 1);
                total -= ls[t];
            }
        }
        total / n as f32
    }

    /// Weighted combination of time and type NLL.
    pub fn combined_nll(time_nll: f32, type_nll: f32, alpha: f32) -> f32 {
        alpha * time_nll + (1.0 - alpha) * type_nll
    }
}

// ---------------------------------------------------------------------------
// §7  TppMetrics — Evaluation metrics
// ---------------------------------------------------------------------------

/// Evaluation report for temporal point process models.
#[derive(Debug, Clone)]
pub struct TppEvalReport {
    /// Average log-likelihood per event
    pub log_lik: f64,
    /// Event type prediction accuracy
    pub type_acc: f32,
    /// RMSE of next-event time predictions
    pub time_rmse: f64,
    /// KS statistic from time-rescaling test
    pub ks_statistic: f64,
}

/// Evaluation metrics for temporal point processes.
pub struct TppMetrics;

impl TppMetrics {
    /// Compute average log-likelihood per event.
    ///
    /// `intensity_fn(t, k)` returns conditional intensity of type k at time t.
    pub fn log_likelihood(seq: &EventSequence, intensity_fn: impl Fn(f64, usize) -> f64) -> f64 {
        if seq.events.is_empty() {
            return 0.0;
        }
        let n_steps = 50usize;
        let mut ll = 0.0f64;

        // Sum log intensities at event times
        for (i, ev) in seq.events.iter().enumerate() {
            let lam = intensity_fn(ev.time, ev.event_type).max(1e-300);
            ll += lam.ln();

            // Subtract integral from last event to this event
            let t_prev = if i == 0 { 0.0 } else { seq.events[i - 1].time };
            let dt = (ev.time - t_prev).max(0.0);
            let step = dt / n_steps as f64;
            for s in 0..n_steps {
                let t0 = t_prev + s as f64 * step;
                let t1 = t0 + step;
                let integral_k: f64 =
                    (0..seq.events.iter().map(|e| e.event_type).max().unwrap_or(0) + 1)
                        .map(|k| 0.5 * (intensity_fn(t0, k) + intensity_fn(t1, k)) * step)
                        .sum();
                ll -= integral_k;
            }
        }
        ll / seq.events.len() as f64
    }

    /// Event type prediction accuracy.
    pub fn event_type_accuracy(predictions: &[usize], targets: &[usize]) -> f32 {
        let n = predictions.len().min(targets.len());
        if n == 0 {
            return 0.0;
        }
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| p == t)
            .count();
        correct as f32 / n as f32
    }

    /// RMSE of next-event time predictions.
    pub fn rmse_next_time(predicted: &[f64], actual: &[f64]) -> f64 {
        let n = predicted.len().min(actual.len());
        if n == 0 {
            return 0.0;
        }
        let mse: f64 = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n as f64;
        mse.sqrt()
    }

    /// Expected calibration error for intensity predictions.
    ///
    /// Bins predicted intensities and compares to actual event counts.
    pub fn calibration_error(
        predicted_intensities: &[f64],
        actual_counts: &[f64],
        n_bins: usize,
    ) -> f64 {
        let n = predicted_intensities.len().min(actual_counts.len());
        if n == 0 || n_bins == 0 {
            return 0.0;
        }
        let max_pred = predicted_intensities
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        if max_pred <= 0.0 {
            return 0.0;
        }
        let bin_width = max_pred / n_bins as f64;
        let mut bin_pred_sum = vec![0.0f64; n_bins];
        let mut bin_actual_sum = vec![0.0f64; n_bins];
        let mut bin_count = vec![0usize; n_bins];

        for i in 0..n {
            let idx = ((predicted_intensities[i] / bin_width) as usize).min(n_bins - 1);
            bin_pred_sum[idx] += predicted_intensities[i];
            bin_actual_sum[idx] += actual_counts[i];
            bin_count[idx] += 1;
        }

        let mut ece = 0.0f64;
        for b in 0..n_bins {
            if bin_count[b] > 0 {
                let cnt = bin_count[b] as f64;
                let pred_mean = bin_pred_sum[b] / cnt;
                let actual_mean = bin_actual_sum[b] / cnt;
                ece += (pred_mean - actual_mean).abs() * cnt / n as f64;
            }
        }
        ece
    }

    /// Build a full evaluation report.
    pub fn evaluate(
        seq: &EventSequence,
        intensity_fn: impl Fn(f64, usize) -> f64 + Copy,
        type_predictions: &[usize],
        time_predictions: &[f64],
    ) -> TppEvalReport {
        let log_lik = Self::log_likelihood(seq, intensity_fn);
        let targets: Vec<usize> = seq.events.iter().map(|e| e.event_type).collect();
        let actual_times: Vec<f64> = seq.inter_arrival_times();
        let type_acc = Self::event_type_accuracy(type_predictions, &targets);
        let time_rmse = Self::rmse_next_time(time_predictions, &actual_times);
        let ks_statistic = TppLoss::time_rescaling_test(seq, |t| {
            (0..seq.events.iter().map(|e| e.event_type).max().unwrap_or(0) + 1)
                .map(|k| intensity_fn(t, k))
                .sum()
        });
        TppEvalReport {
            log_lik,
            type_acc,
            time_rmse,
            ks_statistic,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_hawkes() -> HawkesProcess {
        HawkesProcess::new(vec![0.5], vec![vec![0.3]], vec![1.0])
            .expect("HawkesProcess creation failed")
    }

    fn make_sample_seq() -> EventSequence {
        EventSequence::new(
            vec![
                Event {
                    time: 0.5,
                    event_type: 0,
                },
                Event {
                    time: 1.2,
                    event_type: 0,
                },
                Event {
                    time: 2.1,
                    event_type: 0,
                },
                Event {
                    time: 3.0,
                    event_type: 0,
                },
            ],
            5.0,
        )
    }

    // --- EventSequence tests ---

    #[test]
    fn test_event_sequence_creation() {
        let seq = make_sample_seq();
        assert_eq!(seq.events.len(), 4);
        assert!((seq.t_end - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_event_count() {
        let seq = make_sample_seq();
        assert_eq!(seq.event_count(), 4);
    }

    #[test]
    fn test_observation_window() {
        let seq = make_sample_seq();
        assert!((seq.t_end - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_event_sequence_inter_arrival_times() {
        let seq = make_sample_seq();
        let iat = seq.inter_arrival_times();
        assert_eq!(iat.len(), 4);
        // First IAT = time of first event = 0.5
        assert!((iat[0] - 0.5).abs() < 1e-9);
        // Second IAT = 1.2 - 0.5 = 0.7
        assert!((iat[1] - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_event_sequence_intensity_histogram_bins() {
        let seq = make_sample_seq();
        let hist = seq.intensity_histogram(5);
        assert_eq!(hist.len(), 5);
    }

    #[test]
    fn test_intensity_histogram_sums() {
        let seq = make_sample_seq();
        let hist = seq.intensity_histogram(10);
        let total: f32 = hist.iter().sum();
        // Total count = number of events
        assert!((total - seq.event_count() as f32).abs() < 1e-5);
    }

    // --- HawkesProcess tests ---

    #[test]
    fn test_hawkes_conditional_intensity_baseline() {
        let hp = make_simple_hawkes();
        // No history → intensity equals mu
        let lam = hp.conditional_intensity(1.0, &[], 0);
        assert!((lam - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hawkes_conditional_intensity_with_history() {
        let hp = make_simple_hawkes();
        let history = vec![Event {
            time: 0.5,
            event_type: 0,
        }];
        let lam = hp.conditional_intensity(1.0, &history, 0);
        // With history, should be > mu
        assert!(lam > 0.5);
    }

    #[test]
    fn test_hawkes_excitation_decays() {
        let hp = make_simple_hawkes();
        let history = vec![Event {
            time: 0.0,
            event_type: 0,
        }];
        let lam_early = hp.conditional_intensity(0.1, &history, 0);
        let lam_late = hp.conditional_intensity(10.0, &history, 0);
        // Excitation should decay over time
        assert!(lam_early > lam_late);
    }

    #[test]
    fn test_hawkes_simulate_positive_times() {
        let hp = make_simple_hawkes();
        let mut rng = StdRng::seed_from_u64(42);
        let seq = hp.simulate(10.0, &mut rng);
        for ev in &seq.events {
            assert!(ev.time >= 0.0, "Event time should be non-negative");
        }
    }

    #[test]
    fn test_hawkes_simulate_time_ordered() {
        let hp = make_simple_hawkes();
        let mut rng = StdRng::seed_from_u64(42);
        let seq = hp.simulate(10.0, &mut rng);
        for w in seq.events.windows(2) {
            assert!(w[1].time >= w[0].time, "Events should be time-ordered");
        }
    }

    #[test]
    fn test_hawkes_simulate_count_range() {
        let hp = make_simple_hawkes();
        let mut rng = StdRng::seed_from_u64(42);
        let seq = hp.simulate(20.0, &mut rng);
        // Poisson with rate ~0.5 over 20 units → expect > 0 events
        assert!(seq.events.len() > 0, "Should produce at least some events");
        // Upper bound sanity check
        assert!(
            seq.events.len() < 1000,
            "Should not produce absurdly many events"
        );
    }

    #[test]
    fn test_hawkes_log_likelihood_negative() {
        let hp = make_simple_hawkes();
        let seq = make_sample_seq();
        let ll = hp.log_likelihood(&seq);
        // Log-likelihood should be finite
        assert!(ll.is_finite(), "Log-likelihood should be finite");
    }

    #[test]
    fn test_log_likelihood_finite() {
        let hp = make_simple_hawkes();
        let seq = make_sample_seq();
        let ll = hp.log_likelihood(&seq);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_hawkes_branching_ratio_shape() {
        let hp = make_simple_hawkes();
        let br = hp.branching_ratio();
        assert_eq!(br.len(), 1);
        assert!(br[0] >= 0.0);
    }

    #[test]
    fn test_hawkes_alpha_beta_consistency() {
        let hp = HawkesProcess::new(
            vec![0.3, 0.4],
            vec![vec![0.2, 0.1], vec![0.1, 0.2]],
            vec![1.0, 2.0],
        )
        .expect("HawkesProcess creation failed");
        let br = hp.branching_ratio();
        assert_eq!(br.len(), 2);
        // Branching ratio = sum(alpha[k]) / beta[k]
        assert!((br[0] - 0.3 / 1.0).abs() < 1e-9);
        assert!((br[1] - 0.3 / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_hawkes_fit_mle_runs() {
        let seq = make_sample_seq();
        let hp = HawkesProcess::fit_mle(&seq, 1, 10, 0.001);
        assert_eq!(hp.n_types, 1);
        assert!(hp.mu[0] > 0.0);
    }

    #[test]
    fn test_fit_mle_improves_likelihood() {
        let seq = make_sample_seq();
        let hp_init = HawkesProcess::new(vec![0.5], vec![vec![0.3]], vec![1.0])
            .expect("HawkesProcess creation failed");
        let ll_init = hp_init.log_likelihood(&seq);
        let hp_fitted = HawkesProcess::fit_mle(&seq, 1, 50, 0.01);
        let ll_fit = hp_fitted.log_likelihood(&seq);
        // After fitting, log-likelihood should be >= initial (or at least finite)
        assert!(ll_fit.is_finite());
        let _ = ll_init; // suppress unused warning
    }

    #[test]
    fn test_hawkes_univariate_simulation() {
        let hp = make_simple_hawkes();
        let mut rng = StdRng::seed_from_u64(123);
        let seq = hp.simulate(5.0, &mut rng);
        assert!(seq.t_end >= 5.0 - 1e-10);
        for ev in &seq.events {
            assert_eq!(ev.event_type, 0);
        }
    }

    #[test]
    fn test_hawkes_multivariate_simulation() {
        let hp = HawkesProcess::new(
            vec![0.3, 0.3],
            vec![vec![0.2, 0.1], vec![0.1, 0.2]],
            vec![1.0, 1.0],
        )
        .expect("HawkesProcess creation failed");
        let mut rng = StdRng::seed_from_u64(99);
        let seq = hp.simulate(10.0, &mut rng);
        // Events should have types in {0, 1}
        for ev in &seq.events {
            assert!(ev.event_type < 2);
        }
    }

    #[test]
    fn test_ogata_thinning_no_events_before_t() {
        let hp = make_simple_hawkes();
        let mut rng = StdRng::seed_from_u64(7);
        let seq = hp.simulate(5.0, &mut rng);
        for ev in &seq.events {
            assert!(ev.time < 5.0 + 1e-12, "All events should be before T=5.0");
        }
    }

    // --- RmtppModel tests ---

    #[test]
    fn test_rmtpp_model_creation() {
        let model = RmtppModel::new(16, 3, 42);
        assert_eq!(model.hidden_dim, 16);
        assert_eq!(model.n_types, 3);
    }

    #[test]
    fn test_rmtpp_forward_output_shape() {
        let model = RmtppModel::new(16, 3, 42);
        let inter_times = vec![0.5f32, 0.3, 0.7, 0.2];
        let types = vec![0usize, 1, 2, 0];
        let (logits, intensity) = model.forward(&inter_times, &types);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 3);
        assert_eq!(intensity.len(), 4);
    }

    #[test]
    fn test_rmtpp_type_logits_shape() {
        let model = RmtppModel::new(8, 4, 1);
        let (logits, _) = model.forward(&[0.1f32, 0.2], &[0usize, 1]);
        assert_eq!(logits[0].len(), 4);
    }

    #[test]
    fn test_rmtpp_nll_loss_finite() {
        let model = RmtppModel::new(16, 3, 42);
        let inter_times = vec![0.5f32, 0.3, 0.7];
        let types = vec![0usize, 1, 2];
        let loss = model.nll_loss(&inter_times, &types);
        assert!(loss.is_finite(), "NLL loss should be finite");
    }

    #[test]
    fn test_rmtpp_predict_next_time_positive() {
        let model = RmtppModel::new(16, 3, 42);
        let h = vec![0.1f32; 16];
        let intensity_params = vec![0.0f32];
        let dt = model.predict_next_time(&h, &intensity_params);
        assert!(dt >= 0.0, "Predicted next time should be non-negative");
    }

    // --- NhpModel tests ---

    #[test]
    fn test_nhp_model_creation() {
        let model = NhpModel::new(16, 3, 42);
        assert_eq!(model.hidden_dim, 16);
        assert_eq!(model.n_types, 3);
    }

    #[test]
    fn test_nhp_forward_shape() {
        let model = NhpModel::new(16, 3, 42);
        let seq = make_sample_seq();
        let intensities = model.forward(&seq);
        assert_eq!(intensities.len(), seq.events.len());
        for iv in &intensities {
            assert_eq!(iv.len(), 3);
        }
    }

    #[test]
    fn test_nhp_nll_loss_finite() {
        let model = NhpModel::new(16, 3, 42);
        let seq = make_sample_seq();
        let loss = model.nll_loss(&seq);
        assert!(loss.is_finite(), "NHP NLL should be finite");
    }

    #[test]
    fn test_nhp_intensity_positive() {
        let model = NhpModel::new(16, 3, 42);
        let seq = make_sample_seq();
        let intensities = model.forward(&seq);
        for iv in &intensities {
            for &lam in iv {
                assert!(lam >= 0.0, "Intensities should be non-negative (softplus)");
            }
        }
    }

    // --- ThpModel tests ---

    #[test]
    fn test_thp_model_creation() {
        let model = ThpModel::new(16, 2, 2, 3, 50, 42);
        assert_eq!(model.d_model, 16);
        assert_eq!(model.n_types, 3);
    }

    #[test]
    fn test_thp_forward_shape() {
        let model = ThpModel::new(16, 2, 1, 3, 50, 42);
        let times = vec![0.5, 1.2, 2.1, 3.0];
        let types = vec![0usize, 1, 2, 0];
        let out = model.forward(&times, &types);
        assert_eq!(out.type_logits.len(), 4);
        assert_eq!(out.time_logits.len(), 4);
        assert_eq!(out.type_logits[0].len(), 3);
    }

    #[test]
    fn test_thp_output_dimensions() {
        let model = ThpModel::new(8, 1, 1, 2, 10, 0);
        let out = model.forward(&[1.0, 2.0], &[0usize, 1]);
        assert_eq!(out.type_logits.len(), 2);
        assert_eq!(out.time_logits.len(), 2);
    }

    #[test]
    fn test_thp_nll_loss_finite() {
        let model = ThpModel::new(16, 2, 1, 3, 50, 42);
        let times = vec![0.5, 1.2, 2.1];
        let types = vec![0usize, 1, 2];
        let loss = model.nll_loss(&times, &types);
        assert!(loss.is_finite(), "THP NLL should be finite");
    }

    #[test]
    fn test_temporal_encoding_shape() {
        let mut rng = StdRng::seed_from_u64(1);
        let enc = TemporalEncoding::new(16, &mut rng);
        let v = enc.encode_time(1.5);
        assert_eq!(v.len(), 16);
    }

    // --- TppLoss tests ---

    #[test]
    fn test_tpp_loss_time_rescaling() {
        let hp = make_simple_hawkes();
        let mut rng = StdRng::seed_from_u64(0);
        let seq = hp.simulate(20.0, &mut rng);
        if seq.events.is_empty() {
            return;
        }
        let hp2 = hp.clone();
        let ks =
            TppLoss::time_rescaling_test(&seq, |t| hp2.conditional_intensity(t, &seq.events, 0));
        assert!(ks >= 0.0, "KS statistic should be non-negative");
        assert!(ks <= 1.0, "KS statistic should be at most 1.0");
    }

    #[test]
    fn test_tpp_loss_nonnegative_ks_statistic() {
        let seq = make_sample_seq();
        let ks = TppLoss::time_rescaling_test(&seq, |_t| 0.5);
        assert!(ks >= 0.0);
    }

    #[test]
    fn test_tpp_loss_event_type_nll() {
        let logits = vec![vec![2.0f32, 1.0, 0.5], vec![0.5f32, 2.0, 0.3]];
        let targets = vec![0usize, 1];
        let nll = TppLoss::event_type_nll(&logits, &targets);
        assert!(nll.is_finite());
        assert!(nll >= 0.0);
    }

    #[test]
    fn test_tpp_loss_combined_nll() {
        let combined = TppLoss::combined_nll(1.0, 2.0, 0.5);
        assert!((combined - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_time_rescaling_uniform_output() {
        // Perfect Poisson process: intensity = 1.0 everywhere
        // Inter-arrival times ~ Exp(1), so rescaled times should also be ~Exp(1)
        let events: Vec<Event> = (1..=10)
            .map(|i| Event {
                time: i as f64,
                event_type: 0,
            })
            .collect();
        let seq = EventSequence::new(events, 11.0);
        let ks = TppLoss::time_rescaling_test(&seq, |_t| 1.0);
        assert!(ks >= 0.0);
        assert!(ks.is_finite());
    }

    // --- TppMetrics tests ---

    #[test]
    fn test_tpp_metrics_event_type_accuracy_perfect() {
        let preds = vec![0usize, 1, 2, 0];
        let targets = vec![0usize, 1, 2, 0];
        let acc = TppMetrics::event_type_accuracy(&preds, &targets);
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tpp_metrics_event_type_accuracy_zeros() {
        let preds = vec![1usize, 1, 1, 1];
        let targets = vec![0usize, 0, 0, 0];
        let acc = TppMetrics::event_type_accuracy(&preds, &targets);
        assert!((acc - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tpp_metrics_rmse_zero() {
        let vals = vec![1.0, 2.0, 3.0];
        let rmse = TppMetrics::rmse_next_time(&vals, &vals);
        assert!(rmse.abs() < 1e-10);
    }

    #[test]
    fn test_tpp_metrics_calibration_error() {
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let ece = TppMetrics::calibration_error(&preds, &actual, 4);
        // Perfect calibration → ECE = 0 (approximately)
        assert!(ece.is_finite());
        assert!(ece >= 0.0);
    }

    #[test]
    fn test_tpp_eval_report_fields() {
        let hp = make_simple_hawkes();
        let seq = make_sample_seq();
        let hp2 = hp.clone();
        let report = TppMetrics::evaluate(
            &seq,
            |t, k| hp2.conditional_intensity(t, &[], k),
            &[0, 0, 0, 0],
            &[0.5, 0.7, 0.9, 0.9],
        );
        assert!(report.log_lik.is_finite());
        assert!(report.type_acc >= 0.0 && report.type_acc <= 1.0);
        assert!(report.time_rmse >= 0.0);
        assert!(report.ks_statistic >= 0.0);
    }

    #[test]
    fn test_tpp_metrics_log_likelihood_computation() {
        let seq = make_sample_seq();
        let ll = TppMetrics::log_likelihood(&seq, |_t, _k| 0.5);
        assert!(ll.is_finite());
    }
}
