//! Neuromorphic Computing and Spiking Neural Networks (SNN).
//!
//! Neuron models: [`LifNeuron`] (LIF), [`AdexNeuron`] (AdEx).
//! Spike encoding: [`SpikeEncoder`] (rate/temporal/population), [`PopulationEncoder`].
//! Learning: [`StdpSynapse`] (pair-based STDP).
//! Surrogate gradients: [`FastSigmoid`], [`PiecewiseLinear`], [`SuperSpike`].
//! Reservoir computing: [`LiquidStateMachine`] (LSM / Echo State Network).
//! Metrics: [`SnnMetrics`], [`compute_snn_metrics`], [`compute_membrane_stats`].
//!
//! ```rust,ignore
//! use tenflowers_neural::neuromorphic::{LifConfig, LifNeuron};
//! let cfg = LifConfig::default();
//! let mut neuron = LifNeuron::new(&cfg);
//! let fired = neuron.step(50.0, &cfg);
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ── Internal helpers ──

/// Box-Muller normal sample (f64 version).
#[inline]
fn sample_normal_f64(rng: &mut impl Rng) -> f64 {
    let u1: f64 = (rng.random::<f64>()).max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate `n` i.i.d. N(0,1) samples.
fn sample_normal_vec_f64(n: usize, rng: &mut impl Rng) -> Vec<f64> {
    (0..n).map(|_| sample_normal_f64(rng)).collect()
}

// ── §1 LIF Neuron ──

/// Configuration parameters for the Leaky Integrate-and-Fire neuron model.
///
/// All time constants are in milliseconds; potentials are in millivolts.
#[derive(Debug, Clone)]
pub struct LifConfig {
    /// Membrane time constant (ms).
    pub tau_mem: f64,
    /// Synaptic time constant (ms).
    pub tau_syn: f64,
    /// Resting potential (mV).
    pub v_rest: f64,
    /// Threshold potential (mV).
    pub v_thresh: f64,
    /// Reset potential after a spike (mV).
    pub v_reset: f64,
    /// Integration time step (ms).
    pub dt: f64,
    /// Refractory period in discrete time steps.
    pub refractory_steps: usize,
}

impl Default for LifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 10.0,
            tau_syn: 5.0,
            v_rest: -70.0,
            v_thresh: -55.0,
            v_reset: -80.0,
            dt: 1.0,
            refractory_steps: 2,
        }
    }
}

/// A single Leaky Integrate-and-Fire neuron with exponential synaptic current.
///
/// Dynamics (Euler integration):
/// ```text
/// dV/dt  = (-(V - V_rest) + I_syn) / tau_mem
/// dI/dt  = -I_syn / tau_syn
/// ```
/// On spike: V is clamped to V_reset and the refractory counter is set.
#[derive(Debug, Clone)]
pub struct LifNeuron {
    /// Current membrane potential (mV).
    pub v_mem: f64,
    /// Synaptic current.
    pub i_syn: f64,
    /// Remaining refractory time steps.
    pub refractory: usize,
    /// Record of spike emission times (in discrete steps or ms depending on use).
    pub spike_times: Vec<f64>,
}

impl LifNeuron {
    /// Create a new LIF neuron initialised to resting potential.
    pub fn new(config: &LifConfig) -> Self {
        Self {
            v_mem: config.v_rest,
            i_syn: 0.0,
            refractory: 0,
            spike_times: Vec::new(),
        }
    }

    /// Advance by one time step `dt`, injecting external current `i_ext`.
    ///
    /// Returns `true` if the neuron fired a spike this step.
    pub fn step(&mut self, i_ext: f64, config: &LifConfig) -> bool {
        // Update synaptic current first (exponential decay + external input).
        self.i_syn += i_ext;
        self.i_syn -= (config.dt / config.tau_syn) * self.i_syn;

        if self.refractory > 0 {
            self.refractory -= 1;
            self.v_mem = config.v_reset;
            return false;
        }

        // Euler step for membrane potential.
        let dv = config.dt / config.tau_mem * (-(self.v_mem - config.v_rest) + self.i_syn);
        self.v_mem += dv;

        if self.v_mem >= config.v_thresh {
            self.v_mem = config.v_reset;
            self.refractory = config.refractory_steps;
            true
        } else {
            false
        }
    }

    /// Reset state to resting conditions (e.g., between sequences).
    pub fn reset(&mut self, config: &LifConfig) {
        self.v_mem = config.v_rest;
        self.i_syn = 0.0;
        self.refractory = 0;
    }
}

// ── §2 Spike Encoding ──

/// Strategy used by [`SpikeEncoder`] to represent a scalar value as spike trains.
#[derive(Debug, Clone, PartialEq)]
pub enum SpikeEncoding {
    /// Poisson rate coding: each neuron fires with probability proportional to
    /// the encoded value (scaled to `[0, max_rate]` Hz × dt).
    Rate,
    /// Time-to-first-spike: neuron `k` fires at time proportional to
    /// `1 - value` scaled over the time window.
    Temporal,
    /// Population coding: Gaussian tuning curves over the value range.
    Population,
}

/// Encodes scalar values in `[0, 1]` as multi-neuron spike trains.
///
/// The output shape is `[time_steps][n_neurons]`.
#[derive(Debug, Clone)]
pub struct SpikeEncoder {
    /// Encoding strategy.
    pub encoding: SpikeEncoding,
    /// Number of neurons in the population.
    pub n_neurons: usize,
    /// Number of discrete time steps.
    pub time_steps: usize,
    /// Maximum firing rate in Hz (used for rate coding; assumes dt = 1 ms).
    pub max_rate: f64,
    // Deterministic seed for reproducible encoding.
    seed: u64,
}

impl SpikeEncoder {
    /// Create a new `SpikeEncoder`.
    ///
    /// `value` should be in `[0, 1]`.
    pub fn new(encoding: SpikeEncoding, n_neurons: usize, time_steps: usize) -> Self {
        Self {
            encoding,
            n_neurons,
            time_steps,
            max_rate: 100.0,
            seed: 0xcafe_f64d_u64,
        }
    }

    /// Set the maximum firing rate (Hz).
    pub fn with_max_rate(mut self, max_rate: f64) -> Self {
        self.max_rate = max_rate;
        self
    }

    /// Encode a value in `[0, 1]` into a spike train of shape `[time_steps][n_neurons]`.
    pub fn encode(&self, value: f64) -> Vec<Vec<bool>> {
        let value = value.clamp(0.0, 1.0);
        match self.encoding {
            SpikeEncoding::Rate => self.encode_rate(value),
            SpikeEncoding::Temporal => self.encode_temporal(value),
            SpikeEncoding::Population => self.encode_population(value),
        }
    }

    fn encode_rate(&self, value: f64) -> Vec<Vec<bool>> {
        // Probability of firing per time step: value * max_rate * 1e-3 (Hz × ms).
        let p_fire = (value * self.max_rate * 1e-3).clamp(0.0, 1.0);
        let mut rng = StdRng::seed_from_u64(self.seed);
        (0..self.time_steps)
            .map(|_| {
                (0..self.n_neurons)
                    .map(|_| rng.random::<f64>() < p_fire)
                    .collect()
            })
            .collect()
    }

    fn encode_temporal(&self, value: f64) -> Vec<Vec<bool>> {
        // Neuron k fires at time step proportional to k / n_neurons mapped
        // over the coding window scaled by (1 - value).
        let mut result = vec![vec![false; self.n_neurons]; self.time_steps];
        for k in 0..self.n_neurons {
            // Preferred value of neuron k in [0,1].
            let pref = if self.n_neurons > 1 {
                k as f64 / (self.n_neurons - 1) as f64
            } else {
                0.5
            };
            // Fire at the time step when the preferred value matches;
            // neurons with pref closer to `value` fire earlier.
            let delay_frac = (1.0 - (1.0 - (value - pref).abs())).clamp(0.0, 1.0);
            let fire_t = (delay_frac * (self.time_steps - 1) as f64).round() as usize;
            let fire_t = fire_t.min(self.time_steps - 1);
            result[fire_t][k] = true;
        }
        result
    }

    fn encode_population(&self, value: f64) -> Vec<Vec<bool>> {
        // Build a PopulationEncoder and threshold the rates to produce spikes.
        let encoder = PopulationEncoder::new(self.n_neurons, 0.0, 1.0);
        let rates = encoder.encode(value); // Each in [0, 1].
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(1));
        (0..self.time_steps)
            .map(|_| {
                rates
                    .iter()
                    .map(|&r| rng.random::<f64>() < r * self.max_rate * 1e-3)
                    .collect()
            })
            .collect()
    }

    /// Reconstruct a scalar value from a spike train produced by \[`encode`\].
    pub fn decode(&self, spikes: &[Vec<bool>]) -> f64 {
        match self.encoding {
            SpikeEncoding::Rate => self.decode_rate(spikes),
            SpikeEncoding::Temporal => self.decode_temporal(spikes),
            SpikeEncoding::Population => self.decode_population(spikes),
        }
    }

    fn decode_rate(&self, spikes: &[Vec<bool>]) -> f64 {
        if spikes.is_empty() || self.n_neurons == 0 || self.time_steps == 0 {
            return 0.0;
        }
        let total_spikes: usize = spikes
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&b| b)
            .count();
        let total_slots = self.time_steps * self.n_neurons;
        // Invert: p_fire = value * max_rate * 1e-3.
        let p_fire = total_spikes as f64 / total_slots as f64;
        (p_fire / (self.max_rate * 1e-3)).clamp(0.0, 1.0)
    }

    fn decode_temporal(&self, spikes: &[Vec<bool>]) -> f64 {
        if spikes.is_empty() || self.n_neurons == 0 {
            return 0.0;
        }
        // Find the first-spike time for each neuron and reconstruct the value.
        // Neurons whose preferred stimulus is close to `value` fire earliest
        // (smallest delay_frac).  We decode by weighted average of preferred
        // values, with weight = 1 / (delay_frac + epsilon).
        let mut weighted_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;
        for k in 0..self.n_neurons {
            let pref = if self.n_neurons > 1 {
                k as f64 / (self.n_neurons - 1) as f64
            } else {
                0.5
            };
            // Find first firing time for neuron k.
            let fire_t = spikes.iter().enumerate().find_map(|(t, row)| {
                if k < row.len() && row[k] {
                    Some(t)
                } else {
                    None
                }
            });
            if let Some(t) = fire_t {
                let delay_frac = t as f64 / (self.time_steps - 1).max(1) as f64;
                let w = 1.0 / (delay_frac + 1e-2);
                weighted_sum += pref * w;
                total_weight += w;
            }
        }
        if total_weight < 1e-12 {
            0.5
        } else {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        }
    }

    fn decode_population(&self, spikes: &[Vec<bool>]) -> f64 {
        if spikes.is_empty() || self.n_neurons == 0 {
            return 0.0;
        }
        // Estimate firing rates per neuron, then use population vector decoding.
        let mut rates = vec![0.0_f64; self.n_neurons];
        for row in spikes {
            for (k, &fired) in row.iter().enumerate() {
                if k < self.n_neurons && fired {
                    rates[k] += 1.0;
                }
            }
        }
        let t = self.time_steps as f64;
        for r in &mut rates {
            *r /= t;
        }
        let encoder = PopulationEncoder::new(self.n_neurons, 0.0, 1.0);
        encoder.decode(&rates)
    }
}

// ── §3 STDP Synapse ──

/// Hyper-parameters for pair-based Spike-Timing-Dependent Plasticity.
#[derive(Debug, Clone)]
pub struct StdpConfig {
    /// Long-term potentiation (LTP) amplitude.
    pub a_plus: f64,
    /// Long-term depression (LTD) amplitude.
    pub a_minus: f64,
    /// LTP trace decay time constant (ms).
    pub tau_plus: f64,
    /// LTD trace decay time constant (ms).
    pub tau_minus: f64,
    /// Minimum synaptic weight.
    pub w_min: f64,
    /// Maximum synaptic weight.
    pub w_max: f64,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.0105,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_min: 0.0,
            w_max: 1.0,
        }
    }
}

/// A single synapse that learns its weight via STDP.
///
/// Pair-based STDP rule:
/// - On a pre-synaptic spike: `trace_pre` is incremented, then `w` is updated
///   by `-A_minus * trace_post` (LTD).
/// - On a post-synaptic spike: `trace_post` is incremented, then `w` is updated
///   by `+A_plus * trace_pre` (LTP).
/// - Every time step: both traces decay exponentially.
#[derive(Debug, Clone)]
pub struct StdpSynapse {
    /// Current synaptic weight.
    pub weight: f64,
    /// Pre-synaptic eligibility trace.
    pub trace_pre: f64,
    /// Post-synaptic eligibility trace.
    pub trace_post: f64,
}

impl StdpSynapse {
    /// Create a new synapse with the given initial weight.
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            trace_pre: 0.0,
            trace_post: 0.0,
        }
    }

    /// Called when a pre-synaptic spike is observed (at current time step).
    ///
    /// Increments `trace_pre` and applies LTD (negative weight change).
    pub fn update_pre(&mut self, _dt: f64, config: &StdpConfig) {
        self.trace_pre += 1.0;
        // LTD: weight decreases when pre fires after post.
        let dw = -config.a_minus * self.trace_post;
        self.weight = (self.weight + dw).clamp(config.w_min, config.w_max);
    }

    /// Called when a post-synaptic spike is observed (at current time step).
    ///
    /// Increments `trace_post`, applies LTP (positive weight change), and
    /// returns the weight change `dw`.
    pub fn update_post(&mut self, _dt: f64, config: &StdpConfig) -> f64 {
        self.trace_post += 1.0;
        // LTP: weight increases when post fires after pre.
        let dw = config.a_plus * self.trace_pre;
        self.weight = (self.weight + dw).clamp(config.w_min, config.w_max);
        dw
    }

    /// Exponentially decay both eligibility traces by one time step `dt`.
    pub fn decay_traces(&mut self, dt: f64, config: &StdpConfig) {
        let decay_pre = (-dt / config.tau_plus).exp();
        let decay_post = (-dt / config.tau_minus).exp();
        self.trace_pre *= decay_pre;
        self.trace_post *= decay_post;
    }
}

// ── §4 Spiking Linear Layer ──

/// A fully-connected spiking layer: linear current injection followed by a
/// population of LIF neurons.
///
/// `weights[out][in]` maps input spikes to synaptic currents.
#[derive(Debug, Clone)]
pub struct SpikingLinear {
    /// Weight matrix `[out_size][in_size]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector `[out_size]`.
    pub biases: Vec<f64>,
    /// One LIF neuron per output unit.
    pub neurons: Vec<LifNeuron>,
    /// Shared LIF configuration.
    pub config: LifConfig,
}

impl SpikingLinear {
    /// Construct a spiking linear layer, initialised with small random weights
    /// (Xavier/Glorot uniform, seed = 42).
    pub fn new(in_size: usize, out_size: usize, config: LifConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let limit = if in_size + out_size > 0 {
            (6.0_f64 / (in_size + out_size) as f64).sqrt()
        } else {
            0.1
        };
        let weights: Vec<Vec<f64>> = (0..out_size)
            .map(|_| {
                (0..in_size)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
                    .collect()
            })
            .collect();
        let biases = vec![0.0; out_size];
        let neurons = (0..out_size).map(|_| LifNeuron::new(&config)).collect();
        Self {
            weights,
            biases,
            neurons,
            config,
        }
    }

    /// Process one time step: compute synaptic currents and step each neuron.
    ///
    /// `input_spikes` is a boolean slice of length `in_size`.
    /// Returns output spike vector of length `out_size`.
    pub fn forward_step(&mut self, input_spikes: &[bool]) -> Vec<bool> {
        let out_size = self.neurons.len();
        let in_size = input_spikes.len();
        (0..out_size)
            .map(|o| {
                // Weighted sum of input spikes + bias.
                let i_ext: f64 = self.biases[o]
                    + (0..in_size.min(self.weights[o].len()))
                        .filter(|&i| input_spikes[i])
                        .map(|i| self.weights[o][i])
                        .sum::<f64>();
                self.neurons[o].step(i_ext, &self.config)
            })
            .collect()
    }

    /// Process a full spike-train sequence `[time_steps][in_size]`.
    ///
    /// Returns output spike train `[time_steps][out_size]`.
    pub fn forward_sequence(&mut self, spike_train: &[Vec<bool>]) -> Vec<Vec<bool>> {
        self.reset_state();
        spike_train
            .iter()
            .map(|step_spikes| self.forward_step(step_spikes))
            .collect()
    }

    /// Reset all neuron states to resting conditions.
    pub fn reset_state(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset(&self.config);
        }
    }
}

// ── §5 Surrogate Gradient ──

/// Trait for surrogate (pseudo-)gradient functions used in backpropagation
/// through discontinuous spike events.
pub trait SurrogateGradient {
    /// Heaviside forward: 1 if `v >= threshold`, else 0.
    fn forward(&self, v: f64, threshold: f64) -> f64 {
        if v >= threshold {
            1.0
        } else {
            0.0
        }
    }

    /// Surrogate derivative approximating `d/dv Heaviside(v - threshold)`.
    fn backward(&self, v: f64, threshold: f64) -> f64;
}

/// Fast-Sigmoid surrogate gradient (Neftci et al., 2019).
///
/// `d/dv ≈ beta / (2 * (1 + beta * |v - threshold|))^2`
#[derive(Debug, Clone)]
pub struct FastSigmoid {
    /// Sharpness parameter; larger beta → more spike-like.
    pub beta: f64,
}

impl FastSigmoid {
    /// Create a new `FastSigmoid` with the given sharpness.
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl Default for FastSigmoid {
    fn default() -> Self {
        Self { beta: 25.0 }
    }
}

impl SurrogateGradient for FastSigmoid {
    fn backward(&self, v: f64, threshold: f64) -> f64 {
        let denom = 1.0 + self.beta * (v - threshold).abs();
        self.beta / (2.0 * denom * denom)
    }
}

/// Piecewise-linear surrogate gradient (Esser et al., 2016).
///
/// `d/dv ≈ max(0, 1 - |v - threshold| / delta)`
#[derive(Debug, Clone)]
pub struct PiecewiseLinear {
    /// Half-width of the linear region.
    pub delta: f64,
}

impl PiecewiseLinear {
    /// Create a new `PiecewiseLinear` surrogate with the given width.
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl Default for PiecewiseLinear {
    fn default() -> Self {
        Self { delta: 0.5 }
    }
}

impl SurrogateGradient for PiecewiseLinear {
    fn backward(&self, v: f64, threshold: f64) -> f64 {
        let x = (v - threshold).abs() / self.delta;
        (1.0 - x).max(0.0)
    }
}

/// SuperSpike surrogate gradient (Zenke & Ganguli, 2018).
///
/// `d/dv ≈ 1 / (beta * |v - threshold| + 1)^2`
#[derive(Debug, Clone)]
pub struct SuperSpike {
    /// Steepness parameter.
    pub beta: f64,
}

impl SuperSpike {
    /// Create a new `SuperSpike` surrogate with the given steepness.
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl Default for SuperSpike {
    fn default() -> Self {
        Self { beta: 10.0 }
    }
}

impl SurrogateGradient for SuperSpike {
    fn backward(&self, v: f64, threshold: f64) -> f64 {
        let denom = self.beta * (v - threshold).abs() + 1.0;
        1.0 / (denom * denom)
    }
}

// ── §6 Population Encoder ──

/// Population (place-cell) encoder using Gaussian tuning curves.
///
/// Neuron `i` has preferred stimulus `mu_i` evenly spaced in `[x_min, x_max]`.
/// Its response is `exp(-(x - mu_i)^2 / (2 * sigma^2))`.
#[derive(Debug, Clone)]
pub struct PopulationEncoder {
    /// Number of neurons in the population.
    pub n_neurons: usize,
    /// Lower bound of the coded stimulus range.
    pub x_min: f64,
    /// Upper bound of the coded stimulus range.
    pub x_max: f64,
    /// Width of Gaussian tuning curves.
    pub sigma: f64,
}

impl PopulationEncoder {
    /// Create a new `PopulationEncoder`.
    ///
    /// `sigma` defaults to the inter-neuron spacing for full coverage.
    pub fn new(n_neurons: usize, x_min: f64, x_max: f64) -> Self {
        let spacing = if n_neurons > 1 {
            (x_max - x_min) / (n_neurons - 1) as f64
        } else {
            1.0
        };
        let sigma = spacing.max(1e-8);
        Self {
            n_neurons,
            x_min,
            x_max,
            sigma,
        }
    }

    /// Set a custom tuning-curve width.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Preferred stimulus value of neuron `i`.
    #[inline]
    pub fn preferred(&self, i: usize) -> f64 {
        if self.n_neurons <= 1 {
            (self.x_min + self.x_max) / 2.0
        } else {
            self.x_min + i as f64 * (self.x_max - self.x_min) / (self.n_neurons - 1) as f64
        }
    }

    /// Encode stimulus `x` as a vector of normalised firing rates in `[0, 1]`.
    pub fn encode(&self, x: f64) -> Vec<f64> {
        let two_sigma2 = 2.0 * self.sigma * self.sigma;
        (0..self.n_neurons)
            .map(|i| {
                let mu = self.preferred(i);
                let diff = x - mu;
                (-diff * diff / two_sigma2).exp()
            })
            .collect()
    }

    /// Reconstruct `x` from population rates using weighted-mean decoding.
    ///
    /// Returns `(x_min + x_max) / 2` if all rates are zero.
    pub fn decode(&self, rates: &[f64]) -> f64 {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        let len = rates.len().min(self.n_neurons);
        for i in 0..len {
            let mu = self.preferred(i);
            num += rates[i] * mu;
            den += rates[i];
        }
        if den < 1e-12 {
            (self.x_min + self.x_max) / 2.0
        } else {
            (num / den).clamp(self.x_min, self.x_max)
        }
    }
}

// ── §7 Liquid State Machine / Echo State Network ──

/// Configuration for the Liquid State Machine (LSM) / Echo State Network.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Number of neurons in the reservoir.
    pub n_reservoir: usize,
    /// Dimensionality of the input signal.
    pub n_input: usize,
    /// Dimensionality of the readout output.
    pub n_output: usize,
    /// Target spectral radius of the reservoir weight matrix.
    pub spectral_radius: f64,
    /// Fraction of reservoir-to-reservoir connections that are non-zero.
    pub sparsity: f64,
    /// Scaling factor applied to the input weight matrix.
    pub input_scaling: f64,
    /// Neuron dynamics for each reservoir unit.
    pub lif_config: LifConfig,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            n_reservoir: 100,
            n_input: 1,
            n_output: 1,
            spectral_radius: 0.9,
            sparsity: 0.1,
            input_scaling: 1.0,
            lif_config: LifConfig::default(),
        }
    }
}

/// Liquid State Machine (LSM) / Echo State Network with LIF reservoir.
///
/// The reservoir `W_res` is initialised as a sparse random matrix whose
/// spectral radius is rescaled to `config.spectral_radius` via power iteration.
/// Readout weights `W_out` are trained by ridge regression on reservoir states.
#[derive(Debug, Clone)]
pub struct LiquidStateMachine {
    /// Input-to-reservoir weight matrix `[n_reservoir][n_input]`.
    pub w_in: Vec<Vec<f64>>,
    /// Reservoir-to-reservoir weight matrix `[n_reservoir][n_reservoir]`.
    pub w_res: Vec<Vec<f64>>,
    /// Readout weight matrix `[n_output][n_reservoir]`.
    pub w_out: Vec<Vec<f64>>,
    /// State vector of LIF neurons (one per reservoir unit).
    pub state: Vec<LifNeuron>,
    /// Model configuration.
    pub config: LsmConfig,
}

impl LiquidStateMachine {
    /// Build a new LSM.  `W_res` is initialised as a sparse random matrix
    /// rescaled so that its spectral radius equals `config.spectral_radius`.
    pub fn new(config: LsmConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(0xdead_beef_u64);
        let nr = config.n_reservoir;
        let ni = config.n_input;
        let no = config.n_output;

        // Input weights: uniform in [-input_scaling, input_scaling].
        let w_in: Vec<Vec<f64>> = (0..nr)
            .map(|_| {
                (0..ni)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * config.input_scaling)
                    .collect()
            })
            .collect();

        // Reservoir weights: sparse random, then rescale spectral radius.
        let mut w_res: Vec<Vec<f64>> = (0..nr)
            .map(|_| {
                (0..nr)
                    .map(|_| {
                        if rng.random::<f64>() < config.sparsity {
                            rng.random::<f64>() * 2.0 - 1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Rescale to desired spectral radius (avoid division by zero).
        let rho = Self::spectral_radius(&w_res);
        if rho > 1e-8 {
            let scale = config.spectral_radius / rho;
            for row in &mut w_res {
                for w in row.iter_mut() {
                    *w *= scale;
                }
            }
        }

        // Readout initialised to zero.
        let w_out = vec![vec![0.0; nr]; no];

        let state = (0..nr)
            .map(|_| LifNeuron::new(&config.lif_config))
            .collect();

        Self {
            w_in,
            w_res,
            w_out,
            state,
            config,
        }
    }

    /// Run the reservoir on a time-series of inputs `[time_steps][n_input]`.
    ///
    /// Returns readout activations `[time_steps][n_output]`.
    pub fn run(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let nr = self.config.n_reservoir;
        let no = self.config.n_output;

        // Track spikes from the previous time step for recurrent connections.
        let mut prev_step_spikes = vec![0.0_f64; nr];

        // Collect reservoir spike states as binary vectors.
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            // Compute synaptic input to each reservoir neuron.
            let mut step_output = vec![false; nr];
            for o in 0..nr {
                // Input contribution.
                let i_in: f64 = (0..input.len().min(self.w_in[o].len()))
                    .map(|k| self.w_in[o][k] * input[k])
                    .sum();

                // Recurrent contribution from previous time-step spike states.
                let i_rec: f64 = (0..nr.min(self.w_res[o].len()))
                    .map(|j| self.w_res[o][j] * prev_step_spikes[j])
                    .sum();

                let fired = self.state[o].step(i_in + i_rec, &self.config.lif_config);
                step_output[o] = fired;
            }

            // Update previous spike record.
            for j in 0..nr {
                prev_step_spikes[j] = if step_output[j] { 1.0 } else { 0.0 };
            }

            // Compute readout as linear combination of binary reservoir activity.
            let readout: Vec<f64> = (0..no)
                .map(|r| {
                    (0..nr.min(self.w_out[r].len()))
                        .map(|j| self.w_out[r][j] * prev_step_spikes[j])
                        .sum()
                })
                .collect();
            outputs.push(readout);
        }
        outputs
    }

    /// Train the readout weights `W_out` by ridge regression on reservoir states.
    ///
    /// `inputs` drives the reservoir; `targets` is `[time][n_output]`.
    /// The ridge parameter is fixed at `1e-4`.
    pub fn train_readout(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let nr = self.config.n_reservoir;
        let t = inputs.len().min(targets.len());
        if t == 0 {
            return;
        }

        // Reset and collect reservoir spike states.
        for neuron in &mut self.state {
            neuron.reset(&self.config.lif_config);
        }

        let mut states: Vec<Vec<f64>> = Vec::with_capacity(t); // [t][nr]

        for input in inputs.iter().take(t) {
            let mut step_spikes = vec![false; nr];
            for o in 0..nr {
                let i_in: f64 = (0..input.len().min(self.w_in[o].len()))
                    .map(|k| self.w_in[o][k] * input[k])
                    .sum();
                let prev_f: Vec<f64> = states
                    .last().cloned()
                    .unwrap_or_else(|| vec![0.0; nr]);
                let i_rec: f64 = (0..nr.min(self.w_res[o].len()))
                    .map(|j| self.w_res[o][j] * prev_f[j])
                    .sum();
                let fired = self.state[o].step(i_in + i_rec, &self.config.lif_config);
                step_spikes[o] = fired;
            }
            states.push(
                step_spikes
                    .iter()
                    .map(|&b| if b { 1.0 } else { 0.0 })
                    .collect(),
            );
        }

        // Ridge regression: W_out = (X^T X + lambda I)^{-1} X^T Y  (closed form, small nr).
        // X: [t x nr], Y: [t x n_output].
        let lambda = 1e-4_f64;
        let no = self.config.n_output;

        // XtX: [nr x nr].
        let mut xtx = vec![vec![0.0_f64; nr]; nr];
        for row in &states {
            for i in 0..nr {
                for j in 0..nr {
                    xtx[i][j] += row[i] * row[j];
                }
            }
        }
        // Add ridge regularization.
        for i in 0..nr {
            xtx[i][i] += lambda;
        }

        // XtY: [nr x n_output].
        let mut xty = vec![vec![0.0_f64; no]; nr];
        for (row, target) in states.iter().zip(targets.iter().take(t)) {
            for i in 0..nr {
                for r in 0..no.min(target.len()) {
                    xty[i][r] += row[i] * target[r];
                }
            }
        }

        // Solve (XtX + lambda I) W = XtY via Gaussian elimination (small nr).
        match ridge_solve(&xtx, &xty, nr, no) {
            Some(w) => self.w_out = w,
            None => {
                // Fallback: pseudo-inverse via diagonal.
                for r in 0..no {
                    for i in 0..nr {
                        self.w_out[r][i] = xty[i][r] / xtx[i][i].max(lambda);
                    }
                }
            }
        }
    }

    /// Estimate the spectral radius of `w` via power iteration (50 iterations).
    pub fn spectral_radius(w: &[Vec<f64>]) -> f64 {
        let n = w.len();
        if n == 0 {
            return 0.0;
        }
        let mut rng = StdRng::seed_from_u64(0xbabe);
        let mut v: Vec<f64> = sample_normal_vec_f64(n, &mut rng);
        // Normalise.
        let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }

        let mut eigenvalue = 0.0_f64;
        for _ in 0..50 {
            let mut av = vec![0.0_f64; n];
            for i in 0..n {
                for j in 0..n.min(w[i].len()) {
                    av[i] += w[i][j] * v[j];
                }
            }
            let av_norm = (av.iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-12);
            eigenvalue = av_norm;
            for x in &mut av {
                *x /= av_norm;
            }
            v = av;
        }
        eigenvalue
    }
}

/// Gaussian elimination solver for the normal equations.
/// Returns `W_out` of shape `[n_output][n_reservoir]` or `None` on failure.
fn ridge_solve(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, m: usize) -> Option<Vec<Vec<f64>>> {
    // Augmented matrix [A | B] of size n x (n + m).
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            for j in 0..m {
                row.push(b[i][j]);
            }
            row
        })
        .collect();

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            aug[r1][col]
                .abs()
                .partial_cmp(&aug[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            return None;
        }
        let inv_pivot = 1.0 / pivot;
        for k in col..(n + m) {
            aug[col][k] *= inv_pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for k in col..(n + m) {
                let val = factor * aug[col][k];
                aug[row][k] -= val;
            }
        }
    }

    // Extract solution: W_out[output][reservoir] = aug[reservoir][n + output].
    let w_out: Vec<Vec<f64>> = (0..m)
        .map(|r| (0..n).map(|i| aug[i][n + r]).collect())
        .collect();
    Some(w_out)
}

// ── §8 AdEx Neuron ──

/// Configuration for the Adaptive Exponential Integrate-and-Fire (AdEx) neuron.
///
/// Reference: Brette & Gerstner (2005).
#[derive(Debug, Clone)]
pub struct AdexConfig {
    /// Membrane capacitance (pF).
    pub c_m: f64,
    /// Leak conductance (nS).
    pub g_l: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Threshold slope factor (mV).
    pub v_t: f64,
    /// Sharpness of action potential onset (mV).
    pub delta_t: f64,
    /// Adaptation time constant (ms).
    pub tau_w: f64,
    /// Subthreshold adaptation coupling (nS).
    pub a: f64,
    /// Spike-triggered adaptation increment (pA).
    pub b: f64,
    /// Spike detection threshold (mV).
    pub v_thresh: f64,
    /// Reset potential after spike (mV).
    pub v_reset: f64,
    /// Integration time step (ms).
    pub dt: f64,
}

impl Default for AdexConfig {
    fn default() -> Self {
        Self {
            c_m: 281.0,
            g_l: 30.0,
            e_l: -70.6,
            v_t: -50.4,
            delta_t: 2.0,
            tau_w: 144.0,
            a: 4.0,
            b: 80.5,
            v_thresh: -30.0,
            v_reset: -70.6,
            dt: 0.1,
        }
    }
}

/// A single AdEx neuron with subthreshold adaptation variable `w`.
///
/// Dynamics (Euler integration):
/// ```text
/// dV/dt = (-g_l*(V - E_l) + g_l*delta_t*exp((V - V_t)/delta_t) - w + I) / C_m
/// dw/dt = (a*(V - E_l) - w) / tau_w
/// ```
/// On spike (`V >= V_thresh`): `V → V_reset`, `w += b`.
#[derive(Debug, Clone)]
pub struct AdexNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Adaptation variable (pA).
    pub w: f64,
}

impl AdexNeuron {
    /// Create a new AdEx neuron at resting potential.
    pub fn new(config: &AdexConfig) -> Self {
        Self {
            v: config.e_l,
            w: 0.0,
        }
    }

    /// Advance one Euler step.  Returns `true` on spike.
    pub fn step(&mut self, i_ext: f64, config: &AdexConfig) -> bool {
        // Exponential term — clamp argument to avoid overflow.
        let exp_arg = ((self.v - config.v_t) / config.delta_t).min(30.0);
        let dv = (-config.g_l * (self.v - config.e_l)
            + config.g_l * config.delta_t * exp_arg.exp()
            - self.w
            + i_ext)
            / config.c_m;
        let dw = (config.a * (self.v - config.e_l) - self.w) / config.tau_w;

        self.v += config.dt * dv;
        self.w += config.dt * dw;

        if self.v >= config.v_thresh {
            self.v = config.v_reset;
            self.w += config.b;
            true
        } else {
            false
        }
    }
}

// ── §9 SNN Metrics ──

/// Summary statistics for a population of spike trains.
#[derive(Debug, Clone)]
pub struct SnnMetrics {
    /// Mean firing rate averaged over neurons (spikes per time step).
    pub mean_firing_rate: f64,
    /// Fraction of neurons that did not fire a single spike.
    pub sparsity: f64,
    /// Mean pairwise spike-train correlation (Pearson).
    pub synchrony: f64,
    /// Coefficient of variation (CV) of inter-spike intervals (ISI).
    pub inter_spike_interval_cv: f64,
}

/// Compute SNN population metrics from a spike-train matrix.
///
/// `spike_trains[neuron][time_step]`.
pub fn compute_snn_metrics(spike_trains: &[Vec<bool>]) -> SnnMetrics {
    if spike_trains.is_empty() {
        return SnnMetrics {
            mean_firing_rate: 0.0,
            sparsity: 1.0,
            synchrony: 0.0,
            inter_spike_interval_cv: 0.0,
        };
    }
    let n_neurons = spike_trains.len();
    let t = spike_trains[0].len();

    // Firing rates per neuron.
    let rates: Vec<f64> = spike_trains
        .iter()
        .map(|train| train.iter().filter(|&&b| b).count() as f64 / t.max(1) as f64)
        .collect();

    let mean_firing_rate = rates.iter().sum::<f64>() / n_neurons as f64;
    let sparsity = rates.iter().filter(|&&r| r < 1e-10).count() as f64 / n_neurons as f64;

    // Synchrony: mean pairwise Pearson correlation.
    let synchrony = if n_neurons < 2 || t == 0 {
        0.0
    } else {
        let means: Vec<f64> = rates.clone(); // rate == mean of binary vector.
        let mut sum_corr = 0.0_f64;
        let mut count = 0_usize;
        for i in 0..n_neurons {
            for j in (i + 1)..n_neurons {
                let mi = means[i];
                let mj = means[j];
                let mut cov = 0.0_f64;
                let mut vi = 0.0_f64;
                let mut vj = 0.0_f64;
                for k in 0..t.min(spike_trains[i].len()).min(spike_trains[j].len()) {
                    let xi = if spike_trains[i][k] { 1.0 } else { 0.0 } - mi;
                    let xj = if spike_trains[j][k] { 1.0 } else { 0.0 } - mj;
                    cov += xi * xj;
                    vi += xi * xi;
                    vj += xj * xj;
                }
                let denom = (vi * vj).sqrt();
                if denom > 1e-12 {
                    sum_corr += cov / denom;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            sum_corr / count as f64
        }
    };

    // ISI-CV: pooled across all neurons.
    let mut all_isis: Vec<f64> = Vec::new();
    for train in spike_trains {
        let spike_times: Vec<usize> = train
            .iter()
            .enumerate()
            .filter_map(|(t, &b)| if b { Some(t) } else { None })
            .collect();
        for w in spike_times.windows(2) {
            all_isis.push((w[1] - w[0]) as f64);
        }
    }
    let inter_spike_interval_cv = if all_isis.len() < 2 {
        0.0
    } else {
        let mean_isi = all_isis.iter().sum::<f64>() / all_isis.len() as f64;
        let var_isi = all_isis
            .iter()
            .map(|&x| (x - mean_isi).powi(2))
            .sum::<f64>()
            / all_isis.len() as f64;
        let std_isi = var_isi.sqrt();
        if mean_isi > 1e-12 {
            std_isi / mean_isi
        } else {
            0.0
        }
    };

    SnnMetrics {
        mean_firing_rate,
        sparsity,
        synchrony,
        inter_spike_interval_cv,
    }
}

/// Compute summary statistics of membrane voltage traces.
///
/// `voltages[neuron][time_step]`.
///
/// Returns `(mean, std, max)`.
pub fn compute_membrane_stats(voltages: &[Vec<f64>]) -> (f64, f64, f64) {
    let flat: Vec<f64> = voltages.iter().flat_map(|v| v.iter().copied()).collect();
    if flat.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = flat.len() as f64;
    let mean = flat.iter().sum::<f64>() / n;
    let var = flat.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let max = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, std, max)
}

// ── §10 Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    // ── LIF Neuron ────────────────────────────────────────────────────────────

    #[test]
    fn test_lif_fires_above_threshold() {
        let cfg = LifConfig::default();
        let mut neuron = LifNeuron::new(&cfg);
        // Drive with a large current until it fires.
        let mut fired = false;
        for _ in 0..100 {
            if neuron.step(50.0, &cfg) {
                fired = true;
                break;
            }
        }
        assert!(fired, "LIF neuron should fire with large input current");
    }

    #[test]
    fn test_lif_does_not_fire_below_threshold() {
        let cfg = LifConfig::default();
        let mut neuron = LifNeuron::new(&cfg);
        // Sub-threshold drive — should never fire.
        let fired = (0..20).any(|_| neuron.step(0.0, &cfg));
        assert!(!fired, "LIF neuron should not fire without input");
    }

    #[test]
    fn test_lif_refractory_period() {
        let cfg = LifConfig {
            refractory_steps: 5,
            ..LifConfig::default()
        };
        let mut neuron = LifNeuron::new(&cfg);
        // Force a spike.
        let mut first_spike_t = None;
        for t in 0..200 {
            if neuron.step(100.0, &cfg) && first_spike_t.is_none() {
                first_spike_t = Some(t);
            }
        }
        assert!(first_spike_t.is_some(), "Should have spiked");
        // After the first spike the neuron should have entered refractory.
        // We test that resetting refractory > 0 prevents immediate re-fire.
        // Just check that the neuron eventually recovers (fires again later).
        let mut second_spike = false;
        for _ in 0..100 {
            if neuron.step(100.0, &cfg) {
                second_spike = true;
                break;
            }
        }
        assert!(second_spike, "Should fire again after refractory period");
    }

    #[test]
    fn test_lif_reset() {
        let cfg = LifConfig::default();
        let mut neuron = LifNeuron::new(&cfg);
        // Inject current to change state.
        for _ in 0..5 {
            neuron.step(5.0, &cfg);
        }
        neuron.reset(&cfg);
        assert!(
            (neuron.v_mem - cfg.v_rest).abs() < 1e-9,
            "reset should restore resting potential"
        );
        assert_eq!(neuron.refractory, 0);
    }

    // ── STDP ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stdp_ltp_pre_before_post() {
        let cfg = StdpConfig::default();
        let mut synapse = StdpSynapse::new(0.5);
        // Pre fires first: increment trace_pre.
        synapse.update_pre(1.0, &cfg);
        synapse.decay_traces(5.0, &cfg); // small decay
                                         // Post fires: LTP.
        let dw = synapse.update_post(1.0, &cfg);
        assert!(dw > 0.0, "LTP: pre-before-post should increase weight");
        assert!(synapse.weight > 0.5);
    }

    #[test]
    fn test_stdp_ltd_post_before_pre() {
        let cfg = StdpConfig::default();
        let mut synapse = StdpSynapse::new(0.5);
        // Post fires first.
        synapse.update_post(1.0, &cfg);
        synapse.decay_traces(5.0, &cfg);
        // Pre fires: LTD.
        let weight_before = synapse.weight;
        synapse.update_pre(1.0, &cfg);
        assert!(
            synapse.weight < weight_before,
            "LTD: post-before-pre should decrease weight"
        );
    }

    #[test]
    fn test_stdp_weight_bounds() {
        let cfg = StdpConfig {
            w_min: 0.0,
            w_max: 1.0,
            ..Default::default()
        };
        let mut synapse = StdpSynapse::new(0.99);
        // Repeated LTP should not exceed w_max.
        for _ in 0..100 {
            synapse.update_pre(1.0, &cfg);
            synapse.update_post(1.0, &cfg);
        }
        assert!(synapse.weight <= 1.0, "Weight must not exceed w_max");
        assert!(synapse.weight >= 0.0, "Weight must not go below w_min");
    }

    #[test]
    fn test_stdp_traces_decay() {
        let cfg = StdpConfig::default();
        let mut synapse = StdpSynapse::new(0.5);
        synapse.update_pre(1.0, &cfg);
        let trace_before = synapse.trace_pre;
        synapse.decay_traces(10.0, &cfg);
        assert!(synapse.trace_pre < trace_before, "Trace should decay");
    }

    // ── Spike Encoding ────────────────────────────────────────────────────────

    #[test]
    fn test_rate_encoding_output_shape() {
        let encoder = SpikeEncoder::new(SpikeEncoding::Rate, 8, 50);
        let spikes = encoder.encode(0.5);
        assert_eq!(spikes.len(), 50, "time_steps rows");
        for row in &spikes {
            assert_eq!(row.len(), 8, "n_neurons cols");
        }
    }

    #[test]
    fn test_rate_encoding_high_value_fires_more() {
        let encoder = SpikeEncoder::new(SpikeEncoding::Rate, 10, 1000);
        let low_spikes: usize = encoder
            .encode(0.1)
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&b| b)
            .count();
        let high_spikes: usize = encoder
            .encode(0.9)
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&b| b)
            .count();
        assert!(
            high_spikes > low_spikes,
            "Higher value should produce more spikes in rate coding"
        );
    }

    #[test]
    fn test_rate_encoding_round_trip() {
        let encoder = SpikeEncoder::new(SpikeEncoding::Rate, 20, 2000);
        for &v in &[0.2, 0.5, 0.8] {
            let spikes = encoder.encode(v);
            let decoded = encoder.decode(&spikes);
            assert!(
                (decoded - v).abs() < 0.15,
                "Rate round-trip failed for v={}: decoded={}",
                v,
                decoded
            );
        }
    }

    #[test]
    fn test_encoding_zero_and_one() {
        let encoder = SpikeEncoder::new(SpikeEncoding::Rate, 5, 100);
        let zero_spikes: usize = encoder
            .encode(0.0)
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&b| b)
            .count();
        assert_eq!(zero_spikes, 0, "No spikes for value=0");
    }

    // ── Population Encoder ────────────────────────────────────────────────────

    #[test]
    fn test_population_encoder_peak_at_preferred() {
        let enc = PopulationEncoder::new(5, 0.0, 1.0);
        // For x = preferred[2] = 0.5, neuron 2 should have the highest rate.
        let rates = enc.encode(0.5);
        let max_idx = rates
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_idx, 2, "Center neuron should peak at x=0.5");
    }

    #[test]
    fn test_population_encoder_decode_identity() {
        let enc = PopulationEncoder::new(10, 0.0, 1.0);
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let rates = enc.encode(x);
            let decoded = enc.decode(&rates);
            assert!(
                (decoded - x).abs() < 0.1,
                "PopulationEncoder round-trip: x={}, decoded={}",
                x,
                decoded
            );
        }
    }

    #[test]
    fn test_population_encoder_covers_full_range() {
        let enc = PopulationEncoder::new(8, -1.0, 1.0);
        // Encode boundary values and check that the dominant neuron is at the edges.
        let rates_min = enc.encode(-1.0);
        let rates_max = enc.encode(1.0);
        let idx_min = rates_min
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(99);
        let idx_max = rates_max
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(99);
        assert_eq!(idx_min, 0, "Leftmost neuron should peak at x_min");
        assert_eq!(idx_max, 7, "Rightmost neuron should peak at x_max");
    }

    #[test]
    fn test_population_encoder_zero_rates_decode() {
        let enc = PopulationEncoder::new(5, 0.0, 1.0);
        let rates = vec![0.0; 5];
        let decoded = enc.decode(&rates);
        assert_eq!(decoded, 0.5, "Zero rates should decode to midpoint");
    }

    // ── Spiking Linear Layer ──────────────────────────────────────────────────

    #[test]
    fn test_spiking_linear_forward_step_shape() {
        let cfg = LifConfig::default();
        let mut layer = SpikingLinear::new(4, 3, cfg);
        let input = vec![true, false, true, true];
        let output = layer.forward_step(&input);
        assert_eq!(output.len(), 3, "Output should have out_size elements");
    }

    #[test]
    fn test_spiking_linear_forward_sequence_shape() {
        let cfg = LifConfig::default();
        let mut layer = SpikingLinear::new(5, 4, cfg);
        let spike_train: Vec<Vec<bool>> = vec![vec![true, false, true, false, true]; 10];
        let out = layer.forward_sequence(&spike_train);
        assert_eq!(out.len(), 10, "Should have one output per time step");
        for row in &out {
            assert_eq!(row.len(), 4, "Each step should have out_size outputs");
        }
    }

    #[test]
    fn test_spiking_linear_reset_state() {
        let cfg = LifConfig::default();
        let mut layer = SpikingLinear::new(3, 2, cfg.clone());
        // Run some steps.
        for _ in 0..5 {
            layer.forward_step(&[true, true, true]);
        }
        layer.reset_state();
        for neuron in &layer.neurons {
            assert!((neuron.v_mem - cfg.v_rest).abs() < 1e-9);
            assert_eq!(neuron.refractory, 0);
        }
    }

    #[test]
    fn test_spiking_linear_produces_spikes_with_high_weight() {
        let cfg = LifConfig::default();
        let mut layer = SpikingLinear::new(2, 1, cfg);
        // Override weights to be very large.
        layer.weights[0][0] = 100.0;
        layer.weights[0][1] = 100.0;
        let mut fired = false;
        for _ in 0..20 {
            if layer.forward_step(&[true, true])[0] {
                fired = true;
                break;
            }
        }
        assert!(fired, "Layer should fire with large weights");
    }

    // ── Surrogate Gradients ────────────────────────────────────────────────────

    #[test]
    fn test_fast_sigmoid_forward_above_threshold() {
        let sg = FastSigmoid::default();
        assert_eq!(sg.forward(1.0, 0.5), 1.0);
        assert_eq!(sg.forward(-0.1, 0.5), 0.0);
    }

    #[test]
    fn test_fast_sigmoid_backward_peak_at_threshold() {
        let sg = FastSigmoid::new(25.0);
        // Gradient is maximal at v == threshold.
        let at_threshold = sg.backward(0.5, 0.5);
        let away = sg.backward(0.0, 0.5);
        assert!(
            at_threshold > away,
            "FastSigmoid gradient peaks at threshold"
        );
    }

    #[test]
    fn test_piecewise_linear_backward_zero_outside_delta() {
        let sg = PiecewiseLinear::new(0.5);
        assert_eq!(sg.backward(-1.0, 0.0), 0.0);
        assert_eq!(sg.backward(1.0, 0.0), 0.0);
    }

    #[test]
    fn test_piecewise_linear_backward_one_at_threshold() {
        let sg = PiecewiseLinear::new(0.5);
        assert!((sg.backward(0.5, 0.5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_superspike_backward_positive() {
        let sg = SuperSpike::new(10.0);
        for v in [-1.0, 0.0, 0.5, 1.0] {
            assert!(
                sg.backward(v, 0.5) > 0.0,
                "SuperSpike gradient must be positive"
            );
        }
    }

    #[test]
    fn test_superspike_backward_peak_at_threshold() {
        let sg = SuperSpike::default();
        let peak = sg.backward(0.5, 0.5);
        let off = sg.backward(2.0, 0.5);
        assert!(peak > off, "SuperSpike gradient peaks at threshold");
    }

    // ── AdEx Neuron ───────────────────────────────────────────────────────────

    #[test]
    fn test_adex_fires_with_strong_input() {
        let cfg = AdexConfig::default();
        let mut neuron = AdexNeuron::new(&cfg);
        let mut fired = false;
        for _ in 0..5000 {
            if neuron.step(1000.0, &cfg) {
                fired = true;
                break;
            }
        }
        assert!(fired, "AdEx should fire with strong input current");
    }

    #[test]
    fn test_adex_adaptation_increases_w_after_spike() {
        let cfg = AdexConfig::default();
        let mut neuron = AdexNeuron::new(&cfg);
        let w_before = neuron.w;
        // Run until first spike.
        for _ in 0..5000 {
            if neuron.step(1000.0, &cfg) {
                break;
            }
        }
        assert!(
            neuron.w > w_before,
            "Adaptation variable w should increase after a spike"
        );
    }

    #[test]
    fn test_adex_subthreshold_no_spike() {
        let cfg = AdexConfig::default();
        let mut neuron = AdexNeuron::new(&cfg);
        let fired = (0..100).any(|_| neuron.step(0.0, &cfg));
        assert!(!fired, "AdEx should not fire without input");
    }

    #[test]
    fn test_adex_reset_after_spike() {
        let cfg = AdexConfig::default();
        let mut neuron = AdexNeuron::new(&cfg);
        // Force a spike.
        for _ in 0..5000 {
            if neuron.step(2000.0, &cfg) {
                break;
            }
        }
        assert!(
            (neuron.v - cfg.v_reset).abs() < 1.0,
            "After spike, V should be near V_reset: v={}",
            neuron.v
        );
    }

    // ── LSM ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_lsm_run_output_shape() {
        let cfg = LsmConfig {
            n_reservoir: 20,
            n_input: 2,
            n_output: 3,
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(cfg);
        let inputs: Vec<Vec<f64>> = (0..10).map(|_| vec![0.5, -0.3]).collect();
        let outputs = lsm.run(&inputs);
        assert_eq!(outputs.len(), 10, "One output per time step");
        for row in &outputs {
            assert_eq!(row.len(), 3, "n_output outputs per step");
        }
    }

    #[test]
    fn test_lsm_train_readout_runs() {
        let cfg = LsmConfig {
            n_reservoir: 15,
            n_input: 1,
            n_output: 1,
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(cfg);
        let inputs: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.03]).collect();
        let targets: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.03]).collect();
        lsm.train_readout(&inputs, &targets);
        // After training, the readout weights should be non-trivial.
        let w_sum: f64 = lsm.w_out.iter().flat_map(|r| r.iter().copied()).sum();
        // Just check it ran without panic.
        let _ = w_sum;
    }

    #[test]
    fn test_lsm_empty_input() {
        let cfg = LsmConfig::default();
        let mut lsm = LiquidStateMachine::new(cfg);
        let outputs = lsm.run(&[]);
        assert!(outputs.is_empty());
    }

    // ── SNN Metrics ───────────────────────────────────────────────────────────

    #[test]
    fn test_snn_metrics_empty() {
        let metrics = compute_snn_metrics(&[]);
        assert_eq!(metrics.mean_firing_rate, 0.0);
        assert_eq!(metrics.sparsity, 1.0);
    }

    #[test]
    fn test_snn_metrics_all_silent() {
        let trains = vec![vec![false; 20]; 5];
        let metrics = compute_snn_metrics(&trains);
        assert_eq!(metrics.mean_firing_rate, 0.0);
        assert_eq!(metrics.sparsity, 1.0);
    }

    #[test]
    fn test_snn_metrics_all_active() {
        let trains = vec![vec![true; 10]; 4];
        let metrics = compute_snn_metrics(&trains);
        assert!((metrics.mean_firing_rate - 1.0).abs() < 1e-9);
        assert_eq!(metrics.sparsity, 0.0);
    }

    #[test]
    fn test_snn_metrics_isi_cv() {
        // Regular spiking every 5 steps → ISI = 5 for all intervals → CV ≈ 0.
        let mut train = vec![false; 50];
        for i in (0..50).step_by(5) {
            train[i] = true;
        }
        let metrics = compute_snn_metrics(&[train]);
        assert!(
            metrics.inter_spike_interval_cv < 0.1,
            "Regular spiking should have low ISI CV: {}",
            metrics.inter_spike_interval_cv
        );
    }

    #[test]
    fn test_snn_metrics_synchrony_identical_trains() {
        // Two identical trains → Pearson correlation = 1.
        let train = vec![true, false, true, false, true, false, true, false];
        let metrics = compute_snn_metrics(&[train.clone(), train]);
        assert!(
            (metrics.synchrony - 1.0).abs() < 1e-6,
            "Identical trains must have synchrony 1: {}",
            metrics.synchrony
        );
    }

    #[test]
    fn test_compute_membrane_stats_basic() {
        let voltages = vec![vec![-70.0_f64, -60.0, -50.0], vec![-55.0, -65.0, -75.0]];
        let (mean, std, max) = compute_membrane_stats(&voltages);
        assert!((mean - (-62.5)).abs() < 1e-6, "mean={}", mean);
        assert!(std > 0.0, "std should be positive");
        assert!((max - (-50.0)).abs() < 1e-9, "max={}", max);
    }

    #[test]
    fn test_compute_membrane_stats_empty() {
        let (mean, std, max) = compute_membrane_stats(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
        assert_eq!(max, 0.0);
    }

    // ── Integration: full SNN pipeline ────────────────────────────────────────

    #[test]
    fn test_full_snn_pipeline() {
        let lif_cfg = LifConfig::default();
        let mut layer1 = SpikingLinear::new(4, 8, lif_cfg.clone());
        let mut layer2 = SpikingLinear::new(8, 2, lif_cfg.clone());

        // Override weights for deterministic test.
        for o in 0..8 {
            for i in 0..4 {
                layer1.weights[o][i] = 5.0;
            }
        }
        for o in 0..2 {
            for i in 0..8 {
                layer2.weights[o][i] = 5.0;
            }
        }

        let encoder = SpikeEncoder::new(SpikeEncoding::Rate, 4, 30);
        let input_train = encoder.encode(0.8);

        let hidden_train = layer1.forward_sequence(&input_train);
        let output_train = layer2.forward_sequence(&hidden_train);

        assert_eq!(output_train.len(), 30);
        for row in &output_train {
            assert_eq!(row.len(), 2);
        }

        let metrics = compute_snn_metrics(&output_train);
        // Just check no panic and metrics are in valid ranges.
        assert!(metrics.mean_firing_rate >= 0.0);
        assert!(metrics.sparsity >= 0.0 && metrics.sparsity <= 1.0);
    }

    #[test]
    fn test_population_encoder_single_neuron() {
        let enc = PopulationEncoder::new(1, 0.0, 1.0);
        let rates = enc.encode(0.5);
        assert_eq!(rates.len(), 1);
        assert!(
            (rates[0] - 1.0).abs() < 1e-9,
            "Single neuron should peak at its preferred"
        );
    }

    #[test]
    fn test_lif_multiple_spikes_recorded() {
        let cfg = LifConfig::default();
        let mut neuron = LifNeuron::new(&cfg);
        let mut spike_count = 0;
        for _ in 0..500 {
            if neuron.step(30.0, &cfg) {
                neuron.spike_times.push(spike_count as f64);
                spike_count += 1;
            }
        }
        assert!(spike_count > 3, "Should produce multiple spikes");
    }

    #[test]
    fn test_stdp_multiple_cycles() {
        let cfg = StdpConfig::default();
        let mut synapse = StdpSynapse::new(0.5);
        let initial_weight = synapse.weight;
        // Repeated pre-before-post cycles should potentiate.
        for _ in 0..10 {
            synapse.update_pre(1.0, &cfg);
            synapse.decay_traces(2.0, &cfg);
            synapse.update_post(1.0, &cfg);
            synapse.decay_traces(2.0, &cfg);
        }
        assert!(
            synapse.weight > initial_weight,
            "Repeated pre-before-post should increase weight"
        );
    }

    #[test]
    fn test_adex_spike_rate_increases_with_current() {
        let cfg = AdexConfig::default();
        let mut n_low = AdexNeuron::new(&cfg);
        let mut n_high = AdexNeuron::new(&cfg);
        let (mut low_count, mut high_count) = (0_usize, 0_usize);
        for _ in 0..5000 {
            if n_low.step(800.0, &cfg) {
                low_count += 1;
            }
            if n_high.step(1600.0, &cfg) {
                high_count += 1;
            }
        }
        assert!(
            high_count > low_count,
            "Stronger input should produce more spikes"
        );
    }
}
