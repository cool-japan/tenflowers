//! Quantum-Classical Hybrid Machine Learning (Simulation) — Track Q.
//!
//! Statevector-based quantum circuit simulation and quantum-classical hybrid ML
//! in pure Rust. Provides quantum circuit simulation, VQE, QNN, quantum
//! optimization, and noise models/error mitigation.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

pub mod advanced;
pub use advanced::{
    IqpFeatureMap, MaxCutQaoa, MeasurementErrorMitigation, ProbabilisticErrorCancellation,
    QaoaLayer, QaoaMetrics, QaoaOptimizer, QbmModel, QbmTrainer, QuantumFeatureMapType,
    QuantumKernelAlignment, QuantumKernelFull, ZneExtrapolation, ZzFeatureMap,
};

#[cfg(test)]
mod tests;

// ── Complex Number ────────────────────────────────────────────────────────────

/// 64-bit complex number with basic arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    #[inline]
    pub fn abs_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    #[inline]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    #[inline]
    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }
}

impl std::fmt::Display for Complex64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6}+{:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6}{:.6}i", self.re, self.im)
        }
    }
}

// ── Qubit State Vector ────────────────────────────────────────────────────────

/// Pure statevector for `n_qubits` qubits (2^n complex amplitudes).
/// Qubit 0 is the most-significant bit in the computational basis.
#[derive(Debug, Clone)]
pub struct QubitState {
    pub n_qubits: usize,
    pub amplitudes: Vec<Complex64>,
}

impl QubitState {
    /// Initialize to |0...0⟩.
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1usize << n_qubits;
        let mut amplitudes = vec![Complex64::zero(); dim];
        amplitudes[0] = Complex64::one();
        Self {
            n_qubits,
            amplitudes,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        1usize << self.n_qubits
    }

    /// Apply a gate to the specified qubit(s).
    pub fn apply_gate(&mut self, gate: &QuantumGate, qubits: &[usize]) {
        let matrix = gate.matrix();
        match qubits.len() {
            1 => self.apply_single_qubit_gate(&matrix, qubits[0]),
            2 => self.apply_two_qubit_gate(&matrix, qubits[0], qubits[1]),
            _ => {}
        }
    }

    fn apply_single_qubit_gate(&mut self, matrix: &[Vec<Complex64>], qubit: usize) {
        let bit = self.n_qubits - 1 - qubit;
        let stride = 1usize << bit;
        let block = stride * 2;
        let dim = self.dim();
        let mut new_amps = self.amplitudes.clone();
        let mut base = 0usize;
        while base < dim {
            for inner in 0..stride {
                let i0 = base + inner;
                let i1 = base + inner + stride;
                let (a0, a1) = (self.amplitudes[i0], self.amplitudes[i1]);
                new_amps[i0] = matrix[0][0].mul(a0).add(matrix[0][1].mul(a1));
                new_amps[i1] = matrix[1][0].mul(a0).add(matrix[1][1].mul(a1));
            }
            base += block;
        }
        self.amplitudes = new_amps;
    }

    fn apply_two_qubit_gate(&mut self, matrix: &[Vec<Complex64>], ctrl: usize, tgt: usize) {
        let bit_c = self.n_qubits - 1 - ctrl;
        let bit_t = self.n_qubits - 1 - tgt;
        let dim = self.dim();
        let mut new_amps = self.amplitudes.clone();
        for i in 0..dim {
            let row = ((i >> bit_c) & 1) * 2 + ((i >> bit_t) & 1);
            let val = (0..4).fold(Complex64::zero(), |acc, col| {
                let j = (i & !(1 << bit_c) & !(1 << bit_t))
                    | (((col >> 1) & 1) << bit_c)
                    | ((col & 1) << bit_t);
                acc.add(matrix[row][col].mul(self.amplitudes[j]))
            });
            new_amps[i] = val;
        }
        self.amplitudes = new_amps;
    }

    /// Return measurement probabilities (length 2^n, sums to 1).
    pub fn measure_probs(&self) -> Vec<f64> {
        let probs: Vec<f64> = self.amplitudes.iter().map(|a| a.abs_sq()).collect();
        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            probs.into_iter().map(|p| p / total).collect()
        } else {
            vec![1.0 / self.dim() as f64; self.dim()]
        }
    }

    /// Compute ⟨Z_q⟩ = P(|0⟩) - P(|1⟩) for qubit `q`.
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let bit = self.n_qubits - 1 - qubit;
        self.measure_probs()
            .iter()
            .enumerate()
            .map(|(idx, &p)| if (idx >> bit) & 1 == 0 { p } else { -p })
            .sum()
    }
}

// ── Quantum Gates ─────────────────────────────────────────────────────────────

/// Standard quantum gates.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumGate {
    H,
    X,
    Y,
    Z,
    CNOT,
    RX(f64),
    RY(f64),
    RZ(f64),
    Phase(f64),
    SWAP,
}

impl QuantumGate {
    /// Unitary matrix: 2×2 for single-qubit gates, 4×4 for CNOT/SWAP.
    pub fn matrix(&self) -> Vec<Vec<Complex64>> {
        let c = |re: f64, im: f64| Complex64::new(re, im);
        match self {
            QuantumGate::H => {
                let s = 1.0 / 2.0f64.sqrt();
                vec![vec![c(s, 0.0), c(s, 0.0)], vec![c(s, 0.0), c(-s, 0.0)]]
            }
            QuantumGate::X => vec![vec![c(0., 0.), c(1., 0.)], vec![c(1., 0.), c(0., 0.)]],
            QuantumGate::Y => vec![vec![c(0., 0.), c(0., -1.)], vec![c(0., 1.), c(0., 0.)]],
            QuantumGate::Z => vec![vec![c(1., 0.), c(0., 0.)], vec![c(0., 0.), c(-1., 0.)]],
            QuantumGate::CNOT => vec![
                vec![c(1., 0.), c(0., 0.), c(0., 0.), c(0., 0.)],
                vec![c(0., 0.), c(1., 0.), c(0., 0.), c(0., 0.)],
                vec![c(0., 0.), c(0., 0.), c(0., 0.), c(1., 0.)],
                vec![c(0., 0.), c(0., 0.), c(1., 0.), c(0., 0.)],
            ],
            QuantumGate::RX(t) => {
                let (ct, st) = ((t / 2.).cos(), (t / 2.).sin());
                vec![vec![c(ct, 0.), c(0., -st)], vec![c(0., -st), c(ct, 0.)]]
            }
            QuantumGate::RY(t) => {
                let (ct, st) = ((t / 2.).cos(), (t / 2.).sin());
                vec![vec![c(ct, 0.), c(-st, 0.)], vec![c(st, 0.), c(ct, 0.)]]
            }
            QuantumGate::RZ(t) => {
                let h = t / 2.;
                vec![
                    vec![c(h.cos(), -h.sin()), c(0., 0.)],
                    vec![c(0., 0.), c(h.cos(), h.sin())],
                ]
            }
            QuantumGate::Phase(t) => vec![
                vec![c(1., 0.), c(0., 0.)],
                vec![c(0., 0.), c(t.cos(), t.sin())],
            ],
            QuantumGate::SWAP => vec![
                vec![c(1., 0.), c(0., 0.), c(0., 0.), c(0., 0.)],
                vec![c(0., 0.), c(0., 0.), c(1., 0.), c(0., 0.)],
                vec![c(0., 0.), c(1., 0.), c(0., 0.), c(0., 0.)],
                vec![c(0., 0.), c(0., 0.), c(0., 0.), c(1., 0.)],
            ],
        }
    }

    pub fn is_parameterized(&self) -> bool {
        matches!(
            self,
            QuantumGate::RX(_) | QuantumGate::RY(_) | QuantumGate::RZ(_) | QuantumGate::Phase(_)
        )
    }
    pub fn n_params(&self) -> usize {
        if self.is_parameterized() {
            1
        } else {
            0
        }
    }
}

// ── Circuit Layer & Circuit ───────────────────────────────────────────────────

/// A parallel layer: list of (gate, qubit_indices) applied simultaneously.
#[derive(Debug, Clone)]
pub struct CircuitLayer {
    pub ops: Vec<(QuantumGate, Vec<usize>)>,
}

impl CircuitLayer {
    pub fn new(ops: Vec<(QuantumGate, Vec<usize>)>) -> Self {
        Self { ops }
    }
    pub fn apply(&self, state: &mut QubitState) {
        for (gate, qubits) in &self.ops {
            state.apply_gate(gate, qubits);
        }
    }
}

/// Sequence of circuit layers.
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub n_qubits: usize,
    pub(crate) layers: Vec<CircuitLayer>,
}

impl QuantumCircuit {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            layers: Vec::new(),
        }
    }
    pub fn add_layer(&mut self, layer: CircuitLayer) {
        self.layers.push(layer);
    }
    pub fn run(&self, mut state: QubitState) -> QubitState {
        for layer in &self.layers {
            layer.apply(&mut state);
        }
        state
    }
    pub fn n_parameters(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.ops.iter())
            .map(|(g, _)| g.n_params())
            .sum()
    }
}

// ── Pauli Hamiltonian ─────────────────────────────────────────────────────────

/// H = Σ_k c_k P_k (sum of weighted Pauli string operators).
#[derive(Debug, Clone)]
pub struct PauliHamiltonian {
    pub terms: Vec<(f64, String)>,
}

impl PauliHamiltonian {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }
    pub fn add_term(&mut self, coeff: f64, pauli_string: &str) {
        self.terms.push((coeff, pauli_string.to_string()));
    }
    pub fn expectation(&self, state: &QubitState) -> f64 {
        self.terms
            .iter()
            .map(|(coeff, ps)| coeff * Self::pauli_exp(state, ps))
            .sum()
    }
    fn pauli_exp(state: &QubitState, pauli: &str) -> f64 {
        let mut scratch = state.clone();
        for (q, ch) in pauli.chars().enumerate() {
            match ch {
                'X' => scratch.apply_gate(&QuantumGate::X, &[q]),
                'Y' => scratch.apply_gate(&QuantumGate::Y, &[q]),
                'Z' => scratch.apply_gate(&QuantumGate::Z, &[q]),
                _ => {}
            }
        }
        state
            .amplitudes
            .iter()
            .zip(scratch.amplitudes.iter())
            .map(|(a, b)| a.conj().mul(*b).re)
            .sum()
    }
}

impl Default for PauliHamiltonian {
    fn default() -> Self {
        Self::new()
    }
}

// ── Ansatz Circuit (VQE) ──────────────────────────────────────────────────────

/// Hardware-efficient ansatz: alternating RY + CNOT layers.
/// Parameter count: `n_qubits × (depth + 1)`.
#[derive(Debug, Clone)]
pub struct AnsatzCircuit {
    pub n_qubits: usize,
    pub depth: usize,
}

impl AnsatzCircuit {
    pub fn new(n_qubits: usize, depth: usize) -> Self {
        Self { n_qubits, depth }
    }

    pub fn build(n_qubits: usize, depth: usize, params: &[f64]) -> Result<QuantumCircuit> {
        let expected = n_qubits * (depth + 1);
        if params.len() < expected {
            return Err(TensorError::invalid_argument(format!(
                "AnsatzCircuit::build: need {} params, got {}",
                expected,
                params.len()
            )));
        }
        let mut circuit = QuantumCircuit::new(n_qubits);
        let mut p_idx = 0usize;
        for layer_idx in 0..=depth {
            let ry_ops: Vec<(QuantumGate, Vec<usize>)> = (0..n_qubits)
                .map(|q| (QuantumGate::RY(params[p_idx + q]), vec![q]))
                .collect();
            circuit.add_layer(CircuitLayer::new(ry_ops));
            p_idx += n_qubits;
            if layer_idx < depth && n_qubits >= 2 {
                let cnots: Vec<(QuantumGate, Vec<usize>)> = (0..n_qubits - 1)
                    .map(|q| (QuantumGate::CNOT, vec![q, q + 1]))
                    .collect();
                circuit.add_layer(CircuitLayer::new(cnots));
            }
        }
        Ok(circuit)
    }
}

// ── VQE Optimizer ─────────────────────────────────────────────────────────────

/// VQE optimizer using the parameter-shift rule.
/// ∂⟨H⟩/∂θ_k = [⟨H⟩(θ_k + π/2) − ⟨H⟩(θ_k − π/2)] / 2
#[derive(Debug)]
pub struct VqeOptimizer {
    pub n_qubits: usize,
    pub depth: usize,
    pub params: Vec<f64>,
    pub hamiltonian: PauliHamiltonian,
}

impl VqeOptimizer {
    pub fn new(n_qubits: usize, depth: usize, hamiltonian: PauliHamiltonian, seed: u64) -> Self {
        let n_params = n_qubits * (depth + 1);
        let mut rng = StdRng::seed_from_u64(seed);
        let params: Vec<f64> = (0..n_params)
            .map(|_| rng.random_range(0.0..std::f64::consts::TAU))
            .collect();
        Self {
            n_qubits,
            depth,
            params,
            hamiltonian,
        }
    }

    pub fn energy(&self, params: &[f64]) -> f64 {
        match AnsatzCircuit::build(self.n_qubits, self.depth, params) {
            Ok(circuit) => {
                let state = circuit.run(QubitState::new(self.n_qubits));
                self.hamiltonian.expectation(&state)
            }
            Err(_) => f64::INFINITY,
        }
    }

    pub fn gradient(&self, params: &[f64]) -> Vec<f64> {
        let shift = std::f64::consts::FRAC_PI_2;
        (0..params.len())
            .map(|k| {
                let mut pp = params.to_vec();
                let mut pm = params.to_vec();
                pp[k] += shift;
                pm[k] -= shift;
                (self.energy(&pp) - self.energy(&pm)) / 2.0
            })
            .collect()
    }

    pub fn optimize(&mut self, n_steps: usize, lr: f64) -> Vec<f64> {
        for _ in 0..n_steps {
            let grads = self.gradient(&self.params.clone());
            for (p, g) in self.params.iter_mut().zip(grads.iter()) {
                *p -= lr * g;
            }
        }
        self.params.clone()
    }
}

// ── QAOA ──────────────────────────────────────────────────────────────────────

/// Quantum Approximate Optimization Algorithm circuit builder.
#[derive(Debug, Clone)]
pub struct Qaoa {
    pub n_qubits: usize,
}

impl Qaoa {
    pub fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }

    pub fn build_circuit(
        &self,
        n_qubits: usize,
        p_layers: usize,
        gamma: &[f64],
        beta: &[f64],
    ) -> Result<QuantumCircuit> {
        if gamma.len() < p_layers || beta.len() < p_layers {
            return Err(TensorError::invalid_argument(
                "QAOA: gamma and beta must have length >= p_layers".to_string(),
            ));
        }
        let mut circuit = QuantumCircuit::new(n_qubits);
        circuit.add_layer(CircuitLayer::new(
            (0..n_qubits).map(|q| (QuantumGate::H, vec![q])).collect(),
        ));
        for p in 0..p_layers {
            // Problem unitary: RZ(γ)
            circuit.add_layer(CircuitLayer::new(
                (0..n_qubits)
                    .map(|q| (QuantumGate::RZ(gamma[p]), vec![q]))
                    .collect(),
            ));
            if n_qubits >= 2 {
                circuit.add_layer(CircuitLayer::new(
                    (0..n_qubits - 1)
                        .map(|q| (QuantumGate::CNOT, vec![q, q + 1]))
                        .collect(),
                ));
                circuit.add_layer(CircuitLayer::new(
                    (1..n_qubits)
                        .map(|q| (QuantumGate::RZ(2.0 * gamma[p]), vec![q]))
                        .collect(),
                ));
                circuit.add_layer(CircuitLayer::new(
                    (0..n_qubits - 1)
                        .map(|q| (QuantumGate::CNOT, vec![q, q + 1]))
                        .collect(),
                ));
            }
            // Mixer: RX(β)
            circuit.add_layer(CircuitLayer::new(
                (0..n_qubits)
                    .map(|q| (QuantumGate::RX(beta[p]), vec![q]))
                    .collect(),
            ));
        }
        Ok(circuit)
    }
}

// ── Cost Hamiltonian ──────────────────────────────────────────────────────────

/// Encode combinatorial problems as Pauli Hamiltonians.
#[derive(Debug, Clone)]
pub struct CostHamiltonian;

impl CostHamiltonian {
    /// MaxCut Hamiltonian: H = -Σ_{i<j} w_{ij}/2 Z_i Z_j
    pub fn maxcut_hamiltonian(adj: &[Vec<f64>]) -> PauliHamiltonian {
        let n = adj.len();
        let mut h = PauliHamiltonian::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let w = adj[i][j];
                if w.abs() > 1e-12 {
                    let mut ps = vec!['I'; n];
                    ps[i] = 'Z';
                    ps[j] = 'Z';
                    h.add_term(-w / 2.0, &ps.into_iter().collect::<String>());
                }
            }
        }
        h
    }
}

// ── Data Encoding Layer ───────────────────────────────────────────────────────

/// Encode classical data into quantum rotation angles.
#[derive(Debug, Clone)]
pub struct DataEncodingLayer;

impl DataEncodingLayer {
    /// Normalize `x` then map amplitudes to arccos angles in [0, π].
    pub fn amplitude_encoding(x: &[f64], n_qubits: usize) -> Vec<f64> {
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        let mut angles = vec![0.0f64; n_qubits];
        for i in 0..n_qubits.min(x.len()) {
            angles[i] = (x[i] / norm).clamp(-1.0, 1.0).acos();
        }
        angles
    }

    /// Scale `x` to angles via `π·tanh(x)`.
    pub fn angle_encoding(x: &[f64], n_qubits: usize) -> Vec<f64> {
        let mut angles = vec![0.0f64; n_qubits];
        for (i, &v) in x.iter().take(n_qubits).enumerate() {
            angles[i] = std::f64::consts::PI * v.tanh();
        }
        angles
    }
}

// ── Quantum Layer ─────────────────────────────────────────────────────────────

/// Trainable quantum layer: encode x as RY angles, apply ansatz, return ⟨Z_q⟩.
#[derive(Debug)]
pub struct QuantumLayer {
    pub n_qubits: usize,
    pub depth: usize,
    pub params: Vec<f64>,
}

impl QuantumLayer {
    pub fn new(n_qubits: usize, depth: usize, seed: u64) -> Result<Self> {
        let n_params = n_qubits * (depth + 1);
        let mut rng = StdRng::seed_from_u64(seed);
        let params: Vec<f64> = (0..n_params)
            .map(|_| rng.random_range(0.0..std::f64::consts::TAU))
            .collect();
        Ok(Self {
            n_qubits,
            depth,
            params,
        })
    }

    /// Encode x, run parameterized circuit, return ⟨Z_q⟩ for each qubit.
    pub fn forward(&self, x: &[f64], params: &[f64]) -> Result<Vec<f64>> {
        let angles = DataEncodingLayer::angle_encoding(x, self.n_qubits);
        let mut state = QubitState::new(self.n_qubits);
        let enc: Vec<(QuantumGate, Vec<usize>)> = angles
            .iter()
            .enumerate()
            .map(|(q, &a)| (QuantumGate::RY(a), vec![q]))
            .collect();
        CircuitLayer::new(enc).apply(&mut state);
        let state = AnsatzCircuit::build(self.n_qubits, self.depth, params)?.run(state);
        Ok((0..self.n_qubits).map(|q| state.expectation_z(q)).collect())
    }
}

// ── Hybrid Quantum-Classical Model ────────────────────────────────────────────

/// Quantum feature map followed by a classical ReLU linear head.
#[derive(Debug)]
pub struct HybridQuantumClassical {
    pub quantum_layer: QuantumLayer,
    pub mlp_weights: Vec<Vec<f64>>,
    pub mlp_bias: Vec<f64>,
    pub output_dim: usize,
}

impl HybridQuantumClassical {
    pub fn new(n_qubits: usize, depth: usize, output_dim: usize, seed: u64) -> Result<Self> {
        let quantum_layer = QuantumLayer::new(n_qubits, depth, seed)?;
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
        let scale = (2.0 / n_qubits as f64).sqrt();
        let mlp_weights: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| {
                (0..n_qubits)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        Ok(Self {
            quantum_layer,
            mlp_weights,
            mlp_bias: vec![0.0; output_dim],
            output_dim,
        })
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        let feats = self
            .quantum_layer
            .forward(x, &self.quantum_layer.params.clone())?;
        Ok(self
            .mlp_weights
            .iter()
            .zip(self.mlp_bias.iter())
            .map(|(row, &b)| {
                (row.iter()
                    .zip(feats.iter())
                    .map(|(&w, &f)| w * f)
                    .sum::<f64>()
                    + b)
                    .max(0.0)
            })
            .collect())
    }
}

// ── Quantum Kernel ────────────────────────────────────────────────────────────

/// Quantum kernel k(x1, x2) = |⟨φ(x1)|φ(x2)⟩|² using angle-encoded feature maps.
#[derive(Debug, Clone)]
pub struct QuantumKernel {
    pub n_qubits: usize,
}

impl QuantumKernel {
    pub fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }

    fn feature_state(&self, x: &[f64]) -> QubitState {
        let angles = DataEncodingLayer::angle_encoding(x, self.n_qubits);
        let mut state = QubitState::new(self.n_qubits);
        for (q, &a) in angles.iter().enumerate() {
            state.apply_gate(&QuantumGate::RY(a), &[q]);
        }
        state
    }

    /// Compute k(x1, x2) ∈ [0, 1].
    pub fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let s1 = self.feature_state(x1);
        let s2 = self.feature_state(x2);
        s1.amplitudes
            .iter()
            .zip(s2.amplitudes.iter())
            .fold(Complex64::zero(), |acc, (a, b)| acc.add(a.conj().mul(*b)))
            .abs_sq()
    }
}

// ── Quantum SVM ───────────────────────────────────────────────────────────────

/// SVM with a quantum kernel for binary classification (±1 labels).
#[derive(Debug)]
pub struct QuantumSvm {
    pub kernel: QuantumKernel,
    pub support_vectors: Vec<Vec<f64>>,
    pub alphas: Vec<f64>,
    pub bias: f64,
    pub labels: Vec<f64>,
}

impl QuantumSvm {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            kernel: QuantumKernel::new(n_qubits),
            support_vectors: Vec::new(),
            alphas: Vec::new(),
            bias: 0.0,
            labels: Vec::new(),
        }
    }

    pub fn fit(&mut self, x_train: &[Vec<f64>], y: &[f64]) {
        let n = x_train.len();
        let mut gram = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let k = self.kernel.compute(&x_train[i], &x_train[j]);
                gram[i][j] = k;
                gram[j][i] = k;
            }
        }
        let (lr, c) = (0.1, 1.0);
        let mut alphas = vec![0.0f64; n];
        for _ in 0..100 {
            for i in 0..n {
                let grad = 1.0 - y[i] * (0..n).map(|j| alphas[j] * y[j] * gram[i][j]).sum::<f64>();
                alphas[i] = (alphas[i] + lr * grad).clamp(0.0, c);
            }
        }
        let sv_mask: Vec<bool> = alphas.iter().map(|&a| a > 1e-4).collect();
        let n_sv = sv_mask.iter().filter(|&&b| b).count();
        let bias_sum: f64 = (0..n)
            .filter(|&i| sv_mask[i])
            .map(|i| y[i] - (0..n).map(|j| alphas[j] * y[j] * gram[i][j]).sum::<f64>())
            .sum();
        self.bias = if n_sv > 0 {
            bias_sum / n_sv as f64
        } else {
            0.0
        };
        self.support_vectors = x_train.to_vec();
        self.alphas = alphas;
        self.labels = y.to_vec();
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let d: f64 = self
            .support_vectors
            .iter()
            .zip(self.alphas.iter())
            .zip(self.labels.iter())
            .map(|((sv, &a), &l)| a * l * self.kernel.compute(sv, x))
            .sum::<f64>()
            + self.bias;
        if d >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

// ── Grover's Algorithm ────────────────────────────────────────────────────────

/// Grover's amplitude amplification simulation.
#[derive(Debug)]
pub struct GroverSearch {
    pub n_qubits: usize,
}

impl GroverSearch {
    pub fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }

    pub fn search(
        &self,
        n_qubits: usize,
        oracle_fn: &dyn Fn(usize) -> bool,
        n_iterations: usize,
    ) -> Vec<f64> {
        let dim = 1usize << n_qubits;
        let mut state = QubitState::new(n_qubits);
        for q in 0..n_qubits {
            state.apply_gate(&QuantumGate::H, &[q]);
        }
        for _ in 0..n_iterations {
            for idx in 0..dim {
                if oracle_fn(idx) {
                    state.amplitudes[idx] = state.amplitudes[idx].scale(-1.0);
                }
            }
            let mean = state
                .amplitudes
                .iter()
                .copied()
                .fold(Complex64::zero(), |acc, a| acc.add(a))
                .scale(1.0 / dim as f64);
            for amp in state.amplitudes.iter_mut() {
                *amp = mean.scale(2.0).add(amp.scale(-1.0));
            }
        }
        state.measure_probs()
    }
}

// ── Quantum Annealing Simulator ───────────────────────────────────────────────

/// Simulated quantum annealing for the transverse-field Ising model.
#[derive(Debug)]
pub struct QuantumAnnealingSimulator {
    pub transverse_field: f64,
}

impl QuantumAnnealingSimulator {
    pub fn new(transverse_field: f64) -> Self {
        Self { transverse_field }
    }

    pub fn anneal(
        &self,
        h: &[f64],
        j: &[Vec<f64>],
        t_schedule: &[f64],
        rng: &mut StdRng,
    ) -> Vec<i32> {
        let n = h.len();
        let mut spins: Vec<i32> = (0..n)
            .map(|_| if rng.random_bool(0.5) { 1 } else { -1 })
            .collect();
        let mut best = spins.clone();
        let mut best_e = self.ising_energy(&spins, h, j);
        for &temp in t_schedule {
            for i in 0..n {
                let de = self.flip_delta(&spins, h, j, i);
                let p = if de < 0.0 {
                    1.0
                } else {
                    let qc = self.transverse_field / (temp + 1e-12);
                    (-de / (temp + 1e-12) + qc).exp().min(1.0)
                };
                if rng.random_range(0.0..1.0_f64) < p {
                    spins[i] = -spins[i];
                }
            }
            let e = self.ising_energy(&spins, h, j);
            if e < best_e {
                best_e = e;
                best = spins.clone();
            }
        }
        best
    }

    fn ising_energy(&self, spins: &[i32], h: &[f64], j: &[Vec<f64>]) -> f64 {
        let n = spins.len();
        spins
            .iter()
            .zip(h.iter())
            .map(|(&s, &hi)| -(s as f64) * hi)
            .sum::<f64>()
            + (0..n)
                .flat_map(|i| {
                    ((i + 1)..n).map(move |k| -(spins[i] as f64) * (spins[k] as f64) * j[i][k])
                })
                .sum::<f64>()
    }

    fn flip_delta(&self, spins: &[i32], h: &[f64], j: &[Vec<f64>], i: usize) -> f64 {
        let n = spins.len();
        let lf = h[i]
            + (0..n)
                .map(|k| {
                    if k != i {
                        j[i][k] * spins[k] as f64
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
        2.0 * spins[i] as f64 * lf
    }
}

// ── Parameter-Shift Gradient ──────────────────────────────────────────────────

/// Exact parameter-shift gradient: ∂f/∂θ_k = [f(θ+π/2·ê_k) − f(θ−π/2·ê_k)] / 2
#[derive(Debug, Clone)]
pub struct ParameterShiftGradient {
    pub shift: f64,
}

impl ParameterShiftGradient {
    pub fn new() -> Self {
        Self {
            shift: std::f64::consts::FRAC_PI_2,
        }
    }
    pub fn with_shift(shift: f64) -> Self {
        Self { shift }
    }

    pub fn gradient(&self, f: &dyn Fn(&[f64]) -> f64, params: &[f64]) -> Vec<f64> {
        let s = self.shift;
        (0..params.len())
            .map(|k| {
                let mut pp = params.to_vec();
                let mut pm = params.to_vec();
                pp[k] += s;
                pm[k] -= s;
                (f(&pp) - f(&pm)) / 2.0
            })
            .collect()
    }
}

impl Default for ParameterShiftGradient {
    fn default() -> Self {
        Self::new()
    }
}

// ── Natural Gradient for QNNs ─────────────────────────────────────────────────

/// Quantum natural gradient using diagonal Fisher information approximation.
#[derive(Debug)]
pub struct NaturalGradientQnn {
    pub psg: ParameterShiftGradient,
    pub reg: f64,
}

impl NaturalGradientQnn {
    pub fn new(reg: f64) -> Self {
        Self {
            psg: ParameterShiftGradient::new(),
            reg,
        }
    }

    pub fn natural_gradient(&self, f: &dyn Fn(&[f64]) -> f64, params: &[f64]) -> Vec<f64> {
        let grad = self.psg.gradient(f, params);
        // Diagonal Fisher: F_kk ≈ g_k² + reg; natural gradient = g / F_diag
        grad.iter().map(|&g| g / (g * g + self.reg)).collect()
    }
}

// ── Quantum Backpropagation (Adjoint Differentiation) ─────────────────────────

/// Adjoint differentiation via parameter-shift rule for ansatz circuits.
#[derive(Debug)]
pub struct QuantumBackpropagation {
    pub n_qubits: usize,
}

impl QuantumBackpropagation {
    pub fn new(n_qubits: usize) -> Self {
        Self { n_qubits }
    }

    pub fn gradient(
        &self,
        params: &[f64],
        hamiltonian: &PauliHamiltonian,
        depth: usize,
    ) -> Vec<f64> {
        let psg = ParameterShiftGradient::new();
        let nq = self.n_qubits;
        psg.gradient(
            &|p| match AnsatzCircuit::build(nq, depth, p) {
                Ok(c) => hamiltonian.expectation(&c.run(QubitState::new(nq))),
                Err(_) => 0.0,
            },
            params,
        )
    }
}

// ── Depolarizing Noise ────────────────────────────────────────────────────────

/// Apply depolarizing noise (random Pauli error per qubit with probability p).
#[derive(Debug, Clone)]
pub struct DepolarizingNoise {
    pub p: f64,
}

impl DepolarizingNoise {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
        }
    }

    pub fn apply(&self, state: &mut QubitState, rng: &mut StdRng) {
        for q in 0..state.n_qubits {
            if rng.random_range(0.0..1.0_f64) < self.p {
                match rng.random_range(0..3u32) {
                    0 => state.apply_gate(&QuantumGate::X, &[q]),
                    1 => state.apply_gate(&QuantumGate::Y, &[q]),
                    _ => state.apply_gate(&QuantumGate::Z, &[q]),
                }
            }
        }
    }
}

// ── Bit Flip Channel ──────────────────────────────────────────────────────────

/// Bit flip error channel: flip qubit `q` with probability `p`.
#[derive(Debug, Clone)]
pub struct BitFlipChannel {
    pub p: f64,
}

impl BitFlipChannel {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
        }
    }

    pub fn apply(&self, state: QubitState, qubit: usize, p: f64, rng: &mut StdRng) -> QubitState {
        let mut s = state;
        if rng.random_range(0.0..1.0_f64) < p {
            s.apply_gate(&QuantumGate::X, &[qubit]);
        }
        s
    }
}

// ── Zero-Noise Extrapolation ──────────────────────────────────────────────────

/// ZNE: extrapolate to zero noise by Lagrange polynomial interpolation at c=0.
#[derive(Debug, Clone)]
pub struct ZeroNoiseExtrapolation;

impl ZeroNoiseExtrapolation {
    pub fn extrapolate(results: &[(f64, f64)]) -> f64 {
        match results.len() {
            0 => 0.0,
            1 => results[0].1,
            2 => {
                let (c1, f1) = results[0];
                let (c2, f2) = results[1];
                let d = c2 - c1;
                if d.abs() < 1e-12 {
                    (f1 + f2) / 2.0
                } else {
                    (c2 * f1 - c1 * f2) / d
                }
            }
            _ => {
                let n = results.len();
                let xs: Vec<f64> = results.iter().map(|&(c, _)| c).collect();
                let mut p: Vec<f64> = results.iter().map(|&(_, f)| f).collect();
                for k in 1..n {
                    for i in 0..(n - k) {
                        let d = xs[i] - xs[i + k];
                        if d.abs() > 1e-12 {
                            p[i] = ((0.0 - xs[i + k]) * p[i] - (0.0 - xs[i]) * p[i + 1]) / d;
                        }
                    }
                }
                p[0]
            }
        }
    }
}

// ── Readout Error Mitigation ──────────────────────────────────────────────────

/// Calibration-matrix readout error mitigation.
#[derive(Debug, Clone)]
pub struct ReadoutErrorMitigation {
    pub n_qubits: usize,
    pub cal_matrix: Vec<Vec<Vec<f64>>>,
}

impl ReadoutErrorMitigation {
    /// Build from per-qubit error rates: `p01 = P(1|0)`, `p10 = P(0|1)`.
    pub fn calibrate(n_qubits: usize, p01: f64, p10: f64) -> Self {
        let cal_matrix = vec![vec![vec![1.0 - p01, p10], vec![p01, 1.0 - p10]]; n_qubits];
        Self {
            n_qubits,
            cal_matrix,
        }
    }

    pub fn mitigate(&self, probs: &[f64]) -> Vec<f64> {
        let dim = 1usize << self.n_qubits;
        if probs.len() != dim {
            return probs.to_vec();
        }
        if self.n_qubits == 1 {
            let a = &self.cal_matrix[0];
            let (p00, p01, p10, p11) = (a[0][0], a[0][1], a[1][0], a[1][1]);
            let det = p00 * p11 - p01 * p10;
            if det.abs() < 1e-12 {
                return probs.to_vec();
            }
            let (m0, m1) = (
                (p11 * probs[0] - p01 * probs[1]) / det,
                (-p10 * probs[0] + p00 * probs[1]) / det,
            );
            let (m0c, m1c) = (m0.max(0.0), m1.max(0.0));
            let t = m0c + m1c;
            if t > 0.0 {
                vec![m0c / t, m1c / t]
            } else {
                vec![0.5, 0.5]
            }
        } else {
            let mut mitigated = probs.to_vec();
            for q in 0..self.n_qubits {
                let bit = self.n_qubits - 1 - q;
                let a = &self.cal_matrix[q.min(self.cal_matrix.len() - 1)];
                let (p00, p01, p10, p11) = (a[0][0], a[0][1], a[1][0], a[1][1]);
                let det = p00 * p11 - p01 * p10;
                if det.abs() < 1e-12 {
                    continue;
                }
                let mut new_m = mitigated.clone();
                let stride = 1usize << bit;
                let block = stride * 2;
                let mut base = 0;
                while base < dim {
                    for inner in 0..stride {
                        let i0 = base + inner;
                        let i1 = base + inner + stride;
                        let (v0, v1) = (mitigated[i0], mitigated[i1]);
                        new_m[i0] = ((p11 * v0 - p01 * v1) / det).max(0.0);
                        new_m[i1] = ((-p10 * v0 + p00 * v1) / det).max(0.0);
                    }
                    base += block;
                }
                mitigated = new_m;
            }
            let t: f64 = mitigated.iter().sum();
            if t > 0.0 {
                mitigated.iter().map(|&p| p / t).collect()
            } else {
                vec![1.0 / dim as f64; dim]
            }
        }
    }
}

// ── Noisy Quantum Simulator ───────────────────────────────────────────────────

/// Combine gate noise (depolarizing) and readout noise for realistic simulation.
#[derive(Debug)]
pub struct NoisyQuantumSimulator {
    pub gate_noise: DepolarizingNoise,
    pub readout: ReadoutErrorMitigation,
    pub seed: u64,
}

impl NoisyQuantumSimulator {
    pub fn new(n_qubits: usize, p_gate: f64, p01: f64, p10: f64, seed: u64) -> Self {
        Self {
            gate_noise: DepolarizingNoise::new(p_gate),
            readout: ReadoutErrorMitigation::calibrate(n_qubits, p01, p10),
            seed,
        }
    }

    pub fn run_noisy(&self, circuit: &QuantumCircuit) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut state = QubitState::new(circuit.n_qubits);
        for layer in &circuit.layers {
            layer.apply(&mut state);
            self.gate_noise.apply(&mut state, &mut rng);
        }
        let noisy = state.measure_probs();
        let dim = noisy.len();
        let n = circuit.n_qubits;
        let p01 = self
            .readout
            .cal_matrix
            .first()
            .map(|a| a[1][0])
            .unwrap_or(0.0);
        let p10 = self
            .readout
            .cal_matrix
            .first()
            .map(|a| a[0][1])
            .unwrap_or(0.0);
        let mut out = vec![0.0f64; dim];
        for (ii, &p) in noisy.iter().enumerate() {
            for oi in 0..dim {
                let mut prob = p;
                for bit in 0..n {
                    let ib = (ii >> (n - 1 - bit)) & 1;
                    let ob = (oi >> (n - 1 - bit)) & 1;
                    prob *= match (ib, ob) {
                        (0, 0) => 1.0 - p01,
                        (0, 1) => p01,
                        (1, 0) => p10,
                        (1, 1) => 1.0 - p10,
                        _ => 1.0,
                    };
                }
                out[oi] += prob;
            }
        }
        let t: f64 = out.iter().sum();
        if t > 0.0 {
            out.iter().map(|&p| p / t).collect()
        } else {
            vec![1.0 / dim as f64; dim]
        }
    }
}
