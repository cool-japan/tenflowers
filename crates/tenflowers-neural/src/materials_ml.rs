//! ML for Materials Science — Comprehensive implementation.
//!
//! Provides ML algorithms specifically designed for materials science applications:
//!
//! - [`CrystalGraph`]: Crystal structure as a graph with periodic boundary conditions.
//! - [`Cgcnn`]: Crystal Graph Convolutional Neural Network (Xie & Grossman 2018).
//! - [`SchNetMaterials`]: SchNet for materials property prediction.
//! - [`MattersimModel`]: Universal materials embedder.
//! - [`PhasePredictor`]: Crystal phase and stability prediction.
//! - [`MaterialPropertyPredictor`]: Multi-task materials property prediction.
//! - \[`CrystalSymmetry`\]: Symmetry analysis and lattice system detection.
//! - \[`MaterialAugmentation`\]: Data augmentation for crystal structures.
//! - \[`GenerativeMaterials`\]: Generative models for crystal generation.
//! - \[`MaterialMetrics`\]: Evaluation metrics for materials ML.
//!
//! All fallible operations return `Result<_, TensorError>`.
//! No `unsafe` code; no `unwrap()` calls.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

// ─────────────────────────────────────────────────────────────────────────────
// Math utilities (local to this module)
// ─────────────────────────────────────────────────────────────────────────────

/// ReLU activation (f32).
#[inline]
fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

/// Sigmoid activation (f32).
#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    let xc = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-xc).exp())
}

/// Dot product of two f32 slices.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Linear layer: mat [rows × cols] · v [cols] + bias [rows] → out [rows].
fn linear_f32(mat: &[f32], bias: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &mat[r * cols..(r + 1) * cols];
            dot_f32(row, v) + bias.get(r).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Euclidean distance between two 3D points.
#[inline]
fn dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Vector normalisation (L2). Returns zero vector if norm is ~0.
fn normalize_f32(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 CrystalGraph
// ─────────────────────────────────────────────────────────────────────────────

/// A single atom in a crystal structure.
#[derive(Debug, Clone)]
pub struct Atom {
    /// Element symbol (e.g. "Fe", "O").
    pub element: String,
    /// Fractional or Cartesian position in the unit cell.
    pub position: [f32; 3],
    /// Atomic number (1 = H, 8 = O, 26 = Fe, …).
    pub atomic_number: usize,
    /// Pauling electronegativity.
    pub electronegativity: f32,
    /// Covalent or ionic radius in Å.
    pub radius: f32,
}

/// A bond between two atoms, including the periodic image offset.
///
/// `image` encodes which periodic image of atom `j` the bond connects to:
/// the actual position of the bonded atom is `pos_j + image[0]*a + image[1]*b + image[2]*c`.
#[derive(Debug, Clone)]
pub struct CrystalBond {
    /// Index of the first atom.
    pub i: usize,
    /// Index of the second atom (may be in a periodic image).
    pub j: usize,
    /// Bond distance in Å.
    pub distance: f32,
    /// Periodic image offset vector (fractional lattice coordinates).
    pub image: [i8; 3],
}

/// Crystal structure represented as a graph.
///
/// `lattice` is a 3×3 matrix where each row is a lattice vector (in Å).
#[derive(Debug, Clone)]
pub struct CrystalGraph {
    /// Atoms in the unit cell.
    pub atoms: Vec<Atom>,
    /// Bonds (including periodic images within cutoff).
    pub bonds: Vec<CrystalBond>,
    /// Lattice matrix: rows are lattice vectors a, b, c.
    pub lattice: [[f32; 3]; 3],
}

/// Build a neighbor graph from a list of atoms and a lattice, considering periodic images.
///
/// All pairs whose distance (within ±1 image cell in each direction) are ≤ `cutoff` are connected.
pub fn build_neighbor_graph(atoms: &[Atom], lattice: &[[f32; 3]; 3], cutoff: f32) -> CrystalGraph {
    let mut bonds = Vec::new();
    let n = atoms.len();

    for i in 0..n {
        for j in 0..n {
            // Iterate over periodic images (-1, 0, +1) in each direction
            for &ia in &[-1i8, 0, 1] {
                for &ib in &[-1i8, 0, 1] {
                    for &ic in &[-1i8, 0, 1] {
                        // Skip self in zero image
                        if i == j && ia == 0 && ib == 0 && ic == 0 {
                            continue;
                        }
                        // Cartesian offset from image
                        let offset = [
                            ia as f32 * lattice[0][0]
                                + ib as f32 * lattice[1][0]
                                + ic as f32 * lattice[2][0],
                            ia as f32 * lattice[0][1]
                                + ib as f32 * lattice[1][1]
                                + ic as f32 * lattice[2][1],
                            ia as f32 * lattice[0][2]
                                + ib as f32 * lattice[1][2]
                                + ic as f32 * lattice[2][2],
                        ];
                        let pj = [
                            atoms[j].position[0] + offset[0],
                            atoms[j].position[1] + offset[1],
                            atoms[j].position[2] + offset[2],
                        ];
                        let d = dist3(&atoms[i].position, &pj);
                        if d <= cutoff {
                            bonds.push(CrystalBond {
                                i,
                                j,
                                distance: d,
                                image: [ia, ib, ic],
                            });
                        }
                    }
                }
            }
        }
    }

    CrystalGraph {
        atoms: atoms.to_vec(),
        bonds,
        lattice: *lattice,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 CGCNN — Crystal Graph Convolutional Neural Network
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CGCNN.
#[derive(Debug, Clone)]
pub struct CgcnnConfig {
    /// Atom feature embedding dimension.
    pub atom_fea_len: usize,
    /// Number of graph convolution layers.
    pub n_conv: usize,
    /// Hidden layer size after pooling.
    pub n_h: usize,
    /// Output dimension (number of predicted properties).
    pub output_dim: usize,
}

/// Featurizes atoms by looking up learnable element embeddings.
#[derive(Debug, Clone)]
pub struct AtomFeaturizer {
    /// Output embedding dimension.
    pub embed_dim: usize,
    /// Embedding table: element_table\[atomic_number\] → feature vector.
    /// Indexed by atomic number (1-indexed; entry 0 unused).
    pub element_table: Vec<Vec<f32>>,
}

impl AtomFeaturizer {
    /// Create a new `AtomFeaturizer` with randomly initialized embeddings.
    pub fn new(embed_dim: usize, max_atomic_num: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let element_table = (0..=max_atomic_num)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.random_range(-0.1_f32..0.1_f32))
                    .collect()
            })
            .collect();
        AtomFeaturizer {
            embed_dim,
            element_table,
        }
    }

    /// Featurize a single atom by looking up its atomic number.
    pub fn featurize(&self, atom: &Atom) -> Vec<f32> {
        let idx = atom.atomic_number.min(self.element_table.len() - 1);
        self.element_table[idx].clone()
    }
}

/// Featurizes bond distances using a Gaussian basis expansion.
#[derive(Debug, Clone)]
pub struct EdgeFeaturizer {
    /// Number of Gaussian basis functions.
    pub n_gaussians: usize,
    /// Minimum distance for basis centers.
    pub dmin: f32,
    /// Maximum distance for basis centers.
    pub dmax: f32,
}

impl EdgeFeaturizer {
    /// Create a new `EdgeFeaturizer`.
    pub fn new(n_gaussians: usize, dmin: f32, dmax: f32) -> Self {
        EdgeFeaturizer {
            n_gaussians,
            dmin,
            dmax,
        }
    }

    /// Expand bond distance into a Gaussian basis vector.
    pub fn featurize(&self, distance: f32) -> Vec<f32> {
        let step = if self.n_gaussians > 1 {
            (self.dmax - self.dmin) / (self.n_gaussians as f32 - 1.0)
        } else {
            1.0
        };
        let width = step;
        (0..self.n_gaussians)
            .map(|k| {
                let center = self.dmin + k as f32 * step;
                let diff = distance - center;
                (-diff * diff / (2.0 * width * width)).exp()
            })
            .collect()
    }
}

/// A single CGCNN graph convolution layer.
///
/// Implements the gated convolution from Xie & Grossman (2018):
///   z_i = atom_i + Σ_j softplus( W_core·[atom_i, atom_j, bond_ij] + b_core )
///               ⊙ sigmoid( W_filter·[atom_i, atom_j, bond_ij] + b_filter )
#[derive(Debug, Clone)]
pub struct CgcnnLayer {
    /// Core weight matrix (atom_fea_len × input_concat_dim).
    pub weight_core: Vec<f32>,
    /// Filter (gate) weight matrix (atom_fea_len × input_concat_dim).
    pub weight_filter: Vec<f32>,
    /// Bias for core pathway.
    pub bias_core: Vec<f32>,
    /// Bias for filter pathway.
    pub bias_filter: Vec<f32>,
    /// Output (atom feature) dimension.
    pub atom_fea_len: usize,
    /// Bond feature dimension.
    pub bond_fea_len: usize,
}

impl CgcnnLayer {
    /// Create a new CGCNN layer with Xavier-initialized weights.
    pub fn new(atom_fea_len: usize, bond_fea_len: usize) -> Self {
        let input_dim = 2 * atom_fea_len + bond_fea_len;
        let fan = (atom_fea_len + input_dim) as f32;
        let scale = (6.0_f32 / fan).sqrt();
        let mut rng = StdRng::seed_from_u64(123);
        let mut rand_vec =
            |n: usize| -> Vec<f32> { (0..n).map(|_| rng.random_range(-scale..scale)).collect() };
        let weight_size = atom_fea_len * input_dim;
        CgcnnLayer {
            weight_core: rand_vec(weight_size),
            weight_filter: rand_vec(weight_size),
            bias_core: vec![0.0; atom_fea_len],
            bias_filter: vec![0.0; atom_fea_len],
            atom_fea_len,
            bond_fea_len,
        }
    }

    /// Apply one graph convolution step to atom features.
    pub fn forward(
        &self,
        atom_features: &[Vec<f32>],
        bonds: &[CrystalBond],
        bond_feats: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let n_atoms = atom_features.len();
        let input_dim = 2 * self.atom_fea_len + self.bond_fea_len;
        let mut new_feats: Vec<Vec<f32>> = atom_features.to_vec();

        for i in 0..n_atoms {
            let mut aggregated = vec![0.0f32; self.atom_fea_len];
            let mut count = 0usize;
            for (bond_idx, bond) in bonds.iter().enumerate() {
                if bond.i != i {
                    continue;
                }
                let j = bond.j;
                if j >= n_atoms {
                    continue;
                }
                let bf = bond_feats
                    .get(bond_idx)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.bond_fea_len]);
                // Concatenate [atom_i, atom_j, bond_ij]
                let mut concat = Vec::with_capacity(input_dim);
                concat.extend_from_slice(&atom_features[i]);
                concat.extend_from_slice(&atom_features[j]);
                concat.extend_from_slice(&bf);

                let core_out = linear_f32(
                    &self.weight_core,
                    &self.bias_core,
                    &concat,
                    self.atom_fea_len,
                    input_dim,
                );
                let filter_out = linear_f32(
                    &self.weight_filter,
                    &self.bias_filter,
                    &concat,
                    self.atom_fea_len,
                    input_dim,
                );

                for k in 0..self.atom_fea_len {
                    // softplus * sigmoid gating
                    let sp = (core_out[k].exp() + 1.0).ln();
                    let sg = sigmoid_f32(filter_out[k]);
                    aggregated[k] += sp * sg;
                }
                count += 1;
            }
            if count > 0 {
                let scale = 1.0 / count as f32;
                for k in 0..self.atom_fea_len {
                    new_feats[i][k] = atom_features[i][k] + aggregated[k] * scale;
                }
            }
        }
        new_feats
    }
}

/// CGCNN model for crystal property prediction.
#[derive(Debug, Clone)]
pub struct Cgcnn {
    /// Configuration.
    pub config: CgcnnConfig,
    /// Atom featurizer.
    pub atom_featurizer: AtomFeaturizer,
    /// Edge featurizer.
    pub edge_featurizer: EdgeFeaturizer,
    /// Graph convolution layers.
    pub conv_layers: Vec<CgcnnLayer>,
    /// Post-pooling FC weights (n_h × atom_fea_len).
    pub fc_weight: Vec<f32>,
    /// Post-pooling FC bias.
    pub fc_bias: Vec<f32>,
    /// Output layer weights (output_dim × n_h).
    pub out_weight: Vec<f32>,
    /// Output bias.
    pub out_bias: Vec<f32>,
}

impl Cgcnn {
    /// Create a new CGCNN model.
    pub fn new(config: CgcnnConfig) -> Self {
        let atom_featurizer = AtomFeaturizer::new(config.atom_fea_len, 118);
        let edge_featurizer = EdgeFeaturizer::new(41, 0.0, 8.0);
        let conv_layers: Vec<CgcnnLayer> = (0..config.n_conv)
            .map(|_| CgcnnLayer::new(config.atom_fea_len, 41))
            .collect();

        let mut rng = StdRng::seed_from_u64(7);
        let fc_scale = (2.0 / config.atom_fea_len as f32).sqrt();
        let fc_weight: Vec<f32> = (0..config.n_h * config.atom_fea_len)
            .map(|_| rng.random_range(-fc_scale..fc_scale))
            .collect();
        let fc_bias = vec![0.0; config.n_h];
        let out_scale = (2.0 / config.n_h as f32).sqrt();
        let out_weight: Vec<f32> = (0..config.output_dim * config.n_h)
            .map(|_| rng.random_range(-out_scale..out_scale))
            .collect();
        let out_bias = vec![0.0; config.output_dim];

        Cgcnn {
            config,
            atom_featurizer,
            edge_featurizer,
            conv_layers,
            fc_weight,
            fc_bias,
            out_weight,
            out_bias,
        }
    }

    /// Forward pass: given a crystal graph, return predicted property vector.
    ///
    /// Steps: featurize atoms → featurize bonds → n_conv graph convolutions
    ///       → mean-pool atom features → FC+ReLU → output layer.
    pub fn forward(&self, graph: &CrystalGraph) -> Vec<f32> {
        let n_atoms = graph.atoms.len();
        if n_atoms == 0 {
            return vec![0.0; self.config.output_dim];
        }

        // Featurize atoms and bonds
        let mut atom_features: Vec<Vec<f32>> = graph
            .atoms
            .iter()
            .map(|a| self.atom_featurizer.featurize(a))
            .collect();
        let bond_feats: Vec<Vec<f32>> = graph
            .bonds
            .iter()
            .map(|b| self.edge_featurizer.featurize(b.distance))
            .collect();

        // Graph convolutions
        for layer in &self.conv_layers {
            atom_features = layer.forward(&atom_features, &graph.bonds, &bond_feats);
        }

        // Mean pooling
        let mut pooled = vec![0.0f32; self.config.atom_fea_len];
        for feat in &atom_features {
            for (k, &v) in feat.iter().enumerate() {
                pooled[k] += v;
            }
        }
        let inv_n = 1.0 / n_atoms as f32;
        for v in &mut pooled {
            *v *= inv_n;
        }

        // FC layer with ReLU
        let h = linear_f32(
            &self.fc_weight,
            &self.fc_bias,
            &pooled,
            self.config.n_h,
            self.config.atom_fea_len,
        );
        let h: Vec<f32> = h.into_iter().map(relu_f32).collect();

        // Output layer
        linear_f32(
            &self.out_weight,
            &self.out_bias,
            &h,
            self.config.output_dim,
            self.config.n_h,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 SchNetMaterials
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for SchNet applied to materials.
#[derive(Debug, Clone)]
pub struct SchNetMaterialsConfig {
    /// Atom embedding / feature dimension.
    pub n_atom_basis: usize,
    /// Number of Gaussian RBF centers.
    pub n_gaussians: usize,
    /// Number of interaction blocks.
    pub n_interactions: usize,
    /// Interaction cutoff radius in Å.
    pub cutoff: f32,
}

/// Gaussian radial basis functions (smearing) for pairwise distances.
#[derive(Debug, Clone)]
pub struct GaussianSmearing {
    /// Centers of the Gaussian basis functions.
    pub offsets: Vec<f32>,
    /// Width (variance) parameter.
    pub width: f32,
}

impl GaussianSmearing {
    /// Create evenly spaced Gaussian centers from 0 to `cutoff`.
    pub fn new(n_gaussians: usize, cutoff: f32) -> Self {
        let step = cutoff / n_gaussians.max(1) as f32;
        let offsets: Vec<f32> = (0..n_gaussians).map(|k| k as f32 * step).collect();
        let width = step;
        GaussianSmearing { offsets, width }
    }

    /// Expand distance `d` into a Gaussian-smeared feature vector.
    pub fn expand(&self, d: f32) -> Vec<f32> {
        self.offsets
            .iter()
            .map(|&c| {
                let diff = d - c;
                (-diff * diff / (2.0 * self.width * self.width)).exp()
            })
            .collect()
    }
}

/// Continuous-filter convolution layer (cfconv) from SchNet.
///
/// Applies a filter network (2-layer MLP) to RBF-expanded distances, then
/// multiplies with atom features to produce messages.
#[derive(Debug, Clone)]
pub struct CfConv {
    /// Input atom feature channels.
    pub in_channels: usize,
    /// Output atom feature channels.
    pub out_channels: usize,
    /// Number of Gaussian basis functions.
    pub n_gaussians: usize,
    /// Filter network weights: RBF → n_filters → in_channels.
    /// Packed as [W1 (in_channels × n_gaussians), b1, W2 (in_channels × in_channels), b2].
    pub filter_weights: Vec<f32>,
}

impl CfConv {
    /// Create a new `CfConv` layer.
    pub fn new(in_channels: usize, out_channels: usize, n_gaussians: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(31);
        let scale1 = (2.0 / n_gaussians as f32).sqrt();
        let scale2 = (2.0 / in_channels as f32).sqrt();
        let w1_size = in_channels * n_gaussians;
        let w2_size = in_channels * in_channels;
        let total = w1_size + in_channels + w2_size + in_channels;
        let mut weights = Vec::with_capacity(total);
        for _ in 0..w1_size {
            weights.push(rng.random_range(-scale1..scale1));
        }
        weights.extend(vec![0.0f32; in_channels]); // b1
        for _ in 0..w2_size {
            weights.push(rng.random_range(-scale2..scale2));
        }
        weights.extend(vec![0.0f32; in_channels]); // b2
        CfConv {
            in_channels,
            out_channels,
            n_gaussians,
            filter_weights: weights,
        }
    }

    /// Compute filter vector for a given RBF-expanded distance.
    pub fn filter(&self, rbf: &[f32]) -> Vec<f32> {
        let n = self.n_gaussians;
        let c = self.in_channels;
        let w1 = &self.filter_weights[..c * n];
        let b1_start = c * n;
        let b1 = &self.filter_weights[b1_start..b1_start + c];
        let w2_start = b1_start + c;
        let w2 = &self.filter_weights[w2_start..w2_start + c * c];
        let b2_start = w2_start + c * c;
        let b2 = &self.filter_weights[b2_start..b2_start + c];

        let h1 = linear_f32(w1, b1, rbf, c, n);
        let h1: Vec<f32> = h1.into_iter().map(|x| x.max(0.0)).collect(); // shifted-softplus approx
        linear_f32(w2, b2, &h1, c, c)
    }
}

/// One SchNet interaction block.
#[derive(Debug, Clone)]
pub struct SchNetLayer {
    /// Continuous-filter convolution.
    pub cfconv: CfConv,
    /// Output linear layer weights (n_atom_basis × n_atom_basis).
    pub output_net: Vec<f32>,
    /// Output bias.
    pub output_bias: Vec<f32>,
}

impl SchNetLayer {
    /// Create a new SchNet interaction layer.
    pub fn new(n_atom_basis: usize, n_gaussians: usize) -> Self {
        let cfconv = CfConv::new(n_atom_basis, n_atom_basis, n_gaussians);
        let mut rng = StdRng::seed_from_u64(55);
        let scale = (2.0 / n_atom_basis as f32).sqrt();
        let output_net: Vec<f32> = (0..n_atom_basis * n_atom_basis)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        let output_bias = vec![0.0; n_atom_basis];
        SchNetLayer {
            cfconv,
            output_net,
            output_bias,
        }
    }
}

/// SchNet model for materials property prediction.
#[derive(Debug, Clone)]
pub struct SchNetMaterials {
    /// Configuration.
    pub config: SchNetMaterialsConfig,
    /// Gaussian RBF smearing.
    pub smearing: GaussianSmearing,
    /// Interaction layers.
    pub layers: Vec<SchNetLayer>,
    /// Atom embedding table (max_z × n_atom_basis).
    pub embed_table: Vec<Vec<f32>>,
    /// Readout linear weight (1 × n_atom_basis).
    pub readout_weight: Vec<f32>,
    /// Readout bias.
    pub readout_bias: f32,
}

impl SchNetMaterials {
    /// Create a new SchNet model for materials.
    pub fn new(config: SchNetMaterialsConfig) -> Self {
        let smearing = GaussianSmearing::new(config.n_gaussians, config.cutoff);
        let layers: Vec<SchNetLayer> = (0..config.n_interactions)
            .map(|_| SchNetLayer::new(config.n_atom_basis, config.n_gaussians))
            .collect();

        let mut rng = StdRng::seed_from_u64(99);
        let scale = 0.05_f32;
        let embed_table: Vec<Vec<f32>> = (0..119)
            .map(|_| {
                (0..config.n_atom_basis)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        let rw_scale = (2.0 / config.n_atom_basis as f32).sqrt();
        let readout_weight: Vec<f32> = (0..config.n_atom_basis)
            .map(|_| rng.random_range(-rw_scale..rw_scale))
            .collect();

        SchNetMaterials {
            config,
            smearing,
            layers,
            embed_table,
            readout_weight,
            readout_bias: 0.0,
        }
    }

    /// Predict a scalar property for the given crystal graph.
    pub fn predict_property(&self, graph: &CrystalGraph) -> f32 {
        let n = graph.atoms.len();
        if n == 0 {
            return 0.0;
        }

        // Atom embeddings
        let mut atom_h: Vec<Vec<f32>> = graph
            .atoms
            .iter()
            .map(|a| {
                let idx = a.atomic_number.min(self.embed_table.len() - 1);
                self.embed_table[idx].clone()
            })
            .collect();

        // Interaction layers
        for layer in &self.layers {
            let prev = atom_h.clone();
            for i in 0..n {
                let mut msg = vec![0.0f32; self.config.n_atom_basis];
                for bond in &graph.bonds {
                    if bond.i != i || bond.distance > self.config.cutoff {
                        continue;
                    }
                    let j = bond.j.min(n - 1);
                    let rbf = self.smearing.expand(bond.distance);
                    let w = layer.cfconv.filter(&rbf);
                    for k in 0..self.config.n_atom_basis {
                        msg[k] += w[k] * prev[j][k];
                    }
                }
                // Output network
                let out = linear_f32(
                    &layer.output_net,
                    &layer.output_bias,
                    &msg,
                    self.config.n_atom_basis,
                    self.config.n_atom_basis,
                );
                for k in 0..self.config.n_atom_basis {
                    atom_h[i][k] += out[k];
                }
            }
        }

        // Sum readout
        let mut total = self.readout_bias;
        for feat in &atom_h {
            total += dot_f32(&self.readout_weight, feat);
        }
        total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 MattersimModel
// ─────────────────────────────────────────────────────────────────────────────

/// Element embedding lookup table.
#[derive(Debug, Clone)]
pub struct ElementEmbedding {
    /// Embedding vectors indexed by atomic number.
    pub embeddings: Vec<Vec<f32>>,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl ElementEmbedding {
    /// Create a new `ElementEmbedding` with random initialisation.
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(77);
        let embeddings: Vec<Vec<f32>> = (0..119)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.random_range(-0.1_f32..0.1_f32))
                    .collect()
            })
            .collect();
        ElementEmbedding {
            embeddings,
            embed_dim,
        }
    }

    /// Look up the embedding for a given atomic number.
    pub fn lookup(&self, atomic_number: usize) -> &[f32] {
        let idx = atomic_number.min(self.embeddings.len() - 1);
        &self.embeddings[idx]
    }
}

/// Structure descriptor containing atom-level and pairwise features.
#[derive(Debug, Clone)]
pub struct StructureDescriptor {
    /// Per-atom feature vectors.
    pub atom_features: Vec<Vec<f32>>,
    /// Pairwise (bond) feature vectors.
    pub pair_features: Vec<Vec<f32>>,
}

/// Compute a structure descriptor for a crystal graph.
///
/// Atom features: element embedding + electronegativity + radius.
/// Pair features: RBF-expanded distance.
pub fn compute_descriptor(graph: &CrystalGraph, embed: &ElementEmbedding) -> StructureDescriptor {
    let smear = EdgeFeaturizer::new(16, 0.5, 8.0);
    let atom_features: Vec<Vec<f32>> = graph
        .atoms
        .iter()
        .map(|a| {
            let mut feat = embed.lookup(a.atomic_number).to_vec();
            feat.push(a.electronegativity);
            feat.push(a.radius);
            feat
        })
        .collect();
    let pair_features: Vec<Vec<f32>> = graph
        .bonds
        .iter()
        .map(|b| smear.featurize(b.distance))
        .collect();
    StructureDescriptor {
        atom_features,
        pair_features,
    }
}

/// Universal materials embedder (MattersimModel-style).
#[derive(Debug, Clone)]
pub struct MattersimModel {
    /// Element embedding lookup.
    pub element_embed: ElementEmbedding,
    /// Pair encoder weights (pair_out_dim × rbf_dim).
    pub pair_encoder: Vec<f32>,
    /// Aggregation network weights (embed_dim × (elem_dim + pair_out_dim)).
    pub agg_net: Vec<f32>,
    /// Output bias.
    pub agg_bias: Vec<f32>,
    /// Output (embedding) dimension.
    pub out_dim: usize,
}

impl MattersimModel {
    /// Create a new `MattersimModel`.
    pub fn new(elem_embed_dim: usize, out_dim: usize) -> Self {
        let element_embed = ElementEmbedding::new(elem_embed_dim);
        let rbf_dim = 16usize;
        let pair_out_dim = 32usize;
        let mut rng = StdRng::seed_from_u64(111);
        let pe_scale = (2.0 / rbf_dim as f32).sqrt();
        let pair_encoder: Vec<f32> = (0..pair_out_dim * rbf_dim)
            .map(|_| rng.random_range(-pe_scale..pe_scale))
            .collect();
        let agg_in_dim = elem_embed_dim + 2 + pair_out_dim; // +2 for electro+radius
        let agg_scale = (2.0 / agg_in_dim as f32).sqrt();
        let agg_net: Vec<f32> = (0..out_dim * agg_in_dim)
            .map(|_| rng.random_range(-agg_scale..agg_scale))
            .collect();
        let agg_bias = vec![0.0f32; out_dim];
        MattersimModel {
            element_embed,
            pair_encoder,
            agg_net,
            agg_bias,
            out_dim,
        }
    }

    /// Embed the crystal graph into a fixed-size descriptor vector.
    pub fn embed(&self, graph: &CrystalGraph) -> Vec<f32> {
        let n_atoms = graph.atoms.len();
        if n_atoms == 0 {
            return vec![0.0; self.out_dim];
        }
        let rbf_dim = 16usize;
        let pair_out_dim = 32usize;
        let descriptor = compute_descriptor(graph, &self.element_embed);

        // Encode pairwise features
        let pair_out: Vec<f32> = if descriptor.pair_features.is_empty() {
            vec![0.0; pair_out_dim]
        } else {
            let zero_bias = vec![0.0f32; pair_out_dim];
            let sum: Vec<f32> =
                descriptor
                    .pair_features
                    .iter()
                    .fold(vec![0.0f32; pair_out_dim], |mut acc, pf| {
                        let out =
                            linear_f32(&self.pair_encoder, &zero_bias, pf, pair_out_dim, rbf_dim);
                        for (a, o) in acc.iter_mut().zip(out.iter()) {
                            *a += o;
                        }
                        acc
                    });
            let inv = 1.0 / descriptor.pair_features.len() as f32;
            sum.into_iter().map(|v| relu_f32(v * inv)).collect()
        };

        // Mean-pool atom features
        let atom_dim = descriptor
            .atom_features
            .first()
            .map(|v| v.len())
            .unwrap_or(0);
        let mut atom_mean = vec![0.0f32; atom_dim];
        for af in &descriptor.atom_features {
            for (k, &v) in af.iter().enumerate() {
                atom_mean[k] += v;
            }
        }
        let inv_n = 1.0 / n_atoms as f32;
        for v in &mut atom_mean {
            *v *= inv_n;
        }

        // Concatenate and aggregate
        let mut concat = atom_mean;
        concat.extend_from_slice(&pair_out);
        linear_f32(
            &self.agg_net,
            &self.agg_bias,
            &concat,
            self.out_dim,
            concat.len(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 PhasePredictor
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the phase predictor.
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    /// Number of elements considered.
    pub n_elements: usize,
    /// Number of distinct phases.
    pub n_phases: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

/// A point on (or near) the convex hull of a phase diagram.
#[derive(Debug, Clone)]
pub struct ConvexHullPoint {
    /// Composition vector (sum ≈ 1.0, values in \[0,1\]).
    pub composition: Vec<f32>,
    /// Formation energy per atom (in eV/atom).
    pub energy: f32,
}

/// Test whether a query point lies on the 2D convex hull of binary systems.
///
/// For a binary system (composition\[0\] = x, energy = y), tests if the query
/// point (x_q, e_q) is on or below the convex hull formed by `points`.
/// Returns `true` if the query's energy equals (within tolerance) the hull value at x_q.
pub fn is_on_hull(points: &[ConvexHullPoint], query: &ConvexHullPoint) -> bool {
    let hull_e = energy_above_hull(points, query);
    hull_e.abs() < 1e-4
}

/// Compute the energy above the convex hull for a query point.
///
/// For binary systems (composition dimension ≥ 1), uses linear interpolation
/// between the two hull endpoints bracketing the query composition.
/// Returns the energy difference in eV/atom (0 = on hull, positive = unstable).
pub fn energy_above_hull(points: &[ConvexHullPoint], query: &ConvexHullPoint) -> f32 {
    if points.is_empty() {
        return query.energy;
    }
    // Use first component as the binary coordinate x ∈ [0,1]
    let x_q = query.composition.first().copied().unwrap_or(0.0);
    let e_q = query.energy;

    // Collect (x, energy) pairs from known stable endpoints
    let mut pts: Vec<(f32, f32)> = points
        .iter()
        .map(|p| (p.composition.first().copied().unwrap_or(0.0), p.energy))
        .collect();
    // Add pure-element endpoints at x=0 and x=1 if not present
    if pts.iter().all(|(x, _)| *x > 0.05) {
        pts.push((0.0, 0.0));
    }
    if pts.iter().all(|(x, _)| *x < 0.95) {
        pts.push((1.0, 0.0));
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build lower convex hull
    let hull = lower_convex_hull_2d(&pts);

    // Interpolate hull at x_q
    let hull_e = hull_interpolate(&hull, x_q);
    e_q - hull_e
}

/// Build the lower convex hull of 2D points sorted by x.
///
/// The lower hull is the bottom-facing boundary connecting the leftmost to the
/// rightmost point while always staying as low as possible. Points are removed
/// when they produce a right turn (clockwise), i.e. the middle point is ABOVE
/// the line from its predecessor to the new point.
fn lower_convex_hull_2d(pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut hull: Vec<(f32, f32)> = Vec::new();
    for &p in pts {
        while hull.len() >= 2 {
            let n = hull.len();
            let a = hull[n - 2];
            let b = hull[n - 1];
            // Cross product of (b-a) × (p-a):
            //   > 0 (CCW/left turn)  → b is BELOW the line a→p → keep b (lower hull vertex)
            //   ≤ 0 (CW/right turn or collinear) → b is ABOVE the line a→p → remove b
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull
}

/// Linearly interpolate the convex hull at position x.
fn hull_interpolate(hull: &[(f32, f32)], x: f32) -> f32 {
    if hull.is_empty() {
        return 0.0;
    }
    if hull.len() == 1 {
        return hull[0].1;
    }
    // Find bracketing segment
    for i in 0..hull.len() - 1 {
        let (x0, y0) = hull[i];
        let (x1, y1) = hull[i + 1];
        if x >= x0 && x <= x1 {
            let t = if (x1 - x0).abs() < 1e-10 {
                0.0
            } else {
                (x - x0) / (x1 - x0)
            };
            return y0 + t * (y1 - y0);
        }
    }
    // Extrapolate
    if x < hull[0].0 {
        hull[0].1
    } else {
        hull[hull.len() - 1].1
    }
}

/// Neural phase stability predictor.
#[derive(Debug, Clone)]
pub struct PhasePredictor {
    /// Weight vector for composition → stability score (n_phases × (n_elements + 1)).
    pub weights: Vec<f32>,
    /// Number of phases.
    pub n_phases: usize,
    /// Number of elements.
    pub n_elements: usize,
}

impl PhasePredictor {
    /// Create a new `PhasePredictor`.
    pub fn new(config: &PhaseConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let in_dim = config.n_elements + 1; // composition + energy
        let w_size = config.n_phases * in_dim;
        let scale = (2.0 / in_dim as f32).sqrt();
        let weights: Vec<f32> = (0..w_size)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        PhasePredictor {
            weights,
            n_phases: config.n_phases,
            n_elements: config.n_elements,
        }
    }

    /// Predict stability score (higher = more stable) for a given composition and energy.
    pub fn predict_stability(&self, composition: &[f32], energy: f32) -> f32 {
        let in_dim = self.n_elements + 1;
        // Pad or truncate composition to n_elements, then append energy
        let mut input = vec![0.0f32; in_dim];
        for (k, &v) in composition.iter().enumerate().take(self.n_elements) {
            input[k] = v;
        }
        input[self.n_elements] = energy;

        // Score is max over phase logits (softmax-like)
        let zero_bias = vec![0.0f32; self.n_phases];
        let logits = linear_f32(&self.weights, &zero_bias, &input, self.n_phases, in_dim);
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        // Return the probability of the most stable phase
        exps.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / sum.max(1e-10)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6 MaterialPropertyPredictor (aliased from PropertyPredictor to avoid
//    name collision with molecular_gnn::PropertyPredictor)
// ─────────────────────────────────────────────────────────────────────────────

/// Enumeration of physical materials properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialProperty {
    /// Electronic band gap (eV).
    BandGap,
    /// Formation energy per atom (eV/atom).
    FormationEnergy,
    /// Bulk modulus (GPa).
    BulkModulus,
    /// Shear modulus (GPa).
    ShearModulus,
    /// Mass density (g/cm³).
    Density,
    /// Thermal conductivity (W/m·K).
    ThermalConductivity,
}

impl MaterialProperty {
    /// All property variants, in canonical order.
    pub fn all() -> [MaterialProperty; 6] {
        [
            MaterialProperty::BandGap,
            MaterialProperty::FormationEnergy,
            MaterialProperty::BulkModulus,
            MaterialProperty::ShearModulus,
            MaterialProperty::Density,
            MaterialProperty::ThermalConductivity,
        ]
    }
}

/// Configuration for multi-task materials property prediction.
#[derive(Debug, Clone)]
pub struct MaterialPropertyPredictorConfig {
    /// Number of properties to predict (must be ≤ 6).
    pub n_properties: usize,
    /// Input embedding dimension.
    pub embed_dim: usize,
}

/// Multi-task materials property predictor with per-property linear heads.
#[derive(Debug, Clone)]
pub struct MaterialPropertyPredictor {
    /// Per-property linear head weights (each: 1 × embed_dim).
    pub task_heads: Vec<Vec<f32>>,
    /// Per-property bias.
    pub task_biases: Vec<f32>,
    /// Number of properties.
    pub n_properties: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl MaterialPropertyPredictor {
    /// Create a new `MaterialPropertyPredictor`.
    pub fn new(config: &MaterialPropertyPredictorConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(13);
        let scale = (2.0 / config.embed_dim as f32).sqrt();
        let task_heads: Vec<Vec<f32>> = (0..config.n_properties)
            .map(|_| {
                (0..config.embed_dim)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        MaterialPropertyPredictor {
            task_heads,
            task_biases: vec![0.0; config.n_properties],
            n_properties: config.n_properties,
            embed_dim: config.embed_dim,
        }
    }

    /// Predict all configured properties for the given embedding vector.
    ///
    /// Returns a list of `(MaterialProperty, predicted_value)` pairs.
    pub fn predict_all(&self, embed: &[f32]) -> Vec<(MaterialProperty, f32)> {
        let props = MaterialProperty::all();
        self.task_heads
            .iter()
            .zip(self.task_biases.iter())
            .enumerate()
            .map(|(i, (head, &bias))| {
                let val = dot_f32(head, &embed[..head.len().min(embed.len())]) + bias;
                let prop = *props.get(i).unwrap_or(&MaterialProperty::BandGap);
                (prop, val)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7 CrystalSymmetry
// ─────────────────────────────────────────────────────────────────────────────

/// Crystallographic space group descriptor.
#[derive(Debug, Clone)]
pub struct SpaceGroup {
    /// International space group number (1–230).
    pub number: usize,
    /// Hermann–Mauguin symbol (e.g. "Fm-3m").
    pub symbol: String,
}

/// Compute lattice parameters (a, b, c, α, β, γ) from a 3×3 lattice matrix.
///
/// Each row of `lattice` is a lattice vector. Returns (a, b, c, alpha_deg, beta_deg, gamma_deg).
pub fn compute_lattice_params(lattice: &[[f32; 3]; 3]) -> (f32, f32, f32, f32, f32, f32) {
    let vec_norm = |v: &[f32; 3]| -> f32 { (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() };
    let vec_dot = |a: &[f32; 3], b: &[f32; 3]| -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] };
    let a = vec_norm(&lattice[0]);
    let b = vec_norm(&lattice[1]);
    let c = vec_norm(&lattice[2]);
    let cos_alpha = if b > 1e-10 && c > 1e-10 {
        vec_dot(&lattice[1], &lattice[2]) / (b * c)
    } else {
        0.0
    };
    let cos_beta = if a > 1e-10 && c > 1e-10 {
        vec_dot(&lattice[0], &lattice[2]) / (a * c)
    } else {
        0.0
    };
    let cos_gamma = if a > 1e-10 && b > 1e-10 {
        vec_dot(&lattice[0], &lattice[1]) / (a * b)
    } else {
        0.0
    };
    let to_deg = 180.0 / std::f32::consts::PI;
    let alpha = cos_alpha.clamp(-1.0, 1.0).acos() * to_deg;
    let beta = cos_beta.clamp(-1.0, 1.0).acos() * to_deg;
    let gamma = cos_gamma.clamp(-1.0, 1.0).acos() * to_deg;
    (a, b, c, alpha, beta, gamma)
}

/// Detect the crystal lattice system from a lattice matrix.
///
/// Classification is based on the equality of lattice parameters:
/// - **Cubic**: a=b=c, α=β=γ=90°
/// - **Tetragonal**: a=b≠c, α=β=γ=90°
/// - **Orthorhombic**: a≠b≠c, α=β=γ=90°
/// - **Hexagonal**: a=b≠c, α=β=90°, γ=120°
/// - **Trigonal**: a=b=c, α=β=γ≠90°
/// - **Monoclinic**: a≠b≠c, α=γ=90°, β≠90°
/// - **Triclinic**: all distinct
pub fn detect_lattice_system(lattice: &[[f32; 3]; 3]) -> String {
    let (a, b, c, alpha, beta, gamma) = compute_lattice_params(lattice);
    let tol_len = 0.01_f32;
    let tol_ang = 1.0_f32; // degrees
    let near90 = |ang: f32| (ang - 90.0).abs() < tol_ang;
    let near120 = |ang: f32| (ang - 120.0).abs() < tol_ang;
    let eq_len = |x: f32, y: f32| (x - y).abs() < tol_len;

    if eq_len(a, b) && eq_len(b, c) && near90(alpha) && near90(beta) && near90(gamma) {
        "cubic".to_string()
    } else if eq_len(a, b) && !eq_len(b, c) && near90(alpha) && near90(beta) && near120(gamma) {
        "hexagonal".to_string()
    } else if eq_len(a, b) && eq_len(b, c) && !near90(alpha) {
        "trigonal".to_string()
    } else if eq_len(a, b) && !eq_len(b, c) && near90(alpha) && near90(beta) && near90(gamma) {
        "tetragonal".to_string()
    } else if !eq_len(a, b) && !eq_len(b, c) && near90(alpha) && near90(beta) && near90(gamma) {
        "orthorhombic".to_string()
    } else if near90(alpha) && !near90(beta) && near90(gamma) {
        "monoclinic".to_string()
    } else {
        "triclinic".to_string()
    }
}

/// Estimate the point group order from the distinct atomic environments.
///
/// Uses a simplified heuristic: count how many atoms share the same element
/// and approximate coordination environment (number of bonds within 3 Å).
/// The point group order is approximated as the GCD of the environment multiplicities.
pub fn point_group_order(atoms: &[Atom], bonds: &[CrystalBond]) -> usize {
    if atoms.is_empty() {
        return 1;
    }
    // Count bonds per atom
    let mut coord: Vec<usize> = vec![0; atoms.len()];
    for bond in bonds {
        if bond.distance < 3.0 && bond.i < atoms.len() {
            coord[bond.i] += 1;
        }
    }
    // Group atoms by (element, coordination_number)
    use std::collections::HashMap;
    let mut env_counts: HashMap<(String, usize), usize> = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        let key = (atom.element.clone(), coord[i]);
        *env_counts.entry(key).or_insert(0) += 1;
    }
    // Compute GCD of all multiplicities
    let counts: Vec<usize> = env_counts.values().cloned().collect();
    if counts.is_empty() {
        return 1;
    }
    counts.iter().cloned().fold(counts[0], gcd_usize)
}

fn gcd_usize(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd_usize(b, a % b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8 MaterialAugmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a random SO(3) rotation to all atom positions in a crystal graph.
///
/// The lattice is also rotated so that relative positions are preserved.
pub fn random_rotation(graph: &CrystalGraph, rng: &mut StdRng) -> CrystalGraph {
    // Sample random rotation via quaternion → rotation matrix
    let rot = random_rotation_matrix(rng);
    let rotate_vec = |v: &[f32; 3]| -> [f32; 3] {
        [
            rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
            rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
            rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
        ]
    };
    let new_atoms: Vec<Atom> = graph
        .atoms
        .iter()
        .map(|a| Atom {
            element: a.element.clone(),
            position: rotate_vec(&a.position),
            atomic_number: a.atomic_number,
            electronegativity: a.electronegativity,
            radius: a.radius,
        })
        .collect();
    let new_lattice: [[f32; 3]; 3] = [
        rotate_vec(&graph.lattice[0]),
        rotate_vec(&graph.lattice[1]),
        rotate_vec(&graph.lattice[2]),
    ];
    // Rebuild bonds from scratch for the rotated structure
    CrystalGraph {
        atoms: new_atoms,
        bonds: graph.bonds.clone(), // bond distances unchanged by rotation
        lattice: new_lattice,
    }
}

/// Generate a uniformly random 3×3 rotation matrix using Shoemake's method.
fn random_rotation_matrix(rng: &mut StdRng) -> [[f32; 3]; 3] {
    // Generate a random unit quaternion (Shoemake 1992)
    let u1: f32 = rng.random_range(0.0_f32..1.0_f32);
    let u2: f32 = rng.random_range(0.0_f32..1.0_f32);
    let u3: f32 = rng.random_range(0.0_f32..1.0_f32);
    let tau = 2.0 * std::f32::consts::PI;
    let q0 = (1.0 - u1).sqrt() * (tau * u2).sin();
    let q1 = (1.0 - u1).sqrt() * (tau * u2).cos();
    let q2 = u1.sqrt() * (tau * u3).sin();
    let q3 = u1.sqrt() * (tau * u3).cos();

    // Quaternion → rotation matrix
    let r00 = 1.0 - 2.0 * (q2 * q2 + q3 * q3);
    let r01 = 2.0 * (q1 * q2 - q0 * q3);
    let r02 = 2.0 * (q1 * q3 + q0 * q2);
    let r10 = 2.0 * (q1 * q2 + q0 * q3);
    let r11 = 1.0 - 2.0 * (q1 * q1 + q3 * q3);
    let r12 = 2.0 * (q2 * q3 - q0 * q1);
    let r20 = 2.0 * (q1 * q3 - q0 * q2);
    let r21 = 2.0 * (q2 * q3 + q0 * q1);
    let r22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2);
    [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
}

/// Add isotropic Gaussian displacement noise to atom positions.
pub fn add_gaussian_displacement(graph: &CrystalGraph, std: f32, rng: &mut StdRng) -> CrystalGraph {
    // Box-Muller transform for Gaussian samples
    let gauss = |rng: &mut StdRng| -> f32 {
        let u1: f32 = rng.random_range(1e-10_f32..1.0_f32);
        let u2: f32 = rng.random_range(0.0_f32..1.0_f32);
        let two_pi = 2.0 * std::f32::consts::PI;
        (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos()
    };
    let new_atoms: Vec<Atom> = graph
        .atoms
        .iter()
        .map(|a| Atom {
            element: a.element.clone(),
            position: [
                a.position[0] + std * gauss(rng),
                a.position[1] + std * gauss(rng),
                a.position[2] + std * gauss(rng),
            ],
            atomic_number: a.atomic_number,
            electronegativity: a.electronegativity,
            radius: a.radius,
        })
        .collect();
    CrystalGraph {
        atoms: new_atoms,
        bonds: graph.bonds.clone(),
        lattice: graph.lattice,
    }
}

/// Create a random supercell expansion of the crystal.
///
/// Randomly chooses one of the three lattice directions and doubles it
/// (scale = 2), replicating atoms accordingly. `max_scale` is unused in this
/// simplified implementation (always 2×1×1, 1×2×1, or 1×1×2).
pub fn random_supercell(graph: &CrystalGraph, _max_scale: usize, rng: &mut StdRng) -> CrystalGraph {
    // Pick random axis
    let axis: usize = rng.random_range(0_usize..3_usize);

    let mut new_atoms = graph.atoms.clone();
    // Duplicate atoms along the chosen axis
    for atom in &graph.atoms {
        let pos = atom.position;
        let new_pos = [
            pos[0] + if axis == 0 { graph.lattice[0][0] } else { 0.0 },
            pos[1] + if axis == 1 { graph.lattice[1][1] } else { 0.0 },
            pos[2] + if axis == 2 { graph.lattice[2][2] } else { 0.0 },
        ];
        new_atoms.push(Atom {
            element: atom.element.clone(),
            position: new_pos,
            atomic_number: atom.atomic_number,
            electronegativity: atom.electronegativity,
            radius: atom.radius,
        });
    }

    let mut new_lattice = graph.lattice;
    new_lattice[axis][axis] *= 2.0;

    CrystalGraph {
        atoms: new_atoms,
        bonds: graph.bonds.clone(),
        lattice: new_lattice,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9 GenerativeMaterials
// ─────────────────────────────────────────────────────────────────────────────

/// Composition generator: generates element fractions conditioned on target properties.
#[derive(Debug, Clone)]
pub struct CompositionGenerator {
    /// Element embedding matrix (n_elements × embed_dim).
    pub element_embed: Vec<Vec<f32>>,
    /// Number of elements in the vocabulary.
    pub n_elements: usize,
}

/// Periodic table element symbols (first 94, roughly).
const ELEMENT_SYMBOLS: &[&str] = &[
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
];

impl CompositionGenerator {
    /// Create a new `CompositionGenerator`.
    pub fn new(n_elements: usize, embed_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(17);
        let element_embed: Vec<Vec<f32>> = (0..n_elements)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.random_range(-0.1_f32..0.1_f32))
                    .collect()
            })
            .collect();
        CompositionGenerator {
            element_embed,
            n_elements,
        }
    }

    /// Sample a composition conditioned on target property values.
    ///
    /// Returns a list of `(element_symbol, fraction)` pairs that sum to ~1.
    pub fn sample_composition(&self, target_props: &[f32], rng: &mut StdRng) -> Vec<(String, f32)> {
        if self.n_elements == 0 {
            return Vec::new();
        }
        // Compute scores for each element as dot product with target properties
        let prop_dim = target_props.len();
        let scores: Vec<f32> = self
            .element_embed
            .iter()
            .map(|emb| {
                let eff_len = prop_dim.min(emb.len());
                emb[..eff_len]
                    .iter()
                    .zip(&target_props[..eff_len])
                    .map(|(&e, &t)| e * t)
                    .sum::<f32>()
            })
            .collect();

        // Softmax to get probabilities
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum::<f32>().max(1e-10);

        // Sample 2–4 distinct elements via gumbel-max trick
        let n_sample = rng.random_range(2_usize..=4_usize).min(self.n_elements);
        let mut gumbel_scores: Vec<(f32, usize)> = exps
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let u: f32 = rng.random_range(1e-10_f32..1.0_f32);
                let g = (e / sum_e).ln() - (-u.ln()).ln();
                (g, i)
            })
            .collect();
        gumbel_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Assign uniform-random fractions to top-k elements
        let selected: Vec<usize> = gumbel_scores
            .iter()
            .take(n_sample)
            .map(|&(_, i)| i)
            .collect();
        let mut fracs: Vec<f32> = (0..n_sample)
            .map(|_| rng.random_range(0.1_f32..1.0_f32))
            .collect();
        let frac_sum: f32 = fracs.iter().sum::<f32>().max(1e-10);
        for f in &mut fracs {
            *f /= frac_sum;
        }

        selected
            .into_iter()
            .zip(fracs)
            .map(|(idx, frac)| {
                let sym = ELEMENT_SYMBOLS.get(idx).copied().unwrap_or("X");
                (sym.to_string(), frac)
            })
            .collect()
    }
}

/// Score-based diffusion model for crystal structure generation.
#[derive(Debug, Clone)]
pub struct DiffusionMaterialsModel {
    /// Score network weights (flat: used as a simple linear score approximation).
    pub score_net_weights: Vec<f32>,
    /// Number of diffusion time steps.
    pub n_steps: usize,
}

impl DiffusionMaterialsModel {
    /// Create a new diffusion model for materials generation.
    pub fn new(n_steps: usize, pos_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(199);
        let scale = (2.0 / pos_dim as f32).sqrt();
        let score_net_weights: Vec<f32> = (0..pos_dim * pos_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        DiffusionMaterialsModel {
            score_net_weights,
            n_steps,
        }
    }

    /// Perform one denoising step in the reverse diffusion process.
    ///
    /// `noisy_positions` is a flat array of 3D positions [x0, y0, z0, x1, y1, z1, ...].
    /// `t` is the normalised time step in [0, 1] (1 = pure noise, 0 = clean structure).
    ///
    /// Uses a simplified score estimate: `score ≈ -W · x / σ(t)²` where σ(t) = t.
    pub fn denoise_step(&self, noisy_positions: &[f32], t: f32) -> Vec<f32> {
        let n = noisy_positions.len();
        if n == 0 {
            return Vec::new();
        }
        let sigma_sq = (t * t).max(1e-6);
        let step_size = 0.01 * t; // Euler-Maruyama step

        // Approximate score as -(x - W·x) / sigma^2  (linear score network)
        let w_dim = (self.score_net_weights.len() as f32).sqrt() as usize;
        let eff_n = n.min(w_dim);
        let zero_bias = vec![0.0f32; eff_n];
        let mut wx = if eff_n > 0 {
            linear_f32(
                &self.score_net_weights[..eff_n * eff_n],
                &zero_bias,
                &noisy_positions[..eff_n],
                eff_n,
                eff_n,
            )
        } else {
            vec![]
        };
        // Pad or truncate to n
        wx.resize(n, 0.0);

        noisy_positions
            .iter()
            .zip(wx.iter())
            .map(|(&x, &wx_i)| {
                let score = -(x - wx_i) / sigma_sq;
                x + step_size * score
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10 MaterialMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute mean absolute error between predictions and true values.
pub fn mae(pred: &[f32], true_vals: &[f32]) -> f32 {
    if pred.is_empty() {
        return 0.0;
    }
    let n = pred.len().min(true_vals.len());
    pred[..n]
        .iter()
        .zip(&true_vals[..n])
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f32>()
        / n as f32
}

/// Compute the coefficient of determination (R²).
pub fn r2(pred: &[f32], true_vals: &[f32]) -> f32 {
    if pred.is_empty() {
        return 0.0;
    }
    let n = pred.len().min(true_vals.len());
    let mean_t: f32 = true_vals[..n].iter().sum::<f32>() / n as f32;
    let ss_tot: f32 = true_vals[..n].iter().map(|&t| (t - mean_t).powi(2)).sum();
    if ss_tot < 1e-12 {
        return 1.0; // perfect constant prediction
    }
    let ss_res: f32 = pred[..n]
        .iter()
        .zip(&true_vals[..n])
        .map(|(&p, &t)| (t - p).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

/// Compute Spearman rank correlation between predictions and true values.
pub fn spearman_rank_correlation(pred: &[f32], true_vals: &[f32]) -> f32 {
    if pred.is_empty() {
        return 0.0;
    }
    let n = pred.len().min(true_vals.len());
    let ranks_pred = compute_ranks(&pred[..n]);
    let ranks_true = compute_ranks(&true_vals[..n]);

    let mean_rp: f32 = ranks_pred.iter().sum::<f32>() / n as f32;
    let mean_rt: f32 = ranks_true.iter().sum::<f32>() / n as f32;
    let num: f32 = ranks_pred
        .iter()
        .zip(ranks_true.iter())
        .map(|(&rp, &rt)| (rp - mean_rp) * (rt - mean_rt))
        .sum();
    let den_p: f32 = ranks_pred
        .iter()
        .map(|&rp| (rp - mean_rp).powi(2))
        .sum::<f32>()
        .sqrt();
    let den_t: f32 = ranks_true
        .iter()
        .map(|&rt| (rt - mean_rt).powi(2))
        .sum::<f32>()
        .sqrt();
    let denom = den_p * den_t;
    if denom < 1e-12 {
        return 0.0;
    }
    num / denom
}

/// Compute fractional ranks (average-rank tie-breaking) for a slice.
fn compute_ranks(vals: &[f32]) -> Vec<f32> {
    let n = vals.len();
    let mut indexed: Vec<(f32, usize)> = vals
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f32 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].1] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Compute the top-k screening rate: fraction of true top-k materials that appear
/// in the predicted top-k.
///
/// Returns a value in [0, 1]; 1.0 means perfect top-k recovery.
pub fn top_k_screening_rate(pred_scores: &[f32], true_scores: &[f32], k: usize) -> f32 {
    let n = pred_scores.len().min(true_scores.len());
    if n == 0 || k == 0 {
        return 0.0;
    }
    let k = k.min(n);

    let top_k_indices = |scores: &[f32]| -> std::collections::HashSet<usize> {
        let mut indexed: Vec<(f32, usize)> = scores
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|&(_, i)| i).collect()
    };

    let pred_top = top_k_indices(&pred_scores[..n]);
    let true_top = top_k_indices(&true_scores[..n]);
    let intersection = pred_top.intersection(&true_top).count();
    intersection as f32 / k as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "materials_ml_tests.rs"]
mod tests;
