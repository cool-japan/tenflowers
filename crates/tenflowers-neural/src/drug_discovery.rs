//! Drug Discovery ML Components
//!
//! Provides a complete drug discovery machine-learning pipeline:
//!
//! - [`DdMolecule`]: Molecular representation with atoms, bonds, and SMILES.
//! - [`DdMolecularFingerprint`]: ECFP4 fingerprint via FNV-1a Morgan algorithm.
//! - \[`AdmetPredictor`\]: ADMET property prediction with Lipinski / Veber filters.
//! - [`VirtualScreener`]: Neural docking-proxy screening pipeline.
//! - \[`ScaffoldAnalysis`\]: Bemis–Murcko decomposition, MaxMin diversity picker.
//! - [`SmileRnn`]: Character-level RNN for SMILES generation.
//! - [`ActivityCliffAnalyzer`]: Activity cliff and matched molecular pair detection.
//! - [`DrugDiscoveryReport`]: Campaign evaluation metrics.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §1 Molecular Representation
// ─────────────────────────────────────────────────────────────────────────────

/// Chemical element identity (drug-discovery variant; prefixed `Dd` to avoid
/// collision with the graph-generation crate's `AtomType`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DdAtomType {
    Carbon,
    Nitrogen,
    Oxygen,
    Sulfur,
    Phosphorus,
    Fluorine,
    Chlorine,
    Bromine,
    Iodine,
    Other,
}

impl DdAtomType {
    /// Approximate atomic weight used for molecular-weight estimation.
    fn approx_weight(&self) -> f32 {
        match self {
            DdAtomType::Carbon => 12.011,
            DdAtomType::Nitrogen => 14.007,
            DdAtomType::Oxygen => 15.999,
            DdAtomType::Sulfur => 32.06,
            DdAtomType::Phosphorus => 30.974,
            DdAtomType::Fluorine => 18.998,
            DdAtomType::Chlorine => 35.45,
            DdAtomType::Bromine => 79.904,
            DdAtomType::Iodine => 126.904,
            DdAtomType::Other => 14.0,
        }
    }

    /// Whether the atom can accept a hydrogen bond (N or O).
    fn is_hba(&self) -> bool {
        matches!(self, DdAtomType::Nitrogen | DdAtomType::Oxygen)
    }

    /// Whether the atom can donate a hydrogen bond (N or O with H attached).
    fn is_hbd(&self, n_hydrogens: u8) -> bool {
        n_hydrogens > 0 && matches!(self, DdAtomType::Nitrogen | DdAtomType::Oxygen)
    }
}

/// Single atom in a [`DdMolecule`].
#[derive(Debug, Clone, PartialEq)]
pub struct DdAtom {
    pub element: DdAtomType,
    pub formal_charge: i32,
    pub n_hydrogens: u8,
    pub is_aromatic: bool,
}

/// Bond type for a [`DdMolecule`] (prefixed `Dd` to avoid collision).
#[derive(Debug, Clone, PartialEq)]
pub enum DdBondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl DdBondType {
    fn order(&self) -> f32 {
        match self {
            DdBondType::Single => 1.0,
            DdBondType::Double => 2.0,
            DdBondType::Triple => 3.0,
            DdBondType::Aromatic => 1.5,
        }
    }
}

/// Bond connecting two atoms by index.
#[derive(Debug, Clone, PartialEq)]
pub struct DdBond {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_type: DdBondType,
}

/// Molecular graph with atoms, bonds, and optional SMILES string.
#[derive(Debug, Clone)]
pub struct DdMolecule {
    pub atoms: Vec<DdAtom>,
    pub bonds: Vec<DdBond>,
    pub smiles: String,
}

impl DdMolecule {
    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Number of bonds.
    pub fn n_bonds(&self) -> usize {
        self.bonds.len()
    }

    /// Symmetric adjacency matrix (1.0 where a bond exists, 0.0 otherwise).
    pub fn adjacency_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.n_atoms();
        let mut mat = vec![vec![0.0_f32; n]; n];
        for b in &self.bonds {
            if b.atom1 < n && b.atom2 < n {
                mat[b.atom1][b.atom2] = b.bond_type.order();
                mat[b.atom2][b.atom1] = b.bond_type.order();
            }
        }
        mat
    }

    /// Count of non-hydrogen atoms (here all atoms are already heavy).
    pub fn heavy_atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Approximate molecular weight from atom types plus attached hydrogens.
    pub fn molecular_weight(&self) -> f32 {
        let mut mw = 0.0_f32;
        for atom in &self.atoms {
            mw += atom.element.approx_weight();
            mw += atom.n_hydrogens as f32 * 1.008; // hydrogen weight
        }
        mw
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 Molecular Fingerprint (ECFP4 via FNV-1a)
// ─────────────────────────────────────────────────────────────────────────────

/// Binary circular fingerprint (ECFP-style) stored as a bitvector.
/// Prefixed `Dd` to avoid collision with graph_generation's `MolecularFingerprint`.
#[derive(Debug, Clone)]
pub struct DdMolecularFingerprint {
    pub bits: Vec<bool>,
    pub length: usize,
}

impl DdMolecularFingerprint {
    fn new(length: usize) -> Self {
        DdMolecularFingerprint {
            bits: vec![false; length],
            length,
        }
    }

    fn set_bit(&mut self, pos: usize) {
        if pos < self.length {
            self.bits[pos] = true;
        }
    }
}

/// FNV-1a 64-bit hash constant.
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

fn fnv1a_hash_u64(value: u64, seed: u64) -> u64 {
    let mut hash = seed;
    for byte in value.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn atom_invariant(atom: &DdAtom, idx: usize) -> u64 {
    let elem_code: u64 = match atom.element {
        DdAtomType::Carbon => 6,
        DdAtomType::Nitrogen => 7,
        DdAtomType::Oxygen => 8,
        DdAtomType::Fluorine => 9,
        DdAtomType::Phosphorus => 15,
        DdAtomType::Sulfur => 16,
        DdAtomType::Chlorine => 17,
        DdAtomType::Bromine => 35,
        DdAtomType::Iodine => 53,
        DdAtomType::Other => 0,
    };
    let mut h = FNV_OFFSET;
    h = fnv1a_hash_u64(elem_code, h);
    h = fnv1a_hash_u64(atom.formal_charge.unsigned_abs() as u64, h);
    h = fnv1a_hash_u64(atom.n_hydrogens as u64, h);
    h = fnv1a_hash_u64(if atom.is_aromatic { 1 } else { 0 }, h);
    h = fnv1a_hash_u64(idx as u64, h);
    h
}

/// Generate an ECFP4-like fingerprint using the Morgan algorithm with FNV-1a hashing.
///
/// `radius` controls the neighbourhood depth (2 = ECFP4).
/// `n_bits` must be a power of two for uniform folding.
pub fn ecfp4(mol: &DdMolecule, radius: usize, n_bits: usize) -> DdMolecularFingerprint {
    let n = mol.n_atoms();
    let mut fp = DdMolecularFingerprint::new(n_bits);
    if n == 0 || n_bits == 0 {
        return fp;
    }

    // Build adjacency list from bonds.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in &mol.bonds {
        if b.atom1 < n && b.atom2 < n && b.atom1 != b.atom2 {
            adj[b.atom1].push(b.atom2);
            adj[b.atom2].push(b.atom1);
        }
    }

    // Initial identifiers.
    let mut identifiers: Vec<u64> = (0..n).map(|i| atom_invariant(&mol.atoms[i], i)).collect();

    // Record round-0 identifiers.
    for &id in &identifiers {
        fp.set_bit((id as usize) % n_bits);
    }

    // Iterative Morgan update.
    for _r in 0..radius {
        let mut new_ids = identifiers.clone();
        for atom_idx in 0..n {
            let mut h = FNV_OFFSET;
            h = fnv1a_hash_u64(identifiers[atom_idx], h);
            h = fnv1a_hash_u64(_r as u64, h);
            // Sort neighbour ids for canonical ordering.
            let mut nbr_ids: Vec<u64> = adj[atom_idx].iter().map(|&nb| identifiers[nb]).collect();
            nbr_ids.sort_unstable();
            for nid in nbr_ids {
                h = fnv1a_hash_u64(nid, h);
            }
            new_ids[atom_idx] = h;
            fp.set_bit((h as usize) % n_bits);
        }
        identifiers = new_ids;
    }

    fp
}

/// Tanimoto (Jaccard) similarity between two binary fingerprints.
///
/// Returns 0.0 if both fingerprints are all-zero.
pub fn dd_tanimoto(fp1: &DdMolecularFingerprint, fp2: &DdMolecularFingerprint) -> f32 {
    let len = fp1.length.min(fp2.length);
    if len == 0 {
        return 0.0;
    }
    let (mut inter, mut union_) = (0usize, 0usize);
    for i in 0..len {
        let a = fp1.bits[i];
        let b = fp2.bits[i];
        if a && b {
            inter += 1;
        }
        if a || b {
            union_ += 1;
        }
    }
    if union_ == 0 {
        return 0.0;
    }
    inter as f32 / union_ as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 ADMET Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// ADMET endpoint identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AdmetProperty {
    Solubility,
    Permeability,
    LogP,
    HalfLife,
    Toxicity,
    BBBPenetration,
    HERGInhibition,
}

/// Lipinski-style descriptors derived from a [`DdMolecule`].
#[derive(Debug, Clone)]
pub struct AdmetDescriptors {
    /// Molecular weight.
    pub mw: f32,
    /// Predicted log P (simple atom-contribution model).
    pub logp: f32,
    /// Hydrogen-bond donor count.
    pub hbd: u32,
    /// Hydrogen-bond acceptor count.
    pub hba: u32,
    /// Topological polar surface area (simple O+N contribution estimate).
    pub tpsa: f32,
    /// Number of rotatable bonds (single, non-ring, non-terminal).
    pub rotatable_bonds: u32,
}

impl AdmetDescriptors {
    /// Compute Lipinski descriptors from atom/bond counts.
    pub fn from_molecule(mol: &DdMolecule) -> Self {
        let mw = mol.molecular_weight();

        // Log P: simple Wildman–Crippen-like atom contributions.
        let mut logp = 0.0_f32;
        for atom in &mol.atoms {
            logp += match atom.element {
                DdAtomType::Carbon => 0.5,
                DdAtomType::Nitrogen => -1.0,
                DdAtomType::Oxygen => -1.0,
                DdAtomType::Sulfur => 0.5,
                DdAtomType::Phosphorus => -0.5,
                DdAtomType::Fluorine => 0.14,
                DdAtomType::Chlorine => 0.6,
                DdAtomType::Bromine => 0.8,
                DdAtomType::Iodine => 1.0,
                DdAtomType::Other => 0.0,
            };
        }

        let hbd = mol
            .atoms
            .iter()
            .filter(|a| a.element.is_hbd(a.n_hydrogens))
            .count() as u32;

        let hba = mol.atoms.iter().filter(|a| a.element.is_hba()).count() as u32;

        // TPSA: O contributes ~9 Å², N contributes ~12 Å² (rough estimate).
        let tpsa: f32 = mol
            .atoms
            .iter()
            .map(|a| match a.element {
                DdAtomType::Oxygen => 9.23_f32,
                DdAtomType::Nitrogen => 12.89_f32,
                _ => 0.0_f32,
            })
            .sum();

        // Count rotatable bonds (single bonds, not counted twice).
        let n = mol.n_atoms();
        let rotatable_bonds = mol
            .bonds
            .iter()
            .filter(|b| {
                b.bond_type == DdBondType::Single
                    && b.atom1 < n
                    && b.atom2 < n
                    && b.atom1 != b.atom2
            })
            .count() as u32;

        AdmetDescriptors {
            mw,
            logp,
            hbd,
            hba,
            tpsa,
            rotatable_bonds,
        }
    }

    /// Lipinski Rule-of-Five: MW≤500, LogP≤5, HBD≤5, HBA≤10.
    pub fn lipinski_ro5(&self) -> bool {
        self.mw <= 500.0 && self.logp <= 5.0 && self.hbd <= 5 && self.hba <= 10
    }

    /// Veber oral bioavailability rules: TPSA≤140, RotBonds≤10.
    pub fn veber_rules(&self) -> bool {
        self.tpsa <= 140.0 && self.rotatable_bonds <= 10
    }
}

/// Linear ADMET model (one per endpoint).
#[derive(Debug, Clone)]
pub struct AdmetModel {
    pub property: AdmetProperty,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub input_dim: usize,
}

impl AdmetModel {
    /// Initialise with small random weights (Xavier-like).
    pub fn new(property: AdmetProperty, input_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (1.0_f32 / input_dim.max(1) as f32).sqrt();
        let weights: Vec<f32> = (0..input_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let bias = (rng.random::<f32>() * 2.0 - 1.0) * scale;
        AdmetModel {
            property,
            weights,
            bias,
            input_dim,
        }
    }

    fn dot(weights: &[f32], inputs: &[f32]) -> f32 {
        weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
    }

    /// Predict from a 6-dimensional descriptor vector derived from
    /// [`AdmetDescriptors`] (mw, logp, hbd, hba, tpsa, rotatable_bonds).
    pub fn predict_from_descriptors(&self, desc: &AdmetDescriptors) -> f32 {
        let v = [
            desc.mw / 500.0,
            desc.logp / 5.0,
            desc.hbd as f32 / 5.0,
            desc.hba as f32 / 10.0,
            desc.tpsa / 140.0,
            desc.rotatable_bonds as f32 / 10.0,
        ];
        let dim = self.input_dim.min(6);
        Self::dot(&self.weights[..dim], &v[..dim]) + self.bias
    }

    /// Predict from a binary fingerprint (dot product on bit values).
    pub fn predict_from_fingerprint(&self, fp: &DdMolecularFingerprint) -> f32 {
        let dim = self.input_dim.min(fp.length);
        let inputs: Vec<f32> = fp.bits[..dim]
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        Self::dot(&self.weights[..dim], &inputs) + self.bias
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 Virtual Screening
// ─────────────────────────────────────────────────────────────────────────────

/// Neural-network surrogate for a molecular docking score.
/// Lower score = better predicted binding affinity.
#[derive(Debug, Clone)]
pub struct DockingProxy {
    pub model_weights: Vec<Vec<f32>>,
    pub model_bias: Vec<f32>,
}

impl DockingProxy {
    /// Two-layer MLP surrogate (input_dim → 32 → 1).
    pub fn new(input_dim: usize, rng: &mut impl Rng) -> Self {
        let hidden = 32usize;
        let scale1 = (2.0_f32 / input_dim.max(1) as f32).sqrt();
        let scale2 = (2.0_f32 / hidden as f32).sqrt();

        // Layer 1: input_dim rows × hidden cols stored as Vec<Vec<f32>>.
        let mut model_weights: Vec<Vec<f32>> = Vec::new();
        let mut model_bias: Vec<f32> = Vec::new();

        // Hidden weights: hidden × input_dim.
        for _ in 0..hidden {
            let row: Vec<f32> = (0..input_dim)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale1)
                .collect();
            model_weights.push(row);
            model_bias.push((rng.random::<f32>() * 2.0 - 1.0) * scale1);
        }

        // Output weights: 1 × hidden.
        let out_row: Vec<f32> = (0..hidden)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale2)
            .collect();
        model_weights.push(out_row);
        model_bias.push((rng.random::<f32>() * 2.0 - 1.0) * scale2);

        DockingProxy {
            model_weights,
            model_bias,
        }
    }

    /// Forward pass: input → ReLU(hidden) → output (scalar docking score).
    pub fn score(&self, mol_features: &[f32]) -> f32 {
        if self.model_weights.len() < 2 {
            return 0.0;
        }
        let hidden_size = self.model_weights.len() - 1;

        // Hidden layer.
        let hidden_out: Vec<f32> = (0..hidden_size)
            .map(|h| {
                let w = &self.model_weights[h];
                let b = self.model_bias.get(h).copied().unwrap_or(0.0);
                let z: f32 = w
                    .iter()
                    .zip(mol_features.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f32>()
                    + b;
                z.max(0.0) // ReLU
            })
            .collect();

        // Output layer.
        let out_w = &self.model_weights[hidden_size];
        let out_b = self.model_bias.get(hidden_size).copied().unwrap_or(0.0);
        out_w
            .iter()
            .zip(hidden_out.iter())
            .map(|(wi, hi)| wi * hi)
            .sum::<f32>()
            + out_b
    }
}

/// Virtual screening pipeline.
#[derive(Debug, Clone)]
pub struct VirtualScreener {
    pub scoring_fn: DockingProxy,
    pub n_features: usize,
}

impl VirtualScreener {
    /// Create a screener with a randomly initialised docking proxy.
    pub fn new(n_features: usize, rng: &mut impl Rng) -> Self {
        VirtualScreener {
            scoring_fn: DockingProxy::new(n_features, rng),
            n_features,
        }
    }

    /// Convert a molecule to a feature vector (fingerprint bits folded to n_features).
    fn molecule_features(&self, mol: &DdMolecule) -> Vec<f32> {
        let fp = ecfp4(mol, 2, self.n_features);
        fp.bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
    }

    /// Score all molecules in a library; return `(mol_idx, score)` sorted ascending.
    pub fn screen(&self, library: &[DdMolecule]) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = library
            .iter()
            .enumerate()
            .map(|(i, mol)| {
                let feats = self.molecule_features(mol);
                let score = self.scoring_fn.score(&feats);
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Return indices of molecules passing Lipinski Rule-of-Five.
    pub fn filter_lipinski(library: &[DdMolecule]) -> Vec<usize> {
        library
            .iter()
            .enumerate()
            .filter(|(_, mol)| AdmetDescriptors::from_molecule(mol).lipinski_ro5())
            .map(|(i, _)| i)
            .collect()
    }

    /// Return indices of molecules passing Veber oral bioavailability rules.
    pub fn filter_veber(library: &[DdMolecule]) -> Vec<usize> {
        library
            .iter()
            .enumerate()
            .filter(|(_, mol)| AdmetDescriptors::from_molecule(mol).veber_rules())
            .map(|(i, _)| i)
            .collect()
    }

    /// Enrichment factor at 1%: fraction of actives in top 1% vs. overall.
    pub fn enrich_active(scores: &[(usize, f32)], actives: &[usize]) -> f32 {
        if scores.is_empty() || actives.is_empty() {
            return 0.0;
        }
        let top_n = (scores.len() as f32 * 0.01).ceil() as usize;
        let top_n = top_n.max(1);
        let actives_set: std::collections::HashSet<usize> = actives.iter().copied().collect();
        let hits_in_top = scores
            .iter()
            .take(top_n)
            .filter(|(idx, _)| actives_set.contains(idx))
            .count();
        let frac_actives = actives.len() as f32 / scores.len() as f32;
        if frac_actives <= 0.0 {
            return 0.0;
        }
        (hits_in_top as f32 / top_n as f32) / frac_actives
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 Scaffold Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Bemis–Murcko scaffold decomposition (ring-system based).
pub struct BemisMurckoScaffold;

impl BemisMurckoScaffold {
    /// Identify ring atoms using a DFS-based cycle detection.
    ///
    /// Returns indices of atoms that are part of any ring (the scaffold).
    pub fn get_scaffold(mol: &DdMolecule) -> Vec<usize> {
        let n = mol.n_atoms();
        if n == 0 {
            return Vec::new();
        }
        // Build adjacency list.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in &mol.bonds {
            if b.atom1 < n && b.atom2 < n && b.atom1 != b.atom2 {
                adj[b.atom1].push(b.atom2);
                adj[b.atom2].push(b.atom1);
            }
        }

        // DFS to find back-edges (cycles).
        let mut in_ring = vec![false; n];
        let mut visited = vec![false; n];
        let mut parent = vec![usize::MAX; n];
        let mut path_atoms: Vec<usize> = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            // Iterative DFS with explicit stack.
            let mut stack: Vec<(usize, usize, bool)> = vec![(start, usize::MAX, false)];
            while let Some((node, par, entering)) = stack.pop() {
                if !entering {
                    if visited[node] {
                        // Back edge — mark the cycle.
                        // Walk up the parent chain from par to node.
                        let mut cur = par;
                        while cur != node && cur != usize::MAX {
                            in_ring[cur] = true;
                            cur = parent[cur];
                        }
                        if cur == node {
                            in_ring[node] = true;
                        }
                        continue;
                    }
                    visited[node] = true;
                    parent[node] = par;
                    path_atoms.push(node);
                    // Push leaving marker.
                    stack.push((node, par, true));
                    for &nb in &adj[node] {
                        if nb != par {
                            stack.push((nb, node, false));
                        }
                    }
                } else {
                    // Leaving node.
                    if let Some(pos) = path_atoms.iter().rposition(|&x| x == node) {
                        path_atoms.truncate(pos);
                    }
                }
            }
        }

        (0..n).filter(|&i| in_ring[i]).collect()
    }

    /// Tanimoto similarity between scaffolds of two molecules (ECFP4).
    pub fn scaffold_similarity(mol1: &DdMolecule, mol2: &DdMolecule) -> f32 {
        let scaffold1_idx = Self::get_scaffold(mol1);
        let scaffold2_idx = Self::get_scaffold(mol2);

        // Build sub-molecules from scaffold atoms.
        let sub1 = Self::sub_molecule(mol1, &scaffold1_idx);
        let sub2 = Self::sub_molecule(mol2, &scaffold2_idx);

        let fp1 = ecfp4(&sub1, 2, 1024);
        let fp2 = ecfp4(&sub2, 2, 1024);
        dd_tanimoto(&fp1, &fp2)
    }

    fn sub_molecule(mol: &DdMolecule, atom_indices: &[usize]) -> DdMolecule {
        if atom_indices.is_empty() {
            return DdMolecule {
                atoms: Vec::new(),
                bonds: Vec::new(),
                smiles: String::new(),
            };
        }
        let idx_set: std::collections::HashSet<usize> = atom_indices.iter().copied().collect();
        // Remap indices.
        let mut remap = vec![usize::MAX; mol.n_atoms()];
        for (new_i, &old_i) in atom_indices.iter().enumerate() {
            if old_i < mol.n_atoms() {
                remap[old_i] = new_i;
            }
        }
        let atoms: Vec<DdAtom> = atom_indices
            .iter()
            .filter(|&&i| i < mol.n_atoms())
            .map(|&i| mol.atoms[i].clone())
            .collect();
        let bonds: Vec<DdBond> = mol
            .bonds
            .iter()
            .filter(|b| idx_set.contains(&b.atom1) && idx_set.contains(&b.atom2))
            .map(|b| DdBond {
                atom1: remap[b.atom1],
                atom2: remap[b.atom2],
                bond_type: b.bond_type.clone(),
            })
            .collect();
        DdMolecule {
            atoms,
            bonds,
            smiles: String::new(),
        }
    }
}

/// Diversity-picking utilities.
pub struct DiversityPicker;

impl DiversityPicker {
    /// MaxMin diverse subset selection.
    ///
    /// Iteratively selects the fingerprint most dissimilar to the current pick set.
    pub fn pick(fingerprints: &[DdMolecularFingerprint], n_to_pick: usize) -> Vec<usize> {
        if fingerprints.is_empty() || n_to_pick == 0 {
            return Vec::new();
        }
        let n = fingerprints.len();
        let n_to_pick = n_to_pick.min(n);
        let mut picked: Vec<usize> = Vec::with_capacity(n_to_pick);
        let mut min_dist = vec![0.0_f32; n]; // min Tanimoto distance to picked set

        // Start with the first fingerprint.
        picked.push(0);
        for i in 0..n {
            min_dist[i] = 1.0 - dd_tanimoto(&fingerprints[0], &fingerprints[i]);
        }

        while picked.len() < n_to_pick {
            // Pick the index with maximum min-distance to picked set.
            let next = (0..n)
                .filter(|i| !picked.contains(i))
                .max_by(|&a, &b| {
                    min_dist[a]
                        .partial_cmp(&min_dist[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            picked.push(next);
            // Update min distances.
            for i in 0..n {
                let d = 1.0 - dd_tanimoto(&fingerprints[next], &fingerprints[i]);
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
        }

        picked
    }

    /// Mean pairwise Tanimoto dissimilarity (diversity score ∈ [0, 1]).
    pub fn diversity_score(fingerprints: &[DdMolecularFingerprint]) -> f32 {
        let n = fingerprints.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0_f32;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                total += 1.0 - dd_tanimoto(&fingerprints[i], &fingerprints[j]);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }
}

/// Scaffold-based and Butina clustering.
pub struct ScaffoldCluster;

impl ScaffoldCluster {
    /// Group molecule indices by their Bemis–Murcko scaffold (ring atom set).
    pub fn cluster_by_scaffold(molecules: &[DdMolecule]) -> Vec<Vec<usize>> {
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut scaffold_fps: Vec<DdMolecularFingerprint> = Vec::new();

        for (i, mol) in molecules.iter().enumerate() {
            let scaffold_idx = BemisMurckoScaffold::get_scaffold(mol);
            let sub = BemisMurckoScaffold::sub_molecule(mol, &scaffold_idx);
            let fp = ecfp4(&sub, 2, 512);

            // Find an existing cluster with Tanimoto > 0.8.
            let mut assigned = false;
            for (c, cluster) in clusters.iter_mut().enumerate() {
                if dd_tanimoto(&fp, &scaffold_fps[c]) > 0.8 {
                    cluster.push(i);
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                clusters.push(vec![i]);
                scaffold_fps.push(fp);
            }
        }

        clusters
    }

    /// Butina leader-follower clustering: assign each molecule to the first
    /// existing cluster whose leader has Tanimoto > `threshold`.
    pub fn butina_clustering(
        fingerprints: &[DdMolecularFingerprint],
        threshold: f32,
    ) -> Vec<Vec<usize>> {
        let mut leaders: Vec<usize> = Vec::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for (i, fp) in fingerprints.iter().enumerate() {
            let mut assigned = false;
            for (c, &leader) in leaders.iter().enumerate() {
                if dd_tanimoto(fp, &fingerprints[leader]) > threshold {
                    clusters[c].push(i);
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                leaders.push(i);
                clusters.push(vec![i]);
            }
        }

        clusters
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6 Molecular Generative Models
// ─────────────────────────────────────────────────────────────────────────────

/// Character-level SMILES tokenizer.
#[derive(Debug, Clone)]
pub struct SmileTokenizer {
    pub vocab: Vec<char>,
    pub unk_id: usize,
}

impl SmileTokenizer {
    /// Build a character vocabulary from a list of SMILES strings.
    pub fn from_smiles(smiles_list: &[&str]) -> Self {
        let mut char_set: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
        // Always include special tokens: <pad>=0, <bos>=1, <eos>=2.
        char_set.insert('\0'); // pad
        char_set.insert('^'); // bos
        char_set.insert('$'); // eos
        for s in smiles_list {
            for c in s.chars() {
                char_set.insert(c);
            }
        }
        char_set.insert('?'); // unk
        let vocab: Vec<char> = char_set.into_iter().collect();
        let unk_id = vocab.iter().position(|&c| c == '?').unwrap_or(0);
        SmileTokenizer { vocab, unk_id }
    }

    /// Encode a SMILES string to a token id sequence.
    pub fn encode(&self, smiles: &str) -> Vec<usize> {
        smiles
            .chars()
            .map(|c| {
                self.vocab
                    .iter()
                    .position(|&v| v == c)
                    .unwrap_or(self.unk_id)
            })
            .collect()
    }

    /// Decode a token id sequence back to a SMILES string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&t| self.vocab.get(t).copied().unwrap_or('?'))
            .filter(|&c| c != '\0' && c != '^' && c != '$')
            .collect()
    }
}

/// Character-level RNN for SMILES generation (embedding + single LSTM cell + projection).
#[derive(Debug, Clone)]
pub struct SmileRnn {
    pub embedding: Vec<Vec<f32>>,
    pub lstm_weights: Vec<f32>,
    pub output_proj: Vec<Vec<f32>>,
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

impl SmileRnn {
    /// Initialise the RNN with Xavier weights.
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let embed_scale = (1.0_f32 / embed_dim.max(1) as f32).sqrt();
        let embedding: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * embed_scale)
                    .collect()
            })
            .collect();

        // LSTM: 4 gates × (embed_dim + hidden_dim + 1 bias) weights.
        let lstm_input = embed_dim + hidden_dim;
        let lstm_len = 4 * (lstm_input + 1) * hidden_dim;
        let lstm_scale = (2.0_f32 / lstm_input.max(1) as f32).sqrt();
        let lstm_weights: Vec<f32> = (0..lstm_len)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * lstm_scale)
            .collect();

        // Output projection: hidden_dim → vocab_size.
        let proj_scale = (2.0_f32 / hidden_dim.max(1) as f32).sqrt();
        let output_proj: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * proj_scale)
                    .collect()
            })
            .collect();

        SmileRnn {
            embedding,
            lstm_weights,
            output_proj,
            vocab_size,
            hidden_dim,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    /// LSTM step: returns new (h, c).
    fn lstm_step(
        &self,
        input: &[f32],
        h: &[f32],
        c: &[f32],
        embed_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let hd = self.hidden_dim;
        let lstm_in = embed_dim + hd;
        // Concatenate input and h.
        let xh: Vec<f32> = input.iter().chain(h.iter()).copied().collect();

        // Gate weights layout: [i_gate | f_gate | g_gate | o_gate]
        // Each gate: hd rows × (lstm_in + 1) cols (last = bias).
        let gate_size = (lstm_in + 1) * hd;

        let compute_gate = |gate_offset: usize, activation: fn(f32) -> f32| -> Vec<f32> {
            (0..hd)
                .map(|h_i| {
                    let base = gate_offset + h_i * (lstm_in + 1);
                    let z: f32 = xh
                        .iter()
                        .enumerate()
                        .map(|(xi, &x)| {
                            self.lstm_weights.get(base + xi).copied().unwrap_or(0.0) * x
                        })
                        .sum::<f32>()
                        + self
                            .lstm_weights
                            .get(base + lstm_in)
                            .copied()
                            .unwrap_or(0.0);
                    activation(z)
                })
                .collect()
        };

        let i_gate = compute_gate(0, Self::sigmoid);
        let f_gate = compute_gate(gate_size, Self::sigmoid);
        let g_gate = compute_gate(gate_size * 2, Self::tanh);
        let o_gate = compute_gate(gate_size * 3, Self::sigmoid);

        let new_c: Vec<f32> = (0..hd)
            .map(|k| f_gate[k] * c[k] + i_gate[k] * g_gate[k])
            .collect();
        let new_h: Vec<f32> = (0..hd).map(|k| o_gate[k] * Self::tanh(new_c[k])).collect();

        (new_h, new_c)
    }

    /// Project hidden state to vocab-size logits.
    fn project(&self, h: &[f32]) -> Vec<f32> {
        self.output_proj
            .iter()
            .map(|row| row.iter().zip(h.iter()).map(|(w, hi)| w * hi).sum::<f32>())
            .collect()
    }

    /// Forward pass: returns logits at each position (T × vocab_size).
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        if tokens.is_empty() || self.embedding.is_empty() {
            return Vec::new();
        }
        let embed_dim = self.embedding[0].len();
        let mut h = vec![0.0_f32; self.hidden_dim];
        let mut c = vec![0.0_f32; self.hidden_dim];
        let mut logits_seq: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());

        for &tok in tokens {
            let embed = self
                .embedding
                .get(tok)
                .cloned()
                .unwrap_or_else(|| self.embedding.first().cloned().unwrap_or_default());
            let (new_h, new_c) = self.lstm_step(&embed, &h, &c, embed_dim);
            h = new_h;
            c = new_c;
            logits_seq.push(self.project(&h));
        }

        logits_seq
    }

    /// Sample a token sequence using temperature-scaled softmax.
    pub fn sample(
        &self,
        start_token: usize,
        max_len: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        if self.embedding.is_empty() || max_len == 0 {
            return Vec::new();
        }
        let embed_dim = self.embedding[0].len();
        let temp = temperature.max(1e-6);

        let mut h = vec![0.0_f32; self.hidden_dim];
        let mut c = vec![0.0_f32; self.hidden_dim];
        let mut result = Vec::with_capacity(max_len);
        let mut cur_tok = start_token;

        for _ in 0..max_len {
            result.push(cur_tok);
            let embed = self
                .embedding
                .get(cur_tok)
                .cloned()
                .unwrap_or_else(|| self.embedding.first().cloned().unwrap_or_default());
            let (new_h, new_c) = self.lstm_step(&embed, &h, &c, embed_dim);
            h = new_h;
            c = new_c;

            let logits = self.project(&h);
            // Temperature-scaled softmax sampling.
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&x| ((x - max_l) / temp).exp()).collect();
            let sum_e: f32 = exp_l.iter().sum();
            let probs: Vec<f32> = if sum_e > 0.0 {
                exp_l.iter().map(|&e| e / sum_e).collect()
            } else {
                vec![1.0 / self.vocab_size as f32; self.vocab_size]
            };

            // Categorical sample.
            let u: f32 = rng.random();
            let mut cumsum = 0.0_f32;
            let mut sampled = 0usize;
            for (vi, &p) in probs.iter().enumerate() {
                cumsum += p;
                if u <= cumsum {
                    sampled = vi;
                    break;
                }
            }
            // Set end-of-sequence token if we sample '$' (eos).
            cur_tok = sampled.min(self.vocab_size.saturating_sub(1));
        }

        result
    }
}

/// Property-conditioned SMILES generation utilities.
pub struct PropertyConditionedGeneration;

impl PropertyConditionedGeneration {
    /// Build a conditioning vector from (property, value) pairs.
    ///
    /// The vector has one slot per possible [`AdmetProperty`] variant (7 slots).
    /// Slot indices correspond to the enum variant discriminant order.
    pub fn condition_vector(property_values: &[(AdmetProperty, f32)]) -> Vec<f32> {
        let mut v = vec![0.0_f32; 7];
        for (prop, val) in property_values {
            let idx = match prop {
                AdmetProperty::Solubility => 0,
                AdmetProperty::Permeability => 1,
                AdmetProperty::LogP => 2,
                AdmetProperty::HalfLife => 3,
                AdmetProperty::Toxicity => 4,
                AdmetProperty::BBBPenetration => 5,
                AdmetProperty::HERGInhibition => 6,
            };
            v[idx] = *val;
        }
        v
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7 Activity Cliff Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// A pair of molecules with high structural similarity but large activity difference.
#[derive(Debug, Clone)]
pub struct ActivityCliff {
    pub mol1_idx: usize,
    pub mol2_idx: usize,
    pub similarity: f32,
    pub activity_diff: f32,
}

/// Activity cliff detection and matched molecular pair analysis.
pub struct ActivityCliffAnalyzer;

impl ActivityCliffAnalyzer {
    /// Detect activity cliffs: pairs where Tanimoto > `sim_threshold` and
    /// |activity difference| > `act_threshold`.
    pub fn detect_cliffs(
        fingerprints: &[DdMolecularFingerprint],
        activities: &[f32],
        sim_threshold: f32,
        act_threshold: f32,
    ) -> Vec<ActivityCliff> {
        let n = fingerprints.len().min(activities.len());
        let mut cliffs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = dd_tanimoto(&fingerprints[i], &fingerprints[j]);
                let diff = (activities[i] - activities[j]).abs();
                if sim > sim_threshold && diff > act_threshold {
                    cliffs.push(ActivityCliff {
                        mol1_idx: i,
                        mol2_idx: j,
                        similarity: sim,
                        activity_diff: diff,
                    });
                }
            }
        }
        cliffs
    }

    /// Activity cliff score: similarity × |activity_diff|.
    pub fn cliff_score(similarity: f32, activity_diff: f32) -> f32 {
        similarity * activity_diff.abs()
    }

    /// Matched molecular pair: identify the bond index pairs that differ by exactly
    /// one heavy-atom substitution (simple heuristic: one bond in mol1 has no
    /// counterpart in mol2 and vice versa).
    ///
    /// Returns `Some((bond_idx_mol1, bond_idx_mol2))` for the first such pair found.
    pub fn mmp_transform(mol1: &DdMolecule, mol2: &DdMolecule) -> Option<(usize, usize)> {
        // Simple heuristic: compare bond counts in each molecule.
        // If each molecule has exactly one unique bond (by atom-type signature),
        // return its indices.
        let sig1: Vec<(u8, u8)> = mol1
            .bonds
            .iter()
            .map(|b| {
                let t1 = atom_type_ordinal(&mol1.atoms.get(b.atom1).map(|a| &a.element));
                let t2 = atom_type_ordinal(&mol1.atoms.get(b.atom2).map(|a| &a.element));
                (t1.min(t2), t1.max(t2))
            })
            .collect();
        let sig2: Vec<(u8, u8)> = mol2
            .bonds
            .iter()
            .map(|b| {
                let t1 = atom_type_ordinal(&mol2.atoms.get(b.atom1).map(|a| &a.element));
                let t2 = atom_type_ordinal(&mol2.atoms.get(b.atom2).map(|a| &a.element));
                (t1.min(t2), t1.max(t2))
            })
            .collect();

        // Find bonds in mol1 not in mol2.
        let mut unique1: Vec<usize> = sig1
            .iter()
            .enumerate()
            .filter(|(_, s)| !sig2.contains(s))
            .map(|(i, _)| i)
            .collect();
        let mut unique2: Vec<usize> = sig2
            .iter()
            .enumerate()
            .filter(|(_, s)| !sig1.contains(s))
            .map(|(i, _)| i)
            .collect();

        if unique1.len() == 1 && unique2.len() == 1 {
            Some((unique1.remove(0), unique2.remove(0)))
        } else {
            None
        }
    }
}

fn atom_type_ordinal(atype: &Option<&DdAtomType>) -> u8 {
    match atype {
        Some(DdAtomType::Carbon) => 1,
        Some(DdAtomType::Nitrogen) => 2,
        Some(DdAtomType::Oxygen) => 3,
        Some(DdAtomType::Sulfur) => 4,
        Some(DdAtomType::Phosphorus) => 5,
        Some(DdAtomType::Fluorine) => 6,
        Some(DdAtomType::Chlorine) => 7,
        Some(DdAtomType::Bromine) => 8,
        Some(DdAtomType::Iodine) => 9,
        Some(DdAtomType::Other) | None => 0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8 Drug Discovery Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Area under the ROC curve via the trapezoidal rule.
///
/// `scores` are the raw model scores (lower = better predicted active).
/// `labels` are true active (true) / inactive (false) flags.
pub fn auc_roc_virtual_screening(scores: &[f32], labels: &[bool]) -> f32 {
    if scores.len() != labels.len() || scores.is_empty() {
        return 0.5;
    }
    // Sort descending by score (lower score = more active for docking proxy).
    let mut paired: Vec<(f32, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = paired.iter().filter(|&&(_, l)| l).count();
    let n_neg = paired.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    // Trapezoidal AUC via ROC walk.
    let (mut tp, mut fp) = (0usize, 0usize);
    let (mut prev_tp, mut prev_fp) = (0usize, 0usize);
    let mut auc = 0.0_f32;

    for &(_, label) in &paired {
        if label {
            tp += 1;
        } else {
            fp += 1;
        }
        // Add trapezoid.
        let delta_fp = fp - prev_fp;
        let avg_tp = (tp + prev_tp) as f32 / 2.0;
        auc += (delta_fp as f32) * avg_tp / (n_pos as f32 * n_neg as f32);
        prev_tp = tp;
        prev_fp = fp;
    }

    auc.clamp(0.0, 1.0)
}

/// Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic (BEDROC).
///
/// Uses `alpha = 20.0` (standard for VS benchmarks).
pub fn boltzmann_enhanced_discrimination(scores: &[f32], labels: &[bool]) -> f32 {
    let alpha = 20.0_f32;
    let n = scores.len();
    if n == 0 || scores.len() != labels.len() {
        return 0.0;
    }
    let mut paired: Vec<(f32, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = paired.iter().filter(|&&(_, l)| l).count() as f32;
    if n_pos == 0.0 {
        return 0.0;
    }
    let n_total = n as f32;
    let ra = n_pos / n_total;

    // Sum of exp(-alpha * ri / n) for actives, where ri is 1-based rank.
    let sum_exp: f32 = paired
        .iter()
        .enumerate()
        .filter(|(_, (_, l))| *l)
        .map(|(i, _)| (-(alpha * (i + 1) as f32) / n_total).exp())
        .sum();

    let ri_factor = ra * (1.0 - (-alpha).exp()) / (1.0 - (-alpha / n_total).exp());

    // Normalise and clamp.
    if ri_factor <= 0.0 {
        return 0.0;
    }
    (sum_exp / ri_factor).clamp(0.0, 1.0)
}

/// Fraction of screening hits that have a scaffold different from the original.
pub fn scaffold_hop_rate(original_scaffold: &[usize], hits: &[DdMolecule]) -> f32 {
    if hits.is_empty() {
        return 0.0;
    }
    let original_len = original_scaffold.len();
    let n_hops = hits
        .iter()
        .filter(|mol| {
            let hit_scaffold = BemisMurckoScaffold::get_scaffold(mol);
            // Consider a scaffold hop if the ring-count differs.
            hit_scaffold.len() != original_len
        })
        .count();
    n_hops as f32 / hits.len() as f32
}

/// Summary report for a virtual screening campaign.
#[derive(Debug, Clone)]
pub struct DrugDiscoveryReport {
    pub auc: f32,
    pub bedroc: f32,
    pub enrich_1pct: f32,
    pub n_actives: usize,
    pub n_diverse_hits: usize,
}

/// Evaluate a virtual screening campaign.
pub fn evaluate_campaign(
    scores: &[(usize, f32)],
    actives: &[usize],
    fingerprints: &[DdMolecularFingerprint],
) -> DrugDiscoveryReport {
    // Build flat score and label vectors.
    let n = scores.len();
    let actives_set: std::collections::HashSet<usize> = actives.iter().copied().collect();

    let flat_scores: Vec<f32> = scores.iter().map(|(_, s)| *s).collect();
    let flat_labels: Vec<bool> = scores
        .iter()
        .map(|(i, _)| actives_set.contains(i))
        .collect();

    let auc = auc_roc_virtual_screening(&flat_scores, &flat_labels);
    let bedroc = boltzmann_enhanced_discrimination(&flat_scores, &flat_labels);
    let enrich_1pct = VirtualScreener::enrich_active(scores, actives);

    // Diverse hits: top 10% of hits that are maximally diverse.
    let top_n = (n as f32 * 0.1).ceil() as usize;
    let top_fps: Vec<DdMolecularFingerprint> = scores
        .iter()
        .take(top_n)
        .filter_map(|(idx, _)| fingerprints.get(*idx).cloned())
        .collect();
    let n_diverse_hits = if top_fps.len() > 1 {
        let picked = DiversityPicker::pick(&top_fps, top_fps.len().min(10));
        picked.len()
    } else {
        top_fps.len()
    };

    DrugDiscoveryReport {
        auc,
        bedroc,
        enrich_1pct,
        n_actives: actives.len(),
        n_diverse_hits,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn benzene() -> DdMolecule {
        // 6 aromatic carbons in a ring.
        let atoms: Vec<DdAtom> = (0..6)
            .map(|_| DdAtom {
                element: DdAtomType::Carbon,
                formal_charge: 0,
                n_hydrogens: 1,
                is_aromatic: true,
            })
            .collect();
        let bonds: Vec<DdBond> = (0..6)
            .map(|i| DdBond {
                atom1: i,
                atom2: (i + 1) % 6,
                bond_type: DdBondType::Aromatic,
            })
            .collect();
        DdMolecule {
            atoms,
            bonds,
            smiles: "c1ccccc1".to_string(),
        }
    }

    fn simple_chain() -> DdMolecule {
        // CH3-CH2-OH  (3 heavy atoms, 2 single bonds)
        let atoms = vec![
            DdAtom {
                element: DdAtomType::Carbon,
                formal_charge: 0,
                n_hydrogens: 3,
                is_aromatic: false,
            },
            DdAtom {
                element: DdAtomType::Carbon,
                formal_charge: 0,
                n_hydrogens: 2,
                is_aromatic: false,
            },
            DdAtom {
                element: DdAtomType::Oxygen,
                formal_charge: 0,
                n_hydrogens: 1,
                is_aromatic: false,
            },
        ];
        let bonds = vec![
            DdBond {
                atom1: 0,
                atom2: 1,
                bond_type: DdBondType::Single,
            },
            DdBond {
                atom1: 1,
                atom2: 2,
                bond_type: DdBondType::Single,
            },
        ];
        DdMolecule {
            atoms,
            bonds,
            smiles: "CCO".to_string(),
        }
    }

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // ── §1 Molecular Representation ───────────────────────────────────────

    #[test]
    fn test_atom_creation() {
        let a = DdAtom {
            element: DdAtomType::Carbon,
            formal_charge: 0,
            n_hydrogens: 4,
            is_aromatic: false,
        };
        assert_eq!(a.element, DdAtomType::Carbon);
        assert_eq!(a.formal_charge, 0);
        assert_eq!(a.n_hydrogens, 4);
        assert!(!a.is_aromatic);
    }

    #[test]
    fn test_molecule_n_atoms() {
        let mol = benzene();
        assert_eq!(mol.n_atoms(), 6);
    }

    #[test]
    fn test_molecule_n_bonds() {
        let mol = benzene();
        assert_eq!(mol.n_bonds(), 6);
    }

    #[test]
    fn test_adjacency_matrix_shape() {
        let mol = benzene();
        let adj = mol.adjacency_matrix();
        assert_eq!(adj.len(), 6);
        for row in &adj {
            assert_eq!(row.len(), 6);
        }
    }

    #[test]
    fn test_adjacency_matrix_symmetric() {
        let mol = benzene();
        let adj = mol.adjacency_matrix();
        let n = adj.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (adj[i][j] - adj[j][i]).abs() < 1e-6,
                    "adj[{i}][{j}] != adj[{j}][{i}]"
                );
            }
        }
    }

    #[test]
    fn test_molecular_weight_positive() {
        let mol = benzene();
        assert!(mol.molecular_weight() > 0.0);
    }

    // ── §2 Fingerprints ───────────────────────────────────────────────────

    #[test]
    fn test_ecfp4_length() {
        let mol = benzene();
        let fp = ecfp4(&mol, 2, 1024);
        assert_eq!(fp.length, 1024);
        assert_eq!(fp.bits.len(), 1024);
    }

    #[test]
    fn test_ecfp4_same_molecule_equal() {
        let mol = benzene();
        let fp1 = ecfp4(&mol, 2, 1024);
        let fp2 = ecfp4(&mol, 2, 1024);
        assert_eq!(fp1.bits, fp2.bits);
    }

    #[test]
    fn test_tanimoto_same_molecule() {
        let mol = benzene();
        let fp = ecfp4(&mol, 2, 512);
        let sim = dd_tanimoto(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-6, "Expected 1.0, got {sim}");
    }

    #[test]
    fn test_tanimoto_different_molecules() {
        let mol1 = benzene();
        let mol2 = simple_chain();
        let fp1 = ecfp4(&mol1, 2, 512);
        let fp2 = ecfp4(&mol2, 2, 512);
        let sim = dd_tanimoto(&fp1, &fp2);
        assert!(
            sim < 1.0,
            "Dissimilar molecules should have sim < 1.0, got {sim}"
        );
    }

    #[test]
    fn test_tanimoto_empty_both() {
        let fp1 = DdMolecularFingerprint::new(128);
        let fp2 = DdMolecularFingerprint::new(128);
        let sim = dd_tanimoto(&fp1, &fp2);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_tanimoto_symmetric() {
        let mol1 = benzene();
        let mol2 = simple_chain();
        let fp1 = ecfp4(&mol1, 2, 256);
        let fp2 = ecfp4(&mol2, 2, 256);
        let sim_ab = dd_tanimoto(&fp1, &fp2);
        let sim_ba = dd_tanimoto(&fp2, &fp1);
        assert!((sim_ab - sim_ba).abs() < 1e-6);
    }

    // ── §3 ADMET ──────────────────────────────────────────────────────────

    #[test]
    fn test_admet_descriptors_from_molecule() {
        let mol = simple_chain();
        let desc = AdmetDescriptors::from_molecule(&mol);
        assert!(desc.mw > 0.0);
        assert!(desc.hba >= 1); // oxygen
    }

    #[test]
    fn test_lipinski_ro5_satisfied() {
        let mol = simple_chain();
        let desc = AdmetDescriptors::from_molecule(&mol);
        assert!(desc.lipinski_ro5(), "Simple chain should pass Lipinski RO5");
    }

    #[test]
    fn test_lipinski_ro5_violated_high_mw() {
        // Construct a big molecule manually.
        let atoms: Vec<DdAtom> = (0..50)
            .map(|_| DdAtom {
                element: DdAtomType::Carbon,
                formal_charge: 0,
                n_hydrogens: 2,
                is_aromatic: false,
            })
            .collect();
        let bonds: Vec<DdBond> = (0..49)
            .map(|i| DdBond {
                atom1: i,
                atom2: i + 1,
                bond_type: DdBondType::Single,
            })
            .collect();
        let mol = DdMolecule {
            atoms,
            bonds,
            smiles: String::new(),
        };
        let desc = AdmetDescriptors::from_molecule(&mol);
        assert!(
            !desc.lipinski_ro5(),
            "50-carbon chain should violate MW≤500, desc.mw={}",
            desc.mw
        );
    }

    #[test]
    fn test_veber_rules_satisfied() {
        let mol = simple_chain();
        let desc = AdmetDescriptors::from_molecule(&mol);
        assert!(desc.veber_rules(), "Simple chain should pass Veber rules");
    }

    #[test]
    fn test_admet_model_creation() {
        let mut rng = make_rng();
        let model = AdmetModel::new(AdmetProperty::LogP, 6, &mut rng);
        assert_eq!(model.input_dim, 6);
        assert_eq!(model.weights.len(), 6);
    }

    #[test]
    fn test_admet_predict_from_descriptors_finite() {
        let mut rng = make_rng();
        let model = AdmetModel::new(AdmetProperty::Solubility, 6, &mut rng);
        let mol = simple_chain();
        let desc = AdmetDescriptors::from_molecule(&mol);
        let pred = model.predict_from_descriptors(&desc);
        assert!(pred.is_finite(), "Prediction should be finite, got {pred}");
    }

    #[test]
    fn test_admet_predict_from_fingerprint_finite() {
        let mut rng = make_rng();
        let model = AdmetModel::new(AdmetProperty::BBBPenetration, 512, &mut rng);
        let mol = benzene();
        let fp = ecfp4(&mol, 2, 512);
        let pred = model.predict_from_fingerprint(&fp);
        assert!(
            pred.is_finite(),
            "FP prediction should be finite, got {pred}"
        );
    }

    // ── §4 Virtual Screening ──────────────────────────────────────────────

    #[test]
    fn test_docking_proxy_creation() {
        let mut rng = make_rng();
        let proxy = DockingProxy::new(64, &mut rng);
        assert!(!proxy.model_weights.is_empty());
    }

    #[test]
    fn test_docking_proxy_score_finite() {
        let mut rng = make_rng();
        let proxy = DockingProxy::new(64, &mut rng);
        let feats: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let score = proxy.score(&feats);
        assert!(
            score.is_finite(),
            "Docking score should be finite, got {score}"
        );
    }

    #[test]
    fn test_virtual_screener_screen_sorted() {
        let mut rng = make_rng();
        let screener = VirtualScreener::new(64, &mut rng);
        let library = vec![benzene(), simple_chain(), benzene()];
        let results = screener.screen(&library);
        assert_eq!(results.len(), 3);
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "Results should be sorted ascending");
        }
    }

    #[test]
    fn test_virtual_screener_screen_length() {
        let mut rng = make_rng();
        let screener = VirtualScreener::new(64, &mut rng);
        let library = vec![benzene(), simple_chain()];
        let results = screener.screen(&library);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_lipinski_subset() {
        let library = vec![benzene(), simple_chain()];
        let indices = VirtualScreener::filter_lipinski(&library);
        for &i in &indices {
            assert!(i < library.len(), "Index {i} out of range");
        }
    }

    #[test]
    fn test_enrich_active_range() {
        let scores: Vec<(usize, f32)> = (0..100).map(|i| (i, i as f32)).collect();
        let actives: Vec<usize> = (0..10).collect(); // top 10 are actives
        let ef = VirtualScreener::enrich_active(&scores, &actives);
        assert!(ef >= 0.0, "Enrichment factor must be non-negative");
    }

    // ── §5 Scaffold Analysis ──────────────────────────────────────────────

    #[test]
    fn test_get_scaffold_indices() {
        let mol = benzene();
        let scaffold = BemisMurckoScaffold::get_scaffold(&mol);
        // Benzene is all ring — all 6 atoms.
        assert_eq!(scaffold.len(), 6);
    }

    #[test]
    fn test_scaffold_similarity_range() {
        let mol1 = benzene();
        let mol2 = simple_chain();
        let sim = BemisMurckoScaffold::scaffold_similarity(&mol1, &mol2);
        assert!((0.0..=1.0).contains(&sim), "Similarity out of [0,1]: {sim}");
    }

    #[test]
    fn test_maxmin_picker_count() {
        let mols = [benzene(), simple_chain(), benzene()];
        let fps: Vec<DdMolecularFingerprint> = mols.iter().map(|m| ecfp4(m, 2, 256)).collect();
        let picked = DiversityPicker::pick(&fps, 2);
        assert_eq!(picked.len(), 2);
    }

    #[test]
    fn test_maxmin_picker_distinct() {
        let mols = [benzene(), simple_chain(), benzene()];
        let fps: Vec<DdMolecularFingerprint> = mols.iter().map(|m| ecfp4(m, 2, 256)).collect();
        let picked = DiversityPicker::pick(&fps, 3);
        let unique: std::collections::HashSet<usize> = picked.iter().copied().collect();
        assert_eq!(
            unique.len(),
            picked.len(),
            "Picked indices should be distinct"
        );
    }

    #[test]
    fn test_diversity_score_range() {
        let mols = [benzene(), simple_chain()];
        let fps: Vec<DdMolecularFingerprint> = mols.iter().map(|m| ecfp4(m, 2, 256)).collect();
        let ds = DiversityPicker::diversity_score(&fps);
        assert!((0.0..=1.0).contains(&ds), "Diversity score out of [0,1]: {ds}");
    }

    #[test]
    fn test_butina_clustering_all_assigned() {
        let mols = [benzene(), simple_chain(), benzene()];
        let fps: Vec<DdMolecularFingerprint> = mols.iter().map(|m| ecfp4(m, 2, 256)).collect();
        let clusters = ScaffoldCluster::butina_clustering(&fps, 0.5);
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(
            total,
            mols.len(),
            "All molecules should be assigned to a cluster"
        );
    }

    // ── §6 Generative Models ──────────────────────────────────────────────

    #[test]
    fn test_smile_tokenizer_encode_decode_roundtrip() {
        let smiles = &["CCO", "c1ccccc1", "CC(=O)O"];
        let tok = SmileTokenizer::from_smiles(smiles);
        let encoded = tok.encode("CCO");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "CCO");
    }

    #[test]
    fn test_smile_rnn_forward_shape() {
        let smiles = &["CCO", "c1ccccc1"];
        let tok = SmileTokenizer::from_smiles(smiles);
        let vocab_size = tok.vocab.len();
        let mut rng = make_rng();
        let rnn = SmileRnn::new(vocab_size, 8, 16, &mut rng);
        let tokens = tok.encode("CCO");
        let logits = rnn.forward(&tokens);
        assert_eq!(logits.len(), tokens.len());
        for row in &logits {
            assert_eq!(row.len(), vocab_size);
        }
    }

    #[test]
    fn test_smile_rnn_sample_length() {
        let smiles = &["CCO", "c1ccccc1"];
        let tok = SmileTokenizer::from_smiles(smiles);
        let vocab_size = tok.vocab.len();
        let mut rng = make_rng();
        let rnn = SmileRnn::new(vocab_size, 8, 16, &mut rng);
        let sampled = rnn.sample(0, 10, 1.0, &mut rng);
        assert_eq!(sampled.len(), 10);
    }

    #[test]
    fn test_smile_rnn_sample_within_vocab() {
        let smiles = &["CCO"];
        let tok = SmileTokenizer::from_smiles(smiles);
        let vocab_size = tok.vocab.len();
        let mut rng = make_rng();
        let rnn = SmileRnn::new(vocab_size, 8, 16, &mut rng);
        let sampled = rnn.sample(0, 20, 1.0, &mut rng);
        for &t in &sampled {
            assert!(t < vocab_size, "Token {t} out of vocab range {vocab_size}");
        }
    }

    #[test]
    fn test_property_condition_vector_shape() {
        let props = vec![
            (AdmetProperty::LogP, 2.5_f32),
            (AdmetProperty::Toxicity, 0.1_f32),
        ];
        let v = PropertyConditionedGeneration::condition_vector(&props);
        assert_eq!(v.len(), 7);
        assert!((v[2] - 2.5).abs() < 1e-6); // LogP at slot 2
        assert!((v[4] - 0.1).abs() < 1e-6); // Toxicity at slot 4
    }

    // ── §7 Activity Cliffs ────────────────────────────────────────────────

    #[test]
    fn test_activity_cliff_detection_empty() {
        // Two very dissimilar molecules — no cliffs expected at high sim_threshold.
        let mols = [benzene(), simple_chain()];
        let fps: Vec<DdMolecularFingerprint> = mols.iter().map(|m| ecfp4(m, 2, 512)).collect();
        let activities = vec![5.0_f32, 0.1_f32];
        let cliffs = ActivityCliffAnalyzer::detect_cliffs(&fps, &activities, 0.99, 1.0);
        // With threshold 0.99 and dissimilar mols, expect 0 cliffs.
        assert_eq!(cliffs.len(), 0);
    }

    #[test]
    fn test_activity_cliff_detection_found() {
        // Two very similar molecules (both benzene) with large activity diff.
        let fps: Vec<DdMolecularFingerprint> =
            vec![ecfp4(&benzene(), 2, 512), ecfp4(&benzene(), 2, 512)];
        let activities = vec![0.0_f32, 10.0_f32];
        let cliffs = ActivityCliffAnalyzer::detect_cliffs(&fps, &activities, 0.9, 5.0);
        assert!(
            !cliffs.is_empty(),
            "Identical fingerprints with large activity diff should be a cliff"
        );
    }

    #[test]
    fn test_cliff_score_positive() {
        let score = ActivityCliffAnalyzer::cliff_score(0.9, 5.0);
        assert!(score > 0.0);
    }

    // ── §8 Metrics ────────────────────────────────────────────────────────

    #[test]
    fn test_auc_roc_perfect() {
        // All actives ranked first.
        let scores: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let labels: Vec<bool> = (0..10).map(|i| i < 5).collect();
        let auc = auc_roc_virtual_screening(&scores, &labels);
        assert!(auc > 0.9, "Perfect ranking should give high AUC, got {auc}");
    }

    #[test]
    fn test_auc_roc_random() {
        // Interleaved — should be near 0.5.
        let scores: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let labels: Vec<bool> = (0..20).map(|i| i % 2 == 0).collect();
        let auc = auc_roc_virtual_screening(&scores, &labels);
        assert!((0.0..=1.0).contains(&auc), "AUC must be in [0,1], got {auc}");
    }

    #[test]
    fn test_bedroc_positive() {
        let scores: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let labels: Vec<bool> = (0..20).map(|i| i < 5).collect();
        let bedroc = boltzmann_enhanced_discrimination(&scores, &labels);
        assert!(bedroc >= 0.0, "BEDROC should be non-negative, got {bedroc}");
    }

    #[test]
    fn test_scaffold_hop_rate_range() {
        let mol1 = benzene();
        let scaffold1 = BemisMurckoScaffold::get_scaffold(&mol1);
        let hits = vec![benzene(), simple_chain()];
        let rate = scaffold_hop_rate(&scaffold1, &hits);
        assert!(
            (0.0..=1.0).contains(&rate),
            "Scaffold hop rate out of [0,1]: {rate}"
        );
    }

    #[test]
    fn test_drug_discovery_report_fields() {
        let report = DrugDiscoveryReport {
            auc: 0.75,
            bedroc: 0.5,
            enrich_1pct: 3.0,
            n_actives: 10,
            n_diverse_hits: 5,
        };
        assert_eq!(report.n_actives, 10);
        assert_eq!(report.n_diverse_hits, 5);
        assert!((report.auc - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_campaign_runs() {
        let mols = [benzene(), simple_chain(), benzene(), simple_chain()];
        let fps: Vec<DdMolecularFingerprint> = mols.iter().map(|m| ecfp4(m, 2, 256)).collect();
        let scores: Vec<(usize, f32)> = vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)];
        let actives = vec![0usize, 2];
        let report = evaluate_campaign(&scores, &actives, &fps);
        assert!(report.auc >= 0.0 && report.auc <= 1.0);
        assert!(report.bedroc >= 0.0);
    }
}
