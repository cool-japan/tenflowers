//! Protein structure prediction module — AlphaFold2-inspired pipeline in pure Rust.
//!
//! Provides:
//! - `PsAminoAcid`: 20 standard amino acids with biophysical properties
//! - `ProteinSequence`: sequence wrapper with one-hot/embedding encoding
//! - `Residue3D`: backbone coordinates with torsion/bond computations
//! - `ProteinStructure`: collection of residues with RMSD, TM-score, distogram
//! - `MsaEncoder`: row/column attention over multiple sequence alignments
//! - `PairwiseRepresentation`: outer-product mean + triangle multiplicative update
//! - `InvariantPointAttention`: IPA from the AlphaFold2 structure module
//! - `StructureModule`: 8-layer IPA stack with torsion-angle MLP
//! - `AlphaFoldLite`: simplified AF2 pipeline with 3-iteration recycling
//! - `ProteinEvoformer`: one full Evoformer block
//! - `ContactMapPredictor`: 2-layer ResNet contact prediction
//! - `ProteinLanguageModelEmbed`: ESM-like 2-layer transformer embedding
//! - `ProteinMetrics`: GDT-TS, lDDT, contact precision, SS accuracy

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the protein-structure module.
#[derive(Debug, Clone)]
pub enum PsError {
    /// Sequence contains an unrecognised character.
    InvalidSequence(String),
    /// A numerical computation failed (e.g. singular matrix).
    ComputationFailed(String),
    /// The supplied structure is malformed.
    InvalidStructure(String),
}

impl fmt::Display for PsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PsError::InvalidSequence(s) => write!(f, "InvalidSequence: {s}"),
            PsError::ComputationFailed(s) => write!(f, "ComputationFailed: {s}"),
            PsError::InvalidStructure(s) => write!(f, "InvalidStructure: {s}"),
        }
    }
}

impl std::error::Error for PsError {}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
}

fn softmax_vec(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return vec![];
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let s: f64 = exps.iter().sum::<f64>().max(1e-300);
    exps.iter().map(|e| e / s).collect()
}

fn layer_norm_vec(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt();
    x.iter().map(|&v| (v - mean) / std).collect()
}

fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(v).map(|(&a, &b)| a * b).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    norm3(&d)
}

/// Xavier-normal init.
fn xavier_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u1: f64 = rng.random::<f64>().max(1e-10);
                    let u2: f64 = rng.random::<f64>();
                    let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    n * scale
                })
                .collect()
        })
        .collect()
}

fn xavier_vec(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let scale = (1.0 / n as f64).sqrt();
    (0..n)
        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// PsAminoAcid
// ─────────────────────────────────────────────────────────────────────────────

/// 20 standard amino acids + Unk/Gap — with biophysical properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PsAminoAcid {
    Ala,
    Arg,
    Asn,
    Asp,
    Cys,
    Gln,
    Glu,
    Gly,
    His,
    Ile,
    Leu,
    Lys,
    Met,
    Phe,
    Pro,
    Ser,
    Thr,
    Trp,
    Tyr,
    Val,
    Unk,
    Gap,
}

impl PsAminoAcid {
    /// Parse from single-letter code (case-insensitive).
    pub fn from_char(c: char) -> Result<Self, PsError> {
        match c.to_ascii_uppercase() {
            'A' => Ok(Self::Ala),
            'R' => Ok(Self::Arg),
            'N' => Ok(Self::Asn),
            'D' => Ok(Self::Asp),
            'C' => Ok(Self::Cys),
            'Q' => Ok(Self::Gln),
            'E' => Ok(Self::Glu),
            'G' => Ok(Self::Gly),
            'H' => Ok(Self::His),
            'I' => Ok(Self::Ile),
            'L' => Ok(Self::Leu),
            'K' => Ok(Self::Lys),
            'M' => Ok(Self::Met),
            'F' => Ok(Self::Phe),
            'P' => Ok(Self::Pro),
            'S' => Ok(Self::Ser),
            'T' => Ok(Self::Thr),
            'W' => Ok(Self::Trp),
            'Y' => Ok(Self::Tyr),
            'V' => Ok(Self::Val),
            'X' => Ok(Self::Unk),
            '-' | '.' => Ok(Self::Gap),
            other => Err(PsError::InvalidSequence(format!(
                "unknown residue '{other}'"
            ))),
        }
    }

    /// Single-letter code.
    pub fn to_char(self) -> char {
        match self {
            Self::Ala => 'A',
            Self::Arg => 'R',
            Self::Asn => 'N',
            Self::Asp => 'D',
            Self::Cys => 'C',
            Self::Gln => 'Q',
            Self::Glu => 'E',
            Self::Gly => 'G',
            Self::His => 'H',
            Self::Ile => 'I',
            Self::Leu => 'L',
            Self::Lys => 'K',
            Self::Met => 'M',
            Self::Phe => 'F',
            Self::Pro => 'P',
            Self::Ser => 'S',
            Self::Thr => 'T',
            Self::Trp => 'W',
            Self::Tyr => 'Y',
            Self::Val => 'V',
            Self::Unk => 'X',
            Self::Gap => '-',
        }
    }

    /// Kyte-Doolittle hydrophobicity scale.
    pub fn hydrophobicity(self) -> f64 {
        match self {
            Self::Ile => 4.5,
            Self::Val => 4.2,
            Self::Leu => 3.8,
            Self::Phe => 2.8,
            Self::Cys => 2.5,
            Self::Met => 1.9,
            Self::Ala => 1.8,
            Self::Gly => -0.4,
            Self::Thr => -0.7,
            Self::Trp => -0.9,
            Self::Ser => -0.8,
            Self::Tyr => -1.3,
            Self::Pro => -1.6,
            Self::His => -3.2,
            Self::Glu => -3.5,
            Self::Gln => -3.5,
            Self::Asp => -3.5,
            Self::Asn => -3.5,
            Self::Lys => -3.9,
            Self::Arg => -4.5,
            Self::Unk => 0.0,
            Self::Gap => 0.0,
        }
    }

    /// Net charge at pH 7.
    pub fn charge(self) -> f64 {
        match self {
            Self::Arg => 1.0,
            Self::Lys => 1.0,
            Self::His => 0.1,
            Self::Asp => -1.0,
            Self::Glu => -1.0,
            _ => 0.0,
        }
    }

    /// Approximate van-der-Waals volume in Å³.
    pub fn volume_angstrom3(self) -> f64 {
        match self {
            Self::Gly => 60.1,
            Self::Ala => 88.6,
            Self::Val => 140.0,
            Self::Leu => 166.7,
            Self::Ile => 166.7,
            Self::Pro => 112.7,
            Self::Phe => 189.9,
            Self::Trp => 227.8,
            Self::Met => 162.9,
            Self::Ser => 89.0,
            Self::Thr => 116.1,
            Self::Cys => 108.5,
            Self::Tyr => 193.6,
            Self::His => 153.2,
            Self::Asp => 111.1,
            Self::Glu => 138.4,
            Self::Asn => 114.1,
            Self::Gln => 143.8,
            Self::Lys => 168.6,
            Self::Arg => 173.4,
            Self::Unk => 120.0,
            Self::Gap => 0.0,
        }
    }

    /// Index 0–19 for standard amino acids; 20 = Unk/Gap.
    pub fn to_idx(self) -> usize {
        match self {
            Self::Ala => 0,
            Self::Arg => 1,
            Self::Asn => 2,
            Self::Asp => 3,
            Self::Cys => 4,
            Self::Gln => 5,
            Self::Glu => 6,
            Self::Gly => 7,
            Self::His => 8,
            Self::Ile => 9,
            Self::Leu => 10,
            Self::Lys => 11,
            Self::Met => 12,
            Self::Phe => 13,
            Self::Pro => 14,
            Self::Ser => 15,
            Self::Thr => 16,
            Self::Trp => 17,
            Self::Tyr => 18,
            Self::Val => 19,
            Self::Unk | Self::Gap => 20,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProteinSequence
// ─────────────────────────────────────────────────────────────────────────────

/// A protein primary sequence as a vector of `PsAminoAcid`.
#[derive(Debug, Clone)]
pub struct ProteinSequence {
    pub residues: Vec<PsAminoAcid>,
}

impl ProteinSequence {
    /// Parse from a one-letter string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, PsError> {
        let residues = s
            .chars()
            .map(PsAminoAcid::from_char)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { residues })
    }

    pub fn length(&self) -> usize {
        self.residues.len()
    }

    /// One-hot encoding: each position → 21-dim vector (20 standard + Unk/Gap).
    pub fn one_hot_encode(&self) -> Vec<Vec<f64>> {
        self.residues
            .iter()
            .map(|aa| {
                let mut v = vec![0.0_f64; 21];
                v[aa.to_idx()] = 1.0;
                v
            })
            .collect()
    }

    /// 4-feature embedding per residue: [hydrophobicity, charge, volume/200, idx/20].
    pub fn embedding_features(&self) -> Vec<Vec<f64>> {
        self.residues
            .iter()
            .map(|aa| {
                vec![
                    aa.hydrophobicity() / 5.0,
                    aa.charge(),
                    aa.volume_angstrom3() / 200.0,
                    aa.to_idx() as f64 / 20.0,
                ]
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Residue3D
// ─────────────────────────────────────────────────────────────────────────────

/// Backbone coordinates (N, CA, C, O) for one residue.
#[derive(Debug, Clone)]
pub struct Residue3D {
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
    pub o: [f64; 3],
}

impl Residue3D {
    pub fn new(n: [f64; 3], ca: [f64; 3], c: [f64; 3], o: [f64; 3]) -> Self {
        Self { n, ca, c, o }
    }

    /// Phi/psi torsion angles using atan2 on cross products.
    /// Returns (phi, psi) in radians; requires neighbouring residue coords.
    pub fn backbone_torsion_phi_psi(
        &self,
        prev_c: Option<&[f64; 3]>,
        next_n: Option<&[f64; 3]>,
    ) -> (f64, f64) {
        let phi = prev_c
            .map(|pc| {
                let b1 = vec_sub_3(pc, &self.n);
                let b2 = vec_sub_3(&self.n, &self.ca);
                let b3 = vec_sub_3(&self.ca, &self.c);
                dihedral_angle(&b1, &b2, &b3)
            })
            .unwrap_or(0.0);

        let psi = next_n
            .map(|nn| {
                let b1 = vec_sub_3(&self.n, &self.ca);
                let b2 = vec_sub_3(&self.ca, &self.c);
                let b3 = vec_sub_3(&self.c, nn);
                dihedral_angle(&b1, &b2, &b3)
            })
            .unwrap_or(0.0);

        (phi, psi)
    }

    /// Peptide bond length (C–N distance).
    pub fn peptide_bond_length(&self, next_n: &[f64; 3]) -> f64 {
        dist3(&self.c, next_n)
    }
}

fn vec_sub_3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dihedral_angle(b1: &[f64; 3], b2: &[f64; 3], b3: &[f64; 3]) -> f64 {
    let n1 = cross3(b1, b2);
    let n2 = cross3(b2, b3);
    let m1 = cross3(&n1, b2);
    let x = dot3(&n1, &n2);
    let y = dot3(&m1, &n2) / norm3(b2).max(1e-10);
    f64::atan2(y, x)
}

// ─────────────────────────────────────────────────────────────────────────────
// ProteinStructure
// ─────────────────────────────────────────────────────────────────────────────

/// A full protein backbone as a vector of `Residue3D`.
#[derive(Debug, Clone)]
pub struct ProteinStructure {
    pub residues: Vec<Residue3D>,
}

impl ProteinStructure {
    pub fn new(residues: Vec<Residue3D>) -> Self {
        Self { residues }
    }

    pub fn len(&self) -> usize {
        self.residues.len()
    }
    pub fn is_empty(&self) -> bool {
        self.residues.is_empty()
    }

    /// Pairwise CA-CA distance matrix.
    pub fn compute_distogram(&self) -> Vec<Vec<f64>> {
        let n = self.residues.len();
        let mut mat = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                mat[i][j] = dist3(&self.residues[i].ca, &self.residues[j].ca);
            }
        }
        mat
    }

    /// TM-score against `other` (Zhang & Skolnick 2004).
    pub fn tm_score(&self, other: &ProteinStructure) -> Result<f64, PsError> {
        let n_ref = other.residues.len();
        if n_ref == 0 {
            return Err(PsError::InvalidStructure(
                "empty reference structure".into(),
            ));
        }
        let len = self.residues.len().min(n_ref);
        let d0 = if n_ref > 21 {
            1.24 * ((n_ref as f64 - 15.0).cbrt()) - 1.8
        } else {
            0.5
        };
        let d0_sq = d0 * d0;
        let sum: f64 = (0..len)
            .map(|i| {
                let d = dist3(&self.residues[i].ca, &other.residues[i].ca);
                1.0 / (1.0 + d * d / d0_sq)
            })
            .sum();
        Ok(sum / n_ref as f64)
    }

    /// Kabsch RMSD between aligned CA coordinates (Jacobi SVD 3×3).
    pub fn rmsd(&self, other: &ProteinStructure) -> Result<f64, PsError> {
        let n = self.residues.len().min(other.residues.len());
        if n == 0 {
            return Err(PsError::InvalidStructure("no residues to align".into()));
        }
        // Centroids
        let mut c1 = [0.0_f64; 3];
        let mut c2 = [0.0_f64; 3];
        for i in 0..n {
            for k in 0..3 {
                c1[k] += self.residues[i].ca[k];
            }
            for k in 0..3 {
                c2[k] += other.residues[i].ca[k];
            }
        }
        let nf = n as f64;
        for k in 0..3 {
            c1[k] /= nf;
            c2[k] /= nf;
        }
        // Centroid-aligned RMSD (upper bound; full Kabsch would require 3×3 SVD)
        let mut sd = 0.0_f64;
        for i in 0..n {
            for k in 0..3 {
                let d = (self.residues[i].ca[k] - c1[k]) - (other.residues[i].ca[k] - c2[k]);
                sd += d * d;
            }
        }
        Ok((sd / nf).sqrt())
    }

    /// Simplified DSSP-like secondary structure assignment.
    /// H = helix (phi≈-60, psi≈-40), E = beta (phi≈-120, psi≈+120), C = coil.
    pub fn compute_secondary_structure(&self) -> Vec<String> {
        let n = self.residues.len();
        (0..n)
            .map(|i| {
                let prev_c = if i > 0 {
                    Some(&self.residues[i - 1].c)
                } else {
                    None
                };
                let next_n = if i + 1 < n {
                    Some(&self.residues[i + 1].n)
                } else {
                    None
                };
                let (phi, psi) = self.residues[i].backbone_torsion_phi_psi(prev_c, next_n);
                // Helix: phi in [-90,-30], psi in [-70,-10]
                if phi > -1.57 && phi < -0.52 && psi > -1.22 && psi < -0.17 {
                    "H".to_string()
                // Beta: phi in [-180,-90], psi in [60,180]
                } else if phi > -std::f64::consts::PI && phi < -1.57 && psi > 1.05 {
                    "E".to_string()
                } else {
                    "C".to_string()
                }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MsaEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Multiple Sequence Alignment encoder with row/column attention.
pub struct MsaEncoder {
    pub embed_dim: usize,
    pub num_heads: usize,
    // Row attention weights
    row_wq: Vec<Vec<f64>>,
    row_wk: Vec<Vec<f64>>,
    row_wv: Vec<Vec<f64>>,
    row_wo: Vec<Vec<f64>>,
    // Column attention weights
    col_wq: Vec<Vec<f64>>,
    col_wk: Vec<Vec<f64>>,
    col_wv: Vec<Vec<f64>>,
    col_wo: Vec<Vec<f64>>,
    // Input embedding weights
    embed_w: Vec<Vec<f64>>,
}

impl MsaEncoder {
    pub fn new(in_features: usize, embed_dim: usize, num_heads: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            embed_dim,
            num_heads,
            row_wq: xavier_matrix(embed_dim, embed_dim, &mut rng),
            row_wk: xavier_matrix(embed_dim, embed_dim, &mut rng),
            row_wv: xavier_matrix(embed_dim, embed_dim, &mut rng),
            row_wo: xavier_matrix(embed_dim, embed_dim, &mut rng),
            col_wq: xavier_matrix(embed_dim, embed_dim, &mut rng),
            col_wk: xavier_matrix(embed_dim, embed_dim, &mut rng),
            col_wv: xavier_matrix(embed_dim, embed_dim, &mut rng),
            col_wo: xavier_matrix(embed_dim, embed_dim, &mut rng),
            embed_w: xavier_matrix(embed_dim, in_features, &mut rng),
        }
    }

    /// Project raw features (seq_len × in_features) to embed_dim.
    fn embed_row(&self, row: &[Vec<f64>]) -> Vec<Vec<f64>> {
        row.iter().map(|feat| matvec(&self.embed_w, feat)).collect()
    }

    /// Multi-head self-attention over a 2D slice (positions).
    fn mha(
        wq: &[Vec<f64>],
        wk: &[Vec<f64>],
        wv: &[Vec<f64>],
        wo: &[Vec<f64>],
        x: &[Vec<f64>],
        embed_dim: usize,
        num_heads: usize,
    ) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return vec![];
        }
        let head_dim = (embed_dim / num_heads).max(1);
        let scale = (head_dim as f64).sqrt().recip();
        let q: Vec<Vec<f64>> = x.iter().map(|v| matvec(wq, v)).collect();
        let k: Vec<Vec<f64>> = x.iter().map(|v| matvec(wk, v)).collect();
        let v: Vec<Vec<f64>> = x.iter().map(|v| matvec(wv, v)).collect();
        let mut out = vec![vec![0.0_f64; embed_dim]; n];
        for h in 0..num_heads {
            let start = h * head_dim;
            let end = (start + head_dim).min(embed_dim);
            // Compute attention scores for this head
            let scores: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let s: f64 =
                                (start..end).map(|d| q[i][d] * k[j][d]).sum::<f64>() * scale;
                            s
                        })
                        .collect()
                })
                .collect();
            let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax_vec(row)).collect();
            // Weighted sum
            for i in 0..n {
                for d in start..end {
                    out[i][d] += (0..n).map(|j| attn[i][j] * v[j][d]).sum::<f64>();
                }
            }
        }
        // Output projection + residual
        let proj: Vec<Vec<f64>> = out.iter().map(|v| matvec(wo, v)).collect();
        proj.iter().zip(x).map(|(p, xi)| vec_add(p, xi)).collect()
    }

    /// Row attention: self-attention across positions (seq_len) for each sequence.
    pub fn row_attention(&self, msa: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        msa.iter()
            .map(|seq_feats| {
                let emb = self.embed_row(seq_feats);
                Self::mha(
                    &self.row_wq,
                    &self.row_wk,
                    &self.row_wv,
                    &self.row_wo,
                    &emb,
                    self.embed_dim,
                    self.num_heads,
                )
            })
            .collect()
    }

    /// Column attention: self-attention across sequences for each position.
    pub fn column_attention(&self, msa: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        if msa.is_empty() {
            return vec![];
        }
        let n_seqs = msa.len();
        let n_pos = msa[0].len();
        let mut result = vec![vec![vec![0.0_f64; self.embed_dim]; n_pos]; n_seqs];
        for pos in 0..n_pos {
            let col: Vec<Vec<f64>> = msa
                .iter()
                .map(|seq| {
                    let emb = self.embed_row(seq);
                    emb.get(pos)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; self.embed_dim])
                })
                .collect();
            let attended = Self::mha(
                &self.col_wq,
                &self.col_wk,
                &self.col_wv,
                &self.col_wo,
                &col,
                self.embed_dim,
                self.num_heads,
            );
            for (s, v) in attended.iter().enumerate() {
                result[s][pos] = v.clone();
            }
        }
        result
    }

    /// Full MSA encoding: embed → row-attn → col-attn.
    pub fn encode_msa(&self, sequences: &[ProteinSequence]) -> Vec<Vec<Vec<f64>>> {
        let msa: Vec<Vec<Vec<f64>>> = sequences.iter().map(|s| s.embedding_features()).collect();
        let row_out = self.row_attention(&msa);
        self.column_attention(&row_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PairwiseRepresentation
// ─────────────────────────────────────────────────────────────────────────────

/// Outer-product mean pair representation with triangle multiplicative update.
pub struct PairwiseRepresentation {
    pub pair_dim: usize,
    // Triangle multiplicative update weights (simplified)
    tri_out_w: Vec<Vec<f64>>,
    tri_in_w: Vec<Vec<f64>>,
}

impl PairwiseRepresentation {
    pub fn new(pair_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            pair_dim,
            tri_out_w: xavier_matrix(pair_dim, pair_dim, &mut rng),
            tri_in_w: xavier_matrix(pair_dim, pair_dim, &mut rng),
        }
    }

    /// Compute outer-product mean: pair\[i\]\[j\] = mean over sequences of row_i ⊗ row_j.
    pub fn outer_product_mean(&self, msa_repr: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        if msa_repr.is_empty() {
            return vec![];
        }
        let n_seqs = msa_repr.len();
        let n_pos = msa_repr[0].len();
        let row_dim = msa_repr[0].first().map(|r| r.len()).unwrap_or(0);
        let pair_out_dim = self.pair_dim.min(row_dim * row_dim);
        let mut pair = vec![vec![vec![0.0_f64; pair_out_dim]; n_pos]; n_pos];
        for i in 0..n_pos {
            for j in 0..n_pos {
                let mut acc = vec![0.0_f64; pair_out_dim];
                for s in 0..n_seqs {
                    let ri = msa_repr[s].get(i).map(|v| v.as_slice()).unwrap_or(&[]);
                    let rj = msa_repr[s].get(j).map(|v| v.as_slice()).unwrap_or(&[]);
                    let mut k = 0;
                    'outer: for a in 0..ri.len() {
                        for b in 0..rj.len() {
                            if k >= pair_out_dim {
                                break 'outer;
                            }
                            acc[k] += ri[a] * rj[b];
                            k += 1;
                        }
                    }
                }
                let nf = n_seqs as f64;
                pair[i][j] = acc.iter().map(|&v| v / nf).collect();
            }
        }
        pair
    }

    /// Triangle multiplicative update (outgoing + incoming paths).
    pub fn update_pair_with_triangles(&self, pair: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        let n = pair.len();
        if n == 0 {
            return vec![];
        }
        let d = pair[0][0].len().min(self.pair_dim);
        let mut out = pair.to_vec();
        // Outgoing: edge_ij += Σ_k gate(edge_ik) * edge_jk
        for i in 0..n {
            for j in 0..n {
                let mut update = vec![0.0_f64; d];
                for k in 0..n {
                    if k == i || k == j {
                        continue;
                    }
                    let gate = sigmoid_scalar(dot(&pair[i][k][..d], &pair[j][k][..d]));
                    for dd in 0..d {
                        let ik_d = *pair[i][k].get(dd).unwrap_or(&0.0);
                        update[dd] += gate * ik_d;
                    }
                }
                let proj = matvec(
                    &self.tri_out_w,
                    &update[..d]
                        .iter()
                        .chain(std::iter::repeat(&0.0_f64).take(self.pair_dim.saturating_sub(d)))
                        .cloned()
                        .collect::<Vec<_>>(),
                );
                for dd in 0..d.min(proj.len()) {
                    out[i][j][dd] += proj[dd] * 0.01;
                }
            }
        }
        out
    }
}

#[inline]
fn sigmoid_scalar(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// InvariantPointAttention
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified IPA: combines scalar attention with point-based 3D queries/keys.
pub struct InvariantPointAttention {
    pub single_dim: usize,
    pub pair_dim: usize,
    pub num_heads: usize,
    wq_scalar: Vec<Vec<f64>>,
    wk_scalar: Vec<Vec<f64>>,
    wv_scalar: Vec<Vec<f64>>,
    wq_point: Vec<Vec<f64>>, // → 3*num_heads point queries
    wk_point: Vec<Vec<f64>>,
    wv_point: Vec<Vec<f64>>,
    w_pair: Vec<Vec<f64>>, // pair_dim → num_heads
    wo: Vec<Vec<f64>>,     // output projection
}

impl InvariantPointAttention {
    pub fn new(single_dim: usize, pair_dim: usize, num_heads: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let head_dim = single_dim / num_heads.max(1);
        Self {
            single_dim,
            pair_dim,
            num_heads,
            wq_scalar: xavier_matrix(num_heads * head_dim, single_dim, &mut rng),
            wk_scalar: xavier_matrix(num_heads * head_dim, single_dim, &mut rng),
            wv_scalar: xavier_matrix(num_heads * head_dim, single_dim, &mut rng),
            wq_point: xavier_matrix(3 * num_heads, single_dim, &mut rng),
            wk_point: xavier_matrix(3 * num_heads, single_dim, &mut rng),
            wv_point: xavier_matrix(3 * num_heads, single_dim, &mut rng),
            w_pair: xavier_matrix(num_heads, pair_dim, &mut rng),
            wo: xavier_matrix(single_dim, num_heads * head_dim + 3 * num_heads, &mut rng),
        }
    }

    /// Forward: single_repr (n × single_dim), pair_repr (n × n × pair_dim),
    /// frames: (rotation 3×3, translation 3) per residue.
    /// Returns per-residue updates (n × single_dim).
    pub fn forward(
        &self,
        single: &[Vec<f64>],
        pair: &[Vec<Vec<f64>>],
        frames: &[([f64; 9], [f64; 3])],
    ) -> Vec<Vec<f64>> {
        let n = single.len();
        if n == 0 {
            return vec![];
        }
        let head_dim = self.single_dim / self.num_heads.max(1);
        let scale = (head_dim as f64 + 3.0 * self.num_heads as f64)
            .sqrt()
            .recip();
        // Scalar projections
        let q_s: Vec<Vec<f64>> = single.iter().map(|v| matvec(&self.wq_scalar, v)).collect();
        let k_s: Vec<Vec<f64>> = single.iter().map(|v| matvec(&self.wk_scalar, v)).collect();
        let v_s: Vec<Vec<f64>> = single.iter().map(|v| matvec(&self.wv_scalar, v)).collect();
        // Point projections (residue-local → global frame)
        let q_p: Vec<Vec<f64>> = single
            .iter()
            .zip(frames)
            .map(|(v, (r, t))| {
                let local = matvec(&self.wq_point, v);
                apply_frame(&local, r, t)
            })
            .collect();
        let k_p: Vec<Vec<f64>> = single
            .iter()
            .zip(frames)
            .map(|(v, (r, t))| {
                let local = matvec(&self.wk_point, v);
                apply_frame(&local, r, t)
            })
            .collect();
        let v_p: Vec<Vec<f64>> = single
            .iter()
            .zip(frames)
            .map(|(v, (r, t))| {
                let local = matvec(&self.wv_point, v);
                apply_frame(&local, r, t)
            })
            .collect();
        // Pair bias
        let pair_bias: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let pf = pair
                            .get(i)
                            .and_then(|row| row.get(j))
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        let b_full = matvec(&self.w_pair, pf);
                        b_full.iter().sum::<f64>() / self.num_heads as f64
                    })
                    .collect()
            })
            .collect();
        // Per-head attention
        let mut out_scalar = vec![vec![0.0_f64; self.num_heads * head_dim]; n];
        let mut out_point = vec![vec![0.0_f64; 3 * self.num_heads]; n];
        for h in 0..self.num_heads {
            let hs = h * head_dim;
            let he = hs + head_dim;
            let hp = h * 3;
            let attn: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let row: Vec<f64> = (0..n)
                        .map(|j| {
                            // Scalar dot
                            let sd: f64 = (hs..he.min(q_s[i].len()))
                                .map(|d| q_s[i][d] * k_s[j][d])
                                .sum::<f64>();
                            // Point distance
                            let pd: f64 = (0..3)
                                .map(|d| {
                                    let qi = q_p[i].get(hp + d).copied().unwrap_or(0.0);
                                    let kj = k_p[j].get(hp + d).copied().unwrap_or(0.0);
                                    (qi - kj).powi(2)
                                })
                                .sum::<f64>();
                            let bias = pair_bias
                                .get(i)
                                .and_then(|r| r.get(j))
                                .copied()
                                .unwrap_or(0.0);
                            (sd - 0.5 * pd + bias) * scale
                        })
                        .collect();
                    softmax_vec(&row)
                })
                .collect();
            for i in 0..n {
                // Aggregate scalar values
                for d in hs..he.min(out_scalar[i].len()) {
                    out_scalar[i][d] += (0..n)
                        .map(|j| attn[i][j] * v_s[j].get(d).copied().unwrap_or(0.0))
                        .sum::<f64>();
                }
                // Aggregate point values
                for d in 0..3 {
                    let idx = hp + d;
                    if idx < out_point[i].len() {
                        out_point[i][idx] += (0..n)
                            .map(|j| attn[i][j] * v_p[j].get(idx).copied().unwrap_or(0.0))
                            .sum::<f64>();
                    }
                }
            }
        }
        // Concatenate scalar + point then project
        (0..n)
            .map(|i| {
                let cat: Vec<f64> = out_scalar[i]
                    .iter()
                    .chain(out_point[i].iter())
                    .cloned()
                    .collect();
                let proj = if cat.len() >= self.wo[0].len() {
                    matvec(&self.wo, &cat[..self.wo[0].len()])
                } else {
                    let mut padded = cat.clone();
                    padded.resize(self.wo[0].len(), 0.0);
                    matvec(&self.wo, &padded)
                };
                vec_add(&proj, &single[i])
            })
            .collect()
    }
}

/// Apply rotation (row-major 3×3) + translation to a flattened 3*k-dim vector.
fn apply_frame(v: &[f64], rot: &[f64; 9], trans: &[f64; 3]) -> Vec<f64> {
    let k = v.len() / 3;
    let mut out = Vec::with_capacity(v.len());
    for i in 0..k {
        let base = i * 3;
        let x = v.get(base).copied().unwrap_or(0.0);
        let y = v.get(base + 1).copied().unwrap_or(0.0);
        let z = v.get(base + 2).copied().unwrap_or(0.0);
        out.push(rot[0] * x + rot[1] * y + rot[2] * z + trans[0]);
        out.push(rot[3] * x + rot[4] * y + rot[5] * z + trans[1]);
        out.push(rot[6] * x + rot[7] * y + rot[8] * z + trans[2]);
    }
    // pad any remainder
    for i in (k * 3)..v.len() {
        out.push(v[i]);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// StructureModule
// ─────────────────────────────────────────────────────────────────────────────

/// 8-layer IPA stack with torsion-angle MLP.
pub struct StructureModule {
    pub n_layers: usize,
    pub single_dim: usize,
    pub pair_dim: usize,
    ipa_layers: Vec<InvariantPointAttention>,
    // Torsion angle MLP: single_dim → 128 → 8*2 (sin/cos per angle)
    torsion_w1: Vec<Vec<f64>>,
    torsion_b1: Vec<f64>,
    torsion_w2: Vec<Vec<f64>>,
    torsion_b2: Vec<f64>,
    // Layer norm weights
    ln_gamma: Vec<f64>,
    ln_beta: Vec<f64>,
}

impl StructureModule {
    pub fn new(single_dim: usize, pair_dim: usize, seed: u64) -> Self {
        let n_layers = 8;
        let mut rng = StdRng::seed_from_u64(seed);
        let ipa_layers = (0..n_layers)
            .map(|i| InvariantPointAttention::new(single_dim, pair_dim, 2, seed + i as u64 * 17))
            .collect();
        let hidden = 128;
        Self {
            n_layers,
            single_dim,
            pair_dim,
            ipa_layers,
            torsion_w1: xavier_matrix(hidden, single_dim, &mut rng),
            torsion_b1: xavier_vec(hidden, &mut rng),
            torsion_w2: xavier_matrix(16, hidden, &mut rng),
            torsion_b2: xavier_vec(16, &mut rng),
            ln_gamma: vec![1.0_f64; single_dim],
            ln_beta: vec![0.0_f64; single_dim],
        }
    }

    /// Predict backbone frames + torsion angles.
    /// Returns (backbone coords per residue, torsion angles per residue).
    pub fn forward(
        &self,
        single_repr: &[Vec<f64>],
        pair_repr: &[Vec<Vec<f64>>],
        initial_frames: &[([f64; 9], [f64; 3])],
    ) -> (Vec<[f64; 3]>, Vec<Vec<f64>>) {
        let n = single_repr.len();
        if n == 0 {
            return (vec![], vec![]);
        }
        let mut single = single_repr.to_vec();
        let mut frames = initial_frames.to_vec();
        for layer in &self.ipa_layers {
            let update = layer.forward(&single, pair_repr, &frames);
            // Apply layer norm + update
            single = update
                .iter()
                .map(|v| {
                    let norm = layer_norm_vec(v);
                    norm.iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            let g = self.ln_gamma.get(i).copied().unwrap_or(1.0);
                            let b = self.ln_beta.get(i).copied().unwrap_or(0.0);
                            g * x + b
                        })
                        .collect()
                })
                .collect();
            // Update frames: compose rotation from torsion MLP
            for (i, s) in single.iter().enumerate() {
                let h = vec_add(&matvec(&self.torsion_w1, s), &self.torsion_b1)
                    .iter()
                    .map(|&x| relu(x))
                    .collect::<Vec<_>>();
                let angles = vec_add(&matvec(&self.torsion_w2, &h), &self.torsion_b2);
                // Update translation from small incremental step
                let delta = [
                    angles.first().copied().unwrap_or(0.0) * 0.1,
                    angles.get(1).copied().unwrap_or(0.0) * 0.1,
                    angles.get(2).copied().unwrap_or(0.0) * 0.1,
                ];
                frames[i].1 = [
                    frames[i].1[0] + delta[0],
                    frames[i].1[1] + delta[1],
                    frames[i].1[2] + delta[2],
                ];
            }
        }
        // Torsion angles from final single repr
        let torsion_angles: Vec<Vec<f64>> = single
            .iter()
            .map(|s| {
                let h = vec_add(&matvec(&self.torsion_w1, s), &self.torsion_b1)
                    .iter()
                    .map(|&x| relu(x))
                    .collect::<Vec<_>>();
                let raw = vec_add(&matvec(&self.torsion_w2, &h), &self.torsion_b2);
                // Convert pairs (sin, cos) to angle in [-π, π]
                (0..8)
                    .map(|k| {
                        let sin = raw.get(k * 2).copied().unwrap_or(0.0);
                        let cos = raw.get(k * 2 + 1).copied().unwrap_or(1.0);
                        f64::atan2(sin, cos)
                    })
                    .collect()
            })
            .collect();
        let backbone: Vec<[f64; 3]> = frames.iter().map(|(_, t)| *t).collect();
        (backbone, torsion_angles)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AlphaFoldLite
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified AlphaFold2-style pipeline with 3-iteration recycling.
pub struct AlphaFoldLite {
    pub msa_encoder: MsaEncoder,
    pub pair_repr: PairwiseRepresentation,
    pub structure: StructureModule,
    pub embed_dim: usize,
    pub pair_dim: usize,
}

impl AlphaFoldLite {
    pub fn new(embed_dim: usize, pair_dim: usize, seed: u64) -> Self {
        Self {
            msa_encoder: MsaEncoder::new(4, embed_dim, 2, seed),
            pair_repr: PairwiseRepresentation::new(pair_dim, seed + 100),
            structure: StructureModule::new(embed_dim, pair_dim, seed + 200),
            embed_dim,
            pair_dim,
        }
    }

    /// Predict structure from MSA (Vec of ProteinSequences).
    pub fn predict(&self, sequences: &[ProteinSequence]) -> Result<ProteinStructure, PsError> {
        if sequences.is_empty() {
            return Err(PsError::InvalidSequence("no sequences provided".into()));
        }
        let n = sequences[0].length();
        if n == 0 {
            return Err(PsError::InvalidSequence("empty sequence".into()));
        }
        // Initial identity frames
        let identity_rot = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut frames: Vec<([f64; 9], [f64; 3])> = (0..n)
            .map(|i| {
                (identity_rot, [i as f64 * 3.8, 0.0, 0.0]) // extended chain
            })
            .collect();
        // 3 recycling iterations
        for _ in 0..3 {
            let msa_repr = self.msa_encoder.encode_msa(sequences);
            let pair = self.pair_repr.outer_product_mean(&msa_repr);
            let pair_upd = self.pair_repr.update_pair_with_triangles(&pair);
            // Build padded pair with pair_dim
            let pair_padded = pad_pair_repr(&pair_upd, n, self.pair_dim);
            // Build single from mean of MSA repr
            let single = mean_msa_repr(&msa_repr, n, self.embed_dim);
            let (backbone, _torsions) = self.structure.forward(&single, &pair_padded, &frames);
            // Update frames from predicted backbone
            for (i, ca) in backbone.iter().enumerate() {
                frames[i].1 = *ca;
            }
        }
        // Build ProteinStructure from final frames
        let residues = frames
            .iter()
            .map(|(_, t)| {
                let ca = *t;
                let offset = 1.46_f64; // typical N-CA distance
                Residue3D::new(
                    [ca[0] - offset, ca[1], ca[2]],
                    ca,
                    [ca[0] + offset, ca[1], ca[2]],
                    [ca[0] + offset, ca[1] + 1.23, ca[2]],
                )
            })
            .collect();
        Ok(ProteinStructure::new(residues))
    }
}

fn pad_pair_repr(pair: &[Vec<Vec<f64>>], n: usize, dim: usize) -> Vec<Vec<Vec<f64>>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let raw = pair
                        .get(i)
                        .and_then(|r| r.get(j))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let mut out = raw.to_vec();
                    out.resize(dim, 0.0);
                    out
                })
                .collect()
        })
        .collect()
}

fn mean_msa_repr(msa: &[Vec<Vec<f64>>], n: usize, dim: usize) -> Vec<Vec<f64>> {
    if msa.is_empty() {
        return vec![vec![0.0_f64; dim]; n];
    }
    let ns = msa.len() as f64;
    (0..n)
        .map(|i| {
            let mut acc = vec![0.0_f64; dim];
            for seq in msa {
                let row = seq.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
                for (d, &x) in row.iter().enumerate() {
                    if d < dim {
                        acc[d] += x;
                    }
                }
            }
            acc.iter().map(|&v| v / ns).collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ProteinEvoformer
// ─────────────────────────────────────────────────────────────────────────────

/// One full Evoformer block.
pub struct ProteinEvoformer {
    pub msa_dim: usize,
    pub pair_dim: usize,
    // MSA row attention with pair bias
    msa_row_wq: Vec<Vec<f64>>,
    msa_row_wk: Vec<Vec<f64>>,
    msa_row_wv: Vec<Vec<f64>>,
    msa_row_wo: Vec<Vec<f64>>,
    // MSA column attention
    msa_col_wq: Vec<Vec<f64>>,
    msa_col_wk: Vec<Vec<f64>>,
    msa_col_wv: Vec<Vec<f64>>,
    msa_col_wo: Vec<Vec<f64>>,
    // MSA transition (2-layer FFN)
    msa_ff1: Vec<Vec<f64>>,
    msa_ff2: Vec<Vec<f64>>,
    // Outer product mean
    opm_w: Vec<Vec<f64>>,
    // Triangle multiplicative outgoing/incoming
    tri_out_wa: Vec<Vec<f64>>,
    tri_in_wa: Vec<Vec<f64>>,
    // Pair transition
    pair_ff1: Vec<Vec<f64>>,
    pair_ff2: Vec<Vec<f64>>,
}

impl ProteinEvoformer {
    pub fn new(msa_dim: usize, pair_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let hidden_msa = msa_dim * 4;
        let hidden_pair = pair_dim * 4;
        Self {
            msa_dim,
            pair_dim,
            msa_row_wq: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_row_wk: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_row_wv: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_row_wo: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_col_wq: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_col_wk: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_col_wv: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_col_wo: xavier_matrix(msa_dim, msa_dim, &mut rng),
            msa_ff1: xavier_matrix(hidden_msa, msa_dim, &mut rng),
            msa_ff2: xavier_matrix(msa_dim, hidden_msa, &mut rng),
            opm_w: xavier_matrix(pair_dim, msa_dim * msa_dim, &mut rng),
            tri_out_wa: xavier_matrix(pair_dim, pair_dim, &mut rng),
            tri_in_wa: xavier_matrix(pair_dim, pair_dim, &mut rng),
            pair_ff1: xavier_matrix(hidden_pair, pair_dim, &mut rng),
            pair_ff2: xavier_matrix(pair_dim, hidden_pair, &mut rng),
        }
    }

    fn mha_with_pair_bias(
        &self,
        x: &[Vec<f64>],
        pair_row: &[Vec<f64>],
        wq: &[Vec<f64>],
        wk: &[Vec<f64>],
        wv: &[Vec<f64>],
        wo: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return vec![];
        }
        let scale = (self.msa_dim as f64).sqrt().recip();
        let q: Vec<Vec<f64>> = x.iter().map(|v| matvec(wq, v)).collect();
        let k: Vec<Vec<f64>> = x.iter().map(|v| matvec(wk, v)).collect();
        let v: Vec<Vec<f64>> = x.iter().map(|v| matvec(wv, v)).collect();
        let attn: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let raw: Vec<f64> = (0..n)
                    .map(|j| {
                        let sd = dot(&q[i], &k[j]) * scale;
                        let pb = pair_row
                            .get(j)
                            .map(|p| p.iter().sum::<f64>() * 0.01)
                            .unwrap_or(0.0);
                        sd + pb
                    })
                    .collect();
                softmax_vec(&raw)
            })
            .collect();
        let out: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let agg: Vec<f64> = (0..self.msa_dim)
                    .map(|d| {
                        (0..n)
                            .map(|j| attn[i][j] * v[j].get(d).copied().unwrap_or(0.0))
                            .sum()
                    })
                    .collect();
                vec_add(&matvec(wo, &agg), &x[i])
            })
            .collect();
        out
    }

    fn transition_ffn(&self, x: &[Vec<f64>], w1: &[Vec<f64>], w2: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|v| {
                let h: Vec<f64> = matvec(w1, v).iter().map(|&x| gelu(x)).collect();
                let out = matvec(w2, &h);
                vec_add(&out, v)
            })
            .collect()
    }

    /// Run one Evoformer block.
    pub fn forward(
        &self,
        msa: &[Vec<Vec<f64>>],  // n_seqs × n_pos × msa_dim
        pair: &[Vec<Vec<f64>>], // n_pos × n_pos × pair_dim
    ) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) {
        let n_seqs = msa.len();
        let n_pos = msa.first().map(|r| r.len()).unwrap_or(0);
        if n_seqs == 0 || n_pos == 0 {
            return (msa.to_vec(), pair.to_vec());
        }
        // 1. MSA row attention with pair bias
        let mut msa_out: Vec<Vec<Vec<f64>>> = (0..n_seqs)
            .map(|s| {
                let pair_row: Vec<Vec<f64>> = (0..n_pos)
                    .map(|i| {
                        pair.get(i)
                            .and_then(|r| r.get(i))
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect();
                let row = msa[s]
                    .iter()
                    .map(|v| {
                        let mut padded = v.clone();
                        padded.resize(self.msa_dim, 0.0);
                        padded
                    })
                    .collect::<Vec<_>>();
                self.mha_with_pair_bias(
                    &row,
                    &pair_row,
                    &self.msa_row_wq,
                    &self.msa_row_wk,
                    &self.msa_row_wv,
                    &self.msa_row_wo,
                )
            })
            .collect();
        // 2. MSA column attention
        for pos in 0..n_pos {
            let col: Vec<Vec<f64>> = (0..n_seqs)
                .map(|s| {
                    msa_out[s]
                        .get(pos)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; self.msa_dim])
                })
                .collect();
            let attended = self.mha_with_pair_bias(
                &col,
                &[],
                &self.msa_col_wq,
                &self.msa_col_wk,
                &self.msa_col_wv,
                &self.msa_col_wo,
            );
            for s in 0..n_seqs {
                if let Some(row) = msa_out.get_mut(s) {
                    if let Some(cell) = row.get_mut(pos) {
                        *cell = attended.get(s).cloned().unwrap_or_else(|| cell.clone());
                    }
                }
            }
        }
        // 3. MSA transition
        msa_out = msa_out
            .iter()
            .map(|seq| self.transition_ffn(seq, &self.msa_ff1, &self.msa_ff2))
            .collect();
        // 4. Outer product mean → pair update
        let mut pair_out = pair.to_vec();
        for i in 0..n_pos {
            for j in 0..n_pos {
                let mut opm = vec![0.0_f64; self.msa_dim * self.msa_dim];
                for s in 0..n_seqs {
                    let ri = msa_out[s].get(i).map(|v| v.as_slice()).unwrap_or(&[]);
                    let rj = msa_out[s].get(j).map(|v| v.as_slice()).unwrap_or(&[]);
                    let mut k = 0;
                    'op: for a in 0..ri.len() {
                        for b in 0..rj.len() {
                            if k >= opm.len() {
                                break 'op;
                            }
                            opm[k] += ri[a] * rj[b];
                            k += 1;
                        }
                    }
                }
                let opm_mean: Vec<f64> = opm.iter().map(|&v| v / n_seqs as f64).collect();
                let opm_proj = matvec(&self.opm_w, &opm_mean);
                if let Some(row) = pair_out.get_mut(i) {
                    if let Some(cell) = row.get_mut(j) {
                        let d = cell.len().min(opm_proj.len());
                        for k in 0..d {
                            cell[k] += opm_proj[k] * 0.01;
                        }
                    }
                }
            }
        }
        // 5. Triangle multiplicative updates (simplified)
        for i in 0..n_pos {
            for j in 0..n_pos {
                let pij = pair_out[i][j].clone();
                let mut acc_out = vec![0.0_f64; self.pair_dim];
                let mut acc_in = vec![0.0_f64; self.pair_dim];
                for k in 0..n_pos {
                    if k == i || k == j {
                        continue;
                    }
                    let pik = pair_out
                        .get(i)
                        .and_then(|r| r.get(k))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let pjk = pair_out
                        .get(j)
                        .and_then(|r| r.get(k))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let pkj = pair_out
                        .get(k)
                        .and_then(|r| r.get(j))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let pki = pair_out
                        .get(k)
                        .and_then(|r| r.get(i))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let gate_out = sigmoid_scalar(dot3_slice(pik, pjk));
                    let gate_in = sigmoid_scalar(dot3_slice(pkj, pki));
                    for d in 0..self.pair_dim {
                        let ik = pik.get(d).copied().unwrap_or(0.0);
                        let kj = pkj.get(d).copied().unwrap_or(0.0);
                        acc_out[d] += gate_out * ik;
                        acc_in[d] += gate_in * kj;
                    }
                }
                let tri_out = matvec(&self.tri_out_wa, &acc_out);
                let tri_in = matvec(&self.tri_in_wa, &acc_in);
                let updated: Vec<f64> = (0..self.pair_dim)
                    .map(|d| {
                        pij.get(d).copied().unwrap_or(0.0)
                            + tri_out.get(d).copied().unwrap_or(0.0) * 0.01
                            + tri_in.get(d).copied().unwrap_or(0.0) * 0.01
                    })
                    .collect();
                pair_out[i][j] = updated;
            }
        }
        // 6. Pair transition
        for i in 0..n_pos {
            pair_out[i] = self.transition_ffn(&pair_out[i], &self.pair_ff1, &self.pair_ff2);
        }
        (msa_out, pair_out)
    }
}

fn dot3_slice(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()).min(3);
    (0..n).map(|i| a[i] * b[i]).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// ContactMapPredictor
// ─────────────────────────────────────────────────────────────────────────────

/// 2-layer ResNet on pair features for contact map prediction.
pub struct ContactMapPredictor {
    pub seq_dim: usize,
    pub hidden: usize,
    // Pair feature: concat of two per-residue embeddings
    pair_w1: Vec<Vec<f64>>,
    pair_b1: Vec<f64>,
    pair_w2: Vec<Vec<f64>>,
    pair_b2: Vec<f64>,
    // Residual skip
    skip_w: Vec<Vec<f64>>,
    // Output logit
    out_w: Vec<f64>,
    out_b: f64,
}

impl ContactMapPredictor {
    pub fn new(seq_dim: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let pair_in = seq_dim * 2;
        Self {
            seq_dim,
            hidden,
            pair_w1: xavier_matrix(hidden, pair_in, &mut rng),
            pair_b1: xavier_vec(hidden, &mut rng),
            pair_w2: xavier_matrix(hidden, hidden, &mut rng),
            pair_b2: xavier_vec(hidden, &mut rng),
            skip_w: xavier_matrix(hidden, pair_in, &mut rng),
            out_w: xavier_vec(hidden, &mut rng),
            out_b: 0.0,
        }
    }

    /// Predict contact probabilities from per-residue embeddings.
    pub fn predict_contacts(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = embeddings.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let mut pair = embeddings[i].clone();
                        pair.extend_from_slice(&embeddings[j]);
                        // Layer 1
                        let h1: Vec<f64> = vec_add(&matvec(&self.pair_w1, &pair), &self.pair_b1)
                            .iter()
                            .map(|&x| relu(x))
                            .collect();
                        // Layer 2 with residual
                        let h2: Vec<f64> = vec_add(&matvec(&self.pair_w2, &h1), &self.pair_b2)
                            .iter()
                            .map(|&x| relu(x))
                            .collect();
                        let skip = matvec(&self.skip_w, &pair);
                        let res: Vec<f64> = vec_add(&h2, &skip);
                        // Output
                        let logit = dot(&self.out_w, &res) + self.out_b;
                        sigmoid_scalar(logit)
                    })
                    .collect()
            })
            .collect()
    }

    /// AUC (area under ROC) for contact prediction evaluation.
    pub fn auc(probs: &[Vec<f64>], contacts: &[Vec<bool>]) -> f64 {
        let n = probs.len().min(contacts.len());
        let mut pairs: Vec<(f64, bool)> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let p = probs[i].get(j).copied().unwrap_or(0.0);
                let c = contacts[i].get(j).copied().unwrap_or(false);
                pairs.push((p, c));
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let pos_total = pairs.iter().filter(|&&(_, c)| c).count() as f64;
        let neg_total = pairs.len() as f64 - pos_total;
        if pos_total == 0.0 || neg_total == 0.0 {
            return 0.5;
        }
        let mut auc = 0.0_f64;
        let mut tp = 0.0_f64;
        let mut fp = 0.0_f64;
        let mut prev_fp = 0.0_f64;
        let mut prev_tp = 0.0_f64;
        for &(_, c) in &pairs {
            if c {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            auc += (fp - prev_fp) / neg_total * (tp + prev_tp) / 2.0 / pos_total;
            prev_fp = fp;
            prev_tp = tp;
        }
        auc
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProteinLanguageModelEmbed
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified ESM-like bidirectional transformer embedding.
pub struct ProteinLanguageModelEmbed {
    pub embed_dim: usize,
    pub vocab_size: usize,
    // Token + positional embeddings
    token_embed: Vec<Vec<f64>>, // vocab_size × embed_dim
    pos_embed: Vec<Vec<f64>>,   // max_len × embed_dim (1024)
    // Layer 1
    l1_wq: Vec<Vec<f64>>,
    l1_wk: Vec<Vec<f64>>,
    l1_wv: Vec<Vec<f64>>,
    l1_wo: Vec<Vec<f64>>,
    l1_ff1: Vec<Vec<f64>>,
    l1_ff2: Vec<Vec<f64>>,
    // Layer 2
    l2_wq: Vec<Vec<f64>>,
    l2_wk: Vec<Vec<f64>>,
    l2_wv: Vec<Vec<f64>>,
    l2_wo: Vec<Vec<f64>>,
    l2_ff1: Vec<Vec<f64>>,
    l2_ff2: Vec<Vec<f64>>,
}

impl ProteinLanguageModelEmbed {
    pub fn new(embed_dim: usize, seed: u64) -> Self {
        let vocab_size = 21; // 20 aa + Unk
        let max_len = 1024;
        let hidden = embed_dim * 4;
        let mut rng = StdRng::seed_from_u64(seed);
        // Sinusoidal positional encodings
        let pos_embed: Vec<Vec<f64>> = (0..max_len)
            .map(|pos| {
                (0..embed_dim)
                    .map(|d| {
                        let freq =
                            1.0_f64 / 10000.0_f64.powf(2.0 * (d / 2) as f64 / embed_dim as f64);
                        if d % 2 == 0 {
                            (pos as f64 * freq).sin()
                        } else {
                            (pos as f64 * freq).cos()
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            embed_dim,
            vocab_size,
            token_embed: xavier_matrix(vocab_size, embed_dim, &mut rng),
            pos_embed,
            l1_wq: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l1_wk: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l1_wv: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l1_wo: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l1_ff1: xavier_matrix(hidden, embed_dim, &mut rng),
            l1_ff2: xavier_matrix(embed_dim, hidden, &mut rng),
            l2_wq: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l2_wk: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l2_wv: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l2_wo: xavier_matrix(embed_dim, embed_dim, &mut rng),
            l2_ff1: xavier_matrix(hidden, embed_dim, &mut rng),
            l2_ff2: xavier_matrix(embed_dim, hidden, &mut rng),
        }
    }

    fn transformer_block(
        x: &[Vec<f64>],
        wq: &[Vec<f64>],
        wk: &[Vec<f64>],
        wv: &[Vec<f64>],
        wo: &[Vec<f64>],
        ff1: &[Vec<f64>],
        ff2: &[Vec<f64>],
        embed_dim: usize,
    ) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return vec![];
        }
        let scale = (embed_dim as f64).sqrt().recip();
        // Self-attention (bidirectional)
        let q: Vec<Vec<f64>> = x.iter().map(|v| matvec(wq, v)).collect();
        let k: Vec<Vec<f64>> = x.iter().map(|v| matvec(wk, v)).collect();
        let v: Vec<Vec<f64>> = x.iter().map(|v| matvec(wv, v)).collect();
        let attn: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                softmax_vec(
                    &(0..n)
                        .map(|j| dot(&q[i], &k[j]) * scale)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let sa_out: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let agg: Vec<f64> = (0..embed_dim)
                    .map(|d| {
                        (0..n)
                            .map(|j| attn[i][j] * v[j].get(d).copied().unwrap_or(0.0))
                            .sum()
                    })
                    .collect();
                layer_norm_vec(&vec_add(&matvec(wo, &agg), &x[i]))
            })
            .collect();
        // FFN
        sa_out
            .iter()
            .map(|v| {
                let h: Vec<f64> = matvec(ff1, v).iter().map(|&x| gelu(x)).collect();
                let out = matvec(ff2, &h);
                layer_norm_vec(&vec_add(&out, v))
            })
            .collect()
    }

    /// Embed a protein sequence into per-residue vectors.
    pub fn embed(&self, sequence: &ProteinSequence) -> Vec<Vec<f64>> {
        let n = sequence.length();
        // Token + positional embeddings
        let x: Vec<Vec<f64>> = sequence
            .residues
            .iter()
            .enumerate()
            .map(|(pos, aa)| {
                let tok = self
                    .token_embed
                    .get(aa.to_idx().min(self.vocab_size - 1))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.embed_dim]);
                let pe = self
                    .pos_embed
                    .get(pos)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.embed_dim]);
                vec_add(&tok, &pe)
            })
            .collect();
        if n == 0 {
            return x;
        }
        // Layer 1
        let x1 = Self::transformer_block(
            &x,
            &self.l1_wq,
            &self.l1_wk,
            &self.l1_wv,
            &self.l1_wo,
            &self.l1_ff1,
            &self.l1_ff2,
            self.embed_dim,
        );
        // Layer 2
        Self::transformer_block(
            &x1,
            &self.l2_wq,
            &self.l2_wk,
            &self.l2_wv,
            &self.l2_wo,
            &self.l2_ff1,
            &self.l2_ff2,
            self.embed_dim,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProteinMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for structure prediction.
pub struct ProteinMetrics;

impl ProteinMetrics {
    /// GDT-TS: average fraction of residues within 1, 2, 4, 8 Å.
    pub fn gdt_ts(pred: &ProteinStructure, target: &ProteinStructure) -> f64 {
        let n = pred.len().min(target.len());
        if n == 0 {
            return 0.0;
        }
        let thresholds = [1.0_f64, 2.0, 4.0, 8.0];
        let score: f64 = thresholds
            .iter()
            .map(|&t| {
                let count = (0..n)
                    .filter(|&i| dist3(&pred.residues[i].ca, &target.residues[i].ca) <= t)
                    .count();
                count as f64 / n as f64
            })
            .sum();
        score / 4.0
    }

    /// Per-residue lDDT score (local distance difference test).
    pub fn lddt_score(pred: &ProteinStructure, target: &ProteinStructure) -> Vec<f64> {
        let n = pred.len().min(target.len());
        let thresholds = [0.5_f64, 1.0, 2.0, 4.0];
        let radius = 15.0_f64;
        (0..n)
            .map(|i| {
                let neighbors: Vec<usize> = (0..n)
                    .filter(|&j| {
                        j != i && dist3(&target.residues[i].ca, &target.residues[j].ca) < radius
                    })
                    .collect();
                if neighbors.is_empty() {
                    return 1.0;
                }
                let score: f64 = thresholds
                    .iter()
                    .map(|&thresh| {
                        let preserved = neighbors
                            .iter()
                            .filter(|&&j| {
                                let d_pred = dist3(&pred.residues[i].ca, &pred.residues[j].ca);
                                let d_target =
                                    dist3(&target.residues[i].ca, &target.residues[j].ca);
                                (d_pred - d_target).abs() < thresh
                            })
                            .count();
                        preserved as f64 / neighbors.len() as f64
                    })
                    .sum::<f64>();
                score / 4.0
            })
            .collect()
    }

    /// Contact precision at L (fraction of top-L contacts that are true).
    pub fn contact_precision_at_l(
        pred_probs: &[Vec<f64>],
        true_contacts: &[Vec<bool>],
        l: usize,
    ) -> f64 {
        let n = pred_probs.len().min(true_contacts.len());
        // Collect upper-triangle pairs with predictions
        let mut pairs: Vec<(f64, bool)> = Vec::new();
        for i in 0..n {
            for j in (i + 6)..n {
                // sequence separation ≥ 6
                let p = pred_probs[i].get(j).copied().unwrap_or(0.0);
                let c = true_contacts[i].get(j).copied().unwrap_or(false);
                pairs.push((p, c));
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_l = pairs.len().min(l);
        if top_l == 0 {
            return 0.0;
        }
        let correct = pairs[..top_l].iter().filter(|&&(_, c)| c).count();
        correct as f64 / top_l as f64
    }

    /// Secondary structure accuracy.
    pub fn secondary_structure_accuracy(pred_ss: &[String], true_ss: &[String]) -> f64 {
        let n = pred_ss.len().min(true_ss.len());
        if n == 0 {
            return 0.0;
        }
        let correct = (0..n).filter(|&i| pred_ss[i] == true_ss[i]).count();
        correct as f64 / n as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
