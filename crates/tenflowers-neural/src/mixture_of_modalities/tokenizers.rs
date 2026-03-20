//! Concrete modality tokenizers: text (char-level BPE), image (VQ-VAE),
//! audio (RVQ), and tabular (numeric binning).

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::types::{ModalityData, ModalityTokenizer, MomModalityType};

// ─────────────────────────────────────────────────────────────────────────────
// §3a TextModalityTokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// Text tokenizer using character-level encoding with frequency-based merging.
pub struct TextModalityTokenizer {
    vocab_size: usize,
}

impl TextModalityTokenizer {
    /// Create a new text tokenizer with the given vocabulary size.
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl ModalityTokenizer for TextModalityTokenizer {
    fn modality(&self) -> MomModalityType {
        MomModalityType::Text
    }

    fn encode(&self, data: &ModalityData) -> Vec<usize> {
        let text = match data {
            ModalityData::Text(s) => s.as_str(),
            ModalityData::Code(s) => s.as_str(),
            ModalityData::Math(s) => s.as_str(),
            _ => return Vec::new(),
        };
        // Simplified char-level BPE simulation: map each byte mod vocab_size.
        text.bytes()
            .map(|b| (b as usize) % self.vocab_size)
            .collect()
    }

    fn decode(&self, token_ids: &[usize]) -> ModalityData {
        // Approximate round-trip: map each token ID back to a byte (mod 95 for printable ASCII).
        let s: String = token_ids
            .iter()
            .map(|&id| {
                let byte = ((id % 95) + 32) as u8; // printable ASCII range
                byte as char
            })
            .collect();
        ModalityData::Text(s)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3b ImagePatchTokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// VQ-VAE style image patch tokenizer.
pub struct ImagePatchTokenizer {
    /// Patch size (e.g. 16 for 16×16 patches).
    pub patch_size: usize,
    /// Total number of patches per image.
    pub n_patches: usize,
    /// VQ codebook size.
    pub codebook_size: usize,
    /// Codebook: `[codebook_size][patch_dim]`.
    pub codebook: Vec<Vec<f64>>,
}

impl ImagePatchTokenizer {
    /// Create a new patch tokenizer with a randomly-initialized codebook.
    pub fn new(
        patch_size: usize,
        n_patches: usize,
        codebook_size: usize,
        patch_dim: usize,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let codebook: Vec<Vec<f64>> = (0..codebook_size)
            .map(|_| {
                (0..patch_dim)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        Self {
            patch_size,
            n_patches,
            codebook_size,
            codebook,
        }
    }

    /// Find the nearest codebook entry by L2 distance.
    pub fn nearest_code(&self, patch: &[f64]) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;
        for (i, code) in self.codebook.iter().enumerate() {
            let dist: f64 = patch
                .iter()
                .zip(code.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx
    }
}

impl ModalityTokenizer for ImagePatchTokenizer {
    fn modality(&self) -> MomModalityType {
        MomModalityType::Image
    }

    fn encode(&self, data: &ModalityData) -> Vec<usize> {
        match data {
            ModalityData::ImagePatches(patches) => {
                patches.iter().map(|p| self.nearest_code(p)).collect()
            }
            _ => Vec::new(),
        }
    }

    fn decode(&self, token_ids: &[usize]) -> ModalityData {
        // Reconstruct by returning the codebook vectors for each token ID.
        let patches: Vec<Vec<f64>> = token_ids
            .iter()
            .map(|&id| {
                let idx = id % self.codebook_size;
                self.codebook[idx].clone()
            })
            .collect();
        ModalityData::ImagePatches(patches)
    }

    fn vocab_size(&self) -> usize {
        self.codebook_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3c AudioFrameTokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// Residual Vector Quantization (RVQ) audio frame tokenizer.
pub struct AudioFrameTokenizer {
    /// Number of codebooks (RVQ levels).
    pub n_codebooks: usize,
    /// Size of each codebook.
    pub codebook_size: usize,
    /// Dimension of each frame vector.
    pub frame_dim: usize,
    /// All codebooks: `[n_codebooks][codebook_size][frame_dim]`.
    pub codebooks: Vec<Vec<Vec<f64>>>,
}

impl AudioFrameTokenizer {
    /// Create a new audio tokenizer.
    pub fn new(n_codebooks: usize, codebook_size: usize, frame_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(7);
        let codebooks: Vec<Vec<Vec<f64>>> = (0..n_codebooks)
            .map(|_| {
                (0..codebook_size)
                    .map(|_| {
                        (0..frame_dim)
                            .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            n_codebooks,
            codebook_size,
            frame_dim,
            codebooks,
        }
    }

    /// Quantize a single frame using RVQ: returns one token ID per codebook level.
    pub fn quantize_frame(&self, frame: &[f64]) -> Vec<usize> {
        let mut residual: Vec<f64> = frame.to_vec();
        let mut ids = Vec::with_capacity(self.n_codebooks);
        for cb in &self.codebooks {
            let mut best_idx = 0usize;
            let mut best_dist = f64::INFINITY;
            for (i, code) in cb.iter().enumerate() {
                let dist: f64 = residual
                    .iter()
                    .zip(code.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            ids.push(best_idx);
            // Subtract the quantized vector to get residual for next level.
            for (r, c) in residual.iter_mut().zip(cb[best_idx].iter()) {
                *r -= c;
            }
        }
        ids
    }
}

impl ModalityTokenizer for AudioFrameTokenizer {
    fn modality(&self) -> MomModalityType {
        MomModalityType::Audio
    }

    fn encode(&self, data: &ModalityData) -> Vec<usize> {
        match data {
            ModalityData::AudioFrames(frames) => {
                frames.iter().flat_map(|f| self.quantize_frame(f)).collect()
            }
            _ => Vec::new(),
        }
    }

    fn decode(&self, token_ids: &[usize]) -> ModalityData {
        // Reconstruct frames: each group of n_codebooks token IDs -> one frame.
        let chunks: Vec<&[usize]> = token_ids.chunks(self.n_codebooks).collect();
        let frames: Vec<Vec<f64>> = chunks
            .iter()
            .map(|chunk| {
                let mut frame = vec![0.0f64; self.frame_dim];
                for (level, &tid) in chunk.iter().enumerate() {
                    if level < self.codebooks.len() {
                        let idx = tid % self.codebook_size;
                        for (f, c) in frame.iter_mut().zip(self.codebooks[level][idx].iter()) {
                            *f += c;
                        }
                    }
                }
                frame
            })
            .collect();
        ModalityData::AudioFrames(frames)
    }

    fn vocab_size(&self) -> usize {
        self.codebook_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3d TabularTokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenizer that maps each tabular cell to a token via uniform numeric binning.
pub struct TabularTokenizer {
    /// Number of bins for numeric values.
    pub n_numeric_bins: usize,
    /// Vocabulary size for categorical values.
    pub cat_vocab_size: usize,
}

impl TabularTokenizer {
    /// Create a new tabular tokenizer.
    pub fn new(n_numeric_bins: usize, cat_vocab_size: usize) -> Self {
        Self {
            n_numeric_bins,
            cat_vocab_size,
        }
    }

    /// Map a numeric value to a bin index via uniform binning.
    pub fn quantize_numeric(&self, v: f64, min: f64, max: f64) -> usize {
        if (max - min).abs() < f64::EPSILON {
            return 0;
        }
        let frac = (v - min) / (max - min);
        let bin = (frac * self.n_numeric_bins as f64).floor() as isize;
        bin.clamp(0, self.n_numeric_bins as isize - 1) as usize
    }
}

impl ModalityTokenizer for TabularTokenizer {
    fn modality(&self) -> MomModalityType {
        MomModalityType::Tabular
    }

    fn encode(&self, data: &ModalityData) -> Vec<usize> {
        match data {
            ModalityData::Tabular(rows) => {
                if rows.is_empty() {
                    return Vec::new();
                }
                let n_cols = rows[0].len();
                let mut col_min = vec![f64::INFINITY; n_cols];
                let mut col_max = vec![f64::NEG_INFINITY; n_cols];
                for row in rows {
                    for (j, &v) in row.iter().enumerate() {
                        if j < n_cols {
                            col_min[j] = col_min[j].min(v);
                            col_max[j] = col_max[j].max(v);
                        }
                    }
                }
                rows.iter()
                    .flat_map(|row| {
                        row.iter().enumerate().map(|(j, &v)| {
                            let min = col_min.get(j).copied().unwrap_or(0.0);
                            let max = col_max.get(j).copied().unwrap_or(1.0);
                            self.quantize_numeric(v, min, max)
                        })
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    fn decode(&self, token_ids: &[usize]) -> ModalityData {
        let row: Vec<f64> = token_ids
            .iter()
            .map(|&id| {
                let bin = id % self.n_numeric_bins;
                (bin as f64 + 0.5) / self.n_numeric_bins as f64
            })
            .collect();
        ModalityData::Tabular(vec![row])
    }

    fn vocab_size(&self) -> usize {
        self.n_numeric_bins + self.cat_vocab_size
    }
}
