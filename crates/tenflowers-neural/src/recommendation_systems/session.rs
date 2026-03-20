//! Session-based recommendation: GRU-based session encoder.

use scirs2_core::random::{rngs::StdRng, SeedableRng};

use super::{matvec, sigmoid_f32, xavier_fill, RecResult, RecSysError};

// ─────────────────────────────────────────────────────────────────────────────
// SessionEncoderConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SessionEncoder`].
#[derive(Debug, Clone)]
pub struct SessionEncoderConfig {
    /// Number of items in vocabulary.
    pub n_items: usize,
    /// Item embedding dimension.
    pub emb_dim: usize,
    /// GRU hidden state dimension.
    pub hidden_dim: usize,
    /// Number of GRU layers.
    pub n_layers: usize,
}

impl Default for SessionEncoderConfig {
    fn default() -> Self {
        Self {
            n_items: 1000,
            emb_dim: 64,
            hidden_dim: 128,
            n_layers: 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GruLayer (private)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct GruLayer {
    // Reset gate: r = σ(W_r x + U_r h + b_r)
    wr: Vec<f32>,
    ur: Vec<f32>,
    br: Vec<f32>,
    // Update gate: z = σ(W_z x + U_z h + b_z)
    wz: Vec<f32>,
    uz: Vec<f32>,
    bz: Vec<f32>,
    // New candidate: ñ = tanh(W_n x + U_n (r ⊙ h) + b_n)
    wn: Vec<f32>,
    un: Vec<f32>,
    bn: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
}

impl GruLayer {
    fn new(input_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        let h = hidden_dim;
        let i = input_dim;

        let mut wr = vec![0.0_f32; h * i];
        xavier_fill(&mut wr, i, h, rng);
        let mut ur = vec![0.0_f32; h * h];
        xavier_fill(&mut ur, h, h, rng);
        let mut wz = vec![0.0_f32; h * i];
        xavier_fill(&mut wz, i, h, rng);
        let mut uz = vec![0.0_f32; h * h];
        xavier_fill(&mut uz, h, h, rng);
        let mut wn = vec![0.0_f32; h * i];
        xavier_fill(&mut wn, i, h, rng);
        let mut un = vec![0.0_f32; h * h];
        xavier_fill(&mut un, h, h, rng);

        Self {
            wr,
            ur,
            br: vec![0.0_f32; h],
            wz,
            uz,
            bz: vec![0.0_f32; h],
            wn,
            un,
            bn: vec![0.0_f32; h],
            input_dim,
            hidden_dim,
        }
    }

    /// Single GRU step: returns new hidden state.
    fn step(&self, x: &[f32], h_prev: &[f32]) -> Vec<f32> {
        let hd = self.hidden_dim;
        let id = self.input_dim;

        let wx_r = matvec(&self.wr, x, hd, id);
        let uh_r = matvec(&self.ur, h_prev, hd, hd);
        let r: Vec<f32> = (0..hd)
            .map(|k| sigmoid_f32(wx_r[k] + uh_r[k] + self.br[k]))
            .collect();

        let wx_z = matvec(&self.wz, x, hd, id);
        let uh_z = matvec(&self.uz, h_prev, hd, hd);
        let z: Vec<f32> = (0..hd)
            .map(|k| sigmoid_f32(wx_z[k] + uh_z[k] + self.bz[k]))
            .collect();

        // r ⊙ h_prev
        let rh: Vec<f32> = r
            .iter()
            .zip(h_prev.iter())
            .map(|(ri, hi)| ri * hi)
            .collect();
        let wx_n = matvec(&self.wn, x, hd, id);
        let uh_n = matvec(&self.un, &rh, hd, hd);
        let n_tilde: Vec<f32> = (0..hd)
            .map(|k| (wx_n[k] + uh_n[k] + self.bn[k]).tanh())
            .collect();

        // h = (1 - z) ⊙ n + z ⊙ h_prev
        (0..hd)
            .map(|k| (1.0 - z[k]) * n_tilde[k] + z[k] * h_prev[k])
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// GRU-based session encoder for session-based recommendation.
///
/// Reads a sequence of item IDs and outputs a fixed-size session
/// representation equal to the final hidden state.
#[derive(Debug, Clone)]
pub struct SessionEncoder {
    cfg: SessionEncoderConfig,
    pub(crate) item_emb: Vec<f32>,
    gru_layers: Vec<GruLayer>,
}

impl SessionEncoder {
    /// Create a randomly-initialised SessionEncoder.
    pub fn new(cfg: SessionEncoderConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = cfg.emb_dim;
        let h = cfg.hidden_dim;
        let n = cfg.n_items;

        let mut item_emb = vec![0.0_f32; n * d];
        xavier_fill(&mut item_emb, d, d, &mut rng);

        let mut gru_layers = Vec::new();
        let mut in_dim = d;
        for _ in 0..cfg.n_layers {
            gru_layers.push(GruLayer::new(in_dim, h, &mut rng));
            in_dim = h;
        }

        Self {
            cfg,
            item_emb,
            gru_layers,
        }
    }

    /// Encode a session sequence of item IDs to a hidden representation.
    ///
    /// Returns a vector of length `hidden_dim` representing the session.
    pub fn encode(&self, session: &[usize]) -> RecResult<Vec<f32>> {
        if session.is_empty() {
            return Err(RecSysError::EmptySequence);
        }
        let d = self.cfg.emb_dim;
        let h = self.cfg.hidden_dim;
        let n = self.cfg.n_items;

        let mut hidden = vec![0.0_f32; h];
        for &item in session {
            let item_clamped = item.min(n - 1);
            let emb = &self.item_emb[item_clamped * d..(item_clamped + 1) * d];
            let mut x: Vec<f32> = emb.to_vec();
            for layer in &self.gru_layers {
                let next_h = layer.step(&x, &hidden);
                hidden = next_h;
                x = hidden.clone();
            }
            let _ = x;
        }
        Ok(hidden)
    }
}
