//! Additional LM evaluation metrics and utilities.

use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §A  Distinct-N (diversity metric)
// ─────────────────────────────────────────────────────────────────────────────

/// Distinct-N diversity metric: fraction of unique N-grams in generated text.
#[derive(Debug, Clone)]
pub struct LmeDistinctN {
    pub n: usize,
}

impl LmeDistinctN {
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(TensorError::compute_error_simple(
                "n must be >= 1 for Distinct-N".to_string(),
            ));
        }
        Ok(Self { n })
    }

    /// Compute distinct-N: |unique N-grams| / |total N-grams|.
    pub fn compute(&self, text: &str) -> Result<f64> {
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
        if tokens.len() < self.n {
            return Ok(1.0); // All trivially unique.
        }
        let total = tokens.len() - self.n + 1;
        let mut unique = std::collections::HashSet::new();
        for i in 0..total {
            let gram: Vec<&str> = tokens[i..i + self.n].iter().map(|s| s.as_str()).collect();
            unique.insert(gram.join(" "));
        }
        Ok(unique.len() as f64 / total as f64)
    }

    /// Corpus-level distinct-N across multiple texts.
    pub fn compute_corpus(&self, texts: &[&str]) -> Result<f64> {
        if texts.is_empty() {
            return Err(TensorError::compute_error_simple(
                "texts must be non-empty".to_string(),
            ));
        }
        let mut all_unique = std::collections::HashSet::new();
        let mut all_total = 0usize;
        for &text in texts {
            let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
            if tokens.len() < self.n {
                continue;
            }
            let total = tokens.len() - self.n + 1;
            all_total += total;
            for i in 0..total {
                let gram: Vec<&str> = tokens[i..i + self.n].iter().map(|s| s.as_str()).collect();
                all_unique.insert(gram.join(" "));
            }
        }
        if all_total == 0 {
            return Ok(1.0);
        }
        Ok(all_unique.len() as f64 / all_total as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §B  LmeMeteor (lightweight Meteor-like metric)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight METEOR-like metric: unigram F-mean with γ fragmentation penalty.
#[derive(Debug, Clone)]
pub struct LmeMeteor {
    /// Alpha parameter: harmonic mean weight (default 0.9).
    pub alpha: f64,
    /// Beta: fragmentation penalty exponent (default 3.0).
    pub beta: f64,
    /// Gamma: fragmentation penalty coefficient (default 0.5).
    pub gamma: f64,
}

impl LmeMeteor {
    pub fn new(alpha: f64, beta: f64, gamma: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TensorError::compute_error_simple(
                "alpha must be in (0, 1]".to_string(),
            ));
        }
        Ok(Self { alpha, beta, gamma })
    }

    pub fn compute(&self, hypothesis: &str, reference: &str) -> Result<f64> {
        let hyp_tokens: Vec<String> = hypothesis
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let ref_tokens: Vec<String> = reference
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        if hyp_tokens.is_empty() && ref_tokens.is_empty() {
            return Ok(1.0);
        }
        if hyp_tokens.is_empty() || ref_tokens.is_empty() {
            return Ok(0.0);
        }

        // Build ref token counts.
        let mut ref_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for tok in &ref_tokens {
            *ref_counts.entry(tok.as_str()).or_insert(0) += 1;
        }

        let mut matched = 0usize;
        let mut used_ref: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for tok in &hyp_tokens {
            let avail = ref_counts.get(tok.as_str()).copied().unwrap_or(0);
            let used = used_ref.get(tok.as_str()).copied().unwrap_or(0);
            if used < avail {
                matched += 1;
                *used_ref.entry(tok.as_str()).or_insert(0) += 1;
            }
        }

        let precision = matched as f64 / hyp_tokens.len() as f64;
        let recall = matched as f64 / ref_tokens.len() as f64;

        let denom = self.alpha * precision + (1.0 - self.alpha) * recall;
        if denom < 1e-12 {
            return Ok(0.0);
        }
        let f_mean = precision * recall / denom;

        // Fragmentation: count contiguous matched chunks in hyp.
        let chunks = self.count_chunks(&hyp_tokens, &ref_tokens);
        let frag_penalty = if matched == 0 {
            0.0
        } else {
            self.gamma * (chunks as f64 / matched as f64).powf(self.beta)
        };

        let score = f_mean * (1.0 - frag_penalty);
        Ok(score.max(0.0))
    }

    fn count_chunks(&self, hyp: &[String], ref_set: &[String]) -> usize {
        let ref_set_lower: std::collections::HashSet<&str> =
            ref_set.iter().map(|s| s.as_str()).collect();
        let mut chunks = 0usize;
        let mut in_chunk = false;
        for tok in hyp {
            if ref_set_lower.contains(tok.as_str()) {
                if !in_chunk {
                    chunks += 1;
                    in_chunk = true;
                }
            } else {
                in_chunk = false;
            }
        }
        chunks
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §C  LmeChrF (character n-gram F-score)
// ─────────────────────────────────────────────────────────────────────────────

/// ChrF: character N-gram F-score.
#[derive(Debug, Clone)]
pub struct LmeChrF {
    /// Character N-gram order.
    pub char_n: usize,
    /// Word N-gram order (0 = no word ngrams).
    pub word_n: usize,
    /// Beta for F-score.
    pub beta: f64,
}

impl LmeChrF {
    pub fn new(char_n: usize, word_n: usize, beta: f64) -> Result<Self> {
        if char_n == 0 {
            return Err(TensorError::compute_error_simple(
                "char_n must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            char_n,
            word_n,
            beta,
        })
    }

    pub fn compute(&self, hypothesis: &str, reference: &str) -> Result<f64> {
        let char_score = self.char_ngram_f(hypothesis, reference, self.char_n);
        if self.word_n == 0 {
            return Ok(char_score);
        }
        let word_score = self.word_ngram_f(hypothesis, reference, self.word_n);
        Ok((char_score + word_score) / 2.0)
    }

    fn char_ngrams(text: &str, n: usize) -> std::collections::HashMap<String, usize> {
        let chars: Vec<char> = text.chars().collect();
        let mut map = std::collections::HashMap::new();
        if chars.len() < n {
            return map;
        }
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            *map.entry(gram).or_insert(0) += 1;
        }
        map
    }

    fn char_ngram_f(&self, hyp: &str, ref_: &str, n: usize) -> f64 {
        let hyp_ng = Self::char_ngrams(hyp, n);
        let ref_ng = Self::char_ngrams(ref_, n);
        let ref_total: usize = ref_ng.values().sum();
        let hyp_total: usize = hyp_ng.values().sum();
        if ref_total == 0 && hyp_total == 0 {
            return 1.0;
        }
        if ref_total == 0 || hyp_total == 0 {
            return 0.0;
        }
        let overlap: usize = hyp_ng
            .iter()
            .map(|(g, &hc)| hc.min(*ref_ng.get(g).unwrap_or(&0)))
            .sum();
        let prec = overlap as f64 / hyp_total as f64;
        let rec = overlap as f64 / ref_total as f64;
        let b2 = self.beta * self.beta;
        let denom = b2 * prec + rec;
        if denom < 1e-12 {
            0.0
        } else {
            (1.0 + b2) * prec * rec / denom
        }
    }

    fn word_ngram_f(&self, hyp: &str, ref_: &str, n: usize) -> f64 {
        let hyp_words: Vec<String> = hyp.split_whitespace().map(|s| s.to_lowercase()).collect();
        let ref_words: Vec<String> = ref_.split_whitespace().map(|s| s.to_lowercase()).collect();

        let mut hyp_ng: std::collections::HashMap<Vec<String>, usize> =
            std::collections::HashMap::new();
        let mut ref_ng: std::collections::HashMap<Vec<String>, usize> =
            std::collections::HashMap::new();

        if hyp_words.len() >= n {
            for i in 0..=(hyp_words.len() - n) {
                *hyp_ng.entry(hyp_words[i..i + n].to_vec()).or_insert(0) += 1;
            }
        }
        if ref_words.len() >= n {
            for i in 0..=(ref_words.len() - n) {
                *ref_ng.entry(ref_words[i..i + n].to_vec()).or_insert(0) += 1;
            }
        }

        let ref_total: usize = ref_ng.values().sum();
        let hyp_total: usize = hyp_ng.values().sum();
        if ref_total == 0 && hyp_total == 0 {
            return 1.0;
        }
        if ref_total == 0 || hyp_total == 0 {
            return 0.0;
        }
        let overlap: usize = hyp_ng
            .iter()
            .map(|(g, &hc)| hc.min(*ref_ng.get(g).unwrap_or(&0)))
            .sum();
        let prec = overlap as f64 / hyp_total as f64;
        let rec = overlap as f64 / ref_total as f64;
        let b2 = self.beta * self.beta;
        let denom = b2 * prec + rec;
        if denom < 1e-12 {
            0.0
        } else {
            (1.0 + b2) * prec * rec / denom
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §D  LmeCalibration (Expected Calibration Error)
// ─────────────────────────────────────────────────────────────────────────────

/// Expected calibration error for LM confidence.
#[derive(Debug, Clone)]
pub struct LmeCalibration {
    /// Number of equal-width confidence bins.
    pub n_bins: usize,
}

impl LmeCalibration {
    pub fn new(n_bins: usize) -> Result<Self> {
        if n_bins == 0 {
            return Err(TensorError::compute_error_simple(
                "n_bins must be >= 1".to_string(),
            ));
        }
        Ok(Self { n_bins })
    }

    /// Compute Expected Calibration Error (ECE).
    ///
    /// `confidences` — predicted probability of the chosen answer.
    /// `correct`     — whether the prediction was correct.
    pub fn ece(&self, confidences: &[f64], correct: &[bool]) -> Result<f64> {
        if confidences.is_empty() {
            return Err(TensorError::compute_error_simple(
                "confidences must be non-empty".to_string(),
            ));
        }
        if confidences.len() != correct.len() {
            return Err(TensorError::compute_error_simple(
                "confidences and correct must have equal length".to_string(),
            ));
        }
        let n = confidences.len() as f64;
        let bin_width = 1.0 / self.n_bins as f64;
        let mut ece = 0.0f64;

        for b in 0..self.n_bins {
            let lo = b as f64 * bin_width;
            let hi = lo + bin_width;
            let in_bin: Vec<(f64, bool)> = confidences
                .iter()
                .zip(correct.iter())
                .filter(|(&c, _)| c > lo && c <= hi)
                .map(|(&c, &ok)| (c, ok))
                .collect();
            if in_bin.is_empty() {
                continue;
            }
            let bm = in_bin.len() as f64;
            let avg_conf: f64 = in_bin.iter().map(|(c, _)| c).sum::<f64>() / bm;
            let avg_acc: f64 = in_bin.iter().filter(|(_, ok)| *ok).count() as f64 / bm;
            ece += (bm / n) * (avg_conf - avg_acc).abs();
        }
        Ok(ece)
    }

    /// Maximum Calibration Error (MCE): worst-bin gap.
    pub fn mce(&self, confidences: &[f64], correct: &[bool]) -> Result<f64> {
        if confidences.is_empty() {
            return Err(TensorError::compute_error_simple(
                "confidences must be non-empty".to_string(),
            ));
        }
        if confidences.len() != correct.len() {
            return Err(TensorError::compute_error_simple(
                "confidences and correct must have equal length".to_string(),
            ));
        }
        let bin_width = 1.0 / self.n_bins as f64;
        let mut mce = 0.0f64;

        for b in 0..self.n_bins {
            let lo = b as f64 * bin_width;
            let hi = lo + bin_width;
            let in_bin: Vec<(f64, bool)> = confidences
                .iter()
                .zip(correct.iter())
                .filter(|(&c, _)| c > lo && c <= hi)
                .map(|(&c, &ok)| (c, ok))
                .collect();
            if in_bin.is_empty() {
                continue;
            }
            let bm = in_bin.len() as f64;
            let avg_conf: f64 = in_bin.iter().map(|(c, _)| c).sum::<f64>() / bm;
            let avg_acc: f64 = in_bin.iter().filter(|(_, ok)| *ok).count() as f64 / bm;
            let gap = (avg_conf - avg_acc).abs();
            if gap > mce {
                mce = gap;
            }
        }
        Ok(mce)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §E  LmeWER (Word Error Rate)
// ─────────────────────────────────────────────────────────────────────────────

/// Word Error Rate (WER) and Character Error Rate (CER).
#[derive(Debug, Clone, Default)]
pub struct LmeWER;

impl LmeWER {
    pub fn new() -> Self {
        Self
    }

    /// WER = (S + D + I) / N (Levenshtein on word sequences).
    pub fn wer(&self, hypothesis: &str, reference: &str) -> Result<f64> {
        let hyp: Vec<String> = hypothesis
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let ref_: Vec<String> = reference
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        if ref_.is_empty() {
            if hyp.is_empty() {
                return Ok(0.0);
            }
            return Ok(1.0);
        }
        let edits = self.levenshtein_words(&hyp, &ref_);
        Ok(edits as f64 / ref_.len() as f64)
    }

    /// CER = Levenshtein(chars) / len(reference).
    pub fn cer(&self, hypothesis: &str, reference: &str) -> Result<f64> {
        let hyp: Vec<char> = hypothesis.chars().collect();
        let ref_: Vec<char> = reference.chars().collect();
        if ref_.is_empty() {
            if hyp.is_empty() {
                return Ok(0.0);
            }
            return Ok(1.0);
        }
        let edits = self.levenshtein_chars(&hyp, &ref_);
        Ok(edits as f64 / ref_.len() as f64)
    }

    fn levenshtein_words(&self, a: &[String], b: &[String]) -> usize {
        let m = a.len();
        let n = b.len();
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }
        for i in 1..=m {
            for j in 1..=n {
                dp[i][j] = if a[i - 1] == b[j - 1] {
                    dp[i - 1][j - 1]
                } else {
                    1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
                };
            }
        }
        dp[m][n]
    }

    fn levenshtein_chars(&self, a: &[char], b: &[char]) -> usize {
        let m = a.len();
        let n = b.len();
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }
        for i in 1..=m {
            for j in 1..=n {
                dp[i][j] = if a[i - 1] == b[j - 1] {
                    dp[i - 1][j - 1]
                } else {
                    1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
                };
            }
        }
        dp[m][n]
    }

    /// Batch WER over parallel hypothesis/reference slices.
    pub fn batch_wer(&self, hypotheses: &[&str], references: &[&str]) -> Result<f64> {
        if hypotheses.len() != references.len() || hypotheses.is_empty() {
            return Err(TensorError::compute_error_simple(
                "hypotheses and references must be same non-empty length".to_string(),
            ));
        }
        let mut total_edits = 0usize;
        let mut total_ref = 0usize;
        for (&h, &r) in hypotheses.iter().zip(references.iter()) {
            let hyp: Vec<String> = h.split_whitespace().map(|s| s.to_lowercase()).collect();
            let ref_: Vec<String> = r.split_whitespace().map(|s| s.to_lowercase()).collect();
            total_ref += ref_.len();
            total_edits += self.levenshtein_words(&hyp, &ref_);
        }
        if total_ref == 0 {
            return Ok(0.0);
        }
        Ok(total_edits as f64 / total_ref as f64)
    }
}
