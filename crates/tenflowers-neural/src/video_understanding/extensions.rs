use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

use super::softmax_f32;


// ─────────────────────────────────────────────────────────────────────────────
// 8. HeatmapPoseHead
// ─────────────────────────────────────────────────────────────────────────────

/// 2D Gaussian heatmap pose estimation head.
pub struct HeatmapPoseHead {
    pub n_joints: usize,
}

impl HeatmapPoseHead {
    pub fn new(n_joints: usize) -> Self {
        Self { n_joints }
    }

    /// Generate a 2D Gaussian heatmap for a single joint.
    ///
    /// Returns flattened [H, W] heatmap.
    pub fn gaussian_heatmap(cx: f32, cy: f32, h: usize, w: usize, sigma: f32) -> Vec<f32> {
        let mut hmap = vec![0.0_f32; h * w];
        let two_sigma_sq = 2.0 * sigma * sigma;
        for yi in 0..h {
            for xi in 0..w {
                let dy = yi as f32 - cy;
                let dx = xi as f32 - cx;
                hmap[yi * w + xi] = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
            }
        }
        hmap
    }

    /// Decode heatmap predictions into (x, y, confidence) per joint.
    ///
    /// `heatmap` layout: [n_joints, H, W] flattened.
    /// Returns Vec of (x, y, confidence) tuples.
    pub fn decode_heatmap(
        &self,
        heatmap: &[f32],
        h: usize,
        w: usize,
        n_joints: usize,
    ) -> Vec<(f32, f32, f32)> {
        assert_eq!(heatmap.len(), n_joints * h * w);
        let mut results = Vec::with_capacity(n_joints);
        for ji in 0..n_joints {
            let offset = ji * h * w;
            let joint_map = &heatmap[offset..offset + h * w];
            // Find argmax
            let (max_idx, max_val) = joint_map.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(best_i, best_v), (i, &v)| {
                    if v > best_v {
                        (i, v)
                    } else {
                        (best_i, best_v)
                    }
                },
            );
            let y = (max_idx / w) as f32;
            let x = (max_idx % w) as f32;
            results.push((x, y, max_val));
        }
        results
    }

    /// Soft-argmax: sub-pixel precision pose decoding.
    ///
    /// Applies softmax over the heatmap then computes the expected position.
    pub fn soft_argmax(
        &self,
        heatmap: &[f32],
        h: usize,
        w: usize,
        n_joints: usize,
    ) -> Vec<(f32, f32, f32)> {
        assert_eq!(heatmap.len(), n_joints * h * w);
        let mut results = Vec::with_capacity(n_joints);
        for ji in 0..n_joints {
            let offset = ji * h * w;
            let joint_map = &heatmap[offset..offset + h * w];
            let probs = softmax_f32(joint_map);
            let mut ex = 0.0_f32;
            let mut ey = 0.0_f32;
            for yi in 0..h {
                for xi in 0..w {
                    let p = probs[yi * w + xi];
                    ex += p * xi as f32;
                    ey += p * yi as f32;
                }
            }
            let conf = joint_map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            results.push((ex, ey, conf));
        }
        results
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. TemporalActionDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal action detection: 1D proposal generation with sliding-window NMS.
pub struct TemporalActionDetector {
    pub min_len: usize,
    pub threshold: f32,
    pub iou_threshold: f32,
}

impl TemporalActionDetector {
    pub fn new(min_len: usize, threshold: f32, iou_threshold: f32) -> Self {
        Self {
            min_len,
            threshold,
            iou_threshold,
        }
    }

    /// Temporal IoU between two segments [s1,e1) and [s2,e2).
    pub fn temporal_iou(s1: usize, e1: usize, s2: usize, e2: usize) -> f32 {
        let inter_start = s1.max(s2);
        let inter_end = e1.min(e2);
        if inter_end <= inter_start {
            return 0.0;
        }
        let inter = (inter_end - inter_start) as f32;
        let union = (e1.max(e2) - s1.min(s2)) as f32;
        if union == 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Generate action proposals via connected-component thresholding + NMS.
    ///
    /// `scores` is a 1D array of length `t` (per-frame action scores).
    /// Returns list of (start, end, score) proposals after NMS.
    pub fn generate_proposals(
        &self,
        scores: &[f32],
        t: usize,
        min_len: usize,
        threshold: f32,
    ) -> Vec<(usize, usize, f32)> {
        assert_eq!(scores.len(), t);
        let mut proposals: Vec<(usize, usize, f32)> = Vec::new();

        // Connected-component proposal generation
        let mut in_segment = false;
        let mut seg_start = 0usize;
        for i in 0..t {
            if scores[i] >= threshold {
                if !in_segment {
                    seg_start = i;
                    in_segment = true;
                }
            } else if in_segment {
                in_segment = false;
                let seg_len = i - seg_start;
                if seg_len >= min_len {
                    let seg_score =
                        scores[seg_start..i].iter().cloned().sum::<f32>() / seg_len as f32;
                    proposals.push((seg_start, i, seg_score));
                }
            }
        }
        // Handle segment reaching end of sequence
        if in_segment {
            let seg_len = t - seg_start;
            if seg_len >= min_len {
                let seg_score = scores[seg_start..t].iter().cloned().sum::<f32>() / seg_len as f32;
                proposals.push((seg_start, t, seg_score));
            }
        }

        // Sort by score descending
        proposals.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // 1D NMS
        let mut keep = vec![true; proposals.len()];
        for i in 0..proposals.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..proposals.len() {
                if !keep[j] {
                    continue;
                }
                let iou = Self::temporal_iou(
                    proposals[i].0,
                    proposals[i].1,
                    proposals[j].0,
                    proposals[j].1,
                );
                if iou > self.iou_threshold {
                    keep[j] = false;
                }
            }
        }

        proposals
            .into_iter()
            .enumerate()
            .filter_map(|(i, p)| if keep[i] { Some(p) } else { None })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. VideoAugmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Video augmentation pipeline.
///
/// Input/output convention: clips are [T, H, W, C] flattened.
pub struct VideoAugmentation {
    pub min_frames: usize,
    pub max_frames: usize,
    pub flip_prob: f32,
    pub reverse_prob: f32,
}

impl VideoAugmentation {
    pub fn new(min_frames: usize, max_frames: usize, flip_prob: f32, reverse_prob: f32) -> Self {
        Self {
            min_frames,
            max_frames,
            flip_prob,
            reverse_prob,
        }
    }

    /// Temporal crop: randomly crop `n_frames` consecutive frames.
    pub fn temporal_crop(
        clip: &[f32],
        t: usize,
        h: usize,
        w: usize,
        c: usize,
        n_frames: usize,
        rng: &mut StdRng,
    ) -> (Vec<f32>, usize) {
        let n = n_frames.min(t);
        let max_start = t.saturating_sub(n);
        let start = if max_start > 0 {
            (rng.random::<f64>() * max_start as f64) as usize
        } else {
            0
        };
        let frame_size = h * w * c;
        let out: Vec<f32> = clip[start * frame_size..(start + n) * frame_size].to_vec();
        (out, n)
    }

    /// Frame-rate jitter: randomly drop frames to simulate lower frame rate.
    ///
    /// Keeps every `stride`-th frame, where stride is sampled from [1, max_stride].
    pub fn framerate_jitter(
        clip: &[f32],
        t: usize,
        h: usize,
        w: usize,
        c: usize,
        max_stride: usize,
        rng: &mut StdRng,
    ) -> (Vec<f32>, usize) {
        let stride = if max_stride > 1 {
            1 + (rng.random::<f64>() * max_stride as f64) as usize
        } else {
            1
        };
        let frame_size = h * w * c;
        let mut out = Vec::new();
        let mut count = 0usize;
        let mut fi = 0usize;
        while fi < t {
            let src = &clip[fi * frame_size..(fi + 1) * frame_size];
            out.extend_from_slice(src);
            count += 1;
            fi += stride;
        }
        (out, count)
    }

    /// Random horizontal flip applied consistently across all frames.
    pub fn random_horizontal_flip(
        clip: &[f32],
        t: usize,
        h: usize,
        w: usize,
        c: usize,
        rng: &mut StdRng,
    ) -> Vec<f32> {
        let do_flip: f32 = rng.random();
        if do_flip > 0.5 {
            let mut out = vec![0.0_f32; clip.len()];
            for ti in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        let src = ((ti * h + hi) * w + wi) * c;
                        let dst = ((ti * h + hi) * w + (w - 1 - wi)) * c;
                        out[dst..(c + dst)].copy_from_slice(&clip[src..(c + src)]);
                    }
                }
            }
            out
        } else {
            clip.to_vec()
        }
    }

    /// Temporal reversal: reverse the order of frames.
    pub fn temporal_reversal(clip: &[f32], t: usize, h: usize, w: usize, c: usize) -> Vec<f32> {
        let frame_size = h * w * c;
        let mut out = vec![0.0_f32; clip.len()];
        for ti in 0..t {
            let src_start = (t - 1 - ti) * frame_size;
            let dst_start = ti * frame_size;
            out[dst_start..dst_start + frame_size]
                .copy_from_slice(&clip[src_start..src_start + frame_size]);
        }
        out
    }

    /// 3D Cutout: zero out a random 3D cube [dt, dh, dw] in the clip.
    pub fn cutout_3d(
        clip: &mut [f32],
        t: usize,
        h: usize,
        w: usize,
        c: usize,
        cut_t: usize,
        cut_h: usize,
        cut_w: usize,
        rng: &mut StdRng,
    ) {
        let t0 = if t > cut_t {
            (rng.random::<f64>() * (t - cut_t) as f64) as usize
        } else {
            0
        };
        let h0 = if h > cut_h {
            (rng.random::<f64>() * (h - cut_h) as f64) as usize
        } else {
            0
        };
        let w0 = if w > cut_w {
            (rng.random::<f64>() * (w - cut_w) as f64) as usize
        } else {
            0
        };
        for ti in t0..(t0 + cut_t).min(t) {
            for hi in h0..(h0 + cut_h).min(h) {
                for wi in w0..(w0 + cut_w).min(w) {
                    let base = ((ti * h + hi) * w + wi) * c;
                    for ci in 0..c {
                        clip[base + ci] = 0.0;
                    }
                }
            }
        }
    }

    /// Apply full augmentation pipeline.
    pub fn augment(
        &self,
        clip: &[f32],
        t: usize,
        h: usize,
        w: usize,
        c: usize,
        rng: &mut StdRng,
    ) -> (Vec<f32>, usize) {
        // Temporal crop
        let n_frames = if self.max_frames > self.min_frames {
            self.min_frames
                + (rng.random::<f64>() * (self.max_frames - self.min_frames + 1) as f64) as usize
        } else {
            self.min_frames
        };
        let (cropped, t_crop) = Self::temporal_crop(clip, t, h, w, c, n_frames, rng);

        // Frame rate jitter
        let (jittered, t_jitter) = Self::framerate_jitter(&cropped, t_crop, h, w, c, 3, rng);

        // Temporal reversal
        let reversed_val: f32 = rng.random();
        let after_reversal = if reversed_val < self.reverse_prob {
            Self::temporal_reversal(&jittered, t_jitter, h, w, c)
        } else {
            jittered
        };

        // Random horizontal flip
        let flipped = Self::random_horizontal_flip(&after_reversal, t_jitter, h, w, c, rng);

        // 3D Cutout
        let mut final_clip = flipped;
        let cut_t = (t_jitter / 4).max(1);
        let cut_h = (h / 4).max(1);
        let cut_w = (w / 4).max(1);
        Self::cutout_3d(&mut final_clip, t_jitter, h, w, c, cut_t, cut_h, cut_w, rng);

        (final_clip, t_jitter)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

