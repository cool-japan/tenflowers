//! Visual Scene Graph Generation & Reasoning
//!
//! Implements end-to-end scene graph generation including object detection,
//! relationship prediction, graph neural network refinement, VQA, and metrics.
//!
//! - **SgObjectDetector**: Region-based MLP detector with NMS
//! - **SgRelationshipDetector**: Visual relationship detection (Lu et al. 2016)
//! - **SgMessagePassing**: GNN refinement via GRU-style message passing
//! - **SgSceneGraphGeneration**: Full pipeline (Xu et al. 2017 / Tang et al. 2019)
//! - **SgQueryEngine**: Structured scene graph querying
//! - **SgVQA**: Visual Question Answering over scene graphs
//! - **SgSceneComparison**: Graph edit distance, triplet recall, semantic similarity
//! - **SgMetrics**: Recall@K, mAP, predicate accuracy

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, HashSet};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from scene graph operations.
#[derive(Debug, Clone)]
pub enum SgError {
    InvalidGraph(String),
    DimensionError(String),
    NumericalError(String),
}

impl fmt::Display for SgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SgError::InvalidGraph(s) => write!(f, "SgError::InvalidGraph: {s}"),
            SgError::DimensionError(s) => write!(f, "SgError::DimensionError: {s}"),
            SgError::NumericalError(s) => write!(f, "SgError::NumericalError: {s}"),
        }
    }
}

impl std::error::Error for SgError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn sg_relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn sg_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn sg_softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let denom = if sum < 1e-15 { 1e-15 } else { sum };
    exps.iter().map(|&e| e / denom).collect()
}

/// Matrix-vector multiply: result[i] = sum_j W[i][j] * x[j] + b[i]
fn sg_matvec(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let dot: f64 = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            dot + bi
        })
        .collect()
}

/// Xavier-normal weight initialization [rows × cols].
fn sg_xavier(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u1: f64 = rng.random::<f64>().max(1e-12);
                    let u2: f64 = rng.random::<f64>();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * scale
                })
                .collect()
        })
        .collect()
}

/// Concatenate two slices into a single Vec<f64>.
fn sg_concat(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 SgBoundingBox & SgObject
// ─────────────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in image coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct SgBoundingBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl SgBoundingBox {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }

    /// Box area (w × h).
    pub fn area(&self) -> f64 {
        (self.w * self.h).max(0.0)
    }

    /// Intersection-over-Union with another box.
    pub fn iou(&self, other: &SgBoundingBox) -> f64 {
        let ix1 = self.x.max(other.x);
        let iy1 = self.y.max(other.y);
        let ix2 = (self.x + self.w).min(other.x + other.w);
        let iy2 = (self.y + self.h).min(other.y + other.h);

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let inter = inter_w * inter_h;

        if inter <= 0.0 {
            return 0.0;
        }

        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Center coordinates (cx, cy).
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }

    /// Determine spatial relation of `self` relative to `other`.
    pub fn spatial_relation(&self, other: &SgBoundingBox) -> SgSpatialRelation {
        let (scx, scy) = self.center();
        let (ocx, ocy) = other.center();
        let overlap = self.iou(other);

        // Inside / Contains require strong containment
        let self_inside_other = self.x >= other.x
            && self.y >= other.y
            && (self.x + self.w) <= (other.x + other.w)
            && (self.y + self.h) <= (other.y + other.h);
        let other_inside_self = other.x >= self.x
            && other.y >= self.y
            && (other.x + other.w) <= (self.x + self.w)
            && (other.y + other.h) <= (self.y + self.h);

        if self_inside_other {
            return SgSpatialRelation::Inside;
        }
        if other_inside_self {
            return SgSpatialRelation::Contains;
        }
        if overlap > 0.1 {
            return SgSpatialRelation::Overlaps;
        }

        let dx = scx - ocx;
        let dy = scy - ocy;
        let dist = (dx * dx + dy * dy).sqrt();

        // "Distant" only when centres are extremely far apart relative to image
        // scale (> 10x the average box diagonal) with no dominant direction
        let diag = ((self.w * self.w + self.h * self.h).sqrt()
            + (other.w * other.w + other.h * other.h).sqrt())
            * 0.5;
        // Only mark Distant when truly ambiguous (dist >> diag and roughly equal dx/dy)
        if dist > diag * 10.0 && dx.abs() > 0.0 && dy.abs() > 0.0 {
            let ratio = dx.abs() / dy.abs();
            if ratio > 0.5 && ratio < 2.0 {
                return SgSpatialRelation::Distant;
            }
        }

        if dy.abs() > dx.abs() {
            if dy < 0.0 {
                SgSpatialRelation::Above
            } else {
                SgSpatialRelation::Below
            }
        } else if dx < 0.0 {
            SgSpatialRelation::Left
        } else {
            SgSpatialRelation::Right
        }
    }
}

/// Cardinal spatial relations between scene objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SgSpatialRelation {
    Above,
    Below,
    Left,
    Right,
    Inside,
    Contains,
    Overlaps,
    Distant,
}

/// A detected object node in the scene graph.
#[derive(Debug, Clone)]
pub struct SgObject {
    pub id: usize,
    pub label: String,
    pub confidence: f64,
    pub bbox: SgBoundingBox,
    pub attributes: Vec<String>,
    pub feature: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 SgRelationship & SgSceneGraph
// ─────────────────────────────────────────────────────────────────────────────

/// A directed predicate edge between two objects in the scene graph.
#[derive(Debug, Clone)]
pub struct SgRelationship {
    pub subject_id: usize,
    pub object_id: usize,
    pub predicate: String,
    pub predicate_id: usize,
    pub confidence: f64,
    pub feature: Vec<f64>,
}

/// Complete scene graph: objects + relationships + global image features.
#[derive(Debug, Clone)]
pub struct SgSceneGraph {
    pub objects: Vec<SgObject>,
    pub relationships: Vec<SgRelationship>,
    pub image_features: Vec<f64>,
}

impl Default for SgSceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SgSceneGraph {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            relationships: Vec::new(),
            image_features: Vec::new(),
        }
    }

    /// Add an object and return its assigned id.
    pub fn add_object(&mut self, mut obj: SgObject) -> usize {
        let id = self.objects.len();
        obj.id = id;
        self.objects.push(obj);
        id
    }

    /// Add a relationship edge to the graph.
    pub fn add_relationship(&mut self, rel: SgRelationship) {
        self.relationships.push(rel);
    }

    /// Number of object nodes.
    pub fn n_objects(&self) -> usize {
        self.objects.len()
    }

    /// Number of relationship edges.
    pub fn n_relationships(&self) -> usize {
        self.relationships.len()
    }

    /// Adjacency matrix [n_obj × n_obj] of max relationship confidences.
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.objects.len();
        let mut mat = vec![vec![0.0f64; n]; n];
        for rel in &self.relationships {
            let i = rel.subject_id;
            let j = rel.object_id;
            if i < n && j < n {
                let cur = mat[i][j];
                mat[i][j] = cur.max(rel.confidence);
            }
        }
        mat
    }

    /// Get (subject, object) pair for a given relationship index.
    pub fn get_subject_object(&self, rel_idx: usize) -> Option<(&SgObject, &SgObject)> {
        let rel = self.relationships.get(rel_idx)?;
        let subj = self.objects.iter().find(|o| o.id == rel.subject_id)?;
        let obj = self.objects.iter().find(|o| o.id == rel.object_id)?;
        Some((subj, obj))
    }

    /// All (subject_label, predicate, object_label) triplets.
    pub fn triplets(&self) -> Vec<(String, String, String)> {
        let obj_map: HashMap<usize, &SgObject> = self.objects.iter().map(|o| (o.id, o)).collect();
        self.relationships
            .iter()
            .filter_map(|rel| {
                let subj = obj_map.get(&rel.subject_id)?;
                let obj = obj_map.get(&rel.object_id)?;
                Some((subj.label.clone(), rel.predicate.clone(), obj.label.clone()))
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 SgObjectDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Region-based object detector using linear classification + bbox regression.
pub struct SgObjectDetector {
    pub cls_w: Vec<Vec<f64>>,
    pub cls_b: Vec<f64>,
    pub reg_w: Vec<Vec<f64>>,
    pub reg_b: Vec<f64>,
    pub n_classes: usize,
    pub feature_dim: usize,
    pub score_threshold: f64,
    pub nms_threshold: f64,
}

impl SgObjectDetector {
    pub fn new(feature_dim: usize, n_classes: usize) -> Self {
        let cls_w = sg_xavier(n_classes, feature_dim, 1001);
        let cls_b = vec![0.0; n_classes];
        let reg_w = sg_xavier(4, feature_dim, 1002);
        let reg_b = vec![0.0; 4];
        Self {
            cls_w,
            cls_b,
            reg_w,
            reg_b,
            n_classes,
            feature_dim,
            score_threshold: 0.3,
            nms_threshold: 0.5,
        }
    }

    /// Softmax class scores: softmax(W_cls @ feature + b_cls).
    pub fn classify(&self, feature: &[f64]) -> Vec<f64> {
        let logits = sg_matvec(&self.cls_w, &self.cls_b, feature);
        sg_softmax(&logits)
    }

    /// Predict refined bounding box from anchor + offset regression.
    pub fn regress_bbox(&self, feature: &[f64], anchor: &SgBoundingBox) -> SgBoundingBox {
        let delta = sg_matvec(&self.reg_w, &self.reg_b, feature);
        // Clamp offsets to prevent degenerate boxes
        let dx = delta[0].clamp(-2.0, 2.0);
        let dy = delta[1].clamp(-2.0, 2.0);
        let dw = delta[2].clamp(-2.0, 2.0);
        let dh = delta[3].clamp(-2.0, 2.0);

        let new_w = (anchor.w * dw.exp()).max(1e-3);
        let new_h = (anchor.h * dh.exp()).max(1e-3);
        SgBoundingBox {
            x: anchor.x + dx * anchor.w,
            y: anchor.y + dy * anchor.h,
            w: new_w,
            h: new_h,
        }
    }

    /// Detect objects: classify + regress + NMS.
    pub fn detect(
        &self,
        region_features: &[Vec<f64>],
        anchors: &[SgBoundingBox],
    ) -> Result<Vec<SgObject>, SgError> {
        if region_features.len() != anchors.len() {
            return Err(SgError::DimensionError(format!(
                "region_features.len()={} != anchors.len()={}",
                region_features.len(),
                anchors.len()
            )));
        }
        if region_features.is_empty() {
            return Ok(Vec::new());
        }

        // Build raw detections: (bbox, score, class_id)
        let mut detections: Vec<(SgBoundingBox, f64, usize)> = Vec::new();
        for (feat, anchor) in region_features.iter().zip(anchors.iter()) {
            if feat.len() != self.feature_dim {
                return Err(SgError::DimensionError(format!(
                    "feature len {} != expected {}",
                    feat.len(),
                    self.feature_dim
                )));
            }
            let scores = self.classify(feat);
            // Find best class (skip index 0 which is background)
            let (best_cls, best_score) = scores
                .iter()
                .enumerate()
                .skip(if self.n_classes > 1 { 1 } else { 0 })
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, &s)| (i, s))
                .unwrap_or((0, 0.0));

            if best_score >= self.score_threshold {
                let bbox = self.regress_bbox(feat, anchor);
                detections.push((bbox, best_score, best_cls));
            }
        }

        let kept = self.nms(&detections);

        let objects: Vec<SgObject> = kept
            .into_iter()
            .enumerate()
            .map(|(id, (bbox, confidence, cls_id))| SgObject {
                id,
                label: format!("class_{cls_id}"),
                confidence,
                bbox,
                attributes: Vec::new(),
                feature: region_features
                    .get(id % region_features.len())
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();

        Ok(objects)
    }

    /// Non-Maximum Suppression: keep highest-scoring, suppress IoU > threshold.
    pub fn nms(
        &self,
        detections: &[(SgBoundingBox, f64, usize)],
    ) -> Vec<(SgBoundingBox, f64, usize)> {
        if detections.is_empty() {
            return Vec::new();
        }
        // Sort descending by score
        let mut indices: Vec<usize> = (0..detections.len()).collect();
        indices.sort_by(|&a, &b| {
            detections[b]
                .1
                .partial_cmp(&detections[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept = Vec::new();
        let mut suppressed = vec![false; detections.len()];

        for &i in &indices {
            if suppressed[i] {
                continue;
            }
            kept.push(detections[i].clone());
            for &j in &indices {
                if i == j || suppressed[j] {
                    continue;
                }
                if detections[i].0.iou(&detections[j].0) > self.nms_threshold {
                    suppressed[j] = true;
                }
            }
        }
        kept
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 SgRelationshipDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Visual relationship detector (Lu et al. 2016).
/// Combines visual appearance features with spatial geometry.
pub struct SgRelationshipDetector {
    pub subject_embed: Vec<Vec<f64>>,
    pub object_embed: Vec<Vec<f64>>,
    pub visual_w: Vec<Vec<f64>>,
    pub visual_b: Vec<f64>,
    pub n_predicates: usize,
    pub feature_dim: usize,
    pub embed_dim: usize,
}

impl SgRelationshipDetector {
    pub fn new(feature_dim: usize, n_predicates: usize, embed_dim: usize) -> Self {
        // visual_w: [n_predicates × (2*feature_dim + 4)]
        let input_dim = 2 * feature_dim + 4;
        let subject_embed = sg_xavier(embed_dim, feature_dim, 2001);
        let object_embed = sg_xavier(embed_dim, feature_dim, 2002);
        let visual_w = sg_xavier(n_predicates, input_dim, 2003);
        let visual_b = vec![0.0; n_predicates];
        Self {
            subject_embed,
            object_embed,
            visual_w,
            visual_b,
            n_predicates,
            feature_dim,
            embed_dim,
        }
    }

    /// 4-dimensional spatial feature vector between subject and object.
    pub fn spatial_features(
        &self,
        subj_bbox: &SgBoundingBox,
        obj_bbox: &SgBoundingBox,
    ) -> Vec<f64> {
        let (scx, scy) = subj_bbox.center();
        let (ocx, ocy) = obj_bbox.center();

        // Relative center offset normalised by object size
        let denom_x = (obj_bbox.w).max(1e-6);
        let denom_y = (obj_bbox.h).max(1e-6);
        let rel_x = (scx - ocx) / denom_x;
        let rel_y = (scy - ocy) / denom_y;

        // Log width/height ratio
        let log_w_ratio = (subj_bbox.w.max(1e-6) / obj_bbox.w.max(1e-6)).ln();
        let log_h_ratio = (subj_bbox.h.max(1e-6) / obj_bbox.h.max(1e-6)).ln();

        vec![rel_x, rel_y, log_w_ratio, log_h_ratio]
    }

    /// Predicate probability distribution over n_predicates.
    pub fn predict_predicates(
        &self,
        subj_feature: &[f64],
        obj_feature: &[f64],
        subj_bbox: &SgBoundingBox,
        obj_bbox: &SgBoundingBox,
    ) -> Vec<f64> {
        let spatial = self.spatial_features(subj_bbox, obj_bbox);
        let combined = sg_concat(&sg_concat(subj_feature, obj_feature), &spatial);
        let logits = sg_matvec(&self.visual_w, &self.visual_b, &combined);
        sg_softmax(&logits)
    }

    /// Detect relationships for all object pairs in the scene graph; return top-k.
    pub fn detect_relationships(
        &self,
        scene_graph: &SgSceneGraph,
        top_k: usize,
    ) -> Result<Vec<SgRelationship>, SgError> {
        let n = scene_graph.n_objects();
        if n < 2 {
            return Ok(Vec::new());
        }

        let mut candidates: Vec<(f64, SgRelationship)> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let subj = &scene_graph.objects[i];
                let obj = &scene_graph.objects[j];

                if subj.feature.len() != self.feature_dim {
                    return Err(SgError::DimensionError(format!(
                        "subject feature dim {} != {}",
                        subj.feature.len(),
                        self.feature_dim
                    )));
                }
                if obj.feature.len() != self.feature_dim {
                    return Err(SgError::DimensionError(format!(
                        "object feature dim {} != {}",
                        obj.feature.len(),
                        self.feature_dim
                    )));
                }

                let probs =
                    self.predict_predicates(&subj.feature, &obj.feature, &subj.bbox, &obj.bbox);

                // Take best predicate for this pair
                if let Some((pred_id, &confidence)) = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    let rel = SgRelationship {
                        subject_id: subj.id,
                        object_id: obj.id,
                        predicate: format!("pred_{pred_id}"),
                        predicate_id: pred_id,
                        confidence,
                        feature: probs.clone(),
                    };
                    candidates.push((confidence, rel));
                }
            }
        }

        // Sort by confidence descending
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let k = top_k.min(candidates.len());
        Ok(candidates.into_iter().take(k).map(|(_, r)| r).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 SgMessagePassing
// ─────────────────────────────────────────────────────────────────────────────

/// Graph Neural Network for scene graph refinement via iterative message passing.
/// Uses GRU-style gating: Xu et al. 2017.
pub struct SgMessagePassing {
    pub n_iterations: usize,
    pub obj_transform: Vec<Vec<f64>>,
    pub rel_transform: Vec<Vec<f64>>,
    pub obj_gate: Vec<Vec<f64>>,
    pub obj_gate_b: Vec<f64>,
    pub feature_dim: usize,
}

impl SgMessagePassing {
    pub fn new(feature_dim: usize, n_iter: usize) -> Self {
        let obj_transform = sg_xavier(feature_dim, feature_dim, 3001);
        let rel_transform = sg_xavier(feature_dim, feature_dim, 3002);
        // Gate: [feature_dim × 2*feature_dim]
        let obj_gate = sg_xavier(feature_dim, 2 * feature_dim, 3003);
        let obj_gate_b = vec![0.0; feature_dim];
        Self {
            n_iterations: n_iter,
            obj_transform,
            rel_transform,
            obj_gate,
            obj_gate_b,
            feature_dim,
        }
    }

    /// Aggregate neighbourhood messages for each node:
    /// message_i = Σ_j adj\[i\]\[j\] * (W_rel @ features\[j\])
    pub fn aggregate_messages(&self, features: &[Vec<f64>], adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = features.len();
        (0..n)
            .map(|i| {
                let mut msg = vec![0.0f64; self.feature_dim];
                for j in 0..n {
                    let w = if i < adj.len() && j < adj[i].len() {
                        adj[i][j]
                    } else {
                        0.0
                    };
                    if w.abs() < 1e-15 {
                        continue;
                    }
                    let fj = &features[j];
                    // W_rel @ fj (rel_transform is [feature_dim × feature_dim])
                    for (k, row) in self.rel_transform.iter().enumerate() {
                        msg[k] += w * row
                            .iter()
                            .zip(fj.iter())
                            .map(|(&wi, &fi)| wi * fi)
                            .sum::<f64>();
                    }
                }
                msg
            })
            .collect()
    }

    /// GRU-style feature update:
    /// gate = sigmoid(W_gate @ [feature; message] + b)
    /// new_feature = gate ⊙ feature + (1-gate) ⊙ tanh(W_obj @ message)
    pub fn update_features(&self, features: &[Vec<f64>], messages: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features
            .iter()
            .zip(messages.iter())
            .map(|(feat, msg)| {
                // Compute gate input: concatenate feature and message
                let gate_input = sg_concat(feat, msg);
                let gate = sg_matvec(&self.obj_gate, &self.obj_gate_b, &gate_input)
                    .into_iter()
                    .map(sg_sigmoid)
                    .collect::<Vec<_>>();

                // Proposal: tanh(W_obj @ message)
                let proposal = sg_matvec(&self.obj_transform, &vec![0.0; self.feature_dim], msg)
                    .into_iter()
                    .map(|v| v.tanh())
                    .collect::<Vec<_>>();

                // new_feat = gate * feat + (1-gate) * proposal
                gate.iter()
                    .zip(feat.iter())
                    .zip(proposal.iter())
                    .map(|((&g, &f), &p)| g * f + (1.0 - g) * p)
                    .collect()
            })
            .collect()
    }

    /// Run n_iterations of message passing to refine object features in place.
    pub fn refine_scene_graph(&self, sg: &mut SgSceneGraph) -> Result<(), SgError> {
        let n = sg.n_objects();
        if n == 0 {
            return Ok(());
        }

        // Validate feature dims
        for obj in &sg.objects {
            if obj.feature.len() != self.feature_dim {
                return Err(SgError::DimensionError(format!(
                    "object {} feature dim {} != {}",
                    obj.id,
                    obj.feature.len(),
                    self.feature_dim
                )));
            }
        }

        let adj = sg.adjacency_matrix();

        for _ in 0..self.n_iterations {
            let features: Vec<Vec<f64>> = sg.objects.iter().map(|o| o.feature.clone()).collect();
            let messages = self.aggregate_messages(&features, &adj);
            let updated = self.update_features(&features, &messages);
            for (obj, new_feat) in sg.objects.iter_mut().zip(updated) {
                obj.feature = new_feat;
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6 SgSceneGraphGeneration
// ─────────────────────────────────────────────────────────────────────────────

/// End-to-end scene graph generation pipeline.
/// Integrates object detection → GNN refinement → relationship prediction.
pub struct SgSceneGraphGeneration {
    pub detector: SgObjectDetector,
    pub rel_detector: SgRelationshipDetector,
    pub mp: SgMessagePassing,
    pub feature_dim: usize,
    pub n_classes: usize,
    pub n_predicates: usize,
    pub max_objects: usize,
    pub max_rels: usize,
}

impl SgSceneGraphGeneration {
    pub fn new(feature_dim: usize, n_classes: usize, n_predicates: usize) -> Self {
        let detector = SgObjectDetector::new(feature_dim, n_classes);
        let rel_detector = SgRelationshipDetector::new(feature_dim, n_predicates, feature_dim);
        let mp = SgMessagePassing::new(feature_dim, 2);
        Self {
            detector,
            rel_detector,
            mp,
            feature_dim,
            n_classes,
            n_predicates,
            max_objects: 32,
            max_rels: 64,
        }
    }

    /// Full pipeline: detect objects → GNN refine → predict relationships.
    pub fn generate(
        &self,
        image_features: Vec<f64>,
        region_features: Vec<Vec<f64>>,
        anchors: Vec<SgBoundingBox>,
    ) -> Result<SgSceneGraph, SgError> {
        let mut sg = SgSceneGraph::new();
        sg.image_features = image_features;

        if region_features.is_empty() {
            return Ok(sg);
        }

        // §6.1 Detect objects
        let mut objects = self.detector.detect(&region_features, &anchors)?;
        objects.truncate(self.max_objects);

        for obj in objects {
            sg.add_object(obj);
        }

        if sg.n_objects() < 2 {
            return Ok(sg);
        }

        // §6.2 GNN refinement (message passing)
        self.mp.refine_scene_graph(&mut sg)?;

        // §6.3 Predict relationships
        let rels = self.rel_detector.detect_relationships(&sg, self.max_rels)?;
        for rel in rels {
            sg.add_relationship(rel);
        }

        Ok(sg)
    }

    /// Scene graph training loss: cross-entropy for classes + smooth-L1 for boxes +
    /// cross-entropy for predicates.
    pub fn scene_graph_loss(
        &self,
        predicted: &SgSceneGraph,
        gt_objects: &[(usize, SgBoundingBox)],
        gt_relations: &[(usize, usize, usize)],
    ) -> f64 {
        let mut loss = 0.0;
        let eps = 1e-9;

        // Classification loss (cross-entropy) for each GT object
        for (i, (gt_cls, gt_box)) in gt_objects.iter().enumerate() {
            if let Some(pred_obj) = predicted.objects.get(i) {
                // Recompute class scores from feature
                let scores = self.detector.classify(&pred_obj.feature);
                let cls_id = *gt_cls;
                let p = scores.get(cls_id).cloned().unwrap_or(eps);
                loss -= p.max(eps).ln();

                // Smooth-L1 regression loss
                let dx = pred_obj.bbox.x - gt_box.x;
                let dy = pred_obj.bbox.y - gt_box.y;
                let dw = pred_obj.bbox.w - gt_box.w;
                let dh = pred_obj.bbox.h - gt_box.h;
                for &d in &[dx, dy, dw, dh] {
                    loss += if d.abs() < 1.0 {
                        0.5 * d * d
                    } else {
                        d.abs() - 0.5
                    };
                }
            }
        }

        // Relationship loss (cross-entropy over predicates)
        for (r_idx, (_, gt_pred, _)) in gt_relations.iter().enumerate() {
            if let Some(pred_rel) = predicted.relationships.get(r_idx) {
                let p = pred_rel.feature.get(*gt_pred).cloned().unwrap_or(eps);
                loss -= p.max(eps).ln();
            }
        }

        loss
    }

    /// Recall@K: fraction of GT triplets found in top-K predicted triplets.
    pub fn recall_at_k(
        &self,
        predicted: &SgSceneGraph,
        gt_triplets: &[(String, String, String)],
        k: usize,
    ) -> f64 {
        SgMetrics::recall_at_k(&predicted.triplets(), gt_triplets, k)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7 SgSceneQuerying
// ─────────────────────────────────────────────────────────────────────────────

/// Structured query over a scene graph.
pub enum SgQuery {
    /// Find all objects with the given label.
    FindObjects(String),
    /// Find relationships matching (predicate, optional subject label, optional object label).
    FindRelationships(String, Option<String>, Option<String>),
    /// Count objects with the given label.
    CountObjects(String),
    /// Does an exact (subj, pred, obj) triplet exist?
    ExistsRelationship(String, String, String),
    /// Object ids spatially nearest to the given object id.
    NearestObject(usize),
}

/// Result of a scene graph query.
pub enum SgQueryResult {
    Objects(Vec<usize>),
    Count(usize),
    Boolean(bool),
    Relationships(Vec<usize>),
}

/// Stateless query engine operating on SgSceneGraph references.
pub struct SgQueryEngine;

impl SgQueryEngine {
    pub fn query(sg: &SgSceneGraph, query: &SgQuery) -> SgQueryResult {
        match query {
            SgQuery::FindObjects(label) => {
                let ids: Vec<usize> = sg
                    .objects
                    .iter()
                    .filter(|o| o.label == *label)
                    .map(|o| o.id)
                    .collect();
                SgQueryResult::Objects(ids)
            }

            SgQuery::FindRelationships(pred, subj_label, obj_label) => {
                let obj_map: HashMap<usize, &SgObject> =
                    sg.objects.iter().map(|o| (o.id, o)).collect();
                let ids: Vec<usize> = sg
                    .relationships
                    .iter()
                    .enumerate()
                    .filter(|(_, rel)| {
                        if rel.predicate != *pred {
                            return false;
                        }
                        if let Some(sl) = subj_label {
                            if let Some(subj) = obj_map.get(&rel.subject_id) {
                                if subj.label != *sl {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                        if let Some(ol) = obj_label {
                            if let Some(obj) = obj_map.get(&rel.object_id) {
                                if obj.label != *ol {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                        true
                    })
                    .map(|(i, _)| i)
                    .collect();
                SgQueryResult::Relationships(ids)
            }

            SgQuery::CountObjects(label) => {
                let cnt = sg.objects.iter().filter(|o| o.label == *label).count();
                SgQueryResult::Count(cnt)
            }

            SgQuery::ExistsRelationship(subj_label, pred, obj_label) => {
                let obj_map: HashMap<usize, &SgObject> =
                    sg.objects.iter().map(|o| (o.id, o)).collect();
                let exists = sg.relationships.iter().any(|rel| {
                    if rel.predicate != *pred {
                        return false;
                    }
                    let subj_ok = obj_map
                        .get(&rel.subject_id)
                        .map(|o| o.label == *subj_label)
                        .unwrap_or(false);
                    let obj_ok = obj_map
                        .get(&rel.object_id)
                        .map(|o| o.label == *obj_label)
                        .unwrap_or(false);
                    subj_ok && obj_ok
                });
                SgQueryResult::Boolean(exists)
            }

            SgQuery::NearestObject(target_id) => {
                let target = match sg.objects.iter().find(|o| o.id == *target_id) {
                    Some(t) => t,
                    None => return SgQueryResult::Objects(Vec::new()),
                };
                let (tx, ty) = target.bbox.center();

                let mut distances: Vec<(f64, usize)> = sg
                    .objects
                    .iter()
                    .filter(|o| o.id != *target_id)
                    .map(|o| {
                        let (cx, cy) = o.bbox.center();
                        let d = ((cx - tx).powi(2) + (cy - ty).powi(2)).sqrt();
                        (d, o.id)
                    })
                    .collect();

                distances
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let ids: Vec<usize> = distances.into_iter().map(|(_, id)| id).collect();
                SgQueryResult::Objects(ids)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8 SgVQA — Visual Question Answering
// ─────────────────────────────────────────────────────────────────────────────

/// Visual Question Answering over scene graphs using attention over object features.
pub struct SgVQA {
    pub question_embed_w: Vec<Vec<f64>>,
    pub attention_w: Vec<Vec<f64>>,
    pub answer_w: Vec<Vec<f64>>,
    pub answer_b: Vec<f64>,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_answers: usize,
    pub feature_dim: usize,
}

impl SgVQA {
    pub fn new(vocab_size: usize, embed_dim: usize, n_answers: usize, feature_dim: usize) -> Self {
        // question_embed_w: [embed_dim × vocab_size]
        let question_embed_w = sg_xavier(embed_dim, vocab_size, 4001);
        // attention_w: [1 × (embed_dim + feature_dim)]
        let attention_w = sg_xavier(1, embed_dim + feature_dim, 4002);
        // answer_w: [n_answers × (embed_dim + feature_dim)]
        let answer_w = sg_xavier(n_answers, embed_dim + feature_dim, 4003);
        let answer_b = vec![0.0; n_answers];
        Self {
            question_embed_w,
            attention_w,
            answer_w,
            answer_b,
            vocab_size,
            embed_dim,
            n_answers,
            feature_dim,
        }
    }

    /// Encode question by mean-pooling token embeddings.
    pub fn encode_question(&self, tokens: &[usize]) -> Vec<f64> {
        if tokens.is_empty() {
            return vec![0.0; self.embed_dim];
        }
        let mut q = vec![0.0f64; self.embed_dim];
        let mut count = 0usize;
        for &tok in tokens {
            let col = tok % self.vocab_size;
            for (i, row) in self.question_embed_w.iter().enumerate() {
                q[i] += row.get(col).cloned().unwrap_or(0.0);
            }
            count += 1;
        }
        let n = count.max(1) as f64;
        q.iter_mut().for_each(|v| *v /= n);
        q
    }

    /// Attend to object features using question as query.
    /// Returns attended visual feature vector.
    pub fn attend_to_objects(
        &self,
        question_embed: &[f64],
        object_features: &[Vec<f64>],
    ) -> Vec<f64> {
        if object_features.is_empty() {
            return vec![0.0; self.feature_dim];
        }

        // Attention logit for each object: attention_w[0] @ [q; o_i]
        let attn_row = &self.attention_w[0];
        let logits: Vec<f64> = object_features
            .iter()
            .map(|feat| {
                let combined = sg_concat(question_embed, feat);
                attn_row
                    .iter()
                    .zip(combined.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
            })
            .collect();

        let attn_weights = sg_softmax(&logits);

        // Weighted sum of object features
        let mut attended = vec![0.0f64; self.feature_dim];
        for (weight, feat) in attn_weights.iter().zip(object_features.iter()) {
            for (j, &fv) in feat.iter().enumerate() {
                if j < self.feature_dim {
                    attended[j] += weight * fv;
                }
            }
        }
        attended
    }

    /// Answer a visual question given token sequence and scene graph.
    pub fn answer(&self, question_tokens: &[usize], sg: &SgSceneGraph) -> Vec<f64> {
        let q = self.encode_question(question_tokens);

        // Collect object features (padded to feature_dim if shorter)
        let object_features: Vec<Vec<f64>> = sg
            .objects
            .iter()
            .map(|obj| {
                let mut f = obj.feature.clone();
                f.resize(self.feature_dim, 0.0);
                f
            })
            .collect();

        let v = self.attend_to_objects(&q, &object_features);
        let combined = sg_concat(&q, &v);
        let logits = sg_matvec(&self.answer_w, &self.answer_b, &combined);
        sg_softmax(&logits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9 SgSceneComparison
// ─────────────────────────────────────────────────────────────────────────────

/// Structural and semantic comparison between two scene graphs.
pub struct SgSceneComparison;

impl SgSceneComparison {
    fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
        let na: f64 = a.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if na < 1e-15 || nb < 1e-15 {
            return 0.0;
        }
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    /// Approximate graph edit distance: count unmatched objects + relationships.
    pub fn graph_edit_distance(sg1: &SgSceneGraph, sg2: &SgSceneGraph) -> f64 {
        // Match objects by label
        let labels1: Vec<String> = sg1.objects.iter().map(|o| o.label.clone()).collect();
        let labels2: Vec<String> = sg2.objects.iter().map(|o| o.label.clone()).collect();

        let mut matched = 0usize;
        let mut remaining2 = labels2.clone();
        for l1 in &labels1 {
            if let Some(pos) = remaining2.iter().position(|l2| l2 == l1) {
                matched += 1;
                remaining2.remove(pos);
            }
        }

        let unmatched_obj =
            (labels1.len().saturating_sub(matched)) + (labels2.len().saturating_sub(matched));

        // Match relationships by triplet string
        let trips1: Vec<String> = sg1
            .triplets()
            .into_iter()
            .map(|(s, p, o)| format!("{s}|{p}|{o}"))
            .collect();
        let trips2: Vec<String> = sg2
            .triplets()
            .into_iter()
            .map(|(s, p, o)| format!("{s}|{p}|{o}"))
            .collect();

        let set1: HashSet<&String> = trips1.iter().collect();
        let set2: HashSet<&String> = trips2.iter().collect();
        let common_rels = set1.intersection(&set2).count();
        let unmatched_rel =
            (trips1.len().saturating_sub(common_rels)) + (trips2.len().saturating_sub(common_rels));

        (unmatched_obj + unmatched_rel) as f64
    }

    /// Fraction of triplets in sg1 that also appear in sg2.
    pub fn triplet_recall(sg1: &SgSceneGraph, sg2: &SgSceneGraph) -> f64 {
        let t1 = sg1.triplets();
        if t1.is_empty() {
            return 1.0;
        }
        let t2_set: HashSet<(String, String, String)> = sg2.triplets().into_iter().collect();
        let matched = t1.iter().filter(|t| t2_set.contains(*t)).count();
        matched as f64 / t1.len() as f64
    }

    /// Mean cosine similarity of matched object features.
    pub fn semantic_similarity(sg1: &SgSceneGraph, sg2: &SgSceneGraph) -> f64 {
        if sg1.objects.is_empty() || sg2.objects.is_empty() {
            return 0.0;
        }
        // Build label → feature map for sg2
        let mut label_map: HashMap<&str, &Vec<f64>> = HashMap::new();
        for obj in &sg2.objects {
            label_map.entry(obj.label.as_str()).or_insert(&obj.feature);
        }

        let sims: Vec<f64> = sg1
            .objects
            .iter()
            .filter_map(|obj1| {
                label_map
                    .get(obj1.label.as_str())
                    .map(|feat2| Self::cosine_sim(&obj1.feature, feat2))
            })
            .collect();

        if sims.is_empty() {
            return 0.0;
        }
        sims.iter().sum::<f64>() / sims.len() as f64
    }

    /// Jaccard overlap of relationship sets by predicate label.
    pub fn relationship_overlap(sg1: &SgSceneGraph, sg2: &SgSceneGraph) -> f64 {
        let preds1: HashSet<&str> = sg1
            .relationships
            .iter()
            .map(|r| r.predicate.as_str())
            .collect();
        let preds2: HashSet<&str> = sg2
            .relationships
            .iter()
            .map(|r| r.predicate.as_str())
            .collect();
        let inter = preds1.intersection(&preds2).count();
        let union = preds1.union(&preds2).count();
        if union == 0 {
            return 1.0;
        }
        inter as f64 / union as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10 SgMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for scene graph generation.
pub struct SgMetrics;

impl SgMetrics {
    /// Recall@K: fraction of GT triplets found in top-K predicted triplets.
    pub fn recall_at_k(
        predicted_triplets: &[(String, String, String)],
        gt_triplets: &[(String, String, String)],
        k: usize,
    ) -> f64 {
        if gt_triplets.is_empty() {
            return 1.0;
        }
        let top_k = predicted_triplets.iter().take(k);
        let gt_set: HashSet<&(String, String, String)> = gt_triplets.iter().collect();
        let matched = top_k.filter(|t| gt_set.contains(*t)).count();
        matched as f64 / gt_triplets.len() as f64
    }

    /// Mean over per-predicate recall values.
    pub fn mean_recall(per_predicate_recalls: &[f64]) -> f64 {
        if per_predicate_recalls.is_empty() {
            return 0.0;
        }
        per_predicate_recalls.iter().sum::<f64>() / per_predicate_recalls.len() as f64
    }

    /// Predicate classification accuracy.
    pub fn predicate_classification_accuracy(predicted: &[usize], gt: &[usize]) -> f64 {
        if gt.is_empty() {
            return 1.0;
        }
        let correct = predicted
            .iter()
            .zip(gt.iter())
            .filter(|(&p, &g)| p == g)
            .count();
        correct as f64 / gt.len() as f64
    }

    /// Objects with at least one relationship / total objects.
    pub fn scene_graph_completeness(sg: &SgSceneGraph) -> f64 {
        let n = sg.n_objects();
        if n == 0 {
            return 0.0;
        }
        let mut connected: HashSet<usize> = HashSet::new();
        for rel in &sg.relationships {
            connected.insert(rel.subject_id);
            connected.insert(rel.object_id);
        }
        connected.len() as f64 / n as f64
    }

    /// Relationship density: actual / max_possible relationships.
    pub fn relationship_density(sg: &SgSceneGraph) -> f64 {
        let n = sg.n_objects();
        if n < 2 {
            return 0.0;
        }
        let max_possible = n * (n - 1);
        sg.n_relationships() as f64 / max_possible as f64
    }

    /// Mean Average Precision for object detection at given IoU threshold.
    #[allow(non_snake_case)]
    pub fn bbox_mAP(
        predicted: &[(SgBoundingBox, f64, usize)],
        gt: &[(SgBoundingBox, usize)],
        iou_threshold: f64,
    ) -> f64 {
        if gt.is_empty() || predicted.is_empty() {
            return 0.0;
        }

        // Group by class
        let mut classes: HashSet<usize> = HashSet::new();
        for (_, _, cls) in predicted {
            classes.insert(*cls);
        }
        for (_, cls) in gt {
            classes.insert(*cls);
        }

        let mut ap_sum = 0.0;
        let mut n_classes = 0usize;

        for cls in &classes {
            let cls_preds: Vec<(&SgBoundingBox, f64)> = predicted
                .iter()
                .filter(|(_, _, c)| c == cls)
                .map(|(bb, score, _)| (bb, *score))
                .collect();
            let cls_gt: Vec<&SgBoundingBox> = gt
                .iter()
                .filter(|(_, c)| c == cls)
                .map(|(bb, _)| bb)
                .collect();

            if cls_gt.is_empty() {
                continue;
            }
            n_classes += 1;

            // Sort predictions descending by score
            let mut sorted_preds = cls_preds.clone();
            sorted_preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let n_gt = cls_gt.len();
            let mut gt_matched = vec![false; n_gt];
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut precisions = Vec::new();
            let mut recalls = Vec::new();

            for (pred_bb, _) in &sorted_preds {
                let mut best_iou = 0.0;
                let mut best_gt = None;
                for (gi, gt_bb) in cls_gt.iter().enumerate() {
                    if gt_matched[gi] {
                        continue;
                    }
                    let iou = pred_bb.iou(gt_bb);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gt = Some(gi);
                    }
                }
                if best_iou >= iou_threshold {
                    if let Some(gi) = best_gt {
                        gt_matched[gi] = true;
                        tp += 1;
                    }
                } else {
                    fp += 1;
                }
                let p = if tp + fp > 0 {
                    tp as f64 / (tp + fp) as f64
                } else {
                    0.0
                };
                let r = tp as f64 / n_gt as f64;
                precisions.push(p);
                recalls.push(r);
            }

            // Compute AP via 11-point interpolation
            let ap = Self::compute_ap_11_point(&precisions, &recalls);
            ap_sum += ap;
        }

        if n_classes == 0 {
            0.0
        } else {
            (ap_sum / n_classes as f64).clamp(0.0, 1.0)
        }
    }

    fn compute_ap_11_point(precisions: &[f64], recalls: &[f64]) -> f64 {
        let mut ap = 0.0;
        for t in 0..=10 {
            let recall_thresh = t as f64 / 10.0;
            // Max precision at recall >= thresh
            let max_p = precisions
                .iter()
                .zip(recalls.iter())
                .filter(|(_, &r)| r >= recall_thresh)
                .map(|(&p, _)| p)
                .fold(0.0f64, f64::max);
            ap += max_p / 11.0;
        }
        ap
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    include!("tests.rs");
}
