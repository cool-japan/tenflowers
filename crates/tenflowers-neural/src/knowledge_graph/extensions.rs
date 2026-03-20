//! Extensions for knowledge graph embeddings.
//!
//! This module re-exports the `TrainedKgeModel` enum and provides additional
//! utility types for working with trained KGE models.

// All core types are defined in mod.rs and re-exported through the module root.
// This file provides any additional extension traits or helper functions
// that complement the core KGE types.

use super::{ComplExModel, KgeModel, RotatEModel, TransEModel};

/// Dispatch wrapper that holds any trained KGE model variant.
///
/// This alias type ensures downstream code can use `TrainedKgeModel` without
/// knowing which concrete variant is stored, via the [`KgeModel`] trait.
pub trait KgeModelExt: KgeModel {
    /// Return a human-readable model name.
    fn model_name(&self) -> &'static str;

    /// Estimate storage size in number of f64 values.
    fn storage_size(&self) -> usize {
        self.embedding_dim()
    }
}

impl KgeModelExt for TransEModel {
    fn model_name(&self) -> &'static str {
        "TransE"
    }

    fn storage_size(&self) -> usize {
        let dim = self.config.dim;
        let n_ent = self.entity_emb.len();
        let n_rel = self.relation_emb.len();
        dim * (n_ent + n_rel)
    }
}

impl KgeModelExt for RotatEModel {
    fn model_name(&self) -> &'static str {
        "RotatE"
    }

    fn storage_size(&self) -> usize {
        let half_dim = self.entity_real[0].len();
        let n_ent = self.entity_real.len();
        let n_rel = self.relation_phase.len();
        // 2 * half_dim per entity (real + imag) + half_dim per relation (phases)
        2 * half_dim * n_ent + half_dim * n_rel
    }
}

impl KgeModelExt for ComplExModel {
    fn model_name(&self) -> &'static str {
        "ComplEx"
    }

    fn storage_size(&self) -> usize {
        let dim = self.config.dim;
        let n_ent = self.entity_real.len();
        let n_rel = self.relation_real.len();
        // 2 * dim per entity + 2 * dim per relation
        2 * dim * (n_ent + n_rel)
    }
}

/// Summary statistics about a trained KGE model.
#[derive(Debug, Clone)]
pub struct KgeModelSummary {
    /// Model family name (e.g. "TransE").
    pub model_name: String,
    /// Embedding dimension (logical).
    pub embedding_dim: usize,
    /// Total number of f64 parameters stored.
    pub storage_size: usize,
    /// Number of unique entities.
    pub n_entities: usize,
    /// Number of unique relations.
    pub n_relations: usize,
}

impl KgeModelSummary {
    /// Construct summary from any model implementing `KgeModelExt`.
    pub fn from_model<M: KgeModelExt>(
        model: &M,
        n_entities: usize,
        n_relations: usize,
    ) -> Self {
        Self {
            model_name: model.model_name().to_string(),
            embedding_dim: model.embedding_dim(),
            storage_size: model.storage_size(),
            n_entities,
            n_relations,
        }
    }
}
