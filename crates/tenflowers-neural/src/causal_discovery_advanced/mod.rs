//! Advanced Causal Discovery Methods
//!
//! Implements state-of-the-art algorithms for discovering causal structure from data:
//!
//! - **DirectLiNGAM**: Direct Linear Non-Gaussian Acyclic Model via residual independence.
//! - **NOTEARS**: Continuous optimization for DAG structure learning (Zheng et al. 2018).
//! - **GraNDAG**: Gradient-based non-parametric DAG learning with neural networks.
//! - **DCDI**: Differentiable Causal Discovery with Interventional data.
//! - **FCI**: Fast Causal Inference supporting latent confounders (PAG output).
//! - **CASTLE**: Causal Structure Learning via neural autoencoders with DAG constraint.
//! - **CausalBoosting**: Gradient boosting with built-in causal constraints.
//! - **ICP**: Invariant Causal Prediction for identifying causal parents across environments.
//! - **CausalEffect**: Propensity-score and doubly-robust ATE estimation.
//! - **CdMetrics**: SHD, F1 skeleton, normalized SHD, and AUROC for DAG evaluation.

mod shared;

pub mod castle;
pub mod continuous_opt;
pub mod fci;
pub mod icp_effects;
pub mod lingam;
pub mod metrics;

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — shared math helpers (needed by tests)
// ─────────────────────────────────────────────────────────────────────────────

pub use shared::{
    gauss_solve, invert_sym, mat_exp_approx, mean_f32, normal_cdf_f32, ols_residuals,
    ols_univariate, pearson_corr, std_f32, variance_f32,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — lingam
// ─────────────────────────────────────────────────────────────────────────────

pub use lingam::{direct_lingam, entropy_ica, mutual_information_approx, LinGamConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — continuous_opt (NOTEARS + GraNDAG + DCDI)
// ─────────────────────────────────────────────────────────────────────────────

pub use continuous_opt::{
    acyclicity_penalty, dcdi_step, grandag_loss, h_constraint, interventional_likelihood,
    notears_loss, notears_step, GraNDagConfig, InterventionData, MlpCausalModule, NoTearsConfig,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — fci
// ─────────────────────────────────────────────────────────────────────────────

pub use fci::{fci_rules, orient_v_structures, skeleton_search, Pag, PagEdge, SkeltonEdge};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — castle
// ─────────────────────────────────────────────────────────────────────────────

pub use castle::{
    castle_acyclicity_loss, castle_reconstruction_loss, castle_step, causal_boost, fit_stump,
    CastleConfig, CausalAutoEncoder, CausalBoostConfig, CausalDecisionStump,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — icp_effects
// ─────────────────────────────────────────────────────────────────────────────

pub use icp_effects::{
    doubly_robust_ate, estimate_propensity, icp_find_parents, ipw_ate, is_invariant,
    regression_residuals, sensitivity_analysis_bounds, CdEnvironment, PropensityScoreModel,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — metrics
// ─────────────────────────────────────────────────────────────────────────────

pub use metrics::{auroc_edges, f1_skeleton, normalized_shd, shd};
