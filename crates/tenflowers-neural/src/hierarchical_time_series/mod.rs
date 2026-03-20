//! Hierarchical Time Series Forecasting & Reconciliation.
//!
//! Implements cross-sectional and temporal hierarchy reconciliation methods:
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`HtsHierarchy`] | Tree structure with summing matrix S |
//! | [`HtsBottomUp`] | Bottom-up reconciliation |
//! | [`HtsTopDown`] | Top-down (AHP/PHA) reconciliation |
//! | [`HtsMinTrace`] | MinTrace optimal (Wickramasuriya et al. 2019) |
//! | [`HtsErmReconciliation`] | ERM learned weights |
//! | [`TemporalHierarchy`] | Temporal aggregation & cross-temporal |
//! | [`ProbabilisticForecaster`] | Quantile regression & CRPS |
//! | [`HtsForecaster`] | ETS / Holt-Winters / Theta method |
//! | [`HtsMetrics`] | MASE, RMSSE, CRPS, coherence error |
//!
//! # Design
//!
//! * No `unwrap()` -- all fallible paths return `Result<T, TensorError>`.
//! * Pure Rust, no external C/Fortran dependencies.
//! * Randomness via `scirs2_core::random`.

use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// Section 1 -- Linear Algebra Helpers
// ---------------------------------------------------------------------------

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiply: y = A * x (A is row-major [rows x cols]).
fn mat_vec(a: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    (0..rows)
        .map(|i| {
            let row_start = i * cols;
            dot(&a[row_start..row_start + cols], x)
        })
        .collect()
}

/// Matrix-matrix multiply: C = A * B (row-major).
fn mat_mul(a: &[f64], a_rows: usize, a_cols: usize, b: &[f64], b_cols: usize) -> Vec<f64> {
    let mut c = vec![0.0; a_rows * b_cols];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let a_ik = a[i * a_cols + k];
            for j in 0..b_cols {
                c[i * b_cols + j] += a_ik * b[k * b_cols + j];
            }
        }
    }
    c
}

/// Transpose of a row-major matrix.
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// Solve A x = b via Gaussian elimination with partial pivoting.
/// A is n x n row-major, b is length n. Returns x.
fn solve_linear(a: &[f64], n: usize, b: &[f64]) -> Result<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(TensorError::compute_error_simple(
                "Singular matrix in solve_linear".to_string(),
            ));
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let val = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * val;
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }
    Ok(x)
}

/// Invert an n x n matrix via Gauss-Jordan elimination.
fn invert_matrix(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(TensorError::compute_error_simple(
                "Singular matrix in invert_matrix".to_string(),
            ));
        }
        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }
        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                let val = aug[col * 2 * n + j];
                aug[row * 2 * n + j] -= factor * val;
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Section 2 -- Hierarchy Structure
// ---------------------------------------------------------------------------

/// A node in the hierarchical time series tree.
#[derive(Debug, Clone)]
pub struct HtsNode {
    /// Unique name of this node.
    pub name: String,
    /// Index in the flat node array.
    pub index: usize,
    /// Parent index (None for root).
    pub parent: Option<usize>,
    /// Children indices.
    pub children: Vec<usize>,
    /// Whether this is a bottom-level (leaf) node.
    pub is_leaf: bool,
    /// Level in the hierarchy (0 = root).
    pub level: usize,
}

/// Specification for building a hierarchy from a tree description.
/// Each entry is `(child_name, parent_name)`.
pub type HtsTreeSpec = Vec<(String, String)>;

/// Hierarchical time series structure with summing matrix.
#[derive(Debug, Clone)]
pub struct HtsHierarchy {
    /// All nodes in the hierarchy.
    pub nodes: Vec<HtsNode>,
    /// Name-to-index lookup.
    pub name_to_index: HashMap<String, usize>,
    /// Total number of nodes (all levels).
    pub total_nodes: usize,
    /// Number of bottom-level (leaf) nodes.
    pub num_bottom: usize,
    /// Indices of bottom-level nodes.
    pub bottom_indices: Vec<usize>,
    /// Summing matrix S: maps bottom-level to all levels.
    /// Shape: [total_nodes x num_bottom], row-major.
    pub summing_matrix: Vec<f64>,
    /// Number of levels in the hierarchy.
    pub num_levels: usize,
}

impl HtsHierarchy {
    /// Build hierarchy from a tree specification.
    ///
    /// `tree_spec` is a list of `(child, parent)` pairs. The root node
    /// is inferred as the node that appears as a parent but never as a child.
    pub fn build_from_tree(tree_spec: &HtsTreeSpec) -> Result<Self> {
        if tree_spec.is_empty() {
            return Err(TensorError::compute_error_simple(
                "HtsHierarchy: tree_spec must not be empty".to_string(),
            ));
        }

        // Collect all unique names and determine the root.
        let mut all_names: Vec<String> = Vec::new();
        let mut children_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut parents_set: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (child, parent) in tree_spec {
            children_set.insert(child.clone());
            parents_set.insert(parent.clone());
            if !all_names.contains(child) {
                all_names.push(child.clone());
            }
            if !all_names.contains(parent) {
                all_names.push(parent.clone());
            }
        }

        // Root = appears as parent but never as child.
        let roots: Vec<&String> = parents_set.difference(&children_set).collect();
        if roots.len() != 1 {
            return Err(TensorError::compute_error_simple(format!(
                "HtsHierarchy: expected exactly 1 root, found {}",
                roots.len()
            )));
        }
        let root_name = roots[0].clone();

        // Build adjacency: parent -> children.
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        let mut parent_map: HashMap<String, String> = HashMap::new();
        for (child, parent) in tree_spec {
            adj.entry(parent.clone()).or_default().push(child.clone());
            parent_map.insert(child.clone(), parent.clone());
        }

        // BFS to assign indices and levels.
        let mut nodes: Vec<HtsNode> = Vec::new();
        let mut name_to_index: HashMap<String, usize> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();

        let root_idx = 0;
        nodes.push(HtsNode {
            name: root_name.clone(),
            index: root_idx,
            parent: None,
            children: Vec::new(),
            is_leaf: false,
            level: 0,
        });
        name_to_index.insert(root_name.clone(), root_idx);
        queue.push_back(root_idx);

        while let Some(idx) = queue.pop_front() {
            let node_name = nodes[idx].name.clone();
            if let Some(child_names) = adj.get(&node_name) {
                for cname in child_names {
                    let cidx = nodes.len();
                    let level = nodes[idx].level + 1;
                    nodes.push(HtsNode {
                        name: cname.clone(),
                        index: cidx,
                        parent: Some(idx),
                        children: Vec::new(),
                        is_leaf: false,
                        level,
                    });
                    name_to_index.insert(cname.clone(), cidx);
                    nodes[idx].children.push(cidx);
                    queue.push_back(cidx);
                }
            }
        }

        // Mark leaves and compute num_levels.
        let mut num_levels = 0;
        for node in &mut nodes {
            if node.children.is_empty() {
                node.is_leaf = true;
            }
            if node.level + 1 > num_levels {
                num_levels = node.level + 1;
            }
        }

        let bottom_indices: Vec<usize> = nodes
            .iter()
            .filter(|n| n.is_leaf)
            .map(|n| n.index)
            .collect();
        let num_bottom = bottom_indices.len();
        let total_nodes = nodes.len();

        // Build summing matrix S [total_nodes x num_bottom].
        // For each bottom node b, S[i][b_local] = 1 if node i is an ancestor of b (or b itself).
        let mut summing_matrix = vec![0.0; total_nodes * num_bottom];
        for (b_local, &b_idx) in bottom_indices.iter().enumerate() {
            // Walk up from leaf to root, setting S entries.
            let mut current = Some(b_idx);
            while let Some(ci) = current {
                summing_matrix[ci * num_bottom + b_local] = 1.0;
                current = nodes[ci].parent;
            }
        }

        Ok(Self {
            nodes,
            name_to_index,
            total_nodes,
            num_bottom,
            bottom_indices,
            summing_matrix,
            num_levels,
        })
    }

    /// Get the summing matrix S as a reference.
    pub fn s_matrix(&self) -> &[f64] {
        &self.summing_matrix
    }

    /// Compute S * b where b is a bottom-level forecast vector.
    pub fn aggregate_bottom(&self, bottom: &[f64]) -> Result<Vec<f64>> {
        if bottom.len() != self.num_bottom {
            return Err(TensorError::compute_error_simple(format!(
                "HtsHierarchy::aggregate_bottom: expected {} values, got {}",
                self.num_bottom,
                bottom.len()
            )));
        }
        Ok(mat_vec(
            &self.summing_matrix,
            self.total_nodes,
            self.num_bottom,
            bottom,
        ))
    }

    /// Get indices of nodes at a given level.
    pub fn nodes_at_level(&self, level: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.level == level)
            .map(|n| n.index)
            .collect()
    }
}

/// Grouped time series: combines cross-sectional factors.
#[derive(Debug, Clone)]
pub struct HtsGroupedHierarchy {
    /// Factor names (e.g., "region", "product").
    pub factor_names: Vec<String>,
    /// Factor levels per factor.
    pub factor_levels: Vec<Vec<String>>,
    /// The underlying hierarchy built from all cross-products.
    pub hierarchy: HtsHierarchy,
}

impl HtsGroupedHierarchy {
    /// Build a grouped hierarchy from factor definitions.
    ///
    /// Creates a two-level tree: Total -> each combination of factor levels.
    pub fn build(factor_names: Vec<String>, factor_levels: Vec<Vec<String>>) -> Result<Self> {
        if factor_names.is_empty() || factor_levels.is_empty() {
            return Err(TensorError::compute_error_simple(
                "HtsGroupedHierarchy: need at least one factor".to_string(),
            ));
        }
        if factor_names.len() != factor_levels.len() {
            return Err(TensorError::compute_error_simple(
                "HtsGroupedHierarchy: factor_names and factor_levels must match".to_string(),
            ));
        }

        // Build a simple two-level hierarchy: Total -> bottom combinations.
        // Bottom = cartesian product of all factor levels.
        let mut bottom_names: Vec<String> = vec!["".to_string()];
        for levels in &factor_levels {
            let mut new_names = Vec::new();
            for prefix in &bottom_names {
                for lvl in levels {
                    let name = if prefix.is_empty() {
                        lvl.clone()
                    } else {
                        format!("{}_{}", prefix, lvl)
                    };
                    new_names.push(name);
                }
            }
            bottom_names = new_names;
        }

        let root_name = "Total".to_string();
        let tree_spec: HtsTreeSpec = bottom_names
            .iter()
            .map(|bn| (bn.clone(), root_name.clone()))
            .collect();

        let hierarchy = HtsHierarchy::build_from_tree(&tree_spec)?;
        Ok(Self {
            factor_names,
            factor_levels,
            hierarchy,
        })
    }
}

// ---------------------------------------------------------------------------
// Section 3 -- Bottom-Up Reconciliation
// ---------------------------------------------------------------------------

/// Bottom-up reconciliation: forecast at bottom level, sum up.
pub struct HtsBottomUp;

impl HtsBottomUp {
    /// Reconcile base forecasts using bottom-up approach.
    ///
    /// Only uses the bottom-level forecasts and aggregates upward via S.
    pub fn reconcile(base_forecasts: &[f64], hierarchy: &HtsHierarchy) -> Result<Vec<f64>> {
        if base_forecasts.len() != hierarchy.total_nodes {
            return Err(TensorError::compute_error_simple(format!(
                "HtsBottomUp: expected {} forecasts, got {}",
                hierarchy.total_nodes,
                base_forecasts.len()
            )));
        }
        // Extract bottom-level forecasts.
        let bottom: Vec<f64> = hierarchy
            .bottom_indices
            .iter()
            .map(|&i| base_forecasts[i])
            .collect();
        hierarchy.aggregate_bottom(&bottom)
    }
}

// ---------------------------------------------------------------------------
// Section 4 -- Top-Down Reconciliation
// ---------------------------------------------------------------------------

/// Method for computing top-down proportions.
#[derive(Debug, Clone, Copy)]
pub enum HtsTopDownMethod {
    /// Average Historical Proportions: p_i = mean(y_{i,t} / y_{total,t}).
    Ahp,
    /// Proportions of Historical Averages: p_i = mean(y_{i,t}) / mean(y_{total,t}).
    Pha,
}

/// Top-down reconciliation: forecast at top level, distribute proportionally.
pub struct HtsTopDown;

impl HtsTopDown {
    /// Compute top-down proportions from historical data.
    ///
    /// `historical` is [T x total_nodes] row-major, where T is the number of
    /// historical periods.
    pub fn compute_proportions(
        historical: &[f64],
        num_periods: usize,
        hierarchy: &HtsHierarchy,
        method: HtsTopDownMethod,
    ) -> Result<Vec<f64>> {
        let n = hierarchy.total_nodes;
        if historical.len() != num_periods * n {
            return Err(TensorError::compute_error_simple(format!(
                "HtsTopDown::compute_proportions: expected {} values, got {}",
                num_periods * n,
                historical.len()
            )));
        }
        let nb = hierarchy.num_bottom;
        let mut proportions = vec![0.0; nb];

        match method {
            HtsTopDownMethod::Ahp => {
                // p_i = mean over t of (y_{leaf_i,t} / y_{root,t})
                let root_idx = 0; // root is always index 0
                for (b_local, &b_idx) in hierarchy.bottom_indices.iter().enumerate() {
                    let mut sum_ratio = 0.0;
                    let mut count = 0;
                    for t in 0..num_periods {
                        let y_total = historical[t * n + root_idx];
                        if y_total.abs() > 1e-14 {
                            sum_ratio += historical[t * n + b_idx] / y_total;
                            count += 1;
                        }
                    }
                    proportions[b_local] = if count > 0 {
                        sum_ratio / count as f64
                    } else {
                        1.0 / nb as f64
                    };
                }
            }
            HtsTopDownMethod::Pha => {
                // p_i = mean(y_{leaf_i}) / mean(y_{root})
                let root_idx = 0;
                let mean_total: f64 = (0..num_periods)
                    .map(|t| historical[t * n + root_idx])
                    .sum::<f64>()
                    / num_periods as f64;
                for (b_local, &b_idx) in hierarchy.bottom_indices.iter().enumerate() {
                    let mean_leaf: f64 = (0..num_periods)
                        .map(|t| historical[t * n + b_idx])
                        .sum::<f64>()
                        / num_periods as f64;
                    proportions[b_local] = if mean_total.abs() > 1e-14 {
                        mean_leaf / mean_total
                    } else {
                        1.0 / nb as f64
                    };
                }
            }
        }

        // Normalize to sum to 1.
        let sum: f64 = proportions.iter().sum();
        if sum.abs() > 1e-14 {
            for p in &mut proportions {
                *p /= sum;
            }
        }
        Ok(proportions)
    }

    /// Reconcile using top-down approach with given proportions.
    pub fn reconcile(
        base_forecasts: &[f64],
        hierarchy: &HtsHierarchy,
        proportions: &[f64],
    ) -> Result<Vec<f64>> {
        if base_forecasts.len() != hierarchy.total_nodes {
            return Err(TensorError::compute_error_simple(format!(
                "HtsTopDown: expected {} forecasts, got {}",
                hierarchy.total_nodes,
                base_forecasts.len()
            )));
        }
        if proportions.len() != hierarchy.num_bottom {
            return Err(TensorError::compute_error_simple(format!(
                "HtsTopDown: expected {} proportions, got {}",
                hierarchy.num_bottom,
                proportions.len()
            )));
        }
        // Distribute the top-level forecast.
        let top_forecast = base_forecasts[0];
        let bottom: Vec<f64> = proportions.iter().map(|&p| top_forecast * p).collect();
        hierarchy.aggregate_bottom(&bottom)
    }
}

// ---------------------------------------------------------------------------
// Section 5 -- MinTrace Optimal Reconciliation
// ---------------------------------------------------------------------------

/// Covariance estimation method for MinTrace reconciliation.
#[derive(Debug, Clone, Copy)]
pub enum HtsCovarianceMethod {
    /// OLS: W = I (identity).
    Ols,
    /// WLS: W = diag(variance of residuals).
    Wls,
    /// Shrinkage: Ledoit-Wolf toward diagonal target.
    Shrinkage,
}

/// MinTrace optimal reconciliation (Wickramasuriya et al. 2019).
///
/// Computes the projection matrix P = S (S' W^{-1} S)^{-1} S' W^{-1}
/// that minimizes the trace of the reconciled forecast error covariance.
pub struct HtsMinTrace;

impl HtsMinTrace {
    /// Estimate covariance matrix W from residuals.
    ///
    /// `residuals` is [T x n] row-major, where T = number of periods, n = total nodes.
    fn estimate_covariance(
        residuals: &[f64],
        num_periods: usize,
        n: usize,
        method: HtsCovarianceMethod,
    ) -> Result<Vec<f64>> {
        if residuals.len() != num_periods * n {
            return Err(TensorError::compute_error_simple(format!(
                "MinTrace: expected {} residual values, got {}",
                num_periods * n,
                residuals.len()
            )));
        }
        if num_periods < 2 {
            return Err(TensorError::compute_error_simple(
                "MinTrace: need at least 2 periods for covariance estimation".to_string(),
            ));
        }

        match method {
            HtsCovarianceMethod::Ols => {
                // W = I
                let mut w = vec![0.0; n * n];
                for i in 0..n {
                    w[i * n + i] = 1.0;
                }
                Ok(w)
            }
            HtsCovarianceMethod::Wls => {
                // W = diag(sample variance of each series' residuals)
                let mut w = vec![0.0; n * n];
                for j in 0..n {
                    let mean: f64 = (0..num_periods).map(|t| residuals[t * n + j]).sum::<f64>()
                        / num_periods as f64;
                    let var: f64 = (0..num_periods)
                        .map(|t| {
                            let d = residuals[t * n + j] - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / (num_periods - 1) as f64;
                    w[j * n + j] = var.max(1e-10);
                }
                Ok(w)
            }
            HtsCovarianceMethod::Shrinkage => {
                // Full sample covariance with Ledoit-Wolf shrinkage.
                let means: Vec<f64> = (0..n)
                    .map(|j| {
                        (0..num_periods).map(|t| residuals[t * n + j]).sum::<f64>()
                            / num_periods as f64
                    })
                    .collect();

                // Sample covariance
                let t_f = num_periods as f64;
                let mut s_cov = vec![0.0; n * n];
                for t in 0..num_periods {
                    for i in 0..n {
                        let di = residuals[t * n + i] - means[i];
                        for j in i..n {
                            let dj = residuals[t * n + j] - means[j];
                            let v = di * dj;
                            s_cov[i * n + j] += v;
                            if i != j {
                                s_cov[j * n + i] += v;
                            }
                        }
                    }
                }
                for v in &mut s_cov {
                    *v /= (num_periods - 1) as f64;
                }

                // Shrinkage target: diagonal of s_cov
                let mut target = vec![0.0; n * n];
                for i in 0..n {
                    target[i * n + i] = s_cov[i * n + i];
                }

                // Ledoit-Wolf shrinkage intensity (simplified)
                // rho* = sum_ij var(s_ij) / sum_ij (s_ij - t_ij)^2
                let mut num = 0.0;
                let mut denom = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        let s_ij = s_cov[i * n + j];
                        let t_ij = target[i * n + j];
                        // Approximate var(s_ij) using 4th-moment formula
                        let mut var_sij = 0.0;
                        for t in 0..num_periods {
                            let di = residuals[t * n + i] - means[i];
                            let dj = residuals[t * n + j] - means[j];
                            let prod = di * dj - s_ij;
                            var_sij += prod * prod;
                        }
                        var_sij /= (t_f - 1.0) * (t_f - 1.0);
                        num += var_sij;
                        denom += (s_ij - t_ij) * (s_ij - t_ij);
                    }
                }
                let rho = if denom > 1e-14 {
                    (num / denom).clamp(0.0, 1.0)
                } else {
                    1.0
                };

                // Shrunk covariance: (1-rho)*S + rho*T
                let mut w = vec![0.0; n * n];
                for k in 0..(n * n) {
                    w[k] = (1.0 - rho) * s_cov[k] + rho * target[k];
                }
                Ok(w)
            }
        }
    }

    /// Reconcile forecasts using MinTrace optimal reconciliation.
    ///
    /// P = S (S' W^{-1} S)^{-1} S' W^{-1}
    /// reconciled = P * base_forecasts
    pub fn reconcile(
        base_forecasts: &[f64],
        hierarchy: &HtsHierarchy,
        residuals: &[f64],
        num_periods: usize,
        method: HtsCovarianceMethod,
    ) -> Result<Vec<f64>> {
        let n = hierarchy.total_nodes;
        let m = hierarchy.num_bottom;

        if base_forecasts.len() != n {
            return Err(TensorError::compute_error_simple(format!(
                "MinTrace: expected {} forecasts, got {}",
                n,
                base_forecasts.len()
            )));
        }

        let w = Self::estimate_covariance(residuals, num_periods, n, method)?;
        let w_inv = invert_matrix(&w, n)?;

        let s = &hierarchy.summing_matrix; // [n x m]
        let s_t = transpose(s, n, m); // [m x n]

        // S' W^{-1}: [m x n] * [n x n] = [m x n]
        let st_winv = mat_mul(&s_t, m, n, &w_inv, n);

        // S' W^{-1} S: [m x n] * [n x m] = [m x m]
        let st_winv_s = mat_mul(&st_winv, m, n, s, m);

        // (S' W^{-1} S)^{-1}: [m x m]
        let st_winv_s_inv = invert_matrix(&st_winv_s, m)?;

        // S (S' W^{-1} S)^{-1}: [n x m] * [m x m] = [n x m]
        let s_inv = mat_mul(s, n, m, &st_winv_s_inv, m);

        // P = S (S' W^{-1} S)^{-1} S' W^{-1}: [n x m] * [m x n] = [n x n]
        let p = mat_mul(&s_inv, n, m, &st_winv, n);

        // reconciled = P * base_forecasts
        Ok(mat_vec(&p, n, n, base_forecasts))
    }
}

// ---------------------------------------------------------------------------
// Section 6 -- ERM Reconciliation
// ---------------------------------------------------------------------------

/// ERM (Empirical Risk Minimization) reconciliation.
///
/// Learns reconciliation weights from validation data using cross-validation.
pub struct HtsErmReconciliation {
    /// Learned weight matrix G: [num_bottom x total_nodes].
    pub weight_matrix: Vec<f64>,
    /// Number of bottom-level nodes.
    pub num_bottom: usize,
    /// Total number of nodes.
    pub total_nodes: usize,
}

impl HtsErmReconciliation {
    /// Train ERM weights from base forecasts and actuals.
    ///
    /// `base_forecasts` is [T x n] row-major (T periods, n total nodes).
    /// `actuals` is [T x n] row-major.
    /// Uses ridge regression to learn G that minimizes ||S*G*y_hat - y_actual||^2.
    pub fn train(
        base_forecasts: &[f64],
        actuals: &[f64],
        num_periods: usize,
        hierarchy: &HtsHierarchy,
        regularization: f64,
    ) -> Result<Self> {
        let n = hierarchy.total_nodes;
        let m = hierarchy.num_bottom;

        if base_forecasts.len() != num_periods * n || actuals.len() != num_periods * n {
            return Err(TensorError::compute_error_simple(
                "HtsErm: forecast/actual dimensions mismatch".to_string(),
            ));
        }
        if num_periods < 2 {
            return Err(TensorError::compute_error_simple(
                "HtsErm: need at least 2 periods".to_string(),
            ));
        }

        // Extract bottom-level actuals: [T x m]
        let mut b_actual = vec![0.0; num_periods * m];
        for t in 0..num_periods {
            for (b_local, &b_idx) in hierarchy.bottom_indices.iter().enumerate() {
                b_actual[t * m + b_local] = actuals[t * n + b_idx];
            }
        }

        // Ridge regression: G = (Y_hat' Y_hat + lambda*I)^{-1} Y_hat' B_actual
        // where Y_hat is [T x n] base forecasts, B_actual is [T x m] bottom actuals.

        // Y_hat' Y_hat: [n x n]
        let yht = transpose(base_forecasts, num_periods, n); // [n x T]
        let yht_yh = mat_mul(&yht, n, num_periods, base_forecasts, n); // [n x n]

        // Add regularization
        let mut yht_yh_reg = yht_yh;
        for i in 0..n {
            yht_yh_reg[i * n + i] += regularization;
        }

        // Y_hat' B_actual: [n x T] * [T x m] = [n x m]
        let yht_ba = mat_mul(&yht, n, num_periods, &b_actual, m);

        // Solve for each column of G
        let yht_yh_inv = invert_matrix(&yht_yh_reg, n)?;
        let g = mat_mul(&yht_yh_inv, n, n, &yht_ba, m); // [n x m]

        // G is [n x m], we store as [m x n] (transposed) for reconciliation.
        let weight_matrix = transpose(&g, n, m); // [m x n]

        Ok(Self {
            weight_matrix,
            num_bottom: m,
            total_nodes: n,
        })
    }

    /// Reconcile forecasts using learned ERM weights.
    ///
    /// bottom_hat = G' * base_forecasts, then coherent = S * bottom_hat.
    pub fn reconcile(&self, base_forecasts: &[f64], hierarchy: &HtsHierarchy) -> Result<Vec<f64>> {
        if base_forecasts.len() != self.total_nodes {
            return Err(TensorError::compute_error_simple(format!(
                "HtsErm: expected {} forecasts, got {}",
                self.total_nodes,
                base_forecasts.len()
            )));
        }
        // bottom_hat = G * base_forecasts where G is [m x n]
        let bottom_hat = mat_vec(
            &self.weight_matrix,
            self.num_bottom,
            self.total_nodes,
            base_forecasts,
        );
        hierarchy.aggregate_bottom(&bottom_hat)
    }
}

// ---------------------------------------------------------------------------
// Section 7 -- Temporal Hierarchy
// ---------------------------------------------------------------------------

/// Temporal hierarchy for multi-frequency aggregation.
///
/// Given a base-frequency time series, aggregates to multiple temporal
/// granularities (e.g., daily -> weekly -> monthly) and builds the
/// temporal summing matrix for cross-temporal reconciliation.
#[derive(Debug, Clone)]
pub struct TemporalHierarchy {
    /// Aggregation factors (e.g., [1, 7, 30] for daily/weekly/monthly).
    pub frequencies: Vec<usize>,
    /// Temporal summing matrix: maps finest-level to all levels.
    /// Shape: [total_temporal_nodes x num_base_periods], row-major.
    pub temporal_s: Vec<f64>,
    /// Total number of temporal nodes across all frequencies.
    pub total_temporal: usize,
    /// Number of base-frequency periods.
    pub num_base: usize,
}

impl TemporalHierarchy {
    /// Build temporal hierarchy from aggregation frequencies.
    ///
    /// `base_length` is the number of base-frequency observations.
    /// `frequencies` are the aggregation factors in ascending order.
    /// The first entry should be 1 (base frequency).
    pub fn new(base_length: usize, frequencies: &[usize]) -> Result<Self> {
        if frequencies.is_empty() {
            return Err(TensorError::compute_error_simple(
                "TemporalHierarchy: frequencies must not be empty".to_string(),
            ));
        }
        if base_length == 0 {
            return Err(TensorError::compute_error_simple(
                "TemporalHierarchy: base_length must be > 0".to_string(),
            ));
        }

        let mut sorted_freqs: Vec<usize> = frequencies.to_vec();
        sorted_freqs.sort();
        sorted_freqs.dedup();

        // Count total temporal nodes.
        let mut total = 0;
        for &f in &sorted_freqs {
            if f == 0 {
                return Err(TensorError::compute_error_simple(
                    "TemporalHierarchy: frequency must be > 0".to_string(),
                ));
            }
            total += base_length / f;
        }

        // Build temporal summing matrix.
        // For each frequency level, each aggregated period sums `f` consecutive base periods.
        let num_base = base_length;
        let mut temporal_s = vec![0.0; total * num_base];
        let mut row = 0;
        for &f in &sorted_freqs {
            let num_agg = num_base / f;
            for k in 0..num_agg {
                let start = k * f;
                let end = ((k + 1) * f).min(num_base);
                for col in start..end {
                    temporal_s[row * num_base + col] = 1.0;
                }
                row += 1;
            }
        }

        Ok(Self {
            frequencies: sorted_freqs,
            temporal_s,
            total_temporal: total,
            num_base,
        })
    }

    /// Aggregate a base-frequency series to all temporal levels.
    pub fn aggregate(&self, series: &[f64]) -> Result<Vec<f64>> {
        if series.len() != self.num_base {
            return Err(TensorError::compute_error_simple(format!(
                "TemporalHierarchy::aggregate: expected {} values, got {}",
                self.num_base,
                series.len()
            )));
        }
        Ok(mat_vec(
            &self.temporal_s,
            self.total_temporal,
            self.num_base,
            series,
        ))
    }

    /// Cross-temporal reconciliation using MinTrace-style projection.
    ///
    /// Combines both cross-sectional hierarchy and temporal hierarchy.
    /// `base_forecasts_all_freq` is the concatenation of forecasts at all
    /// temporal frequencies for all cross-sectional nodes.
    pub fn cross_temporal_reconcile(
        &self,
        base_forecasts: &[f64],
        cross_hierarchy: &HtsHierarchy,
    ) -> Result<Vec<f64>> {
        let cs_n = cross_hierarchy.total_nodes;
        let t_n = self.total_temporal;
        let expected = cs_n * t_n;

        if base_forecasts.len() != expected {
            return Err(TensorError::compute_error_simple(format!(
                "cross_temporal: expected {} values, got {}",
                expected,
                base_forecasts.len()
            )));
        }

        // Kronecker product of cross-sectional S and temporal S.
        let cs_m = cross_hierarchy.num_bottom;
        let t_m = self.num_base;
        let cs_s = &cross_hierarchy.summing_matrix; // [cs_n x cs_m]
        let t_s = &self.temporal_s; // [t_n x t_m]

        // Kronecker: S_kron [cs_n*t_n x cs_m*t_m]
        let kron_rows = cs_n * t_n;
        let kron_cols = cs_m * t_m;
        let mut s_kron = vec![0.0; kron_rows * kron_cols];
        for i1 in 0..cs_n {
            for j1 in 0..cs_m {
                let cs_val = cs_s[i1 * cs_m + j1];
                for i2 in 0..t_n {
                    for j2 in 0..t_m {
                        let t_val = t_s[i2 * t_m + j2];
                        let row = i1 * t_n + i2;
                        let col = j1 * t_m + j2;
                        s_kron[row * kron_cols + col] = cs_val * t_val;
                    }
                }
            }
        }

        // Simple OLS reconciliation: P = S (S'S)^{-1} S'
        let s_t = transpose(&s_kron, kron_rows, kron_cols);
        let sts = mat_mul(&s_t, kron_cols, kron_rows, &s_kron, kron_cols);
        let sts_inv = invert_matrix(&sts, kron_cols)?;
        let s_sts_inv = mat_mul(&s_kron, kron_rows, kron_cols, &sts_inv, kron_cols);
        let p = mat_mul(&s_sts_inv, kron_rows, kron_cols, &s_t, kron_rows);

        Ok(mat_vec(&p, kron_rows, kron_rows, base_forecasts))
    }
}

// ---------------------------------------------------------------------------
// Section 8 -- Probabilistic Forecaster
// ---------------------------------------------------------------------------

/// Quantile forecast result.
#[derive(Debug, Clone)]
pub struct HtsQuantileForecast {
    /// Point forecasts (mean or median).
    pub point: Vec<f64>,
    /// Quantile levels (e.g., [0.025, 0.1, 0.5, 0.9, 0.975]).
    pub quantile_levels: Vec<f64>,
    /// Quantile forecasts: [horizon x num_quantiles], row-major.
    pub quantiles: Vec<f64>,
    /// Forecast horizon.
    pub horizon: usize,
}

/// Probabilistic forecaster with quantile regression and prediction intervals.
pub struct ProbabilisticForecaster {
    /// Quantile levels to produce.
    pub quantile_levels: Vec<f64>,
}

impl ProbabilisticForecaster {
    /// Create with specified quantile levels.
    pub fn new(quantile_levels: Vec<f64>) -> Result<Self> {
        for &q in &quantile_levels {
            if !(0.0..=1.0).contains(&q) {
                return Err(TensorError::compute_error_simple(format!(
                    "ProbabilisticForecaster: quantile {} not in [0,1]",
                    q
                )));
            }
        }
        Ok(Self { quantile_levels })
    }

    /// Create with default levels: 2.5%, 10%, 25%, 50%, 75%, 90%, 97.5%.
    pub fn default_levels() -> Self {
        Self {
            quantile_levels: vec![0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
        }
    }

    /// Pinball (quantile) loss for a single quantile level.
    #[inline]
    pub fn pinball_loss(y: f64, q_hat: f64, tau: f64) -> f64 {
        let err = y - q_hat;
        if err >= 0.0 {
            tau * err
        } else {
            (tau - 1.0) * err
        }
    }

    /// Forecast using Gaussian assumption (mean + std).
    ///
    /// Given point forecast (mean) and standard deviation for each horizon step,
    /// produces quantile forecasts via the inverse normal CDF approximation.
    pub fn forecast_gaussian(&self, means: &[f64], stds: &[f64]) -> Result<HtsQuantileForecast> {
        let horizon = means.len();
        if stds.len() != horizon {
            return Err(TensorError::compute_error_simple(
                "ProbabilisticForecaster: means and stds must have same length".to_string(),
            ));
        }
        let nq = self.quantile_levels.len();
        let mut quantiles = vec![0.0; horizon * nq];

        for h in 0..horizon {
            for (qi, &tau) in self.quantile_levels.iter().enumerate() {
                let z = Self::normal_quantile(tau);
                quantiles[h * nq + qi] = means[h] + stds[h] * z;
            }
        }

        Ok(HtsQuantileForecast {
            point: means.to_vec(),
            quantile_levels: self.quantile_levels.clone(),
            quantiles,
            horizon,
        })
    }

    /// Approximate inverse normal CDF (Beasley-Springer-Moro algorithm).
    fn normal_quantile(p: f64) -> f64 {
        if p <= 0.0 {
            return -8.0;
        }
        if p >= 1.0 {
            return 8.0;
        }
        // Rational approximation (Abramowitz & Stegun 26.2.23)
        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        let num = c0 + c1 * t + c2 * t * t;
        let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
        let approx = t - num / den;
        if p < 0.5 {
            -approx
        } else {
            approx
        }
    }

    /// Compute CRPS (Continuous Ranked Probability Score) for Gaussian forecasts.
    ///
    /// CRPS_Gaussian(mu, sigma, y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    /// where z = (y - mu) / sigma.
    pub fn crps_gaussian(y: f64, mu: f64, sigma: f64) -> f64 {
        if sigma <= 0.0 {
            return (y - mu).abs();
        }
        let z = (y - mu) / sigma;
        let phi_z = Self::standard_normal_pdf(z);
        let big_phi_z = Self::standard_normal_cdf(z);
        sigma * (z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z - 1.0 / std::f64::consts::PI.sqrt())
    }

    /// Standard normal PDF.
    #[inline]
    fn standard_normal_pdf(x: f64) -> f64 {
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        inv_sqrt_2pi * (-0.5 * x * x).exp()
    }

    /// Standard normal CDF (approximation).
    #[inline]
    fn standard_normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
    }
}

/// Approximation to the error function erf(x) (Abramowitz & Stegun 7.1.26).
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let t = x.abs();
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let u = 1.0 / (1.0 + p * t);
    let poly = u * (a1 + u * (a2 + u * (a3 + u * (a4 + u * a5))));
    sign * (1.0 - poly * (-t * t).exp())
}

// ---------------------------------------------------------------------------
// Section 9 -- HtsForecaster (ETS / Holt-Winters / Theta)
// ---------------------------------------------------------------------------

/// ETS model type.
#[derive(Debug, Clone, Copy)]
pub enum HtsEtsType {
    /// Simple exponential smoothing (no trend, no season).
    Ses,
    /// Holt's linear method (additive trend, no season).
    HoltLinear,
    /// Damped Holt's method (damped additive trend, no season).
    HoltDamped,
    /// Holt-Winters additive seasonality.
    HoltWintersAdd { period: usize },
    /// Holt-Winters multiplicative seasonality.
    HoltWintersMul { period: usize },
}

/// ETS forecast result.
#[derive(Debug, Clone)]
pub struct HtsForecastResult {
    /// Point forecasts.
    pub point: Vec<f64>,
    /// Fitted values (in-sample one-step-ahead).
    pub fitted: Vec<f64>,
    /// Residuals (actual - fitted).
    pub residuals: Vec<f64>,
    /// Estimated residual standard deviation.
    pub residual_std: f64,
}

/// Base forecasting models for hierarchical time series.
pub struct HtsForecaster;

impl HtsForecaster {
    /// Forecast using ETS (exponential smoothing) models.
    ///
    /// Parameters alpha, beta, gamma, phi are smoothing parameters in (0,1).
    pub fn forecast_ets(
        series: &[f64],
        horizon: usize,
        ets_type: HtsEtsType,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
    ) -> Result<HtsForecastResult> {
        if series.len() < 3 {
            return Err(TensorError::compute_error_simple(
                "HtsForecaster: need at least 3 observations".to_string(),
            ));
        }
        let n = series.len();

        match ets_type {
            HtsEtsType::Ses => {
                // Simple exponential smoothing
                let mut level = series[0];
                let mut fitted = vec![0.0; n];
                let mut residuals = vec![0.0; n];
                fitted[0] = level;
                residuals[0] = series[0] - level;
                for t in 1..n {
                    let forecast = level;
                    fitted[t] = forecast;
                    residuals[t] = series[t] - forecast;
                    level = alpha * series[t] + (1.0 - alpha) * level;
                }
                let point: Vec<f64> = (0..horizon).map(|_| level).collect();
                let residual_std = Self::compute_residual_std(&residuals);
                Ok(HtsForecastResult {
                    point,
                    fitted,
                    residuals,
                    residual_std,
                })
            }
            HtsEtsType::HoltLinear => {
                let b = beta.unwrap_or(0.1);
                let mut level = series[0];
                let mut trend = if n > 1 { series[1] - series[0] } else { 0.0 };
                let mut fitted = vec![0.0; n];
                let mut residuals = vec![0.0; n];
                fitted[0] = level;
                residuals[0] = series[0] - level;
                for t in 1..n {
                    let forecast = level + trend;
                    fitted[t] = forecast;
                    residuals[t] = series[t] - forecast;
                    let new_level = alpha * series[t] + (1.0 - alpha) * (level + trend);
                    trend = b * (new_level - level) + (1.0 - b) * trend;
                    level = new_level;
                }
                let point: Vec<f64> = (1..=horizon).map(|h| level + h as f64 * trend).collect();
                let residual_std = Self::compute_residual_std(&residuals);
                Ok(HtsForecastResult {
                    point,
                    fitted,
                    residuals,
                    residual_std,
                })
            }
            HtsEtsType::HoltDamped => {
                let b = beta.unwrap_or(0.1);
                let p = phi.unwrap_or(0.9);
                let mut level = series[0];
                let mut trend = if n > 1 { series[1] - series[0] } else { 0.0 };
                let mut fitted = vec![0.0; n];
                let mut residuals = vec![0.0; n];
                fitted[0] = level;
                residuals[0] = series[0] - level;
                for t in 1..n {
                    let forecast = level + p * trend;
                    fitted[t] = forecast;
                    residuals[t] = series[t] - forecast;
                    let new_level = alpha * series[t] + (1.0 - alpha) * (level + p * trend);
                    trend = b * (new_level - level) + (1.0 - b) * p * trend;
                    level = new_level;
                }
                // h-step: level + (phi + phi^2 + ... + phi^h) * trend
                let point: Vec<f64> = (1..=horizon)
                    .map(|h| {
                        let mut phi_sum = 0.0;
                        let mut phi_pow = p;
                        for _ in 0..h {
                            phi_sum += phi_pow;
                            phi_pow *= p;
                        }
                        level + phi_sum * trend
                    })
                    .collect();
                let residual_std = Self::compute_residual_std(&residuals);
                Ok(HtsForecastResult {
                    point,
                    fitted,
                    residuals,
                    residual_std,
                })
            }
            HtsEtsType::HoltWintersAdd { period } => {
                if period < 2 || n < 2 * period {
                    return Err(TensorError::compute_error_simple(
                        "HoltWintersAdd: need period >= 2 and n >= 2*period".to_string(),
                    ));
                }
                let b = beta.unwrap_or(0.1);
                let g = gamma.unwrap_or(0.1);

                // Initialize seasonal indices from first full cycle.
                let cycle_mean: f64 = series[..period].iter().sum::<f64>() / period as f64;
                let mut season: Vec<f64> =
                    series[..period].iter().map(|&v| v - cycle_mean).collect();
                let mut level = cycle_mean;
                let mut trend = {
                    let m1: f64 = series[..period].iter().sum::<f64>() / period as f64;
                    let m2: f64 = series[period..2 * period].iter().sum::<f64>() / period as f64;
                    (m2 - m1) / period as f64
                };

                let mut fitted = vec![0.0; n];
                let mut residuals = vec![0.0; n];
                for t in 0..n {
                    let s_idx = t % period;
                    let forecast = level + trend + season[s_idx];
                    fitted[t] = forecast;
                    residuals[t] = series[t] - forecast;
                    if t < n - 1 {
                        let new_level =
                            alpha * (series[t] - season[s_idx]) + (1.0 - alpha) * (level + trend);
                        let new_trend = b * (new_level - level) + (1.0 - b) * trend;
                        season[s_idx] = g * (series[t] - new_level) + (1.0 - g) * season[s_idx];
                        level = new_level;
                        trend = new_trend;
                    }
                }

                let point: Vec<f64> = (0..horizon)
                    .map(|h| {
                        let s_idx = (n + h) % period;
                        level + (h + 1) as f64 * trend + season[s_idx]
                    })
                    .collect();
                let residual_std = Self::compute_residual_std(&residuals);
                Ok(HtsForecastResult {
                    point,
                    fitted,
                    residuals,
                    residual_std,
                })
            }
            HtsEtsType::HoltWintersMul { period } => {
                if period < 2 || n < 2 * period {
                    return Err(TensorError::compute_error_simple(
                        "HoltWintersMul: need period >= 2 and n >= 2*period".to_string(),
                    ));
                }
                let b = beta.unwrap_or(0.1);
                let g = gamma.unwrap_or(0.1);

                let cycle_mean: f64 = series[..period].iter().sum::<f64>() / period as f64;
                if cycle_mean.abs() < 1e-14 {
                    return Err(TensorError::compute_error_simple(
                        "HoltWintersMul: cycle mean too close to zero".to_string(),
                    ));
                }
                let mut season: Vec<f64> =
                    series[..period].iter().map(|&v| v / cycle_mean).collect();
                let mut level = cycle_mean;
                let mut trend = {
                    let m1: f64 = series[..period].iter().sum::<f64>() / period as f64;
                    let m2: f64 = series[period..2 * period].iter().sum::<f64>() / period as f64;
                    (m2 - m1) / period as f64
                };

                let mut fitted = vec![0.0; n];
                let mut residuals = vec![0.0; n];
                for t in 0..n {
                    let s_idx = t % period;
                    let s_val = if season[s_idx].abs() < 1e-14 {
                        1.0
                    } else {
                        season[s_idx]
                    };
                    let forecast = (level + trend) * s_val;
                    fitted[t] = forecast;
                    residuals[t] = series[t] - forecast;
                    if t < n - 1 {
                        let new_level =
                            alpha * (series[t] / s_val) + (1.0 - alpha) * (level + trend);
                        let new_trend = b * (new_level - level) + (1.0 - b) * trend;
                        season[s_idx] = if new_level.abs() < 1e-14 {
                            season[s_idx]
                        } else {
                            g * (series[t] / new_level) + (1.0 - g) * season[s_idx]
                        };
                        level = new_level;
                        trend = new_trend;
                    }
                }

                let point: Vec<f64> = (0..horizon)
                    .map(|h| {
                        let s_idx = (n + h) % period;
                        (level + (h + 1) as f64 * trend) * season[s_idx]
                    })
                    .collect();
                let residual_std = Self::compute_residual_std(&residuals);
                Ok(HtsForecastResult {
                    point,
                    fitted,
                    residuals,
                    residual_std,
                })
            }
        }
    }

    /// Theta method (Assimakopoulos & Nikolopoulos 2000).
    ///
    /// Decomposes series into two theta-lines (theta=0 linear trend + theta=2 SES),
    /// then averages the forecasts.
    pub fn forecast_theta(series: &[f64], horizon: usize, alpha: f64) -> Result<HtsForecastResult> {
        if series.len() < 3 {
            return Err(TensorError::compute_error_simple(
                "Theta: need at least 3 observations".to_string(),
            ));
        }
        let n = series.len();

        // Theta=0 line: linear regression y = a + b*t
        let t_mean = (n - 1) as f64 / 2.0;
        let y_mean: f64 = series.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in series.iter().enumerate() {
            let t = i as f64 - t_mean;
            num += t * (y - y_mean);
            den += t * t;
        }
        let slope = if den.abs() > 1e-14 { num / den } else { 0.0 };
        let intercept = y_mean - slope * t_mean;

        // Linear trend forecasts
        let linear_forecast: Vec<f64> = (0..horizon)
            .map(|h| intercept + slope * (n + h) as f64)
            .collect();

        // Theta=2 line: 2*y - theta0 = 2*y - (a + b*t)
        let theta2: Vec<f64> = (0..n)
            .map(|i| 2.0 * series[i] - (intercept + slope * i as f64))
            .collect();

        // SES on theta2 line
        let mut level = theta2[0];
        let mut fitted = vec![0.0; n];
        let mut residuals = vec![0.0; n];
        for t in 0..n {
            fitted[t] = (intercept + slope * t as f64 + level) / 2.0;
            residuals[t] = series[t] - fitted[t];
            level = alpha * theta2[t] + (1.0 - alpha) * level;
        }

        // Combined forecast: average of theta=0 and theta=2 forecasts.
        let point: Vec<f64> = (0..horizon)
            .map(|h| (linear_forecast[h] + level) / 2.0)
            .collect();

        let residual_std = Self::compute_residual_std(&residuals);
        Ok(HtsForecastResult {
            point,
            fitted,
            residuals,
            residual_std,
        })
    }

    /// Compute residual standard deviation.
    fn compute_residual_std(residuals: &[f64]) -> f64 {
        let n = residuals.len();
        if n < 2 {
            return 0.0;
        }
        let mean: f64 = residuals.iter().sum::<f64>() / n as f64;
        let var: f64 = residuals
            .iter()
            .map(|&r| (r - mean) * (r - mean))
            .sum::<f64>()
            / (n - 1) as f64;
        var.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Section 10 -- Metrics & Report
// ---------------------------------------------------------------------------

/// Hierarchical forecast evaluation metrics.
pub struct HtsMetrics;

impl HtsMetrics {
    /// MASE: Mean Absolute Scaled Error.
    ///
    /// MASE = mean(|e_t|) / mean(|y_t - y_{t-1}|).
    pub fn mase(actuals: &[f64], forecasts: &[f64], in_sample: &[f64]) -> Result<f64> {
        if actuals.len() != forecasts.len() {
            return Err(TensorError::compute_error_simple(
                "MASE: actuals and forecasts length mismatch".to_string(),
            ));
        }
        if in_sample.len() < 2 {
            return Err(TensorError::compute_error_simple(
                "MASE: in_sample needs at least 2 values".to_string(),
            ));
        }
        let naive_mae: f64 = in_sample
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>()
            / (in_sample.len() - 1) as f64;
        if naive_mae < 1e-14 {
            return Err(TensorError::compute_error_simple(
                "MASE: naive MAE is zero (constant series)".to_string(),
            ));
        }
        let mae: f64 = actuals
            .iter()
            .zip(forecasts.iter())
            .map(|(a, f)| (a - f).abs())
            .sum::<f64>()
            / actuals.len() as f64;
        Ok(mae / naive_mae)
    }

    /// RMSSE: Root Mean Squared Scaled Error.
    pub fn rmsse(actuals: &[f64], forecasts: &[f64], in_sample: &[f64]) -> Result<f64> {
        if actuals.len() != forecasts.len() {
            return Err(TensorError::compute_error_simple(
                "RMSSE: actuals and forecasts length mismatch".to_string(),
            ));
        }
        if in_sample.len() < 2 {
            return Err(TensorError::compute_error_simple(
                "RMSSE: in_sample needs at least 2 values".to_string(),
            ));
        }
        let naive_mse: f64 = in_sample
            .windows(2)
            .map(|w| {
                let d = w[1] - w[0];
                d * d
            })
            .sum::<f64>()
            / (in_sample.len() - 1) as f64;
        if naive_mse < 1e-14 {
            return Err(TensorError::compute_error_simple(
                "RMSSE: naive MSE is zero (constant series)".to_string(),
            ));
        }
        let mse: f64 = actuals
            .iter()
            .zip(forecasts.iter())
            .map(|(a, f)| {
                let d = a - f;
                d * d
            })
            .sum::<f64>()
            / actuals.len() as f64;
        Ok((mse / naive_mse).sqrt())
    }

    /// Coherence error: ||y_hat - S * b_hat|| / ||y_hat||.
    ///
    /// Measures how far a set of forecasts deviates from being coherent.
    pub fn coherence_error(forecasts: &[f64], hierarchy: &HtsHierarchy) -> Result<f64> {
        if forecasts.len() != hierarchy.total_nodes {
            return Err(TensorError::compute_error_simple(
                "coherence_error: forecast length mismatch".to_string(),
            ));
        }
        // Extract bottom-level and reconstruct via S.
        let bottom: Vec<f64> = hierarchy
            .bottom_indices
            .iter()
            .map(|&i| forecasts[i])
            .collect();
        let coherent = hierarchy.aggregate_bottom(&bottom)?;
        let norm_y: f64 = forecasts.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm_y < 1e-14 {
            return Ok(0.0);
        }
        let diff_norm: f64 = forecasts
            .iter()
            .zip(coherent.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        Ok(diff_norm / norm_y)
    }

    /// Energy score (multivariate CRPS).
    ///
    /// ES = E||X - y|| - 0.5 * E||X - X'||
    /// Approximated from ensemble samples.
    pub fn energy_score(actuals: &[f64], samples: &[Vec<f64>]) -> Result<f64> {
        if samples.is_empty() {
            return Err(TensorError::compute_error_simple(
                "energy_score: need at least 1 sample".to_string(),
            ));
        }
        let d = actuals.len();
        for s in samples {
            if s.len() != d {
                return Err(TensorError::compute_error_simple(
                    "energy_score: sample dimension mismatch".to_string(),
                ));
            }
        }
        let k = samples.len() as f64;

        // E||X - y||
        let term1: f64 = samples
            .iter()
            .map(|s| {
                s.iter()
                    .zip(actuals.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum::<f64>()
            / k;

        // E||X - X'||
        let mut term2 = 0.0;
        let ns = samples.len();
        for i in 0..ns {
            for j in (i + 1)..ns {
                let dist: f64 = samples[i]
                    .iter()
                    .zip(samples[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                term2 += dist;
            }
        }
        if ns > 1 {
            term2 = 2.0 * term2 / (ns as f64 * (ns as f64 - 1.0));
        }

        Ok(term1 - 0.5 * term2)
    }
}

/// Full hierarchical forecast evaluation report.
#[derive(Debug, Clone)]
pub struct HtsReport {
    /// MASE per hierarchy level.
    pub mase_by_level: Vec<f64>,
    /// RMSSE per hierarchy level.
    pub rmsse_by_level: Vec<f64>,
    /// Overall coherence error.
    pub coherence_error: f64,
    /// Overall CRPS (if probabilistic forecasts available).
    pub crps: Option<f64>,
    /// Overall energy score (if ensemble available).
    pub energy_score: Option<f64>,
}

impl HtsReport {
    /// Generate a full evaluation report.
    ///
    /// `actuals_by_level` and `forecasts_by_level` are indexed by level.
    /// Each entry is a flat vector of (actual, forecast) pairs for that level.
    /// `in_sample_by_level` provides in-sample data for scaling.
    pub fn generate(
        actuals_by_level: &[Vec<f64>],
        forecasts_by_level: &[Vec<f64>],
        in_sample_by_level: &[Vec<f64>],
        coherence_err: f64,
    ) -> Result<Self> {
        let num_levels = actuals_by_level.len();
        if num_levels != forecasts_by_level.len() || num_levels != in_sample_by_level.len() {
            return Err(TensorError::compute_error_simple(
                "HtsReport: level count mismatch".to_string(),
            ));
        }
        let mut mase_by_level = Vec::with_capacity(num_levels);
        let mut rmsse_by_level = Vec::with_capacity(num_levels);
        for l in 0..num_levels {
            let m = HtsMetrics::mase(
                &actuals_by_level[l],
                &forecasts_by_level[l],
                &in_sample_by_level[l],
            )?;
            let r = HtsMetrics::rmsse(
                &actuals_by_level[l],
                &forecasts_by_level[l],
                &in_sample_by_level[l],
            )?;
            mase_by_level.push(m);
            rmsse_by_level.push(r);
        }

        Ok(Self {
            mase_by_level,
            rmsse_by_level,
            coherence_error: coherence_err,
            crps: None,
            energy_score: None,
        })
    }
}

#[cfg(test)]
mod tests;
