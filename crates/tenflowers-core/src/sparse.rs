//! Sparse Tensor Support
//!
//! This module provides production-grade sparse tensor formats for efficient
//! storage and computation of tensors with many zero elements.
//!
//! ## Formats
//!
//! - [`CooTensor`]: Coordinate (COO) format for N-dimensional sparse tensors.
//!   Stores non-zero elements as (indices, value) pairs. Flexible for construction
//!   and format conversion.
//!
//! - [`CsrTensor`]: Compressed Sparse Row (CSR) format for 2D sparse matrices.
//!   Efficient for row-oriented operations such as SpMV and SpMM.
//!
//! ## Operations
//!
//! - [`coo_add`]: Element-wise addition of two COO tensors.
//! - [`coo_scale`]: Scalar multiplication of a COO tensor.
//! - [`csr_add`]: Element-wise addition of two CSR matrices.

use crate::error::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// CooTensor
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse tensor in COO (Coordinate) format.
///
/// Stores non-zero values as `(indices, value)` pairs.
///
/// * `indices` – `nnz` rows, each row has `ndim` coordinate values.
/// * `values`  – `nnz` non-zero values corresponding to each index row.
/// * `shape`   – the dense shape of the tensor.
/// * `nnz`     – number of stored non-zero entries (may include explicit zeros
///               until [`coalesce`](Self::coalesce) is called).
#[derive(Debug, Clone)]
pub struct CooTensor {
    /// Indices: `indices[i]` are the N-dimensional coordinates of the i-th non-zero.
    pub indices: Vec<Vec<usize>>,
    /// Non-zero values.
    pub values: Vec<f32>,
    /// Dense shape of the tensor.
    pub shape: Vec<usize>,
    /// Number of stored entries.
    pub nnz: usize,
}

impl CooTensor {
    /// Create a new COO tensor from raw (indices, values, shape) data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `indices.len() != values.len()`
    /// - Any index row has the wrong dimensionality.
    /// - Any coordinate is out of bounds for the tensor shape.
    pub fn new(
        indices: Vec<Vec<usize>>,
        values: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let nnz = values.len();
        if indices.len() != nnz {
            return Err(TensorError::invalid_argument_op(
                "CooTensor::new",
                &format!(
                    "indices length {} does not match values length {}",
                    indices.len(),
                    nnz
                ),
            ));
        }
        let ndim = shape.len();
        for (i, idx) in indices.iter().enumerate() {
            if idx.len() != ndim {
                return Err(TensorError::invalid_argument_op(
                    "CooTensor::new",
                    &format!(
                        "index row {} has length {} but tensor ndim is {}",
                        i,
                        idx.len(),
                        ndim
                    ),
                ));
            }
            for (dim, (&coord, &dim_size)) in idx.iter().zip(shape.iter()).enumerate() {
                if coord >= dim_size {
                    return Err(TensorError::invalid_argument_op(
                        "CooTensor::new",
                        &format!(
                            "index row {}: coordinate {} in dimension {} is out of bounds \
                             (shape dimension is {})",
                            i, coord, dim, dim_size
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            indices,
            values,
            shape,
            nnz,
        })
    }

    /// Create an empty (all-zeros) sparse tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            shape,
            nnz: 0,
        }
    }

    /// Build a COO tensor from a dense flat array in row-major order.
    ///
    /// Elements whose absolute value is `> threshold` are considered non-zero.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() != shape.iter().product()`.
    pub fn from_dense(data: &[f32], shape: &[usize], threshold: f32) -> Result<Self> {
        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(TensorError::invalid_argument_op(
                "CooTensor::from_dense",
                &format!(
                    "data length {} does not match shape product {}",
                    data.len(),
                    total
                ),
            ));
        }

        let ndim = shape.len();
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (flat_idx, &val) in data.iter().enumerate() {
            if val.abs() > threshold {
                // Convert flat index to N-D coordinates (row-major)
                let mut coords = vec![0usize; ndim];
                let mut remaining = flat_idx;
                for d in (0..ndim).rev() {
                    coords[d] = remaining % shape[d];
                    remaining /= shape[d];
                }
                indices.push(coords);
                values.push(val);
            }
        }

        let nnz = values.len();
        Ok(Self {
            indices,
            values,
            shape: shape.to_vec(),
            nnz,
        })
    }

    /// Convert this COO tensor to a dense flat array (row-major) and its shape.
    ///
    /// Duplicate entries are accumulated (summed).
    pub fn to_dense(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        let total: usize = self.shape.iter().product();
        let mut data = vec![0.0f32; total];
        let ndim = self.shape.len();

        for (idx_row, &val) in self.indices.iter().zip(self.values.iter()) {
            // Convert N-D coordinates to flat row-major index
            let mut flat = 0usize;
            let mut stride = 1usize;
            for d in (0..ndim).rev() {
                flat += idx_row[d] * stride;
                stride *= self.shape[d];
            }
            data[flat] += val;
        }

        Ok((data, self.shape.clone()))
    }

    /// Fraction of elements that are zero in the dense representation.
    ///
    /// Returns `1.0` when the total number of elements is zero.
    pub fn sparsity(&self) -> f32 {
        let total: usize = self.shape.iter().product();
        if total == 0 {
            return 1.0;
        }
        let nnz_clamped = self.nnz.min(total);
        (total - nnz_clamped) as f32 / total as f32
    }

    /// Convert a 2-D COO tensor to CSR format.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2-D.
    pub fn to_csr(&self) -> Result<CsrTensor> {
        if self.shape.len() != 2 {
            return Err(TensorError::invalid_argument_op(
                "CooTensor::to_csr",
                &format!(
                    "to_csr requires a 2-D tensor, got {}-D",
                    self.shape.len()
                ),
            ));
        }

        let nrows = self.shape[0];
        let ncols = self.shape[1];

        // Coalesce first (works on a clone to keep self immutable)
        let mut coalesced = self.clone();
        coalesced.coalesce();

        // Sort by (row, col) — coalesce already sorts lexicographically
        let mut entries: Vec<(usize, usize, f32)> = coalesced
            .indices
            .iter()
            .zip(coalesced.values.iter())
            .map(|(idx, &val)| (idx[0], idx[1], val))
            .collect();
        entries.sort_by_key(|&(r, c, _)| (r, c));

        let nnz = entries.len();
        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for &(row, col, val) in &entries {
            row_ptr[row + 1] += 1;
            col_indices.push(col);
            values.push(val);
        }

        // Prefix-sum the row pointers
        for r in 0..nrows {
            row_ptr[r + 1] += row_ptr[r];
        }

        CsrTensor::new(row_ptr, col_indices, values, nrows, ncols)
    }

    /// Return the value stored at the given N-D coordinates, or `0.0` if absent.
    ///
    /// Returns `0.0` without error when the index length does not match ndim,
    /// matching the semantics of "not present".
    pub fn get(&self, index: &[usize]) -> f32 {
        if index.len() != self.shape.len() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for (idx_row, &val) in self.indices.iter().zip(self.values.iter()) {
            if idx_row.as_slice() == index {
                sum += val;
            }
        }
        sum
    }

    /// Sort entries lexicographically, sum duplicate coordinates, and remove
    /// near-zero entries (absolute value < 1e-9).
    pub fn coalesce(&mut self) {
        if self.nnz == 0 {
            return;
        }

        // Pair up indices and values, then sort lexicographically
        let mut pairs: Vec<(Vec<usize>, f32)> = self
            .indices
            .drain(..)
            .zip(self.values.drain(..))
            .collect();

        pairs.sort_by(|(a, _), (b, _)| a.cmp(b));

        // Merge duplicates
        let mut merged: Vec<(Vec<usize>, f32)> = Vec::with_capacity(pairs.len());
        for (idx, val) in pairs {
            if let Some(last) = merged.last_mut() {
                if last.0 == idx {
                    last.1 += val;
                    continue;
                }
            }
            merged.push((idx, val));
        }

        // Remove near-zeros
        merged.retain(|(_, v)| v.abs() >= 1e-9);

        self.nnz = merged.len();
        for (idx, val) in merged {
            self.indices.push(idx);
            self.values.push(val);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CsrTensor
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse matrix in CSR (Compressed Sparse Row) format — 2-D only.
///
/// * `row_ptr[i]..row_ptr[i+1]` is the range in `col_indices` / `values`
///   belonging to row `i`.
/// * `col_indices[k]` is the column of the k-th non-zero.
/// * `values[k]` is the value of the k-th non-zero.
#[derive(Debug, Clone)]
pub struct CsrTensor {
    /// Row pointers, length `nrows + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices of each non-zero, length `nnz`.
    pub col_indices: Vec<usize>,
    /// Non-zero values, length `nnz`.
    pub values: Vec<f32>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl CsrTensor {
    /// Create a CSR tensor from raw data.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the following invariants are violated:
    /// - `row_ptr.len() != nrows + 1`
    /// - `col_indices.len() != values.len()`
    /// - `*row_ptr.last() != values.len()`
    /// - Any column index `>= ncols`
    pub fn new(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        if row_ptr.len() != nrows + 1 {
            return Err(TensorError::invalid_argument_op(
                "CsrTensor::new",
                &format!(
                    "row_ptr length {} must be nrows+1 = {}",
                    row_ptr.len(),
                    nrows + 1
                ),
            ));
        }
        if col_indices.len() != values.len() {
            return Err(TensorError::invalid_argument_op(
                "CsrTensor::new",
                &format!(
                    "col_indices length {} does not match values length {}",
                    col_indices.len(),
                    values.len()
                ),
            ));
        }
        let nnz = values.len();
        let last_ptr = row_ptr
            .last()
            .copied()
            .unwrap_or(0);
        if last_ptr != nnz {
            return Err(TensorError::invalid_argument_op(
                "CsrTensor::new",
                &format!(
                    "last row_ptr value {} does not match nnz {}",
                    last_ptr, nnz
                ),
            ));
        }
        for (k, &col) in col_indices.iter().enumerate() {
            if col >= ncols {
                return Err(TensorError::invalid_argument_op(
                    "CsrTensor::new",
                    &format!(
                        "col_indices[{}] = {} is out of bounds (ncols = {})",
                        k, col, ncols
                    ),
                ));
            }
        }
        Ok(Self {
            row_ptr,
            col_indices,
            values,
            nrows,
            ncols,
        })
    }

    /// Build a CSR matrix from a dense row-major flat array.
    ///
    /// Elements with `|val| > threshold` are stored as non-zeros.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() != nrows * ncols`.
    pub fn from_dense(
        data: &[f32],
        nrows: usize,
        ncols: usize,
        threshold: f32,
    ) -> Result<Self> {
        if data.len() != nrows * ncols {
            return Err(TensorError::invalid_argument_op(
                "CsrTensor::from_dense",
                &format!(
                    "data length {} does not match nrows*ncols = {}",
                    data.len(),
                    nrows * ncols
                ),
            ));
        }

        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..nrows {
            for col in 0..ncols {
                let val = data[row * ncols + col];
                if val.abs() > threshold {
                    col_indices.push(col);
                    values.push(val);
                }
            }
            row_ptr[row + 1] = col_indices.len();
        }

        Ok(Self {
            row_ptr,
            col_indices,
            values,
            nrows,
            ncols,
        })
    }

    /// Convert this CSR matrix to a dense row-major flat array.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut data = vec![0.0f32; self.nrows * self.ncols];
        for row in 0..self.nrows {
            for k in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_indices[k];
                data[row * self.ncols + col] += self.values[k];
            }
        }
        data
    }

    /// Convert to COO format.
    pub fn to_coo(&self) -> CooTensor {
        let nnz = self.nnz();
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for row in 0..self.nrows {
            for k in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_indices[k];
                indices.push(vec![row, col]);
                values.push(self.values[k]);
            }
        }

        CooTensor {
            indices,
            values,
            shape: vec![self.nrows, self.ncols],
            nnz,
        }
    }

    /// Fraction of elements that are zero.
    pub fn sparsity(&self) -> f32 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 1.0;
        }
        let stored = self.nnz().min(total);
        (total - stored) as f32 / total as f32
    }

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector product: `self * x → result`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x.len() != ncols`.
    pub fn spmv(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.ncols {
            return Err(TensorError::invalid_argument_op(
                "CsrTensor::spmv",
                &format!(
                    "vector length {} does not match ncols {}",
                    x.len(),
                    self.ncols
                ),
            ));
        }

        let mut result = vec![0.0f32; self.nrows];
        for row in 0..self.nrows {
            let mut acc = 0.0f32;
            for k in self.row_ptr[row]..self.row_ptr[row + 1] {
                acc += self.values[k] * x[self.col_indices[k]];
            }
            result[row] = acc;
        }
        Ok(result)
    }

    /// Sparse matrix × dense matrix product: `self [nrows, ncols] * B [ncols, k] → C [nrows, k]`.
    ///
    /// `b` is stored in row-major order with `ncols` rows and `k` columns.
    ///
    /// # Errors
    ///
    /// Returns an error if `b.len() != ncols * k`.
    pub fn spmm(&self, b: &[f32], k: usize) -> Result<Vec<f32>> {
        if b.len() != self.ncols * k {
            return Err(TensorError::invalid_argument_op(
                "CsrTensor::spmm",
                &format!(
                    "B matrix length {} does not match ncols*k = {}",
                    b.len(),
                    self.ncols * k
                ),
            ));
        }

        let mut result = vec![0.0f32; self.nrows * k];
        for row in 0..self.nrows {
            for nz in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_indices[nz];
                let a_val = self.values[nz];
                // result[row, :] += a_val * b[col, :]
                for j in 0..k {
                    result[row * k + j] += a_val * b[col * k + j];
                }
            }
        }
        Ok(result)
    }

    /// Get the value at `(row, col)`, returning `0.0` if not stored.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        if row >= self.nrows || col >= self.ncols {
            return 0.0;
        }
        for k in self.row_ptr[row]..self.row_ptr[row + 1] {
            if self.col_indices[k] == col {
                return self.values[k];
            }
        }
        0.0
    }

    /// Return the transpose of this CSR matrix as a new CSR tensor.
    ///
    /// The result has shape `[ncols, nrows]`.
    pub fn transpose(&self) -> CsrTensor {
        let new_nrows = self.ncols;
        let new_ncols = self.nrows;
        let nnz = self.nnz();

        // Count non-zeros per column of self (= row of transpose)
        let mut new_row_counts = vec![0usize; new_nrows];
        for &col in &self.col_indices {
            new_row_counts[col] += 1;
        }

        // Build row_ptr for the transposed matrix
        let mut new_row_ptr = vec![0usize; new_nrows + 1];
        for r in 0..new_nrows {
            new_row_ptr[r + 1] = new_row_ptr[r] + new_row_counts[r];
        }

        // Fill in col_indices and values for the transposed matrix
        let mut new_col_indices = vec![0usize; nnz];
        let mut new_values = vec![0.0f32; nnz];
        // Position cursor for each new row
        let mut pos = new_row_ptr[..new_nrows].to_vec();

        for old_row in 0..self.nrows {
            for k in self.row_ptr[old_row]..self.row_ptr[old_row + 1] {
                let old_col = self.col_indices[k];
                let val = self.values[k];
                // In the transpose: new_row = old_col, new_col = old_row
                let dest = pos[old_col];
                new_col_indices[dest] = old_row;
                new_values[dest] = val;
                pos[old_col] += 1;
            }
        }

        // The transpose's col_indices within each row are in the order we
        // inserted them (i.e., by old_row, which is sorted). So it is already
        // sorted — valid CSR.

        // SAFETY: all invariants are upheld by construction.
        CsrTensor {
            row_ptr: new_row_ptr,
            col_indices: new_col_indices,
            values: new_values,
            nrows: new_nrows,
            ncols: new_ncols,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse free-function operations
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise addition of two COO tensors with the same shape.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes.
pub fn coo_add(a: &CooTensor, b: &CooTensor) -> Result<CooTensor> {
    if a.shape != b.shape {
        return Err(TensorError::shape_mismatch(
            "coo_add",
            &format!("{:?}", a.shape),
            &format!("{:?}", b.shape),
        ));
    }
    let mut combined_indices = a.indices.clone();
    combined_indices.extend(b.indices.iter().cloned());
    let mut combined_values = a.values.clone();
    combined_values.extend(b.values.iter().copied());

    let mut result = CooTensor {
        nnz: combined_values.len(),
        indices: combined_indices,
        values: combined_values,
        shape: a.shape.clone(),
    };
    result.coalesce();
    Ok(result)
}

/// Scale a COO tensor by a scalar, returning a new tensor.
pub fn coo_scale(a: &CooTensor, scalar: f32) -> CooTensor {
    CooTensor {
        indices: a.indices.clone(),
        values: a.values.iter().map(|&v| v * scalar).collect(),
        shape: a.shape.clone(),
        nnz: a.nnz,
    }
}

/// Element-wise addition of two CSR matrices with the same shape.
///
/// # Errors
///
/// Returns an error if `a` and `b` have different shapes.
pub fn csr_add(a: &CsrTensor, b: &CsrTensor) -> Result<CsrTensor> {
    if a.nrows != b.nrows || a.ncols != b.ncols {
        return Err(TensorError::shape_mismatch(
            "csr_add",
            &format!("[{}, {}]", a.nrows, a.ncols),
            &format!("[{}, {}]", b.nrows, b.ncols),
        ));
    }

    let nrows = a.nrows;
    let ncols = a.ncols;

    let mut row_ptr = vec![0usize; nrows + 1];
    let mut col_indices: Vec<usize> = Vec::new();
    let mut values: Vec<f32> = Vec::new();

    for row in 0..nrows {
        // Merge the two sorted row slices
        let mut ka = a.row_ptr[row];
        let mut kb = b.row_ptr[row];
        let end_a = a.row_ptr[row + 1];
        let end_b = b.row_ptr[row + 1];

        while ka < end_a && kb < end_b {
            let ca = a.col_indices[ka];
            let cb = b.col_indices[kb];
            match ca.cmp(&cb) {
                std::cmp::Ordering::Less => {
                    col_indices.push(ca);
                    values.push(a.values[ka]);
                    ka += 1;
                }
                std::cmp::Ordering::Greater => {
                    col_indices.push(cb);
                    values.push(b.values[kb]);
                    kb += 1;
                }
                std::cmp::Ordering::Equal => {
                    let sum = a.values[ka] + b.values[kb];
                    if sum.abs() >= 1e-9 {
                        col_indices.push(ca);
                        values.push(sum);
                    }
                    ka += 1;
                    kb += 1;
                }
            }
        }
        while ka < end_a {
            col_indices.push(a.col_indices[ka]);
            values.push(a.values[ka]);
            ka += 1;
        }
        while kb < end_b {
            col_indices.push(b.col_indices[kb]);
            values.push(b.values[kb]);
            kb += 1;
        }
        row_ptr[row + 1] = col_indices.len();
    }

    CsrTensor::new(row_ptr, col_indices, values, nrows, ncols)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Dense row-major matrix multiply: A [m,k] × B [k,n] → C [m,n].
    fn dense_matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for l in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        c
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── CooTensor::new ────────────────────────────────────────────────────────

    #[test]
    fn test_coo_new_valid() {
        let idx = vec![vec![0, 1], vec![1, 2]];
        let vals = vec![1.0, 2.0];
        let shape = vec![3, 4];
        let t = CooTensor::new(idx, vals, shape).expect("valid coo");
        assert_eq!(t.nnz, 2);
        assert_eq!(t.shape, vec![3, 4]);
    }

    #[test]
    fn test_coo_new_invalid_index_length() {
        // inner index vec has wrong length
        let idx = vec![vec![0]]; // ndim=1 but shape is 2D
        let vals = vec![1.0];
        let shape = vec![3, 4];
        assert!(CooTensor::new(idx, vals, shape).is_err());
    }

    #[test]
    fn test_coo_new_index_out_of_bounds() {
        let idx = vec![vec![5, 0]]; // row 5 out of bounds for shape [3,4]
        let vals = vec![1.0];
        let shape = vec![3, 4];
        assert!(CooTensor::new(idx, vals, shape).is_err());
    }

    #[test]
    fn test_coo_new_mismatched_counts() {
        let idx = vec![vec![0, 0], vec![1, 1]];
        let vals = vec![1.0]; // only one value, two index rows
        let shape = vec![3, 4];
        assert!(CooTensor::new(idx, vals, shape).is_err());
    }

    // ── CooTensor::zeros ─────────────────────────────────────────────────────

    #[test]
    fn test_coo_zeros() {
        let t = CooTensor::zeros(vec![4, 5]);
        assert_eq!(t.nnz, 0);
        assert!(t.values.is_empty());
        assert_eq!(t.shape, vec![4, 5]);
    }

    // ── CooTensor::from_dense / to_dense ─────────────────────────────────────

    #[test]
    fn test_coo_from_dense_basic() {
        // 2×3 matrix with one zero element
        let data = vec![1.0, 0.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2usize, 3];
        let t = CooTensor::from_dense(&data, &shape, 0.0).expect("from_dense");
        assert_eq!(t.nnz, 5); // element 0.0 at [0,1] is excluded (|0| not > 0.0)
    }

    #[test]
    fn test_coo_from_dense_threshold() {
        // Only values > 0.5 in absolute value stored
        let data = vec![0.1, 0.0, -0.3, 1.0, -2.0, 0.4];
        let shape = vec![2usize, 3];
        let t = CooTensor::from_dense(&data, &shape, 0.5).expect("from_dense");
        // 1.0 and -2.0 exceed threshold; others do not
        assert_eq!(t.nnz, 2);
    }

    #[test]
    fn test_coo_to_dense_roundtrip() {
        let data = vec![1.0f32, 0.0, 3.0, 4.0, 0.0, 6.0];
        let shape_arr = vec![2usize, 3];
        let t = CooTensor::from_dense(&data, &shape_arr, 0.0).expect("from_dense");
        let (recovered, shape_out) = t.to_dense().expect("to_dense");
        assert_eq!(shape_out, shape_arr);
        assert!(approx_eq(&recovered, &data, 1e-6));
    }

    #[test]
    fn test_coo_from_dense_wrong_length() {
        let data = vec![1.0, 2.0]; // only 2 elements but shape needs 6
        assert!(CooTensor::from_dense(&data, &[2, 3], 0.0).is_err());
    }

    // ── CooTensor::sparsity ───────────────────────────────────────────────────

    #[test]
    fn test_coo_sparsity() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let t = CooTensor::from_dense(&data, &[10], 0.0).expect("from_dense");
        let sp = t.sparsity();
        // 1 non-zero out of 10 → sparsity = 0.9
        assert!((sp - 0.9).abs() < 1e-6, "sparsity={}", sp);
    }

    #[test]
    fn test_coo_sparsity_all_zeros() {
        let t = CooTensor::zeros(vec![5, 5]);
        let sp = t.sparsity();
        assert!((sp - 1.0).abs() < 1e-6);
    }

    // ── CooTensor::get ────────────────────────────────────────────────────────

    #[test]
    fn test_coo_get() {
        let idx = vec![vec![1, 2], vec![0, 0]];
        let vals = vec![7.0, 3.0];
        let t = CooTensor::new(idx, vals, vec![3, 4]).expect("new");
        assert!((t.get(&[1, 2]) - 7.0).abs() < 1e-6);
        assert!((t.get(&[0, 0]) - 3.0).abs() < 1e-6);
        assert_eq!(t.get(&[2, 3]), 0.0); // not present
    }

    // ── CooTensor::to_csr ─────────────────────────────────────────────────────

    #[test]
    fn test_coo_to_csr_basic() {
        // 3×4 sparse matrix
        //   [1 0 2 0]
        //   [0 3 0 0]
        //   [0 0 0 4]
        let idx = vec![
            vec![0usize, 0],
            vec![0, 2],
            vec![1, 1],
            vec![2, 3],
        ];
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let coo = CooTensor::new(idx, vals, vec![3, 4]).expect("new");
        let csr = coo.to_csr().expect("to_csr");

        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 4);
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.row_ptr, vec![0, 2, 3, 4]);
        assert_eq!(csr.col_indices, vec![0, 2, 1, 3]);
        assert!(approx_eq(&csr.values, &[1.0, 2.0, 3.0, 4.0], 1e-6));
    }

    #[test]
    fn test_coo_to_csr_not_2d() {
        let idx = vec![vec![0, 0, 0]];
        let vals = vec![1.0];
        let coo = CooTensor::new(idx, vals, vec![2, 2, 2]).expect("new");
        assert!(coo.to_csr().is_err());
    }

    // ── CsrTensor::new ────────────────────────────────────────────────────────

    #[test]
    fn test_csr_new_valid() {
        let row_ptr = vec![0, 2, 3, 4];
        let col_idx = vec![0, 2, 1, 3];
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let csr = CsrTensor::new(row_ptr, col_idx, vals, 3, 4).expect("new");
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 4);
        assert_eq!(csr.nnz(), 4);
    }

    #[test]
    fn test_csr_new_invalid_row_ptr() {
        // row_ptr length must be nrows+1 = 4
        let row_ptr = vec![0, 2, 4]; // length 3, not 4
        let col_idx = vec![0, 1, 2, 3];
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        assert!(CsrTensor::new(row_ptr, col_idx, vals, 3, 4).is_err());
    }

    #[test]
    fn test_csr_new_col_out_of_bounds() {
        let row_ptr = vec![0, 1];
        let col_idx = vec![10]; // ncols = 4 → out of bounds
        let vals = vec![1.0];
        assert!(CsrTensor::new(row_ptr, col_idx, vals, 1, 4).is_err());
    }

    // ── CsrTensor::from_dense / to_dense ────────────────────────────────────

    #[test]
    fn test_csr_from_dense_roundtrip() {
        let dense = vec![
            1.0f32, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 4.0,
        ];
        let csr = CsrTensor::from_dense(&dense, 3, 4, 0.0).expect("from_dense");
        let recovered = csr.to_dense();
        assert!(approx_eq(&recovered, &dense, 1e-6));
    }

    // ── CsrTensor::to_coo / to_csr roundtrip ───────────────────────────────

    #[test]
    fn test_csr_to_coo_roundtrip() {
        let dense = vec![
            0.0f32, 5.0, 0.0,
            1.0, 0.0, 3.0,
        ];
        let csr = CsrTensor::from_dense(&dense, 2, 3, 0.0).expect("from_dense");
        let coo = csr.to_coo();
        let csr2 = coo.to_csr().expect("to_csr");
        let recovered = csr2.to_dense();
        assert!(approx_eq(&recovered, &dense, 1e-6));
    }

    // ── CsrTensor::spmv ───────────────────────────────────────────────────────

    #[test]
    fn test_csr_spmv_basic() {
        // A = [[2, 0, 1], [0, 3, 0]], x = [1, 2, 3]
        // A*x = [2*1+0*2+1*3, 0*1+3*2+0*3] = [5, 6]
        let dense = vec![2.0f32, 0.0, 1.0, 0.0, 3.0, 0.0];
        let csr = CsrTensor::from_dense(&dense, 2, 3, 0.0).expect("csr");
        let x = vec![1.0f32, 2.0, 3.0];
        let result = csr.spmv(&x).expect("spmv");
        let reference = dense_matmul(&dense, 2, 3, &x, 1);
        assert!(approx_eq(&result, &reference, 1e-5), "spmv={result:?}, ref={reference:?}");
    }

    #[test]
    fn test_csr_spmv_shape_error() {
        let dense = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let csr = CsrTensor::from_dense(&dense, 2, 3, 0.0).expect("csr");
        assert!(csr.spmv(&[1.0, 2.0]).is_err()); // wrong length
    }

    // ── CsrTensor::spmm ───────────────────────────────────────────────────────

    #[test]
    fn test_csr_spmm_basic() {
        // A [2×3], B [3×2]
        let a_dense = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 1.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0]; // B [3×2]
        let csr = CsrTensor::from_dense(&a_dense, 2, 3, 0.0).expect("csr");
        let result = csr.spmm(&b, 2).expect("spmm");
        let reference = dense_matmul(&a_dense, 2, 3, &b, 2);
        assert!(approx_eq(&result, &reference, 1e-5), "spmm={result:?}, ref={reference:?}");
    }

    #[test]
    fn test_csr_spmm_shape_error() {
        let dense = vec![1.0f32, 0.0, 0.0, 1.0];
        let csr = CsrTensor::from_dense(&dense, 2, 2, 0.0).expect("csr");
        // B should have length ncols*k = 2*3 = 6, but we give 4
        assert!(csr.spmm(&[1.0, 2.0, 3.0, 4.0], 3).is_err());
    }

    // ── CsrTensor::transpose ─────────────────────────────────────────────────

    #[test]
    fn test_csr_transpose_correctness() {
        // A = [[1,2,0],[0,0,3]]  shape [2,3]
        // A^T = [[1,0],[2,0],[0,3]] shape [3,2]
        let dense = vec![1.0f32, 2.0, 0.0, 0.0, 0.0, 3.0];
        let csr = CsrTensor::from_dense(&dense, 2, 3, 0.0).expect("csr");
        let t = csr.transpose();
        assert_eq!(t.nrows, 3);
        assert_eq!(t.ncols, 2);
        let t_dense = t.to_dense();
        let expected = vec![1.0f32, 0.0, 2.0, 0.0, 0.0, 3.0];
        assert!(approx_eq(&t_dense, &expected, 1e-6));
    }

    #[test]
    fn test_csr_transpose_double_inverse() {
        let dense = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 4.0, 0.0, 0.0, 5.0, 6.0];
        let csr = CsrTensor::from_dense(&dense, 3, 4, 0.0).expect("csr");
        let tt = csr.transpose().transpose();
        assert!(approx_eq(&tt.to_dense(), &dense, 1e-6));
    }

    // ── CsrTensor::get ────────────────────────────────────────────────────────

    #[test]
    fn test_csr_get() {
        let dense = vec![0.0f32, 5.0, 1.0, 0.0];
        let csr = CsrTensor::from_dense(&dense, 2, 2, 0.0).expect("csr");
        assert!((csr.get(0, 1) - 5.0).abs() < 1e-6);
        assert!((csr.get(1, 0) - 1.0).abs() < 1e-6);
        assert_eq!(csr.get(0, 0), 0.0);
        assert_eq!(csr.get(1, 1), 0.0);
    }

    // ── CooTensor::coalesce ───────────────────────────────────────────────────

    #[test]
    fn test_coo_coalesce_dedup() {
        // Two entries at [1,2] with values 3.0 and 4.0 → should merge to 7.0
        let idx = vec![vec![0usize, 0], vec![1, 2], vec![1, 2]];
        let vals = vec![1.0, 3.0, 4.0];
        let mut coo = CooTensor {
            nnz: 3,
            indices: idx,
            values: vals,
            shape: vec![3, 4],
        };
        coo.coalesce();
        assert_eq!(coo.nnz, 2);
        assert!((coo.get(&[1, 2]) - 7.0).abs() < 1e-6);
        assert!((coo.get(&[0, 0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_coo_coalesce_cancellation() {
        // Two entries that cancel each other out → removed
        let idx = vec![vec![0usize, 0], vec![0, 0]];
        let vals = vec![1.0, -1.0];
        let mut coo = CooTensor {
            nnz: 2,
            indices: idx,
            values: vals,
            shape: vec![2, 2],
        };
        coo.coalesce();
        assert_eq!(coo.nnz, 0);
    }

    // ── csr_add ───────────────────────────────────────────────────────────────

    #[test]
    fn test_csr_add_basic() {
        let a_dense = vec![1.0f32, 0.0, 0.0, 2.0];
        let b_dense = vec![0.0f32, 3.0, 4.0, 0.0];
        let a = CsrTensor::from_dense(&a_dense, 2, 2, 0.0).expect("a");
        let b = CsrTensor::from_dense(&b_dense, 2, 2, 0.0).expect("b");
        let c = csr_add(&a, &b).expect("csr_add");
        let expected = vec![1.0f32, 3.0, 4.0, 2.0];
        assert!(approx_eq(&c.to_dense(), &expected, 1e-6));
    }

    #[test]
    fn test_csr_add_commutativity() {
        let a_dense = vec![1.0f32, 2.0, 0.0, 3.0];
        let b_dense = vec![4.0f32, 0.0, 5.0, 0.0];
        let a = CsrTensor::from_dense(&a_dense, 2, 2, 0.0).expect("a");
        let b = CsrTensor::from_dense(&b_dense, 2, 2, 0.0).expect("b");
        let ab = csr_add(&a, &b).expect("ab");
        let ba = csr_add(&b, &a).expect("ba");
        assert!(approx_eq(&ab.to_dense(), &ba.to_dense(), 1e-6));
    }

    #[test]
    fn test_csr_add_shape_mismatch() {
        let a = CsrTensor::from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2, 0.0).expect("a");
        let b = CsrTensor::from_dense(&[1.0, 2.0, 3.0], 1, 3, 0.0).expect("b");
        assert!(csr_add(&a, &b).is_err());
    }

    // ── coo_add ───────────────────────────────────────────────────────────────

    #[test]
    fn test_coo_add_basic() {
        let a_dense = vec![1.0f32, 0.0, 0.0, 2.0];
        let b_dense = vec![0.0f32, 3.0, 4.0, 0.0];
        let a = CooTensor::from_dense(&a_dense, &[2, 2], 0.0).expect("a");
        let b = CooTensor::from_dense(&b_dense, &[2, 2], 0.0).expect("b");
        let c = coo_add(&a, &b).expect("coo_add");
        let (c_dense, _) = c.to_dense().expect("to_dense");
        let expected = vec![1.0f32, 3.0, 4.0, 2.0];
        assert!(approx_eq(&c_dense, &expected, 1e-6));
    }

    #[test]
    fn test_coo_add_shape_mismatch() {
        let a = CooTensor::zeros(vec![2, 3]);
        let b = CooTensor::zeros(vec![3, 2]);
        assert!(coo_add(&a, &b).is_err());
    }

    // ── coo_scale ────────────────────────────────────────────────────────────

    #[test]
    fn test_coo_scale() {
        let data = vec![1.0f32, 0.0, 2.0, 3.0];
        let coo = CooTensor::from_dense(&data, &[2, 2], 0.0).expect("from_dense");
        let scaled = coo_scale(&coo, 2.0);
        let (s_dense, _) = scaled.to_dense().expect("to_dense");
        let expected = vec![2.0f32, 0.0, 4.0, 6.0];
        assert!(approx_eq(&s_dense, &expected, 1e-6));
    }

    #[test]
    fn test_coo_scale_zero() {
        let data = vec![1.0f32, 2.0, 3.0];
        let coo = CooTensor::from_dense(&data, &[3], 0.0).expect("from_dense");
        let scaled = coo_scale(&coo, 0.0);
        assert_eq!(scaled.nnz, 3); // nnz unchanged; values are 0 but not coalesced
        let (s_dense, _) = scaled.to_dense().expect("to_dense");
        assert!(approx_eq(&s_dense, &[0.0, 0.0, 0.0], 1e-6));
    }
}
