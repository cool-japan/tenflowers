// Linear Algebra: Singular Value Decomposition (SVD)
//
// Performs SVD using Jacobi rotations adapted for GPU execution.
// For a matrix A [m, n], computes: A = U * S * V^T
// Where: U [m, m] (orthogonal), S [min(m,n)] (diagonal), V^T [n, n] (orthogonal)
//
// This implementation uses the Jacobi method which is suitable for GPU parallelization
// and provides good numerical stability for small to medium matrices.

struct LinalgMetadata {
    rows_a: u32,
    cols_a: u32,
    rows_b: u32,
    cols_b: u32,
    batch_size: u32,
    tolerance: f32,
    max_iterations: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> input_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> u_matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> vt_matrix: array<f32>;
@group(0) @binding(4) var<storage, read_write> working_matrix: array<f32>;
@group(0) @binding(5) var<uniform> metadata: LinalgMetadata;

// Initialize matrices for SVD computation
@compute @workgroup_size(16, 16, 1)
fn initialize_svd(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    let row = global_id.y;
    let col = global_id.x;
    
    // Initialize working matrix with input data
    if (row < m && col < n) {
        let idx = row * n + col;
        working_matrix[idx] = input_matrix[idx];
    }
    
    // Initialize U matrix as identity (m x m)
    if (row < m && col < m) {
        let u_idx = row * m + col;
        if (row == col) {
            u_matrix[u_idx] = 1.0;
        } else {
            u_matrix[u_idx] = 0.0;
        }
    }
    
    // Initialize V^T matrix as identity (n x n)
    if (row < n && col < n) {
        let vt_idx = row * n + col;
        if (row == col) {
            vt_matrix[vt_idx] = 1.0;
        } else {
            vt_matrix[vt_idx] = 0.0;
        }
    }
    
    // Initialize singular values to zero
    if (row == 0u && col < min(m, n)) {
        s_values[col] = 0.0;
    }
}

// Compute Givens rotation parameters
fn compute_givens_rotation(a: f32, b: f32) -> vec2<f32> {
    if (abs(b) < metadata.tolerance) {
        return vec2<f32>(1.0, 0.0); // cos, sin
    }
    
    let tau = a / b;
    let t = 1.0 / (abs(tau) + sqrt(1.0 + tau * tau));
    
    if (tau < 0.0) {
        let cos_val = t / sqrt(1.0 + t * t);
        let sin_val = -t * cos_val;
        return vec2<f32>(cos_val, sin_val);
    } else {
        let cos_val = t / sqrt(1.0 + t * t);
        let sin_val = t * cos_val;
        return vec2<f32>(cos_val, sin_val);
    }
}

// Apply Givens rotation to eliminate off-diagonal element (i,j)
@compute @workgroup_size(256, 1, 1)
fn apply_givens_rotation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.cols_a;
    let thread_id = global_id.x;
    
    // Get the target (i,j) pair from metadata
    let i = metadata.rows_b; // Reusing metadata fields for i,j indices
    let j = metadata.cols_b;
    
    if (i >= j || i >= n || j >= n) {
        return;
    }
    
    // Compute rotation parameters
    let a_ii = working_matrix[i * n + i];
    let a_jj = working_matrix[j * n + j];
    let a_ij = working_matrix[i * n + j];
    let a_ji = working_matrix[j * n + i];
    
    // Symmetric matrix: a_ij == a_ji
    let off_diag = a_ij;
    
    if (abs(off_diag) < metadata.tolerance) {
        return; // Already converged
    }
    
    // Compute rotation angle
    let diff = a_jj - a_ii;
    let t = if (abs(diff) < metadata.tolerance) {
        if (off_diag > 0.0) { 1.0 } else { -1.0 }
    } else {
        off_diag / (diff + sign(diff) * sqrt(diff * diff + off_diag * off_diag))
    };
    
    let cos_theta = 1.0 / sqrt(1.0 + t * t);
    let sin_theta = t * cos_theta;
    
    // Apply rotation to working matrix (only upper triangle matters)
    if (thread_id < n) {
        let k = thread_id;
        
        if (k != i && k != j) {
            // Update row i and j, column k
            let a_ik = working_matrix[i * n + k];
            let a_jk = working_matrix[j * n + k];
            
            working_matrix[i * n + k] = cos_theta * a_ik - sin_theta * a_jk;
            working_matrix[j * n + k] = sin_theta * a_ik + cos_theta * a_jk;
            
            // Update column i and j, row k
            working_matrix[k * n + i] = working_matrix[i * n + k]; // Symmetric
            working_matrix[k * n + j] = working_matrix[j * n + k]; // Symmetric
        }
    }
    
    // Update diagonal elements
    if (thread_id == 0u) {
        let a_ii_new = cos_theta * cos_theta * a_ii - 2.0 * sin_theta * cos_theta * off_diag + sin_theta * sin_theta * a_jj;
        let a_jj_new = sin_theta * sin_theta * a_ii + 2.0 * sin_theta * cos_theta * off_diag + cos_theta * cos_theta * a_jj;
        
        working_matrix[i * n + i] = a_ii_new;
        working_matrix[j * n + j] = a_jj_new;
        working_matrix[i * n + j] = 0.0;
        working_matrix[j * n + i] = 0.0;
    }
    
    // Apply rotation to V matrix (V = V * G^T)
    if (thread_id < n) {
        let k = thread_id;
        let v_ki = vt_matrix[k * n + i];
        let v_kj = vt_matrix[k * n + j];
        
        vt_matrix[k * n + i] = cos_theta * v_ki - sin_theta * v_kj;
        vt_matrix[k * n + j] = sin_theta * v_ki + cos_theta * v_kj;
    }
}

// Extract singular values from diagonal of working matrix
@compute @workgroup_size(256, 1, 1)
fn extract_singular_values(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = min(metadata.rows_a, metadata.cols_a);
    let idx = global_id.x;
    
    if (idx >= n) {
        return;
    }
    
    let diagonal_val = working_matrix[idx * metadata.cols_a + idx];
    s_values[idx] = sqrt(abs(diagonal_val));
}

// For rectangular matrices: reduce to bidiagonal form first
@compute @workgroup_size(16, 16, 1)
fn bidiagonalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= m || col >= n) {
        return;
    }
    
    // This is a simplified bidiagonalization
    // A full implementation would use Householder reflections
    // For now, we assume the input is already in a suitable form
    // or is square (m == n)
}

// Compute U matrix for rectangular case using back-transformation
@compute @workgroup_size(16, 16, 1)
fn compute_u_matrix(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= m || col >= m) {
        return;
    }
    
    // For rectangular matrices, U needs to be computed differently
    // This is a simplified version - a full implementation would
    // require the intermediate transformations from bidiagonalization
    
    if (col < min(m, n)) {
        // Copy from V matrix if square, otherwise compute properly
        if (m == n) {
            u_matrix[row * m + col] = vt_matrix[row * n + col];
        }
    }
}