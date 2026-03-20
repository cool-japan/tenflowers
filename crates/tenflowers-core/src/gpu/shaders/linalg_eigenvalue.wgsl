// Linear Algebra: Eigenvalue Computation using QR Algorithm
//
// Computes eigenvalues and eigenvectors of a symmetric matrix using the QR algorithm.
// For a symmetric matrix A [n, n], computes eigenvalues λ and eigenvectors V such that:
// A * V = V * Λ (where Λ is diagonal matrix of eigenvalues)
//
// This implementation uses the symmetric QR algorithm with Givens rotations,
// which is suitable for GPU parallelization and provides good numerical stability.

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
@group(0) @binding(1) var<storage, read_write> eigenvalues: array<f32>;
@group(0) @binding(2) var<storage, read_write> eigenvectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> working_matrix: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_matrix: array<f32>;
@group(0) @binding(5) var<uniform> metadata: LinalgMetadata;

// Initialize matrices for eigenvalue computation
@compute @workgroup_size(16, 16, 1)
fn initialize_eigen(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= n || col >= n) {
        return;
    }
    
    let idx = row * n + col;
    
    // Copy input to working matrix
    working_matrix[idx] = input_matrix[idx];
    
    // Initialize eigenvectors as identity matrix
    if (row == col) {
        eigenvectors[idx] = 1.0;
        q_matrix[idx] = 1.0;
    } else {
        eigenvectors[idx] = 0.0;
        q_matrix[idx] = 0.0;
    }
    
    // Initialize eigenvalues
    if (row == 0u && col < n) {
        eigenvalues[col] = 0.0;
    }
}

// Check convergence by examining off-diagonal elements
@compute @workgroup_size(256, 1, 1)
fn check_convergence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let thread_id = global_id.x;
    
    if (thread_id >= n * (n - 1u) / 2u) {
        return;
    }
    
    // Convert thread_id to (i,j) with i < j
    var count = 0u;
    for (var i = 0u; i < n - 1u; i = i + 1u) {
        for (var j = i + 1u; j < n; j = j + 1u) {
            if (count == thread_id) {
                let off_diag = abs(working_matrix[i * n + j]);
                // Set a convergence flag (simplified)
                if (off_diag < metadata.tolerance) {
                    // Mark as converged (implementation specific)
                }
                return;
            }
            count = count + 1u;
        }
    }
}

// Apply Givens rotation to eliminate off-diagonal element (i,j)
@compute @workgroup_size(256, 1, 1)
fn apply_givens_eigen(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let thread_id = global_id.x;
    
    // Get the target (i,j) pair from metadata
    let i = metadata.rows_b; // Reusing metadata fields for i,j indices
    let j = metadata.cols_b;
    
    if (i >= j || i >= n || j >= n) {
        return;
    }
    
    // Get matrix elements
    let a_ii = working_matrix[i * n + i];
    let a_jj = working_matrix[j * n + j];
    let a_ij = working_matrix[i * n + j];
    
    if (abs(a_ij) < metadata.tolerance) {
        return; // Already converged
    }
    
    // Compute rotation parameters for symmetric case
    let diff = a_jj - a_ii;
    let t = if (abs(diff) < metadata.tolerance) {
        if (a_ij > 0.0) { 1.0 } else { -1.0 }
    } else {
        a_ij / (diff + sign(diff) * sqrt(diff * diff + a_ij * a_ij))
    };
    
    let cos_theta = 1.0 / sqrt(1.0 + t * t);
    let sin_theta = t * cos_theta;
    
    // Apply rotation to working matrix: A = G^T * A * G
    if (thread_id < n) {
        let k = thread_id;
        
        if (k != i && k != j) {
            // Transform row k: A[k,i] and A[k,j]
            let a_ki = working_matrix[k * n + i];
            let a_kj = working_matrix[k * n + j];
            
            working_matrix[k * n + i] = cos_theta * a_ki - sin_theta * a_kj;
            working_matrix[k * n + j] = sin_theta * a_ki + cos_theta * a_kj;
            
            // Transform column k: A[i,k] and A[j,k] (symmetric)
            working_matrix[i * n + k] = working_matrix[k * n + i];
            working_matrix[j * n + k] = working_matrix[k * n + j];
        }
    }
    
    // Update diagonal elements
    if (thread_id == 0u) {
        let a_ii_new = cos_theta * cos_theta * a_ii - 2.0 * sin_theta * cos_theta * a_ij + sin_theta * sin_theta * a_jj;
        let a_jj_new = sin_theta * sin_theta * a_ii + 2.0 * sin_theta * cos_theta * a_ij + cos_theta * cos_theta * a_jj;
        
        working_matrix[i * n + i] = a_ii_new;
        working_matrix[j * n + j] = a_jj_new;
        working_matrix[i * n + j] = 0.0;
        working_matrix[j * n + i] = 0.0;
    }
    
    // Accumulate eigenvectors: V = V * G
    if (thread_id < n) {
        let k = thread_id;
        let v_ki = eigenvectors[k * n + i];
        let v_kj = eigenvectors[k * n + j];
        
        eigenvectors[k * n + i] = cos_theta * v_ki - sin_theta * v_kj;
        eigenvectors[k * n + j] = sin_theta * v_ki + cos_theta * v_kj;
    }
}

// Extract eigenvalues from diagonal
@compute @workgroup_size(256, 1, 1)
fn extract_eigenvalues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let idx = global_id.x;
    
    if (idx >= n) {
        return;
    }
    
    eigenvalues[idx] = working_matrix[idx * n + idx];
}

// Sort eigenvalues and eigenvectors in descending order
@compute @workgroup_size(256, 1, 1)
fn sort_eigenvalues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let thread_id = global_id.x;
    
    // Simple bubble sort (not optimal for GPU, but works for small matrices)
    // For larger matrices, a parallel sorting algorithm would be needed
    
    if (thread_id != 0u) {
        return;
    }
    
    // Perform sorting in a single thread for simplicity
    for (var i = 0u; i < n - 1u; i = i + 1u) {
        for (var j = 0u; j < n - i - 1u; j = j + 1u) {
            if (eigenvalues[j] < eigenvalues[j + 1u]) {
                // Swap eigenvalues
                let temp_val = eigenvalues[j];
                eigenvalues[j] = eigenvalues[j + 1u];
                eigenvalues[j + 1u] = temp_val;
                
                // Swap corresponding eigenvectors
                for (var k = 0u; k < n; k = k + 1u) {
                    let temp_vec = eigenvectors[k * n + j];
                    eigenvectors[k * n + j] = eigenvectors[k * n + (j + 1u)];
                    eigenvectors[k * n + (j + 1u)] = temp_vec;
                }
            }
        }
    }
}

// Normalize eigenvectors to unit length
@compute @workgroup_size(256, 1, 1)
fn normalize_eigenvectors(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let col = global_id.x;
    
    if (col >= n) {
        return;
    }
    
    // Compute norm of column 'col'
    var norm_squared = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let val = eigenvectors[i * n + col];
        norm_squared = norm_squared + val * val;
    }
    
    let norm = sqrt(norm_squared);
    
    if (norm > metadata.tolerance) {
        // Normalize the column
        for (var i = 0u; i < n; i = i + 1u) {
            eigenvectors[i * n + col] = eigenvectors[i * n + col] / norm;
        }
    }
}

// Compute residual for convergence checking: ||A*v - λ*v||
@compute @workgroup_size(256, 1, 1)
fn compute_residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let col = global_id.x;
    
    if (col >= n) {
        return;
    }
    
    let lambda = eigenvalues[col];
    var residual = 0.0;
    
    for (var i = 0u; i < n; i = i + 1u) {
        var av_i = 0.0;
        for (var j = 0u; j < n; j = j + 1u) {
            av_i = av_i + input_matrix[i * n + j] * eigenvectors[j * n + col];
        }
        
        let lv_i = lambda * eigenvectors[i * n + col];
        let diff = av_i - lv_i;
        residual = residual + diff * diff;
    }
    
    // Store residual (could be used for convergence checking)
    // For now, we just compute it
}

// Alternative: Power iteration for largest eigenvalue (simpler, more GPU-friendly)
@compute @workgroup_size(256, 1, 1)
fn power_iteration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let idx = global_id.x;
    
    if (idx >= n) {
        return;
    }
    
    // This is a simplified power iteration step
    // Full implementation would require multiple iterations and proper normalization
    
    var sum = 0.0;
    for (var j = 0u; j < n; j = j + 1u) {
        sum = sum + input_matrix[idx * n + j] * eigenvectors[j * n + 0u];
    }
    
    q_matrix[idx] = sum;
}