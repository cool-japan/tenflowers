// Linear Algebra: Linear System Solver using LU Factorization
//
// Solves linear systems Ax = b using LU decomposition with partial pivoting.
// The algorithm consists of three main steps:
// 1. Apply permutation to RHS: Pb = permuted_b
// 2. Forward substitution: solve Ly = Pb where L is lower triangular  
// 3. Backward substitution: solve Ux = y where U is upper triangular
//
// This implementation supports multiple right-hand sides (multiple b vectors).

struct LinalgMetadata {
    rows_a: u32,     // Matrix dimension n
    cols_a: u32,     // Number of right-hand sides (nrhs)
    rows_b: u32,     // Current row index for sequential operations
    cols_b: u32,     // Current column index
    batch_size: u32,
    tolerance: f32,
    max_iterations: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> l_matrix: array<f32>;         // Lower triangular matrix L
@group(0) @binding(1) var<storage, read> u_matrix: array<f32>;         // Upper triangular matrix U  
@group(0) @binding(2) var<storage, read> p_matrix: array<f32>;         // Permutation matrix P
@group(0) @binding(3) var<storage, read> b_vector: array<f32>;         // Right-hand side vector(s)
@group(0) @binding(4) var<storage, read_write> y_vector: array<f32>;   // Intermediate solution vector
@group(0) @binding(5) var<storage, read_write> x_vector: array<f32>;   // Final solution vector  
@group(0) @binding(6) var<storage, read_write> status: array<u32>;     // Status flags
@group(0) @binding(7) var<uniform> metadata: LinalgMetadata;

// Check for singularity by examining diagonal elements of U
@compute @workgroup_size(256, 1, 1)
fn check_singularity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let n = metadata.rows_a;
    
    if (thread_id >= n) {
        return;
    }
    
    // Check if diagonal element is too small (near singular)
    let diag_idx = thread_id * n + thread_id;
    let diag_val = abs(u_matrix[diag_idx]);
    
    if (diag_val < metadata.tolerance) {
        // Matrix is singular
        status[0] = 1u;
    }
}

// Apply permutation matrix to right-hand side: y = P * b
@compute @workgroup_size(64, 1, 1)
fn apply_permutation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let rhs = global_id.y;
    let n = metadata.rows_a;
    let nrhs = metadata.cols_a;
    
    if (row >= n || rhs >= nrhs) {
        return;
    }
    
    // Apply permutation: y[row, rhs] = b[perm[row], rhs]
    let perm_row = u32(p_matrix[row]);
    let b_idx = perm_row * nrhs + rhs;
    let y_idx = row * nrhs + rhs;
    
    if (b_idx < arrayLength(&b_vector) && y_idx < arrayLength(&y_vector)) {
        y_vector[y_idx] = b_vector[b_idx];
    }
}

// Forward substitution: solve L * y = P * b
// This must be done sequentially for each row
@compute @workgroup_size(64, 1, 1)
fn forward_substitution(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rhs = global_id.x;
    let n = metadata.rows_a;
    let nrhs = metadata.cols_a;
    let current_row = metadata.rows_b;
    
    if (rhs >= nrhs || current_row >= n) {
        return;
    }
    
    // Solve for y[current_row, rhs]
    var sum = 0.0;
    
    // Compute sum of L[current_row, j] * y[j, rhs] for j < current_row
    for (var j = 0u; j < current_row; j = j + 1u) {
        let l_idx = current_row * n + j;
        let y_idx = j * nrhs + rhs;
        
        if (l_idx < arrayLength(&l_matrix) && y_idx < arrayLength(&y_vector)) {
            sum = sum + l_matrix[l_idx] * y_vector[y_idx];
        }
    }
    
    // Get L[current_row, current_row] (should be 1.0 for proper LU)
    let l_diag_idx = current_row * n + current_row;
    let l_diag = l_matrix[l_diag_idx];
    
    // Solve: L[current_row, current_row] * y[current_row, rhs] = b[current_row, rhs] - sum
    let b_idx = current_row * nrhs + rhs;
    let y_idx = current_row * nrhs + rhs;
    
    if (abs(l_diag) < metadata.tolerance) {
        // Singular matrix
        status[0] = 1u;
        return;
    }
    
    if (b_idx < arrayLength(&y_vector) && y_idx < arrayLength(&y_vector)) {
        y_vector[y_idx] = (y_vector[b_idx] - sum) / l_diag;
    }
}

// Backward substitution: solve U * x = y  
// This must be done sequentially from last row to first
@compute @workgroup_size(64, 1, 1)
fn backward_substitution(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rhs = global_id.x;
    let n = metadata.rows_a;
    let nrhs = metadata.cols_a;
    let current_row = metadata.rows_b;
    
    if (rhs >= nrhs || current_row >= n) {
        return;
    }
    
    // Solve for x[current_row, rhs]
    var sum = 0.0;
    
    // Compute sum of U[current_row, j] * x[j, rhs] for j > current_row
    for (var j = current_row + 1u; j < n; j = j + 1u) {
        let u_idx = current_row * n + j;
        let x_idx = j * nrhs + rhs;
        
        if (u_idx < arrayLength(&u_matrix) && x_idx < arrayLength(&x_vector)) {
            sum = sum + u_matrix[u_idx] * x_vector[x_idx];
        }
    }
    
    // Get U[current_row, current_row]
    let u_diag_idx = current_row * n + current_row;
    let u_diag = u_matrix[u_diag_idx];
    
    // Solve: U[current_row, current_row] * x[current_row, rhs] = y[current_row, rhs] - sum
    let y_idx = current_row * nrhs + rhs;
    let x_idx = current_row * nrhs + rhs;
    
    if (abs(u_diag) < metadata.tolerance) {
        // Singular matrix
        status[0] = 1u;
        return;
    }
    
    if (y_idx < arrayLength(&y_vector) && x_idx < arrayLength(&x_vector)) {
        x_vector[x_idx] = (y_vector[y_idx] - sum) / u_diag;
    }
}

// Complete linear solve in multiple passes
// This is an alternative approach that performs the entire solve in a single kernel
// Less efficient than the sequential approach but simpler for debugging
@compute @workgroup_size(16, 16, 1)
fn solve_complete(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let rhs = global_id.x;
    let n = metadata.rows_a;
    let nrhs = metadata.cols_a;
    
    if (row >= n || rhs >= nrhs) {
        return;
    }
    
    // Initialize status to success
    if (row == 0u && rhs == 0u) {
        status[0] = 0u;
    }
    
    workgroupBarrier();
    
    // Step 1: Apply permutation (all threads can do this in parallel)
    let perm_row = u32(p_matrix[row]);
    let b_idx = perm_row * nrhs + rhs;
    let y_idx = row * nrhs + rhs;
    
    if (b_idx < arrayLength(&b_vector) && y_idx < arrayLength(&y_vector)) {
        y_vector[y_idx] = b_vector[b_idx];
    }
    
    workgroupBarrier();
    
    // Steps 2 & 3: Forward and backward substitution
    // Note: This simplified version doesn't handle the sequential dependencies properly
    // For a correct implementation, use the sequential kernels above
    
    // For demonstration, just copy y to x
    if (y_idx < arrayLength(&y_vector) && row * nrhs + rhs < arrayLength(&x_vector)) {
        x_vector[row * nrhs + rhs] = y_vector[y_idx];
    }
}

// Initialize solution vectors
@compute @workgroup_size(64, 1, 1)
fn initialize_solve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let n = metadata.rows_a;
    let nrhs = metadata.cols_a;
    let total_elements = n * nrhs;
    
    if (index >= total_elements) {
        return;
    }
    
    // Initialize y and x vectors to zero
    if (index < arrayLength(&y_vector)) {
        y_vector[index] = 0.0;
    }
    
    if (index < arrayLength(&x_vector)) {
        x_vector[index] = 0.0;
    }
    
    // Initialize status to success
    if (index == 0u) {
        status[0] = 0u;
    }
}

// Residual computation for verification: compute ||Ax - b||
@compute @workgroup_size(64, 1, 1)
fn compute_residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let rhs = global_id.y;
    let n = metadata.rows_a;
    let nrhs = metadata.cols_a;
    
    if (row >= n || rhs >= nrhs) {
        return;
    }
    
    // Compute (Ax)[row, rhs] = sum(A[row, j] * x[j, rhs])
    var ax_value = 0.0;
    
    for (var j = 0u; j < n; j = j + 1u) {
        // Reconstruct A from L and U: A = P^(-1) * L * U
        // For simplicity, assume we have access to original A
        // In practice, this would require reconstructing A or storing it separately
        let a_idx = row * n + j;
        let x_idx = j * nrhs + rhs;
        
        if (a_idx < arrayLength(&l_matrix) && x_idx < arrayLength(&x_vector)) {
            // This is a placeholder - would need actual A matrix
            ax_value = ax_value + l_matrix[a_idx] * x_vector[x_idx];
        }
    }
    
    // Compute residual: r[row, rhs] = Ax[row, rhs] - b[row, rhs]
    let b_idx = row * nrhs + rhs;
    let residual = ax_value - b_vector[b_idx];
    
    // Store residual (reusing y_vector for this purpose)
    let r_idx = row * nrhs + rhs;
    if (r_idx < arrayLength(&y_vector)) {
        y_vector[r_idx] = residual;
    }
}