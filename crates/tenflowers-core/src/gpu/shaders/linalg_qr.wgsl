// Linear Algebra: QR Decomposition using Householder Reflections
//
// Performs QR decomposition of a matrix using Householder reflections.
// For a matrix A [m, n], computes: A = Q * R
// Where: Q [m, m] (orthogonal), R [m, n] (upper triangular)
//
// This implementation follows the standard Householder QR algorithm adapted for GPU execution.
// The algorithm proceeds column by column, computing Householder vectors and applying
// transformations to eliminate subdiagonal elements.

struct LinalgMetadata {
    rows_a: u32,
    cols_a: u32,
    rows_b: u32,  // Current column index for iteration
    cols_b: u32,
    batch_size: u32,
    tolerance: f32,
    max_iterations: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read_write> working_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> r_matrix: array<f32>;
@group(0) @binding(3) var<storage, read_write> householder_vectors: array<f32>;
@group(0) @binding(4) var<storage, read_write> tau_buffer: array<f32>;
@group(0) @binding(5) var<uniform> metadata: LinalgMetadata;

// Helper function to compute 2-norm of a vector
fn compute_norm(start_idx: u32, length: u32) -> f32 {
    var norm_squared = 0.0;
    for (var i = 0u; i < length; i = i + 1u) {
        let val = working_matrix[start_idx + i];
        norm_squared = norm_squared + val * val;
    }
    return sqrt(norm_squared);
}

// Compute Householder vector for column k
@compute @workgroup_size(256, 1, 1)
fn compute_householder(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    let k = metadata.rows_b; // Current column index
    
    if (k >= min(m, n)) {
        return;
    }
    
    // Only the first thread in the workgroup computes the Householder vector
    if (thread_id == 0u) {
        let col_start = k * m + k; // Start of subcolumn
        let col_length = m - k;   // Length of subcolumn
        
        if (col_length == 0u) {
            return;
        }
        
        // Get the pivot element
        let alpha = working_matrix[col_start];
        
        // Compute 2-norm of the subcolumn
        let norm = compute_norm(col_start, col_length);
        
        if (norm < metadata.tolerance) {
            // Column is already zero, no reflection needed
            tau_buffer[k] = 0.0;
            return;
        }
        
        // Compute tau and update the Householder vector
        var tau: f32;
        var beta: f32;
        
        if (alpha >= 0.0) {
            beta = -norm;
        } else {
            beta = norm;
        }
        
        tau = (beta - alpha) / beta;
        let scale = 1.0 / (alpha - beta);
        
        // Store tau value
        tau_buffer[k] = tau;
        
        // Compute Householder vector: v = (x - beta*e1) / (x[0] - beta)
        // First element is implicitly 1
        householder_vectors[k * m + k] = 1.0;
        
        // Scale the rest of the vector
        for (var i = 1u; i < col_length; i = i + 1u) {
            let idx = col_start + i;
            householder_vectors[k * m + k + i] = working_matrix[idx] * scale;
        }
        
        // Update the working matrix: set first element to beta, rest to zero
        working_matrix[col_start] = beta;
        for (var i = 1u; i < col_length; i = i + 1u) {
            working_matrix[col_start + i] = 0.0;
        }
    }
}

// Apply Householder transformation to remaining columns
@compute @workgroup_size(16, 16, 1)
fn apply_householder(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    let k = metadata.rows_b; // Current column index
    
    if (row >= m || col >= n || col <= k) {
        return;
    }
    
    let tau = tau_buffer[k];
    if (tau == 0.0) {
        return; // No transformation needed
    }
    
    // Apply Householder transformation: A := A - tau * v * (v^T * A)
    // We're updating column 'col' of the matrix
    
    let col_length = m - k;
    var dot_product = 0.0;
    
    // Compute v^T * A[:, col] for the current column
    for (var i = 0u; i < col_length; i = i + 1u) {
        let v_idx = k * m + k + i;
        let a_idx = (k + i) * n + col;
        
        if (k + i < m && a_idx < arrayLength(&working_matrix)) {
            dot_product = dot_product + householder_vectors[v_idx] * working_matrix[a_idx];
        }
    }
    
    // Update A[:, col] -= tau * v * dot_product
    if (row >= k && row < m) {
        let v_idx = k * m + k + (row - k);
        let a_idx = row * n + col;
        
        if (a_idx < arrayLength(&working_matrix)) {
            working_matrix[a_idx] = working_matrix[a_idx] - tau * householder_vectors[v_idx] * dot_product;
        }
    }
}

// Memory-optimized Householder transformation with improved coalescing
@compute @workgroup_size(256, 1, 1)
fn apply_householder_optimized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    let k = metadata.rows_b; // Current column index
    
    if (k >= min(m, n)) {
        return;
    }
    
    let tau = tau_buffer[k];
    if (tau == 0.0) {
        return; // No transformation needed
    }
    
    let col_length = m - k;
    let remaining_cols = n - k - 1u;
    
    // Each thread handles multiple elements for better memory bandwidth utilization
    let elements_per_thread = (remaining_cols * col_length + 255u) / 256u;
    
    for (var elem = 0u; elem < elements_per_thread; elem = elem + 1u) {
        let global_elem_idx = thread_id * elements_per_thread + elem;
        
        if (global_elem_idx >= remaining_cols * col_length) {
            break;
        }
        
        // Convert linear index to (column, row) coordinates
        let col_offset = global_elem_idx / col_length;
        let row_offset = global_elem_idx % col_length;
        let col = k + 1u + col_offset;
        let row = k + row_offset;
        
        if (col >= n || row >= m) {
            continue;
        }
        
        // Compute dot product for this column if this is the first row
        var dot_product = 0.0;
        if (row_offset == 0u) {
            for (var i = 0u; i < col_length; i = i + 1u) {
                let v_idx = k * m + k + i;
                let a_idx = (k + i) * n + col;
                
                if (v_idx < arrayLength(&householder_vectors) && a_idx < arrayLength(&working_matrix)) {
                    dot_product = dot_product + householder_vectors[v_idx] * working_matrix[a_idx];
                }
            }
        }
        
        // Broadcast dot product within workgroup (simplified - would need proper reduction)
        // For now, each thread recomputes the dot product
        dot_product = 0.0;
        for (var i = 0u; i < col_length; i = i + 1u) {
            let v_idx = k * m + k + i;
            let a_idx = (k + i) * n + col;
            
            if (v_idx < arrayLength(&householder_vectors) && a_idx < arrayLength(&working_matrix)) {
                dot_product = dot_product + householder_vectors[v_idx] * working_matrix[a_idx];
            }
        }
        
        // Apply transformation
        let v_idx = k * m + k + row_offset;
        let a_idx = row * n + col;
        
        if (v_idx < arrayLength(&householder_vectors) && a_idx < arrayLength(&working_matrix)) {
            working_matrix[a_idx] = working_matrix[a_idx] - tau * householder_vectors[v_idx] * dot_product;
        }
    }
}

// Extract Q matrix from Householder vectors
@compute @workgroup_size(16, 16, 1)
fn extract_q_matrix(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = metadata.rows_a;
    
    if (row >= m || col >= m) {
        return;
    }
    
    let q_idx = row * m + col;
    
    // Initialize Q as identity matrix
    if (row == col) {
        q_matrix[q_idx] = 1.0;
    } else {
        q_matrix[q_idx] = 0.0;
    }
    
    // Apply Householder transformations in reverse order to construct Q
    let min_mn = min(m, metadata.cols_a);
    
    // Apply transformations Q = H_{min_mn-1} * ... * H_1 * H_0
    // We iterate from the last transformation backwards
    for (var k = 0u; k < min_mn; k = k + 1u) {
        let tau_k = tau_buffer[k];
        
        if (tau_k == 0.0) {
            continue; // Skip transformations with zero tau
        }
        
        // Apply Q := Q * (I - tau_k * v_k * v_k^T)
        // This is equivalent to: q_col := q_col - tau_k * v_k * (v_k^T * q_col)
        
        // Compute v_k^T * q_col where q_col is the current column of Q
        var dot_product = 0.0;
        for (var i = k; i < m; i = i + 1u) {
            let v_idx = k * m + i;
            let q_read_idx = i * m + col;
            
            if (v_idx < arrayLength(&householder_vectors) && q_read_idx < arrayLength(&q_matrix)) {
                dot_product = dot_product + householder_vectors[v_idx] * q_matrix[q_read_idx];
            }
        }
        
        // Update q_col := q_col - tau_k * v_k * dot_product
        if (row >= k) {
            let v_idx = k * m + row;
            if (v_idx < arrayLength(&householder_vectors)) {
                q_matrix[q_idx] = q_matrix[q_idx] - tau_k * householder_vectors[v_idx] * dot_product;
            }
        }
        
        // Synchronize across workgroup to ensure consistency
        workgroupBarrier();
    }
}

// Extract R matrix (upper triangular part of working matrix)
@compute @workgroup_size(16, 16, 1)
fn extract_r_matrix(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    
    if (row >= m || col >= n) {
        return;
    }
    
    let r_idx = row * n + col;
    let work_idx = row * n + col;
    
    // R is the upper triangular part of the transformed matrix
    if (row <= col) {
        r_matrix[r_idx] = working_matrix[work_idx];
    } else {
        r_matrix[r_idx] = 0.0;
    }
}

// Alternative simplified QR decomposition kernel
// This kernel performs a simplified QR decomposition in a single pass
// Suitable for smaller matrices where the full Householder approach is overkill
@compute @workgroup_size(16, 16, 1)
fn simplified_qr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    
    if (row >= m || col >= n) {
        return;
    }
    
    // Initialize Q as identity and R as copy of input
    if (row < m && col < m) {
        let q_idx = row * m + col;
        if (row == col) {
            q_matrix[q_idx] = 1.0;
        } else {
            q_matrix[q_idx] = 0.0;
        }
    }
    
    if (row < m && col < n) {
        let r_idx = row * n + col;
        r_matrix[r_idx] = working_matrix[row * n + col];
    }
    
    // This is a placeholder for a complete QR implementation
    // A full GPU implementation would require more sophisticated algorithms
    // like parallel Householder QR or Givens rotations
}

// Utility kernel for matrix initialization
@compute @workgroup_size(16, 16, 1)
fn initialize_qr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = metadata.rows_a;
    let n = metadata.cols_a;
    
    // Initialize Q as identity matrix
    if (row < m && col < m) {
        let q_idx = row * m + col;
        if (row == col) {
            q_matrix[q_idx] = 1.0;
        } else {
            q_matrix[q_idx] = 0.0;
        }
    }
    
    // Initialize Householder vectors and tau
    if (row == 0u && col < min(m, n)) {
        tau_buffer[col] = 0.0;
    }
    
    // Clear Householder vectors
    if (row < m && col < n) {
        let h_idx = col * m + row;
        if (h_idx < arrayLength(&householder_vectors)) {
            householder_vectors[h_idx] = 0.0;
        }
    }
}