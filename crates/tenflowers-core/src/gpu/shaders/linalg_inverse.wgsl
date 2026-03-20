// Linear Algebra: Matrix Inverse using Gauss-Jordan Elimination
//
// Computes the inverse of a square matrix using Gauss-Jordan elimination
// with partial pivoting. The implementation uses an augmented matrix [A|I]
// and transforms it to [I|A^(-1)].
//
// Input:  Matrix A [n, n]
// Output: Matrix A^(-1) [n, n]

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

@group(0) @binding(0) var<storage, read_write> augmented_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> inverse_matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> pivot_info: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;
@group(0) @binding(4) var<uniform> metadata: LinalgMetadata;

// Kernel for initializing the augmented matrix [A|I]
@compute @workgroup_size(16, 16, 1)
fn initialize_augmented(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= n || col >= 2u * n) {
        return;
    }
    
    let idx = row * 2u * n + col;
    
    if (col < n) {
        // Left half: copy original matrix A
        let original_idx = row * n + col;
        augmented_matrix[idx] = augmented_matrix[original_idx];
    } else {
        // Right half: identity matrix I
        let identity_col = col - n;
        if (row == identity_col) {
            augmented_matrix[idx] = 1.0;
        } else {
            augmented_matrix[idx] = 0.0;
        }
    }
}

// Kernel for finding pivot element in column k
@compute @workgroup_size(256, 1, 1)
fn find_pivot(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let thread_id = global_id.x;
    let k = pivot_info[0]; // Current column index
    
    if (thread_id >= n || k >= n) {
        return;
    }
    
    // Find maximum absolute value in column k from diagonal down
    var max_val = abs(augmented_matrix[k * 2u * n + k]);
    var max_row = k;
    
    for (var i = k + 1u; i < n; i = i + 1u) {
        let val = abs(augmented_matrix[i * 2u * n + k]);
        if (val > max_val) {
            max_val = val;
            max_row = i;
        }
    }
    
    // Store pivot row index
    if (thread_id == 0u) {
        pivot_info[1] = max_row;
        
        // Check for singularity
        if (max_val < metadata.tolerance) {
            status[0] = 1u; // Mark as singular
        }
    }
}

// Kernel for swapping rows in the augmented matrix
@compute @workgroup_size(256, 1, 1)
fn swap_rows(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let col = global_id.x;
    let k = pivot_info[0]; // Current column index
    let pivot_row = pivot_info[1]; // Pivot row index
    
    if (col >= 2u * n || k >= n) {
        return;
    }
    
    // Swap rows k and pivot_row in augmented matrix
    if (k != pivot_row) {
        let idx_k = k * 2u * n + col;
        let idx_pivot = pivot_row * 2u * n + col;
        
        let temp = augmented_matrix[idx_k];
        augmented_matrix[idx_k] = augmented_matrix[idx_pivot];
        augmented_matrix[idx_pivot] = temp;
    }
}

// Kernel for scaling the pivot row
@compute @workgroup_size(256, 1, 1)
fn scale_pivot_row(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let col = global_id.x;
    let k = pivot_info[0]; // Current column index
    
    if (col >= 2u * n || k >= n) {
        return;
    }
    
    // Scale row k by 1/pivot_element
    let pivot_idx = k * 2u * n + k;
    let pivot_val = augmented_matrix[pivot_idx];
    
    if (abs(pivot_val) < metadata.tolerance) {
        return; // Skip if pivot is too small
    }
    
    let row_idx = k * 2u * n + col;
    augmented_matrix[row_idx] = augmented_matrix[row_idx] / pivot_val;
}

// Kernel for eliminating column k in all other rows
@compute @workgroup_size(16, 16, 1)
fn eliminate_column(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let row = global_id.y;
    let col = global_id.x;
    let k = pivot_info[0]; // Current column index
    
    if (row >= n || col >= 2u * n || k >= n) {
        return;
    }
    
    // Skip the pivot row
    if (row == k) {
        return;
    }
    
    // Get elimination factor
    let factor_idx = row * 2u * n + k;
    let factor = augmented_matrix[factor_idx];
    
    // Eliminate element
    let curr_idx = row * 2u * n + col;
    let pivot_row_idx = k * 2u * n + col;
    augmented_matrix[curr_idx] = augmented_matrix[curr_idx] - factor * augmented_matrix[pivot_row_idx];
}

// Kernel for extracting the inverse matrix from the right half
@compute @workgroup_size(16, 16, 1)
fn extract_inverse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = metadata.rows_a;
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= n || col >= n) {
        return;
    }
    
    // Extract from right half of augmented matrix
    let aug_idx = row * 2u * n + n + col;
    let inv_idx = row * n + col;
    
    inverse_matrix[inv_idx] = augmented_matrix[aug_idx];
}

// Main kernel for initialization
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        status[0] = 0u; // Initialize status as success
    }
    
    // The actual Gauss-Jordan elimination is handled by multiple kernel dispatches
    // from the host code due to synchronization requirements
}