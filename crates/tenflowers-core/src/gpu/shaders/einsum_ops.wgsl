// Einstein summation GPU compute shaders
// Supports common tensor contraction patterns

// Input tensors (up to 4 for complex contractions)
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read> input_c: array<f32>;
@group(0) @binding(3) var<storage, read> input_d: array<f32>;

// Output tensor
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Metadata: dimensions, strides, and contraction info
@group(0) @binding(5) var<storage, read> metadata: array<u32>;
// Layout: [a_rank, b_rank, output_rank, a_dims..., b_dims..., output_dims..., 
//          a_strides..., b_strides..., output_strides..., contraction_info...]

// Helper function to compute flat index from multidimensional indices
fn compute_index(indices: ptr<function, array<u32, 8>>, strides: ptr<function, array<u32, 8>>, rank: u32) -> u32 {
    var idx = 0u;
    for (var i = 0u; i < rank; i++) {
        idx += (*indices)[i] * (*strides)[i];
    }
    return idx;
}

// Matrix multiplication: "ij,jk->ik"
@compute @workgroup_size(16, 16)
fn matmul_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let k = global_id.y;
    
    // Extract dimensions from metadata
    let M = metadata[3]; // a_dims[0] (i dimension)
    let N = metadata[4]; // a_dims[1] (j dimension) 
    let K = metadata[6]; // b_dims[1] (k dimension)
    
    if (i >= M || k >= K) {
        return;
    }
    
    var sum = 0.0;
    
    // Compute dot product along j dimension
    for (var j = 0u; j < N; j++) {
        let a_idx = i * N + j; // Row-major: A[i,j]
        let b_idx = j * K + k; // Row-major: B[j,k]
        sum += input_a[a_idx] * input_b[b_idx];
    }
    
    let output_idx = i * K + k; // Row-major: C[i,k]
    output[output_idx] = sum;
}

// Batched matrix multiplication: "bij,bjk->bik"
@compute @workgroup_size(8, 8, 4)
fn batched_matmul_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let i = global_id.y;
    let k = global_id.z;
    
    // Extract dimensions
    let B = metadata[3]; // batch size
    let M = metadata[4]; // i dimension
    let N = metadata[5]; // j dimension (shared)
    let K = metadata[7]; // k dimension
    
    if (b >= B || i >= M || k >= K) {
        return;
    }
    
    var sum = 0.0;
    
    // Compute dot product along j dimension for this batch
    for (var j = 0u; j < N; j++) {
        let a_idx = b * M * N + i * N + j; // A[b,i,j]
        let b_idx = b * N * K + j * K + k; // B[b,j,k]
        sum += input_a[a_idx] * input_b[b_idx];
    }
    
    let output_idx = b * M * K + i * K + k; // C[b,i,k]
    output[output_idx] = sum;
}

// Transpose: "ij->ji"
@compute @workgroup_size(16, 16)
fn transpose_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    
    let M = metadata[3]; // original rows
    let N = metadata[4]; // original cols
    
    if (i >= M || j >= N) {
        return;
    }
    
    let input_idx = i * N + j;   // A[i,j]
    let output_idx = j * M + i;  // B[j,i]
    
    output[output_idx] = input_a[input_idx];
}

// Diagonal extraction: "ii->i"
@compute @workgroup_size(64)
fn diagonal_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    let N = metadata[3]; // matrix dimension (assuming square)
    
    if (i >= N) {
        return;
    }
    
    let input_idx = i * N + i; // A[i,i]
    output[i] = input_a[input_idx];
}

// Element-wise multiplication: "ij,ij->ij"
@compute @workgroup_size(64)
fn elementwise_mul_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    output[idx] = input_a[idx] * input_b[idx];
}

// Element-wise multiplication and sum: "ij,ij->"
@compute @workgroup_size(64)
fn elementwise_mul_sum_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = arrayLength(&input_a);
    
    if (idx >= total_elements) {
        return;
    }
    
    let local_sum = input_a[idx] * input_b[idx];
    
    // Use workgroup shared memory for reduction
    var shared_data: array<f32, 64>;
    let local_idx = global_id.x % 64u;
    shared_data[local_idx] = local_sum;
    
    workgroupBarrier();
    
    // Parallel reduction within workgroup
    var s = 32u;
    while (s > 0u) {
        if (local_idx < s) {
            shared_data[local_idx] += shared_data[local_idx + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    
    // First thread in workgroup writes partial sum
    if (local_idx == 0u) {
        let workgroup_idx = global_id.x / 64u;
        if (workgroup_idx == 0u) {
            output[0] = shared_data[0];
        } else {
            // Atomic add for multiple workgroups
            output[0] += shared_data[0]; // Note: This needs proper atomics
        }
    }
}

// Outer product: "i,j->ij"
@compute @workgroup_size(16, 16)
fn outer_product_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    
    let M = metadata[3]; // size of first vector
    let N = metadata[4]; // size of second vector
    
    if (i >= M || j >= N) {
        return;
    }
    
    let output_idx = i * N + j;
    output[output_idx] = input_a[i] * input_b[j];
}

// Trace (sum of diagonal): "ii->"
@compute @workgroup_size(64)
fn trace_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let N = metadata[3]; // matrix dimension
    
    if (i >= N) {
        return;
    }
    
    let diag_element = input_a[i * N + i];
    
    // Use shared memory for reduction
    var shared_data: array<f32, 64>;
    let local_idx = i % 64u;
    shared_data[local_idx] = diag_element;
    
    workgroupBarrier();
    
    // Parallel reduction
    var s = 32u;
    while (s > 0u) {
        if (local_idx < s) {
            shared_data[local_idx] += shared_data[local_idx + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    
    if (local_idx == 0u) {
        let workgroup_idx = i / 64u;
        if (workgroup_idx == 0u) {
            output[0] = shared_data[0];
        } else {
            output[0] += shared_data[0]; // Note: Needs proper atomics
        }
    }
}

// Sum along specific axes: "ijk->ik" (sum over j)
@compute @workgroup_size(8, 8, 8)
fn sum_axis_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let k = global_id.y;
    
    let I = metadata[3]; // i dimension
    let J = metadata[4]; // j dimension (summed over)
    let K = metadata[5]; // k dimension
    
    if (i >= I || k >= K) {
        return;
    }
    
    var sum = 0.0;
    
    // Sum over j dimension
    for (var j = 0u; j < J; j++) {
        let input_idx = i * J * K + j * K + k; // A[i,j,k]
        sum += input_a[input_idx];
    }
    
    let output_idx = i * K + k; // B[i,k]
    output[output_idx] = sum;
}

// Vector dot product: "i,i->"
@compute @workgroup_size(64)
fn vector_dot_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let N = metadata[3]; // vector length
    
    if (i >= N) {
        return;
    }
    
    let product = input_a[i] * input_b[i];
    
    // Use shared memory for reduction
    var shared_data: array<f32, 64>;
    let local_idx = i % 64u;
    shared_data[local_idx] = product;
    
    workgroupBarrier();
    
    // Parallel reduction
    var s = 32u;
    while (s > 0u) {
        if (local_idx < s) {
            shared_data[local_idx] += shared_data[local_idx + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    
    if (local_idx == 0u) {
        let workgroup_idx = i / 64u;
        if (workgroup_idx == 0u) {
            output[0] = shared_data[0];
        } else {
            output[0] += shared_data[0]; // Note: Needs proper atomics
        }
    }
}

// General tensor contraction with arbitrary indices
// This is a more complex kernel that interprets contraction metadata
@compute @workgroup_size(64)
fn general_einsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let total_output_elements = metadata[0]; // First metadata element is output size
    
    if (thread_id >= total_output_elements) {
        return;
    }
    
    // This would require more complex metadata parsing
    // For now, implement as a placeholder that can be extended
    // The metadata would include:
    // - Input/output dimension mapping
    // - Contraction patterns
    // - Stride information
    
    // Implementation would:
    // 1. Decode output index to multidimensional coordinates
    // 2. Map to input coordinates based on contraction pattern
    // 3. Perform the required summation/multiplication
    
    // For demonstration, just copy input to output
    output[thread_id] = input_a[thread_id % arrayLength(&input_a)];
}