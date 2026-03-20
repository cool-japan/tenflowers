// Binary operation compute shaders for i32 type

@group(0) @binding(0) var<storage, read> input_a: array<i32>;
@group(0) @binding(1) var<storage, read> input_b: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;

// Enhanced broadcasting bindings
@group(0) @binding(3) var<storage, read> shape_metadata: array<u32>; // [a_rank, b_rank, output_rank, a_shape..., b_shape..., output_shape...]

// Helper functions for broadcasting

// Compute flat index from multidimensional indices
fn compute_flat_index_broadcast(indices: ptr<function, array<u32, 8>>, shape_offset: u32, rank: u32) -> u32 {
    var flat_idx = 0u;
    var stride = 1u;
    
    for (var i = 0u; i < rank; i++) {
        let dim_idx = rank - 1u - i;
        flat_idx += (*indices)[dim_idx] * stride;
        stride *= shape_metadata[shape_offset + dim_idx];
    }
    
    return flat_idx;
}

// Compute multidimensional indices from flat index
fn compute_multi_index_broadcast(flat_idx: u32, indices: ptr<function, array<u32, 8>>, shape_offset: u32, rank: u32) {
    var remaining = flat_idx;
    
    for (var i = 0u; i < rank; i++) {
        let dim_idx = rank - 1u - i;
        (*indices)[dim_idx] = remaining % shape_metadata[shape_offset + dim_idx];
        remaining = remaining / shape_metadata[shape_offset + dim_idx];
    }
}

// Convert output indices to input indices with broadcasting
fn broadcast_indices(output_indices: ptr<function, array<u32, 8>>, input_indices: ptr<function, array<u32, 8>>, 
                     input_shape_offset: u32, input_rank: u32, output_rank: u32) {
    // Initialize input indices to 0
    for (var i = 0u; i < 8u; i++) {
        (*input_indices)[i] = 0u;
    }
    
    // Align the rightmost dimensions and broadcast
    let rank_diff = output_rank - input_rank;
    for (var i = 0u; i < input_rank; i++) {
        let input_dim_idx = i;
        let output_dim_idx = i + rank_diff;
        let input_size = shape_metadata[input_shape_offset + input_dim_idx];
        
        if (input_size == 1u) {
            // Broadcasting: use index 0 for size-1 dimensions
            (*input_indices)[input_dim_idx] = 0u;
        } else {
            // Normal indexing
            (*input_indices)[input_dim_idx] = (*output_indices)[output_dim_idx];
        }
    }
}

// Basic binary operations

// Addition operation
@compute @workgroup_size(64)
fn add_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Support broadcasting by cycling through smaller array
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    output[index] = input_a[a_idx] + input_b[b_idx];
}

// Subtraction operation
@compute @workgroup_size(64)
fn sub_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    output[index] = input_a[a_idx] - input_b[b_idx];
}

// Multiplication operation
@compute @workgroup_size(64)
fn mul_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    output[index] = input_a[a_idx] * input_b[b_idx];
}

// Division operation
@compute @workgroup_size(64)
fn div_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    output[index] = input_a[a_idx] / input_b[b_idx];
}

// Enhanced broadcasting operations

@compute @workgroup_size(64)
fn add_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output)) {
        return;
    }
    
    let a_rank = shape_metadata[0];
    let b_rank = shape_metadata[1];
    let output_rank = shape_metadata[2];
    
    let a_shape_offset = 3u;
    let b_shape_offset = 3u + a_rank;
    let output_shape_offset = 3u + a_rank + b_rank;
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_broadcast(output_idx, &output_indices, output_shape_offset, output_rank);
    
    // Convert to input indices with broadcasting
    var a_indices: array<u32, 8>;
    var b_indices: array<u32, 8>;
    broadcast_indices(&output_indices, &a_indices, a_shape_offset, a_rank, output_rank);
    broadcast_indices(&output_indices, &b_indices, b_shape_offset, b_rank, output_rank);
    
    // Compute flat input indices
    let a_flat_idx = compute_flat_index_broadcast(&a_indices, a_shape_offset, a_rank);
    let b_flat_idx = compute_flat_index_broadcast(&b_indices, b_shape_offset, b_rank);
    
    // Perform operation
    output[output_idx] = input_a[a_flat_idx] + input_b[b_flat_idx];
}

@compute @workgroup_size(64)
fn sub_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output)) {
        return;
    }
    
    let a_rank = shape_metadata[0];
    let b_rank = shape_metadata[1];
    let output_rank = shape_metadata[2];
    
    let a_shape_offset = 3u;
    let b_shape_offset = 3u + a_rank;
    let output_shape_offset = 3u + a_rank + b_rank;
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_broadcast(output_idx, &output_indices, output_shape_offset, output_rank);
    
    // Convert to input indices with broadcasting
    var a_indices: array<u32, 8>;
    var b_indices: array<u32, 8>;
    broadcast_indices(&output_indices, &a_indices, a_shape_offset, a_rank, output_rank);
    broadcast_indices(&output_indices, &b_indices, b_shape_offset, b_rank, output_rank);
    
    // Compute flat input indices
    let a_flat_idx = compute_flat_index_broadcast(&a_indices, a_shape_offset, a_rank);
    let b_flat_idx = compute_flat_index_broadcast(&b_indices, b_shape_offset, b_rank);
    
    // Perform operation
    output[output_idx] = input_a[a_flat_idx] - input_b[b_flat_idx];
}

@compute @workgroup_size(64)
fn mul_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output)) {
        return;
    }
    
    let a_rank = shape_metadata[0];
    let b_rank = shape_metadata[1];
    let output_rank = shape_metadata[2];
    
    let a_shape_offset = 3u;
    let b_shape_offset = 3u + a_rank;
    let output_shape_offset = 3u + a_rank + b_rank;
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_broadcast(output_idx, &output_indices, output_shape_offset, output_rank);
    
    // Convert to input indices with broadcasting
    var a_indices: array<u32, 8>;
    var b_indices: array<u32, 8>;
    broadcast_indices(&output_indices, &a_indices, a_shape_offset, a_rank, output_rank);
    broadcast_indices(&output_indices, &b_indices, b_shape_offset, b_rank, output_rank);
    
    // Compute flat input indices
    let a_flat_idx = compute_flat_index_broadcast(&a_indices, a_shape_offset, a_rank);
    let b_flat_idx = compute_flat_index_broadcast(&b_indices, b_shape_offset, b_rank);
    
    // Perform operation
    output[output_idx] = input_a[a_flat_idx] * input_b[b_flat_idx];
}

@compute @workgroup_size(64)
fn div_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output)) {
        return;
    }
    
    let a_rank = shape_metadata[0];
    let b_rank = shape_metadata[1];
    let output_rank = shape_metadata[2];
    
    let a_shape_offset = 3u;
    let b_shape_offset = 3u + a_rank;
    let output_shape_offset = 3u + a_rank + b_rank;
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_broadcast(output_idx, &output_indices, output_shape_offset, output_rank);
    
    // Convert to input indices with broadcasting
    var a_indices: array<u32, 8>;
    var b_indices: array<u32, 8>;
    broadcast_indices(&output_indices, &a_indices, a_shape_offset, a_rank, output_rank);
    broadcast_indices(&output_indices, &b_indices, b_shape_offset, b_rank, output_rank);
    
    // Compute flat input indices
    let a_flat_idx = compute_flat_index_broadcast(&a_indices, a_shape_offset, a_rank);
    let b_flat_idx = compute_flat_index_broadcast(&b_indices, b_shape_offset, b_rank);
    
    // Perform operation
    output[output_idx] = input_a[a_flat_idx] / input_b[b_flat_idx];
}