// Logical operation compute shaders
// Operates on boolean tensors represented as u32 arrays

@group(0) @binding(0) var<storage, read> input_a: array<u32>; // Boolean inputs stored as u32
@group(0) @binding(1) var<storage, read> input_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>; // Boolean output

// Broadcasting metadata
@group(0) @binding(3) var<storage, read> shape_metadata: array<u32>;

// Helper functions for broadcasting
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

fn compute_multi_index_broadcast(flat_idx: u32, indices: ptr<function, array<u32, 8>>, shape_offset: u32, rank: u32) {
    var remaining = flat_idx;
    
    for (var i = 0u; i < rank; i++) {
        let dim_idx = rank - 1u - i;
        (*indices)[dim_idx] = remaining % shape_metadata[shape_offset + dim_idx];
        remaining = remaining / shape_metadata[shape_offset + dim_idx];
    }
}

fn broadcast_indices(output_indices: ptr<function, array<u32, 8>>, input_indices: ptr<function, array<u32, 8>>, 
                     input_shape_offset: u32, input_rank: u32, output_rank: u32) {
    for (var i = 0u; i < 8u; i++) {
        (*input_indices)[i] = 0u;
    }
    
    let rank_diff = output_rank - input_rank;
    for (var i = 0u; i < input_rank; i++) {
        let input_dim_idx = i;
        let output_dim_idx = i + rank_diff;
        let input_size = shape_metadata[input_shape_offset + input_dim_idx];
        
        if (input_size == 1u) {
            (*input_indices)[input_dim_idx] = 0u;
        } else {
            (*input_indices)[input_dim_idx] = (*output_indices)[output_dim_idx];
        }
    }
}

// Basic logical operations

@compute @workgroup_size(64)
fn and_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    // Convert to boolean, perform AND, convert back to u32
    let a_bool = input_a[a_idx] != 0u;
    let b_bool = input_b[b_idx] != 0u;
    output[index] = select(0u, 1u, a_bool && b_bool);
}

@compute @workgroup_size(64)
fn or_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    let a_bool = input_a[a_idx] != 0u;
    let b_bool = input_b[b_idx] != 0u;
    output[index] = select(0u, 1u, a_bool || b_bool);
}

@compute @workgroup_size(64)
fn xor_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    let a_bool = input_a[a_idx] != 0u;
    let b_bool = input_b[b_idx] != 0u;
    output[index] = select(0u, 1u, a_bool != b_bool);
}

// Unary logical operation (NOT)
@compute @workgroup_size(64)
fn not_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let a_bool = input_a[a_idx] != 0u;
    output[index] = select(1u, 0u, a_bool); // NOT operation
}

// Broadcasting versions

@compute @workgroup_size(64)
fn and_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    var output_indices: array<u32, 8>;
    compute_multi_index_broadcast(output_idx, &output_indices, output_shape_offset, output_rank);
    
    var a_indices: array<u32, 8>;
    var b_indices: array<u32, 8>;
    broadcast_indices(&output_indices, &a_indices, a_shape_offset, a_rank, output_rank);
    broadcast_indices(&output_indices, &b_indices, b_shape_offset, b_rank, output_rank);
    
    let a_flat_idx = compute_flat_index_broadcast(&a_indices, a_shape_offset, a_rank);
    let b_flat_idx = compute_flat_index_broadcast(&b_indices, b_shape_offset, b_rank);
    
    let a_bool = input_a[a_flat_idx] != 0u;
    let b_bool = input_b[b_flat_idx] != 0u;
    output[output_idx] = select(0u, 1u, a_bool && b_bool);
}

@compute @workgroup_size(64)
fn or_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    var output_indices: array<u32, 8>;
    compute_multi_index_broadcast(output_idx, &output_indices, output_shape_offset, output_rank);
    
    var a_indices: array<u32, 8>;
    var b_indices: array<u32, 8>;
    broadcast_indices(&output_indices, &a_indices, a_shape_offset, a_rank, output_rank);
    broadcast_indices(&output_indices, &b_indices, b_shape_offset, b_rank, output_rank);
    
    let a_flat_idx = compute_flat_index_broadcast(&a_indices, a_shape_offset, a_rank);
    let b_flat_idx = compute_flat_index_broadcast(&b_indices, b_shape_offset, b_rank);
    
    let a_bool = input_a[a_flat_idx] != 0u;
    let b_bool = input_b[b_flat_idx] != 0u;
    output[output_idx] = select(0u, 1u, a_bool || b_bool);
}

@compute @workgroup_size(64)
fn xor_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    var output_indices: array<u32, 8>;
    compute_multi_index_broadcast(output_idx, &output_indices, output_shape_offset, output_rank);
    
    var a_indices: array<u32, 8>;
    var b_indices: array<u32, 8>;
    broadcast_indices(&output_indices, &a_indices, a_shape_offset, a_rank, output_rank);
    broadcast_indices(&output_indices, &b_indices, b_shape_offset, b_rank, output_rank);
    
    let a_flat_idx = compute_flat_index_broadcast(&a_indices, a_shape_offset, a_rank);
    let b_flat_idx = compute_flat_index_broadcast(&b_indices, b_shape_offset, b_rank);
    
    let a_bool = input_a[a_flat_idx] != 0u;
    let b_bool = input_b[b_flat_idx] != 0u;
    output[output_idx] = select(0u, 1u, a_bool != b_bool);
}

// Short-circuit optimized operations
// These operations can potentially skip computation when one operand guarantees the result

@compute @workgroup_size(64)
fn and_short_circuit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    // Early exit if first operand is false
    if (input_a[a_idx] == 0u) {
        output[index] = 0u;
        return;
    }
    
    // Check second operand
    output[index] = select(0u, 1u, input_b[b_idx] != 0u);
}

@compute @workgroup_size(64)
fn or_short_circuit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    // Early exit if first operand is true
    if (input_a[a_idx] != 0u) {
        output[index] = 1u;
        return;
    }
    
    // Check second operand
    output[index] = select(0u, 1u, input_b[b_idx] != 0u);
}