// Multi-type binary operation compute shaders
// Supports f32, i32, and f64 data types

// f32 operations
@group(0) @binding(0) var<storage, read> input_a_f32: array<f32>;
@group(0) @binding(1) var<storage, read> input_b_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_f32: array<f32>;

// i32 operations
@group(0) @binding(3) var<storage, read> input_a_i32: array<i32>;
@group(0) @binding(4) var<storage, read> input_b_i32: array<i32>;
@group(0) @binding(5) var<storage, read_write> output_i32: array<i32>;

// f64 operations (if supported by device)
@group(0) @binding(6) var<storage, read> input_a_f64: array<f64>;
@group(0) @binding(7) var<storage, read> input_b_f64: array<f64>;
@group(0) @binding(8) var<storage, read_write> output_f64: array<f64>;

// Broadcasting metadata
@group(0) @binding(9) var<storage, read> shape_metadata: array<u32>;

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

// f32 arithmetic operations

@compute @workgroup_size(64)
fn add_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_f32[index] = input_a_f32[a_idx] + input_b_f32[b_idx];
}

@compute @workgroup_size(64)
fn sub_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_f32[index] = input_a_f32[a_idx] - input_b_f32[b_idx];
}

@compute @workgroup_size(64)
fn mul_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_f32[index] = input_a_f32[a_idx] * input_b_f32[b_idx];
}

@compute @workgroup_size(64)
fn div_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    // Add safe division with NaN handling
    let denominator = input_b_f32[b_idx];
    if (abs(denominator) < 1e-7) {
        output_f32[index] = select(f32(1.0 / 0.0), f32(-1.0 / 0.0), input_a_f32[a_idx] >= 0.0); // +/-Inf
    } else {
        output_f32[index] = input_a_f32[a_idx] / denominator;
    }
}

@compute @workgroup_size(64)
fn pow_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_f32[index] = pow(input_a_f32[a_idx], input_b_f32[b_idx]);
}

// i32 arithmetic operations

@compute @workgroup_size(64)
fn add_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_i32[index] = input_a_i32[a_idx] + input_b_i32[b_idx];
}

@compute @workgroup_size(64)
fn sub_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_i32[index] = input_a_i32[a_idx] - input_b_i32[b_idx];
}

@compute @workgroup_size(64)
fn mul_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_i32[index] = input_a_i32[a_idx] * input_b_i32[b_idx];
}

@compute @workgroup_size(64)
fn div_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    // Integer division with zero check
    let denominator = input_b_i32[b_idx];
    if (denominator == 0) {
        output_i32[index] = 0; // or could be max_value/min_value
    } else {
        output_i32[index] = input_a_i32[a_idx] / denominator;
    }
}

@compute @workgroup_size(64)
fn mod_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i32)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    let denominator = input_b_i32[b_idx];
    if (denominator == 0) {
        output_i32[index] = 0;
    } else {
        output_i32[index] = input_a_i32[a_idx] % denominator;
    }
}

// f64 arithmetic operations (if f64 is supported)

@compute @workgroup_size(64)
fn add_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f64)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_f64[index] = input_a_f64[a_idx] + input_b_f64[b_idx];
}

@compute @workgroup_size(64)
fn sub_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f64)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_f64[index] = input_a_f64[a_idx] - input_b_f64[b_idx];
}

@compute @workgroup_size(64)
fn mul_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f64)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_f64[index] = input_a_f64[a_idx] * input_b_f64[b_idx];
}

@compute @workgroup_size(64)
fn div_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f64)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    let denominator = input_b_f64[b_idx];
    if (abs(denominator) < 1e-15) {
        output_f64[index] = select(f64(1.0 / 0.0), f64(-1.0 / 0.0), input_a_f64[a_idx] >= 0.0);
    } else {
        output_f64[index] = input_a_f64[a_idx] / denominator;
    }
}

@compute @workgroup_size(64)
fn pow_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_f64)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_f64[index] = pow(input_a_f64[a_idx], input_b_f64[b_idx]);
}

// Broadcasting operations for f32

@compute @workgroup_size(64)
fn add_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_f32)) {
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
    
    output_f32[output_idx] = input_a_f32[a_flat_idx] + input_b_f32[b_flat_idx];
}

@compute @workgroup_size(64)
fn sub_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_f32)) {
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
    
    output_f32[output_idx] = input_a_f32[a_flat_idx] - input_b_f32[b_flat_idx];
}

@compute @workgroup_size(64)
fn mul_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_f32)) {
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
    
    output_f32[output_idx] = input_a_f32[a_flat_idx] * input_b_f32[b_flat_idx];
}

@compute @workgroup_size(64)
fn div_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_f32)) {
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
    
    let denominator = input_b_f32[b_flat_idx];
    if (abs(denominator) < 1e-7) {
        output_f32[output_idx] = select(f32(1.0 / 0.0), f32(-1.0 / 0.0), input_a_f32[a_flat_idx] >= 0.0);
    } else {
        output_f32[output_idx] = input_a_f32[a_flat_idx] / denominator;
    }
}

@compute @workgroup_size(64)
fn pow_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_f32)) {
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
    
    output_f32[output_idx] = pow(input_a_f32[a_flat_idx], input_b_f32[b_flat_idx]);
}