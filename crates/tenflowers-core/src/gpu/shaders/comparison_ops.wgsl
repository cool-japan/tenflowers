// Comparison operation compute shaders
// Supports multiple data types through specialization

// f32 comparison operations
@group(0) @binding(0) var<storage, read> input_a_f32: array<f32>;
@group(0) @binding(1) var<storage, read> input_b_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_bool: array<u32>; // Boolean results stored as u32 (converted to u8 in Rust)

// i32 comparison operations  
@group(0) @binding(3) var<storage, read> input_a_i32: array<i32>;
@group(0) @binding(4) var<storage, read> input_b_i32: array<i32>;

// i64 comparison operations
@group(0) @binding(5) var<storage, read> input_a_i64: array<i64>;
@group(0) @binding(6) var<storage, read> input_b_i64: array<i64>;

// f64 comparison operations
@group(0) @binding(7) var<storage, read> input_a_f64: array<f64>;
@group(0) @binding(8) var<storage, read> input_b_f64: array<f64>;

// i16 comparison operations
@group(0) @binding(9) var<storage, read> input_a_i16: array<i32>; // WGSL doesn't have i16, use i32
@group(0) @binding(10) var<storage, read> input_b_i16: array<i32>;

// u16 comparison operations  
@group(0) @binding(11) var<storage, read> input_a_u16: array<u32>; // WGSL doesn't have u16, use u32
@group(0) @binding(12) var<storage, read> input_b_u16: array<u32>;

// i8 comparison operations
@group(0) @binding(13) var<storage, read> input_a_i8: array<i32>; // WGSL doesn't have i8, use i32
@group(0) @binding(14) var<storage, read> input_b_i8: array<i32>;

// u8 comparison operations
@group(0) @binding(15) var<storage, read> input_a_u8: array<u32>; // WGSL doesn't have u8, use u32
@group(0) @binding(16) var<storage, read> input_b_u8: array<u32>;

// u32 comparison operations
@group(0) @binding(17) var<storage, read> input_a_u32: array<u32>;
@group(0) @binding(18) var<storage, read> input_b_u32: array<u32>;

// u64 comparison operations
@group(0) @binding(19) var<storage, read> input_a_u64: array<u64>;
@group(0) @binding(20) var<storage, read> input_b_u64: array<u64>;

// Broadcasting metadata
@group(0) @binding(21) var<storage, read> shape_metadata: array<u32>;

// Helper functions for broadcasting (same as binary_ops.wgsl)
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

// f32 comparison operations

@compute @workgroup_size(64)
fn eq_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_bool[index] = select(0u, 1u, input_a_f32[a_idx] == input_b_f32[b_idx]);
}

@compute @workgroup_size(64)
fn ne_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_bool[index] = select(0u, 1u, input_a_f32[a_idx] != input_b_f32[b_idx]);
}

@compute @workgroup_size(64)
fn lt_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_bool[index] = select(0u, 1u, input_a_f32[a_idx] < input_b_f32[b_idx]);
}

@compute @workgroup_size(64)
fn le_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_bool[index] = select(0u, 1u, input_a_f32[a_idx] <= input_b_f32[b_idx]);
}

@compute @workgroup_size(64)
fn gt_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_bool[index] = select(0u, 1u, input_a_f32[a_idx] > input_b_f32[b_idx]);
}

@compute @workgroup_size(64)
fn ge_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f32);
    let b_idx = index % arrayLength(&input_b_f32);
    
    output_bool[index] = select(0u, 1u, input_a_f32[a_idx] >= input_b_f32[b_idx]);
}

// i32 comparison operations

@compute @workgroup_size(64)
fn eq_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_bool[index] = select(0u, 1u, input_a_i32[a_idx] == input_b_i32[b_idx]);
}

@compute @workgroup_size(64)
fn ne_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_bool[index] = select(0u, 1u, input_a_i32[a_idx] != input_b_i32[b_idx]);
}

@compute @workgroup_size(64)
fn lt_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_bool[index] = select(0u, 1u, input_a_i32[a_idx] < input_b_i32[b_idx]);
}

@compute @workgroup_size(64)
fn le_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_bool[index] = select(0u, 1u, input_a_i32[a_idx] <= input_b_i32[b_idx]);
}

@compute @workgroup_size(64)
fn gt_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_bool[index] = select(0u, 1u, input_a_i32[a_idx] > input_b_i32[b_idx]);
}

@compute @workgroup_size(64)
fn ge_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i32);
    let b_idx = index % arrayLength(&input_b_i32);
    
    output_bool[index] = select(0u, 1u, input_a_i32[a_idx] >= input_b_i32[b_idx]);
}

// Broadcasting versions for f32

@compute @workgroup_size(64)
fn eq_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f32[a_flat_idx] == input_b_f32[b_flat_idx]);
}

@compute @workgroup_size(64)
fn ne_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f32[a_flat_idx] != input_b_f32[b_flat_idx]);
}

@compute @workgroup_size(64)
fn lt_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f32[a_flat_idx] < input_b_f32[b_flat_idx]);
}

@compute @workgroup_size(64)
fn le_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f32[a_flat_idx] <= input_b_f32[b_flat_idx]);
}

@compute @workgroup_size(64)
fn gt_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f32[a_flat_idx] > input_b_f32[b_flat_idx]);
}

@compute @workgroup_size(64)
fn ge_f32_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f32[a_flat_idx] >= input_b_f32[b_flat_idx]);
}

// i64 comparison operations

@compute @workgroup_size(64)
fn eq_i64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i64);
    let b_idx = index % arrayLength(&input_b_i64);
    
    output_bool[index] = select(0u, 1u, input_a_i64[a_idx] == input_b_i64[b_idx]);
}

@compute @workgroup_size(64)
fn ne_i64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i64);
    let b_idx = index % arrayLength(&input_b_i64);
    
    output_bool[index] = select(0u, 1u, input_a_i64[a_idx] != input_b_i64[b_idx]);
}

@compute @workgroup_size(64)
fn lt_i64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i64);
    let b_idx = index % arrayLength(&input_b_i64);
    
    output_bool[index] = select(0u, 1u, input_a_i64[a_idx] < input_b_i64[b_idx]);
}

@compute @workgroup_size(64)
fn le_i64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i64);
    let b_idx = index % arrayLength(&input_b_i64);
    
    output_bool[index] = select(0u, 1u, input_a_i64[a_idx] <= input_b_i64[b_idx]);
}

@compute @workgroup_size(64)
fn gt_i64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i64);
    let b_idx = index % arrayLength(&input_b_i64);
    
    output_bool[index] = select(0u, 1u, input_a_i64[a_idx] > input_b_i64[b_idx]);
}

@compute @workgroup_size(64)
fn ge_i64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i64);
    let b_idx = index % arrayLength(&input_b_i64);
    
    output_bool[index] = select(0u, 1u, input_a_i64[a_idx] >= input_b_i64[b_idx]);
}

// f64 comparison operations

@compute @workgroup_size(64)
fn eq_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_bool[index] = select(0u, 1u, input_a_f64[a_idx] == input_b_f64[b_idx]);
}

@compute @workgroup_size(64)
fn ne_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_bool[index] = select(0u, 1u, input_a_f64[a_idx] != input_b_f64[b_idx]);
}

@compute @workgroup_size(64)
fn lt_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_bool[index] = select(0u, 1u, input_a_f64[a_idx] < input_b_f64[b_idx]);
}

@compute @workgroup_size(64)
fn le_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_bool[index] = select(0u, 1u, input_a_f64[a_idx] <= input_b_f64[b_idx]);
}

@compute @workgroup_size(64)
fn gt_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_bool[index] = select(0u, 1u, input_a_f64[a_idx] > input_b_f64[b_idx]);
}

@compute @workgroup_size(64)
fn ge_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_f64);
    let b_idx = index % arrayLength(&input_b_f64);
    
    output_bool[index] = select(0u, 1u, input_a_f64[a_idx] >= input_b_f64[b_idx]);
}

// u32 comparison operations

@compute @workgroup_size(64)
fn eq_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u32);
    let b_idx = index % arrayLength(&input_b_u32);
    
    output_bool[index] = select(0u, 1u, input_a_u32[a_idx] == input_b_u32[b_idx]);
}

@compute @workgroup_size(64)
fn ne_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u32);
    let b_idx = index % arrayLength(&input_b_u32);
    
    output_bool[index] = select(0u, 1u, input_a_u32[a_idx] != input_b_u32[b_idx]);
}

@compute @workgroup_size(64)
fn lt_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u32);
    let b_idx = index % arrayLength(&input_b_u32);
    
    output_bool[index] = select(0u, 1u, input_a_u32[a_idx] < input_b_u32[b_idx]);
}

@compute @workgroup_size(64)
fn le_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u32);
    let b_idx = index % arrayLength(&input_b_u32);
    
    output_bool[index] = select(0u, 1u, input_a_u32[a_idx] <= input_b_u32[b_idx]);
}

@compute @workgroup_size(64)
fn gt_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u32);
    let b_idx = index % arrayLength(&input_b_u32);
    
    output_bool[index] = select(0u, 1u, input_a_u32[a_idx] > input_b_u32[b_idx]);
}

@compute @workgroup_size(64)
fn ge_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u32);
    let b_idx = index % arrayLength(&input_b_u32);
    
    output_bool[index] = select(0u, 1u, input_a_u32[a_idx] >= input_b_u32[b_idx]);
}

// u64 comparison operations

@compute @workgroup_size(64)
fn eq_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u64);
    let b_idx = index % arrayLength(&input_b_u64);
    
    output_bool[index] = select(0u, 1u, input_a_u64[a_idx] == input_b_u64[b_idx]);
}

@compute @workgroup_size(64)
fn ne_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u64);
    let b_idx = index % arrayLength(&input_b_u64);
    
    output_bool[index] = select(0u, 1u, input_a_u64[a_idx] != input_b_u64[b_idx]);
}

@compute @workgroup_size(64)
fn lt_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u64);
    let b_idx = index % arrayLength(&input_b_u64);
    
    output_bool[index] = select(0u, 1u, input_a_u64[a_idx] < input_b_u64[b_idx]);
}

@compute @workgroup_size(64)
fn le_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u64);
    let b_idx = index % arrayLength(&input_b_u64);
    
    output_bool[index] = select(0u, 1u, input_a_u64[a_idx] <= input_b_u64[b_idx]);
}

@compute @workgroup_size(64)
fn gt_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u64);
    let b_idx = index % arrayLength(&input_b_u64);
    
    output_bool[index] = select(0u, 1u, input_a_u64[a_idx] > input_b_u64[b_idx]);
}

@compute @workgroup_size(64)
fn ge_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u64);
    let b_idx = index % arrayLength(&input_b_u64);
    
    output_bool[index] = select(0u, 1u, input_a_u64[a_idx] >= input_b_u64[b_idx]);
}

// Smaller integer types (i16, u16, i8, u8) - using expanded 32-bit storage
// These use masking to handle the smaller value ranges

@compute @workgroup_size(64)
fn eq_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i16);
    let b_idx = index % arrayLength(&input_b_i16);
    
    // Mask to 16-bit signed values
    let a_val = input_a_i16[a_idx] & 0xFFFF;
    let b_val = input_b_i16[b_idx] & 0xFFFF;
    let a_signed = select(a_val, a_val | 0xFFFF0000, (a_val & 0x8000) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFF0000, (b_val & 0x8000) != 0u);
    
    output_bool[index] = select(0u, 1u, a_signed == b_signed);
}

@compute @workgroup_size(64)
fn ne_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i16);
    let b_idx = index % arrayLength(&input_b_i16);
    
    let a_val = input_a_i16[a_idx] & 0xFFFF;
    let b_val = input_b_i16[b_idx] & 0xFFFF;
    let a_signed = select(a_val, a_val | 0xFFFF0000, (a_val & 0x8000) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFF0000, (b_val & 0x8000) != 0u);
    
    output_bool[index] = select(0u, 1u, a_signed != b_signed);
}

@compute @workgroup_size(64)
fn lt_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i16);
    let b_idx = index % arrayLength(&input_b_i16);
    
    let a_val = input_a_i16[a_idx] & 0xFFFF;
    let b_val = input_b_i16[b_idx] & 0xFFFF;
    let a_signed = select(a_val, a_val | 0xFFFF0000, (a_val & 0x8000) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFF0000, (b_val & 0x8000) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) < i32(b_signed));
}

@compute @workgroup_size(64)
fn le_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i16);
    let b_idx = index % arrayLength(&input_b_i16);
    
    let a_val = input_a_i16[a_idx] & 0xFFFF;
    let b_val = input_b_i16[b_idx] & 0xFFFF;
    let a_signed = select(a_val, a_val | 0xFFFF0000, (a_val & 0x8000) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFF0000, (b_val & 0x8000) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) <= i32(b_signed));
}

@compute @workgroup_size(64)
fn gt_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i16);
    let b_idx = index % arrayLength(&input_b_i16);
    
    let a_val = input_a_i16[a_idx] & 0xFFFF;
    let b_val = input_b_i16[b_idx] & 0xFFFF;
    let a_signed = select(a_val, a_val | 0xFFFF0000, (a_val & 0x8000) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFF0000, (b_val & 0x8000) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) > i32(b_signed));
}

@compute @workgroup_size(64)
fn ge_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i16);
    let b_idx = index % arrayLength(&input_b_i16);
    
    let a_val = input_a_i16[a_idx] & 0xFFFF;
    let b_val = input_b_i16[b_idx] & 0xFFFF;
    let a_signed = select(a_val, a_val | 0xFFFF0000, (a_val & 0x8000) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFF0000, (b_val & 0x8000) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) >= i32(b_signed));
}

// u16 comparison operations

@compute @workgroup_size(64)
fn eq_u16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u16);
    let b_idx = index % arrayLength(&input_b_u16);
    
    let a_val = input_a_u16[a_idx] & 0xFFFF;
    let b_val = input_b_u16[b_idx] & 0xFFFF;
    
    output_bool[index] = select(0u, 1u, a_val == b_val);
}

@compute @workgroup_size(64)
fn ne_u16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u16);
    let b_idx = index % arrayLength(&input_b_u16);
    
    let a_val = input_a_u16[a_idx] & 0xFFFF;
    let b_val = input_b_u16[b_idx] & 0xFFFF;
    
    output_bool[index] = select(0u, 1u, a_val != b_val);
}

@compute @workgroup_size(64)
fn lt_u16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u16);
    let b_idx = index % arrayLength(&input_b_u16);
    
    let a_val = input_a_u16[a_idx] & 0xFFFF;
    let b_val = input_b_u16[b_idx] & 0xFFFF;
    
    output_bool[index] = select(0u, 1u, a_val < b_val);
}

@compute @workgroup_size(64)
fn le_u16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u16);
    let b_idx = index % arrayLength(&input_b_u16);
    
    let a_val = input_a_u16[a_idx] & 0xFFFF;
    let b_val = input_b_u16[b_idx] & 0xFFFF;
    
    output_bool[index] = select(0u, 1u, a_val <= b_val);
}

@compute @workgroup_size(64)
fn gt_u16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u16);
    let b_idx = index % arrayLength(&input_b_u16);
    
    let a_val = input_a_u16[a_idx] & 0xFFFF;
    let b_val = input_b_u16[b_idx] & 0xFFFF;
    
    output_bool[index] = select(0u, 1u, a_val > b_val);
}

@compute @workgroup_size(64)
fn ge_u16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u16);
    let b_idx = index % arrayLength(&input_b_u16);
    
    let a_val = input_a_u16[a_idx] & 0xFFFF;
    let b_val = input_b_u16[b_idx] & 0xFFFF;
    
    output_bool[index] = select(0u, 1u, a_val >= b_val);
}

// i8 comparison operations

@compute @workgroup_size(64)
fn eq_i8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i8);
    let b_idx = index % arrayLength(&input_b_i8);
    
    let a_val = input_a_i8[a_idx] & 0xFF;
    let b_val = input_b_i8[b_idx] & 0xFF;
    let a_signed = select(a_val, a_val | 0xFFFFFF00, (a_val & 0x80) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFFFF00, (b_val & 0x80) != 0u);
    
    output_bool[index] = select(0u, 1u, a_signed == b_signed);
}

@compute @workgroup_size(64)
fn ne_i8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i8);
    let b_idx = index % arrayLength(&input_b_i8);
    
    let a_val = input_a_i8[a_idx] & 0xFF;
    let b_val = input_b_i8[b_idx] & 0xFF;
    let a_signed = select(a_val, a_val | 0xFFFFFF00, (a_val & 0x80) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFFFF00, (b_val & 0x80) != 0u);
    
    output_bool[index] = select(0u, 1u, a_signed != b_signed);
}

@compute @workgroup_size(64)
fn lt_i8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i8);
    let b_idx = index % arrayLength(&input_b_i8);
    
    let a_val = input_a_i8[a_idx] & 0xFF;
    let b_val = input_b_i8[b_idx] & 0xFF;
    let a_signed = select(a_val, a_val | 0xFFFFFF00, (a_val & 0x80) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFFFF00, (b_val & 0x80) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) < i32(b_signed));
}

@compute @workgroup_size(64)
fn le_i8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i8);
    let b_idx = index % arrayLength(&input_b_i8);
    
    let a_val = input_a_i8[a_idx] & 0xFF;
    let b_val = input_b_i8[b_idx] & 0xFF;
    let a_signed = select(a_val, a_val | 0xFFFFFF00, (a_val & 0x80) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFFFF00, (b_val & 0x80) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) <= i32(b_signed));
}

@compute @workgroup_size(64)
fn gt_i8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i8);
    let b_idx = index % arrayLength(&input_b_i8);
    
    let a_val = input_a_i8[a_idx] & 0xFF;
    let b_val = input_b_i8[b_idx] & 0xFF;
    let a_signed = select(a_val, a_val | 0xFFFFFF00, (a_val & 0x80) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFFFF00, (b_val & 0x80) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) > i32(b_signed));
}

@compute @workgroup_size(64)
fn ge_i8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_i8);
    let b_idx = index % arrayLength(&input_b_i8);
    
    let a_val = input_a_i8[a_idx] & 0xFF;
    let b_val = input_b_i8[b_idx] & 0xFF;
    let a_signed = select(a_val, a_val | 0xFFFFFF00, (a_val & 0x80) != 0u);
    let b_signed = select(b_val, b_val | 0xFFFFFF00, (b_val & 0x80) != 0u);
    
    output_bool[index] = select(0u, 1u, i32(a_signed) >= i32(b_signed));
}

// u8 comparison operations

@compute @workgroup_size(64)
fn eq_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u8);
    let b_idx = index % arrayLength(&input_b_u8);
    
    let a_val = input_a_u8[a_idx] & 0xFF;
    let b_val = input_b_u8[b_idx] & 0xFF;
    
    output_bool[index] = select(0u, 1u, a_val == b_val);
}

@compute @workgroup_size(64)
fn ne_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u8);
    let b_idx = index % arrayLength(&input_b_u8);
    
    let a_val = input_a_u8[a_idx] & 0xFF;
    let b_val = input_b_u8[b_idx] & 0xFF;
    
    output_bool[index] = select(0u, 1u, a_val != b_val);
}

@compute @workgroup_size(64)
fn lt_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u8);
    let b_idx = index % arrayLength(&input_b_u8);
    
    let a_val = input_a_u8[a_idx] & 0xFF;
    let b_val = input_b_u8[b_idx] & 0xFF;
    
    output_bool[index] = select(0u, 1u, a_val < b_val);
}

@compute @workgroup_size(64)
fn le_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u8);
    let b_idx = index % arrayLength(&input_b_u8);
    
    let a_val = input_a_u8[a_idx] & 0xFF;
    let b_val = input_b_u8[b_idx] & 0xFF;
    
    output_bool[index] = select(0u, 1u, a_val <= b_val);
}

@compute @workgroup_size(64)
fn gt_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u8);
    let b_idx = index % arrayLength(&input_b_u8);
    
    let a_val = input_a_u8[a_idx] & 0xFF;
    let b_val = input_b_u8[b_idx] & 0xFF;
    
    output_bool[index] = select(0u, 1u, a_val > b_val);
}

@compute @workgroup_size(64)
fn ge_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_bool)) {
        return;
    }
    
    let a_idx = index % arrayLength(&input_a_u8);
    let b_idx = index % arrayLength(&input_b_u8);
    
    let a_val = input_a_u8[a_idx] & 0xFF;
    let b_val = input_b_u8[b_idx] & 0xFF;
    
    output_bool[index] = select(0u, 1u, a_val >= b_val);
}

// f64 broadcast comparison operations

@compute @workgroup_size(64)
fn eq_f64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f64[a_flat_idx] == input_b_f64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn ne_f64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f64[a_flat_idx] != input_b_f64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn lt_f64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f64[a_flat_idx] < input_b_f64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn le_f64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f64[a_flat_idx] <= input_b_f64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn gt_f64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f64[a_flat_idx] > input_b_f64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn ge_f64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_f64[a_flat_idx] >= input_b_f64[b_flat_idx]);
}

// i64 broadcast comparison operations

@compute @workgroup_size(64)
fn eq_i64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_i64[a_flat_idx] == input_b_i64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn ne_i64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_i64[a_flat_idx] != input_b_i64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn lt_i64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_i64[a_flat_idx] < input_b_i64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn le_i64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_i64[a_flat_idx] <= input_b_i64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn gt_i64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_i64[a_flat_idx] > input_b_i64[b_flat_idx]);
}

@compute @workgroup_size(64)
fn ge_i64_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= arrayLength(&output_bool)) {
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
    
    output_bool[output_idx] = select(0u, 1u, input_a_i64[a_flat_idx] >= input_b_i64[b_flat_idx]);
}