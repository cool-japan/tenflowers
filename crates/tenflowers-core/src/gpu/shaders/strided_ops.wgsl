// Strided tensor operations compute shader
// Handles non-contiguous tensor view materialization

struct StridedInfo {
    ndim: u32,
    total_elements: u32,
    offset: u32,
    element_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: StridedInfo;
@group(0) @binding(3) var<storage, read> shape: array<u32>;
@group(0) @binding(4) var<storage, read> strides: array<u32>;

// Convert flat index to multi-dimensional coordinates
fn flat_to_coords(flat_idx: u32, shape_arr: ptr<storage, array<u32>, read>, ndim: u32) -> array<u32, 8> {
    var coords: array<u32, 8>;
    var remaining = flat_idx;
    
    for (var i = 0u; i < ndim; i = i + 1u) {
        let dim_idx = ndim - 1u - i;
        coords[dim_idx] = remaining % shape_arr[dim_idx];
        remaining = remaining / shape_arr[dim_idx];
    }
    
    return coords;
}

// Convert multi-dimensional coordinates to strided index
fn coords_to_strided_index(coords: ptr<function, array<u32, 8>>, strides_arr: ptr<storage, array<u32>, read>, ndim: u32, offset: u32) -> u32 {
    var strided_idx = offset;
    
    for (var i = 0u; i < ndim; i = i + 1u) {
        strided_idx = strided_idx + coords[i] * strides_arr[i];
    }
    
    return strided_idx;
}

@compute @workgroup_size(64)
fn strided_materialize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_elements) {
        return;
    }
    
    // Convert flat output index to multi-dimensional coordinates
    var coords = flat_to_coords(out_index, &shape, info.ndim);
    
    // Convert coordinates to strided input index
    let strided_index = coords_to_strided_index(&coords, &strides, info.ndim, info.offset);
    
    // Copy data from strided input to contiguous output
    if (strided_index < arrayLength(&input)) {
        output[out_index] = input[strided_index];
    } else {
        output[out_index] = 0.0; // Zero-pad if out of bounds
    }
}

// Additional shader for strided assignment (for future use)
@compute @workgroup_size(64)
fn strided_assign(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_elements) {
        return;
    }
    
    // Convert flat input index to multi-dimensional coordinates
    var coords = flat_to_coords(out_index, &shape, info.ndim);
    
    // Convert coordinates to strided output index
    let strided_index = coords_to_strided_index(&coords, &strides, info.ndim, info.offset);
    
    // Copy data from contiguous input to strided output
    if (strided_index < arrayLength(&output)) {
        output[strided_index] = input[out_index];
    }
}