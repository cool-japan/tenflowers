// Additional tensor manipulation compute shaders

// Concatenate operation
struct ConcatInfo {
    axis: u32,
    ndim: u32,
    num_inputs: u32,
    total_size: u32,
}

@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> info: ConcatInfo;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;
@group(0) @binding(5) var<storage, read> input1_shape: array<u32>;
@group(0) @binding(6) var<storage, read> input2_shape: array<u32>;

@compute @workgroup_size(64)
fn concat_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % output_shape[i];
        remaining = remaining / output_shape[i];
    }
    
    // Determine which input tensor to read from
    let axis_coord = out_coords[info.axis];
    let input1_axis_size = input1_shape[info.axis];
    
    if (axis_coord < input1_axis_size) {
        // Read from input1
        var in_coords: array<u32, 8>;
        for (var i = 0u; i < info.ndim; i = i + 1u) {
            in_coords[i] = out_coords[i];
        }
        
        // Convert to flat index
        var in_index = 0u;
        var stride = 1u;
        for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
            in_index = in_index + in_coords[i] * stride;
            if (i > 0u) {
                stride = stride * input1_shape[i];
            } else {
                break;
            }
        }
        
        output[out_index] = input1[in_index];
    } else {
        // Read from input2
        var in_coords: array<u32, 8>;
        for (var i = 0u; i < info.ndim; i = i + 1u) {
            if (i == info.axis) {
                in_coords[i] = out_coords[i] - input1_axis_size;
            } else {
                in_coords[i] = out_coords[i];
            }
        }
        
        // Convert to flat index
        var in_index = 0u;
        var stride = 1u;
        for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
            in_index = in_index + in_coords[i] * stride;
            if (i > 0u) {
                stride = stride * input2_shape[i];
            } else {
                break;
            }
        }
        
        output[out_index] = input2[in_index];
    }
}

// Expand dims is just a reshape/view change
@compute @workgroup_size(64)
fn expand_dims_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Expand dims is just a view change, so direct copy
    output[index] = input[index];
}

// Cast operations for different types
@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_i32: array<i32>;

@compute @workgroup_size(64)
fn cast_f32_to_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i32)) {
        return;
    }
    
    output_i32[index] = i32(input_f32[index]);
}

@compute @workgroup_size(64)
fn cast_i32_to_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = f32(input_i32[index]);
}

// Gather operation
struct GatherInfo {
    axis: u32,
    ndim: u32,
    indices_size: u32,
    total_size: u32,
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> info: GatherInfo;
@group(0) @binding(4) var<storage, read> params_shape: array<u32>;
@group(0) @binding(5) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn gather_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % output_shape[i];
        remaining = remaining / output_shape[i];
    }
    
    // Get the index to gather
    let gather_idx = indices[out_coords[info.axis]];
    
    // Build params coordinates
    var params_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            params_coords[i] = u32(gather_idx);
        } else {
            params_coords[i] = out_coords[i];
        }
    }
    
    // Convert to flat index
    var params_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        params_index = params_index + params_coords[i] * stride;
        if (i > 0u) {
            stride = stride * params_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = params[params_index];
}

// Scatter operation
struct ScatterInfo {
    axis: u32,
    ndim: u32,
    updates_size: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> tensor_in: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> updates: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> info: ScatterInfo;
@group(0) @binding(5) var<storage, read> tensor_shape: array<u32>;
@group(0) @binding(6) var<storage, read> updates_shape: array<u32>;

@compute @workgroup_size(64)
fn scatter_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= arrayLength(&output)) {
        return;
    }
    
    // First, copy the original tensor
    output[out_index] = tensor_in[out_index];
    
    // Now we need to check if this position should be updated
    // Convert flat index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % tensor_shape[i];
        remaining = remaining / tensor_shape[i];
    }
    
    // Check all indices to see if any match our axis coordinate
    for (var idx = 0u; idx < info.updates_size; idx = idx + 1u) {
        let scatter_idx = indices[idx];
        if (u32(scatter_idx) == out_coords[info.axis]) {
            // This position should be updated
            // Build updates coordinates
            var update_coords: array<u32, 8>;
            for (var i = 0u; i < info.ndim; i = i + 1u) {
                if (i == info.axis) {
                    update_coords[i] = idx;
                } else {
                    update_coords[i] = out_coords[i];
                }
            }
            
            // Convert to flat index
            var update_index = 0u;
            var stride = 1u;
            for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
                update_index = update_index + update_coords[i] * stride;
                if (i > 0u) {
                    stride = stride * updates_shape[i];
                } else {
                    break;
                }
            }
            
            output[out_index] = updates[update_index];
            break;
        }
    }
}

// Roll operation
struct RollInfo {
    shift: i32,
    axis: i32,  // -1 for flattened roll
    total_size: u32,
    ndim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RollInfo;
@group(0) @binding(3) var<storage, read> shape: array<u32>;

@compute @workgroup_size(64)
fn roll_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    var in_index: u32;
    
    if (info.axis < 0) {
        // Flattened roll
        let size = i32(info.total_size);
        let normalized_shift = ((info.shift % size) + size) % size;
        in_index = u32((i32(out_index) - normalized_shift + size) % size);
    } else {
        // Roll along specific axis
        // Convert flat output index to coordinates
        var out_coords: array<u32, 8>;
        var remaining = out_index;
        for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
            if (i == 0u) {
                out_coords[i] = remaining;
                break;
            }
            out_coords[i] = remaining % shape[i];
            remaining = remaining / shape[i];
        }
        
        // Apply roll to the specific axis
        var in_coords: array<u32, 8>;
        for (var i = 0u; i < info.ndim; i = i + 1u) {
            if (i == u32(info.axis)) {
                let axis_size = i32(shape[i]);
                let normalized_shift = ((info.shift % axis_size) + axis_size) % axis_size;
                in_coords[i] = u32((i32(out_coords[i]) - normalized_shift + axis_size) % axis_size);
            } else {
                in_coords[i] = out_coords[i];
            }
        }
        
        // Convert to flat index
        in_index = 0u;
        var stride = 1u;
        for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
            in_index = in_index + in_coords[i] * stride;
            if (i > 0u) {
                stride = stride * shape[i];
            } else {
                break;
            }
        }
    }
    
    output[out_index] = input[in_index];
}

// Where operation
@group(0) @binding(0) var<storage, read> condition: array<u32>;  // bool as u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn where_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Handle broadcasting by using modulo for indexing
    let cond_idx = index % arrayLength(&condition);
    let x_idx = index % arrayLength(&x);
    let y_idx = index % arrayLength(&y);
    
    if (condition[cond_idx] != 0u) {
        output[index] = x[x_idx];
    } else {
        output[index] = y[y_idx];
    }
}

// One-hot operation
struct OneHotInfo {
    depth: u32,
    on_value: f32,
    off_value: f32,
    indices_size: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: OneHotInfo;

@compute @workgroup_size(64)
fn one_hot_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    let total_size = info.indices_size * info.depth;
    
    if (out_index >= total_size) {
        return;
    }
    
    let idx_pos = out_index / info.depth;
    let depth_pos = out_index % info.depth;
    
    let index_value = indices[idx_pos];
    
    if (u32(index_value) == depth_pos) {
        output[out_index] = info.on_value;
    } else {
        output[out_index] = info.off_value;
    }
}

// Flip operation
struct FlipInfo {
    axis: u32,
    ndim: u32,
    total_size: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: FlipInfo;
@group(0) @binding(3) var<storage, read> shape: array<u32>;

@compute @workgroup_size(64)
fn flip_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % shape[i];
        remaining = remaining / shape[i];
    }
    
    // Flip the coordinate along the specified axis
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            in_coords[i] = shape[i] - 1u - out_coords[i];
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Broadcast operation
struct BroadcastInfo {
    input_ndim: u32,
    output_ndim: u32,
    total_size: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: BroadcastInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn broadcast_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    for (var i = info.output_ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % output_shape[i];
        remaining = remaining / output_shape[i];
    }
    
    // Map to input coordinates using broadcasting rules
    var in_coords: array<u32, 8>;
    let offset = info.output_ndim - info.input_ndim;
    
    for (var i = 0u; i < info.input_ndim; i = i + 1u) {
        let out_dim = i + offset;
        if (input_shape[i] == 1u) {
            in_coords[i] = 0u;  // Broadcast dimension
        } else {
            in_coords[i] = out_coords[out_dim];
        }
    }
    
    // Convert to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.input_ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}