// Tensor manipulation compute shaders

// Reshape operation - just a memory copy since reshape doesn't change data layout
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn reshape_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Reshape is just a view change, so direct copy
    output[index] = input[index];
}

// Transpose operation with shape information
struct TransposeInfo {
    ndim: u32,
    total_size: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: TransposeInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;
@group(0) @binding(5) var<storage, read> permutation: array<u32>;

@compute @workgroup_size(64)
fn transpose_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to multi-dimensional coordinates
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
    
    // Apply permutation to get input coordinates
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        in_coords[permutation[i]] = out_coords[i];
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Repeat operation
struct RepeatInfo {
    ndim: u32,
    total_size: u32,
    repeats: u32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RepeatInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn repeat_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates - divide by repeats for the specified axis
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            in_coords[i] = out_coords[i] / info.repeats;
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Roll operation
struct RollInfo {
    ndim: u32,
    total_size: u32,
    shift: i32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RollInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn roll_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates with rolling
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            let axis_size = i32(input_shape[i]);
            let shifted = (i32(out_coords[i]) - info.shift) % axis_size;
            in_coords[i] = u32((shifted + axis_size) % axis_size);
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Gather operation
struct GatherInfo {
    ndim: u32,
    total_size: u32,
    axis: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> info: GatherInfo;
@group(0) @binding(4) var<storage, read> params_shape: array<u32>;
@group(0) @binding(5) var<storage, read> indices_shape: array<u32>;

@compute @workgroup_size(64)
fn gather_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    let indices_ndim = arrayLength(&indices_shape);
    
    // First, extract indices coordinates
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % indices_shape[i];
        remaining = remaining / indices_shape[i];
    }
    
    // Get index value
    var idx_flat = 0u;
    var stride = 1u;
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        idx_flat = idx_flat + out_coords[i] * stride;
        if (i > 0u) {
            stride = stride * indices_shape[i];
        } else {
            break;
        }
    }
    
    let index_val = indices[idx_flat];
    
    // Map to params coordinates
    var params_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            params_coords[i] = u32(index_val);
        } else if (i < indices_ndim) {
            params_coords[i] = out_coords[i];
        } else {
            params_coords[i] = 0u;
        }
    }
    
    // Convert params coordinates to flat index
    var params_index = 0u;
    stride = 1u;
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

// Where operation (conditional selection)
struct WhereInfo {
    total_size: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<storage, read> condition: array<u32>; // bool as u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> info: WhereInfo;

@compute @workgroup_size(64)
fn where_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= info.total_size) {
        return;
    }
    
    if (condition[index] != 0u) {
        output[index] = x[index];
    } else {
        output[index] = y[index];
    }
}

// OneHot operation
struct OneHotInfo {
    total_size: u32,
    depth: u32,
    on_value: f32,
    off_value: f32,
}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: OneHotInfo;

@compute @workgroup_size(64)
fn one_hot_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    let indices_size = arrayLength(&indices);
    let idx_index = out_index / info.depth;
    let class_index = out_index % info.depth;
    
    if (idx_index < indices_size) {
        let target_class = indices[idx_index];
        if (class_index == u32(target_class)) {
            output[out_index] = info.on_value;
        } else {
            output[out_index] = info.off_value;
        }
    } else {
        output[out_index] = info.off_value;
    }
}

// Slice operation with ranges
struct SliceInfo {
    ndim: u32,
    total_size: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: SliceInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;
@group(0) @binding(5) var<storage, read> slice_starts: array<u32>;

@compute @workgroup_size(64)
fn slice_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Add slice offsets to get input coordinates
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        in_coords[i] = out_coords[i] + slice_starts[i];
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Repeat operation
struct RepeatInfo {
    ndim: u32,
    total_size: u32,
    repeats: u32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RepeatInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn repeat_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates - divide by repeats for the specified axis
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            in_coords[i] = out_coords[i] / info.repeats;
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Roll operation
struct RollInfo {
    ndim: u32,
    total_size: u32,
    shift: i32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RollInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn roll_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates with rolling
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            let axis_size = i32(input_shape[i]);
            let shifted = (i32(out_coords[i]) - info.shift) % axis_size;
            in_coords[i] = u32((shifted + axis_size) % axis_size);
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Gather operation
struct GatherInfo {
    ndim: u32,
    total_size: u32,
    axis: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> info: GatherInfo;
@group(0) @binding(4) var<storage, read> params_shape: array<u32>;
@group(0) @binding(5) var<storage, read> indices_shape: array<u32>;

@compute @workgroup_size(64)
fn gather_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    let indices_ndim = arrayLength(&indices_shape);
    
    // First, extract indices coordinates
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % indices_shape[i];
        remaining = remaining / indices_shape[i];
    }
    
    // Get index value
    var idx_flat = 0u;
    var stride = 1u;
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        idx_flat = idx_flat + out_coords[i] * stride;
        if (i > 0u) {
            stride = stride * indices_shape[i];
        } else {
            break;
        }
    }
    
    let index_val = indices[idx_flat];
    
    // Map to params coordinates
    var params_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            params_coords[i] = u32(index_val);
        } else if (i < indices_ndim) {
            params_coords[i] = out_coords[i];
        } else {
            params_coords[i] = 0u;
        }
    }
    
    // Convert params coordinates to flat index
    var params_index = 0u;
    stride = 1u;
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

// Where operation (conditional selection)
struct WhereInfo {
    total_size: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<storage, read> condition: array<u32>; // bool as u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> info: WhereInfo;

@compute @workgroup_size(64)
fn where_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= info.total_size) {
        return;
    }
    
    if (condition[index] != 0u) {
        output[index] = x[index];
    } else {
        output[index] = y[index];
    }
}

// OneHot operation
struct OneHotInfo {
    total_size: u32,
    depth: u32,
    on_value: f32,
    off_value: f32,
}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: OneHotInfo;

@compute @workgroup_size(64)
fn one_hot_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    let indices_size = arrayLength(&indices);
    let idx_index = out_index / info.depth;
    let class_index = out_index % info.depth;
    
    if (idx_index < indices_size) {
        let target_class = indices[idx_index];
        if (class_index == u32(target_class)) {
            output[out_index] = info.on_value;
        } else {
            output[out_index] = info.off_value;
        }
    } else {
        output[out_index] = info.off_value;
    }
}

// Pad operation
struct PadInfo {
    ndim: u32,
    total_size: u32,
    constant_value: f32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: PadInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;
@group(0) @binding(5) var<storage, read> pad_before: array<u32>;
@group(0) @binding(6) var<storage, read> pad_after: array<u32>;

@compute @workgroup_size(64)
fn pad_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Check if we're in the padded region
    var in_bounds = true;
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (out_coords[i] < pad_before[i] || out_coords[i] >= pad_before[i] + input_shape[i]) {
            in_bounds = false;
            break;
        }
        in_coords[i] = out_coords[i] - pad_before[i];
    }
    
    if (!in_bounds) {
        output[out_index] = info.constant_value;
        return;
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Repeat operation
struct RepeatInfo {
    ndim: u32,
    total_size: u32,
    repeats: u32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RepeatInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn repeat_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates - divide by repeats for the specified axis
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            in_coords[i] = out_coords[i] / info.repeats;
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Roll operation
struct RollInfo {
    ndim: u32,
    total_size: u32,
    shift: i32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RollInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn roll_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates with rolling
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            let axis_size = i32(input_shape[i]);
            let shifted = (i32(out_coords[i]) - info.shift) % axis_size;
            in_coords[i] = u32((shifted + axis_size) % axis_size);
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Gather operation
struct GatherInfo {
    ndim: u32,
    total_size: u32,
    axis: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> info: GatherInfo;
@group(0) @binding(4) var<storage, read> params_shape: array<u32>;
@group(0) @binding(5) var<storage, read> indices_shape: array<u32>;

@compute @workgroup_size(64)
fn gather_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    let indices_ndim = arrayLength(&indices_shape);
    
    // First, extract indices coordinates
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % indices_shape[i];
        remaining = remaining / indices_shape[i];
    }
    
    // Get index value
    var idx_flat = 0u;
    var stride = 1u;
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        idx_flat = idx_flat + out_coords[i] * stride;
        if (i > 0u) {
            stride = stride * indices_shape[i];
        } else {
            break;
        }
    }
    
    let index_val = indices[idx_flat];
    
    // Map to params coordinates
    var params_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            params_coords[i] = u32(index_val);
        } else if (i < indices_ndim) {
            params_coords[i] = out_coords[i];
        } else {
            params_coords[i] = 0u;
        }
    }
    
    // Convert params coordinates to flat index
    var params_index = 0u;
    stride = 1u;
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

// Where operation (conditional selection)
struct WhereInfo {
    total_size: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<storage, read> condition: array<u32>; // bool as u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> info: WhereInfo;

@compute @workgroup_size(64)
fn where_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= info.total_size) {
        return;
    }
    
    if (condition[index] != 0u) {
        output[index] = x[index];
    } else {
        output[index] = y[index];
    }
}

// OneHot operation
struct OneHotInfo {
    total_size: u32,
    depth: u32,
    on_value: f32,
    off_value: f32,
}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: OneHotInfo;

@compute @workgroup_size(64)
fn one_hot_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    let indices_size = arrayLength(&indices);
    let idx_index = out_index / info.depth;
    let class_index = out_index % info.depth;
    
    if (idx_index < indices_size) {
        let target_class = indices[idx_index];
        if (class_index == u32(target_class)) {
            output[out_index] = info.on_value;
        } else {
            output[out_index] = info.off_value;
        }
    } else {
        output[out_index] = info.off_value;
    }
}

// Tile operation
struct TileInfo {
    ndim: u32,
    total_size: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: TileInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn tile_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates using modulo
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        in_coords[i] = out_coords[i] % input_shape[i];
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Repeat operation
struct RepeatInfo {
    ndim: u32,
    total_size: u32,
    repeats: u32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RepeatInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn repeat_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates - divide by repeats for the specified axis
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            in_coords[i] = out_coords[i] / info.repeats;
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Roll operation
struct RollInfo {
    ndim: u32,
    total_size: u32,
    shift: i32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: RollInfo;
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

@compute @workgroup_size(64)
fn roll_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Map to input coordinates with rolling
    var in_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            let axis_size = i32(input_shape[i]);
            let shifted = (i32(out_coords[i]) - info.shift) % axis_size;
            in_coords[i] = u32((shifted + axis_size) % axis_size);
        } else {
            in_coords[i] = out_coords[i];
        }
    }
    
    // Convert input coordinates to flat index
    var in_index = 0u;
    var stride = 1u;
    for (var i = info.ndim - 1u; i >= 0u; i = i - 1u) {
        in_index = in_index + in_coords[i] * stride;
        if (i > 0u) {
            stride = stride * input_shape[i];
        } else {
            break;
        }
    }
    
    output[out_index] = input[in_index];
}

// Gather operation
struct GatherInfo {
    ndim: u32,
    total_size: u32,
    axis: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> info: GatherInfo;
@group(0) @binding(4) var<storage, read> params_shape: array<u32>;
@group(0) @binding(5) var<storage, read> indices_shape: array<u32>;

@compute @workgroup_size(64)
fn gather_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    // Convert flat output index to coordinates
    var out_coords: array<u32, 8>;
    var remaining = out_index;
    let indices_ndim = arrayLength(&indices_shape);
    
    // First, extract indices coordinates
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        if (i == 0u) {
            out_coords[i] = remaining;
            break;
        }
        out_coords[i] = remaining % indices_shape[i];
        remaining = remaining / indices_shape[i];
    }
    
    // Get index value
    var idx_flat = 0u;
    var stride = 1u;
    for (var i = indices_ndim - 1u; i >= 0u; i = i - 1u) {
        idx_flat = idx_flat + out_coords[i] * stride;
        if (i > 0u) {
            stride = stride * indices_shape[i];
        } else {
            break;
        }
    }
    
    let index_val = indices[idx_flat];
    
    // Map to params coordinates
    var params_coords: array<u32, 8>;
    for (var i = 0u; i < info.ndim; i = i + 1u) {
        if (i == info.axis) {
            params_coords[i] = u32(index_val);
        } else if (i < indices_ndim) {
            params_coords[i] = out_coords[i];
        } else {
            params_coords[i] = 0u;
        }
    }
    
    // Convert params coordinates to flat index
    var params_index = 0u;
    stride = 1u;
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

// Where operation (conditional selection)
struct WhereInfo {
    total_size: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<storage, read> condition: array<u32>; // bool as u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> info: WhereInfo;

@compute @workgroup_size(64)
fn where_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= info.total_size) {
        return;
    }
    
    if (condition[index] != 0u) {
        output[index] = x[index];
    } else {
        output[index] = y[index];
    }
}

// OneHot operation
struct OneHotInfo {
    total_size: u32,
    depth: u32,
    on_value: f32,
    off_value: f32,
}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> info: OneHotInfo;

@compute @workgroup_size(64)
fn one_hot_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_index = global_id.x;
    if (out_index >= info.total_size) {
        return;
    }
    
    let indices_size = arrayLength(&indices);
    let idx_index = out_index / info.depth;
    let class_index = out_index % info.depth;
    
    if (idx_index < indices_size) {
        let target_class = indices[idx_index];
        if (class_index == u32(target_class)) {
            output[out_index] = info.on_value;
        } else {
            output[out_index] = info.off_value;
        }
    } else {
        output[out_index] = info.off_value;
    }
}