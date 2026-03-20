// Reduction operation compute shaders
// These kernels implement parallel reduction algorithms

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> metadata: array<u32>; // [input_size, output_size, axis_info]

// Shared memory for workgroup-level reductions
var<workgroup> shared_data: array<f32, 256>;

// Sum reduction
@compute @workgroup_size(256)
fn sum_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>,
                 @builtin(workgroup_id) group_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory
    if (global_idx < input_size) {
        shared_data[thread_id] = input[global_idx];
    } else {
        shared_data[thread_id] = 0.0;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write result - using simple atomic store (works for single workgroup output)
    // For multi-workgroup, a two-stage reduction would be needed
    if (thread_id == 0u && group_id.x == 0u) {
        atomicStore(&output[0], bitcast<u32>(shared_data[0]));
    }
}

// Enhanced axis-specific reduction bindings
@group(0) @binding(0) var<storage, read> axis_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> axis_output: array<f32>;
@group(0) @binding(2) var<storage, read> axis_metadata: array<u32>; // [input_size, output_size, input_rank, num_axes, axis0, axis1, ...]
@group(0) @binding(3) var<storage, read> input_shape: array<u32>;
@group(0) @binding(4) var<storage, read> output_shape: array<u32>;

// Helper function to compute flat index from multidimensional indices using input_shape
fn compute_flat_index_input(indices: ptr<function, array<u32, 8>>, rank: u32) -> u32 {
    var flat_idx = 0u;
    var stride = 1u;

    for (var i = 0u; i < rank; i++) {
        let dim_idx = rank - 1u - i;
        flat_idx += (*indices)[dim_idx] * stride;
        stride *= input_shape[dim_idx];
    }

    return flat_idx;
}

// Helper function to compute multidimensional indices from flat index using output_shape
fn compute_multi_index_output(flat_idx: u32, indices: ptr<function, array<u32, 8>>, rank: u32) {
    var remaining = flat_idx;

    for (var i = 0u; i < rank; i++) {
        let dim_idx = rank - 1u - i;
        (*indices)[dim_idx] = remaining % output_shape[dim_idx];
        remaining = remaining / output_shape[dim_idx];
    }
}

// Enhanced sum reduction along specific axes
@compute @workgroup_size(64)
fn sum_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize sum
    var sum_val = 0.0;
    
    // Iterate over all reduction axes
    // For simplicity, we'll implement a brute force approach
    // In a production system, this would be optimized based on the specific axes
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            sum_val += axis_input[input_flat_idx];
        }
    }
    
    axis_output[output_idx] = sum_val;
}

// Enhanced mean reduction along specific axes
@compute @workgroup_size(64)
fn mean_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize sum and count
    var sum_val = 0.0;
    var count = 0u;
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            sum_val += axis_input[input_flat_idx];
            count += 1u;
        }
    }
    
    if (count > 0u) {
        axis_output[output_idx] = sum_val / f32(count);
    } else {
        axis_output[output_idx] = 0.0;
    }
}

// Enhanced max reduction along specific axes
@compute @workgroup_size(64)
fn max_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize max value
    var max_val = -3.40282347e+38; // -FLT_MAX
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            max_val = max(max_val, axis_input[input_flat_idx]);
        }
    }
    
    axis_output[output_idx] = max_val;
}

// Enhanced min reduction along specific axes
@compute @workgroup_size(64)
fn min_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize min value
    var min_val = 3.40282347e+38; // FLT_MAX
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            min_val = min(min_val, axis_input[input_flat_idx]);
        }
    }
    
    axis_output[output_idx] = min_val;
}

// Enhanced argmax reduction along specific axes
@compute @workgroup_size(64)
fn argmax_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize max value and index
    var max_val = -3.40282347e+38; // -FLT_MAX
    var max_idx = 0u;
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            let val = axis_input[input_flat_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = input_indices[axis_metadata[4]]; // Index along first reduction axis
            }
        }
    }
    
    axis_output[output_idx] = f32(max_idx);
}

// Enhanced argmin reduction along specific axes
@compute @workgroup_size(64)
fn argmin_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize min value and index
    var min_val = 3.40282347e+38; // FLT_MAX
    var min_idx = 0u;
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            let val = axis_input[input_flat_idx];
            if (val < min_val) {
                min_val = val;
                min_idx = input_indices[axis_metadata[4]]; // Index along first reduction axis
            }
        }
    }
    
    axis_output[output_idx] = f32(min_idx);
}

// Enhanced all reduction along specific axes
@compute @workgroup_size(64)
fn all_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize all value (true = 1.0)
    var all_val = 1.0;
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            let val = axis_input[input_flat_idx];
            if (val == 0.0) {
                all_val = 0.0;
                break;
            }
        }
    }
    
    axis_output[output_idx] = all_val;
}

// Enhanced any reduction along specific axes
@compute @workgroup_size(64)
fn any_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize any value (false = 0.0)
    var any_val = 0.0;
    
    let reduction_size = input_size / output_size;
    
    for (var r = 0u; r < reduction_size; r++) {
        var input_indices: array<u32, 8>;
        
        // Copy output indices to input indices
        for (var i = 0u; i < 8u; i++) {
            input_indices[i] = output_indices[i];
        }
        
        // Compute which reduction iteration this is and update input indices accordingly
        var remaining_r = r;
        for (var axis_idx = 0u; axis_idx < num_axes; axis_idx++) {
            let axis = axis_metadata[4u + axis_idx];
            if (axis < input_rank) {
                let axis_size = input_shape[axis];
                input_indices[axis] = remaining_r % axis_size;
                remaining_r = remaining_r / axis_size;
            }
        }
        
        // Compute flat input index
        let input_flat_idx = compute_flat_index_input(&input_indices, input_rank);
        
        if (input_flat_idx < input_size) {
            let val = axis_input[input_flat_idx];
            if (val != 0.0) {
                any_val = 1.0;
                break;
            }
        }
    }
    
    axis_output[output_idx] = any_val;
}

// Enhanced legacy sum reduction with axis-aware logic
@compute @workgroup_size(64)
fn sum_axis_reduction_legacy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = metadata[0];
    let output_size = metadata[1];
    let axis_info = metadata[2]; // Axis to reduce along (0 for last axis, 1 for all axes)
    
    if (output_idx >= output_size) {
        return;
    }
    
    var sum_val = 0.0;
    
    if (axis_info == 1u) {
        // Reduce along all axes (global reduction)
        let stride = output_size;
        var i = output_idx;
        
        while (i < input_size) {
            sum_val += input[i];
            i += stride;
        }
    } else {
        // Reduce along specific axis (axis_info contains the stride pattern)
        // This implements a simple strided reduction pattern
        let reduction_stride = input_size / output_size;
        let start_idx = output_idx * reduction_stride;
        
        for (var i = 0u; i < reduction_stride; i++) {
            let input_idx = start_idx + i;
            if (input_idx < input_size) {
                sum_val += input[input_idx];
            }
        }
    }

    atomicStore(&output[output_idx], bitcast<u32>(sum_val));
}

// Max reduction
@compute @workgroup_size(256)
fn max_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory
    if (global_idx < input_size) {
        shared_data[thread_id] = input[global_idx];
    } else {
        shared_data[thread_id] = -3.40282347e+38; // -FLT_MAX
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result (note: this is a simplified version)
    if (thread_id == 0u) {
        // For atomic max, we need to implement atomic compare-and-swap
        // For now, just write the result (works for single workgroup)
        atomicStore(&output[0], bitcast<u32>(shared_data[0]));
    }
}

// Min reduction
@compute @workgroup_size(256)
fn min_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory
    if (global_idx < input_size) {
        shared_data[thread_id] = input[global_idx];
    } else {
        shared_data[thread_id] = 3.40282347e+38; // FLT_MAX
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] = min(shared_data[thread_id], shared_data[thread_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (thread_id == 0u) {
        atomicStore(&output[0], bitcast<u32>(shared_data[0]));
    }
}

// ArgMax reduction - returns index of maximum value
@group(0) @binding(0) var<storage, read> argmax_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> argmax_output: array<u32>;
@group(0) @binding(2) var<storage, read> argmax_metadata: array<u32>;

// Shared memory for argmax (value, index pairs)
var<workgroup> shared_values: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = argmax_metadata[0];
    
    // Load data into shared memory
    if (global_idx < input_size) {
        shared_values[thread_id] = argmax_input[global_idx];
        shared_indices[thread_id] = global_idx;
    } else {
        shared_values[thread_id] = -3.40282347e+38; // -FLT_MAX
        shared_indices[thread_id] = 0u;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            if (shared_values[thread_id + stride] > shared_values[thread_id]) {
                shared_values[thread_id] = shared_values[thread_id + stride];
                shared_indices[thread_id] = shared_indices[thread_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (thread_id == 0u) {
        argmax_output[0] = shared_indices[0];
    }
}

// ArgMin reduction - returns index of minimum value
@compute @workgroup_size(256)
fn argmin_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = argmax_metadata[0];
    
    // Load data into shared memory
    if (global_idx < input_size) {
        shared_values[thread_id] = argmax_input[global_idx];
        shared_indices[thread_id] = global_idx;
    } else {
        shared_values[thread_id] = 3.40282347e+38; // FLT_MAX
        shared_indices[thread_id] = 0u;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            if (shared_values[thread_id + stride] < shared_values[thread_id]) {
                shared_values[thread_id] = shared_values[thread_id + stride];
                shared_indices[thread_id] = shared_indices[thread_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (thread_id == 0u) {
        argmax_output[0] = shared_indices[0];
    }
}

// Mean reduction (sum followed by division)
@compute @workgroup_size(256)
fn mean_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory
    if (global_idx < input_size) {
        shared_data[thread_id] = input[global_idx];
    } else {
        shared_data[thread_id] = 0.0;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result divided by input size - using simple atomic store (works for single workgroup output)
    // For multi-workgroup, a two-stage reduction would be needed
    if (thread_id == 0u && group_id.x == 0u) {
        atomicStore(&output[0], bitcast<u32>(shared_data[0] / f32(input_size)));
    }
}

// All reduction (logical AND for boolean-like operations)
@compute @workgroup_size(256)
fn all_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory (treating as boolean: non-zero = true)
    if (global_idx < input_size) {
        shared_data[thread_id] = select(0.0, 1.0, input[global_idx] != 0.0);
    } else {
        shared_data[thread_id] = 1.0; // Identity for AND
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory (AND operation)
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] = shared_data[thread_id] * shared_data[thread_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (thread_id == 0u) {
        atomicStore(&output[0], bitcast<u32>(shared_data[0]));
    }
}

// Any reduction (logical OR for boolean-like operations)
@compute @workgroup_size(256)
fn any_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory (treating as boolean: non-zero = true)
    if (global_idx < input_size) {
        shared_data[thread_id] = select(0.0, 1.0, input[global_idx] != 0.0);
    } else {
        shared_data[thread_id] = 0.0; // Identity for OR
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory (OR operation using max)
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (thread_id == 0u) {
        atomicStore(&output[0], bitcast<u32>(shared_data[0]));
    }
}

// Inf/NaN detection reduction - returns 1.0 if any inf/nan found, 0.0 otherwise
@compute @workgroup_size(256)
fn inf_nan_detection(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory, checking for inf/nan
    if (global_idx < input_size) {
        let value = input[global_idx];
        // Check if value is infinite or NaN
        // In IEEE 754: infinity has exponent bits all 1 and mantissa all 0
        // NaN has exponent bits all 1 and mantissa non-zero
        let bits = bitcast<u32>(value);
        let exponent = (bits >> 23u) & 0xFFu;
        let mantissa = bits & 0x7FFFFFu;

        // Exponent = 0xFF (255) means inf or NaN
        let is_inf_or_nan = (exponent == 0xFFu);

        shared_data[thread_id] = select(0.0, 1.0, is_inf_or_nan);
    } else {
        shared_data[thread_id] = 0.0; // Identity for OR
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory (OR operation using max)
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (thread_id == 0u) {
        atomicStore(&output[0], bitcast<u32>(shared_data[0]));
    }
}

// GPU Histogram computation using atomic operations
// Bindings: 
// @group(0) @binding(0) - input data (array<f32>)
// @group(0) @binding(1) - histogram bins (array<atomic<u32>>)
// @group(0) @binding(2) - histogram metadata (array<f32>) [min_val, max_val, num_bins]
@compute @workgroup_size(256)
fn histogram_computation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_idx = global_id.x;
    let input_size = arrayLength(&input);
    
    if (global_idx >= input_size) {
        return;
    }
    
    let value = input[global_idx];

    // Get histogram parameters from metadata (stored as u32 bits, interpreted as f32)
    let min_val = bitcast<f32>(metadata[0]);
    let max_val = bitcast<f32>(metadata[1]);
    let num_bins = metadata[2];
    
    // Check if value is within range
    if (value < min_val || value > max_val) {
        return;
    }
    
    // Calculate bin index
    let range = max_val - min_val;
    let normalized_val = (value - min_val) / range;
    var bin_idx = u32(normalized_val * f32(num_bins));
    
    // Clamp to valid range (handle edge case where value == max_val)
    bin_idx = min(bin_idx, num_bins - 1u);
    
    // Atomically increment the bin count
    atomicAdd(&output[bin_idx], 1u);
}
// Product reduction kernel
// Computes the product of all elements in the input
@compute @workgroup_size(256)
fn product_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                      @builtin(local_invocation_id) local_id: vec3<u32>,
                      @builtin(workgroup_id) group_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // Load data into shared memory with multiplicative identity (1.0)
    if (global_idx < input_size) {
        shared_data[thread_id] = input[global_idx];
    } else {
        shared_data[thread_id] = 1.0;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory with multiplication
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] *= shared_data[thread_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result - note: atomic multiply not available, so we need a workaround
    // For now, we'll use a simple non-atomic approach which works for single workgroup
    if (thread_id == 0u) {
        // For multi-workgroup, this would need atomic operations or a two-pass approach
        atomicStore(&output[group_id.x], bitcast<u32>(shared_data[0]));
    }
}

// Variance reduction kernel  
// Computes the variance of all elements: Var(X) = E[X^2] - E[X]^2
// This requires a two-pass algorithm or Welford's online algorithm
// For now, implementing a simple two-pass approach
@compute @workgroup_size(256)
fn variance_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_idx = global_id.x;
    let input_size = metadata[0];
    
    // First pass: compute mean (assuming it's precomputed and stored in metadata[3])
    let mean = bitcast<f32>(metadata[3]);
    
    // Load squared differences into shared memory
    if (global_idx < input_size) {
        let diff = input[global_idx] - mean;
        shared_data[thread_id] = diff * diff;
    } else {
        shared_data[thread_id] = 0.0;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (thread_id < stride && (thread_id + stride) < 256u) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write result - using simple atomic store (works for single workgroup output)
    // For multi-workgroup, a two-stage reduction would be needed
    if (thread_id == 0u && group_id.x == 0u) {
        atomicStore(&output[0], bitcast<u32>(shared_data[0]));
    }
}

// Product reduction along specific axes
@compute @workgroup_size(64)
fn product_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Initialize product with multiplicative identity
    var product_val = 1.0;
    
    // Iterate over reduction dimensions
    // This is a simplified version; production would optimize based on specific axes
    let reduction_size = input_size / output_size;
    
    for (var i = 0u; i < reduction_size; i++) {
        // Map to input index (simplified - assumes last axis reduction)
        let input_idx = output_idx * reduction_size + i;
        if (input_idx < input_size) {
            product_val *= axis_input[input_idx];
        }
    }
    
    // Store result
    axis_output[output_idx] = product_val;
}

// Variance reduction along specific axes
@compute @workgroup_size(64)
fn variance_axis_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let input_size = axis_metadata[0];
    let output_size = axis_metadata[1];
    let input_rank = axis_metadata[2];
    let num_axes = axis_metadata[3];
    
    if (output_idx >= output_size) {
        return;
    }
    
    // Get output indices
    var output_indices: array<u32, 8>;
    compute_multi_index_output(output_idx, &output_indices, arrayLength(&output_shape));
    
    // Two-pass algorithm for variance
    // Pass 1: Compute mean
    var sum_val = 0.0;
    let reduction_size = input_size / output_size;
    var count = 0.0;
    
    for (var i = 0u; i < reduction_size; i++) {
        let input_idx = output_idx * reduction_size + i;
        if (input_idx < input_size) {
            sum_val += axis_input[input_idx];
            count += 1.0;
        }
    }
    
    let mean_val = sum_val / count;
    
    // Pass 2: Compute variance
    var variance_val = 0.0;
    
    for (var i = 0u; i < reduction_size; i++) {
        let input_idx = output_idx * reduction_size + i;
        if (input_idx < input_size) {
            let diff = axis_input[input_idx] - mean_val;
            variance_val += diff * diff;
        }
    }
    
    // Store result (population variance)
    axis_output[output_idx] = variance_val / count;
}
