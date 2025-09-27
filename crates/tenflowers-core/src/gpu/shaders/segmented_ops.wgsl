// Segmented operations for ragged tensor support

struct SegmentParams {
    data_size: u32,
    num_segments: u32,
    _padding: array<u32, 2>,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> segment_ids: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: SegmentParams;

// Segmented sum operation
// Each thread processes one data element and atomically adds to its segment
@compute @workgroup_size(256)
fn segment_sum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.data_size) {
        return;
    }
    
    let segment_id = segment_ids[idx];
    if (segment_id >= 0 && segment_id < i32(params.num_segments)) {
        let data_val = data[idx];
        
        // Atomic add to accumulate sum for this segment
        atomicAdd(&output[segment_id], data_val);
    }
}

// Segmented max operation
// Uses atomic compare-exchange to find maximum
@compute @workgroup_size(256)
fn segment_max(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.data_size) {
        return;
    }
    
    let segment_id = segment_ids[idx];
    if (segment_id >= 0 && segment_id < i32(params.num_segments)) {
        let data_val = data[idx];
        
        // Atomic max operation
        var current_max = atomicLoad(&output[segment_id]);
        var new_max = max(current_max, data_val);
        
        // Keep trying until we successfully update or find a larger value
        while (new_max > current_max) {
            let old_val = atomicCompareExchangeWeak(&output[segment_id], current_max, new_max);
            if (old_val == current_max) {
                break; // Successfully updated
            }
            current_max = old_val;
            new_max = max(current_max, data_val);
        }
    }
}

// Segmented mean operation - uses two buffers for sum and count
@group(0) @binding(3) var<storage, read_write> count_buffer: array<u32>;
@group(0) @binding(4) var<uniform> mean_params: SegmentParams;

@compute @workgroup_size(256)
fn segment_mean(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= mean_params.data_size) {
        return;
    }
    
    let segment_id = segment_ids[idx];
    if (segment_id >= 0 && segment_id < i32(mean_params.num_segments)) {
        let data_val = data[idx];
        
        // Atomic add to accumulate sum for this segment
        atomicAdd(&output[segment_id], data_val);
        
        // Atomic add to count for this segment
        atomicAdd(&count_buffer[segment_id], 1u);
    }
}

// Second pass for segment_mean: divide sum by count
@compute @workgroup_size(64)
fn segment_mean_finalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let segment_id = global_id.x;
    
    if (segment_id >= mean_params.num_segments) {
        return;
    }
    
    let count = count_buffer[segment_id];
    if (count > 0u) {
        output[segment_id] = output[segment_id] / f32(count);
    }
}

// Optimized segmented reduction using workgroup shared memory
// This version processes segments in parallel within workgroups
var<workgroup> shared_sums: array<f32, 256>;
var<workgroup> shared_counts: array<u32, 256>;

@compute @workgroup_size(256)
fn segment_sum_optimized(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let workgroup_size = 256u;
    
    // Initialize shared memory
    shared_sums[tid] = 0.0;
    shared_counts[tid] = 0u;
    workgroupBarrier();
    
    // Each thread processes multiple elements
    for (var i = gid; i < params.data_size; i += workgroup_size * 16u) {
        if (i < params.data_size) {
            let segment_id = segment_ids[i];
            if (segment_id >= 0 && segment_id < i32(params.num_segments)) {
                let data_val = data[i];
                
                // Accumulate in shared memory if segment maps to this thread
                if (u32(segment_id) % workgroup_size == tid) {
                    shared_sums[tid] += data_val;
                }
            }
        }
    }
    
    workgroupBarrier();
    
    // Write results from shared memory to global memory
    if (tid < params.num_segments) {
        atomicAdd(&output[tid], shared_sums[tid]);
    }
}

// Segmented argmax operation - finds index of maximum value in each segment
@group(0) @binding(2) var<storage, read_write> argmax_output: array<u32>;

@compute @workgroup_size(256)
fn segment_argmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.data_size) {
        return;
    }
    
    let segment_id = segment_ids[idx];
    if (segment_id >= 0 && segment_id < i32(params.num_segments)) {
        let data_val = data[idx];
        
        // Try to update both max value and argmax index
        var current_max = atomicLoad(&output[segment_id]);
        
        while (data_val > current_max) {
            // Try to update max value
            let old_max = atomicCompareExchangeWeak(&output[segment_id], current_max, data_val);
            if (old_max == current_max) {
                // Successfully updated max, now update argmax
                atomicStore(&argmax_output[segment_id], idx);
                break;
            }
            current_max = old_max;
        }
    }
}

// Segmented scan operations - cumulative operations within segments
@compute @workgroup_size(256)
fn segment_cumsum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.data_size) {
        return;
    }
    
    let segment_id = segment_ids[idx];
    if (segment_id >= 0 && segment_id < i32(params.num_segments)) {
        let data_val = data[idx];
        
        // Simple implementation: sum all previous elements in same segment
        var cum_sum = 0.0;
        for (var i = 0u; i <= idx; i++) {
            if (i < params.data_size && segment_ids[i] == segment_id) {
                cum_sum += data[i];
            }
        }
        
        output[idx] = cum_sum;
    }
}

// Segmented prefix sum with efficient segment boundary handling
@compute @workgroup_size(256)
fn segment_prefix_sum(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let workgroup_size = 256u;
    
    if (gid >= params.data_size) {
        return;
    }
    
    // Load data into shared memory
    shared_sums[tid] = data[gid];
    workgroupBarrier();
    
    // Parallel prefix sum within workgroup
    var step = 1u;
    while (step < workgroup_size) {
        if (tid >= step && gid >= step) {
            // Only add if we're in the same segment
            if (segment_ids[gid] == segment_ids[gid - step]) {
                shared_sums[tid] += shared_sums[tid - step];
            }
        }
        step *= 2u;
        workgroupBarrier();
    }
    
    // Store result
    output[gid] = shared_sums[tid];
}