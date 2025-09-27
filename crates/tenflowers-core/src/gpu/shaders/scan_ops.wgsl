// Parallel scan (prefix sum) operations for GPU

struct ScanParams {
    size: u32,
    axis: u32,
    stride: u32,
    axis_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp: array<f32>;
@group(0) @binding(3) var<uniform> params: ScanParams;

// Workgroup-level shared memory for parallel scan
var<workgroup> shared_data: array<f32, 256>;

// Up-sweep (reduce) phase of parallel scan
@compute @workgroup_size(256)
fn cumsum_up_sweep(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let workgroup_size = 256u;
    
    if (gid >= params.size) {
        return;
    }
    
    // Load data into shared memory
    shared_data[tid] = input[gid];
    workgroupBarrier();
    
    // Up-sweep phase
    var step = 1u;
    while (step < workgroup_size) {
        if (tid % (2u * step) == 0u && tid + step < workgroup_size) {
            shared_data[tid + step] += shared_data[tid];
        }
        step *= 2u;
        workgroupBarrier();
    }
    
    // Store the partial sum for this workgroup
    if (tid == 0u) {
        temp[workgroup_id.x] = shared_data[workgroup_size - 1u];
    }
}

// Down-sweep phase of parallel scan
@compute @workgroup_size(256)
fn cumsum_down_sweep(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let workgroup_size = 256u;
    
    if (gid >= params.size) {
        return;
    }
    
    // Load data into shared memory
    shared_data[tid] = input[gid];
    workgroupBarrier();
    
    // Up-sweep phase (same as above)
    var step = 1u;
    while (step < workgroup_size) {
        if (tid % (2u * step) == 0u && tid + step < workgroup_size) {
            shared_data[tid + step] += shared_data[tid];
        }
        step *= 2u;
        workgroupBarrier();
    }
    
    // Clear the last element for down-sweep
    if (tid == 0u) {
        shared_data[workgroup_size - 1u] = 0.0;
    }
    workgroupBarrier();
    
    // Down-sweep phase
    step = workgroup_size / 2u;
    while (step > 0u) {
        if (tid % (2u * step) == 0u && tid + step < workgroup_size) {
            let temp_val = shared_data[tid];
            shared_data[tid] = shared_data[tid + step];
            shared_data[tid + step] += temp_val;
        }
        step /= 2u;
        workgroupBarrier();
    }
    
    // Add the prefix sum from previous workgroups
    var prefix_sum = 0.0;
    if (workgroup_id.x > 0u) {
        prefix_sum = temp[workgroup_id.x - 1u];
    }
    
    // Store result (exclusive scan) + input value for inclusive scan
    output[gid] = shared_data[tid] + prefix_sum + input[gid];
}

// Simple cumulative sum for small arrays (single workgroup)
@compute @workgroup_size(256)
fn cumsum_simple(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    if (gid >= params.size) {
        return;
    }
    
    // Load data into shared memory
    shared_data[tid] = input[gid];
    workgroupBarrier();
    
    // Simple prefix sum using shared memory
    var step = 1u;
    while (step < 256u) {
        if (tid >= step && tid < params.size) {
            shared_data[tid] += shared_data[tid - step];
        }
        step *= 2u;
        workgroupBarrier();
    }
    
    // Store result
    output[gid] = shared_data[tid];
}

// Cumulative product operations
@compute @workgroup_size(256)
fn cumprod_simple(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    if (gid >= params.size) {
        return;
    }
    
    // Load data into shared memory
    shared_data[tid] = input[gid];
    workgroupBarrier();
    
    // Simple prefix product using shared memory
    var step = 1u;
    while (step < 256u) {
        if (tid >= step && tid < params.size) {
            shared_data[tid] *= shared_data[tid - step];
        }
        step *= 2u;
        workgroupBarrier();
    }
    
    // Store result
    output[gid] = shared_data[tid];
}

// Axis-aware cumulative sum
@compute @workgroup_size(64)
fn cumsum_axis(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    
    if (gid >= params.size) {
        return;
    }
    
    // Calculate position within the tensor
    let outer_idx = gid / (params.axis_size * params.stride);
    let inner_idx = gid % params.stride;
    let axis_idx = (gid / params.stride) % params.axis_size;
    
    // Base index for this slice
    let base_idx = outer_idx * params.axis_size * params.stride + inner_idx;
    
    // Compute cumulative sum along the axis
    var sum = 0.0;
    for (var i = 0u; i <= axis_idx; i++) {
        let idx = base_idx + i * params.stride;
        sum += input[idx];
    }
    
    output[gid] = sum;
}

// Axis-aware cumulative product
@compute @workgroup_size(64)
fn cumprod_axis(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    
    if (gid >= params.size) {
        return;
    }
    
    // Calculate position within the tensor
    let outer_idx = gid / (params.axis_size * params.stride);
    let inner_idx = gid % params.stride;
    let axis_idx = (gid / params.stride) % params.axis_size;
    
    // Base index for this slice
    let base_idx = outer_idx * params.axis_size * params.stride + inner_idx;
    
    // Compute cumulative product along the axis
    var prod = 1.0;
    for (var i = 0u; i <= axis_idx; i++) {
        let idx = base_idx + i * params.stride;
        prod *= input[idx];
    }
    
    output[gid] = prod;
}

// Optimized scan for power-of-2 sizes
@compute @workgroup_size(256)
fn cumsum_optimized(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x * 2u;
    
    if (gid >= params.size) {
        return;
    }
    
    // Load two elements per thread
    shared_data[tid] = 0.0;
    if (gid < params.size) {
        shared_data[tid] += input[gid];
    }
    if (gid + 1u < params.size) {
        shared_data[tid] += input[gid + 1u];
    }
    
    workgroupBarrier();
    
    // Efficient tree-based scan
    var step = 1u;
    while (step < 256u) {
        if (tid >= step) {
            shared_data[tid] += shared_data[tid - step];
        }
        step *= 2u;
        workgroupBarrier();
    }
    
    // Store results
    if (gid < params.size) {
        output[gid] = shared_data[tid] - (gid + 1u < params.size ? input[gid + 1u] : 0.0);
    }
    if (gid + 1u < params.size) {
        output[gid + 1u] = shared_data[tid];
    }
}