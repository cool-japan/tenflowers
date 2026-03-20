// Normalization operation compute shaders

struct NormalizationParams {
    normalized_size: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: NormalizationParams;

// Layer normalization kernel
// Each workgroup processes one normalization group
@compute @workgroup_size(256)
fn layer_norm(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    
    let normalized_size = params.normalized_size;
    let epsilon = params.epsilon;
    let group_idx = group_id.x;
    let local_idx = local_id.x;
    let workgroup_size = 256u;
    
    // Shared memory for reduction
    var<workgroup> shared_sum: array<f32, 256>;
    var<workgroup> shared_sum_sq: array<f32, 256>;
    
    // Calculate start index for this normalization group
    let group_start = group_idx * normalized_size;
    
    // Phase 1: Calculate local sums
    var local_sum = 0.0;
    var local_sum_sq = 0.0;
    
    // Each thread processes multiple elements
    for (var i = local_idx; i < normalized_size; i += workgroup_size) {
        let idx = group_start + i;
        if (idx < arrayLength(&input)) {
            let val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // Store in shared memory
    shared_sum[local_idx] = local_sum;
    shared_sum_sq[local_idx] = local_sum_sq;
    workgroupBarrier();
    
    // Phase 2: Reduction to calculate mean and variance
    // Tree reduction
    for (var stride = workgroup_size / 2u; stride > 0u; stride >>= 1u) {
        if (local_idx < stride) {
            shared_sum[local_idx] += shared_sum[local_idx + stride];
            shared_sum_sq[local_idx] += shared_sum_sq[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // Thread 0 has the final sum
    var mean: f32;
    var variance: f32;
    if (local_idx == 0u) {
        mean = shared_sum[0] / f32(normalized_size);
        variance = shared_sum_sq[0] / f32(normalized_size) - mean * mean;
    }
    
    // Broadcast mean and variance to all threads
    workgroupBarrier();
    if (local_idx == 0u) {
        shared_sum[0] = mean;
        shared_sum[1] = variance;
    }
    workgroupBarrier();
    
    mean = shared_sum[0];
    variance = shared_sum[1];
    let std_dev = sqrt(variance + epsilon);
    
    // Phase 3: Apply normalization
    for (var i = local_idx; i < normalized_size; i += workgroup_size) {
        let idx = group_start + i;
        if (idx < arrayLength(&input)) {
            let normalized = (input[idx] - mean) / std_dev;
            output[idx] = gamma[i] * normalized + beta[i];
        }
    }
}

// Simple layer norm for small normalized sizes (no shared memory optimization)
@compute @workgroup_size(64)
fn layer_norm_simple(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let group_idx = global_id.x;
    let normalized_size = params.normalized_size;
    let epsilon = params.epsilon;
    
    // Calculate start index for this normalization group
    let group_start = group_idx * normalized_size;
    
    // Check bounds
    if (group_start >= arrayLength(&input)) {
        return;
    }
    
    // Calculate mean
    var sum = 0.0;
    for (var i = 0u; i < normalized_size; i++) {
        let idx = group_start + i;
        if (idx < arrayLength(&input)) {
            sum += input[idx];
        }
    }
    let mean = sum / f32(normalized_size);
    
    // Calculate variance
    var sum_sq = 0.0;
    for (var i = 0u; i < normalized_size; i++) {
        let idx = group_start + i;
        if (idx < arrayLength(&input)) {
            let diff = input[idx] - mean;
            sum_sq += diff * diff;
        }
    }
    let variance = sum_sq / f32(normalized_size);
    let std_dev = sqrt(variance + epsilon);
    
    // Apply normalization
    for (var i = 0u; i < normalized_size; i++) {
        let idx = group_start + i;
        if (idx < arrayLength(&input)) {
            let normalized = (input[idx] - mean) / std_dev;
            output[idx] = gamma[i] * normalized + beta[i];
        }
    }
}

// Group normalization kernel
struct GroupNormParams {
    batch_size: u32,
    num_channels: u32,
    num_groups: u32,
    spatial_size: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> group_input: array<f32>;
@group(0) @binding(1) var<storage, read> group_gamma: array<f32>;
@group(0) @binding(2) var<storage, read> group_beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> group_output: array<f32>;
@group(0) @binding(4) var<uniform> group_params: GroupNormParams;

@compute @workgroup_size(256)
fn group_norm(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    
    let batch_idx = group_id.x;
    let group_idx = group_id.y;
    let local_idx = local_id.x;
    
    let channels_per_group = group_params.num_channels / group_params.num_groups;
    let group_size = channels_per_group * group_params.spatial_size;
    let workgroup_size = 256u;
    
    // Shared memory for reduction
    var<workgroup> shared_sum: array<f32, 256>;
    var<workgroup> shared_sum_sq: array<f32, 256>;
    
    // Calculate local sums
    var local_sum = 0.0;
    var local_sum_sq = 0.0;
    
    let batch_offset = batch_idx * group_params.num_channels * group_params.spatial_size;
    let group_offset = group_idx * channels_per_group * group_params.spatial_size;
    
    // Each thread processes multiple elements
    for (var i = local_idx; i < group_size; i += workgroup_size) {
        let idx = batch_offset + group_offset + i;
        if (idx < arrayLength(&group_input)) {
            let val = group_input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // Store in shared memory
    shared_sum[local_idx] = local_sum;
    shared_sum_sq[local_idx] = local_sum_sq;
    workgroupBarrier();
    
    // Tree reduction
    for (var stride = workgroup_size / 2u; stride > 0u; stride >>= 1u) {
        if (local_idx < stride) {
            shared_sum[local_idx] += shared_sum[local_idx + stride];
            shared_sum_sq[local_idx] += shared_sum_sq[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // Calculate mean and variance
    var mean: f32;
    var variance: f32;
    if (local_idx == 0u) {
        mean = shared_sum[0] / f32(group_size);
        variance = shared_sum_sq[0] / f32(group_size) - mean * mean;
    }
    
    // Broadcast to all threads
    workgroupBarrier();
    if (local_idx == 0u) {
        shared_sum[0] = mean;
        shared_sum[1] = variance;
    }
    workgroupBarrier();
    
    mean = shared_sum[0];
    variance = shared_sum[1];
    let std_dev = sqrt(variance + group_params.epsilon);
    
    // Apply normalization
    for (var i = local_idx; i < group_size; i += workgroup_size) {
        let idx = batch_offset + group_offset + i;
        if (idx < arrayLength(&group_input)) {
            let channel_idx = (group_idx * channels_per_group + i / group_params.spatial_size) % group_params.num_channels;
            let normalized = (group_input[idx] - mean) / std_dev;
            group_output[idx] = group_gamma[channel_idx] * normalized + group_beta[channel_idx];
        }
    }
}

// Synchronized Batch Normalization shaders

struct BatchStatsParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    spatial_size: u32,
}

// Compute mean for synchronized batch norm
@group(0) @binding(0) var<storage, read> sync_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> sync_means: array<f32>;
@group(0) @binding(2) var<uniform> sync_params: BatchStatsParams;

@compute @workgroup_size(256)
fn sync_batch_norm_compute_mean(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let channel_idx = global_id.x;
    
    if (channel_idx >= sync_params.channels) {
        return;
    }
    
    let batch_size = sync_params.batch_size;
    let height = sync_params.height;
    let width = sync_params.width;
    let spatial_size = sync_params.spatial_size;
    let elements_per_channel = batch_size * spatial_size;
    
    var sum = 0.0;
    var count = 0u;
    
    // Sum all elements for this channel across all batches and spatial dimensions
    for (var b = 0u; b < batch_size; b++) {
        for (var s = 0u; s < spatial_size; s++) {
            let idx = b * sync_params.channels * spatial_size + channel_idx * spatial_size + s;
            if (idx < arrayLength(&sync_input)) {
                sum += sync_input[idx];
                count += 1u;
            }
        }
    }
    
    // Store the mean for this channel
    if (count > 0u) {
        sync_means[channel_idx] = sum / f32(count);
    }
}

// Compute variance for synchronized batch norm
@group(0) @binding(0) var<storage, read> sync_var_input: array<f32>;
@group(0) @binding(1) var<storage, read> sync_channel_means: array<f32>;
@group(0) @binding(2) var<storage, read_write> sync_vars: array<f32>;
@group(0) @binding(3) var<uniform> sync_var_params: BatchStatsParams;

@compute @workgroup_size(256)
fn sync_batch_norm_compute_var(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let channel_idx = global_id.x;
    
    if (channel_idx >= sync_var_params.channels) {
        return;
    }
    
    let batch_size = sync_var_params.batch_size;
    let height = sync_var_params.height;
    let width = sync_var_params.width;
    let spatial_size = sync_var_params.spatial_size;
    let mean = sync_channel_means[channel_idx];
    
    var sum_squared_diff = 0.0;
    var count = 0u;
    
    // Sum squared differences for this channel across all batches and spatial dimensions
    for (var b = 0u; b < batch_size; b++) {
        for (var s = 0u; s < spatial_size; s++) {
            let idx = b * sync_var_params.channels * spatial_size + channel_idx * spatial_size + s;
            if (idx < arrayLength(&sync_var_input)) {
                let diff = sync_var_input[idx] - mean;
                sum_squared_diff += diff * diff;
                count += 1u;
            }
        }
    }
    
    // Store the variance for this channel
    if (count > 0u) {
        sync_vars[channel_idx] = sum_squared_diff / f32(count);
    }
}

// Apply synchronized batch normalization
struct SyncBatchNormParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> sync_apply_input: array<f32>;
@group(0) @binding(1) var<storage, read> sync_apply_means: array<f32>;
@group(0) @binding(2) var<storage, read> sync_apply_vars: array<f32>;
@group(0) @binding(3) var<storage, read> sync_apply_gamma: array<f32>;
@group(0) @binding(4) var<storage, read> sync_apply_beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> sync_apply_output: array<f32>;
@group(0) @binding(6) var<uniform> sync_apply_params: SyncBatchNormParams;

@compute @workgroup_size(256)
fn sync_batch_norm_apply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&sync_apply_input)) {
        return;
    }
    
    let batch_size = sync_apply_params.batch_size;
    let channels = sync_apply_params.channels;
    let height = sync_apply_params.height;
    let width = sync_apply_params.width;
    let spatial_size = height * width;
    let epsilon = sync_apply_params.epsilon;
    
    // Calculate which channel this element belongs to
    let elements_per_batch = channels * spatial_size;
    let batch_idx = idx / elements_per_batch;
    let idx_in_batch = idx % elements_per_batch;
    let channel_idx = idx_in_batch / spatial_size;
    
    if (channel_idx >= channels) {
        return;
    }
    
    // Get statistics for this channel
    let mean = sync_apply_means[channel_idx];
    let variance = sync_apply_vars[channel_idx];
    let std_dev = sqrt(variance + epsilon);
    let gamma = sync_apply_gamma[channel_idx];
    let beta = sync_apply_beta[channel_idx];
    
    // Apply normalization
    let normalized = (sync_apply_input[idx] - mean) / std_dev;
    sync_apply_output[idx] = gamma * normalized + beta;
}