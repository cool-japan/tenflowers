// Pooling operations compute shaders

struct PoolingParams {
    batch_size: u32,
    channels: u32,
    input_height: u32,
    input_width: u32,
    output_height: u32,
    output_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolingParams;

// Max pooling 2D kernel
@compute @workgroup_size(8, 8, 1)
fn max_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    var max_val = -3.4028235e+38; // -FLT_MAX
    var found_valid = false;
    
    // Pooling window
    for (var kh: u32 = 0u; kh < params.kernel_height; kh++) {
        for (var kw: u32 = 0u; kw < params.kernel_width; kw++) {
            let in_y = out_y * params.stride_h + kh;
            let in_x = out_x * params.stride_w + kw;
            
            // Check bounds with padding
            if (in_y >= params.pad_h && in_x >= params.pad_w) {
                let actual_y = in_y - params.pad_h;
                let actual_x = in_x - params.pad_w;
                
                if (actual_y < params.input_height && actual_x < params.input_width) {
                    let input_idx = ((batch_idx * params.channels + channel) * params.input_height + actual_y) * params.input_width + actual_x;
                    let val = input[input_idx];
                    
                    if (!found_valid || val > max_val) {
                        max_val = val;
                        found_valid = true;
                    }
                }
            }
        }
    }
    
    // Set output value (0.0 if no valid input found)
    output[output_idx] = select(0.0, max_val, found_valid);
}

// Average pooling 2D kernel
@compute @workgroup_size(8, 8, 1)
fn avg_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    var sum = 0.0;
    var count = 0u;
    
    // Pooling window
    for (var kh: u32 = 0u; kh < params.kernel_height; kh++) {
        for (var kw: u32 = 0u; kw < params.kernel_width; kw++) {
            let in_y = out_y * params.stride_h + kh;
            let in_x = out_x * params.stride_w + kw;
            
            // Check bounds with padding
            if (in_y >= params.pad_h && in_x >= params.pad_w) {
                let actual_y = in_y - params.pad_h;
                let actual_x = in_x - params.pad_w;
                
                if (actual_y < params.input_height && actual_x < params.input_width) {
                    let input_idx = ((batch_idx * params.channels + channel) * params.input_height + actual_y) * params.input_width + actual_x;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
    }
    
    // Set output value (0.0 if no valid input found)
    output[output_idx] = select(0.0, sum / f32(count), count > 0u);
}

// Global average pooling kernel
@compute @workgroup_size(256, 1, 1)
fn global_avg_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_channel_idx = global_id.x;
    let total_batch_channels = params.batch_size * params.channels;
    
    if (batch_channel_idx >= total_batch_channels) {
        return;
    }
    
    let batch_idx = batch_channel_idx / params.channels;
    let channel_idx = batch_channel_idx % params.channels;
    
    var sum = 0.0;
    let spatial_size = params.input_height * params.input_width;
    
    // Sum all spatial locations for this batch and channel
    for (var i: u32 = 0u; i < spatial_size; i++) {
        let input_idx = batch_channel_idx * spatial_size + i;
        sum += input[input_idx];
    }
    
    // Average and store result
    let avg = sum / f32(spatial_size);
    output[batch_channel_idx] = avg;
}

// Global max pooling kernel
@compute @workgroup_size(256, 1, 1)
fn global_max_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_channel_idx = global_id.x;
    let total_batch_channels = params.batch_size * params.channels;
    
    if (batch_channel_idx >= total_batch_channels) {
        return;
    }
    
    let batch_idx = batch_channel_idx / params.channels;
    let channel_idx = batch_channel_idx % params.channels;
    
    var max_val = -3.4028235e+38; // -FLT_MAX
    let spatial_size = params.input_height * params.input_width;
    
    // Find max across all spatial locations for this batch and channel
    for (var i: u32 = 0u; i < spatial_size; i++) {
        let input_idx = batch_channel_idx * spatial_size + i;
        let val = input[input_idx];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Store result
    output[batch_channel_idx] = max_val;
}

// Adaptive average pooling kernel - pools to specified output size
@compute @workgroup_size(8, 8, 1)
fn adaptive_avg_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    // Calculate adaptive pooling region
    let start_h = (out_y * params.input_height) / params.output_height;
    let end_h = ((out_y + 1u) * params.input_height) / params.output_height;
    let start_w = (out_x * params.input_width) / params.output_width;
    let end_w = ((out_x + 1u) * params.input_width) / params.output_width;
    
    var sum = 0.0;
    var count = 0u;
    
    // Pool over the adaptive region
    for (var h: u32 = start_h; h < end_h; h++) {
        for (var w: u32 = start_w; w < end_w; w++) {
            let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
            sum += input[input_idx];
            count++;
        }
    }
    
    // Set output value
    output[output_idx] = select(0.0, sum / f32(count), count > 0u);
}

// Adaptive max pooling kernel - pools to specified output size
@compute @workgroup_size(8, 8, 1)
fn adaptive_max_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    // Calculate adaptive pooling region
    let start_h = (out_y * params.input_height) / params.output_height;
    let end_h = ((out_y + 1u) * params.input_height) / params.output_height;
    let start_w = (out_x * params.input_width) / params.output_width;
    let end_w = ((out_x + 1u) * params.input_width) / params.output_width;
    
    var max_val = -3.4028235e+38; // -FLT_MAX
    var found_valid = false;
    
    // Pool over the adaptive region
    for (var h: u32 = start_h; h < end_h; h++) {
        for (var w: u32 = start_w; w < end_w; w++) {
            let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
            let val = input[input_idx];
            
            if (!found_valid || val > max_val) {
                max_val = val;
                found_valid = true;
            }
        }
    }
    
    // Set output value
    output[output_idx] = select(0.0, max_val, found_valid);
}

// ======== 3D ADAPTIVE POOLING OPERATIONS ========

struct Pooling3DParams {
    batch_size: u32,
    channels: u32,
    input_depth: u32,
    input_height: u32,
    input_width: u32,
    output_depth: u32,
    output_height: u32,
    output_width: u32,
    kernel_depth: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
}

@group(0) @binding(0) var<storage, read> input3d: array<f32>;
@group(0) @binding(1) var<storage, read_write> output3d: array<f32>;
@group(0) @binding(2) var<uniform> params3d: Pooling3DParams;

// Adaptive average pooling 3D kernel
@compute @workgroup_size(4, 4, 4)
fn adaptive_avg_pool3d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params3d.batch_size;
    let channel = global_id.z / params3d.batch_size;
    let out_d = global_id.y % params3d.output_depth;
    let out_h = global_id.y / params3d.output_depth;
    let out_w = global_id.x;
    
    if (channel >= params3d.channels || out_d >= params3d.output_depth || 
        out_h >= params3d.output_height || out_w >= params3d.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params3d.channels + channel) * params3d.output_depth + out_d) * 
                     params3d.output_height * params3d.output_width + out_h * params3d.output_width + out_w;
    
    // Calculate adaptive pooling region
    let start_d = (out_d * params3d.input_depth) / params3d.output_depth;
    let end_d = ((out_d + 1u) * params3d.input_depth) / params3d.output_depth;
    let start_h = (out_h * params3d.input_height) / params3d.output_height;
    let end_h = ((out_h + 1u) * params3d.input_height) / params3d.output_height;
    let start_w = (out_w * params3d.input_width) / params3d.output_width;
    let end_w = ((out_w + 1u) * params3d.input_width) / params3d.output_width;
    
    var sum = 0.0;
    var count = 0u;
    
    // Pool over the adaptive region
    for (var d: u32 = start_d; d < end_d; d++) {
        for (var h: u32 = start_h; h < end_h; h++) {
            for (var w: u32 = start_w; w < end_w; w++) {
                let input_idx = ((batch_idx * params3d.channels + channel) * params3d.input_depth + d) * 
                               params3d.input_height * params3d.input_width + h * params3d.input_width + w;
                sum += input3d[input_idx];
                count++;
            }
        }
    }
    
    // Set output value
    output3d[output_idx] = select(0.0, sum / f32(count), count > 0u);
}

// Adaptive max pooling 3D kernel
@compute @workgroup_size(4, 4, 4)
fn adaptive_max_pool3d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params3d.batch_size;
    let channel = global_id.z / params3d.batch_size;
    let out_d = global_id.y % params3d.output_depth;
    let out_h = global_id.y / params3d.output_depth;
    let out_w = global_id.x;
    
    if (channel >= params3d.channels || out_d >= params3d.output_depth || 
        out_h >= params3d.output_height || out_w >= params3d.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params3d.channels + channel) * params3d.output_depth + out_d) * 
                     params3d.output_height * params3d.output_width + out_h * params3d.output_width + out_w;
    
    // Calculate adaptive pooling region
    let start_d = (out_d * params3d.input_depth) / params3d.output_depth;
    let end_d = ((out_d + 1u) * params3d.input_depth) / params3d.output_depth;
    let start_h = (out_h * params3d.input_height) / params3d.output_height;
    let end_h = ((out_h + 1u) * params3d.input_height) / params3d.output_height;
    let start_w = (out_w * params3d.input_width) / params3d.output_width;
    let end_w = ((out_w + 1u) * params3d.input_width) / params3d.output_width;
    
    var max_val = -3.4028235e+38; // -FLT_MAX
    var found_valid = false;
    
    // Pool over the adaptive region
    for (var d: u32 = start_d; d < end_d; d++) {
        for (var h: u32 = start_h; h < end_h; h++) {
            for (var w: u32 = start_w; w < end_w; w++) {
                let input_idx = ((batch_idx * params3d.channels + channel) * params3d.input_depth + d) * 
                               params3d.input_height * params3d.input_width + h * params3d.input_width + w;
                let val = input3d[input_idx];
                
                if (!found_valid || val > max_val) {
                    max_val = val;
                    found_valid = true;
                }
            }
        }
    }
    
    // Set output value
    output3d[output_idx] = select(0.0, max_val, found_valid);
}

// ======== OPTIMIZED ADAPTIVE POOLING OPERATIONS ========

// Optimized adaptive average pooling with shared memory
@compute @workgroup_size(8, 8, 1)
fn adaptive_avg_pool2d_optimized(@builtin(global_invocation_id) global_id: vec3<u32>, 
                                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    // Shared memory for tile-based processing
    var shared_data: array<f32, 64>; // 8x8 workgroup
    let local_idx = local_id.y * 8u + local_id.x;
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    // Calculate adaptive pooling region with optimized bounds
    let start_h = (out_y * params.input_height) / params.output_height;
    let end_h = min(((out_y + 1u) * params.input_height + params.output_height - 1u) / params.output_height, params.input_height);
    let start_w = (out_x * params.input_width) / params.output_width;
    let end_w = min(((out_x + 1u) * params.input_width + params.output_width - 1u) / params.output_width, params.input_width);
    
    var sum = 0.0;
    var count = 0u;
    
    // Efficient pooling with loop unrolling for small regions
    let region_h = end_h - start_h;
    let region_w = end_w - start_w;
    
    if (region_h <= 2u && region_w <= 2u) {
        // Small region - unroll loops
        for (var h: u32 = start_h; h < end_h; h++) {
            for (var w: u32 = start_w; w < end_w; w++) {
                let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
                sum += input[input_idx];
                count++;
            }
        }
    } else {
        // Large region - use vectorized operations
        for (var h: u32 = start_h; h < end_h; h++) {
            for (var w: u32 = start_w; w < end_w; w++) {
                let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
                sum += input[input_idx];
                count++;
            }
        }
    }
    
    // Set output value with improved precision
    output[output_idx] = select(0.0, sum / f32(count), count > 0u);
}

// Optimized adaptive max pooling with shared memory
@compute @workgroup_size(8, 8, 1)
fn adaptive_max_pool2d_optimized(@builtin(global_invocation_id) global_id: vec3<u32>, 
                                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    // Calculate adaptive pooling region with optimized bounds
    let start_h = (out_y * params.input_height) / params.output_height;
    let end_h = min(((out_y + 1u) * params.input_height + params.output_height - 1u) / params.output_height, params.input_height);
    let start_w = (out_x * params.input_width) / params.output_width;
    let end_w = min(((out_x + 1u) * params.input_width + params.output_width - 1u) / params.output_width, params.input_width);
    
    var max_val = -3.4028235e+38; // -FLT_MAX
    var found_valid = false;
    
    // Efficient pooling with loop unrolling for small regions
    let region_h = end_h - start_h;
    let region_w = end_w - start_w;
    
    if (region_h <= 2u && region_w <= 2u) {
        // Small region - unroll loops for better performance
        for (var h: u32 = start_h; h < end_h; h++) {
            for (var w: u32 = start_w; w < end_w; w++) {
                let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
                let val = input[input_idx];
                
                if (!found_valid || val > max_val) {
                    max_val = val;
                    found_valid = true;
                }
            }
        }
    } else {
        // Large region - use vectorized operations
        for (var h: u32 = start_h; h < end_h; h++) {
            for (var w: u32 = start_w; w < end_w; w++) {
                let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
                let val = input[input_idx];
                
                max_val = max(max_val, val);
                found_valid = true;
            }
        }
    }
    
    // Set output value
    output[output_idx] = select(0.0, max_val, found_valid);
}

// Fractional adaptive pooling - supports non-integer scaling ratios
@compute @workgroup_size(8, 8, 1)
fn fractional_adaptive_pool2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    // Calculate fractional pooling region with sub-pixel precision
    let h_ratio = f32(params.input_height) / f32(params.output_height);
    let w_ratio = f32(params.input_width) / f32(params.output_width);
    
    let start_h_f = f32(out_y) * h_ratio;
    let end_h_f = f32(out_y + 1u) * h_ratio;
    let start_w_f = f32(out_x) * w_ratio;
    let end_w_f = f32(out_x + 1u) * w_ratio;
    
    let start_h = u32(floor(start_h_f));
    let end_h = min(u32(ceil(end_h_f)), params.input_height);
    let start_w = u32(floor(start_w_f));
    let end_w = min(u32(ceil(end_w_f)), params.input_width);
    
    var sum = 0.0;
    var weight_sum = 0.0;
    
    // Fractional pooling with bilinear interpolation weights
    for (var h: u32 = start_h; h < end_h; h++) {
        for (var w: u32 = start_w; w < end_w; w++) {
            let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
            let val = input[input_idx];
            
            // Calculate overlap weight
            let h_f = f32(h);
            let w_f = f32(w);
            let h_weight = min(h_f + 1.0, end_h_f) - max(h_f, start_h_f);
            let w_weight = min(w_f + 1.0, end_w_f) - max(w_f, start_w_f);
            let weight = h_weight * w_weight;
            
            sum += val * weight;
            weight_sum += weight;
        }
    }
    
    // Set output value with fractional weights
    output[output_idx] = select(0.0, sum / weight_sum, weight_sum > 0.0);
}

// Fractional max pooling - supports non-integer scaling ratios with max operation
@compute @workgroup_size(8, 8, 1)
fn fractional_max_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (channel >= params.channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.channels + channel) * params.output_height + out_y) * params.output_width + out_x;
    
    // Calculate fractional pooling region with sub-pixel precision
    let h_ratio = f32(params.input_height) / f32(params.output_height);
    let w_ratio = f32(params.input_width) / f32(params.output_width);
    
    let start_h_f = f32(out_y) * h_ratio;
    let end_h_f = f32(out_y + 1u) * h_ratio;
    let start_w_f = f32(out_x) * w_ratio;
    let end_w_f = f32(out_x + 1u) * w_ratio;
    
    let start_h = u32(floor(start_h_f));
    let end_h = min(u32(ceil(end_h_f)), params.input_height);
    let start_w = u32(floor(start_w_f));
    let end_w = min(u32(ceil(end_w_f)), params.input_width);
    
    var max_val = -3.4028235e+38; // -FLT_MAX
    var found_valid = false;
    
    // Fractional max pooling with overlap consideration
    for (var h: u32 = start_h; h < end_h; h++) {
        for (var w: u32 = start_w; w < end_w; w++) {
            let input_idx = ((batch_idx * params.channels + channel) * params.input_height + h) * params.input_width + w;
            let val = input[input_idx];
            
            // Calculate overlap weight for boundary consideration
            let h_f = f32(h);
            let w_f = f32(w);
            let h_weight = min(h_f + 1.0, end_h_f) - max(h_f, start_h_f);
            let w_weight = min(w_f + 1.0, end_w_f) - max(w_f, start_w_f);
            let weight = h_weight * w_weight;
            
            // Only consider pixels with significant overlap
            if (weight > 0.1 && (!found_valid || val > max_val)) {
                max_val = val;
                found_valid = true;
            }
        }
    }
    
    // Set output value
    output[output_idx] = select(0.0, max_val, found_valid);
}

// ======== ROI POOLING OPERATIONS ========

struct ROIParams {
    batch_size: u32,
    channels: u32,
    input_height: u32,
    input_width: u32,
    pooled_height: u32,
    pooled_width: u32,
    num_rois: u32,
    spatial_scale: f32,
    sampling_ratio: i32,
}

@group(0) @binding(0) var<storage, read> feature_maps: array<f32>;
@group(0) @binding(1) var<storage, read> rois: array<f32>;
@group(0) @binding(2) var<storage, read_write> roi_output: array<f32>;
@group(0) @binding(3) var<uniform> roi_params: ROIParams;

// ROI Pooling kernel for object detection
@compute @workgroup_size(8, 8, 1)
fn roi_pool2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let roi_idx = global_id.z;
    let ph = global_id.y;
    let pw = global_id.x;
    
    if (roi_idx >= roi_params.num_rois || ph >= roi_params.pooled_height || pw >= roi_params.pooled_width) {
        return;
    }
    
    // ROI format: [batch_id, x1, y1, x2, y2]
    let roi_start = roi_idx * 5u;
    let batch_id = u32(rois[roi_start]);
    let roi_start_w = rois[roi_start + 1u] * roi_params.spatial_scale;
    let roi_start_h = rois[roi_start + 2u] * roi_params.spatial_scale;
    let roi_end_w = rois[roi_start + 3u] * roi_params.spatial_scale;
    let roi_end_h = rois[roi_start + 4u] * roi_params.spatial_scale;
    
    // Force malformed ROIs to be 1x1
    let roi_width = max(roi_end_w - roi_start_w, 1.0);
    let roi_height = max(roi_end_h - roi_start_h, 1.0);
    
    let bin_size_h = roi_height / f32(roi_params.pooled_height);
    let bin_size_w = roi_width / f32(roi_params.pooled_width);
    
    // Find the start and end of the pooling region
    let hstart = u32(floor(roi_start_h + f32(ph) * bin_size_h));
    let wstart = u32(floor(roi_start_w + f32(pw) * bin_size_w));
    let hend = u32(ceil(roi_start_h + f32(ph + 1u) * bin_size_h));
    let wend = u32(ceil(roi_start_w + f32(pw + 1u) * bin_size_w));
    
    // Clamp to input boundaries
    let hstart_clamped = min(hstart, roi_params.input_height);
    let wstart_clamped = min(wstart, roi_params.input_width);
    let hend_clamped = min(hend, roi_params.input_height);
    let wend_clamped = min(wend, roi_params.input_width);
    
    // Pool over all channels
    for (var c: u32 = 0u; c < roi_params.channels; c++) {
        var max_val = -3.4028235e+38; // -FLT_MAX
        var found_valid = false;
        
        // Pool over the region
        for (var h: u32 = hstart_clamped; h < hend_clamped; h++) {
            for (var w: u32 = wstart_clamped; w < wend_clamped; w++) {
                let feature_idx = ((batch_id * roi_params.channels + c) * roi_params.input_height + h) * roi_params.input_width + w;
                let val = feature_maps[feature_idx];
                
                if (!found_valid || val > max_val) {
                    max_val = val;
                    found_valid = true;
                }
            }
        }
        
        // Store result
        let output_idx = ((roi_idx * roi_params.channels + c) * roi_params.pooled_height + ph) * roi_params.pooled_width + pw;
        roi_output[output_idx] = select(0.0, max_val, found_valid);
    }
}

// ROI Align kernel with bilinear interpolation
@compute @workgroup_size(8, 8, 1)
fn roi_align2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let roi_idx = global_id.z;
    let ph = global_id.y;
    let pw = global_id.x;
    
    if (roi_idx >= roi_params.num_rois || ph >= roi_params.pooled_height || pw >= roi_params.pooled_width) {
        return;
    }
    
    // ROI format: [batch_id, x1, y1, x2, y2]
    let roi_start = roi_idx * 5u;
    let batch_id = u32(rois[roi_start]);
    let roi_start_w = rois[roi_start + 1u] * roi_params.spatial_scale;
    let roi_start_h = rois[roi_start + 2u] * roi_params.spatial_scale;
    let roi_end_w = rois[roi_start + 3u] * roi_params.spatial_scale;
    let roi_end_h = rois[roi_start + 4u] * roi_params.spatial_scale;
    
    // Force malformed ROIs to be 1x1
    let roi_width = max(roi_end_w - roi_start_w, 1.0);
    let roi_height = max(roi_end_h - roi_start_h, 1.0);
    
    let bin_size_h = roi_height / f32(roi_params.pooled_height);
    let bin_size_w = roi_width / f32(roi_params.pooled_width);
    
    // We use roi_bin_grid to sample the grid and mimic integral over bin
    let roi_bin_grid_h = select(i32(ceil(bin_size_h)), roi_params.sampling_ratio, roi_params.sampling_ratio > 0);
    let roi_bin_grid_w = select(i32(ceil(bin_size_w)), roi_params.sampling_ratio, roi_params.sampling_ratio > 0);
    
    // Count is the number of sampling points
    let count = f32(roi_bin_grid_h * roi_bin_grid_w);
    
    // Pool over all channels
    for (var c: u32 = 0u; c < roi_params.channels; c++) {
        var output_val = 0.0;
        
        // Sample points within the bin
        for (var iy: i32 = 0; iy < roi_bin_grid_h; iy++) {
            let y = roi_start_h + (f32(ph) + (f32(iy) + 0.5) / f32(roi_bin_grid_h)) * bin_size_h;
            
            for (var ix: i32 = 0; ix < roi_bin_grid_w; ix++) {
                let x = roi_start_w + (f32(pw) + (f32(ix) + 0.5) / f32(roi_bin_grid_w)) * bin_size_w;
                
                // Bilinear interpolation
                let val = bilinear_interpolate(x, y, batch_id, c);
                output_val += val;
            }
        }
        output_val /= count;
        
        // Store result
        let output_idx = ((roi_idx * roi_params.channels + c) * roi_params.pooled_height + ph) * roi_params.pooled_width + pw;
        roi_output[output_idx] = output_val;
    }
}

// Helper function for bilinear interpolation
fn bilinear_interpolate(x: f32, y: f32, batch_id: u32, channel: u32) -> f32 {
    let x_low = i32(floor(x));
    let y_low = i32(floor(y));
    let x_high = x_low + 1;
    let y_high = y_low + 1;
    
    let lx = x - f32(x_low);
    let ly = y - f32(y_low);
    let hx = 1.0 - lx;
    let hy = 1.0 - ly;
    
    var v1 = 0.0;
    var v2 = 0.0;
    var v3 = 0.0;
    var v4 = 0.0;
    
    // Get values at the four corners with bounds checking
    if (x_low >= 0 && x_low < i32(roi_params.input_width) && y_low >= 0 && y_low < i32(roi_params.input_height)) {
        let idx = ((batch_id * roi_params.channels + channel) * roi_params.input_height + u32(y_low)) * roi_params.input_width + u32(x_low);
        v1 = feature_maps[idx];
    }
    
    if (x_high >= 0 && x_high < i32(roi_params.input_width) && y_low >= 0 && y_low < i32(roi_params.input_height)) {
        let idx = ((batch_id * roi_params.channels + channel) * roi_params.input_height + u32(y_low)) * roi_params.input_width + u32(x_high);
        v2 = feature_maps[idx];
    }
    
    if (x_low >= 0 && x_low < i32(roi_params.input_width) && y_high >= 0 && y_high < i32(roi_params.input_height)) {
        let idx = ((batch_id * roi_params.channels + channel) * roi_params.input_height + u32(y_high)) * roi_params.input_width + u32(x_low);
        v3 = feature_maps[idx];
    }
    
    if (x_high >= 0 && x_high < i32(roi_params.input_width) && y_high >= 0 && y_high < i32(roi_params.input_height)) {
        let idx = ((batch_id * roi_params.channels + channel) * roi_params.input_height + u32(y_high)) * roi_params.input_width + u32(x_high);
        v4 = feature_maps[idx];
    }
    
    // Bilinear interpolation
    let w1 = hx * hy;
    let w2 = lx * hy;
    let w3 = hx * ly;
    let w4 = lx * ly;
    
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}