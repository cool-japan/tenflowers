// Convolution compute shaders

// 1D Convolution parameters
struct Conv1DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_length: u32,
    out_length: u32,
    kernel_length: u32,
    stride: u32,
    pad_left: u32,
}

// 2D Convolution parameters
struct ConvParams {
    batch_size: u32,
    in_channels: u32,
    input_height: u32,
    input_width: u32,
    out_channels: u32,
    kernel_height: u32,
    kernel_width: u32,
    output_height: u32,
    output_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvParams;

// 1D Convolution kernel
@compute @workgroup_size(8, 8, 1)
fn conv1d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let out_channel = global_id.y;
    let out_pos = global_id.x;
    
    // Read 1D parameters (reusing ConvParams for compatibility)
    let batch_size = params.batch_size;
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;
    let in_length = params.input_height; // Reuse height for length
    let out_length = params.output_height; // Reuse output_height for out_length
    let kernel_length = params.kernel_height; // Reuse kernel_height for kernel_length
    let stride = params.stride_h; // Reuse stride_h for 1D stride
    let pad_left = params.pad_h; // Reuse pad_h for left padding
    
    if (batch_idx >= batch_size || out_channel >= out_channels || out_pos >= out_length) {
        return;
    }
    
    let output_idx = (batch_idx * out_channels + out_channel) * out_length + out_pos;
    
    var sum = 0.0;
    
    // 1D Convolution loop
    for (var ic: u32 = 0u; ic < in_channels; ic++) {
        for (var k: u32 = 0u; k < kernel_length; k++) {
            let in_pos = out_pos * stride + k;
            
            // Check bounds with padding
            if (in_pos >= pad_left) {
                let actual_pos = in_pos - pad_left;
                
                if (actual_pos < in_length) {
                    let input_idx = (batch_idx * in_channels + ic) * in_length + actual_pos;
                    let weight_idx = (out_channel * in_channels + ic) * kernel_length + k;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_channel];
    output[output_idx] = sum;
}

// 2D Convolution kernel
@compute @workgroup_size(8, 8, 1)
fn conv2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let out_channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= params.out_channels || out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * params.out_channels + out_channel) * params.output_height + out_y) * params.output_width + out_x;
    
    var sum = 0.0;
    
    // Convolution loop
    for (var ic: u32 = 0u; ic < params.in_channels; ic++) {
        for (var kh: u32 = 0u; kh < params.kernel_height; kh++) {
            for (var kw: u32 = 0u; kw < params.kernel_width; kw++) {
                let in_y = out_y * params.stride_h + kh;
                let in_x = out_x * params.stride_w + kw;
                
                // Check bounds with padding
                if (in_y >= params.pad_h && in_x >= params.pad_w) {
                    let actual_y = in_y - params.pad_h;
                    let actual_x = in_x - params.pad_w;
                    
                    if (actual_y < params.input_height && actual_x < params.input_width) {
                        let input_idx = ((batch_idx * params.in_channels + ic) * params.input_height + actual_y) * params.input_width + actual_x;
                        let weight_idx = ((out_channel * params.in_channels + ic) * params.kernel_height + kh) * params.kernel_width + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_channel];
    output[output_idx] = sum;
}

// Depthwise Convolution parameters
struct DepthwiseConvParams {
    batch_size: u32,
    in_channels: u32,
    input_height: u32,
    input_width: u32,
    multiplier: u32,
    kernel_height: u32,
    kernel_width: u32,
    output_height: u32,
    output_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}

// Note: For DepthwiseConv2D, we reuse the same input/weight/bias/output bindings but with a different uniform
@group(0) @binding(4) var<uniform> depthwise_params: DepthwiseConvParams;

// Depthwise 2D Convolution kernel
@compute @workgroup_size(8, 8, 1)
fn depthwise_conv2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % depthwise_params.batch_size;
    let out_channel = global_id.z / depthwise_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    let total_out_channels = depthwise_params.in_channels * depthwise_params.multiplier;
    
    if (out_channel >= total_out_channels || out_y >= depthwise_params.output_height || out_x >= depthwise_params.output_width) {
        return;
    }
    
    let output_idx = ((batch_idx * total_out_channels + out_channel) * depthwise_params.output_height + out_y) * depthwise_params.output_width + out_x;
    
    // Decode input channel and multiplier index from output channel
    let in_channel = out_channel / depthwise_params.multiplier;
    let multiplier_idx = out_channel % depthwise_params.multiplier;
    
    var sum = 0.0;
    
    // Depthwise convolution loop (only convolve with single input channel)
    for (var kh: u32 = 0u; kh < depthwise_params.kernel_height; kh++) {
        for (var kw: u32 = 0u; kw < depthwise_params.kernel_width; kw++) {
            let in_y = out_y * depthwise_params.stride_h + kh;
            let in_x = out_x * depthwise_params.stride_w + kw;
            
            // Check bounds with padding
            if (in_y >= depthwise_params.pad_h && in_x >= depthwise_params.pad_w) {
                let actual_y = in_y - depthwise_params.pad_h;
                let actual_x = in_x - depthwise_params.pad_w;
                
                if (actual_y < depthwise_params.input_height && actual_x < depthwise_params.input_width) {
                    let input_idx = ((batch_idx * depthwise_params.in_channels + in_channel) * depthwise_params.input_height + actual_y) * depthwise_params.input_width + actual_x;
                    let weight_idx = ((in_channel * depthwise_params.multiplier + multiplier_idx) * depthwise_params.kernel_height + kh) * depthwise_params.kernel_width + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_channel];
    output[output_idx] = sum;
}

// Batch normalization forward pass
struct BatchNormParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> bn_input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> running_mean: array<f32>;
@group(0) @binding(4) var<storage, read> running_var: array<f32>;
@group(0) @binding(5) var<storage, read_write> bn_output: array<f32>;
@group(0) @binding(6) var<uniform> bn_params: BatchNormParams;

@compute @workgroup_size(64)
fn batch_norm_inference(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_size = bn_params.batch_size * bn_params.channels * bn_params.height * bn_params.width;
    
    if (index >= total_size) {
        return;
    }
    
    // Calculate channel index
    let spatial_size = bn_params.height * bn_params.width;
    let channel_idx = (index / spatial_size) % bn_params.channels;
    
    // Normalize
    let mean = running_mean[channel_idx];
    let var = running_var[channel_idx];
    let normalized = (bn_input[index] - mean) / sqrt(var + bn_params.epsilon);
    
    // Scale and shift
    bn_output[index] = gamma[channel_idx] * normalized + beta[channel_idx];
}

// Batch Normalization Training Mode Shaders

// First pass: Compute channel means
@group(0) @binding(0) var<storage, read> bn_train_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> channel_means: array<f32>;
@group(0) @binding(2) var<uniform> bn_train_params: BatchNormParams;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn batch_norm_compute_mean(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let channel_idx = workgroup_id.x;
    let local_idx = local_id.x;
    let workgroup_size = 256u;
    
    if (channel_idx >= bn_train_params.channels) {
        return;
    }
    
    let spatial_size = bn_train_params.height * bn_train_params.width;
    let channel_size = bn_train_params.batch_size * spatial_size;
    
    // Initialize shared memory
    shared_sum[local_idx] = 0.0;
    
    // Each thread processes multiple elements
    let stride = workgroup_size;
    for (var i = local_idx; i < channel_size; i += stride) {
        let batch_idx = i / spatial_size;
        let spatial_idx = i % spatial_size;
        let tensor_idx = ((batch_idx * bn_train_params.channels + channel_idx) * spatial_size) + spatial_idx;
        shared_sum[local_idx] += bn_train_input[tensor_idx];
    }
    
    workgroupBarrier();
    
    // Reduction in shared memory
    var step = workgroup_size / 2u;
    while (step > 0u) {
        if (local_idx < step) {
            shared_sum[local_idx] += shared_sum[local_idx + step];
        }
        workgroupBarrier();
        step /= 2u;
    }
    
    // Write result
    if (local_idx == 0u) {
        channel_means[channel_idx] = shared_sum[0] / f32(channel_size);
    }
}

// Second pass: Compute channel variances
@group(0) @binding(3) var<storage, read_write> channel_vars: array<f32>;

@compute @workgroup_size(256)
fn batch_norm_compute_var(@builtin(global_invocation_id) global_id: vec3<u32>,
                         @builtin(local_invocation_id) local_id: vec3<u32>,
                         @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let channel_idx = workgroup_id.x;
    let local_idx = local_id.x;
    let workgroup_size = 256u;
    
    if (channel_idx >= bn_train_params.channels) {
        return;
    }
    
    let spatial_size = bn_train_params.height * bn_train_params.width;
    let channel_size = bn_train_params.batch_size * spatial_size;
    let mean = channel_means[channel_idx];
    
    // Initialize shared memory
    shared_sum[local_idx] = 0.0;
    
    // Each thread processes multiple elements
    let stride = workgroup_size;
    for (var i = local_idx; i < channel_size; i += stride) {
        let batch_idx = i / spatial_size;
        let spatial_idx = i % spatial_size;
        let tensor_idx = ((batch_idx * bn_train_params.channels + channel_idx) * spatial_size) + spatial_idx;
        let diff = bn_train_input[tensor_idx] - mean;
        shared_sum[local_idx] += diff * diff;
    }
    
    workgroupBarrier();
    
    // Reduction in shared memory
    var step = workgroup_size / 2u;
    while (step > 0u) {
        if (local_idx < step) {
            shared_sum[local_idx] += shared_sum[local_idx + step];
        }
        workgroupBarrier();
        step /= 2u;
    }
    
    // Write result
    if (local_idx == 0u) {
        channel_vars[channel_idx] = shared_sum[0] / f32(channel_size);
    }
}

// Third pass: Apply normalization using computed statistics
@group(0) @binding(4) var<storage, read> bn_train_gamma: array<f32>;
@group(0) @binding(5) var<storage, read> bn_train_beta: array<f32>;
@group(0) @binding(6) var<storage, read_write> bn_train_output: array<f32>;

@compute @workgroup_size(256)
fn batch_norm_apply_training(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_size = bn_train_params.batch_size * bn_train_params.channels * bn_train_params.height * bn_train_params.width;
    
    if (index >= total_size) {
        return;
    }
    
    // Calculate channel index
    let spatial_size = bn_train_params.height * bn_train_params.width;
    let channel_idx = (index / spatial_size) % bn_train_params.channels;
    
    // Get computed batch statistics
    let mean = channel_means[channel_idx];
    let var = channel_vars[channel_idx];
    
    // Normalize
    let normalized = (bn_train_input[index] - mean) / sqrt(var + bn_train_params.epsilon);
    
    // Scale and shift
    bn_train_output[index] = bn_train_gamma[channel_idx] * normalized + bn_train_beta[channel_idx];
}

// Fourth pass: Update running statistics with exponential moving average
struct MomentumParams {
    momentum: f32,
    channels: u32,
    _padding: array<u32, 2>,
}

@group(0) @binding(0) var<storage, read> batch_means: array<f32>;
@group(0) @binding(1) var<storage, read> batch_vars: array<f32>;
@group(0) @binding(2) var<storage, read_write> running_means: array<f32>;
@group(0) @binding(3) var<storage, read_write> running_vars: array<f32>;
@group(0) @binding(4) var<uniform> momentum_params: MomentumParams;

@compute @workgroup_size(64)
fn batch_norm_update_running_stats(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let channel_idx = global_id.x;
    
    if (channel_idx >= momentum_params.channels) {
        return;
    }
    
    let momentum = momentum_params.momentum;
    let batch_mean = batch_means[channel_idx];
    let batch_var = batch_vars[channel_idx];
    
    // Exponential moving average update
    // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    running_means[channel_idx] = (1.0 - momentum) * running_means[channel_idx] + momentum * batch_mean;
    
    // running_var = (1 - momentum) * running_var + momentum * batch_var
    running_vars[channel_idx] = (1.0 - momentum) * running_vars[channel_idx] + momentum * batch_var;
}

// 3D Convolution parameters
struct Conv3dParams {
    batch_size: u32,
    in_channels: u32,
    input_depth: u32,
    input_height: u32,
    input_width: u32,
    out_channels: u32,
    kernel_depth: u32,
    kernel_height: u32,
    kernel_width: u32,
    output_depth: u32,
    output_height: u32,
    output_width: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
}

// Note: For Conv3D, we reuse the same input/weight/bias/output bindings but with a different uniform
@group(0) @binding(4) var<uniform> conv3d_params: Conv3dParams;

// 3D Convolution kernel
@compute @workgroup_size(8, 8, 4)
fn conv3d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let depth_batch_channel = global_id.z;
    
    // Decode depth, batch, and channel from z dimension
    let out_depth = depth_batch_channel % conv3d_params.output_depth;
    let batch_channel = depth_batch_channel / conv3d_params.output_depth;
    let batch_idx = batch_channel / conv3d_params.out_channels;
    let out_channel = batch_channel % conv3d_params.out_channels;
    
    if (batch_idx >= conv3d_params.batch_size || 
        out_channel >= conv3d_params.out_channels || 
        out_depth >= conv3d_params.output_depth ||
        out_y >= conv3d_params.output_height || 
        out_x >= conv3d_params.output_width) {
        return;
    }
    
    let output_idx = ((((batch_idx * conv3d_params.out_channels + out_channel) * conv3d_params.output_depth + out_depth) 
                      * conv3d_params.output_height + out_y) * conv3d_params.output_width + out_x);
    
    var sum = 0.0;
    
    // 3D Convolution loop
    for (var ic: u32 = 0u; ic < conv3d_params.in_channels; ic++) {
        for (var kd: u32 = 0u; kd < conv3d_params.kernel_depth; kd++) {
            for (var kh: u32 = 0u; kh < conv3d_params.kernel_height; kh++) {
                for (var kw: u32 = 0u; kw < conv3d_params.kernel_width; kw++) {
                    let in_d = out_depth * conv3d_params.stride_d + kd;
                    let in_y = out_y * conv3d_params.stride_h + kh;
                    let in_x = out_x * conv3d_params.stride_w + kw;
                    
                    // Check bounds with padding
                    if (in_d >= conv3d_params.pad_d && in_y >= conv3d_params.pad_h && in_x >= conv3d_params.pad_w) {
                        let actual_d = in_d - conv3d_params.pad_d;
                        let actual_y = in_y - conv3d_params.pad_h;
                        let actual_x = in_x - conv3d_params.pad_w;
                        
                        if (actual_d < conv3d_params.input_depth && 
                            actual_y < conv3d_params.input_height && 
                            actual_x < conv3d_params.input_width) {
                            
                            let input_idx = ((((batch_idx * conv3d_params.in_channels + ic) * conv3d_params.input_depth + actual_d) 
                                             * conv3d_params.input_height + actual_y) * conv3d_params.input_width + actual_x);
                                             
                            let weight_idx = ((((out_channel * conv3d_params.in_channels + ic) * conv3d_params.kernel_depth + kd) 
                                              * conv3d_params.kernel_height + kh) * conv3d_params.kernel_width + kw);
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_channel];
    output[output_idx] = sum;
}

// ======== GPU CONVOLUTION OPTIMIZATIONS ========

// Winograd Convolution F(2x2, 3x3) Implementation
// Transforms 2x2 output tiles using 3x3 filters
struct WinogradParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_height: u32,
    input_width: u32,
    output_height: u32,
    output_width: u32,
    tile_h: u32,
    tile_w: u32,
}

@group(0) @binding(0) var<storage, read> winograd_input: array<f32>;
@group(0) @binding(1) var<storage, read> winograd_filter: array<f32>;
@group(0) @binding(2) var<storage, read> winograd_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> winograd_output: array<f32>;
@group(0) @binding(4) var<uniform> winograd_params: WinogradParams;

// Winograd transform matrices for F(2x2, 3x3)
// B^T matrix (4x4) for input transform
let BT_MATRIX = array<array<f32, 4>, 4>(
    array<f32, 4>(1.0, 0.0, -1.0, 0.0),
    array<f32, 4>(0.0, 1.0, 1.0, 0.0),
    array<f32, 4>(0.0, -1.0, 1.0, 0.0),
    array<f32, 4>(0.0, 1.0, 0.0, -1.0)
);

// G matrix (4x3) for filter transform
let G_MATRIX = array<array<f32, 3>, 4>(
    array<f32, 3>(1.0, 0.0, 0.0),
    array<f32, 3>(0.5, 0.5, 0.5),
    array<f32, 3>(0.5, -0.5, 0.5),
    array<f32, 3>(0.0, 0.0, 1.0)
);

// A^T matrix (2x4) for output transform
let AT_MATRIX = array<array<f32, 4>, 2>(
    array<f32, 4>(1.0, 1.0, 1.0, 0.0),
    array<f32, 4>(0.0, 1.0, -1.0, -1.0)
);

// Input transform: d = B^T * input_tile * B
@compute @workgroup_size(8, 8, 1)
fn winograd_input_transform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % winograd_params.batch_size;
    let channel_idx = global_id.z / winograd_params.batch_size;
    let tile_y = global_id.y;
    let tile_x = global_id.x;
    
    if (channel_idx >= winograd_params.in_channels || 
        tile_y >= winograd_params.tile_h || 
        tile_x >= winograd_params.tile_w) {
        return;
    }
    
    // Extract 4x4 input tile
    var input_tile: array<array<f32, 4>, 4>;
    let base_y = tile_y * 2u;
    let base_x = tile_x * 2u;
    
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let y = base_y + i;
            let x = base_x + j;
            
            if (y < winograd_params.input_height && x < winograd_params.input_width) {
                let input_idx = ((batch_idx * winograd_params.in_channels + channel_idx) * 
                                winograd_params.input_height + y) * winograd_params.input_width + x;
                input_tile[i][j] = winograd_input[input_idx];
            } else {
                input_tile[i][j] = 0.0; // Padding
            }
        }
    }
    
    // Compute B^T * input_tile * B
    var temp: array<array<f32, 4>, 4>;
    var transformed: array<array<f32, 4>, 4>;
    
    // First: temp = B^T * input_tile
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            temp[i][j] = 0.0;
            for (var k = 0u; k < 4u; k++) {
                temp[i][j] += BT_MATRIX[i][k] * input_tile[k][j];
            }
        }
    }
    
    // Second: transformed = temp * B
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            transformed[i][j] = 0.0;
            for (var k = 0u; k < 4u; k++) {
                transformed[i][j] += temp[i][k] * BT_MATRIX[j][k]; // Note: B = B^T for F(2x2,3x3)
            }
        }
    }
    
    // Store transformed tile
    let tile_idx = (batch_idx * winograd_params.in_channels + channel_idx) * 
                   winograd_params.tile_h * winograd_params.tile_w + 
                   tile_y * winograd_params.tile_w + tile_x;
    
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let element_idx = (i * 4u + j) * winograd_params.batch_size * 
                             winograd_params.in_channels * winograd_params.tile_h * 
                             winograd_params.tile_w + tile_idx;
            winograd_output[element_idx] = transformed[i][j];
        }
    }
}

// Filter transform: g = G * filter * G^T
@compute @workgroup_size(8, 8, 1)
fn winograd_filter_transform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_channel = global_id.y;
    let in_channel = global_id.x;
    
    if (out_channel >= winograd_params.out_channels || in_channel >= winograd_params.in_channels) {
        return;
    }
    
    // Extract 3x3 filter
    var filter_tile: array<array<f32, 3>, 3>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            let filter_idx = ((out_channel * winograd_params.in_channels + in_channel) * 3u + i) * 3u + j;
            filter_tile[i][j] = winograd_filter[filter_idx];
        }
    }
    
    // Compute G * filter * G^T
    var temp: array<array<f32, 3>, 4>;
    var transformed: array<array<f32, 4>, 4>;
    
    // First: temp = G * filter
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 3u; j++) {
            temp[i][j] = 0.0;
            for (var k = 0u; k < 3u; k++) {
                temp[i][j] += G_MATRIX[i][k] * filter_tile[k][j];
            }
        }
    }
    
    // Second: transformed = temp * G^T
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            transformed[i][j] = 0.0;
            for (var k = 0u; k < 3u; k++) {
                transformed[i][j] += temp[i][k] * G_MATRIX[j][k];
            }
        }
    }
    
    // Store transformed filter
    let filter_idx = out_channel * winograd_params.in_channels + in_channel;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let element_idx = (i * 4u + j) * winograd_params.out_channels * 
                             winograd_params.in_channels + filter_idx;
            winograd_output[element_idx] = transformed[i][j];
        }
    }
}

// Element-wise multiplication in frequency domain
@compute @workgroup_size(64)
fn winograd_element_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let element_idx = global_id.x;
    let total_elements = 16u * winograd_params.batch_size * winograd_params.out_channels * 
                        winograd_params.tile_h * winograd_params.tile_w;
    
    if (element_idx >= total_elements) {
        return;
    }
    
    // Decode indices
    let spatial_idx = element_idx % (winograd_params.batch_size * winograd_params.out_channels * 
                                    winograd_params.tile_h * winograd_params.tile_w);
    let freq_idx = element_idx / (winograd_params.batch_size * winograd_params.out_channels * 
                                 winograd_params.tile_h * winograd_params.tile_w);
    
    let tile_idx = spatial_idx % (winograd_params.tile_h * winograd_params.tile_w);
    let channel_batch_idx = spatial_idx / (winograd_params.tile_h * winograd_params.tile_w);
    let out_channel = channel_batch_idx % winograd_params.out_channels;
    let batch_idx = channel_batch_idx / winograd_params.out_channels;
    
    // Accumulate over input channels
    var sum = 0.0;
    for (var ic = 0u; ic < winograd_params.in_channels; ic++) {
        let input_idx = freq_idx * winograd_params.batch_size * winograd_params.in_channels * 
                       winograd_params.tile_h * winograd_params.tile_w + 
                       (batch_idx * winograd_params.in_channels + ic) * 
                       winograd_params.tile_h * winograd_params.tile_w + tile_idx;
        
        let filter_idx = freq_idx * winograd_params.out_channels * winograd_params.in_channels + 
                        out_channel * winograd_params.in_channels + ic;
        
        sum += winograd_input[input_idx] * winograd_filter[filter_idx];
    }
    
    winograd_output[element_idx] = sum;
}

// Output transform: output = A^T * frequency_output * A
@compute @workgroup_size(8, 8, 1)
fn winograd_output_transform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % winograd_params.batch_size;
    let channel_idx = global_id.z / winograd_params.batch_size;
    let tile_y = global_id.y;
    let tile_x = global_id.x;
    
    if (channel_idx >= winograd_params.out_channels || 
        tile_y >= winograd_params.tile_h || 
        tile_x >= winograd_params.tile_w) {
        return;
    }
    
    // Load 4x4 frequency domain tile
    var freq_tile: array<array<f32, 4>, 4>;
    let tile_idx = (batch_idx * winograd_params.out_channels + channel_idx) * 
                   winograd_params.tile_h * winograd_params.tile_w + 
                   tile_y * winograd_params.tile_w + tile_x;
    
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let element_idx = (i * 4u + j) * winograd_params.batch_size * 
                             winograd_params.out_channels * winograd_params.tile_h * 
                             winograd_params.tile_w + tile_idx;
            freq_tile[i][j] = winograd_input[element_idx];
        }
    }
    
    // Compute A^T * freq_tile * A
    var temp: array<array<f32, 4>, 2>;
    var output_tile: array<array<f32, 2>, 2>;
    
    // First: temp = A^T * freq_tile
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 4u; j++) {
            temp[i][j] = 0.0;
            for (var k = 0u; k < 4u; k++) {
                temp[i][j] += AT_MATRIX[i][k] * freq_tile[k][j];
            }
        }
    }
    
    // Second: output_tile = temp * A
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            output_tile[i][j] = 0.0;
            for (var k = 0u; k < 4u; k++) {
                output_tile[i][j] += temp[i][k] * AT_MATRIX[j][k];
            }
        }
    }
    
    // Store 2x2 output tile with bias
    let base_y = tile_y * 2u;
    let base_x = tile_x * 2u;
    
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            let y = base_y + i;
            let x = base_x + j;
            
            if (y < winograd_params.output_height && x < winograd_params.output_width) {
                let output_idx = ((batch_idx * winograd_params.out_channels + channel_idx) * 
                                 winograd_params.output_height + y) * winograd_params.output_width + x;
                winograd_output[output_idx] = output_tile[i][j] + winograd_bias[channel_idx];
            }
        }
    }
}

// FFT-based Convolution using frequency domain multiplication
// This requires pre-transformed input and filters in frequency domain
struct FFTConvParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    fft_height: u32,
    fft_width: u32,
    output_height: u32,
    output_width: u32,
    pad_h: u32,
    pad_w: u32,
}

@group(0) @binding(0) var<storage, read> fft_input_real: array<f32>;
@group(0) @binding(1) var<storage, read> fft_input_imag: array<f32>;
@group(0) @binding(2) var<storage, read> fft_filter_real: array<f32>;
@group(0) @binding(3) var<storage, read> fft_filter_imag: array<f32>;
@group(0) @binding(4) var<storage, read> fft_bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> fft_output_real: array<f32>;
@group(0) @binding(6) var<storage, read_write> fft_output_imag: array<f32>;
@group(0) @binding(7) var<uniform> fft_params: FFTConvParams;

// Complex multiplication in frequency domain
@compute @workgroup_size(8, 8, 1)
fn fft_conv_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % fft_params.batch_size;
    let out_channel = global_id.z / fft_params.batch_size;
    let freq_y = global_id.y;
    let freq_x = global_id.x;
    
    if (out_channel >= fft_params.out_channels || 
        freq_y >= fft_params.fft_height || 
        freq_x >= fft_params.fft_width) {
        return;
    }
    
    let freq_idx = freq_y * fft_params.fft_width + freq_x;
    let output_idx = ((batch_idx * fft_params.out_channels + out_channel) * 
                     fft_params.fft_height + freq_y) * fft_params.fft_width + freq_x;
    
    var sum_real = 0.0;
    var sum_imag = 0.0;
    
    // Accumulate over input channels
    for (var ic = 0u; ic < fft_params.in_channels; ic++) {
        let input_idx = ((batch_idx * fft_params.in_channels + ic) * 
                        fft_params.fft_height + freq_y) * fft_params.fft_width + freq_x;
        
        let filter_idx = ((out_channel * fft_params.in_channels + ic) * 
                         fft_params.fft_height + freq_y) * fft_params.fft_width + freq_x;
        
        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let input_real = fft_input_real[input_idx];
        let input_imag = fft_input_imag[input_idx];
        let filter_real = fft_filter_real[filter_idx];
        let filter_imag = fft_filter_imag[filter_idx];
        
        sum_real += input_real * filter_real - input_imag * filter_imag;
        sum_imag += input_real * filter_imag + input_imag * filter_real;
    }
    
    fft_output_real[output_idx] = sum_real;
    fft_output_imag[output_idx] = sum_imag;
}

// Extract real part and add bias after inverse FFT
@compute @workgroup_size(8, 8, 1)
fn fft_conv_finalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % fft_params.batch_size;
    let out_channel = global_id.z / fft_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= fft_params.out_channels || 
        out_y >= fft_params.output_height || 
        out_x >= fft_params.output_width) {
        return;
    }
    
    // Account for padding when reading from FFT output
    let fft_y = out_y + fft_params.pad_h;
    let fft_x = out_x + fft_params.pad_w;
    
    let fft_idx = ((batch_idx * fft_params.out_channels + out_channel) * 
                  fft_params.fft_height + fft_y) * fft_params.fft_width + fft_x;
    
    let output_idx = ((batch_idx * fft_params.out_channels + out_channel) * 
                     fft_params.output_height + out_y) * fft_params.output_width + out_x;
    
    // Extract real part and add bias
    let conv_result = fft_output_real[fft_idx];
    let bias_value = fft_bias[out_channel];
    
    fft_output_real[output_idx] = conv_result + bias_value;
}

// Optimized bias addition kernel
@compute @workgroup_size(64)
fn optimized_bias_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.batch_size * params.out_channels * params.output_height * params.output_width;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Decode channel index efficiently
    let spatial_size = params.output_height * params.output_width;
    let channel_idx = (idx / spatial_size) % params.out_channels;
    
    // Vectorized bias addition
    output[idx] = output[idx] + bias[channel_idx];
}

// Tiled convolution for memory efficiency
@compute @workgroup_size(8, 8, 1)
fn tiled_conv2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let out_channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= params.out_channels || 
        out_y >= params.output_height || 
        out_x >= params.output_width) {
        return;
    }
    
    // Use shared memory for input tile caching
    var shared_input: array<f32, 64>; // 8x8 tile
    
    let tile_y = out_y & ~7u; // Round down to tile boundary
    let tile_x = out_x & ~7u;
    let local_y = out_y & 7u;
    let local_x = out_x & 7u;
    
    var sum = 0.0;
    
    // Process input channels in tiles
    for (var ic = 0u; ic < params.in_channels; ic++) {
        // Load input tile into shared memory
        for (var ky = 0u; ky < params.kernel_height; ky++) {
            for (var kx = 0u; kx < params.kernel_width; kx++) {
                let in_y = tile_y + ky;
                let in_x = tile_x + kx;
                
                if (in_y < params.input_height && in_x < params.input_width) {
                    let input_idx = ((batch_idx * params.in_channels + ic) * 
                                    params.input_height + in_y) * params.input_width + in_x;
                    shared_input[ky * 8u + kx] = input[input_idx];
                } else {
                    shared_input[ky * 8u + kx] = 0.0;
                }
            }
        }
        
        // Compute convolution using shared memory
        for (var ky = 0u; ky < params.kernel_height; ky++) {
            for (var kx = 0u; kx < params.kernel_width; kx++) {
                let weight_idx = ((out_channel * params.in_channels + ic) * 
                                 params.kernel_height + ky) * params.kernel_width + kx;
                sum += shared_input[ky * 8u + kx] * weight[weight_idx];
            }
        }
    }
    
    let output_idx = ((batch_idx * params.out_channels + out_channel) * 
                     params.output_height + out_y) * params.output_width + out_x;
    output[output_idx] = sum + bias[out_channel];
}

// ===== FUSED CONV + BATCHNORM + ACTIVATION KERNELS =====

// Parameters for fused Conv + BatchNorm + ReLU operations
struct ConvBnReluParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_height: u32,
    input_width: u32,
    output_height: u32,
    output_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride_y: u32,
    stride_x: u32,
    pad_y: u32,
    pad_x: u32,
    eps: f32,  // BatchNorm epsilon for numerical stability
}

@group(0) @binding(0) var<storage, read> conv_bn_input: array<f32>;
@group(0) @binding(1) var<storage, read> conv_bn_weight: array<f32>;
@group(0) @binding(2) var<storage, read> conv_bn_bias: array<f32>;
@group(0) @binding(3) var<storage, read> bn_running_mean: array<f32>;
@group(0) @binding(4) var<storage, read> bn_running_var: array<f32>;
@group(0) @binding(5) var<storage, read> bn_weight: array<f32>;
@group(0) @binding(6) var<storage, read> bn_bias: array<f32>;
@group(0) @binding(7) var<storage, read_write> conv_bn_output: array<f32>;
@group(0) @binding(8) var<uniform> conv_bn_params: ConvBnReluParams;

// Fused Conv2D + BatchNorm + ReLU kernel
@compute @workgroup_size(8, 8, 1)
fn conv_bn_relu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % conv_bn_params.batch_size;
    let out_channel = global_id.z / conv_bn_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= conv_bn_params.out_channels || 
        out_y >= conv_bn_params.output_height || 
        out_x >= conv_bn_params.output_width) {
        return;
    }
    
    var sum = 0.0;
    
    // Convolution computation
    for (var ic = 0u; ic < conv_bn_params.in_channels; ic++) {
        for (var ky = 0u; ky < conv_bn_params.kernel_height; ky++) {
            for (var kx = 0u; kx < conv_bn_params.kernel_width; kx++) {
                let in_y = out_y * conv_bn_params.stride_y + ky;
                let in_x = out_x * conv_bn_params.stride_x + kx;
                
                // Apply padding
                if (in_y >= conv_bn_params.pad_y && 
                    in_x >= conv_bn_params.pad_x && 
                    (in_y - conv_bn_params.pad_y) < conv_bn_params.input_height && 
                    (in_x - conv_bn_params.pad_x) < conv_bn_params.input_width) {
                    
                    let adj_in_y = in_y - conv_bn_params.pad_y;
                    let adj_in_x = in_x - conv_bn_params.pad_x;
                    
                    let input_idx = ((batch_idx * conv_bn_params.in_channels + ic) * 
                                    conv_bn_params.input_height + adj_in_y) * 
                                   conv_bn_params.input_width + adj_in_x;
                    let weight_idx = ((out_channel * conv_bn_params.in_channels + ic) * 
                                     conv_bn_params.kernel_height + ky) * 
                                    conv_bn_params.kernel_width + kx;
                    
                    sum += conv_bn_input[input_idx] * conv_bn_weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bn_bias[out_channel];
    
    // Apply BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    let bn_mean = bn_running_mean[out_channel];
    let bn_var = bn_running_var[out_channel];
    let bn_scale = bn_weight[out_channel];
    let bn_shift = bn_bias[out_channel];
    
    let normalized = (sum - bn_mean) / sqrt(bn_var + conv_bn_params.eps);
    let bn_output = normalized * bn_scale + bn_shift;
    
    // Apply ReLU activation
    let final_output = max(0.0, bn_output);
    
    let output_idx = ((batch_idx * conv_bn_params.out_channels + out_channel) * 
                     conv_bn_params.output_height + out_y) * 
                    conv_bn_params.output_width + out_x;
    conv_bn_output[output_idx] = final_output;
}

// Fused Conv2D + BatchNorm + GELU kernel
@compute @workgroup_size(8, 8, 1)
fn conv_bn_gelu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % conv_bn_params.batch_size;
    let out_channel = global_id.z / conv_bn_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= conv_bn_params.out_channels || 
        out_y >= conv_bn_params.output_height || 
        out_x >= conv_bn_params.output_width) {
        return;
    }
    
    var sum = 0.0;
    
    // Convolution computation
    for (var ic = 0u; ic < conv_bn_params.in_channels; ic++) {
        for (var ky = 0u; ky < conv_bn_params.kernel_height; ky++) {
            for (var kx = 0u; kx < conv_bn_params.kernel_width; kx++) {
                let in_y = out_y * conv_bn_params.stride_y + ky;
                let in_x = out_x * conv_bn_params.stride_x + kx;
                
                // Apply padding
                if (in_y >= conv_bn_params.pad_y && 
                    in_x >= conv_bn_params.pad_x && 
                    (in_y - conv_bn_params.pad_y) < conv_bn_params.input_height && 
                    (in_x - conv_bn_params.pad_x) < conv_bn_params.input_width) {
                    
                    let adj_in_y = in_y - conv_bn_params.pad_y;
                    let adj_in_x = in_x - conv_bn_params.pad_x;
                    
                    let input_idx = ((batch_idx * conv_bn_params.in_channels + ic) * 
                                    conv_bn_params.input_height + adj_in_y) * 
                                   conv_bn_params.input_width + adj_in_x;
                    let weight_idx = ((out_channel * conv_bn_params.in_channels + ic) * 
                                     conv_bn_params.kernel_height + ky) * 
                                    conv_bn_params.kernel_width + kx;
                    
                    sum += conv_bn_input[input_idx] * conv_bn_weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bn_bias[out_channel];
    
    // Apply BatchNorm
    let bn_mean = bn_running_mean[out_channel];
    let bn_var = bn_running_var[out_channel];
    let bn_scale = bn_weight[out_channel];
    let bn_shift = bn_bias[out_channel];
    
    let normalized = (sum - bn_mean) / sqrt(bn_var + conv_bn_params.eps);
    let bn_output = normalized * bn_scale + bn_shift;
    
    // Apply GELU activation
    let x_cubed = bn_output * bn_output * bn_output;
    let inner = sqrt(2.0 / 3.14159265359) * (bn_output + 0.044715 * x_cubed);
    let final_output = 0.5 * bn_output * (1.0 + tanh(inner));
    
    let output_idx = ((batch_idx * conv_bn_params.out_channels + out_channel) * 
                     conv_bn_params.output_height + out_y) * 
                    conv_bn_params.output_width + out_x;
    conv_bn_output[output_idx] = final_output;
}

// Fused Conv2D + BatchNorm + Swish kernel
@compute @workgroup_size(8, 8, 1)
fn conv_bn_swish_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % conv_bn_params.batch_size;
    let out_channel = global_id.z / conv_bn_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= conv_bn_params.out_channels || 
        out_y >= conv_bn_params.output_height || 
        out_x >= conv_bn_params.output_width) {
        return;
    }
    
    var sum = 0.0;
    
    // Convolution computation
    for (var ic = 0u; ic < conv_bn_params.in_channels; ic++) {
        for (var ky = 0u; ky < conv_bn_params.kernel_height; ky++) {
            for (var kx = 0u; kx < conv_bn_params.kernel_width; kx++) {
                let in_y = out_y * conv_bn_params.stride_y + ky;
                let in_x = out_x * conv_bn_params.stride_x + kx;
                
                // Apply padding
                if (in_y >= conv_bn_params.pad_y && 
                    in_x >= conv_bn_params.pad_x && 
                    (in_y - conv_bn_params.pad_y) < conv_bn_params.input_height && 
                    (in_x - conv_bn_params.pad_x) < conv_bn_params.input_width) {
                    
                    let adj_in_y = in_y - conv_bn_params.pad_y;
                    let adj_in_x = in_x - conv_bn_params.pad_x;
                    
                    let input_idx = ((batch_idx * conv_bn_params.in_channels + ic) * 
                                    conv_bn_params.input_height + adj_in_y) * 
                                   conv_bn_params.input_width + adj_in_x;
                    let weight_idx = ((out_channel * conv_bn_params.in_channels + ic) * 
                                     conv_bn_params.kernel_height + ky) * 
                                    conv_bn_params.kernel_width + kx;
                    
                    sum += conv_bn_input[input_idx] * conv_bn_weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bn_bias[out_channel];
    
    // Apply BatchNorm
    let bn_mean = bn_running_mean[out_channel];
    let bn_var = bn_running_var[out_channel];
    let bn_scale = bn_weight[out_channel];
    let bn_shift = bn_bias[out_channel];
    
    let normalized = (sum - bn_mean) / sqrt(bn_var + conv_bn_params.eps);
    let bn_output = normalized * bn_scale + bn_shift;
    
    // Apply Swish activation (x * sigmoid(x))
    let sigmoid = 1.0 / (1.0 + exp(-bn_output));
    let final_output = bn_output * sigmoid;
    
    let output_idx = ((batch_idx * conv_bn_params.out_channels + out_channel) * 
                     conv_bn_params.output_height + out_y) * 
                    conv_bn_params.output_width + out_x;
    conv_bn_output[output_idx] = final_output;
}

// Fused Conv2D + ReLU kernel (without BatchNorm)
@compute @workgroup_size(8, 8, 1)
fn conv_relu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % params.batch_size;
    let out_channel = global_id.z / params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= params.out_channels || 
        out_y >= params.output_height || 
        out_x >= params.output_width) {
        return;
    }
    
    var sum = 0.0;
    
    // Convolution computation
    for (var ic = 0u; ic < params.in_channels; ic++) {
        for (var ky = 0u; ky < params.kernel_height; ky++) {
            for (var kx = 0u; kx < params.kernel_width; kx++) {
                let in_y = out_y * params.stride_y + ky;
                let in_x = out_x * params.stride_x + kx;
                
                // Apply padding
                if (in_y >= params.pad_y && 
                    in_x >= params.pad_x && 
                    (in_y - params.pad_y) < params.input_height && 
                    (in_x - params.pad_x) < params.input_width) {
                    
                    let adj_in_y = in_y - params.pad_y;
                    let adj_in_x = in_x - params.pad_x;
                    
                    let input_idx = ((batch_idx * params.in_channels + ic) * 
                                    params.input_height + adj_in_y) * 
                                   params.input_width + adj_in_x;
                    let weight_idx = ((out_channel * params.in_channels + ic) * 
                                     params.kernel_height + ky) * 
                                    params.kernel_width + kx;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias and apply ReLU (double fusion optimization)
    let final_output = max(0.0, sum + bias[out_channel]);
    
    let output_idx = ((batch_idx * params.out_channels + out_channel) * 
                     params.output_height + out_y) * params.output_width + out_x;
    output[output_idx] = final_output;
}

// ===== IM2COL CONVOLUTION IMPLEMENTATION =====

// Im2col parameters structure
struct Im2ColParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
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
    dilation_h: u32,
    dilation_w: u32,
}

@group(0) @binding(0) var<storage, read> im2col_input: array<f32>;
@group(0) @binding(1) var<storage, read> im2col_weight: array<f32>;
@group(0) @binding(2) var<storage, read> im2col_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> im2col_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> im2col_matrix: array<f32>;
@group(0) @binding(5) var<uniform> im2col_params: Im2ColParams;

// Im2col transformation kernel
// Converts input tensor to column matrix for matrix multiplication
@compute @workgroup_size(16, 16, 1)
fn im2col_transform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % im2col_params.batch_size;
    let channel_idx = global_id.z / im2col_params.batch_size;
    let col_idx = global_id.y;
    let row_idx = global_id.x;
    
    let kernel_size = im2col_params.kernel_height * im2col_params.kernel_width;
    let output_size = im2col_params.output_height * im2col_params.output_width;
    let matrix_height = im2col_params.in_channels * kernel_size;
    
    if (channel_idx >= im2col_params.in_channels || 
        col_idx >= output_size || 
        row_idx >= kernel_size) {
        return;
    }
    
    // Decode kernel position
    let kernel_y = row_idx / im2col_params.kernel_width;
    let kernel_x = row_idx % im2col_params.kernel_width;
    
    // Decode output position
    let out_y = col_idx / im2col_params.output_width;
    let out_x = col_idx % im2col_params.output_width;
    
    // Calculate input position with dilation
    let in_y = out_y * im2col_params.stride_h + kernel_y * im2col_params.dilation_h;
    let in_x = out_x * im2col_params.stride_w + kernel_x * im2col_params.dilation_w;
    
    // Check bounds with padding
    var value = 0.0;
    if (in_y >= im2col_params.pad_h && in_x >= im2col_params.pad_w) {
        let actual_y = in_y - im2col_params.pad_h;
        let actual_x = in_x - im2col_params.pad_w;
        
        if (actual_y < im2col_params.input_height && actual_x < im2col_params.input_width) {
            let input_idx = ((batch_idx * im2col_params.in_channels + channel_idx) * 
                            im2col_params.input_height + actual_y) * 
                           im2col_params.input_width + actual_x;
            value = im2col_input[input_idx];
        }
    }
    
    // Store in im2col matrix
    // Matrix layout: [batch_size, in_channels * kernel_size, output_height * output_width]
    let matrix_row = channel_idx * kernel_size + row_idx;
    let matrix_idx = (batch_idx * matrix_height + matrix_row) * output_size + col_idx;
    im2col_matrix[matrix_idx] = value;
}

// Matrix multiplication kernel for im2col convolution
// Multiplies weight matrix with im2col matrix
@compute @workgroup_size(16, 16, 1)  
fn im2col_gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % im2col_params.batch_size;
    let out_channel = global_id.z / im2col_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= im2col_params.out_channels || 
        out_y >= im2col_params.output_height || 
        out_x >= im2col_params.output_width) {
        return;
    }
    
    let kernel_size = im2col_params.kernel_height * im2col_params.kernel_width;
    let output_size = im2col_params.output_height * im2col_params.output_width;
    let matrix_height = im2col_params.in_channels * kernel_size;
    
    let col_idx = out_y * im2col_params.output_width + out_x;
    
    var sum = 0.0;
    
    // Matrix multiplication: weight[out_channel, :] * im2col_matrix[:, col_idx]
    for (var k = 0u; k < matrix_height; k++) {
        let weight_idx = out_channel * matrix_height + k;
        let matrix_idx = (batch_idx * matrix_height + k) * output_size + col_idx;
        sum += im2col_weight[weight_idx] * im2col_matrix[matrix_idx];
    }
    
    // Add bias
    sum += im2col_bias[out_channel];
    
    let output_idx = ((batch_idx * im2col_params.out_channels + out_channel) * 
                     im2col_params.output_height + out_y) * 
                    im2col_params.output_width + out_x;
    im2col_output[output_idx] = sum;
}

// Tiled matrix multiplication for large convolutions
@compute @workgroup_size(16, 16, 1)
fn im2col_tiled_gemm(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch_idx = global_id.z % im2col_params.batch_size;
    let out_channel = global_id.z / im2col_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= im2col_params.out_channels || 
        out_y >= im2col_params.output_height || 
        out_x >= im2col_params.output_width) {
        return;
    }
    
    let kernel_size = im2col_params.kernel_height * im2col_params.kernel_width;
    let output_size = im2col_params.output_height * im2col_params.output_width;
    let matrix_height = im2col_params.in_channels * kernel_size;
    
    // Shared memory for tiling
    var tile_a: array<array<f32, 16>, 16>;
    var tile_b: array<array<f32, 16>, 16>;
    
    let col_idx = out_y * im2col_params.output_width + out_x;
    let tile_size = 16u;
    let num_tiles = (matrix_height + tile_size - 1u) / tile_size;
    
    var sum = 0.0;
    
    // Tiled matrix multiplication
    for (var tile = 0u; tile < num_tiles; tile++) {
        let k_base = tile * tile_size;
        let local_k = local_id.x;
        let k_idx = k_base + local_k;
        
        // Load tile A (weight matrix)
        if (k_idx < matrix_height) {
            let weight_idx = out_channel * matrix_height + k_idx;
            tile_a[local_id.y][local_id.x] = im2col_weight[weight_idx];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }
        
        // Load tile B (im2col matrix)
        if (k_idx < matrix_height) {
            let matrix_idx = (batch_idx * matrix_height + k_idx) * output_size + col_idx;
            tile_b[local_id.x][local_id.y] = im2col_matrix[matrix_idx];
        } else {
            tile_b[local_id.x][local_id.y] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial sum
        for (var k = 0u; k < tile_size; k++) {
            sum += tile_a[local_id.y][k] * tile_b[k][local_id.y];
        }
        
        workgroupBarrier();
    }
    
    // Add bias and store result
    sum += im2col_bias[out_channel];
    
    let output_idx = ((batch_idx * im2col_params.out_channels + out_channel) * 
                     im2col_params.output_height + out_y) * 
                    im2col_params.output_width + out_x;
    im2col_output[output_idx] = sum;
}

// Fused im2col + activation kernels

// Im2col + ReLU kernel
@compute @workgroup_size(16, 16, 1)
fn im2col_relu_gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % im2col_params.batch_size;
    let out_channel = global_id.z / im2col_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= im2col_params.out_channels || 
        out_y >= im2col_params.output_height || 
        out_x >= im2col_params.output_width) {
        return;
    }
    
    let kernel_size = im2col_params.kernel_height * im2col_params.kernel_width;
    let output_size = im2col_params.output_height * im2col_params.output_width;
    let matrix_height = im2col_params.in_channels * kernel_size;
    
    let col_idx = out_y * im2col_params.output_width + out_x;
    
    var sum = 0.0;
    
    // Matrix multiplication
    for (var k = 0u; k < matrix_height; k++) {
        let weight_idx = out_channel * matrix_height + k;
        let matrix_idx = (batch_idx * matrix_height + k) * output_size + col_idx;
        sum += im2col_weight[weight_idx] * im2col_matrix[matrix_idx];
    }
    
    // Add bias and apply ReLU
    let final_output = max(0.0, sum + im2col_bias[out_channel]);
    
    let output_idx = ((batch_idx * im2col_params.out_channels + out_channel) * 
                     im2col_params.output_height + out_y) * 
                    im2col_params.output_width + out_x;
    im2col_output[output_idx] = final_output;
}

// Im2col + GELU kernel
@compute @workgroup_size(16, 16, 1)
fn im2col_gelu_gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % im2col_params.batch_size;
    let out_channel = global_id.z / im2col_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_channel >= im2col_params.out_channels || 
        out_y >= im2col_params.output_height || 
        out_x >= im2col_params.output_width) {
        return;
    }
    
    let kernel_size = im2col_params.kernel_height * im2col_params.kernel_width;
    let output_size = im2col_params.output_height * im2col_params.output_width;
    let matrix_height = im2col_params.in_channels * kernel_size;
    
    let col_idx = out_y * im2col_params.output_width + out_x;
    
    var sum = 0.0;
    
    // Matrix multiplication
    for (var k = 0u; k < matrix_height; k++) {
        let weight_idx = out_channel * matrix_height + k;
        let matrix_idx = (batch_idx * matrix_height + k) * output_size + col_idx;
        sum += im2col_weight[weight_idx] * im2col_matrix[matrix_idx];
    }
    
    // Add bias
    sum += im2col_bias[out_channel];
    
    // Apply GELU activation
    let x_cubed = sum * sum * sum;
    let inner = sqrt(2.0 / 3.14159265359) * (sum + 0.044715 * x_cubed);
    let final_output = 0.5 * sum * (1.0 + tanh(inner));
    
    let output_idx = ((batch_idx * im2col_params.out_channels + out_channel) * 
                     im2col_params.output_height + out_y) * 
                    im2col_params.output_width + out_x;
    im2col_output[output_idx] = final_output;
}

// Memory coalescing optimized im2col kernel
@compute @workgroup_size(256, 1, 1)
fn im2col_coalesced_transform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_idx = global_id.x;
    let batch_idx = global_id.z % im2col_params.batch_size;
    let channel_idx = global_id.z / im2col_params.batch_size;
    
    let kernel_size = im2col_params.kernel_height * im2col_params.kernel_width;
    let output_size = im2col_params.output_height * im2col_params.output_width;
    let total_elements = kernel_size * output_size;
    
    if (channel_idx >= im2col_params.in_channels || global_idx >= total_elements) {
        return;
    }
    
    // Decode positions with coalesced memory access pattern
    let kernel_pos = global_idx / output_size;
    let output_pos = global_idx % output_size;
    
    let kernel_y = kernel_pos / im2col_params.kernel_width;
    let kernel_x = kernel_pos % im2col_params.kernel_width;
    
    let out_y = output_pos / im2col_params.output_width;
    let out_x = output_pos % im2col_params.output_width;
    
    // Calculate input position with dilation
    let in_y = out_y * im2col_params.stride_h + kernel_y * im2col_params.dilation_h;
    let in_x = out_x * im2col_params.stride_w + kernel_x * im2col_params.dilation_w;
    
    // Check bounds with padding
    var value = 0.0;
    if (in_y >= im2col_params.pad_h && in_x >= im2col_params.pad_w) {
        let actual_y = in_y - im2col_params.pad_h;
        let actual_x = in_x - im2col_params.pad_w;
        
        if (actual_y < im2col_params.input_height && actual_x < im2col_params.input_width) {
            let input_idx = ((batch_idx * im2col_params.in_channels + channel_idx) * 
                            im2col_params.input_height + actual_y) * 
                           im2col_params.input_width + actual_x;
            value = im2col_input[input_idx];
        }
    }
    
    // Store with coalesced memory access
    let matrix_row = channel_idx * kernel_size + kernel_pos;
    let matrix_idx = (batch_idx * im2col_params.in_channels * kernel_size + matrix_row) * 
                    output_size + output_pos;
    im2col_matrix[matrix_idx] = value;
}

// ConvTranspose2D (Deconvolution) parameters
struct ConvTransposeParams {
    batch_size: u32,
    in_channels: u32,
    input_height: u32,
    input_width: u32,
    out_channels: u32,
    kernel_height: u32,
    kernel_width: u32,
    output_height: u32,
    output_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    output_pad_h: u32,
    output_pad_w: u32,
}

@group(0) @binding(4) var<uniform> conv_transpose_params: ConvTransposeParams;

// ConvTranspose2D kernel
@compute @workgroup_size(8, 8, 1)
fn conv_transpose2d_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z % conv_transpose_params.batch_size;
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (batch_idx >= conv_transpose_params.batch_size || 
        out_y >= conv_transpose_params.output_height || 
        out_x >= conv_transpose_params.output_width) {
        return;
    }
    
    // For each output channel
    for (var oc: u32 = 0u; oc < conv_transpose_params.out_channels; oc++) {
        var sum: f32 = 0.0;
        
        // For each input channel
        for (var ic: u32 = 0u; ic < conv_transpose_params.in_channels; ic++) {
            // For each kernel position
            for (var kh: u32 = 0u; kh < conv_transpose_params.kernel_height; kh++) {
                for (var kw: u32 = 0u; kw < conv_transpose_params.kernel_width; kw++) {
                    // Calculate corresponding input position
                    // For transposed convolution: input_pos = (output_pos + pad - kernel_pos) / stride
                    
                    let input_y_raw = i32(out_y + conv_transpose_params.pad_h) - i32(kh);
                    let input_x_raw = i32(out_x + conv_transpose_params.pad_w) - i32(kw);
                    
                    // Check if input position is valid and aligned with stride
                    if (input_y_raw >= 0 && input_x_raw >= 0 && 
                        u32(input_y_raw) % conv_transpose_params.stride_h == 0u &&
                        u32(input_x_raw) % conv_transpose_params.stride_w == 0u) {
                        
                        let input_y = u32(input_y_raw) / conv_transpose_params.stride_h;
                        let input_x = u32(input_x_raw) / conv_transpose_params.stride_w;
                        
                        if (input_y < conv_transpose_params.input_height && 
                            input_x < conv_transpose_params.input_width) {
                            
                            // Calculate indices
                            let input_idx = ((batch_idx * conv_transpose_params.in_channels + ic) * 
                                           conv_transpose_params.input_height + input_y) * 
                                           conv_transpose_params.input_width + input_x;
                            
                            // Weight format: [in_channels, out_channels, kernel_height, kernel_width]
                            let weight_idx = (((ic * conv_transpose_params.out_channels + oc) * 
                                             conv_transpose_params.kernel_height + kh) * 
                                             conv_transpose_params.kernel_width + kw);
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias and store output
        let output_idx = ((batch_idx * conv_transpose_params.out_channels + oc) * 
                         conv_transpose_params.output_height + out_y) * 
                         conv_transpose_params.output_width + out_x;
        
        output[output_idx] = sum + bias[oc];
    }
}