// Quantization operation compute shaders
// These kernels implement quantization and dequantization operations

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read> params: array<f32>; // [scale, zero_point, qmin, qmax]

// Quantization kernel - converts float to quantized integer
@compute @workgroup_size(64)
fn quantize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let scale = params[0];
    let zero_point = params[1];
    let qmin = params[2];
    let qmax = params[3];
    
    let val = input[index];
    let quantized = round((val / scale) + zero_point);
    output[index] = i32(clamp(quantized, qmin, qmax));
}

// INT8 specific quantization shader
@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_i8: array<i32>; // Using i32 for i8 values

@compute @workgroup_size(64)
fn quantize_int8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i8)) {
        return;
    }
    
    let scale = params[0];
    let zero_point = params[1];
    
    let val = input_f32[index];
    let quantized = round((val / scale) + zero_point);
    output_i8[index] = i32(clamp(quantized, -128.0, 127.0));
}

// INT4 specific quantization shader
@compute @workgroup_size(64)
fn quantize_int4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_i8)) {
        return;
    }
    
    let scale = params[0];
    let zero_point = params[1];
    
    let val = input_f32[index];
    let quantized = round((val / scale) + zero_point);
    output_i8[index] = i32(clamp(quantized, -8.0, 7.0));
}

// Dequantization shader - converts quantized integer back to float
@group(0) @binding(0) var<storage, read> input_quantized: array<i32>;
@group(0) @binding(1) var<storage, read_write> output_float: array<f32>;

@compute @workgroup_size(64)
fn dequantize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_float)) {
        return;
    }
    
    let scale = params[0];
    let zero_point = params[1];
    
    let quantized_val = input_quantized[index];
    output_float[index] = (f32(quantized_val) - zero_point) * scale;
}

// Fake quantization shader - simulates quantization effects during training
@group(0) @binding(0) var<storage, read> input_fake: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_fake: array<f32>;

@compute @workgroup_size(64)
fn fake_quantize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_fake)) {
        return;
    }
    
    let scale = params[0];
    let zero_point = params[1];
    let qmin = params[2];
    let qmax = params[3];
    
    let val = input_fake[index];
    let quantized = round((val / scale) + zero_point);
    let clamped = clamp(quantized, qmin, qmax);
    output_fake[index] = (clamped - zero_point) * scale;
}

// Dynamic quantization shader - calculates min/max on GPU
@group(0) @binding(0) var<storage, read> input_dynamic: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_dynamic: array<i32>;
@group(0) @binding(2) var<storage, read_write> min_max_output: array<f32>; // [min, max]

var<workgroup> shared_data: array<f32, 64>;

@compute @workgroup_size(64)
fn dynamic_quantize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let local_idx = global_id.x % 64u;
    
    // Initialize shared memory
    shared_data[local_idx] = 0.0;
    
    // Load data if within bounds
    var val = 0.0;
    if (index < arrayLength(&input_dynamic)) {
        val = input_dynamic[index];
    }
    
    shared_data[local_idx] = val;
    workgroupBarrier();
    
    // Find min/max within workgroup
    var min_val = val;
    var max_val = val;
    
    for (var i = 0u; i < 64u; i++) {
        let shared_val = shared_data[i];
        min_val = min(min_val, shared_val);
        max_val = max(max_val, shared_val);
    }
    
    // Store results (first thread in workgroup)
    if (local_idx == 0u) {
        min_max_output[0] = min_val;
        min_max_output[1] = max_val;
    }
    
    workgroupBarrier();
    
    // Perform quantization with calculated min/max
    let abs_max = max(abs(min_val), abs(max_val));
    let scale = abs_max / 127.0;
    
    if (index < arrayLength(&output_dynamic)) {
        let quantized = round(val / scale);
        output_dynamic[index] = i32(clamp(quantized, -128.0, 127.0));
    }
}

// Per-channel quantization shader
@group(0) @binding(0) var<storage, read> input_per_channel: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_per_channel: array<i32>;
@group(0) @binding(2) var<storage, read> channel_params: array<f32>; // [scale0, zero_point0, scale1, zero_point1, ...]
@group(0) @binding(3) var<storage, read> channel_metadata: array<u32>; // [channel_axis, num_channels, stride]

@compute @workgroup_size(64)
fn per_channel_quantize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output_per_channel)) {
        return;
    }
    
    let channel_axis = channel_metadata[0];
    let num_channels = channel_metadata[1];
    let stride = channel_metadata[2];
    
    // Calculate which channel this element belongs to
    let channel_idx = (index / stride) % num_channels;
    
    let scale = channel_params[channel_idx * 2u];
    let zero_point = channel_params[channel_idx * 2u + 1u];
    
    let val = input_per_channel[index];
    let quantized = round((val / scale) + zero_point);
    output_per_channel[index] = i32(clamp(quantized, -128.0, 127.0));
}