// Fused operation compute shaders for kernel fusion optimization
// These kernels combine multiple operations to reduce memory bandwidth and improve performance

// ===== FUSED ADD + ACTIVATION KERNELS =====

struct BinaryOpParams {
    size: u32,
    broadcast_a: u32,  // 1 if tensor A should be broadcast, 0 otherwise
    broadcast_b: u32,  // 1 if tensor B should be broadcast, 0 otherwise
    size_a: u32,       // Size of tensor A
    size_b: u32,       // Size of tensor B
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: BinaryOpParams;

// Fused Add + ReLU kernel
@compute @workgroup_size(256)
fn add_relu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Add + ReLU
    let sum = input_a[idx_a] + input_b[idx_b];
    output[idx] = max(0.0, sum);
}

// Fused Add + Sigmoid kernel
@compute @workgroup_size(256)
fn add_sigmoid_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Add + Sigmoid
    let sum = input_a[idx_a] + input_b[idx_b];
    output[idx] = 1.0 / (1.0 + exp(-sum));
}

// Fused Add + Tanh kernel
@compute @workgroup_size(256)
fn add_tanh_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Add + Tanh
    let sum = input_a[idx_a] + input_b[idx_b];
    output[idx] = tanh(sum);
}

// Fused Add + GELU kernel
@compute @workgroup_size(256)
fn add_gelu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Add + GELU (approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))))
    let sum = input_a[idx_a] + input_b[idx_b];
    let x_cubed = sum * sum * sum;
    let inner = sqrt(2.0 / 3.14159265359) * (sum + 0.044715 * x_cubed);
    output[idx] = 0.5 * sum * (1.0 + tanh(inner));
}

// Fused Add + Swish (SiLU) kernel
@compute @workgroup_size(256)
fn add_swish_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Add + Swish (x * sigmoid(x))
    let sum = input_a[idx_a] + input_b[idx_b];
    let sigmoid = 1.0 / (1.0 + exp(-sum));
    output[idx] = sum * sigmoid;
}

// Fused Add + Mish kernel
@compute @workgroup_size(256)
fn add_mish_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Add + Mish (x * tanh(softplus(x)))
    let sum = input_a[idx_a] + input_b[idx_b];
    let softplus = log(1.0 + exp(sum));
    output[idx] = sum * tanh(softplus);
}

// Fused Add + LeakyReLU kernel
struct LeakyReLUParams {
    size: u32,
    broadcast_a: u32,
    broadcast_b: u32,
    size_a: u32,
    size_b: u32,
    negative_slope: f32,
}

@group(0) @binding(3) var<uniform> leaky_params: LeakyReLUParams;

@compute @workgroup_size(256)
fn add_leaky_relu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= leaky_params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % leaky_params.size_a, leaky_params.broadcast_a == 1u);
    let idx_b = select(idx, idx % leaky_params.size_b, leaky_params.broadcast_b == 1u);
    
    // Fused Add + LeakyReLU
    let sum = input_a[idx_a] + input_b[idx_b];
    output[idx] = select(leaky_params.negative_slope * sum, sum, sum > 0.0);
}

// ===== FUSED MULTIPLY + ACTIVATION KERNELS =====

// Fused Mul + ReLU kernel
@compute @workgroup_size(256)
fn mul_relu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Mul + ReLU
    let product = input_a[idx_a] * input_b[idx_b];
    output[idx] = max(0.0, product);
}

// Fused Mul + Sigmoid kernel
@compute @workgroup_size(256)
fn mul_sigmoid_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Mul + Sigmoid
    let product = input_a[idx_a] * input_b[idx_b];
    output[idx] = 1.0 / (1.0 + exp(-product));
}

// Fused Mul + Tanh kernel
@compute @workgroup_size(256)
fn mul_tanh_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting
    let idx_a = select(idx, idx % params.size_a, params.broadcast_a == 1u);
    let idx_b = select(idx, idx % params.size_b, params.broadcast_b == 1u);
    
    // Fused Mul + Tanh
    let product = input_a[idx_a] * input_b[idx_b];
    output[idx] = tanh(product);
}

// ===== FUSED NORMALIZATION + ACTIVATION KERNELS =====

struct NormActivationParams {
    size: u32,
    normalized_shape: u32,  // Size of the last dimension being normalized
    eps: f32,              // Epsilon for numerical stability
}

@group(0) @binding(0) var<storage, read> norm_input: array<f32>;
@group(0) @binding(1) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(2) var<storage, read> norm_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_output: array<f32>;
@group(0) @binding(4) var<uniform> norm_params: NormActivationParams;

// Fused LayerNorm + ReLU kernel
@compute @workgroup_size(256)
fn layernorm_relu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= norm_params.size) {
        return;
    }
    
    let normalized_shape = norm_params.normalized_shape;
    let batch_idx = idx / normalized_shape;
    let feature_idx = idx % normalized_shape;
    
    // Compute mean and variance for this batch
    var sum = 0.0;
    var sum_sq = 0.0;
    
    for (var i: u32 = 0u; i < normalized_shape; i++) {
        let val = norm_input[batch_idx * normalized_shape + i];
        sum += val;
        sum_sq += val * val;
    }
    
    let mean = sum / f32(normalized_shape);
    let variance = (sum_sq / f32(normalized_shape)) - (mean * mean);
    let std_dev = sqrt(variance + norm_params.eps);
    
    // Normalize and apply scale/bias
    let normalized = (norm_input[idx] - mean) / std_dev;
    let scaled = normalized * norm_weight[feature_idx] + norm_bias[feature_idx];
    
    // Apply ReLU activation
    norm_output[idx] = max(0.0, scaled);
}

// Fused LayerNorm + GELU kernel
@compute @workgroup_size(256)
fn layernorm_gelu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= norm_params.size) {
        return;
    }
    
    let normalized_shape = norm_params.normalized_shape;
    let batch_idx = idx / normalized_shape;
    let feature_idx = idx % normalized_shape;
    
    // Compute mean and variance for this batch
    var sum = 0.0;
    var sum_sq = 0.0;
    
    for (var i: u32 = 0u; i < normalized_shape; i++) {
        let val = norm_input[batch_idx * normalized_shape + i];
        sum += val;
        sum_sq += val * val;
    }
    
    let mean = sum / f32(normalized_shape);
    let variance = (sum_sq / f32(normalized_shape)) - (mean * mean);
    let std_dev = sqrt(variance + norm_params.eps);
    
    // Normalize and apply scale/bias
    let normalized = (norm_input[idx] - mean) / std_dev;
    let scaled = normalized * norm_weight[feature_idx] + norm_bias[feature_idx];
    
    // Apply GELU activation
    let x_cubed = scaled * scaled * scaled;
    let inner = sqrt(2.0 / 3.14159265359) * (scaled + 0.044715 * x_cubed);
    norm_output[idx] = 0.5 * scaled * (1.0 + tanh(inner));
}