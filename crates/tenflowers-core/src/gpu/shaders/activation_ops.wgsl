// Activation function compute shaders

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// ReLU activation
@compute @workgroup_size(64)
fn relu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    output[index] = max(input[index], 0.0);
}

// Sigmoid activation
@compute @workgroup_size(64)
fn sigmoid_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    output[index] = 1.0 / (1.0 + exp(-input[index]));
}

// Tanh activation
@compute @workgroup_size(64)
fn tanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    output[index] = tanh(input[index]);
}

// GELU activation (approximation)
@compute @workgroup_size(64)
fn gelu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    let x = input[index];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608;
    let a = 0.044715;
    let inner = sqrt_2_over_pi * (x + a * x * x * x);
    output[index] = 0.5 * x * (1.0 + tanh(inner));
}

// Swish activation
@compute @workgroup_size(64)
fn swish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    let x = input[index];
    output[index] = x / (1.0 + exp(-x));
}

// ELU activation
@compute @workgroup_size(64)
fn elu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    let x = input[index];
    if (x >= 0.0) {
        output[index] = x;
    } else {
        output[index] = exp(x) - 1.0;
    }
}

// Leaky ReLU activation
@compute @workgroup_size(64)
fn leaky_relu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    let x = input[index];
    let negative_slope = 0.01;
    output[index] = max(x, negative_slope * x);
}

// Mish activation
@compute @workgroup_size(64)
fn mish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    let x = input[index];
    // mish(x) = x * tanh(ln(1 + exp(x)))
    // For numerical stability, we use ln(1 + exp(x)) = softplus(x)
    let softplus = log(1.0 + exp(x));
    output[index] = x * tanh(softplus);
}

// HardSwish activation
@compute @workgroup_size(64)
fn hard_swish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    
    let x = input[index];
    // hard_swish(x) = x * relu6(x + 3) / 6
    // relu6(y) = min(max(y, 0), 6)
    let y = x + 3.0;
    let relu6_val = clamp(y, 0.0, 6.0);
    output[index] = x * relu6_val / 6.0;
}

// PReLU activation with separate alpha buffer and metadata for channel-wise alpha
@group(0) @binding(2) var<storage, read> alpha: array<f32>;

// Metadata struct for tensor shape information
struct ActivationMetadata {
    total_elements: u32,
    batch_size: u32,
    channels: u32, 
    height: u32,
    width: u32,
    is_channelwise: u32, // 0 = scalar alpha, 1 = channel-wise alpha
    padding1: u32,
    padding2: u32,
}

@group(0) @binding(3) var<uniform> metadata: ActivationMetadata;

@compute @workgroup_size(64)
fn prelu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= metadata.total_elements) {
        return;
    }
    
    let x = input[index];
    
    var alpha_val: f32;
    
    if (metadata.is_channelwise == 1u) {
        // Calculate channel index for channel-wise alpha
        // Assuming NCHW format: [batch, channels, height, width]
        let spatial_size = metadata.height * metadata.width;
        let channel_idx = (index / spatial_size) % metadata.channels;
        alpha_val = alpha[channel_idx];
    } else {
        // Scalar alpha - use alpha[0] for all elements
        alpha_val = alpha[0];
    }
    
    if (x >= 0.0) {
        output[index] = x;
    } else {
        output[index] = alpha_val * x;
    }
}