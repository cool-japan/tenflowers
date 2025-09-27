// Unary operation compute shaders
// These kernels implement element-wise unary operations

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Logarithm operation (natural log)
@compute @workgroup_size(256)
fn log_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = log(input[index]);
}

// Negation operation
@compute @workgroup_size(256)
fn neg_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = -input[index];
}

// Square root operation
@compute @workgroup_size(256)
fn sqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = sqrt(input[index]);
}

// Absolute value operation
@compute @workgroup_size(256)
fn abs_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = abs(input[index]);
}

// Exponential operation
@compute @workgroup_size(256)
fn exp_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = exp(input[index]);
}

// Sine operation
@compute @workgroup_size(256)
fn sin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = sin(input[index]);
}

// Cosine operation
@compute @workgroup_size(256)
fn cos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = cos(input[index]);
}

// Tangent operation
@compute @workgroup_size(256)
fn tan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = tan(input[index]);
}

// Reciprocal operation (1/x)
@compute @workgroup_size(256)
fn recip_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = 1.0 / input[index];
}

// Floor operation
@compute @workgroup_size(256)
fn floor_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = floor(input[index]);
}

// Ceiling operation
@compute @workgroup_size(256)
fn ceil_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = ceil(input[index]);
}

// Round operation
@compute @workgroup_size(256)
fn round_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = round(input[index]);
}