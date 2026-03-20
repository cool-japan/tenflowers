// Unary operation compute shaders for f64 type

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn log_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = log(input[index]);
}

@compute @workgroup_size(64)
fn neg_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = -input[index];
}

@compute @workgroup_size(64)
fn sqrt_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = sqrt(input[index]);
}

@compute @workgroup_size(64)
fn abs_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = abs(input[index]);
}

@compute @workgroup_size(64)
fn exp_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = exp(input[index]);
}

@compute @workgroup_size(64)
fn sin_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = sin(input[index]);
}

@compute @workgroup_size(64)
fn cos_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = cos(input[index]);
}

@compute @workgroup_size(64)
fn tan_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = tan(input[index]);
}

@compute @workgroup_size(64)
fn recip_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = 1.0 / input[index];
}

@compute @workgroup_size(64)
fn floor_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = floor(input[index]);
}

@compute @workgroup_size(64)
fn ceil_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = ceil(input[index]);
}

@compute @workgroup_size(64)
fn round_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = round(input[index]);
}