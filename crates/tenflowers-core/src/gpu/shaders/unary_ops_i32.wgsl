// Unary operation compute shaders for i32 type

@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(64)
fn neg_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = -input[index];
}

@compute @workgroup_size(64)
fn abs_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = abs(input[index]);
}

// For i32, we can't do log, sqrt, sin, cos, etc. as they're floating-point operations
// and would require casting to f32/f64 first