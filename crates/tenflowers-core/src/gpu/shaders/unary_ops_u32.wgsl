// Unary operation compute shaders for u32 type

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn abs_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    // For unsigned integers, abs is identity operation
    output[index] = input[index];
}

// For u32, we can't do neg (would result in underflow)
// We can't do log, sqrt, sin, cos, etc. as they're floating-point operations
// and would require casting to f32/f64 first