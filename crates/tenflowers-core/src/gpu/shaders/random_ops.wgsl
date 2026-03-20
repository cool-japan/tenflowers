// Random number generation compute shaders
// These kernels implement various random number generation algorithms

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>; // [mean, std, seed_low, seed_high]

// Simple Linear Congruential Generator state
var<private> rng_state: u32;

// Initialize RNG state with seed and index
fn init_rng(seed: u32, index: u32) {
    rng_state = seed ^ (index * 0x9e3779b9u);
}

// Generate next random u32
fn next_u32() -> u32 {
    rng_state = rng_state * 0x19660du + 0x3c6ef35fu;
    return rng_state;
}

// Generate random f32 in [0, 1)
fn next_f32() -> f32 {
    return f32(next_u32()) / 4294967296.0;
}

// Box-Muller transform for normal distribution
// Returns two independent normal samples
fn box_muller(mean: f32, std: f32) -> vec2<f32> {
    let u1 = next_f32();
    let u2 = next_f32();
    
    // Ensure u1 is not zero to avoid log(0)
    let u1_safe = max(u1, 1e-8);
    
    let r = sqrt(-2.0 * log(u1_safe));
    let theta = 2.0 * 3.14159265359 * u2;
    
    let z0 = r * cos(theta);
    let z1 = r * sin(theta);
    
    return vec2<f32>(mean + std * z0, mean + std * z1);
}

// Normal distribution random number generation
@compute @workgroup_size(64)
fn random_normal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = arrayLength(&output);
    
    if (index >= total_elements) {
        return;
    }
    
    let mean = params[0];
    let std = params[1];
    let seed_low = bitcast<u32>(params[2]);
    let seed_high = bitcast<u32>(params[3]);
    let seed = seed_low ^ (seed_high << 16u);
    
    init_rng(seed, index);
    
    // Generate pairs of normal samples
    let pair_index = index / 2u;
    let is_first = (index % 2u) == 0u;
    
    if (is_first && (index + 1u) < total_elements) {
        // Generate pair of normal samples
        let samples = box_muller(mean, std);
        output[index] = samples.x;
        output[index + 1u] = samples.y;
    } else if (!is_first) {
        // Second element of pair already generated
        return;
    } else {
        // Odd number of elements, generate single sample
        let samples = box_muller(mean, std);
        output[index] = samples.x;
    }
}

// Uniform distribution random number generation
@compute @workgroup_size(64)
fn random_uniform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let min_val = params[0];
    let max_val = params[1];
    let seed_low = bitcast<u32>(params[2]);
    let seed_high = bitcast<u32>(params[3]);
    let seed = seed_low ^ (seed_high << 16u);
    
    init_rng(seed, index);
    
    let uniform_sample = next_f32();
    output[index] = min_val + (max_val - min_val) * uniform_sample;
}

// Randn (standard normal) generation
@compute @workgroup_size(64)
fn randn(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = arrayLength(&output);
    
    if (index >= total_elements) {
        return;
    }
    
    let seed_low = bitcast<u32>(params[0]);
    let seed_high = bitcast<u32>(params[1]);
    let seed = seed_low ^ (seed_high << 16u);
    
    init_rng(seed, index);
    
    // Generate pairs of standard normal samples
    let pair_index = index / 2u;
    let is_first = (index % 2u) == 0u;
    
    if (is_first && (index + 1u) < total_elements) {
        // Generate pair of standard normal samples
        let samples = box_muller(0.0, 1.0);
        output[index] = samples.x;
        output[index + 1u] = samples.y;
    } else if (!is_first) {
        // Second element of pair already generated
        return;
    } else {
        // Odd number of elements, generate single sample
        let samples = box_muller(0.0, 1.0);
        output[index] = samples.x;
    }
}

// Rand (uniform [0, 1)) generation
@compute @workgroup_size(64)
fn rand(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let seed_low = bitcast<u32>(params[0]);
    let seed_high = bitcast<u32>(params[1]);
    let seed = seed_low ^ (seed_high << 16u);
    
    init_rng(seed, index);
    
    output[index] = next_f32();
}