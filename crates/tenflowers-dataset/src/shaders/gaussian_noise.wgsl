// GPU-accelerated Gaussian noise shader
// Adds Gaussian noise to image for data augmentation

struct Uniforms {
    width: u32,
    height: u32,
    channels: u32,
    seed: u32,
    mean: f32,
    std_dev: f32,
    padding1: f32,
    padding2: f32,
};

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn get_input_pixel(channel: u32, y: u32, x: u32) -> f32 {
    let idx = channel * uniforms.height * uniforms.width + y * uniforms.width + x;
    if (idx >= arrayLength(&input_data)) {
        return 0.0;
    }
    return input_data[idx];
}

fn set_output_pixel(channel: u32, y: u32, x: u32, value: f32) {
    let idx = channel * uniforms.height * uniforms.width + y * uniforms.width + x;
    if (idx < arrayLength(&output_data)) {
        output_data[idx] = value;
    }
}

// Simple pseudo-random number generator (LCG)
fn rand(seed: u32) -> u32 {
    let a = 1664525u;
    let c = 1013904223u;
    return a * seed + c;
}

fn rand_float(seed: u32) -> f32 {
    return f32(rand(seed)) / 4294967295.0;
}

// Box-Muller transform for Gaussian distribution
fn gaussian_noise(seed1: u32, seed2: u32, mean: f32, std_dev: f32) -> f32 {
    let u1 = rand_float(seed1);
    let u2 = rand_float(seed2);
    
    // Ensure u1 is not zero to avoid log(0)
    let u1_safe = max(u1, 1e-10);
    
    let z0 = sqrt(-2.0 * log(u1_safe)) * cos(2.0 * 3.14159265 * u2);
    return mean + std_dev * z0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let channel = global_id.z;
    
    if (x >= uniforms.width || y >= uniforms.height || channel >= uniforms.channels) {
        return;
    }
    
    // Get original pixel value
    let original_value = get_input_pixel(channel, y, x);
    
    // Generate unique seeds for this pixel
    let pixel_id = channel * uniforms.height * uniforms.width + y * uniforms.width + x;
    let seed1 = uniforms.seed + pixel_id;
    let seed2 = uniforms.seed + pixel_id + 1000000u;
    
    // Generate Gaussian noise
    let noise = gaussian_noise(seed1, seed2, uniforms.mean, uniforms.std_dev);
    
    // Add noise to original pixel and clamp to [0, 1]
    let noisy_value = clamp(original_value + noise, 0.0, 1.0);
    
    // Store result
    set_output_pixel(channel, y, x, noisy_value);
}