// GPU-accelerated Gaussian blur shader
// Applies Gaussian blur filter to image for smoothing effect

struct Uniforms {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma: f32,
    padding1: f32,
    padding2: f32,
    padding3: f32,
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

fn gaussian_weight(x: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    return exp(-(x * x) / (2.0 * sigma_sq)) / (sqrt(2.0 * 3.14159265) * sigma);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let channel = global_id.z;
    
    if (x >= uniforms.width || y >= uniforms.height || channel >= uniforms.channels) {
        return;
    }
    
    let kernel_radius = uniforms.kernel_size / 2u;
    var sum = 0.0;
    var weight_sum = 0.0;
    
    // Apply Gaussian kernel
    for (var ky = 0u; ky < uniforms.kernel_size; ky = ky + 1u) {
        for (var kx = 0u; kx < uniforms.kernel_size; kx = kx + 1u) {
            let offset_x = i32(kx) - i32(kernel_radius);
            let offset_y = i32(ky) - i32(kernel_radius);
            
            let sample_x = i32(x) + offset_x;
            let sample_y = i32(y) + offset_y;
            
            // Clamp to image boundaries
            let clamped_x = max(0, min(sample_x, i32(uniforms.width) - 1));
            let clamped_y = max(0, min(sample_y, i32(uniforms.height) - 1));
            
            // Calculate Gaussian weight
            let distance = sqrt(f32(offset_x * offset_x + offset_y * offset_y));
            let weight = gaussian_weight(distance, uniforms.sigma);
            
            // Sample pixel and accumulate
            let pixel_value = get_input_pixel(channel, u32(clamped_y), u32(clamped_x));
            sum = sum + pixel_value * weight;
            weight_sum = weight_sum + weight;
        }
    }
    
    // Normalize and store result
    let result = sum / weight_sum;
    set_output_pixel(channel, y, x, result);
}