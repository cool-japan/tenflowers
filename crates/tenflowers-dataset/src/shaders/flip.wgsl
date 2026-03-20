// GPU-accelerated horizontal flip shader
// Flips image horizontally by reversing pixel order along width dimension

struct Uniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let channel = global_id.z;
    
    if (x >= uniforms.width || y >= uniforms.height || channel >= uniforms.channels) {
        return;
    }
    
    // Calculate flipped x coordinate
    let flipped_x = uniforms.width - 1u - x;
    
    // Get pixel from original position and store at flipped position
    let pixel_value = get_input_pixel(channel, y, flipped_x);
    set_output_pixel(channel, y, x, pixel_value);
}