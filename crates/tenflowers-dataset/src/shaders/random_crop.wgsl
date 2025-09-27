// Random crop compute shader for GPU-accelerated image cropping

struct Uniforms {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    channels: u32,
    crop_x: u32,
    crop_y: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_x = global_id.x;
    let output_y = global_id.y;
    let channel = global_id.z;

    // Check bounds
    if (output_x >= uniforms.output_width || output_y >= uniforms.output_height || channel >= uniforms.channels) {
        return;
    }

    // Calculate input coordinates
    let input_x = output_x + uniforms.crop_x;
    let input_y = output_y + uniforms.crop_y;

    // Calculate indices
    let input_index = channel * uniforms.input_width * uniforms.input_height + 
                     input_y * uniforms.input_width + input_x;
    let output_index = channel * uniforms.output_width * uniforms.output_height + 
                      output_y * uniforms.output_width + output_x;

    // Copy the pixel value
    output_data[output_index] = input_data[input_index];
}