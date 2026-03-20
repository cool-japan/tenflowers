// GPU-accelerated image resize shader
// Performs bilinear interpolation for smooth resizing

struct Uniforms {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    channels: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn get_input_pixel(channel: u32, y: u32, x: u32) -> f32 {
    let idx = channel * uniforms.input_height * uniforms.input_width + y * uniforms.input_width + x;
    if (idx >= arrayLength(&input_data)) {
        return 0.0;
    }
    return input_data[idx];
}

fn set_output_pixel(channel: u32, y: u32, x: u32, value: f32) {
    let idx = channel * uniforms.output_height * uniforms.output_width + y * uniforms.output_width + x;
    if (idx < arrayLength(&output_data)) {
        output_data[idx] = value;
    }
}

fn bilinear_interpolate(channel: u32, x: f32, y: f32) -> f32 {
    let x0 = u32(floor(x));
    let x1 = min(x0 + 1u, uniforms.input_width - 1u);
    let y0 = u32(floor(y));
    let y1 = min(y0 + 1u, uniforms.input_height - 1u);
    
    let dx = x - f32(x0);
    let dy = y - f32(y0);
    
    let top_left = get_input_pixel(channel, y0, x0);
    let top_right = get_input_pixel(channel, y0, x1);
    let bottom_left = get_input_pixel(channel, y1, x0);
    let bottom_right = get_input_pixel(channel, y1, x1);
    
    let top = top_left * (1.0 - dx) + top_right * dx;
    let bottom = bottom_left * (1.0 - dx) + bottom_right * dx;
    
    return top * (1.0 - dy) + bottom * dy;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_x = global_id.x;
    let output_y = global_id.y;
    let channel = global_id.z;
    
    if (output_x >= uniforms.output_width || output_y >= uniforms.output_height || channel >= uniforms.channels) {
        return;
    }
    
    // Calculate source coordinates
    let scale_x = f32(uniforms.input_width) / f32(uniforms.output_width);
    let scale_y = f32(uniforms.input_height) / f32(uniforms.output_height);
    
    let src_x = (f32(output_x) + 0.5) * scale_x - 0.5;
    let src_y = (f32(output_y) + 0.5) * scale_y - 0.5;
    
    // Clamp to input bounds
    let clamped_x = max(0.0, min(src_x, f32(uniforms.input_width - 1u)));
    let clamped_y = max(0.0, min(src_y, f32(uniforms.input_height - 1u)));
    
    // Perform bilinear interpolation
    let interpolated_value = bilinear_interpolate(channel, clamped_x, clamped_y);
    
    // Store result
    set_output_pixel(channel, output_y, output_x, interpolated_value);
}