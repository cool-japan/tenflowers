// Rotation compute shader for GPU-accelerated image rotation

struct Uniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
    cos_angle: f32,
    sin_angle: f32,
    center_x: f32,
    center_y: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn bilinear_interpolate(x: f32, y: f32, channel: u32) -> f32 {
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, uniforms.width - 1u);
    let y1 = min(y0 + 1u, uniforms.height - 1u);
    
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    
    if (x0 >= uniforms.width || y0 >= uniforms.height) {
        return 0.0;
    }
    
    let idx00 = channel * uniforms.width * uniforms.height + y0 * uniforms.width + x0;
    let idx01 = channel * uniforms.width * uniforms.height + y0 * uniforms.width + x1;
    let idx10 = channel * uniforms.width * uniforms.height + y1 * uniforms.width + x0;
    let idx11 = channel * uniforms.width * uniforms.height + y1 * uniforms.width + x1;
    
    let v00 = input_data[idx00];
    let v01 = input_data[idx01];
    let v10 = input_data[idx10];
    let v11 = input_data[idx11];
    
    let v0 = v00 * (1.0 - fx) + v01 * fx;
    let v1 = v10 * (1.0 - fx) + v11 * fx;
    
    return v0 * (1.0 - fy) + v1 * fy;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let channel = global_id.z;

    // Check bounds
    if (x >= uniforms.width || y >= uniforms.height || channel >= uniforms.channels) {
        return;
    }

    // Translate to center
    let centered_x = f32(x) - uniforms.center_x;
    let centered_y = f32(y) - uniforms.center_y;

    // Apply rotation
    let rotated_x = centered_x * uniforms.cos_angle - centered_y * uniforms.sin_angle;
    let rotated_y = centered_x * uniforms.sin_angle + centered_y * uniforms.cos_angle;

    // Translate back
    let source_x = rotated_x + uniforms.center_x;
    let source_y = rotated_y + uniforms.center_y;

    // Calculate output index
    let output_index = channel * uniforms.width * uniforms.height + y * uniforms.width + x;

    // Sample with bilinear interpolation
    if (source_x >= 0.0 && source_x < f32(uniforms.width) && 
        source_y >= 0.0 && source_y < f32(uniforms.height)) {
        output_data[output_index] = bilinear_interpolate(source_x, source_y, channel);
    } else {
        // Fill with black for out-of-bounds pixels
        output_data[output_index] = 0.0;
    }
}