// GPU-accelerated color jitter shader
// Adjusts brightness, contrast, saturation, and hue of RGB images

struct Uniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
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

fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    let delta = max_val - min_val;
    
    var h: f32;
    var s: f32;
    let v = max_val;
    
    if (delta == 0.0) {
        h = 0.0;
        s = 0.0;
    } else {
        s = delta / max_val;
        
        if (rgb.r == max_val) {
            h = (rgb.g - rgb.b) / delta;
        } else if (rgb.g == max_val) {
            h = 2.0 + (rgb.b - rgb.r) / delta;
        } else {
            h = 4.0 + (rgb.r - rgb.g) / delta;
        }
        
        h = h * 60.0;
        if (h < 0.0) {
            h = h + 360.0;
        }
    }
    
    return vec3<f32>(h, s, v);
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;
    
    if (s == 0.0) {
        return vec3<f32>(v, v, v);
    }
    
    let h_sector = h / 60.0;
    let i = u32(floor(h_sector));
    let f = h_sector - f32(i);
    
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    
    switch (i % 6u) {
        case 0u: { return vec3<f32>(v, t, p); }
        case 1u: { return vec3<f32>(q, v, p); }
        case 2u: { return vec3<f32>(p, v, t); }
        case 3u: { return vec3<f32>(p, q, v); }
        case 4u: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }
    
    // Get RGB values
    let r = get_input_pixel(0u, y, x);
    let g = get_input_pixel(1u, y, x);
    let b = get_input_pixel(2u, y, x);
    
    var rgb = vec3<f32>(r, g, b);
    
    // Apply brightness adjustment
    rgb = rgb + uniforms.brightness;
    
    // Apply contrast adjustment
    rgb = (rgb - 0.5) * uniforms.contrast + 0.5;
    
    // Convert to HSV for saturation and hue adjustments
    var hsv = rgb_to_hsv(rgb);
    
    // Apply saturation adjustment
    hsv.y = hsv.y * uniforms.saturation;
    
    // Apply hue adjustment
    hsv.x = hsv.x + uniforms.hue;
    if (hsv.x < 0.0) {
        hsv.x = hsv.x + 360.0;
    } else if (hsv.x >= 360.0) {
        hsv.x = hsv.x - 360.0;
    }
    
    // Convert back to RGB
    rgb = hsv_to_rgb(hsv);
    
    // Clamp values to [0, 1]
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Store result
    set_output_pixel(0u, y, x, rgb.r);
    set_output_pixel(1u, y, x, rgb.g);
    set_output_pixel(2u, y, x, rgb.b);
}