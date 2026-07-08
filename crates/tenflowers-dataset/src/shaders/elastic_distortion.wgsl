// GPU-accelerated elastic distortion shader
// Applies pre-computed displacement fields (dx, dy) per pixel.
// The displacement fields are passed as part of the input buffer:
//   input layout: [image_data (C*H*W), dx_field (H*W), dy_field (H*W)]
// Output layout: [image_data (C*H*W)]
//
// Each output pixel (x,y) samples from source at (x + dx[y,x], y + dy[y,x]).

struct ElasticUniforms {
    width:    u32,
    height:   u32,
    channels: u32,
    padding:  u32,
    // alpha: intensity multiplier (already baked into the displacement field by CPU)
    alpha:    f32,
    pad0:     f32,
    pad1:     f32,
    pad2:     f32,
}

@group(0) @binding(0) var<storage, read>       input_data  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data : array<f32>;
@group(0) @binding(2) var<uniform>             uniforms    : ElasticUniforms;

fn get_image_pixel(channel: u32, iy: u32, ix: u32) -> f32 {
    let idx = channel * uniforms.height * uniforms.width + iy * uniforms.width + ix;
    if (idx >= arrayLength(&input_data)) {
        return 0.0;
    }
    return input_data[idx];
}

fn get_displacement_x(iy: u32, ix: u32) -> f32 {
    // dx field starts after image data: offset = channels * H * W
    let offset = uniforms.channels * uniforms.height * uniforms.width;
    let idx = offset + iy * uniforms.width + ix;
    if (idx >= arrayLength(&input_data)) {
        return 0.0;
    }
    return input_data[idx] * uniforms.alpha;
}

fn get_displacement_y(iy: u32, ix: u32) -> f32 {
    // dy field starts after dx field: offset = (channels + 1) * H * W
    let offset = (uniforms.channels + 1u) * uniforms.height * uniforms.width;
    let idx = offset + iy * uniforms.width + ix;
    if (idx >= arrayLength(&input_data)) {
        return 0.0;
    }
    return input_data[idx] * uniforms.alpha;
}

fn bilinear_sample(sx: f32, sy: f32, channel: u32) -> f32 {
    let w = uniforms.width;
    let h = uniforms.height;

    if (sx < 0.0 || sx >= f32(w) || sy < 0.0 || sy >= f32(h)) {
        return 0.0;
    }

    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);

    let fx = sx - f32(x0);
    let fy = sy - f32(y0);

    let v00 = get_image_pixel(channel, y0, x0);
    let v01 = get_image_pixel(channel, y0, x1);
    let v10 = get_image_pixel(channel, y1, x0);
    let v11 = get_image_pixel(channel, y1, x1);

    let row0 = mix(v00, v01, fx);
    let row1 = mix(v10, v11, fx);
    return mix(row0, row1, fy);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if (x >= uniforms.width || y >= uniforms.height || c >= uniforms.channels) {
        return;
    }

    let dx = get_displacement_x(y, x);
    let dy = get_displacement_y(y, x);

    let src_x = f32(x) + dx;
    let src_y = f32(y) + dy;

    let out_idx = c * uniforms.width * uniforms.height + y * uniforms.width + x;
    output_data[out_idx] = bilinear_sample(src_x, src_y, c);
}
