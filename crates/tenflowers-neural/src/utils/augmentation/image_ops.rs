//! Image augmentation operation implementations.
//!
//! Functions in this module apply spatial and colour transforms to tensors
//! of shape `[C, H, W]` or `[N, C, H, W]`.

use scirs2_core::num_traits::{Float, FromPrimitive, Zero};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::RngExt;
use tenflowers_core::{Result, Tensor};

use super::{box_muller, f64_to_t, fisher_yates_shuffle, sample_beta};

// ---------------------------------------------------------------------------
// Flips
// ---------------------------------------------------------------------------

/// Flip image left-right along the width axis.
pub(super) fn apply_horizontal_flip<T>(
    input: &Tensor<T>,
    dims: &[usize],
    _height: usize,
    width: usize,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let mut out = data.to_vec();
    let ndim = dims.len();

    if ndim == 3 {
        let (c, h, w) = (dims[0], dims[1], dims[2]);
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src_idx = ci * h * w + hi * w + wi;
                    let dst_idx = ci * h * w + hi * w + (w - 1 - wi);
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    } else {
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let chw = c * h * w;
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let base = ni * chw + ci * h * w + hi * w;
                        out[base + (w - 1 - wi)] = data[base + wi];
                    }
                }
            }
        }
    }
    let _ = width;
    Tensor::from_vec(out, dims)
}

/// Flip image top-bottom along the height axis.
pub(super) fn apply_vertical_flip<T>(
    input: &Tensor<T>,
    dims: &[usize],
    _height: usize,
    _width: usize,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let mut out = data.to_vec();
    let ndim = dims.len();

    if ndim == 3 {
        let (c, h, w) = (dims[0], dims[1], dims[2]);
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src_idx = ci * h * w + hi * w + wi;
                    let dst_idx = ci * h * w + (h - 1 - hi) * w + wi;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    } else {
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let chw = c * h * w;
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let src = ni * chw + ci * h * w + hi * w + wi;
                        let dst = ni * chw + ci * h * w + (h - 1 - hi) * w + wi;
                        out[dst] = data[src];
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, dims)
}

// ---------------------------------------------------------------------------
// Spatial transforms (rotation, scale, translation, shear, crop)
// ---------------------------------------------------------------------------

/// Rotate the spatial plane by `angle` degrees using bilinear interpolation.
pub(super) fn apply_rotation<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    angle_deg: f64,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let rad = angle_deg * std::f64::consts::PI / 180.0;
    let cos_a = rad.cos();
    let sin_a = rad.sin();
    let cy = (height as f64 - 1.0) / 2.0;
    let cx = (width as f64 - 1.0) / 2.0;

    apply_spatial_transform(input, dims, height, width, |y, x| {
        let yf = y as f64 - cy;
        let xf = x as f64 - cx;
        let src_y = cos_a * yf + sin_a * xf + cy;
        let src_x = -sin_a * yf + cos_a * xf + cx;
        (src_y, src_x)
    })
}

/// Scale the image by `scale` factor around the centre.
pub(super) fn apply_scaling<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    scale: f64,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let cy = (height as f64 - 1.0) / 2.0;
    let cx = (width as f64 - 1.0) / 2.0;
    let inv = if scale.abs() < 1e-12 {
        1.0
    } else {
        1.0 / scale
    };

    apply_spatial_transform(input, dims, height, width, |y, x| {
        let src_y = (y as f64 - cy) * inv + cy;
        let src_x = (x as f64 - cx) * inv + cx;
        (src_y, src_x)
    })
}

/// Translate the image by (dx, dy) pixels.
pub(super) fn apply_translation<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    dx: f64,
    dy: f64,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    apply_spatial_transform(input, dims, height, width, |y, x| {
        (y as f64 - dy, x as f64 - dx)
    })
}

/// Shear the image.
pub(super) fn apply_shear<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    shear_x: f64,
    shear_y: f64,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let cy = (height as f64 - 1.0) / 2.0;
    let cx = (width as f64 - 1.0) / 2.0;

    apply_spatial_transform(input, dims, height, width, |y, x| {
        let yf = y as f64 - cy;
        let xf = x as f64 - cx;
        let src_y = yf - shear_y * xf + cy;
        let src_x = xf - shear_x * yf + cx;
        (src_y, src_x)
    })
}

/// Random crop to `crop_frac` of original size, then resize back via bilinear interpolation.
pub(super) fn apply_random_crop_and_resize<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    crop_frac: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let crop_h = ((height as f64) * crop_frac).max(1.0) as usize;
    let crop_w = ((width as f64) * crop_frac).max(1.0) as usize;

    let max_y = height.saturating_sub(crop_h);
    let max_x = width.saturating_sub(crop_w);

    let off_y = if max_y > 0 {
        rng.random_range(0..=max_y)
    } else {
        0
    };
    let off_x = if max_x > 0 {
        rng.random_range(0..=max_x)
    } else {
        0
    };

    apply_spatial_transform(input, dims, height, width, |y, x| {
        let src_y = off_y as f64 + (y as f64 / height.max(1) as f64) * crop_h as f64;
        let src_x = off_x as f64 + (x as f64 / width.max(1) as f64) * crop_w as f64;
        (
            src_y.min((height - 1) as f64),
            src_x.min((width - 1) as f64),
        )
    })
}

// ---------------------------------------------------------------------------
// Colour / intensity transforms
// ---------------------------------------------------------------------------

/// Add a scalar brightness offset to every element.
pub(super) fn apply_brightness<T>(input: &Tensor<T>, delta: f64) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let delta_t = f64_to_t::<T>(delta)?;
    let out: Vec<T> = data.iter().map(|&v| v + delta_t).collect();
    Tensor::from_vec(out, input.shape().dims())
}

/// Adjust contrast by scaling towards the per-channel mean.
pub(super) fn apply_contrast<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    alpha: f64,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let mut out = data.to_vec();
    let ndim = dims.len();
    let hw = height * width;

    let adjust_channel = |out: &mut [T], data: &[T], offset: usize| -> Result<()> {
        let mut sum = 0.0_f64;
        for i in 0..hw {
            sum += data[offset + i].to_f64().unwrap_or(0.0);
        }
        let mean = sum / hw as f64;
        for i in 0..hw {
            let v = data[offset + i].to_f64().unwrap_or(0.0);
            let adjusted = mean + alpha * (v - mean);
            out[offset + i] = f64_to_t(adjusted)?;
        }
        Ok(())
    };

    if ndim == 3 {
        let c = dims[0];
        for ci in 0..c {
            adjust_channel(&mut out, data, ci * hw)?;
        }
    } else {
        let (n, c) = (dims[0], dims[1]);
        let chw = c * hw;
        for ni in 0..n {
            for ci in 0..c {
                adjust_channel(&mut out, data, ni * chw + ci * hw)?;
            }
        }
    }

    Tensor::from_vec(out, dims)
}

/// Adjust saturation by blending towards the luminance channel.
///
/// Works on tensors with >= 3 channels. Channels 0,1,2 are treated as RGB.
pub(super) fn apply_saturation<T>(
    input: &Tensor<T>,
    dims: &[usize],
    alpha: f64,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let ndim = dims.len();
    if ndim < 3 {
        return Ok(input.clone());
    }

    let data = input.data();
    let mut out = data.to_vec();

    let (n, c, h, w) = if ndim == 3 {
        (1, dims[0], dims[1], dims[2])
    } else {
        (dims[0], dims[1], dims[2], dims[3])
    };

    if c < 3 {
        return Ok(input.clone());
    }

    let hw = h * w;
    let chw = c * hw;

    for ni in 0..n {
        let base = if ndim == 3 { 0 } else { ni * chw };
        for i in 0..hw {
            let r = data[base + i].to_f64().unwrap_or(0.0);
            let g = data[base + hw + i].to_f64().unwrap_or(0.0);
            let b = data[base + 2 * hw + i].to_f64().unwrap_or(0.0);
            let lum = 0.2989 * r + 0.5870 * g + 0.1140 * b;
            out[base + i] = T::from_f64(lum + alpha * (r - lum)).unwrap_or_else(T::zero);
            out[base + hw + i] = T::from_f64(lum + alpha * (g - lum)).unwrap_or_else(T::zero);
            out[base + 2 * hw + i] = T::from_f64(lum + alpha * (b - lum)).unwrap_or_else(T::zero);
        }
    }

    Tensor::from_vec(out, dims)
}

/// Simple hue-shift by rotating in the RGB colour-opponent plane.
///
/// The shift is applied by rotating the (R-lum, G-lum, B-lum) vector.
pub(super) fn apply_hue_shift<T>(input: &Tensor<T>, delta: f64) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let dims = input.shape().dims().to_vec();
    let ndim = dims.len();
    if ndim < 3 {
        return Ok(input.clone());
    }

    let data = input.data();
    let mut out = data.to_vec();

    let (n, c, h, w) = if ndim == 3 {
        (1, dims[0], dims[1], dims[2])
    } else {
        (dims[0], dims[1], dims[2], dims[3])
    };

    if c < 3 {
        return Ok(input.clone());
    }

    let hw = h * w;
    let chw = c * hw;
    let cos_d = (delta * 2.0 * std::f64::consts::PI).cos();
    let sin_d = (delta * 2.0 * std::f64::consts::PI).sin();

    // Rotation matrix in the colour-opponent plane (simplified Rodrigues around the
    // grey axis (1,1,1)/sqrt(3)).
    let k = 1.0 / 3.0;
    let m00 = k + (1.0 - k) * cos_d;
    let m01 = k * (1.0 - cos_d) - (1.0 / 3.0_f64.sqrt()) * sin_d;
    let m02 = k * (1.0 - cos_d) + (1.0 / 3.0_f64.sqrt()) * sin_d;
    let m10 = k * (1.0 - cos_d) + (1.0 / 3.0_f64.sqrt()) * sin_d;
    let m11 = k + (1.0 - k) * cos_d;
    let m12 = k * (1.0 - cos_d) - (1.0 / 3.0_f64.sqrt()) * sin_d;
    let m20 = k * (1.0 - cos_d) - (1.0 / 3.0_f64.sqrt()) * sin_d;
    let m21 = k * (1.0 - cos_d) + (1.0 / 3.0_f64.sqrt()) * sin_d;
    let m22 = k + (1.0 - k) * cos_d;

    for ni in 0..n {
        let base = if ndim == 3 { 0 } else { ni * chw };
        for i in 0..hw {
            let r = data[base + i].to_f64().unwrap_or(0.0);
            let g = data[base + hw + i].to_f64().unwrap_or(0.0);
            let b = data[base + 2 * hw + i].to_f64().unwrap_or(0.0);
            let nr = m00 * r + m01 * g + m02 * b;
            let ng = m10 * r + m11 * g + m12 * b;
            let nb = m20 * r + m21 * g + m22 * b;
            out[base + i] = T::from_f64(nr).unwrap_or_else(T::zero);
            out[base + hw + i] = T::from_f64(ng).unwrap_or_else(T::zero);
            out[base + 2 * hw + i] = T::from_f64(nb).unwrap_or_else(T::zero);
        }
    }

    Tensor::from_vec(out, &dims)
}

// ---------------------------------------------------------------------------
// Noise transforms
// ---------------------------------------------------------------------------

/// Add Gaussian noise using Box-Muller transform.
pub(super) fn apply_gaussian_noise<T>(
    input: &Tensor<T>,
    std_dev: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let (z0, z1) = box_muller(rng);
        let v0 = data[i].to_f64().unwrap_or(0.0) + z0 * std_dev;
        out.push(f64_to_t(v0)?);
        i += 1;
        if i < data.len() {
            let v1 = data[i].to_f64().unwrap_or(0.0) + z1 * std_dev;
            out.push(f64_to_t(v1)?);
            i += 1;
        }
    }
    Tensor::from_vec(out, input.shape().dims())
}

/// Salt-and-pepper noise: randomly set pixels to 0 or 1.
pub(super) fn apply_salt_pepper_noise<T>(
    input: &Tensor<T>,
    noise_prob: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let one_t = f64_to_t::<T>(1.0)?;
    let out: Vec<T> = data
        .iter()
        .map(|&v| {
            let coin: f64 = rng.random_range(0.0..1.0);
            if coin < noise_prob / 2.0 {
                T::zero()
            } else if coin < noise_prob {
                one_t
            } else {
                v
            }
        })
        .collect();
    Tensor::from_vec(out, input.shape().dims())
}

// ---------------------------------------------------------------------------
// Erasing / cutout
// ---------------------------------------------------------------------------

/// Random erasing: zero-out a rectangular patch whose area is roughly `area_ratio * H * W`.
pub(super) fn apply_random_erasing<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    area_ratio: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let target_area = (height as f64 * width as f64 * area_ratio).max(1.0);
    let aspect: f64 = rng.random_range(0.3..3.3);
    let eh = (target_area * aspect).sqrt().min(height as f64) as usize;
    let ew = (target_area / aspect.max(1e-6)).sqrt().min(width as f64) as usize;
    let eh = eh.max(1).min(height);
    let ew = ew.max(1).min(width);
    let y0 = if height > eh {
        rng.random_range(0..height - eh)
    } else {
        0
    };
    let x0 = if width > ew {
        rng.random_range(0..width - ew)
    } else {
        0
    };

    erase_rect(input, dims, height, width, y0, x0, eh, ew)
}

/// Cutout: zero-out a square patch of given `size`.
pub(super) fn apply_cutout<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    size: usize,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let half = size / 2;
    let cy: usize = rng.random_range(0..height);
    let cx: usize = rng.random_range(0..width);
    let y0 = cy.saturating_sub(half);
    let x0 = cx.saturating_sub(half);
    let y1 = (cy + half).min(height);
    let x1 = (cx + half).min(width);
    let eh = y1 - y0;
    let ew = x1 - x0;

    erase_rect(input, dims, height, width, y0, x0, eh, ew)
}

/// Zero-out a rectangular region across all channels.
fn erase_rect<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    y0: usize,
    x0: usize,
    eh: usize,
    ew: usize,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let mut out = data.to_vec();
    let ndim = dims.len();
    let hw = height * width;

    let erase_channel = |out: &mut [T], offset: usize| {
        for dy in 0..eh {
            for dx in 0..ew {
                let y = y0 + dy;
                let x = x0 + dx;
                if y < height && x < width {
                    out[offset + y * width + x] = T::zero();
                }
            }
        }
    };

    if ndim == 3 {
        let c = dims[0];
        for ci in 0..c {
            erase_channel(&mut out, ci * hw);
        }
    } else {
        let (n, c) = (dims[0], dims[1]);
        let chw = c * hw;
        for ni in 0..n {
            for ci in 0..c {
                erase_channel(&mut out, ni * chw + ci * hw);
            }
        }
    }

    Tensor::from_vec(out, dims)
}

// ---------------------------------------------------------------------------
// Mixup / CutMix
// ---------------------------------------------------------------------------

/// Self-mixup: blend each element with a shuffled copy of itself.
///
/// For a batch `[N, C, H, W]`, samples are shuffled along the batch dimension.
/// For a single image `[C, H, W]`, a random permutation of pixels is used.
pub(super) fn apply_self_mixup<T>(
    input: &Tensor<T>,
    alpha: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let dims = input.shape().dims();
    let ndim = dims.len();

    let lam = sample_beta(alpha, alpha, rng);
    let lam_t = f64_to_t::<T>(lam)?;
    let one_minus = f64_to_t::<T>(1.0 - lam)?;

    if ndim == 4 {
        let n = dims[0];
        let sample_size = data.len() / n;
        let mut perm: Vec<usize> = (0..n).collect();
        fisher_yates_shuffle(&mut perm, rng);

        let out: Vec<T> = (0..data.len())
            .map(|i| {
                let batch_idx = i / sample_size;
                let inner_idx = i % sample_size;
                let shuffled_val = data[perm[batch_idx] * sample_size + inner_idx];
                data[i] * lam_t + shuffled_val * one_minus
            })
            .collect();
        Tensor::from_vec(out, dims)
    } else {
        let mut perm: Vec<usize> = (0..data.len()).collect();
        fisher_yates_shuffle(&mut perm, rng);

        let out: Vec<T> = (0..data.len())
            .map(|i| data[i] * lam_t + data[perm[i]] * one_minus)
            .collect();
        Tensor::from_vec(out, dims)
    }
}

/// Self-CutMix: paste a rectangular region from a shuffled copy.
pub(super) fn apply_self_cutmix<T>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    alpha: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let ndim = dims.len();

    let lam = sample_beta(alpha, alpha, rng);
    let cut_ratio = (1.0 - lam).sqrt();
    let cut_h = ((height as f64) * cut_ratio).max(1.0) as usize;
    let cut_w = ((width as f64) * cut_ratio).max(1.0) as usize;
    let cut_h = cut_h.min(height);
    let cut_w = cut_w.min(width);

    let cy: usize = rng.random_range(0..height);
    let cx: usize = rng.random_range(0..width);
    let y0 = cy.saturating_sub(cut_h / 2);
    let x0 = cx.saturating_sub(cut_w / 2);
    let y1 = (y0 + cut_h).min(height);
    let x1 = (x0 + cut_w).min(width);

    let mut out = data.to_vec();
    let hw = height * width;

    if ndim == 4 {
        let n = dims[0];
        let c = dims[1];
        let chw = c * hw;
        let sample_size = chw;
        let mut perm: Vec<usize> = (0..n).collect();
        fisher_yates_shuffle(&mut perm, rng);

        for ni in 0..n {
            let si = perm[ni];
            for ci in 0..c {
                let dst_base = ni * sample_size + ci * hw;
                let src_base = si * sample_size + ci * hw;
                for y in y0..y1 {
                    for x in x0..x1 {
                        out[dst_base + y * width + x] = data[src_base + y * width + x];
                    }
                }
            }
        }
    } else {
        let c = dims[0];
        for ci in 0..c {
            let src_ci = c - 1 - ci;
            let dst_base = ci * hw;
            let src_base = src_ci * hw;
            for y in y0..y1 {
                for x in x0..x1 {
                    out[dst_base + y * width + x] = data[src_base + y * width + x];
                }
            }
        }
    }

    Tensor::from_vec(out, dims)
}

// ---------------------------------------------------------------------------
// Internal: spatial transform + bilinear interpolation
// ---------------------------------------------------------------------------

/// Generic spatial transform using bilinear interpolation.
fn apply_spatial_transform<T, F>(
    input: &Tensor<T>,
    dims: &[usize],
    height: usize,
    width: usize,
    map_fn: F,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
    F: Fn(usize, usize) -> (f64, f64),
{
    let data = input.data();
    let total = data.len();
    let mut out = vec![T::zero(); total];
    let ndim = dims.len();

    let process_channel = |out: &mut [T], data: &[T], offset: usize| {
        for y in 0..height {
            for x in 0..width {
                let (src_y, src_x) = map_fn(y, x);
                let val = bilinear_sample(data, offset, height, width, src_y, src_x);
                out[offset + y * width + x] = val;
            }
        }
    };

    if ndim == 3 {
        let c = dims[0];
        for ci in 0..c {
            let offset = ci * height * width;
            process_channel(&mut out, data, offset);
        }
    } else {
        let (n, c) = (dims[0], dims[1]);
        let chw = c * height * width;
        for ni in 0..n {
            for ci in 0..c {
                let offset = ni * chw + ci * height * width;
                process_channel(&mut out, data, offset);
            }
        }
    }

    Tensor::from_vec(out, dims)
}

/// Bilinear interpolation sampling from a single channel plane.
fn bilinear_sample<T: Float + FromPrimitive + Zero>(
    data: &[T],
    offset: usize,
    height: usize,
    width: usize,
    src_y: f64,
    src_x: f64,
) -> T {
    if src_y < 0.0 || src_x < 0.0 || src_y > (height as f64 - 1.0) || src_x > (width as f64 - 1.0) {
        return T::zero();
    }

    let y0 = src_y.floor() as usize;
    let x0 = src_x.floor() as usize;
    let y1 = (y0 + 1).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);

    let fy = src_y - src_y.floor();
    let fx = src_x - src_x.floor();

    let idx = |y: usize, x: usize| offset + y * width + x;

    let v00 = data[idx(y0, x0)].to_f64().unwrap_or(0.0);
    let v01 = data[idx(y0, x1)].to_f64().unwrap_or(0.0);
    let v10 = data[idx(y1, x0)].to_f64().unwrap_or(0.0);
    let v11 = data[idx(y1, x1)].to_f64().unwrap_or(0.0);

    let val = v00 * (1.0 - fy) * (1.0 - fx)
        + v01 * (1.0 - fy) * fx
        + v10 * fy * (1.0 - fx)
        + v11 * fy * fx;

    T::from_f64(val).unwrap_or_else(T::zero)
}
