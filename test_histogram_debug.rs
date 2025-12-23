fn compute_scalar_debug(data: &[f32], histogram: &mut [u32], bin_width: f32, min_val: f32, max_val: f32, bins: usize) {
    println!("Debug scalar histogram computation:");
    println!("min_val: {}, max_val: {}, bin_width: {}, bins: {}", min_val, max_val, bin_width, bins);
    for (i, &value) in data.iter().enumerate() {
        let clamped = value.clamp(min_val, max_val);
        let bin_idx = ((clamped - min_val) / bin_width) as usize;
        let bin_idx = bin_idx.min(bins - 1);
        println!("data[{}] = {}, clamped = {}, bin_idx = {}", i, value, clamped, bin_idx);
        histogram[bin_idx] += 1;
    }
}

fn main() {
    // Test data from the failing test
    let data = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
    let bins = 10;
    let min_val = 0.0;
    let max_val = 10.0;
    let bin_width = (max_val - min_val) / bins as f32;

    let mut histogram = vec![0u32; bins];
    compute_scalar_debug(&data, &mut histogram, bin_width, min_val, max_val, bins);

    println!("Final histogram: {:?}", histogram);
}