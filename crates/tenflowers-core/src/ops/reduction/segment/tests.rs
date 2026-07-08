//! Tests for segment reduction operations

#[cfg(test)]
mod tests {
    use crate::ops::reduction::segment::{
        segment_all, segment_any, segment_max, segment_mean, segment_min, segment_prod,
        segment_sum, unsorted_segment_max,
    };
    use crate::Tensor;

    #[test]
    fn test_segment_sum_basic() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
            .expect("tensor creation should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("tensor creation should succeed");

        let result = segment_sum(&data, &segment_ids, 3).expect("segment_sum should succeed");
        let result_data = result.to_vec().expect("to_vec should succeed");

        assert_eq!(result_data.len(), 3);
        assert!((result_data[0] - 3.0).abs() < 1e-6);
        assert!((result_data[1] - 7.0).abs() < 1e-6);
        assert!((result_data[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_segment_sum_empty_segments() {
        let data =
            Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 2], &[2]).expect("test: from_vec should succeed");

        let result = segment_sum(&data, &segment_ids, 4).expect("test: segment_sum should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 4);
        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!((result_data[1] - 0.0).abs() < 1e-6);
        assert!((result_data[2] - 2.0).abs() < 1e-6);
        assert!((result_data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_segment_sum_single_segment() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 0, 0], &[4]).expect("test: from_vec should succeed");

        let result = segment_sum(&data, &segment_ids, 1).expect("test: segment_sum should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 1);
        assert!((result_data[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_segment_mean_basic() {
        let data = Tensor::from_vec(vec![2.0_f32, 4.0, 6.0, 8.0, 10.0, 12.0], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result =
            segment_mean(&data, &segment_ids, 3).expect("test: segment_mean should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 3);
        assert!((result_data[0] - 3.0).abs() < 1e-6);
        assert!((result_data[1] - 7.0).abs() < 1e-6);
        assert!((result_data[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_segment_mean_variable_length() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0], &[5])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 0, 1, 1], &[5]).expect("test: from_vec should succeed");

        let result =
            segment_mean(&data, &segment_ids, 2).expect("test: segment_mean should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 2);
        assert!((result_data[0] - 2.0).abs() < 1e-6);
        assert!((result_data[1] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_segment_max_basic() {
        let data = Tensor::from_vec(vec![1.0_f32, 5.0, 2.0, 8.0, 3.0, 6.0], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, 3).expect("test: segment_max should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 3);
        assert!((result_data[0] - 5.0).abs() < 1e-6);
        assert!((result_data[1] - 8.0).abs() < 1e-6);
        assert!((result_data[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_segment_max_negative_values() {
        let data = Tensor::from_vec(vec![-5.0_f32, -2.0, -8.0, -1.0], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, 2).expect("test: segment_max should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 2);
        assert!((result_data[0] - (-2.0)).abs() < 1e-6);
        assert!((result_data[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_segment_sum_large_input() {
        let size = 2000;
        let num_segments = 10;

        let data_vec: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let segment_ids_vec: Vec<i32> = (0..size).map(|i| (i % num_segments) as i32).collect();

        let data =
            Tensor::from_vec(data_vec.clone(), &[size]).expect("test: operation should succeed");
        let segment_ids = Tensor::from_vec(segment_ids_vec.clone(), &[size])
            .expect("test: operation should succeed");

        let result = segment_sum(&data, &segment_ids, num_segments)
            .expect("test: segment_sum should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), num_segments);

        let mut expected = vec![0.0_f32; num_segments];
        for (i, &val) in data_vec.iter().enumerate() {
            expected[segment_ids_vec[i] as usize] += val;
        }

        for i in 0..num_segments {
            assert!(
                (result_data[i] - expected[i]).abs() < 1e-3,
                "Segment {} mismatch: got {}, expected {}",
                i,
                result_data[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_segment_operations_i32() {
        let data = Tensor::from_vec(vec![1_i32, 2, 3, 4, 5, 6], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result = segment_sum(&data, &segment_ids, 3).expect("test: segment_sum should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 3);
        assert_eq!(result_data[0], 3);
        assert_eq!(result_data[1], 7);
        assert_eq!(result_data[2], 11);
    }

    #[test]
    fn test_segment_operations_f64() {
        let data = Tensor::from_vec(vec![1.5_f64, 2.5, 3.5, 4.5], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_sum(&data, &segment_ids, 2).expect("test: segment_sum should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 2);
        assert!((result_data[0] - 4.0).abs() < 1e-10);
        assert!((result_data[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_segment_sum_shape_mismatch() {
        let data =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]).expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 1], &[2]).expect("test: from_vec should succeed");

        let result = segment_sum(&data, &segment_ids, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_segment_operations_consistency() {
        let data = Tensor::from_vec(vec![2.0_f32, 4.0, 6.0, 8.0], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let sum_result =
            segment_sum(&data, &segment_ids, 2).expect("test: segment_sum should succeed");
        let mean_result =
            segment_mean(&data, &segment_ids, 2).expect("test: segment_mean should succeed");

        let sum_data = sum_result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");
        let mean_data = mean_result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert!((mean_data[0] - sum_data[0] / 2.0).abs() < 1e-6);
        assert!((mean_data[1] - sum_data[1] / 2.0).abs() < 1e-6);
    }

    /// `segment_sum` over a `[4, 2]` input must keep the feature dimension and
    /// return `[num_segments, 2]`. The OLD implementation flattened the result
    /// to 1-D `[num_segments]`, dropping the feature dimension entirely.
    #[test]
    fn test_segment_sum_2d_feature_dim() {
        // 4 rows, width 2.
        //   row 0: [1, 2]  -> seg 0
        //   row 1: [3, 4]  -> seg 0
        //   row 2: [5, 6]  -> seg 1
        //   row 3: [7, 8]  -> seg 1
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_sum(&data, &segment_ids, 2).expect("test: segment_sum should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0: [1+3, 2+4] = [4, 6]; seg 1: [5+7, 6+8] = [12, 14].
        assert_eq!(got, vec![4.0, 6.0, 12.0, 14.0]);
    }

    /// `segment_mean` over a `[4, 2]` input must also keep the feature
    /// dimension and return `[num_segments, 2]`.
    #[test]
    fn test_segment_mean_2d_feature_dim() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result =
            segment_mean(&data, &segment_ids, 2).expect("test: segment_mean should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0: mean([1,3], [2,4]) = [2, 3]; seg 1: mean([5,7],[6,8]) = [6, 7].
        assert_eq!(got, vec![2.0, 3.0, 6.0, 7.0]);
    }

    /// An empty segment in a 2-D reduction must yield a zero feature row.
    #[test]
    fn test_segment_sum_2d_empty_segment() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 2], &[2]).expect("test: from_vec should succeed");

        let result = segment_sum(&data, &segment_ids, 3).expect("test: segment_sum should succeed");

        assert_eq!(result.shape().dims(), &[3, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0: [1, 2]; seg 1: [0, 0] (empty); seg 2: [3, 4].
        assert_eq!(got, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_segment_max_single_element_segments() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 1, 2, 3], &[4]).expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, 4).expect("test: segment_max should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert_eq!(result_data.len(), 4);
        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!((result_data[1] - 2.0).abs() < 1e-6);
        assert!((result_data[2] - 3.0).abs() < 1e-6);
        assert!((result_data[3] - 4.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // D > 1 (feature-width) coverage for segment_max/min/prod/any/all.
    //
    // The old implementations flattened the *entire* data array to 1-D
    // length `N*D` and zipped it against a length-`N` id array, silently
    // truncating to `N` elements and misaligning flat index `k` (row `k/D`)
    // against `ids[k]` (segment of row `k`, not row `k/D`). Only `D == 1`
    // happened to work. These tests pin down the correct row-wise behavior.
    // ------------------------------------------------------------------

    /// `segment_max` over a `[4, 2]` input must keep the feature dimension
    /// and return `[num_segments, 2]`.
    #[test]
    fn test_segment_max_2d_feature_dim() {
        // row 0: [1, 5] -> seg 0   row 1: [2, 8] -> seg 0
        // row 2: [3, 6] -> seg 1   row 3: [4, 9] -> seg 1
        let data = Tensor::from_vec(vec![1.0_f32, 5.0, 2.0, 8.0, 3.0, 6.0, 4.0, 9.0], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, 2).expect("test: segment_max should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0: [max(1,2), max(5,8)] = [2, 8]; seg 1: [max(3,4), max(6,9)] = [4, 9].
        assert_eq!(got, vec![2.0, 8.0, 4.0, 9.0]);
    }

    /// An empty segment in `segment_max` must be filled with `T::min_value()`.
    #[test]
    fn test_segment_max_empty_segment() {
        let data =
            Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 2], &[2]).expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, 4).expect("test: segment_max should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1.0, f32::MIN, 2.0, f32::MIN]);
    }

    #[test]
    fn test_segment_max_i32() {
        let data = Tensor::from_vec(vec![1_i32, 5, 2, 8, 3, 6], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, 3).expect("test: segment_max should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![5, 8, 6]);
    }

    #[test]
    fn test_segment_max_f64() {
        let data = Tensor::from_vec(vec![1.5_f64, 5.5, 2.5, 8.5], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, 2).expect("test: segment_max should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![5.5, 8.5]);
    }

    /// Exercises `reduce_rows`'s row-chunked parallel branch (`rows > 1000`)
    /// together with a `D > 1` feature width, checking the parallel
    /// chunk-merge logic for a "pick one side" combine (max) against a
    /// brute-force reference.
    #[test]
    fn test_segment_max_large_input_2d() {
        let rows = 1200_usize;
        let feature_width = 2_usize;
        let num_segments = 10_usize;

        let mut data_vec = Vec::with_capacity(rows * feature_width);
        let mut segment_ids_vec = Vec::with_capacity(rows);
        for i in 0..rows {
            data_vec.push(((i * 37 + 11) % 997) as f32);
            data_vec.push(((i * 53 + 5) % 811) as f32);
            segment_ids_vec.push((i % num_segments) as i32);
        }

        let data = Tensor::from_vec(data_vec.clone(), &[rows, feature_width])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(segment_ids_vec.clone(), &[rows])
            .expect("test: from_vec should succeed");

        let result = segment_max(&data, &segment_ids, num_segments)
            .expect("test: segment_max should succeed");
        assert_eq!(result.shape().dims(), &[num_segments, feature_width]);
        let got = result.to_vec().expect("test: to_vec should succeed");

        let mut expected = vec![f32::MIN; num_segments * feature_width];
        let mut seen = vec![false; num_segments];
        for (row, &seg_id) in segment_ids_vec.iter().enumerate() {
            let seg = seg_id as usize;
            for col in 0..feature_width {
                let val = data_vec[row * feature_width + col];
                let cell = seg * feature_width + col;
                if !seen[seg] || val > expected[cell] {
                    expected[cell] = val;
                }
            }
            seen[seg] = true;
        }

        assert_eq!(got, expected);
    }

    #[test]
    fn test_segment_min_basic() {
        let data = Tensor::from_vec(vec![1.0_f32, 5.0, 2.0, 8.0, 3.0, 6.0], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result = segment_min(&data, &segment_ids, 3).expect("test: segment_min should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_segment_min_negative_values() {
        let data = Tensor::from_vec(vec![-5.0_f32, -2.0, -8.0, -1.0], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_min(&data, &segment_ids, 2).expect("test: segment_min should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![-5.0, -8.0]);
    }

    /// `segment_min` over a `[4, 2]` input must keep the feature dimension
    /// and return `[num_segments, 2]`.
    #[test]
    fn test_segment_min_2d_feature_dim() {
        let data = Tensor::from_vec(vec![1.0_f32, 5.0, 2.0, 8.0, 3.0, 6.0, 4.0, 9.0], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_min(&data, &segment_ids, 2).expect("test: segment_min should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0: [min(1,2), min(5,8)] = [1, 5]; seg 1: [min(3,4), min(6,9)] = [3, 6].
        assert_eq!(got, vec![1.0, 5.0, 3.0, 6.0]);
    }

    /// An empty segment in `segment_min` must be filled with `T::max_value()`.
    #[test]
    fn test_segment_min_empty_segment() {
        let data =
            Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 2], &[2]).expect("test: from_vec should succeed");

        let result = segment_min(&data, &segment_ids, 4).expect("test: segment_min should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1.0, f32::MAX, 2.0, f32::MAX]);
    }

    #[test]
    fn test_segment_min_i32() {
        let data = Tensor::from_vec(vec![1_i32, 5, 2, 8, 3, 6], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result = segment_min(&data, &segment_ids, 3).expect("test: segment_min should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1, 2, 3]);
    }

    #[test]
    fn test_segment_min_f64() {
        let data = Tensor::from_vec(vec![1.5_f64, 5.5, 2.5, 8.5], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_min(&data, &segment_ids, 2).expect("test: segment_min should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1.5, 2.5]);
    }

    #[test]
    fn test_segment_prod_basic() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result =
            segment_prod(&data, &segment_ids, 3).expect("test: segment_prod should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![2.0, 12.0, 30.0]);
    }

    /// `segment_prod` over a `[4, 2]` input must keep the feature dimension
    /// and return `[num_segments, 2]`.
    #[test]
    fn test_segment_prod_2d_feature_dim() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result =
            segment_prod(&data, &segment_ids, 2).expect("test: segment_prod should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0: [1*3, 2*4] = [3, 8]; seg 1: [5*7, 6*8] = [35, 48].
        assert_eq!(got, vec![3.0, 8.0, 35.0, 48.0]);
    }

    /// `[N, 3]` (width-3 feature) coverage: `segment_prod` over a `[3, 3]`
    /// input must return `[num_segments, 3]`.
    #[test]
    fn test_segment_prod_nx3_feature_dim() {
        // row 0: [1,2,3] -> seg 0   row 1: [4,5,6] -> seg 0   row 2: [7,8,9] -> seg 1
        let data = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        )
        .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1], &[3]).expect("test: from_vec should succeed");

        let result =
            segment_prod(&data, &segment_ids, 2).expect("test: segment_prod should succeed");

        assert_eq!(result.shape().dims(), &[2, 3]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0: [1*4, 2*5, 3*6] = [4, 10, 18]; seg 1: [7, 8, 9] (single row).
        assert_eq!(got, vec![4.0, 10.0, 18.0, 7.0, 8.0, 9.0]);
    }

    /// An empty segment in `segment_prod` must be filled with `T::one()`.
    #[test]
    fn test_segment_prod_empty_segment() {
        let data =
            Tensor::from_vec(vec![2.0_f32, 3.0], &[2]).expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 2], &[2]).expect("test: from_vec should succeed");

        let result =
            segment_prod(&data, &segment_ids, 4).expect("test: segment_prod should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![2.0, 1.0, 3.0, 1.0]);
    }

    #[test]
    fn test_segment_prod_i32() {
        let data = Tensor::from_vec(vec![1_i32, 2, 3, 4, 5, 6], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result =
            segment_prod(&data, &segment_ids, 3).expect("test: segment_prod should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![2, 12, 30]);
    }

    #[test]
    fn test_segment_prod_f64() {
        let data = Tensor::from_vec(vec![1.5_f64, 2.0, 2.5, 4.0], &[4])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result =
            segment_prod(&data, &segment_ids, 2).expect("test: segment_prod should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![3.0, 10.0]);
    }

    /// Exercises `reduce_rows`'s row-chunked parallel branch with a
    /// multiplicative combine, whose chunk-merge step differs from
    /// max/min's "pick one side" combine. Uses exactly-representable `+-1.0`
    /// values so the result can be compared for exact equality.
    #[test]
    fn test_segment_prod_large_input() {
        let rows = 1200_usize;
        let num_segments = 10_usize;

        let data_vec: Vec<f32> = (0..rows)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let segment_ids_vec: Vec<i32> = (0..rows).map(|i| (i % num_segments) as i32).collect();

        let data =
            Tensor::from_vec(data_vec.clone(), &[rows]).expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(segment_ids_vec.clone(), &[rows])
            .expect("test: from_vec should succeed");

        let result = segment_prod(&data, &segment_ids, num_segments)
            .expect("test: segment_prod should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        let mut expected = vec![1.0_f32; num_segments];
        for (row, &val) in data_vec.iter().enumerate() {
            let seg = segment_ids_vec[row] as usize;
            expected[seg] *= val;
        }

        assert_eq!(got, expected);
    }

    #[test]
    fn test_segment_any_basic() {
        let data = Tensor::from_vec(vec![0_u8, 0, 1, 0, 0, 0], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result = segment_any(&data, &segment_ids, 3).expect("test: segment_any should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![0, 1, 0]);
    }

    /// `segment_any` over a `[4, 2]` input must keep the feature dimension
    /// and return `[num_segments, 2]`; also checks that a later row (not
    /// just the first) can flip a cell to true.
    #[test]
    fn test_segment_any_2d_feature_dim() {
        // row 0: [0,0] -> seg 0   row 1: [0,1] -> seg 0
        // row 2: [0,0] -> seg 1   row 3: [0,0] -> seg 1
        let data = Tensor::from_vec(vec![0_u8, 0, 0, 1, 0, 0, 0, 0], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_any(&data, &segment_ids, 2).expect("test: segment_any should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(got, vec![0, 1, 0, 0]);
    }

    /// An empty segment in `segment_any` must be filled with `0` (false).
    #[test]
    fn test_segment_any_empty_segment() {
        let data = Tensor::from_vec(vec![1_u8, 0], &[2]).expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 2], &[2]).expect("test: from_vec should succeed");

        let result = segment_any(&data, &segment_ids, 4).expect("test: segment_any should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_segment_all_basic() {
        let data = Tensor::from_vec(vec![1_u8, 1, 1, 0, 1, 1], &[6])
            .expect("test: from_vec should succeed");
        let segment_ids = Tensor::from_vec(vec![0_i32, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let result = segment_all(&data, &segment_ids, 3).expect("test: segment_all should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1, 0, 1]);
    }

    /// `segment_all` over a `[4, 2]` input must keep the feature dimension
    /// and return `[num_segments, 2]`; also checks that a zero on a later
    /// row (not just the first) forces the cell to false.
    #[test]
    fn test_segment_all_2d_feature_dim() {
        // row 0: [1,1] -> seg 0   row 1: [1,0] -> seg 0
        // row 2: [1,1] -> seg 1   row 3: [1,1] -> seg 1
        let data = Tensor::from_vec(vec![1_u8, 1, 1, 0, 1, 1, 1, 1], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let result = segment_all(&data, &segment_ids, 2).expect("test: segment_all should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(got, vec![1, 0, 1, 1]);
    }

    /// An empty segment in `segment_all` must be filled with `1`
    /// (vacuously true).
    #[test]
    fn test_segment_all_empty_segment() {
        let data = Tensor::from_vec(vec![1_u8, 1], &[2]).expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![0_i32, 2], &[2]).expect("test: from_vec should succeed");

        let result = segment_all(&data, &segment_ids, 4).expect("test: segment_all should succeed");
        let got = result.to_vec().expect("test: to_vec should succeed");

        assert_eq!(got, vec![1, 1, 1, 1]);
    }

    /// `unsorted_segment_max` over a `[4, 2]` input with genuinely unsorted
    /// (interleaved) segment ids must still keep the feature dimension and
    /// group rows correctly regardless of order.
    #[test]
    fn test_unsorted_segment_max_2d_feature_dim() {
        // row 0: [1,5] -> seg 1   row 1: [2,8] -> seg 0
        // row 2: [3,6] -> seg 1   row 3: [4,9] -> seg 0
        let data = Tensor::from_vec(vec![1.0_f32, 5.0, 2.0, 8.0, 3.0, 6.0, 4.0, 9.0], &[4, 2])
            .expect("test: from_vec should succeed");
        let segment_ids =
            Tensor::from_vec(vec![1_i32, 0, 1, 0], &[4]).expect("test: from_vec should succeed");

        let result = unsorted_segment_max(&data, &segment_ids, 2)
            .expect("test: unsorted_segment_max should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        let got = result.to_vec().expect("test: to_vec should succeed");
        // seg 0 (rows 1,3): [max(2,4), max(8,9)] = [4, 9].
        // seg 1 (rows 0,2): [max(1,3), max(5,6)] = [3, 6].
        assert_eq!(got, vec![4.0, 9.0, 3.0, 6.0]);
    }
}
