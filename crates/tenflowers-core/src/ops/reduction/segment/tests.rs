//! Tests for segment reduction operations

#[cfg(test)]
mod tests {
    use crate::ops::reduction::segment::{segment_max, segment_mean, segment_sum};
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
}
