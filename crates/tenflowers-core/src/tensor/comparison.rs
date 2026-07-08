//! Comparison and Logical Operations
//!
//! This module provides tensor comparison operations (==, !=, >, <, etc.)
//! and logical operations for boolean tensors. All operations support
//! broadcasting and both CPU and GPU execution.

use super::core::{Tensor, TensorStorage};
use crate::Result;

impl<T> Tensor<T>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Element-wise equality comparison
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>>
    where
        T: PartialEq,
    {
        if self.device() != other.device() {
            return Err(crate::TensorError::device_mismatch(
                "comparison",
                &self.device().to_string(),
                &other.device().to_string(),
            ));
        }

        let broadcast_shape = self.shape().broadcast_shape(other.shape()).ok_or_else(|| {
            crate::TensorError::ShapeMismatch {
                operation: "broadcast".to_string(),
                expected: self.shape().to_string(),
                got: other.shape().to_string(),
                context: None,
            }
        })?;

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
                use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};

                let a_broadcast =
                    arr_a
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast first tensor".to_string(),
                            )
                        })?;
                let b_broadcast =
                    arr_b
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast second tensor".to_string(),
                            )
                        })?;

                let mut result = ArrayD::default(a_broadcast.raw_dim());
                Zip::from(&mut result)
                    .and(&a_broadcast)
                    .and(&b_broadcast)
                    .for_each(|r, a_val, b_val| {
                        *r = a_val == b_val;
                    });

                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Use the high-level comparison function which handles GPU operations
                let result = crate::ops::comparison::eq(self, other)?;
                // Convert from u8 to bool tensor
                match result.storage {
                    TensorStorage::Cpu(arr) => {
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                    #[cfg(feature = "gpu")]
                    TensorStorage::Gpu(ref gpu_buf) => {
                        // For GPU, we need to convert u8 to bool
                        let cpu_result = gpu_buf.to_cpu()?;
                        let arr = scirs2_core::ndarray::ArrayD::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(result.shape().dims()),
                            cpu_result,
                        )
                        .map_err(|e| crate::TensorError::invalid_shape_simple(e.to_string()))?;
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                }
            }
        }
    }

    /// Element-wise not-equal comparison
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>>
    where
        T: PartialEq,
    {
        let eq_result = self.eq(other)?;
        match &eq_result.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| !x);
                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Use the high-level comparison function which handles GPU operations
                let result = crate::ops::comparison::ne(self, other)?;
                // Convert from u8 to bool tensor
                match result.storage {
                    TensorStorage::Cpu(arr) => {
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                    #[cfg(feature = "gpu")]
                    TensorStorage::Gpu(ref gpu_buf) => {
                        // For GPU, we need to convert u8 to bool
                        let cpu_result = gpu_buf.to_cpu()?;
                        let arr = scirs2_core::ndarray::ArrayD::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(result.shape().dims()),
                            cpu_result,
                        )
                        .map_err(|e| crate::TensorError::invalid_shape_simple(e.to_string()))?;
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                }
            }
        }
    }

    /// Element-wise greater-than comparison
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>>
    where
        T: PartialOrd,
    {
        if self.device() != other.device() {
            return Err(crate::TensorError::device_mismatch(
                "comparison",
                &self.device().to_string(),
                &other.device().to_string(),
            ));
        }

        let broadcast_shape = self.shape().broadcast_shape(other.shape()).ok_or_else(|| {
            crate::TensorError::ShapeMismatch {
                operation: "broadcast".to_string(),
                expected: self.shape().to_string(),
                got: other.shape().to_string(),
                context: None,
            }
        })?;

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
                use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};

                let a_broadcast =
                    arr_a
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast first tensor".to_string(),
                            )
                        })?;
                let b_broadcast =
                    arr_b
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast second tensor".to_string(),
                            )
                        })?;

                let mut result = ArrayD::default(a_broadcast.raw_dim());
                Zip::from(&mut result)
                    .and(&a_broadcast)
                    .and(&b_broadcast)
                    .for_each(|r, a_val, b_val| {
                        *r = a_val > b_val;
                    });

                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Use the high-level comparison function which handles GPU operations
                let result = crate::ops::comparison::gt(self, other)?;
                // Convert from u8 to bool tensor
                match result.storage {
                    TensorStorage::Cpu(arr) => {
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                    #[cfg(feature = "gpu")]
                    TensorStorage::Gpu(ref gpu_buf) => {
                        // For GPU, we need to convert u8 to bool
                        let cpu_result = gpu_buf.to_cpu()?;
                        let arr = scirs2_core::ndarray::ArrayD::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(result.shape().dims()),
                            cpu_result,
                        )
                        .map_err(|e| crate::TensorError::invalid_shape_simple(e.to_string()))?;
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                }
            }
        }
    }

    /// Element-wise greater-than-or-equal comparison
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>>
    where
        T: PartialOrd,
    {
        if self.device() != other.device() {
            return Err(crate::TensorError::device_mismatch(
                "comparison",
                &self.device().to_string(),
                &other.device().to_string(),
            ));
        }

        let broadcast_shape = self.shape().broadcast_shape(other.shape()).ok_or_else(|| {
            crate::TensorError::ShapeMismatch {
                operation: "broadcast".to_string(),
                expected: self.shape().to_string(),
                got: other.shape().to_string(),
                context: None,
            }
        })?;

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
                use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};

                let a_broadcast =
                    arr_a
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast first tensor".to_string(),
                            )
                        })?;
                let b_broadcast =
                    arr_b
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast second tensor".to_string(),
                            )
                        })?;

                let mut result = ArrayD::default(a_broadcast.raw_dim());
                Zip::from(&mut result)
                    .and(&a_broadcast)
                    .and(&b_broadcast)
                    .for_each(|r, a_val, b_val| {
                        *r = a_val >= b_val;
                    });

                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Use the high-level comparison function which handles GPU operations
                let result = crate::ops::comparison::ge(self, other)?;
                // Convert from u8 to bool tensor
                match result.storage {
                    TensorStorage::Cpu(arr) => {
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                    #[cfg(feature = "gpu")]
                    TensorStorage::Gpu(ref gpu_buf) => {
                        // For GPU, we need to convert u8 to bool
                        let cpu_result = gpu_buf.to_cpu()?;
                        let arr = scirs2_core::ndarray::ArrayD::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(result.shape().dims()),
                            cpu_result,
                        )
                        .map_err(|e| crate::TensorError::invalid_shape_simple(e.to_string()))?;
                        let bool_arr = arr.mapv(|x| x != 0);
                        Ok(Tensor::<bool>::from_array(bool_arr))
                    }
                }
            }
        }
    }

    /// Element-wise less-than comparison
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>>
    where
        T: PartialOrd,
    {
        other.gt(self)
    }

    /// Element-wise less-than-or-equal comparison
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>>
    where
        T: PartialOrd,
    {
        other.ge(self)
    }
}

// Boolean tensor specific operations
impl Tensor<bool> {
    /// Cast boolean tensor to u8 tensor (false -> 0, true -> 1)
    pub fn cast_to_u8(&self) -> Result<Tensor<u8>> {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let u8_arr = arr.mapv(|x| if x { 1u8 } else { 0u8 });
                Ok(Tensor::<u8>::from_array(u8_arr))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For now, GPU bool->u8 casting not implemented
                Err(crate::TensorError::unsupported_operation_simple(
                    "GPU bool to u8 casting not yet implemented".to_string(),
                ))
            }
        }
    }
    /// Element-wise logical AND operation
    pub fn logical_and(&self, other: &Self) -> Result<Self> {
        if self.device() != other.device() {
            return Err(crate::TensorError::device_mismatch(
                "comparison",
                &self.device().to_string(),
                &other.device().to_string(),
            ));
        }

        let broadcast_shape = self.shape().broadcast_shape(other.shape()).ok_or_else(|| {
            crate::TensorError::ShapeMismatch {
                operation: "broadcast".to_string(),
                expected: self.shape().to_string(),
                got: other.shape().to_string(),
                context: None,
            }
        })?;

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
                use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};

                let a_broadcast =
                    arr_a
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast first tensor".to_string(),
                            )
                        })?;
                let b_broadcast =
                    arr_b
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast second tensor".to_string(),
                            )
                        })?;

                let mut result = ArrayD::default(a_broadcast.raw_dim());
                Zip::from(&mut result)
                    .and(&a_broadcast)
                    .and(&b_broadcast)
                    .for_each(|r, a_val, b_val| {
                        *r = *a_val && *b_val;
                    });

                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Convert bool tensors to u8 for GPU logical operations
                let self_u8 = self.cast_to_u8()?;
                let other_u8 = other.cast_to_u8()?;

                // Use the high-level logical function which handles GPU operations
                let result_u8 = crate::ops::logical::logical_and(&self_u8, &other_u8)?;

                // Convert result back to bool tensor
                result_u8.cast_to_bool()
            }
        }
    }

    /// Element-wise logical OR operation
    pub fn logical_or(&self, other: &Self) -> Result<Self> {
        if self.device() != other.device() {
            return Err(crate::TensorError::device_mismatch(
                "comparison",
                &self.device().to_string(),
                &other.device().to_string(),
            ));
        }

        let broadcast_shape = self.shape().broadcast_shape(other.shape()).ok_or_else(|| {
            crate::TensorError::ShapeMismatch {
                operation: "broadcast".to_string(),
                expected: self.shape().to_string(),
                got: other.shape().to_string(),
                context: None,
            }
        })?;

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
                use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};

                let a_broadcast =
                    arr_a
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast first tensor".to_string(),
                            )
                        })?;
                let b_broadcast =
                    arr_b
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast second tensor".to_string(),
                            )
                        })?;

                let mut result = ArrayD::default(a_broadcast.raw_dim());
                Zip::from(&mut result)
                    .and(&a_broadcast)
                    .and(&b_broadcast)
                    .for_each(|r, a_val, b_val| {
                        *r = *a_val || *b_val;
                    });

                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Convert bool tensors to u8 for GPU logical operations
                let self_u8 = self.cast_to_u8()?;
                let other_u8 = other.cast_to_u8()?;

                // Use the high-level logical function which handles GPU operations
                let result_u8 = crate::ops::logical::logical_or(&self_u8, &other_u8)?;

                // Convert result back to bool tensor
                result_u8.cast_to_bool()
            }
        }
    }

    /// Element-wise logical NOT operation
    pub fn logical_not(&self) -> Result<Self> {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| !x);
                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Convert bool tensor to u8 for GPU logical operations
                let self_u8 = self.cast_to_u8()?;

                // Use the high-level logical function which handles GPU operations
                let result_u8 = crate::ops::logical::logical_not(&self_u8)?;

                // Convert result back to bool tensor
                result_u8.cast_to_bool()
            }
        }
    }

    /// Element-wise logical XOR operation
    pub fn logical_xor(&self, other: &Self) -> Result<Self> {
        if self.device() != other.device() {
            return Err(crate::TensorError::device_mismatch(
                "comparison",
                &self.device().to_string(),
                &other.device().to_string(),
            ));
        }

        let broadcast_shape = self.shape().broadcast_shape(other.shape()).ok_or_else(|| {
            crate::TensorError::ShapeMismatch {
                operation: "broadcast".to_string(),
                expected: self.shape().to_string(),
                got: other.shape().to_string(),
                context: None,
            }
        })?;

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
                use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};

                let a_broadcast =
                    arr_a
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast first tensor".to_string(),
                            )
                        })?;
                let b_broadcast =
                    arr_b
                        .broadcast(IxDyn(broadcast_shape.dims()))
                        .ok_or_else(|| {
                            crate::TensorError::invalid_argument(
                                "Cannot broadcast second tensor".to_string(),
                            )
                        })?;

                let mut result = ArrayD::default(a_broadcast.raw_dim());
                Zip::from(&mut result)
                    .and(&a_broadcast)
                    .and(&b_broadcast)
                    .for_each(|r, a_val, b_val| {
                        *r = *a_val ^ *b_val;
                    });

                Ok(Tensor::<bool>::from_array(result))
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Convert bool tensors to u8 for GPU logical operations
                let self_u8 = self.cast_to_u8()?;
                let other_u8 = other.cast_to_u8()?;

                // Use the high-level logical function which handles GPU operations
                let result_u8 = crate::ops::logical::logical_xor(&self_u8, &other_u8)?;

                // Convert result back to bool tensor
                result_u8.cast_to_bool()
            }
        }
    }

    /// Reduce tensor using logical AND along specified axes
    pub fn all(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self> {
        crate::ops::reduction::all(self, axes, keepdims)
    }

    /// Reduce tensor using logical OR along specified axes
    pub fn any(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self> {
        crate::ops::reduction::any(self, axes, keepdims)
    }
}

// U8 tensor specific operations
impl Tensor<u8> {
    /// Cast u8 tensor to boolean tensor (0 -> false, non-zero -> true)
    pub fn cast_to_bool(&self) -> Result<Tensor<bool>> {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let bool_arr = arr.mapv(|x| x != 0);
                Ok(Tensor::<bool>::from_array(bool_arr))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // No native GPU u8->bool cast kernel exists. Read the tensor
                // back to the host (a real device->host transfer) and
                // recurse into the CPU arm above, which is known-correct.
                let cpu_self = self.to_cpu()?;
                cpu_self.cast_to_bool()
            }
        }
    }
}

// GPU-resident correctness test for the readback+delegate fix in
// `cast_to_bool()`'s GPU arm. Note: the result is `Tensor<bool>`, and `bool`
// does not implement `bytemuck::Pod` (restricted valid bit patterns), so
// `to_vec()` (which requires `Pod`-adjacent numeric bounds) cannot be used
// on the result - use `as_slice()` instead, which carries no such bound.
// Skips gracefully (without failing the suite) if no GPU adapter is
// available.
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_cast_to_bool_matches_cpu_reference() {
        // Element count is a multiple of 4 (`wgpu::COPY_BUFFER_ALIGNMENT`):
        // `u8` is 1 byte wide, so an unaligned element count (e.g. 5, giving
        // a 5-byte copy) trips a pre-existing `COPY_BUFFER_ALIGNMENT`
        // validation error inside `GpuBuffer::to_cpu()`'s host readback -
        // a limitation of the shared buffer-transfer machinery, not of the
        // `cast_to_bool` readback+delegate fix under test here (every other
        // GPU test in this PR happens to use `f32`, whose 4-byte width keeps
        // any element count aligned, which is why only this `u8` test needs
        // to pick its size deliberately).
        let src = Tensor::<u8>::from_vec(vec![0, 1, 2, 0, 5, 255, 3, 0], &[8])
            .expect("test: from_vec should succeed");

        let src_gpu = match src.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };

        let result = src_gpu
            .cast_to_bool()
            .expect("test: gpu cast_to_bool should succeed with a real adapter");
        let data = result
            .as_slice()
            .expect("cast_to_bool result must be CPU-resident and contiguous");
        assert_eq!(data, &[false, true, true, false, true, true, true, false]);
    }
}
