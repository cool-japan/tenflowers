//! Zero-copy operations for memory-efficient dataset loading
//!
//! This module provides utilities for zero-copy data access and tensor views
//! that avoid unnecessary data copying during dataset operations.

use crate::Dataset;
use std::ops::Range;
use std::sync::Arc;
use tenflowers_core::{Result, Shape, Tensor, TensorError};

#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapOptions};
#[cfg(feature = "mmap")]
use std::fs::File;
#[cfg(feature = "mmap")]
use std::marker::PhantomData;
#[cfg(feature = "mmap")]
use std::path::Path;

/// A zero-copy view into a tensor that shares memory with the original tensor
#[derive(Debug, Clone)]
pub struct TensorView<T> {
    /// Reference to the original tensor
    source: Arc<Tensor<T>>,
    /// Offset into the source tensor data
    offset: usize,
    /// Shape of this view
    shape: Shape,
    /// Strides for accessing data
    strides: Vec<usize>,
}

impl<T> TensorView<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a new tensor view from a source tensor
    pub fn new(
        source: Arc<Tensor<T>>,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(TensorError::invalid_argument(
                "Shape and strides must have the same length".to_string(),
            ));
        }

        let shape = Shape::new(shape);

        Ok(Self {
            source,
            offset,
            shape,
            strides,
        })
    }

    /// Create a view that slices the tensor along specified dimensions
    pub fn slice(source: Arc<Tensor<T>>, ranges: &[Range<usize>]) -> Result<Self> {
        let source_shape = source.shape();

        if ranges.len() != source_shape.rank() {
            return Err(TensorError::invalid_argument(format!(
                "Number of ranges ({}) must match tensor rank ({})",
                ranges.len(),
                source_shape.rank()
            )));
        }

        // Calculate new shape and offset
        let mut new_shape = Vec::new();
        let mut offset = 0;
        let mut stride = 1;

        // Calculate strides (row-major order)
        let mut strides = vec![1; ranges.len()];
        for i in (0..ranges.len()).rev() {
            strides[i] = stride;
            stride *= source_shape.dims()[i];
        }

        // Calculate offset and new shape
        for (i, range) in ranges.iter().enumerate() {
            if range.end > source_shape.dims()[i] {
                return Err(TensorError::invalid_argument(format!(
                    "Range end {} exceeds dimension size {}",
                    range.end,
                    source_shape.dims()[i]
                )));
            }

            offset += range.start * strides[i];
            new_shape.push(range.end - range.start);
        }

        Self::new(source, offset, new_shape, strides)
    }

    /// Create a view that reshapes the tensor without copying data
    pub fn reshape(source: Arc<Tensor<T>>, new_shape: Vec<usize>) -> Result<Self> {
        let total_elements = new_shape.iter().product::<usize>();
        let source_elements = source.shape().size();

        if total_elements != source_elements {
            return Err(TensorError::invalid_argument(
                format!("Cannot reshape tensor with {source_elements} elements to shape with {total_elements} elements")
            ));
        }

        // Calculate row-major strides for new shape
        let mut strides = vec![1; new_shape.len()];
        let mut stride = 1;
        for i in (0..new_shape.len()).rev() {
            strides[i] = stride;
            stride *= new_shape[i];
        }

        Self::new(source, 0, new_shape, strides)
    }

    /// Get the shape of this view
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the strides of this view
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the offset into the source tensor
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Check if this view is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for i in (0..self.shape.rank()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape.dims()[i];
        }
        true
    }

    /// Materialize this view into a concrete tensor (performs copy)
    pub fn materialize(&self) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable,
    {
        if self.is_contiguous() {
            // If contiguous, we can potentially slice directly
            if let Some(slice) = self.source.as_slice() {
                let start = self.offset;
                let end = start + self.shape.size();
                let data = slice[start..end].to_vec();
                return Tensor::from_vec(data, self.shape.dims());
            }
        }

        // Non-contiguous case: need to copy elements individually
        let mut data = Vec::with_capacity(self.shape.size());
        let indices = self.iter_indices();

        if let Some(slice) = self.source.as_slice() {
            for linear_idx in indices {
                let source_idx = self.offset + linear_idx;
                data.push(slice[source_idx]);
            }
        } else {
            return Err(TensorError::invalid_argument(
                "Cannot access GPU tensor data for materialization".to_string(),
            ));
        }

        Tensor::from_vec(data, self.shape.dims())
    }

    /// Iterate over linear indices for this view
    fn iter_indices(&self) -> LinearIndexIterator {
        LinearIndexIterator::new(&self.shape, &self.strides)
    }
}

/// Iterator over linear indices for a tensor view
struct LinearIndexIterator {
    shape: Vec<usize>,
    strides: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl LinearIndexIterator {
    fn new(shape: &Shape, strides: &[usize]) -> Self {
        Self {
            shape: shape.dims().to_vec(),
            strides: strides.to_vec(),
            current: vec![0; shape.rank()],
            done: shape.size() == 0,
        }
    }
}

impl Iterator for LinearIndexIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Calculate current linear index
        let linear_idx = self
            .current
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum();

        // Advance to next index
        let mut carry = 1;
        for i in (0..self.current.len()).rev() {
            self.current[i] += carry;
            if self.current[i] < self.shape[i] {
                carry = 0;
                break;
            } else {
                self.current[i] = 0;
            }
        }

        if carry == 1 {
            self.done = true;
        }

        Some(linear_idx)
    }
}

/// Zero-copy dataset wrapper that provides views into a large tensor
pub struct ZeroCopyDataset<T> {
    /// Source tensor containing all data
    source: Arc<Tensor<T>>,
    /// Number of samples
    num_samples: usize,
    /// Size of each sample (number of elements)
    sample_size: usize,
    /// Features shape (without batch dimension)
    feature_shape: Vec<usize>,
    /// Labels shape (without batch dimension)  
    label_shape: Vec<usize>,
    /// Offset to labels in the source tensor
    labels_offset: usize,
}

impl<T> ZeroCopyDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a new zero-copy dataset from features and labels tensors
    pub fn new(features: Tensor<T>, labels: Tensor<T>) -> Result<Self> {
        let features_shape = features.shape();
        let labels_shape = labels.shape();

        if features_shape.dims()[0] != labels_shape.dims()[0] {
            return Err(TensorError::invalid_argument(
                "Features and labels must have same batch size".to_string(),
            ));
        }

        let num_samples = features_shape.dims()[0];
        let feature_elements = features_shape.size() / num_samples;
        let label_elements = labels_shape.size() / num_samples;

        // Concatenate features and labels into a single tensor for zero-copy access
        let mut combined_data = Vec::new();

        if let (Some(feat_slice), Some(label_slice)) = (features.as_slice(), labels.as_slice()) {
            // Interleave features and labels for each sample
            for i in 0..num_samples {
                let feat_start = i * feature_elements;
                let feat_end = feat_start + feature_elements;
                combined_data.extend_from_slice(&feat_slice[feat_start..feat_end]);

                let label_start = i * label_elements;
                let label_end = label_start + label_elements;
                combined_data.extend_from_slice(&label_slice[label_start..label_end]);
            }
        } else {
            return Err(TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensors not supported for zero-copy dataset)"
                    .to_string(),
            ));
        }

        let sample_size = feature_elements + label_elements;
        let combined_shape = vec![num_samples * sample_size];
        let source = Arc::new(Tensor::from_vec(combined_data, &combined_shape)?);

        Ok(Self {
            source,
            num_samples,
            sample_size,
            feature_shape: features_shape.dims()[1..].to_vec(),
            label_shape: labels_shape.dims()[1..].to_vec(),
            labels_offset: feature_elements,
        })
    }

    /// Get a zero-copy view of a sample
    pub fn get_view(&self, index: usize) -> Result<(TensorView<T>, TensorView<T>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset with {} samples",
                index, self.num_samples
            )));
        }

        let sample_offset = index * self.sample_size;

        // Create feature view
        let feature_strides = self.calculate_strides(&self.feature_shape);
        let feature_view = TensorView::new(
            Arc::clone(&self.source),
            sample_offset,
            self.feature_shape.clone(),
            feature_strides,
        )?;

        // Create label view
        let label_strides = self.calculate_strides(&self.label_shape);
        let label_view = TensorView::new(
            Arc::clone(&self.source),
            sample_offset + self.labels_offset,
            self.label_shape.clone(),
            label_strides,
        )?;

        Ok((feature_view, label_view))
    }

    fn calculate_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides
    }
}

impl<T> Dataset<T> for ZeroCopyDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        let (feature_view, label_view) = self.get_view(index)?;

        // Materialize views into concrete tensors
        let features = feature_view.materialize()?;
        let labels = label_view.materialize()?;

        Ok((features, labels))
    }
}

/// Memory-mapped zero-copy dataset for large datasets
pub struct MemoryMappedDataset<T> {
    /// Memory-mapped data
    data: Arc<[T]>,
    /// Number of samples
    num_samples: usize,
    /// Feature size per sample
    feature_size: usize,
    /// Label size per sample
    label_size: usize,
    /// Feature shape (without batch dimension)
    feature_shape: Vec<usize>,
    /// Label shape (without batch dimension)
    label_shape: Vec<usize>,
}

impl<T> MemoryMappedDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a memory-mapped dataset from raw data
    /// Data layout: [sample0_features, sample0_labels, sample1_features, sample1_labels, ...]
    pub fn new(
        data: Arc<[T]>,
        num_samples: usize,
        feature_shape: Vec<usize>,
        label_shape: Vec<usize>,
    ) -> Result<Self> {
        let feature_size = feature_shape.iter().product();
        let label_size = label_shape.iter().product();
        let expected_size = num_samples * (feature_size + label_size);

        if data.len() != expected_size {
            return Err(TensorError::invalid_argument(format!(
                "Data size {} doesn't match expected size {} for {} samples",
                data.len(),
                expected_size,
                num_samples
            )));
        }

        Ok(Self {
            data,
            num_samples,
            feature_size,
            label_size,
            feature_shape,
            label_shape,
        })
    }

    /// Get a zero-copy slice for a sample
    pub fn get_raw_sample(&self, index: usize) -> Result<(&[T], &[T])> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {index} out of bounds"
            )));
        }

        let sample_size = self.feature_size + self.label_size;
        let start = index * sample_size;

        let features = &self.data[start..start + self.feature_size];
        let labels = &self.data[start + self.feature_size..start + sample_size];

        Ok((features, labels))
    }
}

impl<T> Dataset<T> for MemoryMappedDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        let (feat_slice, label_slice) = self.get_raw_sample(index)?;

        // Create tensors from slices (this involves copying)
        let features = Tensor::from_vec(feat_slice.to_vec(), &self.feature_shape)?;
        let labels = Tensor::from_vec(label_slice.to_vec(), &self.label_shape)?;

        Ok((features, labels))
    }
}

/// Enhanced memory-mapped dataset that loads data directly from files
/// This provides true zero-copy access to very large datasets that exceed available RAM
#[cfg(feature = "mmap")]
#[allow(unsafe_code)] // Required for memory mapping
pub struct MemoryMappedFileDataset<T> {
    /// Memory-mapped file
    mmap: Mmap,
    /// Number of samples  
    num_samples: usize,
    /// Feature size per sample (in bytes)
    feature_size_bytes: usize,
    /// Label size per sample (in bytes)
    label_size_bytes: usize,
    /// Feature shape (without batch dimension)
    feature_shape: Vec<usize>,
    /// Label shape (without batch dimension)
    label_shape: Vec<usize>,
    /// Size of each element in bytes
    #[allow(dead_code)] // Used for future validation features
    element_size: usize,
    /// Total size per sample (features + labels) in bytes
    sample_size_bytes: usize,
    /// File path for debugging
    file_path: String,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

#[cfg(feature = "mmap")]
impl<T> MemoryMappedFileDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a memory-mapped dataset directly from a file
    /// The file should contain binary data in the format:
    /// [sample0_features, sample0_labels, sample1_features, sample1_labels, ...]
    #[allow(unsafe_code)] // Required for memory mapping
    pub fn from_file<P: AsRef<Path>>(
        file_path: P,
        num_samples: usize,
        feature_shape: Vec<usize>,
        label_shape: Vec<usize>,
    ) -> Result<Self> {
        let file_path = file_path.as_ref();
        let file = File::open(file_path).map_err(|e| {
            TensorError::io_error_simple(format!(
                "Failed to open file {}: {}",
                file_path.display(),
                e
            ))
        })?;

        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                TensorError::io_error_simple(format!(
                    "Failed to memory map file {}: {}",
                    file_path.display(),
                    e
                ))
            })?
        };

        let element_size = std::mem::size_of::<T>();
        let feature_size = feature_shape.iter().product::<usize>();
        let label_size = label_shape.iter().product::<usize>();
        let feature_size_bytes = feature_size * element_size;
        let label_size_bytes = label_size * element_size;
        let sample_size_bytes = feature_size_bytes + label_size_bytes;
        let expected_file_size = num_samples * sample_size_bytes;

        if mmap.len() < expected_file_size {
            return Err(TensorError::invalid_argument(format!(
                "File {} size {} is smaller than expected size {} for {} samples",
                file_path.display(),
                mmap.len(),
                expected_file_size,
                num_samples
            )));
        }

        Ok(Self {
            mmap,
            num_samples,
            feature_size_bytes,
            label_size_bytes,
            feature_shape,
            label_shape,
            element_size,
            sample_size_bytes,
            file_path: file_path.display().to_string(),
            _phantom: PhantomData,
        })
    }

    /// Create a memory-mapped dataset from an existing file with automatic shape detection
    /// Assumes all samples have the same shape
    pub fn auto_detect<P: AsRef<Path>>(
        file_path: P,
        feature_shape: Vec<usize>,
        label_shape: Vec<usize>,
    ) -> Result<Self> {
        let file_path_ref = file_path.as_ref();
        let metadata = std::fs::metadata(file_path_ref).map_err(|e| {
            TensorError::io_error_simple(format!(
                "Failed to get metadata for {}: {}",
                file_path_ref.display(),
                e
            ))
        })?;

        let element_size = std::mem::size_of::<T>();
        let feature_size = feature_shape.iter().product::<usize>();
        let label_size = label_shape.iter().product::<usize>();
        let sample_size_bytes = (feature_size + label_size) * element_size;

        let num_samples = metadata.len() as usize / sample_size_bytes;

        if metadata.len() as usize % sample_size_bytes != 0 {
            return Err(TensorError::invalid_argument(format!(
                "File {} size {} is not evenly divisible by sample size {}",
                file_path_ref.display(),
                metadata.len(),
                sample_size_bytes
            )));
        }

        Self::from_file(file_path, num_samples, feature_shape, label_shape)
    }

    /// Get raw byte slices for a sample (features and labels)
    fn get_raw_sample_bytes(&self, index: usize) -> Result<(&[u8], &[u8])> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset with {} samples",
                index, self.num_samples
            )));
        }

        let sample_offset = index * self.sample_size_bytes;
        let feature_start = sample_offset;
        let feature_end = feature_start + self.feature_size_bytes;
        let label_start = feature_end;
        let label_end = label_start + self.label_size_bytes;

        let feature_bytes = &self.mmap[feature_start..feature_end];
        let label_bytes = &self.mmap[label_start..label_end];

        Ok((feature_bytes, label_bytes))
    }

    /// Get file statistics for monitoring
    pub fn file_stats(&self) -> MemoryMappedFileStats {
        MemoryMappedFileStats {
            file_path: self.file_path.clone(),
            file_size: self.mmap.len(),
            num_samples: self.num_samples,
            sample_size_bytes: self.sample_size_bytes,
            feature_shape: self.feature_shape.clone(),
            label_shape: self.label_shape.clone(),
        }
    }

    /// Create a view that spans multiple samples for batch processing
    pub fn get_batch_view(&self, start_index: usize, batch_size: usize) -> Result<&[u8]> {
        if start_index + batch_size > self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Batch {}..{} out of bounds for dataset with {} samples",
                start_index,
                start_index + batch_size,
                self.num_samples
            )));
        }

        let start_offset = start_index * self.sample_size_bytes;
        let end_offset = (start_index + batch_size) * self.sample_size_bytes;

        Ok(&self.mmap[start_offset..end_offset])
    }
}

#[cfg(feature = "mmap")]
impl<T> Dataset<T> for MemoryMappedFileDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + bytemuck::Pod + 'static,
{
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        let (feature_bytes, label_bytes) = self.get_raw_sample_bytes(index)?;

        // Convert bytes to typed slices safely using bytemuck
        let feature_data: &[T] = bytemuck::cast_slice(feature_bytes);
        let label_data: &[T] = bytemuck::cast_slice(label_bytes);

        // Create tensors from slices
        let features = Tensor::from_vec(feature_data.to_vec(), &self.feature_shape)?;
        let labels = Tensor::from_vec(label_data.to_vec(), &self.label_shape)?;

        Ok((features, labels))
    }
}

/// Statistics for memory-mapped file datasets
#[cfg(feature = "mmap")]
#[derive(Debug, Clone)]
pub struct MemoryMappedFileStats {
    pub file_path: String,
    pub file_size: usize,
    pub num_samples: usize,
    pub sample_size_bytes: usize,
    pub feature_shape: Vec<usize>,
    pub label_shape: Vec<usize>,
}

#[cfg(feature = "mmap")]
impl MemoryMappedFileStats {
    /// Calculate memory efficiency (how much of mapped memory is actually used)
    pub fn memory_efficiency(&self) -> f64 {
        let used_size = self.num_samples * self.sample_size_bytes;
        used_size as f64 / self.file_size as f64
    }

    /// Get human-readable file size
    pub fn human_readable_size(&self) -> String {
        let size = self.file_size as f64;
        if size < 1024.0 {
            format!("{size:.1} B")
        } else if size < 1024.0 * 1024.0 {
            let kb = size / 1024.0;
            format!("{kb:.1} KB")
        } else if size < 1024.0 * 1024.0 * 1024.0 {
            let mb = size / (1024.0 * 1024.0);
            format!("{mb:.1} MB")
        } else {
            let gb = size / (1024.0 * 1024.0 * 1024.0);
            format!("{gb:.1} GB")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Arc::new(Tensor::from_vec(data, &[2, 3]).unwrap());

        // Slice first row: [1, 2, 3]
        let view = TensorView::slice(tensor, &[0..1, 0..3]).unwrap();
        assert_eq!(view.shape().dims(), &[1, 3]);
        assert_eq!(view.offset(), 0);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_tensor_view_reshape() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Arc::new(Tensor::from_vec(data, &[2, 3]).unwrap());

        // Reshape to [6, 1]
        let view = TensorView::reshape(tensor, vec![6, 1]).unwrap();
        assert_eq!(view.shape().dims(), &[6, 1]);
        assert_eq!(view.offset(), 0);
    }

    #[test]
    fn test_zero_copy_dataset() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = ZeroCopyDataset::new(features, labels).unwrap();
        assert_eq!(dataset.len(), 2);

        let (feat, label) = dataset.get(0).unwrap();
        assert_eq!(feat.shape().dims(), &[2]);
        assert_eq!(label.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_memory_mapped_dataset() {
        // Data layout: [feat0_0, feat0_1, label0, feat1_0, feat1_1, label1]
        let data: Arc<[f32]> = Arc::from(vec![1.0, 2.0, 0.0, 3.0, 4.0, 1.0]);

        let dataset = MemoryMappedDataset::new(
            data,
            2,       // 2 samples
            vec![2], // 2 features per sample
            vec![],  // scalar labels
        )
        .unwrap();

        assert_eq!(dataset.len(), 2);

        let (feat0, label0) = dataset.get(0).unwrap();
        assert_eq!(feat0.shape().dims(), &[2]);
        assert_eq!(label0.shape().dims(), &[] as &[usize]);

        let (feat1, label1) = dataset.get(1).unwrap();
        assert_eq!(feat1.shape().dims(), &[2]);
        assert_eq!(label1.shape().dims(), &[] as &[usize]);
    }
}
