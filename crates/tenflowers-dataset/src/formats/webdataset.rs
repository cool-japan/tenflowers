//! WebDataset format support for streaming large datasets
//!
//! WebDataset is a format for storing datasets as TAR files with specific naming conventions.
//! It's designed for efficient streaming of large datasets, especially in distributed training scenarios.
//!
//! The format stores samples as sets of files with the same basename but different extensions:
//! - `sample001.jpg` (image data)
//! - `sample001.txt` (text/metadata)
//! - `sample001.cls` (class label)
//! - etc.

use crate::Dataset;
use scirs2_core::rand_prelude::SliceRandom;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

/// A sample from a WebDataset containing multiple modalities
#[derive(Debug, Clone)]
pub struct WebDatasetSample {
    /// Sample key (basename without extension)
    pub key: String,
    /// Raw data for each extension/modality
    pub data: HashMap<String, Vec<u8>>,
}

impl WebDatasetSample {
    /// Create a new WebDataset sample
    pub fn new(key: String) -> Self {
        Self {
            key,
            data: HashMap::new(),
        }
    }

    /// Add data for a specific extension
    pub fn add_data(&mut self, extension: String, data: Vec<u8>) {
        self.data.insert(extension, data);
    }

    /// Get data for a specific extension
    pub fn get_data(&self, extension: &str) -> Option<&[u8]> {
        self.data.get(extension).map(|v| v.as_slice())
    }

    /// Check if sample has data for a specific extension
    pub fn has_extension(&self, extension: &str) -> bool {
        self.data.contains_key(extension)
    }

    /// Get all available extensions
    pub fn extensions(&self) -> Vec<&String> {
        self.data.keys().collect()
    }
}

/// Configuration for WebDataset loading
#[derive(Debug, Clone)]
pub struct WebDatasetConfig {
    /// Image extension to use as features (e.g., "jpg", "png")
    pub image_extension: String,
    /// Label extension to use (e.g., "cls", "txt")
    pub label_extension: String,
    /// Whether to shuffle samples within each tar file
    pub shuffle: bool,
    /// Maximum number of samples to load (None for all)
    pub max_samples: Option<usize>,
    /// Image decoding configuration
    pub decode_images: bool,
    /// Target image size for resizing (width, height)
    pub target_size: Option<(u32, u32)>,
}

impl Default for WebDatasetConfig {
    fn default() -> Self {
        Self {
            image_extension: "jpg".to_string(),
            label_extension: "cls".to_string(),
            shuffle: false,
            max_samples: None,
            decode_images: true,
            target_size: None,
        }
    }
}

/// WebDataset implementation for streaming large datasets
pub struct WebDataset<T> {
    samples: Vec<WebDatasetSample>,
    config: WebDatasetConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> WebDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a WebDataset from a TAR file
    pub fn from_tar<P: AsRef<Path>>(tar_path: P, config: WebDatasetConfig) -> Result<Self> {
        let file = std::fs::File::open(tar_path.as_ref())
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open tar file: {e}")))?;

        let mut archive = tar::Archive::new(file);
        let mut sample_map: HashMap<String, WebDatasetSample> = HashMap::new();

        // Parse tar entries and group by sample key
        for entry in archive.entries().map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read tar entries: {e}"))
        })? {
            let mut entry = entry.map_err(|e| {
                TensorError::invalid_argument(format!("Failed to read tar entry: {e}"))
            })?;

            let path = entry.path().map_err(|e| {
                TensorError::invalid_argument(format!("Failed to get entry path: {e}"))
            })?;

            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some((basename, extension)) = Self::parse_filename(file_name) {
                    let mut data = Vec::new();
                    entry.read_to_end(&mut data).map_err(|e| {
                        TensorError::invalid_argument(format!("Failed to read entry data: {e}"))
                    })?;

                    let sample = sample_map
                        .entry(basename.clone())
                        .or_insert_with(|| WebDatasetSample::new(basename));
                    sample.add_data(extension, data);
                }
            }
        }

        let mut samples: Vec<_> = sample_map.into_values().collect();

        // Filter samples that have required extensions
        samples.retain(|sample| {
            sample.has_extension(&config.image_extension)
                && sample.has_extension(&config.label_extension)
        });

        // Shuffle if requested
        if config.shuffle {
            let mut rng = scirs2_core::random::rng();
            samples.shuffle(&mut rng);
        }

        // Limit samples if requested
        if let Some(max_samples) = config.max_samples {
            samples.truncate(max_samples);
        }

        Ok(Self {
            samples,
            config,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create WebDataset from multiple TAR files (for sharding)
    pub fn from_tar_files<P: AsRef<Path>>(
        tar_paths: Vec<P>,
        config: WebDatasetConfig,
    ) -> Result<Self> {
        let mut all_samples = Vec::new();

        for tar_path in tar_paths {
            let dataset = Self::from_tar(tar_path, config.clone())?;
            all_samples.extend(dataset.samples);
        }

        // Global shuffle if requested
        if config.shuffle {
            let mut rng = scirs2_core::random::rng();
            all_samples.shuffle(&mut rng);
        }

        // Global limit if requested
        if let Some(max_samples) = config.max_samples {
            all_samples.truncate(max_samples);
        }

        Ok(Self {
            samples: all_samples,
            config,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Parse filename into basename and extension
    fn parse_filename(filename: &str) -> Option<(String, String)> {
        if let Some(dot_pos) = filename.rfind('.') {
            let basename = filename[..dot_pos].to_string();
            let extension = filename[dot_pos + 1..].to_string();
            Some((basename, extension))
        } else {
            None
        }
    }

    /// Convert WebDataset sample to tensor pair
    fn sample_to_tensors(&self, sample: &WebDatasetSample) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: std::str::FromStr + scirs2_core::num_traits::cast::FromPrimitive + Copy,
        T::Err: std::fmt::Debug,
    {
        // Get image data
        let image_data = sample
            .get_data(&self.config.image_extension)
            .ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Sample {} missing image extension {}",
                    sample.key, self.config.image_extension
                ))
            })?;

        // Get label data
        let label_data = sample
            .get_data(&self.config.label_extension)
            .ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Sample {} missing label extension {}",
                    sample.key, self.config.label_extension
                ))
            })?;

        // Process image
        let feature_tensor = if self.config.decode_images {
            self.decode_image_tensor(image_data)?
        } else {
            // Return raw bytes as tensor
            let bytes: Vec<T> = image_data
                .iter()
                .map(|&b| T::from_u8(b).unwrap_or_default())
                .collect();
            Tensor::from_vec(bytes, &[image_data.len()])?
        };

        // Process label
        let label_str = String::from_utf8_lossy(label_data).trim().to_string();
        let label_value = label_str.parse::<T>().map_err(|e| {
            TensorError::invalid_argument(format!("Failed to parse label '{label_str}': {e:?}"))
        })?;
        let label_tensor = Tensor::from_vec(vec![label_value], &[])?;

        Ok((feature_tensor, label_tensor))
    }

    /// Decode image data to tensor
    #[cfg(feature = "images")]
    fn decode_image_tensor(&self, image_data: &[u8]) -> Result<Tensor<T>>
    where
        T: scirs2_core::num_traits::cast::FromPrimitive + Copy,
    {
        let img = image::load_from_memory(image_data)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to decode image: {e}")))?;

        let img = if let Some((width, height)) = self.config.target_size {
            img.resize(width, height, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };

        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();

        let pixels: Vec<T> = rgb_img
            .pixels()
            .flat_map(|p| p.0.iter())
            .map(|&pixel| T::from_u8(pixel).unwrap_or_default())
            .collect();

        let shape = vec![3, height as usize, width as usize]; // CHW format
        Tensor::from_vec(pixels, &shape)
    }

    /// Fallback for when images feature is not enabled
    #[cfg(not(feature = "images"))]
    fn decode_image_tensor(&self, image_data: &[u8]) -> Result<Tensor<T>>
    where
        T: scirs2_core::num_traits::cast::FromPrimitive + Copy,
    {
        // Return raw bytes as tensor when image decoding is not available
        let bytes: Vec<T> = image_data
            .iter()
            .map(|&b| T::from_u8(b).unwrap_or_default())
            .collect();
        Tensor::from_vec(bytes, &[image_data.len()])
    }
}

impl<T> Dataset<T> for WebDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + std::str::FromStr
        + scirs2_core::num_traits::cast::FromPrimitive
        + Copy
        + Send
        + Sync
        + 'static,
    T::Err: std::fmt::Debug,
{
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.samples.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for WebDataset of length {}",
                index,
                self.samples.len()
            )));
        }

        self.sample_to_tensors(&self.samples[index])
    }
}

/// Builder for WebDataset with fluent configuration
pub struct WebDatasetBuilder {
    config: WebDatasetConfig,
}

impl WebDatasetBuilder {
    /// Create a new WebDataset builder
    pub fn new() -> Self {
        Self {
            config: WebDatasetConfig::default(),
        }
    }

    /// Set image extension
    pub fn image_extension<S: Into<String>>(mut self, ext: S) -> Self {
        self.config.image_extension = ext.into();
        self
    }

    /// Set label extension
    pub fn label_extension<S: Into<String>>(mut self, ext: S) -> Self {
        self.config.label_extension = ext.into();
        self
    }

    /// Enable shuffling
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    /// Set maximum number of samples
    pub fn max_samples(mut self, max_samples: usize) -> Self {
        self.config.max_samples = Some(max_samples);
        self
    }

    /// Enable/disable image decoding
    pub fn decode_images(mut self, decode: bool) -> Self {
        self.config.decode_images = decode;
        self
    }

    /// Set target image size for resizing
    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        self.config.target_size = Some((width, height));
        self
    }

    /// Build WebDataset from a single TAR file
    pub fn from_tar<T, P: AsRef<Path>>(self, tar_path: P) -> Result<WebDataset<T>>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        WebDataset::from_tar(tar_path, self.config)
    }

    /// Build WebDataset from multiple TAR files
    pub fn from_tar_files<T, P: AsRef<Path>>(self, tar_paths: Vec<P>) -> Result<WebDataset<T>>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        WebDataset::from_tar_files(tar_paths, self.config)
    }
}

impl Default for WebDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming WebDataset that can handle very large datasets efficiently
pub struct StreamingWebDataset<T> {
    tar_paths: Vec<PathBuf>,
    config: WebDatasetConfig,
    current_dataset: Option<WebDataset<T>>,
    current_tar_index: usize,
    current_sample_index: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StreamingWebDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a new streaming WebDataset
    pub fn new<P: AsRef<Path>>(tar_paths: Vec<P>, config: WebDatasetConfig) -> Self {
        let tar_paths = tar_paths
            .into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        Self {
            tar_paths,
            config,
            current_dataset: None,
            current_tar_index: 0,
            current_sample_index: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Load the next tar file
    fn load_next_tar(&mut self) -> Result<bool>
    where
        T: std::str::FromStr + scirs2_core::num_traits::cast::FromPrimitive + Copy,
        T::Err: std::fmt::Debug,
    {
        if self.current_tar_index >= self.tar_paths.len() {
            return Ok(false);
        }

        let tar_path = &self.tar_paths[self.current_tar_index];
        let dataset = WebDataset::from_tar(tar_path, self.config.clone())?;

        self.current_dataset = Some(dataset);
        self.current_tar_index += 1;
        self.current_sample_index = 0;

        Ok(true)
    }
}

impl<T> Iterator for StreamingWebDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + std::str::FromStr
        + scirs2_core::num_traits::cast::FromPrimitive
        + Copy
        + Send
        + Sync
        + 'static,
    T::Err: std::fmt::Debug,
{
    type Item = Result<(Tensor<T>, Tensor<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we need to load a new tar file
            if self.current_dataset.is_none() {
                match self.load_next_tar() {
                    Ok(true) => continue,
                    Ok(false) => return None, // No more tar files
                    Err(e) => return Some(Err(e)),
                }
            }

            // Try to get next sample from current dataset
            if let Some(ref dataset) = self.current_dataset {
                if self.current_sample_index < dataset.len() {
                    let result = dataset.get(self.current_sample_index);
                    self.current_sample_index += 1;
                    return Some(result);
                } else {
                    // Current dataset exhausted, load next
                    self.current_dataset = None;
                    continue;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webdataset_sample() {
        let mut sample = WebDatasetSample::new("sample001".to_string());

        sample.add_data("jpg".to_string(), vec![1, 2, 3]);
        sample.add_data("txt".to_string(), vec![4, 5, 6]);

        assert!(sample.has_extension("jpg"));
        assert!(sample.has_extension("txt"));
        assert!(!sample.has_extension("png"));

        assert_eq!(sample.get_data("jpg"), Some([1, 2, 3].as_slice()));
        assert_eq!(sample.extensions().len(), 2);
    }

    #[test]
    fn test_parse_filename() {
        assert_eq!(
            WebDataset::<f32>::parse_filename("sample001.jpg"),
            Some(("sample001".to_string(), "jpg".to_string()))
        );

        assert_eq!(
            WebDataset::<f32>::parse_filename("test.image.png"),
            Some(("test.image".to_string(), "png".to_string()))
        );

        assert_eq!(WebDataset::<f32>::parse_filename("noextension"), None);
    }

    #[test]
    fn test_webdataset_config() {
        let config = WebDatasetConfig::default();
        assert_eq!(config.image_extension, "jpg");
        assert_eq!(config.label_extension, "cls");
        assert!(!config.shuffle);
        assert!(config.decode_images);
    }

    #[test]
    fn test_webdataset_builder() {
        let builder = WebDatasetBuilder::new()
            .image_extension("png")
            .label_extension("txt")
            .shuffle(true)
            .max_samples(100)
            .target_size(224, 224);

        assert_eq!(builder.config.image_extension, "png");
        assert_eq!(builder.config.label_extension, "txt");
        assert!(builder.config.shuffle);
        assert_eq!(builder.config.max_samples, Some(100));
        assert_eq!(builder.config.target_size, Some((224, 224)));
    }

    // Note: Full integration tests with actual TAR files would require
    // more complex setup and are better suited for integration test suites
}
