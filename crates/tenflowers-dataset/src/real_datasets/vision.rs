//! Computer vision datasets: MNIST, CIFAR-10, ImageNet

use crate::Dataset;
use bytemuck::{Pod, Zeroable};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "images")]
use image::DynamicImage;

use super::common::{byte_utils, error_utils, validation, DatasetConfig};
use super::download::Downloader;

/// Real MNIST dataset loader with automatic downloading
pub struct RealMnistDataset<T> {
    images: Tensor<T>,
    labels: Tensor<T>,
    num_samples: usize,
    is_train: bool,
}

/// Configuration for MNIST dataset loading
#[derive(Debug, Clone)]
pub struct MnistConfig {
    /// Root directory to store downloaded data
    pub root: PathBuf,
    /// Whether to use training set (true) or test set (false)
    pub train: bool,
    /// Whether to download if not found
    pub download: bool,
}

impl Default for MnistConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("./data"),
            train: true,
            download: true,
        }
    }
}

impl DatasetConfig for MnistConfig {
    fn root(&self) -> &PathBuf {
        &self.root
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn should_download(&self) -> bool {
        self.download
    }
}

impl<T> RealMnistDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::FromPrimitive
        + Send
        + Sync
        + 'static,
{
    /// Create a new MNIST dataset
    pub fn new(config: MnistConfig) -> Result<Self> {
        let mnist_dir = config.root.join("MNIST").join("raw");

        // Create directory if it doesn't exist
        validation::ensure_directory_exists(&mnist_dir, "MNIST")?;

        let (images_file, labels_file) = if config.train {
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        } else {
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        };

        let images_path = mnist_dir.join(images_file);
        let labels_path = mnist_dir.join(labels_file);

        // Download files if they don't exist and download is enabled
        if config.download && (!images_path.exists() || !labels_path.exists()) {
            let downloader = Downloader::new();
            downloader.download_mnist(&mnist_dir, config.train)?;
        }

        // Verify files exist
        validation::validate_file_exists(&images_path, "MNIST images")?;
        validation::validate_file_exists(&labels_path, "MNIST labels")?;

        // Load the data
        let (images, num_samples) = Self::load_images(&images_path)?;
        let labels = Self::load_labels(&labels_path, num_samples)?;

        Ok(Self {
            images,
            labels,
            num_samples,
            is_train: config.train,
        })
    }

    /// Load MNIST images from IDX file format
    fn load_images(path: &Path) -> Result<(Tensor<T>, usize)> {
        let mut file = File::open(path).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to open MNIST images file")
        })?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to read magic number"))?;

        if byte_utils::bytes_to_u32_be(&magic) != 2051 {
            return Err(error_utils::invalid_format_error(
                "MNIST images",
                "Invalid magic number",
            ));
        }

        let mut num_images_bytes = [0u8; 4];
        let mut num_rows_bytes = [0u8; 4];
        let mut num_cols_bytes = [0u8; 4];

        file.read_exact(&mut num_images_bytes).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to read number of images")
        })?;
        file.read_exact(&mut num_rows_bytes)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to read number of rows"))?;
        file.read_exact(&mut num_cols_bytes).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to read number of columns")
        })?;

        let num_images = byte_utils::bytes_to_u32_be(&num_images_bytes) as usize;
        let num_rows = byte_utils::bytes_to_u32_be(&num_rows_bytes) as usize;
        let num_cols = byte_utils::bytes_to_u32_be(&num_cols_bytes) as usize;

        // Read all pixel data
        let total_pixels = num_images * num_rows * num_cols;
        let mut buffer = vec![0u8; total_pixels];
        file.read_exact(&mut buffer)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to read pixel data"))?;

        // Convert to tensor data (normalize to [0, 1])
        let tensor_data: Vec<T> = buffer
            .into_iter()
            .map(|pixel| T::from_f32(pixel as f32 / 255.0).unwrap_or_default())
            .collect();

        let images = Tensor::from_vec(tensor_data, &[num_images, 1, num_rows, num_cols])?;

        Ok((images, num_images))
    }

    /// Load MNIST labels from IDX file format
    fn load_labels(path: &Path, expected_count: usize) -> Result<Tensor<T>> {
        let mut file = File::open(path).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to open MNIST labels file")
        })?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to read magic number"))?;

        if byte_utils::bytes_to_u32_be(&magic) != 2049 {
            return Err(error_utils::invalid_format_error(
                "MNIST labels",
                "Invalid magic number",
            ));
        }

        let mut num_labels_bytes = [0u8; 4];
        file.read_exact(&mut num_labels_bytes).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to read number of labels")
        })?;

        let num_labels = byte_utils::bytes_to_u32_be(&num_labels_bytes) as usize;

        if num_labels != expected_count {
            return Err(error_utils::invalid_format_error(
                "MNIST labels",
                &format!(
                    "Label count {} doesn't match image count {}",
                    num_labels, expected_count
                ),
            ));
        }

        // Read all labels
        let mut buffer = vec![0u8; num_labels];
        file.read_exact(&mut buffer)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to read label data"))?;

        // Convert to tensor data
        let tensor_data: Vec<T> = buffer
            .into_iter()
            .map(|label| T::from_u8(label).unwrap_or_default())
            .collect();

        let labels = Tensor::from_vec(tensor_data, &[num_labels])?;

        Ok(labels)
    }

    /// Get the images tensor
    pub fn images(&self) -> &Tensor<T> {
        &self.images
    }

    /// Get the labels tensor
    pub fn labels(&self) -> &Tensor<T> {
        &self.labels
    }

    /// Get whether this is training set
    pub fn is_train(&self) -> bool {
        self.is_train
    }

    /// Get the number of classes (10 for MNIST)
    pub fn num_classes(&self) -> usize {
        10
    }
}

impl<
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::One
            + scirs2_core::numeric::FromPrimitive
            + Pod
            + Zeroable
            + Send
            + Sync
            + 'static,
    > Dataset<T> for RealMnistDataset<T>
{
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        // Extract single image and label
        let image = self.images.slice(&[index..index + 1, 0..1, 0..28, 0..28])?;
        #[allow(clippy::single_range_in_vec_init)]
        let label = self.labels.slice(&[index..index + 1])?;

        Ok((image, label))
    }
}

/// Real CIFAR-10 dataset loader with automatic downloading
pub struct RealCifar10Dataset<T> {
    images: Tensor<T>,
    labels: Tensor<T>,
    num_samples: usize,
    is_train: bool,
}

/// Configuration for CIFAR-10 dataset loading
#[derive(Debug, Clone)]
pub struct Cifar10Config {
    /// Root directory to store downloaded data
    pub root: PathBuf,
    /// Whether to use training set (true) or test set (false)
    pub train: bool,
    /// Whether to download if not found
    pub download: bool,
}

impl Default for Cifar10Config {
    fn default() -> Self {
        Self {
            root: PathBuf::from("./data"),
            train: true,
            download: true,
        }
    }
}

impl DatasetConfig for Cifar10Config {
    fn root(&self) -> &PathBuf {
        &self.root
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn should_download(&self) -> bool {
        self.download
    }
}

impl<T> RealCifar10Dataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::FromPrimitive
        + Send
        + Sync
        + 'static,
{
    /// Create a new CIFAR-10 dataset
    pub fn new(config: Cifar10Config) -> Result<Self> {
        let cifar_dir = config.root.join("cifar-10-batches-bin");

        // Create directory if it doesn't exist
        validation::ensure_directory_exists(&cifar_dir, "CIFAR-10")?;

        // Download if needed
        if config.download && !Self::check_cifar10_files(&cifar_dir, config.train) {
            let downloader = Downloader::new();
            downloader.download_cifar10(&config.root.join("CIFAR-10"))?;
        }

        // Load the data
        let (images, labels, num_samples) = Self::load_cifar10_data(&cifar_dir, config.train)?;

        Ok(Self {
            images,
            labels,
            num_samples,
            is_train: config.train,
        })
    }

    /// Check if CIFAR-10 files exist
    fn check_cifar10_files(data_dir: &Path, train: bool) -> bool {
        let batch_files = if train {
            vec![
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ]
        } else {
            vec!["test_batch.bin"]
        };

        batch_files.iter().all(|file| data_dir.join(file).exists())
    }

    /// Load CIFAR-10 data from binary files
    fn load_cifar10_data(data_dir: &Path, train: bool) -> Result<(Tensor<T>, Tensor<T>, usize)> {
        let batch_files = if train {
            vec![
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ]
        } else {
            vec!["test_batch.bin"]
        };

        let mut all_images = Vec::new();
        let mut all_labels = Vec::new();
        let mut total_samples = 0;

        for batch_file in batch_files {
            let file_path = data_dir.join(batch_file);

            validation::validate_file_exists(&file_path, "CIFAR-10 batch file")?;

            let mut file = File::open(&file_path).map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to open CIFAR-10 batch file")
            })?;

            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to read CIFAR-10 batch file")
            })?;

            // Each CIFAR-10 sample is 3073 bytes: 1 label + 3072 image bytes (32x32x3)
            const SAMPLE_SIZE: usize = 3073;
            const IMAGE_SIZE: usize = 3072; // 32 * 32 * 3

            if buffer.len() % SAMPLE_SIZE != 0 {
                return Err(error_utils::invalid_format_error(
                    "CIFAR-10",
                    &format!("Invalid file size: {} bytes", buffer.len()),
                ));
            }

            let num_samples = buffer.len() / SAMPLE_SIZE;

            for i in 0..num_samples {
                let start = i * SAMPLE_SIZE;

                // First byte is the label
                let label = buffer[start];
                all_labels.push(T::from_u8(label).unwrap_or_default());

                // Next 3072 bytes are the image (R, G, B channels)
                let image_data = &buffer[start + 1..start + 1 + IMAGE_SIZE];

                for &pixel in image_data {
                    // Normalize pixel values to [0, 1]
                    let normalized_pixel = T::from_f32(pixel as f32 / 255.0).unwrap_or_default();
                    all_images.push(normalized_pixel);
                }
            }

            total_samples += num_samples;
        }

        // Create tensors - images are [N, 3, 32, 32] (channels first)
        let images = Tensor::from_vec(all_images, &[total_samples, 3, 32, 32])?;
        let labels = Tensor::from_vec(all_labels, &[total_samples])?;

        Ok((images, labels, total_samples))
    }

    /// Get CIFAR-10 class names
    pub fn class_names() -> Vec<String> {
        vec![
            "airplane".to_string(),
            "automobile".to_string(),
            "bird".to_string(),
            "cat".to_string(),
            "deer".to_string(),
            "dog".to_string(),
            "frog".to_string(),
            "horse".to_string(),
            "ship".to_string(),
            "truck".to_string(),
        ]
    }

    /// Get the images tensor
    pub fn images(&self) -> &Tensor<T> {
        &self.images
    }

    /// Get the labels tensor
    pub fn labels(&self) -> &Tensor<T> {
        &self.labels
    }

    /// Get whether this is training set
    pub fn is_train(&self) -> bool {
        self.is_train
    }

    /// Get the number of classes (10 for CIFAR-10)
    pub fn num_classes(&self) -> usize {
        10
    }
}

impl<
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::One
            + scirs2_core::numeric::FromPrimitive
            + Pod
            + Zeroable
            + Send
            + Sync
            + 'static,
    > Dataset<T> for RealCifar10Dataset<T>
{
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        // Extract single image and label
        let image = self.images.slice(&[index..index + 1, 0..3, 0..32, 0..32])?;
        #[allow(clippy::single_range_in_vec_init)]
        let label = self.labels.slice(&[index..index + 1])?;

        Ok((image, label))
    }
}

/// Real ImageNet dataset loader (validation set)
pub struct RealImageNetDataset<T> {
    images: Vec<Tensor<T>>,
    labels: Vec<usize>,
    class_names: Vec<String>,
    num_samples: usize,
    is_train: bool,
}

/// Configuration for ImageNet dataset loading
#[derive(Debug, Clone)]
pub struct ImageNetConfig {
    /// Root directory to store downloaded data
    pub root: PathBuf,
    /// Whether to use training set (true) or validation set (false)
    pub train: bool,
    /// Whether to download if not found
    pub download: bool,
    /// Maximum number of samples to load (None for all)
    pub max_samples: Option<usize>,
    /// Target image size for preprocessing
    pub image_size: (u32, u32),
}

impl Default for ImageNetConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("./data"),
            train: false,            // Default to validation set (smaller)
            download: false,         // ImageNet is large, require explicit download
            max_samples: Some(1000), // Limit for testing
            image_size: (224, 224),
        }
    }
}

impl DatasetConfig for ImageNetConfig {
    fn root(&self) -> &PathBuf {
        &self.root
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn should_download(&self) -> bool {
        self.download
    }
}

impl<T> RealImageNetDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::FromPrimitive
        + Send
        + Sync
        + 'static,
{
    /// Create a new ImageNet dataset (validation set only for now)
    pub fn new(config: ImageNetConfig) -> Result<Self> {
        if config.train {
            return Err(TensorError::invalid_argument(
                "ImageNet training set not supported yet. Use train=false for validation set."
                    .to_string(),
            ));
        }

        let imagenet_dir = config.root.join("ImageNet");
        let val_dir = imagenet_dir.join("val");

        validation::ensure_directory_exists(&imagenet_dir, "ImageNet")?;

        // Download if needed
        if config.download && !val_dir.exists() {
            let downloader = Downloader::new();
            downloader.download_imagenet_val(&imagenet_dir)?;
        }

        // Load validation data
        Self::load_validation_data(&config)
    }

    #[cfg(feature = "images")]
    fn load_validation_data(config: &ImageNetConfig) -> Result<Self> {
        let imagenet_dir = config.root.join("ImageNet");
        let val_dir = imagenet_dir.join("val");
        let labels_path = imagenet_dir.join("ILSVRC2012_validation_ground_truth.txt");

        validation::validate_file_exists(&val_dir, "ImageNet validation directory")?;
        validation::validate_file_exists(&labels_path, "ImageNet validation labels")?;

        // Load ground truth labels
        let labels_content = std::fs::read_to_string(&labels_path)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to read ImageNet labels"))?;

        let labels: Vec<usize> = labels_content
            .lines()
            .filter_map(|line| line.trim().parse::<usize>().ok())
            .map(|label| label - 1) // Convert to 0-based indexing
            .collect();

        // Get validation image files
        let mut image_files: Vec<_> = std::fs::read_dir(&val_dir)
            .map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to read validation directory")
            })?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    let path = e.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("JPEG") {
                        Some(path)
                    } else {
                        None
                    }
                })
            })
            .collect();

        image_files.sort();

        // Limit number of samples if specified
        let max_samples = config
            .max_samples
            .unwrap_or(image_files.len())
            .min(labels.len());
        image_files.truncate(max_samples);

        // Load and preprocess images
        let mut images = Vec::new();
        let mut processed_labels = Vec::new();

        for (i, image_path) in image_files.iter().enumerate() {
            if i >= labels.len() {
                break;
            }

            match image::open(image_path) {
                Ok(img) => {
                    let processed_img = Self::preprocess_image(img, config)?;
                    images.push(processed_img);
                    processed_labels.push(labels[i]);
                }
                Err(e) => {
                    println!(
                        "Warning: Failed to load image {}: {}",
                        image_path.display(),
                        e
                    );
                    continue;
                }
            }
        }

        let num_samples = images.len();

        // Generate class names (simplified - in practice you'd load from metadata)
        let class_names: Vec<String> = (0..1000).map(|i| format!("class_{:04}", i)).collect();

        Ok(Self {
            images,
            labels: processed_labels,
            class_names,
            num_samples,
            is_train: false,
        })
    }

    #[cfg(not(feature = "images"))]
    fn load_validation_data(_config: &ImageNetConfig) -> Result<Self> {
        Err(TensorError::invalid_argument(
            "Images feature not enabled. Please enable the 'images' feature to load ImageNet data."
                .to_string(),
        ))
    }

    #[cfg(feature = "images")]
    fn preprocess_image(img: DynamicImage, config: &ImageNetConfig) -> Result<Tensor<T>> {
        // Resize image
        let (target_width, target_height) = config.image_size;
        let final_img = if img.width() != target_width || img.height() != target_height {
            img.resize(
                target_width,
                target_height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            img
        };

        // Convert to RGB if needed
        let rgb_img = final_img.to_rgb8();

        // Convert to tensor format (C, H, W) with normalization
        let (width, height) = rgb_img.dimensions();
        let mut tensor_data = Vec::with_capacity(3 * width as usize * height as usize);

        // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        let mean = [0.485f32, 0.456f32, 0.406f32];
        let std = [0.229f32, 0.224f32, 0.225f32];

        // Process each channel separately (CHW format)
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb_img.get_pixel(x, y);
                    let value = pixel[channel] as f32 / 255.0; // Normalize to [0, 1]
                    let normalized = (value - mean[channel]) / std[channel]; // ImageNet normalization
                    tensor_data.push(T::from_f32(normalized).unwrap_or_default());
                }
            }
        }

        Tensor::from_vec(tensor_data, &[3, height as usize, width as usize])
    }

    #[cfg(not(feature = "images"))]
    fn preprocess_image(_img: DynamicImage, _config: &ImageNetConfig) -> Result<Tensor<T>> {
        Err(TensorError::invalid_argument(
            "Images feature not enabled. Please enable the 'images' feature to preprocess ImageNet images.".to_string()
        ))
    }

    /// Get all images
    pub fn all_images(&self) -> &[Tensor<T>] {
        &self.images
    }

    /// Get all labels
    pub fn all_labels(&self) -> &[usize] {
        &self.labels
    }

    /// Get class names
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    /// Get whether this is training set
    pub fn is_train(&self) -> bool {
        self.is_train
    }

    /// Get the number of classes (1000 for ImageNet)
    pub fn num_classes(&self) -> usize {
        1000
    }
}

impl<
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::FromPrimitive
            + Send
            + Sync
            + 'static,
    > Dataset<T> for RealImageNetDataset<T>
{
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        let image = self.images[index].clone();
        let label_value = self.labels[index];

        // Convert label to scalar tensor
        let label = Tensor::from_vec(vec![T::from_usize(label_value).unwrap_or_default()], &[])?;

        Ok((image, label))
    }
}

/// Builder patterns for easy dataset construction
pub struct RealMnistBuilder {
    config: MnistConfig,
}

impl RealMnistBuilder {
    pub fn new() -> Self {
        Self {
            config: MnistConfig::default(),
        }
    }

    pub fn root<P: AsRef<Path>>(mut self, root: P) -> Self {
        self.config.root = root.as_ref().to_path_buf();
        self
    }

    pub fn train(mut self, train: bool) -> Self {
        self.config.train = train;
        self
    }

    pub fn download(mut self, download: bool) -> Self {
        self.config.download = download;
        self
    }

    pub fn build<T>(self) -> Result<RealMnistDataset<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        RealMnistDataset::new(self.config)
    }
}

impl Default for RealMnistBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RealCifar10Builder {
    config: Cifar10Config,
}

impl RealCifar10Builder {
    pub fn new() -> Self {
        Self {
            config: Cifar10Config::default(),
        }
    }

    pub fn root<P: AsRef<Path>>(mut self, root: P) -> Self {
        self.config.root = root.as_ref().to_path_buf();
        self
    }

    pub fn train(mut self, train: bool) -> Self {
        self.config.train = train;
        self
    }

    pub fn download(mut self, download: bool) -> Self {
        self.config.download = download;
        self
    }

    pub fn build<T>(self) -> Result<RealCifar10Dataset<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        RealCifar10Dataset::new(self.config)
    }
}

impl Default for RealCifar10Builder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RealImageNetBuilder {
    config: ImageNetConfig,
}

impl RealImageNetBuilder {
    pub fn new() -> Self {
        Self {
            config: ImageNetConfig::default(),
        }
    }

    pub fn root<P: AsRef<Path>>(mut self, root: P) -> Self {
        self.config.root = root.as_ref().to_path_buf();
        self
    }

    pub fn train(mut self, train: bool) -> Self {
        self.config.train = train;
        self
    }

    pub fn download(mut self, download: bool) -> Self {
        self.config.download = download;
        self
    }

    pub fn max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.config.max_samples = max_samples;
        self
    }

    pub fn image_size(mut self, width: u32, height: u32) -> Self {
        self.config.image_size = (width, height);
        self
    }

    pub fn build<T>(self) -> Result<RealImageNetDataset<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        RealImageNetDataset::new(self.config)
    }
}

impl Default for RealImageNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_mnist_builder() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RealMnistBuilder::new()
            .root(temp_dir.path())
            .train(true)
            .download(false); // Don't actually download in tests

        // This should fail since we don't have files and download is disabled
        let result = builder.build::<f32>();
        assert!(result.is_err());
    }

    #[test]
    fn test_mnist_config_default() {
        let config = MnistConfig::default();
        assert_eq!(config.root, PathBuf::from("./data"));
        assert!(config.train);
        assert!(config.download);
    }

    #[test]
    fn test_cifar10_builder() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RealCifar10Builder::new()
            .root(temp_dir.path())
            .train(true)
            .download(false); // Don't actually download in tests

        // This should fail since we don't have files and download is disabled
        let result = builder.build::<f32>();
        assert!(result.is_err());
    }

    #[test]
    fn test_cifar10_config_default() {
        let config = Cifar10Config::default();
        assert_eq!(config.root, PathBuf::from("./data"));
        assert!(config.train);
        assert!(config.download);
    }

    #[test]
    fn test_cifar10_class_names() {
        let class_names = RealCifar10Dataset::<f32>::class_names();
        assert_eq!(class_names.len(), 10);
        assert_eq!(class_names[0], "airplane");
        assert_eq!(class_names[9], "truck");
    }
}
