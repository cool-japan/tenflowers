//! Image folder dataset implementation
//!
//! This module provides functionality for loading images from folder structures
//! where each subdirectory represents a different class.

use crate::formats::common::{MissingValueStrategy, NamingPattern};
use crate::{Dataset, Transform, TransformedDataset};
use std::collections::HashMap;
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for ImageFolderDataset
#[derive(Debug, Clone)]
pub struct ImageFolderConfig {
    /// Valid image extensions
    pub extensions: Vec<String>,
    /// Whether to follow symbolic links
    pub follow_symlinks: bool,
    /// How to handle missing or corrupted images
    pub missing_strategy: MissingValueStrategy,
    /// How to extract class information
    pub naming_pattern: NamingPattern,
    /// Whether to sort files alphabetically
    pub sort_files: bool,
    /// Maximum number of images to load (None for all)
    pub max_images: Option<usize>,
    /// Resize images to (height, width) if specified
    pub resize: Option<(usize, usize)>,
    /// Whether to normalize pixel values to [0, 1] range (default: true)
    pub normalize: Option<bool>,
}

impl Default for ImageFolderConfig {
    fn default() -> Self {
        Self {
            extensions: vec![
                "jpg".to_string(),
                "jpeg".to_string(),
                "png".to_string(),
                "bmp".to_string(),
                "gif".to_string(),
                "tiff".to_string(),
                "webp".to_string(),
            ],
            follow_symlinks: false,
            missing_strategy: MissingValueStrategy::default(),
            naming_pattern: NamingPattern::default(),
            sort_files: true,
            max_images: None,
            resize: None,
            normalize: Some(true),
        }
    }
}

/// Dataset for loading images from a folder structure
pub struct ImageFolderDataset<T> {
    /// Root directory path
    root_path: PathBuf,
    /// List of image paths with their class indices
    image_paths: Vec<(PathBuf, usize)>,
    /// Mapping from class names to indices
    class_to_idx: HashMap<String, usize>,
    /// Mapping from indices to class names
    idx_to_class: HashMap<usize, String>,
    /// Dataset configuration
    config: ImageFolderConfig,
    /// Optional transform to apply to images
    transform: Option<Box<dyn Transform<T> + Send + Sync>>,
    _phantom: PhantomData<T>,
}

impl<T> ImageFolderDataset<T> {
    /// Create a new ImageFolderDataset from a root directory
    pub fn new<P: AsRef<Path>>(root_path: P) -> Result<Self> {
        Self::with_config(root_path, ImageFolderConfig::default())
    }

    /// Create a new ImageFolderDataset with custom configuration
    pub fn with_config<P: AsRef<Path>>(root_path: P, config: ImageFolderConfig) -> Result<Self> {
        let root_path = root_path.as_ref().to_path_buf();

        if !root_path.exists() {
            return Err(TensorError::invalid_argument(format!(
                "Root directory does not exist: {}",
                root_path.display()
            )));
        }

        if !root_path.is_dir() {
            return Err(TensorError::invalid_argument(format!(
                "Root path is not a directory: {}",
                root_path.display()
            )));
        }

        let mut dataset = Self {
            root_path,
            image_paths: Vec::new(),
            class_to_idx: HashMap::new(),
            idx_to_class: HashMap::new(),
            config,
            transform: None,
            _phantom: PhantomData,
        };

        dataset.scan_directory()?;
        Ok(dataset)
    }

    /// Add a transform to be applied to images
    pub fn with_transform<Tr>(mut self, transform: Tr) -> Self
    where
        Tr: Transform<T> + Send + Sync + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Get the class to index mapping
    pub fn class_to_idx(&self) -> &HashMap<String, usize> {
        &self.class_to_idx
    }

    /// Get the index to class mapping
    pub fn idx_to_class(&self) -> &HashMap<usize, String> {
        &self.idx_to_class
    }

    /// Get the number of classes
    pub fn num_classes(&self) -> usize {
        self.class_to_idx.len()
    }

    /// Get the list of class names
    pub fn classes(&self) -> Vec<String> {
        let mut classes: Vec<_> = self.class_to_idx.keys().cloned().collect();
        classes.sort();
        classes
    }

    /// Get the image path for a given index
    pub fn image_path(&self, index: usize) -> Option<&PathBuf> {
        self.image_paths.get(index).map(|(path, _)| path)
    }

    /// Get the class index for a given dataset index
    pub fn class_index(&self, index: usize) -> Option<usize> {
        self.image_paths.get(index).map(|(_, class_idx)| *class_idx)
    }

    /// Get the class name for a given dataset index
    pub fn class_name(&self, index: usize) -> Option<&String> {
        self.class_index(index)
            .and_then(|class_idx| self.idx_to_class.get(&class_idx))
    }

    /// Scan the directory structure to build the image list
    fn scan_directory(&mut self) -> Result<()> {
        match self.config.naming_pattern {
            NamingPattern::DirectoryAsClass => {
                self.scan_directory_structure()?;
            }
            NamingPattern::FilenamePrefix(_) | NamingPattern::FilenameSuffix(_) => {
                self.scan_flat_structure()?;
            }
            NamingPattern::CustomMapping(ref mapping) => {
                let mapping_clone = mapping.clone();
                self.scan_with_custom_mapping(&mapping_clone)?;
            }
        }

        if self.config.sort_files {
            self.image_paths.sort_by(|a, b| a.0.cmp(&b.0));
        }

        if let Some(max_images) = self.config.max_images {
            self.image_paths.truncate(max_images);
        }

        Ok(())
    }

    /// Scan directory structure where each subdirectory is a class
    fn scan_directory_structure(&mut self) -> Result<()> {
        let entries = fs::read_dir(&self.root_path).map_err(|e| {
            TensorError::invalid_argument(format!(
                "Cannot read directory {}: {}",
                self.root_path.display(),
                e
            ))
        })?;

        let mut class_dirs = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                TensorError::invalid_argument(format!("Error reading directory entry: {e}"))
            })?;

            let path = entry.path();
            if path.is_dir() || (self.config.follow_symlinks && path.is_symlink()) {
                class_dirs.push(path);
            }
        }

        // Sort directories for consistent class ordering
        class_dirs.sort();

        // Build class mappings
        for (class_idx, class_dir) in class_dirs.iter().enumerate() {
            let class_name = class_dir
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    TensorError::invalid_argument(format!(
                        "Invalid directory name: {}",
                        class_dir.display()
                    ))
                })?
                .to_string();

            self.class_to_idx.insert(class_name.clone(), class_idx);
            self.idx_to_class.insert(class_idx, class_name);

            // Scan images in this class directory
            self.scan_images_in_directory(class_dir, class_idx)?;
        }

        Ok(())
    }

    /// Scan flat structure where class is extracted from filename
    fn scan_flat_structure(&mut self) -> Result<()> {
        let mut class_names = std::collections::HashSet::new();
        let mut temp_paths = Vec::new();

        // First pass: collect all images and extract class names
        self.collect_images_recursive(&self.root_path, &mut temp_paths)?;

        for path in &temp_paths {
            if let Some(class_name) = self.extract_class_from_filename(path)? {
                class_names.insert(class_name);
            }
        }

        // Build class mappings
        let mut sorted_classes: Vec<_> = class_names.into_iter().collect();
        sorted_classes.sort();

        for (class_idx, class_name) in sorted_classes.iter().enumerate() {
            self.class_to_idx.insert(class_name.clone(), class_idx);
            self.idx_to_class.insert(class_idx, class_name.clone());
        }

        // Second pass: build image paths with class indices
        for path in temp_paths {
            if let Some(class_name) = self.extract_class_from_filename(&path)? {
                if let Some(&class_idx) = self.class_to_idx.get(&class_name) {
                    self.image_paths.push((path, class_idx));
                }
            }
        }

        Ok(())
    }

    /// Scan with custom class mapping
    fn scan_with_custom_mapping(&mut self, mapping: &HashMap<String, usize>) -> Result<()> {
        self.class_to_idx = mapping.clone();
        for (class_name, &class_idx) in mapping {
            self.idx_to_class.insert(class_idx, class_name.clone());
        }

        let mut temp_paths = Vec::new();
        self.collect_images_recursive(&self.root_path, &mut temp_paths)?;

        for path in temp_paths {
            // For custom mapping, try to find the class by checking if any key matches
            // This is a simple implementation - could be extended for more complex patterns
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                for (pattern, &class_idx) in mapping {
                    if filename.contains(pattern) {
                        self.image_paths.push((path.clone(), class_idx));
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Scan images in a specific directory
    fn scan_images_in_directory(&mut self, dir_path: &Path, class_idx: usize) -> Result<()> {
        let mut image_paths = Vec::new();
        self.collect_images_recursive(dir_path, &mut image_paths)?;

        for path in image_paths {
            self.image_paths.push((path, class_idx));
        }

        Ok(())
    }

    /// Recursively collect image files from a directory
    fn collect_images_recursive(
        &self,
        dir_path: &Path,
        image_paths: &mut Vec<PathBuf>,
    ) -> Result<()> {
        let entries = fs::read_dir(dir_path).map_err(|e| {
            TensorError::invalid_argument(format!(
                "Cannot read directory {}: {}",
                dir_path.display(),
                e
            ))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                TensorError::invalid_argument(format!("Error reading directory entry: {e}"))
            })?;

            let path = entry.path();
            if path.is_file() || (self.config.follow_symlinks && path.is_symlink()) {
                if self.is_valid_image(&path) {
                    image_paths.push(path);
                }
            } else if path.is_dir() {
                // Recursively scan subdirectories (only for flat structure scanning)
                match self.config.naming_pattern {
                    NamingPattern::FilenamePrefix(_)
                    | NamingPattern::FilenameSuffix(_)
                    | NamingPattern::CustomMapping(_) => {
                        self.collect_images_recursive(&path, image_paths)?;
                    }
                    _ => {} // Don't recurse for directory-based structure
                }
            }
        }

        Ok(())
    }

    /// Check if a file is a valid image based on extension
    fn is_valid_image(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            let extension = extension.to_lowercase();
            self.config.extensions.contains(&extension)
        } else {
            false
        }
    }

    /// Extract class name from filename based on naming pattern
    fn extract_class_from_filename(&self, path: &Path) -> Result<Option<String>> {
        let filename = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                TensorError::invalid_argument(format!("Invalid filename: {}", path.display()))
            })?;

        match &self.config.naming_pattern {
            NamingPattern::FilenamePrefix(separator) => {
                if let Some(pos) = filename.find(separator) {
                    Ok(Some(filename[..pos].to_string()))
                } else {
                    Ok(None)
                }
            }
            NamingPattern::FilenameSuffix(separator) => {
                if let Some(pos) = filename.rfind(separator) {
                    Ok(Some(filename[pos + separator.len()..].to_string()))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Load an image from a file path
    #[cfg(feature = "images")]
    fn load_image(&self, path: &Path) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::ScientificNumber
            + Send
            + Sync
            + 'static,
    {
        use image::GenericImageView;

        // Check if file exists
        if !path.exists() {
            match self.config.missing_strategy {
                MissingValueStrategy::SkipRow => {
                    return Err(TensorError::invalid_argument(format!(
                        "Image file not found: {}",
                        path.display()
                    )));
                }
                _ => {
                    // Return a zero tensor if using fill strategies
                    let shape = match self.config.resize {
                        Some((h, w)) => vec![3, h, w],
                        None => vec![3, 224, 224], // Default size
                    };
                    let size: usize = shape.iter().product();
                    return Tensor::from_vec(vec![T::default(); size], &shape);
                }
            }
        }

        // Load the image
        let img = image::open(path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to load image {}: {}", path.display(), e))
        })?;

        // Convert to RGB8 first
        let rgb_img = img.to_rgb8();

        // Resize if requested
        let rgb_img = if let Some((height, width)) = self.config.resize {
            image::imageops::resize(
                &rgb_img,
                width as u32,
                height as u32,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            rgb_img
        };
        let (width, height) = rgb_img.dimensions();

        // Convert image data to tensor
        // Image data is in format [height, width, channels]
        // We want [channels, height, width] (CHW format)
        let mut data: Vec<T> = Vec::with_capacity((3 * width * height) as usize);

        // Normalize to [0, 1] range by default
        let normalize = self.config.normalize.unwrap_or(true);

        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb_img.get_pixel(x, y);
                    let value = pixel[c as usize];
                    let normalized_value = if normalize {
                        value as f64 / 255.0
                    } else {
                        value as f64
                    };
                    use scirs2_core::numeric::ScientificNumber;
                    let value_t = T::from_f64(normalized_value).ok_or_else(|| {
                        TensorError::invalid_argument(
                            "Failed to convert image data to target type".to_string(),
                        )
                    })?;
                    data.push(value_t);
                }
            }
        }

        Tensor::from_vec(data, &[3, height as usize, width as usize])
    }

    #[cfg(not(feature = "images"))]
    fn load_image(&self, _path: &Path) -> Result<Tensor<T>>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        Err(TensorError::InvalidOperation {
            operation: "load_image".to_string(),
            reason: "Image loading requires 'images' feature to be enabled".to_string(),
            context: None,
        })
    }
}

impl<T> Dataset<T> for ImageFolderDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + Send
        + Sync
        + 'static
        + scirs2_core::numeric::ScientificNumber,
{
    fn len(&self) -> usize {
        self.image_paths.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.len()
            )));
        }

        let (image_path, class_idx) = &self.image_paths[index];

        // Load the image
        let image = self.load_image(image_path)?;

        // Create class label tensor
        let label_data = if let Some(class_val) = scirs2_core::num_traits::NumCast::from(*class_idx)
        {
            vec![class_val]
        } else {
            vec![T::default()]
        };
        let label = Tensor::from_vec(label_data, &[])?;

        let mut sample = (image, label);

        // Apply transform if present
        if let Some(ref transform) = self.transform {
            sample = transform.apply(sample)?;
        }

        Ok(sample)
    }
}

/// Builder for ImageFolderDataset
pub struct ImageFolderDatasetBuilder<T> {
    root_path: Option<PathBuf>,
    config: ImageFolderConfig,
    _phantom: PhantomData<T>,
}

impl<T> ImageFolderDatasetBuilder<T> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            root_path: None,
            config: ImageFolderConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the root path
    pub fn root_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.root_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set valid image extensions
    pub fn extensions(mut self, extensions: Vec<String>) -> Self {
        self.config.extensions = extensions;
        self
    }

    /// Set whether to follow symbolic links
    pub fn follow_symlinks(mut self, follow: bool) -> Self {
        self.config.follow_symlinks = follow;
        self
    }

    /// Set the missing value strategy
    pub fn missing_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.config.missing_strategy = strategy;
        self
    }

    /// Set the naming pattern
    pub fn naming_pattern(mut self, pattern: NamingPattern) -> Self {
        self.config.naming_pattern = pattern;
        self
    }

    /// Set whether to sort files
    pub fn sort_files(mut self, sort: bool) -> Self {
        self.config.sort_files = sort;
        self
    }

    /// Set maximum number of images to load
    pub fn max_images(mut self, max: usize) -> Self {
        self.config.max_images = Some(max);
        self
    }

    /// Build the dataset
    pub fn build(self) -> Result<ImageFolderDataset<T>> {
        let root_path = self
            .root_path
            .ok_or_else(|| TensorError::invalid_argument("Root path is required".to_string()))?;

        ImageFolderDataset::with_config(root_path, self.config)
    }
}

impl<T> Default for ImageFolderDatasetBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create an ImageFolderDataset with transforms
pub fn image_folder_dataset_with_transform<T, Tr>(
    root_path: impl AsRef<Path>,
    transform: Tr,
) -> Result<TransformedDataset<T, ImageFolderDataset<T>, Tr>>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + Send
        + Sync
        + 'static
        + scirs2_core::numeric::ScientificNumber,
    Tr: Transform<T> + 'static,
{
    let dataset = ImageFolderDataset::new(root_path)?;
    Ok(TransformedDataset::new(dataset, transform))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_dataset_structure() -> (TempDir, PathBuf) {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path().to_path_buf();

        // Create class directories
        let class_a_dir = root_path.join("class_a");
        let class_b_dir = root_path.join("class_b");
        fs::create_dir_all(&class_a_dir).unwrap();
        fs::create_dir_all(&class_b_dir).unwrap();

        // Create dummy image files
        let files = [
            class_a_dir.join("image1.jpg"),
            class_a_dir.join("image2.png"),
            class_b_dir.join("image3.jpg"),
            class_b_dir.join("image4.bmp"),
        ];

        for file_path in &files {
            let mut file = fs::File::create(file_path).unwrap();
            file.write_all(b"dummy image data").unwrap();
        }

        (temp_dir, root_path)
    }

    #[test]
    fn test_image_folder_dataset_creation() {
        let (_temp_dir, root_path) = create_test_dataset_structure();
        let dataset = ImageFolderDataset::<f32>::new(&root_path).unwrap();

        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.num_classes(), 2);

        let classes = dataset.classes();
        assert_eq!(classes, vec!["class_a".to_string(), "class_b".to_string()]);
    }

    #[test]
    fn test_image_folder_dataset_builder() {
        let (_temp_dir, root_path) = create_test_dataset_structure();

        let dataset = ImageFolderDatasetBuilder::<f32>::new()
            .root_path(&root_path)
            .extensions(vec!["jpg".to_string(), "png".to_string()])
            .max_images(2)
            .build()
            .unwrap();

        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_naming_pattern_default() {
        let pattern = NamingPattern::default();
        matches!(pattern, NamingPattern::DirectoryAsClass);
    }

    #[test]
    fn test_config_default() {
        let config = ImageFolderConfig::default();
        assert!(config.extensions.contains(&"jpg".to_string()));
        assert!(config.extensions.contains(&"png".to_string()));
        assert!(!config.follow_symlinks);
        assert!(config.sort_files);
        assert!(config.max_images.is_none());
    }

    #[test]
    fn test_invalid_root_path() {
        let result = ImageFolderDataset::<f32>::new("/nonexistent/path");
        assert!(result.is_err());
    }
}
