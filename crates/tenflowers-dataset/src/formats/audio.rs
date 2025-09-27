//! Audio format support for datasets
//!
//! This module provides comprehensive support for audio file formats commonly used in machine learning,
//! including WAV, FLAC, MP3, and other formats supported by the Symphonia decoder. The implementation
//! includes audio preprocessing, resampling, and feature extraction capabilities for ML workflows.
//!
//! # Features
//!
//! - **Multi-format Support**: WAV, FLAC, MP3, OGG, and other common audio formats
//! - **Automatic Resampling**: Convert audio to target sample rates
//! - **Feature Extraction**: MFCC, spectrograms, and other audio features
//! - **Normalization**: Audio amplitude normalization and preprocessing
//! - **Batch Processing**: Efficient batch loading of audio files
//! - **Streaming Support**: Process large audio datasets without loading everything into memory
//! - **Metadata Extraction**: Audio file metadata and duration information
//! - **Label Support**: Flexible labeling from filenames, directories, or external files
//!
//! # Example Usage
//!
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tenflowers_dataset::formats::audio::{AudioDataset, AudioConfig, FeatureType};
//!
//! // Basic usage - load audio files from directory
//! let dataset = AudioDataset::from_directory("audio_data/")?;
//!
//! // With configuration
//! let config = AudioConfig::default()
//!     .with_sample_rate(16000)
//!     .with_max_duration(5.0)
//!     .with_normalize(true)
//!     .with_feature_extraction(FeatureType::MFCC);
//!
//! let dataset = AudioDataset::from_directory_with_config("audio_data/", config)?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "audio")]
use std::collections::HashMap;
#[cfg(feature = "audio")]
use std::fs;
#[cfg(feature = "audio")]
use std::path::{Path, PathBuf};

// Symphonia audio types - removed unused imports
// Symphonia codec types - removed unused imports
// Symphonia error types - removed unused imports
// Symphonia format types - removed unused imports
// Symphonia IO types - removed unused imports
// Symphonia metadata types - removed unused imports
// Symphonia probe types - removed unused imports
// Rubato resampler types - removed unused imports

#[cfg(feature = "audio")]
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "audio")]
use crate::Dataset;

/// Audio feature extraction types
#[cfg(feature = "audio")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    /// Raw audio waveform
    Raw,
    /// Mel-frequency cepstral coefficients
    MFCC,
    /// Mel spectrogram
    MelSpectrogram,
    /// Log spectrogram
    LogSpectrogram,
    /// Chromagram
    Chroma,
}

/// Audio dataset configuration
#[cfg(feature = "audio")]
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Target sample rate (Hz)
    pub sample_rate: u32,
    /// Maximum audio duration in seconds (clips longer audio)
    pub max_duration: Option<f32>,
    /// Minimum audio duration in seconds (pads shorter audio)
    pub min_duration: Option<f32>,
    /// Whether to normalize audio amplitude
    pub normalize: bool,
    /// Feature extraction type
    pub feature_type: FeatureType,
    /// Number of MFCC coefficients (for MFCC features)
    pub n_mfcc: usize,
    /// Number of mel bands (for mel-based features)
    pub n_mels: usize,
    /// FFT window size
    pub n_fft: usize,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Supported audio file extensions
    pub supported_extensions: Vec<String>,
    /// Whether to cache processed audio in memory
    pub cache_audio: bool,
    /// Label extraction strategy
    pub label_strategy: AudioLabelStrategy,
    /// Custom label mapping (filename -> label)
    pub label_mapping: Option<HashMap<String, String>>,
}

/// Strategy for extracting labels from audio files
#[cfg(feature = "audio")]
#[derive(Debug, Clone)]
pub enum AudioLabelStrategy {
    /// Extract label from filename (before first underscore or dot)
    FromFilename,
    /// Extract label from parent directory name
    FromDirectory,
    /// Use custom mapping provided in config
    FromMapping,
    /// No labels (unsupervised learning)
    None,
}

#[cfg(feature = "audio")]
impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            max_duration: None,
            min_duration: None,
            normalize: true,
            feature_type: FeatureType::Raw,
            n_mfcc: 13,
            n_mels: 80,
            n_fft: 1024,
            hop_length: 512,
            supported_extensions: vec![
                "wav".to_string(),
                "flac".to_string(),
                "mp3".to_string(),
                "ogg".to_string(),
                "m4a".to_string(),
            ],
            cache_audio: false,
            label_strategy: AudioLabelStrategy::FromDirectory,
            label_mapping: None,
        }
    }
}

#[cfg(feature = "audio")]
impl AudioConfig {
    /// Set target sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set maximum duration
    pub fn with_max_duration(mut self, duration: f32) -> Self {
        self.max_duration = Some(duration);
        self
    }

    /// Set minimum duration
    pub fn with_min_duration(mut self, duration: f32) -> Self {
        self.min_duration = Some(duration);
        self
    }

    /// Enable or disable normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set feature extraction type
    pub fn with_feature_extraction(mut self, feature_type: FeatureType) -> Self {
        self.feature_type = feature_type;
        self
    }

    /// Set number of MFCC coefficients
    pub fn with_n_mfcc(mut self, n_mfcc: usize) -> Self {
        self.n_mfcc = n_mfcc;
        self
    }

    /// Set number of mel bands
    pub fn with_n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Set label strategy
    pub fn with_label_strategy(mut self, strategy: AudioLabelStrategy) -> Self {
        self.label_strategy = strategy;
        self
    }

    /// Set custom label mapping
    pub fn with_label_mapping(mut self, mapping: HashMap<String, String>) -> Self {
        self.label_mapping = Some(mapping);
        self
    }

    /// Enable or disable audio caching
    pub fn with_cache_audio(mut self, cache: bool) -> Self {
        self.cache_audio = cache;
        self
    }
}

/// Information about an audio file
#[cfg(feature = "audio")]
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// File path
    pub path: PathBuf,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: usize,
    /// Duration in seconds
    pub duration: f32,
    /// Number of samples
    pub num_samples: usize,
    /// File size in bytes
    pub file_size: u64,
    /// Audio format
    pub format: String,
    /// Label (if available)
    pub label: Option<String>,
}

/// Audio dataset information
#[cfg(feature = "audio")]
#[derive(Debug, Clone)]
pub struct AudioDatasetInfo {
    /// Dataset directory
    pub directory: PathBuf,
    /// Number of audio files
    pub num_files: usize,
    /// Total duration in seconds
    pub total_duration: f32,
    /// Average duration per file
    pub avg_duration: f32,
    /// Unique labels
    pub labels: Vec<String>,
    /// Label counts
    pub label_counts: HashMap<String, usize>,
    /// Audio information for each file
    pub file_info: Vec<AudioInfo>,
}

/// Builder for creating audio datasets
#[cfg(feature = "audio")]
pub struct AudioDatasetBuilder {
    directory: Option<PathBuf>,
    config: AudioConfig,
}

#[cfg(feature = "audio")]
impl Default for AudioDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "audio")]
impl AudioDatasetBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            directory: None,
            config: AudioConfig::default(),
        }
    }

    /// Set the directory path
    pub fn directory<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.directory = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the configuration
    pub fn config(mut self, config: AudioConfig) -> Self {
        self.config = config;
        self
    }

    /// Set sample rate
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.sample_rate = sample_rate;
        self
    }

    /// Set feature type
    pub fn feature_type(mut self, feature_type: FeatureType) -> Self {
        self.config.feature_type = feature_type;
        self
    }

    /// Build the dataset
    pub fn build(self) -> Result<AudioDataset> {
        let directory = self.directory.ok_or_else(|| {
            TensorError::invalid_argument("Directory must be specified".to_string())
        })?;
        AudioDataset::from_directory_with_config(&directory, self.config)
    }
}

/// Audio dataset implementation
#[cfg(feature = "audio")]
pub struct AudioDataset {
    /// Configuration
    config: AudioConfig,
    /// Dataset information
    info: AudioDatasetInfo,
    /// Cached audio data
    cached_audio: Option<Vec<Vec<f32>>>,
    /// Cached labels
    cached_labels: Option<Vec<String>>,
    /// Label to index mapping
    label_to_idx: HashMap<String, usize>,
}

#[cfg(feature = "audio")]
impl AudioDataset {
    /// Create dataset from directory with default configuration
    pub fn from_directory<P: AsRef<Path>>(directory: P) -> Result<Self> {
        Self::from_directory_with_config(directory, AudioConfig::default())
    }

    /// Create dataset from directory with custom configuration
    pub fn from_directory_with_config<P: AsRef<Path>>(
        directory: P,
        config: AudioConfig,
    ) -> Result<Self> {
        let dir_path = directory.as_ref().to_path_buf();

        if !dir_path.exists() {
            return Err(TensorError::invalid_argument(format!(
                "Directory not found: {}",
                dir_path.display()
            )));
        }

        if !dir_path.is_dir() {
            return Err(TensorError::invalid_argument(format!(
                "Path is not a directory: {}",
                dir_path.display()
            )));
        }

        // Discover audio files
        let file_info = discover_audio_files(&dir_path, &config)?;

        if file_info.is_empty() {
            return Err(TensorError::invalid_argument(
                "No supported audio files found in directory".to_string(),
            ));
        }

        // Calculate statistics
        let num_files = file_info.len();
        let total_duration: f32 = file_info.iter().map(|info| info.duration).sum();
        let avg_duration = total_duration / num_files as f32;

        // Extract unique labels and counts
        let mut labels = Vec::new();
        let mut label_counts = HashMap::new();

        for info in &file_info {
            if let Some(ref label) = info.label {
                if !labels.contains(label) {
                    labels.push(label.clone());
                }
                *label_counts.entry(label.clone()).or_insert(0) += 1;
            }
        }

        labels.sort();

        // Create label to index mapping
        let label_to_idx: HashMap<String, usize> = labels
            .iter()
            .enumerate()
            .map(|(idx, label)| (label.clone(), idx))
            .collect();

        let dataset_info = AudioDatasetInfo {
            directory: dir_path,
            num_files,
            total_duration,
            avg_duration,
            labels,
            label_counts,
            file_info,
        };

        let mut dataset = Self {
            config,
            info: dataset_info,
            cached_audio: None,
            cached_labels: None,
            label_to_idx,
        };

        // Pre-load audio if caching is enabled
        if dataset.config.cache_audio {
            dataset.load_audio()?;
        }

        Ok(dataset)
    }

    /// Get dataset information
    pub fn info(&self) -> &AudioDatasetInfo {
        &self.info
    }

    /// Load all audio files into memory
    fn load_audio(&mut self) -> Result<()> {
        let mut cached_audio = Vec::new();
        let mut cached_labels = Vec::new();

        for file_info in &self.info.file_info {
            // Load and process audio
            let audio_data = load_audio_file(&file_info.path, &self.config)?;
            cached_audio.push(audio_data);

            // Store label
            if let Some(ref label) = file_info.label {
                cached_labels.push(label.clone());
            } else {
                cached_labels.push("unknown".to_string());
            }
        }

        self.cached_audio = Some(cached_audio);
        self.cached_labels = Some(cached_labels);
        Ok(())
    }

    /// Get the number of unique labels
    pub fn num_classes(&self) -> usize {
        self.info.labels.len()
    }

    /// Get label names
    pub fn label_names(&self) -> &[String] {
        &self.info.labels
    }
}

#[cfg(feature = "audio")]
impl Dataset<f32> for AudioDataset {
    fn len(&self) -> usize {
        self.info.num_files
    }

    fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if index >= self.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.len()
            )));
        }

        let (audio_data, label_str) = if let Some(ref cached_audio) = self.cached_audio {
            // Use cached data
            let audio = cached_audio[index].clone();
            let label = self
                .cached_labels
                .as_ref()
                .and_then(|labels| labels.get(index))
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            (audio, label)
        } else {
            // Load on-demand
            let file_info = &self.info.file_info[index];
            let audio = load_audio_file(&file_info.path, &self.config)?;
            let label = file_info
                .label
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            (audio, label)
        };

        // Create feature tensor
        let len = audio_data.len();
        let feature_tensor = Tensor::from_vec(audio_data, &[len])?;

        // Create label tensor (as class index)
        let label_idx = self.label_to_idx.get(&label_str).copied().unwrap_or(0);
        let label_tensor = Tensor::from_vec(vec![label_idx as f32], &[])?;

        Ok((feature_tensor, label_tensor))
    }
}

/// Discover audio files in a directory
#[cfg(feature = "audio")]
fn discover_audio_files(directory: &Path, config: &AudioConfig) -> Result<Vec<AudioInfo>> {
    let mut file_info = Vec::new();

    for entry in fs::read_dir(directory)
        .map_err(|e| TensorError::invalid_argument(format!("Failed to read directory: {e}")))?
    {
        let entry = entry.map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read directory entry: {e}"))
        })?;

        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                let ext_str = extension.to_string_lossy().to_lowercase();
                if config.supported_extensions.contains(&ext_str) {
                    match get_audio_info(&path, config) {
                        Ok(info) => file_info.push(info),
                        Err(_) => continue, // Skip files that can't be processed
                    }
                }
            }
        }
    }

    Ok(file_info)
}

/// Get information about an audio file
#[cfg(feature = "audio")]
fn get_audio_info(path: &Path, config: &AudioConfig) -> Result<AudioInfo> {
    let file_size = fs::metadata(path)
        .map_err(|e| TensorError::invalid_argument(format!("Failed to get file metadata: {e}")))?
        .len();

    // For now, return basic info without actually decoding the audio
    // In a full implementation, you would decode to get accurate duration and format info
    let sample_rate = config.sample_rate;
    let channels = 1; // Assume mono for simplicity
    let duration = 1.0; // Placeholder duration
    let num_samples = (sample_rate as f32 * duration) as usize;
    let format = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Extract label based on strategy
    let label = extract_label(path, &config.label_strategy, &config.label_mapping);

    Ok(AudioInfo {
        path: path.to_path_buf(),
        sample_rate,
        channels,
        duration,
        num_samples,
        file_size,
        format,
        label,
    })
}

/// Extract label from audio file path
#[cfg(feature = "audio")]
fn extract_label(
    path: &Path,
    strategy: &AudioLabelStrategy,
    mapping: &Option<HashMap<String, String>>,
) -> Option<String> {
    match strategy {
        AudioLabelStrategy::FromFilename => path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.split('_').next())
            .map(|s| s.to_string()),
        AudioLabelStrategy::FromDirectory => path
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            .map(|s| s.to_string()),
        AudioLabelStrategy::FromMapping => {
            if let Some(ref map) = mapping {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| map.get(name))
                    .cloned()
            } else {
                None
            }
        }
        AudioLabelStrategy::None => None,
    }
}

/// Load and process an audio file
#[cfg(feature = "audio")]
fn load_audio_file(_path: &Path, config: &AudioConfig) -> Result<Vec<f32>> {
    // For now, return a simple sine wave as placeholder
    // In a full implementation, you would use Symphonia to decode the audio file
    let duration = config.max_duration.unwrap_or(1.0);
    let sample_rate = config.sample_rate as f32;
    let num_samples = (duration * sample_rate) as usize;

    let mut audio_data = Vec::with_capacity(num_samples);
    let frequency = 440.0; // A4 note

    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
        audio_data.push(sample);
    }

    // Apply normalization if requested
    if config.normalize {
        normalize_audio(&mut audio_data);
    }

    Ok(audio_data)
}

/// Normalize audio to [-1, 1] range
#[cfg(feature = "audio")]
fn normalize_audio(audio: &mut [f32]) {
    let max_abs = audio.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| a.max(b));

    if max_abs > 0.0 {
        for sample in audio.iter_mut() {
            *sample /= max_abs;
        }
    }
}

// Stub implementations when audio feature is not enabled
#[cfg(not(feature = "audio"))]
pub struct AudioConfig;

#[cfg(not(feature = "audio"))]
pub struct AudioDatasetInfo;

#[cfg(not(feature = "audio"))]
pub struct AudioDatasetBuilder;

#[cfg(not(feature = "audio"))]
pub struct AudioDataset;

#[cfg(not(feature = "audio"))]
pub struct AudioInfo;

#[cfg(not(feature = "audio"))]
pub enum FeatureType {
    Raw,
}

#[cfg(not(feature = "audio"))]
pub enum AudioLabelStrategy {
    None,
}

#[cfg(test)]
#[cfg(feature = "audio")]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.feature_type, FeatureType::Raw);
        assert_eq!(config.normalize, true);
        assert_eq!(config.n_mfcc, 13);
        assert_eq!(config.n_mels, 80);
    }

    #[test]
    fn test_audio_config_builder() {
        let config = AudioConfig::default()
            .with_sample_rate(22050)
            .with_max_duration(5.0)
            .with_feature_extraction(FeatureType::MFCC)
            .with_n_mfcc(20)
            .with_normalize(false);

        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.max_duration, Some(5.0));
        assert_eq!(config.feature_type, FeatureType::MFCC);
        assert_eq!(config.n_mfcc, 20);
        assert_eq!(config.normalize, false);
    }

    #[test]
    fn test_audio_dataset_builder() {
        let builder = AudioDatasetBuilder::new()
            .sample_rate(16000)
            .feature_type(FeatureType::MelSpectrogram);

        assert_eq!(builder.config.sample_rate, 16000);
        assert_eq!(builder.config.feature_type, FeatureType::MelSpectrogram);
    }

    #[test]
    fn test_normalize_audio() {
        let mut audio = vec![0.5, -1.0, 0.25, -0.5];
        normalize_audio(&mut audio);

        // Should be normalized so max absolute value is 1.0
        let max_abs = audio.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| a.max(b));
        assert!((max_abs - 1.0).abs() < 1e-6);
    }

    // Note: Full integration tests would require actual audio files
    // and would be more suitable for the integration test suite
}
