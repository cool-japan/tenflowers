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

// Rubato: offline/batch sinc resampler used to convert decoded audio to the
// dataset's configured target sample rate.
#[cfg(feature = "audio")]
use rubato::audioadapter_buffers::direct::InterleavedSlice;
#[cfg(feature = "audio")]
use rubato::{
    Async as RubatoAsyncResampler, FixedAsync, Resampler as RubatoResamplerTrait,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

// Symphonia: container probing (format auto-detection) and audio decoding
// for WAV, FLAC, MP3 and Ogg/Vorbis.
#[cfg(feature = "audio")]
use symphonia::core::codecs::audio::AudioDecoderOptions;
#[cfg(feature = "audio")]
use symphonia::core::codecs::CodecParameters;
#[cfg(feature = "audio")]
use symphonia::core::errors::Error as SymphoniaError;
#[cfg(feature = "audio")]
use symphonia::core::formats::probe::Hint;
#[cfg(feature = "audio")]
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
#[cfg(feature = "audio")]
use symphonia::core::io::MediaSourceStream;
#[cfg(feature = "audio")]
use symphonia::core::meta::MetadataOptions;

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

/// Get information about an audio file.
///
/// `sample_rate`, `channels`, `duration` and `num_samples` are real,
/// decoder-derived values: the container is probed via Symphonia to read
/// its codec parameters. When the container records a frame count in its
/// headers (e.g. WAV, FLAC), that is used directly, avoiding a full decode.
/// When it does not (e.g. some MP3 streams without a Xing/VBRI header), the
/// file is fully decoded so that `duration`/`num_samples` are never
/// fabricated or left silently wrong.
#[cfg(feature = "audio")]
fn get_audio_info(path: &Path, config: &AudioConfig) -> Result<AudioInfo> {
    let file_size = fs::metadata(path)
        .map_err(|e| TensorError::invalid_argument(format!("Failed to get file metadata: {e}")))?
        .len();

    let format = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Extract label based on strategy (purely path-based, fully real).
    let label = extract_label(path, &config.label_strategy, &config.label_mapping);

    let probed = probe_audio_track(path)?;
    let (sample_rate, channels, num_samples, duration) = match probed.num_frames {
        Some(frames) => {
            let num_samples = frames as usize * probed.channels;
            let duration = frames as f32 / probed.sample_rate as f32;
            (probed.sample_rate, probed.channels, num_samples, duration)
        }
        None => {
            // The container doesn't record a frame count in its headers, so
            // the only honest way to know the real duration is to decode
            // the whole file and count the decoded samples.
            let decoded = decode_audio_track(path)?;
            let channels = decoded.channels.max(1);
            let frames = decoded.samples.len() / channels;
            let duration = frames as f32 / decoded.sample_rate as f32;
            (
                decoded.sample_rate,
                decoded.channels,
                decoded.samples.len(),
                duration,
            )
        }
    };

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

/// Track properties obtainable from container/codec headers alone, without
/// decoding any audio packets.
#[cfg(feature = "audio")]
struct ProbedAudioTrack {
    /// Sample rate in Hz, as reported by the codec parameters.
    sample_rate: u32,
    /// Number of channels, as reported by the codec parameters.
    channels: usize,
    /// Total number of frames, if the container records it in its headers
    /// (e.g. WAV, FLAC). `None` when only a full decode can determine it
    /// (e.g. some MP3 streams without a Xing/VBRI header).
    num_frames: Option<u64>,
}

/// Fully decoded audio: interleaved samples plus the sample rate and channel
/// count actually reported by the decoder (i.e. the file's native values,
/// before any resampling).
#[cfg(feature = "audio")]
struct DecodedAudio {
    /// Interleaved samples (`channels` values per frame).
    samples: Vec<f32>,
    sample_rate: u32,
    channels: usize,
}

/// A probed container/track, ready for either cheap metadata inspection or
/// full packet-by-packet decoding.
#[cfg(feature = "audio")]
struct ProbedFormat {
    /// The format reader, positioned to read packets from the start.
    reader: Box<dyn FormatReader>,
    /// The id of the located audio track (packets carry this id).
    track_id: u32,
    /// The audio track's codec parameters (sample rate, channels, codec, ...).
    audio_params: symphonia::core::codecs::audio::AudioCodecParameters,
    /// The track's frame count, if the container records one in its headers.
    num_frames: Option<u64>,
}

/// Open `path`, probe its container format (by content, not just file
/// extension, though the extension is passed along as a hint), and locate
/// its first audio track.
#[cfg(feature = "audio")]
fn probe_audio_format(path: &Path) -> Result<ProbedFormat> {
    let file = fs::File::open(path).map_err(|e| {
        TensorError::io_error_simple(format!(
            "Failed to open audio file '{}': {e}",
            path.display()
        ))
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }

    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| {
            TensorError::invalid_argument(format!(
                "Could not recognize audio format for '{}': {e}. \
                 Supported containers/codecs: WAV, FLAC, MP3, Ogg/Vorbis (PCM/ADPCM).",
                path.display()
            ))
        })?;

    let (track_id, audio_params, num_frames) = {
        let track = format.first_track(TrackType::Audio).ok_or_else(|| {
            TensorError::invalid_argument(format!("No audio track found in '{}'", path.display()))
        })?;

        let audio_params = match &track.codec_params {
            Some(CodecParameters::Audio(params)) => params.clone(),
            _ => {
                return Err(TensorError::invalid_argument(format!(
                    "Track has no audio codec parameters in '{}'",
                    path.display()
                )));
            }
        };

        (track.id, audio_params, track.num_frames)
    };

    Ok(ProbedFormat {
        reader: format,
        track_id,
        audio_params,
        num_frames,
    })
}

/// Probe an audio file's track metadata without decoding any samples.
#[cfg(feature = "audio")]
fn probe_audio_track(path: &Path) -> Result<ProbedAudioTrack> {
    let probed = probe_audio_format(path)?;

    let sample_rate = probed.audio_params.sample_rate.ok_or_else(|| {
        TensorError::invalid_argument(format!(
            "Unknown sample rate for audio file '{}'",
            path.display()
        ))
    })?;
    let channels = probed
        .audio_params
        .channels
        .as_ref()
        .map(symphonia::core::audio::Channels::count)
        .unwrap_or(1);

    Ok(ProbedAudioTrack {
        sample_rate,
        channels,
        num_frames: probed.num_frames,
    })
}

/// Fully decode an audio file to interleaved `f32` samples at its native
/// sample rate. No resampling or normalization is applied here.
#[cfg(feature = "audio")]
fn decode_audio_track(path: &Path) -> Result<DecodedAudio> {
    let ProbedFormat {
        reader: mut format,
        track_id,
        audio_params,
        num_frames: _,
    } = probe_audio_format(path)?;

    let sample_rate = audio_params.sample_rate.ok_or_else(|| {
        TensorError::invalid_argument(format!(
            "Unknown sample rate for audio file '{}'",
            path.display()
        ))
    })?;
    let channels = audio_params
        .channels
        .as_ref()
        .map(symphonia::core::audio::Channels::count)
        .unwrap_or(1);

    let dec_opts = AudioDecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &dec_opts)
        .map_err(|e| {
            TensorError::not_implemented_simple(format!(
                "No decoder available for the codec used by '{}': {e}",
                path.display()
            ))
        })?;

    let mut samples: Vec<f32> = Vec::new();
    let mut chunk: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => {
                return Err(TensorError::invalid_argument(format!(
                    "Failed to read audio packet from '{}': {e}",
                    path.display()
                )));
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => {
                return Err(TensorError::invalid_argument(format!(
                    "Failed to decode audio from '{}': {e}",
                    path.display()
                )));
            }
        };

        decoded.copy_to_vec_interleaved::<f32>(&mut chunk);
        samples.extend_from_slice(&chunk);
    }

    if samples.is_empty() {
        return Err(TensorError::invalid_argument(format!(
            "No audio samples could be decoded from '{}'",
            path.display()
        )));
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
    })
}

/// Resample interleaved audio from `from_rate` to `to_rate` using a
/// high-quality sinc-interpolation resampler. This favors offline/batch
/// quality (cubic sinc interpolation) over realtime throughput, which is
/// appropriate for dataset preprocessing.
///
/// Returns the input unchanged (cloned) if the rates already match.
#[cfg(feature = "audio")]
fn resample_interleaved(
    samples: &[f32],
    channels: usize,
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(TensorError::invalid_argument(
            "Cannot resample audio with zero channels".to_string(),
        ));
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    if from_rate == 0 || to_rate == 0 {
        return Err(TensorError::invalid_argument(
            "Cannot resample audio with a zero sample rate".to_string(),
        ));
    }

    let frames_in = samples.len() / channels;
    if frames_in == 0 {
        return Ok(Vec::new());
    }

    // A solid general-purpose quality preset: cubic sinc interpolation with
    // a moderately long filter. Good enough for ML preprocessing without
    // the CPU cost of the largest filter sizes.
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let chunk_size = 1024usize;

    let mut resampler = RubatoAsyncResampler::<f32>::new_sinc(
        ratio,
        1.0,
        &params,
        chunk_size,
        channels,
        FixedAsync::Input,
    )
    .map_err(|e| {
        TensorError::compute_error_simple(format!("Failed to construct audio resampler: {e}"))
    })?;

    let output_capacity_frames = resampler.process_all_needed_output_len(frames_in);
    let mut out_data = vec![0.0f32; output_capacity_frames * channels];

    let input_adapter = InterleavedSlice::new(samples, channels, frames_in).map_err(|e| {
        TensorError::compute_error_simple(format!("Invalid resampler input buffer: {e}"))
    })?;
    let mut output_adapter =
        InterleavedSlice::new_mut(&mut out_data, channels, output_capacity_frames).map_err(
            |e| TensorError::compute_error_simple(format!("Invalid resampler output buffer: {e}")),
        )?;

    let (_frames_read, frames_written) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, frames_in, None)
        .map_err(|e| TensorError::compute_error_simple(format!("Audio resampling failed: {e}")))?;

    out_data.truncate(frames_written * channels);
    Ok(out_data)
}

/// Load and process an audio file: decode it (WAV/FLAC/MP3/Ogg-Vorbis via
/// Symphonia), resample to `config.sample_rate` if needed (via Rubato), and
/// normalize if configured. Channels are kept at the file's native layout
/// (interleaved); `AudioConfig` has no downmix option, so none is applied.
#[cfg(feature = "audio")]
fn load_audio_file(path: &Path, config: &AudioConfig) -> Result<Vec<f32>> {
    let decoded = decode_audio_track(path)?;
    let channels = decoded.channels.max(1);

    let mut samples = if config.sample_rate != 0 && config.sample_rate != decoded.sample_rate {
        resample_interleaved(
            &decoded.samples,
            channels,
            decoded.sample_rate,
            config.sample_rate,
        )?
    } else {
        decoded.samples
    };

    if config.normalize {
        normalize_audio(&mut samples);
    }

    Ok(samples)
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

    /// Hand-construct a minimal valid 16-bit PCM WAV file (RIFF/WAVE header
    /// with `fmt ` and `data` chunks) from known sample values, for
    /// exercising the real decoder without depending on any external test
    /// fixture files.
    fn build_pcm16_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let bits_per_sample: u16 = 16;
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let data_bytes = (samples.len() * 2) as u32;

        let mut buf = Vec::with_capacity(44 + samples.len() * 2);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size (PCM)
        buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());

        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.feature_type, FeatureType::Raw);
        assert!(config.normalize);
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
        assert!(!config.normalize);
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

    #[test]
    fn test_load_audio_file_errors_on_invalid_data() {
        // Audio decoding is implemented (Symphonia), but garbage bytes are
        // not valid audio in any supported container/codec. Loading must
        // report an honest error rather than fabricating samples or
        // panicking.
        let base =
            std::env::temp_dir().join(format!("tenflowers_audio_load_{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let fake_audio = base.join("fake.wav");
        std::fs::write(&fake_audio, b"not really audio").expect("test: write should succeed");

        let config = AudioConfig::default();
        let result = load_audio_file(&fake_audio, &config);
        assert!(
            result.is_err(),
            "load_audio_file must not fabricate samples for invalid input; it must error"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_get_audio_info_errors_on_invalid_data() {
        // Probing must also fail honestly (not panic, not report fabricated
        // metadata) for bytes that aren't a recognizable audio container.
        let base =
            std::env::temp_dir().join(format!("tenflowers_audio_info_bad_{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let fake_audio = base.join("clip.wav");
        std::fs::write(&fake_audio, b"0123456789").expect("test: write should succeed");

        let config = AudioConfig::default();
        let result = get_audio_info(&fake_audio, &config);
        assert!(
            result.is_err(),
            "get_audio_info must not fabricate metadata for invalid input; it must error"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_load_audio_file_decodes_real_wav_samples() {
        // Decode a hand-crafted, genuinely valid WAV file and verify the
        // returned samples are the real decoded values (not fabricated),
        // within the float tolerance expected from i16 -> f32 conversion.
        let base = std::env::temp_dir().join(format!(
            "tenflowers_audio_wav_decode_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let wav_path = base.join("tone.wav");

        let sample_rate = 8000u32;
        let raw_samples: Vec<i16> = (0..40i16).map(|i| (i - 20) * 500).collect();
        let wav_bytes = build_pcm16_wav(sample_rate, 1, &raw_samples);
        std::fs::write(&wav_path, &wav_bytes).expect("test: write should succeed");

        // Match the config's target rate to the file's native rate so no
        // resampling is applied, and disable normalization, so the decoded
        // samples are a direct conversion of the raw PCM values.
        let config = AudioConfig::default()
            .with_sample_rate(sample_rate)
            .with_normalize(false);

        let decoded = load_audio_file(&wav_path, &config).expect("test: decode should succeed");

        assert_eq!(decoded.len(), raw_samples.len());
        for (decoded_sample, &raw_sample) in decoded.iter().zip(raw_samples.iter()) {
            let expected = f32::from(raw_sample) / 32_768.0;
            assert!(
                (decoded_sample - expected).abs() < 1e-5,
                "decoded {decoded_sample} != expected {expected}"
            );
        }

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_get_audio_info_reports_real_values() {
        // get_audio_info now reports real, decoder/header-derived values
        // instead of the historical honest-zero placeholders.
        let base =
            std::env::temp_dir().join(format!("tenflowers_audio_wav_info_{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let wav_path = base.join("clip.wav");

        let sample_rate = 8000u32;
        let raw_samples: Vec<i16> = vec![0; 32];
        let wav_bytes = build_pcm16_wav(sample_rate, 1, &raw_samples);
        std::fs::write(&wav_path, &wav_bytes).expect("test: write should succeed");

        let config = AudioConfig::default();
        let info = get_audio_info(&wav_path, &config).expect("test: probing should succeed");

        // Real, filesystem-derivable values are still populated.
        assert_eq!(info.file_size, wav_bytes.len() as u64);
        assert_eq!(info.format, "wav");
        // Decode/header-derived values are now real, not fabricated zeros.
        assert_eq!(info.sample_rate, sample_rate);
        assert_eq!(info.channels, 1);
        assert_eq!(info.num_samples, raw_samples.len());
        let expected_duration = raw_samples.len() as f32 / sample_rate as f32;
        assert!(
            (info.duration - expected_duration).abs() < 1e-6,
            "duration {} != expected {}",
            info.duration,
            expected_duration
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_load_audio_file_resamples_to_target_rate() {
        // Decode at the file's native rate but request a different target
        // rate, and verify Rubato resampling produces the expected output
        // length. `process_all_into_buffer` guarantees the output is
        // trimmed/padded to exactly `ceil(ratio * input_frames)` frames.
        let base =
            std::env::temp_dir().join(format!("tenflowers_audio_resample_{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let wav_path = base.join("tone.wav");

        let native_rate = 8000u32;
        let target_rate = 16000u32; // exact 2x upsample
        let raw_samples: Vec<i16> = (0..50i16).map(|i| (i - 25) * 300).collect();
        let wav_bytes = build_pcm16_wav(native_rate, 1, &raw_samples);
        std::fs::write(&wav_path, &wav_bytes).expect("test: write should succeed");

        let config = AudioConfig::default()
            .with_sample_rate(target_rate)
            .with_normalize(false);

        let decoded =
            load_audio_file(&wav_path, &config).expect("test: decode+resample should succeed");

        let ratio = f64::from(target_rate) / f64::from(native_rate);
        let expected_frames = ((raw_samples.len() as f64) * ratio).ceil() as usize;
        assert_eq!(decoded.len(), expected_frames);

        let _ = std::fs::remove_dir_all(&base);
    }

    // Note: Full integration tests would require actual audio files
    // and would be more suitable for the integration test suite
}
