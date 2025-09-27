//! Online Learning module for concept drift detection and real-time processing
//!
//! This module provides functionality for detecting concept drift in streaming data
//! and adapting to changes in data distribution over time.

use crate::Dataset;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tenflowers_core::{Result, Tensor, TensorError};

/// Concept drift detection method
#[derive(Debug, Clone)]
pub enum DriftDetectionMethod {
    /// ADWIN (Adaptive Windowing) for detecting changes in data distribution
    ADWIN { confidence: f64 },
    /// Page-Hinkley test for detecting changes in mean
    PageHinkley { threshold: f64, alpha: f64 },
    /// Kolmogorov-Smirnov test for detecting distribution changes
    KolmogorovSmirnov { window_size: usize, confidence: f64 },
    /// Statistical test based on error rate changes
    ErrorRate { window_size: usize, threshold: f64 },
}

/// Online learning configuration
#[derive(Debug, Clone)]
pub struct OnlineLearningConfig {
    /// Maximum window size for storing recent samples
    pub max_window_size: usize,
    /// Minimum samples required before drift detection
    pub min_samples_for_detection: usize,
    /// Real-time processing timeout
    pub processing_timeout: Duration,
    /// Enable adaptive windowing
    pub adaptive_windowing: bool,
    /// Drift detection method
    pub drift_method: DriftDetectionMethod,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            max_window_size: 1000,
            min_samples_for_detection: 30,
            processing_timeout: Duration::from_millis(100),
            adaptive_windowing: true,
            drift_method: DriftDetectionMethod::ADWIN { confidence: 0.95 },
        }
    }
}

/// Statistics for online learning
#[derive(Debug, Clone)]
pub struct OnlineStats {
    /// Number of samples processed
    pub samples_processed: usize,
    /// Number of drift events detected
    pub drift_events: usize,
    /// Average processing time per sample
    pub avg_processing_time: Duration,
    /// Current window size
    pub current_window_size: usize,
    /// Last drift detection time
    pub last_drift_time: Option<Instant>,
}

/// Online learning dataset for streaming data with concept drift detection
pub struct OnlineLearningDataset<T, D: Dataset<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    #[allow(dead_code)]
    dataset: D,
    config: OnlineLearningConfig,
    sample_window: VecDeque<(Tensor<T>, Tensor<T>)>,
    error_window: VecDeque<f64>,
    stats: OnlineStats,
    drift_detector: Box<dyn DriftDetector>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, D: Dataset<T>> OnlineLearningDataset<T, D>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new online learning dataset
    pub fn new(dataset: D, config: OnlineLearningConfig) -> Self {
        let drift_detector = create_drift_detector(&config.drift_method);

        Self {
            dataset,
            config,
            sample_window: VecDeque::new(),
            error_window: VecDeque::new(),
            stats: OnlineStats {
                samples_processed: 0,
                drift_events: 0,
                avg_processing_time: Duration::from_millis(0),
                current_window_size: 0,
                last_drift_time: None,
            },
            drift_detector,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process a new sample and detect concept drift
    pub fn process_sample(
        &mut self,
        sample: (Tensor<T>, Tensor<T>),
        prediction_error: f64,
    ) -> Result<bool> {
        let start_time = Instant::now();

        // Add sample to window
        self.sample_window.push_back(sample);
        self.error_window.push_back(prediction_error);

        // Maintain window size
        if self.sample_window.len() > self.config.max_window_size {
            self.sample_window.pop_front();
        }
        if self.error_window.len() > self.config.max_window_size {
            self.error_window.pop_front();
        }

        // Detect drift
        let drift_detected =
            if self.stats.samples_processed >= self.config.min_samples_for_detection {
                self.drift_detector.detect_drift(&self.error_window)?
            } else {
                false
            };

        if drift_detected {
            self.handle_drift_detection()?;
        }

        // Update statistics
        self.stats.samples_processed += 1;
        self.stats.current_window_size = self.sample_window.len();

        let processing_time = start_time.elapsed();
        self.stats.avg_processing_time = if self.stats.samples_processed == 1 {
            processing_time
        } else {
            let avg_nanos = ((self.stats.avg_processing_time.as_nanos()
                * (self.stats.samples_processed - 1) as u128)
                + processing_time.as_nanos())
                / self.stats.samples_processed as u128;
            Duration::from_nanos(avg_nanos.min(u64::MAX as u128) as u64)
        };

        Ok(drift_detected)
    }

    /// Handle drift detection event
    fn handle_drift_detection(&mut self) -> Result<()> {
        self.stats.drift_events += 1;
        self.stats.last_drift_time = Some(Instant::now());

        // Reset drift detector state
        self.drift_detector.reset();

        // Optionally adjust window size based on drift
        if self.config.adaptive_windowing {
            self.adjust_window_size();
        }

        Ok(())
    }

    /// Adjust window size based on drift detection
    fn adjust_window_size(&mut self) {
        // Simple adaptive strategy: reduce window size after drift
        let new_size =
            (self.sample_window.len() * 3 / 4).max(self.config.min_samples_for_detection);

        while self.sample_window.len() > new_size {
            self.sample_window.pop_front();
        }
        while self.error_window.len() > new_size {
            self.error_window.pop_front();
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Get current sample window
    pub fn get_current_window(&self) -> &VecDeque<(Tensor<T>, Tensor<T>)> {
        &self.sample_window
    }

    /// Reset the online learning state
    pub fn reset(&mut self) {
        self.sample_window.clear();
        self.error_window.clear();
        self.drift_detector.reset();
        self.stats = OnlineStats {
            samples_processed: 0,
            drift_events: 0,
            avg_processing_time: Duration::from_millis(0),
            current_window_size: 0,
            last_drift_time: None,
        };
    }
}

impl<T, D: Dataset<T>> Dataset<T> for OnlineLearningDataset<T, D>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.sample_window.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.sample_window.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for online dataset of length {}",
                index,
                self.sample_window.len()
            )));
        }

        // Since Tensor doesn't implement Clone, we need to recreate it
        let (ref features, ref labels) = self.sample_window[index];
        Ok((features.clone(), labels.clone()))
    }
}

/// Trait for drift detection algorithms
pub trait DriftDetector: Send + Sync {
    /// Detect drift based on error sequence
    fn detect_drift(&mut self, errors: &VecDeque<f64>) -> Result<bool>;

    /// Reset detector state
    fn reset(&mut self);
}

/// ADWIN (Adaptive Windowing) drift detector
pub struct ADWINDetector {
    confidence: f64,
    window: VecDeque<f64>,
    sum: f64,
    sum_squared: f64,
}

impl ADWINDetector {
    pub fn new(confidence: f64) -> Self {
        Self {
            confidence,
            window: VecDeque::new(),
            sum: 0.0,
            sum_squared: 0.0,
        }
    }

    fn calculate_cut_threshold(&self, n1: usize, n2: usize) -> f64 {
        let n = n1 + n2;
        let delta = 1.0 - self.confidence;

        if n1 == 0 || n2 == 0 {
            return f64::INFINITY;
        }

        let m = 1.0 / (n1 as f64) + 1.0 / (n2 as f64);
        let variance = self.sum_squared / (n as f64) - (self.sum / (n as f64)).powi(2);

        ((2.0 * variance * m * (2.0 / delta).ln()) + (2.0 * m * (2.0 / delta).ln() / 3.0)).sqrt()
    }
}

impl DriftDetector for ADWINDetector {
    fn detect_drift(&mut self, errors: &VecDeque<f64>) -> Result<bool> {
        // Add new errors to window
        for &error in errors.iter().skip(self.window.len()) {
            self.window.push_back(error);
            self.sum += error;
            self.sum_squared += error * error;
        }

        if self.window.len() < 2 {
            return Ok(false);
        }

        // Check for drift by testing different window cuts
        let n = self.window.len();

        for i in 1..n {
            let n1 = i;
            let n2 = n - i;

            let sum1: f64 = self.window.iter().take(n1).sum();
            let sum2: f64 = self.window.iter().skip(n1).sum();

            let mean1 = sum1 / n1 as f64;
            let mean2 = sum2 / n2 as f64;

            let threshold = self.calculate_cut_threshold(n1, n2);

            if (mean1 - mean2).abs() > threshold {
                // Remove old samples up to the cut point
                for _ in 0..i {
                    if let Some(old_error) = self.window.pop_front() {
                        self.sum -= old_error;
                        self.sum_squared -= old_error * old_error;
                    }
                }
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sum_squared = 0.0;
    }
}

/// Page-Hinkley drift detector
pub struct PageHinkleyDetector {
    threshold: f64,
    alpha: f64,
    mean: f64,
    sum: f64,
    min_sum: f64,
    sample_count: usize,
}

impl PageHinkleyDetector {
    pub fn new(threshold: f64, alpha: f64) -> Self {
        Self {
            threshold,
            alpha,
            mean: 0.0,
            sum: 0.0,
            min_sum: 0.0,
            sample_count: 0,
        }
    }
}

impl DriftDetector for PageHinkleyDetector {
    fn detect_drift(&mut self, errors: &VecDeque<f64>) -> Result<bool> {
        if errors.len() <= self.sample_count {
            return Ok(false);
        }

        // Process new samples
        for &error in errors.iter().skip(self.sample_count) {
            self.sample_count += 1;

            // Update running mean
            self.mean += (error - self.mean) / self.sample_count as f64;

            // Update cumulative sum
            self.sum += error - self.mean - self.alpha;
            self.min_sum = self.min_sum.min(self.sum);

            // Check for drift
            if self.sum - self.min_sum > self.threshold {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn reset(&mut self) {
        self.mean = 0.0;
        self.sum = 0.0;
        self.min_sum = 0.0;
        self.sample_count = 0;
    }
}

/// Kolmogorov-Smirnov drift detector
pub struct KSDetector {
    window_size: usize,
    confidence: f64,
    reference_window: VecDeque<f64>,
    current_window: VecDeque<f64>,
}

impl KSDetector {
    pub fn new(window_size: usize, confidence: f64) -> Self {
        Self {
            window_size,
            confidence,
            reference_window: VecDeque::new(),
            current_window: VecDeque::new(),
        }
    }

    fn ks_test(&self, sample1: &[f64], sample2: &[f64]) -> f64 {
        let mut combined: Vec<f64> = sample1.iter().chain(sample2.iter()).cloned().collect();
        combined.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        let mut max_diff: f64 = 0.0;

        for &value in &combined {
            let cdf1 = sample1.iter().filter(|&&x| x <= value).count() as f64 / n1;
            let cdf2 = sample2.iter().filter(|&&x| x <= value).count() as f64 / n2;
            max_diff = max_diff.max((cdf1 - cdf2).abs());
        }

        max_diff
    }
}

impl DriftDetector for KSDetector {
    fn detect_drift(&mut self, errors: &VecDeque<f64>) -> Result<bool> {
        // Initialize reference window
        if self.reference_window.is_empty() && errors.len() >= self.window_size {
            self.reference_window
                .extend(errors.iter().take(self.window_size));
        }

        // Update current window
        for &error in errors.iter().skip(self.current_window.len()) {
            self.current_window.push_back(error);
            if self.current_window.len() > self.window_size {
                self.current_window.pop_front();
            }
        }

        // Perform KS test when both windows are full
        if self.reference_window.len() == self.window_size
            && self.current_window.len() == self.window_size
        {
            let ref_slice: Vec<f64> = self.reference_window.iter().cloned().collect();
            let cur_slice: Vec<f64> = self.current_window.iter().cloned().collect();

            let ks_statistic = self.ks_test(&ref_slice, &cur_slice);

            // Critical value for KS test (approximation)
            let n = self.window_size as f64;
            let critical_value = (-0.5 * (1.0 - self.confidence).ln() / n).sqrt();

            if ks_statistic > critical_value {
                // Update reference window with current window
                self.reference_window = self.current_window.clone();
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn reset(&mut self) {
        self.reference_window.clear();
        self.current_window.clear();
    }
}

/// Error rate drift detector
pub struct ErrorRateDetector {
    window_size: usize,
    threshold: f64,
    reference_error_rate: Option<f64>,
    current_errors: VecDeque<f64>,
}

impl ErrorRateDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
            reference_error_rate: None,
            current_errors: VecDeque::new(),
        }
    }
}

impl DriftDetector for ErrorRateDetector {
    fn detect_drift(&mut self, errors: &VecDeque<f64>) -> Result<bool> {
        // Update current errors
        for &error in errors.iter().skip(self.current_errors.len()) {
            self.current_errors.push_back(error);
            if self.current_errors.len() > self.window_size {
                self.current_errors.pop_front();
            }
        }

        if self.current_errors.len() < self.window_size {
            return Ok(false);
        }

        let current_error_rate =
            self.current_errors.iter().sum::<f64>() / self.current_errors.len() as f64;

        // Initialize reference error rate
        if self.reference_error_rate.is_none() {
            self.reference_error_rate = Some(current_error_rate);
            return Ok(false);
        }

        let reference_rate = self.reference_error_rate.unwrap();

        // Check if error rate has changed significantly
        if (current_error_rate - reference_rate).abs() > self.threshold {
            self.reference_error_rate = Some(current_error_rate);
            return Ok(true);
        }

        Ok(false)
    }

    fn reset(&mut self) {
        self.reference_error_rate = None;
        self.current_errors.clear();
    }
}

/// Create a drift detector based on the specified method
fn create_drift_detector(method: &DriftDetectionMethod) -> Box<dyn DriftDetector> {
    match method {
        DriftDetectionMethod::ADWIN { confidence } => Box::new(ADWINDetector::new(*confidence)),
        DriftDetectionMethod::PageHinkley { threshold, alpha } => {
            Box::new(PageHinkleyDetector::new(*threshold, *alpha))
        }
        DriftDetectionMethod::KolmogorovSmirnov {
            window_size,
            confidence,
        } => Box::new(KSDetector::new(*window_size, *confidence)),
        DriftDetectionMethod::ErrorRate {
            window_size,
            threshold,
        } => Box::new(ErrorRateDetector::new(*window_size, *threshold)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_adwin_detector() {
        let mut detector = ADWINDetector::new(0.8); // Lower confidence for easier detection

        // Create stable data
        let mut errors = VecDeque::new();
        for i in 0..30 {
            errors.push_back(0.1 + (i as f64) * 0.005);
        }

        // Should not detect drift in stable data
        assert!(!detector.detect_drift(&errors).unwrap());

        // Add significantly drifted data
        for i in 0..20 {
            errors.push_back(0.8 + (i as f64) * 0.01);
        }

        // Should detect drift with larger difference
        assert!(detector.detect_drift(&errors).unwrap());
    }

    #[test]
    fn test_page_hinkley_detector() {
        let mut detector = PageHinkleyDetector::new(2.0, 0.001); // Lower threshold for easier detection

        // Create gradual drift
        let mut errors = VecDeque::new();
        for i in 0..50 {
            let error = if i < 25 { 0.1 } else { 0.5 }; // Larger drift
            errors.push_back(error);
        }

        // Should detect drift
        assert!(detector.detect_drift(&errors).unwrap());
    }

    #[test]
    fn test_online_learning_dataset() {
        // Create test dataset
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = OnlineLearningConfig::default();
        let mut online_dataset = OnlineLearningDataset::new(dataset, config);

        // Process samples
        let sample1 = (
            Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![0.0], &[]).unwrap(),
        );
        let sample2 = (
            Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![1.0], &[]).unwrap(),
        );

        assert!(!online_dataset.process_sample(sample1, 0.1).unwrap());
        assert!(!online_dataset.process_sample(sample2, 0.15).unwrap());

        assert_eq!(online_dataset.len(), 2);
        assert_eq!(online_dataset.get_stats().samples_processed, 2);
    }

    #[test]
    fn test_error_rate_detector() {
        let mut detector = ErrorRateDetector::new(10, 0.1);

        // Create stable low error rate
        let mut errors = VecDeque::new();
        for _ in 0..10 {
            errors.push_back(0.05);
        }

        assert!(!detector.detect_drift(&errors).unwrap());

        // Add high error rate (drift)
        for _ in 0..10 {
            errors.push_back(0.25);
        }

        assert!(detector.detect_drift(&errors).unwrap());
    }

    #[test]
    fn test_ks_detector() {
        let mut detector = KSDetector::new(5, 0.95);

        // Create two different distributions
        let mut errors = VecDeque::new();

        // First distribution (normal)
        for i in 0..5 {
            errors.push_back(0.1 + (i as f64) * 0.01);
        }

        // Second distribution (different)
        for i in 0..5 {
            errors.push_back(0.5 + (i as f64) * 0.01);
        }

        // Should detect drift between different distributions
        assert!(detector.detect_drift(&errors).unwrap());
    }
}
