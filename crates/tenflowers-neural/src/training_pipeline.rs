#[cfg(test)]
use crate::trainer::TrainingMetrics;
use crate::{
    benchmarks::{BenchmarkConfig, BenchmarkResults, ModelBenchmark},
    model::Model,
    optimizers::Optimizer,
    trainer::{EarlyStopping, LearningRateReduction, ModelCheckpoint, Trainer, TrainingState},
};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
#[cfg(feature = "serialize")]
use tenflowers_core::TensorError;
use tenflowers_core::{Result, Tensor};

/// Comprehensive training pipeline configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TrainingPipelineConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Whether to use early stopping
    pub use_early_stopping: bool,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Whether to use learning rate reduction
    pub use_lr_reduction: bool,
    /// Learning rate reduction patience
    pub lr_reduction_patience: usize,
    /// Learning rate reduction factor
    pub lr_reduction_factor: f32,
    /// Whether to save model checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint save path
    pub checkpoint_path: String,
    /// Whether to validate during training
    pub validate_during_training: bool,
    /// Validation frequency (every N epochs)
    pub validation_frequency: usize,
    /// Whether to run benchmarks after training
    pub run_benchmarks: bool,
    /// Whether to compute detailed metrics
    pub compute_detailed_metrics: bool,
    /// Whether to be verbose during training
    pub verbose: bool,
}

impl Default for TrainingPipelineConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            use_early_stopping: true,
            early_stopping_patience: 10,
            use_lr_reduction: true,
            lr_reduction_patience: 5,
            lr_reduction_factor: 0.5,
            save_checkpoints: true,
            checkpoint_path: "model_checkpoint.ckpt".to_string(),
            validate_during_training: true,
            validation_frequency: 1,
            run_benchmarks: false,
            compute_detailed_metrics: true,
            verbose: true,
        }
    }
}

impl TrainingPipelineConfig {
    /// Create a quick training configuration for experimentation
    pub fn quick() -> Self {
        Self {
            epochs: 10,
            batch_size: 64,
            use_early_stopping: false,
            save_checkpoints: false,
            run_benchmarks: false,
            compute_detailed_metrics: false,
            ..Default::default()
        }
    }

    /// Create a production training configuration
    pub fn production() -> Self {
        Self {
            epochs: 200,
            batch_size: 32,
            use_early_stopping: true,
            early_stopping_patience: 15,
            use_lr_reduction: true,
            lr_reduction_patience: 7,
            save_checkpoints: true,
            run_benchmarks: true,
            compute_detailed_metrics: true,
            verbose: true,
            ..Default::default()
        }
    }

    /// Create a configuration for large models
    pub fn large_model() -> Self {
        Self {
            epochs: 50,
            batch_size: 16, // Smaller batch size for memory
            use_early_stopping: true,
            early_stopping_patience: 5,
            use_lr_reduction: true,
            lr_reduction_patience: 3,
            save_checkpoints: true,
            validation_frequency: 5, // Less frequent validation
            run_benchmarks: false,   // Skip benchmarks for large models
            compute_detailed_metrics: false,
            ..Default::default()
        }
    }

    /// Builder methods
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_early_stopping(mut self, patience: usize) -> Self {
        self.use_early_stopping = true;
        self.early_stopping_patience = patience;
        self
    }

    pub fn with_lr_reduction(mut self, patience: usize, factor: f32) -> Self {
        self.use_lr_reduction = true;
        self.lr_reduction_patience = patience;
        self.lr_reduction_factor = factor;
        self
    }

    pub fn with_checkpoints(mut self, path: String) -> Self {
        self.save_checkpoints = true;
        self.checkpoint_path = path;
        self
    }

    pub fn with_benchmarks(mut self, run_benchmarks: bool) -> Self {
        self.run_benchmarks = run_benchmarks;
        self
    }
}

/// Comprehensive training results
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TrainingResults<T> {
    /// Final training state
    pub final_state: TrainingState,
    /// Training duration
    pub training_duration: Duration,
    /// Best validation loss achieved
    pub best_val_loss: Option<f32>,
    /// Best validation accuracy achieved
    pub best_val_accuracy: Option<f32>,
    /// Final model parameters count
    pub parameter_count: usize,
    /// Benchmark results (if enabled)
    pub benchmark_results: Option<BenchmarkResults>,
    /// Detailed metrics (if enabled)
    pub detailed_metrics: HashMap<String, f32>,
    /// Training configuration used
    pub config: TrainingPipelineConfig,
    /// Whether training completed successfully
    pub completed_successfully: bool,
    /// Reason for training termination
    pub termination_reason: String,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TrainingResults<T> {
    /// Generate a comprehensive training report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== TRAINING RESULTS REPORT ===\n\n");

        // Training Overview
        report.push_str(&format!(
            "Training Status: {}\n",
            if self.completed_successfully {
                "✅ Completed Successfully"
            } else {
                "❌ Failed"
            }
        ));
        report.push_str(&format!(
            "Termination Reason: {}\n",
            self.termination_reason
        ));
        report.push_str(&format!(
            "Training Duration: {:.2} seconds\n",
            self.training_duration.as_secs_f64()
        ));
        report.push_str(&format!("Epochs Completed: {}\n", self.final_state.epoch));
        report.push_str(&format!("Total Steps: {}\n", self.final_state.step));
        report.push_str(&format!("Model Parameters: {}\n\n", self.parameter_count));

        // Performance Metrics
        report.push_str("Performance Metrics:\n");
        if let Some(final_train) = self.final_state.history.last() {
            report.push_str(&format!("- Final Training Loss: {:.6}\n", final_train.loss));
        }

        if let Some(final_val) = self.final_state.val_history.last() {
            report.push_str(&format!("- Final Validation Loss: {:.6}\n", final_val.loss));
            if let Some(accuracy) = final_val.metrics.get("accuracy") {
                report.push_str(&format!(
                    "- Final Validation Accuracy: {:.2}%\n",
                    accuracy * 100.0
                ));
            }
        }

        if let Some(best_loss) = self.best_val_loss {
            report.push_str(&format!("- Best Validation Loss: {best_loss:.6}\n"));
        }

        if let Some(best_acc) = self.best_val_accuracy {
            report.push_str(&format!(
                "- Best Validation Accuracy: {:.4}%\n",
                best_acc * 100.0
            ));
        }

        // Configuration
        report.push_str("\nConfiguration:\n");
        report.push_str(&format!("- Epochs: {}\n", self.config.epochs));
        report.push_str(&format!("- Batch Size: {}\n", self.config.batch_size));
        report.push_str(&format!(
            "- Learning Rate: {:.6}\n",
            self.config.learning_rate
        ));
        report.push_str(&format!(
            "- Early Stopping: {}\n",
            if self.config.use_early_stopping {
                "Enabled"
            } else {
                "Disabled"
            }
        ));
        report.push_str(&format!(
            "- LR Reduction: {}\n",
            if self.config.use_lr_reduction {
                "Enabled"
            } else {
                "Disabled"
            }
        ));

        // Benchmark Results
        if let Some(ref benchmark) = self.benchmark_results {
            report.push_str("\nBenchmark Results:\n");
            report.push_str(&format!(
                "- Forward Pass: {:.2} ms\n",
                benchmark.avg_metrics.forward_time_ms
            ));
            report.push_str(&format!(
                "- Backward Pass: {:.2} ms\n",
                benchmark.avg_metrics.backward_time_ms
            ));
            report.push_str(&format!(
                "- Throughput: {:.1} samples/sec\n",
                benchmark.avg_metrics.throughput_samples_per_sec
            ));
        }

        // Detailed Metrics
        if !self.detailed_metrics.is_empty() {
            report.push_str("\nDetailed Metrics:\n");
            for (metric, value) in &self.detailed_metrics {
                report.push_str(&format!("- {metric}: {value:.6}\n"));
            }
        }

        report.push_str("\n=== END REPORT ===\n");
        report
    }

    /// Export results as JSON (if serialize feature is enabled)
    #[cfg(feature = "serialize")]
    pub fn to_json(&self) -> Result<String> {
        use serde_json;
        serde_json::to_string_pretty(self)
            .map_err(|e| TensorError::invalid_argument(format!("JSON serialization failed: {}", e)))
    }
}

/// Advanced training pipeline that combines all training components
pub struct TrainingPipeline<T> {
    config: TrainingPipelineConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TrainingPipeline<T>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + scirs2_core::num_traits::Signed
        + bytemuck::Pod,
{
    /// Create a new training pipeline with the provided configuration
    pub fn new(config: TrainingPipelineConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a training pipeline with default configuration
    pub fn with_defaults() -> Self {
        Self::new(TrainingPipelineConfig::default())
    }

    /// Train a model using the complete pipeline
    pub fn train<M, O, D>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        train_data: D,
        val_data: Option<D>,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
        model_name: Option<String>,
    ) -> Result<TrainingResults<T>>
    where
        M: Model<T>,
        O: Optimizer<T>,
        D: Iterator<Item = (Tensor<T>, Tensor<T>)> + Clone,
    {
        let training_start = Instant::now();
        let model_name = model_name.unwrap_or_else(|| "Model".to_string());

        if self.config.verbose {
            println!("🚀 Starting training pipeline for: {model_name}");
            println!(
                "Configuration: {} epochs, batch size {}, lr {:.6}",
                self.config.epochs, self.config.batch_size, self.config.learning_rate
            );
        }

        // Set up the trainer with callbacks
        let mut trainer = Trainer::new();
        trainer.set_verbose(self.config.verbose);

        // Add early stopping callback
        if self.config.use_early_stopping {
            let early_stopping = EarlyStopping::new(
                self.config.early_stopping_patience,
                0.0001, // min_delta
                "val_loss".to_string(),
                "min".to_string(),
            );
            trainer.add_callback(Box::new(early_stopping));

            if self.config.verbose {
                println!(
                    "📊 Early stopping enabled (patience: {})",
                    self.config.early_stopping_patience
                );
            }
        }

        // Add learning rate reduction callback
        if self.config.use_lr_reduction {
            let lr_reduction = LearningRateReduction::new(
                "val_loss".to_string(),
                self.config.lr_reduction_factor,
                self.config.lr_reduction_patience,
                0.0001, // min_delta
                0,      // cooldown
                0.0,    // min_lr
                "min".to_string(),
                self.config.verbose,
            );
            trainer.add_callback(Box::new(lr_reduction));

            if self.config.verbose {
                println!(
                    "📉 Learning rate reduction enabled (patience: {}, factor: {:.2})",
                    self.config.lr_reduction_patience, self.config.lr_reduction_factor
                );
            }
        }

        // Add model checkpoint callback
        if self.config.save_checkpoints {
            let checkpoint = ModelCheckpoint::new(
                self.config.checkpoint_path.clone(),
                "val_loss".to_string(),
                "min".to_string(),
                true, // save_best_only
            );
            trainer.add_callback(Box::new(checkpoint));

            if self.config.verbose {
                println!(
                    "💾 Model checkpointing enabled (path: {})",
                    self.config.checkpoint_path
                );
            }
        }

        // Set optimizer learning rate
        optimizer.set_learning_rate(self.config.learning_rate);

        // Count model parameters
        let parameter_count = model
            .parameters()
            .iter()
            .map(|p| p.shape().dims().iter().product::<usize>())
            .sum();

        if self.config.verbose {
            println!("🏗️  Model has {parameter_count} parameters");
            println!("🎯 Beginning training...\n");
        }

        // Run training
        let training_result = trainer.fit(
            model,
            optimizer,
            train_data,
            val_data.clone(),
            self.config.epochs,
            loss_fn,
        );

        let training_duration = training_start.elapsed();

        // Handle training results
        let (final_state, completed_successfully, termination_reason) = match training_result {
            Ok(state) => (state, true, "Training completed successfully".to_string()),
            Err(e) => {
                let empty_state = TrainingState::new();
                (empty_state, false, format!("Training failed: {e}"))
            }
        };

        // Calculate best metrics
        let best_val_loss = final_state
            .val_history
            .iter()
            .map(|m| m.loss)
            .fold(f32::INFINITY, f32::min)
            .finite_or(None);

        let best_val_accuracy = final_state
            .val_history
            .iter()
            .filter_map(|m| m.metrics.get("accuracy"))
            .fold(0.0f32, |acc, &x| acc.max(x))
            .finite_or(None);

        // Run benchmarks if enabled
        let benchmark_results = if self.config.run_benchmarks && completed_successfully {
            if self.config.verbose {
                println!("\n🔬 Running performance benchmarks...");
            }

            let benchmark_config = BenchmarkConfig {
                batch_size: self.config.batch_size,
                warmup_iterations: 5,
                measurement_iterations: 20,
                ..Default::default()
            };

            let benchmark = ModelBenchmark::new(benchmark_config);
            match benchmark.benchmark_model(model, optimizer, loss_fn, model_name.clone()) {
                Ok(results) => {
                    if self.config.verbose {
                        println!("📊 Benchmarks completed successfully");
                    }
                    Some(results)
                }
                Err(e) => {
                    if self.config.verbose {
                        println!("⚠️  Benchmark failed: {e}");
                    }
                    None
                }
            }
        } else {
            None
        };

        // Compute detailed metrics if enabled
        let detailed_metrics = if self.config.compute_detailed_metrics && completed_successfully {
            self.compute_detailed_metrics(&final_state, val_data)
        } else {
            HashMap::new()
        };

        let results = TrainingResults::<T> {
            final_state,
            training_duration,
            best_val_loss,
            best_val_accuracy,
            parameter_count,
            benchmark_results,
            detailed_metrics,
            config: self.config.clone(),
            completed_successfully,
            termination_reason,
            _phantom: std::marker::PhantomData,
        };

        if self.config.verbose {
            println!("\n{}", results.generate_report());
        }

        Ok(results)
    }

    /// Compute detailed metrics from training results
    fn compute_detailed_metrics<D>(
        &self,
        state: &TrainingState,
        val_data: Option<D>,
    ) -> HashMap<String, f32>
    where
        D: Iterator<Item = (Tensor<T>, Tensor<T>)> + Clone,
    {
        let mut metrics = HashMap::new();

        // Training loss statistics
        if !state.history.is_empty() {
            let losses: Vec<f32> = state.history.iter().map(|m| m.loss).collect();
            let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;
            let min_loss = losses.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
            let max_loss = losses.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

            metrics.insert("avg_training_loss".to_string(), avg_loss);
            metrics.insert("min_training_loss".to_string(), min_loss);
            metrics.insert("max_training_loss".to_string(), max_loss);

            // Loss improvement
            if losses.len() > 1 {
                let initial_loss = losses[0];
                let final_loss = losses[losses.len() - 1];
                let improvement = (initial_loss - final_loss) / initial_loss;
                metrics.insert("training_loss_improvement".to_string(), improvement);
            }
        }

        // Validation loss statistics
        if !state.val_history.is_empty() {
            let val_losses: Vec<f32> = state.val_history.iter().map(|m| m.loss).collect();
            let avg_val_loss = val_losses.iter().sum::<f32>() / val_losses.len() as f32;
            let min_val_loss = val_losses.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));

            metrics.insert("avg_validation_loss".to_string(), avg_val_loss);
            metrics.insert("min_validation_loss".to_string(), min_val_loss);

            // Training efficiency (ratio of validation to training loss)
            if let Some(latest_train) = state.history.last() {
                let efficiency = latest_train.loss / avg_val_loss;
                metrics.insert("training_efficiency".to_string(), efficiency);
            }
        }

        // Convergence metrics
        if state.history.len() > 10 {
            let recent_losses: Vec<f32> = state
                .history
                .iter()
                .rev()
                .take(10)
                .map(|m| m.loss)
                .collect();

            let variance = self.calculate_variance(&recent_losses);
            metrics.insert("recent_loss_variance".to_string(), variance);
        }

        metrics
    }

    /// Calculate variance for a slice of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }
}

/// Helper trait for Option<f32> to handle infinity values
trait FiniteOrNone {
    fn finite_or(self, default: Option<f32>) -> Option<f32>;
}

impl FiniteOrNone for f32 {
    fn finite_or(self, default: Option<f32>) -> Option<f32> {
        if self.is_finite() && self != f32::INFINITY && self != f32::NEG_INFINITY {
            Some(self)
        } else {
            default
        }
    }
}

/// Utility functions for quick training setups
pub mod quick_train {
    use super::*;
    use crate::loss::mse;
    use crate::optimizers::{Adam, SGD};

    /// Quick training setup with Adam optimizer and MSE loss
    pub fn train_with_adam<M, D>(
        model: &mut M,
        train_data: D,
        val_data: Option<D>,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<TrainingResults<f32>>
    where
        M: Model<f32>,
        D: Iterator<Item = (Tensor<f32>, Tensor<f32>)> + Clone,
    {
        let mut optimizer = Adam::new(learning_rate);
        let config = TrainingPipelineConfig::default()
            .with_epochs(epochs)
            .with_learning_rate(learning_rate);

        let pipeline = TrainingPipeline::new(config);
        pipeline.train(model, &mut optimizer, train_data, val_data, mse, None)
    }

    /// Quick training setup with SGD optimizer and MSE loss
    pub fn train_with_sgd<M, D>(
        model: &mut M,
        train_data: D,
        val_data: Option<D>,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<TrainingResults<f32>>
    where
        M: Model<f32>,
        D: Iterator<Item = (Tensor<f32>, Tensor<f32>)> + Clone,
    {
        let mut optimizer = SGD::new(learning_rate);
        let config = TrainingPipelineConfig::default()
            .with_epochs(epochs)
            .with_learning_rate(learning_rate);

        let pipeline = TrainingPipeline::new(config);
        pipeline.train(model, &mut optimizer, train_data, val_data, mse, None)
    }

    /// Quick experimentation setup (fast training for testing)
    pub fn quick_experiment<M, O, D>(
        model: &mut M,
        optimizer: &mut O,
        train_data: D,
        val_data: Option<D>,
        loss_fn: fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>>,
    ) -> Result<TrainingResults<f32>>
    where
        M: Model<f32>,
        O: Optimizer<f32>,
        D: Iterator<Item = (Tensor<f32>, Tensor<f32>)> + Clone,
    {
        let config = TrainingPipelineConfig::quick();
        let pipeline = TrainingPipeline::new(config);
        pipeline.train(
            model,
            optimizer,
            train_data,
            val_data,
            loss_fn,
            Some("QuickExperiment".to_string()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::sgd::SGD;
    use crate::{layers::dense::Dense, Sequential};

    #[test]
    fn test_training_pipeline_config_creation() {
        let config = TrainingPipelineConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert!(config.use_early_stopping);
    }

    #[test]
    fn test_training_pipeline_config_builders() {
        let config = TrainingPipelineConfig::default()
            .with_epochs(50)
            .with_batch_size(64)
            .with_learning_rate(0.01)
            .with_early_stopping(15)
            .with_benchmarks(true);

        assert_eq!(config.epochs, 50);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.early_stopping_patience, 15);
        assert!(config.run_benchmarks);
    }

    #[test]
    fn test_training_pipeline_presets() {
        let quick_config = TrainingPipelineConfig::quick();
        assert_eq!(quick_config.epochs, 10);
        assert!(!quick_config.use_early_stopping);
        assert!(!quick_config.save_checkpoints);

        let prod_config = TrainingPipelineConfig::production();
        assert_eq!(prod_config.epochs, 200);
        assert!(prod_config.use_early_stopping);
        assert!(prod_config.run_benchmarks);

        let large_config = TrainingPipelineConfig::large_model();
        assert_eq!(large_config.batch_size, 16);
        assert_eq!(large_config.validation_frequency, 5);
        assert!(!large_config.run_benchmarks);
    }

    #[test]
    fn test_training_pipeline_creation() {
        let config = TrainingPipelineConfig::default();
        let pipeline = TrainingPipeline::<f32>::new(config.clone());
        assert_eq!(pipeline.config.epochs, config.epochs);
    }

    #[test]
    fn test_variance_calculation() {
        let pipeline = TrainingPipeline::<f32>::with_defaults();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = pipeline.calculate_variance(&values);
        assert!((variance - 2.0).abs() < 0.01); // Expected variance is 2.0

        let empty_values = vec![];
        let empty_variance = pipeline.calculate_variance(&empty_values);
        assert_eq!(empty_variance, 0.0);

        let single_value = vec![1.0];
        let single_variance = pipeline.calculate_variance(&single_value);
        assert_eq!(single_variance, 0.0);
    }

    #[test]
    fn test_finite_or_none_trait() {
        assert_eq!(1.0f32.finite_or(None), Some(1.0));
        assert_eq!(f32::INFINITY.finite_or(None), None);
        assert_eq!(f32::NEG_INFINITY.finite_or(None), None);
        assert_eq!(f32::NAN.finite_or(Some(0.0)), Some(0.0));
    }

    #[test]
    fn test_training_results_report_generation() {
        let config = TrainingPipelineConfig::default();
        let mut state = TrainingState::new();

        // Add some mock training history
        state.history.push(TrainingMetrics {
            epoch: 0,
            step: 10,
            loss: 1.0,
            metrics: HashMap::new(),
        });

        let mut val_metrics = HashMap::new();
        val_metrics.insert("accuracy".to_string(), 0.85);
        state.val_history.push(TrainingMetrics {
            epoch: 0,
            step: 10,
            loss: 0.8,
            metrics: val_metrics,
        });

        let results = TrainingResults::<f32> {
            final_state: state,
            training_duration: Duration::from_secs(300),
            best_val_loss: Some(0.75),
            best_val_accuracy: Some(0.90),
            parameter_count: 1000,
            benchmark_results: None,
            detailed_metrics: HashMap::new(),
            config,
            completed_successfully: true,
            termination_reason: "Training completed successfully".to_string(),
            _phantom: std::marker::PhantomData,
        };

        let report = results.generate_report();
        assert!(report.contains("✅ Completed Successfully"));
        assert!(report.contains("Training Duration: 300.00 seconds"));
        assert!(report.contains("Model Parameters: 1000"));
        assert!(report.contains("Final Validation Accuracy: 85.00%"));
        assert!(report.contains("Best Validation Loss: 0.750000"));
    }
}
