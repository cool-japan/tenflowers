use crate::{model::Model, optimizers::Optimizer};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tenflowers_core::{Result, Tensor, TensorError};

/// Performance metrics for model benchmarking
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkMetrics {
    /// Forward pass time in milliseconds
    pub forward_time_ms: f32,
    /// Backward pass time in milliseconds
    pub backward_time_ms: f32,
    /// Total training step time in milliseconds
    pub total_step_time_ms: f32,
    /// Memory usage in MB (if available)
    pub memory_usage_mb: Option<f32>,
    /// Throughput in samples per second
    pub throughput_samples_per_sec: f32,
    /// Loss value
    pub loss: f32,
    /// Model accuracy (if available)
    pub accuracy: Option<f32>,
}

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before measuring
    pub warmup_iterations: usize,
    /// Number of iterations to measure
    pub measurement_iterations: usize,
    /// Batch size for benchmarking
    pub batch_size: usize,
    /// Input dimensions
    pub input_shape: Vec<usize>,
    /// Number of classes (for classification tasks)  
    pub num_classes: usize,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Whether to include validation metrics
    pub include_validation: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            batch_size: 32,
            input_shape: vec![3, 224, 224], // Default ImageNet-like input
            num_classes: 1000,
            measure_memory: false,
            include_validation: true,
        }
    }
}

impl BenchmarkConfig {
    /// Create a configuration for image classification benchmarking
    pub fn image_classification(
        batch_size: usize,
        input_shape: Vec<usize>,
        num_classes: usize,
    ) -> Self {
        Self {
            batch_size,
            input_shape,
            num_classes,
            ..Default::default()
        }
    }

    /// Create a configuration for NLP benchmarking
    pub fn nlp(batch_size: usize, sequence_length: usize, vocab_size: usize) -> Self {
        Self {
            batch_size,
            input_shape: vec![sequence_length],
            num_classes: vocab_size,
            ..Default::default()
        }
    }

    /// Builder method to set warmup iterations
    pub fn with_warmup(mut self, warmup_iterations: usize) -> Self {
        self.warmup_iterations = warmup_iterations;
        self
    }

    /// Builder method to set measurement iterations
    pub fn with_measurements(mut self, measurement_iterations: usize) -> Self {
        self.measurement_iterations = measurement_iterations;
        self
    }

    /// Builder method to enable memory measurement
    pub fn with_memory_measurement(mut self, measure_memory: bool) -> Self {
        self.measure_memory = measure_memory;
        self
    }
}

/// Comprehensive model benchmark results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkResults {
    /// Average metrics across all measurement iterations
    pub avg_metrics: BenchmarkMetrics,
    /// Standard deviation of metrics
    pub std_metrics: BenchmarkMetrics,
    /// Minimum observed metrics
    pub min_metrics: BenchmarkMetrics,
    /// Maximum observed metrics
    pub max_metrics: BenchmarkMetrics,
    /// Configuration used for benchmarking
    pub config: BenchmarkConfig,
    /// Model name/identifier
    pub model_name: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl BenchmarkResults {
    /// Generate a human-readable summary report
    pub fn summary_report(&self) -> String {
        format!(
            "Model: {}\n\
             Batch Size: {}\n\
             Input Shape: {:?}\n\
             \n\
             Performance Metrics (avg ± std):\n\
             - Forward Pass: {:.2} ± {:.2} ms\n\
             - Backward Pass: {:.2} ± {:.2} ms\n\
             - Total Step: {:.2} ± {:.2} ms\n\
             - Throughput: {:.1} ± {:.1} samples/sec\n\
             - Loss: {:.4} ± {:.4}\n\
             {}\n\
             Memory Usage: {}\n",
            self.model_name,
            self.config.batch_size,
            self.config.input_shape,
            self.avg_metrics.forward_time_ms,
            self.std_metrics.forward_time_ms,
            self.avg_metrics.backward_time_ms,
            self.std_metrics.backward_time_ms,
            self.avg_metrics.total_step_time_ms,
            self.std_metrics.total_step_time_ms,
            self.avg_metrics.throughput_samples_per_sec,
            self.std_metrics.throughput_samples_per_sec,
            self.avg_metrics.loss,
            self.std_metrics.loss,
            if let Some(acc) = self.avg_metrics.accuracy {
                format!("- Accuracy: {:.2}%", acc * 100.0)
            } else {
                String::new()
            },
            if let Some(mem) = self.avg_metrics.memory_usage_mb {
                format!("{mem:.1} MB")
            } else {
                "Not measured".to_string()
            }
        )
    }

    /// Export results as JSON string
    #[cfg(feature = "serialize")]
    pub fn to_json(&self) -> Result<String> {
        let serializable = SerializableBenchmarkResults {
            avg_metrics: SerializableBenchmarkMetrics {
                forward_time_ms: self.avg_metrics.forward_time_ms,
                backward_time_ms: self.avg_metrics.backward_time_ms,
                total_step_time_ms: self.avg_metrics.total_step_time_ms,
                memory_usage_mb: self.avg_metrics.memory_usage_mb,
                throughput_samples_per_sec: self.avg_metrics.throughput_samples_per_sec,
                loss: self.avg_metrics.loss,
                accuracy: self.avg_metrics.accuracy,
            },
            std_metrics: SerializableBenchmarkMetrics {
                forward_time_ms: self.std_metrics.forward_time_ms,
                backward_time_ms: self.std_metrics.backward_time_ms,
                total_step_time_ms: self.std_metrics.total_step_time_ms,
                memory_usage_mb: self.std_metrics.memory_usage_mb,
                throughput_samples_per_sec: self.std_metrics.throughput_samples_per_sec,
                loss: self.std_metrics.loss,
                accuracy: self.std_metrics.accuracy,
            },
            min_metrics: SerializableBenchmarkMetrics {
                forward_time_ms: self.min_metrics.forward_time_ms,
                backward_time_ms: self.min_metrics.backward_time_ms,
                total_step_time_ms: self.min_metrics.total_step_time_ms,
                memory_usage_mb: self.min_metrics.memory_usage_mb,
                throughput_samples_per_sec: self.min_metrics.throughput_samples_per_sec,
                loss: self.min_metrics.loss,
                accuracy: self.min_metrics.accuracy,
            },
            max_metrics: SerializableBenchmarkMetrics {
                forward_time_ms: self.max_metrics.forward_time_ms,
                backward_time_ms: self.max_metrics.backward_time_ms,
                total_step_time_ms: self.max_metrics.total_step_time_ms,
                memory_usage_mb: self.max_metrics.memory_usage_mb,
                throughput_samples_per_sec: self.max_metrics.throughput_samples_per_sec,
                loss: self.max_metrics.loss,
                accuracy: self.max_metrics.accuracy,
            },
            config: SerializableBenchmarkConfig {
                warmup_iterations: self.config.warmup_iterations,
                measurement_iterations: self.config.measurement_iterations,
                batch_size: self.config.batch_size,
                input_shape: self.config.input_shape.clone(),
                num_classes: self.config.num_classes,
                measure_memory: self.config.measure_memory,
                include_validation: self.config.include_validation,
            },
            model_name: self.model_name.clone(),
            metadata: self.metadata.clone(),
        };

        serde_json::to_string_pretty(&serializable)
            .map_err(|e| TensorError::invalid_argument(format!("JSON serialization failed: {}", e)))
    }
}

/// Model benchmarking utility
pub struct ModelBenchmark<T> {
    config: BenchmarkConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ModelBenchmark<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + num_traits::Zero
        + num_traits::One,
{
    /// Create a new benchmark instance
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Benchmark a model with the given optimizer and loss function
    pub fn benchmark_model<M, O>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
        model_name: String,
    ) -> Result<BenchmarkResults>
    where
        M: Model<T>,
        O: Optimizer<T>,
    {
        println!("Starting benchmark for model: {model_name}");
        println!(
            "Configuration: batch_size={}, iterations={}",
            self.config.batch_size, self.config.measurement_iterations
        );

        // Generate synthetic data for benchmarking
        let (input_data, target_data) = self.generate_synthetic_data()?;

        model.set_training(true);

        // Warmup phase
        println!(
            "Warming up ({} iterations)...",
            self.config.warmup_iterations
        );
        for _ in 0..self.config.warmup_iterations {
            self.run_single_iteration(model, optimizer, &input_data, &target_data, loss_fn)?;
        }

        // Measurement phase
        println!(
            "Measuring performance ({} iterations)...",
            self.config.measurement_iterations
        );
        let mut measurements = Vec::new();

        for i in 0..self.config.measurement_iterations {
            if i % 20 == 0 {
                println!("Progress: {}/{}", i, self.config.measurement_iterations);
            }

            let metrics = self.measure_single_iteration(
                model,
                optimizer,
                &input_data,
                &target_data,
                loss_fn,
            )?;
            measurements.push(metrics);
        }

        // Calculate statistics
        let results = self.calculate_statistics(measurements, model_name)?;

        println!("Benchmark completed!");
        println!("{}", results.summary_report());

        Ok(results)
    }

    /// Generate synthetic data for benchmarking
    fn generate_synthetic_data(&self) -> Result<(Tensor<T>, Tensor<T>)> {
        use rand_distr::{Distribution, StandardNormal};

        // Create input tensor with batch dimension
        let mut input_shape = vec![self.config.batch_size];
        input_shape.extend_from_slice(&self.config.input_shape);

        // Generate random normal data for input
        let total_elements: usize = input_shape.iter().product();
        let mut rng = rand::rng();
        let mut input_data = Vec::with_capacity(total_elements);

        for _ in 0..total_elements {
            let val: f64 = StandardNormal.sample(&mut rng);
            input_data.push(T::from(val).unwrap_or(T::zero()));
        }

        let input_tensor = Tensor::from_vec(input_data, &input_shape)?;

        // Create target tensor (one-hot encoded for classification)
        let target_shape = vec![self.config.batch_size, self.config.num_classes];
        let target_elements = target_shape.iter().product::<usize>();
        let mut target_data = vec![T::zero(); target_elements];

        // Fill with one-hot vectors
        for i in 0..self.config.batch_size {
            let class_idx = i % self.config.num_classes; // Simple deterministic assignment
            let row_start = i * self.config.num_classes;
            if row_start + class_idx < target_data.len() {
                target_data[row_start + class_idx] = T::one();
            }
        }

        let target_tensor = Tensor::from_vec(target_data, &target_shape)?;

        Ok((input_tensor, target_tensor))
    }

    /// Run a single training iteration without measuring
    fn run_single_iteration<M, O>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        input: &Tensor<T>,
        target: &Tensor<T>,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    ) -> Result<()>
    where
        M: Model<T>,
        O: Optimizer<T>,
    {
        model.zero_grad();
        let predictions = model.forward(input)?;
        let _loss = loss_fn(&predictions, target)?;

        // Simple gradient computation (simplified for benchmarking)
        self.compute_gradients_simple(model, input, target, loss_fn)?;
        optimizer.step(model)?;

        Ok(())
    }

    /// Measure a single training iteration
    fn measure_single_iteration<M, O>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        input: &Tensor<T>,
        target: &Tensor<T>,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    ) -> Result<BenchmarkMetrics>
    where
        M: Model<T>,
        O: Optimizer<T>,
    {
        let total_start = Instant::now();

        // Zero gradients
        model.zero_grad();

        // Forward pass timing
        let forward_start = Instant::now();
        let predictions = model.forward(input)?;
        let loss = loss_fn(&predictions, target)?;
        let forward_time = forward_start.elapsed();

        // Extract loss value
        let loss_value = loss.get(&[]).unwrap_or(T::zero());

        // Backward pass timing
        let backward_start = Instant::now();
        self.compute_gradients_simple(model, input, target, loss_fn)?;
        let backward_time = backward_start.elapsed();

        // Optimizer step
        optimizer.step(model)?;

        let total_time = total_start.elapsed();

        // Calculate accuracy if this is a classification task
        let accuracy = self.calculate_accuracy(&predictions, target).ok();

        // Calculate throughput
        let throughput = self.config.batch_size as f32 / total_time.as_secs_f32();

        // Memory usage (simplified - would need platform-specific implementation)
        let memory_usage = if self.config.measure_memory {
            Some(self.estimate_memory_usage())
        } else {
            None
        };

        Ok(BenchmarkMetrics {
            forward_time_ms: forward_time.as_secs_f32() * 1000.0,
            backward_time_ms: backward_time.as_secs_f32() * 1000.0,
            total_step_time_ms: total_time.as_secs_f32() * 1000.0,
            memory_usage_mb: memory_usage,
            throughput_samples_per_sec: throughput,
            loss: loss_value.to_f32().unwrap_or(0.0),
            accuracy,
        })
    }

    /// Simple gradient computation for benchmarking
    fn compute_gradients_simple<M>(
        &self,
        model: &mut M,
        input: &Tensor<T>,
        target: &Tensor<T>,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    ) -> Result<()>
    where
        M: Model<T>,
    {
        use rand_distr::{Distribution, StandardNormal};

        // Simplified gradient computation for benchmarking purposes
        // In a real implementation, this would use proper automatic differentiation
        let mut params = model.parameters_mut();
        let mut rng = rand::rng();

        for param in params.iter_mut() {
            if param.requires_grad() {
                // Create a simple gradient (normally computed by autograd)
                let grad_shape = param.shape().dims();
                let total_elements: usize = grad_shape.iter().product();
                let mut grad_data = Vec::with_capacity(total_elements);

                for _ in 0..total_elements {
                    let val: f64 = StandardNormal.sample(&mut rng);
                    let scaled_val = val * 0.001f64;
                    grad_data.push(T::from(scaled_val).unwrap_or(T::zero()));
                }

                let grad = Tensor::from_vec(grad_data, grad_shape)?;
                param.set_grad(Some(grad));
            }
        }
        Ok(())
    }

    /// Calculate classification accuracy
    fn calculate_accuracy(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32> {
        // Simplified accuracy calculation
        let pred_data = predictions.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("Cannot access prediction data".to_string())
        })?;
        let target_data = targets.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("Cannot access target data".to_string())
        })?;

        let batch_size = self.config.batch_size;
        let num_classes = self.config.num_classes;
        let mut correct = 0;

        for i in 0..batch_size {
            // Find predicted class (argmax)
            let pred_start = i * num_classes;
            let mut max_idx = 0;
            let mut max_val = pred_data[pred_start].to_f32().unwrap_or(f32::NEG_INFINITY);

            for j in 1..num_classes {
                let val = pred_data[pred_start + j]
                    .to_f32()
                    .unwrap_or(f32::NEG_INFINITY);
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }

            // Find target class (argmax)
            let target_start = i * num_classes;
            let mut target_idx = 0;
            let mut target_max = target_data[target_start].to_f32().unwrap_or(0.0);

            for j in 1..num_classes {
                let val = target_data[target_start + j].to_f32().unwrap_or(0.0);
                if val > target_max {
                    target_max = val;
                    target_idx = j;
                }
            }

            if max_idx == target_idx {
                correct += 1;
            }
        }

        Ok(correct as f32 / batch_size as f32)
    }

    /// Estimate memory usage (simplified)
    fn estimate_memory_usage(&self) -> f32 {
        // Simplified memory estimation - would need platform-specific implementation
        // This is just a placeholder
        let input_size = self.config.batch_size * self.config.input_shape.iter().product::<usize>();
        let estimated_mb = (input_size * std::mem::size_of::<T>()) as f32 / (1024.0 * 1024.0);
        estimated_mb * 4.0 // Rough estimate including gradients and intermediate results
    }

    /// Calculate statistics from measurements
    fn calculate_statistics(
        &self,
        measurements: Vec<BenchmarkMetrics>,
        model_name: String,
    ) -> Result<BenchmarkResults> {
        let n = measurements.len() as f32;

        // Calculate averages
        let avg_forward = measurements.iter().map(|m| m.forward_time_ms).sum::<f32>() / n;
        let avg_backward = measurements.iter().map(|m| m.backward_time_ms).sum::<f32>() / n;
        let avg_total = measurements
            .iter()
            .map(|m| m.total_step_time_ms)
            .sum::<f32>()
            / n;
        let avg_throughput = measurements
            .iter()
            .map(|m| m.throughput_samples_per_sec)
            .sum::<f32>()
            / n;
        let avg_loss = measurements.iter().map(|m| m.loss).sum::<f32>() / n;
        let avg_accuracy = measurements.iter().filter_map(|m| m.accuracy).sum::<f32>()
            / measurements.iter().filter(|m| m.accuracy.is_some()).count() as f32;

        // Calculate standard deviations
        let std_forward = self.calculate_std(&measurements, |m| m.forward_time_ms, avg_forward);
        let std_backward = self.calculate_std(&measurements, |m| m.backward_time_ms, avg_backward);
        let std_total = self.calculate_std(&measurements, |m| m.total_step_time_ms, avg_total);
        let std_throughput = self.calculate_std(
            &measurements,
            |m| m.throughput_samples_per_sec,
            avg_throughput,
        );
        let std_loss = self.calculate_std(&measurements, |m| m.loss, avg_loss);

        // Calculate min/max
        let min_forward = measurements
            .iter()
            .map(|m| m.forward_time_ms)
            .fold(f32::INFINITY, f32::min);
        let max_forward = measurements
            .iter()
            .map(|m| m.forward_time_ms)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_backward = measurements
            .iter()
            .map(|m| m.backward_time_ms)
            .fold(f32::INFINITY, f32::min);
        let max_backward = measurements
            .iter()
            .map(|m| m.backward_time_ms)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_total = measurements
            .iter()
            .map(|m| m.total_step_time_ms)
            .fold(f32::INFINITY, f32::min);
        let max_total = measurements
            .iter()
            .map(|m| m.total_step_time_ms)
            .fold(f32::NEG_INFINITY, f32::max);

        let avg_memory = measurements
            .iter()
            .filter_map(|m| m.memory_usage_mb)
            .sum::<f32>()
            / measurements
                .iter()
                .filter(|m| m.memory_usage_mb.is_some())
                .count()
                .max(1) as f32;

        Ok(BenchmarkResults {
            avg_metrics: BenchmarkMetrics {
                forward_time_ms: avg_forward,
                backward_time_ms: avg_backward,
                total_step_time_ms: avg_total,
                memory_usage_mb: if self.config.measure_memory {
                    Some(avg_memory)
                } else {
                    None
                },
                throughput_samples_per_sec: avg_throughput,
                loss: avg_loss,
                accuracy: if avg_accuracy.is_finite() {
                    Some(avg_accuracy)
                } else {
                    None
                },
            },
            std_metrics: BenchmarkMetrics {
                forward_time_ms: std_forward,
                backward_time_ms: std_backward,
                total_step_time_ms: std_total,
                memory_usage_mb: None,
                throughput_samples_per_sec: std_throughput,
                loss: std_loss,
                accuracy: None,
            },
            min_metrics: BenchmarkMetrics {
                forward_time_ms: min_forward,
                backward_time_ms: min_backward,
                total_step_time_ms: min_total,
                memory_usage_mb: None,
                throughput_samples_per_sec: 0.0,
                loss: 0.0,
                accuracy: None,
            },
            max_metrics: BenchmarkMetrics {
                forward_time_ms: max_forward,
                backward_time_ms: max_backward,
                total_step_time_ms: max_total,
                memory_usage_mb: None,
                throughput_samples_per_sec: 0.0,
                loss: 0.0,
                accuracy: None,
            },
            config: self.config.clone(),
            model_name,
            metadata: HashMap::new(),
        })
    }

    /// Calculate standard deviation for a given field
    fn calculate_std<F>(&self, measurements: &[BenchmarkMetrics], field_getter: F, mean: f32) -> f32
    where
        F: Fn(&BenchmarkMetrics) -> f32,
    {
        let variance = measurements
            .iter()
            .map(|m| {
                let diff = field_getter(m) - mean;
                diff * diff
            })
            .sum::<f32>()
            / measurements.len() as f32;

        variance.sqrt()
    }
}

/// Utility function to compare multiple models
pub fn compare_models<T, M, O>(
    models: Vec<(&str, &mut M)>,
    optimizers: Vec<&mut O>,
    loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    config: BenchmarkConfig,
) -> Result<Vec<BenchmarkResults>>
where
    T: Clone
        + Default
        + num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + num_traits::Zero
        + num_traits::One,
    M: Model<T>,
    O: Optimizer<T>,
{
    let benchmark = ModelBenchmark::new(config);
    let mut results = Vec::new();

    for ((name, model), optimizer) in models.into_iter().zip(optimizers.into_iter()) {
        println!("\n--- Benchmarking {name} ---");
        let result = benchmark.benchmark_model(model, optimizer, loss_fn, name.to_string())?;
        results.push(result);
    }

    // Print comparison summary
    println!("\n=== COMPARISON SUMMARY ===");
    for result in &results {
        println!(
            "{}: {:.2} ms/step, {:.1} samples/sec",
            result.model_name,
            result.avg_metrics.total_step_time_ms,
            result.avg_metrics.throughput_samples_per_sec
        );
    }

    Ok(results)
}

// Serde support is already imported above

#[cfg(feature = "serialize")]
#[derive(Serialize, Deserialize)]
struct SerializableBenchmarkResults {
    avg_metrics: SerializableBenchmarkMetrics,
    std_metrics: SerializableBenchmarkMetrics,
    min_metrics: SerializableBenchmarkMetrics,
    max_metrics: SerializableBenchmarkMetrics,
    config: SerializableBenchmarkConfig,
    model_name: String,
    metadata: HashMap<String, String>,
}

#[cfg(feature = "serialize")]
#[derive(Serialize, Deserialize)]
struct SerializableBenchmarkMetrics {
    forward_time_ms: f32,
    backward_time_ms: f32,
    total_step_time_ms: f32,
    memory_usage_mb: Option<f32>,
    throughput_samples_per_sec: f32,
    loss: f32,
    accuracy: Option<f32>,
}

#[cfg(feature = "serialize")]
#[derive(Serialize, Deserialize)]
struct SerializableBenchmarkConfig {
    warmup_iterations: usize,
    measurement_iterations: usize,
    batch_size: usize,
    input_shape: Vec<usize>,
    num_classes: usize,
    measure_memory: bool,
    include_validation: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::sgd::SGD;
    use crate::{layers::dense::Dense, Sequential};

    #[test]
    fn test_benchmark_config_creation() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.measurement_iterations, 100);
    }

    #[test]
    fn test_benchmark_config_builders() {
        let config = BenchmarkConfig::image_classification(64, vec![3, 256, 256], 10)
            .with_warmup(5)
            .with_measurements(50)
            .with_memory_measurement(true);

        assert_eq!(config.batch_size, 64);
        assert_eq!(config.input_shape, vec![3, 256, 256]);
        assert_eq!(config.num_classes, 10);
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 50);
        assert!(config.measure_memory);
    }

    #[test]
    fn test_nlp_config() {
        let config = BenchmarkConfig::nlp(16, 512, 50000);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.input_shape, vec![512]);
        assert_eq!(config.num_classes, 50000);
    }

    #[test]
    fn test_model_benchmark_creation() {
        let config = BenchmarkConfig::default();
        let benchmark = ModelBenchmark::<f32>::new(config);
        assert_eq!(benchmark.config.batch_size, 32);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let config = BenchmarkConfig::image_classification(4, vec![3, 32, 32], 10);
        let benchmark = ModelBenchmark::<f32>::new(config);

        let (input, target) = benchmark.generate_synthetic_data().unwrap();

        // Check input shape: [batch_size, channels, height, width]
        assert_eq!(input.shape().dims(), &[4, 3, 32, 32]);

        // Check target shape: [batch_size, num_classes]
        assert_eq!(target.shape().dims(), &[4, 10]);
    }

    #[test]
    fn test_statistics_calculation() {
        let config = BenchmarkConfig::default();
        let benchmark = ModelBenchmark::<f32>::new(config);

        let measurements = vec![
            BenchmarkMetrics {
                forward_time_ms: 10.0,
                backward_time_ms: 20.0,
                total_step_time_ms: 30.0,
                memory_usage_mb: Some(100.0),
                throughput_samples_per_sec: 1000.0,
                loss: 0.5,
                accuracy: Some(0.8),
            },
            BenchmarkMetrics {
                forward_time_ms: 12.0,
                backward_time_ms: 18.0,
                total_step_time_ms: 30.0,
                memory_usage_mb: Some(110.0),
                throughput_samples_per_sec: 1000.0,
                loss: 0.4,
                accuracy: Some(0.9),
            },
        ];

        let results = benchmark
            .calculate_statistics(measurements, "TestModel".to_string())
            .unwrap();

        assert_eq!(results.model_name, "TestModel");
        assert_eq!(results.avg_metrics.forward_time_ms, 11.0);
        assert_eq!(results.avg_metrics.backward_time_ms, 19.0);
        assert_eq!(results.avg_metrics.total_step_time_ms, 30.0);
        assert_eq!(results.avg_metrics.loss, 0.45);
    }

    #[test]
    fn test_benchmark_results_summary() {
        let config = BenchmarkConfig::default();
        let results = BenchmarkResults {
            avg_metrics: BenchmarkMetrics {
                forward_time_ms: 10.5,
                backward_time_ms: 15.2,
                total_step_time_ms: 25.7,
                memory_usage_mb: Some(256.0),
                throughput_samples_per_sec: 1243.5,
                loss: 0.1234,
                accuracy: Some(0.9567),
            },
            std_metrics: BenchmarkMetrics {
                forward_time_ms: 0.5,
                backward_time_ms: 1.2,
                total_step_time_ms: 1.7,
                memory_usage_mb: None,
                throughput_samples_per_sec: 25.3,
                loss: 0.0123,
                accuracy: None,
            },
            min_metrics: BenchmarkMetrics {
                forward_time_ms: 10.0,
                backward_time_ms: 14.0,
                total_step_time_ms: 24.0,
                memory_usage_mb: None,
                throughput_samples_per_sec: 0.0,
                loss: 0.0,
                accuracy: None,
            },
            max_metrics: BenchmarkMetrics {
                forward_time_ms: 11.0,
                backward_time_ms: 16.4,
                total_step_time_ms: 27.4,
                memory_usage_mb: None,
                throughput_samples_per_sec: 0.0,
                loss: 0.0,
                accuracy: None,
            },
            config,
            model_name: "ResNet18".to_string(),
            metadata: HashMap::new(),
        };

        let summary = results.summary_report();
        assert!(summary.contains("ResNet18"));
        assert!(summary.contains("10.50 ± 0.50 ms"));
        assert!(summary.contains("1243.5 ± 25.3 samples/sec"));
        assert!(summary.contains("95.67%"));
        assert!(summary.contains("256.0 MB"));
    }
}
