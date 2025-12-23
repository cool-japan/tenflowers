//! # Automatic Mixed Precision (AMP) Policy for Gradient Computation
//!
//! This module provides comprehensive automatic mixed precision support for gradient computation,
//! building on the core mixed precision functionality from tenflowers-core and integrating
//! with the GradientTape for seamless automatic differentiation.
//!
//! ## Features
//!
//! - **Dynamic Loss Scaling**: Adaptive loss scaling with sophisticated overflow detection and recovery
//! - **Granular Control**: Operation-level precision policy with customizable whitelist/blacklist
//! - **GradientTape Integration**: Seamless integration with GradientTape for mixed precision training
//! - **Stability Monitoring**: Comprehensive stability metrics and anomaly detection
//! - **Performance Tracking**: Track precision transitions and their impact on performance
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::{GradientTape, AMPPolicy, AMPConfig};
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure AMP policy
//! let amp_config = AMPConfig::default()
//!     .with_initial_scale(65536.0)
//!     .with_growth_interval(2000)
//!     .with_backoff_factor(0.5);
//!
//! let mut amp_policy = AMPPolicy::new(amp_config);
//!
//! // Use with GradientTape
//! let mut tape = GradientTape::new();
//!
//! // Forward pass with mixed precision
//! let loss = Tensor::<f32>::ones(&[1]);
//! let scaled_loss = amp_policy.scale_loss(&loss)?;
//!
//! // Backward pass
//! let mut gradients = vec![Tensor::<f32>::ones(&[10, 10])];
//! let should_step = amp_policy.unscale_and_check(&mut gradients)?;
//!
//! if should_step {
//!     // Apply gradients (no overflow detected)
//! } else {
//!     // Skip step due to overflow
//! }
//!
//! // Get stability metrics
//! let metrics = amp_policy.get_stability_metrics();
//! println!("Overflow rate: {:.2}%", metrics.overflow_rate * 100.0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Dynamic Loss Scaling
//!
//! The AMP policy implements sophisticated dynamic loss scaling that:
//!
//! 1. **Starts with a high initial scale** (default 2^16) to prevent underflow
//! 2. **Detects gradient overflow** (inf/nan values) after backpropagation
//! 3. **Backs off on overflow** by reducing scale (typically by 0.5x)
//! 4. **Grows after stability** by increasing scale after N stable steps
//! 5. **Adapts to model behavior** with configurable growth intervals and backoff factors
//!
//! ## Operation-Level Control
//!
//! Some operations are numerically sensitive and should always use FP32:
//!
//! - **Normalization**: BatchNorm, LayerNorm, GroupNorm
//! - **Softmax**: Softmax, LogSoftmax
//! - **Loss Functions**: CrossEntropy, KLDivergence
//! - **Transcendental**: Exp, Log, Sqrt, Pow
//!
//! The policy automatically enforces FP32 for these operations while allowing
//! FP16/BF16 for matrix multiplications, convolutions, and other compute-heavy ops.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tenflowers_core::{
    mixed_precision::{
        AutocastContext, GradientScaler, MixedPrecisionConfig, MixedPrecisionState,
        OptimizationLevel,
    },
    DType, Result, Tensor, TensorError,
};

/// Configuration for AMP policy with enhanced dynamic loss scaling
#[derive(Debug, Clone)]
pub struct AMPConfig {
    /// Enable automatic mixed precision
    pub enabled: bool,
    /// Initial loss scaling factor
    pub initial_scale: f32,
    /// Minimum loss scale (safety floor)
    pub min_scale: f32,
    /// Maximum loss scale (safety ceiling)
    pub max_scale: f32,
    /// Number of consecutive steps without overflow before increasing scale
    pub growth_interval: usize,
    /// Factor to multiply scale by when growing (default: 2.0)
    pub growth_factor: f32,
    /// Factor to multiply scale by when backing off on overflow (default: 0.5)
    pub backoff_factor: f32,
    /// Target precision for computation (Float16 or BFloat16)
    pub target_dtype: DType,
    /// Operations that must use FP32 for stability
    pub fp32_operations: Vec<String>,
    /// Maximum consecutive overflows before triggering warning
    pub max_consecutive_overflows: usize,
    /// Enable detailed stability tracking
    pub track_stability: bool,
}

impl Default for AMPConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_scale: 65536.0, // 2^16
            min_scale: 1.0,
            max_scale: 65536.0 * 65536.0, // 2^32
            growth_interval: 2000,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            target_dtype: DType::Float16,
            fp32_operations: vec![
                // Normalization operations
                "batch_norm".to_string(),
                "layer_norm".to_string(),
                "group_norm".to_string(),
                "instance_norm".to_string(),
                // Softmax and loss functions
                "softmax".to_string(),
                "log_softmax".to_string(),
                "cross_entropy".to_string(),
                "kl_divergence".to_string(),
                "nll_loss".to_string(),
                // Numerically sensitive transcendental functions
                "exp".to_string(),
                "log".to_string(),
                "sqrt".to_string(),
                "rsqrt".to_string(),
                "pow".to_string(),
                "reciprocal".to_string(),
            ],
            max_consecutive_overflows: 5,
            track_stability: true,
        }
    }
}

impl AMPConfig {
    /// Set initial loss scale
    pub fn with_initial_scale(mut self, scale: f32) -> Self {
        self.initial_scale = scale;
        self
    }

    /// Set growth interval for dynamic loss scaling
    pub fn with_growth_interval(mut self, interval: usize) -> Self {
        self.growth_interval = interval;
        self
    }

    /// Set backoff factor for overflow recovery
    pub fn with_backoff_factor(mut self, factor: f32) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Set growth factor for scale increases
    pub fn with_growth_factor(mut self, factor: f32) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Set target precision (Float16 or BFloat16)
    pub fn with_target_dtype(mut self, dtype: DType) -> Self {
        self.target_dtype = dtype;
        self
    }

    /// Enable BFloat16 precision
    pub fn with_bfloat16(mut self) -> Self {
        self.target_dtype = DType::BFloat16;
        self
    }

    /// Enable Float16 precision
    pub fn with_float16(mut self) -> Self {
        self.target_dtype = DType::Float16;
        self
    }

    /// Add operation to FP32 whitelist
    pub fn with_fp32_operation(mut self, operation: impl Into<String>) -> Self {
        self.fp32_operations.push(operation.into());
        self
    }

    /// Enable stability tracking
    pub fn with_stability_tracking(mut self, enabled: bool) -> Self {
        self.track_stability = enabled;
        self
    }
}

/// Stability metrics for AMP policy
#[derive(Debug, Clone)]
pub struct AMPStabilityMetrics {
    /// Total number of steps processed
    pub total_steps: usize,
    /// Number of steps with gradient overflow
    pub overflow_steps: usize,
    /// Overflow rate (overflow_steps / total_steps)
    pub overflow_rate: f64,
    /// Current loss scale value
    pub current_scale: f32,
    /// Minimum scale reached
    pub min_scale_reached: f32,
    /// Maximum scale reached
    pub max_scale_reached: f32,
    /// Number of scale adjustments (up or down)
    pub scale_adjustments: usize,
    /// Number of consecutive overflows
    pub consecutive_overflows: usize,
    /// Average steps between overflows
    pub avg_steps_between_overflows: f64,
    /// Time spent in AMP operations
    pub amp_overhead_ms: u64,
}

impl Default for AMPStabilityMetrics {
    fn default() -> Self {
        Self {
            total_steps: 0,
            overflow_steps: 0,
            overflow_rate: 0.0,
            current_scale: 65536.0,
            min_scale_reached: 65536.0,
            max_scale_reached: 65536.0,
            scale_adjustments: 0,
            consecutive_overflows: 0,
            avg_steps_between_overflows: 0.0,
            amp_overhead_ms: 0,
        }
    }
}

/// Scale adjustment event for tracking
#[derive(Debug, Clone)]
pub struct ScaleAdjustment {
    /// Step number when adjustment occurred
    pub step: usize,
    /// Scale before adjustment
    pub old_scale: f32,
    /// Scale after adjustment
    pub new_scale: f32,
    /// Reason for adjustment
    pub reason: ScaleAdjustmentReason,
}

/// Reason for scale adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleAdjustmentReason {
    /// Scale increased after stable period
    Growth,
    /// Scale decreased due to overflow
    Backoff,
    /// Scale clamped to minimum
    MinClamp,
    /// Scale clamped to maximum
    MaxClamp,
}

/// Automatic Mixed Precision Policy
///
/// Manages mixed precision training with dynamic loss scaling, gradient overflow detection,
/// and operation-level precision control integrated with GradientTape.
pub struct AMPPolicy {
    config: AMPConfig,
    scaler: GradientScaler,
    autocast_context: AutocastContext,
    metrics: AMPStabilityMetrics,
    scale_history: Vec<ScaleAdjustment>,
    last_overflow_step: Option<usize>,
    operation_precisions: HashMap<String, DType>,
    start_time: Instant,
}

impl AMPPolicy {
    /// Create a new AMP policy with the given configuration
    pub fn new(config: AMPConfig) -> Self {
        // Convert to core MixedPrecisionConfig
        let core_config = MixedPrecisionConfig {
            enabled: config.enabled,
            opt_level: OptimizationLevel::O1, // Use conservative mixed precision
            loss_scale: config.initial_scale,
            dynamic_loss_scaling: true,
            min_loss_scale: config.min_scale,
            max_loss_scale: config.max_scale,
            scale_growth_interval: config.growth_interval,
            scale_growth_factor: 2.0,  // Double scale when growing
            scale_backoff_factor: 0.5, // Halve scale on overflow
            fp32_operations: config.fp32_operations.clone(),
            fp16_blacklist: vec![],
            keep_master_weights: true,       // Keep FP32 master weights
            enable_gradient_clipping: false, // Disable by default
            gradient_clip_norm: 1.0,         // Default clip threshold
        };

        let scaler = GradientScaler::new(core_config);
        let autocast_context = AutocastContext::new(config.enabled, config.target_dtype);

        let metrics = AMPStabilityMetrics {
            current_scale: config.initial_scale,
            min_scale_reached: config.initial_scale,
            max_scale_reached: config.initial_scale,
            ..Default::default()
        };

        // Initialize operation precision map
        let mut operation_precisions = HashMap::new();
        for op in &config.fp32_operations {
            operation_precisions.insert(op.clone(), DType::Float32);
        }

        Self {
            config,
            scaler,
            autocast_context,
            metrics,
            scale_history: Vec::new(),
            last_overflow_step: None,
            operation_precisions,
            start_time: Instant::now(),
        }
    }

    /// Create a new disabled AMP policy (pass-through mode)
    pub fn disabled() -> Self {
        Self::new(AMPConfig {
            enabled: false,
            ..Default::default()
        })
    }

    /// Scale loss before backpropagation
    pub fn scale_loss<T>(&self, loss: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if !self.config.enabled {
            return Ok(loss.clone());
        }

        self.scaler.scale(loss)
    }

    /// Unscale gradients and check for overflow, returning whether step should proceed
    ///
    /// Returns `Ok(true)` if gradients are clean and step should proceed
    /// Returns `Ok(false)` if overflow detected and step should be skipped
    pub fn unscale_and_check<T>(&mut self, gradients: &mut [Tensor<T>]) -> Result<bool>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let start = Instant::now();

        if !self.config.enabled {
            return Ok(true);
        }

        // Unscale and check using core scaler
        let should_step = self.scaler.unscale_gradients_and_check(gradients)?;

        // Update metrics
        self.metrics.total_steps += 1;

        let old_scale = self.metrics.current_scale;
        let new_scale = self.scaler.get_scale();

        if !should_step {
            // Overflow detected
            self.metrics.overflow_steps += 1;
            self.metrics.consecutive_overflows += 1;
            self.last_overflow_step = Some(self.metrics.total_steps);

            // Record scale adjustment if it changed
            if (old_scale - new_scale).abs() > 1e-6 {
                self.metrics.scale_adjustments += 1;
                self.metrics.min_scale_reached = self.metrics.min_scale_reached.min(new_scale);

                self.scale_history.push(ScaleAdjustment {
                    step: self.metrics.total_steps,
                    old_scale,
                    new_scale,
                    reason: if new_scale < old_scale {
                        ScaleAdjustmentReason::Backoff
                    } else {
                        ScaleAdjustmentReason::Growth
                    },
                });
            }

            // Check for excessive consecutive overflows
            if self.metrics.consecutive_overflows >= self.config.max_consecutive_overflows {
                return Err(TensorError::numerical_error(
                    "gradient_overflow",
                    &format!(
                        "AMP Policy: {} consecutive gradient overflows detected",
                        self.metrics.consecutive_overflows
                    ),
                    vec![
                        "Consider reducing the learning rate".to_string(),
                        "Consider disabling mixed precision".to_string(),
                        "Check model architecture for numerical instabilities".to_string(),
                    ],
                ));
            }
        } else {
            // Successful step
            self.metrics.consecutive_overflows = 0;

            // Check if scale increased
            if (new_scale - old_scale).abs() > 1e-6 {
                self.metrics.scale_adjustments += 1;
                self.metrics.max_scale_reached = self.metrics.max_scale_reached.max(new_scale);

                self.scale_history.push(ScaleAdjustment {
                    step: self.metrics.total_steps,
                    old_scale,
                    new_scale,
                    reason: if new_scale > old_scale {
                        ScaleAdjustmentReason::Growth
                    } else {
                        ScaleAdjustmentReason::Backoff
                    },
                });
            }
        }

        self.metrics.current_scale = new_scale;

        // Update derived metrics
        if self.metrics.overflow_steps > 0 {
            self.metrics.overflow_rate =
                self.metrics.overflow_steps as f64 / self.metrics.total_steps as f64;

            if self.metrics.overflow_steps > 1 {
                self.metrics.avg_steps_between_overflows =
                    self.metrics.total_steps as f64 / self.metrics.overflow_steps as f64;
            }
        }

        // Track overhead
        let elapsed = start.elapsed();
        self.metrics.amp_overhead_ms += elapsed.as_millis() as u64;

        Ok(should_step)
    }

    /// Get the appropriate dtype for an operation
    pub fn get_operation_dtype(&self, operation_name: &str, default_dtype: DType) -> DType {
        if !self.config.enabled {
            return default_dtype;
        }

        // Check custom operation precision map
        if let Some(&dtype) = self.operation_precisions.get(operation_name) {
            return dtype;
        }

        // Delegate to autocast context
        self.autocast_context
            .get_operation_dtype(operation_name, default_dtype)
    }

    /// Check if operation should use mixed precision
    pub fn should_autocast(&self, operation_name: &str, input_dtype: DType) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Don't autocast if operation is in FP32 whitelist
        if self.operation_precisions.contains_key(operation_name) {
            return false;
        }

        self.autocast_context
            .should_autocast(operation_name, input_dtype)
    }

    /// Get current stability metrics
    pub fn get_stability_metrics(&self) -> AMPStabilityMetrics {
        self.metrics.clone()
    }

    /// Get scale adjustment history
    pub fn get_scale_history(&self) -> &[ScaleAdjustment] {
        &self.scale_history
    }

    /// Check if AMP is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get current loss scale
    pub fn get_current_scale(&self) -> f32 {
        self.scaler.get_scale()
    }

    /// Get configuration
    pub fn get_config(&self) -> &AMPConfig {
        &self.config
    }

    /// Reset metrics (useful for new training phase)
    pub fn reset_metrics(&mut self) {
        self.metrics = AMPStabilityMetrics::default();
        self.metrics.current_scale = self.scaler.get_scale();
        self.metrics.min_scale_reached = self.metrics.current_scale;
        self.metrics.max_scale_reached = self.metrics.current_scale;
        self.scale_history.clear();
        self.last_overflow_step = None;
        self.start_time = Instant::now();
    }

    /// Print stability report
    pub fn print_stability_report(&self) {
        println!("\n=== AMP Stability Report ===");
        println!("Total steps: {}", self.metrics.total_steps);
        println!("Overflow rate: {:.2}%", self.metrics.overflow_rate * 100.0);
        println!("Current scale: {:.1}", self.metrics.current_scale);
        println!(
            "Scale range: [{:.1}, {:.1}]",
            self.metrics.min_scale_reached, self.metrics.max_scale_reached
        );
        println!("Scale adjustments: {}", self.metrics.scale_adjustments);
        println!(
            "Consecutive overflows: {}",
            self.metrics.consecutive_overflows
        );

        if self.metrics.avg_steps_between_overflows > 0.0 {
            println!(
                "Average steps between overflows: {:.1}",
                self.metrics.avg_steps_between_overflows
            );
        }

        println!("AMP overhead: {} ms", self.metrics.amp_overhead_ms);

        if !self.scale_history.is_empty() {
            println!("\nRecent scale adjustments:");
            let recent = self.scale_history.iter().rev().take(5);
            for adjustment in recent {
                println!(
                    "  Step {}: {:.1} -> {:.1} ({:?})",
                    adjustment.step, adjustment.old_scale, adjustment.new_scale, adjustment.reason
                );
            }
        }
        println!("===========================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amp_config_builder() {
        let config = AMPConfig::default()
            .with_initial_scale(32768.0)
            .with_growth_interval(1000)
            .with_backoff_factor(0.25)
            .with_bfloat16();

        assert_eq!(config.initial_scale, 32768.0);
        assert_eq!(config.growth_interval, 1000);
        assert_eq!(config.backoff_factor, 0.25);
        assert_eq!(config.target_dtype, DType::BFloat16);
    }

    #[test]
    fn test_amp_policy_creation() {
        let config = AMPConfig::default();
        let policy = AMPPolicy::new(config);

        assert!(!policy.is_enabled());
        assert_eq!(policy.get_current_scale(), 65536.0);
    }

    #[test]
    fn test_amp_policy_disabled() {
        let policy = AMPPolicy::disabled();
        assert!(!policy.is_enabled());
    }

    #[test]
    fn test_scale_loss_passthrough_when_disabled() {
        let policy = AMPPolicy::disabled();
        let loss = Tensor::<f32>::from_scalar(1.0);
        let scaled = policy.scale_loss(&loss).unwrap();

        // Should be unchanged
        assert_eq!(scaled.as_slice().unwrap()[0], loss.as_slice().unwrap()[0]);
    }

    #[test]
    fn test_scale_loss_when_enabled() {
        let config = AMPConfig {
            enabled: true,
            initial_scale: 1024.0,
            ..Default::default()
        };
        let policy = AMPPolicy::new(config);
        let loss = Tensor::<f32>::from_scalar(1.0);
        let scaled = policy.scale_loss(&loss).unwrap();

        // Should be scaled by 1024
        assert_eq!(scaled.as_slice().unwrap()[0], 1024.0);
    }

    #[test]
    fn test_unscale_and_check_no_overflow() {
        let config = AMPConfig {
            enabled: true,
            initial_scale: 1024.0,
            ..Default::default()
        };
        let mut policy = AMPPolicy::new(config);

        // Create gradients without overflow
        let mut gradients = vec![Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap()];

        let should_step = policy.unscale_and_check(&mut gradients).unwrap();

        assert!(should_step);
        assert_eq!(policy.metrics.total_steps, 1);
        assert_eq!(policy.metrics.overflow_steps, 0);
        assert_eq!(policy.metrics.consecutive_overflows, 0);
    }

    #[test]
    fn test_operation_dtype_fp32_whitelist() {
        let config = AMPConfig {
            enabled: true,
            ..Default::default()
        };
        let policy = AMPPolicy::new(config);

        // Operations in FP32 whitelist should return Float32
        assert_eq!(
            policy.get_operation_dtype("softmax", DType::Float32),
            DType::Float32
        );
        assert_eq!(
            policy.get_operation_dtype("batch_norm", DType::Float32),
            DType::Float32
        );
        assert_eq!(
            policy.get_operation_dtype("exp", DType::Float32),
            DType::Float32
        );
    }

    #[test]
    fn test_operation_dtype_autocast() {
        let config = AMPConfig {
            enabled: true,
            target_dtype: DType::Float16,
            ..Default::default()
        };
        let policy = AMPPolicy::new(config);

        // Regular operations should use target dtype
        assert_eq!(
            policy.get_operation_dtype("matmul", DType::Float32),
            DType::Float16
        );
        assert_eq!(
            policy.get_operation_dtype("conv2d", DType::Float32),
            DType::Float16
        );
    }

    #[test]
    fn test_stability_metrics_tracking() {
        let config = AMPConfig {
            enabled: true,
            track_stability: true,
            ..Default::default()
        };
        let mut policy = AMPPolicy::new(config);

        // Simulate multiple successful steps
        for _ in 0..10 {
            let mut gradients = vec![Tensor::<f32>::ones(&[5])];
            policy.unscale_and_check(&mut gradients).unwrap();
        }

        let metrics = policy.get_stability_metrics();
        assert_eq!(metrics.total_steps, 10);
        assert_eq!(metrics.overflow_steps, 0);
        assert_eq!(metrics.overflow_rate, 0.0);
    }

    #[test]
    fn test_reset_metrics() {
        let config = AMPConfig {
            enabled: true,
            ..Default::default()
        };
        let mut policy = AMPPolicy::new(config);

        // Simulate some steps
        for _ in 0..5 {
            let mut gradients = vec![Tensor::<f32>::ones(&[5])];
            policy.unscale_and_check(&mut gradients).unwrap();
        }

        assert_eq!(policy.metrics.total_steps, 5);

        // Reset
        policy.reset_metrics();

        assert_eq!(policy.metrics.total_steps, 0);
        assert_eq!(policy.metrics.overflow_steps, 0);
        assert!(policy.scale_history.is_empty());
    }

    #[test]
    fn test_should_autocast() {
        let config = AMPConfig {
            enabled: true,
            ..Default::default()
        };
        let policy = AMPPolicy::new(config);

        // Should autocast regular ops with float input
        assert!(policy.should_autocast("matmul", DType::Float32));

        // Should not autocast FP32 whitelist ops
        assert!(!policy.should_autocast("softmax", DType::Float32));

        // Should not autocast integer ops
        assert!(!policy.should_autocast("matmul", DType::Int32));
    }

    #[test]
    fn test_bfloat16_config() {
        let config = AMPConfig::default().with_bfloat16();
        let policy = AMPPolicy::new(config);

        // When enabled, regular ops should use BFloat16
        let mut enabled_config = AMPConfig::default().with_bfloat16();
        enabled_config.enabled = true;
        let enabled_policy = AMPPolicy::new(enabled_config);

        assert_eq!(
            enabled_policy.get_operation_dtype("conv2d", DType::Float32),
            DType::BFloat16
        );
    }
}
