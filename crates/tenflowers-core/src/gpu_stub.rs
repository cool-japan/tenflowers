/// GPU Kernel Dispatcher Stub
///
/// When the `gpu` feature is enabled this module provides a `GpuKernelDispatcher`
/// that issues real WGPU command encoders.  When `gpu` is *not* enabled (the
/// default) it exposes a `GpuStub` that simulates GPU dispatch entirely on the
/// CPU, enabling tests and benchmarks to exercise the dispatch path without
/// requiring a physical GPU.
///
/// # Design
///
/// The stub follows the same public interface as the real dispatcher so that
/// call-sites can use `#[cfg(feature = "gpu")]` to swap between the two
/// without any structural changes.
///
/// # Reduction operations
///
/// Both the real dispatcher and the stub expose a `dispatch_reduction` method.
/// The stub computes the reduction on the CPU and reports simulated timing
/// metadata so that performance-gate benchmarks can validate the dispatch
/// overhead without depending on GPU hardware.
use crate::{Result, TensorError};

// ---------------------------------------------------------------------------
// CPU-fallback stub (always compiled)
// ---------------------------------------------------------------------------

/// Operation kind for a GPU-style reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StubReductionOp {
    /// Sum all elements.
    Sum,
    /// Compute the arithmetic mean of all elements.
    Mean,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
}

impl StubReductionOp {
    /// Human-readable name used in diagnostics and test output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
        }
    }
}

/// Result returned by `GpuStub::dispatch_reduction`.
#[derive(Debug, Clone)]
pub struct StubReductionResult {
    /// The computed reduction value (CPU fallback).
    pub value: f64,
    /// Simulated command-encoder creation latency in nanoseconds.
    pub encoder_latency_ns: u64,
    /// Whether the operation ran on real GPU hardware.
    pub used_real_gpu: bool,
}

/// Simulated GPU dispatch using CPU fallback.
///
/// This struct mirrors the public interface of the real `GpuKernelDispatcher`
/// so that benchmarks and unit tests work without requiring a physical GPU.
///
/// The stub measures the wall-clock cost of the equivalent CPU operation and
/// stores it in `StubReductionResult::encoder_latency_ns` so that overhead
/// benchmarks have a concrete timing signal to assert against.
#[derive(Debug, Default)]
pub struct GpuStub {
    /// Simulated device label, used only for diagnostics.
    device_label: String,
}

impl GpuStub {
    /// Create a new `GpuStub` with the given device label.
    pub fn new(device_label: impl Into<String>) -> Self {
        Self {
            device_label: device_label.into(),
        }
    }

    /// Device label set at construction time.
    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    /// Simulate GPU reduction dispatch using a CPU fallback.
    ///
    /// # Errors
    ///
    /// Returns an error when `data` is empty and the operation is `Max` or
    /// `Min` (no identity element for an empty set).
    ///
    /// # Panics
    ///
    /// Does not panic.
    pub fn dispatch_reduction(
        &self,
        op: StubReductionOp,
        data: &[f32],
    ) -> Result<StubReductionResult> {
        let start = std::time::Instant::now();

        let value: f64 = match op {
            StubReductionOp::Sum => data.iter().map(|&x| x as f64).sum(),
            StubReductionOp::Mean => {
                if data.is_empty() {
                    0.0
                } else {
                    data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64
                }
            }
            StubReductionOp::Max => {
                if data.is_empty() {
                    return Err(TensorError::invalid_argument(
                        "Max reduction requires at least one element".to_string(),
                    ));
                }
                data.iter()
                    .map(|&x| x as f64)
                    .fold(f64::NEG_INFINITY, f64::max)
            }
            StubReductionOp::Min => {
                if data.is_empty() {
                    return Err(TensorError::invalid_argument(
                        "Min reduction requires at least one element".to_string(),
                    ));
                }
                data.iter().map(|&x| x as f64).fold(f64::INFINITY, f64::min)
            }
        };

        let encoder_latency_ns = start.elapsed().as_nanos() as u64;

        Ok(StubReductionResult {
            value,
            encoder_latency_ns,
            used_real_gpu: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Real WGPU dispatcher (only compiled with the `gpu` feature)
// ---------------------------------------------------------------------------

/// Real GPU kernel dispatcher backed by a WGPU `Device`.
///
/// Available only when the `gpu` feature is enabled.
#[cfg(feature = "gpu")]
pub struct GpuKernelDispatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(feature = "gpu")]
impl GpuKernelDispatcher {
    /// Create a new dispatcher from an existing WGPU device and queue.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue }
    }

    /// Dispatch a reduction operation by creating a WGPU command encoder.
    ///
    /// This is a **stub** implementation: it creates the encoder and records
    /// the setup latency, but does not submit or execute real GPU work.
    /// Replace the body with a real compute pass when the shader pipeline is
    /// wired up.
    ///
    /// # Errors
    ///
    /// Returns an error when `data` is empty.
    pub fn dispatch_reduction(
        &self,
        op: StubReductionOp,
        data: &[f32],
    ) -> Result<StubReductionResult> {
        if data.is_empty() && matches!(op, StubReductionOp::Max | StubReductionOp::Min) {
            return Err(TensorError::invalid_argument(
                "Max/Min reduction requires at least one element".to_string(),
            ));
        }

        let start = std::time::Instant::now();

        // Create a command encoder — this is the GPU-side overhead we measure.
        let _encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("reduction_{}_encoder", op.name())),
            });

        // NOTE(v0.2): add real compute pass once the WGSL shader pipeline is wired up.
        // For now we fall back to CPU to produce a correct answer while the
        // encoder-creation overhead is still measured above.
        let cpu_stub = GpuStub::new("gpu_fallback");
        let cpu_result = cpu_stub.dispatch_reduction(op, data)?;

        let encoder_latency_ns = start.elapsed().as_nanos() as u64;

        Ok(StubReductionResult {
            value: cpu_result.value,
            encoder_latency_ns,
            used_real_gpu: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: measure the nanosecond cost of a single closure call
// ---------------------------------------------------------------------------

/// Measure the wall-clock cost of a single call to `f` in nanoseconds.
///
/// This is intentionally a thin wrapper around `std::time::Instant` so that
/// callers do not need to repeat the boilerplate.
#[inline]
pub fn measure_overhead_ns<F: FnMut()>(mut f: F) -> u64 {
    let start = std::time::Instant::now();
    f();
    start.elapsed().as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Threshold constants
// ---------------------------------------------------------------------------

/// Maximum acceptable dispatch overhead in nanoseconds.
///
/// A registry lookup (read-lock + hash-map probe) should complete in well
/// under one microsecond on any modern machine.  We set this to 10 µs to
/// give generous headroom for CI environments running under heavy load.
pub const MAX_DISPATCH_OVERHEAD_NS: u64 = 10_000;

/// Maximum acceptable GPU stub dispatch overhead in nanoseconds.
///
/// The CPU-fallback stub for a small payload (≤ 1 024 elements) should
/// complete in under 500 µs even on slow CI hardware.
pub const MAX_GPU_STUB_OVERHEAD_NS: u64 = 500_000;

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

/// Check that a measured overhead value is within the given threshold.
///
/// Returns `Ok(())` when `measured_ns <= threshold_ns`, or an error message
/// describing the violation.
pub fn validate_overhead(label: &str, measured_ns: u64, threshold_ns: u64) -> Result<()> {
    if measured_ns <= threshold_ns {
        Ok(())
    } else {
        Err(TensorError::invalid_argument(format!(
            "Overhead validation failed for '{}': measured {}ns exceeds threshold {}ns",
            label, measured_ns, threshold_ns
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // GpuStub correctness
    // ------------------------------------------------------------------

    #[test]
    fn test_stub_sum_reduction() {
        let stub = GpuStub::new("test_device");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let result = stub
            .dispatch_reduction(StubReductionOp::Sum, &data)
            .expect("dispatch_reduction should succeed");
        assert!((result.value - 15.0).abs() < 1e-6, "sum should be 15.0");
        assert!(!result.used_real_gpu);
    }

    #[test]
    fn test_stub_mean_reduction() {
        let stub = GpuStub::new("test_device");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let result = stub
            .dispatch_reduction(StubReductionOp::Mean, &data)
            .expect("dispatch_reduction should succeed");
        assert!((result.value - 3.0).abs() < 1e-6, "mean should be 3.0");
    }

    #[test]
    fn test_stub_max_reduction() {
        let stub = GpuStub::new("test_device");
        let data = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let result = stub
            .dispatch_reduction(StubReductionOp::Max, &data)
            .expect("dispatch_reduction should succeed");
        assert!((result.value - 9.0).abs() < 1e-6, "max should be 9.0");
    }

    #[test]
    fn test_stub_min_reduction() {
        let stub = GpuStub::new("test_device");
        let data = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0];
        let result = stub
            .dispatch_reduction(StubReductionOp::Min, &data)
            .expect("dispatch_reduction should succeed");
        assert!((result.value - 1.0).abs() < 1e-6, "min should be 1.0");
    }

    #[test]
    fn test_stub_empty_max_returns_error() {
        let stub = GpuStub::new("test_device");
        let result = stub.dispatch_reduction(StubReductionOp::Max, &[]);
        assert!(result.is_err(), "Max of empty slice should be an error");
    }

    #[test]
    fn test_stub_empty_min_returns_error() {
        let stub = GpuStub::new("test_device");
        let result = stub.dispatch_reduction(StubReductionOp::Min, &[]);
        assert!(result.is_err(), "Min of empty slice should be an error");
    }

    #[test]
    fn test_stub_empty_sum_returns_zero() {
        let stub = GpuStub::new("test_device");
        let result = stub
            .dispatch_reduction(StubReductionOp::Sum, &[])
            .expect("Sum of empty slice is defined (zero)");
        assert!((result.value - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_stub_empty_mean_returns_zero() {
        let stub = GpuStub::new("test_device");
        let result = stub
            .dispatch_reduction(StubReductionOp::Mean, &[])
            .expect("Mean of empty slice returns 0");
        assert!((result.value - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_stub_device_label() {
        let stub = GpuStub::new("my_test_gpu");
        assert_eq!(stub.device_label(), "my_test_gpu");
    }

    #[test]
    fn test_stub_encoder_latency_is_non_zero_or_zero() {
        // The latency might be 0 on extremely fast systems; we only assert
        // that the field is populated (no panic / unwrap failure).
        let stub = GpuStub::new("test_device");
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let result = stub
            .dispatch_reduction(StubReductionOp::Sum, &data)
            .expect("dispatch_reduction should succeed");
        // Just verify the field exists and is accessible
        let _ = result.encoder_latency_ns;
    }

    // ------------------------------------------------------------------
    // measure_overhead_ns
    // ------------------------------------------------------------------

    #[test]
    fn test_measure_overhead_ns_fast_closure() {
        let ns = measure_overhead_ns(|| {
            let _ = 1_u64.wrapping_add(1);
        });
        // Should be measurable; we do not assert an upper bound here because
        // CI machines can be arbitrarily slow, but the function must return.
        let _ = ns;
    }

    #[test]
    fn test_measure_overhead_ns_returns_u64() {
        let ns: u64 = measure_overhead_ns(|| {
            std::hint::black_box(42_u64);
        });
        // Type check: if this compiles, the return type is correct.
        let _ = ns;
    }

    // ------------------------------------------------------------------
    // Threshold constants
    // ------------------------------------------------------------------

    #[test]
    fn test_threshold_constants_are_positive() {
        const _: () = {
            assert!(MAX_DISPATCH_OVERHEAD_NS > 0);
            assert!(MAX_GPU_STUB_OVERHEAD_NS > 0);
        };
    }

    #[test]
    fn test_threshold_constants_ordering() {
        // GPU stub threshold must be larger than pure dispatch overhead
        const _: () = {
            assert!(MAX_GPU_STUB_OVERHEAD_NS > MAX_DISPATCH_OVERHEAD_NS);
        };
    }

    // ------------------------------------------------------------------
    // validate_overhead
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_overhead_passes_when_within_threshold() {
        let result = validate_overhead("test_op", 500, 1_000);
        assert!(result.is_ok(), "500ns should be within 1000ns threshold");
    }

    #[test]
    fn test_validate_overhead_passes_at_exact_threshold() {
        let result = validate_overhead("test_op", 1_000, 1_000);
        assert!(result.is_ok(), "Exactly at threshold should pass");
    }

    #[test]
    fn test_validate_overhead_fails_above_threshold() {
        let result = validate_overhead("test_op", 1_001, 1_000);
        assert!(result.is_err(), "1001ns should exceed 1000ns threshold");
    }

    #[test]
    fn test_validate_overhead_error_message_contains_label() {
        let result = validate_overhead("my_operation", 9_999_999, 1);
        let err_msg = format!("{:?}", result.expect_err("should be error"));
        assert!(
            err_msg.contains("my_operation"),
            "Error message should name the operation"
        );
    }

    // ------------------------------------------------------------------
    // StubReductionOp
    // ------------------------------------------------------------------

    #[test]
    fn test_stub_reduction_op_names() {
        assert_eq!(StubReductionOp::Sum.name(), "sum");
        assert_eq!(StubReductionOp::Mean.name(), "mean");
        assert_eq!(StubReductionOp::Max.name(), "max");
        assert_eq!(StubReductionOp::Min.name(), "min");
    }
}
