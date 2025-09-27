use num_traits::{Float, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Debug information for gradient checking
#[derive(Debug, Clone)]
pub struct GradientDebugInfo {
    pub tensor_id: usize,
    pub operation: String,
    pub has_nan: bool,
    pub has_inf: bool,
    pub max_magnitude: f64,
    pub min_magnitude: f64,
    pub gradient_norm: f64,
}

/// Debugging utilities for gradient computation
pub struct GradientDebugger {
    debug_info: HashMap<usize, GradientDebugInfo>,
    check_nan_inf: bool,
    check_gradient_magnitude: bool,
    gradient_magnitude_threshold: f64,
}

impl GradientDebugger {
    /// Create a new gradient debugger
    pub fn new() -> Self {
        Self {
            debug_info: HashMap::new(),
            check_nan_inf: true,
            check_gradient_magnitude: true,
            gradient_magnitude_threshold: 1e6,
        }
    }

    /// Enable or disable NaN/Inf checking
    pub fn set_nan_inf_checking(&mut self, enabled: bool) {
        self.check_nan_inf = enabled;
    }

    /// Enable or disable gradient magnitude checking
    pub fn set_gradient_magnitude_checking(&mut self, enabled: bool) {
        self.check_gradient_magnitude = enabled;
    }

    /// Set the threshold for gradient magnitude warnings
    pub fn set_gradient_magnitude_threshold(&mut self, threshold: f64) {
        self.gradient_magnitude_threshold = threshold;
    }

    /// Check a gradient tensor for debugging issues
    pub fn check_gradient<T>(
        &mut self,
        tensor_id: usize,
        operation: &str,
        gradient: &Tensor<T>,
    ) -> Result<()>
    where
        T: Float + Clone + Default + Zero + Send + Sync + 'static,
    {
        let mut debug_info = GradientDebugInfo {
            tensor_id,
            operation: operation.to_string(),
            has_nan: false,
            has_inf: false,
            max_magnitude: 0.0,
            min_magnitude: f64::INFINITY,
            gradient_norm: 0.0,
        };

        if let Some(data) = gradient.as_slice() {
            let mut sum_squares = T::zero();

            for &val in data {
                // Check for NaN
                if self.check_nan_inf && val.is_nan() {
                    debug_info.has_nan = true;
                }

                // Check for infinity
                if self.check_nan_inf && val.is_infinite() {
                    debug_info.has_inf = true;
                }

                // Track magnitude
                let magnitude = val.abs().to_f64().unwrap_or(0.0);
                debug_info.max_magnitude = debug_info.max_magnitude.max(magnitude);
                debug_info.min_magnitude = debug_info.min_magnitude.min(magnitude);

                // Accumulate for norm calculation
                sum_squares = sum_squares + val * val;
            }

            // Calculate gradient norm
            debug_info.gradient_norm = sum_squares.sqrt().to_f64().unwrap_or(0.0);

            // Check for gradient explosion
            if self.check_gradient_magnitude
                && debug_info.gradient_norm > self.gradient_magnitude_threshold
            {
                eprintln!("WARNING: Large gradient detected in operation '{}' (tensor {}). Gradient norm: {:.6}", 
                         operation, tensor_id, debug_info.gradient_norm);
            }

            // Report NaN/Inf if found
            if debug_info.has_nan {
                eprintln!("ERROR: NaN detected in gradient for operation '{operation}' (tensor {tensor_id})");
            }
            if debug_info.has_inf {
                eprintln!("ERROR: Inf detected in gradient for operation '{operation}' (tensor {tensor_id})");
            }
        }

        self.debug_info.insert(tensor_id, debug_info);
        Ok(())
    }

    /// Get debug information for a specific tensor
    pub fn get_debug_info(&self, tensor_id: usize) -> Option<&GradientDebugInfo> {
        self.debug_info.get(&tensor_id)
    }

    /// Get all debug information
    pub fn get_all_debug_info(&self) -> &HashMap<usize, GradientDebugInfo> {
        &self.debug_info
    }

    /// Clear all debug information
    pub fn clear(&mut self) {
        self.debug_info.clear();
    }

    /// Generate a debug report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Gradient Debug Report ===\n\n");

        if self.debug_info.is_empty() {
            report.push_str("No gradient debug information available.\n");
            return report;
        }

        // Summary statistics
        let total_tensors = self.debug_info.len();
        let nan_count = self.debug_info.values().filter(|info| info.has_nan).count();
        let inf_count = self.debug_info.values().filter(|info| info.has_inf).count();
        let large_gradient_count = self
            .debug_info
            .values()
            .filter(|info| info.gradient_norm > self.gradient_magnitude_threshold)
            .count();

        report.push_str(&format!("Total tensors analyzed: {total_tensors}\n"));
        report.push_str(&format!("Tensors with NaN gradients: {nan_count}\n"));
        report.push_str(&format!("Tensors with Inf gradients: {inf_count}\n"));
        report.push_str(&format!(
            "Tensors with large gradients (>{:.1e}): {large_gradient_count}\n\n",
            self.gradient_magnitude_threshold
        ));

        // Detailed information for problematic tensors
        if nan_count > 0 || inf_count > 0 || large_gradient_count > 0 {
            report.push_str("Problematic tensors:\n");
            for info in self.debug_info.values() {
                if info.has_nan
                    || info.has_inf
                    || info.gradient_norm > self.gradient_magnitude_threshold
                {
                    let tensor_id = info.tensor_id;
                    let operation = &info.operation;
                    report.push_str(&format!("  Tensor {tensor_id}: {operation}\n"));
                    if info.has_nan {
                        report.push_str("    - Contains NaN values\n");
                    }
                    if info.has_inf {
                        report.push_str("    - Contains Inf values\n");
                    }
                    if info.gradient_norm > self.gradient_magnitude_threshold {
                        report.push_str(&format!(
                            "    - Large gradient norm: {:.6}\n",
                            info.gradient_norm
                        ));
                    }
                    report.push_str(&format!("    - Max magnitude: {:.6}\n", info.max_magnitude));
                    report.push_str(&format!("    - Min magnitude: {:.6}\n", info.min_magnitude));
                }
            }
        }

        report
    }

    /// Check for any critical issues (NaN or Inf)
    pub fn has_critical_issues(&self) -> bool {
        self.debug_info
            .values()
            .any(|info| info.has_nan || info.has_inf)
    }

    /// Get the maximum gradient norm across all tensors
    pub fn max_gradient_norm(&self) -> f64 {
        self.debug_info
            .values()
            .map(|info| info.gradient_norm)
            .fold(0.0, f64::max)
    }
}

impl Default for GradientDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for gradient debugging
pub mod utils {
    use super::*;

    /// Check if a tensor contains any NaN values
    pub fn has_nan<T>(tensor: &Tensor<T>) -> bool
    where
        T: Float + Clone + Default,
    {
        if let Some(data) = tensor.as_slice() {
            data.iter().any(|&val| val.is_nan())
        } else {
            false
        }
    }

    /// Check if a tensor contains any infinite values
    pub fn has_inf<T>(tensor: &Tensor<T>) -> bool
    where
        T: Float + Clone + Default,
    {
        if let Some(data) = tensor.as_slice() {
            data.iter().any(|&val| val.is_infinite())
        } else {
            false
        }
    }

    /// Compute the L2 norm of a tensor
    pub fn tensor_norm<T>(tensor: &Tensor<T>) -> T
    where
        T: Float + Clone + Default + Zero,
    {
        if let Some(data) = tensor.as_slice() {
            let sum_squares = data.iter().fold(T::zero(), |acc, &val| acc + val * val);
            sum_squares.sqrt()
        } else {
            T::zero()
        }
    }

    /// Compute gradient magnitude statistics
    pub fn gradient_statistics<T>(tensor: &Tensor<T>) -> Option<(T, T, T)>
    where
        T: Float + Clone + Default + PartialOrd,
    {
        if let Some(data) = tensor.as_slice() {
            if data.is_empty() {
                return None;
            }

            let mut min_val = data[0].abs();
            let mut max_val = min_val;
            let mut sum_squares = T::zero();

            for &val in data {
                let abs_val = val.abs();
                if abs_val < min_val {
                    min_val = abs_val;
                }
                if abs_val > max_val {
                    max_val = abs_val;
                }
                sum_squares = sum_squares + val * val;
            }

            let norm = sum_squares.sqrt();
            Some((min_val, max_val, norm))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_gradient_debugger_basic() {
        let mut debugger = GradientDebugger::new();

        // Create a normal gradient tensor
        let gradient = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        debugger.check_gradient(1, "test_op", &gradient).unwrap();

        let info = debugger.get_debug_info(1).unwrap();
        assert!(!info.has_nan);
        assert!(!info.has_inf);
        assert!(info.gradient_norm > 0.0);
    }

    #[test]
    fn test_nan_detection() {
        use std::f32;

        let tensor = Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0], &[3]).unwrap();
        assert!(utils::has_nan(&tensor));
        assert!(!utils::has_inf(&tensor));
    }

    #[test]
    fn test_inf_detection() {
        use std::f32;

        let tensor = Tensor::from_vec(vec![1.0f32, f32::INFINITY, 3.0], &[3]).unwrap();
        assert!(!utils::has_nan(&tensor));
        assert!(utils::has_inf(&tensor));
    }
}
