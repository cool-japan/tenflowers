//! JIT Compilation System for Gradient Kernels
//!
//! This module provides runtime compilation and optimization of gradient kernels.
//! It generates WebGPU compute shaders optimized for specific tensor shapes and operations.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Unique identifier for a kernel configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelSignature {
    pub operation: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub dtype: String,
    pub device_features: DeviceFeatures,
}

impl Hash for KernelSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operation.hash(state);
        self.input_shapes.hash(state);
        self.output_shape.hash(state);
        self.dtype.hash(state);
        // Hash device features manually, excluding f32 fields
        self.device_features.max_workgroup_size.hash(state);
        self.device_features.max_workgroups_per_dim.hash(state);
        self.device_features.supports_f64.hash(state);
        self.device_features.supports_i64.hash(state);
        self.device_features.compute_units.hash(state);
        // Skip memory_bandwidth_gb_s since f32 doesn't implement Hash
    }
}

/// Device-specific features that affect kernel compilation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceFeatures {
    pub max_workgroup_size: u32,
    pub max_workgroups_per_dim: u32,
    pub supports_f64: bool,
    pub supports_i64: bool,
    pub memory_bandwidth_gb_s: f32,
    pub compute_units: u32,
}

impl Eq for DeviceFeatures {}

impl Eq for KernelSignature {}

/// Compiled kernel with metadata
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub signature: KernelSignature,
    pub wgsl_source: String,
    pub workgroup_size: (u32, u32, u32),
    pub specializations: HashMap<String, String>,
    pub compile_time_ms: f64,
    pub estimated_performance: KernelPerformance,
}

/// Performance characteristics of a compiled kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPerformance {
    pub estimated_flops: u64,
    pub estimated_memory_accesses: u64,
    pub estimated_execution_time_us: f64,
    pub memory_bound_ratio: f32, // 0.0 = compute bound, 1.0 = memory bound
}

/// Template for generating gradient kernels
#[derive(Debug, Clone)]
pub struct GradientKernelTemplate {
    pub operation_name: String,
    pub forward_expression: String,
    pub backward_expression: String,
    pub requires_intermediate_values: bool,
    pub broadcasting_support: bool,
}

/// JIT compiler for gradient kernels
#[derive(Debug)]
pub struct JitCompiler {
    kernel_cache: Arc<RwLock<HashMap<KernelSignature, CompiledKernel>>>,
    template_registry: Arc<RwLock<HashMap<String, GradientKernelTemplate>>>,
    device_features: DeviceFeatures,
    optimization_level: OptimizationLevel,
}

/// Optimization levels for kernel compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    Debug,      // No optimizations, fast compilation
    Balanced,   // Moderate optimizations
    Aggressive, // Maximum optimizations, slower compilation
}

impl JitCompiler {
    /// Create a new JIT compiler
    pub fn new(device_features: DeviceFeatures, optimization_level: OptimizationLevel) -> Self {
        let mut compiler = Self {
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            template_registry: Arc::new(RwLock::new(HashMap::new())),
            device_features,
            optimization_level,
        };

        compiler.register_builtin_templates();
        compiler
    }

    /// Register built-in gradient kernel templates
    fn register_builtin_templates(&mut self) {
        let templates = vec![
            // Binary operation gradients
            GradientKernelTemplate {
                operation_name: "add_backward".to_string(),
                forward_expression: "a + b".to_string(),
                backward_expression: "grad_output".to_string(),
                requires_intermediate_values: false,
                broadcasting_support: true,
            },
            GradientKernelTemplate {
                operation_name: "mul_backward".to_string(),
                forward_expression: "a * b".to_string(),
                backward_expression: "grad_output * other_input".to_string(),
                requires_intermediate_values: true,
                broadcasting_support: true,
            },
            GradientKernelTemplate {
                operation_name: "div_backward".to_string(),
                forward_expression: "a / b".to_string(),
                backward_expression: "grad_output / denominator - grad_output * numerator / (denominator * denominator)".to_string(),
                requires_intermediate_values: true,
                broadcasting_support: true,
            },
            // Activation function gradients
            GradientKernelTemplate {
                operation_name: "relu_backward".to_string(),
                forward_expression: "max(0.0, x)".to_string(),
                backward_expression: "grad_output * select(0.0, 1.0, x > 0.0)".to_string(),
                requires_intermediate_values: true,
                broadcasting_support: false,
            },
            GradientKernelTemplate {
                operation_name: "sigmoid_backward".to_string(),
                forward_expression: "1.0 / (1.0 + exp(-x))".to_string(),
                backward_expression: "grad_output * sigmoid_out * (1.0 - sigmoid_out)".to_string(),
                requires_intermediate_values: true,
                broadcasting_support: false,
            },
            GradientKernelTemplate {
                operation_name: "tanh_backward".to_string(),
                forward_expression: "tanh(x)".to_string(),
                backward_expression: "grad_output * (1.0 - tanh_out * tanh_out)".to_string(),
                requires_intermediate_values: true,
                broadcasting_support: false,
            },
            // Reduction operation gradients
            GradientKernelTemplate {
                operation_name: "sum_backward".to_string(),
                forward_expression: "sum(x)".to_string(),
                backward_expression: "grad_output".to_string(), // Broadcast to original shape
                requires_intermediate_values: false,
                broadcasting_support: true,
            },
            GradientKernelTemplate {
                operation_name: "mean_backward".to_string(),
                forward_expression: "mean(x)".to_string(),
                backward_expression: "grad_output / f32(reduced_elements)".to_string(),
                requires_intermediate_values: false,
                broadcasting_support: true,
            },
        ];

        let mut registry = self.template_registry.write().unwrap();
        for template in templates {
            registry.insert(template.operation_name.clone(), template);
        }
    }

    /// Compile or retrieve cached gradient kernel
    pub fn compile_gradient_kernel(&self, signature: KernelSignature) -> Result<CompiledKernel> {
        // Check cache first
        {
            let cache = self.kernel_cache.read().unwrap();
            if let Some(cached_kernel) = cache.get(&signature) {
                return Ok(cached_kernel.clone());
            }
        }

        // Compile new kernel
        let start_time = std::time::Instant::now();
        let compiled_kernel = self.compile_kernel_from_signature(&signature)?;
        let compile_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let mut kernel = compiled_kernel;
        kernel.compile_time_ms = compile_time;

        // Cache the compiled kernel
        {
            let mut cache = self.kernel_cache.write().unwrap();
            cache.insert(signature.clone(), kernel.clone());
        }

        Ok(kernel)
    }

    /// Generate optimized WGSL shader code
    fn compile_kernel_from_signature(&self, signature: &KernelSignature) -> Result<CompiledKernel> {
        let template = {
            let registry = self.template_registry.read().unwrap();
            registry
                .get(&signature.operation)
                .ok_or_else(|| {
                    tenflowers_core::TensorError::unsupported_operation_simple(format!(
                        "Unknown operation for JIT compilation: {}",
                        signature.operation
                    ))
                })?
                .clone()
        };

        let optimized_workgroup_size = self.calculate_optimal_workgroup_size(signature);
        let wgsl_source =
            self.generate_wgsl_shader(signature, &template, optimized_workgroup_size)?;
        let specializations = self.generate_specializations(signature, &template);
        let performance = self.estimate_kernel_performance(signature, &template);

        Ok(CompiledKernel {
            signature: signature.clone(),
            wgsl_source,
            workgroup_size: optimized_workgroup_size,
            specializations,
            compile_time_ms: 0.0, // Will be set by caller
            estimated_performance: performance,
        })
    }

    /// Calculate optimal workgroup size for given tensor shapes
    fn calculate_optimal_workgroup_size(&self, signature: &KernelSignature) -> (u32, u32, u32) {
        let total_elements = signature.output_shape.iter().product::<usize>() as u32;
        let max_workgroup_size = self.device_features.max_workgroup_size;

        match self.optimization_level {
            OptimizationLevel::Debug => (64, 1, 1),
            OptimizationLevel::Balanced => {
                // Balance between occupancy and resource usage
                let size = if total_elements < 1024 {
                    64
                } else if total_elements < 65536 {
                    128
                } else {
                    256
                }
                .min(max_workgroup_size);
                (size, 1, 1)
            }
            OptimizationLevel::Aggressive => {
                // Optimize for memory coalescing and occupancy
                match signature.output_shape.len() {
                    1 => (256.min(max_workgroup_size), 1, 1),
                    2 => {
                        let w = 16.min(signature.output_shape[1] as u32);
                        let h = (256 / w).min(signature.output_shape[0] as u32);
                        (w, h, 1)
                    }
                    3 => {
                        let w = 8.min(signature.output_shape[2] as u32);
                        let h = 8.min(signature.output_shape[1] as u32);
                        let d = (256 / (w * h)).min(signature.output_shape[0] as u32);
                        (w, h, d)
                    }
                    _ => (8, 8, 4), // Conservative for higher dimensions
                }
            }
        }
    }

    /// Generate WGSL shader source code
    fn generate_wgsl_shader(
        &self,
        signature: &KernelSignature,
        template: &GradientKernelTemplate,
        workgroup_size: (u32, u32, u32),
    ) -> Result<String> {
        let mut shader = String::new();

        // Header with workgroup size
        shader.push_str(&format!(
            "// JIT-compiled gradient kernel for {}\n",
            signature.operation
        ));
        shader.push_str(&format!(
            "// Optimized for shapes: {:?} -> {:?}\n",
            signature.input_shapes, signature.output_shape
        ));
        shader.push_str(&format!(
            "@compute @workgroup_size({}, {}, {})\n",
            workgroup_size.0, workgroup_size.1, workgroup_size.2
        ));

        // Storage bindings
        shader.push_str("fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n");
        shader.push_str("    let index = global_id.x;\n");

        // Shape-specific optimizations
        if signature.output_shape.len() == 1 {
            // 1D optimized path
            shader.push_str(&format!(
                "    if (index >= {}u) {{ return; }}\n",
                signature.output_shape[0]
            ));
        } else {
            // Multi-dimensional with bounds checking
            shader.push_str(&format!(
                "    let total_elements = {}u;\n",
                signature.output_shape.iter().product::<usize>()
            ));
            shader.push_str("    if (index >= total_elements) { return; }\n");
        }

        // Operation-specific kernel body
        match signature.operation.as_str() {
            "add_backward" => {
                shader.push_str("    // Addition gradient: pass through\n");
                shader.push_str("    output[index] = grad_output[index];\n");
            }
            "mul_backward" => {
                shader.push_str("    // Multiplication gradient: multiply by other input\n");
                if signature.input_shapes.len() >= 2 {
                    shader.push_str("    output[index] = grad_output[index] * input_b[index];\n");
                }
            }
            "relu_backward" => {
                shader
                    .push_str("    // ReLU gradient: zero if input <= 0, pass through otherwise\n");
                shader.push_str("    let input_val = input[index];\n");
                shader.push_str(
                    "    output[index] = select(0.0, grad_output[index], input_val > 0.0);\n",
                );
            }
            "sigmoid_backward" => {
                shader.push_str(
                    "    // Sigmoid gradient: sigmoid_out * (1 - sigmoid_out) * grad_output\n",
                );
                shader.push_str("    let sigmoid_val = sigmoid_output[index];\n");
                shader.push_str(
                    "    output[index] = grad_output[index] * sigmoid_val * (1.0 - sigmoid_val);\n",
                );
            }
            _ => {
                shader.push_str("    // Generic gradient computation\n");
                shader.push_str("    output[index] = grad_output[index];\n");
            }
        }

        shader.push_str("}\n");

        // Add storage buffer declarations at the beginning
        let mut full_shader = String::new();
        full_shader.push_str("@group(0) @binding(0) var<storage, read> input: array<f32>;\n");
        full_shader.push_str("@group(0) @binding(1) var<storage, read> grad_output: array<f32>;\n");
        full_shader
            .push_str("@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n");

        if template.requires_intermediate_values {
            match signature.operation.as_str() {
                "mul_backward" => {
                    full_shader.push_str(
                        "@group(0) @binding(3) var<storage, read> input_b: array<f32>;\n",
                    );
                }
                "sigmoid_backward" => {
                    full_shader.push_str(
                        "@group(0) @binding(3) var<storage, read> sigmoid_output: array<f32>;\n",
                    );
                }
                _ => {}
            }
        }

        full_shader.push('\n');
        full_shader.push_str(&shader);

        Ok(full_shader)
    }

    /// Generate kernel specializations for different configurations
    fn generate_specializations(
        &self,
        signature: &KernelSignature,
        template: &GradientKernelTemplate,
    ) -> HashMap<String, String> {
        let mut specializations = HashMap::new();

        // Data type specializations
        specializations.insert("dtype".to_string(), signature.dtype.clone());

        // Shape specializations
        specializations.insert(
            "total_elements".to_string(),
            signature.output_shape.iter().product::<usize>().to_string(),
        );

        // Broadcasting specializations
        if template.broadcasting_support && signature.input_shapes.len() > 1 {
            let needs_broadcasting = signature
                .input_shapes
                .iter()
                .any(|shape| shape != &signature.output_shape);
            specializations.insert(
                "needs_broadcasting".to_string(),
                needs_broadcasting.to_string(),
            );
        }

        // Memory access pattern specializations
        let is_contiguous = self.is_contiguous_access(&signature.output_shape);
        specializations.insert("contiguous_access".to_string(), is_contiguous.to_string());

        specializations
    }

    /// Estimate kernel performance characteristics
    fn estimate_kernel_performance(
        &self,
        signature: &KernelSignature,
        _template: &GradientKernelTemplate,
    ) -> KernelPerformance {
        let total_elements = signature.output_shape.iter().product::<usize>() as u64;

        // Estimate FLOPs based on operation
        let estimated_flops = match signature.operation.as_str() {
            "add_backward" | "sum_backward" => total_elements, // Simple pass-through
            "mul_backward" => total_elements * 2,              // One multiplication
            "div_backward" => total_elements * 4,              // Multiple ops for division gradient
            "relu_backward" => total_elements,                 // Conditional assignment
            "sigmoid_backward" => total_elements * 3,          // Multiple arithmetic ops
            "tanh_backward" => total_elements * 3,
            _ => total_elements * 2, // Default estimate
        };

        // Estimate memory accesses
        let input_accesses = signature
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>() as u64)
            .sum::<u64>();
        let output_accesses = total_elements;
        let estimated_memory_accesses = input_accesses + output_accesses;

        // Estimate execution time based on device characteristics
        let compute_time_us =
            estimated_flops as f64 / (self.device_features.compute_units as f64 * 1e9) * 1e6; // Assuming 1 GFLOP/s per compute unit
        let memory_time_us = (estimated_memory_accesses * 4) as f64 / // 4 bytes per f32
            (self.device_features.memory_bandwidth_gb_s as f64 * 1e9)
            * 1e6;

        let estimated_execution_time_us = compute_time_us.max(memory_time_us);
        let memory_bound_ratio = memory_time_us / estimated_execution_time_us;

        KernelPerformance {
            estimated_flops,
            estimated_memory_accesses,
            estimated_execution_time_us,
            memory_bound_ratio: memory_bound_ratio as f32,
        }
    }

    /// Check if memory access pattern is contiguous
    fn is_contiguous_access(&self, shape: &[usize]) -> bool {
        // Simple heuristic: consider 1D or last dimension > 64 as likely contiguous
        shape.len() == 1 || shape.last().unwrap_or(&0) > &64
    }

    /// Get kernel cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.kernel_cache.read().unwrap();
        let total_kernels = cache.len();
        let total_size_estimate = cache.values().map(|k| k.wgsl_source.len()).sum::<usize>();
        (total_kernels, total_size_estimate)
    }

    /// Clear kernel cache
    pub fn clear_cache(&self) {
        let mut cache = self.kernel_cache.write().unwrap();
        cache.clear();
    }

    /// Register a custom gradient kernel template
    pub fn register_template(&self, template: GradientKernelTemplate) {
        let mut registry = self.template_registry.write().unwrap();
        registry.insert(template.operation_name.clone(), template);
    }

    /// Export cached kernels for persistence
    pub fn export_cache(&self) -> Result<String> {
        let cache = self.kernel_cache.read().unwrap();
        let serializable_cache: HashMap<String, serde_json::Value> = cache
            .iter()
            .map(|(sig, kernel)| {
                let key = format!("{sig:?}"); // Simple serialization of signature
                let value = serde_json::json!({
                    "wgsl_source": kernel.wgsl_source,
                    "workgroup_size": kernel.workgroup_size,
                    "compile_time_ms": kernel.compile_time_ms,
                    "performance": kernel.estimated_performance
                });
                (key, value)
            })
            .collect();

        serde_json::to_string_pretty(&serializable_cache).map_err(|e| {
            tenflowers_core::TensorError::unsupported_operation_simple(format!(
                "Failed to serialize kernel cache: {e}"
            ))
        })
    }
}

impl Default for DeviceFeatures {
    fn default() -> Self {
        Self {
            max_workgroup_size: 256,
            max_workgroups_per_dim: 65535,
            supports_f64: false,
            supports_i64: false,
            memory_bandwidth_gb_s: 100.0, // Conservative estimate
            compute_units: 16,            // Conservative estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let device_features = DeviceFeatures::default();
        let compiler = JitCompiler::new(device_features, OptimizationLevel::Balanced);

        let (kernel_count, _) = compiler.cache_stats();
        assert_eq!(kernel_count, 0); // Cache should be empty initially
    }

    #[test]
    fn test_kernel_signature_hash() {
        let sig1 = KernelSignature {
            operation: "add_backward".to_string(),
            input_shapes: vec![vec![100, 100]],
            output_shape: vec![100, 100],
            dtype: "f32".to_string(),
            device_features: DeviceFeatures::default(),
        };

        let sig2 = sig1.clone();
        assert_eq!(sig1, sig2);

        let mut hasher1 = std::collections::hash_map::DefaultHasher::new();
        let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
        sig1.hash(&mut hasher1);
        sig2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_workgroup_size_calculation() {
        let device_features = DeviceFeatures::default();
        let compiler = JitCompiler::new(device_features.clone(), OptimizationLevel::Aggressive);

        let signature = KernelSignature {
            operation: "add_backward".to_string(),
            input_shapes: vec![vec![1024, 1024]],
            output_shape: vec![1024, 1024],
            dtype: "f32".to_string(),
            device_features: DeviceFeatures::default(),
        };

        let workgroup_size = compiler.calculate_optimal_workgroup_size(&signature);
        assert!(workgroup_size.0 > 0);
        assert!(workgroup_size.0 <= device_features.max_workgroup_size);
    }

    #[test]
    fn test_template_registration() {
        let device_features = DeviceFeatures::default();
        let compiler = JitCompiler::new(device_features, OptimizationLevel::Balanced);

        let custom_template = GradientKernelTemplate {
            operation_name: "custom_op_backward".to_string(),
            forward_expression: "custom_forward(x)".to_string(),
            backward_expression: "custom_backward(grad_output, x)".to_string(),
            requires_intermediate_values: true,
            broadcasting_support: false,
        };

        compiler.register_template(custom_template);

        // Verify template was registered by checking the registry
        let registry = compiler.template_registry.read().unwrap();
        assert!(registry.contains_key("custom_op_backward"));
    }

    #[test]
    fn test_performance_estimation() {
        let device_features = DeviceFeatures::default();
        let compiler = JitCompiler::new(device_features, OptimizationLevel::Balanced);

        let signature = KernelSignature {
            operation: "mul_backward".to_string(),
            input_shapes: vec![vec![1000], vec![1000]],
            output_shape: vec![1000],
            dtype: "f32".to_string(),
            device_features: DeviceFeatures::default(),
        };

        let template = GradientKernelTemplate {
            operation_name: "mul_backward".to_string(),
            forward_expression: "a * b".to_string(),
            backward_expression: "grad_output * other_input".to_string(),
            requires_intermediate_values: true,
            broadcasting_support: true,
        };

        let performance = compiler.estimate_kernel_performance(&signature, &template);
        assert!(performance.estimated_flops > 0);
        assert!(performance.estimated_memory_accesses > 0);
        assert!(performance.estimated_execution_time_us > 0.0);
        assert!(performance.memory_bound_ratio >= 0.0 && performance.memory_bound_ratio <= 1.0);
    }
}
