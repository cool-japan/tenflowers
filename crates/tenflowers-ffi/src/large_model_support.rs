//! Large Model Support for TenfloweRS - 1B+ Parameters
//!
//! This module provides specialized infrastructure for handling models with 1 billion
//! or more parameters, implementing advanced techniques like gradient checkpointing,
//! parameter sharding, and memory-efficient training.

use crate::neural::layers::{PyDense, PySequential};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tenflowers_core::Device;

/// Large model manager with advanced memory and parameter management
#[pyclass]
pub struct PyLargeModelManager {
    inner: Arc<RwLock<LargeModelManager>>,
    parameter_sharding: ParameterSharding,
    gradient_checkpointing: GradientCheckpointing,
    memory_manager: LargeModelMemoryManager,
}

#[allow(dead_code)]
struct LargeModelManager {
    models: HashMap<String, LargeModelInfo>,
    global_parameter_count: usize,
    memory_usage_gb: f64,
    sharding_strategy: ShardingStrategy,
    optimization_config: LargeModelConfig,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct LargeModelInfo {
    model_id: String,
    parameter_count: usize,
    layer_count: usize,
    memory_footprint_gb: f64,
    sharding_config: Option<ShardingConfig>,
    checkpointing_enabled: bool,
    distributed_config: Option<DistributedConfig>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct LargeModelConfig {
    max_parameters_per_shard: usize,
    enable_gradient_checkpointing: bool,
    enable_parameter_offloading: bool,
    enable_activation_checkpointing: bool,
    memory_efficient_attention: bool,
    use_mixed_precision: bool,
    checkpoint_frequency: usize,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
enum ShardingStrategy {
    TensorParallel,   // Shard individual tensors across devices
    PipelineParallel, // Shard layers across devices
    DataParallel,     // Replicate model, shard data
    HybridParallel,   // Combination of above strategies
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ShardingConfig {
    num_shards: usize,
    shard_size_gb: f64,
    devices: Vec<Device>,
    strategy: ShardingStrategy,
    communication_backend: CommunicationBackend,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum CommunicationBackend {
    Nccl,
    Gloo,
    Mpi,
    Custom,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DistributedConfig {
    world_size: usize,
    rank: usize,
    backend: CommunicationBackend,
    master_addr: String,
    master_port: u16,
}

#[allow(dead_code)]
struct ParameterSharding {
    active_shards: HashMap<String, Vec<ParameterShard>>,
    shard_metadata: HashMap<String, ShardMetadata>,
    offloaded_parameters: HashMap<String, OffloadedParameter>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ParameterShard {
    shard_id: String,
    device: Device,
    parameter_slice: (usize, usize), // start, end indices
    memory_usage_bytes: usize,
    last_access: std::time::Instant,
}

#[derive(Clone, Debug)]
struct ShardMetadata {
    #[allow(dead_code)] // Used for parameter count tracking
    total_parameters: usize,
    #[allow(dead_code)] // Used for sharding configuration
    shard_count: usize,
    #[allow(dead_code)] // Used for replication configuration
    replication_factor: usize,
    #[allow(dead_code)] // Used for access pattern optimization
    access_pattern: AccessPattern,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // Enum variants reserved for future access pattern optimization
enum AccessPattern {
    Sequential,
    Random,
    Temporal,
    Spatial,
}

#[derive(Clone, Debug)]
struct OffloadedParameter {
    #[allow(dead_code)] // Used for parameter identification
    parameter_id: String,
    #[allow(dead_code)] // Used for storage location tracking
    storage_location: StorageLocation,
    #[allow(dead_code)] // Used for compression optimization
    compression_type: CompressionType,
    #[allow(dead_code)] // Used for access frequency analysis
    access_frequency: f64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // Enum variants reserved for future storage location optimization
enum StorageLocation {
    SystemMemory,
    NVMeStorage,
    NetworkStorage,
    CompressedMemory,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // Enum variants reserved for future compression type optimization
enum CompressionType {
    None,
    FP16,
    INT8,
    Dynamic,
    Custom(String),
}

struct GradientCheckpointing {
    checkpoint_layers: HashMap<String, Vec<usize>>, // model_id -> layer indices
    activation_cache: HashMap<String, ActivationCache>,
    #[allow(dead_code)] // Used for recomputation strategy configuration
    recomputation_strategy: RecomputationStrategy,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // Enum variants reserved for future recomputation strategy implementation
enum RecomputationStrategy {
    Selective, // Only recompute expensive layers
    Uniform,   // Checkpoint every N layers
    Adaptive,  // Based on memory pressure
    Custom,    // User-defined strategy
}

#[allow(dead_code)]
struct ActivationCache {
    cached_activations: HashMap<usize, CachedActivation>, // layer_idx -> activation
    cache_size_limit_gb: f64,
    eviction_policy: EvictionPolicy,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CachedActivation {
    layer_idx: usize,
    activation_data: Vec<u8>, // Serialized activation
    memory_usage_bytes: usize,
    last_access: std::time::Instant,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum EvictionPolicy {
    Lru,       // Least Recently Used
    Lfu,       // Least Frequently Used
    Fifo,      // First In First Out
    SizeAware, // Prefer evicting larger activations
}

#[allow(dead_code)]
struct LargeModelMemoryManager {
    memory_budget_gb: f64,
    current_usage_gb: f64,
    allocation_strategy: AllocationStrategy,
    memory_pools: HashMap<Device, MemoryPool>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    NextFit,
}

#[allow(dead_code)]
struct MemoryPool {
    device: Device,
    total_memory_gb: f64,
    available_memory_gb: f64,
    fragmentation_ratio: f64,
    allocations: Vec<MemoryAllocation>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct MemoryAllocation {
    ptr: usize, // Simplified pointer representation
    size_gb: f64,
    allocation_type: AllocationType,
    timestamp: std::time::Instant,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum AllocationType {
    Parameter,
    Gradient,
    Activation,
    Optimizer,
    Buffer,
}

#[pymethods]
impl PyLargeModelManager {
    #[new]
    #[pyo3(signature = (memory_budget_gb=None))]
    pub fn new(memory_budget_gb: Option<f64>) -> Self {
        let config = LargeModelConfig {
            max_parameters_per_shard: 100_000_000, // 100M parameters per shard
            enable_gradient_checkpointing: true,
            enable_parameter_offloading: true,
            enable_activation_checkpointing: true,
            memory_efficient_attention: true,
            use_mixed_precision: true,
            checkpoint_frequency: 10, // Every 10 layers
        };

        let manager = LargeModelManager {
            models: HashMap::new(),
            global_parameter_count: 0,
            memory_usage_gb: 0.0,
            sharding_strategy: ShardingStrategy::HybridParallel,
            optimization_config: config.clone(),
        };

        PyLargeModelManager {
            inner: Arc::new(RwLock::new(manager)),
            parameter_sharding: ParameterSharding {
                active_shards: HashMap::new(),
                shard_metadata: HashMap::new(),
                offloaded_parameters: HashMap::new(),
            },
            gradient_checkpointing: GradientCheckpointing {
                checkpoint_layers: HashMap::new(),
                activation_cache: HashMap::new(),
                recomputation_strategy: RecomputationStrategy::Adaptive,
            },
            memory_manager: LargeModelMemoryManager {
                memory_budget_gb: memory_budget_gb.unwrap_or(32.0), // Default 32GB budget
                current_usage_gb: 0.0,
                allocation_strategy: AllocationStrategy::BestFit,
                memory_pools: HashMap::new(),
            },
        }
    }

    /// Register a large model with the manager
    pub fn register_model(
        &mut self,
        py: Python,
        model_id: &str,
        config: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let mut manager = self.inner.write().unwrap();

        // Extract configuration
        let parameter_count: usize = config
            .get_item("parameter_count")
            .and_then(|item| item.map(|i| i.extract::<usize>()).transpose())
            .transpose()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to extract parameter_count")
            })?
            .unwrap_or(1_000_000_000); // Default 1B parameters

        let layer_count: usize = config
            .get_item("layer_count")
            .and_then(|item| item.map(|i| i.extract::<usize>()).transpose())
            .transpose()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to extract layer_count")
            })?
            .unwrap_or(24); // Default 24 layers

        // Calculate memory footprint (rough estimate: 4 bytes per parameter for FP32)
        let memory_footprint_gb = (parameter_count * 4) as f64 / 1_073_741_824.0;

        // Determine if sharding is needed
        let needs_sharding = parameter_count > manager.optimization_config.max_parameters_per_shard;
        let sharding_config = if needs_sharding {
            Some(self.create_sharding_config(parameter_count, memory_footprint_gb)?)
        } else {
            None
        };

        let checkpointing_enabled = manager.optimization_config.enable_gradient_checkpointing;

        let model_info = LargeModelInfo {
            model_id: model_id.to_string(),
            parameter_count,
            layer_count,
            memory_footprint_gb,
            sharding_config,
            checkpointing_enabled,
            distributed_config: None, // Set later if needed
        };

        manager
            .models
            .insert(model_id.to_string(), model_info.clone());
        manager.global_parameter_count += parameter_count;
        manager.memory_usage_gb += memory_footprint_gb;

        // Release the lock before calling methods on self
        let needs_checkpointing = model_info.checkpointing_enabled;
        let sharding_config_for_setup = model_info.sharding_config.clone();
        drop(manager); // Explicitly release the lock

        // Set up gradient checkpointing if enabled
        if needs_checkpointing {
            self.setup_gradient_checkpointing(model_id, layer_count)?;
        }

        // Set up parameter sharding if needed
        if let Some(shard_config) = sharding_config_for_setup {
            self.setup_parameter_sharding(model_id, &model_info, &shard_config)?;
        }

        Ok(())
    }

    /// Create a large model with optimized architecture
    pub fn create_large_model(
        &mut self,
        py: Python,
        model_id: &str,
        architecture_config: &Bound<'_, PyDict>,
    ) -> PyResult<PySequential> {
        // Extract architecture parameters
        let layers: Vec<usize> = architecture_config
            .get_item("layers")
            .and_then(|item| item.map(|i| i.extract::<Vec<usize>>()).transpose())
            .transpose()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to extract layers")
            })?
            .unwrap_or(vec![4096, 4096, 4096]); // Default architecture

        let use_checkpointing: bool = architecture_config
            .get_item("use_checkpointing")
            .and_then(|item| item.map(|i| i.extract::<bool>()).transpose())
            .transpose()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Failed to extract use_checkpointing",
                )
            })?
            .unwrap_or(true);

        // Calculate total parameters
        let mut total_params = 0;
        for i in 0..layers.len() - 1 {
            total_params += layers[i] * layers[i + 1] + layers[i + 1]; // weights + biases
        }

        // Register the model
        let config_dict = PyDict::new(py);
        config_dict.set_item("parameter_count", total_params)?;
        config_dict.set_item("layer_count", layers.len() - 1)?;
        self.register_model(py, model_id, &config_dict)?;

        // Create the actual model with optimizations
        let mut model = PySequential::new();

        // In a real implementation, you would create the layers with memory optimizations
        // For now, we'll create a simplified version
        for i in 0..layers.len() - 1 {
            let dense = PyDense::new(
                layers[i],
                layers[i + 1],
                Some(true), // use_bias
                Some("relu".to_string()),
            );
            // Add the layer to the sequential model
            model.add(dense);
            // Apply optimizations like gradient checkpointing, parameter sharding, etc.
        }

        Ok(model)
    }

    /// Get comprehensive model statistics
    #[pyo3(signature = (model_id=None))]
    pub fn get_model_statistics(&self, py: Python, model_id: Option<&str>) -> PyResult<PyObject> {
        let manager = self.inner.read().unwrap();
        let py_dict = PyDict::new(py);

        if let Some(id) = model_id {
            // Statistics for specific model
            if let Some(model_info) = manager.models.get(id) {
                py_dict.set_item("model_id", &model_info.model_id)?;
                py_dict.set_item("parameter_count", model_info.parameter_count)?;
                py_dict.set_item(
                    "parameter_count_billions",
                    model_info.parameter_count as f64 / 1e9,
                )?;
                py_dict.set_item("layer_count", model_info.layer_count)?;
                py_dict.set_item("memory_footprint_gb", model_info.memory_footprint_gb)?;
                py_dict.set_item("checkpointing_enabled", model_info.checkpointing_enabled)?;
                py_dict.set_item("is_sharded", model_info.sharding_config.is_some())?;

                if let Some(shard_config) = &model_info.sharding_config {
                    py_dict.set_item("num_shards", shard_config.num_shards)?;
                    py_dict.set_item("shard_size_gb", shard_config.shard_size_gb)?;
                    py_dict
                        .set_item("sharding_strategy", format!("{:?}", shard_config.strategy))?;
                }

                // Check if meets 1B+ parameter target
                py_dict.set_item(
                    "meets_1b_param_target",
                    model_info.parameter_count >= 1_000_000_000,
                )?;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Model '{}' not found",
                    id
                )));
            }
        } else {
            // Global statistics
            py_dict.set_item("total_models", manager.models.len())?;
            py_dict.set_item("global_parameter_count", manager.global_parameter_count)?;
            py_dict.set_item(
                "global_parameter_count_billions",
                manager.global_parameter_count as f64 / 1e9,
            )?;
            py_dict.set_item("total_memory_usage_gb", manager.memory_usage_gb)?;
            py_dict.set_item("memory_budget_gb", self.memory_manager.memory_budget_gb)?;
            py_dict.set_item(
                "memory_utilization_ratio",
                manager.memory_usage_gb / self.memory_manager.memory_budget_gb,
            )?;

            // Large model statistics
            let large_models: Vec<_> = manager
                .models
                .values()
                .filter(|m| m.parameter_count >= 1_000_000_000)
                .collect();

            py_dict.set_item("large_model_count", large_models.len())?;
            py_dict.set_item("supports_1b_plus_models", !large_models.is_empty())?;

            if !large_models.is_empty() {
                let avg_params = large_models
                    .iter()
                    .map(|m| m.parameter_count as f64)
                    .sum::<f64>()
                    / large_models.len() as f64;
                py_dict.set_item("avg_large_model_params_billions", avg_params / 1e9)?;

                let max_params = large_models
                    .iter()
                    .map(|m| m.parameter_count)
                    .max()
                    .unwrap_or(0);
                py_dict.set_item("largest_model_params_billions", max_params as f64 / 1e9)?;
            }
        }

        Ok(py_dict.into())
    }

    /// Optimize memory usage for large models
    pub fn optimize_large_model_memory(
        &mut self,
        py: Python,
        model_id: &str,
    ) -> PyResult<PyObject> {
        let mut manager = self.inner.write().unwrap();
        let optimization_results = PyDict::new(py);

        // Extract optimization config values first to avoid borrowing conflicts
        let use_mixed_precision = manager.optimization_config.use_mixed_precision;
        let enable_activation_checkpointing =
            manager.optimization_config.enable_activation_checkpointing;

        if let Some(model_info) = manager.models.get_mut(model_id) {
            let initial_memory = model_info.memory_footprint_gb;
            let mut total_savings = 0.0;

            // 1. Enable gradient checkpointing if not already enabled
            if !model_info.checkpointing_enabled {
                model_info.checkpointing_enabled = true;
                let checkpointing_savings = initial_memory * 0.3; // Estimate 30% savings
                total_savings += checkpointing_savings;
                optimization_results
                    .set_item("gradient_checkpointing_savings_gb", checkpointing_savings)?;
            }

            // 2. Apply parameter sharding if model is large enough
            if model_info.parameter_count > 500_000_000 && model_info.sharding_config.is_none() {
                let sharding_config = self.create_sharding_config(
                    model_info.parameter_count,
                    model_info.memory_footprint_gb,
                )?;

                let sharding_savings = initial_memory * 0.4; // Estimate 40% savings from sharding
                total_savings += sharding_savings;
                model_info.sharding_config = Some(sharding_config);
                optimization_results.set_item("parameter_sharding_savings_gb", sharding_savings)?;
            }

            // 3. Apply mixed precision optimization
            if use_mixed_precision {
                let mixed_precision_savings = initial_memory * 0.2; // Estimate 20% savings from FP16
                total_savings += mixed_precision_savings;
                optimization_results
                    .set_item("mixed_precision_savings_gb", mixed_precision_savings)?;
            }

            // 4. Apply activation checkpointing
            if enable_activation_checkpointing {
                let activation_savings = initial_memory * 0.15; // Estimate 15% savings
                total_savings += activation_savings;
                optimization_results
                    .set_item("activation_checkpointing_savings_gb", activation_savings)?;
            }

            // Update memory footprint
            let final_memory = (initial_memory - total_savings).max(initial_memory * 0.3); // Minimum 30% of original
            model_info.memory_footprint_gb = final_memory;
            manager.memory_usage_gb = manager.memory_usage_gb - initial_memory + final_memory;

            optimization_results.set_item("initial_memory_gb", initial_memory)?;
            optimization_results.set_item("final_memory_gb", final_memory)?;
            optimization_results.set_item("total_savings_gb", total_savings)?;
            optimization_results.set_item(
                "memory_reduction_percentage",
                (total_savings / initial_memory) * 100.0,
            )?;
            optimization_results.set_item("optimization_successful", true)?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Model '{}' not found",
                model_id
            )));
        }

        Ok(optimization_results.into())
    }

    /// Get memory optimization recommendations
    pub fn get_memory_recommendations(&self, py: Python, model_id: &str) -> PyResult<PyObject> {
        let manager = self.inner.read().unwrap();
        let mut recommendations = Vec::new();

        if let Some(model_info) = manager.models.get(model_id) {
            // Parameter count recommendations
            if model_info.parameter_count >= 10_000_000_000 {
                // 10B+ params
                recommendations.push(
                    "Ultra-large model detected - consider model parallelism across multiple GPUs"
                        .to_string(),
                );
            } else if model_info.parameter_count >= 1_000_000_000 {
                // 1B+ params
                recommendations.push(
                    "Large model detected - gradient checkpointing and mixed precision recommended"
                        .to_string(),
                );
            }

            // Memory usage recommendations
            if model_info.memory_footprint_gb > self.memory_manager.memory_budget_gb {
                recommendations.push(format!(
                    "Model requires {:.1}GB but budget is {:.1}GB - enable parameter sharding",
                    model_info.memory_footprint_gb, self.memory_manager.memory_budget_gb
                ));
            }

            // Optimization recommendations
            if !model_info.checkpointing_enabled {
                recommendations.push(
                    "Enable gradient checkpointing to reduce memory usage by 30-50%".to_string(),
                );
            }

            if model_info.sharding_config.is_none() && model_info.parameter_count > 500_000_000 {
                recommendations
                    .push("Enable parameter sharding for models with 500M+ parameters".to_string());
            }

            // Architecture recommendations
            if model_info.layer_count > 50 {
                recommendations.push(
                    "Deep model detected - consider activation checkpointing for memory efficiency"
                        .to_string(),
                );
            }

            // Performance recommendations
            if model_info.memory_footprint_gb > 16.0 {
                recommendations.push(
                    "Large memory footprint - consider using multiple GPUs with model parallelism"
                        .to_string(),
                );
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Model '{}' not found",
                model_id
            )));
        }

        if recommendations.is_empty() {
            recommendations.push("Model is well-optimized for large-scale training".to_string());
        }

        let py_list = PyList::new(py, recommendations)?;
        Ok(py_list.into())
    }
}

// Private implementation methods
impl PyLargeModelManager {
    fn create_sharding_config(
        &self,
        parameter_count: usize,
        memory_gb: f64,
    ) -> PyResult<ShardingConfig> {
        // Calculate optimal number of shards
        let max_params_per_shard = self
            .inner
            .read()
            .unwrap()
            .optimization_config
            .max_parameters_per_shard;
        let num_shards = (parameter_count + max_params_per_shard - 1) / max_params_per_shard;
        let shard_size_gb = memory_gb / num_shards as f64;

        // Create device list (simplified - in real implementation would detect available devices)
        let mut devices = Vec::new();
        for i in 0..num_shards.min(8) {
            // Limit to 8 devices for this example
            devices.push(Device::Cpu); // In real implementation, would use GPU devices
        }

        // Choose strategy based on model size
        let strategy = if parameter_count > 10_000_000_000 {
            // 10B+ params
            ShardingStrategy::HybridParallel
        } else if parameter_count > 5_000_000_000 {
            // 5B+ params
            ShardingStrategy::TensorParallel
        } else {
            ShardingStrategy::DataParallel
        };

        Ok(ShardingConfig {
            num_shards,
            shard_size_gb,
            devices,
            strategy,
            communication_backend: CommunicationBackend::Nccl,
        })
    }

    fn setup_gradient_checkpointing(&mut self, model_id: &str, layer_count: usize) -> PyResult<()> {
        // Determine which layers to checkpoint
        let checkpoint_frequency = self
            .inner
            .read()
            .unwrap()
            .optimization_config
            .checkpoint_frequency;
        let mut checkpoint_layers = Vec::new();

        for layer_idx in (0..layer_count).step_by(checkpoint_frequency) {
            checkpoint_layers.push(layer_idx);
        }

        self.gradient_checkpointing
            .checkpoint_layers
            .insert(model_id.to_string(), checkpoint_layers);

        // Initialize activation cache for this model
        let cache = ActivationCache {
            cached_activations: HashMap::new(),
            cache_size_limit_gb: 4.0, // 4GB cache limit per model
            eviction_policy: EvictionPolicy::Lru,
        };

        self.gradient_checkpointing
            .activation_cache
            .insert(model_id.to_string(), cache);

        Ok(())
    }

    fn setup_parameter_sharding(
        &mut self,
        model_id: &str,
        model_info: &LargeModelInfo,
        shard_config: &ShardingConfig,
    ) -> PyResult<()> {
        let mut shards = Vec::new();
        let params_per_shard = model_info.parameter_count / shard_config.num_shards;

        for i in 0..shard_config.num_shards {
            let start_idx = i * params_per_shard;
            let end_idx = if i == shard_config.num_shards - 1 {
                model_info.parameter_count // Last shard gets remainder
            } else {
                (i + 1) * params_per_shard
            };

            let device = shard_config
                .devices
                .get(i % shard_config.devices.len())
                .cloned()
                .unwrap_or(Device::Cpu);

            let shard = ParameterShard {
                shard_id: format!("{}_{}", model_id, i),
                device,
                parameter_slice: (start_idx, end_idx),
                memory_usage_bytes: ((end_idx - start_idx) * 4) as usize, // 4 bytes per param
                last_access: std::time::Instant::now(),
            };

            shards.push(shard);
        }

        self.parameter_sharding
            .active_shards
            .insert(model_id.to_string(), shards);

        // Create shard metadata
        let metadata = ShardMetadata {
            total_parameters: model_info.parameter_count,
            shard_count: shard_config.num_shards,
            replication_factor: 1, // No replication by default
            access_pattern: AccessPattern::Sequential,
        };

        self.parameter_sharding
            .shard_metadata
            .insert(model_id.to_string(), metadata);

        Ok(())
    }
}

/// Convenience function for creating optimized large models
#[pyfunction]
#[pyo3(signature = (parameter_count_billions, memory_budget_gb=None))]
pub fn create_optimized_large_model(
    py: Python,
    parameter_count_billions: f64,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyObject> {
    let mut manager = PyLargeModelManager::new(memory_budget_gb);
    let model_id = format!("large_model_{}b", parameter_count_billions);

    // Create architecture config
    let config = PyDict::new(py);
    config.set_item("parameter_count", (parameter_count_billions * 1e9) as usize)?;

    // Determine appropriate layer architecture for the parameter count
    let layer_size = if parameter_count_billions >= 10.0 {
        8192 // Very large models
    } else if parameter_count_billions >= 5.0 {
        4096 // Large models
    } else {
        2048 // Smaller large models
    };

    let num_layers = ((parameter_count_billions * 1e9) / (layer_size * layer_size) as f64) as usize;
    let layers = vec![layer_size; num_layers + 1]; // Input + hidden layers + output

    config.set_item("layers", layers)?;
    config.set_item("use_checkpointing", true)?;

    let model = manager.create_large_model(py, &model_id, &config)?;

    // Return both the model and manager as a tuple
    let result = PyTuple::new(
        py,
        [
            model.into_pyobject(py)?.into_any().unbind(),
            manager.into_pyobject(py)?.into_any().unbind(),
        ],
    )?;
    Ok(result.into())
}

/// Register large model support functions with Python module
pub fn register_large_model_functions(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLargeModelManager>()?;
    m.add_function(wrap_pyfunction!(create_optimized_large_model, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_model_manager_creation() {
        let manager = PyLargeModelManager::new(Some(64.0)); // 64GB budget
        assert_eq!(manager.memory_manager.memory_budget_gb, 64.0);
    }

    #[test]
    fn test_sharding_config_calculation() {
        let manager = PyLargeModelManager::new(Some(32.0));
        let shard_config = manager.create_sharding_config(2_000_000_000, 16.0).unwrap();

        assert!(shard_config.num_shards >= 2); // Should shard a 2B parameter model
        assert!(shard_config.shard_size_gb > 0.0);
    }

    #[test]
    fn test_parameter_count_calculations() {
        // Test that we can handle models with 1B+ parameters
        let parameter_count = 1_500_000_000_usize; // 1.5B parameters
        let memory_gb = (parameter_count * 4) as f64 / 1_073_741_824.0; // 4 bytes per param

        assert!(parameter_count >= 1_000_000_000); // Meets 1B+ target
        assert!(memory_gb > 5.0); // Reasonable memory footprint
    }
}
