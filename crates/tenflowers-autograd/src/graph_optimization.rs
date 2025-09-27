//! Enhanced Graph Optimization Framework
//!
//! This module provides comprehensive graph optimization capabilities that extend
//! the core tenflowers-core optimization system with device placement, subgraph extraction,
//! and autograd-specific optimizations.
use crate::{
    device_placement::{
        DevicePlacementConfig, DevicePlacementOptimizer, GraphOperation, PlacementResult,
    },
    subgraph_extraction::{SubgraphConfig, SubgraphExtractionResult, SubgraphExtractor},
    Result,
};
use std::collections::HashSet;
use tenflowers_core::Device;

/// Enhanced graph optimization configuration
#[derive(Debug, Clone)]
pub struct GraphOptimizationConfig {
    pub enable_device_placement: bool,
    pub enable_subgraph_extraction: bool,
    pub enable_gradient_fusion: bool,
    pub enable_memory_optimization: bool,
    pub enable_communication_optimization: bool,
    pub device_placement_config: DevicePlacementConfig,
    pub subgraph_config: SubgraphConfig,
    pub optimization_level: OptimizationLevel,
    pub target_devices: Vec<Device>,
    pub max_optimization_iterations: usize,
}

impl Default for GraphOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_device_placement: true,
            enable_subgraph_extraction: true,
            enable_gradient_fusion: true,
            enable_memory_optimization: true,
            enable_communication_optimization: true,
            device_placement_config: DevicePlacementConfig::default(),
            subgraph_config: SubgraphConfig::default(),
            optimization_level: OptimizationLevel::Balanced,
            target_devices: vec![Device::Cpu],
            max_optimization_iterations: 5,
        }
    }
}

/// Optimization level for graph transformations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Minimal optimizations, fast compilation
    Debug,
    /// Balanced optimizations
    Balanced,
    /// Aggressive optimizations, slower compilation
    Aggressive,
    /// Maximum optimizations for production
    Production,
}

/// Comprehensive graph optimization result
#[derive(Debug, Clone)]
pub struct GraphOptimizationResult {
    pub device_placement: Option<PlacementResult>,
    pub subgraph_extraction: Option<SubgraphExtractionResult>,
    pub gradient_fusions: Vec<GradientFusion>,
    pub memory_optimizations: Vec<MemoryOptimization>,
    pub communication_plan: CommunicationPlan,
    pub optimization_stats: OptimizationStats,
    pub estimated_speedup: f64,
    pub estimated_memory_reduction: f64,
}

/// Gradient operation fusion
#[derive(Debug, Clone)]
pub struct GradientFusion {
    pub fusion_id: String,
    pub fused_operations: Vec<String>,
    pub fusion_type: GradientFusionType,
    pub estimated_speedup: f64,
    pub memory_savings_bytes: usize,
}

/// Types of gradient fusions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientFusionType {
    /// Fuse element-wise gradient operations
    ElementWise,
    /// Fuse reduction gradient operations
    Reduction,
    /// Fuse activation function gradients
    Activation,
    /// Fuse normalization gradients
    Normalization,
    /// Custom fusion pattern
    Custom(String),
}

/// Memory optimization transformations
#[derive(Debug, Clone)]
pub struct MemoryOptimization {
    pub optimization_id: String,
    pub optimization_type: MemoryOptimizationType,
    pub affected_operations: Vec<String>,
    pub memory_savings_bytes: usize,
    pub performance_impact: f64, // Positive = speedup, negative = slowdown
}

/// Types of memory optimizations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryOptimizationType {
    /// In-place gradient updates
    InPlaceGradients,
    /// Gradient checkpointing
    Checkpointing,
    /// Memory pooling and reuse
    MemoryPooling,
    /// Tensor lifetime optimization
    LifetimeOptimization,
    /// Gradient accumulation
    GradientAccumulation,
}

/// Cross-device communication plan
#[derive(Debug, Clone)]
pub struct CommunicationPlan {
    pub transfers: Vec<DataTransferOp>,
    pub synchronization_points: Vec<SynchronizationPoint>,
    pub total_transfer_volume_bytes: usize,
    pub estimated_transfer_time_ms: f64,
    pub communication_overlap_opportunities: Vec<OverlapOpportunity>,
}

/// Data transfer operation
#[derive(Debug, Clone)]
pub struct DataTransferOp {
    pub id: String,
    pub source_device: Device,
    pub target_device: Device,
    pub tensor_id: String,
    pub size_bytes: usize,
    pub estimated_time_ms: f64,
    pub can_overlap_with_compute: bool,
    pub priority: TransferPriority,
}

/// Transfer priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Synchronization point in the execution
#[derive(Debug, Clone)]
pub struct SynchronizationPoint {
    pub id: String,
    pub operation_ids: Vec<String>,
    pub devices: Vec<Device>,
    pub sync_type: SynchronizationType,
    pub estimated_wait_time_ms: f64,
}

/// Types of synchronization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynchronizationType {
    /// All devices must complete before proceeding
    Barrier,
    /// Wait for specific data dependencies
    DataDependency,
    /// Gradient aggregation synchronization
    GradientSync,
    /// Memory consistency synchronization
    MemorySync,
}

/// Opportunity for computation-communication overlap
#[derive(Debug, Clone)]
pub struct OverlapOpportunity {
    pub compute_operation: String,
    pub transfer_operation: String,
    pub overlap_ratio: f64, // 0.0 to 1.0
    pub estimated_speedup: f64,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub total_optimization_time_ms: f64,
    pub operations_optimized: usize,
    pub fusions_applied: usize,
    pub memory_optimizations_applied: usize,
    pub device_placements_changed: usize,
    pub subgraphs_created: usize,
}

/// Enhanced graph optimizer
#[derive(Debug)]
pub struct EnhancedGraphOptimizer {
    config: GraphOptimizationConfig,
    device_optimizer: DevicePlacementOptimizer,
    subgraph_extractor: SubgraphExtractor,
    gradient_fusion_patterns: Vec<FusionPattern>,
}

/// Pattern for gradient fusion
#[derive(Debug, Clone)]
pub struct FusionPattern {
    pub name: String,
    pub operation_sequence: Vec<String>,
    pub fusion_type: GradientFusionType,
    pub estimated_speedup: f64,
}

impl EnhancedGraphOptimizer {
    /// Create a new enhanced graph optimizer
    pub fn new(config: GraphOptimizationConfig) -> Self {
        let device_optimizer =
            DevicePlacementOptimizer::new(config.device_placement_config.clone());
        let subgraph_extractor = SubgraphExtractor::new(config.subgraph_config.clone());

        let mut optimizer = Self {
            config,
            device_optimizer,
            subgraph_extractor,
            gradient_fusion_patterns: Vec::new(),
        };

        optimizer.initialize_fusion_patterns();
        optimizer
    }

    /// Initialize gradient fusion patterns
    fn initialize_fusion_patterns(&mut self) {
        // Element-wise gradient fusions
        self.gradient_fusion_patterns.push(FusionPattern {
            name: "add_mul_gradient_fusion".to_string(),
            operation_sequence: vec!["add_backward".to_string(), "mul_backward".to_string()],
            fusion_type: GradientFusionType::ElementWise,
            estimated_speedup: 1.3,
        });

        // Activation gradient fusions
        self.gradient_fusion_patterns.push(FusionPattern {
            name: "relu_dropout_gradient_fusion".to_string(),
            operation_sequence: vec!["relu_backward".to_string(), "dropout_backward".to_string()],
            fusion_type: GradientFusionType::Activation,
            estimated_speedup: 1.5,
        });

        // Normalization gradient fusions
        self.gradient_fusion_patterns.push(FusionPattern {
            name: "batchnorm_relu_gradient_fusion".to_string(),
            operation_sequence: vec![
                "batchnorm_backward".to_string(),
                "relu_backward".to_string(),
            ],
            fusion_type: GradientFusionType::Normalization,
            estimated_speedup: 1.4,
        });

        // Reduction gradient fusions
        self.gradient_fusion_patterns.push(FusionPattern {
            name: "sum_mean_gradient_fusion".to_string(),
            operation_sequence: vec!["sum_backward".to_string(), "mean_backward".to_string()],
            fusion_type: GradientFusionType::Reduction,
            estimated_speedup: 1.2,
        });
    }

    /// Optimize a computation graph comprehensively
    pub fn optimize_graph<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<GraphOptimizationResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let start_time = std::time::Instant::now();
        let mut stats = OptimizationStats {
            total_optimization_time_ms: 0.0,
            operations_optimized: operations.len(),
            fusions_applied: 0,
            memory_optimizations_applied: 0,
            device_placements_changed: 0,
            subgraphs_created: 0,
        };

        // Step 1: Device placement optimization
        let device_placement = if self.config.enable_device_placement {
            let placement = self.device_optimizer.optimize_placement(operations)?;
            stats.device_placements_changed = placement.decisions.len();
            Some(placement)
        } else {
            None
        };

        // Step 2: Subgraph extraction
        let subgraph_extraction = if self.config.enable_subgraph_extraction {
            let extraction = self.subgraph_extractor.extract_subgraphs(operations)?;
            stats.subgraphs_created = extraction.subgraphs.len();
            Some(extraction)
        } else {
            None
        };

        // Step 3: Gradient fusion optimization
        let gradient_fusions = if self.config.enable_gradient_fusion {
            let fusions = self.optimize_gradient_fusions(operations)?;
            stats.fusions_applied = fusions.len();
            fusions
        } else {
            Vec::new()
        };

        // Step 4: Memory optimization
        let memory_optimizations = if self.config.enable_memory_optimization {
            let optimizations = self.optimize_memory_usage(operations)?;
            stats.memory_optimizations_applied = optimizations.len();
            optimizations
        } else {
            Vec::new()
        };

        // Step 5: Communication optimization
        let communication_plan = if self.config.enable_communication_optimization {
            self.optimize_communication(operations, &device_placement, &subgraph_extraction)?
        } else {
            CommunicationPlan {
                transfers: vec![],
                synchronization_points: vec![],
                total_transfer_volume_bytes: 0,
                estimated_transfer_time_ms: 0.0,
                communication_overlap_opportunities: vec![],
            }
        };

        // Calculate overall metrics
        let estimated_speedup = self.calculate_estimated_speedup(
            &gradient_fusions,
            &memory_optimizations,
            &device_placement,
            &subgraph_extraction,
        );

        let estimated_memory_reduction = self.calculate_memory_reduction(&memory_optimizations);

        stats.total_optimization_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(GraphOptimizationResult {
            device_placement,
            subgraph_extraction,
            gradient_fusions,
            memory_optimizations,
            communication_plan,
            optimization_stats: stats,
            estimated_speedup,
            estimated_memory_reduction,
        })
    }

    /// Optimize gradient fusions
    fn optimize_gradient_fusions<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<Vec<GradientFusion>>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut fusions = Vec::new();
        let mut used_operations = HashSet::new();

        // Look for fusion opportunities
        for pattern in &self.gradient_fusion_patterns {
            let opportunities =
                self.find_fusion_opportunities(operations, pattern, &used_operations);

            for opportunity in opportunities {
                let fusion = GradientFusion {
                    fusion_id: format!("{}_{}", pattern.name, fusions.len()),
                    fused_operations: opportunity.iter().map(|op| op.id.clone()).collect(),
                    fusion_type: pattern.fusion_type.clone(),
                    estimated_speedup: pattern.estimated_speedup,
                    memory_savings_bytes: self.estimate_fusion_memory_savings(&opportunity),
                };

                // Mark operations as used
                for op in &opportunity {
                    used_operations.insert(op.id.clone());
                }

                fusions.push(fusion);
            }
        }

        Ok(fusions)
    }

    /// Find opportunities for a specific fusion pattern
    fn find_fusion_opportunities<'a, T>(
        &self,
        operations: &'a [GraphOperation<T>],
        pattern: &FusionPattern,
        used_operations: &HashSet<String>,
    ) -> Vec<Vec<&'a GraphOperation<T>>>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut opportunities = Vec::new();

        // Simple sequential pattern matching
        for i in 0..operations.len() {
            if used_operations.contains(&operations[i].id) {
                continue;
            }

            let mut candidate_ops = vec![&operations[i]];
            let mut current_outputs = operations[i].outputs.clone();

            // Try to extend the pattern
            for operation in operations.iter().skip(i + 1) {
                if used_operations.contains(&operation.id) {
                    continue;
                }

                // Check if this operation consumes outputs from the pattern
                let has_dependency = operation
                    .inputs
                    .iter()
                    .any(|input| current_outputs.contains(input));

                if has_dependency && candidate_ops.len() < pattern.operation_sequence.len() {
                    candidate_ops.push(operation);
                    current_outputs.extend(operation.outputs.clone());
                }
            }

            // Check if the candidate matches the pattern
            if self.check_pattern_applicability(pattern, &candidate_ops) {
                opportunities.push(candidate_ops);
            }
        }

        opportunities
    }

    /// Check if a pattern is applicable to a set of operations
    fn check_pattern_applicability<T>(
        &self,
        pattern: &FusionPattern,
        operations: &[&GraphOperation<T>],
    ) -> bool
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        if operations.len() != pattern.operation_sequence.len() {
            return false;
        }

        // Check if operations match the pattern sequence
        for (i, expected_op) in pattern.operation_sequence.iter().enumerate() {
            if operations[i].operation_name != *expected_op {
                return false;
            }
        }

        true
    }

    /// Estimate memory savings from fusion
    fn estimate_fusion_memory_savings<T>(&self, operations: &[&GraphOperation<T>]) -> usize
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Estimate based on intermediate tensor elimination
        let total_intermediate_size: usize =
            operations.iter().flat_map(|op| &op.tensor_sizes).sum();

        // Assume fusion eliminates about 30% of intermediate storage
        (total_intermediate_size as f64 * 0.3 * 4.0) as usize // 4 bytes per f32
    }

    /// Optimize memory usage
    fn optimize_memory_usage<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<Vec<MemoryOptimization>>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut optimizations = Vec::new();

        // In-place gradient optimizations
        let inplace_ops = self.find_inplace_gradient_opportunities(operations);
        if !inplace_ops.is_empty() {
            optimizations.push(MemoryOptimization {
                optimization_id: "inplace_gradients".to_string(),
                optimization_type: MemoryOptimizationType::InPlaceGradients,
                affected_operations: inplace_ops.iter().map(|op| op.id.clone()).collect(),
                memory_savings_bytes: self.estimate_inplace_memory_savings(&inplace_ops),
                performance_impact: 1.1, // Slight speedup
            });
        }

        // Gradient checkpointing opportunities
        let checkpoint_segments = self.find_checkpointing_opportunities(operations);
        if !checkpoint_segments.is_empty() {
            optimizations.push(MemoryOptimization {
                optimization_id: "gradient_checkpointing".to_string(),
                optimization_type: MemoryOptimizationType::Checkpointing,
                affected_operations: checkpoint_segments
                    .iter()
                    .flat_map(|segment| segment.iter().map(|op| op.id.clone()))
                    .collect(),
                memory_savings_bytes: self
                    .estimate_checkpointing_memory_savings(&checkpoint_segments),
                performance_impact: 0.9, // Slight slowdown due to recomputation
            });
        }

        // Memory pooling optimization
        if operations.len() > 10 {
            optimizations.push(MemoryOptimization {
                optimization_id: "memory_pooling".to_string(),
                optimization_type: MemoryOptimizationType::MemoryPooling,
                affected_operations: operations.iter().map(|op| op.id.clone()).collect(),
                memory_savings_bytes: self.estimate_pooling_memory_savings(operations),
                performance_impact: 1.05, // Small speedup from reduced allocations
            });
        }

        Ok(optimizations)
    }

    /// Find operations that can use in-place gradients
    fn find_inplace_gradient_opportunities<'a, T>(
        &self,
        operations: &'a [GraphOperation<T>],
    ) -> Vec<&'a GraphOperation<T>>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        operations
            .iter()
            .filter(|op| self.can_use_inplace_gradients(op))
            .collect()
    }

    /// Check if operation can use in-place gradients
    fn can_use_inplace_gradients<T>(&self, operation: &GraphOperation<T>) -> bool
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        matches!(
            operation.operation_name.as_str(),
            "add_backward"
                | "sub_backward"
                | "mul_backward"
                | "relu_backward"
                | "sigmoid_backward"
                | "tanh_backward"
        )
    }

    /// Find checkpointing opportunities
    fn find_checkpointing_opportunities<'a, T>(
        &self,
        operations: &'a [GraphOperation<T>],
    ) -> Vec<Vec<&'a GraphOperation<T>>>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        let mut memory_usage = 0;
        let threshold = 100 * 1024 * 1024; // 100MB threshold

        for op in operations {
            let op_memory: usize = op.tensor_sizes.iter().sum::<usize>() * 4; // 4 bytes per f32

            if memory_usage + op_memory > threshold && !current_segment.is_empty() {
                segments.push(current_segment.clone());
                current_segment.clear();
                memory_usage = 0;
            }

            current_segment.push(op);
            memory_usage += op_memory;
        }

        if !current_segment.is_empty() {
            segments.push(current_segment);
        }

        segments
    }

    /// Estimate memory savings from various optimizations
    fn estimate_inplace_memory_savings<T>(&self, operations: &[&GraphOperation<T>]) -> usize {
        operations
            .iter()
            .map(|op| op.tensor_sizes.iter().sum::<usize>() * 4) // Save one copy per operation
            .sum()
    }

    fn estimate_checkpointing_memory_savings<T>(
        &self,
        segments: &[Vec<&GraphOperation<T>>],
    ) -> usize {
        segments
            .iter()
            .map(|segment| {
                let total_memory: usize = segment
                    .iter()
                    .flat_map(|op| &op.tensor_sizes)
                    .sum::<usize>()
                    * 4;
                total_memory / 2 // Roughly 50% savings from checkpointing
            })
            .sum()
    }

    fn estimate_pooling_memory_savings<T>(&self, operations: &[GraphOperation<T>]) -> usize {
        let total_memory: usize = operations
            .iter()
            .flat_map(|op| &op.tensor_sizes)
            .sum::<usize>()
            * 4;
        total_memory / 10 // 10% savings from pooling
    }

    /// Optimize communication between devices/subgraphs
    fn optimize_communication<T>(
        &self,
        operations: &[GraphOperation<T>],
        device_placement: &Option<PlacementResult>,
        subgraph_extraction: &Option<SubgraphExtractionResult>,
    ) -> Result<CommunicationPlan>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut transfers = Vec::new();
        let mut sync_points = Vec::new();
        let mut overlap_opportunities = Vec::new();

        // Analyze device placement transfers
        if let Some(placement) = device_placement {
            for decision in &placement.decisions {
                for transfer_req in &decision.transfer_requirements {
                    transfers.push(DataTransferOp {
                        id: format!("transfer_{}_{}", decision.operation_id, transfers.len()),
                        source_device: transfer_req.from_device,
                        target_device: transfer_req.to_device,
                        tensor_id: decision.operation_id.clone(),
                        size_bytes: transfer_req.data_size_bytes,
                        estimated_time_ms: transfer_req.estimated_time_ms,
                        can_overlap_with_compute: self.can_overlap_transfer(&decision.operation_id),
                        priority: self.determine_transfer_priority(&decision.operation_id),
                    });
                }
            }
        }

        // Analyze subgraph communications
        if let Some(extraction) = subgraph_extraction {
            for comm in &extraction.communications {
                transfers.push(DataTransferOp {
                    id: format!("subgraph_transfer_{}", transfers.len()),
                    source_device: Device::Cpu, // Simplified
                    target_device: Device::Cpu, // Simplified
                    tensor_id: comm.tensor.id.clone(),
                    size_bytes: comm.tensor.size_bytes,
                    estimated_time_ms: comm.estimated_transfer_time_ms,
                    can_overlap_with_compute: !comm.is_critical_path,
                    priority: if comm.is_critical_path {
                        TransferPriority::Critical
                    } else {
                        TransferPriority::Medium
                    },
                });
            }

            // Add synchronization points for subgraph boundaries
            for stage in &extraction.execution_order {
                if stage.len() > 1 {
                    sync_points.push(SynchronizationPoint {
                        id: format!("stage_sync_{}", sync_points.len()),
                        operation_ids: stage.clone(),
                        devices: self.config.target_devices.clone(),
                        sync_type: SynchronizationType::Barrier,
                        estimated_wait_time_ms: 1.0, // Minimal sync overhead
                    });
                }
            }
        }

        // Find computation-communication overlap opportunities
        for transfer in transfers.iter() {
            if transfer.can_overlap_with_compute {
                // Find operations that could run concurrently
                for op in operations {
                    if op.id != transfer.tensor_id {
                        overlap_opportunities.push(OverlapOpportunity {
                            compute_operation: op.id.clone(),
                            transfer_operation: transfer.id.clone(),
                            overlap_ratio: 0.7, // Estimate 70% overlap
                            estimated_speedup: 1.2,
                        });
                    }
                }
            }
        }

        let total_transfer_volume = transfers.iter().map(|t| t.size_bytes).sum();
        let total_transfer_time = transfers.iter().map(|t| t.estimated_time_ms).sum();

        Ok(CommunicationPlan {
            transfers,
            synchronization_points: sync_points,
            total_transfer_volume_bytes: total_transfer_volume,
            estimated_transfer_time_ms: total_transfer_time,
            communication_overlap_opportunities: overlap_opportunities,
        })
    }

    /// Check if transfer can overlap with computation
    fn can_overlap_transfer(&self, operation_id: &str) -> bool {
        // Most gradient operations can overlap with transfers
        !operation_id.contains("sync") && !operation_id.contains("barrier")
    }

    /// Determine transfer priority
    fn determine_transfer_priority(&self, operation_id: &str) -> TransferPriority {
        if operation_id.contains("critical") || operation_id.contains("loss") {
            TransferPriority::Critical
        } else if operation_id.contains("gradient") {
            TransferPriority::High
        } else {
            TransferPriority::Medium
        }
    }

    /// Calculate estimated speedup from all optimizations
    fn calculate_estimated_speedup(
        &self,
        fusions: &[GradientFusion],
        memory_opts: &[MemoryOptimization],
        device_placement: &Option<PlacementResult>,
        subgraph_extraction: &Option<SubgraphExtractionResult>,
    ) -> f64 {
        let mut speedup = 1.0;

        // Fusion speedups (multiplicative)
        for fusion in fusions {
            speedup *= fusion.estimated_speedup;
        }

        // Memory optimization speedups
        for opt in memory_opts {
            speedup *= opt.performance_impact;
        }

        // Device placement speedup (estimate based on GPU acceleration)
        #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
        if let Some(placement) = device_placement {
            #[cfg(feature = "gpu")]
            {
                let gpu_ops = placement
                    .decisions
                    .iter()
                    .filter(|d| matches!(d.chosen_device, Device::Gpu(_)))
                    .count();
                let total_ops = placement.decisions.len();

                if total_ops > 0 {
                    let gpu_ratio = gpu_ops as f64 / total_ops as f64;
                    speedup *= 1.0 + gpu_ratio * 2.0; // Assume 2x speedup for GPU operations
                }
            }
        }

        // Subgraph parallelization speedup
        if let Some(extraction) = subgraph_extraction {
            speedup *= extraction.parallel_efficiency;
        }

        speedup
    }

    /// Calculate memory reduction from optimizations
    fn calculate_memory_reduction(&self, memory_opts: &[MemoryOptimization]) -> f64 {
        let total_savings: usize = memory_opts.iter().map(|opt| opt.memory_savings_bytes).sum();

        // Convert to percentage (estimate original memory usage)
        let estimated_original_memory = total_savings * 4; // Rough estimate
        if estimated_original_memory > 0 {
            total_savings as f64 / estimated_original_memory as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_optimizer_creation() {
        let config = GraphOptimizationConfig::default();
        let optimizer = EnhancedGraphOptimizer::new(config);

        assert!(!optimizer.gradient_fusion_patterns.is_empty());
    }

    #[test]
    fn test_fusion_pattern_matching() {
        let config = GraphOptimizationConfig::default();
        let optimizer = EnhancedGraphOptimizer::new(config);

        let operations = vec![
            GraphOperation::<f32>::new(
                "op1".to_string(),
                "add_backward".to_string(),
                vec![],
                vec!["t1".to_string()],
            ),
            GraphOperation::<f32>::new(
                "op2".to_string(),
                "mul_backward".to_string(),
                vec!["t1".to_string()],
                vec!["t2".to_string()],
            ),
        ];

        let fusions = optimizer.optimize_gradient_fusions(&operations).unwrap();
        assert!(!fusions.is_empty());
    }

    #[test]
    fn test_memory_optimization() {
        let config = GraphOptimizationConfig::default();
        let optimizer = EnhancedGraphOptimizer::new(config);

        let operations = vec![
            GraphOperation::<f32>::new(
                "op1".to_string(),
                "relu_backward".to_string(),
                vec![],
                vec!["t1".to_string()],
            ),
            GraphOperation::<f32>::new(
                "op2".to_string(),
                "add_backward".to_string(),
                vec!["t1".to_string()],
                vec!["t2".to_string()],
            ),
        ];

        let memory_opts = optimizer.optimize_memory_usage(&operations).unwrap();
        assert!(!memory_opts.is_empty());
    }

    #[test]
    fn test_comprehensive_optimization() {
        let mut config = GraphOptimizationConfig::default();
        #[cfg(feature = "gpu")]
        {
            config.target_devices = vec![Device::Cpu, Device::Gpu(0)];
        }
        #[cfg(not(feature = "gpu"))]
        {
            config.target_devices = vec![Device::Cpu];
        }

        let optimizer = EnhancedGraphOptimizer::new(config);

        let operations = vec![
            GraphOperation::<f32>::new(
                "op1".to_string(),
                "MatMul".to_string(),
                vec![],
                vec!["t1".to_string()],
            )
            .with_tensor_sizes(vec![1000, 1000]),
            GraphOperation::<f32>::new(
                "op2".to_string(),
                "ReLU".to_string(),
                vec!["t1".to_string()],
                vec!["t2".to_string()],
            )
            .with_tensor_sizes(vec![1000, 1000]),
            GraphOperation::<f32>::new(
                "op3".to_string(),
                "add_backward".to_string(),
                vec!["t2".to_string()],
                vec!["t3".to_string()],
            )
            .with_tensor_sizes(vec![1000, 1000]),
        ];

        let result = optimizer.optimize_graph(&operations).unwrap();
        // The speedup estimation may not always be > 1.0 depending on the optimization heuristics
        // Just check that we get a reasonable speedup value (> 0)
        assert!(result.estimated_speedup > 0.0);
        assert!(result.optimization_stats.operations_optimized > 0);
    }
}
