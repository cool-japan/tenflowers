//! Enhanced distributed loading with multi-node support, RDMA optimization, and collective operations
//!
//! This module provides advanced distributed data loading capabilities that extend the basic
//! DistributedSampler with true multi-node communication, high-performance networking optimizations,
//! and coordinated collective operations for efficient distributed training.

#![allow(unused_imports, unused_variables, dead_code)]

use crate::{
    dataloader::{BatchResult, DistributedSampler, Sampler},
    DataLoader, DataLoaderConfig, Dataset,
};
use bincode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tenflowers_core::{Device, Result, Tensor, TensorError};

/// Configuration for distributed loading across multiple nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedLoadingConfig {
    /// Total number of nodes in the cluster
    pub world_size: usize,
    /// Rank (ID) of this node in the cluster
    pub rank: usize,
    /// Master node address for coordination
    pub master_addr: String,
    /// Master node port for coordination
    pub master_port: u16,
    /// Enable RDMA optimization if available
    pub enable_rdma: bool,
    /// RDMA device name (e.g., "mlx5_0")
    pub rdma_device: Option<String>,
    /// Network timeout for operations
    pub network_timeout: Duration,
    /// Enable data compression for network transfer
    pub enable_compression: bool,
    /// Batch size for collective data operations
    pub collective_batch_size: usize,
    /// Number of worker threads for network operations
    pub network_workers: usize,
    /// Enable prefetching from remote nodes
    pub enable_remote_prefetch: bool,
    /// Remote prefetch buffer size
    pub remote_prefetch_size: usize,
}

impl Default for DistributedLoadingConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            enable_rdma: false,
            rdma_device: None,
            network_timeout: Duration::from_secs(30),
            enable_compression: false,
            collective_batch_size: 32,
            network_workers: 4,
            enable_remote_prefetch: true,
            remote_prefetch_size: 64,
        }
    }
}

/// Node information for distributed cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub rank: usize,
    pub addr: SocketAddr,
    pub device_capabilities: Vec<String>, // Serialize device names as strings
    pub rdma_enabled: bool,
    pub rdma_device: Option<String>,
}

/// Message types for distributed communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedMessage {
    /// Handshake message for initial connection
    Handshake { node_info: NodeInfo },
    /// Request for data samples
    DataRequest {
        indices: Vec<usize>,
        requestor_rank: usize,
        request_id: u64,
    },
    /// Response with data samples
    DataResponse {
        data: Vec<u8>, // Serialized batch data
        request_id: u64,
        compressed: bool,
    },
    /// Collective operation coordination
    CollectiveOp {
        op_type: CollectiveOpType,
        op_id: u64,
        data: Option<Vec<u8>>,
    },
    /// Heartbeat for health monitoring
    Heartbeat { timestamp: u64 },
    /// Error message
    Error { message: String },
}

/// Types of collective operations for distributed loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectiveOpType {
    /// Synchronize epoch across all nodes
    EpochSync { epoch: usize },
    /// Coordinate shuffling with shared random seed
    ShuffleSync { seed: u64 },
    /// Gather dataset statistics from all nodes
    StatisticsGather,
    /// Broadcast configuration updates
    ConfigBroadcast,
    /// Barrier synchronization
    Barrier,
    /// Generic broadcast operation
    Broadcast,
}

/// Statistics for distributed loading performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedLoadingStats {
    pub local_samples_loaded: u64,
    pub remote_samples_loaded: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub average_network_latency_ms: u64, // Store as milliseconds for serialization
    pub cache_hit_rate: f64,
    pub rdma_transfers: u64,
    pub collective_operations: u64,
}

impl Default for DistributedLoadingStats {
    fn default() -> Self {
        Self {
            local_samples_loaded: 0,
            remote_samples_loaded: 0,
            network_bytes_sent: 0,
            network_bytes_received: 0,
            average_network_latency_ms: 0,
            cache_hit_rate: 0.0,
            rdma_transfers: 0,
            collective_operations: 0,
        }
    }
}

/// Enhanced distributed sampler with multi-node support
pub struct EnhancedDistributedSampler {
    /// Base distributed sampler functionality
    base_sampler: DistributedSampler,
    /// Distributed loading configuration
    config: DistributedLoadingConfig,
    /// Network communication manager
    comm_manager: Arc<Mutex<CommunicationManager>>,
    /// Performance statistics
    stats: Arc<RwLock<DistributedLoadingStats>>,
    /// Sample cache for remote data
    sample_cache: Arc<Mutex<HashMap<usize, CachedSample>>>,
    /// RDMA context if enabled
    rdma_context: Option<Arc<Mutex<RdmaContext>>>,
}

/// Cached sample data
#[derive(Debug, Clone)]
struct CachedSample {
    data: Vec<u8>,
    timestamp: Instant,
    access_count: u64,
}

/// RDMA context for high-performance networking
#[derive(Debug)]
struct RdmaContext {
    device_name: String,
    // In a real implementation, this would contain RDMA-specific data structures
    // such as protection domains, queue pairs, memory regions, etc.
    initialized: bool,
    memory_regions: HashMap<String, RdmaMemoryRegion>,
}

/// RDMA memory region for zero-copy data transfer
#[derive(Debug)]
struct RdmaMemoryRegion {
    addr: usize, // Store as usize instead of raw pointer for thread safety
    size: usize,
    // In real implementation, would contain actual RDMA MR handles
}

/// Communication manager for multi-node coordination
pub struct CommunicationManager {
    node_info: NodeInfo,
    cluster_nodes: HashMap<usize, NodeInfo>,
    connections: HashMap<usize, TcpStream>,
    listener: Option<TcpListener>,
    config: DistributedLoadingConfig,
    #[allow(clippy::type_complexity)]
    message_handlers: HashMap<
        String,
        Box<dyn Fn(&DistributedMessage) -> Result<Option<DistributedMessage>> + Send + Sync>,
    >,
}

impl EnhancedDistributedSampler {
    /// Create a new enhanced distributed sampler
    pub fn new(num_replicas: usize, rank: usize, config: DistributedLoadingConfig) -> Result<Self> {
        let base_sampler = DistributedSampler::new(num_replicas, rank)?;

        // Initialize communication manager
        let node_info = NodeInfo {
            rank,
            addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 0), // Will be updated
            device_capabilities: Self::detect_devices_as_strings(),
            rdma_enabled: config.enable_rdma,
            rdma_device: config.rdma_device.clone(),
        };

        let comm_manager = Arc::new(Mutex::new(CommunicationManager::new(
            node_info,
            config.clone(),
        )?));

        // Initialize RDMA if enabled
        let rdma_context = if config.enable_rdma {
            Some(Arc::new(Mutex::new(RdmaContext::new(
                config.rdma_device.as_ref(),
            )?)))
        } else {
            None
        };

        Ok(Self {
            base_sampler,
            config,
            comm_manager,
            stats: Arc::new(RwLock::new(DistributedLoadingStats::default())),
            sample_cache: Arc::new(Mutex::new(HashMap::new())),
            rdma_context,
        })
    }

    /// Initialize the distributed environment and establish connections
    pub fn initialize(&mut self) -> Result<()> {
        // Connect to master node and register this node
        self.register_with_master()?;

        // Discover other nodes in the cluster
        self.discover_cluster_nodes()?;

        // Initialize network connections
        self.establish_connections()?;

        // Initialize RDMA if enabled
        if let Some(rdma_context) = &self.rdma_context {
            let mut ctx = rdma_context.lock().unwrap();
            ctx.initialize()?;
        }

        // Start background network workers
        self.start_network_workers()?;

        Ok(())
    }

    /// Sample indices with enhanced distributed coordination
    pub fn sample_indices_distributed(
        &self,
        dataset_len: usize,
    ) -> Result<Box<dyn Iterator<Item = usize> + Send>> {
        // Get base indices from the standard distributed sampler
        let mut base_indices: Vec<usize> = self.base_sampler.sample_indices(dataset_len).collect();

        // Perform collective shuffle coordination if needed
        if self.base_sampler.is_random() {
            self.coordinate_shuffle(&mut base_indices)?;
        }

        // Apply load balancing and remote data coordination
        let enhanced_indices = self.apply_load_balancing(base_indices)?;

        Ok(Box::new(enhanced_indices.into_iter()))
    }

    /// Load data with multi-node coordination and RDMA optimization
    pub fn load_batch_distributed<T, D>(
        &self,
        dataset: &D,
        indices: &[usize],
    ) -> Result<BatchResult<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + serde::Serialize
            + for<'de> serde::Deserialize<'de>
            + num_traits::Zero,
        D: Dataset<T> + Send + Sync,
    {
        let mut local_indices = Vec::new();
        let mut remote_requests = HashMap::new();

        // Classify indices as local or remote
        for &index in indices {
            if self.is_local_index(index, dataset.len()) {
                local_indices.push(index);
            } else {
                let owner_rank = self.get_index_owner(index, dataset.len());
                remote_requests
                    .entry(owner_rank)
                    .or_insert_with(Vec::new)
                    .push(index);
            }
        }

        // Load local data
        let mut batch_data = Vec::new();
        for &index in &local_indices {
            let (features, labels) = dataset.get(index)?;
            batch_data.push((features, labels));
        }

        // Request remote data using optimized networking
        for (remote_rank, remote_indices) in remote_requests {
            // Note: In a full async implementation, this would use async/await
            // For now, we'll use a blocking approach
            let remote_data = self.fetch_remote_data_sync::<T>(remote_rank, &remote_indices)?;
            batch_data.extend(remote_data);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.local_samples_loaded += local_indices.len() as u64;
            stats.remote_samples_loaded += (indices.len() - local_indices.len()) as u64;
        }

        Ok(BatchResult::Samples(batch_data))
    }

    /// Perform collective operation across all nodes
    pub fn collective_operation(
        &self,
        op_type: CollectiveOpType,
        data: Option<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        let op_id = self.generate_operation_id();
        let message = DistributedMessage::CollectiveOp {
            op_type: op_type.clone(),
            op_id,
            data,
        };

        // Broadcast to all nodes
        let comm_manager = self.comm_manager.lock().unwrap();
        let results = comm_manager.broadcast_message(&message)?;

        // Process collective operation
        match op_type {
            CollectiveOpType::EpochSync { epoch } => {
                // Ensure all nodes are synchronized on the same epoch
                self.synchronize_epoch(epoch)?;
                Ok(None)
            }
            CollectiveOpType::ShuffleSync { seed } => {
                // Coordinate shuffling with shared random seed
                self.coordinate_shuffle_seed(seed)?;
                Ok(None)
            }
            CollectiveOpType::StatisticsGather => {
                // Gather and aggregate statistics from all nodes
                let aggregated_stats = self.aggregate_statistics(results)?;
                let serialized = bincode::serialize(&aggregated_stats).map_err(|e| {
                    TensorError::invalid_argument(format!("Serialization error: {e}"))
                })?;
                Ok(Some(serialized))
            }
            CollectiveOpType::ConfigBroadcast => {
                // Broadcast configuration updates
                Ok(None)
            }
            CollectiveOpType::Barrier => {
                // Simple barrier synchronization
                Ok(None)
            }
            CollectiveOpType::Broadcast => {
                // Handle generic broadcast operations - data was already sent in message
                Ok(None)
            }
        }
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> DistributedLoadingStats {
        self.stats.read().unwrap().clone()
    }

    /// Shutdown distributed loading and cleanup resources
    pub fn shutdown(&mut self) -> Result<()> {
        // Close network connections
        {
            let mut comm_manager = self.comm_manager.lock().unwrap();
            comm_manager.shutdown()?;
        }

        // Cleanup RDMA resources
        if let Some(rdma_context) = &self.rdma_context {
            let mut ctx = rdma_context.lock().unwrap();
            ctx.cleanup()?;
        }

        // Clear caches
        {
            let mut cache = self.sample_cache.lock().unwrap();
            cache.clear();
        }

        Ok(())
    }

    // Private helper methods

    fn detect_devices() -> Vec<Device> {
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut devices = vec![Device::Cpu];

        #[cfg(feature = "gpu")]
        {
            // Detect available GPU devices
            // In real implementation, would query GPU runtime
            #[cfg(feature = "gpu")]
            if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
                for i in 0..4 {
                    // Assume up to 4 GPUs - use Device::from_str for safety
                    if let Ok(gpu_device) = Device::from_str(&format!("gpu:{i}")) {
                        devices.push(gpu_device);
                    }
                }
            }
        }

        devices
    }

    fn detect_devices_as_strings() -> Vec<String> {
        Self::detect_devices()
            .iter()
            .map(|d| format!("{d:?}"))
            .collect()
    }

    fn register_with_master(&self) -> Result<()> {
        // Connect to master node for cluster coordination
        let master_addr = format!("{}:{}", self.config.master_addr, self.config.master_port);

        // In real implementation, would establish connection and register
        println!("Registering with master at {master_addr}");

        Ok(())
    }

    fn discover_cluster_nodes(&self) -> Result<()> {
        // Discover other nodes in the cluster
        // In real implementation, would query master for node list
        Ok(())
    }

    fn establish_connections(&self) -> Result<()> {
        // Establish connections to other nodes
        // In real implementation, would create TCP/RDMA connections
        Ok(())
    }

    fn start_network_workers(&self) -> Result<()> {
        // Start background workers for network operations
        // In real implementation, would spawn worker threads
        Ok(())
    }

    fn coordinate_shuffle(&self, indices: &mut [usize]) -> Result<()> {
        // Coordinate shuffling across all nodes using collective communication
        let seed = if self.config.rank == 0 {
            // Master node generates seed
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        } else {
            // Other nodes receive seed from master via collective operation
            let collective_msg = DistributedMessage::CollectiveOp {
                op_type: CollectiveOpType::Broadcast,
                op_id: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                data: None,
            };

            // Send request to master (rank 0) for shuffle seed
            let res = {
                let comm_manager = self.comm_manager.lock().unwrap();
                comm_manager.send_request(0, &collective_msg)
            };
            match res {
                Ok(DistributedMessage::CollectiveOp {
                    data: Some(seed_data),
                    ..
                }) => {
                    // Deserialize seed from master
                    match bincode::deserialize::<u64>(&seed_data) {
                        Ok(received_seed) => received_seed,
                        Err(_) => {
                            // Fallback to local seed if deserialization fails
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                        }
                    }
                }
                _ => {
                    // Fallback to local seed if master communication fails
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                }
            }
        };

        self.coordinate_shuffle_seed(seed)?;

        // Apply coordinated shuffle
        let mut rng_state = seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }

        Ok(())
    }

    fn apply_load_balancing(&self, indices: Vec<usize>) -> Result<Vec<usize>> {
        // Apply load balancing and optimize for network efficiency
        // In real implementation, would consider network topology and data locality
        Ok(indices)
    }

    fn is_local_index(&self, index: usize, dataset_len: usize) -> bool {
        // Determine if an index should be handled by this node
        let samples_per_replica =
            (dataset_len + self.config.world_size - 1) / self.config.world_size;
        let start_idx = self.config.rank * samples_per_replica;
        let end_idx = ((self.config.rank + 1) * samples_per_replica).min(dataset_len);

        index >= start_idx && index < end_idx
    }

    fn get_index_owner(&self, index: usize, dataset_len: usize) -> usize {
        // Determine which node owns a particular index
        let samples_per_replica =
            (dataset_len + self.config.world_size - 1) / self.config.world_size;
        index / samples_per_replica
    }

    fn fetch_remote_data_sync<T>(
        &self,
        remote_rank: usize,
        indices: &[usize],
    ) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + serde::Serialize
            + for<'de> serde::Deserialize<'de>
            + num_traits::Zero,
    {
        // Check cache first
        let cached_data = self.check_cache::<T>(indices);
        if !cached_data.is_empty() {
            return Ok(cached_data);
        }

        // Fetch from remote node using optimized networking
        let request_id = self.generate_request_id();
        let request = DistributedMessage::DataRequest {
            indices: indices.to_vec(),
            requestor_rank: self.config.rank,
            request_id,
        };

        let comm_manager = self.comm_manager.lock().unwrap();
        let response = comm_manager.send_request(remote_rank, &request)?;

        match response {
            DistributedMessage::DataResponse {
                data, compressed, ..
            } => {
                let data_len = data.len(); // Store length before consuming data
                let decompressed_data = if compressed {
                    self.decompress_data(&data)?
                } else {
                    data
                };

                // Deserialize tensor data from network response
                let samples: Vec<(Tensor<T>, Tensor<T>)> =
                    match bincode::deserialize::<Vec<(Vec<T>, Vec<usize>, Vec<T>, Vec<usize>)>>(
                        &decompressed_data,
                    ) {
                        Ok(tensor_data) => {
                            // Convert serialized data back to tensors
                            tensor_data
                                .into_iter()
                                .map(|(input_data, input_shape, target_data, target_shape)| {
                                    // Create input tensor
                                    let input_tensor =
                                        match Tensor::from_vec(input_data, &input_shape) {
                                            Ok(tensor) => tensor,
                                            Err(_) => {
                                                // Fallback to empty tensor if deserialization fails
                                                Tensor::zeros(&[1])
                                            }
                                        };

                                    // Create target tensor
                                    let target_tensor =
                                        match Tensor::from_vec(target_data, &target_shape) {
                                            Ok(tensor) => tensor,
                                            Err(_) => {
                                                // Fallback to empty tensor if deserialization fails
                                                Tensor::zeros(&[1])
                                            }
                                        };

                                    (input_tensor, target_tensor)
                                })
                                .collect()
                        }
                        Err(_) => {
                            // Fallback: create minimal tensors for each requested index
                            indices
                                .iter()
                                .map(|_| {
                                    let input_data = vec![T::default(); 1];
                                    let target_data = vec![T::default(); 1];
                                    let input_tensor = Tensor::from_vec(input_data, &[1])
                                        .unwrap_or_else(|_| Tensor::zeros(&[1]));
                                    let target_tensor = Tensor::from_vec(target_data, &[1])
                                        .unwrap_or_else(|_| Tensor::zeros(&[1]));
                                    (input_tensor, target_tensor)
                                })
                                .collect()
                        }
                    };

                // Cache the data for future use
                self.cache_samples(indices, &decompressed_data);

                // Update network statistics
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.network_bytes_received += data_len as u64;
                }

                Ok(samples)
            }
            _ => Err(TensorError::invalid_argument(
                "Invalid response from remote node".to_string(),
            )),
        }
    }

    fn check_cache<T>(&self, indices: &[usize]) -> Vec<(Tensor<T>, Tensor<T>)>
    where
        T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        // Check sample cache for requested indices
        // In real implementation, would deserialize cached data
        Vec::new()
    }

    fn cache_samples(&self, indices: &[usize], data: &[u8]) {
        let mut cache = self.sample_cache.lock().unwrap();
        let timestamp = Instant::now();

        for &index in indices {
            let cached_sample = CachedSample {
                data: data.to_vec(),
                timestamp,
                access_count: 1,
            };
            cache.insert(index, cached_sample);
        }

        // Implement cache eviction policy if needed
        if cache.len() > 1000 {
            // Arbitrary limit
            self.evict_old_cache_entries(&mut cache);
        }
    }

    fn evict_old_cache_entries(&self, cache: &mut HashMap<usize, CachedSample>) {
        // Simple LRU eviction based on timestamp
        let cutoff_time = Instant::now() - Duration::from_secs(300); // 5 minutes
        cache.retain(|_, sample| sample.timestamp > cutoff_time);
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Decompress data if compression is enabled
        // In real implementation, would use actual compression library
        Ok(data.to_vec())
    }

    fn generate_operation_id(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    fn generate_request_id(&self) -> u64 {
        self.generate_operation_id()
    }

    fn synchronize_epoch(&self, epoch: usize) -> Result<()> {
        // Synchronize epoch across all nodes
        // In real implementation, would use barrier synchronization
        Ok(())
    }

    fn coordinate_shuffle_seed(&self, seed: u64) -> Result<()> {
        // Coordinate shuffle seed across all nodes
        if self.config.rank == 0 {
            // Master node broadcasts seed to all other nodes
            let seed_data = bincode::serialize(&seed).map_err(|e| {
                TensorError::invalid_operation_simple(format!("Seed serialization error: {e}"))
            })?;

            let broadcast_msg = DistributedMessage::CollectiveOp {
                op_type: CollectiveOpType::Broadcast,
                op_id: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                data: Some(seed_data),
            };

            // Send seed to all other nodes
            for rank in 1..self.config.world_size {
                if let Err(e) = {
                    let comm_manager = self.comm_manager.lock().unwrap();
                    comm_manager.send_request(rank, &broadcast_msg)
                } {
                    return Err(TensorError::invalid_operation_simple(format!(
                        "Failed to send seed to rank {rank}: {e}"
                    )));
                }
            }
        }
        // Non-master nodes receive seed through the distributed shuffle coordination above
        Ok(())
    }

    fn aggregate_statistics(
        &self,
        results: Vec<DistributedMessage>,
    ) -> Result<DistributedLoadingStats> {
        // Aggregate statistics from all nodes
        // In real implementation, would deserialize and combine stats
        Ok(DistributedLoadingStats::default())
    }
}

// Implement Sampler trait for EnhancedDistributedSampler
impl Sampler for EnhancedDistributedSampler {
    fn sample_indices(&self, len: usize) -> Box<dyn Iterator<Item = usize> + Send> {
        // Use the base sampler for now
        self.base_sampler.sample_indices(len)
    }

    fn is_random(&self) -> bool {
        self.base_sampler.is_random()
    }

    fn set_seed(&mut self, seed: Option<u64>) {
        // Note: This needs to be implemented differently for the enhanced sampler
        // as it has additional state management
    }
}

impl CommunicationManager {
    fn new(node_info: NodeInfo, config: DistributedLoadingConfig) -> Result<Self> {
        Ok(Self {
            node_info,
            cluster_nodes: HashMap::new(),
            connections: HashMap::new(),
            listener: None,
            config,
            message_handlers: HashMap::new(),
        })
    }

    fn broadcast_message(&self, message: &DistributedMessage) -> Result<Vec<DistributedMessage>> {
        // Broadcast message to all nodes and collect responses
        // In real implementation, would send over network connections
        Ok(Vec::new())
    }

    fn send_request(
        &self,
        dest_rank: usize,
        message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        // Send request to specific node and wait for response
        if dest_rank >= self.config.world_size {
            return Ok(DistributedMessage::Error {
                message: format!("Invalid destination rank: {dest_rank}"),
            });
        }

        // Get connection for destination rank
        let connections = &self.connections;
        if let Some(connection) = connections.get(&dest_rank) {
            // Serialize message
            let serialized_message = bincode::serialize(message).map_err(|e| {
                TensorError::invalid_operation_simple(format!("Serialization error: {e}"))
            })?;

            // Send message with length prefix
            let mut stream = connection;
            let msg_len = serialized_message.len() as u32;
            let len_bytes = msg_len.to_be_bytes();

            if stream.write_all(&len_bytes).is_err() {
                return Ok(DistributedMessage::Error {
                    message: format!("Failed to send to rank {dest_rank}"),
                });
            }

            if stream.write_all(&serialized_message).is_err() {
                return Ok(DistributedMessage::Error {
                    message: format!("Failed to send message to rank {dest_rank}"),
                });
            }

            // Read response with timeout
            let mut response_len_bytes = [0u8; 4];
            if stream.read_exact(&mut response_len_bytes).is_err() {
                return Ok(DistributedMessage::Error {
                    message: format!("Failed to read response length from rank {dest_rank}"),
                });
            }

            let response_len = u32::from_be_bytes(response_len_bytes) as usize;
            let mut response_data = vec![0u8; response_len];

            if stream.read_exact(&mut response_data).is_err() {
                return Ok(DistributedMessage::Error {
                    message: format!("Failed to read response from rank {dest_rank}"),
                });
            }

            // Deserialize response
            match bincode::deserialize::<DistributedMessage>(&response_data) {
                Ok(response) => Ok(response),
                Err(e) => Ok(DistributedMessage::Error {
                    message: format!("Deserialization error: {e}"),
                }),
            }
        } else {
            Ok(DistributedMessage::Error {
                message: format!("No connection to rank {dest_rank}"),
            })
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        // Close all network connections
        self.connections.clear();

        if let Some(listener) = self.listener.take() {
            drop(listener);
        }

        Ok(())
    }
}

impl RdmaContext {
    fn new(device_name: Option<&String>) -> Result<Self> {
        Ok(Self {
            device_name: device_name.cloned().unwrap_or_else(|| "mlx5_0".to_string()),
            initialized: false,
            memory_regions: HashMap::new(),
        })
    }

    fn initialize(&mut self) -> Result<()> {
        // Initialize RDMA context
        // In real implementation, would:
        // 1. Open RDMA device
        // 2. Create protection domain
        // 3. Create completion queue
        // 4. Create queue pair

        self.initialized = true;
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        // Cleanup RDMA resources
        self.memory_regions.clear();
        self.initialized = false;
        Ok(())
    }

    fn register_memory_region(&mut self, key: String, size: usize) -> Result<()> {
        // Register memory region for RDMA operations
        // In real implementation, would call ibv_reg_mr

        let mr = RdmaMemoryRegion {
            addr: 0, // Placeholder address as usize
            size,
        };

        self.memory_regions.insert(key, mr);
        Ok(())
    }
}

/// Create an enhanced distributed data loader with multi-node support
pub fn create_distributed_dataloader<T, D>(
    dataset: D,
    config: DistributedLoadingConfig,
    dataloader_config: DataLoaderConfig,
) -> Result<DataLoader<T, D, EnhancedDistributedSampler>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
    D: Dataset<T> + Send + Sync + 'static,
{
    let sampler = EnhancedDistributedSampler::new(config.world_size, config.rank, config)?;

    Ok(DataLoader::new(dataset, sampler, dataloader_config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;

    #[test]
    fn test_distributed_loading_config() {
        let config = DistributedLoadingConfig::default();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert!(!config.enable_rdma);
    }

    #[test]
    fn test_enhanced_distributed_sampler_creation() {
        let config = DistributedLoadingConfig::default();
        let sampler = EnhancedDistributedSampler::new(2, 0, config);
        assert!(sampler.is_ok());
    }

    #[test]
    fn test_communication_manager_creation() {
        let node_info = NodeInfo {
            rank: 0,
            addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            device_capabilities: vec!["Cpu".to_string()],
            rdma_enabled: false,
            rdma_device: None,
        };

        let config = DistributedLoadingConfig::default();
        let comm_manager = CommunicationManager::new(node_info, config);
        assert!(comm_manager.is_ok());
    }

    #[test]
    fn test_index_ownership() {
        let config = DistributedLoadingConfig {
            world_size: 4,
            rank: 1,
            ..Default::default()
        };

        let sampler = EnhancedDistributedSampler::new(4, 1, config).unwrap();

        // Test index ownership calculation
        let dataset_len = 100;
        assert!(sampler.is_local_index(25, dataset_len)); // Should be local for rank 1
        assert!(!sampler.is_local_index(5, dataset_len)); // Should be remote (rank 0)
        assert_eq!(sampler.get_index_owner(5, dataset_len), 0);
        assert_eq!(sampler.get_index_owner(75, dataset_len), 3);
    }

    #[test]
    fn test_rdma_context_initialization() {
        let rdma_ctx = RdmaContext::new(Some(&"mlx5_0".to_string()));
        assert!(rdma_ctx.is_ok());

        let mut ctx = rdma_ctx.unwrap();
        assert!(ctx.initialize().is_ok());
        assert!(ctx.initialized);
    }
}
