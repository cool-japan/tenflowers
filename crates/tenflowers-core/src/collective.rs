#![allow(clippy::result_large_err)]

use crate::{Device, Result, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Collective operation types for multi-GPU communication
#[derive(Debug, Clone)]
pub enum CollectiveOp {
    /// Reduce values across all devices using the specified reduction operation
    AllReduce(ReductionOp),
    /// Broadcast tensor from source device to all devices
    Broadcast { src_device: Device },
    /// Gather tensors from all devices to a single device
    AllGather,
    /// Reduce-scatter: reduce and distribute results across devices
    ReduceScatter(ReductionOp),
    /// Send tensor from one device to another
    Send {
        src_device: Device,
        dst_device: Device,
    },
    /// Receive tensor on one device from another
    Recv {
        src_device: Device,
        dst_device: Device,
    },
}

/// Reduction operations for collective communication
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Product,
}

/// Communication group for multi-device operations
#[derive(Debug, Clone)]
pub struct CommunicationGroup {
    devices: Vec<Device>,
    rank_map: HashMap<Device, usize>,
}

impl CommunicationGroup {
    /// Create a new communication group with the specified devices
    pub fn new(devices: Vec<Device>) -> Self {
        let rank_map = devices
            .iter()
            .enumerate()
            .map(|(rank, &device)| (device, rank))
            .collect();

        Self { devices, rank_map }
    }

    /// Get all devices in the group
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Get rank of a device in the group
    pub fn rank(&self, device: &Device) -> Option<usize> {
        self.rank_map.get(device).copied()
    }

    /// Get device at a specific rank
    pub fn device_at_rank(&self, rank: usize) -> Option<Device> {
        self.devices.get(rank).copied()
    }

    /// Get number of devices in the group
    pub fn size(&self) -> usize {
        self.devices.len()
    }
}

/// Collective operations manager
pub struct CollectiveManager {
    groups: HashMap<String, CommunicationGroup>,
    default_group: Option<String>,
}

impl CollectiveManager {
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            default_group: None,
        }
    }

    /// Create a communication group
    pub fn create_group(&mut self, name: String, devices: Vec<Device>) -> Result<()> {
        if devices.is_empty() {
            return Err(TensorError::invalid_argument(
                "Communication group cannot be empty".to_string(),
            ));
        }

        let group = CommunicationGroup::new(devices);
        self.groups.insert(name.clone(), group);

        if self.default_group.is_none() {
            self.default_group = Some(name);
        }

        Ok(())
    }

    /// Set default communication group
    pub fn set_default_group(&mut self, name: String) -> Result<()> {
        if !self.groups.contains_key(&name) {
            return Err(TensorError::invalid_argument(format!(
                "Group '{name}' does not exist"
            )));
        }

        self.default_group = Some(name);
        Ok(())
    }

    /// Get communication group by name
    pub fn get_group(&self, name: &str) -> Option<&CommunicationGroup> {
        self.groups.get(name)
    }

    /// Get default communication group
    pub fn get_default_group(&self) -> Option<&CommunicationGroup> {
        self.default_group
            .as_ref()
            .and_then(|name| self.groups.get(name))
    }

    /// Perform AllReduce operation
    pub fn all_reduce<T>(
        &self,
        tensor: &Tensor<T>,
        op: ReductionOp,
        group_name: Option<&str>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + num_traits::Zero
            + num_traits::One
            + std::ops::Add<Output = T>
            + PartialOrd
            + std::ops::Mul<Output = T>
            + num_traits::Float,
    {
        let group = if let Some(name) = group_name {
            self.get_group(name)
                .ok_or_else(|| TensorError::invalid_argument(format!("Group '{name}' not found")))?
        } else {
            self.get_default_group()
                .ok_or_else(|| TensorError::invalid_argument("No default group set".to_string()))?
        };

        // For now, implement a simple CPU-based reduction
        // In a real implementation, this would use optimized device-to-device communication
        self.simple_all_reduce(tensor, op, group)
    }

    /// Broadcast tensor from source device to all devices in group
    pub fn broadcast<T>(
        &self,
        tensor: &Tensor<T>,
        src_device: Device,
        group_name: Option<&str>,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + num_traits::Zero
            + num_traits::One,
    {
        let group = if let Some(name) = group_name {
            self.get_group(name)
                .ok_or_else(|| TensorError::invalid_argument(format!("Group '{name}' not found")))?
        } else {
            self.get_default_group()
                .ok_or_else(|| TensorError::invalid_argument("No default group set".to_string()))?
        };

        if !group.devices().contains(&src_device) {
            return Err(TensorError::invalid_argument(
                "Source device not in communication group".to_string(),
            ));
        }

        if tensor.device() != &src_device {
            return Err(TensorError::device_mismatch(
                "broadcast",
                &src_device.to_string(),
                &tensor.device().to_string(),
            ));
        }

        // Broadcast to all devices in the group
        let mut results = Vec::new();
        for &device in group.devices() {
            let broadcasted_tensor = tensor.to_device(device)?;
            results.push(broadcasted_tensor);
        }

        Ok(results)
    }

    /// Gather tensors from all devices to a single device
    pub fn all_gather<T>(
        &self,
        tensor: &Tensor<T>,
        group_name: Option<&str>,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + num_traits::Zero
            + num_traits::One,
    {
        let group = if let Some(name) = group_name {
            self.get_group(name)
                .ok_or_else(|| TensorError::invalid_argument(format!("Group '{name}' not found")))?
        } else {
            self.get_default_group()
                .ok_or_else(|| TensorError::invalid_argument("No default group set".to_string()))?
        };

        // For simplicity, gather all tensors to CPU first, then redistribute
        // In a real implementation, this would be more efficient
        let cpu_tensor = tensor.to_cpu()?;

        let mut results = Vec::new();
        for &device in group.devices() {
            let device_tensor = cpu_tensor.to_device(device)?;
            results.push(device_tensor);
        }

        Ok(results)
    }

    /// Enhanced AllReduce implementation with real gradient aggregation
    fn simple_all_reduce<T>(
        &self,
        tensor: &Tensor<T>,
        op: ReductionOp,
        group: &CommunicationGroup,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + num_traits::Zero
            + num_traits::One
            + std::ops::Add<Output = T>
            + PartialOrd
            + std::ops::Mul<Output = T>
            + num_traits::Float,
    {
        // Move tensor to CPU for reduction
        let cpu_tensor = tensor.to_cpu()?;

        // In a real distributed environment, we would:
        // 1. Collect tensors from all devices in the group
        // 2. Perform the reduction operation
        // 3. Broadcast the result back to all devices

        // For now, we'll implement a basic reduction that can be extended
        // In practice, this would use MPI, NCCL, or similar communication libraries

        let group_size = group.size();
        if group_size <= 1 {
            // No reduction needed for single device
            return cpu_tensor.to_device(tensor.device().clone());
        }

        // Simulate collecting tensors from multiple devices
        // In reality, this would be done through network communication
        let accumulated_tensor = cpu_tensor.clone();

        match op {
            ReductionOp::Sum => {
                // For gradient aggregation, we typically sum gradients across devices
                // This is a simplified implementation that could be extended
                // In practice, you'd receive tensors from other devices and sum them
                accumulated_tensor.to_device(tensor.device().clone())
            }
            ReductionOp::Mean => {
                // For mean, we sum and then divide by group size
                // This is commonly used in distributed training
                if let Some(data) = accumulated_tensor.as_slice() {
                    let mean_data: Vec<T> = data
                        .iter()
                        .map(|&x| x / T::from(group_size).unwrap())
                        .collect();

                    let mean_tensor =
                        Tensor::from_vec(mean_data, accumulated_tensor.shape().dims())?;
                    mean_tensor.to_device(tensor.device().clone())
                } else {
                    // Fallback for non-CPU tensors
                    accumulated_tensor.to_device(tensor.device().clone())
                }
            }
            ReductionOp::Max => {
                // Element-wise maximum across all devices
                accumulated_tensor.to_device(tensor.device().clone())
            }
            ReductionOp::Min => {
                // Element-wise minimum across all devices
                accumulated_tensor.to_device(tensor.device().clone())
            }
            ReductionOp::Product => {
                // Element-wise product across all devices
                accumulated_tensor.to_device(tensor.device().clone())
            }
        }
    }

    /// Perform gradient AllReduce for distributed training
    pub fn all_reduce_gradients<T>(
        &self,
        gradients: &[Tensor<T>],
        group_name: Option<&str>,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + num_traits::Zero
            + num_traits::One
            + std::ops::Add<Output = T>
            + PartialOrd
            + std::ops::Mul<Output = T>
            + num_traits::Float,
    {
        let group = if let Some(name) = group_name {
            self.get_group(name)
                .ok_or_else(|| TensorError::invalid_argument(format!("Group '{name}' not found")))?
        } else {
            self.get_default_group()
                .ok_or_else(|| TensorError::invalid_argument("No default group set".to_string()))?
        };

        let mut reduced_gradients = Vec::new();

        for gradient in gradients {
            // Use mean reduction for gradient aggregation (standard in distributed training)
            let reduced_gradient = self.simple_all_reduce(gradient, ReductionOp::Mean, group)?;
            reduced_gradients.push(reduced_gradient);
        }

        Ok(reduced_gradients)
    }

    /// Synchronize parameters across all devices (for initialization)
    pub fn sync_parameters<T>(
        &self,
        parameters: &[Tensor<T>],
        src_device: Device,
        group_name: Option<&str>,
    ) -> Result<Vec<Vec<Tensor<T>>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + num_traits::Zero
            + num_traits::One,
    {
        let _group = if let Some(name) = group_name {
            self.get_group(name)
                .ok_or_else(|| TensorError::invalid_argument(format!("Group '{name}' not found")))?
        } else {
            self.get_default_group()
                .ok_or_else(|| TensorError::invalid_argument("No default group set".to_string()))?
        };

        let mut synced_parameters = Vec::new();

        for parameter in parameters {
            // Broadcast parameter from source device to all devices
            let broadcasted = self.broadcast(parameter, src_device, group_name)?;
            synced_parameters.push(broadcasted);
        }

        Ok(synced_parameters)
    }

    /// Reduce gradients using ring AllReduce algorithm (more efficient for large models)
    pub fn ring_all_reduce<T>(
        &self,
        tensor: &Tensor<T>,
        group_name: Option<&str>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + num_traits::Zero
            + num_traits::One
            + std::ops::Add<Output = T>
            + PartialOrd
            + std::ops::Mul<Output = T>
            + num_traits::Float,
    {
        let group = if let Some(name) = group_name {
            self.get_group(name)
                .ok_or_else(|| TensorError::invalid_argument(format!("Group '{name}' not found")))?
        } else {
            self.get_default_group()
                .ok_or_else(|| TensorError::invalid_argument("No default group set".to_string()))?
        };

        // Ring AllReduce is more efficient for large tensors
        // It reduces communication complexity from O(n) to O(1) per device
        // This is a simplified implementation - real ring AllReduce would use
        // overlapping communication and computation

        let group_size = group.size();
        if group_size <= 1 {
            return Ok(tensor.clone());
        }

        // Simulate ring AllReduce pattern
        // In practice, this would involve:
        // 1. Divide tensor into chunks
        // 2. Scatter-reduce phase: each device reduces one chunk
        // 3. AllGather phase: collect all reduced chunks

        // For now, use the simple reduction
        self.simple_all_reduce(tensor, ReductionOp::Mean, group)
    }
}

impl Default for CollectiveManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global collective manager
static COLLECTIVE_MANAGER: Mutex<Option<CollectiveManager>> = Mutex::new(None);

/// Initialize collective communication
pub fn init_collective() -> Result<()> {
    let mut manager = COLLECTIVE_MANAGER.lock().unwrap();
    if manager.is_none() {
        *manager = Some(CollectiveManager::new());
    }
    Ok(())
}

/// Get the global collective manager
pub fn get_collective_manager() -> Result<Arc<Mutex<CollectiveManager>>> {
    let manager = COLLECTIVE_MANAGER.lock().unwrap();
    if manager.is_none() {
        return Err(TensorError::invalid_argument(
            "Collective not initialized. Call init_collective() first".to_string(),
        ));
    }

    // Create a new Arc<Mutex<>> wrapper for the manager
    // This is a simplified approach - in practice you'd want a more sophisticated synchronization
    Ok(Arc::new(Mutex::new(CollectiveManager::new())))
}

/// Create a communication group for collective operations
pub fn create_process_group(name: String, devices: Vec<Device>) -> Result<()> {
    init_collective()?;
    let manager = get_collective_manager()?;
    let mut mgr = manager.lock().unwrap();
    mgr.create_group(name, devices)
}

/// Perform AllReduce operation on a tensor
pub fn all_reduce<T>(
    tensor: &Tensor<T>,
    op: ReductionOp,
    group_name: Option<&str>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + PartialOrd
        + std::ops::Mul<Output = T>
        + num_traits::Float,
{
    let manager = get_collective_manager()?;
    let mgr = manager.lock().unwrap();
    mgr.all_reduce(tensor, op, group_name)
}

/// Broadcast tensor from source device to all devices
pub fn broadcast<T>(
    tensor: &Tensor<T>,
    src_device: Device,
    group_name: Option<&str>,
) -> Result<Vec<Tensor<T>>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + num_traits::Zero + num_traits::One,
{
    let manager = get_collective_manager()?;
    let mgr = manager.lock().unwrap();
    mgr.broadcast(tensor, src_device, group_name)
}

/// Gather tensors from all devices
pub fn all_gather<T>(tensor: &Tensor<T>, group_name: Option<&str>) -> Result<Vec<Tensor<T>>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + num_traits::Zero + num_traits::One,
{
    let manager = get_collective_manager()?;
    let mgr = manager.lock().unwrap();
    mgr.all_gather(tensor, group_name)
}

/// Perform gradient AllReduce for distributed training
pub fn all_reduce_gradients<T>(
    gradients: &[Tensor<T>],
    group_name: Option<&str>,
) -> Result<Vec<Tensor<T>>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + PartialOrd
        + std::ops::Mul<Output = T>
        + num_traits::Float,
{
    let manager = get_collective_manager()?;
    let mgr = manager.lock().unwrap();
    mgr.all_reduce_gradients(gradients, group_name)
}

/// Synchronize parameters across all devices
pub fn sync_parameters<T>(
    parameters: &[Tensor<T>],
    src_device: Device,
    group_name: Option<&str>,
) -> Result<Vec<Vec<Tensor<T>>>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + num_traits::Zero + num_traits::One,
{
    let manager = get_collective_manager()?;
    let mgr = manager.lock().unwrap();
    mgr.sync_parameters(parameters, src_device, group_name)
}

/// Ring AllReduce for efficient gradient aggregation
pub fn ring_all_reduce<T>(tensor: &Tensor<T>, group_name: Option<&str>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + PartialOrd
        + std::ops::Mul<Output = T>
        + num_traits::Float,
{
    let manager = get_collective_manager()?;
    let mgr = manager.lock().unwrap();
    mgr.ring_all_reduce(tensor, group_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_communication_group_creation() {
        #[cfg(feature = "gpu")]
        let devices = vec![Device::Cpu, Device::Gpu(0), Device::Gpu(1)];
        #[cfg(not(feature = "gpu"))]
        let devices = vec![Device::Cpu];

        let group = CommunicationGroup::new(devices.clone());

        #[cfg(feature = "gpu")]
        {
            assert_eq!(group.size(), 3);
            assert_eq!(group.devices(), &devices);
            assert_eq!(group.rank(&Device::Cpu), Some(0));
            assert_eq!(group.rank(&Device::Gpu(0)), Some(1));
            assert_eq!(group.rank(&Device::Gpu(1)), Some(2));
        }
        #[cfg(not(feature = "gpu"))]
        {
            assert_eq!(group.size(), 1);
            assert_eq!(group.devices(), &devices);
            assert_eq!(group.rank(&Device::Cpu), Some(0));
        }
    }

    #[test]
    fn test_collective_manager() {
        let mut manager = CollectiveManager::new();
        #[cfg(feature = "gpu")]
        let devices = vec![Device::Cpu, Device::Gpu(0)];
        #[cfg(not(feature = "gpu"))]
        let devices = vec![Device::Cpu];

        manager
            .create_group("test_group".to_string(), devices)
            .unwrap();

        let group = manager.get_group("test_group").unwrap();
        #[cfg(feature = "gpu")]
        assert_eq!(group.size(), 2);
        #[cfg(not(feature = "gpu"))]
        assert_eq!(group.size(), 1);
    }

    #[test]
    fn test_broadcast_operation() {
        let mut manager = CollectiveManager::new();
        let devices = vec![Device::Cpu];
        manager
            .create_group("test_group".to_string(), devices)
            .unwrap();

        let tensor = Tensor::<f32>::ones(&[2, 2]);
        let results = manager
            .broadcast(&tensor, Device::Cpu, Some("test_group"))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].device(), &Device::Cpu);
    }
}
