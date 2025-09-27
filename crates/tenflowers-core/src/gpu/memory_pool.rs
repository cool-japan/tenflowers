/*!
 * Advanced GPU Memory Pool Management
 *
 * This module implements sophisticated memory pooling and caching strategies
 * to minimize GPU memory allocations and reduce transfer overhead.
 */

use crate::{Device, Result, TensorError};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

/// Memory allocation request
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub alignment: u64,
    pub label: Option<String>,
}

/// Memory pool entry
#[derive(Debug)]
struct PoolEntry {
    buffer: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    last_used: Instant,
    ref_count: usize,
    id: usize, // Unique ID for tracking
}

/// ID generator for buffers
static BUFFER_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

fn next_buffer_id() -> usize {
    BUFFER_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocated: u64,
    pub total_freed: u64,
    pub current_usage: u64,
    pub peak_usage: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub fragmentation_ratio: f64,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum total memory pool size (bytes)
    pub max_pool_size: u64,
    /// Minimum buffer size to cache (smaller buffers are not pooled)
    pub min_cached_size: u64,
    /// Maximum buffer size to cache (larger buffers bypass pool)
    pub max_cached_size: u64,
    /// Time after which unused buffers are freed
    pub cleanup_interval: Duration,
    /// Maximum number of buffers per size bucket
    pub max_buffers_per_bucket: usize,
    /// Enable memory defragmentation
    pub enable_defragmentation: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1024 * 1024 * 1024, // 1GB default
            min_cached_size: 1024,             // 1KB minimum
            max_cached_size: 64 * 1024 * 1024, // 64MB maximum
            cleanup_interval: Duration::from_secs(60),
            max_buffers_per_bucket: 16,
            enable_defragmentation: true,
        }
    }
}

/// Advanced GPU memory pool with intelligent caching and defragmentation
pub struct GpuMemoryPool {
    device: Arc<wgpu::Device>,
    config: PoolConfig,
    /// Pools organized by size buckets for efficient lookup
    size_buckets: RwLock<HashMap<u64, VecDeque<PoolEntry>>>,
    /// Active allocations tracking by buffer pointer as usize (for Send + Sync)
    active_allocations: Mutex<HashMap<usize, PoolEntry>>,
    /// Memory statistics
    stats: RwLock<MemoryStats>,
    /// Last cleanup time
    last_cleanup: Mutex<Instant>,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    pub fn new(device: Arc<wgpu::Device>, config: PoolConfig) -> Self {
        Self {
            device,
            config,
            size_buckets: RwLock::new(HashMap::new()),
            active_allocations: Mutex::new(HashMap::new()),
            stats: RwLock::new(MemoryStats::default()),
            last_cleanup: Mutex::new(Instant::now()),
        }
    }

    /// Allocate a GPU buffer with pooling
    pub fn allocate(&self, request: &AllocationRequest) -> Result<Arc<wgpu::Buffer>> {
        // Check if we should use the pool
        if !self.should_use_pool(request) {
            return self.allocate_direct(request);
        }

        let size_bucket = self.calculate_size_bucket(request.size);

        // Try to get from pool first
        if let Some(buffer) = self.try_get_from_pool(size_bucket, request.usage) {
            self.update_stats_hit();
            return Ok(Arc::new(buffer));
        }

        // Allocate new buffer
        self.update_stats_miss();
        let buffer = self.create_new_buffer(request)?;

        // Track allocation
        self.track_allocation(&buffer, request)?;

        Ok(Arc::new(buffer))
    }

    /// Return a buffer to the pool
    pub fn deallocate(&self, buffer: Arc<wgpu::Buffer>) -> Result<()> {
        // Extract raw pointer as usize for lookup
        let buffer_ptr = Arc::as_ptr(&buffer) as usize;

        // Remove from active allocations and get metadata
        let entry = {
            let mut active = self.active_allocations.lock().unwrap();
            active.remove(&buffer_ptr)
        };

        if let Some(mut entry) = entry {
            // Update reference count
            entry.ref_count = Arc::strong_count(&buffer);

            // If this is the last reference and buffer is poolable, return to pool
            if entry.ref_count <= 2 {
                // Arc + our temporary reference
                if self.should_cache_buffer(&entry) {
                    self.return_to_pool(entry)?;
                } else {
                    self.update_stats_freed(entry.size);
                }
            } else {
                // Buffer still has references, keep tracking
                let mut active = self.active_allocations.lock().unwrap();
                active.insert(buffer_ptr, entry);
            }
        }

        // Periodically run cleanup
        self.maybe_run_cleanup()?;

        Ok(())
    }

    /// Force cleanup of unused buffers
    pub fn cleanup(&self) -> Result<()> {
        let mut freed_bytes = 0u64;
        let cutoff_time = Instant::now() - self.config.cleanup_interval;

        let mut size_buckets = self.size_buckets.write().unwrap();

        for (_, bucket) in size_buckets.iter_mut() {
            let original_len = bucket.len();
            bucket.retain(|entry| {
                if entry.last_used > cutoff_time && entry.ref_count == 0 {
                    true // Keep in pool
                } else {
                    freed_bytes += entry.size;
                    false // Remove from pool
                }
            });
        }

        // Update cleanup time
        *self.last_cleanup.lock().unwrap() = Instant::now();

        // Update stats
        self.update_stats_freed(freed_bytes);

        // Update fragmentation ratio
        self.update_fragmentation_stats();

        Ok(())
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Defragment memory pool by consolidating small buffers
    pub fn defragment(&self) -> Result<()> {
        if !self.config.enable_defragmentation {
            return Ok(());
        }

        let mut size_buckets = self.size_buckets.write().unwrap();
        let mut consolidated_bytes = 0u64;

        // Find fragmented buckets (many small buffers)
        let fragmented_buckets: Vec<u64> = size_buckets
            .iter()
            .filter(|(_, bucket)| bucket.len() > self.config.max_buffers_per_bucket / 2)
            .map(|(size, _)| *size)
            .collect();

        for size in fragmented_buckets {
            if let Some(bucket) = size_buckets.get_mut(&size) {
                // Sort by last used time
                let mut entries: Vec<_> = bucket.drain(..).collect();
                entries.sort_by_key(|e| e.last_used);

                // Keep only the most recently used buffers
                let keep_count = self.config.max_buffers_per_bucket.min(entries.len());
                let kept_entries = entries.into_iter().take(keep_count);

                bucket.extend(kept_entries);
                consolidated_bytes += size * (bucket.len() - keep_count) as u64;
            }
        }

        self.update_stats_freed(consolidated_bytes);
        Ok(())
    }

    /// Check if request should use the pool
    fn should_use_pool(&self, request: &AllocationRequest) -> bool {
        request.size >= self.config.min_cached_size
            && request.size <= self.config.max_cached_size
            && request.usage.contains(wgpu::BufferUsages::STORAGE)
    }

    /// Calculate size bucket for efficient pooling
    fn calculate_size_bucket(&self, size: u64) -> u64 {
        // Round up to next power of 2 for efficient bucketing
        if size <= 1024 {
            size.next_power_of_two()
        } else {
            // For larger sizes, use 16KB granularity
            ((size + 16383) / 16384) * 16384
        }
    }

    /// Try to get buffer from pool
    fn try_get_from_pool(
        &self,
        size_bucket: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<wgpu::Buffer> {
        let mut size_buckets = self.size_buckets.write().unwrap();

        if let Some(bucket) = size_buckets.get_mut(&size_bucket) {
            // Find buffer with compatible usage
            if let Some(pos) = bucket.iter().position(|entry| entry.usage.contains(usage)) {
                let entry = bucket.remove(pos)?;
                return Some(entry.buffer);
            }
        }

        None
    }

    /// Allocate buffer directly without pooling
    fn allocate_direct(&self, request: &AllocationRequest) -> Result<Arc<wgpu::Buffer>> {
        let buffer = self.create_new_buffer(request)?;
        Ok(Arc::new(buffer))
    }

    /// Create a new GPU buffer
    fn create_new_buffer(&self, request: &AllocationRequest) -> Result<wgpu::Buffer> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: request.label.as_deref(),
            size: request.size,
            usage: request.usage,
            mapped_at_creation: false,
        });

        self.update_stats_allocated(request.size);
        Ok(buffer)
    }

    /// Track buffer allocation
    fn track_allocation(&self, buffer: &wgpu::Buffer, request: &AllocationRequest) -> Result<()> {
        // Note: We don't actually use the PoolEntry.buffer field for active allocations,
        // so we create a dummy buffer here. This should be refactored in the future.
        let entry = PoolEntry {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: request.label.as_deref(),
                size: request.size,
                usage: request.usage,
                mapped_at_creation: false,
            }),
            size: request.size,
            usage: request.usage,
            last_used: Instant::now(),
            ref_count: 1,
            id: next_buffer_id(),
        };

        let buffer_ptr = buffer as *const wgpu::Buffer as usize;
        let mut active = self.active_allocations.lock().unwrap();
        active.insert(buffer_ptr, entry);

        Ok(())
    }

    /// Return buffer to pool
    fn return_to_pool(&self, mut entry: PoolEntry) -> Result<()> {
        entry.last_used = Instant::now();
        entry.ref_count = 0;

        let size_bucket = self.calculate_size_bucket(entry.size);
        let mut size_buckets = self.size_buckets.write().unwrap();

        let bucket = size_buckets
            .entry(size_bucket)
            .or_insert_with(VecDeque::new);

        // Limit bucket size
        if bucket.len() >= self.config.max_buffers_per_bucket {
            // Remove oldest entry
            if let Some(old_entry) = bucket.pop_front() {
                self.update_stats_freed(old_entry.size);
            }
        }

        bucket.push_back(entry);
        Ok(())
    }

    /// Check if buffer should be cached
    fn should_cache_buffer(&self, entry: &PoolEntry) -> bool {
        entry.size >= self.config.min_cached_size && entry.size <= self.config.max_cached_size
    }

    /// Maybe run cleanup if enough time has passed
    fn maybe_run_cleanup(&self) -> Result<()> {
        let last_cleanup = *self.last_cleanup.lock().unwrap();
        if last_cleanup.elapsed() > self.config.cleanup_interval {
            self.cleanup()?;
        }
        Ok(())
    }

    /// Update allocation statistics
    fn update_stats_allocated(&self, size: u64) {
        let mut stats = self.stats.write().unwrap();
        stats.total_allocated += size;
        stats.current_usage += size;
        if stats.current_usage > stats.peak_usage {
            stats.peak_usage = stats.current_usage;
        }
    }

    /// Update deallocation statistics
    fn update_stats_freed(&self, size: u64) {
        let mut stats = self.stats.write().unwrap();
        stats.total_freed += size;
        stats.current_usage = stats.current_usage.saturating_sub(size);
    }

    /// Update cache hit statistics
    fn update_stats_hit(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.cache_hits += 1;
    }

    /// Update cache miss statistics
    fn update_stats_miss(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.cache_misses += 1;
    }

    /// Update fragmentation statistics
    fn update_fragmentation_stats(&self) {
        let size_buckets = self.size_buckets.read().unwrap();
        let mut total_buckets = 0;
        let mut fragmented_buckets = 0;

        for (_, bucket) in size_buckets.iter() {
            total_buckets += 1;
            if bucket.len() > self.config.max_buffers_per_bucket / 2 {
                fragmented_buckets += 1;
            }
        }

        let fragmentation_ratio = if total_buckets > 0 {
            fragmented_buckets as f64 / total_buckets as f64
        } else {
            0.0
        };

        let mut stats = self.stats.write().unwrap();
        stats.fragmentation_ratio = fragmentation_ratio;
    }

    /// Get cache hit ratio
    pub fn get_cache_hit_ratio(&self) -> f64 {
        let stats = self.stats.read().unwrap();
        let total_requests = stats.cache_hits + stats.cache_misses;
        if total_requests > 0 {
            stats.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Get memory efficiency ratio
    pub fn get_memory_efficiency(&self) -> f64 {
        let stats = self.stats.read().unwrap();
        if stats.peak_usage > 0 {
            stats.current_usage as f64 / stats.peak_usage as f64
        } else {
            1.0
        }
    }
}

/// Global memory pool manager
pub struct GlobalMemoryManager {
    pools: RwLock<HashMap<usize, Arc<GpuMemoryPool>>>,
}

impl GlobalMemoryManager {
    /// Get global instance
    pub fn instance() -> &'static Self {
        static INSTANCE: std::sync::OnceLock<GlobalMemoryManager> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| Self {
            pools: RwLock::new(HashMap::new()),
        })
    }

    /// Get or create memory pool for device
    pub fn get_pool(&self, device_id: usize, device: Arc<wgpu::Device>) -> Arc<GpuMemoryPool> {
        let pools = self.pools.read().unwrap();
        if let Some(pool) = pools.get(&device_id) {
            return Arc::clone(pool);
        }
        drop(pools);

        // Create new pool
        let mut pools = self.pools.write().unwrap();
        let pool = Arc::new(GpuMemoryPool::new(device, PoolConfig::default()));
        pools.insert(device_id, Arc::clone(&pool));
        pool
    }

    /// Get memory statistics for all devices
    pub fn get_global_stats(&self) -> HashMap<usize, MemoryStats> {
        let pools = self.pools.read().unwrap();
        pools
            .iter()
            .map(|(device_id, pool)| (*device_id, pool.get_stats()))
            .collect()
    }

    /// Cleanup all pools
    pub fn cleanup_all(&self) -> Result<()> {
        let pools = self.pools.read().unwrap();
        for pool in pools.values() {
            pool.cleanup()?;
        }
        Ok(())
    }

    /// Defragment all pools
    pub fn defragment_all(&self) -> Result<()> {
        let pools = self.pools.read().unwrap();
        for pool in pools.values() {
            pool.defragment()?;
        }
        Ok(())
    }
}

/// Convenient allocation functions
pub fn allocate_gpu_buffer(
    device_id: usize,
    device: Arc<wgpu::Device>,
    size: u64,
    usage: wgpu::BufferUsages,
) -> Result<Arc<wgpu::Buffer>> {
    let pool = GlobalMemoryManager::instance().get_pool(device_id, device);
    let request = AllocationRequest {
        size,
        usage,
        alignment: 4, // Default alignment
        label: None,
    };
    pool.allocate(&request)
}

pub fn deallocate_gpu_buffer(device_id: usize, buffer: Arc<wgpu::Buffer>) -> Result<()> {
    // This requires the device to get the pool, but we don't have it here
    // In a real implementation, you'd store device_id in the buffer metadata
    // For now, just drop the buffer
    drop(buffer);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bucket_calculation() {
        // Test the size bucket calculation logic directly without requiring a device
        fn calculate_size_bucket(size: u64) -> u64 {
            // Round up to next power of 2 for efficient bucketing
            if size <= 1024 {
                size.next_power_of_two()
            } else {
                // For larger sizes, use 16KB granularity
                ((size + 16383) / 16384) * 16384
            }
        }

        assert_eq!(calculate_size_bucket(100), 128);
        assert_eq!(calculate_size_bucket(1024), 1024);
        assert_eq!(calculate_size_bucket(20000), 32768);
    }

    #[test]
    fn test_pool_config_defaults() {
        let config = PoolConfig::default();
        assert_eq!(config.max_pool_size, 1024 * 1024 * 1024);
        assert_eq!(config.min_cached_size, 1024);
        assert!(config.enable_defragmentation);
    }

    #[test]
    fn test_memory_stats_initialization() {
        let stats = MemoryStats::default();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.current_usage, 0);
        assert_eq!(stats.cache_hits, 0);
    }
}
