//! Persistent caching implementations
//!
//! This module provides disk-based caching that persists across program runs.

use crate::cache::dataset::CacheStats;
use crate::Dataset;
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::hash::Hash;
use std::io::{BufReader, BufWriter};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
/// Persistent cache that stores data on disk with LRU eviction
pub struct PersistentCache<K, V> {
    cache_dir: std::path::PathBuf,
    capacity: usize,
    index: HashMap<K, (String, usize)>, // (filename, access_order)
    access_counter: usize,
    _phantom: PhantomData<V>,
}

impl<K, V> PersistentCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Display + std::str::FromStr,
    V: Clone + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Create a new persistent cache with the specified directory and capacity
    pub fn new<P: AsRef<Path>>(cache_dir: P, capacity: usize) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            create_dir_all(&cache_dir).map_err(|e| {
                TensorError::invalid_argument(format!("Failed to create cache directory: {e}"))
            })?;
        }

        let mut cache = Self {
            cache_dir,
            capacity,
            index: HashMap::new(),
            access_counter: 0,
            _phantom: PhantomData,
        };

        // Load existing cache index
        cache.load_index()?;

        Ok(cache)
    }

    /// Load cache index from disk
    fn load_index(&mut self) -> Result<()> {
        let index_path = self.cache_dir.join("cache_index.json");

        if !index_path.exists() {
            return Ok(()); // No existing index
        }

        let file = File::open(&index_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to open cache index: {e}"))
        })?;

        let reader = BufReader::new(file);

        // Simple JSON format: {"key": {"filename": "...", "access_order": 123}, ...}
        let index_data: HashMap<String, (String, usize)> = serde_json::from_reader(reader)
            .map_err(|e| {
                TensorError::invalid_argument(format!("Failed to parse cache index: {e}"))
            })?;

        // Convert string keys back to original type (simplified approach)
        for (key_str, (filename, access_order)) in index_data {
            if let Ok(key) = key_str.parse::<K>() {
                self.index.insert(key, (filename, access_order));
                self.access_counter = self.access_counter.max(access_order);
            }
        }

        self.access_counter += 1; // Ensure next access has higher number

        Ok(())
    }

    /// Save cache index to disk
    fn save_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("cache_index.json");

        let file = File::create(&index_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to create cache index: {e}"))
        })?;

        let writer = BufWriter::new(file);

        // Convert to string keys for JSON serialization
        let index_data: HashMap<String, (String, usize)> = self
            .index
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        serde_json::to_writer(writer, &index_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to save cache index: {e}"))
        })?;

        Ok(())
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &K) -> Result<Option<V>> {
        if let Some((filename, access_time)) = self.index.get_mut(key) {
            // Update access time
            self.access_counter += 1;
            *access_time = self.access_counter;

            // Load value from disk
            let file_path = self.cache_dir.join(filename);

            if !file_path.exists() {
                // File was deleted, remove from index
                self.index.remove(key);
                return Ok(None);
            }

            let file = File::open(&file_path).map_err(|e| {
                TensorError::invalid_argument(format!("Failed to open cache file: {e}"))
            })?;

            let reader = BufReader::new(file);

            let value: V = bincode::deserialize_from(reader).map_err(|e| {
                TensorError::invalid_argument(format!("Failed to deserialize cached value: {e}"))
            })?;

            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Insert a value into the cache
    pub fn insert(&mut self, key: K, value: V) -> Result<()> {
        self.access_counter += 1;

        // Check if we need to evict items
        if self.index.len() >= self.capacity && !self.index.contains_key(&key) {
            self.evict_lru()?;
        }

        // Generate filename for this entry
        let filename = format!("cache_{}_{}.bin", key, self.access_counter);
        let file_path = self.cache_dir.join(&filename);

        // Serialize and save to disk
        let file = File::create(&file_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to create cache file: {e}"))
        })?;

        let writer = BufWriter::new(file);

        bincode::serialize_into(writer, &value).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize value: {e}"))
        })?;

        // Update index
        if let Some((old_filename, _)) = self.index.insert(key, (filename, self.access_counter)) {
            // Remove old file if it exists
            let old_path = self.cache_dir.join(old_filename);
            let _ = std::fs::remove_file(old_path); // Ignore errors
        }

        // Save updated index
        self.save_index()?;

        Ok(())
    }

    /// Evict the least recently used item
    fn evict_lru(&mut self) -> Result<()> {
        if let Some((lru_key, (filename, _))) = self
            .index
            .iter()
            .min_by_key(|(_, (_, access_time))| *access_time)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            // Remove file
            let file_path = self.cache_dir.join(&filename);
            let _ = std::fs::remove_file(file_path); // Ignore errors

            // Remove from index
            self.index.remove(&lru_key);
        }

        Ok(())
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Clear all cached items
    pub fn clear(&mut self) -> Result<()> {
        // Remove all cache files
        for (filename, _) in self.index.values() {
            let file_path = self.cache_dir.join(filename);
            let _ = std::fs::remove_file(file_path); // Ignore errors
        }

        // Clear index
        self.index.clear();
        self.access_counter = 0;

        // Save empty index
        self.save_index()?;

        Ok(())
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

#[cfg(feature = "serialize")]
/// Persistent cache that works with byte arrays for tensor data
pub struct TensorPersistentCache {
    cache: PersistentCache<usize, (Vec<u8>, Vec<u8>)>, // Serialized tensor data
}

impl TensorPersistentCache {
    /// Create a new tensor persistent cache
    pub fn new<P: AsRef<Path>>(cache_dir: P, capacity: usize) -> Result<Self> {
        Ok(Self {
            cache: PersistentCache::new(cache_dir, capacity)?,
        })
    }

    /// Get tensors from cache
    pub fn get<T>(&mut self, index: &usize) -> Result<Option<(Tensor<T>, Tensor<T>)>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::cast::NumCast,
    {
        if let Some((features_bytes, labels_bytes)) = self.cache.get(index)? {
            // Deserialize tensors from byte arrays
            let features_tensor = Self::deserialize_tensor(&features_bytes)?;
            let labels_tensor = Self::deserialize_tensor(&labels_bytes)?;
            Ok(Some((features_tensor, labels_tensor)))
        } else {
            Ok(None)
        }
    }

    /// Insert tensors into cache
    pub fn insert<T>(
        &mut self,
        index: usize,
        features: &Tensor<T>,
        labels: &Tensor<T>,
    ) -> Result<()>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::cast::NumCast,
    {
        // Serialize tensors to byte arrays
        let features_bytes = Self::serialize_tensor(features)?;
        let labels_bytes = Self::serialize_tensor(labels)?;

        // Store in persistent cache
        self.cache.insert(index, (features_bytes, labels_bytes))?;
        Ok(())
    }

    /// Clear cache
    pub fn clear(&mut self) -> Result<()> {
        self.cache.clear()
    }

    /// Serialize a tensor to bytes
    /// Format: [type_id: u8][shape_len: u32][shape: u32...][data: T...]
    fn serialize_tensor<T>(tensor: &Tensor<T>) -> Result<Vec<u8>>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        let mut bytes = Vec::new();

        // Determine type ID based on size of T (simple heuristic)
        let type_id = std::mem::size_of::<T>() as u8;
        bytes.push(type_id);

        // Serialize shape
        let shape = tensor.shape().dims();
        let shape_len = shape.len() as u32;
        bytes.extend_from_slice(&shape_len.to_le_bytes());

        for &dim in shape {
            bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        // Serialize data - try to get raw data
        if let Some(data_slice) = tensor.as_slice() {
            // For CPU tensors, convert each element to bytes safely
            for element in data_slice.iter() {
                // Use a safe approach to get bytes representation
                let element_ptr = element as *const T as *const u8;
                let element_bytes = std::mem::size_of::<T>();
                // SAFETY: We're reading from a valid T reference for exactly size_of::<T>() bytes
                #[allow(unsafe_code)]
                let element_data =
                    unsafe { std::slice::from_raw_parts(element_ptr, element_bytes) };
                bytes.extend_from_slice(element_data);
            }
        } else {
            return Err(TensorError::invalid_argument(
                "Cannot serialize GPU tensors or tensors without CPU data".to_string(),
            ));
        }

        Ok(bytes)
    }

    /// Deserialize a tensor from bytes
    fn deserialize_tensor<T>(bytes: &[u8]) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::cast::NumCast,
    {
        if bytes.len() < 5 {
            // At least type_id + shape_len
            return Err(TensorError::invalid_argument(
                "Invalid tensor serialization: too few bytes".to_string(),
            ));
        }

        let mut offset = 0;

        // Read type ID (for validation)
        let _type_id = bytes[offset];
        offset += 1;

        // Read shape length
        let shape_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + shape_len * 4 {
            return Err(TensorError::invalid_argument(
                "Invalid tensor serialization: insufficient bytes for shape".to_string(),
            ));
        }

        // Read shape
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let dim = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            shape.push(dim);
            offset += 4;
        }

        // Calculate expected data size
        let total_elements = shape.iter().product::<usize>();
        let element_size = std::mem::size_of::<T>();
        let expected_data_bytes = total_elements * element_size;

        if bytes.len() < offset + expected_data_bytes {
            return Err(TensorError::invalid_argument(
                "Invalid tensor serialization: insufficient bytes for data".to_string(),
            ));
        }

        // Deserialize data
        let data_bytes = &bytes[offset..offset + expected_data_bytes];

        // Convert bytes back to T values
        let mut data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            let element_offset = i * element_size;

            // Simple conversion based on element size
            let value = match element_size {
                1 => {
                    // u8 or i8
                    let byte_val = data_bytes[element_offset];
                    scirs2_core::num_traits::cast::NumCast::from(byte_val)
                        .unwrap_or_else(T::default)
                }
                2 => {
                    // u16 or i16
                    if element_offset + 2 <= data_bytes.len() {
                        let val = u16::from_le_bytes([
                            data_bytes[element_offset],
                            data_bytes[element_offset + 1],
                        ]);
                        scirs2_core::num_traits::cast::NumCast::from(val).unwrap_or_else(T::default)
                    } else {
                        T::default()
                    }
                }
                4 => {
                    // u32, i32, or f32
                    if element_offset + 4 <= data_bytes.len() {
                        let val = f32::from_le_bytes([
                            data_bytes[element_offset],
                            data_bytes[element_offset + 1],
                            data_bytes[element_offset + 2],
                            data_bytes[element_offset + 3],
                        ]);
                        scirs2_core::num_traits::cast::NumCast::from(val).unwrap_or_else(T::default)
                    } else {
                        T::default()
                    }
                }
                8 => {
                    // u64, i64, or f64
                    if element_offset + 8 <= data_bytes.len() {
                        let val = f64::from_le_bytes([
                            data_bytes[element_offset],
                            data_bytes[element_offset + 1],
                            data_bytes[element_offset + 2],
                            data_bytes[element_offset + 3],
                            data_bytes[element_offset + 4],
                            data_bytes[element_offset + 5],
                            data_bytes[element_offset + 6],
                            data_bytes[element_offset + 7],
                        ]);
                        scirs2_core::num_traits::cast::NumCast::from(val).unwrap_or_else(T::default)
                    } else {
                        T::default()
                    }
                }
                _ => {
                    // Unsupported size, use default
                    T::default()
                }
            };

            data.push(value);
        }

        // Create tensor from deserialized data
        Tensor::from_vec(data, &shape)
    }
}

#[cfg(feature = "serialize")]
/// Dataset wrapper that uses persistent caching with simplified implementation
pub struct PersistentlyCachedDataset<T, D: Dataset<T>> {
    dataset: D,
    cache: Arc<Mutex<TensorPersistentCache>>,
    cache_stats: Arc<Mutex<CacheStats>>,
    _phantom: PhantomData<T>,
}

impl<T, D: Dataset<T>> PersistentlyCachedDataset<T, D>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::cast::NumCast,
{
    /// Create a new persistently cached dataset
    pub fn new<P: AsRef<Path>>(dataset: D, cache_dir: P, cache_capacity: usize) -> Result<Self> {
        let cache = TensorPersistentCache::new(cache_dir, cache_capacity)?;

        Ok(Self {
            dataset,
            cache: Arc::new(Mutex::new(cache)),
            cache_stats: Arc::new(Mutex::new(CacheStats::default())),
            _phantom: PhantomData,
        })
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<CacheStats> {
        match self.cache_stats.lock() {
            Ok(stats) => Ok(stats.clone()),
            Err(_) => Err(TensorError::CacheError {
                operation: "persistent_cache_stats".to_string(),
                details: "Persistent cache stats mutex poisoned".to_string(),
                recoverable: true,
                context: None,
            }),
        }
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<()> {
        match self.cache.lock() {
            Ok(mut cache) => cache.clear()?,
            Err(_) => {
                return Err(TensorError::CacheError {
                    operation: "persistent_cache_clear".to_string(),
                    details: "Persistent cache mutex poisoned during clear".to_string(),
                    recoverable: false,
                    context: None,
                })
            }
        }

        match self.cache_stats.lock() {
            Ok(mut stats) => {
                *stats = CacheStats::default();
                Ok(())
            }
            Err(_) => Err(TensorError::CacheError {
                operation: "persistent_cache_clear_stats".to_string(),
                details: "Persistent cache stats mutex poisoned during clear".to_string(),
                recoverable: false,
                context: None,
            }),
        }
    }

    /// Get underlying dataset
    pub fn into_inner(self) -> D {
        self.dataset
    }

    /// Get reference to underlying dataset
    pub fn inner(&self) -> &D {
        &self.dataset
    }
}

impl<T, D: Dataset<T>> Dataset<T> for PersistentlyCachedDataset<T, D>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::cast::NumCast,
{
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        // Update stats
        match self.cache_stats.lock() {
            Ok(mut stats) => stats.total_requests += 1,
            Err(_) => {
                return Err(TensorError::CacheError {
                    operation: "persistent_cache_stats_update".to_string(),
                    details: "Persistent cache stats mutex poisoned during total requests update"
                        .to_string(),
                    recoverable: false,
                    context: None,
                })
            }
        }

        // Try cache first
        let cache_result = match self.cache.lock() {
            Ok(mut cache) => cache.get(&index),
            Err(_) => {
                return Err(TensorError::CacheError {
                    operation: "persistent_cache_get".to_string(),
                    details: "Persistent cache mutex poisoned during get operation".to_string(),
                    recoverable: false,
                    context: None,
                })
            }
        };

        if let Ok(Some(cached_sample)) = cache_result {
            // Cache hit - update hit stats
            match self.cache_stats.lock() {
                Ok(mut stats) => stats.hits += 1,
                Err(_) => {
                    return Err(TensorError::CacheError {
                        operation: "persistent_cache_hit_stats".to_string(),
                        details: "Persistent cache stats mutex poisoned during hit update"
                            .to_string(),
                        recoverable: false,
                        context: None,
                    })
                }
            }
            return Ok(cached_sample);
        }

        // Cache miss - load from dataset
        let sample = self.dataset.get(index)?;

        // Cache the result (currently a no-op due to serialization limitations)
        match self.cache.lock() {
            Ok(mut cache) => {
                if let Err(e) = cache.insert(index, &sample.0, &sample.1) {
                    // Log warning but don't fail the operation
                    eprintln!("Warning: Failed to cache sample {index}: {e}");
                }
            }
            Err(_) => {
                // Log warning but don't fail the operation
                eprintln!("Warning: Cache mutex poisoned during insert for sample {index}");
            }
        }

        // Update miss stats
        match self.cache_stats.lock() {
            Ok(mut stats) => stats.misses += 1,
            Err(_) => {
                return Err(TensorError::CacheError {
                    operation: "persistent_cache_miss_stats".to_string(),
                    details: "Persistent cache stats mutex poisoned during miss update".to_string(),
                    recoverable: false,
                    context: None,
                })
            }
        }

        Ok(sample)
    }
}
