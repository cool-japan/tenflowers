//! Model serialization module for TenfloweRS FFI
//!
//! This module provides utilities for saving and loading models, including
//! weights, optimizer states, and training configurations.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Save a tensor to a file
///
/// Saves tensor data in a simple binary format: [ndim, shape..., data...]
///
/// # Arguments
///
/// * `tensor` - Tensor to save
/// * `path` - File path to save to
#[pyfunction]
pub fn save_tensor(tensor: &PyTensor, path: String) -> PyResult<()> {
    let tensor_data = tensor
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e)))?;

    let shape = tensor.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let mut file = File::create(&path)
        .map_err(|e| PyIOError::new_err(format!("Failed to create file {}: {}", path, e)))?;

    // Write number of dimensions
    let ndim = shape_vec.len() as u32;
    file.write_all(&ndim.to_le_bytes())
        .map_err(|e| PyIOError::new_err(format!("Failed to write ndim: {}", e)))?;

    // Write shape
    for &dim in &shape_vec {
        file.write_all(&(dim as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write shape: {}", e)))?;
    }

    // Write data
    for &val in &tensor_data {
        file.write_all(&val.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write data: {}", e)))?;
    }

    Ok(())
}

/// Load a tensor from a file
///
/// Loads tensor data saved with save_tensor.
///
/// # Arguments
///
/// * `path` - File path to load from
///
/// # Returns
///
/// Loaded tensor
#[pyfunction]
pub fn load_tensor(path: String) -> PyResult<PyTensor> {
    let mut file = File::open(&path)
        .map_err(|e| PyIOError::new_err(format!("Failed to open file {}: {}", path, e)))?;

    // Read number of dimensions
    let mut ndim_bytes = [0u8; 4];
    file.read_exact(&mut ndim_bytes)
        .map_err(|e| PyIOError::new_err(format!("Failed to read ndim: {}", e)))?;
    let ndim = u32::from_le_bytes(ndim_bytes) as usize;

    // Read shape
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_bytes = [0u8; 8];
        file.read_exact(&mut dim_bytes)
            .map_err(|e| PyIOError::new_err(format!("Failed to read shape: {}", e)))?;
        shape.push(u64::from_le_bytes(dim_bytes) as usize);
    }

    // Calculate total elements
    let total_elements: usize = shape.iter().product();

    // Read data
    let mut data = Vec::with_capacity(total_elements);
    for _ in 0..total_elements {
        let mut val_bytes = [0u8; 4];
        file.read_exact(&mut val_bytes)
            .map_err(|e| PyIOError::new_err(format!("Failed to read data: {}", e)))?;
        data.push(f32::from_le_bytes(val_bytes));
    }

    // Create tensor
    let tensor = Tensor::from_vec(data, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Save a state dictionary to a file
///
/// Saves a dictionary of named tensors (model weights, optimizer state, etc.)
///
/// # Arguments
///
/// * `state_dict` - Dictionary mapping names to tensors
/// * `path` - File path to save to
#[pyfunction]
pub fn save_state_dict(state_dict: &Bound<'_, PyDict>, path: String) -> PyResult<()> {
    let mut file = File::create(&path)
        .map_err(|e| PyIOError::new_err(format!("Failed to create file {}: {}", path, e)))?;

    // Get all keys
    let keys: Vec<String> = state_dict
        .keys()
        .iter()
        .map(|k| k.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    // Write number of entries
    let num_entries = keys.len() as u32;
    file.write_all(&num_entries.to_le_bytes())
        .map_err(|e| PyIOError::new_err(format!("Failed to write num_entries: {}", e)))?;

    // Write each entry
    for key in keys {
        // Write key length and key
        let key_bytes = key.as_bytes();
        let key_len = key_bytes.len() as u32;
        file.write_all(&key_len.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write key length: {}", e)))?;
        file.write_all(key_bytes)
            .map_err(|e| PyIOError::new_err(format!("Failed to write key: {}", e)))?;

        // Get tensor
        let tensor_obj = state_dict.get_item(&key)?.ok_or_else(|| {
            PyValueError::new_err(format!("Key '{}' not found in state_dict", key))
        })?;
        let tensor = tensor_obj.extract::<PyTensor>()?;

        // Write tensor data
        let tensor_data = tensor.tensor.to_vec().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get tensor data for '{}': {}", key, e))
        })?;

        let shape = tensor.tensor.shape();
        let shape_vec: Vec<usize> = shape.iter().copied().collect();

        // Write number of dimensions
        let ndim = shape_vec.len() as u32;
        file.write_all(&ndim.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write ndim: {}", e)))?;

        // Write shape
        for &dim in &shape_vec {
            file.write_all(&(dim as u64).to_le_bytes())
                .map_err(|e| PyIOError::new_err(format!("Failed to write shape: {}", e)))?;
        }

        // Write data
        for &val in &tensor_data {
            file.write_all(&val.to_le_bytes())
                .map_err(|e| PyIOError::new_err(format!("Failed to write data: {}", e)))?;
        }
    }

    Ok(())
}

/// Load a state dictionary from a file
///
/// Loads a dictionary of named tensors saved with save_state_dict.
///
/// # Arguments
///
/// * `path` - File path to load from
///
/// # Returns
///
/// Dictionary mapping names to tensors
#[pyfunction]
pub fn load_state_dict(path: String, py: Python) -> PyResult<Py<PyDict>> {
    let mut file = File::open(&path)
        .map_err(|e| PyIOError::new_err(format!("Failed to open file {}: {}", path, e)))?;

    // Read number of entries
    let mut num_entries_bytes = [0u8; 4];
    file.read_exact(&mut num_entries_bytes)
        .map_err(|e| PyIOError::new_err(format!("Failed to read num_entries: {}", e)))?;
    let num_entries = u32::from_le_bytes(num_entries_bytes);

    let state_dict = PyDict::new(py);

    // Read each entry
    for _ in 0..num_entries {
        // Read key length and key
        let mut key_len_bytes = [0u8; 4];
        file.read_exact(&mut key_len_bytes)
            .map_err(|e| PyIOError::new_err(format!("Failed to read key length: {}", e)))?;
        let key_len = u32::from_le_bytes(key_len_bytes) as usize;

        let mut key_bytes = vec![0u8; key_len];
        file.read_exact(&mut key_bytes)
            .map_err(|e| PyIOError::new_err(format!("Failed to read key: {}", e)))?;
        let key = String::from_utf8(key_bytes)
            .map_err(|e| PyValueError::new_err(format!("Invalid UTF-8 in key: {}", e)))?;

        // Read tensor data
        // Read number of dimensions
        let mut ndim_bytes = [0u8; 4];
        file.read_exact(&mut ndim_bytes)
            .map_err(|e| PyIOError::new_err(format!("Failed to read ndim: {}", e)))?;
        let ndim = u32::from_le_bytes(ndim_bytes) as usize;

        // Read shape
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_bytes = [0u8; 8];
            file.read_exact(&mut dim_bytes)
                .map_err(|e| PyIOError::new_err(format!("Failed to read shape: {}", e)))?;
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }

        // Calculate total elements
        let total_elements: usize = shape.iter().product();

        // Read data
        let mut data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            let mut val_bytes = [0u8; 4];
            file.read_exact(&mut val_bytes)
                .map_err(|e| PyIOError::new_err(format!("Failed to read data: {}", e)))?;
            data.push(f32::from_le_bytes(val_bytes));
        }

        // Create tensor
        let tensor = Tensor::from_vec(data, &shape).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create tensor for '{}': {}", key, e))
        })?;

        let py_tensor = PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        };

        state_dict.set_item(key, py_tensor)?;
    }

    Ok(state_dict.unbind())
}

/// Model Checkpoint Manager
///
/// Manages saving and loading model checkpoints including optimizer state.
#[pyclass(name = "CheckpointManager")]
#[derive(Debug, Clone)]
pub struct PyCheckpointManager {
    /// Directory to save checkpoints
    pub checkpoint_dir: String,
    /// Maximum number of checkpoints to keep
    pub max_to_keep: usize,
    /// List of saved checkpoint paths
    pub saved_checkpoints: Vec<String>,
}

#[pymethods]
impl PyCheckpointManager {
    /// Create a new checkpoint manager
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Directory to save checkpoints
    /// * `max_to_keep` - Maximum number of checkpoints to keep (default: 5)
    #[new]
    #[pyo3(signature = (checkpoint_dir, max_to_keep=None))]
    pub fn new(checkpoint_dir: String, max_to_keep: Option<usize>) -> PyResult<Self> {
        let max_to_keep = max_to_keep.unwrap_or(5);

        if max_to_keep == 0 {
            return Err(PyValueError::new_err("max_to_keep must be positive"));
        }

        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&checkpoint_dir).map_err(|e| {
            PyIOError::new_err(format!(
                "Failed to create checkpoint directory {}: {}",
                checkpoint_dir, e
            ))
        })?;

        Ok(PyCheckpointManager {
            checkpoint_dir,
            max_to_keep,
            saved_checkpoints: Vec::new(),
        })
    }

    /// Save a checkpoint
    ///
    /// # Arguments
    ///
    /// * `model_state` - Model state dictionary
    /// * `optimizer_state` - Optional optimizer state dictionary
    /// * `epoch` - Current epoch number
    /// * `step` - Current step number
    ///
    /// # Returns
    ///
    /// Path to saved checkpoint
    pub fn save(
        &mut self,
        model_state: &Bound<'_, PyDict>,
        optimizer_state: Option<&Bound<'_, PyDict>>,
        epoch: usize,
        step: usize,
    ) -> PyResult<String> {
        let checkpoint_name = format!("checkpoint_epoch{}_step{}.bin", epoch, step);
        let checkpoint_path = Path::new(&self.checkpoint_dir).join(&checkpoint_name);
        let checkpoint_path_str = checkpoint_path
            .to_str()
            .ok_or_else(|| PyValueError::new_err("Invalid checkpoint path"))?
            .to_string();

        let mut file = File::create(&checkpoint_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to create checkpoint file: {}", e)))?;

        // Write epoch and step
        file.write_all(&(epoch as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write epoch: {}", e)))?;
        file.write_all(&(step as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write step: {}", e)))?;

        // Save model state to temporary buffer
        let mut model_buffer = Vec::new();
        save_state_dict_to_buffer(model_state, &mut model_buffer)?;

        // Write model state size and data
        file.write_all(&(model_buffer.len() as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write model size: {}", e)))?;
        file.write_all(&model_buffer)
            .map_err(|e| PyIOError::new_err(format!("Failed to write model state: {}", e)))?;

        // Save optimizer state if provided
        if let Some(opt_state) = optimizer_state {
            file.write_all(&1u8.to_le_bytes())
                .map_err(|e| PyIOError::new_err(format!("Failed to write has_optimizer: {}", e)))?;

            let mut opt_buffer = Vec::new();
            save_state_dict_to_buffer(opt_state, &mut opt_buffer)?;

            file.write_all(&(opt_buffer.len() as u64).to_le_bytes())
                .map_err(|e| {
                    PyIOError::new_err(format!("Failed to write optimizer size: {}", e))
                })?;
            file.write_all(&opt_buffer).map_err(|e| {
                PyIOError::new_err(format!("Failed to write optimizer state: {}", e))
            })?;
        } else {
            file.write_all(&0u8.to_le_bytes())
                .map_err(|e| PyIOError::new_err(format!("Failed to write has_optimizer: {}", e)))?;
        }

        // Track saved checkpoints
        self.saved_checkpoints.push(checkpoint_path_str.clone());

        // Remove old checkpoints if exceeding max_to_keep
        while self.saved_checkpoints.len() > self.max_to_keep {
            if let Some(old_path) = self.saved_checkpoints.first() {
                let _ = std::fs::remove_file(old_path);
                self.saved_checkpoints.remove(0);
            }
        }

        Ok(checkpoint_path_str)
    }

    /// Get the latest checkpoint path
    pub fn latest_checkpoint(&self) -> Option<String> {
        self.saved_checkpoints.last().cloned()
    }

    fn __repr__(&self) -> String {
        format!(
            "CheckpointManager(checkpoint_dir='{}', max_to_keep={}, saved={})",
            self.checkpoint_dir,
            self.max_to_keep,
            self.saved_checkpoints.len()
        )
    }
}

// Helper function to save state dict to a buffer
fn save_state_dict_to_buffer(state_dict: &Bound<'_, PyDict>, buffer: &mut Vec<u8>) -> PyResult<()> {
    let keys: Vec<String> = state_dict
        .keys()
        .iter()
        .map(|k| k.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    // Write number of entries
    let num_entries = keys.len() as u32;
    buffer.extend_from_slice(&num_entries.to_le_bytes());

    for key in keys {
        let key_bytes = key.as_bytes();
        let key_len = key_bytes.len() as u32;
        buffer.extend_from_slice(&key_len.to_le_bytes());
        buffer.extend_from_slice(key_bytes);

        let tensor_obj = state_dict.get_item(&key)?.ok_or_else(|| {
            PyValueError::new_err(format!("Key '{}' not found in state_dict", key))
        })?;
        let tensor = tensor_obj.extract::<PyTensor>()?;

        let tensor_data = tensor.tensor.to_vec().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get tensor data for '{}': {}", key, e))
        })?;

        let shape = tensor.tensor.shape();
        let shape_vec: Vec<usize> = shape.iter().copied().collect();

        let ndim = shape_vec.len() as u32;
        buffer.extend_from_slice(&ndim.to_le_bytes());

        for &dim in &shape_vec {
            buffer.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        for &val in &tensor_data {
            buffer.extend_from_slice(&val.to_le_bytes());
        }
    }

    Ok(())
}
