//! Neural network hook system for debugging and monitoring
//!
//! This module provides a comprehensive hook system that allows users to register
//! callbacks for forward and backward passes through neural network layers.

use crate::tensor_ops::PyTensor;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Hook function type for forward hooks
pub type ForwardHook = Box<dyn Fn(&PyTensor, &PyTensor) -> PyResult<()> + Send + Sync>;

/// Hook function type for backward hooks
pub type BackwardHook = Box<dyn Fn(&PyTensor) -> PyResult<()> + Send + Sync>;

/// Hook handle for tracking registered hooks
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyHookHandle {
    id: usize,
    hook_type: String,
}

#[pymethods]
impl PyHookHandle {
    /// Get the hook ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the hook type
    pub fn hook_type(&self) -> String {
        self.hook_type.clone()
    }
}

impl PyHookHandle {
    pub fn new(id: usize, hook_type: String) -> Self {
        Self { id, hook_type }
    }
}

/// Hook management functionality for neural network layers
#[derive(Debug)]
pub struct HookManager {
    pub forward_hooks: Arc<std::sync::RwLock<HashMap<usize, PyObject>>>,
    pub backward_hooks: Arc<std::sync::RwLock<HashMap<usize, PyObject>>>,
    pub next_hook_id: Arc<std::sync::RwLock<usize>>,
}

impl HookManager {
    /// Create a new hook manager
    pub fn new() -> Self {
        Self {
            forward_hooks: Arc::new(std::sync::RwLock::new(HashMap::new())),
            backward_hooks: Arc::new(std::sync::RwLock::new(HashMap::new())),
            next_hook_id: Arc::new(std::sync::RwLock::new(0)),
        }
    }

    /// Register a forward hook
    pub fn register_forward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        let mut hooks = self.forward_hooks.write().unwrap();
        let mut next_id = self.next_hook_id.write().unwrap();

        let id = *next_id;
        *next_id += 1;

        hooks.insert(id, hook);

        Ok(PyHookHandle::new(id, "forward".to_string()))
    }

    /// Register a backward hook
    pub fn register_backward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        let mut hooks = self.backward_hooks.write().unwrap();
        let mut next_id = self.next_hook_id.write().unwrap();

        let id = *next_id;
        *next_id += 1;

        hooks.insert(id, hook);

        Ok(PyHookHandle::new(id, "backward".to_string()))
    }

    /// Remove a hook by ID
    pub fn remove_hook(&self, handle: &PyHookHandle) -> PyResult<()> {
        match handle.hook_type.as_str() {
            "forward" => {
                let mut hooks = self.forward_hooks.write().unwrap();
                hooks.remove(&handle.id);
            }
            "backward" => {
                let mut hooks = self.backward_hooks.write().unwrap();
                hooks.remove(&handle.id);
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid hook type",
                ));
            }
        }
        Ok(())
    }

    /// Clear all hooks
    pub fn clear_hooks(&self) {
        self.forward_hooks.write().unwrap().clear();
        self.backward_hooks.write().unwrap().clear();
    }

    /// Get hook count
    pub fn hook_count(&self) -> (usize, usize) {
        let forward_count = self.forward_hooks.read().unwrap().len();
        let backward_count = self.backward_hooks.read().unwrap().len();
        (forward_count, backward_count)
    }

    /// Execute forward hooks
    pub fn execute_forward_hooks(&self, input: &PyTensor, output: &PyTensor) -> PyResult<()> {
        let hooks = self.forward_hooks.read().unwrap();

        Python::with_gil(|py| {
            for (_id, hook) in hooks.iter() {
                if let Err(e) = hook.call1(py, (input.clone(), output.clone())) {
                    eprintln!("Warning: Forward hook failed: {}", e);
                    // Continue execution but log the error
                    e.print(py);
                }
            }
        });

        Ok(())
    }

    /// Execute backward hooks
    pub fn execute_backward_hooks(&self, grad_input: &PyTensor) -> PyResult<()> {
        let hooks = self.backward_hooks.read().unwrap();

        Python::with_gil(|py| {
            for (_id, hook) in hooks.iter() {
                if let Err(e) = hook.call1(py, (grad_input.clone(),)) {
                    eprintln!("Warning: Backward hook failed: {}", e);
                    // Continue execution but log the error
                    e.print(py);
                }
            }
        });

        Ok(())
    }
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HookManager {
    fn clone(&self) -> Self {
        // Create a new hook manager without copying hooks
        // This is appropriate since hooks are typically layer-specific
        Self::new()
    }
}

/// Global hook registry for cross-layer hook management
#[pyclass]
pub struct PyGlobalHookRegistry {
    forward_hooks: Arc<std::sync::RwLock<HashMap<usize, PyObject>>>,
    backward_hooks: Arc<std::sync::RwLock<HashMap<usize, PyObject>>>,
    next_hook_id: Arc<std::sync::RwLock<usize>>,
}

impl Default for PyGlobalHookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyGlobalHookRegistry {
    #[new]
    pub fn new() -> Self {
        Self {
            forward_hooks: Arc::new(std::sync::RwLock::new(HashMap::new())),
            backward_hooks: Arc::new(std::sync::RwLock::new(HashMap::new())),
            next_hook_id: Arc::new(std::sync::RwLock::new(0)),
        }
    }

    /// Register a global forward hook
    pub fn register_forward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        let mut hooks = self.forward_hooks.write().unwrap();
        let mut next_id = self.next_hook_id.write().unwrap();

        let id = *next_id;
        *next_id += 1;

        hooks.insert(id, hook);

        Ok(PyHookHandle::new(id, "forward".to_string()))
    }

    /// Register a global backward hook
    pub fn register_backward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        let mut hooks = self.backward_hooks.write().unwrap();
        let mut next_id = self.next_hook_id.write().unwrap();

        let id = *next_id;
        *next_id += 1;

        hooks.insert(id, hook);

        Ok(PyHookHandle::new(id, "backward".to_string()))
    }

    /// Remove a hook by ID
    pub fn remove_hook(&self, handle: &PyHookHandle) -> PyResult<()> {
        match handle.hook_type.as_str() {
            "forward" => {
                let mut hooks = self.forward_hooks.write().unwrap();
                hooks.remove(&handle.id);
            }
            "backward" => {
                let mut hooks = self.backward_hooks.write().unwrap();
                hooks.remove(&handle.id);
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid hook type",
                ));
            }
        }
        Ok(())
    }

    /// Clear all global hooks
    pub fn clear_hooks(&self) {
        self.forward_hooks.write().unwrap().clear();
        self.backward_hooks.write().unwrap().clear();
    }

    /// Get global hook count
    pub fn hook_count(&self) -> (usize, usize) {
        let forward_count = self.forward_hooks.read().unwrap().len();
        let backward_count = self.backward_hooks.read().unwrap().len();
        (forward_count, backward_count)
    }

    /// Execute global forward hooks
    pub fn execute_forward_hooks(&self, input: &PyTensor, output: &PyTensor) -> PyResult<()> {
        let hooks = self.forward_hooks.read().unwrap();

        Python::with_gil(|py| {
            for (_id, hook) in hooks.iter() {
                if let Err(e) = hook.call1(py, (input.clone(), output.clone())) {
                    eprintln!("Warning: Global forward hook failed: {}", e);
                    e.print(py);
                }
            }
        });

        Ok(())
    }

    /// Execute global backward hooks
    pub fn execute_backward_hooks(&self, grad_input: &PyTensor) -> PyResult<()> {
        let hooks = self.backward_hooks.read().unwrap();

        Python::with_gil(|py| {
            for (_id, hook) in hooks.iter() {
                if let Err(e) = hook.call1(py, (grad_input.clone(),)) {
                    eprintln!("Warning: Global backward hook failed: {}", e);
                    e.print(py);
                }
            }
        });

        Ok(())
    }

    /// List all registered hooks with metadata
    pub fn list_hooks(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        let result = PyDict::new(py);

        let forward_hooks = self.forward_hooks.read().unwrap();
        let backward_hooks = self.backward_hooks.read().unwrap();

        result.set_item("forward_count", forward_hooks.len())?;
        result.set_item("backward_count", backward_hooks.len())?;
        result.set_item("total_count", forward_hooks.len() + backward_hooks.len())?;

        Ok(result.into())
    }
}
