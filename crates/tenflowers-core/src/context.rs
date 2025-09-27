use crate::device::context::{DeviceContext, DEVICE_MANAGER};
use crate::{Device, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Execution context for TenfloweRS operations
pub struct Context {
    /// Default device for operations
    default_device: Device,
    /// Device contexts cache
    device_contexts: RwLock<HashMap<Device, Arc<dyn DeviceContext>>>,
    /// Context attributes
    attributes: RwLock<HashMap<String, String>>,
    /// Eager execution mode
    eager_mode: bool,
    /// Enable profiling
    profiling_enabled: bool,
}

impl Context {
    /// Create a new execution context
    pub fn new() -> Result<Self> {
        Ok(Self {
            default_device: Device::Cpu,
            device_contexts: RwLock::new(HashMap::new()),
            attributes: RwLock::new(HashMap::new()),
            eager_mode: true,
            profiling_enabled: false,
        })
    }

    /// Create a context with specific device
    pub fn with_device(device: Device) -> Result<Self> {
        let mut ctx = Self::new()?;
        ctx.default_device = device;
        Ok(ctx)
    }

    /// Get the default device
    pub fn default_device(&self) -> Device {
        self.default_device
    }

    /// Set the default device
    pub fn set_default_device(&mut self, device: Device) {
        self.default_device = device;
    }

    /// Check if eager execution is enabled
    pub fn is_eager(&self) -> bool {
        self.eager_mode
    }

    /// Set eager execution mode
    pub fn set_eager_mode(&mut self, eager: bool) {
        self.eager_mode = eager;
    }

    /// Enable/disable profiling
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    /// Get device context
    pub fn get_device_context(&self, device: &Device) -> Result<Arc<dyn DeviceContext>> {
        // Check cache first
        {
            let contexts = self.device_contexts.read().unwrap();
            if let Some(ctx) = contexts.get(device) {
                return Ok(Arc::clone(ctx));
            }
        }

        // Get from global manager
        let ctx = DEVICE_MANAGER.get_context(device)?;

        // Cache it
        {
            let mut contexts = self.device_contexts.write().unwrap();
            contexts.insert(*device, Arc::clone(&ctx));
        }

        Ok(ctx)
    }

    /// Set a context attribute
    pub fn set_attribute(&self, key: String, value: String) {
        let mut attrs = self.attributes.write().unwrap();
        attrs.insert(key, value);
    }

    /// Get a context attribute
    pub fn get_attribute(&self, key: &str) -> Option<String> {
        let attrs = self.attributes.read().unwrap();
        attrs.get(key).cloned()
    }
}

// Global context for eager execution
lazy_static::lazy_static! {
    static ref GLOBAL_CONTEXT: RwLock<Option<Arc<Context>>> = RwLock::new(None);
}

/// Get the current global context
pub fn get_context() -> Result<Arc<Context>> {
    let ctx_opt = GLOBAL_CONTEXT.read().unwrap();
    if let Some(ctx) = ctx_opt.as_ref() {
        Ok(Arc::clone(ctx))
    } else {
        drop(ctx_opt);

        // Create new context
        let ctx = Arc::new(Context::new()?);
        let mut ctx_opt = GLOBAL_CONTEXT.write().unwrap();
        *ctx_opt = Some(Arc::clone(&ctx));
        Ok(ctx)
    }
}

/// Set the global context
pub fn set_context(ctx: Arc<Context>) {
    let mut ctx_opt = GLOBAL_CONTEXT.write().unwrap();
    *ctx_opt = Some(ctx);
}

/// Context scope for temporary device placement
pub struct DeviceScope {
    previous_device: Device,
    context: Arc<Context>,
}

impl DeviceScope {
    /// Create a new device scope
    pub fn new(device: Device) -> Result<Self> {
        let ctx = get_context()?;
        let previous = ctx.default_device();

        // Clone context and modify
        let mut new_ctx = (*ctx).clone();
        new_ctx.set_default_device(device);
        set_context(Arc::new(new_ctx));

        Ok(Self {
            previous_device: previous,
            context: ctx,
        })
    }
}

impl Drop for DeviceScope {
    fn drop(&mut self) {
        // Restore previous context
        let mut restored_ctx = (*self.context).clone();
        restored_ctx.set_default_device(self.previous_device);
        set_context(Arc::new(restored_ctx));
    }
}

// Make Context cloneable for DeviceScope
impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            default_device: self.default_device,
            device_contexts: RwLock::new(HashMap::new()), // Don't clone cache
            attributes: RwLock::new(self.attributes.read().unwrap().clone()),
            eager_mode: self.eager_mode,
            profiling_enabled: self.profiling_enabled,
        }
    }
}

/// Macro for device scope
#[macro_export]
macro_rules! with_device {
    ($device:expr, $body:block) => {{
        let _scope = $crate::context::DeviceScope::new($device)?;
        $body
    }};
}
