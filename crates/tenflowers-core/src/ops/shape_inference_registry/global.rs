//! Global registry access: OnceLock-based singleton for shape inference.

use super::registry::ShapeInferenceRegistry;
use std::sync::OnceLock;

static GLOBAL_REGISTRY: OnceLock<ShapeInferenceRegistry> = OnceLock::new();

/// Get the global shape inference registry.
///
/// Initializes the registry on first access (lazy initialization).
pub fn get_registry() -> &'static ShapeInferenceRegistry {
    GLOBAL_REGISTRY.get_or_init(ShapeInferenceRegistry::new)
}

/// Initialize the global registry (called automatically on first access).
///
/// This function can be called explicitly to eagerly initialize the registry
/// during application startup, avoiding first-call latency in hot paths.
pub fn initialize_registry() {
    let _ = get_registry();
}
