//! Unit tests for the operation registry.

#[cfg(test)]
mod tests {
    use super::super::core::OP_REGISTRY;
    use super::super::types::{ArgDef, AttrValue, OpDef, OpRegistry, OpVersion};
    use std::collections::HashMap;

    #[test]
    fn test_op_registry() {
        let registry = OpRegistry::new();

        // Register test op
        let op_def = OpDef {
            name: "TestOp".to_string(),
            version: OpVersion::new(1, 0, 0),
            inputs: vec![],
            outputs: vec![],
            attrs: HashMap::new(),
            shape_fn: None,
            grad_fn: None,
            doc: "Test operation".to_string(),
            deprecated: false,
            deprecation_message: None,
        };

        registry
            .register_op(op_def.clone())
            .expect("test: operation should succeed");

        // Get op
        let retrieved = registry
            .get_op("TestOp")
            .expect("test: get_op should succeed");
        assert_eq!(retrieved.name, "TestOp");
        assert_eq!(retrieved.version, OpVersion::new(1, 0, 0));

        // List ops
        let ops = registry.list_ops();
        assert!(ops.contains(&"TestOp".to_string()));
    }

    #[test]
    fn test_builtin_ops() {
        // Check that built-in ops are registered
        assert!(OP_REGISTRY.get_op("Add").is_some());
        assert!(OP_REGISTRY.get_op("MatMul").is_some());
    }

    #[test]
    fn test_op_versioning() {
        let registry = OpRegistry::new();

        // Register multiple versions of the same operation
        let op_v1 = OpDef {
            name: "TestVersionOp".to_string(),
            version: OpVersion::new(1, 0, 0),
            inputs: vec![],
            outputs: vec![],
            attrs: HashMap::new(),
            shape_fn: None,
            grad_fn: None,
            doc: "Test operation v1.0.0".to_string(),
            deprecated: false,
            deprecation_message: None,
        };

        let op_v1_1 = OpDef {
            name: "TestVersionOp".to_string(),
            version: OpVersion::new(1, 1, 0),
            inputs: vec![],
            outputs: vec![],
            attrs: HashMap::new(),
            shape_fn: None,
            grad_fn: None,
            doc: "Test operation v1.1.0".to_string(),
            deprecated: false,
            deprecation_message: None,
        };

        let op_v2 = OpDef {
            name: "TestVersionOp".to_string(),
            version: OpVersion::new(2, 0, 0),
            inputs: vec![],
            outputs: vec![],
            attrs: HashMap::new(),
            shape_fn: None,
            grad_fn: None,
            doc: "Test operation v2.0.0".to_string(),
            deprecated: false,
            deprecation_message: None,
        };

        registry
            .register_op(op_v1)
            .expect("test: register_op should succeed");
        registry
            .register_op(op_v1_1)
            .expect("test: register_op should succeed");
        registry
            .register_op(op_v2)
            .expect("test: register_op should succeed");

        // Test latest version retrieval
        let latest = registry
            .get_op("TestVersionOp")
            .expect("test: get_op should succeed");
        assert_eq!(latest.version, OpVersion::new(2, 0, 0));

        // Test specific version retrieval
        let v1 = registry
            .get_op_version("TestVersionOp", &OpVersion::new(1, 0, 0))
            .expect("operation should succeed");
        assert_eq!(v1.version, OpVersion::new(1, 0, 0));

        // Test compatible version resolution
        let compatible = registry
            .get_op_compatible("TestVersionOp", &OpVersion::new(1, 0, 0))
            .expect("operation should succeed");
        assert_eq!(compatible.version, OpVersion::new(1, 1, 0)); // Should get highest compatible

        // Test cross-major version incompatibility
        let compatible_v2 = registry
            .get_op_compatible("TestVersionOp", &OpVersion::new(2, 0, 0))
            .expect("operation should succeed");
        assert_eq!(compatible_v2.version, OpVersion::new(2, 0, 0));

        // Test version listing
        let versions = registry.list_op_versions("TestVersionOp");
        assert_eq!(versions.len(), 3);
        assert!(versions.contains(&OpVersion::new(1, 0, 0)));
        assert!(versions.contains(&OpVersion::new(1, 1, 0)));
        assert!(versions.contains(&OpVersion::new(2, 0, 0)));
    }

    #[test]
    fn test_version_compatibility() {
        let v1_0_0 = OpVersion::new(1, 0, 0);
        let v1_1_0 = OpVersion::new(1, 1, 0);
        let v1_2_0 = OpVersion::new(1, 2, 0);
        let v2_0_0 = OpVersion::new(2, 0, 0);

        // Test compatibility within same major version
        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(v1_2_0.is_compatible_with(&v1_0_0));
        assert!(v1_2_0.is_compatible_with(&v1_1_0));

        // Test incompatibility with lower minor versions
        assert!(!v1_0_0.is_compatible_with(&v1_1_0));

        // Test incompatibility across major versions
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v2_0_0));
    }

    #[test]
    fn test_deprecated_operations() {
        let registry = OpRegistry::new();

        // Register a deprecated operation
        let deprecated_op = OpDef {
            name: "DeprecatedOp".to_string(),
            version: OpVersion::new(1, 0, 0),
            inputs: vec![],
            outputs: vec![],
            attrs: HashMap::new(),
            shape_fn: None,
            grad_fn: None,
            doc: "Deprecated operation".to_string(),
            deprecated: true,
            deprecation_message: Some("Use NewOp instead".to_string()),
        };

        registry
            .register_op(deprecated_op)
            .expect("test: register_op should succeed");

        let retrieved = registry
            .get_op("DeprecatedOp")
            .expect("test: get_op should succeed");
        assert!(retrieved.deprecated);
        assert_eq!(
            retrieved.deprecation_message,
            Some("Use NewOp instead".to_string())
        );
    }
}
