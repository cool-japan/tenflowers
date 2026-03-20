//! Tests for the shape inference registry.

#[cfg(test)]
mod tests {
    use super::super::global::get_registry;
    use super::super::registry::ShapeInferenceRegistry;
    use super::super::types::{MetadataValue, OperationCategory};
    use crate::Shape;
    use std::collections::HashMap;

    #[test]
    fn test_registry_creation() {
        let registry = ShapeInferenceRegistry::new();
        let ops = registry.list_operations();
        assert!(!ops.is_empty(), "Registry should have builtin operations");
        assert!(ops.contains(&"add".to_string()));
        assert!(ops.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_binary_elementwise_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[2, 3]);
        let metadata = HashMap::new();

        let result = registry.infer("add", &[shape1, shape2], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[2, 3]
        );
    }

    #[test]
    fn test_broadcasting_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 1, 3]);
        let shape2 = Shape::from_slice(&[1, 4, 3]);
        let metadata = HashMap::new();

        let result = registry.infer("mul", &[shape1, shape2], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[2, 4, 3]
        );
    }

    #[test]
    fn test_matmul_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[3, 4]);
        let mut metadata = HashMap::new();
        metadata.insert("transpose_a".to_string(), MetadataValue::Bool(false));
        metadata.insert("transpose_b".to_string(), MetadataValue::Bool(false));

        let result = registry.infer("matmul", &[shape1, shape2], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[2, 4]
        );
    }

    #[test]
    fn test_reduction_inference() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);

        // Reduce on axis 1, no keepdims
        let mut metadata = HashMap::new();
        metadata.insert("axis".to_string(), MetadataValue::Int(1));
        metadata.insert("keepdims".to_string(), MetadataValue::Bool(false));

        let result = registry.infer("sum", std::slice::from_ref(&shape), &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[2, 4]
        );

        // Reduce on axis 1, with keepdims
        metadata.insert("keepdims".to_string(), MetadataValue::Bool(true));
        let result = registry.infer("sum", std::slice::from_ref(&shape), &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[2, 1, 4]
        );

        // Reduce all dimensions
        let metadata_all = HashMap::new();
        let result = registry.infer("mean", &[shape], &metadata_all);
        assert!(result.is_ok());
        let result_shape = result.expect("test: operation should succeed");
        assert_eq!(result_shape.dims().len(), 0); // Scalar shape has no dimensions
    }

    #[test]
    fn test_reshape_inference() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);
        let mut metadata = HashMap::new();
        metadata.insert("shape".to_string(), MetadataValue::IntVec(vec![6, 4]));

        let result = registry.infer("reshape", &[shape], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[6, 4]
        );
    }

    #[test]
    fn test_reshape_with_infer() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);
        let mut metadata = HashMap::new();
        metadata.insert("shape".to_string(), MetadataValue::IntVec(vec![-1, 4]));

        let result = registry.infer("reshape", &[shape], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[6, 4]
        );
    }

    #[test]
    fn test_transpose_inference() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);
        let metadata = HashMap::new();

        let result = registry.infer("transpose", &[shape], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[2, 4, 3]
        );
    }

    #[test]
    fn test_concat_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[2, 4]);
        let shape3 = Shape::from_slice(&[2, 5]);
        let mut metadata = HashMap::new();
        metadata.insert("axis".to_string(), MetadataValue::Int(1));

        let result = registry.infer("concat", &[shape1, shape2, shape3], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[2, 12]
        );
    }

    #[test]
    fn test_stack_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[2, 3]);
        let shape3 = Shape::from_slice(&[2, 3]);
        let mut metadata = HashMap::new();
        metadata.insert("axis".to_string(), MetadataValue::Int(0));

        let result = registry.infer("stack", &[shape1, shape2, shape3], &metadata);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: operation should succeed").dims(),
            &[3, 2, 3]
        );
    }

    #[test]
    fn test_operations_by_category() {
        let registry = get_registry();
        let binary_ops = registry.operations_by_category(OperationCategory::BinaryElementwise);
        assert!(binary_ops.contains(&"add".to_string()));
        assert!(binary_ops.contains(&"mul".to_string()));

        let matrix_ops = registry.operations_by_category(OperationCategory::MatrixOps);
        assert!(matrix_ops.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_error_for_unknown_operation() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3]);
        let metadata = HashMap::new();

        let result = registry.infer("unknown_op", &[shape], &metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_standardization() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[4, 5]);
        let metadata = HashMap::new();

        // Should produce standardized error message
        let result = registry.infer("add", &[shape1, shape2], &metadata);
        assert!(result.is_err());
        let err = result.expect_err("test: incompatible shapes should produce an error");
        let err_msg = format!("{}", err);
        // Error should mention broadcasting
        assert!(err_msg.contains("broadcast") || err_msg.contains("Broadcast"));
    }
}
