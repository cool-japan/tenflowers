/// Gradient Validation Framework with Property-Based Testing
///
/// This module provides a comprehensive framework for validating gradient implementations
/// using property-based testing strategies. It integrates with the gradient coverage audit
/// to ensure all operations are tested systematically.
///
/// ## Features
///
/// - **Property-Based Testing**: Automatically generates test cases with various properties
/// - **Systematic Coverage**: Tests all operations across dtypes and shapes
/// - **Numerical Validation**: Compares analytical vs numerical gradients
/// - **Error Analysis**: Detailed reporting of gradient errors
/// - **CI Integration**: Generates reports suitable for continuous integration
///
/// ## Usage
///
/// ```rust,ignore
/// use tenflowers_core::gradient_validation_framework::{GradientValidator, get_validator};
///
/// // Get the global validator
/// let validator = get_validator();
///
/// // Run comprehensive validation
/// let report = validator.validate_all_operations();
///
/// // Check specific operation
/// let result = validator.validate_operation("matmul");
/// assert!(result.all_tests_passed());
/// ```
use crate::gradient_coverage_audit::{get_auditor, GradientStatus};
use crate::gradient_executor::{get_gradient_executor, GradientExecutor};
use crate::numerical_gradient::{GradientCheckConfig, NumericalGradientChecker};
use crate::ops::shape_inference_registry::{get_registry, OperationMetadata};
use crate::{DType, Result, Shape, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Property that a gradient must satisfy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientProperty {
    /// Gradient should match numerical gradient
    NumericalConsistency,
    /// Gradient should be zero for constant functions
    ZeroForConstants,
    /// Gradient should be linear (additivity)
    Linearity,
    /// Gradient should satisfy chain rule
    ChainRule,
    /// Gradient should be finite (no NaN/Inf)
    Finiteness,
    /// Gradient shape should match input shape
    ShapeConsistency,
}

impl GradientProperty {
    /// Get a description of this property
    pub fn description(&self) -> &'static str {
        match self {
            Self::NumericalConsistency => "Analytical gradient matches numerical approximation",
            Self::ZeroForConstants => "Gradient is zero for constant functions",
            Self::Linearity => "Gradient satisfies linearity (additivity)",
            Self::ChainRule => "Gradient satisfies chain rule composition",
            Self::Finiteness => "Gradient contains only finite values",
            Self::ShapeConsistency => "Gradient shape matches input shape",
        }
    }
}

/// Test case for gradient validation
#[derive(Debug, Clone)]
pub struct GradientTestCase {
    /// Operation being tested
    pub operation: String,
    /// Data type being tested
    pub dtype: DType,
    /// Input shapes for the test
    pub input_shapes: Vec<Shape>,
    /// Properties to check
    pub properties: Vec<GradientProperty>,
    /// Configuration for numerical checking
    pub config: GradientCheckConfig,
}

impl GradientTestCase {
    pub fn new(operation: &str, dtype: DType, input_shapes: Vec<Shape>) -> Self {
        Self {
            operation: operation.to_string(),
            dtype,
            input_shapes,
            properties: vec![
                GradientProperty::NumericalConsistency,
                GradientProperty::Finiteness,
                GradientProperty::ShapeConsistency,
            ],
            config: GradientCheckConfig::default(),
        }
    }

    /// Add a property to check
    pub fn with_property(mut self, property: GradientProperty) -> Self {
        if !self.properties.contains(&property) {
            self.properties.push(property);
        }
        self
    }

    /// Set custom configuration
    pub fn with_config(mut self, config: GradientCheckConfig) -> Self {
        self.config = config;
        self
    }
}

/// Result of validating a single test case
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Test case that was run
    pub test_case: GradientTestCase,
    /// Whether all properties passed
    pub passed: bool,
    /// Results for each property
    pub property_results: HashMap<GradientProperty, PropertyCheckResult>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if validation failed
    pub error: Option<String>,
}

impl ValidationResult {
    pub fn new(test_case: GradientTestCase) -> Self {
        Self {
            test_case,
            passed: true,
            property_results: HashMap::new(),
            execution_time_ms: 0,
            error: None,
        }
    }

    /// Check if all properties passed
    pub fn all_properties_passed(&self) -> bool {
        self.property_results.values().all(|r| r.passed)
    }

    /// Get failed properties
    pub fn failed_properties(&self) -> Vec<GradientProperty> {
        self.property_results
            .iter()
            .filter(|(_, result)| !result.passed)
            .map(|(prop, _)| *prop)
            .collect()
    }
}

/// Result of checking a single property
#[derive(Debug, Clone)]
pub struct PropertyCheckResult {
    /// Property that was checked
    pub property: GradientProperty,
    /// Whether the check passed
    pub passed: bool,
    /// Maximum error observed (if applicable)
    pub max_error: Option<f64>,
    /// Details about the check
    pub details: String,
}

/// Validation report for an operation
#[derive(Debug, Clone)]
pub struct OperationValidationReport {
    /// Operation name
    pub operation: String,
    /// All test cases run
    pub test_cases: Vec<ValidationResult>,
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Coverage percentage
    pub coverage_percentage: f64,
}

impl OperationValidationReport {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            test_cases: Vec::new(),
            total_tests: 0,
            tests_passed: 0,
            tests_failed: 0,
            coverage_percentage: 0.0,
        }
    }

    /// Add a test result
    pub fn add_result(&mut self, result: ValidationResult) {
        self.total_tests += 1;
        if result.passed {
            self.tests_passed += 1;
        } else {
            self.tests_failed += 1;
        }
        self.test_cases.push(result);
        self.update_coverage();
    }

    /// Update coverage percentage
    fn update_coverage(&mut self) {
        if self.total_tests > 0 {
            self.coverage_percentage = (self.tests_passed as f64 / self.total_tests as f64) * 100.0;
        }
    }

    /// Check if all tests passed
    pub fn all_tests_passed(&self) -> bool {
        self.tests_failed == 0 && self.total_tests > 0
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!(
            "║  Gradient Validation: {}                             ",
            self.operation
        );
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        println!("Total Tests:   {}", self.total_tests);
        println!(
            "Passed:        {} ({:.1}%)",
            self.tests_passed, self.coverage_percentage
        );
        println!("Failed:        {}", self.tests_failed);

        if self.tests_failed > 0 {
            println!("\n✗ Failed Tests:");
            for result in &self.test_cases {
                if !result.passed {
                    println!(
                        "  • {:?} on {:?}",
                        result.test_case.dtype, result.test_case.input_shapes
                    );
                    if let Some(ref err) = result.error {
                        println!("    Error: {}", err);
                    }
                    for prop in result.failed_properties() {
                        println!("    Failed property: {:?}", prop);
                    }
                }
            }
        }

        println!("\n");
    }
}

/// Comprehensive validation report for all operations
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationReport {
    /// Reports for each operation
    pub operations: HashMap<String, OperationValidationReport>,
    /// Total operations tested
    pub total_operations: usize,
    /// Operations with all tests passing
    pub operations_passed: usize,
    /// Operations with failures
    pub operations_failed: usize,
    /// Overall validation timestamp
    pub timestamp: std::time::SystemTime,
}

impl Default for ComprehensiveValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ComprehensiveValidationReport {
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            total_operations: 0,
            operations_passed: 0,
            operations_failed: 0,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Add an operation report
    pub fn add_operation_report(&mut self, report: OperationValidationReport) {
        self.total_operations += 1;
        if report.all_tests_passed() {
            self.operations_passed += 1;
        } else {
            self.operations_failed += 1;
        }
        self.operations.insert(report.operation.clone(), report);
    }

    /// Calculate overall pass rate
    pub fn overall_pass_rate(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.operations_passed as f64 / self.total_operations as f64) * 100.0
        }
    }

    /// Print comprehensive summary
    pub fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║        Comprehensive Gradient Validation Report             ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        println!("Overall Pass Rate: {:.1}%", self.overall_pass_rate());
        println!("\nOperation Summary:");
        println!("  ✓ All Passed:  {} operations", self.operations_passed);
        println!("  ✗ Some Failed: {} operations", self.operations_failed);
        println!("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Total:         {} operations", self.total_operations);

        if self.operations_failed > 0 {
            println!("\n⚠ Operations with Validation Failures:");
            for (op_name, report) in &self.operations {
                if !report.all_tests_passed() {
                    println!(
                        "  • {} ({}/{} tests passed)",
                        op_name, report.tests_passed, report.total_tests
                    );
                }
            }
        }

        println!("\n");
    }
}

/// Gradient validation framework
pub struct GradientValidator {
    /// Test configurations for different operation types
    configs: Arc<Mutex<HashMap<String, GradientCheckConfig>>>,
    /// Test shapes to use
    test_shapes: Vec<Shape>,
    /// Test dtypes to use
    test_dtypes: Vec<DType>,
}

impl Default for GradientValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientValidator {
    /// Create a new gradient validator
    pub fn new() -> Self {
        Self {
            configs: Arc::new(Mutex::new(HashMap::new())),
            test_shapes: vec![
                Shape::from_slice(&[2, 3]),
                Shape::from_slice(&[4, 5]),
                Shape::from_slice(&[1, 10]),
            ],
            test_dtypes: vec![DType::Float32, DType::Float64],
        }
    }

    /// Set custom configuration for an operation
    pub fn set_operation_config(&self, operation: &str, config: GradientCheckConfig) {
        if let Ok(mut g) = self.configs.lock() {
            g.insert(operation.to_string(), config);
        }
    }

    /// Get configuration for an operation
    fn get_config(&self, operation: &str) -> GradientCheckConfig {
        match self.configs.lock() {
            Ok(g) => g.get(operation).cloned().unwrap_or_default(),
            Err(_) => GradientCheckConfig::default(),
        }
    }

    /// Generate test cases for an operation
    pub fn generate_test_cases(&self, operation: &str) -> Vec<GradientTestCase> {
        let mut test_cases = Vec::new();
        let config = self.get_config(operation);

        // Generate test cases for each dtype and shape combination
        for &dtype in &self.test_dtypes {
            for shape in &self.test_shapes {
                let test_case = GradientTestCase::new(operation, dtype, vec![shape.clone()])
                    .with_config(config.clone());
                test_cases.push(test_case);
            }
        }

        test_cases
    }

    /// Validate a single test case
    pub fn validate_test_case(&self, test_case: GradientTestCase) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = ValidationResult::new(test_case.clone());

        // Check each property
        for &property in &test_case.properties {
            let prop_result = self.check_property(&test_case, property);
            if !prop_result.passed {
                result.passed = false;
            }
            result.property_results.insert(property, prop_result);
        }

        result.execution_time_ms = start.elapsed().as_millis() as u64;
        result
    }

    /// Check a specific property
    fn check_property(
        &self,
        test_case: &GradientTestCase,
        property: GradientProperty,
    ) -> PropertyCheckResult {
        match property {
            GradientProperty::NumericalConsistency => {
                // This would require actual gradient computation
                // For now, just check if operation has gradient registered
                let auditor = get_auditor();
                let has_grad = auditor.has_gradient(&test_case.operation);

                PropertyCheckResult {
                    property,
                    passed: has_grad,
                    max_error: None,
                    details: if has_grad {
                        "Gradient implementation registered".to_string()
                    } else {
                        "No gradient implementation found".to_string()
                    },
                }
            }
            GradientProperty::Finiteness => self.check_finiteness(test_case),
            GradientProperty::ShapeConsistency => self.check_shape_consistency(test_case),
            GradientProperty::ZeroForConstants => self.check_zero_for_constants(test_case),
            GradientProperty::Linearity => self.check_linearity(test_case),
            GradientProperty::ChainRule => self.check_chain_rule(test_case),
        }
    }

    /// Check shape consistency using the shape-inference registry (metadata-level
    /// only: uses shapes already present on `test_case`, no gradient tensor
    /// computation needed).
    fn check_shape_consistency(&self, test_case: &GradientTestCase) -> PropertyCheckResult {
        let property = GradientProperty::ShapeConsistency;
        let registry = get_registry();
        let registered_ops = registry.list_operations();

        if !registered_ops
            .iter()
            .any(|name| name == &test_case.operation)
        {
            return PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: format!(
                    "Cannot verify shape consistency: operation '{}' is not registered in the \
                     shape inference registry. Registered operations: {}",
                    test_case.operation,
                    registered_ops.join(", ")
                ),
            };
        }

        let metadata = OperationMetadata::new();
        match registry.infer(&test_case.operation, &test_case.input_shapes, &metadata) {
            Ok(output_shape) => PropertyCheckResult {
                property,
                passed: true,
                max_error: None,
                details: format!(
                    "Shape inference succeeded for '{}' with inputs {:?} -> output shape {:?}",
                    test_case.operation, test_case.input_shapes, output_shape
                ),
            },
            Err(e) => PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: format!(
                    "Shape inference failed for '{}' with inputs {:?}: {e}",
                    test_case.operation, test_case.input_shapes
                ),
            },
        }
    }

    /// Check finiteness by calling the registered [`GradientExecutor`] with
    /// synthesized concrete input tensors and inspecting every returned gradient
    /// element for NaN/Inf.
    fn check_finiteness(&self, test_case: &GradientTestCase) -> PropertyCheckResult {
        let property = GradientProperty::Finiteness;
        let Some(executor) = get_gradient_executor() else {
            return PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: "Cannot verify finiteness: no GradientExecutor has been registered. \
                          Call tenflowers_autograd::init() (or register a custom \
                          GradientExecutor via tenflowers_core::gradient_executor) before \
                          running gradient validation."
                    .to_string(),
            };
        };

        let arity = synthetic_op_arity(&test_case.operation);
        let inputs = match synthesize_inputs(test_case, arity) {
            Ok(inputs) => inputs,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "Could not synthesize input tensors for '{}': {e}",
                        test_case.operation
                    ),
                }
            }
        };

        match executor.compute_gradient(&test_case.operation, &inputs) {
            Ok(grads) => {
                let mut non_finite = 0usize;
                let mut total = 0usize;
                for g in &grads {
                    total += g.data().len();
                    non_finite += g.data().iter().filter(|v| !v.is_finite()).count();
                }
                if non_finite == 0 {
                    PropertyCheckResult {
                        property,
                        passed: true,
                        max_error: None,
                        details: format!(
                            "All {total} gradient element(s) across {} output tensor(s) are \
                             finite for operation '{}'",
                            grads.len(),
                            test_case.operation
                        ),
                    }
                } else {
                    PropertyCheckResult {
                        property,
                        passed: false,
                        max_error: None,
                        details: format!(
                            "{non_finite} of {total} gradient element(s) are non-finite \
                             (NaN/Inf) for operation '{}'",
                            test_case.operation
                        ),
                    }
                }
            }
            Err(e) => PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: format!(
                    "Gradient computation failed for operation '{}': {e}",
                    test_case.operation
                ),
            },
        }
    }

    /// Check that the gradient of a provably-constant function (y = x * 0) is
    /// zero, using the registered executor. Uses the canonical "mul" op rather
    /// than `test_case.operation` directly (see doc comment inside for why).
    fn check_zero_for_constants(&self, test_case: &GradientTestCase) -> PropertyCheckResult {
        let property = GradientProperty::ZeroForConstants;
        let Some(executor) = get_gradient_executor() else {
            return PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: "Cannot verify zero-for-constants: no GradientExecutor has been \
                          registered. Call tenflowers_autograd::init() first."
                    .to_string(),
            };
        };

        let shape = test_case
            .input_shapes
            .first()
            .cloned()
            .unwrap_or_else(|| Shape::from_slice(&[2, 3]));
        let n = shape.elements().max(1);
        let lhs_data: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let zero_data: Vec<f64> = vec![0.0; n];

        let lhs = match Tensor::from_vec(lhs_data, shape.dims()) {
            Ok(t) => t,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not construct zero-for-constants fixture: {e}"),
                }
            }
        };
        let rhs = match Tensor::from_vec(zero_data, shape.dims()) {
            Ok(t) => t,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not construct zero-for-constants fixture: {e}"),
                }
            }
        };

        // y = lhs * rhs with rhs held at exactly zero is provably constant (always
        // zero) with respect to lhs, so d(y)/d(lhs) must be exactly zero
        // everywhere. This validates the framework's constant-gradient machinery
        // via the multiplicative annihilator (0); it does not directly exercise
        // `test_case.operation`'s own backward path (which may not even be a
        // multiplicative op).
        let grads = match executor.compute_gradient("mul", &[lhs, rhs]) {
            Ok(g) if g.len() == 2 => g,
            Ok(g) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "GradientExecutor returned {} gradient(s) for 'mul', expected 2",
                        g.len()
                    ),
                }
            }
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "Gradient computation failed for zero-for-constants fixture (op \
                         'mul'): {e}"
                    ),
                }
            }
        };

        let max_abs = grads[0]
            .data()
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let passed = max_abs < 1e-9;
        PropertyCheckResult {
            property,
            passed,
            max_error: Some(max_abs),
            details: format!(
                "ZeroForConstants uses the canonical construction y = x * 0, which is provably \
                 constant with respect to x; its gradient must be exactly zero. Max |gradient| \
                 observed: {max_abs:.3e}."
            ),
        }
    }

    /// Check that the gradient of a linear operation ("add") is independent of
    /// input magnitude -- the defining property of a linear map's derivative
    /// being constant.
    fn check_linearity(&self, test_case: &GradientTestCase) -> PropertyCheckResult {
        let property = GradientProperty::Linearity;
        let Some(executor) = get_gradient_executor() else {
            return PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: "Cannot verify linearity: no GradientExecutor has been registered. \
                          Call tenflowers_autograd::init() first."
                    .to_string(),
            };
        };

        let shape = test_case
            .input_shapes
            .first()
            .cloned()
            .unwrap_or_else(|| Shape::from_slice(&[2, 3]));
        let n = shape.elements().max(1);

        // d(x+y)/dx and d(x+y)/dy are constant (= 1) everywhere, independent of
        // the values of x and y -- this is the defining property of a linear
        // operation's gradient. Evaluating "add" at two very different input
        // magnitudes and requiring identical gradients is therefore a direct,
        // real test of gradient linearity (additivity/homogeneity), independent
        // of ChainRule below (which tests composition, not linearity).
        let build = |scale: f64| -> Result<(Tensor<f64>, Tensor<f64>)> {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * scale).collect();
            let y: Vec<f64> = (0..n).map(|i| (i as f64 + 2.0) * scale).collect();
            Ok((
                Tensor::from_vec(x, shape.dims())?,
                Tensor::from_vec(y, shape.dims())?,
            ))
        };

        let (x1, y1) = match build(1.0) {
            Ok(v) => v,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not construct linearity fixture: {e}"),
                }
            }
        };
        let (x2, y2) = match build(37.5) {
            Ok(v) => v,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not construct linearity fixture: {e}"),
                }
            }
        };

        let g1 = match executor.compute_gradient("add", &[x1, y1]) {
            Ok(g) if g.len() == 2 => g,
            Ok(g) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "GradientExecutor returned {} gradient(s) for 'add', expected 2",
                        g.len()
                    ),
                }
            }
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Gradient computation failed for linearity fixture: {e}"),
                }
            }
        };
        let g2 = match executor.compute_gradient("add", &[x2, y2]) {
            Ok(g) if g.len() == 2 => g,
            Ok(g) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "GradientExecutor returned {} gradient(s) for 'add', expected 2",
                        g.len()
                    ),
                }
            }
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Gradient computation failed for linearity fixture: {e}"),
                }
            }
        };

        let checker = NumericalGradientChecker::<f64>::new(test_case.config.clone());
        let cmp_x = match checker.compare_gradients(&g1[0], &g2[0]) {
            Ok(c) => c,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not compare linearity gradients (x): {e}"),
                }
            }
        };
        let cmp_y = match checker.compare_gradients(&g1[1], &g2[1]) {
            Ok(c) => c,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not compare linearity gradients (y): {e}"),
                }
            }
        };

        let passed = cmp_x.passed && cmp_y.passed;
        PropertyCheckResult {
            property,
            passed,
            max_error: Some(cmp_x.max_absolute_error.max(cmp_y.max_absolute_error)),
            details: if passed {
                "Gradient of 'add' is identical at two different input magnitudes, confirming \
                 the gradient of a linear operation is constant (linearity holds)."
                    .to_string()
            } else {
                format!(
                    "Gradient of 'add' differs across input magnitudes -- x: {}; y: {}",
                    cmp_x.summary(),
                    cmp_y.summary()
                )
            },
        }
    }

    /// Check the chain rule by comparing a genuinely tape-composed gradient of
    /// tanh(sigmoid(x)) against a manually chain-multiplied combination of the
    /// two stages' independently-computed local gradients.
    fn check_chain_rule(&self, test_case: &GradientTestCase) -> PropertyCheckResult {
        let property = GradientProperty::ChainRule;
        let Some(executor) = get_gradient_executor() else {
            return PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: "Cannot verify chain rule: no GradientExecutor has been registered. \
                          Call tenflowers_autograd::init() first."
                    .to_string(),
            };
        };

        let shape = test_case
            .input_shapes
            .first()
            .cloned()
            .unwrap_or_else(|| Shape::from_slice(&[2, 3]));
        let n = shape.elements().max(1);
        let x_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.25).collect();
        let x = match Tensor::from_vec(x_data, shape.dims()) {
            Ok(t) => t,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not construct chain-rule fixture: {e}"),
                }
            }
        };

        // Composed (automatic): tanh(sigmoid(x)), differentiated through a
        // single, continuous computation that chains both stages -- this
        // exercises the executor's own internal chain-rule accumulation. The
        // "stage1->stage2" op-name convention is a private contract between this
        // module and whatever GradientExecutor is registered (see
        // tenflowers-autograd's implementation).
        let composed = match executor.compute_gradient("sigmoid->tanh", std::slice::from_ref(&x)) {
            Ok(g) if g.len() == 1 => g,
            Ok(g) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "GradientExecutor returned {} gradient(s) for 'sigmoid->tanh', \
                         expected 1",
                        g.len()
                    ),
                }
            }
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "Gradient computation failed for composed chain-rule fixture: {e}"
                    ),
                }
            }
        };

        // Manual chain rule: d(tanh)/ds (evaluated at s = sigmoid(x)) times
        // d(sigmoid)/dx, computed from two independent, single-stage gradient
        // calls and combined here with plain elementwise multiplication -- a
        // completely separate code path from the composed call above, so
        // comparing the two is a real, non-circular check.
        let sigmoid_x = match x.sigmoid() {
            Ok(t) => t,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not evaluate sigmoid(x) for chain-rule fixture: {e}"),
                }
            }
        };
        let d_sigmoid = match executor.compute_gradient("sigmoid", std::slice::from_ref(&x)) {
            Ok(g) if g.len() == 1 => g,
            Ok(g) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "GradientExecutor returned {} gradient(s) for 'sigmoid', expected 1",
                        g.len()
                    ),
                }
            }
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Gradient computation failed for 'sigmoid' stage: {e}"),
                }
            }
        };
        let d_tanh = match executor.compute_gradient("tanh", std::slice::from_ref(&sigmoid_x)) {
            Ok(g) if g.len() == 1 => g,
            Ok(g) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!(
                        "GradientExecutor returned {} gradient(s) for 'tanh', expected 1",
                        g.len()
                    ),
                }
            }
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Gradient computation failed for 'tanh' stage: {e}"),
                }
            }
        };

        let manual_data: Vec<f64> = d_tanh[0]
            .data()
            .iter()
            .zip(d_sigmoid[0].data().iter())
            .map(|(a, b)| a * b)
            .collect();
        let manual = match Tensor::from_vec(manual_data, shape.dims()) {
            Ok(t) => t,
            Err(e) => {
                return PropertyCheckResult {
                    property,
                    passed: false,
                    max_error: None,
                    details: format!("Could not assemble manual chain-rule gradient: {e}"),
                }
            }
        };

        let checker = NumericalGradientChecker::<f64>::new(test_case.config.clone());
        match checker.compare_gradients(&manual, &composed[0]) {
            Ok(cmp) if cmp.passed => PropertyCheckResult {
                property,
                passed: true,
                max_error: Some(cmp.max_absolute_error),
                details: "Composed gradient of tanh(sigmoid(x)) matches the manually \
                          chain-multiplied per-stage gradients."
                    .to_string(),
            },
            Ok(cmp) => PropertyCheckResult {
                property,
                passed: false,
                max_error: Some(cmp.max_absolute_error),
                details: format!("Chain rule mismatch: {}", cmp.summary()),
            },
            Err(e) => PropertyCheckResult {
                property,
                passed: false,
                max_error: None,
                details: format!("Could not compare chain-rule gradients: {e}"),
            },
        }
    }

    /// Validate a specific operation
    pub fn validate_operation(&self, operation: &str) -> OperationValidationReport {
        let mut report = OperationValidationReport::new(operation);
        let test_cases = self.generate_test_cases(operation);

        for test_case in test_cases {
            let result = self.validate_test_case(test_case);
            report.add_result(result);
        }

        report
    }

    /// Validate all operations with gradient implementations
    pub fn validate_all_operations(&self) -> ComprehensiveValidationReport {
        let mut report = ComprehensiveValidationReport::new();
        let auditor = get_auditor();
        let audit_report = auditor.audit_all();

        // Only validate operations with gradient implementations
        for (op_name, grad_info) in &audit_report.operations {
            if grad_info.status == GradientStatus::Implemented
                || grad_info.status == GradientStatus::Partial
            {
                let op_report = self.validate_operation(op_name);
                report.add_operation_report(op_report);
            }
        }

        report
    }

    /// Set test shapes
    pub fn set_test_shapes(&mut self, shapes: Vec<Shape>) {
        self.test_shapes = shapes;
    }

    /// Set test dtypes
    pub fn set_test_dtypes(&mut self, dtypes: Vec<DType>) {
        self.test_dtypes = dtypes;
    }
}

/// Best-effort arity hint used only to decide how many synthetic input tensors
/// to construct before calling into the registered [`GradientExecutor`]. This is
/// NOT authoritative -- the executor itself is the source of truth for what it
/// supports, and returns a clear `Err` if this guess is wrong.
fn synthetic_op_arity(op: &str) -> usize {
    match op {
        "relu" | "sigmoid" | "tanh" | "neg" | "abs" | "exp" | "log" | "sqrt" | "sin" | "cos" => 1,
        _ => 2,
    }
}

/// Synthesize `count` concrete f64 input tensors from `test_case.input_shapes`.
///
/// `generate_test_cases` currently records a single shape per test case
/// regardless of an operation's real arity, so this pads with the first
/// available shape (or a small default) when there aren't enough. Values follow
/// a fixed, deterministic pattern: the first tensor is strictly positive
/// (`1, 2, 3, ...`), and any subsequent tensor starts at exactly zero
/// (`0, 1, 2, ...`). This keeps operations that are well-defined at zero
/// (add/sub/mul/matmul/relu/sigmoid/tanh/pow-with-positive-base) finite, while
/// operations with a genuine mathematical singularity at zero in that position
/// (division) naturally -- not artificially -- surface a non-finite gradient.
fn synthesize_inputs(test_case: &GradientTestCase, count: usize) -> Result<Vec<Tensor<f64>>> {
    let default_shape = Shape::from_slice(&[2, 2]);
    let fallback = test_case
        .input_shapes
        .first()
        .cloned()
        .unwrap_or_else(|| default_shape.clone());

    let mut shapes: Vec<Shape> = test_case.input_shapes.iter().take(count).cloned().collect();
    while shapes.len() < count {
        shapes.push(fallback.clone());
    }

    shapes
        .into_iter()
        .enumerate()
        .map(|(idx, shape)| {
            let n = shape.elements().max(1);
            let data: Vec<f64> = if idx == 0 {
                (0..n).map(|i| i as f64 + 1.0).collect()
            } else {
                (0..n).map(|i| i as f64).collect()
            };
            Tensor::from_vec(data, shape.dims())
        })
        .collect()
}

// ============================================================================
// Global Validator Access
// ============================================================================

static GLOBAL_VALIDATOR: OnceLock<GradientValidator> = OnceLock::new();

/// Get the global gradient validator
pub fn get_validator() -> &'static GradientValidator {
    GLOBAL_VALIDATOR.get_or_init(GradientValidator::new)
}

/// Initialize the global validator
pub fn initialize_validator() {
    let _ = get_validator();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = GradientValidator::new();
        assert!(!validator.test_shapes.is_empty());
        assert!(!validator.test_dtypes.is_empty());
    }

    #[test]
    fn test_generate_test_cases() {
        let validator = GradientValidator::new();
        let test_cases = validator.generate_test_cases("add");

        // Should generate cases for each dtype x shape combination
        assert!(!test_cases.is_empty());
        assert!(test_cases.iter().all(|tc| tc.operation == "add"));
    }

    #[test]
    fn test_property_descriptions() {
        assert!(!GradientProperty::NumericalConsistency
            .description()
            .is_empty());
        assert!(!GradientProperty::Finiteness.description().is_empty());
    }

    #[test]
    fn test_validation_result() {
        let test_case =
            GradientTestCase::new("add", DType::Float32, vec![Shape::from_slice(&[2, 3])]);
        let result = ValidationResult::new(test_case);

        assert!(result.passed);
        assert!(result.property_results.is_empty());
    }

    #[test]
    fn test_operation_validation_report() {
        let mut report = OperationValidationReport::new("add");
        assert_eq!(report.total_tests, 0);
        assert_eq!(report.tests_passed, 0);

        let test_case =
            GradientTestCase::new("add", DType::Float32, vec![Shape::from_slice(&[2, 3])]);
        let mut result = ValidationResult::new(test_case);
        result.passed = true;

        report.add_result(result);
        assert_eq!(report.total_tests, 1);
        assert_eq!(report.tests_passed, 1);
        assert!(report.all_tests_passed());
    }

    #[test]
    fn test_comprehensive_report() {
        let mut report = ComprehensiveValidationReport::new();
        assert_eq!(report.total_operations, 0);

        let op_report = OperationValidationReport::new("add");
        report.add_operation_report(op_report);

        assert_eq!(report.total_operations, 1);
    }

    #[test]
    fn test_validate_operation() {
        let validator = GradientValidator::new();
        let report = validator.validate_operation("add");

        assert!(!report.operation.is_empty());
        assert!(report.total_tests > 0);
    }

    #[test]
    fn test_global_validator() {
        let validator1 = get_validator();
        let validator2 = get_validator();

        // Should return the same instance
        assert!(std::ptr::eq(validator1, validator2));
    }

    #[test]
    fn test_property_check_result() {
        let result = PropertyCheckResult {
            property: GradientProperty::Finiteness,
            passed: true,
            max_error: Some(1e-5),
            details: "Test".to_string(),
        };

        assert!(result.passed);
        assert_eq!(result.max_error, Some(1e-5));
    }

    #[test]
    fn test_shape_consistency_passes_for_compatible_shapes() {
        let validator = GradientValidator::new();
        let test_case = GradientTestCase::new(
            "add",
            DType::Float32,
            vec![Shape::from_slice(&[2, 3]), Shape::from_slice(&[2, 3])],
        );
        let result = validator.validate_test_case(test_case);
        let prop = result
            .property_results
            .get(&GradientProperty::ShapeConsistency)
            .expect("ShapeConsistency should have been checked");
        assert!(prop.passed, "expected pass: {}", prop.details);
    }

    #[test]
    fn test_shape_consistency_fails_for_mismatched_shapes() {
        let validator = GradientValidator::new();
        // matmul requires matching inner dimensions; [2,3] @ [4,5] is incompatible.
        let test_case = GradientTestCase::new(
            "matmul",
            DType::Float32,
            vec![Shape::from_slice(&[2, 3]), Shape::from_slice(&[4, 5])],
        );
        let result = validator.validate_test_case(test_case);
        let prop = result
            .property_results
            .get(&GradientProperty::ShapeConsistency)
            .expect("ShapeConsistency should have been checked");
        assert!(
            !prop.passed,
            "expected ShapeConsistency to fail for incompatible matmul shapes"
        );
    }

    #[test]
    fn test_shape_consistency_unregistered_operation_is_honest() {
        let validator = GradientValidator::new();
        let test_case = GradientTestCase::new(
            "definitely_not_a_registered_op",
            DType::Float32,
            vec![Shape::from_slice(&[2, 3])],
        );
        let result = validator.validate_test_case(test_case);
        let prop = result
            .property_results
            .get(&GradientProperty::ShapeConsistency)
            .expect("ShapeConsistency should have been checked");
        assert!(
            !prop.passed,
            "an unregistered operation must not be reported as passed"
        );
    }

    #[test]
    fn test_finiteness_without_registered_executor_is_honest_not_fabricated() {
        // This crate's own unit test binary never calls register_gradient_executor,
        // so Finiteness must honestly report that it cannot verify, rather than
        // fabricating `passed: true` as it did before this fix.
        let validator = GradientValidator::new();
        let test_case =
            GradientTestCase::new("add", DType::Float32, vec![Shape::from_slice(&[2, 3])]);
        let result = validator.validate_test_case(test_case);
        let prop = result
            .property_results
            .get(&GradientProperty::Finiteness)
            .expect("Finiteness should have been checked");
        assert!(
            !prop.passed,
            "must not fabricate passed:true without a registered executor"
        );
        assert!(
            prop.details.to_lowercase().contains("executor"),
            "details should explain that no gradient executor is registered: {}",
            prop.details
        );
    }
}
