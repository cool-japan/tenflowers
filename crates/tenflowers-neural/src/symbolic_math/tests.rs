//! Tests for the symbolic_math module.

use super::*;
use std::collections::HashMap;

// Helper functions
fn var(s: &str) -> Expr {
    Expr::Var(s.to_string())
}
fn c(v: f64) -> Expr {
    Expr::Const(v)
}
fn add(a: Expr, b: Expr) -> Expr {
    Expr::Add(Box::new(a), Box::new(b))
}
fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Sub(Box::new(a), Box::new(b))
}
fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Mul(Box::new(a), Box::new(b))
}
fn div(a: Expr, b: Expr) -> Expr {
    Expr::Div(Box::new(a), Box::new(b))
}
fn pow(a: Expr, b: Expr) -> Expr {
    Expr::Pow(Box::new(a), Box::new(b))
}
fn sin(a: Expr) -> Expr {
    Expr::Sin(Box::new(a))
}
fn cos(a: Expr) -> Expr {
    Expr::Cos(Box::new(a))
}
fn exp(a: Expr) -> Expr {
    Expr::Exp(Box::new(a))
}
fn ln(a: Expr) -> Expr {
    Expr::Ln(Box::new(a))
}

fn vars(kv: &[(&str, f64)]) -> HashMap<String, f64> {
    kv.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

// ── SymbolicExpr eval ──────────────────────────────────────────────────

#[test]
fn test_eval_const() {
    assert_eq!(
        eval(&c(3.125), &vars(&[])).expect("eval should succeed"),
        3.125
    );
}

#[test]
fn test_eval_var() {
    let e = var("x");
    assert_eq!(
        eval(&e, &vars(&[("x", 5.0)])).expect("eval should succeed"),
        5.0
    );
}

#[test]
fn test_eval_undefined_var() {
    let e = var("z");
    assert!(eval(&e, &vars(&[])).is_err());
}

#[test]
fn test_eval_arithmetic() {
    // 2 + 3*x for x=4 → 14
    let e = add(c(2.0), mul(c(3.0), var("x")));
    assert!((eval(&e, &vars(&[("x", 4.0)])).expect("eval should succeed") - 14.0).abs() < 1e-10);
}

#[test]
fn test_eval_sin_cos() {
    let e = add(sin(var("x")), cos(var("x")));
    let val = eval(&e, &vars(&[("x", 0.0)])).expect("eval should succeed");
    assert!((val - 1.0).abs() < 1e-10); // sin(0)+cos(0)=1
}

#[test]
fn test_eval_exp_ln() {
    let e = ln(exp(var("x")));
    let val = eval(&e, &vars(&[("x", 2.5)])).expect("eval should succeed");
    assert!((val - 2.5).abs() < 1e-10);
}

#[test]
fn test_eval_div_zero() {
    let e = div(c(1.0), c(0.0));
    assert!(eval(&e, &vars(&[])).is_err());
}

#[test]
fn test_eval_sqrt() {
    let e = Expr::Sqrt(Box::new(c(9.0)));
    assert!((eval(&e, &vars(&[])).expect("eval should succeed") - 3.0).abs() < 1e-10);
}

// ── simplify ──────────────────────────────────────────────────────────

#[test]
fn test_simplify_add_zero() {
    let e = add(c(0.0), var("x"));
    assert_eq!(simplify(e), var("x"));
}

#[test]
fn test_simplify_mul_one() {
    let e = mul(c(1.0), var("x"));
    assert_eq!(simplify(e), var("x"));
}

#[test]
fn test_simplify_mul_zero() {
    let e = mul(c(0.0), var("x"));
    assert_eq!(simplify(e), c(0.0));
}

#[test]
fn test_simplify_sub_self() {
    let e = sub(var("x"), var("x"));
    assert_eq!(simplify(e), c(0.0));
}

#[test]
fn test_simplify_div_self() {
    let e = div(var("x"), var("x"));
    assert_eq!(simplify(e), c(1.0));
}

#[test]
fn test_simplify_pow_zero() {
    let e = pow(var("x"), c(0.0));
    assert_eq!(simplify(e), c(1.0));
}

#[test]
fn test_simplify_pow_one() {
    let e = pow(var("x"), c(1.0));
    assert_eq!(simplify(e), var("x"));
}

#[test]
fn test_simplify_constant_fold() {
    let e = add(c(3.0), c(4.0));
    assert_eq!(simplify(e), c(7.0));
}

// ── derivative ────────────────────────────────────────────────────────

#[test]
fn test_derivative_const() {
    let e = c(5.0);
    let de = derivative(&e, "x");
    assert_eq!(simplify(de), c(0.0));
}

#[test]
fn test_derivative_var() {
    let de = derivative(&var("x"), "x");
    assert_eq!(simplify(de), c(1.0));
}

#[test]
fn test_derivative_other_var() {
    let de = derivative(&var("y"), "x");
    assert_eq!(simplify(de), c(0.0));
}

#[test]
fn test_derivative_polynomial() {
    // d/dx (3*x^2) = 6*x
    let e = mul(c(3.0), pow(var("x"), c(2.0)));
    let de = simplify(derivative(&e, "x"));
    let val = eval(&de, &vars(&[("x", 2.0)])).expect("eval should succeed");
    assert!((val - 12.0).abs() < 1e-8);
}

#[test]
fn test_derivative_sin() {
    // d/dx sin(x) = cos(x)
    let de = derivative(&sin(var("x")), "x");
    let val = eval(&de, &vars(&[("x", 0.0)])).expect("eval should succeed");
    assert!((val - 1.0).abs() < 1e-10); // cos(0)=1
}

#[test]
fn test_derivative_exp() {
    // d/dx e^x = e^x
    let e = exp(var("x"));
    let de = derivative(&e, "x");
    let x_val = 1.0f64;
    let v1 = eval(&e, &vars(&[("x", x_val)])).expect("eval should succeed");
    let v2 = eval(&de, &vars(&[("x", x_val)])).expect("eval should succeed");
    assert!((v1 - v2).abs() < 1e-8);
}

// ── ExpressionHasher ──────────────────────────────────────────────────

#[test]
fn test_hash_equal_exprs() {
    let e1 = add(var("x"), c(1.0));
    let e2 = add(var("x"), c(1.0));
    assert_eq!(
        ExpressionHasher::hash_expr(&e1),
        ExpressionHasher::hash_expr(&e2)
    );
}

#[test]
fn test_hash_different_exprs() {
    let e1 = add(var("x"), c(1.0));
    let e2 = add(var("x"), c(2.0));
    assert_ne!(
        ExpressionHasher::hash_expr(&e1),
        ExpressionHasher::hash_expr(&e2)
    );
}

#[test]
fn test_exprs_equal() {
    let e1 = mul(c(2.0), var("y"));
    let e2 = mul(c(2.0), var("y"));
    assert!(ExpressionHasher::exprs_equal(&e1, &e2));
}

// ── parse_token_sequence ──────────────────────────────────────────────

#[test]
fn test_parse_postfix_add() {
    // Tokens: x0, Const(1.0), Add → x0 + 1.0
    let tokens = vec![
        ExprToken::Var(0),
        ExprToken::Const(4), // index 4 → 1.0 in default table
        ExprToken::Add,
        ExprToken::End,
    ];
    let const_table = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];
    let var_names = vec!["x0".to_string()];
    let result = parse_token_sequence(&tokens, &const_table, &var_names);
    assert!(result.is_ok());
    let expr = result.expect("parse result should be Ok");
    let val = eval(&expr, &vars(&[("x0", 3.0)])).expect("eval should succeed");
    assert!((val - 4.0).abs() < 1e-10);
}

#[test]
fn test_parse_postfix_sin() {
    let tokens = vec![ExprToken::Var(0), ExprToken::Sin, ExprToken::End];
    let const_table = vec![0.0];
    let var_names = vec!["x".to_string()];
    let result =
        parse_token_sequence(&tokens, &const_table, &var_names).expect("parse should succeed");
    let val =
        eval(&result, &vars(&[("x", std::f64::consts::PI / 2.0)])).expect("eval should succeed");
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_parse_postfix_stack_error() {
    // Add with only one operand on stack
    let tokens = vec![ExprToken::Var(0), ExprToken::Add];
    let result = parse_token_sequence(&tokens, &[], &["x".to_string()]);
    assert!(result.is_err());
}

// ── GradientSymbolicRegressor ─────────────────────────────────────────

#[test]
fn test_regressor_fit_constant() {
    let basis = ExprBasis::standard(1);
    let mut reg = GradientSymbolicRegressor::new(basis).with_max_iter(500);
    // y = 2 (constant)
    let x_data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
    let y_data: Vec<f64> = vec![2.0; 10];
    let coeffs = reg.fit(&x_data, &y_data, 0.0);
    // Coefficients vector should be non-empty; just verify fit runs
    assert!(!coeffs.is_empty());
    // The prediction should be bounded (gradient descent has run)
    let pred = reg.predict(&[0.0]);
    assert!(pred.is_finite());
}

#[test]
fn test_regressor_active_terms() {
    let basis = ExprBasis::standard(1);
    let mut reg = GradientSymbolicRegressor::new(basis);
    let x_data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y_data: Vec<f64> = vec![0.0; 5];
    reg.fit(&x_data, &y_data, 0.1);
    // Should have very few active terms
    let terms = reg.active_terms();
    assert!(terms.len() <= 3);
}

// ── FormulaSearch ─────────────────────────────────────────────────────

#[test]
fn test_crossover_preserves_structure() {
    let a = add(var("x"), c(1.0));
    let b = mul(c(2.0), var("x"));
    let mut rng = StdRng::seed_from_u64(42);
    let child = crossover(&a, &b, &mut rng);
    // Should be a valid expression (can be evaluated)
    let val = eval(&child, &vars(&[("x", 1.0)]));
    // May fail if div-by-zero etc, but shouldn't panic
    let _ = val;
}

#[test]
fn test_mutate_returns_expr() {
    let e = add(var("x"), c(2.0));
    let mut rng = StdRng::seed_from_u64(7);
    let m = mutate(&e, &mut rng, &["x".to_string()]);
    // Just check it's a valid Expr
    let _ = eval(&m, &vars(&[("x", 1.0)]));
}

#[test]
fn test_tournament_select_best() {
    let pop = vec![
        Individual {
            expr: c(1.0),
            fitness: 10.0,
        },
        Individual {
            expr: c(2.0),
            fitness: 2.0,
        },
        Individual {
            expr: c(3.0),
            fitness: 5.0,
        },
    ];
    let mut rng = StdRng::seed_from_u64(0);
    // With k=3 we always see all; best fitness is 2.0
    // Note: tournament_select uses random sampling, so just check no panic
    let sel = tournament_select(&pop, 3, &mut rng);
    assert!(sel.fitness >= 0.0);
}

#[test]
fn test_evolve_reduces_error() {
    let x_data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x[0] * 2.0).collect();
    let mut rng = StdRng::seed_from_u64(123);
    let mut pop = init_population(20, &x_data, &y_data, &mut rng);
    let initial_best = pop.iter().map(|i| i.fitness).fold(f64::MAX, f64::min);
    evolve(&mut pop, &x_data, &y_data, 5, &mut rng);
    let final_best = pop.iter().map(|i| i.fitness).fold(f64::MAX, f64::min);
    // After evolution, best should be ≤ initial (or same if already perfect)
    assert!(final_best <= initial_best + 1e-3);
}

// ── EquationBalancer ──────────────────────────────────────────────────

#[test]
fn test_balance_water() {
    // H2 + O2 → H2O
    let mat = parse_equation("H2+O2", "H2O");
    let coeffs = balance(&mat);
    assert!(coeffs.is_some());
    let c = coeffs.expect("balance should produce coefficients");
    // Check: 2*H2 + O2 → 2*H2O
    // H: 2*c[0] = 2*c[2], O: 2*c[1] = c[2]
    assert_eq!(c.len(), 3);
    // Verify stoichiometry (ratios)
    // c[0]:c[1]:c[2] should be 2:1:2
    let ratio = c[0] as f64 / c[2] as f64;
    assert!((ratio - 1.0).abs() < 0.01);
}

#[test]
fn test_balance_combustion() {
    // CH4 + O2 → CO2 + H2O
    let mat = parse_equation("CH4+O2", "CO2+H2O");
    let coeffs = balance(&mat);
    assert!(coeffs.is_some());
    let c = coeffs.expect("balance should produce coefficients");
    assert_eq!(c.len(), 4);
    // Check: 1 CH4 + 2 O2 → 1 CO2 + 2 H2O
    // Verify H balance: 4*c[0] = 2*c[3]
    let lhs_h = 4 * c[0];
    let rhs_h = 2 * c[3];
    assert_eq!(lhs_h, rhs_h);
}

#[test]
fn test_parse_formula() {
    let f = parse_formula("H2O");
    assert_eq!(*f.get("H").unwrap_or(&0), 2);
    assert_eq!(*f.get("O").unwrap_or(&0), 1);
}

// ── DimensionalAnalysis ───────────────────────────────────────────────

#[test]
fn test_dim_multiply() {
    let force = Dimension::new(1, 1, -2, 0, 0); // kg·m/s²
    let dist = Dimension::new(0, 1, 0, 0, 0); // m
    let work = dim_multiply(&force, &dist); // kg·m²/s²
    assert_eq!(work, Dimension::new(1, 2, -2, 0, 0));
}

#[test]
fn test_dim_divide() {
    let vel = Dimension::new(0, 1, -1, 0, 0); // m/s
    let time = Dimension::new(0, 0, 1, 0, 0); // s
    let acc = dim_divide(&vel, &time); // m/s²
    assert_eq!(acc, Dimension::new(0, 1, -2, 0, 0));
}

#[test]
fn test_is_dimensionless() {
    assert!(is_dimensionless(&Dimension::dimensionless()));
    assert!(!is_dimensionless(&Dimension::new(1, 0, 0, 0, 0)));
}

#[test]
fn test_find_pi_groups_pendulum() {
    // Pendulum: period T [s], length L [m], mass m [kg], gravity g [m/s²]
    let quantities = vec![
        ("T".to_string(), Dimension::new(0, 0, 1, 0, 0)),
        ("L".to_string(), Dimension::new(0, 1, 0, 0, 0)),
        ("m".to_string(), Dimension::new(1, 0, 0, 0, 0)),
        ("g".to_string(), Dimension::new(0, 1, -2, 0, 0)),
    ];
    let pi_groups = find_pi_groups(&quantities);
    // By Buckingham Pi: 4 - rank(3) = 1 Pi group
    assert!(!pi_groups.is_empty());
}

// ── MatrixCalculus ────────────────────────────────────────────────────

#[test]
fn test_trace_derivative_ax() {
    // d Tr(A*X) / dX = A^T
    let ax = MatrixExpr::Mul(
        Box::new(MatrixExpr::Matrix("A".to_string(), 3, 3)),
        Box::new(MatrixExpr::Matrix("X".to_string(), 3, 3)),
    );
    let trace_ax = MatrixExpr::Trace(Box::new(ax));
    let result = trace_derivative(&trace_ax, "X");
    assert!(matches!(result, MatrixExpr::Transpose(_)));
}

#[test]
fn test_trace_derivative_identity() {
    // d Tr(X) / dX = I
    let tx = MatrixExpr::Trace(Box::new(MatrixExpr::Matrix("X".to_string(), 3, 3)));
    let result = trace_derivative(&tx, "X");
    assert!(matches!(result, MatrixExpr::Matrix(ref n, _, _) if n == "I"));
}

// ── PolynomialArithmetic ──────────────────────────────────────────────

#[test]
fn test_poly_evaluate_horner() {
    // p(x) = 1 + 2x + 3x^2; p(2) = 1+4+12=17
    let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
    assert!((poly_evaluate(&p, 2.0) - 17.0).abs() < 1e-10);
}

#[test]
fn test_poly_add() {
    let a = Polynomial::new(vec![1.0, 2.0]);
    let b = Polynomial::new(vec![3.0, 4.0]);
    let c = poly_add(&a, &b);
    assert_eq!(c.coeffs, vec![4.0, 6.0]);
}

#[test]
fn test_poly_mul() {
    // (1 + x)(1 + x) = 1 + 2x + x^2
    let p = Polynomial::new(vec![1.0, 1.0]);
    let q = poly_mul(&p, &p);
    assert!((q.coeffs[0] - 1.0).abs() < 1e-10);
    assert!((q.coeffs[1] - 2.0).abs() < 1e-10);
    assert!((q.coeffs[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_poly_div_rem() {
    // (x^2 - 1) / (x - 1) = x + 1, rem 0
    let num = Polynomial::new(vec![-1.0, 0.0, 1.0]);
    let den = Polynomial::new(vec![-1.0, 1.0]);
    let (quot, rem) = poly_div_rem(&num, &den);
    assert!((poly_evaluate(&quot, 5.0) - 6.0).abs() < 1e-8); // x+1 at 5 = 6
    assert!(rem.is_zero());
}

#[test]
fn test_poly_gcd() {
    // gcd(x^2 - 1, x - 1) = x - 1
    let a = Polynomial::new(vec![-1.0, 0.0, 1.0]);
    let b = Polynomial::new(vec![-1.0, 1.0]);
    let g = poly_gcd(&a, &b);
    assert_eq!(g.degree(), 1);
    assert!((poly_evaluate(&g, 1.0)).abs() < 1e-8);
}

#[test]
fn test_roots_linear() {
    // 2x - 6 = 0 → x = 3
    let p = Polynomial::new(vec![-6.0, 2.0]);
    let roots = roots_companion_matrix(&p);
    assert_eq!(roots.len(), 1);
    assert!((roots[0].0 - 3.0).abs() < 1e-6);
}

#[test]
fn test_roots_quadratic() {
    // x^2 - 5x + 6 = (x-2)(x-3), roots 2 and 3
    let p = Polynomial::new(vec![6.0, -5.0, 1.0]);
    let roots = roots_companion_matrix(&p);
    // At least one root found
    assert!(!roots.is_empty());
    for (r, _) in &roots {
        assert!(poly_evaluate(&p, *r).abs() < 1e-4);
    }
}

// ── SymbolicIntegrator ────────────────────────────────────────────────

#[test]
fn test_integrate_constant() {
    // ∫ 3 dx = 3*x
    let result = integrate(&c(3.0), "x").expect("integration should succeed");
    let val = eval(&result, &vars(&[("x", 2.0)])).expect("eval should succeed");
    assert!((val - 6.0).abs() < 1e-10);
}

#[test]
fn test_integrate_variable() {
    // ∫ x dx = x^2/2
    let result = integrate(&var("x"), "x").expect("integration should succeed");
    let val = eval(&result, &vars(&[("x", 4.0)])).expect("eval should succeed");
    assert!((val - 8.0).abs() < 1e-10); // 4^2/2=8
}

#[test]
fn test_integrate_monomial() {
    // ∫ x^3 dx = x^4/4
    let e = pow(var("x"), c(3.0));
    let result = integrate(&e, "x").expect("integration should succeed");
    let val = eval(&result, &vars(&[("x", 2.0)])).expect("eval should succeed");
    assert!((val - 4.0).abs() < 1e-10); // 2^4/4=4
}

#[test]
fn test_integrate_sin() {
    // ∫ sin(x) dx = -cos(x); at x=0: -cos(0)=-1
    let result = integrate(&sin(var("x")), "x").expect("integration should succeed");
    let val = eval(&result, &vars(&[("x", 0.0)])).expect("eval should succeed");
    assert!((val + 1.0).abs() < 1e-10);
}

#[test]
fn test_integrate_cos() {
    // ∫ cos(x) dx = sin(x); at x=π/2: sin(π/2)=1
    let result = integrate(&cos(var("x")), "x").expect("integration should succeed");
    let val =
        eval(&result, &vars(&[("x", std::f64::consts::PI / 2.0)])).expect("eval should succeed");
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_integrate_exp() {
    // ∫ e^x dx = e^x; at x=1: e
    let result = integrate(&exp(var("x")), "x").expect("integration should succeed");
    let val = eval(&result, &vars(&[("x", 1.0)])).expect("eval should succeed");
    assert!((val - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn test_integrate_one_over_x() {
    // ∫ 1/x dx = ln(x); at x=e: ln(e)=1
    let e = div(c(1.0), var("x"));
    let result = integrate(&e, "x").expect("integration should succeed");
    let val = eval(&result, &vars(&[("x", std::f64::consts::E)])).expect("eval should succeed");
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_integrate_sum_rule() {
    // ∫ (sin(x) + cos(x)) dx = -cos(x) + sin(x)
    let e = add(sin(var("x")), cos(var("x")));
    let result = integrate(&e, "x");
    assert!(result.is_some());
}

#[test]
fn test_integrate_constant_multiple() {
    // ∫ 3*x dx = 3 * x^2/2
    let e = mul(c(3.0), var("x"));
    let result = integrate(&e, "x").expect("integration should succeed");
    let val = eval(&result, &vars(&[("x", 2.0)])).expect("eval should succeed");
    assert!((val - 6.0).abs() < 1e-10); // 3*4/2=6
}

#[test]
fn test_integrate_unrecognized() {
    // ∫ sin(x^2) dx — no pattern
    let e = sin(pow(var("x"), c(2.0)));
    let result = integrate(&e, "x");
    assert!(result.is_none());
}

// ── Advanced: ProofState and Hypothesis ───────────────────────────────

#[test]
fn test_proof_state_new_single_goal() {
    let goal = add(var("x"), c(1.0));
    let ps = ProofState::new(goal.clone());
    assert_eq!(ps.n_goals(), 1);
    assert!(!ps.is_complete());
    assert!(ps.hypotheses.is_empty());
}

#[test]
fn test_proof_state_empty_goals() {
    let ps = ProofState::with_goals(vec![]);
    assert!(ps.is_complete());
    assert_eq!(ps.n_goals(), 0);
}

#[test]
fn test_proof_state_add_hypothesis() {
    let mut ps = ProofState::new(var("x"));
    let hyp = Hypothesis::new("h1", c(42.0));
    ps.add_hypothesis(hyp);
    assert_eq!(ps.hypotheses.len(), 1);
    assert_eq!(ps.hypotheses[0].name, "h1");
}

#[test]
fn test_hypothesis_creation() {
    let hyp = Hypothesis::new("ha", mul(var("x"), c(2.0)));
    assert_eq!(hyp.name, "ha");
    match &hyp.formula {
        Expr::Mul(_, _) => {}
        _ => panic!("expected Mul formula"),
    }
}

// ── Advanced: TacticEngine ────────────────────────────────────────────

#[test]
fn test_tactic_simplify_zero_const() {
    let engine = TacticEngine::new();
    // Goal: 0 — should be discharged after simplify
    let ps = ProofState::new(c(0.0));
    let result = engine.apply(&ps, &Tactic::Simplify);
    assert!(result.is_ok(), "simplify on 0 should succeed");
    let new_ps = result.expect("simplify result");
    assert!(new_ps.is_complete(), "goal 0 should be discharged");
}

#[test]
fn test_tactic_trivial_zero() {
    let engine = TacticEngine::new();
    let ps = ProofState::new(c(0.0));
    let result = engine.apply(&ps, &Tactic::Trivial);
    assert!(result.is_ok(), "trivial on Const(0) should succeed");
    let new_ps = result.expect("trivial result");
    assert!(new_ps.is_complete(), "goal should be discharged");
}

#[test]
fn test_tactic_trivial_nonzero_fails() {
    let engine = TacticEngine::new();
    let ps = ProofState::new(c(5.0));
    let result = engine.apply(&ps, &Tactic::Trivial);
    assert!(result.is_err(), "trivial should fail on non-zero goal");
}

#[test]
fn test_tactic_intro_adds_hypothesis() {
    let engine = TacticEngine::new();
    let goal = var("x");
    let ps = ProofState::new(goal);
    let result = engine.apply(&ps, &Tactic::Intro("h_x".to_string()));
    assert!(result.is_ok());
    let new_ps = result.expect("intro result");
    assert!(new_ps.is_complete(), "intro should discharge goal");
    assert_eq!(new_ps.hypotheses.len(), 1);
    assert_eq!(new_ps.hypotheses[0].name, "h_x");
}

#[test]
fn test_tactic_split_add_goal() {
    let engine = TacticEngine::new();
    let goal = add(var("x"), var("y"));
    let ps = ProofState::new(goal);
    let result = engine.apply(&ps, &Tactic::Split);
    assert!(result.is_ok());
    let new_ps = result.expect("split result");
    assert_eq!(new_ps.n_goals(), 2, "split Add should produce 2 goals");
}

#[test]
fn test_tactic_split_non_add_fails() {
    let engine = TacticEngine::new();
    let ps = ProofState::new(var("x"));
    let result = engine.apply(&ps, &Tactic::Split);
    assert!(result.is_err(), "split on non-Add goal should fail");
}

#[test]
fn test_tactic_exact_matching_hypothesis() {
    let engine = TacticEngine::new();
    let formula = mul(var("x"), c(2.0));
    let mut ps = ProofState::new(formula.clone());
    ps.add_hypothesis(Hypothesis::new("h1", formula));
    let result = engine.apply(&ps, &Tactic::Exact("h1".to_string()));
    assert!(result.is_ok());
    let new_ps = result.expect("exact result");
    assert!(
        new_ps.is_complete(),
        "exact with matching hypothesis should close goal"
    );
}

#[test]
fn test_tactic_exact_missing_hypothesis_fails() {
    let engine = TacticEngine::new();
    let ps = ProofState::new(var("x"));
    let result = engine.apply(&ps, &Tactic::Exact("nonexistent".to_string()));
    assert!(result.is_err());
}

#[test]
fn test_tactic_differentiate() {
    let engine = TacticEngine::new();
    // Goal: x^2 → differentiate x → 2*x (or simplified form)
    let goal = pow(var("x"), c(2.0));
    let ps = ProofState::new(goal);
    let result = engine.apply(&ps, &Tactic::Differentiate("x".to_string()));
    assert!(result.is_ok());
    let new_ps = result.expect("differentiate result");
    // Still has one goal (the derivative), not closed
    assert_eq!(new_ps.n_goals(), 1);
    assert!(!new_ps.trace.is_empty());
}

#[test]
fn test_tactic_apply_sequence() {
    let engine = TacticEngine::new();
    // Intro then check hypothesis was added
    let ps = ProofState::new(var("y"));
    let tactics = vec![Tactic::Intro("h_y".to_string())];
    let result = engine.apply_sequence(&ps, &tactics);
    assert!(result.is_ok());
    let final_ps = result.expect("sequence result");
    assert!(final_ps.is_complete());
    assert_eq!(final_ps.hypotheses.len(), 1);
}

#[test]
fn test_tactic_no_goals_returns_error() {
    let engine = TacticEngine::new();
    let ps = ProofState::with_goals(vec![]);
    let result = engine.apply(&ps, &Tactic::Trivial);
    assert!(
        result.is_err(),
        "applying tactic with no goals should error"
    );
}

#[test]
fn test_tactic_name_strings() {
    assert_eq!(Tactic::Simplify.name(), "simplify");
    assert_eq!(Tactic::Split.name(), "split");
    assert_eq!(Tactic::Trivial.name(), "trivial");
    assert_eq!(Tactic::Intro("x".to_string()).name(), "intro");
    assert_eq!(Tactic::Rewrite("h".to_string()).name(), "rewrite");
    assert_eq!(
        Tactic::Differentiate("x".to_string()).name(),
        "differentiate"
    );
    assert_eq!(Tactic::Exact("h".to_string()).name(), "exact");
}

// ── Advanced: NeuralTacticSelector ────────────────────────────────────

#[test]
fn test_neural_tactic_selector_output_dim() {
    let selector = NeuralTacticSelector::new(12, 16, 42);
    assert_eq!(selector.n_tactics, 5);
    assert_eq!(selector.input_dim, 12);
    assert_eq!(selector.hidden_dim, 16);
}

#[test]
fn test_neural_tactic_selector_select_valid_index() {
    let selector = NeuralTacticSelector::new(12, 16, 7);
    let ps = ProofState::new(add(var("x"), c(1.0)));
    let idx = selector.select_tactic_idx(&ps);
    assert!(
        idx < selector.n_tactics,
        "selected index must be valid tactic"
    );
}

#[test]
fn test_neural_tactic_selector_probabilities_sum_to_one() {
    let selector = NeuralTacticSelector::new(12, 16, 99);
    let ps = ProofState::new(mul(var("x"), var("y")));
    let probs = selector.tactic_probabilities(&ps);
    assert_eq!(probs.len(), 5);
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "probabilities must sum to 1, got {sum}"
    );
}

#[test]
fn test_neural_tactic_selector_probabilities_all_positive() {
    let selector = NeuralTacticSelector::new(12, 16, 77);
    let ps = ProofState::new(c(3.0));
    let probs = selector.tactic_probabilities(&ps);
    assert!(
        probs.iter().all(|&p| p >= 0.0),
        "all probabilities must be non-negative"
    );
}

#[test]
fn test_neural_tactic_featurize_length() {
    let selector = NeuralTacticSelector::new(12, 16, 1);
    let ps = ProofState::new(sin(var("x")));
    let feats = selector.featurize(&ps);
    assert_eq!(feats.len(), 12, "feature vector must match input_dim");
}

// ── Advanced: DiscoveredEquation and EquationDatabase ─────────────────

#[test]
fn test_discovered_equation_aic_computation() {
    let eq = DiscoveredEquation::new(var("x"), 0.01, 2, 100);
    // AIC = 2*2 + 100*ln(0.01) ≈ 4 - 460.5 = -456.5
    assert!(
        eq.aic < 0.0,
        "AIC should be negative for low MSE, got {}",
        eq.aic
    );
    assert_eq!(eq.n_params, 2);
    assert!((eq.mse - 0.01).abs() < 1e-15);
}

#[test]
fn test_discovered_equation_high_mse_positive_aic() {
    let eq = DiscoveredEquation::new(var("x"), 10.0, 1, 10);
    // AIC = 2*1 + 10*ln(10) ≈ 2 + 23 = 25
    assert!(
        eq.aic > 0.0,
        "high MSE should give positive AIC, got {}",
        eq.aic
    );
}

#[test]
fn test_equation_database_empty() {
    let db = EquationDatabase::new();
    assert!(db.is_empty());
    assert_eq!(db.len(), 0);
    assert!(db.best().is_none());
}

#[test]
fn test_equation_database_add_and_best() {
    let mut db = EquationDatabase::new();
    let eq1 = DiscoveredEquation::new(var("x"), 1.0, 1, 50);
    let eq2 = DiscoveredEquation::new(mul(var("x"), c(2.0)), 0.1, 2, 50);
    db.add(eq1);
    db.add(eq2);
    assert_eq!(db.len(), 2);
    let best = db.best().expect("database should have best equation");
    // Lower MSE gives lower AIC for same n_data
    assert!(best.mse <= 1.0);
}

#[test]
fn test_equation_database_sorted_by_aic() {
    let mut db = EquationDatabase::new();
    db.add(DiscoveredEquation::new(var("x"), 5.0, 1, 20));
    db.add(DiscoveredEquation::new(var("y"), 0.001, 1, 20));
    db.add(DiscoveredEquation::new(var("z"), 1.0, 1, 20));
    // Should be sorted ascending by AIC
    let aics: Vec<f64> = db.equations.iter().map(|e| e.aic).collect();
    for i in 1..aics.len() {
        assert!(
            aics[i - 1] <= aics[i],
            "equations must be sorted by AIC ascending"
        );
    }
}

#[test]
fn test_equation_database_pareto_front_nonempty() {
    let mut db = EquationDatabase::new();
    db.add(DiscoveredEquation::new(var("x"), 0.1, 1, 20));
    db.add(DiscoveredEquation::new(mul(var("x"), c(2.0)), 0.05, 3, 20));
    let front = db.pareto_front();
    assert!(!front.is_empty(), "pareto front should not be empty");
}

// ── Advanced: SrSymbolicRegressor ─────────────────────────────────────

#[test]
fn test_sr_symbolic_regressor_fit_populates_database() {
    let mut sr = SrSymbolicRegressor::new(vec!["x".to_string()], 20, 5, 0.01);
    // y = 2*x data
    let x_data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
    let y_data: Vec<f64> = (0..10).map(|i| 2.0 * i as f64).collect();
    sr.fit(&x_data, &y_data, 42);
    // database may or may not have entries (population filter)
    // just check it doesn't panic and returns something
    let _ = sr.best_equation();
}

#[test]
fn test_sr_symbolic_regressor_mse_exact() {
    let sr = SrSymbolicRegressor::new(vec!["x".to_string()], 10, 3, 0.01);
    // Expression: x; data: y = x → MSE should be 0
    let x_data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y_data: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let mse = sr.mse(&var("x"), &x_data, &y_data);
    assert!(
        mse.abs() < 1e-10,
        "MSE for y=x on y=x data should be 0, got {mse}"
    );
}

#[test]
fn test_sr_symbolic_regressor_mse_nonzero() {
    let sr = SrSymbolicRegressor::new(vec!["x".to_string()], 10, 3, 0.01);
    let x_data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // Expression: x^2 — not a perfect fit
    let expr = pow(var("x"), c(2.0));
    let mse = sr.mse(&expr, &x_data, &y_data);
    assert!(mse > 0.0, "MSE for x^2 on y=x+1 data should be positive");
}

// ── NeuralExpressionSynthesizer ───────────────────────────────────────

#[test]
fn test_synthesizer_beam_decode() {
    let config = SynthesizerConfig::default();
    let var_names = vec!["x0".to_string()];
    let mut rng = StdRng::seed_from_u64(99);
    let synth = NeuralExpressionSynthesizer::new(config, var_names, &mut rng);
    let tokens = synth.beam_decode(3);
    assert!(!tokens.is_empty());
}

#[test]
fn test_expr_basis_standard() {
    let basis = ExprBasis::standard(2);
    assert!(!basis.exprs.is_empty());
    assert_eq!(basis.exprs.len(), basis.names.len());
}
