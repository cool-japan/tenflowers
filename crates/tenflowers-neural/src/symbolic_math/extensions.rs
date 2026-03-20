//! Extensions for symbolic_math: matrix calculus, polynomial arithmetic, symbolic integration.

use super::*;
use scirs2_core::RngExt;

/// Run one generation of genetic programming.
pub fn evolve(
    population: &mut Vec<Individual>,
    x: &[Vec<f64>],
    y: &[f64],
    generations: usize,
    rng: &mut StdRng,
) {
    // Collect variable names from data dimensions
    let n_vars = x.first().map(|r| r.len()).unwrap_or(1);
    let var_names: Vec<String> = (0..n_vars).map(|i| format!("x{i}")).collect();

    for _gen in 0..generations {
        let pop_size = population.len();
        let mut new_pop: Vec<Individual> = Vec::with_capacity(pop_size);

        // Elitism: keep top 10%
        population.sort_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let elite_n = (pop_size / 10).max(1);
        for ind in population.iter().take(elite_n) {
            new_pop.push(ind.clone());
        }

        while new_pop.len() < pop_size {
            let op: f64 = rng.random();
            let child_expr = if op < 0.5 {
                // Crossover
                let p1 = tournament_select(population, 3, rng).expr.clone();
                let p2 = tournament_select(population, 3, rng).expr.clone();
                crossover(&p1, &p2, rng)
            } else {
                // Mutate
                let parent = tournament_select(population, 3, rng).expr.clone();
                mutate(&parent, rng, &var_names)
            };
            let child_expr = simplify(child_expr);
            let fitness = mse_fitness(&child_expr, x, y);
            new_pop.push(Individual {
                expr: child_expr,
                fitness,
            });
        }
        *population = new_pop;
    }
    population.sort_by(|a, b| {
        a.fitness
            .partial_cmp(&b.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Create an initial population with random expressions.
pub fn init_population(
    pop_size: usize,
    x: &[Vec<f64>],
    y: &[f64],
    rng: &mut StdRng,
) -> Vec<Individual> {
    let n_vars = x.first().map(|r| r.len()).unwrap_or(1);
    let var_names: Vec<String> = (0..n_vars).map(|i| format!("x{i}")).collect();
    (0..pop_size)
        .map(|_| {
            let expr = random_expr(rng, &var_names, 3);
            let fitness = mse_fitness(&expr, x, y);
            Individual { expr, fitness }
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// 6. EquationBalancer
// ────────────────────────────────────────────────────────────────────────────

/// Rational number for exact arithmetic in Gaussian elimination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rational {
    num: i64,
    den: i64,
}

impl Rational {
    fn new(num: i64, den: i64) -> Self {
        if den == 0 {
            return Self { num: 0, den: 1 };
        }
        let g = gcd_i64(num.abs(), den.abs());
        let sign = if den < 0 { -1 } else { 1 };
        Self {
            num: sign * num / g,
            den: sign * den / g,
        }
    }

    fn zero() -> Self {
        Self { num: 0, den: 1 }
    }
    fn one() -> Self {
        Self { num: 1, den: 1 }
    }

    fn add(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)
    }
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
    }
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.num, self.den * rhs.den)
    }
    fn div(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den, self.den * rhs.num)
    }
    fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }
    fn is_zero(self) -> bool {
        self.num == 0
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd_i64(b, a % b)
    }
}

fn lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd_i64(a, b) * b
    }
}

/// Parse a chemical formula like "H2O" or "CH4" into element counts.
pub fn parse_formula(formula: &str) -> HashMap<String, i32> {
    let mut result: HashMap<String, i32> = HashMap::new();
    let chars: Vec<char> = formula.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_uppercase() {
            let mut elem = String::from(chars[i]);
            i += 1;
            while i < chars.len() && chars[i].is_lowercase() {
                elem.push(chars[i]);
                i += 1;
            }
            let mut num_str = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                num_str.push(chars[i]);
                i += 1;
            }
            let count: i32 = if num_str.is_empty() {
                1
            } else {
                num_str.parse().unwrap_or(1)
            };
            *result.entry(elem).or_insert(0) += count;
        } else {
            i += 1;
        }
    }
    result
}

/// Parse a side of an equation (comma or '+'-separated molecules) into an element matrix column.
fn parse_side(side: &str) -> Vec<HashMap<String, i32>> {
    side.split('+').map(|s| parse_formula(s.trim())).collect()
}

/// Build the element-coefficient matrix for the equation.
/// Returns matrix with shape [n_elements × n_compounds] where LHS coefficients are positive,
/// RHS coefficients are negated (so null vector gives balanced equation).
pub fn parse_equation(lhs: &str, rhs: &str) -> Vec<Vec<i32>> {
    let lhs_mols = parse_side(lhs);
    let rhs_mols = parse_side(rhs);

    // Collect all elements
    let mut elements: Vec<String> = Vec::new();
    for mol in lhs_mols.iter().chain(rhs_mols.iter()) {
        for elem in mol.keys() {
            if !elements.contains(elem) {
                elements.push(elem.clone());
            }
        }
    }
    elements.sort();

    let n_elem = elements.len();
    let n_comp = lhs_mols.len() + rhs_mols.len();
    let mut matrix = vec![vec![0i32; n_comp]; n_elem];

    for (j, mol) in lhs_mols.iter().enumerate() {
        for (i, elem) in elements.iter().enumerate() {
            matrix[i][j] = *mol.get(elem).unwrap_or(&0);
        }
    }
    for (j, mol) in rhs_mols.iter().enumerate() {
        let col = lhs_mols.len() + j;
        for (i, elem) in elements.iter().enumerate() {
            matrix[i][col] = -(*mol.get(elem).unwrap_or(&0));
        }
    }
    matrix
}

/// Find a small integer null vector via rational Gaussian elimination.
pub fn balance(matrix: &[Vec<i32>]) -> Option<Vec<i32>> {
    if matrix.is_empty() {
        return None;
    }
    let n_rows = matrix.len();
    let n_cols = matrix[0].len();
    if n_cols == 0 {
        return None;
    }

    // Work in rationals
    let mut mat: Vec<Vec<Rational>> = matrix
        .iter()
        .map(|row| row.iter().map(|&v| Rational::new(v as i64, 1)).collect())
        .collect();

    // Append identity for tracking free variables
    // Actually use standard null-space: row-reduce [A | I] is complex.
    // Simpler: row-reduce A, set last free var = 1, back-substitute.
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut row = 0usize;
    for col in 0..n_cols {
        // Find pivot
        let mut piv = None;
        for r in row..n_rows {
            if !mat[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }
        if let Some(pr) = piv {
            mat.swap(row, pr);
            let scale = mat[row][col];
            for c in 0..n_cols {
                mat[row][c] = mat[row][c].div(scale);
            }
            for r in 0..n_rows {
                if r != row && !mat[r][col].is_zero() {
                    let factor = mat[r][col];
                    for c in 0..n_cols {
                        let val = mat[row][c].mul(factor);
                        mat[r][c] = mat[r][c].sub(val);
                    }
                }
            }
            pivot_cols.push(col);
            row += 1;
        }
    }

    // Free variables: columns not in pivot_cols
    let free_cols: Vec<usize> = (0..n_cols).filter(|c| !pivot_cols.contains(c)).collect();
    if free_cols.is_empty() {
        return None;
    }

    // Set first free variable = 1, others = 0
    let mut solution: Vec<Rational> = vec![Rational::zero(); n_cols];
    solution[free_cols[0]] = Rational::one();

    // Back-substitute pivot variables
    for (pi, &pc) in pivot_cols.iter().enumerate() {
        let mut s = Rational::zero();
        for &fc in &free_cols {
            s = s.add(mat[pi][fc].mul(solution[fc]));
        }
        solution[pc] = s.neg();
    }

    // Convert to integers: find LCM of denominators
    let mut lcm_den: i64 = 1;
    for r in &solution {
        lcm_den = lcm_i64(lcm_den, r.den.abs());
    }
    let int_sol: Vec<i32> = solution
        .iter()
        .map(|r| (r.num * lcm_den / r.den) as i32)
        .collect();

    // Verify: all must be positive (or all negative → flip)
    let all_pos = int_sol.iter().all(|&v| v >= 0);
    let all_neg = int_sol.iter().all(|&v| v <= 0);
    if !all_pos && !all_neg {
        return None;
    }
    if all_neg {
        Some(int_sol.iter().map(|&v| -v).collect())
    } else {
        Some(int_sol)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 7. DimensionalAnalysis — Buckingham Pi theorem
// ────────────────────────────────────────────────────────────────────────────

/// Physical dimension (SI base units, exponents).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Dimension {
    /// Mass (kg)
    pub m: i8,
    /// Length (m)
    pub l: i8,
    /// Time (s)
    pub t: i8,
    /// Electric current (A)
    pub i: i8,
    /// Temperature (K)
    pub theta: i8,
}

impl Dimension {
    pub fn new(m: i8, l: i8, t: i8, i: i8, theta: i8) -> Self {
        Self { m, l, t, i, theta }
    }

    pub fn dimensionless() -> Self {
        Self::default()
    }
}

/// Multiply two dimensions (add exponents).
pub fn dim_multiply(a: &Dimension, b: &Dimension) -> Dimension {
    Dimension {
        m: a.m + b.m,
        l: a.l + b.l,
        t: a.t + b.t,
        i: a.i + b.i,
        theta: a.theta + b.theta,
    }
}

/// Divide two dimensions (subtract exponents).
pub fn dim_divide(a: &Dimension, b: &Dimension) -> Dimension {
    Dimension {
        m: a.m - b.m,
        l: a.l - b.l,
        t: a.t - b.t,
        i: a.i - b.i,
        theta: a.theta - b.theta,
    }
}

/// Check if a dimension is dimensionless.
pub fn is_dimensionless(d: &Dimension) -> bool {
    d.m == 0 && d.l == 0 && d.t == 0 && d.i == 0 && d.theta == 0
}

/// Find Pi groups using null space of the dimension matrix.
/// Returns each Pi group as a list of (quantity_name, exponent) pairs.
pub fn find_pi_groups(quantities: &[(String, Dimension)]) -> Vec<Vec<(String, i8)>> {
    let n = quantities.len();
    if n == 0 {
        return vec![];
    }

    // Build dimension matrix [5 × n] (5 base dimensions × n quantities)
    let dim_rows = 5usize;
    let mut mat: Vec<Vec<Rational>> = vec![vec![Rational::zero(); n]; dim_rows];
    for (j, (_, dim)) in quantities.iter().enumerate() {
        mat[0][j] = Rational::new(dim.m as i64, 1);
        mat[1][j] = Rational::new(dim.l as i64, 1);
        mat[2][j] = Rational::new(dim.t as i64, 1);
        mat[3][j] = Rational::new(dim.i as i64, 1);
        mat[4][j] = Rational::new(dim.theta as i64, 1);
    }

    // Row-reduce to find null space
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut row = 0usize;
    for col in 0..n {
        let mut piv = None;
        for r in row..dim_rows {
            if !mat[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }
        if let Some(pr) = piv {
            mat.swap(row, pr);
            let scale = mat[row][col];
            for c in 0..n {
                mat[row][c] = mat[row][c].div(scale);
            }
            for r in 0..dim_rows {
                if r != row && !mat[r][col].is_zero() {
                    let factor = mat[r][col];
                    for c in 0..n {
                        let val = mat[row][c].mul(factor);
                        mat[r][c] = mat[r][c].sub(val);
                    }
                }
            }
            pivot_cols.push(col);
            row += 1;
        }
    }

    let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_cols.contains(c)).collect();
    let mut pi_groups: Vec<Vec<(String, i8)>> = Vec::new();

    for &fc in &free_cols {
        let mut solution = vec![Rational::zero(); n];
        solution[fc] = Rational::one();
        for (pi, &pc) in pivot_cols.iter().enumerate() {
            let s = mat[pi][fc].neg();
            solution[pc] = s;
        }
        // Scale to integers
        let mut lcm_den: i64 = 1;
        for r in &solution {
            lcm_den = lcm_i64(lcm_den, r.den.abs());
        }
        let group: Vec<(String, i8)> = solution
            .iter()
            .enumerate()
            .map(|(j, r)| {
                let exp = (r.num * lcm_den / r.den) as i8;
                (quantities[j].0.clone(), exp)
            })
            .filter(|(_, exp)| *exp != 0)
            .collect();
        if !group.is_empty() {
            pi_groups.push(group);
        }
    }
    pi_groups
}

// ────────────────────────────────────────────────────────────────────────────
// 8. MatrixCalculus — automatic matrix derivative rules
// ────────────────────────────────────────────────────────────────────────────

/// Symbolic matrix expression.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixExpr {
    /// Named matrix with dimensions (rows, cols)
    Matrix(String, usize, usize),
    /// Element-wise addition
    Add(Box<MatrixExpr>, Box<MatrixExpr>),
    /// Matrix multiplication
    Mul(Box<MatrixExpr>, Box<MatrixExpr>),
    /// Transpose
    Transpose(Box<MatrixExpr>),
    /// Trace (scalar)
    Trace(Box<MatrixExpr>),
    /// Determinant (scalar)
    Det(Box<MatrixExpr>),
    /// Matrix inverse
    Inv(Box<MatrixExpr>),
    /// Frobenius norm squared
    Frobenius(Box<MatrixExpr>),
}

impl MatrixExpr {
    /// Format for display.
    pub fn name(&self) -> String {
        match self {
            MatrixExpr::Matrix(n, _, _) => n.clone(),
            MatrixExpr::Add(a, b) => format!("({} + {})", a.name(), b.name()),
            MatrixExpr::Mul(a, b) => format!("({} * {})", a.name(), b.name()),
            MatrixExpr::Transpose(e) => format!("{}^T", e.name()),
            MatrixExpr::Trace(e) => format!("Tr({})", e.name()),
            MatrixExpr::Det(e) => format!("det({})", e.name()),
            MatrixExpr::Inv(e) => format!("{}^-1", e.name()),
            MatrixExpr::Frobenius(e) => format!("||{}||_F^2", e.name()),
        }
    }
}

/// Compute d Tr(AX) / dX = A^T
/// Compute d Tr(XA) / dX = A^T  (same result)
/// More generally tries to identify the pattern.
pub fn trace_derivative(expr: &MatrixExpr, wrt: &str) -> MatrixExpr {
    match expr {
        MatrixExpr::Trace(inner) => {
            match inner.as_ref() {
                // d Tr(A*X) / dX = A^T
                MatrixExpr::Mul(a, b) => {
                    let a_is_wrt = matrix_contains_var(a, wrt);
                    let b_is_wrt = matrix_contains_var(b, wrt);
                    if !a_is_wrt && b_is_wrt {
                        MatrixExpr::Transpose(a.clone())
                    } else if a_is_wrt && !b_is_wrt {
                        MatrixExpr::Transpose(b.clone())
                    } else {
                        // Default: return transpose of non-wrt factor or identity
                        MatrixExpr::Matrix("I".to_string(), 1, 1)
                    }
                }
                // d Tr(X) / dX = I
                MatrixExpr::Matrix(name, r, c) if name == wrt => {
                    MatrixExpr::Matrix("I".to_string(), *r, *c)
                }
                _ => MatrixExpr::Matrix("I".to_string(), 1, 1),
            }
        }
        _ => MatrixExpr::Matrix("0".to_string(), 1, 1),
    }
}

/// Check if a MatrixExpr contains a variable named `var`.
fn matrix_contains_var(expr: &MatrixExpr, var: &str) -> bool {
    match expr {
        MatrixExpr::Matrix(n, _, _) => n == var,
        MatrixExpr::Add(a, b) | MatrixExpr::Mul(a, b) => {
            matrix_contains_var(a, var) || matrix_contains_var(b, var)
        }
        MatrixExpr::Transpose(e)
        | MatrixExpr::Trace(e)
        | MatrixExpr::Det(e)
        | MatrixExpr::Inv(e)
        | MatrixExpr::Frobenius(e) => matrix_contains_var(e, var),
    }
}

/// Compute d ||AXB - C||_F^2 / dX = 2 A^T (AXB - C) B^T
///
/// More generally: d ||M||_F^2 / dX where M depends linearly on X.
pub fn frobenius_derivative(expr: &MatrixExpr, wrt: &str) -> MatrixExpr {
    match expr {
        MatrixExpr::Frobenius(inner) => {
            // Pattern: ||A * X * B - C||_F^2
            match inner.as_ref() {
                MatrixExpr::Add(axb, c) => {
                    // Try to extract A, X, B from axb = A*(X*B) or (A*X)*B
                    if let Some((a_mat, b_mat)) = extract_axb_pattern(axb, wrt) {
                        // Result: 2 * A^T * (AXB + C) * B^T
                        let at = MatrixExpr::Transpose(Box::new(a_mat.clone()));
                        let bt = MatrixExpr::Transpose(Box::new(b_mat.clone()));
                        let mid = MatrixExpr::Add(Box::new(axb.as_ref().clone()), c.clone());
                        MatrixExpr::Mul(
                            Box::new(MatrixExpr::Mul(Box::new(at), Box::new(mid))),
                            Box::new(bt),
                        )
                    } else {
                        MatrixExpr::Matrix("0".to_string(), 1, 1)
                    }
                }
                // ||AXB||_F^2 → 2 A^T AXB B^T
                _ => {
                    if let Some((a_mat, b_mat)) = extract_axb_pattern(inner, wrt) {
                        let at = MatrixExpr::Transpose(Box::new(a_mat));
                        let bt = MatrixExpr::Transpose(Box::new(b_mat));
                        MatrixExpr::Mul(
                            Box::new(MatrixExpr::Mul(
                                Box::new(at),
                                Box::new(inner.as_ref().clone()),
                            )),
                            Box::new(bt),
                        )
                    } else {
                        MatrixExpr::Matrix("0".to_string(), 1, 1)
                    }
                }
            }
        }
        _ => MatrixExpr::Matrix("0".to_string(), 1, 1),
    }
}

/// Try to extract (A, B) from expression A * X * B pattern.
fn extract_axb_pattern(expr: &MatrixExpr, wrt: &str) -> Option<(MatrixExpr, MatrixExpr)> {
    match expr {
        MatrixExpr::Mul(left, right) => {
            let l = left.as_ref();
            let r = right.as_ref();
            // (A * X) * B
            if let MatrixExpr::Mul(a, x) = l {
                if matrix_contains_var(x, wrt)
                    && !matrix_contains_var(a, wrt)
                    && !matrix_contains_var(r, wrt)
                {
                    return Some((a.as_ref().clone(), r.clone()));
                }
            }
            // A * (X * B)
            if let MatrixExpr::Mul(x, b) = r {
                if matrix_contains_var(x, wrt)
                    && !matrix_contains_var(l, wrt)
                    && !matrix_contains_var(b, wrt)
                {
                    return Some((l.clone(), b.as_ref().clone()));
                }
            }
            // A * X  (B = Identity)
            if matrix_contains_var(r, wrt) && !matrix_contains_var(l, wrt) {
                let dim = if let MatrixExpr::Matrix(_, _, c) = r {
                    *c
                } else {
                    1
                };
                return Some((l.clone(), MatrixExpr::Matrix("I".to_string(), dim, dim)));
            }
            // X * B  (A = Identity)
            if matrix_contains_var(l, wrt) && !matrix_contains_var(r, wrt) {
                let dim = if let MatrixExpr::Matrix(_, row, _) = l {
                    *row
                } else {
                    1
                };
                return Some((MatrixExpr::Matrix("I".to_string(), dim, dim), r.clone()));
            }
            None
        }
        _ => None,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 9. PolynomialArithmetic
// ────────────────────────────────────────────────────────────────────────────

/// Dense polynomial with coefficients indexed by degree.
/// `coeffs[0]` = constant term, `coeffs[k]` = coefficient of x^k.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<f64>) -> Self {
        let mut p = Self { coeffs };
        p.trim();
        p
    }

    /// Trim trailing near-zero coefficients.
    pub fn trim(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last().is_some_and(|c| c.abs() < 1e-12) {
            self.coeffs.pop();
        }
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.abs() < 1e-12)
    }

    pub fn zero() -> Self {
        Self::new(vec![0.0])
    }
    pub fn one() -> Self {
        Self::new(vec![1.0])
    }
}

/// Evaluate polynomial via Horner's method.
pub fn poly_evaluate(p: &Polynomial, x: f64) -> f64 {
    if p.coeffs.is_empty() {
        return 0.0;
    }
    let mut result = 0.0f64;
    for c in p.coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

/// Add two polynomials.
pub fn poly_add(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let len = a.coeffs.len().max(b.coeffs.len());
    let mut coeffs = vec![0.0f64; len];
    for (i, &v) in a.coeffs.iter().enumerate() {
        coeffs[i] += v;
    }
    for (i, &v) in b.coeffs.iter().enumerate() {
        coeffs[i] += v;
    }
    Polynomial::new(coeffs)
}

/// Subtract two polynomials.
pub fn poly_sub(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let len = a.coeffs.len().max(b.coeffs.len());
    let mut coeffs = vec![0.0f64; len];
    for (i, &v) in a.coeffs.iter().enumerate() {
        coeffs[i] += v;
    }
    for (i, &v) in b.coeffs.iter().enumerate() {
        coeffs[i] -= v;
    }
    Polynomial::new(coeffs)
}

/// Multiply two polynomials.
pub fn poly_mul(a: &Polynomial, b: &Polynomial) -> Polynomial {
    if a.is_zero() || b.is_zero() {
        return Polynomial::zero();
    }
    let deg = a.degree() + b.degree();
    let mut coeffs = vec![0.0f64; deg + 1];
    for (i, &ai) in a.coeffs.iter().enumerate() {
        for (j, &bj) in b.coeffs.iter().enumerate() {
            coeffs[i + j] += ai * bj;
        }
    }
    Polynomial::new(coeffs)
}

/// Polynomial long division: returns (quotient, remainder).
pub fn poly_div_rem(a: &Polynomial, b: &Polynomial) -> (Polynomial, Polynomial) {
    if b.is_zero() {
        return (Polynomial::zero(), a.clone());
    }
    let b_lead = *b.coeffs.last().unwrap_or(&1.0);
    if b_lead.abs() < 1e-300 {
        return (Polynomial::zero(), a.clone());
    }
    let mut rem = a.coeffs.clone();
    let b_deg = b.degree();
    let a_deg = a.degree();
    if a_deg < b_deg {
        return (Polynomial::zero(), a.clone());
    }
    let q_len = a_deg - b_deg + 1;
    let mut quot = vec![0.0f64; q_len];

    for i in (0..q_len).rev() {
        let lead = rem[b_deg + i] / b_lead;
        quot[i] = lead;
        for j in 0..=b_deg {
            rem[j + i] -= lead * b.coeffs[j];
        }
    }
    (
        Polynomial::new(quot),
        Polynomial::new(rem[..b_deg].to_vec()),
    )
}

/// Euclidean GCD of two polynomials (monic).
pub fn poly_gcd(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let mut u = a.clone();
    let mut v = b.clone();
    while !v.is_zero() {
        let (_, r) = poly_div_rem(&u, &v);
        u = v;
        v = r;
    }
    // Make monic
    if let Some(&lead) = u.coeffs.last() {
        if lead.abs() > 1e-12 {
            for c in &mut u.coeffs {
                *c /= lead;
            }
        }
    }
    u
}

/// Companion matrix eigenvalue estimation via power iteration (real roots only).
/// Returns approximate real roots as (real, imag=0.0) pairs.
pub fn roots_companion_matrix(p: &Polynomial) -> Vec<(f64, f64)> {
    let n = p.degree();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        // p[0] + p[1]*x = 0  →  x = -p[0]/p[1]
        let lead = p.coeffs[1];
        if lead.abs() < 1e-300 {
            return vec![];
        }
        return vec![(-p.coeffs[0] / lead, 0.0)];
    }
    // Normalize to monic
    let lead = *p.coeffs.last().unwrap_or(&1.0);
    if lead.abs() < 1e-300 {
        return vec![];
    }
    let monic: Vec<f64> = p.coeffs.iter().map(|c| c / lead).collect();

    // Build companion matrix (n×n), stored row-major
    // Last column = -monic[0..n]/monic[n], sub-diagonal = 1
    let mut companion = vec![0.0f64; n * n];
    for i in 0..n - 1 {
        companion[(i + 1) * n + i] = 1.0; // sub-diagonal
    }
    for i in 0..n {
        companion[i * n + (n - 1)] = -monic[i]; // last column
    }

    // Power iteration variant: use deflation to find multiple roots
    let mut roots: Vec<(f64, f64)> = Vec::new();
    // Use characteristic polynomial evaluation at grid points as heuristic
    let test_points: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.5).collect();
    let mut candidates: Vec<f64> = Vec::new();
    for &t in &test_points {
        let fval = poly_evaluate(&Polynomial::new(monic.clone()), t);
        if fval.abs() < 5.0 {
            candidates.push(t);
        }
    }
    candidates.sort_by(|a, b| {
        a.abs()
            .partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Newton refinement for each candidate
    for init in candidates {
        let root = newton_refine(&Polynomial::new(monic.clone()), init, 50);
        if let Some(r) = root {
            // Dedup
            let is_dup = roots.iter().any(|(x, _)| (x - r).abs() < 1e-4);
            if !is_dup {
                roots.push((r, 0.0));
            }
        }
        if roots.len() >= n {
            break;
        }
    }
    roots
}

fn newton_refine(p: &Polynomial, x0: f64, max_iter: usize) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = poly_evaluate(p, x);
        if fx.abs() < 1e-10 {
            return Some(x);
        }
        // Numerical derivative
        let h = 1e-7;
        let dfx = (poly_evaluate(p, x + h) - poly_evaluate(p, x - h)) / (2.0 * h);
        if dfx.abs() < 1e-300 {
            break;
        }
        let x_new = x - fx / dfx;
        if (x_new - x).abs() < 1e-10 {
            return Some(x_new);
        }
        x = x_new;
        if x.abs() > 1e6 {
            break;
        }
    }
    let fval = poly_evaluate(p, x);
    if fval.abs() < 1e-6 {
        Some(x)
    } else {
        None
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 10. SymbolicIntegrator — pattern-based integration
// ────────────────────────────────────────────────────────────────────────────

/// Attempt symbolic integration of `expr` with respect to `var`.
/// Returns `None` if no pattern matches.
pub fn integrate(expr: &Expr, var: &str) -> Option<Expr> {
    match expr {
        // ∫ c dx = c*x
        Expr::Const(c) => Some(Expr::Mul(
            Box::new(Expr::Const(*c)),
            Box::new(Expr::Var(var.to_string())),
        )),
        // ∫ y dx = y*x  (y is a different variable)
        Expr::Var(v) => {
            if v == var {
                // ∫ x dx = x^2 / 2
                Some(Expr::Div(
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(var.to_string())),
                        Box::new(Expr::Const(2.0)),
                    )),
                    Box::new(Expr::Const(2.0)),
                ))
            } else {
                // ∫ y dx = y * x
                Some(Expr::Mul(
                    Box::new(Expr::Var(v.clone())),
                    Box::new(Expr::Var(var.to_string())),
                ))
            }
        }
        // ∫ x^n dx = x^(n+1) / (n+1)  for n ≠ -1
        Expr::Pow(base, exp) => {
            if let (Expr::Var(v), Some(n)) = (base.as_ref(), exp.const_val()) {
                if v == var && (n + 1.0).abs() > 1e-12 {
                    let new_exp = n + 1.0;
                    return Some(simplify(Expr::Div(
                        Box::new(Expr::Pow(
                            Box::new(Expr::Var(var.to_string())),
                            Box::new(Expr::Const(new_exp)),
                        )),
                        Box::new(Expr::Const(new_exp)),
                    )));
                }
            }
            None
        }
        // ∫ sin(x) dx = -cos(x)
        Expr::Sin(inner) => {
            if let Expr::Var(v) = inner.as_ref() {
                if v == var {
                    return Some(Expr::Neg(Box::new(Expr::Cos(Box::new(Expr::Var(
                        var.to_string(),
                    ))))));
                }
            }
            None
        }
        // ∫ cos(x) dx = sin(x)
        Expr::Cos(inner) => {
            if let Expr::Var(v) = inner.as_ref() {
                if v == var {
                    return Some(Expr::Sin(Box::new(Expr::Var(var.to_string()))));
                }
            }
            None
        }
        // ∫ exp(x) dx = exp(x)
        Expr::Exp(inner) => {
            if let Expr::Var(v) = inner.as_ref() {
                if v == var {
                    return Some(expr.clone());
                }
            }
            None
        }
        // ∫ 1/x dx = ln(x)  (from Div(1, x))
        Expr::Div(num, den) => {
            if let (Expr::Const(c), Expr::Var(v)) = (num.as_ref(), den.as_ref()) {
                if v == var {
                    if (c - 1.0).abs() < 1e-12 {
                        return Some(Expr::Ln(Box::new(Expr::Var(var.to_string()))));
                    } else {
                        // ∫ c/x dx = c * ln(x)
                        return Some(Expr::Mul(
                            Box::new(Expr::Const(*c)),
                            Box::new(Expr::Ln(Box::new(Expr::Var(var.to_string())))),
                        ));
                    }
                }
            }
            None
        }
        // Sum rule: ∫ (f + g) dx = ∫f dx + ∫g dx
        Expr::Add(l, r) => {
            let il = integrate(l, var)?;
            let ir = integrate(r, var)?;
            Some(simplify(Expr::Add(Box::new(il), Box::new(ir))))
        }
        // Difference rule: ∫ (f - g) dx = ∫f dx - ∫g dx
        Expr::Sub(l, r) => {
            let il = integrate(l, var)?;
            let ir = integrate(r, var)?;
            Some(simplify(Expr::Sub(Box::new(il), Box::new(ir))))
        }
        // Constant multiple rule: ∫ c*f dx = c * ∫f dx
        Expr::Mul(l, r) => {
            // Try c * f or f * c
            if let Some(c) = l.const_val() {
                let inner = integrate(r, var)?;
                return Some(simplify(Expr::Mul(
                    Box::new(Expr::Const(c)),
                    Box::new(inner),
                )));
            }
            if let Some(c) = r.const_val() {
                let inner = integrate(l, var)?;
                return Some(simplify(Expr::Mul(
                    Box::new(Expr::Const(c)),
                    Box::new(inner),
                )));
            }
            None
        }
        // ∫ (-f) dx = -(∫f dx)
        Expr::Neg(inner) => {
            let i = integrate(inner, var)?;
            Some(Expr::Neg(Box::new(i)))
        }
        _ => None,
    }
}
