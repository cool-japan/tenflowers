//! Hyperparameter space definitions.
//!
//! Provides the core types for specifying hyperparameter search spaces,
//! including categorical choices, integer ranges, continuous ranges, and
//! log-uniform ranges.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// HParamValue
// ─────────────────────────────────────────────────────────────────────────────

/// A concrete value that a hyperparameter can take.
#[derive(Debug, Clone, PartialEq)]
pub enum HParamValue {
    /// A 64-bit signed integer value.
    Int(i64),
    /// A 64-bit floating point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// A string value.
    String(String),
}

impl HParamValue {
    /// Return the value as `f64`, if this is an `Int` or `Float` variant.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            HParamValue::Float(v) => Some(*v),
            HParamValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Return the value as `i64`, if this is an `Int` variant.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            HParamValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the value as `bool`, if this is a `Bool` variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            HParamValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the value as `&str`, if this is a `String` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            HParamValue::String(v) => Some(v.as_str()),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HParamSpec
// ─────────────────────────────────────────────────────────────────────────────

/// Specification of the domain of a single hyperparameter.
#[derive(Debug, Clone)]
pub enum HParamSpec {
    /// A fixed set of candidate values (can be of mixed types).
    Categorical(Vec<HParamValue>),

    /// A range of integer values `[low, high)` with a given `step`.
    ///
    /// The discrete grid is: `low, low + step, low + 2*step, ...` up to (but
    /// not including) `high`.
    IntRange { low: i64, high: i64, step: i64 },

    /// A continuous real-valued range `[low, high)`.
    FloatRange { low: f64, high: f64 },

    /// A log-uniform range `[low, high)`.
    ///
    /// Values are sampled uniformly in log-space, i.e. `exp(uniform(ln(low),
    /// ln(high)))`.  Both `low` and `high` must be strictly positive.
    LogRange { low: f64, high: f64 },

    /// A boolean hyperparameter (equivalent to `Categorical([false, true])`).
    Bool,
}

impl HParamSpec {
    /// Return the discrete grid of values implied by this spec, if any.
    ///
    /// For `FloatRange` and `LogRange` this method returns `None` because the
    /// domain is continuous.  All other variants have a finite grid.
    pub fn discrete_values(&self) -> Option<Vec<HParamValue>> {
        match self {
            HParamSpec::Categorical(vals) => Some(vals.clone()),
            HParamSpec::IntRange { low, high, step } => {
                if *step <= 0 || *low >= *high {
                    return Some(vec![]);
                }
                let mut vals = Vec::new();
                let mut v = *low;
                while v < *high {
                    vals.push(HParamValue::Int(v));
                    v += step;
                }
                Some(vals)
            }
            HParamSpec::Bool => Some(vec![HParamValue::Bool(false), HParamValue::Bool(true)]),
            HParamSpec::FloatRange { .. } | HParamSpec::LogRange { .. } => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HParamConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Associates a named hyperparameter with its search space specification.
#[derive(Debug, Clone)]
pub struct HParamConfig {
    /// The unique name of this hyperparameter (used as key in `HParamSet`).
    pub name: String,
    /// The domain of this hyperparameter.
    pub spec: HParamSpec,
}

impl HParamConfig {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, spec: HParamSpec) -> Self {
        HParamConfig {
            name: name.into(),
            spec,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HParamSet
// ─────────────────────────────────────────────────────────────────────────────

/// A concrete assignment of values to hyperparameters for a single trial.
#[derive(Debug, Clone, Default)]
pub struct HParamSet {
    params: HashMap<String, HParamValue>,
}

impl HParamSet {
    /// Create a new, empty `HParamSet`.
    pub fn new() -> Self {
        HParamSet {
            params: HashMap::new(),
        }
    }

    /// Insert or overwrite a hyperparameter value.
    pub fn set(&mut self, name: &str, value: HParamValue) {
        self.params.insert(name.to_owned(), value);
    }

    /// Retrieve the value for the given hyperparameter name.
    pub fn get(&self, name: &str) -> Option<&HParamValue> {
        self.params.get(name)
    }

    /// Retrieve the value as `f64` (works for both `Int` and `Float` variants).
    pub fn get_float(&self, name: &str) -> Option<f64> {
        self.params.get(name)?.as_float()
    }

    /// Retrieve the value as `i64`.
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.params.get(name)?.as_int()
    }

    /// Retrieve the value as `bool`.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.params.get(name)?.as_bool()
    }

    /// Retrieve the value as `&str`.
    pub fn get_str(&self, name: &str) -> Option<&str> {
        self.params.get(name)?.as_str()
    }

    /// Return the names of all hyperparameters stored in this set.
    pub fn names(&self) -> Vec<&str> {
        self.params.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of hyperparameters in this set.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Return `true` if no hyperparameters are stored.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hparam_set_roundtrip_float() {
        let mut s = HParamSet::new();
        s.set("lr", HParamValue::Float(0.001));
        assert_eq!(s.get_float("lr"), Some(0.001));
        assert_eq!(s.get_int("lr"), None);
        assert_eq!(s.get_bool("lr"), None);
        assert_eq!(s.get_str("lr"), None);
    }

    #[test]
    fn test_hparam_set_roundtrip_int() {
        let mut s = HParamSet::new();
        s.set("batch", HParamValue::Int(32));
        assert_eq!(s.get_int("batch"), Some(32));
        // Int can also be retrieved as float
        assert_eq!(s.get_float("batch"), Some(32.0));
        assert_eq!(s.get_bool("batch"), None);
    }

    #[test]
    fn test_hparam_set_roundtrip_bool() {
        let mut s = HParamSet::new();
        s.set("dropout", HParamValue::Bool(true));
        assert_eq!(s.get_bool("dropout"), Some(true));
        assert_eq!(s.get_float("dropout"), None);
        assert_eq!(s.get_int("dropout"), None);
    }

    #[test]
    fn test_hparam_set_roundtrip_string() {
        let mut s = HParamSet::new();
        s.set("activation", HParamValue::String("relu".to_owned()));
        assert_eq!(s.get_str("activation"), Some("relu"));
        assert_eq!(s.get_float("activation"), None);
    }

    #[test]
    fn test_hparam_set_names_and_len() {
        let mut s = HParamSet::new();
        assert!(s.is_empty());
        s.set("a", HParamValue::Int(1));
        s.set("b", HParamValue::Float(2.0));
        assert_eq!(s.len(), 2);
        let mut names = s.names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_hparam_spec_discrete_categorical() {
        let spec = HParamSpec::Categorical(vec![
            HParamValue::String("adam".to_owned()),
            HParamValue::String("sgd".to_owned()),
        ]);
        let vals = spec.discrete_values().expect("should have discrete values");
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_hparam_spec_int_range_grid() {
        let spec = HParamSpec::IntRange {
            low: 0,
            high: 10,
            step: 2,
        };
        let vals = spec.discrete_values().expect("should have discrete values");
        // 0, 2, 4, 6, 8
        assert_eq!(vals.len(), 5);
        assert_eq!(vals[0], HParamValue::Int(0));
        assert_eq!(vals[4], HParamValue::Int(8));
    }

    #[test]
    fn test_hparam_spec_bool_grid() {
        let spec = HParamSpec::Bool;
        let vals = spec.discrete_values().expect("should have discrete values");
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], HParamValue::Bool(false));
        assert_eq!(vals[1], HParamValue::Bool(true));
    }

    #[test]
    fn test_hparam_spec_float_range_no_discrete() {
        let spec = HParamSpec::FloatRange {
            low: 0.0,
            high: 1.0,
        };
        assert!(spec.discrete_values().is_none());
    }

    #[test]
    fn test_hparam_spec_log_range_no_discrete() {
        let spec = HParamSpec::LogRange {
            low: 1e-5,
            high: 1e-1,
        };
        assert!(spec.discrete_values().is_none());
    }
}
