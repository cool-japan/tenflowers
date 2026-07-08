//! Advanced profiling tool integration for TenfloweRS FFI.
//!
//! This module provides a session-based profiler that records operation
//! timings, memory usage, and device information. Actual measurement is the
//! caller's responsibility — the profiler stores whatever it is told, so it
//! can be wired to any instrumentation source.
//!
//! # Python Usage
//!
//! ```python
//! import tenflowers as tf
//!
//! profiler = tf.PyProfiler()
//! profiler.start_session()
//! profiler.record("matmul", 12_000, 1_048_576)   # op_name, ns, bytes
//! profiler.record("relu",   3_000,  524_288)
//! report = profiler.end_session()
//!
//! print(f"total ops: {report.total_ops}")
//! print(f"total time: {report.total_time_ns} ns")
//! for rec in report.top_ops(1):
//!     print(rec.op_name, rec.duration_ns)
//! ```

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

// ─── Data types ───────────────────────────────────────────────────────────────

/// A single profiling record for one operation.
///
/// Stores the operation name, wall-clock duration (nanoseconds), memory
/// consumed, and the device on which the operation ran.
#[derive(Debug, Clone)]
pub struct ProfileRecord {
    /// Human-readable name of the operation (e.g. `"matmul"`, `"relu"`).
    pub op_name: String,
    /// Wall-clock duration in nanoseconds.
    pub duration_ns: u64,
    /// Memory allocated / consumed for this operation in bytes.
    pub memory_bytes: usize,
    /// Device identifier string (e.g. `"cpu"`, `"gpu:0"`).
    pub device: String,
}

// ─── Python-visible record ────────────────────────────────────────────────────

/// Python-facing view of a single [`ProfileRecord`].
#[pyclass(name = "PyProfileRecord")]
#[derive(Debug, Clone)]
pub struct PyProfileRecord {
    /// Operation name.
    #[pyo3(get)]
    pub op_name: String,
    /// Duration in nanoseconds.
    #[pyo3(get)]
    pub duration_ns: u64,
    /// Memory bytes consumed.
    #[pyo3(get)]
    pub memory_bytes: usize,
    /// Device string.
    #[pyo3(get)]
    pub device: String,
}

impl From<&ProfileRecord> for PyProfileRecord {
    fn from(r: &ProfileRecord) -> Self {
        Self {
            op_name: r.op_name.clone(),
            duration_ns: r.duration_ns,
            memory_bytes: r.memory_bytes,
            device: r.device.clone(),
        }
    }
}

#[pymethods]
impl PyProfileRecord {
    fn __repr__(&self) -> String {
        format!(
            "PyProfileRecord(op_name='{}', duration_ns={}, memory_bytes={}, device='{}')",
            self.op_name, self.duration_ns, self.memory_bytes, self.device
        )
    }
}

// ─── Session state ────────────────────────────────────────────────────────────

/// Tracks whether a profiling session is active and its accumulated records.
#[derive(Debug, Default)]
struct SessionState {
    active: bool,
    records: Vec<ProfileRecord>,
}

// ─── PyProfiler ───────────────────────────────────────────────────────────────

/// Thread-safe, session-based operation profiler.
///
/// Call [`start_session`] before recording operations and [`end_session`] to
/// retrieve a [`PyProfileReport`] containing aggregated statistics.
///
/// The profiler does **not** perform any timing itself — callers are expected
/// to measure durations externally and pass them to [`record`].
#[pyclass(name = "PyProfiler")]
pub struct PyProfiler {
    state: Arc<Mutex<SessionState>>,
}

#[pymethods]
impl PyProfiler {
    /// Create a new, idle profiler.
    #[new]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(SessionState::default())),
        }
    }

    /// Begin a new profiling session, discarding any previously recorded data.
    ///
    /// Returns an error if a session is already active.
    pub fn start_session(&self) -> PyResult<()> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("profiler lock poisoned: {e}")))?;
        if guard.active {
            return Err(PyRuntimeError::new_err(
                "profiling session is already active; call end_session() first",
            ));
        }
        guard.active = true;
        guard.records.clear();
        Ok(())
    }

    /// End the active profiling session and return a [`PyProfileReport`].
    ///
    /// Returns an error if no session is active.
    pub fn end_session(&self) -> PyResult<PyProfileReport> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("profiler lock poisoned: {e}")))?;
        if !guard.active {
            return Err(PyRuntimeError::new_err(
                "no active profiling session; call start_session() first",
            ));
        }
        guard.active = false;
        let report = build_report(&guard.records);
        guard.records.clear();
        Ok(report)
    }

    /// Record a single operation.
    ///
    /// # Arguments
    ///
    /// * `op_name`      — name of the operation
    /// * `duration_ns`  — measured duration in nanoseconds
    /// * `memory_bytes` — memory consumed by the operation in bytes
    ///
    /// The device defaults to `"cpu"`. Use [`record_on_device`] to specify one
    /// explicitly.
    ///
    /// Returns an error if called outside an active session.
    #[pyo3(signature = (op_name, duration_ns, memory_bytes))]
    pub fn record(&self, op_name: String, duration_ns: u64, memory_bytes: usize) -> PyResult<()> {
        self.record_on_device(op_name, duration_ns, memory_bytes, "cpu".to_string())
    }

    /// Record a single operation with an explicit device identifier.
    #[pyo3(signature = (op_name, duration_ns, memory_bytes, device))]
    pub fn record_on_device(
        &self,
        op_name: String,
        duration_ns: u64,
        memory_bytes: usize,
        device: String,
    ) -> PyResult<()> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("profiler lock poisoned: {e}")))?;
        if !guard.active {
            return Err(PyRuntimeError::new_err(
                "cannot record: no active profiling session",
            ));
        }
        guard.records.push(ProfileRecord {
            op_name,
            duration_ns,
            memory_bytes,
            device,
        });
        Ok(())
    }

    /// Return the number of records in the current session without ending it.
    pub fn current_record_count(&self) -> PyResult<usize> {
        let guard = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("profiler lock poisoned: {e}")))?;
        Ok(guard.records.len())
    }

    /// Return `True` if a session is currently active.
    pub fn is_active(&self) -> PyResult<bool> {
        let guard = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("profiler lock poisoned: {e}")))?;
        Ok(guard.active)
    }

    fn __repr__(&self) -> String {
        let active = self.state.lock().map(|g| g.active).unwrap_or(false);
        format!("PyProfiler(active={})", active)
    }
}

// ─── Report ───────────────────────────────────────────────────────────────────

/// Aggregated profiling statistics produced by [`PyProfiler::end_session`].
#[pyclass(name = "PyProfileReport")]
#[derive(Debug, Clone)]
pub struct PyProfileReport {
    /// Total number of operations recorded.
    #[pyo3(get)]
    pub total_ops: usize,
    /// Sum of all recorded durations in nanoseconds.
    #[pyo3(get)]
    pub total_time_ns: u64,
    /// Maximum single-operation memory footprint observed.
    #[pyo3(get)]
    pub peak_memory_bytes: usize,
    /// All records, preserved for detailed inspection.
    records: Vec<ProfileRecord>,
}

#[pymethods]
impl PyProfileReport {
    /// Return the `n` operations with the highest cumulative duration.
    ///
    /// Operations are grouped by name; the returned list is sorted from most
    /// expensive to least expensive.
    pub fn top_ops(&self, n: usize) -> Vec<PyProfileRecord> {
        use std::collections::HashMap;

        // Accumulate totals per op name.
        let mut totals: HashMap<&str, (u64, usize, &str)> = HashMap::new();
        for rec in &self.records {
            let entry = totals
                .entry(rec.op_name.as_str())
                .or_insert((0, 0, rec.device.as_str()));
            entry.0 += rec.duration_ns;
            if rec.memory_bytes > entry.1 {
                entry.1 = rec.memory_bytes;
            }
        }

        // Sort descending by total duration.
        let mut sorted: Vec<(&str, u64, usize, &str)> = totals
            .into_iter()
            .map(|(name, (dur, mem, dev))| (name, dur, mem, dev))
            .collect();
        sorted.sort_by_key(|entry| std::cmp::Reverse(entry.1));

        sorted
            .into_iter()
            .take(n)
            .map(|(name, dur, mem, dev)| PyProfileRecord {
                op_name: name.to_string(),
                duration_ns: dur,
                memory_bytes: mem,
                device: dev.to_string(),
            })
            .collect()
    }

    /// Average duration per operation in nanoseconds.
    pub fn average_duration_ns(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.total_time_ns as f64 / self.total_ops as f64
        }
    }

    /// Total memory footprint across all recorded operations, in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.records.iter().map(|r| r.memory_bytes).sum()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProfileReport(total_ops={}, total_time_ns={}, peak_memory_bytes={})",
            self.total_ops, self.total_time_ns, self.peak_memory_bytes
        )
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn build_report(records: &[ProfileRecord]) -> PyProfileReport {
    let total_ops = records.len();
    let total_time_ns = records.iter().map(|r| r.duration_ns).sum();
    let peak_memory_bytes = records.iter().map(|r| r.memory_bytes).max().unwrap_or(0);
    PyProfileReport {
        total_ops,
        total_time_ns,
        peak_memory_bytes,
        records: records.to_vec(),
    }
}

// ─── Registration helper ──────────────────────────────────────────────────────

/// Register profiling classes and functions into the given Python module.
pub fn register_profiling_classes(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProfileRecord>()?;
    m.add_class::<PyProfiler>()?;
    m.add_class::<PyProfileReport>()?;
    let _ = py; // kept for API symmetry with other register_* fns
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_records() -> Vec<ProfileRecord> {
        vec![
            ProfileRecord {
                op_name: "matmul".to_string(),
                duration_ns: 10_000,
                memory_bytes: 2_048,
                device: "cpu".to_string(),
            },
            ProfileRecord {
                op_name: "relu".to_string(),
                duration_ns: 3_000,
                memory_bytes: 1_024,
                device: "cpu".to_string(),
            },
            ProfileRecord {
                op_name: "matmul".to_string(),
                duration_ns: 8_000,
                memory_bytes: 2_048,
                device: "cpu".to_string(),
            },
        ]
    }

    #[test]
    fn test_build_report_aggregates_correctly() {
        let records = make_records();
        let report = build_report(&records);
        assert_eq!(report.total_ops, 3);
        assert_eq!(report.total_time_ns, 21_000);
        assert_eq!(report.peak_memory_bytes, 2_048);
    }

    #[test]
    fn test_top_ops_returns_sorted_by_duration() {
        let records = make_records();
        let report = build_report(&records);
        let top = report.top_ops(2);
        assert_eq!(top.len(), 2);
        // matmul has 18 000 ns total; relu has 3 000 ns
        assert_eq!(top[0].op_name, "matmul");
        assert_eq!(top[1].op_name, "relu");
    }

    #[test]
    fn test_top_ops_limited_by_n() {
        let records = make_records();
        let report = build_report(&records);
        let top = report.top_ops(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_ops_n_larger_than_set() {
        let records = make_records();
        let report = build_report(&records);
        let top = report.top_ops(100);
        // only 2 distinct op names
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_average_duration_zero_when_empty() {
        let report = build_report(&[]);
        assert_eq!(report.average_duration_ns(), 0.0);
    }

    #[test]
    fn test_average_duration_nonzero() {
        let records = make_records();
        let report = build_report(&records);
        let avg = report.average_duration_ns();
        assert!((avg - 7_000.0).abs() < 1.0);
    }

    #[test]
    fn test_total_memory_bytes() {
        let records = make_records();
        let report = build_report(&records);
        assert_eq!(report.total_memory_bytes(), 2_048 + 1_024 + 2_048);
    }

    #[test]
    fn test_profiler_new_is_inactive() {
        let profiler = PyProfiler::new();
        let guard = profiler.state.lock().unwrap();
        assert!(!guard.active);
        assert!(guard.records.is_empty());
    }

    #[test]
    fn test_profiler_session_lifecycle() {
        let profiler = PyProfiler::new();
        profiler.start_session().unwrap();
        {
            let guard = profiler.state.lock().unwrap();
            assert!(guard.active);
        }
        profiler.record("conv".to_string(), 5_000, 512).unwrap();
        let report = profiler.end_session().unwrap();
        assert_eq!(report.total_ops, 1);
        assert_eq!(report.total_time_ns, 5_000);
        // After end_session the state is cleared.
        let guard = profiler.state.lock().unwrap();
        assert!(!guard.active);
        assert!(guard.records.is_empty());
    }

    #[test]
    fn test_profiler_double_start_returns_error() {
        let profiler = PyProfiler::new();
        profiler.start_session().unwrap();
        let result = profiler.start_session();
        assert!(result.is_err());
    }

    #[test]
    fn test_profiler_record_outside_session_returns_error() {
        let profiler = PyProfiler::new();
        let result = profiler.record("op".to_string(), 1_000, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_profiler_end_session_without_start_returns_error() {
        let profiler = PyProfiler::new();
        let result = profiler.end_session();
        assert!(result.is_err());
    }
}
