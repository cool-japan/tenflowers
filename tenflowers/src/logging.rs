//! Unified logging and diagnostic interface for TenfloweRS.
//!
//! This module provides a framework-wide configurable log level and a set of
//! macros that integrate naturally with the Rust ecosystem. No external
//! logging crate is required — the module is self-contained and adds zero
//! runtime overhead when logging is disabled.
//!
//! # Design
//!
//! - Log level is controlled by a single `AtomicU8` that can be set from any
//!   thread at any time via [`crate::logging::set_log_level`].
//! - The level is checked **before** any string formatting, so disabled log
//!   sites cost a single atomic load.
//! - Output goes to `stderr` by default.  Applications may install a custom
//!   backend with [`crate::logging::set_log_backend`].
//!
//! # Quick Start
//!
//! ```rust
//! use tenflowers::logging::{LogLevel, set_log_level};
//! // log_info! / log_debug! are #[macro_export] macros at crate root.
//! use tenflowers::{log_info, log_debug};
//!
//! // Enable INFO-level logging (default is WARN).
//! set_log_level(LogLevel::Info);
//!
//! log_info!("Framework initialised");
//! log_debug!("This only prints when level is DEBUG or TRACE");
//! ```

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// Log verbosity level (ordered from least to most verbose).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    /// Only critical errors are printed.
    Error = 0,
    /// Warnings and errors (default).
    Warn = 1,
    /// Informational messages, warnings, and errors.
    Info = 2,
    /// Verbose debug output.
    Debug = 3,
    /// Maximum verbosity — internal trace messages.
    Trace = 4,
}

impl LogLevel {
    /// Parse a string level name (case-insensitive).
    ///
    /// Returns `None` for unrecognised strings. For the standard, fallible
    /// `str::parse` form, use the [`FromStr`](std::str::FromStr) implementation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers::logging::LogLevel;
    ///
    /// assert_eq!(LogLevel::from_name("info"),  Some(LogLevel::Info));
    /// assert_eq!(LogLevel::from_name("DEBUG"), Some(LogLevel::Debug));
    /// assert_eq!(LogLevel::from_name("???"),   None);
    /// ```
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "error" => Some(Self::Error),
            "warn" | "warning" => Some(Self::Warn),
            "info" => Some(Self::Info),
            "debug" => Some(Self::Debug),
            "trace" => Some(Self::Trace),
            _ => None,
        }
    }

    /// Returns the short upper-case label for this level.
    pub fn label(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    /// Parse a level name (case-insensitive); see [`LogLevel::from_name`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| format!("unknown log level: {s:?}"))
    }
}

/// A custom log-message handler.  Receives `(level, message)`.
pub type LogBackend = Box<dyn Fn(LogLevel, &str) + Send + Sync + 'static>;

/// Global log level — stored as `u8` so it can live in a `const`.
static GLOBAL_LOG_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Warn as u8);

/// Optional custom backend.  `None` means "print to stderr".
fn global_backend() -> &'static Mutex<Option<LogBackend>> {
    static BACKEND: OnceLock<Mutex<Option<LogBackend>>> = OnceLock::new();
    BACKEND.get_or_init(|| Mutex::new(None))
}

/// Set the global log level.
///
/// Thread-safe; the new level takes effect immediately across all threads.
///
/// # Example
///
/// ```rust
/// use tenflowers::logging::{LogLevel, set_log_level, current_log_level};
///
/// set_log_level(LogLevel::Debug);
/// assert_eq!(current_log_level(), LogLevel::Debug);
/// ```
pub fn set_log_level(level: LogLevel) {
    GLOBAL_LOG_LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Returns the current global log level.
pub fn current_log_level() -> LogLevel {
    match GLOBAL_LOG_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Error,
        1 => LogLevel::Warn,
        2 => LogLevel::Info,
        3 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// Install a custom logging backend.
///
/// The backend is called for every message whose level is at or below the
/// current global log level.  Pass `None` to restore the default
/// stderr backend.
///
/// # Example
///
/// ```rust
/// use tenflowers::logging::{set_log_backend, LogLevel, set_log_level};
/// use tenflowers::log_warn;  // #[macro_export] macro lives at crate root
///
/// set_log_level(LogLevel::Warn);
/// set_log_backend(Some(Box::new(|level, msg| {
///     println!("[{level}] {msg}");
/// })));
///
/// log_warn!("custom backend installed");
/// // Restore default.
/// set_log_backend(None);
/// ```
pub fn set_log_backend(backend: Option<LogBackend>) {
    if let Ok(mut guard) = global_backend().lock() {
        *guard = backend;
    }
}

/// Emit a single log record at the given level.
///
/// Called by the `log_*!` macros — prefer those for normal use.
#[inline]
pub fn emit(level: LogLevel, message: &str) {
    if level > current_log_level() {
        return;
    }
    match global_backend().lock() {
        Ok(guard) if guard.is_some() => {
            (guard.as_ref().expect("checked is_some"))(level, message);
        }
        _ => {
            eprintln!("[tenflowers::{level}] {message}");
        }
    }
}

/// Initialize logging from the `TENFLOWERS_LOG` environment variable.
///
/// Recognised values: `error`, `warn`, `info`, `debug`, `trace` (case-insensitive).
/// If the variable is absent or unrecognised, the level is left at its
/// current value.
///
/// # Example
///
/// ```rust
/// // Usually called once at application startup:
/// tenflowers::logging::init_from_env();
/// ```
pub fn init_from_env() {
    if let Ok(val) = std::env::var("TENFLOWERS_LOG") {
        if let Some(level) = LogLevel::from_name(&val) {
            set_log_level(level);
        }
    }
}

/// Log a message at [`LogLevel::Error`].
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        $crate::logging::emit($crate::logging::LogLevel::Error, &format!($($arg)*));
    }};
}

/// Log a message at [`LogLevel::Warn`].
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{
        $crate::logging::emit($crate::logging::LogLevel::Warn, &format!($($arg)*));
    }};
}

/// Log a message at [`LogLevel::Info`].
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {{
        $crate::logging::emit($crate::logging::LogLevel::Info, &format!($($arg)*));
    }};
}

/// Log a message at [`LogLevel::Debug`].
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        $crate::logging::emit($crate::logging::LogLevel::Debug, &format!($($arg)*));
    }};
}

/// Log a message at [`LogLevel::Trace`].
#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => {{
        $crate::logging::emit($crate::logging::LogLevel::Trace, &format!($($arg)*));
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn reset_level() {
        set_log_level(LogLevel::Warn);
    }

    #[test]
    fn test_set_and_get_level() {
        set_log_level(LogLevel::Info);
        assert_eq!(current_log_level(), LogLevel::Info);
        reset_level();
    }

    #[test]
    fn test_level_ordering() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Trace);
    }

    #[test]
    fn test_level_from_str() {
        assert_eq!(LogLevel::from_name("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_name("WARN"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_name("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::from_name("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_name("???"), None);
    }

    #[test]
    fn test_level_display() {
        assert_eq!(LogLevel::Error.to_string(), "ERROR");
        assert_eq!(LogLevel::Debug.to_string(), "DEBUG");
    }

    #[test]
    fn test_emit_below_threshold_skipped() {
        let count = Arc::new(Mutex::new(0u32));
        let count_clone = Arc::clone(&count);
        set_log_level(LogLevel::Warn);
        set_log_backend(Some(Box::new(move |_, _| {
            *count_clone.lock().unwrap() += 1;
        })));
        emit(LogLevel::Debug, "this should be suppressed");
        assert_eq!(*count.lock().unwrap(), 0);
        set_log_backend(None);
        reset_level();
    }

    #[test]
    fn test_emit_at_threshold_fires() {
        let recorded = Arc::new(Mutex::new(Vec::<String>::new()));
        let rec_clone = Arc::clone(&recorded);
        set_log_level(LogLevel::Info);
        set_log_backend(Some(Box::new(move |_, msg: &str| {
            rec_clone.lock().unwrap().push(msg.to_owned());
        })));
        emit(LogLevel::Info, "hello from test");
        let msgs = recorded.lock().unwrap().clone();
        assert!(msgs.iter().any(|m| m.contains("hello from test")));
        set_log_backend(None);
        reset_level();
    }

    #[test]
    fn test_init_from_env_no_variable() {
        // Should not panic even if TENFLOWERS_LOG is absent.
        std::env::remove_var("TENFLOWERS_LOG");
        init_from_env();
    }
}
