//! Demonstrates the `tenflowers::logging` module: setting the log level and
//! emitting messages through the `log_*!` macros.
//!
//! Run with:
//!
//! ```text
//! cargo run --example logging_example -p tenflowers
//! # To see all levels:
//! TENFLOWERS_LOG=trace cargo run --example logging_example -p tenflowers
//! ```

use std::sync::{Arc, Mutex};
use tenflowers::logging::{
    current_log_level, init_from_env, set_log_backend, set_log_level, LogLevel,
};
use tenflowers::{log_debug, log_error, log_info, log_trace, log_warn};

fn main() {
    println!("=== TenfloweRS: logging example ===\n");

    // ── 1. Read level from the environment variable TENFLOWERS_LOG ───────────
    init_from_env();
    println!(
        "Initial log level (from env or default): {}",
        current_log_level()
    );

    // ── 2. Set level programmatically ────────────────────────────────────────
    set_log_level(LogLevel::Info);
    println!("Set level to: {}", current_log_level());

    // ── 3. Emit messages at various levels ───────────────────────────────────
    // With level=Info, ERROR/WARN/INFO will appear; DEBUG/TRACE will be suppressed.
    log_error!("This is an ERROR message");
    log_warn!("This is a WARN  message");
    log_info!("This is an INFO  message");
    log_debug!("This DEBUG message is suppressed at Info level");
    log_trace!("This TRACE message is suppressed at Info level");

    // ── 4. Install a custom backend to capture messages ───────────────────────
    println!("\n--- Custom backend ---");
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);

    set_log_backend(Some(Box::new(move |level, msg| {
        let record = format!("[{level}] {msg}");
        if let Ok(mut guard) = captured_clone.lock() {
            guard.push(record.clone());
        }
        eprintln!("{record}");
    })));

    set_log_level(LogLevel::Debug);
    log_info!("Framework initialized (custom backend)");
    log_debug!("Debug output visible with custom backend");
    log_warn!("A warning via custom backend");

    let messages = captured
        .lock()
        .expect("lock should not be poisoned")
        .clone();
    println!("Captured {} messages via custom backend.", messages.len());
    for m in &messages {
        println!("  {m}");
    }

    // ── 5. Restore default backend ────────────────────────────────────────────
    set_log_backend(None);
    set_log_level(LogLevel::Warn);
    println!("\nRestored default backend and Warn level.");
    println!("Logging example completed successfully.");
}
