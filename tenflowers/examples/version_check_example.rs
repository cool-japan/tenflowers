//! Demonstrates `tenflowers::version_check::subcrate_versions()` and
//! `assert_versions_consistent()`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example version_check_example -p tenflowers
//! ```

use tenflowers::version_check::{
    assert_versions_consistent, check_version_consistency, subcrate_versions,
};

fn main() {
    println!("=== TenfloweRS: version check example ===\n");

    // ── Enumerate all subcrate versions ──────────────────────────────────────
    println!("Subcrate versions:");
    for sv in subcrate_versions() {
        println!("  {:<30} v{}", sv.name, sv.version);
    }

    // ── Consistency report ────────────────────────────────────────────────────
    let report = check_version_consistency();
    println!("\nConsistency report:");
    println!("  reference version: {}", report.reference_version);
    println!("  is_consistent:     {}", report.is_consistent);
    println!("  summary:           {}", report.summary());

    if !report.mismatches.is_empty() {
        println!("\nMismatches:");
        for m in &report.mismatches {
            println!(
                "  {} (expected {} got {})",
                m.name, report.reference_version, m.version
            );
        }
    }

    // ── Assert — panics in debug builds if any version drifts ────────────────
    assert_versions_consistent();
    println!("\nassert_versions_consistent() passed — all subcrates are in sync.");
}
