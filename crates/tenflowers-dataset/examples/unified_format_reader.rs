#![allow(clippy::result_large_err)]

//! Example demonstrating unified format reader abstraction
//!
//! This example shows how to use the unified format reader system to:
//! - Automatically detect data formats
//! - Read data from multiple formats with a single interface
//! - Validate schemas across different formats
//! - Perform cross-format operations

use std::path::Path;
use tenflowers_dataset::formats::registry::global;
use tenflowers_dataset::formats::{
    CrossFormatIterator, FormatConverter, FormatReaderBuilder, SchemaCompatibility,
    SchemaValidator, UnifiedBatchReader, ValidationConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified Format Reader Example ===\n");

    // 1. List all registered formats
    println!("1. Registered Formats:");
    let formats = global::list_formats();
    println!("   Available formats: {:?}", formats);

    let extensions = global::list_extensions();
    println!("   Supported extensions: {:?}\n", extensions);

    // 2. Auto-detect format from file extension
    println!("2. Format Detection:");
    let csv_path = Path::new("data.csv");
    match global::detect_format(csv_path) {
        Ok(detection) => {
            println!("   Format: {}", detection.format_name);
            println!("   Confidence: {:.2}", detection.confidence);
            println!("   Method: {:?}\n", detection.method);
        }
        Err(e) => {
            println!("   Could not detect format (file may not exist): {}\n", e);
        }
    }

    // 3. Create format reader using builder pattern
    println!("3. Format Reader Builder:");
    println!("   You can create readers with automatic format detection:");
    println!(
        "
    let reader = FormatReaderBuilder::new(\"data.json\")
        .build(&registry)?;

    // Or specify format explicitly:
    let reader = FormatReaderBuilder::new(\"data.csv\")
        .with_format(\"CSV\")
        .build(&registry)?;
    "
    );

    // 4. Schema validation example
    println!("4. Schema Validation:");
    println!("   Validators can check schema compatibility:");
    let config = ValidationConfig {
        strict_types: false,
        enforce_field_order: false,
        allow_nullable: true,
        validate_shapes: true,
        max_fields: Some(100),
    };
    let validator = SchemaValidator::with_config(config);
    println!("   Validator configured with flexible type matching\n");

    // 5. Cross-format operations
    println!("5. Cross-Format Operations:");
    println!("   The system supports:");
    println!("   - CrossFormatIterator: Iterate across multiple formats");
    println!("   - UnifiedBatchReader: Read batches from any format");
    println!("   - FormatConverter: Convert between formats");
    println!("   - SchemaCompatibility: Check schema compatibility\n");

    // 6. Example workflow (pseudo-code since we don't have actual files)
    println!("6. Example Workflow:");
    println!(
        "
// Load data from multiple formats
let csv_reader = global::auto_create_reader(Path::new(\"train.csv\"))?;
let json_reader = global::auto_create_reader(Path::new(\"val.json\"))?;
let parquet_reader = global::auto_create_reader(Path::new(\"test.parquet\"))?;

// Check schema compatibility
let csv_meta = csv_reader.metadata()?;
let json_meta = json_reader.metadata()?;
let compatible = SchemaCompatibility::are_compatible(&csv_meta, &json_meta)?;

// Iterate across all formats
let readers = vec![csv_reader, json_reader, parquet_reader];
let mut iter = CrossFormatIterator::new(readers);

for sample in iter {{
    let sample = sample?;
    // Process sample (features, labels)
    println!(\"Sample {{}}: {{:?}}\", sample.source_index, sample.metadata);
}}

// Batch processing
let mut batch_reader = UnifiedBatchReader::new(reader, 32);
while let Some(batch) = batch_reader.next_batch()? {{
    let (features, labels) = FormatConverter::samples_to_tensors(&batch)?;
    // Train model with batch
}}
    "
    );

    println!("\n=== Features Demonstrated ===");
    println!("✓ Automatic format detection");
    println!("✓ Unified reader interface");
    println!("✓ Schema validation and compatibility checking");
    println!("✓ Cross-format iteration");
    println!("✓ Batch processing");
    println!("✓ Format conversion utilities");

    Ok(())
}
