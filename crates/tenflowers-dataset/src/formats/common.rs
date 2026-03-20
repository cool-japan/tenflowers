//! Common utilities and types for data formats

use std::collections::HashMap;

/// Strategy for handling missing values in dataset
#[derive(Debug, Clone, Default)]
pub enum MissingValueStrategy {
    #[default]
    /// Skip rows containing any missing values
    SkipRow,
    /// Fill missing values with a default value
    FillValue(String),
    /// Fill missing values with the mean of the column (for numeric types)
    FillMean,
    /// Forward fill (use the last valid value)
    ForwardFill,
    /// Backward fill (use the next valid value)
    BackwardFill,
}

/// Naming pattern for files in datasets
#[derive(Debug, Clone, Default)]
pub enum NamingPattern {
    /// Use directory name as class name
    #[default]
    DirectoryAsClass,
    /// Extract class from filename prefix (e.g., "class_001.jpg" -> "class")
    FilenamePrefix(String),
    /// Extract class from filename suffix (e.g., "001_class.jpg" -> "class")
    FilenameSuffix(String),
    /// Use custom mapping from directory/filename to class
    CustomMapping(HashMap<String, usize>),
}
