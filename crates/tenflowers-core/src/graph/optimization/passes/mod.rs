//! Core optimization passes and trait definitions
//!
//! This module provides the fundamental optimization pass trait and basic passes
//! like constant folding, common subexpression elimination, and dead code elimination.

pub mod algebraic;
pub mod constant_folding;
pub mod cse;
pub mod dead_code;
pub mod pass_support;
pub mod scheduling;
pub mod strength_reduction;

// Re-export all types
pub use algebraic::*;
pub use constant_folding::*;
pub use cse::*;
pub use dead_code::*;
pub use pass_support::*;
pub use scheduling::*;
pub use strength_reduction::*;

#[cfg(test)]
mod tests;
