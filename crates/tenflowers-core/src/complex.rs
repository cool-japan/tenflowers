//! Complex number types and operations for TenfloweRS
//!
//! This module provides Complex32 and Complex64 types to support FFT operations
//! and other complex number computations.

use num_traits::{One, Zero};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// 32-bit complex number (f32 real and imaginary parts)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex32 {
    pub real: f32,
    pub imag: f32,
}

/// 64-bit complex number (f64 real and imaginary parts)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub real: f64,
    pub imag: f64,
}

impl Complex32 {
    /// Create a new complex number
    pub fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }

    /// Create a complex number from a real number
    pub fn from_real(real: f32) -> Self {
        Self { real, imag: 0.0 }
    }

    /// Create a complex number from an imaginary number
    pub fn from_imag(imag: f32) -> Self {
        Self { real: 0.0, imag }
    }

    /// Get the magnitude (absolute value) of the complex number
    pub fn magnitude(&self) -> f32 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    /// Get the phase (argument) of the complex number
    pub fn phase(&self) -> f32 {
        self.imag.atan2(self.real)
    }

    /// Get the complex conjugate
    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }

    /// Convert to polar form (magnitude, phase)
    pub fn to_polar(&self) -> (f32, f32) {
        (self.magnitude(), self.phase())
    }

    /// Create from polar form (magnitude, phase)
    pub fn from_polar(magnitude: f32, phase: f32) -> Self {
        Self {
            real: magnitude * phase.cos(),
            imag: magnitude * phase.sin(),
        }
    }
}

impl Complex64 {
    /// Create a new complex number
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    /// Create a complex number from a real number
    pub fn from_real(real: f64) -> Self {
        Self { real, imag: 0.0 }
    }

    /// Create a complex number from an imaginary number
    pub fn from_imag(imag: f64) -> Self {
        Self { real: 0.0, imag }
    }

    /// Get the magnitude (absolute value) of the complex number
    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    /// Get the phase (argument) of the complex number
    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    /// Get the complex conjugate
    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }

    /// Convert to polar form (magnitude, phase)
    pub fn to_polar(&self) -> (f64, f64) {
        (self.magnitude(), self.phase())
    }

    /// Create from polar form (magnitude, phase)
    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            real: magnitude * phase.cos(),
            imag: magnitude * phase.sin(),
        }
    }
}

// Arithmetic operations for Complex32
impl Add for Complex32 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl Sub for Complex32 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl Mul for Complex32 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl Div for Complex32 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.real * other.real + other.imag * other.imag;
        Self {
            real: (self.real * other.real + self.imag * other.imag) / denom,
            imag: (self.imag * other.real - self.real * other.imag) / denom,
        }
    }
}

impl Neg for Complex32 {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

// Arithmetic operations for Complex64
impl Add for Complex64 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl Sub for Complex64 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl Mul for Complex64 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl Div for Complex64 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.real * other.real + other.imag * other.imag;
        Self {
            real: (self.real * other.real + self.imag * other.imag) / denom,
            imag: (self.imag * other.real - self.real * other.imag) / denom,
        }
    }
}

impl Neg for Complex64 {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

// Zero and One traits for Complex32
impl Zero for Complex32 {
    fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imag == 0.0
    }
}

impl One for Complex32 {
    fn one() -> Self {
        Self {
            real: 1.0,
            imag: 0.0,
        }
    }
}

// Zero and One traits for Complex64
impl Zero for Complex64 {
    fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imag == 0.0
    }
}

impl One for Complex64 {
    fn one() -> Self {
        Self {
            real: 1.0,
            imag: 0.0,
        }
    }
}

// Default implementations
impl Default for Complex32 {
    fn default() -> Self {
        Self::zero()
    }
}

impl Default for Complex64 {
    fn default() -> Self {
        Self::zero()
    }
}

// Display implementations
impl fmt::Display for Complex32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.imag >= 0.0 {
            write!(f, "{}+{}i", self.real, self.imag)
        } else {
            write!(f, "{}{}i", self.real, self.imag)
        }
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.imag >= 0.0 {
            write!(f, "{}+{}i", self.real, self.imag)
        } else {
            write!(f, "{}{}i", self.real, self.imag)
        }
    }
}

// Conversions
impl From<f32> for Complex32 {
    fn from(real: f32) -> Self {
        Self::from_real(real)
    }
}

impl From<f64> for Complex64 {
    fn from(real: f64) -> Self {
        Self::from_real(real)
    }
}

impl From<Complex32> for Complex64 {
    fn from(c: Complex32) -> Self {
        Self {
            real: c.real as f64,
            imag: c.imag as f64,
        }
    }
}

impl From<Complex64> for Complex32 {
    fn from(c: Complex64) -> Self {
        Self {
            real: c.real as f32,
            imag: c.imag as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex32_arithmetic() {
        let a = Complex32::new(3.0, 4.0);
        let b = Complex32::new(1.0, 2.0);

        let sum = a + b;
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);

        let product = a * b;
        assert_eq!(product.real, -5.0);
        assert_eq!(product.imag, 10.0);
    }

    #[test]
    fn test_complex64_magnitude() {
        let c = Complex64::new(3.0, 4.0);
        assert_eq!(c.magnitude(), 5.0);
    }

    #[test]
    fn test_complex_conjugate() {
        let c = Complex32::new(3.0, 4.0);
        let conj = c.conjugate();
        assert_eq!(conj.real, 3.0);
        assert_eq!(conj.imag, -4.0);
    }
}
