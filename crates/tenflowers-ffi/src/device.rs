//! Device abstraction module for TenfloweRS FFI
//!
//! Exposes `Device` and `DeviceKind` as Python classes mirroring
//! `tenflowers_core::Device`, keeping the existing string-based
//! `set_default_device("gpu:0")` API intact.

use pyo3::prelude::*;
use tenflowers_core::Device;

/// The kind of compute device.
#[pyclass(name = "DeviceKind", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDeviceKind {
    /// CPU (host memory)
    Cpu = 0,
    /// NVIDIA GPU via CUDA
    Gpu = 1,
    /// AMD GPU via ROCm
    Rocm = 2,
}

#[pymethods]
impl PyDeviceKind {
    fn __repr__(&self) -> &'static str {
        match self {
            PyDeviceKind::Cpu => "DeviceKind.Cpu",
            PyDeviceKind::Gpu => "DeviceKind.Gpu",
            PyDeviceKind::Rocm => "DeviceKind.Rocm",
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            PyDeviceKind::Cpu => "cpu",
            PyDeviceKind::Gpu => "gpu",
            PyDeviceKind::Rocm => "rocm",
        }
    }
}

/// A compute device (CPU or a GPU/ROCm device with a numeric ID).
///
/// # Examples (Python)
/// ```python
/// from tenflowers import Device
/// cpu = Device.cpu()
/// assert cpu.is_cpu
/// gpu0 = Device.gpu(0)
/// assert gpu0.is_gpu
/// assert repr(gpu0) == "Device.gpu(0)"
/// ```
#[pyclass(name = "Device")]
#[derive(Debug, Clone)]
pub struct PyDevice {
    /// Kind of the device.
    pub kind: PyDeviceKind,
    /// Device ordinal / ID (0 for CPU).
    pub id: usize,
}

#[pymethods]
impl PyDevice {
    /// Construct the CPU device.
    #[staticmethod]
    pub fn cpu() -> Self {
        Self {
            kind: PyDeviceKind::Cpu,
            id: 0,
        }
    }

    /// Construct a CUDA GPU device with the given ordinal.
    #[staticmethod]
    pub fn gpu(id: usize) -> Self {
        Self {
            kind: PyDeviceKind::Gpu,
            id,
        }
    }

    /// Construct a ROCm GPU device with the given ordinal.
    #[staticmethod]
    pub fn rocm(id: usize) -> Self {
        Self {
            kind: PyDeviceKind::Rocm,
            id,
        }
    }

    /// Parse a device string such as `"cpu"`, `"gpu:0"`, `"rocm:1"`.
    #[staticmethod]
    pub fn from_string(s: &str) -> PyResult<Self> {
        let s = s.trim().to_lowercase();
        if s == "cpu" {
            return Ok(Self::cpu());
        }
        if let Some(rest) = s.strip_prefix("gpu:") {
            let id: usize = rest.parse().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid GPU id in '{}'", s))
            })?;
            return Ok(Self::gpu(id));
        }
        if let Some(rest) = s.strip_prefix("rocm:") {
            let id: usize = rest.parse().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid ROCm id in '{}'", s))
            })?;
            return Ok(Self::rocm(id));
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unrecognised device string '{}'; expected 'cpu', 'gpu:N', or 'rocm:N'",
            s
        )))
    }

    pub fn __repr__(&self) -> String {
        match self.kind {
            PyDeviceKind::Cpu => "Device.cpu()".to_string(),
            PyDeviceKind::Gpu => format!("Device.gpu({})", self.id),
            PyDeviceKind::Rocm => format!("Device.rocm({})", self.id),
        }
    }

    pub fn __str__(&self) -> String {
        match self.kind {
            PyDeviceKind::Cpu => "cpu".to_string(),
            PyDeviceKind::Gpu => format!("gpu:{}", self.id),
            PyDeviceKind::Rocm => format!("rocm:{}", self.id),
        }
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.kind == other.kind && self.id == other.id
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        !self.__eq__(other)
    }

    pub fn __hash__(&self) -> u64 {
        let kind_val = self.kind as u64;
        // simple hash: pack kind + id
        kind_val
            .wrapping_mul(1_000_003)
            .wrapping_add(self.id as u64)
    }

    /// `True` if this is the CPU device.
    #[getter]
    pub fn is_cpu(&self) -> bool {
        matches!(self.kind, PyDeviceKind::Cpu)
    }

    /// `True` if this is a CUDA GPU device.
    #[getter]
    pub fn is_gpu(&self) -> bool {
        matches!(self.kind, PyDeviceKind::Gpu)
    }

    /// `True` if this is a ROCm device.
    #[getter]
    pub fn is_rocm(&self) -> bool {
        matches!(self.kind, PyDeviceKind::Rocm)
    }

    /// The device ordinal (0 for CPU, N for GPU/ROCm).
    #[getter]
    pub fn device_id(&self) -> usize {
        self.id
    }

    /// The device kind.
    #[getter]
    pub fn kind(&self) -> PyDeviceKind {
        self.kind
    }
}

// ── Conversions ──────────────────────────────────────────────────────────────

impl From<Device> for PyDevice {
    fn from(d: Device) -> Self {
        match d {
            Device::Cpu => PyDevice::cpu(),
            #[cfg(feature = "gpu")]
            Device::Gpu(id) => PyDevice::gpu(id),
            #[cfg(feature = "rocm")]
            Device::Rocm(id) => PyDevice::rocm(id),
        }
    }
}

impl From<PyDevice> for Device {
    fn from(d: PyDevice) -> Self {
        match d.kind {
            PyDeviceKind::Cpu => Device::Cpu,
            PyDeviceKind::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    Device::Gpu(d.id)
                }
                #[cfg(not(feature = "gpu"))]
                Device::Cpu // graceful fallback when GPU feature is disabled
            }
            PyDeviceKind::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    Device::Rocm(d.id)
                }
                #[cfg(not(feature = "rocm"))]
                Device::Cpu // graceful fallback when ROCm feature is disabled
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_repr() {
        let d = PyDevice::cpu();
        assert_eq!(d.__repr__(), "Device.cpu()");
        assert_eq!(d.__str__(), "cpu");
        assert!(d.is_cpu());
        assert!(!d.is_gpu());
        assert!(!d.is_rocm());
        assert_eq!(d.device_id(), 0);
    }

    #[test]
    fn gpu_device_repr() {
        let d = PyDevice::gpu(1);
        assert_eq!(d.__repr__(), "Device.gpu(1)");
        assert_eq!(d.__str__(), "gpu:1");
        assert!(!d.is_cpu());
        assert!(d.is_gpu());
        assert_eq!(d.device_id(), 1);
    }

    #[test]
    fn rocm_device_repr() {
        let d = PyDevice::rocm(2);
        assert_eq!(d.__repr__(), "Device.rocm(2)");
        assert_eq!(d.__str__(), "rocm:2");
        assert!(d.is_rocm());
        assert_eq!(d.device_id(), 2);
    }

    #[test]
    fn device_equality() {
        let a = PyDevice::cpu();
        let b = PyDevice::cpu();
        assert!(a.__eq__(&b));
        assert!(!a.__ne__(&b));

        let c = PyDevice::gpu(0);
        assert!(!a.__eq__(&c));
    }

    #[test]
    fn device_hash_differs_by_kind_and_id() {
        let cpu = PyDevice::cpu();
        let gpu0 = PyDevice::gpu(0);
        let gpu1 = PyDevice::gpu(1);
        // hashes should differ (not guaranteed for all hash functions but good sanity check)
        assert_ne!(cpu.__hash__(), gpu0.__hash__());
        assert_ne!(gpu0.__hash__(), gpu1.__hash__());
    }

    #[test]
    fn from_string_cpu() {
        let d = PyDevice::from_string("cpu").expect("test: cpu parse should succeed");
        assert!(d.is_cpu());
    }

    #[test]
    fn from_string_gpu() {
        let d = PyDevice::from_string("gpu:0").expect("test: gpu:0 parse should succeed");
        assert!(d.is_gpu());
        assert_eq!(d.device_id(), 0);
    }

    #[test]
    fn from_string_rocm() {
        let d = PyDevice::from_string("rocm:3").expect("test: rocm:3 parse should succeed");
        assert!(d.is_rocm());
        assert_eq!(d.device_id(), 3);
    }

    #[test]
    fn from_string_invalid_returns_err() {
        assert!(PyDevice::from_string("fpga:0").is_err());
        assert!(PyDevice::from_string("gpu:notanumber").is_err());
    }

    #[test]
    fn from_core_device_cpu() {
        let core = Device::Cpu;
        let py: PyDevice = core.into();
        assert!(py.is_cpu());
    }

    #[test]
    fn round_trip_cpu() {
        let py = PyDevice::cpu();
        let core: Device = py.into();
        assert_eq!(core, Device::Cpu);
    }
}
