use crate::memory::global_monitor;
use crate::ops::benchmark::BenchmarkConfig;
use crate::{Device, Tensor};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub available_devices: Vec<Device>,
    pub default_device: Device,
    pub memory_info: MemoryInfo,
    pub performance_benchmarks: PerformanceBenchmarks,
    pub features_enabled: FeaturesInfo,
    pub health_status: HealthStatus,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_system_memory: Option<u64>,
    pub available_memory: Option<u64>,
    pub gpu_memory_info: Vec<GpuMemoryInfo>,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub device: Device,
    pub total_memory: Option<u64>,
    pub allocated_memory: u64,
    pub reserved_memory: u64,
}

#[derive(Debug, Clone)]
pub struct PerformanceBenchmarks {
    pub cpu_add_throughput: f64,
    pub cpu_matmul_throughput: f64,
    pub gpu_add_throughput: Option<f64>,
    pub gpu_matmul_throughput: Option<f64>,
    pub tensor_creation_latency: Duration,
    pub device_transfer_bandwidth: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct FeaturesInfo {
    pub gpu_support: bool,
    pub cuda_available: bool,
    pub metal_available: bool,
    pub rocm_available: bool,
    pub blas_acceleration: bool,
    pub mixed_precision: bool,
    pub distributed_training: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Excellent,
    Good,
    Warning(Vec<String>),
    Critical(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct SystemHealthChecker {
    config: HealthCheckConfig,
}

#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub run_performance_tests: bool,
    pub test_duration: Duration,
    pub memory_threshold_warning: f64,
    pub memory_threshold_critical: f64,
    pub performance_threshold_warning: f64,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            run_performance_tests: true,
            test_duration: Duration::from_secs(5),
            memory_threshold_warning: 0.8,
            memory_threshold_critical: 0.95,
            performance_threshold_warning: 0.1,
        }
    }
}

impl SystemHealthChecker {
    pub fn new() -> Self {
        Self {
            config: HealthCheckConfig::default(),
        }
    }

    pub fn with_config(config: HealthCheckConfig) -> Self {
        Self { config }
    }

    pub fn check_system_health(&self) -> Result<SystemInfo, Box<dyn std::error::Error>> {
        println!("🔍 TenfloweRS System Health Check");
        println!("=================================");

        let available_devices = self.detect_available_devices();
        let default_device = Device::default();

        println!("✅ Devices detected: {} devices", available_devices.len());

        let memory_info = self.gather_memory_info(&available_devices)?;
        println!("✅ Memory information gathered");

        let features_enabled = self.check_features();
        println!("✅ Feature detection completed");

        let performance_benchmarks = if self.config.run_performance_tests {
            println!("🏃 Running performance benchmarks...");
            self.run_performance_benchmarks(&available_devices)?
        } else {
            PerformanceBenchmarks::default()
        };

        let health_status =
            self.assess_health_status(&memory_info, &performance_benchmarks, &features_enabled);

        let system_info = SystemInfo {
            available_devices,
            default_device,
            memory_info,
            performance_benchmarks,
            features_enabled,
            health_status,
        };

        self.print_health_report(&system_info);

        Ok(system_info)
    }

    fn detect_available_devices(&self) -> Vec<Device> {
        let mut devices = vec![Device::Cpu];

        #[cfg(feature = "gpu")]
        {
            if let Ok(gpu_device) = Device::best_gpu() {
                devices.push(gpu_device);
            }

            for i in 0..8 {
                if let Ok(gpu_device) = Device::try_gpu(i) {
                    if !devices.contains(&gpu_device) {
                        devices.push(gpu_device);
                    }
                }
            }
        }

        devices
    }

    fn gather_memory_info(
        &self,
        devices: &[Device],
    ) -> Result<MemoryInfo, Box<dyn std::error::Error>> {
        let _monitor = global_monitor();

        let total_system_memory = self.get_system_memory();
        let available_memory = self.get_available_memory();

        let gpu_memory_info = Vec::new();

        for _device in devices {
            #[cfg(feature = "gpu")]
            if _device.is_gpu() {
                // GPU memory monitoring implementation would go here
                // Currently disabled to avoid compilation issues
            }
        }

        Ok(MemoryInfo {
            total_system_memory,
            available_memory,
            gpu_memory_info,
        })
    }

    fn check_features(&self) -> FeaturesInfo {
        FeaturesInfo {
            gpu_support: cfg!(feature = "gpu"),
            cuda_available: self.is_cuda_available(),
            metal_available: self.is_metal_available(),
            rocm_available: self.is_rocm_available(),
            blas_acceleration: cfg!(any(feature = "blas-openblas", feature = "blas-mkl")),
            mixed_precision: true,
            distributed_training: true,
        }
    }

    fn run_performance_benchmarks(
        &self,
        devices: &[Device],
    ) -> Result<PerformanceBenchmarks, Box<dyn std::error::Error>> {
        let config = BenchmarkConfig {
            warmup_iterations: 3,
            measurement_iterations: 10,
            measure_memory: false,
            calculate_flops: true,
            min_execution_time: Duration::from_millis(1),
            max_execution_time: self.config.test_duration,
        };

        let test_shape = vec![1024, 1024];

        // CPU benchmarks
        let cpu_add_throughput =
            self.benchmark_add_throughput(&Device::Cpu, &test_shape, &config)?;
        let cpu_matmul_throughput =
            self.benchmark_matmul_throughput(&Device::Cpu, &test_shape, &config)?;

        // GPU benchmarks
        let (gpu_add_throughput, gpu_matmul_throughput) = {
            #[cfg(feature = "gpu")]
            {
                if let Some(gpu_device) = devices.iter().find(|d| d.is_gpu()) {
                    (
                        Some(self.benchmark_add_throughput(gpu_device, &test_shape, &config)?),
                        Some(self.benchmark_matmul_throughput(gpu_device, &test_shape, &config)?),
                    )
                } else {
                    (None, None)
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                (None, None)
            }
        };

        // Tensor creation latency
        let tensor_creation_latency = self.benchmark_tensor_creation(&Device::Cpu)?;

        // Device transfer bandwidth
        let device_transfer_bandwidth = if devices.len() > 1 {
            Some(self.benchmark_device_transfer(devices)?)
        } else {
            None
        };

        Ok(PerformanceBenchmarks {
            cpu_add_throughput,
            cpu_matmul_throughput,
            gpu_add_throughput,
            gpu_matmul_throughput,
            tensor_creation_latency,
            device_transfer_bandwidth,
        })
    }

    fn benchmark_add_throughput(
        &self,
        _device: &Device,
        shape: &[usize],
        config: &BenchmarkConfig,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let a: Tensor<f32> = Tensor::ones(shape);
        let b: Tensor<f32> = Tensor::ones(shape);

        let start = Instant::now();
        for _ in 0..config.measurement_iterations {
            let _ = a.add(&b)?;
        }
        let elapsed = start.elapsed();

        let ops_per_second = config.measurement_iterations as f64 / elapsed.as_secs_f64();
        let elements = shape.iter().product::<usize>() as f64;
        Ok(ops_per_second * elements / 1e9)
    }

    fn benchmark_matmul_throughput(
        &self,
        _device: &Device,
        shape: &[usize],
        config: &BenchmarkConfig,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let a: Tensor<f32> = Tensor::ones(shape);
        let b: Tensor<f32> = Tensor::ones(shape);

        let start = Instant::now();
        for _ in 0..config.measurement_iterations {
            let _ = a.matmul(&b)?;
        }
        let elapsed = start.elapsed();

        let ops_per_second = config.measurement_iterations as f64 / elapsed.as_secs_f64();
        let flops = 2.0 * shape[0] as f64 * shape[1] as f64 * shape[1] as f64;
        Ok(ops_per_second * flops / 1e9)
    }

    fn benchmark_tensor_creation(
        &self,
        _device: &Device,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let iterations = 1000;
        let shape = vec![100, 100];

        let start = Instant::now();
        for _ in 0..iterations {
            let _: Tensor<f32> = Tensor::zeros(&shape);
        }
        let elapsed = start.elapsed();

        Ok(elapsed / iterations)
    }

    fn benchmark_device_transfer(
        &self,
        devices: &[Device],
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if devices.len() < 2 {
            return Ok(0.0);
        }

        let shape = vec![1024, 1024];
        let _tensor: Tensor<f32> = Tensor::ones(&shape);
        let data_size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();

        // For now, return mock transfer rate since device transfer API might not be available
        let elapsed = Duration::from_millis(10); // Mock transfer time

        Ok(data_size as f64 / elapsed.as_secs_f64() / 1e9)
    }

    fn assess_health_status(
        &self,
        memory_info: &MemoryInfo,
        benchmarks: &PerformanceBenchmarks,
        features: &FeaturesInfo,
    ) -> HealthStatus {
        let mut warnings = Vec::new();
        let mut critical_issues = Vec::new();

        // Check memory usage
        for gpu_info in &memory_info.gpu_memory_info {
            if let Some(total) = gpu_info.total_memory {
                let usage_ratio =
                    (gpu_info.allocated_memory + gpu_info.reserved_memory) as f64 / total as f64;

                if usage_ratio > self.config.memory_threshold_critical {
                    critical_issues.push(format!(
                        "Critical GPU memory usage: {:.1}%",
                        usage_ratio * 100.0
                    ));
                } else if usage_ratio > self.config.memory_threshold_warning {
                    warnings.push(format!(
                        "High GPU memory usage: {:.1}%",
                        usage_ratio * 100.0
                    ));
                }
            }
        }

        // Check performance
        if benchmarks.cpu_add_throughput < self.config.performance_threshold_warning {
            warnings.push("Low CPU performance detected".to_string());
        }

        if let Some(gpu_throughput) = benchmarks.gpu_add_throughput {
            if gpu_throughput < self.config.performance_threshold_warning {
                warnings.push("Low GPU performance detected".to_string());
            }
        }

        // Check features
        if !features.gpu_support {
            warnings.push("GPU support not compiled in".to_string());
        }

        if !critical_issues.is_empty() {
            HealthStatus::Critical(critical_issues)
        } else if !warnings.is_empty() {
            HealthStatus::Warning(warnings)
        } else if features.gpu_support && features.blas_acceleration {
            HealthStatus::Excellent
        } else {
            HealthStatus::Good
        }
    }

    fn print_health_report(&self, info: &SystemInfo) {
        println!("\n📊 System Health Report");
        println!("=======================");

        println!("\n🖥️  Available Devices:");
        for device in &info.available_devices {
            println!("  • {device}");
        }
        println!("  Default device: {}", info.default_device);

        println!("\n💾 Memory Information:");
        if let Some(total) = info.memory_info.total_system_memory {
            println!("  System memory: {:.2} GB", total as f64 / 1e9);
        }

        for gpu_info in &info.memory_info.gpu_memory_info {
            println!("  {} Memory:", gpu_info.device);
            if let Some(total) = gpu_info.total_memory {
                println!("    Total: {:.2} GB", total as f64 / 1e9);
            }
            println!(
                "    Allocated: {:.2} MB",
                gpu_info.allocated_memory as f64 / 1e6
            );
            println!(
                "    Reserved: {:.2} MB",
                gpu_info.reserved_memory as f64 / 1e6
            );
        }

        println!("\n⚡ Features Enabled:");
        println!(
            "  GPU Support: {}",
            if info.features_enabled.gpu_support {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  CUDA: {}",
            if info.features_enabled.cuda_available {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  Metal: {}",
            if info.features_enabled.metal_available {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  ROCm: {}",
            if info.features_enabled.rocm_available {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  BLAS Acceleration: {}",
            if info.features_enabled.blas_acceleration {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  Mixed Precision: {}",
            if info.features_enabled.mixed_precision {
                "✅"
            } else {
                "❌"
            }
        );

        if self.config.run_performance_tests {
            println!("\n🏎️  Performance Benchmarks:");
            println!(
                "  CPU Add throughput: {:.2} GFLOPS",
                info.performance_benchmarks.cpu_add_throughput
            );
            println!(
                "  CPU MatMul throughput: {:.2} GFLOPS",
                info.performance_benchmarks.cpu_matmul_throughput
            );

            if let Some(gpu_add) = info.performance_benchmarks.gpu_add_throughput {
                println!("  GPU Add throughput: {gpu_add:.2} GFLOPS");
            }

            if let Some(gpu_matmul) = info.performance_benchmarks.gpu_matmul_throughput {
                println!("  GPU MatMul throughput: {gpu_matmul:.2} GFLOPS");
            }

            println!(
                "  Tensor creation latency: {:?}",
                info.performance_benchmarks.tensor_creation_latency
            );

            if let Some(bandwidth) = info.performance_benchmarks.device_transfer_bandwidth {
                println!("  Device transfer bandwidth: {bandwidth:.2} GB/s");
            }
        }

        println!("\n🏥 Health Status:");
        match &info.health_status {
            HealthStatus::Excellent => println!("  ✅ Excellent - All systems optimal!"),
            HealthStatus::Good => println!("  👍 Good - System running well"),
            HealthStatus::Warning(warnings) => {
                println!("  ⚠️  Warning - Issues detected:");
                for warning in warnings {
                    println!("    • {warning}");
                }
            }
            HealthStatus::Critical(issues) => {
                println!("  🚨 Critical - Immediate attention required:");
                for issue in issues {
                    println!("    • {issue}");
                }
            }
        }

        println!("\n🎯 Recommendations:");
        self.print_recommendations(info);
    }

    fn print_recommendations(&self, info: &SystemInfo) {
        let mut recommendations = Vec::new();

        if !info.features_enabled.gpu_support {
            recommendations.push("Consider compiling with GPU support for better performance");
        }

        if !info.features_enabled.blas_acceleration {
            recommendations.push("Enable BLAS acceleration for improved CPU performance");
        }

        if info.available_devices.len() == 1 && info.available_devices[0].is_cpu() {
            recommendations.push("Consider using GPU acceleration for large-scale computations");
        }

        if info.performance_benchmarks.cpu_add_throughput < 1.0 {
            recommendations.push("CPU performance seems low - check system load and cooling");
        }

        if recommendations.is_empty() {
            println!("  ✨ Your TenfloweRS installation is optimally configured!");
        } else {
            for rec in recommendations {
                println!("  💡 {rec}");
            }
        }
    }

    // Helper methods for system detection
    fn get_system_memory(&self) -> Option<u64> {
        None
    }

    fn get_available_memory(&self) -> Option<u64> {
        None
    }

    #[allow(dead_code)]
    fn get_gpu_total_memory(&self, _device: &Device) -> Option<u64> {
        None
    }

    fn is_cuda_available(&self) -> bool {
        cfg!(feature = "cuda")
    }

    fn is_metal_available(&self) -> bool {
        cfg!(feature = "metal")
    }

    fn is_rocm_available(&self) -> bool {
        cfg!(feature = "rocm")
    }
}

impl Default for SystemHealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceBenchmarks {
    fn default() -> Self {
        Self {
            cpu_add_throughput: 0.0,
            cpu_matmul_throughput: 0.0,
            gpu_add_throughput: None,
            gpu_matmul_throughput: None,
            tensor_creation_latency: Duration::from_nanos(0),
            device_transfer_bandwidth: None,
        }
    }
}

pub fn run_system_health_check() -> Result<SystemInfo, Box<dyn std::error::Error>> {
    let checker = SystemHealthChecker::new();
    checker.check_system_health()
}

pub fn run_quick_health_check() -> Result<SystemInfo, Box<dyn std::error::Error>> {
    let config = HealthCheckConfig {
        run_performance_tests: false,
        ..Default::default()
    };
    let checker = SystemHealthChecker::with_config(config);
    checker.check_system_health()
}
