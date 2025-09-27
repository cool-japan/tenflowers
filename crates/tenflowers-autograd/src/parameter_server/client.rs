use super::ParameterServer;
#[cfg(test)]
use super::{FaultToleranceMode, LoadBalancingStrategy, ParameterServerConfig};
use crate::tape::TensorId;
use tenflowers_core::{Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

pub struct ParameterServerClient {
    server: ParameterServer,
    worker_id: usize,
}

impl ParameterServerClient {
    /// Create a new client for the specified worker
    pub fn new(server: ParameterServer, worker_id: usize) -> Self {
        Self { server, worker_id }
    }

    /// Push gradients to the parameter server
    pub fn push_gradients<T>(&self, gradients: &[(TensorId, Tensor<T>, u64)]) -> Result<()>
    where
        T: Clone + Send + Sync + 'static,
    {
        for (tensor_id, gradient, version) in gradients {
            self.server
                .submit_gradient(self.worker_id, *tensor_id, gradient, *version)?;
        }
        Ok(())
    }

    /// Pull parameters from the parameter server
    pub fn pull_parameters<T>(&self, tensor_ids: &[TensorId]) -> Result<Vec<Tensor<T>>>
    where
        T: Clone + Send + Sync + 'static,
    {
        self.server.pull_parameters(self.worker_id, tensor_ids)
    }

    /// Send heartbeat to parameter server
    pub fn send_heartbeat(&self, computational_load: f64) -> Result<()> {
        self.server.heartbeat(self.worker_id, computational_load)
    }

    /// Update worker capacity information
    pub fn update_capacity(&self, capacity: f64, latency_ms: f64) -> Result<()> {
        self.server
            .update_worker_capacity(self.worker_id, capacity, latency_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{GradientTape, TrackedTensor};
    use tenflowers_core::Tensor;

    #[test]
    fn test_parameter_server_creation() {
        let config = ParameterServerConfig::default();
        let server = ParameterServer::new(config);
        let stats = server.get_stats();
        assert_eq!(stats.total_parameters, 0);
    }

    #[test]
    fn test_parameter_registration() {
        let config = ParameterServerConfig::default();
        let server = ParameterServer::new(config);

        let tape = GradientTape::new();
        let x: TrackedTensor<f32> = tape.watch(Tensor::zeros(&[2, 2]));

        let result = server.register_parameter(x.id, &x.tensor);
        assert!(result.is_ok());

        let stats = server.get_stats();
        assert_eq!(stats.total_parameters, 1);
    }

    #[test]
    fn test_gradient_submission() {
        let mut config = ParameterServerConfig::default();
        config.num_workers = 2;
        let server = ParameterServer::new(config);

        let tape = GradientTape::new();
        let x: TrackedTensor<f32> = tape.watch(Tensor::zeros(&[2, 2]));
        let grad: Tensor<f32> = Tensor::ones(&[2, 2]);

        server.register_parameter(x.id, &x.tensor).unwrap();

        let result = server.submit_gradient(0, x.id, &grad, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parameter_retrieval() {
        let config = ParameterServerConfig::default();
        let server = ParameterServer::new(config);

        let tape = GradientTape::new();
        let x: TrackedTensor<f32> = tape.watch(Tensor::zeros(&[2, 2]));

        server.register_parameter(x.id, &x.tensor).unwrap();

        let retrieved: Option<Tensor<f32>> = server.get_parameter(x.id).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_parameter_server_client() {
        let mut config = ParameterServerConfig::default();
        config.num_workers = 2;
        let server = ParameterServer::new(config);
        let client = ParameterServerClient::new(server.clone(), 0);

        let tape = GradientTape::new();
        let x: TrackedTensor<f32> = tape.watch(Tensor::zeros(&[2, 2]));
        let grad: Tensor<f32> = Tensor::ones(&[2, 2]);

        server.register_parameter(x.id, &x.tensor).unwrap();

        // Test heartbeat
        let result = client.send_heartbeat(0.5);
        assert!(result.is_ok());

        // Test gradient push
        let gradients = vec![(x.id, grad, 0u64)];
        let result = client.push_gradients(&gradients);
        assert!(result.is_ok());

        // Test parameter pull
        let result: Result<Vec<Tensor<f32>>> = client.pull_parameters(&[x.id]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_balancing_strategies() {
        let strategies = vec![
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::CapacityBased,
            LoadBalancingStrategy::LoadBased,
            LoadBalancingStrategy::Dynamic,
        ];

        for strategy in strategies {
            let mut config = ParameterServerConfig::default();
            config.load_balancing = strategy;
            config.num_workers = 3;

            let server = ParameterServer::new(config);
            let tape = GradientTape::new();
            let x: TrackedTensor<f32> = tape.watch(Tensor::zeros(&[2, 2]));

            let result = server.register_parameter(x.id, &x.tensor);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_worker_capacity_update() {
        let mut config = ParameterServerConfig::default();
        config.num_workers = 2;
        let server = ParameterServer::new(config);

        let result = server.update_worker_capacity(0, 1.5, 10.0);
        assert!(result.is_ok());

        let result = server.update_worker_capacity(10, 1.0, 5.0);
        assert!(result.is_err()); // Invalid worker ID
    }

    #[test]
    fn test_fault_tolerance_modes() {
        let modes = vec![
            FaultToleranceMode::None,
            FaultToleranceMode::Checkpoint,
            FaultToleranceMode::Replication,
            FaultToleranceMode::Hybrid,
        ];

        for mode in modes {
            let mut config = ParameterServerConfig::default();
            config.fault_tolerance = mode;
            let _server = ParameterServer::new(config);
            // Just ensure it creates successfully
        }
    }
}
