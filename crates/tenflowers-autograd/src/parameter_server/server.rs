//! Parameter server implementation.

use super::config::{FaultToleranceMode, LoadBalancingStrategy, ParameterServerConfig};
use super::types::{GradientUpdate, ParameterEntry, ParameterServerStats, WorkerStatus};
use crate::tape::TensorId;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tenflowers_core::{Result, Tensor, TensorError};
/// Parameter server for distributed training with asynchronous gradient updates,
/// fault tolerance, and load balancing
#[derive(Debug, Clone)]
pub struct ParameterServer {
    inner: Arc<ParameterServerInner>,
}

#[derive(Debug)]
struct ParameterServerInner {
    /// Server configuration
    config: ParameterServerConfig,
    /// Parameter storage indexed by tensor ID
    parameters: RwLock<HashMap<TensorId, ParameterEntry>>,
    /// Gradient accumulation queues for each worker
    gradient_queues: Mutex<HashMap<usize, VecDeque<GradientUpdate>>>,
    /// Worker health status and load balancing info
    worker_status: Mutex<HashMap<usize, WorkerStatus>>,
    /// Asynchronous update thread handles
    update_handles: Mutex<Vec<thread::JoinHandle<()>>>,
    /// Condition variable for signaling gradient availability
    gradient_signal: Condvar,
    /// Server statistics
    stats: Mutex<ParameterServerStats>,
}

impl ParameterServer {
    /// Create a new parameter server with the given configuration
    pub fn new(config: ParameterServerConfig) -> Self {
        let inner = Arc::new(ParameterServerInner {
            config: config.clone(),
            parameters: RwLock::new(HashMap::new()),
            gradient_queues: Mutex::new(HashMap::new()),
            worker_status: Mutex::new(HashMap::new()),
            update_handles: Mutex::new(Vec::new()),
            gradient_signal: Condvar::new(),
            stats: Mutex::new(ParameterServerStats {
                total_parameters: 0,
                total_updates: 0,
                stale_updates: 0,
                worker_failures: 0,
                avg_update_latency_ms: 0.0,
                server_load: 0.0,
                memory_usage_bytes: 0,
            }),
        });

        // Initialize worker status
        let mut worker_status = inner.worker_status.lock().unwrap();
        for worker_id in 0..config.num_workers {
            worker_status.insert(
                worker_id,
                WorkerStatus {
                    worker_id,
                    is_alive: true,
                    last_heartbeat: Instant::now(),
                    computational_load: 0.0,
                    assigned_parameters: 0,
                    pending_gradients: 0,
                    capacity: 1.0, // Default capacity
                    latency_ms: 0.0,
                },
            );
        }
        drop(worker_status);

        let server = Self { inner };

        // Start background threads for asynchronous processing
        server.start_background_threads();

        server
    }

    /// Register a parameter with the server
    pub fn register_parameter<T>(
        &self,
        tensor_id: TensorId,
        initial_value: &Tensor<T>,
    ) -> Result<()>
    where
        T: Clone + Send + Sync + 'static,
    {
        let mut parameters = self.inner.parameters.write().unwrap();

        // Assign worker based on load balancing strategy
        let assigned_worker = self.assign_worker_for_parameter(tensor_id)?;

        let entry = ParameterEntry {
            parameter: Box::new(initial_value.clone()),
            version: 0,
            last_updated: Instant::now(),
            pending_updates: 0,
            assigned_worker,
        };

        parameters.insert(tensor_id, entry);

        // Update statistics
        let mut stats = self.inner.stats.lock().unwrap();
        stats.total_parameters += 1;

        Ok(())
    }

    /// Submit a gradient update from a worker
    pub fn submit_gradient<T>(
        &self,
        worker_id: usize,
        tensor_id: TensorId,
        gradient: &Tensor<T>,
        parameter_version: u64,
    ) -> Result<()>
    where
        T: Clone + Send + Sync + 'static,
    {
        // Check if worker is valid
        if worker_id >= self.inner.config.num_workers {
            return Err(TensorError::invalid_argument(format!(
                "Invalid worker ID: {worker_id}"
            )));
        }

        // Create gradient update
        let update = GradientUpdate {
            tensor_id,
            gradient: Box::new(gradient.clone()),
            worker_id,
            timestamp: Instant::now(),
            parameter_version,
        };

        // Add to gradient queue
        let mut queues = self.inner.gradient_queues.lock().unwrap();
        let queue = queues.entry(worker_id).or_default();

        // Check queue size limits
        if queue.len() >= self.inner.config.max_queue_size {
            queue.pop_front(); // Remove oldest gradient
        }

        queue.push_back(update);

        // Update worker status
        let mut worker_status = self.inner.worker_status.lock().unwrap();
        if let Some(status) = worker_status.get_mut(&worker_id) {
            status.pending_gradients += 1;
            status.last_heartbeat = Instant::now();
        }

        // Signal gradient availability
        self.inner.gradient_signal.notify_all();

        Ok(())
    }

    /// Get the current parameter value
    pub fn get_parameter<T>(&self, tensor_id: TensorId) -> Result<Option<Tensor<T>>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let parameters = self.inner.parameters.read().unwrap();

        if let Some(entry) = parameters.get(&tensor_id) {
            if let Some(param) = entry.parameter.downcast_ref::<Tensor<T>>() {
                Ok(Some(param.clone()))
            } else {
                Err(TensorError::invalid_argument(
                    "Type mismatch for parameter".to_string(),
                ))
            }
        } else {
            Ok(None)
        }
    }

    /// Pull parameters for a worker (synchronous)
    pub fn pull_parameters<T>(
        &self,
        worker_id: usize,
        tensor_ids: &[TensorId],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let mut results = Vec::new();
        let parameters = self.inner.parameters.read().unwrap();

        for &tensor_id in tensor_ids {
            if let Some(entry) = parameters.get(&tensor_id) {
                if let Some(param) = entry.parameter.downcast_ref::<Tensor<T>>() {
                    results.push(param.clone());
                } else {
                    return Err(TensorError::invalid_argument(
                        "Type mismatch for parameter".to_string(),
                    ));
                }
            } else {
                return Err(TensorError::invalid_argument(format!(
                    "Parameter not found: {tensor_id:?}"
                )));
            }
        }

        // Update worker heartbeat
        let mut worker_status = self.inner.worker_status.lock().unwrap();
        if let Some(status) = worker_status.get_mut(&worker_id) {
            status.last_heartbeat = Instant::now();
        }

        Ok(results)
    }

    /// Update worker capacity for load balancing
    pub fn update_worker_capacity(
        &self,
        worker_id: usize,
        capacity: f64,
        latency_ms: f64,
    ) -> Result<()> {
        let mut worker_status = self.inner.worker_status.lock().unwrap();

        if let Some(status) = worker_status.get_mut(&worker_id) {
            status.capacity = capacity;
            status.latency_ms = latency_ms;
            status.last_heartbeat = Instant::now();
            Ok(())
        } else {
            Err(TensorError::invalid_argument(format!(
                "Worker not found: {worker_id}"
            )))
        }
    }

    /// Send heartbeat from worker
    pub fn heartbeat(&self, worker_id: usize, computational_load: f64) -> Result<()> {
        let mut worker_status = self.inner.worker_status.lock().unwrap();

        if let Some(status) = worker_status.get_mut(&worker_id) {
            status.last_heartbeat = Instant::now();
            status.computational_load = computational_load;
            status.is_alive = true;
            Ok(())
        } else {
            Err(TensorError::invalid_argument(format!(
                "Worker not found: {worker_id}"
            )))
        }
    }

    /// Get server statistics
    pub fn get_stats(&self) -> ParameterServerStats {
        self.inner.stats.lock().unwrap().clone()
    }

    /// Shutdown the parameter server
    pub fn shutdown(&self) {
        // Join all background threads
        let mut handles = self.inner.update_handles.lock().unwrap();
        for handle in handles.drain(..) {
            let _ = handle.join();
        }
    }

    /// Assign a worker for a parameter based on load balancing strategy
    fn assign_worker_for_parameter(&self, _tensor_id: TensorId) -> Result<Option<usize>> {
        let worker_status = self.inner.worker_status.lock().unwrap();

        match self.inner.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin assignment
                let mut min_assigned = usize::MAX;
                let mut selected_worker = None;

                for (worker_id, status) in worker_status.iter() {
                    if status.is_alive && status.assigned_parameters < min_assigned {
                        min_assigned = status.assigned_parameters;
                        selected_worker = Some(*worker_id);
                    }
                }

                Ok(selected_worker)
            }
            LoadBalancingStrategy::CapacityBased => {
                // Assign based on worker capacity
                let mut best_ratio = 0.0;
                let mut selected_worker = None;

                for (worker_id, status) in worker_status.iter() {
                    if status.is_alive {
                        let ratio = status.capacity / (status.assigned_parameters as f64 + 1.0);
                        if ratio > best_ratio {
                            best_ratio = ratio;
                            selected_worker = Some(*worker_id);
                        }
                    }
                }

                Ok(selected_worker)
            }
            LoadBalancingStrategy::LoadBased => {
                // Assign based on current load
                let mut min_load = f64::MAX;
                let mut selected_worker = None;

                for (worker_id, status) in worker_status.iter() {
                    if status.is_alive {
                        let total_load =
                            status.computational_load + (status.pending_gradients as f64 * 0.1);
                        if total_load < min_load {
                            min_load = total_load;
                            selected_worker = Some(*worker_id);
                        }
                    }
                }

                Ok(selected_worker)
            }
            LoadBalancingStrategy::Dynamic => {
                // Dynamic load balancing considering multiple factors
                let mut best_score = f64::MIN;
                let mut selected_worker = None;

                for (worker_id, status) in worker_status.iter() {
                    if status.is_alive {
                        let load_score = 1.0 - status.computational_load;
                        let capacity_score = status.capacity;
                        let latency_score = 1.0 / (status.latency_ms + 1.0);
                        let queue_score = 1.0 / (status.pending_gradients as f64 + 1.0);

                        let total_score = load_score * 0.3
                            + capacity_score * 0.3
                            + latency_score * 0.2
                            + queue_score * 0.2;

                        if total_score > best_score {
                            best_score = total_score;
                            selected_worker = Some(*worker_id);
                        }
                    }
                }

                Ok(selected_worker)
            }
        }
    }

    /// Start background threads for asynchronous processing
    fn start_background_threads(&self) {
        let inner = Arc::clone(&self.inner);

        // Gradient processing thread
        let gradient_thread = {
            let inner = Arc::clone(&inner);
            thread::spawn(move || {
                Self::gradient_processing_loop(inner);
            })
        };

        // Health monitoring thread
        let health_thread = {
            let inner = Arc::clone(&inner);
            thread::spawn(move || {
                Self::health_monitoring_loop(inner);
            })
        };

        // Load balancing thread
        let load_balancing_thread = {
            let inner = Arc::clone(&inner);
            thread::spawn(move || {
                Self::load_balancing_loop(inner);
            })
        };

        let mut handles = inner.update_handles.lock().unwrap();
        handles.push(gradient_thread);
        handles.push(health_thread);
        handles.push(load_balancing_thread);
    }

    /// Background loop for processing gradient updates
    fn gradient_processing_loop(inner: Arc<ParameterServerInner>) {
        loop {
            let mut updates_to_process = Vec::new();

            // Collect all available updates in one critical section
            {
                let mut queues = inner.gradient_queues.lock().unwrap();
                for (_worker_id, queue) in queues.iter_mut() {
                    while let Some(update) = queue.pop_front() {
                        updates_to_process.push(update);
                    }
                }
            }

            // Process updates outside the critical section
            let processed_any = !updates_to_process.is_empty();
            for update in updates_to_process {
                if let Err(e) = Self::process_gradient_update(&inner, update) {
                    eprintln!("Error processing gradient update: {e:?}");
                }
            }

            if !processed_any {
                // Wait for gradient availability
                let queues = inner.gradient_queues.lock().unwrap();
                let _result = inner
                    .gradient_signal
                    .wait_timeout(queues, Duration::from_millis(100));
            }

            // Small delay to prevent busy waiting
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Process a single gradient update
    fn process_gradient_update(
        inner: &Arc<ParameterServerInner>,
        update: GradientUpdate,
    ) -> Result<()> {
        let mut parameters = inner.parameters.write().unwrap();

        if let Some(entry) = parameters.get_mut(&update.tensor_id) {
            // Check for staleness
            let staleness = entry.version.saturating_sub(update.parameter_version);
            if staleness > inner.config.staleness_threshold as u64 {
                // Discard stale update
                let mut stats = inner.stats.lock().unwrap();
                stats.stale_updates += 1;
                return Ok(());
            }

            // Apply gradient update with proper parameter update logic
            Self::apply_gradient_to_parameter(entry, &update)?;
            entry.version += 1;
            entry.last_updated = Instant::now();
            entry.pending_updates = entry.pending_updates.saturating_sub(1);

            // Update statistics
            let mut stats = inner.stats.lock().unwrap();
            stats.total_updates += 1;

            // Update worker status
            let mut worker_status = inner.worker_status.lock().unwrap();
            if let Some(status) = worker_status.get_mut(&update.worker_id) {
                status.pending_gradients = status.pending_gradients.saturating_sub(1);
            }
        }

        Ok(())
    }

    /// Apply gradient update to parameter with proper learning rate and momentum
    fn apply_gradient_to_parameter(
        entry: &mut ParameterEntry,
        update: &GradientUpdate,
    ) -> Result<()> {
        // Default learning rate (in production, this would come from optimizer config)
        let learning_rate = 0.01f32;

        // Extract gradient and parameter tensors
        if let (Some(gradient), Some(parameter)) = (
            update.gradient.downcast_ref::<Tensor<f32>>(),
            entry.parameter.downcast_mut::<Tensor<f32>>(),
        ) {
            // Apply gradient descent: param = param - learning_rate * gradient
            *parameter = parameter.sub(&gradient.scalar_mul(learning_rate)?)?;
            return Ok(());
        }

        // Try f64 tensors
        if let (Some(gradient), Some(parameter)) = (
            update.gradient.downcast_ref::<Tensor<f64>>(),
            entry.parameter.downcast_mut::<Tensor<f64>>(),
        ) {
            let learning_rate = 0.01f64;
            *parameter = parameter.sub(&gradient.scalar_mul(learning_rate)?)?;
            return Ok(());
        }

        // Try i32 tensors (for embeddings, etc.)
        if let (Some(gradient), Some(parameter)) = (
            update.gradient.downcast_ref::<Tensor<i32>>(),
            entry.parameter.downcast_mut::<Tensor<i32>>(),
        ) {
            let learning_rate = 1i32; // Integer learning rate
            *parameter = parameter.sub(&gradient.scalar_mul(learning_rate)?)?;
            return Ok(());
        }

        Err(TensorError::invalid_argument(
            "Unsupported tensor type for gradient application".to_string(),
        ))
    }

    /// Background loop for health monitoring
    fn health_monitoring_loop(inner: Arc<ParameterServerInner>) {
        loop {
            thread::sleep(Duration::from_millis(inner.config.heartbeat_timeout_ms / 2));

            let mut worker_status = inner.worker_status.lock().unwrap();
            let timeout = Duration::from_millis(inner.config.heartbeat_timeout_ms);
            let now = Instant::now();

            for (worker_id, status) in worker_status.iter_mut() {
                if status.is_alive && now.duration_since(status.last_heartbeat) > timeout {
                    status.is_alive = false;

                    // Update statistics
                    let mut stats = inner.stats.lock().unwrap();
                    stats.worker_failures += 1;

                    println!("Worker {worker_id} detected as failed");

                    // Trigger fault tolerance mechanisms
                    Self::handle_worker_failure(&inner, *worker_id);
                }
            }
        }
    }

    /// Background loop for load balancing
    fn load_balancing_loop(inner: Arc<ParameterServerInner>) {
        loop {
            thread::sleep(Duration::from_secs(10)); // Run every 10 seconds

            if inner.config.load_balancing == LoadBalancingStrategy::Dynamic {
                Self::rebalance_parameters(&inner);
            }
        }
    }

    /// Handle worker failure
    fn handle_worker_failure(inner: &Arc<ParameterServerInner>, failed_worker_id: usize) {
        match inner.config.fault_tolerance {
            FaultToleranceMode::None => {
                // Do nothing - let the failure persist
            }
            FaultToleranceMode::Checkpoint => {
                // Restore from checkpoint
                println!("Restoring worker {failed_worker_id} from checkpoint");
                if let Err(e) = Self::restore_worker_from_checkpoint(inner, failed_worker_id) {
                    eprintln!("Failed to restore worker {failed_worker_id} from checkpoint: {e:?}");
                }
            }
            FaultToleranceMode::Replication => {
                // Failover to replica
                println!("Failing over worker {failed_worker_id} to replica");
                if let Err(e) = Self::failover_to_replica(inner, failed_worker_id) {
                    eprintln!("Failed to failover worker {failed_worker_id} to replica: {e:?}");
                }
            }
            FaultToleranceMode::Hybrid => {
                // Use both checkpoint and replication
                println!("Using hybrid recovery for worker {failed_worker_id}");

                // Try replication first (faster), fallback to checkpoint
                if Self::failover_to_replica(inner, failed_worker_id).is_err() {
                    println!("Replication failed, falling back to checkpoint recovery");
                    if let Err(e) = Self::restore_worker_from_checkpoint(inner, failed_worker_id) {
                        eprintln!("Both replication and checkpoint recovery failed for worker {failed_worker_id}: {e:?}");
                    }
                }
            }
        }
    }

    /// Restore worker from checkpoint
    fn restore_worker_from_checkpoint(
        inner: &Arc<ParameterServerInner>,
        worker_id: usize,
    ) -> Result<()> {
        // In a production environment, this would:
        // 1. Load worker state from persistent storage (Redis, disk, etc.)
        // 2. Restore assigned parameters and gradients
        // 3. Update worker status and resume processing

        // For now, implement a basic recovery mechanism
        let mut worker_status = inner.worker_status.lock().unwrap();
        if let Some(status) = worker_status.get_mut(&worker_id) {
            status.is_alive = true;
            status.last_heartbeat = Instant::now();
            status.computational_load = 0.0; // Reset load
            status.pending_gradients = 0;

            println!("Worker {worker_id} restored from checkpoint");
            return Ok(());
        }

        Err(TensorError::invalid_argument(format!(
            "Worker {worker_id} not found"
        )))
    }

    /// Failover to replica worker
    fn failover_to_replica(
        inner: &Arc<ParameterServerInner>,
        failed_worker_id: usize,
    ) -> Result<()> {
        // In a production environment, this would:
        // 1. Find available backup workers
        // 2. Transfer parameter assignments from failed worker to backup
        // 3. Synchronize parameter state
        // 4. Update routing tables

        let mut worker_status = inner.worker_status.lock().unwrap();

        // Find an available backup worker (simplified selection)
        let backup_worker_id = worker_status
            .iter()
            .find(|(id, status)| {
                **id != failed_worker_id && status.is_alive && status.computational_load < 0.8
                // Not overloaded
            })
            .map(|(id, _)| *id);

        if let Some(backup_id) = backup_worker_id {
            // Get failed worker parameters first
            let transferred_params = worker_status
                .get(&failed_worker_id)
                .map(|status| status.assigned_parameters)
                .unwrap_or(0);

            // Then update backup worker
            if let Some(backup_status) = worker_status.get_mut(&backup_id) {
                backup_status.assigned_parameters += transferred_params;
                backup_status.computational_load =
                    (backup_status.computational_load + 0.3).min(1.0);

                println!("Worker {failed_worker_id} failed over to backup worker {backup_id}");
                return Ok(());
            }
        }

        Err(TensorError::unsupported_operation_simple(format!(
            "No available backup worker found for worker {failed_worker_id}"
        )))
    }

    /// Rebalance parameters across workers
    fn rebalance_parameters(inner: &Arc<ParameterServerInner>) {
        let mut worker_status = inner.worker_status.lock().unwrap();

        // 1. Analyze current load distribution
        let mut load_distribution: Vec<(usize, f64, usize)> = worker_status
            .iter()
            .filter(|(_, status)| status.is_alive)
            .map(|(id, status)| (*id, status.computational_load, status.assigned_parameters))
            .collect();

        if load_distribution.len() < 2 {
            return; // Need at least 2 workers for balancing
        }

        // Sort by computational load
        load_distribution.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // 2. Identify overloaded and underloaded workers
        let avg_load: f64 = load_distribution
            .iter()
            .map(|(_, load, _)| *load)
            .sum::<f64>()
            / load_distribution.len() as f64;
        let load_threshold = 0.2; // 20% difference from average

        let mut migrations = Vec::new();
        let mut i = 0;
        let mut j = load_distribution.len() - 1;

        // 3. Plan parameter migrations from overloaded to underloaded workers
        while i < j {
            let (underloaded_id, underloaded_load, _) = load_distribution[i];
            let (overloaded_id, overloaded_load, overloaded_params) = load_distribution[j];

            if underloaded_load < avg_load - load_threshold
                && overloaded_load > avg_load + load_threshold
                && overloaded_params > 1
            {
                // Calculate how many parameters to migrate
                let load_diff = overloaded_load - underloaded_load;
                let params_to_migrate = ((load_diff / 2.0) * overloaded_params as f64) as usize;
                let params_to_migrate = params_to_migrate.max(1).min(overloaded_params / 2);

                migrations.push((overloaded_id, underloaded_id, params_to_migrate));

                // Update load distribution for next iteration
                let load_per_param = overloaded_load / overloaded_params as f64;
                load_distribution[i].1 += params_to_migrate as f64 * load_per_param;
                load_distribution[j].1 -= params_to_migrate as f64 * load_per_param;
                load_distribution[j].2 -= params_to_migrate;

                if load_distribution[i].1 >= avg_load - load_threshold {
                    i += 1;
                }
                if load_distribution[j].1 <= avg_load + load_threshold {
                    j -= 1;
                }
            } else {
                if underloaded_load >= avg_load - load_threshold {
                    i += 1;
                }
                if overloaded_load <= avg_load + load_threshold {
                    j -= 1;
                }
            }
        }

        // 4. Apply migrations
        let num_migrations = migrations.len();
        for (from_worker, to_worker, param_count) in &migrations {
            // Update from worker first
            if let Some(from_status) = worker_status.get_mut(from_worker) {
                from_status.assigned_parameters -= param_count;
                let load_per_param = from_status.computational_load
                    / (from_status.assigned_parameters + param_count) as f64;
                from_status.computational_load -= *param_count as f64 * load_per_param;
            }

            // Update to worker second
            if let Some(to_status) = worker_status.get_mut(to_worker) {
                to_status.assigned_parameters += param_count;
                // Estimate load increase for migrated parameters
                to_status.computational_load += (*param_count as f64 * 0.1).min(0.3);
                to_status.computational_load = to_status.computational_load.min(1.0);
            }

            println!(
                "Migrated {param_count} parameters from worker {from_worker} to worker {to_worker}"
            );
        }

        if num_migrations > 0 {
            println!("Completed dynamic load balancing with {num_migrations} migrations");
        }
    }
}
