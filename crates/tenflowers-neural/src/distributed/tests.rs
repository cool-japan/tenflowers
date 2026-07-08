//! Tests for the distributed module

#[cfg(test)]
mod tests {
    use crate::backends::thread::ThreadBackend;
    use crate::distributed::types::{
        BackendConfig, CollectiveOp, CommunicationBackend, CommunicationBackendImpl,
        CommunicationGroup, CommunicationRuntime, ReductionOp,
    };
    use tenflowers_core::{Device, Tensor};

    #[test]
    fn test_communication_runtime_creation() {
        let runtime = CommunicationRuntime::new();
        assert!(runtime.groups.is_empty());
        assert!(runtime.default_group.is_none());
    }

    #[test]
    fn test_thread_backend() {
        let mut backend = ThreadBackend::new();
        let config = BackendConfig::default();

        assert!(backend.initialize(&config).is_ok());
        assert_eq!(backend.name(), "thread");
    }

    #[test]
    fn test_communication_group_creation() {
        let mut runtime = CommunicationRuntime::new();
        runtime.register_backend(CommunicationBackend::Thread, Box::new(ThreadBackend::new()));

        let config = BackendConfig::default();
        runtime
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test_group".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Cpu],
            backend: CommunicationBackend::Thread,
        };

        assert!(runtime.create_group(group).is_ok());
        assert!(runtime.get_group("test_group").is_some());
    }

    #[test]
    fn test_all_reduce_operation_returns_honest_error() {
        // `ThreadBackend::all_reduce_f32` (via `simulate_all_reduce`) cannot perform
        // a genuine cross-rank reduction: separate `ThreadBackend` instances never
        // share `shared_state`, so this call only ever sees its own local `tensor`
        // and has no other rank's data to combine it with. `Sum`/`Average` used to
        // fabricate a result by scaling or echoing back that local tensor as if it
        // were a real cross-rank reduction; now they honestly error out just like
        // `Min`/`Max`/`Product` already did, and `CommunicationRuntime::
        // collective_op_f32` must propagate that error rather than reporting
        // success. See `ThreadBackend::simulate_all_reduce` and
        // `backends::thread::tests::test_thread_all_reduce_returns_honest_error`.
        let mut runtime = CommunicationRuntime::new();
        runtime.register_backend(CommunicationBackend::Thread, Box::new(ThreadBackend::new()));

        let config = BackendConfig::default();
        runtime
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test_group".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Cpu],
            backend: CommunicationBackend::Thread,
        };

        runtime
            .create_group(group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[2, 3]);
        let op = CollectiveOp::AllReduce {
            reduction_op: ReductionOp::Sum,
        };

        let result = runtime.collective_op_f32(op, &tensor, Some("test_group"));

        assert!(
            result.is_err(),
            "all-reduce must never silently fabricate a cross-rank result for the \
             thread backend; expected an honest error, got {result:?}"
        );
    }
}
