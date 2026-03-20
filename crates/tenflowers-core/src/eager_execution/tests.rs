//! Tests for eager execution module

#[cfg(test)]
mod tests {
    use super::super::config::EagerExecutionConfig;
    use super::super::engine::EagerExecutionEngine;

    #[test]
    fn test_eager_execution_config() {
        let config = EagerExecutionConfig::default();
        assert!(config.enable_op_cache);
        assert!(config.enable_memory_pool);
        assert_eq!(config.target_overhead_ns, 1_000_000); // 1ms
    }

    #[test]
    fn test_op_signature_creation() {
        let engine = EagerExecutionEngine::new(EagerExecutionConfig::default());

        // Mock tensor creation would be needed for real test
        // This test validates the concept
        assert_eq!(engine.config.target_overhead_ns, 1_000_000);
    }

    #[test]
    fn test_performance_report_generation() {
        let engine = EagerExecutionEngine::new(EagerExecutionConfig::default());
        let report = engine.generate_performance_report();

        // Empty report should have default values
        assert_eq!(report.total_operations, 0);
        assert_eq!(report.success_rate, 0.0);
    }

    #[test]
    fn test_cache_statistics() {
        let engine = EagerExecutionEngine::new(EagerExecutionConfig::default());
        let stats = engine.get_cache_stats();

        // Initial cache should be empty
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }
}
