//! Debug test to understand the NUMA dataloader issue

#[cfg(test)]
mod debug_test {
    use tenflowers_dataset::{
        TensorDataset, EnhancedDataLoaderBuilder, SequentialSampler,
        NumaConfig, Dataset
    };
    use tenflowers_core::Tensor;

    #[test]
    fn debug_numa_dataloader_issue() {
        println!("Starting debug test...");
        
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            &[4, 2]
        ).unwrap();
        let labels = Tensor::<f32>::from_vec(
            vec![0.0, 1.0, 0.0, 1.0], 
            &[4]
        ).unwrap();

        let dataset = TensorDataset::new(features, labels);
        println!("Dataset created with length: {}", dataset.len());
        
        let sampler = SequentialSampler::new();

        let loader = EnhancedDataLoaderBuilder::new()
            .batch_size(2)
            .num_workers(2)
            .numa_config(NumaConfig::default())
            .build(dataset, sampler);

        assert!(loader.is_ok());
        
        let loader = loader.unwrap();
        
        // Check queue stats before iterating
        let (queue_lengths, total_tasks, is_empty) = loader.get_queue_stats();
        println!("Queue stats: lengths={:?}, total_tasks={}, is_empty={}", queue_lengths, total_tasks, is_empty);
        
        // Test that loader can process at least one batch
        let mut batch_count = 0;
        
        for (i, batch) in loader.enumerate() {
            println!("Processing batch {}", i);
            match batch {
                Ok(batch_result) => {
                    println!("Batch {} success: features shape = {:?}, labels shape = {:?}", 
                        i, batch_result.features.shape(), batch_result.labels.shape());
                    batch_count += 1;
                }
                Err(e) => {
                    println!("Batch {} error: {:?}", i, e);
                }
            }
            
            // Just ensure we can process at least one batch successfully
            if batch_count >= 2 {
                break;
            }
        }
        
        println!("Processed {} batches", batch_count);
        
        // We should have processed at least one batch
        assert!(batch_count >= 1, "Expected at least 1 batch, got {}", batch_count);
        
        println!("Test passed!");
    }
}