#![allow(clippy::result_large_err)]

//! Modern ML Dataset Examples
//!
//! This example demonstrates how to use the modern ML dataset generators
//! for few-shot learning, meta-learning, contrastive learning, and self-supervised learning.

use tenflowers_dataset::{
    ContrastiveLearningDataset, FewShotDataset, MetaLearningDataset, ModernMLConfig,
    SelfSupervisedDataset,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Modern ML Dataset Generation Examples\n");

    // Configuration for all examples
    let config = ModernMLConfig {
        seed: 42,
        feature_dim: 64,
        noise_level: 0.1,
    };

    // 1. Few-Shot Learning Dataset
    println!("📚 Few-Shot Learning Dataset:");
    let few_shot_dataset = FewShotDataset::<f32>::new(
        10, // 10 episodes
        5,  // 5-way classification
        3,  // 3-shot (3 examples per class in support set)
        2,  // 2 query examples per class
        &config,
    )?;

    println!("  • Created {} episodes", few_shot_dataset.num_episodes());

    let mut few_shot_mut = few_shot_dataset; // Make it mutable for iteration
    if let Some(episode) = few_shot_mut.next_episode() {
        println!(
            "  • Episode 1: {}-way {}-shot",
            episode.n_way, episode.k_shot
        );
        println!("  • Support set size: {}", episode.support_set.len());
        println!("  • Query set size: {}", episode.query_set.len());
    }

    // 2. Contrastive Learning Dataset
    println!("\n🔗 Contrastive Learning Dataset:");
    let contrastive_dataset = ContrastiveLearningDataset::<f32>::new(
        50,  // 50 positive pairs
        100, // 100 negative pairs
        &config,
    )?;

    println!(
        "  • Positive pairs: {}",
        contrastive_dataset.positive_pairs().len()
    );
    println!(
        "  • Negative pairs: {}",
        contrastive_dataset.negative_pairs().len()
    );

    if let Some((anchor, positive)) = contrastive_dataset.get_positive_pair(0) {
        println!(
            "  • First positive pair shapes: {:?}, {:?}",
            anchor.shape().dims(),
            positive.shape().dims()
        );
    }

    // 3. Self-Supervised Learning Dataset
    println!("\n🔄 Self-Supervised Learning Dataset:");
    let ssl_dataset = SelfSupervisedDataset::<f32>::new(
        25, // 25 original samples
        4,  // 4 augmentations per sample
        &config,
    )?;

    println!("  • Original samples: {}", ssl_dataset.len());
    if let Some(augmentations) = ssl_dataset.get_augmentations(0) {
        println!("  • Augmentations per sample: {}", augmentations.len());
        println!(
            "  • Augmentation shape: {:?}",
            augmentations[0].shape().dims()
        );
    }

    // 4. Meta-Learning Dataset
    println!("\n🧠 Meta-Learning Dataset:");
    let meta_dataset = MetaLearningDataset::<f32>::new(
        8,   // 8 different tasks
        100, // 100 samples per task
        0.2, // 20% test split
        &config,
    )?;

    println!("  • Number of tasks: {}", meta_dataset.num_tasks());
    if let Some(task) = meta_dataset.get_task(0) {
        println!("  • Task 0 - Train samples: {}", task.train_data.len());
        println!("  • Task 0 - Test samples: {}", task.test_data.len());
        if let Some((features, label)) = task.train_data.first() {
            println!("  • Feature shape: {:?}", features.shape().dims());
            println!("  • Label shape: {:?}", label.shape().dims());
        }
    }

    println!("\n✅ All modern ML datasets created successfully!");
    println!("\nThese datasets are ready for:");
    println!("  • Few-shot learning experiments (MAML, ProtoNet, etc.)");
    println!("  • Contrastive learning (SimCLR, MoCo, etc.)");
    println!("  • Self-supervised pretraining");
    println!("  • Meta-learning and transfer learning research");

    Ok(())
}
