//! Tests for curriculum_learning module.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions for tests
// ─────────────────────────────────────────────────────────────────────────────

fn make_predictions(n: usize, n_classes: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let raw: Vec<f64> = (0..n_classes).map(|_| rng.random_range(0.0..1.0)).collect();
            cl_softmax(&raw)
        })
        .collect()
}

fn make_labels(n: usize, n_classes: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(0..n_classes)).collect()
}

fn make_losses(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(0.01..2.0)).collect()
}

fn make_features(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: ClDifficultyScorer tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_difficulty_scorer_creation() {
    let scorer = ClDifficultyScorer::new(100, ClDifficultyMethod::LossBased);
    assert!(scorer.is_ok());
    let s = scorer.expect("should succeed");
    assert_eq!(s.n_samples, 100);
}

#[test]
fn test_difficulty_scorer_zero_samples() {
    let result = ClDifficultyScorer::new(0, ClDifficultyMethod::LossBased);
    assert!(result.is_err());
}

#[test]
fn test_difficulty_score_batch_loss_based() {
    let scorer =
        ClDifficultyScorer::new(10, ClDifficultyMethod::LossBased).expect("should succeed");
    let losses = vec![0.1, 0.5, 0.9, 0.3, 0.7];
    let preds = make_predictions(5, 3, 42);
    let labels = make_labels(5, 3, 42);
    let scores = scorer.score_batch(&losses, &preds, &labels).expect("ok");
    assert_eq!(scores.len(), 5);
    // Scores should be normalized to [0, 1]
    for &s in &scores {
        assert!((0.0..=1.0).contains(&s));
    }
    // Highest loss (0.9) should get highest difficulty
    let max_idx = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    assert_eq!(max_idx, 2); // index of 0.9
}

#[test]
fn test_difficulty_score_batch_margin() {
    let scorer =
        ClDifficultyScorer::new(10, ClDifficultyMethod::PredictionMargin).expect("should succeed");
    // Create predictions with clear margin differences
    let preds = vec![
        vec![0.9, 0.05, 0.05],  // Large margin -> easy
        vec![0.34, 0.33, 0.33], // Small margin -> hard
        vec![0.6, 0.3, 0.1],    // Medium margin
    ];
    let losses = vec![0.1, 0.5, 0.3];
    let labels = vec![0, 1, 2];
    let scores = scorer.score_batch(&losses, &preds, &labels).expect("ok");
    assert_eq!(scores.len(), 3);
    // Small margin (sample 1) should be hardest
    assert!(scores[1] >= scores[0]);
}

#[test]
fn test_difficulty_score_batch_combined() {
    let scorer = ClDifficultyScorer::new(10, ClDifficultyMethod::Combined).expect("should succeed");
    let losses = vec![0.1, 0.5, 0.9];
    let preds = make_predictions(3, 3, 42);
    let labels = make_labels(3, 3, 42);
    let scores = scorer.score_batch(&losses, &preds, &labels).expect("ok");
    assert_eq!(scores.len(), 3);
}

#[test]
fn test_difficulty_scorer_update_batch() {
    let mut scorer =
        ClDifficultyScorer::new(5, ClDifficultyMethod::LossBased).expect("should succeed");
    let indices = vec![0, 1, 2];
    let losses = vec![0.5, 1.0, 0.1];
    let preds = vec![vec![0.8, 0.2], vec![0.3, 0.7], vec![0.9, 0.1]];
    let labels = vec![0, 0, 0];
    let result = scorer.update_batch(&indices, &losses, &preds, &labels);
    assert!(result.is_ok());
    assert_eq!(scorer.loss_counts[0], 1);
    assert_eq!(scorer.loss_counts[1], 1);
    assert_eq!(scorer.loss_counts[2], 1);
}

#[test]
fn test_difficulty_scorer_forgetting_events() {
    let mut scorer =
        ClDifficultyScorer::new(3, ClDifficultyMethod::ForgettingEvents).expect("should succeed");
    // First: sample 0 is correct
    let _ = scorer.update_batch(&[0], &[0.1], &[vec![0.9, 0.1]], &[0]);
    assert!(scorer.prev_correct[0]);
    // Second: sample 0 becomes incorrect -> forgetting event
    let _ = scorer.update_batch(&[0], &[0.9], &[vec![0.3, 0.7]], &[0]);
    assert_eq!(scorer.forgetting_counts[0], 1);
}

#[test]
fn test_difficulty_scorer_ensemble_disagreement() {
    let scorer = ClDifficultyScorer::new(10, ClDifficultyMethod::EnsembleDisagreement)
        .expect("should succeed");
    let preds = vec![
        vec![0.5, 0.5],   // High entropy -> high difficulty
        vec![0.99, 0.01], // Low entropy -> low difficulty
    ];
    let losses = vec![0.5, 0.1];
    let labels = vec![0, 0];
    let scores = scorer.score_batch(&losses, &preds, &labels).expect("ok");
    assert!(scores[0] > scores[1]); // Higher entropy = harder
}

#[test]
fn test_difficulty_scorer_reset() {
    let mut scorer =
        ClDifficultyScorer::new(5, ClDifficultyMethod::LossBased).expect("should succeed");
    let _ = scorer.update_batch(&[0], &[0.5], &[vec![0.8, 0.2]], &[0]);
    scorer.reset();
    assert_eq!(scorer.accumulated_losses[0], 0.0);
    assert_eq!(scorer.loss_counts[0], 0);
}

#[test]
fn test_difficulty_accumulated_difficulties() {
    let mut scorer =
        ClDifficultyScorer::new(3, ClDifficultyMethod::LossBased).expect("should succeed");
    let _ = scorer.update_batch(
        &[0, 1, 2],
        &[0.1, 0.5, 0.9],
        &make_predictions(3, 2, 42),
        &[0, 1, 0],
    );
    let diffs = scorer.get_accumulated_difficulties();
    assert_eq!(diffs.len(), 3);
    // Highest loss sample should have highest difficulty
    assert!(diffs[2] >= diffs[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: ClCurriculumScheduler tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scheduler_creation() {
    let s = ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 100, 42);
    assert!(s.is_ok());
}

#[test]
fn test_scheduler_zero_steps() {
    let result = ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 0, 42);
    assert!(result.is_err());
}

#[test]
fn test_scheduler_linear_competence() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 100, 42).expect("ok");
    assert!(s.competence() <= 0.02); // Near 0 at start
    for _ in 0..50 {
        s.step();
    }
    let c = s.competence();
    assert!((c - 0.5).abs() < 0.05);
    for _ in 0..50 {
        s.step();
    }
    assert!((s.competence() - 1.0).abs() < 0.02);
}

#[test]
fn test_scheduler_root_competence() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Root, 100, 42).expect("ok");
    for _ in 0..25 {
        s.step();
    }
    let c = s.competence();
    assert!((c - 0.5).abs() < 0.1); // sqrt(0.25) = 0.5
}

#[test]
fn test_scheduler_exponential_competence() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Exponential, 100, 42).expect("ok");
    for _ in 0..100 {
        s.step();
    }
    assert!(s.competence() > 0.99);
}

#[test]
fn test_scheduler_step_competence() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Step, 100, 42).expect("ok");
    // At step 10 (< 25%), competence = 0.25
    for _ in 0..10 {
        s.step();
    }
    assert!((s.competence() - 0.25).abs() < 0.01);
    // At step 60 (> 50%, < 75%), competence = 0.75
    for _ in 0..50 {
        s.step();
    }
    assert!((s.competence() - 0.75).abs() < 0.01);
}

#[test]
fn test_scheduler_logarithmic_competence() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Logarithmic, 100, 42).expect("ok");
    for _ in 0..100 {
        s.step();
    }
    assert!((s.competence() - 1.0).abs() < 0.02);
}

#[test]
fn test_scheduler_select_batch() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 100, 42).expect("ok");
    // Move to mid-training
    for _ in 0..50 {
        s.step();
    }
    let difficulties = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let batch = s.select_batch(&difficulties, 3).expect("ok");
    assert_eq!(batch.len(), 3);
    // All selected should have difficulty <= competence (~0.5)
    let c = s.competence();
    for &idx in &batch {
        assert!(difficulties[idx] <= c + 0.01);
    }
}

#[test]
fn test_scheduler_anti_curriculum() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 100, 42).expect("ok");
    s.set_anti_curriculum(true);
    for _ in 0..50 {
        s.step();
    }
    let difficulties = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let batch = s.select_batch(&difficulties, 3).expect("ok");
    assert_eq!(batch.len(), 3);
    // Anti-curriculum: should select hard samples
    let c = s.competence();
    let threshold = 1.0 - c;
    for &idx in &batch {
        assert!(difficulties[idx] >= threshold - 0.01);
    }
}

#[test]
fn test_scheduler_baby_step() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 100, 42).expect("ok");
    let baby = ClBabyStepConfig::new(4, 5).expect("ok");
    s.set_baby_step(baby);
    let difficulties = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    // First batch: only bin 1 accessible (difficulty <= 0.25)
    let batch = s.select_batch(&difficulties, 10).expect("ok");
    assert!(!batch.is_empty());
}

#[test]
fn test_scheduler_reset() {
    let mut s = ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 100, 42).expect("ok");
    for _ in 0..50 {
        s.step();
    }
    s.reset();
    assert_eq!(s.current_step, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: SelfPacedLearning tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_spl_creation() {
    let spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 10);
    assert!(spl.is_ok());
}

#[test]
fn test_spl_invalid_params() {
    assert!(SelfPacedLearning::new(ClSplWeightingStrategy::Hard, -0.1, 5.0, 10).is_err());
    assert!(SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 0.3, 10).is_err());
    assert!(SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 0).is_err());
}

#[test]
fn test_spl_hard_weights() {
    let spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 10).expect("ok");
    let losses = vec![0.1, 0.3, 0.5, 0.7, 1.0];
    let weights = spl.compute_weights(&losses).expect("ok");
    // loss < lambda=0.5: weight=1, else weight=0
    assert!((weights[0] - 1.0).abs() < 1e-10);
    assert!((weights[1] - 1.0).abs() < 1e-10);
    assert!((weights[2] - 0.0).abs() < 1e-10); // 0.5 is not < 0.5
    assert!((weights[3] - 0.0).abs() < 1e-10);
    assert!((weights[4] - 0.0).abs() < 1e-10);
}

#[test]
fn test_spl_soft_weights() {
    let spl = SelfPacedLearning::new(ClSplWeightingStrategy::Soft, 1.0, 5.0, 10).expect("ok");
    let losses = vec![0.0, 0.5, 1.0, 1.5];
    let weights = spl.compute_weights(&losses).expect("ok");
    // Soft: max(0, 1 - loss/lambda)
    assert!((weights[0] - 1.0).abs() < 1e-10); // 1 - 0/1 = 1
    assert!((weights[1] - 0.5).abs() < 1e-10); // 1 - 0.5/1 = 0.5
    assert!((weights[2] - 0.0).abs() < 1e-10); // 1 - 1/1 = 0
    assert!((weights[3] - 0.0).abs() < 1e-10); // max(0, -0.5) = 0
}

#[test]
fn test_spl_mixture_weights() {
    let mut spl =
        SelfPacedLearning::new(ClSplWeightingStrategy::Mixture, 1.0, 5.0, 10).expect("ok");
    spl.set_mixture_alpha(0.5);
    let losses = vec![0.5];
    let weights = spl.compute_weights(&losses).expect("ok");
    // Mixture: alpha * hard + (1-alpha) * soft
    // hard: 1 (0.5 < 1.0), soft: 0.5 (1 - 0.5/1.0)
    // 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    assert!((weights[0] - 0.75).abs() < 1e-10);
}

#[test]
fn test_spl_logarithmic_weights() {
    let spl =
        SelfPacedLearning::new(ClSplWeightingStrategy::Logarithmic, 1.0, 5.0, 10).expect("ok");
    let losses = vec![0.0, 1.0, 3.0];
    let weights = spl.compute_weights(&losses).expect("ok");
    // lambda/(lambda+loss)
    assert!((weights[0] - 1.0).abs() < 1e-10); // 1/(1+0)
    assert!((weights[1] - 0.5).abs() < 1e-10); // 1/(1+1)
    assert!((weights[2] - 0.25).abs() < 1e-10); // 1/(1+3)
}

#[test]
fn test_spl_step_epoch_linear() {
    let mut spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 10).expect("ok");
    spl.set_schedule(ClLambdaSchedule::Linear);
    for _ in 0..5 {
        spl.step_epoch();
    }
    // lambda should be at midpoint: 0.5 + (5.0-0.5)*5/10 = 2.75
    assert!((spl.lambda - 2.75).abs() < 0.1);
}

#[test]
fn test_spl_step_epoch_geometric() {
    let mut spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 10).expect("ok");
    spl.set_schedule(ClLambdaSchedule::Geometric);
    for _ in 0..10 {
        spl.step_epoch();
    }
    // At T, lambda should approach lambda_max
    assert!((spl.lambda - 5.0).abs() < 0.1);
}

#[test]
fn test_spl_step_epoch_step_double() {
    let mut spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 20).expect("ok");
    spl.set_schedule(ClLambdaSchedule::StepDouble);
    spl.set_step_interval(3);
    // After 3 epochs: lambda = 0.5 * 2^1 = 1.0
    for _ in 0..3 {
        spl.step_epoch();
    }
    assert!((spl.lambda - 1.0).abs() < 0.01);
}

#[test]
fn test_spl_weighted_loss() {
    let spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 1.0, 5.0, 10).expect("ok");
    let losses = vec![0.2, 0.8, 1.5];
    let wl = spl.weighted_loss(&losses).expect("ok");
    // Only first two have weight=1 (< 1.0): (0.2+0.8)/2 = 0.5
    assert!((wl - 0.5).abs() < 1e-10);
}

#[test]
fn test_spl_reset() {
    let mut spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 10).expect("ok");
    for _ in 0..5 {
        spl.step_epoch();
    }
    spl.reset();
    assert!((spl.lambda - 0.5).abs() < 1e-10);
    assert_eq!(spl.current_epoch, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: CompetenceLearning tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_competence_creation() {
    let cl = CompetenceLearning::new(0.1, 0.9);
    assert!(cl.is_ok());
}

#[test]
fn test_competence_invalid_momentum() {
    assert!(CompetenceLearning::new(0.1, 1.5).is_err());
    assert!(CompetenceLearning::new(0.1, -0.1).is_err());
}

#[test]
fn test_competence_update_accuracy() {
    let mut cl = CompetenceLearning::new(0.1, 0.0).expect("ok");
    cl.set_confidence_method(ClConfidenceMethod::Accuracy);
    let preds = vec![
        vec![0.9, 0.1], // Correct (label=0)
        vec![0.3, 0.7], // Incorrect (label=0)
        vec![0.8, 0.2], // Correct (label=0)
    ];
    let labels = vec![0, 0, 0];
    let c = cl.update_competence(&preds, &labels).expect("ok");
    // 2 out of 3 correct = 0.667
    assert!((c - 2.0 / 3.0).abs() < 0.01);
}

#[test]
fn test_competence_update_mean_confidence() {
    let mut cl = CompetenceLearning::new(0.1, 0.0).expect("ok");
    cl.set_confidence_method(ClConfidenceMethod::MeanConfidence);
    let preds = vec![vec![0.8, 0.2], vec![0.6, 0.4]];
    let labels = vec![0, 0];
    let c = cl.update_competence(&preds, &labels).expect("ok");
    // Mean of (0.8, 0.6) = 0.7
    assert!((c - 0.7).abs() < 0.01);
}

#[test]
fn test_competence_update_inverse_entropy() {
    let mut cl = CompetenceLearning::new(0.1, 0.0).expect("ok");
    cl.set_confidence_method(ClConfidenceMethod::InverseEntropy);
    // Very confident predictions -> low entropy -> high competence
    let preds = vec![vec![0.99, 0.01], vec![0.98, 0.02]];
    let labels = vec![0, 0];
    let c = cl.update_competence(&preds, &labels).expect("ok");
    assert!(c > 0.5); // Should be high
}

#[test]
fn test_competence_filter_by_competence() {
    let mut cl = CompetenceLearning::new(0.1, 0.0).expect("ok");
    cl.competence = 0.5;
    let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let eligible = cl.filter_by_competence(&difficulties).expect("ok");
    // Indices 0, 1, 2 have difficulty <= 0.5
    assert_eq!(eligible.len(), 3);
    assert!(eligible.contains(&0));
    assert!(eligible.contains(&1));
    assert!(eligible.contains(&2));
}

#[test]
fn test_competence_filter_all_too_hard() {
    let mut cl = CompetenceLearning::new(0.0, 0.0).expect("ok");
    cl.competence = 0.0;
    let difficulties = vec![0.5, 0.6, 0.7];
    let eligible = cl.filter_by_competence(&difficulties).expect("ok");
    // Should return at least the easiest sample
    assert!(!eligible.is_empty());
}

#[test]
fn test_competence_estimate_difficulty_from_agreement() {
    let cl = CompetenceLearning::new(0.1, 0.0).expect("ok");
    let model_preds = vec![
        // Model 1
        vec![vec![0.9, 0.1], vec![0.7, 0.3], vec![0.8, 0.2]],
        // Model 2
        vec![vec![0.9, 0.1], vec![0.2, 0.8], vec![0.8, 0.2]],
    ];
    let difficulties = cl
        .estimate_difficulty_from_agreement(&model_preds)
        .expect("ok");
    assert_eq!(difficulties.len(), 3);
    // Sample 0 and 2: both models agree -> low difficulty
    // Sample 1: models disagree -> high difficulty
    assert!(difficulties[1] > difficulties[0]);
}

#[test]
fn test_competence_ema_smoothing() {
    let mut cl = CompetenceLearning::new(0.1, 0.9).expect("ok");
    let preds_good = vec![vec![0.9, 0.1]];
    let preds_bad = vec![vec![0.1, 0.9]];
    let labels = vec![0];

    let _ = cl.update_competence(&preds_good, &labels);
    let c1 = cl.competence;
    let _ = cl.update_competence(&preds_bad, &labels);
    let c2 = cl.competence;
    // With high momentum, c2 shouldn't drop too much from c1
    assert!(c2 > 0.5);
    let _ = c1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: DataShapley tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_data_shapley_creation() {
    let ds = DataShapley::new(10, 0.01);
    assert!(ds.is_ok());
}

#[test]
fn test_data_shapley_zero_perms() {
    assert!(DataShapley::new(0, 0.01).is_err());
}

#[test]
fn test_data_shapley_compute() {
    let ds = DataShapley::new(5, 0.001).expect("ok");
    // Simple model_fn: performance = fraction of useful samples included
    let useful_samples = [0, 2, 4]; // Odd-indexed samples are "noise"
    let model_fn = |subset: &[usize]| -> Result<f64, TensorError> {
        let useful_count = subset
            .iter()
            .filter(|&&s| useful_samples.contains(&s))
            .count();
        Ok(useful_count as f64 / useful_samples.len() as f64)
    };
    let values = ds.compute(5, &model_fn).expect("ok");
    assert_eq!(values.len(), 5);
    // Useful samples should have higher Shapley values
    assert!(values[0] > values[1] || values[2] > values[1] || values[4] > values[3]);
}

#[test]
fn test_data_shapley_identify_groups() {
    let ds = DataShapley::new(10, 0.01).expect("ok");
    let values = vec![0.01, 0.02, 0.5, 0.6, 0.9, -0.1, 0.0, 0.3, 0.4, 0.8];
    let (high, low) = ds.identify_value_groups(&values, 20.0, 20.0).expect("ok");
    // Top 20% (2 samples): should be the highest values
    assert!(!high.is_empty());
    // Bottom 20% (2 samples): should be the lowest values
    assert!(!low.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: KnnShapley tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_knn_shapley_creation() {
    let ks = KnnShapley::new(3);
    assert!(ks.is_ok());
}

#[test]
fn test_knn_shapley_zero_k() {
    assert!(KnnShapley::new(0).is_err());
}

#[test]
fn test_knn_shapley_compute() {
    let ks = KnnShapley::new(2).expect("ok");
    // Simple 2D data: two clusters
    let train_features = vec![
        vec![0.0, 0.0], // Class 0
        vec![0.1, 0.1], // Class 0
        vec![1.0, 1.0], // Class 1
        vec![1.1, 1.1], // Class 1
    ];
    let train_labels = vec![0, 0, 1, 1];
    let test_features = vec![
        vec![0.05, 0.05], // Near class 0
        vec![1.05, 1.05], // Near class 1
    ];
    let test_labels = vec![0, 1];
    let values = ks
        .compute(&train_features, &train_labels, &test_features, &test_labels)
        .expect("ok");
    assert_eq!(values.len(), 4);
    // All values should be positive (all samples contribute)
    let sum: f64 = values.iter().sum();
    assert!(sum > 0.0);
}

#[test]
fn test_knn_shapley_mismatched_dims() {
    let ks = KnnShapley::new(2).expect("ok");
    let result = ks.compute(&[vec![1.0, 2.0]], &[0], &[], &[]);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: LavaValuation tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lava_creation() {
    let lava = LavaValuation::new(0.1, 50);
    assert!(lava.is_ok());
}

#[test]
fn test_lava_invalid_params() {
    assert!(LavaValuation::new(-0.1, 50).is_err());
    assert!(LavaValuation::new(0.1, 0).is_err());
}

#[test]
fn test_lava_compute_dual() {
    let lava = LavaValuation::new(0.5, 20).expect("ok");
    let train = vec![
        vec![0.0, 0.0],
        vec![0.5, 0.5],
        vec![10.0, 10.0], // Far from validation
    ];
    let val = vec![vec![0.1, 0.1], vec![0.4, 0.4]];
    let scores = lava.compute(&train, &val).expect("ok");
    assert_eq!(scores.len(), 3);
    // Scores are normalized [0,1]
    for &s in &scores {
        assert!((0.0..=1.0).contains(&s));
    }
    // Point 2 (far) should have lower value than point 0 or 1 (close)
    assert!(scores[2] < scores[0] || scores[2] < scores[1]);
}

#[test]
fn test_lava_compute_nn() {
    let mut lava = LavaValuation::new(0.5, 20).expect("ok");
    lava.use_dual = false;
    let train = vec![vec![0.0, 0.0], vec![5.0, 5.0]];
    let val = vec![vec![0.1, 0.1]];
    let scores = lava.compute(&train, &val).expect("ok");
    assert_eq!(scores.len(), 2);
    // Point 0 closer to val -> higher score
    assert!(scores[0] > scores[1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: AutoCurriculum tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_auto_curriculum_creation() {
    let ac = AutoCurriculum::new(5, 1.41);
    assert!(ac.is_ok());
    let ac = ac.expect("ok");
    assert_eq!(ac.n_bins, 5);
    assert_eq!(ac.bin_boundaries.len(), 6);
}

#[test]
fn test_auto_curriculum_zero_bins() {
    assert!(AutoCurriculum::new(0, 1.0).is_err());
}

#[test]
fn test_auto_curriculum_select_bin_round_robin() {
    let ac = AutoCurriculum::new(3, 1.41).expect("ok");
    // Should select each bin once before using UCB1
    let b0 = ac.select_bin().expect("ok");
    assert_eq!(b0, 0);
}

#[test]
fn test_auto_curriculum_update_reward() {
    let mut ac = AutoCurriculum::new(3, 1.41).expect("ok");
    ac.update_reward(0, 0.5).expect("ok");
    ac.update_reward(1, 0.3).expect("ok");
    ac.update_reward(2, 0.1).expect("ok");
    assert_eq!(ac.total_pulls, 3);
    assert_eq!(ac.arms[0].pull_count, 1);
    assert!((ac.arms[0].total_reward - 0.5).abs() < 1e-10);
}

#[test]
fn test_auto_curriculum_ucb1_selection() {
    let mut ac = AutoCurriculum::new(3, 1.41).expect("ok");
    // Give bin 0 high reward, bins 1,2 low
    for _ in 0..5 {
        ac.update_reward(0, 1.0).expect("ok");
        ac.update_reward(1, 0.1).expect("ok");
        ac.update_reward(2, 0.05).expect("ok");
    }
    let selected = ac.select_bin().expect("ok");
    // Bin 0 should be preferred due to higher mean reward
    assert_eq!(selected, 0);
}

#[test]
fn test_auto_curriculum_get_bin_samples() {
    let ac = AutoCurriculum::new(4, 1.0).expect("ok");
    let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let bin0 = ac.get_bin_samples(&difficulties, 0).expect("ok");
    // Bin 0: [0, 0.25) -> sample 0 (0.1)
    assert!(bin0.contains(&0));
    assert!(!bin0.contains(&2));
}

#[test]
fn test_auto_curriculum_invalid_bin() {
    let ac = AutoCurriculum::new(3, 1.0).expect("ok");
    assert!(ac.get_bin_samples(&[0.5], 5).is_err());
}

#[test]
fn test_auto_curriculum_mean_rewards() {
    let mut ac = AutoCurriculum::new(2, 1.0).expect("ok");
    ac.update_reward(0, 1.0).expect("ok");
    ac.update_reward(0, 3.0).expect("ok");
    ac.update_reward(1, 0.5).expect("ok");
    let means = ac.mean_rewards();
    assert!((means[0] - 2.0).abs() < 1e-10);
    assert!((means[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_auto_curriculum_reset() {
    let mut ac = AutoCurriculum::new(3, 1.0).expect("ok");
    ac.update_reward(0, 1.0).expect("ok");
    ac.reset();
    assert_eq!(ac.total_pulls, 0);
    assert_eq!(ac.arms[0].pull_count, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: CurriculumTrainer tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_trainer_creation() {
    let t = CurriculumTrainer::new(ClTrainingMode::Curriculum, 4, 10, 100, 42);
    assert!(t.is_ok());
}

#[test]
fn test_trainer_invalid_params() {
    assert!(CurriculumTrainer::new(ClTrainingMode::Curriculum, 0, 10, 100, 42).is_err());
    assert!(CurriculumTrainer::new(ClTrainingMode::Curriculum, 4, 0, 100, 42).is_err());
    assert!(CurriculumTrainer::new(ClTrainingMode::Curriculum, 4, 10, 0, 42).is_err());
}

#[test]
fn test_trainer_random_epoch() {
    let mut trainer = CurriculumTrainer::new(ClTrainingMode::Random, 4, 10, 20, 42).expect("ok");
    let difficulties = vec![0.1; 20];
    let losses = vec![0.5; 20];
    let train_fn =
        |indices: &[usize]| -> Result<f64, TensorError> { Ok(indices.len() as f64 * 0.1) };
    let result = trainer
        .train_epoch(&difficulties, &losses, &train_fn, None, None, None, None)
        .expect("ok");
    assert_eq!(result.n_samples_used, 4);
    assert_eq!(result.n_samples_total, 20);
    assert!(result.avg_loss > 0.0);
}

#[test]
fn test_trainer_curriculum_epoch() {
    let mut trainer =
        CurriculumTrainer::new(ClTrainingMode::Curriculum, 4, 10, 20, 42).expect("ok");
    let mut scheduler =
        ClCurriculumScheduler::new(ClCompetenceStrategy::Linear, 10, 42).expect("ok");
    for _ in 0..5 {
        scheduler.step();
    }
    let difficulties: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
    let losses = vec![0.5; 20];
    let train_fn = |indices: &[usize]| -> Result<f64, TensorError> { Ok(0.3) };
    let result = trainer
        .train_epoch(
            &difficulties,
            &losses,
            &train_fn,
            Some(&mut scheduler),
            None,
            None,
            None,
        )
        .expect("ok");
    assert!(result.n_samples_used > 0);
    assert!(result.n_samples_used <= 4);
}

#[test]
fn test_trainer_spl_epoch() {
    let mut trainer = CurriculumTrainer::new(ClTrainingMode::SelfPaced, 4, 10, 20, 42).expect("ok");
    let spl = SelfPacedLearning::new(ClSplWeightingStrategy::Hard, 0.5, 5.0, 10).expect("ok");
    let difficulties = vec![0.5; 20];
    let losses: Vec<f64> = (0..20).map(|i| i as f64 / 10.0).collect();
    let train_fn = |_: &[usize]| -> Result<f64, TensorError> { Ok(0.2) };
    let result = trainer
        .train_epoch(
            &difficulties,
            &losses,
            &train_fn,
            None,
            Some(&spl),
            None,
            None,
        )
        .expect("ok");
    assert!(result.n_samples_used > 0);
}

#[test]
fn test_trainer_reset() {
    let mut trainer = CurriculumTrainer::new(ClTrainingMode::Random, 4, 10, 20, 42).expect("ok");
    let difficulties = vec![0.1; 20];
    let losses = vec![0.5; 20];
    let train_fn = |_: &[usize]| -> Result<f64, TensorError> { Ok(0.5) };
    let _ = trainer.train_epoch(&difficulties, &losses, &train_fn, None, None, None, None);
    trainer.reset();
    assert_eq!(trainer.current_epoch, 0);
    assert!(trainer.loss_history.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: ClMetrics tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_convergence_speed() {
    let losses = vec![2.0, 1.5, 1.0, 0.8, 0.5, 0.4, 0.35, 0.3];
    let speed = CurriculumMetrics::convergence_speed(&losses, 0.1);
    // Best loss = 0.3, target = 0.33
    // First epoch with loss <= 0.33 is epoch 7 (0.3)
    assert!(speed <= 8);
}

#[test]
fn test_metrics_convergence_speed_empty() {
    assert_eq!(CurriculumMetrics::convergence_speed(&[], 0.1), 0);
}

#[test]
fn test_metrics_selection_gini_uniform() {
    let counts = vec![10, 10, 10, 10, 10];
    let gini = CurriculumMetrics::selection_gini(&counts);
    assert!(gini < 0.1); // Should be near 0 for uniform
}

#[test]
fn test_metrics_selection_gini_concentrated() {
    let counts = vec![100, 0, 0, 0, 0];
    let gini = CurriculumMetrics::selection_gini(&counts);
    assert!(gini > 0.5); // Should be high for concentrated
}

#[test]
fn test_metrics_avg_utilization() {
    let history = vec![0.2, 0.4, 0.6, 0.8, 1.0];
    let avg = CurriculumMetrics::avg_utilization(&history);
    assert!((avg - 0.6).abs() < 1e-10);
}

#[test]
fn test_metrics_difficulty_histogram() {
    let difficulties = vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
    let hist = CurriculumMetrics::difficulty_histogram(&difficulties, 5);
    assert_eq!(hist.len(), 5);
    // Sum should equal number of samples
    let total: usize = hist.iter().sum();
    assert_eq!(total, 6);
}

#[test]
fn test_metrics_learning_curve_ratio() {
    let method = vec![1.0, 0.5, 0.3];
    let baseline = vec![1.0, 0.7, 0.5];
    let ratio = CurriculumMetrics::learning_curve_ratio(&method, &baseline);
    // Method converges faster -> lower AUC -> ratio < 1
    assert!(ratio < 1.0);
}

#[test]
fn test_metrics_generate_report() {
    let losses = vec![2.0, 1.5, 1.0, 0.8, 0.5];
    let utilization = vec![0.2, 0.4, 0.6, 0.8, 1.0];
    let counts = vec![2, 3, 4, 5, 1, 0, 2, 3, 4, 5];
    let report =
        CurriculumMetrics::generate_report("linear_curriculum", &losses, &utilization, &counts);
    assert_eq!(report.strategy, "linear_curriculum");
    assert!((report.best_loss - 0.5).abs() < 1e-10);
    assert_eq!(report.best_epoch, 4);
    assert!(report.avg_utilization > 0.0);
}

#[test]
fn test_metrics_compare_reports() {
    let report_a = ClReport {
        strategy: "curriculum".to_string(),
        final_loss: 0.3,
        best_loss: 0.2,
        best_epoch: 8,
        convergence_speed: 5,
        avg_utilization: 0.6,
        selection_gini: 0.3,
        learning_curve: vec![1.0, 0.5, 0.3, 0.2],
    };
    let report_b = ClReport {
        strategy: "random".to_string(),
        final_loss: 0.5,
        best_loss: 0.4,
        best_epoch: 9,
        convergence_speed: 7,
        avg_utilization: 1.0,
        selection_gini: 0.0,
        learning_curve: vec![1.0, 0.7, 0.5, 0.4],
    };
    let better = CurriculumMetrics::compare_reports(&report_a, &report_b);
    assert_eq!(better, "curriculum");
}

#[test]
fn test_metrics_effective_dataset_size() {
    let counts = vec![5, 0, 3, 0, 1];
    assert_eq!(CurriculumMetrics::effective_dataset_size(&counts), 3);
}

#[test]
fn test_metrics_sample_efficiency() {
    let losses = vec![2.0, 1.0];
    let samples = vec![10, 10];
    let eff = CurriculumMetrics::sample_efficiency(&losses, &samples);
    // improvement = 2.0 - 1.0 = 1.0, total_samples = 20
    assert!((eff - 0.05).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_integration_full_curriculum_pipeline() {
    let n_samples = 50;
    let n_classes = 3;
    let total_epochs = 5;
    let batch_size = 10;

    // Create components
    let mut scorer = ClDifficultyScorer::new(n_samples, ClDifficultyMethod::LossBased).expect("ok");
    let mut scheduler = ClCurriculumScheduler::new(
        ClCompetenceStrategy::Root,
        total_epochs * (n_samples / batch_size),
        42,
    )
    .expect("ok");
    let mut trainer = CurriculumTrainer::new(
        ClTrainingMode::Curriculum,
        batch_size,
        total_epochs,
        n_samples,
        42,
    )
    .expect("ok");

    let predictions = make_predictions(n_samples, n_classes, 42);
    let labels = make_labels(n_samples, n_classes, 42);

    // Score initial difficulties
    let losses = make_losses(n_samples, 42);
    let difficulties = scorer
        .score_batch(&losses, &predictions, &labels)
        .expect("ok");

    // Run training epochs
    let train_fn =
        |indices: &[usize]| -> Result<f64, TensorError> { Ok(indices.len() as f64 * 0.02) };

    for _epoch in 0..total_epochs {
        let result = trainer
            .train_epoch(
                &difficulties,
                &losses,
                &train_fn,
                Some(&mut scheduler),
                None,
                None,
                None,
            )
            .expect("ok");
        assert!(result.n_samples_used > 0);
        assert!(result.n_samples_used <= batch_size);
    }

    // Generate report
    let report = CurriculumMetrics::generate_report(
        "root_curriculum",
        &trainer.loss_history,
        &trainer.utilization_history,
        &trainer.selection_counts,
    );
    assert_eq!(report.learning_curve.len(), total_epochs);
}

#[test]
fn test_integration_spl_with_annealing() {
    let mut spl = SelfPacedLearning::new(ClSplWeightingStrategy::Soft, 0.3, 3.0, 10).expect("ok");
    spl.set_schedule(ClLambdaSchedule::Geometric);

    let losses = vec![0.1, 0.3, 0.5, 0.8, 1.2, 2.0];

    // Initially, only easy samples should have weight
    let w0 = spl.compute_weights(&losses).expect("ok");
    let active0 = w0.iter().filter(|&&w| w > 0.01).count();

    // After annealing, more samples should be included
    for _ in 0..8 {
        spl.step_epoch();
    }
    let w8 = spl.compute_weights(&losses).expect("ok");
    let active8 = w8.iter().filter(|&&w| w > 0.01).count();

    assert!(active8 >= active0);
}

#[test]
fn test_integration_knn_shapley_identifies_useful_data() {
    let ks = KnnShapley::new(2).expect("ok");
    // Create clusters where one cluster is noisy (mislabeled)
    let train = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0], // Class 0 cluster
        vec![1.0, 1.0],
        vec![1.1, 1.0], // Class 1 cluster
        vec![0.5, 0.5], // Noisy point labeled as class 0 (between clusters)
    ];
    let train_labels = vec![0, 0, 1, 1, 0];
    let test = vec![
        vec![0.05, 0.0], // Near class 0
        vec![1.05, 1.0], // Near class 1
    ];
    let test_labels = vec![0, 1];
    let values = ks
        .compute(&train, &train_labels, &test, &test_labels)
        .expect("ok");
    assert_eq!(values.len(), 5);
}
