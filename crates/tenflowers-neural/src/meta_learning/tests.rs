//! Tests for meta-learning algorithms.

#[cfg(test)]
mod tests {
    use crate::meta_learning::{
        few_shot::{attention_kernel, matching_networks_predict, EpisodeSampler},
        maml::{maml_inner_update, maml_meta_gradient, MamlConfig},
        prototypical::{
            cosine_similarity, dot_product, euclidean_distance, DistanceMetric, PrototypicalNetwork,
        },
        reptile::{reptile_meta_update, ReptileConfig},
        types::MetaLearningError,
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    /// A trivial loss: sum of (params[0][i] - target)^2 averaged over params.
    fn toy_loss(params: &[Vec<f32>], input: &[f32]) -> f32 {
        let target = input.first().copied().unwrap_or(0.0);
        let layer = &params[0];
        layer.iter().map(|p| (p - target).powi(2)).sum::<f32>() / layer.len() as f32
    }

    /// Gradient of `toy_loss`: dL/dp_i = 2*(p_i - target)/n.
    fn toy_grad(params: &[Vec<f32>], input: &[f32]) -> Vec<Vec<f32>> {
        let target = input.first().copied().unwrap_or(0.0);
        let layer = &params[0];
        let n = layer.len() as f32;
        let grad_layer: Vec<f32> = layer.iter().map(|p| 2.0 * (p - target) / n).collect();
        vec![grad_layer]
    }

    // ── MAML tests ────────────────────────────────────────────────────────────

    #[test]
    fn maml_inner_update_moves_params_toward_target() {
        let params = vec![vec![1.0_f32, 1.0, 1.0]];
        let support_inputs = vec![vec![0.0_f32]]; // target = 0.0
        let support_targets = vec![0.0_f32];
        let config = MamlConfig {
            inner_lr: 0.5,
            meta_lr: 0.01,
            num_inner_steps: 1,
            first_order: true,
        };
        let adapted = maml_inner_update(
            &params,
            &support_inputs,
            &support_targets,
            &config,
            toy_loss,
            toy_grad,
        );
        // Params should move toward 0.0.
        for p in &adapted[0] {
            assert!(*p < 1.0, "adapted param {p} should be < 1.0");
        }
    }

    #[test]
    fn maml_inner_update_multiple_steps_reduce_loss() {
        let params = vec![vec![2.0_f32, 2.0]];
        let support_inputs = vec![vec![0.0_f32]];
        let support_targets = vec![0.0_f32];
        let config_1 = MamlConfig {
            inner_lr: 0.1,
            meta_lr: 0.01,
            num_inner_steps: 1,
            first_order: false,
        };
        let config_5 = MamlConfig {
            num_inner_steps: 5,
            ..config_1.clone()
        };
        let adapted_1 = maml_inner_update(
            &params,
            &support_inputs,
            &support_targets,
            &config_1,
            toy_loss,
            toy_grad,
        );
        let adapted_5 = maml_inner_update(
            &params,
            &support_inputs,
            &support_targets,
            &config_5,
            toy_loss,
            toy_grad,
        );
        let loss_1 = toy_loss(&adapted_1, &[0.0]);
        let loss_5 = toy_loss(&adapted_5, &[0.0]);
        assert!(
            loss_5 <= loss_1,
            "5-step adapted loss {loss_5} should be <= 1-step loss {loss_1}"
        );
    }

    #[test]
    fn maml_meta_gradient_meta_loss_is_mean_of_task_losses() {
        let params = vec![vec![1.0_f32, 1.0]];
        let tasks = vec![
            (
                vec![vec![0.0_f32]],
                vec![0.0_f32],
                vec![vec![0.0_f32]],
                vec![0.0_f32],
            ),
            (
                vec![vec![2.0_f32]],
                vec![2.0_f32],
                vec![vec![2.0_f32]],
                vec![2.0_f32],
            ),
        ];
        let config = MamlConfig {
            inner_lr: 0.1,
            meta_lr: 0.01,
            num_inner_steps: 1,
            first_order: true,
        };
        let result = maml_meta_gradient(&params, &tasks, &config, toy_loss, toy_grad);
        let expected_meta_loss =
            result.task_losses.iter().sum::<f32>() / result.task_losses.len() as f32;
        assert!(
            (result.meta_loss - expected_meta_loss).abs() < 1e-5,
            "meta_loss should equal mean of task losses"
        );
    }

    #[test]
    fn maml_meta_gradient_returns_correct_number_of_task_losses() {
        let params = vec![vec![0.0_f32]];
        let tasks: Vec<(Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> = (0..5)
            .map(|i| {
                (
                    vec![vec![i as f32]],
                    vec![i as f32],
                    vec![vec![i as f32]],
                    vec![i as f32],
                )
            })
            .collect();
        let config = MamlConfig::new(0.01, 0.001);
        let result = maml_meta_gradient(&params, &tasks, &config, toy_loss, toy_grad);
        assert_eq!(result.task_losses.len(), 5);
    }

    #[test]
    fn maml_meta_gradient_gradient_shape_matches_params() {
        let params = vec![vec![1.0_f32; 4], vec![0.0_f32; 2]];
        let tasks = vec![(
            vec![vec![1.0_f32]],
            vec![1.0_f32],
            vec![vec![1.0_f32]],
            vec![1.0_f32],
        )];
        let config = MamlConfig::new(0.1, 0.01);
        let result = maml_meta_gradient(&params, &tasks, &config, toy_loss, toy_grad);
        assert_eq!(result.param_gradients.len(), params.len());
        assert_eq!(result.param_gradients[0].len(), 4);
        assert_eq!(result.param_gradients[1].len(), 2);
    }

    // ── Reptile tests ─────────────────────────────────────────────────────────

    #[test]
    fn reptile_meta_update_moves_params_toward_task_adapted() {
        let mut params = vec![vec![0.0_f32, 0.0]];
        let task_params = vec![vec![vec![1.0_f32, 2.0]], vec![vec![3.0_f32, 4.0]]];
        let config = ReptileConfig {
            inner_lr: 0.1,
            meta_lr: 0.5,
            num_inner_steps: 5,
            meta_batch_size: 2,
        };
        reptile_meta_update(&mut params, &task_params, &config);
        // Expected: θ = θ + 0.5 * (mean − θ) = 0 + 0.5 * (2.0 - 0) = 1.0
        assert!(
            (params[0][0] - 1.0).abs() < 1e-5,
            "param[0] should be ~1.0, got {}",
            params[0][0]
        );
        assert!(
            (params[0][1] - 1.5).abs() < 1e-5,
            "param[1] should be ~1.5, got {}",
            params[0][1]
        );
    }

    #[test]
    fn reptile_meta_update_empty_task_params_no_change() {
        let mut params = vec![vec![5.0_f32]];
        let config = ReptileConfig::new(0.1, 0.5);
        reptile_meta_update(&mut params, &[], &config);
        assert_eq!(params[0][0], 5.0);
    }

    #[test]
    fn reptile_meta_update_lr_one_converges_to_mean() {
        let mut params = vec![vec![0.0_f32]];
        let task_params = vec![
            vec![vec![4.0_f32]],
            vec![vec![8.0_f32]],
            vec![vec![12.0_f32]],
        ];
        let config = ReptileConfig {
            inner_lr: 0.1,
            meta_lr: 1.0,
            num_inner_steps: 1,
            meta_batch_size: 3,
        };
        reptile_meta_update(&mut params, &task_params, &config);
        // With meta_lr=1.0: θ ← mean = (4+8+12)/3 = 8.0
        assert!((params[0][0] - 8.0).abs() < 1e-4);
    }

    // ── PrototypicalNetwork tests ─────────────────────────────────────────────

    #[test]
    fn proto_net_compute_prototypes_mean_of_support() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);
        // 2-way, 2-shot: class 0 → [1,0],[3,0]; class 1 → [0,1],[0,3]
        let support = vec![
            vec![1.0_f32, 0.0],
            vec![3.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 3.0],
        ];
        let labels = vec![0usize, 0, 1, 1];
        let protos = net.compute_prototypes(&support, &labels, 2);
        assert_eq!(protos.len(), 2);
        // Class 0 prototype ≈ [2.0, 0.0]
        assert!((protos[0][0] - 2.0).abs() < 1e-5);
        assert!((protos[0][1] - 0.0).abs() < 1e-5);
        // Class 1 prototype ≈ [0.0, 2.0]
        assert!((protos[1][0] - 0.0).abs() < 1e-5);
        assert!((protos[1][1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn proto_net_euclidean_distances_correct_shape() {
        let net = PrototypicalNetwork::new(3, DistanceMetric::Euclidean);
        let queries = vec![vec![1.0_f32, 0.0, 0.0]; 4];
        let protos = vec![vec![0.0_f32; 3]; 3];
        let dists = net.query_distances(&queries, &protos);
        assert_eq!(dists.len(), 4);
        assert_eq!(dists[0].len(), 3);
    }

    #[test]
    fn proto_net_cosine_distances_correct_shape() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Cosine);
        let queries = vec![vec![1.0_f32, 0.0]];
        let protos = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let dists = net.query_distances(&queries, &protos);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].len(), 2);
        // cosine_similarity([1,0], [1,0]) = 1.0 → score = 1.0 - 1.0 = 0.0
        assert!((dists[0][0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn proto_net_dot_product_distances() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::DotProduct);
        let queries = vec![vec![2.0_f32, 3.0]];
        let protos = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let dists = net.query_distances(&queries, &protos);
        // dot([2,3],[1,0]) = 2.0; dot([2,3],[0,1]) = 3.0
        assert!((dists[0][0] - 2.0).abs() < 1e-5);
        assert!((dists[0][1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn proto_net_predict_assigns_nearest_prototype() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);
        let protos = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let queries = vec![
            vec![0.1_f32, 0.1], // closer to proto 0
            vec![9.9, 9.9],     // closer to proto 1
        ];
        let preds = net.predict(&queries, &protos);
        assert_eq!(preds[0], 0);
        assert_eq!(preds[1], 1);
    }

    #[test]
    fn proto_net_accuracy_perfect() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);
        let preds = vec![0usize, 1, 0, 1];
        let targets = vec![0usize, 1, 0, 1];
        assert!((net.accuracy(&preds, &targets) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn proto_net_accuracy_half() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);
        let preds = vec![0usize, 0, 0, 0];
        let targets = vec![0usize, 1, 0, 1];
        assert!((net.accuracy(&preds, &targets) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn proto_net_accuracy_empty() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);
        assert_eq!(net.accuracy(&[], &[]), 0.0);
    }

    #[test]
    fn proto_net_loss_zero_when_perfectly_separated() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);
        let protos = vec![vec![-100.0_f32, 0.0], vec![100.0_f32, 0.0]];
        let queries = vec![vec![-100.0_f32, 0.0], vec![100.0_f32, 0.0]];
        let labels = vec![0usize, 1];
        let loss = net.loss(&queries, &protos, &labels);
        assert!(
            loss < 0.01,
            "loss should be near 0 for perfectly separated classes, got {loss}"
        );
    }

    #[test]
    fn proto_net_loss_non_negative() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);
        let protos = vec![vec![0.0_f32, 1.0], vec![1.0, 0.0]];
        let queries = vec![vec![0.5_f32, 0.5], vec![0.3, 0.7]];
        let labels = vec![0usize, 0];
        let loss = net.loss(&queries, &protos, &labels);
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
    }

    // ── EpisodeSampler tests ──────────────────────────────────────────────────

    fn make_dataset(num_classes: usize, per_class: usize) -> Vec<(Vec<f32>, usize)> {
        (0..num_classes)
            .flat_map(|c| {
                (0..per_class).map(move |i| {
                    let feat = vec![c as f32, i as f32];
                    (feat, c)
                })
            })
            .collect()
    }

    #[test]
    fn episode_sampler_basic() {
        let dataset = make_dataset(5, 10);
        let sampler = EpisodeSampler::new(3, 2, 4);
        let ep = sampler
            .sample_episode(&dataset, 5)
            .expect("sampling failed");
        assert_eq!(ep.n_way, 3);
        assert_eq!(ep.k_shot, 2);
        assert_eq!(ep.support_features.len(), 6); // 3 * 2
        assert_eq!(ep.query_features.len(), 12); // 3 * 4
        assert_eq!(ep.support_relabeled.len(), 6);
        assert_eq!(ep.query_labels.len(), 12);
    }

    #[test]
    fn episode_sampler_relabels_in_range() {
        let dataset = make_dataset(10, 8);
        let sampler = EpisodeSampler::new(5, 3, 2);
        let ep = sampler
            .sample_episode(&dataset, 10)
            .expect("sampling failed");
        for &lbl in &ep.support_relabeled {
            assert!(lbl < ep.n_way, "support relabel {lbl} out of range");
        }
        for &lbl in &ep.query_labels {
            assert!(lbl < ep.n_way, "query label {lbl} out of range");
        }
    }

    #[test]
    fn episode_sampler_error_invalid_n_way() {
        let dataset = make_dataset(3, 10);
        let sampler = EpisodeSampler::new(5, 2, 2);
        let result = sampler.sample_episode(&dataset, 3);
        assert!(
            matches!(result, Err(MetaLearningError::InvalidNWay { .. })),
            "expected InvalidNWay error"
        );
    }

    #[test]
    fn episode_sampler_error_insufficient_data() {
        let dataset = make_dataset(5, 2); // only 2 examples per class
        let sampler = EpisodeSampler::new(5, 2, 2); // needs 4 per class
        let result = sampler.sample_episode(&dataset, 5);
        assert!(
            matches!(
                result,
                Err(MetaLearningError::InsufficientData { .. })
                    | Err(MetaLearningError::InvalidNWay { .. })
            ),
            "expected InsufficientData or InvalidNWay error"
        );
    }

    // ── Matching Networks tests ───────────────────────────────────────────────

    #[test]
    fn attention_kernel_sums_to_one() {
        let query = vec![1.0_f32, 0.0];
        let support = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let weights = attention_kernel(&query, &support);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn attention_kernel_empty_support_returns_empty() {
        let query = vec![1.0_f32, 0.0];
        let weights = attention_kernel(&query, &[]);
        assert!(weights.is_empty());
    }

    #[test]
    fn matching_networks_predict_correct_class() {
        let support = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let support_labels = vec![0usize, 1];
        let query = vec![0.9_f32, 0.1];
        let pred = matching_networks_predict(&query, &support, &support_labels, 2);
        assert_eq!(pred, 0, "should predict class 0");
    }

    #[test]
    fn matching_networks_predict_empty_support_returns_zero() {
        let pred = matching_networks_predict(&[1.0_f32], &[], &[], 2);
        assert_eq!(pred, 0);
    }

    // ── Distance function tests ───────────────────────────────────────────────

    #[test]
    fn euclidean_distance_zero_for_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(euclidean_distance(&v, &v) < 1e-6);
    }

    #[test]
    fn euclidean_distance_known_value() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn dot_product_known_value() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-5);
    }

    // ── MetaLearningError display ─────────────────────────────────────────────

    #[test]
    fn error_display_insufficient_data() {
        let e = MetaLearningError::InsufficientData {
            needed: 10,
            found: 3,
        };
        let msg = e.to_string();
        assert!(msg.contains("10"), "message: {msg}");
        assert!(msg.contains("3"), "message: {msg}");
    }

    #[test]
    fn error_display_invalid_n_way() {
        let e = MetaLearningError::InvalidNWay {
            n_way: 5,
            available_classes: 2,
        };
        let msg = e.to_string();
        assert!(msg.contains("5"), "message: {msg}");
        assert!(msg.contains("2"), "message: {msg}");
    }

    #[test]
    fn error_display_dimension_mismatch() {
        let e = MetaLearningError::DimensionMismatch {
            expected: 64,
            found: 32,
        };
        let msg = e.to_string();
        assert!(msg.contains("64"), "message: {msg}");
        assert!(msg.contains("32"), "message: {msg}");
    }

    #[test]
    fn error_display_empty_embeddings() {
        let e = MetaLearningError::EmptyEmbeddings;
        let msg = e.to_string();
        assert!(!msg.is_empty());
    }
}
