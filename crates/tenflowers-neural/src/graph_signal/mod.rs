//! Graph Signal Processing & Spectral Methods.

pub mod dynamic;
pub mod pooling;
pub mod spectral;
pub mod topological;
mod utils;

pub use dynamic::{
    AdaptiveGraphConv, EdgeWeightLearner, GraphStructureLearning, IterativeGraphRefinement,
    NgramGraphBuilder,
};
pub use pooling::{
    AsapPooling, DiffPoolLayer, GraphCoarsening, HierarchicalPool, PoolLevel, SagPoolLayer,
};
pub use spectral::{
    BandpassGraphFilter, ChebyshevConv, DiffusionProcess, GraphFilter, GraphLaplacian,
    GraphWienerFilter, HarmonicAnalysis, SignalInterpolation, SpectralConv, WaveletTransformGraph,
};
pub use topological::{
    EdgeFlow, HodgeLaplacian, PersistencePair, PersistentHomologyLayer, SimplicialComplexData,
    SimplicialSignalDenoise, TdaFeatureExtractor,
};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
    use scirs2_core::RngExt;

    fn simple_adj() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ]
    }

    fn simple_x() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ]
    }

    fn triangle_sc() -> SimplicialComplexData {
        SimplicialComplexData::new(3, vec![(0, 1), (1, 2), (0, 2)], vec![(0, 1, 2)])
    }

    #[test]
    fn test_laplacian_symmetric() {
        let adj = simple_adj();
        let l = GraphLaplacian::compute(&adj);
        let n = l.len();
        for i in 0..n {
            for j in 0..n {
                let diff = (l[i][j] - l[j][i]).abs();
                assert!(diff < 1e-12, "L not symmetric at ({i},{j}): diff={diff}");
            }
        }
    }

    #[test]
    fn test_laplacian_diagonal_zero_colsum() {
        let adj = simple_adj();
        let l = GraphLaplacian::compute(&adj);
        for (i, row) in l.iter().enumerate() {
            assert!(
                row[i] >= 0.0 && row[i] <= 1.0 + 1e-10,
                "diagonal[{i}]={} not in [0,1]",
                row[i]
            );
        }
    }

    #[test]
    fn test_chebyshev_conv_shape() {
        let adj = simple_adj();
        let l = GraphLaplacian::compute(&adj);
        let l_tilde = GraphLaplacian::rescale(&l, 2.0);
        let conv = ChebyshevConv::new(2, 4, 3, 42);
        let x = simple_x();
        let out = conv.forward(&x, &l_tilde, 3);
        assert_eq!(out.len(), 4);
        for row in &out {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_spectral_conv_shape() {
        let adj = simple_adj();
        let x = simple_x();
        let conv = SpectralConv::new(2, 7);
        let out = conv.forward(&x, &adj, 2);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn test_harmonic_gft_reconstruction() {
        let mut rng = StdRng::seed_from_u64(99);
        let n = 4;
        let mut evecs: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.random::<f64>() - 0.5).collect())
            .collect();
        for i in 0..n {
            for j in 0..i {
                let proj: f64 = evecs[i].iter().zip(&evecs[j]).map(|(x, y)| x * y).sum();
                let ej = evecs[j].clone();
                for (x, e) in evecs[i].iter_mut().zip(&ej) {
                    *x -= proj * e;
                }
            }
            let norm: f64 = evecs[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            evecs[i].iter_mut().for_each(|x| *x /= norm.max(1e-12));
        }
        let signal: Vec<f64> = (0..n).map(|i| i as f64 * 0.3 + 0.1).collect();
        let coeffs = HarmonicAnalysis::gft(&signal, &evecs);
        let recon = HarmonicAnalysis::inverse_gft(&coeffs, &evecs);
        for (s, r) in signal.iter().zip(&recon) {
            assert!(
                (s - r).abs() < 1e-10,
                "GFT reconstruction failed: {s} vs {r}"
            );
        }
    }

    #[test]
    fn test_wavelet_transform_shape() {
        let adj = simple_adj();
        let scales = vec![0.5, 1.0, 2.0];
        let wavelets = WaveletTransformGraph::compute_wavelets(&adj, &scales);
        assert_eq!(wavelets.len(), 3);
        for w in &wavelets {
            assert_eq!(w.len(), 4);
            assert_eq!(w[0].len(), 4);
        }
    }

    #[test]
    fn test_graph_filter_output_shape() {
        let adj = simple_adj();
        let signal = vec![1.0, 0.5, 0.3, 0.1];
        let filter = GraphFilter::new(vec![1.0, 0.5, 0.25]);
        let out = filter.apply(&signal, &adj);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_diffusion_process_step() {
        let adj = simple_adj();
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let out = DiffusionProcess::step(&x, &adj, 0.01);
        assert_eq!(out.len(), 4);
        assert!(out[0] > 0.0);
    }

    #[test]
    fn test_diffusion_energy_decreasing() {
        let adj = simple_adj();
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let out = DiffusionProcess::diffuse(&x, &adj, 2.0, 100);
        let mean = out.iter().sum::<f64>() / out.len() as f64;
        let var_out: f64 = out.iter().map(|v| (v - mean) * (v - mean)).sum();
        let mean0 = x.iter().sum::<f64>() / x.len() as f64;
        let var_in: f64 = x.iter().map(|v| (v - mean0) * (v - mean0)).sum();
        assert!(
            var_out <= var_in + 1e-8,
            "variance did not decrease: {var_in} -> {var_out}"
        );
    }

    #[test]
    fn test_signal_interpolation() {
        let adj = simple_adj();
        let known = vec![Some(1.0), None, Some(0.0), None];
        let interp = SignalInterpolation::new(10.0, 50);
        let out = interp.interpolate(&known, &adj);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 0.5, "node 0: {}", out[0]);
        assert!((out[2] - 0.0).abs() < 0.5, "node 2: {}", out[2]);
    }

    #[test]
    fn test_bandpass_filter_shape() {
        let adj = simple_adj();
        let signal = vec![1.0, 0.5, 0.3, 0.1];
        let bpf = BandpassGraphFilter::new(0.1, 1.5, 6);
        let out = bpf.apply(&signal, &adj);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_wiener_filter() {
        let n = 4;
        let signal = vec![1.0, 0.5, 0.3, 0.1];
        let s_xx = vec![1.0; n];
        let s_nn = vec![0.1; n];
        let wf = GraphWienerFilter::new(s_xx, s_nn);
        let evecs: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0; n];
                v[i] = 1.0;
                v
            })
            .collect();
        let out = wf.apply(&signal, &evecs);
        assert_eq!(out.len(), n);
    }

    #[test]
    fn test_diffpool_output_shape() {
        let x = simple_x();
        let adj = simple_adj();
        let pool = DiffPoolLayer::new(2, 2, 3, 11);
        let s = vec![
            vec![0.8, 0.2],
            vec![0.1, 0.9],
            vec![0.6, 0.4],
            vec![0.3, 0.7],
        ];
        let (x_pool, a_pool) = pool.pool(&x, &adj, &s);
        assert_eq!(x_pool.len(), 2);
        assert_eq!(x_pool[0].len(), 3);
        assert_eq!(a_pool.len(), 2);
        assert_eq!(a_pool[0].len(), 2);
    }

    #[test]
    fn test_sagpool_keeps_ratio() {
        let x = vec![
            vec![1.0, 0.0, 0.5],
            vec![0.0, 1.0, 0.5],
            vec![0.5, 0.5, 1.0],
            vec![0.3, 0.7, 0.8],
            vec![0.9, 0.1, 0.2],
            vec![0.2, 0.8, 0.6],
        ];
        let n = x.len();
        let adj: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i != j && (i + j) % 2 == 0 { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let sag = SagPoolLayer::new(3, 13);
        let (out_x, sel) = sag.forward(&x, &adj, 0.5);
        let k = ((n as f64 * 0.5).ceil() as usize).max(1);
        assert_eq!(out_x.len(), k);
        assert_eq!(sel.len(), k);
        assert!(sel.iter().all(|&i| i < n));
    }

    #[test]
    fn test_graph_coarsening() {
        let adj = simple_adj();
        let x = simple_x();
        let (adj_c, x_c, mapping) = GraphCoarsening::coarsen(&adj, &x);
        assert!(!adj_c.is_empty());
        assert!(!x_c.is_empty());
        assert_eq!(mapping.len(), 4);
        let max_c = mapping.iter().copied().max().unwrap_or(0);
        assert_eq!(adj_c.len(), max_c + 1);
    }

    #[test]
    fn test_hierarchical_pool() {
        let x = simple_x();
        let adj = simple_adj();
        let hp = HierarchicalPool::new(2, 0.5);
        let (out_x, history) = hp.forward(&x, &adj, 42);
        assert!(!out_x.is_empty());
        assert!(!history.is_empty());
    }

    #[test]
    fn test_asap_pooling_shape() {
        let x = simple_x();
        let adj = simple_adj();
        let asap = AsapPooling::new(2, 7);
        let out = asap.forward(&x, &adj, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn test_edge_weight_learner() {
        let x = simple_x();
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let ewl = EdgeWeightLearner::new(2, 4, 99);
        let weights = ewl.forward(&x, &edges);
        assert_eq!(weights.len(), 3);
        for &w in &weights {
            assert!((0.0..=1.0).contains(&w), "weight {w} out of [0,1]");
        }
    }

    #[test]
    fn test_graph_structure_learning_k() {
        let x = simple_x();
        let gsl = GraphStructureLearning::new(2, 0.0);
        let adj = gsl.learn_adj(&x, 2);
        assert_eq!(adj.len(), 4);
        for i in 0..4 {
            for j in 0..4 {
                assert!((adj[i][j] - adj[j][i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_adaptive_graph_conv() {
        let x = simple_x();
        let adj = simple_adj();
        let gsl = GraphStructureLearning::new(2, 0.0);
        let adj_learned = gsl.learn_adj(&x, 2);
        let conv = AdaptiveGraphConv::new(2, 3, 17);
        let out = conv.forward(&x, &adj_learned, &adj, 0.5);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 3);
    }

    #[test]
    fn test_ngram_graph_builder() {
        let seqs = vec![vec![0usize, 1, 2, 3], vec![1, 2, 3, 0], vec![2, 3, 1, 0]];
        let adj = NgramGraphBuilder::build(&seqs, 2);
        assert_eq!(adj.len(), 4);
        assert_eq!(adj[0].len(), 4);
        for i in 0..4 {
            for j in 0..4 {
                assert!((adj[i][j] - adj[j][i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_iterative_refinement() {
        let x = simple_x();
        let adj = simple_adj();
        let igr = IterativeGraphRefinement::new(2, 2, 3, 55);
        let out = igr.forward(&x, &adj);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_hodge_laplacian_l0_symmetric() {
        let sc = triangle_sc();
        let l0 = HodgeLaplacian::l0(&sc);
        let n = l0.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (l0[i][j] - l0[j][i]).abs() < 1e-12,
                    "L0 not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_hodge_laplacian_l1_symmetric() {
        let sc = triangle_sc();
        let l1 = HodgeLaplacian::l1(&sc);
        if l1.is_empty() {
            return;
        }
        let n = l1.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (l1[i][j] - l1[j][i]).abs() < 1e-12,
                    "L1 not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_edge_flow_gradient() {
        let sc = triangle_sc();
        let ef = EdgeFlow::new(sc.clone());
        let vertex_signal = vec![1.0, 0.0, -1.0];
        let grad = ef.gradient(&vertex_signal);
        assert_eq!(grad.len(), sc.edges.len());
    }

    #[test]
    fn test_edge_flow_divergence() {
        let sc = triangle_sc();
        let ef = EdgeFlow::new(sc.clone());
        let edge_signal = vec![1.0, -1.0, 0.5];
        let div = ef.divergence(&edge_signal);
        assert_eq!(div.len(), 3);
    }

    #[test]
    fn test_edge_flow_curl() {
        let sc = triangle_sc();
        let ef = EdgeFlow::new(sc.clone());
        let edge_signal = vec![1.0, 1.0, 1.0];
        let curl = ef.curl(&edge_signal);
        assert_eq!(curl.len(), sc.triangles.len());
    }

    #[test]
    fn test_simplicial_denoise() {
        let sc = triangle_sc();
        let noisy = vec![0.5, -0.3, 0.8];
        let denoiser = SimplicialSignalDenoise::new(0.1, 20, 0.01);
        let out = denoiser.denoise(&noisy, &sc);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_tda_feature_extractor() {
        let adj = simple_adj();
        let tda = TdaFeatureExtractor::new(vec![0.1, 0.5, 1.0]);
        let features = tda.extract(&adj);
        assert_eq!(features.len(), 3);
        for f in &features {
            assert!(f[0] >= 1, "beta_0 should be >= 1");
        }
    }

    #[test]
    fn test_betti_numbers() {
        let n = 4;
        let adj: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i != j { 1.0 } else { 0.0 }).collect())
            .collect();
        let tda = TdaFeatureExtractor::new(vec![0.5]);
        let features = tda.extract(&adj);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0][0], 1, "K4 should be connected");
        assert_eq!(features[0][1], 3, "K4 beta_1 should be 3");
    }

    #[test]
    fn test_persistent_homology_diagram() {
        let adj = simple_adj();
        let phl = PersistentHomologyLayer::new(0.01);
        let diag = phl.compute_diagram(&adj);
        assert!(!diag.is_empty());
        for pair in &diag {
            assert!(
                pair.birth <= pair.death + 1e-12,
                "birth {} > death {}",
                pair.birth,
                pair.death
            );
        }
    }

    #[test]
    fn test_persistent_homology_loss() {
        let adj = simple_adj();
        let phl = PersistentHomologyLayer::new(0.1);
        let diag = phl.compute_diagram(&adj);
        let loss = phl.persistence_loss(&diag);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_spectral_conv_empty() {
        let conv = SpectralConv::new(2, 7);
        let out = conv.forward(&[], &[], 2);
        assert!(out.is_empty());
    }

    #[test]
    fn test_chebyshev_k1() {
        let adj = simple_adj();
        let lap = GraphLaplacian::compute(&adj);
        let l_tilde = GraphLaplacian::rescale(&lap, 2.0);
        let conv = ChebyshevConv::new(2, 2, 1, 1);
        let x = simple_x();
        let out = conv.forward(&x, &l_tilde, 1);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn test_diffpool_empty_s_fallback() {
        let x = simple_x();
        let adj = simple_adj();
        let pool = DiffPoolLayer::new(2, 2, 2, 1);
        let (x_pool, a_pool) = pool.pool(&x, &adj, &[]);
        assert_eq!(x_pool.len(), 2);
        assert_eq!(a_pool.len(), 2);
    }

    #[test]
    fn test_ngram_graph_trigrams() {
        let seqs = vec![vec![0usize, 1, 2, 3, 4]];
        let adj = NgramGraphBuilder::build(&seqs, 3);
        assert_eq!(adj.len(), 5);
        assert!(adj[0][1] > 0.0 || adj[1][0] > 0.0);
    }

    #[test]
    fn test_hierarchical_pool_unpool() {
        let x = simple_x();
        let adj = simple_adj();
        let hp = HierarchicalPool::new(1, 0.5);
        let (out_x, history) = hp.forward(&x, &adj, 42);
        if !history.is_empty() {
            let unpooled = HierarchicalPool::unpool(&out_x, &history[0]);
            assert_eq!(unpooled.len(), history[0].n_original);
        }
    }

    #[test]
    fn test_gft_identity_basis() {
        let n = 3;
        let evecs: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0; n];
                v[i] = 1.0;
                v
            })
            .collect();
        let signal = vec![3.0, 1.0, 4.0];
        let coeffs = HarmonicAnalysis::gft(&signal, &evecs);
        let recon = HarmonicAnalysis::inverse_gft(&coeffs, &evecs);
        for (s, r) in signal.iter().zip(&recon) {
            assert!((s - r).abs() < 1e-12);
        }
    }

    #[test]
    fn test_laplacian_3node_ring() {
        let adj = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let l = GraphLaplacian::compute(&adj);
        for i in 0..3 {
            assert!((l[i][i] - 1.0).abs() < 1e-12, "diag[{i}]={}", l[i][i]);
        }
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        (l[i][j] + 0.5).abs() < 1e-12,
                        "L[{i}][{j}]={} expected -0.5",
                        l[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_edge_weight_all_sigmoid_range() {
        let x = vec![
            vec![0.1, 0.9, 0.5],
            vec![0.8, 0.2, 0.3],
            vec![0.4, 0.6, 0.7],
        ];
        let edges: Vec<(usize, usize)> = (0..3)
            .flat_map(|i| (0..3).filter_map(move |j| if i != j { Some((i, j)) } else { None }))
            .collect();
        let ewl = EdgeWeightLearner::new(3, 5, 7);
        let ws = ewl.forward(&x, &edges);
        for &w in &ws {
            assert!(w > 0.0 && w < 1.0, "weight {w} not in (0,1)");
        }
    }
}
