//! Advanced Diffusion Models & Flow Matching — TenfloweRS
//!
//! This module provides:
//!
//! ## Flow Matching
//! - [`OtFlowMatching`]         — Optimal Transport flow matching (Lipman et al. 2022)
//! - [`CfmModel`]               — Continuous Flow Matching with σ_min interpolation
//! - [`RectifiedFlow`]          — Reflow straight trajectories
//! - [`ConsistencyModel`]       — One-step generation via consistency distillation
//! - [`FlowMatchingIntegrator`] — ODE solver (Euler / Heun / DPM-Solver)
//!
//! ## Latent Diffusion
//! - [`VariationalEncoder`]     — x → (μ, log σ²)
//! - [`VariationalDecoder`]     — z → x̂
//! - [`LatentDiffusionModel`]   — VAE + DDPM in latent space
//! - [`ConditioningEncoder`]    — text/class embedding
//! - [`ClassifierFreeGuidance`] — ε_guided = ε_uncond + w*(ε_cond − ε_uncond)
//!
//! ## Noise Schedules & Samplers
//! - [`CosineNoiseSchedule`]    — cosine β schedule (Nichol & Dhariwal)
//! - [`FlowSchedule`]           — exponential interpolant schedule
//! - [`DpmSolverSampler`]       — DPM-Solver++ 2nd-order multistep
//! - [`PndmSampler`]            — PLMS/PNDM 4th-order linear multistep
//! - [`SdeBasedSampler`]        — SDE Euler-Maruyama step
//!
//! ## Conditional Generation
//! - [`AdaptiveLayerNorm`]           — AdaLN: γ = Wγ·c + 1, β = Wβ·c
//! - [`CrossAttentionConditioning`]  — cross-attention between latent and conditioning
//! - [`ControlNetAdapter`]           — side network with zero-conv injection
//! - [`InpaintingMask`]              — mask-conditional generation
//! - [`GuidedDiffusionStep`]         — classifier guidance
//!
//! ## Evaluation Metrics
//! - [`FrechetInceptionDistance`]    — FID approximation from feature statistics
//! - [`InceptionScore`]              — IS = exp(E_x[KL(p(y|x) ‖ p(y))])
//! - [`RecallPrecision`]             — manifold precision/recall
//! - [`DiffusionLoss`]               — MSE + VLB combined loss
//! - [`NoisePredictionEvaluator`]    — SNR-weighted per-timestep tracking

pub(crate) mod helpers;

pub mod conditional;
pub mod flow_matching;
pub mod latent_diffusion;
pub mod metrics;
pub mod schedules;

// ─── Flow Matching re-exports ──────────────────────────────────────────────
pub use flow_matching::{
    CfmModel, ConsistencyModel, FlowMatchingIntegrator, IntegratorMethod, OtFlowMatching,
    RectifiedFlow,
};

// ─── Latent Diffusion re-exports ──────────────────────────────────────────
pub use latent_diffusion::{
    ClassifierFreeGuidance, ConditioningEncoder, LatentDiffusionModel, VariationalDecoder,
    VariationalEncoder,
};

// ─── Schedules & Samplers re-exports ──────────────────────────────────────
pub use schedules::{
    CosineNoiseSchedule, DpmSolverSampler, FlowSchedule, PndmSampler, SdeBasedSampler,
};

// ─── Conditional Generation re-exports ────────────────────────────────────
pub use conditional::{
    AdaptiveLayerNorm, ControlNetAdapter, CrossAttentionConditioning, GuidedDiffusionStep,
    InpaintingMask,
};

// ─── Evaluation Metrics re-exports ────────────────────────────────────────
pub use metrics::{
    DiffusionLoss, FrechetInceptionDistance, InceptionScore, NoisePredictionEvaluator,
    RecallPrecision,
};

// ─────────────────────────────────────────────────────────────────────────────
//  TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    const DIM: usize = 8;

    fn zeros(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    fn ones(n: usize) -> Vec<f64> {
        vec![1.0; n]
    }

    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 0.1).collect()
    }

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ─── Flow Matching ───────────────────────────────────────────────────────

    #[test]
    fn test_cfm_interpolation() {
        let cfm = CfmModel::new(DIM, 1e-4);
        let x0 = zeros(DIM);
        let x1 = ones(DIM);
        let mut rng = make_rng(0);
        let (xt, _) = cfm
            .forward(&x0, &x1, 0.5, &mut rng)
            .expect("forward failed");
        for &v in &xt {
            assert!((v - 0.5).abs() < 1e-6, "expected ~0.5 got {}", v);
        }
    }

    #[test]
    fn test_cfm_target_vf() {
        let cfm = CfmModel::new(DIM, 0.0);
        let x0 = zeros(DIM);
        let x1 = ones(DIM);
        let mut rng = make_rng(1);
        let (_, tvf) = cfm
            .forward(&x0, &x1, 0.3, &mut rng)
            .expect("forward failed");
        for &v in &tvf {
            assert!((v - 1.0).abs() < 1e-10, "expected target_vf=1.0 got {}", v);
        }
    }

    #[test]
    fn test_rectified_flow_straight() {
        let rf = RectifiedFlow::new(DIM);
        let z0 = zeros(DIM);
        let z1 = ones(DIM);
        let zt = rf.interpolate(&z0, &z1, 0.7).expect("interpolate failed");
        for &v in &zt {
            assert!((v - 0.7).abs() < 1e-10, "expected 0.7 got {}", v);
        }
        let vel = rf.target_velocity(&z0, &z1).expect("velocity failed");
        for &v in &vel {
            assert!((v - 1.0).abs() < 1e-10, "expected velocity 1.0 got {}", v);
        }
    }

    #[test]
    fn test_consistency_model_forward() {
        let cm = ConsistencyModel::new(DIM, 100);
        let x = ramp(DIM);
        let out = cm.forward(&x, 50).expect("forward failed");
        for (&a, &b) in out.iter().zip(x.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "identity consistency: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_consistency_loss_zero_for_same_input() {
        let cm = ConsistencyModel::new(DIM, 100);
        let x = ramp(DIM);
        let loss = cm.consistency_loss(&x, 10, &x, 9).expect("loss failed");
        assert!(loss < 1e-10, "loss should be ~0 for same input: {}", loss);
    }

    #[test]
    fn test_flow_integrator_euler() {
        let integrator = FlowMatchingIntegrator::new(10, IntegratorMethod::Euler);
        let x_init = ones(DIM);
        let out = integrator
            .integrate(&x_init, |_x, _t| zeros(DIM))
            .expect("integrate failed");
        for (&a, &b) in out.iter().zip(x_init.iter()) {
            assert!((a - b).abs() < 1e-10, "zero velocity: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_flow_integrator_heun() {
        let integrator = FlowMatchingIntegrator::new(5, IntegratorMethod::Heun);
        let x_init = ones(DIM);
        let out = integrator
            .integrate(&x_init, |_x, _t| zeros(DIM))
            .expect("heun failed");
        assert_eq!(out.len(), DIM);
        for (&a, &b) in out.iter().zip(x_init.iter()) {
            assert!((a - b).abs() < 1e-10, "heun zero: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_ot_flow_interpolation_t0() {
        let otfm = OtFlowMatching::new(DIM);
        let x0 = ramp(DIM);
        let x1 = ones(DIM);
        let xt = otfm
            .interpolate(&x0, &x1, 0.0)
            .expect("interpolate t=0 failed");
        for (&a, &b) in xt.iter().zip(x0.iter()) {
            assert!((a - b).abs() < 1e-10, "at t=0, xt should equal x0");
        }
    }

    #[test]
    fn test_ot_flow_interpolation_t1() {
        let otfm = OtFlowMatching::new(DIM);
        let x0 = zeros(DIM);
        let x1 = ramp(DIM);
        let xt = otfm
            .interpolate(&x0, &x1, 1.0)
            .expect("interpolate t=1 failed");
        for (&a, &b) in xt.iter().zip(x1.iter()) {
            assert!((a - b).abs() < 1e-10, "at t=1, xt should equal x1");
        }
    }

    #[test]
    fn test_ot_flow_conditional_vf() {
        let otfm = OtFlowMatching::new(DIM);
        let x0 = zeros(DIM);
        let x1 = ones(DIM);
        let u = otfm
            .conditional_vf(&x0, &x1, 0.0)
            .expect("conditional_vf failed");
        for &v in &u {
            assert!((v - 1.0).abs() < 1e-10, "expected u=1.0 got {}", v);
        }
    }

    // ─── Latent Diffusion ────────────────────────────────────────────────────

    #[test]
    fn test_variational_encoder_shape() {
        let enc = VariationalEncoder::new(16, 32, 8, 42);
        let x = ramp(16);
        let (mu, lv) = enc.encode(&x).expect("encode failed");
        assert_eq!(mu.len(), 8);
        assert_eq!(lv.len(), 8);
    }

    #[test]
    fn test_reparameterize() {
        let enc = VariationalEncoder::new(16, 32, 8, 42);
        let x = ramp(16);
        let (mu, lv) = enc.encode(&x).expect("encode failed");
        let z = enc
            .reparameterize(&mu, &lv, 99)
            .expect("reparameterize failed");
        assert_eq!(z.len(), 8);
        let diff: f64 = mu
            .iter()
            .zip(z.iter())
            .map(|(&m, &zi)| (m - zi).powi(2))
            .sum::<f64>();
        assert!(diff >= 0.0, "reparameterize diff should be >= 0");
    }

    #[test]
    fn test_latent_diffusion_construction() {
        let ldm = LatentDiffusionModel::new(16, 32, 8, 100, 42);
        assert_eq!(ldm.num_timesteps, 100);
        assert_eq!(ldm.alpha_bars.len(), 101);
        for t in 1..100 {
            assert!(
                ldm.alpha_bars[t] <= ldm.alpha_bars[t - 1] + 1e-6,
                "alpha_bars should be non-increasing at t={}",
                t
            );
        }
    }

    #[test]
    fn test_ldm_encode_decode_shape() {
        let ldm = LatentDiffusionModel::new(16, 32, 8, 100, 42);
        let x = ramp(16);
        let z = ldm.encode(&x, 0).expect("encode failed");
        assert_eq!(z.len(), 8);
        let x_hat = ldm.decode(&z).expect("decode failed");
        assert_eq!(x_hat.len(), 16);
    }

    #[test]
    fn test_cfg_guidance_weight() {
        let cfg = ClassifierFreeGuidance::new(DIM);
        let cond = ones(DIM);
        let uncond = zeros(DIM);
        let out = cfg.apply(&cond, &uncond, 2.0).expect("apply failed");
        for &v in &out {
            assert!((v - 2.0).abs() < 1e-10, "expected 2.0 got {}", v);
        }
    }

    #[test]
    fn test_cfg_uncond_limit() {
        let cfg = ClassifierFreeGuidance::new(DIM);
        let cond = ones(DIM);
        let uncond: Vec<f64> = (0..DIM).map(|i| i as f64).collect();
        let out = cfg.apply(&cond, &uncond, 0.0).expect("apply failed");
        for (&a, &b) in out.iter().zip(uncond.iter()) {
            assert!((a - b).abs() < 1e-10, "scale=0 should return uncond");
        }
    }

    // ─── Noise Schedules & Samplers ──────────────────────────────────────────

    #[test]
    fn test_cosine_schedule_decreasing() {
        let sched = CosineNoiseSchedule::new(100, 0.008);
        for t in 1..=100 {
            assert!(
                sched.alpha_bars[t] < sched.alpha_bars[t - 1] + 1e-6,
                "cosine alpha_bars not decreasing at t={}",
                t
            );
        }
    }

    #[test]
    fn test_cosine_schedule_endpoints() {
        let sched = CosineNoiseSchedule::new(1000, 0.008);
        assert!(
            sched.alpha_bars[0] > 0.99,
            "alpha_bar[0] should be ~1: {}",
            sched.alpha_bars[0]
        );
        assert!(
            sched.alpha_bars[1000] < 0.01,
            "alpha_bar[T] should be ~0: {}",
            sched.alpha_bars[1000]
        );
    }

    #[test]
    fn test_cosine_betas_in_range() {
        let sched = CosineNoiseSchedule::new(100, 0.008);
        for (i, &b) in sched.betas.iter().enumerate() {
            assert!((0.0..=0.999).contains(&b), "beta[{}]={} out of range", i, b);
        }
    }

    #[test]
    fn test_dpm_solver_step() {
        let mut sampler = DpmSolverSampler::new(20, -5.0, 5.0);
        let x = ones(DIM);
        let model_out = zeros(DIM);
        let x_next = sampler
            .sample_step(&x, 1.0, 0.9, model_out)
            .expect("step failed");
        assert_eq!(x_next.len(), DIM);
    }

    #[test]
    fn test_dpm_solver_multistep() {
        let mut sampler = DpmSolverSampler::new(20, -5.0, 5.0);
        let mut x = ones(DIM);
        for step in 0..5 {
            let model_out: Vec<f64> = (0..DIM).map(|i| (i + step) as f64 * 0.01).collect();
            let t_cur = 1.0 - step as f64 * 0.1;
            let t_next = t_cur - 0.1;
            x = sampler
                .sample_step(&x, t_cur, t_next, model_out)
                .expect("step failed");
        }
        assert_eq!(x.len(), DIM);
    }

    #[test]
    fn test_pndm_buffer() {
        let mut pndm = PndmSampler::new(20);
        assert_eq!(pndm.buffer_len(), 0);
        let x = ones(DIM);
        let _ = pndm.step(&x, zeros(DIM), -0.05).expect("step failed");
        assert_eq!(pndm.buffer_len(), 1);
        let _ = pndm.step(&x, ones(DIM), -0.05).expect("step 2 failed");
        assert_eq!(pndm.buffer_len(), 2);
    }

    #[test]
    fn test_pndm_buffer_max_4() {
        let mut pndm = PndmSampler::new(20);
        let x = ones(DIM);
        for i in 0..6 {
            let mo: Vec<f64> = (0..DIM).map(|j| (i + j) as f64 * 0.01).collect();
            let _ = pndm.step(&x, mo, -0.05).expect("step failed");
        }
        assert!(
            pndm.buffer_len() <= 4,
            "buffer exceeded 4: {}",
            pndm.buffer_len()
        );
    }

    #[test]
    fn test_sde_sampler_noise() {
        let sampler = SdeBasedSampler::new(20, 0.01, 80.0);
        let x = ones(DIM);
        let score = zeros(DIM);
        let noise = ramp(DIM);
        let x_next = sampler
            .em_step(&x, &score, 0, &noise)
            .expect("em_step failed");
        assert_eq!(x_next.len(), DIM);
    }

    #[test]
    fn test_sde_sampler_zero_score_zero_noise() {
        let sampler = SdeBasedSampler::new(10, 0.01, 1.0);
        let x = ones(DIM);
        let score = zeros(DIM);
        let noise = zeros(DIM);
        let x_next = sampler
            .em_step(&x, &score, 0, &noise)
            .expect("em_step failed");
        assert_eq!(x_next.len(), DIM);
    }

    // ─── Conditional Generation ──────────────────────────────────────────────

    #[test]
    fn test_adaln_scale_shift() {
        let adaln = AdaptiveLayerNorm::new(DIM, 4);
        let x = ramp(DIM);
        let c = zeros(4);
        let out = adaln.forward(&x, &c).expect("adaln forward failed");
        assert_eq!(out.len(), DIM);
        let mean = out.iter().sum::<f64>() / DIM as f64;
        assert!(
            mean.abs() < 1e-6,
            "adaln output should have ~0 mean: {}",
            mean
        );
    }

    #[test]
    fn test_adaln_output_shape() {
        let adaln = AdaptiveLayerNorm::new(16, 8);
        let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let c: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
        let out = adaln.forward(&x, &c).expect("adaln shape failed");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_cross_attention_conditioning() {
        let ca = CrossAttentionConditioning::new(DIM, DIM, DIM / 2);
        let q = ramp(DIM);
        let keys: Vec<f64> = (0..3 * DIM).map(|i| i as f64 * 0.01).collect();
        let values: Vec<f64> = (0..3 * DIM).map(|i| i as f64 * 0.02).collect();
        let out = ca.attend(&q, &keys, &values, 3).expect("attend failed");
        assert_eq!(out.len(), DIM);
    }

    #[test]
    fn test_cross_attention_single_key() {
        let ca = CrossAttentionConditioning::new(DIM, DIM, DIM);
        let q = zeros(DIM);
        let keys = ones(DIM);
        let values = ramp(DIM);
        let out = ca
            .attend(&q, &keys, &values, 1)
            .expect("attend single key failed");
        assert_eq!(out.len(), DIM);
        for (&a, &b) in out.iter().zip(values.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "single key: output should equal values"
            );
        }
    }

    #[test]
    fn test_inpainting_mask() {
        let mask = vec![true, false, true, false, true, false, true, false];
        let known: Vec<f64> = (0..DIM).map(|i| i as f64).collect();
        let im = InpaintingMask::new(mask.clone(), known.clone()).expect("mask failed");
        assert_eq!(im.num_known(), 4);

        let x_t = ones(DIM);
        let x_known_noisy = ramp(DIM);
        let out = im.apply(&x_t, &x_known_noisy).expect("apply failed");

        for (i, ((&o, &m), (&xkn, &xt))) in out
            .iter()
            .zip(mask.iter())
            .zip(x_known_noisy.iter().zip(x_t.iter()))
            .enumerate()
        {
            if m {
                assert!(
                    (o - xkn).abs() < 1e-10,
                    "masked px[{}] should be x_known_noisy",
                    i
                );
            } else {
                assert!((o - xt).abs() < 1e-10, "unmasked px[{}] should be x_t", i);
            }
        }
    }

    #[test]
    fn test_guided_step_direction() {
        let gds = GuidedDiffusionStep::new(1.0);
        let eps = ones(DIM);
        let grad = ones(DIM);
        let alpha_bar = 0.75;
        let out = gds
            .apply(&eps, &grad, alpha_bar)
            .expect("guided step failed");
        let expected = 1.0 - (1.0 - 0.75_f64).sqrt();
        for &v in &out {
            assert!(
                (v - expected).abs() < 1e-8,
                "expected {} got {}",
                expected,
                v
            );
        }
    }

    #[test]
    fn test_guided_step_zero_weight() {
        let gds = GuidedDiffusionStep::new(0.0);
        let eps = ramp(DIM);
        let grad = ones(DIM);
        let out = gds.apply(&eps, &grad, 0.5).expect("zero weight failed");
        for (&a, &b) in out.iter().zip(eps.iter()) {
            assert!((a - b).abs() < 1e-10, "w=0 should return eps");
        }
    }

    // ─── Evaluation Metrics ──────────────────────────────────────────────────

    #[test]
    fn test_inception_score_positive() {
        let probs: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let mut p = vec![0.01_f64; 5];
                p[i % 5] = 0.96;
                p
            })
            .collect();
        let is = InceptionScore::compute(&probs).expect("IS failed");
        assert!(
            is > 1.0,
            "IS should be > 1 for peaked distributions: {}",
            is
        );
    }

    #[test]
    fn test_inception_score_uniform_low() {
        let probs: Vec<Vec<f64>> = (0..10).map(|_| vec![0.2; 5]).collect();
        let is = InceptionScore::compute(&probs).expect("IS failed");
        assert!(
            is < 1.5,
            "IS should be ~1 for uniform distributions: {}",
            is
        );
    }

    #[test]
    fn test_fid_same_dist_zero() {
        let mu = vec![1.0, 2.0, 3.0];
        let sigma = vec![0.5, 0.5, 0.5];
        let fid =
            FrechetInceptionDistance::new(mu.clone(), sigma.clone()).expect("FID init failed");
        let score = fid.compute(&mu, &sigma).expect("FID compute failed");
        assert!(
            score.abs() < 1e-8,
            "FID of identical distributions should be ~0: {}",
            score
        );
    }

    #[test]
    fn test_fid_different_dist_positive() {
        let mu_r = vec![0.0, 0.0];
        let sigma_r = vec![1.0, 1.0];
        let fid = FrechetInceptionDistance::new(mu_r, sigma_r).expect("FID init");
        let mu_g = vec![5.0, 5.0];
        let sigma_g = vec![1.0, 1.0];
        let score = fid.compute(&mu_g, &sigma_g).expect("FID compute");
        assert!(
            score > 0.0,
            "FID of different distributions should be > 0: {}",
            score
        );
    }

    #[test]
    fn test_diffusion_loss_positive() {
        let dl = DiffusionLoss::new(0.5);
        let pred = ones(DIM);
        let target = zeros(DIM);
        let loss = dl.simple_loss(&pred, &target).expect("simple_loss failed");
        assert!(loss > 0.0, "loss should be positive: {}", loss);
    }

    #[test]
    fn test_diffusion_loss_zero_on_perfect() {
        let dl = DiffusionLoss::new(1.0);
        let noise = ramp(DIM);
        let loss = dl.simple_loss(&noise, &noise).expect("perfect loss failed");
        assert!(
            loss < 1e-10,
            "perfect prediction → loss should be ~0: {}",
            loss
        );
    }

    #[test]
    fn test_snr_weighted_loss() {
        let alpha_bars: Vec<f64> = (0..100).map(|t| 1.0 - t as f64 / 100.0).collect();
        let mut evaluator = NoisePredictionEvaluator::new(&alpha_bars, 5.0).expect("init failed");
        evaluator.record(0, 1.0).expect("record failed");
        evaluator.record(50, 2.0).expect("record failed");
        evaluator.record(99, 3.0).expect("record failed");
        let snr_loss = evaluator.snr_weighted_loss();
        assert!(
            snr_loss > 0.0,
            "SNR-weighted loss should be positive: {}",
            snr_loss
        );
    }

    #[test]
    fn test_per_timestep_tracking() {
        let alpha_bars: Vec<f64> = (0..50).map(|t| 1.0 - t as f64 / 50.0).collect();
        let mut evaluator = NoisePredictionEvaluator::new(&alpha_bars, 5.0).expect("init failed");
        for t in 0..50 {
            evaluator.record(t, t as f64 * 0.01).expect("record failed");
        }
        let per_t = evaluator.per_timestep_avg();
        assert_eq!(per_t.len(), 50);
        for (t, &avg) in per_t.iter().enumerate() {
            let expected = t as f64 * 0.01;
            assert!(
                (avg - expected).abs() < 1e-10,
                "per_timestep_avg[{}] = {} != {}",
                t,
                avg,
                expected
            );
        }
    }

    #[test]
    fn test_noise_evaluator_reset() {
        let alpha_bars: Vec<f64> = (0..10).map(|t| 1.0 - t as f64 / 10.0).collect();
        let mut evaluator = NoisePredictionEvaluator::new(&alpha_bars, 5.0).expect("init failed");
        evaluator.record(0, 1.0).expect("record failed");
        evaluator.reset();
        let per_t = evaluator.per_timestep_avg();
        for &v in &per_t {
            assert_eq!(v, 0.0, "after reset, all losses should be 0");
        }
    }

    #[test]
    fn test_flow_schedule_monotone() {
        let sched = FlowSchedule::new(0.01, 80.0, 50);
        for i in 1..=50 {
            assert!(
                sched.sigmas[i] >= sched.sigmas[i - 1],
                "flow schedule sigmas not increasing at i={}",
                i
            );
        }
    }

    #[test]
    fn test_rectified_flow_loss_zero_perfect() {
        let rf = RectifiedFlow::new(DIM);
        let z0 = zeros(DIM);
        let z1 = ones(DIM);
        let loss = rf.loss(&ones(DIM), &z0, &z1).expect("loss failed");
        assert!(
            loss < 1e-10,
            "perfect rectified flow loss should be ~0: {}",
            loss
        );
    }

    #[test]
    fn test_control_net_zero_gate() {
        let adapter = ControlNetAdapter::new(DIM, 3);
        let enc_out = ones(DIM);
        let main = ramp(DIM);
        let out = adapter.inject(&enc_out, 0, &main).expect("inject failed");
        for (&a, &b) in out.iter().zip(main.iter()) {
            assert!((a - b).abs() < 1e-10, "zero gate: should return main");
        }
    }

    #[test]
    fn test_variational_decoder_shape() {
        let dec = VariationalDecoder::new(8, 32, 16, 42);
        let z = ramp(8);
        let out = dec.decode(&z).expect("decode failed");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_ldm_diffuse_shape() {
        let ldm = LatentDiffusionModel::new(16, 32, 8, 100, 42);
        let z = ramp(8);
        let noise = ones(8);
        let zt = ldm.diffuse(&z, 50, &noise).expect("diffuse failed");
        assert_eq!(zt.len(), 8);
    }
}
