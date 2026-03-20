//! Tests for the robotics module (original + advanced algorithms).

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ═══════════════════ ORIGINAL TESTS ═════════════════════════════════════════

#[test]
fn test_demonstration_buffer_sample() {
    let mut buf = DemonstrationBuffer::new(100).expect("operation should succeed");
    for i in 0..50 {
        buf.add(DemoTransition::new(
            vec![i as f32, i as f32 + 1.0],
            vec![0.0],
            i as f32 * 0.1,
            false,
        ));
    }
    assert_eq!(buf.len(), 50);
    let mut rng = StdRng::seed_from_u64(42);
    assert_eq!(
        buf.sample_batch(10, &mut rng)
            .expect("operation should succeed")
            .len(),
        10
    );
}

#[test]
fn test_demonstration_buffer_overflow() {
    let mut buf = DemonstrationBuffer::new(5).expect("operation should succeed");
    for i in 0..10 {
        buf.add(DemoTransition::new(vec![i as f32], vec![0.0], 0.0, false));
    }
    assert_eq!(buf.len(), 5);
}

#[test]
fn test_behavioral_cloning_loss() {
    let mut bc =
        BehavioralCloning::new(4, 3, 16, 0.01, true, 42).expect("operation should succeed");
    let loss = bc
        .train_step(&[1.0_f32, 0.5, -0.3, 0.8], &[0.0_f32, 1.0, 0.0])
        .expect("computation failed");
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_behavioral_cloning_mse_loss() {
    let mut bc =
        BehavioralCloning::new(4, 2, 8, 0.001, false, 7).expect("operation should succeed");
    let loss = bc
        .train_step(&[0.1_f32, 0.2, 0.3, 0.4], &[1.0_f32, -1.0])
        .expect("computation failed");
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_dagger_beta_decay() {
    let dag = DaggerPolicy::new(1.0, 0.0, 100);
    assert!((dag.beta_schedule(0) - 1.0).abs() < 1e-9);
    assert!(dag.beta_schedule(50) < dag.beta_schedule(0));
    assert!(dag.beta_schedule(100) <= dag.beta_schedule(50));
}

#[test]
fn test_dagger_aggregate() {
    let mut dag = DaggerPolicy::new(1.0, 0.0, 10);
    dag.aggregate(
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![0.0], vec![1.0]],
    );
    assert_eq!(dag.dataset_size(), 2);
}

#[test]
fn test_gail_discriminator_loss() {
    let mut disc = GailDiscriminator::new(4, 16, 0.01, 42).expect("operation should succeed");
    let expert = vec![
        DemoTransition::new(vec![0.1, 0.2], vec![0.3, 0.4], 1.0, false),
        DemoTransition::new(vec![0.5, 0.6], vec![0.7, 0.8], 0.5, false),
    ];
    let policy = vec![DemoTransition::new(
        vec![-0.1, -0.2],
        vec![-0.3, -0.4],
        -0.5,
        false,
    )];
    let loss = disc.discriminator_loss(&expert, &policy);
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_gail_predict_range() {
    let disc = GailDiscriminator::new(4, 8, 0.001, 1).expect("operation should succeed");
    let s = disc.predict(&[1.0, 0.5], &[-0.3, 0.8]);
    assert!((0.0..=1.0).contains(&s));
}

#[test]
fn test_offline_rl_cql_loss() {
    let agent = OfflineRlAgent::new(4, 2, 16, 0.5, 42).expect("operation should succeed");
    let obs = vec![0.1_f32, 0.2, 0.3, 0.4];
    let pol = vec![vec![0.1_f32, 0.2], vec![0.3_f32, 0.4]];
    let rnd = vec![vec![0.9_f32, -0.9], vec![-0.5_f32, 0.5], vec![0.0_f32, 1.0]];
    let loss = agent
        .cql_loss(&obs, &pol, &rnd)
        .expect("operation should succeed");
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_rssm_deterministic() {
    let rssm = RecurrentStateSpaceModel::new(8, 4, 6, 2, 42).expect("operation should succeed");
    let h_next = rssm.deterministic_update(&[0.0_f32; 8], &[0.1_f32; 4], &[0.5_f32; 2]);
    assert_eq!(h_next.len(), 8);
    assert!(h_next.iter().all(|&v| v.abs() <= 1.0));
}

#[test]
fn test_rssm_posterior_shape() {
    let rssm = RecurrentStateSpaceModel::new(8, 4, 6, 2, 1).expect("operation should succeed");
    let (mu, lv) = rssm.posterior(&[0.0_f32; 8], &[0.1_f32; 6]);
    assert_eq!(mu.len(), 4);
    assert_eq!(lv.len(), 4);
}

#[test]
fn test_world_model_encoder_shape() {
    let enc = WorldModelEncoder::new(8, 16, 4, 99).expect("operation should succeed");
    let (mu, lv) = enc.encode(&[1.0_f32; 8]);
    assert_eq!(mu.len(), 4);
    assert_eq!(lv.len(), 4);
}

#[test]
fn test_world_model_decoder_shape() {
    let dec = WorldModelDecoder::new(8, 4, 6, 7).expect("operation should succeed");
    assert_eq!(dec.decode(&[0.0_f32; 8], &[0.5_f32; 4]).len(), 6);
}

#[test]
fn test_reward_predictor_scalar() {
    let rp = RewardPredictor::new(8, 4, 55).expect("operation should succeed");
    assert!(rp.predict(&[0.1_f32; 8], &[0.2_f32; 4]).is_finite());
}

#[test]
fn test_dreamer_kl_positive() {
    let trainer = DreamerTrainer::new(6, 8, 4, 2, 0.5, 1.0, 42).expect("operation should succeed");
    let (total, recon, kl, rew) = trainer
        .world_model_loss(&[0.5_f32; 6], &[0.0_f32; 8], &[0.1_f32; 2], 0.3, 100)
        .expect("computation failed");
    assert!(total.is_finite());
    assert!(recon >= 0.0);
    assert!(kl >= 0.0);
    assert!(rew >= 0.0);
}

#[test]
fn test_dh_forward_kinematics_zero() {
    let params = vec![DhParam::new(1.0, 0.0, 0.0, 0.0); 3];
    let pos =
        RobotKinematics::forward_kinematics(&params, &[0.0; 3]).expect("operation should succeed");
    assert!((pos[0] - 3.0).abs() < 1e-9);
    assert!(pos[1].abs() < 1e-9);
    assert!(pos[2].abs() < 1e-9);
}

#[test]
fn test_dh_forward_kinematics_shape() {
    let params = vec![DhParam::new(0.5, 0.1, 0.0, 0.0); 6];
    let pos = RobotKinematics::forward_kinematics(&params, &[std::f64::consts::PI / 4.0; 6])
        .expect("operation should succeed");
    assert_eq!(pos.len(), 3);
    assert!(pos.iter().all(|v| v.is_finite()));
}

#[test]
fn test_dh_mismatch_error() {
    assert!(RobotKinematics::forward_kinematics(
        &vec![DhParam::new(1.0, 0.0, 0.0, 0.0); 3],
        &[0.0; 2]
    )
    .is_err());
}

#[test]
fn test_motion_primitive_execute() {
    let mp = MotionPrimitive::new(1.0, 10).expect("operation should succeed");
    assert!(mp.execute(1.0, 1.0, 0.0, 0.5).is_finite());
}

#[test]
fn test_motion_primitive_phase_boundary() {
    let mp = MotionPrimitive::new(2.0, 5).expect("operation should succeed");
    assert!(mp.execute(1.0, 10.0, 0.0, 0.0).is_finite());
}

#[test]
fn test_grasp_quality_positive() {
    let normals = vec![[0., 0., 1.], [0., 0., -1.], [1., 0., 0.], [-1., 0., 0.]];
    assert!(GraspQuality::epsilon_quality(&normals) > 0.0);
}

#[test]
fn test_grasp_quality_insufficient_contacts() {
    assert_eq!(GraspQuality::epsilon_quality(&[[1.0_f64, 0.0, 0.0]]), 0.0);
}

#[test]
fn test_task_motion_planning() {
    let objs = vec![Symbol {
        name: "block_A".to_string(),
    }];
    let state = vec![Predicate {
        name: "ON".to_string(),
        args: vec!["block_A".to_string(), "table".to_string()],
    }];
    let mut tamp = TaskAndMotionPlanning::new(objs, state.clone());
    assert!(tamp.holds(&state[0]));
    tamp.block_predicate("STACK");
    assert!(!tamp.is_feasible(&Predicate {
        name: "STACK".to_string(),
        args: vec!["block_A".to_string()]
    }));
    let pick = Predicate {
        name: "HOLDING".to_string(),
        args: vec!["block_A".to_string()],
    };
    assert!(tamp.apply_action(vec![pick.clone()], vec![state[0].clone()]));
    assert!(tamp.holds(&pick));
}

#[test]
fn test_occupancy_grid_update() {
    let mut grid = OccupancyGrid::new(10, 10).expect("operation should succeed");
    assert!(grid.is_free(5, 5), "initially all cells are free");
    for _ in 0..10 {
        grid.update(5, 5, true);
    }
    assert!(!grid.is_free(5, 5));
}

#[test]
fn test_occupancy_grid_probability() {
    let grid = OccupancyGrid::new(4, 4).expect("operation should succeed");
    assert!((grid.probability(2, 2) - 0.5).abs() < 0.05);
}

#[test]
fn test_bresenham_ray() {
    let grid = OccupancyGrid::new(20, 20).expect("operation should succeed");
    let ray = grid.bresenham_ray((0, 0), (5, 5));
    assert!(!ray.is_empty());
    assert!(!ray.contains(&(5, 5)));
}

#[test]
fn test_astar_finds_path() {
    let mut grid = OccupancyGrid::new(5, 5).expect("operation should succeed");
    for x in 0..5 {
        for y in 0..5 {
            grid.update(x, y, false);
        }
    }
    let planner = AStarPlanner::new(false);
    let path = planner
        .plan(&grid, (0, 0), (4, 4))
        .expect("operation should succeed");
    assert_eq!(*path.first().expect("operation should succeed"), (0, 0));
    assert_eq!(*path.last().expect("operation should succeed"), (4, 4));
}

#[test]
fn test_astar_obstacle_blocked() {
    let mut grid = OccupancyGrid::new(5, 5).expect("operation should succeed");
    for x in 0..5 {
        for y in 0..5 {
            grid.update(x, y, false);
        }
    }
    for y in 0..5 {
        for _ in 0..20 {
            grid.update(2, y, true);
        }
    }
    assert!(AStarPlanner::new(false)
        .plan(&grid, (0, 0), (4, 4))
        .is_none());
}

#[test]
fn test_particle_filter_predict() {
    let mut pf =
        ParticleFilter::new(100, (0.0, 10.0), (0.0, 10.0), 42).expect("operation should succeed");
    let mut rng = StdRng::seed_from_u64(1);
    let n = pf.particles.len();
    pf.predict((0.5, 0.1, 0.05), &mut rng);
    assert_eq!(pf.particles.len(), n);
}

#[test]
fn test_particle_filter_resample() {
    let mut pf =
        ParticleFilter::new(50, (0.0, 5.0), (0.0, 5.0), 7).expect("operation should succeed");
    let mut rng = StdRng::seed_from_u64(2);
    pf.resample(&mut rng);
    assert_eq!(pf.particles.len(), 50);
    for p in &pf.particles {
        assert!((p.weight - 1.0 / 50.0).abs() < 1e-9);
    }
}

#[test]
fn test_potential_field_direction() {
    let nav = PotentialFieldNavigator::new(1.0, 100.0, 2.0);
    let f = nav.compute_force([0.0, 0.0], [5.0, 0.0], &[]);
    assert!(f[0] > 0.0);
    assert!(f[1].abs() < 1e-9);
}

#[test]
fn test_potential_field_repulsion() {
    let nav = PotentialFieldNavigator::new(0.1, 10.0, 3.0);
    let f = nav.compute_force([1.0, 0.0], [10.0, 0.0], &[[0.5, 0.0]]);
    assert!(f[0] > 0.0);
}

#[test]
fn test_neural_map_builder() {
    let builder = NeuralMapBuilder::new(4, 8, 42).expect("operation should succeed");
    assert_eq!(builder.embed_cell(&[1.0_f32; 4]).len(), 8);
}

#[test]
fn test_domain_randomizer_range() {
    let dr = extensions::DomainRandomizer::new((0.5, 1.5), (0.1, 0.9), (0.01, 0.1), (0, 3));
    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..50 {
        let p = dr.sample_params(&mut rng);
        assert!(p.mass >= 0.5 && p.mass <= 1.5);
        assert!(p.friction >= 0.1 && p.friction <= 0.9);
        assert!(p.damping >= 0.01 && p.damping <= 0.1);
        assert!(p.action_delay <= 3);
    }
}

#[test]
fn test_physics_params() {
    let p = extensions::PhysicsParams {
        mass: 1.0,
        friction: 0.5,
        damping: 0.1,
        action_delay: 2,
    };
    assert_eq!(p.action_delay, 2);
    assert!((p.mass - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_adaptation_module() {
    let am = extensions::AdaptationModule::new(6, 4, 42).expect("operation should succeed");
    let mut rng = StdRng::seed_from_u64(1);
    let ctx = am
        .adapt(
            &[
                vec![0.1_f32, 0.2, 0.3, -0.1, 0.5, 1.0],
                vec![-0.1_f32, 0.5, 0.2, 0.3, -0.3, 0.0],
            ],
            &mut rng,
        )
        .expect("computation failed");
    assert_eq!(ctx.len(), 4);
    assert!(ctx.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mmd_loss_same_zero() {
    let mmd = extensions::DomainAdaptationLoss::new(1.0);
    let f = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0], vec![5.0_f32, 6.0]];
    assert!(mmd.mmd_loss(&f, &f) < 1e-6);
}

#[test]
fn test_mmd_loss_different() {
    let mmd = extensions::DomainAdaptationLoss::new(1.0);
    let sim = vec![vec![0.0_f32, 0.0], vec![0.1_f32, 0.1], vec![0.2_f32, 0.2]];
    let real = vec![
        vec![10.0_f32, 10.0],
        vec![10.1_f32, 10.1],
        vec![10.2_f32, 10.2],
    ];
    assert!(mmd.mmd_loss(&sim, &real) > 0.0);
}

#[test]
fn test_sim_to_real_evaluator() {
    let mut eval = extensions::SimToRealEvaluator::new();
    eval.record_sim_reward(10.0);
    eval.record_sim_reward(12.0);
    eval.record_real_reward(8.0);
    eval.record_real_reward(5.0);
    eval.record_mmd(0.05);
    assert!((eval.transfer_success_rate(0.6) - 0.5).abs() < 1e-9);
    assert!((eval.mean_domain_shift() - 0.05).abs() < 1e-9);
}

#[test]
fn test_demo_buffer_empty_error() {
    let buf = DemonstrationBuffer::new(10).expect("operation should succeed");
    let mut rng = StdRng::seed_from_u64(0);
    assert!(buf.sample_batch(1, &mut rng).is_err());
}

#[test]
fn test_behavioral_cloning_dim_error() {
    let mut bc = BehavioralCloning::new(4, 3, 8, 0.01, true, 1).expect("operation should succeed");
    assert!(bc.train_step(&[0.0_f32; 3], &[0.0_f32, 1.0, 0.0]).is_err());
}

#[test]
fn test_occupancy_grid_oob() {
    let mut grid = OccupancyGrid::new(5, 5).expect("operation should succeed");
    grid.update(10, 10, true); // should not panic
    assert!(grid.probability(10, 10) > 0.0);
}

#[test]
fn test_rssm_sample_latent() {
    let rssm = RecurrentStateSpaceModel::new(8, 4, 6, 2, 10).expect("operation should succeed");
    let z = rssm.sample_latent(&[0.0_f32; 4], &[0.0_f32; 4], 42);
    assert_eq!(z.len(), 4);
    assert!(z.iter().all(|v| v.is_finite()));
}

#[test]
fn test_grasp_force_closure() {
    let normals = vec![[0., 0., 1.], [0., 0., -1.], [1., 0., 0.], [-1., 0., 0.]];
    assert!(GraspQuality::is_force_closed(&normals));
}

// ═══════════════════ ADVANCED ALGORITHM TESTS ═══════════════════════════════

// --- AABB tests ---

#[test]
fn test_aabb_contains() {
    let aabb = Aabb::new(vec![0.0, 0.0], vec![1.0, 1.0]).expect("valid aabb");
    assert!(aabb.contains(&[0.5, 0.5]));
    assert!(!aabb.contains(&[1.5, 0.5]));
    assert!(!aabb.contains(&[-0.1, 0.5]));
}

#[test]
fn test_aabb_segment_intersection() {
    let aabb = Aabb::new(vec![0.0, 0.0], vec![1.0, 1.0]).expect("valid aabb");
    // Segment through the box
    assert!(aabb.intersects_segment(&[-1.0, 0.5], &[2.0, 0.5]));
    // Segment clearly outside
    assert!(!aabb.intersects_segment(&[2.0, 0.0], &[3.0, 1.0]));
}

#[test]
fn test_aabb_invalid() {
    assert!(Aabb::new(vec![], vec![]).is_err());
    assert!(Aabb::new(vec![0.0], vec![1.0, 2.0]).is_err());
}

// --- RRT* tests ---

#[test]
fn test_rrt_star_plan_open_space() {
    let planner = RrtStarPlanner::new(2, vec![0.0, 0.0], vec![10.0, 10.0], 0.5, 1.5, 500, vec![])
        .expect("valid planner");
    let result = planner
        .plan(&[0.0, 0.0], &[9.0, 9.0], 42)
        .expect("no error");
    // In open space the planner should find a path
    assert!(result.is_some());
    let path = result.expect("path exists");
    assert!(!path.is_empty());
    // Path should start near origin
    let start = &path[0];
    assert!(start[0].abs() < 0.01 && start[1].abs() < 0.01);
}

#[test]
fn test_rrt_star_dim_error() {
    let planner = RrtStarPlanner::new(2, vec![0.0, 0.0], vec![5.0, 5.0], 0.5, 1.5, 100, vec![])
        .expect("valid");
    assert!(planner.plan(&[0.0], &[4.0, 4.0], 1).is_err());
}

#[test]
fn test_rrt_star_invalid_dim() {
    assert!(RrtStarPlanner::new(0, vec![], vec![], 0.5, 1.0, 100, vec![]).is_err());
}

#[test]
fn test_rrt_star_with_obstacle() {
    // Large blocking obstacle in the middle — planner may or may not find path
    let obs = Aabb::new(vec![3.0, 0.0], vec![7.0, 10.0]).expect("valid aabb");
    let planner = RrtStarPlanner::new(
        2,
        vec![0.0, 0.0],
        vec![10.0, 10.0],
        0.3,
        1.0,
        200,
        vec![obs],
    )
    .expect("valid planner");
    // Just ensure no panic
    let _ = planner.plan(&[1.0, 5.0], &[9.0, 5.0], 7);
}

// --- NeuralRRT tests ---

#[test]
fn test_neural_rrt_suggest_sample() {
    let rrt = RrtStarPlanner::new(2, vec![0.0, 0.0], vec![5.0, 5.0], 0.5, 1.5, 200, vec![])
        .expect("valid");
    let nrrt = NeuralRrt::new(rrt, 0.3, 16, 42).expect("valid neural rrt");
    let sample = nrrt.suggest_sample(&[0.0, 0.0], &[4.0, 4.0]);
    assert_eq!(sample.len(), 2);
    assert!(sample[0] >= 0.0 && sample[0] <= 5.0);
    assert!(sample[1] >= 0.0 && sample[1] <= 5.0);
}

#[test]
fn test_neural_rrt_plan() {
    let rrt = RrtStarPlanner::new(2, vec![0.0, 0.0], vec![5.0, 5.0], 0.5, 1.5, 400, vec![])
        .expect("valid");
    let nrrt = NeuralRrt::new(rrt, 0.5, 8, 99).expect("valid neural rrt");
    let result = nrrt.plan(&[0.0, 0.0], &[4.0, 4.0], 10);
    assert!(result.is_ok());
}

#[test]
fn test_neural_rrt_invalid() {
    let rrt = RrtStarPlanner::new(2, vec![0.0, 0.0], vec![5.0, 5.0], 0.5, 1.5, 100, vec![])
        .expect("valid");
    assert!(NeuralRrt::new(rrt, 0.3, 0, 1).is_err());
}

// --- PRM tests ---

#[test]
fn test_prm_build_and_query() {
    let prm =
        Prm::build(2, vec![0.0, 0.0], vec![5.0, 5.0], 1.5, vec![], 50, 42).expect("valid prm");
    assert_eq!(prm.nodes.len(), 50);
    assert_eq!(prm.dim, 2);
}

#[test]
fn test_prm_invalid_params() {
    assert!(Prm::build(0, vec![], vec![], 1.0, vec![], 10, 1).is_err());
    assert!(Prm::build(2, vec![0.0, 0.0], vec![5.0, 5.0], 1.0, vec![], 0, 1).is_err());
}

#[test]
fn test_prm_edges_symmetric() {
    let prm = Prm::build(2, vec![0.0, 0.0], vec![3.0, 3.0], 2.0, vec![], 20, 7).expect("valid");
    // Every edge (i,j) should have a corresponding (j,i)
    for (i, neighbours) in prm.edges.iter().enumerate() {
        for &(j, _) in neighbours {
            let reverse = prm.edges[j].iter().any(|&(k, _)| k == i);
            assert!(reverse, "edge {i}->{j} has no reverse");
        }
    }
}

// --- ContactModel tests ---

#[test]
fn test_contact_model_friction_cone() {
    let cm = ContactModel::new(0.5, 4).expect("valid");
    let rays = cm.friction_cone_rays();
    assert_eq!(rays.len(), 4);
    for r in &rays {
        assert!(r[2].abs() < 1.01); // z-component = 1
    }
}

#[test]
fn test_contact_model_in_cone() {
    let cm = ContactModel::new(0.5, 8).expect("valid");
    // Pure normal force is always in cone
    assert!(cm.is_in_cone(&[0.0, 0.0, 1.0]));
    // Tangential force larger than mu * fn is outside
    assert!(!cm.is_in_cone(&[1.0, 0.0, 0.1]));
}

#[test]
fn test_contact_model_invalid() {
    assert!(ContactModel::new(0.0, 4).is_err());
    assert!(ContactModel::new(0.5, 2).is_err());
}

// --- GraspQualityMetric tests ---

#[test]
fn test_grasp_quality_metric_construction() {
    let contacts = vec![
        ([0.05, 0.0, 0.0], [1.0, 0.0, 0.0_f64]),
        ([-0.05, 0.0, 0.0], [-1.0, 0.0, 0.0_f64]),
        ([0.0, 0.05, 0.0], [0.0, 1.0, 0.0_f64]),
    ];
    let metric = GraspQualityMetric::from_contacts(&contacts, 0.5).expect("valid contacts");
    assert!(!metric.primitives.is_empty());
    let q = metric.epsilon_quality();
    assert!(q >= 0.0 && q.is_finite());
}

#[test]
fn test_grasp_quality_metric_empty_error() {
    assert!(GraspQualityMetric::from_contacts(&[], 0.5).is_err());
}

// --- DexterousGraspPlanner tests ---

#[test]
fn test_dexterous_grasp_planner_basic() {
    let planner = DexterousGraspPlanner::new(3, 0.5, 10).expect("valid planner");
    let candidates = vec![
        ([0.05, 0.0, 0.0], [1.0, 0.0, 0.0_f64]),
        ([-0.05, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.05, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, -0.05, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, 0.05], [0.0, 0.0, 1.0]),
    ];
    let (indices, quality) = planner.plan(&candidates, 42).expect("planning succeeded");
    assert_eq!(indices.len(), 3);
    assert!(quality >= 0.0);
}

#[test]
fn test_dexterous_grasp_planner_too_few_contacts() {
    let planner = DexterousGraspPlanner::new(3, 0.5, 5).expect("valid");
    let contacts = vec![
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0_f64]),
        ([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
    ];
    assert!(planner.plan(&contacts, 1).is_err());
}

#[test]
fn test_dexterous_grasp_planner_invalid() {
    assert!(DexterousGraspPlanner::new(1, 0.5, 10).is_err());
}

// --- TactileSensorModel tests ---

#[test]
fn test_tactile_sensor_no_contact() {
    let sensor = TactileSensorModel::new(4, 4, 0.01, 1.0).expect("valid sensor");
    let readings = sensor.simulate(&[]);
    assert_eq!(readings.len(), 16);
    assert!(readings.iter().all(|&v| v == 0.0));
}

#[test]
fn test_tactile_sensor_center_contact() {
    let sensor = TactileSensorModel::new(5, 5, 0.01, 0.5).expect("valid sensor");
    // Place contact at center (2,2) with unit force
    let readings = sensor.simulate(&[([0.02, 0.02], 1.0)]);
    assert_eq!(readings.len(), 25);
    let peak = sensor.peak_taxel(&readings);
    assert!(peak.is_some());
    let total = sensor.total_load(&readings);
    assert!(total > 0.0 && total.is_finite());
}

#[test]
fn test_tactile_sensor_multiple_contacts() {
    let sensor = TactileSensorModel::new(8, 8, 0.01, 1.0).expect("valid sensor");
    let contacts = vec![([0.01, 0.01], 2.0), ([0.07, 0.07], 1.0)];
    let readings = sensor.simulate(&contacts);
    assert_eq!(readings.len(), 64);
    assert!(sensor.total_load(&readings) > 0.0);
}

#[test]
fn test_tactile_sensor_invalid() {
    assert!(TactileSensorModel::new(0, 4, 0.01, 1.0).is_err());
}

// --- WbcTask tests ---

#[test]
fn test_wbc_task_pseudoinverse() {
    // Simple 2-DOF, 2-task system
    // J = I, desired = [1, 0]
    let j = vec![1.0, 0.0, 0.0, 1.0_f64]; // 2×2 identity
    let task = WbcTask::new("test", j, vec![1.0, 0.0], 2, 2, 1.0).expect("valid task");
    let sol = task.pseudoinverse_solution();
    assert_eq!(sol.len(), 2);
    assert!(sol.iter().all(|v| v.is_finite()));
}

#[test]
fn test_wbc_task_null_space() {
    let j = vec![1.0, 0.0, 0.0, 1.0_f64];
    let task = WbcTask::new("ns_test", j, vec![0.0, 0.0], 2, 2, 1.0).expect("valid task");
    let ns = task.null_space_projector();
    assert_eq!(ns.len(), 4);
    // For full-rank square J, null space should be ≈ zero matrix
    for v in &ns {
        assert!(
            v.abs() < 0.1,
            "null space should be near zero for full-rank J"
        );
    }
}

#[test]
fn test_wbc_task_invalid() {
    assert!(WbcTask::new("bad", vec![1.0, 0.0], vec![1.0, 0.0], 2, 3, 1.0).is_err());
    assert!(WbcTask::new("bad2", vec![1.0, 0.0, 0.0, 1.0], vec![1.0], 2, 2, 1.0).is_err());
}

// --- HierarchicalQP tests ---

#[test]
fn test_hierarchical_qp_solve() {
    let j1 = vec![1.0, 0.0, 0.0, 1.0_f64]; // identity 2×2
    let j2 = vec![0.0, 1.0, 1.0, 0.0_f64]; // swap 2×2
    let t1 = WbcTask::new("t1", j1, vec![1.0, 0.0], 2, 2, 2.0).expect("valid");
    let t2 = WbcTask::new("t2", j2, vec![0.0, 1.0], 2, 2, 1.0).expect("valid");
    let hqp = HierarchicalQp::new(vec![t1, t2]).expect("valid hqp");
    let cmd = hqp.solve();
    assert_eq!(cmd.len(), 2);
    assert!(cmd.iter().all(|v| v.is_finite()));
}

#[test]
fn test_hierarchical_qp_single_task() {
    let j = vec![1.0, 0.0, 0.0, 1.0_f64];
    let t = WbcTask::new("only", j, vec![2.0, -1.0], 2, 2, 1.0).expect("valid");
    let hqp = HierarchicalQp::new(vec![t]).expect("valid");
    let cmd = hqp.solve();
    assert_eq!(cmd.len(), 2);
}

#[test]
fn test_hierarchical_qp_empty_error() {
    assert!(HierarchicalQp::new(vec![]).is_err());
}

// --- CentroidalDynamics tests ---

#[test]
fn test_centroidal_dynamics_gravity() {
    let mut cd = CentroidalDynamics::new(10.0, [0.0, 0.0, -9.81]).expect("valid");
    // With no contact forces, gravity should change linear momentum
    cd.update(&[], &[0.0, 0.0, 0.0], 0.01);
    let lm = cd.linear_momentum();
    assert!(lm[2] < 0.0); // downward impulse from gravity
}

#[test]
fn test_centroidal_dynamics_contact() {
    let mut cd = CentroidalDynamics::new(1.0, [0.0, 0.0, -9.81]).expect("valid");
    // Contact exactly balances gravity
    cd.update(
        &[([0.0, 0.0, 0.0], [0.0, 0.0, 9.81])],
        &[0.0, 0.0, 0.0],
        0.1,
    );
    let vel = cd.com_velocity();
    assert!(vel[2].abs() < 0.01); // approximately stationary
}

#[test]
fn test_centroidal_dynamics_angular_momentum() {
    let cd = CentroidalDynamics::new(5.0, [0.0, 0.0, -9.81]).expect("valid");
    let am = cd.angular_momentum();
    assert_eq!(am, [0.0, 0.0, 0.0]);
}

#[test]
fn test_centroidal_dynamics_invalid_mass() {
    assert!(CentroidalDynamics::new(-1.0, [0.0, 0.0, -9.81]).is_err());
    assert!(CentroidalDynamics::new(0.0, [0.0, 0.0, -9.81]).is_err());
}

// --- AdvancedDomainRandomizer tests ---

#[test]
fn test_advanced_domain_randomizer() {
    let physics = extensions::DomainRandomizer::new((1.0, 2.0), (0.1, 0.5), (0.01, 0.1), (0, 2));
    let adr = AdvancedDomainRandomizer::new(physics, (0.5, 2.0), (0.1, 1.0), (0.0, 0.05));
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let (tex, light, noise) = adr.sample_visual(&mut rng);
        assert!((0.5..=2.0).contains(&tex));
        assert!((0.1..=1.0).contains(&light));
        assert!((0.0..=0.05).contains(&noise));
    }
}

// --- SimToRealAdapter tests ---

#[test]
fn test_sim_to_real_adapter_features() {
    let adapter = SimToRealAdapter::new(4, 8, 16, 42).expect("valid adapter");
    let obs = vec![0.1_f32, 0.2, -0.3, 0.4];
    let feats = adapter.extract_features(&obs);
    assert_eq!(feats.len(), 8);
    assert!(feats.iter().all(|v| v.is_finite()));
}

#[test]
fn test_sim_to_real_adapter_domain_logit() {
    let adapter = SimToRealAdapter::new(4, 8, 16, 7).expect("valid adapter");
    let feats = vec![0.5_f32; 8];
    let logit = adapter.domain_logit(&feats);
    assert!(logit.is_finite());
}

#[test]
fn test_sim_to_real_adapter_loss() {
    let adapter = SimToRealAdapter::new(4, 6, 12, 99).expect("valid adapter");
    let sim_obs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1; 4]).collect();
    let real_obs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.5 + 2.0; 4]).collect();
    let loss = adapter.adaptation_loss(&sim_obs, &real_obs);
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_sim_to_real_adapter_invalid() {
    assert!(SimToRealAdapter::new(0, 8, 16, 1).is_err());
}

// --- RoboticsPolicyMetrics tests ---

#[test]
fn test_robotics_policy_metrics_success_rate() {
    let mut m = RoboticsPolicyMetrics::new();
    m.record_episode(true, 10.0, 8.0, false);
    m.record_episode(false, 12.0, 8.0, true);
    m.record_episode(true, 9.0, 8.0, false);
    assert!((m.success_rate() - 2.0 / 3.0).abs() < 1e-9);
}

#[test]
fn test_robotics_policy_metrics_path_efficiency() {
    let mut m = RoboticsPolicyMetrics::new();
    m.record_episode(true, 10.0, 10.0, false); // perfect efficiency
    m.record_episode(true, 20.0, 10.0, false); // 50% efficiency
    let eff = m.path_length_efficiency();
    assert!((eff - 0.75).abs() < 1e-9);
}

#[test]
fn test_robotics_policy_metrics_force_violation() {
    let mut m = RoboticsPolicyMetrics::new();
    m.record_episode(true, 5.0, 5.0, false);
    m.record_episode(true, 5.0, 5.0, true);
    m.record_episode(true, 5.0, 5.0, true);
    assert!((m.force_violation_rate() - 2.0 / 3.0).abs() < 1e-9);
}

#[test]
fn test_robotics_policy_metrics_sim2real_gap() {
    let mut m = RoboticsPolicyMetrics::new();
    m.record_rewards(100.0, 80.0);
    m.record_rewards(100.0, 80.0);
    let gap = m.sim2real_gap();
    assert!((gap - 0.2).abs() < 1e-6);
}

#[test]
fn test_robotics_policy_metrics_empty() {
    let m = RoboticsPolicyMetrics::new();
    assert_eq!(m.success_rate(), 0.0);
    assert_eq!(m.path_length_efficiency(), 0.0);
    assert_eq!(m.force_violation_rate(), 0.0);
    assert_eq!(m.sim2real_gap(), 0.0);
    assert_eq!(m.n_episodes(), 0);
}

#[test]
fn test_robotics_policy_metrics_summary() {
    let mut m = RoboticsPolicyMetrics::new();
    m.record_episode(true, 5.0, 5.0, false);
    m.record_rewards(10.0, 9.0);
    let summary = m.summary();
    assert!(summary.contains_key("success_rate"));
    assert!(summary.contains_key("path_length_efficiency"));
    assert!(summary.contains_key("force_violation_rate"));
    assert!(summary.contains_key("sim2real_gap"));
    assert!(summary.contains_key("n_episodes"));
}
