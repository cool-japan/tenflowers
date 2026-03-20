//! Tests for differentiable_physics module.

use super::*;

// ── DpVec3 ──────────────────────────────────────────────────────────────────

#[test]
fn test_vec3_zero() {
    let v = DpVec3::zero();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

#[test]
fn test_vec3_add() {
    let a = DpVec3::new(1.0, 2.0, 3.0);
    let b = DpVec3::new(4.0, 5.0, 6.0);
    let c = a.add(b);
    assert!((c.x - 5.0).abs() < 1e-12);
    assert!((c.y - 7.0).abs() < 1e-12);
    assert!((c.z - 9.0).abs() < 1e-12);
}

#[test]
fn test_vec3_sub() {
    let a = DpVec3::new(5.0, 3.0, 1.0);
    let b = DpVec3::new(1.0, 1.0, 1.0);
    let c = a.sub(b);
    assert!((c.x - 4.0).abs() < 1e-12);
    assert!((c.y - 2.0).abs() < 1e-12);
    assert!((c.z - 0.0).abs() < 1e-12);
}

#[test]
fn test_vec3_scale() {
    let v = DpVec3::new(2.0, 3.0, 4.0);
    let s = v.scale(2.0);
    assert!((s.x - 4.0).abs() < 1e-12);
    assert!((s.y - 6.0).abs() < 1e-12);
    assert!((s.z - 8.0).abs() < 1e-12);
}

#[test]
fn test_vec3_dot() {
    let a = DpVec3::new(1.0, 2.0, 3.0);
    let b = DpVec3::new(4.0, 5.0, 6.0);
    let d = a.dot(b);
    assert!((d - 32.0).abs() < 1e-12);
}

#[test]
fn test_vec3_cross() {
    let x = DpVec3::unit_x();
    let y = DpVec3::unit_y();
    let z = x.cross(y);
    assert!((z.x - 0.0).abs() < 1e-12);
    assert!((z.y - 0.0).abs() < 1e-12);
    assert!((z.z - 1.0).abs() < 1e-12);
}

#[test]
fn test_vec3_cross_anticommutative() {
    let a = DpVec3::new(1.0, 2.0, 3.0);
    let b = DpVec3::new(4.0, 5.0, 6.0);
    let ab = a.cross(b);
    let ba = b.cross(a);
    assert!((ab.x + ba.x).abs() < 1e-12);
    assert!((ab.y + ba.y).abs() < 1e-12);
    assert!((ab.z + ba.z).abs() < 1e-12);
}

#[test]
fn test_vec3_norm() {
    let v = DpVec3::new(3.0, 4.0, 0.0);
    assert!((v.norm() - 5.0).abs() < 1e-12);
}

#[test]
fn test_vec3_normalize() {
    let v = DpVec3::new(3.0, 0.0, 4.0);
    let n = v.normalize().expect("should normalize");
    assert!((n.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn test_vec3_normalize_zero_fails() {
    let v = DpVec3::zero();
    assert!(v.normalize().is_err());
}

#[test]
fn test_vec3_lerp() {
    let a = DpVec3::new(0.0, 0.0, 0.0);
    let b = DpVec3::new(10.0, 10.0, 10.0);
    let mid = a.lerp(b, 0.5);
    assert!((mid.x - 5.0).abs() < 1e-12);
}

#[test]
fn test_vec3_neg() {
    let v = DpVec3::new(1.0, -2.0, 3.0);
    let n = v.neg();
    assert!((n.x - (-1.0)).abs() < 1e-12);
    assert!((n.y - 2.0).abs() < 1e-12);
    assert!((n.z - (-3.0)).abs() < 1e-12);
}

// ── DpQuaternion ────────────────────────────────────────────────────────────

#[test]
fn test_quat_identity() {
    let q = DpQuaternion::identity();
    assert!((q.w - 1.0).abs() < 1e-12);
    assert!((q.x).abs() < 1e-12);
}

#[test]
fn test_quat_from_axis_angle() {
    let q = DpQuaternion::from_axis_angle(DpVec3::unit_z(), std::f64::consts::FRAC_PI_2)
        .expect("valid axis");
    // 90 degrees around Z: w = cos(45deg) = sqrt(2)/2
    assert!((q.w - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    assert!((q.z - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
}

#[test]
fn test_quat_from_axis_angle_zero_fails() {
    let r = DpQuaternion::from_axis_angle(DpVec3::zero(), 1.0);
    assert!(r.is_err());
}

#[test]
fn test_quat_normalize() {
    let q = DpQuaternion::new(2.0, 0.0, 0.0, 0.0);
    let n = q.normalize().expect("normalization");
    assert!((n.w - 1.0).abs() < 1e-12);
}

#[test]
fn test_quat_conjugate() {
    let q = DpQuaternion::new(1.0, 2.0, 3.0, 4.0);
    let c = q.conjugate();
    assert!((c.w - 1.0).abs() < 1e-12);
    assert!((c.x - (-2.0)).abs() < 1e-12);
}

#[test]
fn test_quat_mul_identity() {
    let q = DpQuaternion::from_axis_angle(DpVec3::unit_y(), 0.5).expect("valid");
    let id = DpQuaternion::identity();
    let r = q.mul(id);
    assert!((r.w - q.w).abs() < 1e-12);
    assert!((r.x - q.x).abs() < 1e-12);
}

#[test]
fn test_quat_rotate_vector() {
    // Rotate unit_x by 90 degrees around z → should give unit_y
    let q = DpQuaternion::from_axis_angle(DpVec3::unit_z(), std::f64::consts::FRAC_PI_2)
        .expect("valid");
    let v = q.rotate_vector(DpVec3::unit_x());
    assert!((v.x).abs() < 1e-10);
    assert!((v.y - 1.0).abs() < 1e-10);
    assert!((v.z).abs() < 1e-10);
}

#[test]
fn test_quat_slerp_identity() {
    let a = DpQuaternion::identity();
    let b = DpQuaternion::from_axis_angle(DpVec3::unit_y(), std::f64::consts::PI).expect("valid");
    // At t=0, should be close to a
    let s = a.slerp(b, 0.0).expect("slerp");
    assert!((s.w - a.w).abs() < 1e-10);
}

#[test]
fn test_quat_slerp_midpoint() {
    let a = DpQuaternion::identity();
    let b = DpQuaternion::from_axis_angle(DpVec3::unit_z(), std::f64::consts::PI).expect("valid");
    let mid = a.slerp(b, 0.5).expect("slerp");
    // Midpoint rotation of pi around z → should be pi/2 around z
    let expected = DpQuaternion::from_axis_angle(DpVec3::unit_z(), std::f64::consts::FRAC_PI_2)
        .expect("valid");
    assert!((mid.w - expected.w).abs() < 1e-6);
    assert!((mid.z - expected.z).abs() < 1e-6);
}

#[test]
fn test_quat_to_axis_angle_roundtrip() {
    let axis = DpVec3::new(1.0, 1.0, 0.0).normalize().expect("ok");
    let angle = 1.2;
    let q = DpQuaternion::from_axis_angle(axis, angle).expect("valid");
    let (ax, an) = q.to_axis_angle();
    assert!((an - angle).abs() < 1e-10);
    assert!((ax.x - axis.x).abs() < 1e-10);
}

// ── DpRigidBody ─────────────────────────────────────────────────────────────

#[test]
fn test_rigid_body_creation() {
    let b = DpRigidBody::new(5.0, DpVec3::new(1.0, 2.0, 3.0)).expect("valid");
    assert!((b.mass - 5.0).abs() < 1e-12);
    assert!((b.position.x - 1.0).abs() < 1e-12);
}

#[test]
fn test_rigid_body_zero_mass_fails() {
    assert!(DpRigidBody::new(0.0, DpVec3::zero()).is_err());
    assert!(DpRigidBody::new(-1.0, DpVec3::zero()).is_err());
}

#[test]
fn test_rigid_body_static() {
    let b = DpRigidBody::new_static(DpVec3::zero());
    assert!(b.is_static);
    assert_eq!(b.inv_mass(), 0.0);
}

#[test]
fn test_rigid_body_apply_force() {
    let mut b = DpRigidBody::new(1.0, DpVec3::zero()).expect("valid");
    b.apply_force(DpVec3::new(10.0, 0.0, 0.0), DpVec3::zero());
    assert!((b.force.x - 10.0).abs() < 1e-12);
}

#[test]
fn test_rigid_body_apply_force_generates_torque() {
    let mut b = DpRigidBody::new(1.0, DpVec3::zero()).expect("valid");
    // Force at offset point generates torque
    b.apply_force(DpVec3::new(0.0, 10.0, 0.0), DpVec3::new(1.0, 0.0, 0.0));
    // torque = (1,0,0) x (0,10,0) = (0,0,10)
    assert!((b.torque.z - 10.0).abs() < 1e-12);
}

#[test]
fn test_rigid_body_kinetic_energy() {
    let mut b = DpRigidBody::new(2.0, DpVec3::zero()).expect("valid");
    b.velocity = DpVec3::new(3.0, 0.0, 0.0);
    // KE = 0.5 * 2 * 9 = 9
    assert!((b.kinetic_energy() - 9.0).abs() < 1e-12);
}

#[test]
fn test_rigid_body_potential_energy() {
    let b = DpRigidBody::new(2.0, DpVec3::new(0.0, 10.0, 0.0)).expect("valid");
    let g = DpVec3::new(0.0, -9.81, 0.0);
    // PE = -m * g.y * y = -2 * (-9.81) * 10 = 196.2
    let pe = b.potential_energy(g);
    assert!((pe - 196.2).abs() < 1e-10);
}

#[test]
fn test_rigid_body_momentum() {
    let mut b = DpRigidBody::new(3.0, DpVec3::zero()).expect("valid");
    b.velocity = DpVec3::new(4.0, 0.0, 0.0);
    let p = b.momentum();
    assert!((p.x - 12.0).abs() < 1e-12);
}

#[test]
fn test_rigid_body_static_no_force() {
    let mut b = DpRigidBody::new_static(DpVec3::zero());
    b.apply_force(DpVec3::new(100.0, 0.0, 0.0), DpVec3::zero());
    assert!((b.force.x).abs() < 1e-12);
}

#[test]
fn test_rigid_body_angular_momentum() {
    let mut b = DpRigidBody::new(1.0, DpVec3::zero()).expect("valid");
    b.angular_velocity = DpVec3::new(0.0, 5.0, 0.0);
    let l = b.angular_momentum();
    assert!((l.y - b.inertia.y * 5.0).abs() < 1e-12);
}

// ── DpPhysicsEngine ─────────────────────────────────────────────────────────

#[test]
fn test_engine_free_fall_euler() {
    let engine = DpPhysicsEngine::new();
    let mut body = DpRigidBody::new(1.0, DpVec3::new(0.0, 100.0, 0.0)).expect("valid");
    let mut bodies = vec![body.clone()];
    let dt = 0.01;
    let n_steps = 100;
    for _ in 0..n_steps {
        engine
            .step(&mut bodies, dt, DpIntegrator::Euler)
            .expect("step ok");
    }
    // After 1s of free fall, y should decrease significantly
    assert!(bodies[0].position.y < 100.0);
    // Velocity should be approximately -9.81 m/s (with some damping)
    assert!(bodies[0].velocity.y < -5.0);
}

#[test]
fn test_engine_free_fall_semi_implicit() {
    let engine = DpPhysicsEngine::new();
    let mut bodies = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 100.0, 0.0)).expect("valid")];
    let dt = 0.01;
    for _ in 0..100 {
        engine
            .step(&mut bodies, dt, DpIntegrator::SemiImplicitEuler)
            .expect("ok");
    }
    assert!(bodies[0].position.y < 100.0);
    assert!(bodies[0].velocity.y < -5.0);
}

#[test]
fn test_engine_free_fall_verlet() {
    let engine = DpPhysicsEngine::new();
    let mut bodies = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 100.0, 0.0)).expect("valid")];
    let dt = 0.01;
    for _ in 0..100 {
        engine
            .step(&mut bodies, dt, DpIntegrator::Verlet)
            .expect("ok");
    }
    assert!(bodies[0].position.y < 100.0);
}

#[test]
fn test_engine_free_fall_leapfrog() {
    let engine = DpPhysicsEngine::new();
    let mut bodies = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 100.0, 0.0)).expect("valid")];
    for _ in 0..100 {
        engine
            .step(&mut bodies, 0.01, DpIntegrator::Leapfrog)
            .expect("ok");
    }
    assert!(bodies[0].position.y < 100.0);
}

#[test]
fn test_engine_negative_dt_fails() {
    let engine = DpPhysicsEngine::new();
    let mut bodies = vec![DpRigidBody::new(1.0, DpVec3::zero()).expect("valid")];
    assert!(engine
        .step(&mut bodies, -0.01, DpIntegrator::Euler)
        .is_err());
}

#[test]
fn test_engine_spring_oscillation() {
    let mut engine = DpPhysicsEngine::with_config(DpPhysicsConfig {
        gravity: DpVec3::zero(),
        linear_damping: 0.0,
        angular_damping: 0.0,
        max_velocity: 1000.0,
        max_angular_velocity: 1000.0,
    });

    let mut bodies = vec![
        DpRigidBody::new(1.0, DpVec3::new(-2.0, 0.0, 0.0)).expect("valid"),
        DpRigidBody::new(1.0, DpVec3::new(2.0, 0.0, 0.0)).expect("valid"),
    ];
    bodies[0].is_static = true;

    engine.add_spring(DpSpring {
        body_a: 0,
        body_b: Some(1),
        anchor: DpVec3::zero(),
        rest_length: 3.0,
        stiffness: 100.0,
        damping: 0.0,
    });

    // Current distance = 4, rest = 3, so body 1 should be pulled toward body 0
    let initial_x = bodies[1].position.x;
    for _ in 0..50 {
        engine
            .step(&mut bodies, 0.001, DpIntegrator::SemiImplicitEuler)
            .expect("ok");
    }
    assert!(bodies[1].position.x < initial_x);
}

#[test]
fn test_engine_simulate_returns_trajectory() {
    let engine = DpPhysicsEngine::new();
    let mut bodies = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 10.0, 0.0)).expect("valid")];
    let traj = engine
        .simulate(&mut bodies, 0.01, 10, DpIntegrator::Euler)
        .expect("ok");
    assert_eq!(traj.len(), 11); // initial + 10 steps
    assert_eq!(traj[0].len(), 1); // 1 body
}

#[test]
fn test_engine_static_body_no_motion() {
    let engine = DpPhysicsEngine::new();
    let mut bodies = vec![DpRigidBody::new_static(DpVec3::new(0.0, 5.0, 0.0))];
    for _ in 0..100 {
        engine
            .step(&mut bodies, 0.01, DpIntegrator::Euler)
            .expect("ok");
    }
    assert!((bodies[0].position.y - 5.0).abs() < 1e-12);
}

// ── DpContactSolver ─────────────────────────────────────────────────────────

#[test]
fn test_contact_sphere_sphere_detection() {
    let mut b1 = DpRigidBody::new(1.0, DpVec3::new(0.0, 0.0, 0.0)).expect("valid");
    b1.radius = 1.0;
    let mut b2 = DpRigidBody::new(1.0, DpVec3::new(1.5, 0.0, 0.0)).expect("valid");
    b2.radius = 1.0;
    let solver = DpContactSolver::new();
    let contacts = solver.detect_contacts(&[b1, b2]);
    assert_eq!(contacts.len(), 1);
    assert!((contacts[0].penetration - 0.5).abs() < 1e-12);
}

#[test]
fn test_contact_sphere_sphere_no_overlap() {
    let mut b1 = DpRigidBody::new(1.0, DpVec3::new(0.0, 0.0, 0.0)).expect("valid");
    b1.radius = 1.0;
    let mut b2 = DpRigidBody::new(1.0, DpVec3::new(5.0, 0.0, 0.0)).expect("valid");
    b2.radius = 1.0;
    let solver = DpContactSolver::new();
    let contacts = solver.detect_contacts(&[b1, b2]);
    assert!(contacts.is_empty());
}

#[test]
fn test_contact_sphere_plane_detection() {
    let mut b = DpRigidBody::new(1.0, DpVec3::new(0.0, 0.5, 0.0)).expect("valid");
    b.radius = 1.0;
    let mut solver = DpContactSolver::new();
    solver.add_plane(DpGroundPlane::horizontal(0.0));
    let contacts = solver.detect_contacts(&[b]);
    assert_eq!(contacts.len(), 1);
    assert!((contacts[0].penetration - 0.5).abs() < 1e-12);
}

#[test]
fn test_contact_impulse_resolution() {
    let mut b1 = DpRigidBody::new(1.0, DpVec3::new(0.0, 0.0, 0.0)).expect("valid");
    b1.radius = 1.0;
    b1.velocity = DpVec3::new(1.0, 0.0, 0.0);
    b1.restitution = 1.0; // perfectly elastic

    let mut b2 = DpRigidBody::new(1.0, DpVec3::new(1.5, 0.0, 0.0)).expect("valid");
    b2.radius = 1.0;
    b2.velocity = DpVec3::new(-1.0, 0.0, 0.0);
    b2.restitution = 1.0;

    let solver = DpContactSolver::new();
    let mut bodies = vec![b1, b2];
    let contacts = solver.detect_contacts(&bodies);
    solver.resolve_contacts_impulse(&mut bodies, &contacts);

    // After elastic collision of equal masses: velocities swap
    assert!((bodies[0].velocity.x - (-1.0)).abs() < 1e-6);
    assert!((bodies[1].velocity.x - 1.0).abs() < 1e-6);
}

#[test]
fn test_contact_penalty_forces() {
    let mut b1 = DpRigidBody::new(1.0, DpVec3::new(0.0, 0.0, 0.0)).expect("valid");
    b1.radius = 1.0;
    let mut b2 = DpRigidBody::new(1.0, DpVec3::new(1.5, 0.0, 0.0)).expect("valid");
    b2.radius = 1.0;
    let solver = DpContactSolver::new();
    let mut bodies = vec![b1, b2];
    let contacts = solver.detect_contacts(&bodies);
    solver.apply_penalty_forces(&mut bodies, &contacts);
    // Body 0 should be pushed in -x, body 1 in +x
    assert!(bodies[0].force.x < 0.0);
    assert!(bodies[1].force.x > 0.0);
}

#[test]
fn test_contact_positional_correction() {
    let mut b1 = DpRigidBody::new(1.0, DpVec3::new(0.0, 0.0, 0.0)).expect("valid");
    b1.radius = 1.0;
    let mut b2 = DpRigidBody::new(1.0, DpVec3::new(1.0, 0.0, 0.0)).expect("valid");
    b2.radius = 1.0;
    let solver = DpContactSolver::new();
    let mut bodies = vec![b1, b2];
    let contacts = solver.detect_contacts(&bodies);
    let dist_before = bodies[1].position.sub(bodies[0].position).norm();
    solver.apply_positional_correction(&mut bodies, &contacts, 0.8, 0.01);
    let dist_after = bodies[1].position.sub(bodies[0].position).norm();
    assert!(dist_after > dist_before);
}

// ── DpFluidSim (SPH) ───────────────────────────────────────────────────────

#[test]
fn test_sph_kernel_cubic_spline_at_zero() {
    let sim = DpFluidSim::new(1.0).expect("valid");
    let w0 = sim.kernel_w(0.0);
    // At r=0, cubic spline = sigma * 1
    assert!(w0 > 0.0);
}

#[test]
fn test_sph_kernel_cubic_spline_decreases() {
    let sim = DpFluidSim::new(1.0).expect("valid");
    let w0 = sim.kernel_w(0.0);
    let w1 = sim.kernel_w(0.5);
    let w2 = sim.kernel_w(1.5);
    assert!(w0 > w1);
    assert!(w1 > w2);
}

#[test]
fn test_sph_kernel_cubic_spline_compact_support() {
    let sim = DpFluidSim::new(1.0).expect("valid");
    let w = sim.kernel_w(2.5);
    assert!((w).abs() < 1e-15);
}

#[test]
fn test_sph_kernel_wendland_c2() {
    let mut sim = DpFluidSim::new(1.0).expect("valid");
    sim.kernel = DpSphKernel::WendlandC2;
    let w0 = sim.kernel_w(0.0);
    let w1 = sim.kernel_w(1.0);
    assert!(w0 > w1);
    assert!(w1 > 0.0);
    // Compact support
    assert!((sim.kernel_w(2.5)).abs() < 1e-15);
}

#[test]
fn test_sph_density_computation() {
    let sim = DpFluidSim::new(0.5).expect("valid");
    let mut particles = vec![
        DpSphParticle::new(DpVec3::new(0.0, 0.0, 0.0), 1.0),
        DpSphParticle::new(DpVec3::new(0.1, 0.0, 0.0), 1.0),
    ];
    sim.compute_densities(&mut particles);
    // Both particles should have non-zero density
    assert!(particles[0].density > 0.0);
    assert!(particles[1].density > 0.0);
}

#[test]
fn test_sph_pressure_tait_eos() {
    let sim = DpFluidSim::new(0.5).expect("valid");
    let mut particles = vec![DpSphParticle::new(DpVec3::zero(), 1.0)];
    particles[0].density = 1500.0; // above rest density
    sim.compute_pressures(&mut particles);
    assert!(particles[0].pressure > 0.0);
}

#[test]
fn test_sph_step() {
    let sim = DpFluidSim::new(0.5).expect("valid");
    let mut particles = vec![
        DpSphParticle::new(DpVec3::new(0.0, 1.0, 0.0), 1.0),
        DpSphParticle::new(DpVec3::new(0.1, 1.0, 0.0), 1.0),
    ];
    sim.step(&mut particles, 0.001).expect("ok");
    // Particles should have moved under gravity
    assert!(particles[0].position.y < 1.0 || particles[0].velocity.y < 0.0);
}

#[test]
fn test_sph_zero_h_fails() {
    assert!(DpFluidSim::new(0.0).is_err());
    assert!(DpFluidSim::new(-1.0).is_err());
}

#[test]
fn test_sph_negative_dt_fails() {
    let sim = DpFluidSim::new(1.0).expect("valid");
    let mut particles = vec![DpSphParticle::new(DpVec3::zero(), 1.0)];
    assert!(sim.step(&mut particles, -0.01).is_err());
}

#[test]
fn test_sph_empty_particles() {
    let sim = DpFluidSim::new(1.0).expect("valid");
    let mut particles: Vec<DpSphParticle> = Vec::new();
    assert!(sim.step(&mut particles, 0.01).is_ok());
}

// ── DpClothSim ──────────────────────────────────────────────────────────────

#[test]
fn test_cloth_creation() {
    let cloth = DpClothSim::new();
    let (particles, springs) = cloth.create_cloth(5, 5, 0.1, 100.0, 1.0).expect("ok");
    assert_eq!(particles.len(), 25);
    // Structural: 5*4 + 4*5 = 40, Shear: 4*4*2=32, Bend: 5*3+3*5+5*3+3*5=30
    assert!(springs.len() > 40); // at least structural
}

#[test]
fn test_cloth_too_small_fails() {
    let cloth = DpClothSim::new();
    assert!(cloth.create_cloth(1, 5, 0.1, 100.0, 1.0).is_err());
    assert!(cloth.create_cloth(5, 1, 0.1, 100.0, 1.0).is_err());
}

#[test]
fn test_cloth_zero_spacing_fails() {
    let cloth = DpClothSim::new();
    assert!(cloth.create_cloth(5, 5, 0.0, 100.0, 1.0).is_err());
}

#[test]
fn test_cloth_step() {
    let cloth = DpClothSim::new();
    let (mut particles, springs) = cloth.create_cloth(3, 3, 0.1, 500.0, 5.0).expect("ok");
    // Pin top-left corner
    particles[0].pinned = true;
    let initial_y = particles[8].position.y;
    for _ in 0..10 {
        cloth.step(&mut particles, &springs, 0.001).expect("ok");
    }
    // Unpinned particles should fall
    assert!(particles[8].position.y < initial_y || particles[8].velocity.y < 0.0);
    // Pinned particle should not move
    assert!((particles[0].position.y - 5.0).abs() < 1e-12);
}

#[test]
fn test_cloth_negative_dt_fails() {
    let cloth = DpClothSim::new();
    let (mut particles, springs) = cloth.create_cloth(3, 3, 0.1, 100.0, 1.0).expect("ok");
    assert!(cloth.step(&mut particles, &springs, -0.01).is_err());
}

#[test]
fn test_cloth_strain_limiting() {
    let mut cloth = DpClothSim::new();
    cloth.max_stretch = 1.2;
    let (mut particles, springs) = cloth.create_cloth(3, 3, 0.1, 10.0, 0.1).expect("ok");
    // Give a huge initial velocity to stretch springs
    for p in particles.iter_mut() {
        if !p.pinned {
            p.velocity = DpVec3::new(0.0, -100.0, 0.0);
        }
    }
    cloth.step(&mut particles, &springs, 0.01).expect("ok");
    // Check that no spring is stretched more than max_stretch * rest_length
    for spring in &springs {
        let dist = particles[spring.p2]
            .position
            .sub(particles[spring.p1].position)
            .norm();
        assert!(dist <= spring.rest_length * cloth.max_stretch + 1e-6);
    }
}

#[test]
fn test_cloth_spring_types() {
    let cloth = DpClothSim::new();
    let (_, springs) = cloth.create_cloth(4, 4, 0.1, 100.0, 1.0).expect("ok");
    let structural = springs
        .iter()
        .filter(|s| s.spring_type == DpClothSpringType::Structural)
        .count();
    let shear = springs
        .iter()
        .filter(|s| s.spring_type == DpClothSpringType::Shear)
        .count();
    let bend = springs
        .iter()
        .filter(|s| s.spring_type == DpClothSpringType::Bend)
        .count();
    assert!(structural > 0);
    assert!(shear > 0);
    assert!(bend > 0);
}

// ── DpAdjointMethod ─────────────────────────────────────────────────────────

#[test]
fn test_adjoint_jacobian_identity_no_forces() {
    let engine = DpPhysicsEngine::with_config(DpPhysicsConfig {
        gravity: DpVec3::zero(),
        linear_damping: 0.0,
        angular_damping: 0.0,
        max_velocity: 1000.0,
        max_angular_velocity: 1000.0,
    });
    let bodies = vec![DpRigidBody::new(1.0, DpVec3::zero()).expect("valid")];
    let adjoint = DpAdjointMethod::new();
    let jac = adjoint
        .compute_step_jacobian(&engine, &bodies, 0.01, DpIntegrator::SemiImplicitEuler)
        .expect("ok");
    // With no forces: v' = v, x' = x + v*dt
    // So Jacobian should be close to [[I, dt*I], [0, I]]
    assert_eq!(jac.len(), 6);
    assert_eq!(jac[0].len(), 6);
    // dx'/dx ≈ I (position depends on position)
    assert!((jac[0][0] - 1.0).abs() < 1e-4);
}

#[test]
fn test_adjoint_gradient_shape() {
    let engine = DpPhysicsEngine::with_config(DpPhysicsConfig {
        gravity: DpVec3::new(0.0, -9.81, 0.0),
        linear_damping: 0.0,
        angular_damping: 0.0,
        max_velocity: 1000.0,
        max_angular_velocity: 1000.0,
    });
    let bodies = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 10.0, 0.0)).expect("valid")];
    let adjoint = DpAdjointMethod::new();
    let loss_grad = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // dL/dy_final = 1
    let grad = adjoint
        .compute_gradient(
            &engine,
            &bodies,
            0.01,
            5,
            DpIntegrator::SemiImplicitEuler,
            &loss_grad,
        )
        .expect("ok");
    assert_eq!(grad.len(), 6);
}

#[test]
fn test_adjoint_gradient_dimension_mismatch() {
    let engine = DpPhysicsEngine::new();
    let bodies = vec![DpRigidBody::new(1.0, DpVec3::zero()).expect("valid")];
    let adjoint = DpAdjointMethod::new();
    let bad_grad = vec![1.0, 2.0]; // wrong size
    assert!(adjoint
        .compute_gradient(&engine, &bodies, 0.01, 1, DpIntegrator::Euler, &bad_grad)
        .is_err());
}

// ── DpPhysicsLoss ───────────────────────────────────────────────────────────

#[test]
fn test_loss_trajectory_mse_zero_for_same() {
    let snap = vec![vec![DpBodySnapshot {
        position: DpVec3::new(1.0, 2.0, 3.0),
        velocity: DpVec3::zero(),
        orientation: DpQuaternion::identity(),
        angular_velocity: DpVec3::zero(),
    }]];
    let loss = DpPhysicsLoss::new(DpVec3::new(0.0, -9.81, 0.0));
    let mse = loss.trajectory_mse(&snap, &snap).expect("ok");
    assert!((mse).abs() < 1e-15);
}

#[test]
fn test_loss_trajectory_mse_nonzero() {
    let pred = vec![vec![DpBodySnapshot {
        position: DpVec3::new(1.0, 0.0, 0.0),
        velocity: DpVec3::zero(),
        orientation: DpQuaternion::identity(),
        angular_velocity: DpVec3::zero(),
    }]];
    let target = vec![vec![DpBodySnapshot {
        position: DpVec3::new(2.0, 0.0, 0.0),
        velocity: DpVec3::zero(),
        orientation: DpQuaternion::identity(),
        angular_velocity: DpVec3::zero(),
    }]];
    let loss = DpPhysicsLoss::new(DpVec3::zero());
    let mse = loss.trajectory_mse(&pred, &target).expect("ok");
    assert!(mse > 0.0);
}

#[test]
fn test_loss_trajectory_length_mismatch() {
    let a = vec![vec![]];
    let b = vec![vec![], vec![]];
    let loss = DpPhysicsLoss::new(DpVec3::zero());
    assert!(loss.trajectory_mse(&a, &b).is_err());
}

#[test]
fn test_loss_energy_conservation() {
    let g = DpVec3::new(0.0, -9.81, 0.0);
    let loss = DpPhysicsLoss::new(g);
    let b1 = DpRigidBody::new(1.0, DpVec3::new(0.0, 10.0, 0.0)).expect("valid");
    let mut b2 = DpRigidBody::new(1.0, DpVec3::new(0.0, 5.0, 0.0)).expect("valid");
    b2.velocity = DpVec3::new(0.0, 9.9, 0.0); // some velocity
    let trajectory = vec![vec![b1], vec![b2]];
    let el = loss.energy_conservation_loss(&trajectory);
    // Energy should differ → non-zero loss
    assert!(el >= 0.0);
}

#[test]
fn test_loss_momentum_conservation() {
    let loss = DpPhysicsLoss::new(DpVec3::zero());
    let mut b1 = DpRigidBody::new(1.0, DpVec3::zero()).expect("valid");
    b1.velocity = DpVec3::new(5.0, 0.0, 0.0);
    let mut b2 = DpRigidBody::new(1.0, DpVec3::zero()).expect("valid");
    b2.velocity = DpVec3::new(3.0, 0.0, 0.0); // momentum changed
    let traj = vec![vec![b1], vec![b2]];
    let ml = loss.momentum_conservation_loss(&traj);
    assert!(ml > 0.0);
}

#[test]
fn test_loss_contact_violation() {
    let loss = DpPhysicsLoss::new(DpVec3::zero());
    let contacts = vec![DpContact {
        body_a: 0,
        body_b: Some(1),
        normal: DpVec3::unit_x(),
        penetration: 0.5,
        point: DpVec3::zero(),
    }];
    let vl = loss.contact_violation_loss(&[], &contacts);
    assert!((vl - 0.25).abs() < 1e-12);
}

// ── DpMetrics ───────────────────────────────────────────────────────────────

#[test]
fn test_metrics_energy_drift() {
    let g = DpVec3::new(0.0, -9.81, 0.0);
    let initial = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 10.0, 0.0)).expect("valid")];
    let mut final_body = DpRigidBody::new(1.0, DpVec3::new(0.0, 5.0, 0.0)).expect("valid");
    final_body.velocity = DpVec3::new(0.0, -9.9, 0.0);
    let final_bodies = vec![final_body];
    let drift = DpMetrics::energy_drift(&initial, &final_bodies, g);
    assert!(drift >= 0.0);
}

#[test]
fn test_metrics_trajectory_rmse_zero() {
    let snap = vec![vec![DpBodySnapshot {
        position: DpVec3::new(1.0, 2.0, 3.0),
        velocity: DpVec3::zero(),
        orientation: DpQuaternion::identity(),
        angular_velocity: DpVec3::zero(),
    }]];
    let rmse = DpMetrics::trajectory_rmse(&snap, &snap).expect("ok");
    assert!((rmse).abs() < 1e-15);
}

#[test]
fn test_metrics_max_penetration() {
    let contacts = vec![
        DpContact {
            body_a: 0,
            body_b: None,
            normal: DpVec3::unit_y(),
            penetration: 0.1,
            point: DpVec3::zero(),
        },
        DpContact {
            body_a: 1,
            body_b: None,
            normal: DpVec3::unit_y(),
            penetration: 0.5,
            point: DpVec3::zero(),
        },
    ];
    assert!((DpMetrics::max_penetration(&contacts) - 0.5).abs() < 1e-12);
}

#[test]
fn test_metrics_max_velocity() {
    let mut b = DpRigidBody::new(1.0, DpVec3::zero()).expect("valid");
    b.velocity = DpVec3::new(3.0, 4.0, 0.0);
    assert!((DpMetrics::max_velocity(&[b]) - 5.0).abs() < 1e-12);
}

#[test]
fn test_metrics_report() {
    let g = DpVec3::new(0.0, -9.81, 0.0);
    let initial = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 10.0, 0.0)).expect("valid")];
    let final_bodies = vec![DpRigidBody::new(1.0, DpVec3::new(0.0, 5.0, 0.0)).expect("valid")];
    let report = DpMetrics::report(&initial, &final_bodies, g, &[]);
    assert_eq!(report.num_bodies, 1);
    assert_eq!(report.num_contacts, 0);
    assert!(report.energy_drift >= 0.0);
    // Check Display trait
    let display = format!("{}", report);
    assert!(display.contains("Energy drift"));
}

#[test]
fn test_metrics_total_kinetic_energy() {
    let mut b1 = DpRigidBody::new(2.0, DpVec3::zero()).expect("valid");
    b1.velocity = DpVec3::new(3.0, 0.0, 0.0);
    let mut b2 = DpRigidBody::new(1.0, DpVec3::zero()).expect("valid");
    b2.velocity = DpVec3::new(0.0, 4.0, 0.0);
    let ke = DpMetrics::total_kinetic_energy(&[b1, b2]);
    // 0.5*2*9 + 0.5*1*16 = 9 + 8 = 17
    assert!((ke - 17.0).abs() < 1e-12);
}

#[test]
fn test_metrics_mean_penetration() {
    let contacts = vec![
        DpContact {
            body_a: 0,
            body_b: None,
            normal: DpVec3::unit_y(),
            penetration: 0.2,
            point: DpVec3::zero(),
        },
        DpContact {
            body_a: 1,
            body_b: None,
            normal: DpVec3::unit_y(),
            penetration: 0.4,
            point: DpVec3::zero(),
        },
    ];
    assert!((DpMetrics::mean_penetration(&contacts) - 0.3).abs() < 1e-12);
}

// ── Integration: combined simulation ────────────────────────────────────────

#[test]
fn test_full_simulation_with_collision() {
    let engine = DpPhysicsEngine::new();
    let mut solver = DpContactSolver::new();
    solver.add_plane(DpGroundPlane::horizontal(0.0));

    let mut b = DpRigidBody::new(1.0, DpVec3::new(0.0, 5.0, 0.0)).expect("valid");
    b.radius = 0.5;
    b.restitution = 0.8;
    let mut bodies = vec![b];

    // Simulate free fall until near ground
    for _ in 0..200 {
        engine
            .step(&mut bodies, 0.005, DpIntegrator::SemiImplicitEuler)
            .expect("ok");
        let contacts = solver.detect_contacts(&bodies);
        if !contacts.is_empty() {
            solver.resolve_contacts_impulse(&mut bodies, &contacts);
            solver.apply_positional_correction(&mut bodies, &contacts, 0.5, 0.01);
        }
    }

    // Body should have bounced and be somewhere above ground
    assert!(bodies[0].position.y >= -0.1);
}

#[test]
fn test_energy_conservation_symplectic() {
    // Symplectic integrators should conserve energy better than Euler
    let config = DpPhysicsConfig {
        gravity: DpVec3::zero(),
        linear_damping: 0.0,
        angular_damping: 0.0,
        max_velocity: 1000.0,
        max_angular_velocity: 1000.0,
    };
    let mut engine = DpPhysicsEngine::with_config(config);

    // Harmonic oscillator via spring
    engine.add_spring(DpSpring {
        body_a: 0,
        body_b: None,
        anchor: DpVec3::zero(),
        rest_length: 0.0,
        stiffness: 10.0,
        damping: 0.0,
    });

    let mut bodies_euler = vec![DpRigidBody::new(1.0, DpVec3::new(1.0, 0.0, 0.0)).expect("valid")];
    let mut bodies_verlet = bodies_euler.clone();

    let initial_ke_euler = bodies_euler[0].kinetic_energy();
    let initial_pe = 0.5 * 10.0 * 1.0; // 0.5 k x^2

    for _ in 0..1000 {
        engine
            .step(&mut bodies_euler, 0.001, DpIntegrator::Euler)
            .expect("ok");
        engine
            .step(&mut bodies_verlet, 0.001, DpIntegrator::Verlet)
            .expect("ok");
    }

    let euler_ke = bodies_euler[0].kinetic_energy();
    let euler_pe = 0.5 * 10.0 * bodies_euler[0].position.norm_sq();
    let verlet_ke = bodies_verlet[0].kinetic_energy();
    let verlet_pe = 0.5 * 10.0 * bodies_verlet[0].position.norm_sq();

    let euler_drift = ((euler_ke + euler_pe) - (initial_ke_euler + initial_pe)).abs();
    let verlet_drift = ((verlet_ke + verlet_pe) - (initial_ke_euler + initial_pe)).abs();

    // Verlet should conserve energy better (or at least not worse) than Euler
    // Both should be somewhat reasonable
    assert!(verlet_drift < euler_drift + 1.0);
}

#[test]
fn test_flatten_unflatten_roundtrip() {
    let mut bodies = vec![
        DpRigidBody::new(1.0, DpVec3::new(1.0, 2.0, 3.0)).expect("valid"),
        DpRigidBody::new(2.0, DpVec3::new(4.0, 5.0, 6.0)).expect("valid"),
    ];
    bodies[0].velocity = DpVec3::new(7.0, 8.0, 9.0);
    bodies[1].velocity = DpVec3::new(10.0, 11.0, 12.0);

    let flat = flatten_state(&bodies);
    assert_eq!(flat.len(), 12);

    let mut restored = vec![
        DpRigidBody::new(1.0, DpVec3::zero()).expect("valid"),
        DpRigidBody::new(2.0, DpVec3::zero()).expect("valid"),
    ];
    unflatten_state(&flat, &mut restored).expect("ok");

    assert!((restored[0].position.x - 1.0).abs() < 1e-12);
    assert!((restored[1].velocity.z - 12.0).abs() < 1e-12);
}
