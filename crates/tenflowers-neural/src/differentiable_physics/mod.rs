//! Differentiable Physics Simulation — production-grade pure-Rust module.
//!
//! Implements differentiable rigid-body dynamics, fluid simulation (SPH), cloth
//! simulation, and adjoint-method gradient computation through physics. All types
//! are prefixed with `Dp` to avoid namespace collisions.
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`DpVec3`] | 3D vector with full arithmetic |
//! | [`DpQuaternion`] | Unit quaternion for rotations (SLERP, axis-angle) |
//! | [`DpRigidBody`] | Rigid body state (position, velocity, orientation, angular) |
//! | [`DpPhysicsEngine`] | Differentiable integrators (Euler, semi-implicit, Verlet, symplectic) |
//! | [`DpContactSolver`] | Contact detection and resolution (penalty, impulse, Coulomb friction) |
//! | [`DpSphParticle`] / [`DpFluidSim`] | SPH fluid with cubic-spline / Wendland C2 kernels |
//! | [`DpClothSim`] | Mass-spring cloth (structural + shear + bend springs) |
//! | [`DpAdjointMethod`] | Gradient through physics via adjoint ODE |
//! | [`DpPhysicsLoss`] | Physics-informed loss functions |
//! | [`DpMetrics`] / [`DpReport`] | Simulation quality metrics |
//!
//! All computation uses `f64` for numerical precision. No `unwrap()` anywhere.

use tenflowers_core::{Result, TensorError};

#[cfg(test)]
mod tests;

#[inline]
fn dp_err(op: &str, reason: &str) -> TensorError {
    TensorError::InvalidArgument {
        operation: op.to_string(),
        reason: reason.to_string(),
        context: None,
    }
}

// ── 1. DpVec3 — 3D vector type ─────────────────────────────────────────────

/// 3D vector used throughout the differentiable physics module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl DpVec3 {
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    #[inline]
    pub fn unit_x() -> Self {
        Self {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }
    #[inline]
    pub fn unit_y() -> Self {
        Self {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }
    #[inline]
    pub fn unit_z() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
    #[inline]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    #[inline]
    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize to unit length. Returns error if near-zero.
    pub fn normalize(self) -> Result<Self> {
        let n = self.norm();
        if n < 1e-15 {
            return Err(dp_err(
                "DpVec3::normalize",
                "cannot normalize near-zero vector",
            ));
        }
        Ok(self.scale(1.0 / n))
    }

    #[inline]
    pub fn lerp(self, o: Self, t: f64) -> Self {
        self.scale(1.0 - t).add(o.scale(t))
    }
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }
    #[inline]
    pub fn max_component(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }
}

impl Default for DpVec3 {
    fn default() -> Self {
        Self::zero()
    }
}

// ── 2. DpQuaternion — rotation quaternion ───────────────────────────────────

/// Unit quaternion representing a 3D rotation. Stored as (w, x, y, z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpQuaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl DpQuaternion {
    #[inline]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }
    #[inline]
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create from axis-angle. `axis` must be non-zero; `angle` in radians.
    pub fn from_axis_angle(axis: DpVec3, angle: f64) -> Result<Self> {
        let n = axis.norm();
        if n < 1e-15 {
            return Err(dp_err(
                "DpQuaternion::from_axis_angle",
                "axis must be non-zero",
            ));
        }
        let half = angle * 0.5;
        let s = half.sin() / n;
        Ok(Self {
            w: half.cos(),
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
        })
    }

    #[inline]
    pub fn norm(self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(self) -> Result<Self> {
        let n = self.norm();
        if n < 1e-15 {
            return Err(dp_err(
                "DpQuaternion::normalize",
                "cannot normalize zero quaternion",
            ));
        }
        let inv = 1.0 / n;
        Ok(Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        })
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Hamilton product: self * other.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, o: Self) -> Self {
        Self {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    /// Rotate a 3D vector: q v q*.
    pub fn rotate_vector(self, v: DpVec3) -> DpVec3 {
        let qv = DpQuaternion::new(0.0, v.x, v.y, v.z);
        let r = self.mul(qv).mul(self.conjugate());
        DpVec3::new(r.x, r.y, r.z)
    }

    /// Spherical linear interpolation.
    pub fn slerp(self, other: Self, t: f64) -> Result<Self> {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let mut o = other;
        if dot < 0.0 {
            o = DpQuaternion::new(-o.w, -o.x, -o.y, -o.z);
            dot = -dot;
        }
        if dot > 0.9995 {
            return DpQuaternion::new(
                self.w + t * (o.w - self.w),
                self.x + t * (o.x - self.x),
                self.y + t * (o.y - self.y),
                self.z + t * (o.z - self.z),
            )
            .normalize();
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        if sin_theta.abs() < 1e-15 {
            return self.normalize();
        }
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;
        DpQuaternion::new(
            a * self.w + b * o.w,
            a * self.x + b * o.x,
            a * self.y + b * o.y,
            a * self.z + b * o.z,
        )
        .normalize()
    }

    /// Extract axis and angle from this quaternion.
    pub fn to_axis_angle(self) -> (DpVec3, f64) {
        let n = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n < 1e-15 {
            return (DpVec3::unit_x(), 0.0);
        }
        (
            DpVec3::new(self.x / n, self.y / n, self.z / n),
            2.0 * n.atan2(self.w),
        )
    }
}

impl Default for DpQuaternion {
    fn default() -> Self {
        Self::identity()
    }
}

// ── 3. DpRigidBody ──────────────────────────────────────────────────────────

/// Rigid body state for differentiable simulation.
#[derive(Debug, Clone)]
pub struct DpRigidBody {
    pub position: DpVec3,
    pub velocity: DpVec3,
    pub orientation: DpQuaternion,
    pub angular_velocity: DpVec3,
    pub mass: f64,
    /// Inertia tensor diagonal (Ixx, Iyy, Izz).
    pub inertia: DpVec3,
    pub force: DpVec3,
    pub torque: DpVec3,
    pub restitution: f64,
    pub friction: f64,
    pub radius: f64,
    pub is_static: bool,
}

impl DpRigidBody {
    pub fn new(mass: f64, position: DpVec3) -> Result<Self> {
        if mass <= 0.0 {
            return Err(dp_err("DpRigidBody::new", "mass must be positive"));
        }
        let iv = 0.4 * mass;
        Ok(Self {
            position,
            velocity: DpVec3::zero(),
            orientation: DpQuaternion::identity(),
            angular_velocity: DpVec3::zero(),
            mass,
            inertia: DpVec3::new(iv, iv, iv),
            force: DpVec3::zero(),
            torque: DpVec3::zero(),
            restitution: 0.5,
            friction: 0.3,
            radius: 1.0,
            is_static: false,
        })
    }

    pub fn new_static(position: DpVec3) -> Self {
        Self {
            position,
            velocity: DpVec3::zero(),
            orientation: DpQuaternion::identity(),
            angular_velocity: DpVec3::zero(),
            mass: f64::INFINITY,
            inertia: DpVec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            force: DpVec3::zero(),
            torque: DpVec3::zero(),
            restitution: 0.5,
            friction: 0.5,
            radius: 1.0,
            is_static: true,
        }
    }

    /// Apply a force at a given world-space point.
    pub fn apply_force(&mut self, force: DpVec3, point: DpVec3) {
        if self.is_static {
            return;
        }
        self.force = self.force.add(force);
        self.torque = self.torque.add(point.sub(self.position).cross(force));
    }

    pub fn apply_torque(&mut self, torque: DpVec3) {
        if !self.is_static {
            self.torque = self.torque.add(torque);
        }
    }

    pub fn clear_forces(&mut self) {
        self.force = DpVec3::zero();
        self.torque = DpVec3::zero();
    }

    #[inline]
    pub fn inv_mass(&self) -> f64 {
        if self.is_static {
            0.0
        } else {
            1.0 / self.mass
        }
    }

    #[inline]
    pub fn inv_inertia(&self) -> DpVec3 {
        if self.is_static {
            return DpVec3::zero();
        }
        DpVec3::new(
            if self.inertia.x.abs() > 1e-15 {
                1.0 / self.inertia.x
            } else {
                0.0
            },
            if self.inertia.y.abs() > 1e-15 {
                1.0 / self.inertia.y
            } else {
                0.0
            },
            if self.inertia.z.abs() > 1e-15 {
                1.0 / self.inertia.z
            } else {
                0.0
            },
        )
    }

    /// Kinetic energy: (1/2)mv^2 + (1/2)Iw^2.
    pub fn kinetic_energy(&self) -> f64 {
        if self.is_static {
            return 0.0;
        }
        let w = self.angular_velocity;
        0.5 * self.mass * self.velocity.norm_sq()
            + 0.5
                * (self.inertia.x * w.x * w.x
                    + self.inertia.y * w.y * w.y
                    + self.inertia.z * w.z * w.z)
    }

    /// Potential energy under uniform gravity: -m * g . position.
    pub fn potential_energy(&self, gravity: DpVec3) -> f64 {
        if self.is_static {
            return 0.0;
        }
        -self.mass * gravity.dot(self.position)
    }

    pub fn total_energy(&self, gravity: DpVec3) -> f64 {
        self.kinetic_energy() + self.potential_energy(gravity)
    }

    pub fn momentum(&self) -> DpVec3 {
        if self.is_static {
            DpVec3::zero()
        } else {
            self.velocity.scale(self.mass)
        }
    }

    pub fn angular_momentum(&self) -> DpVec3 {
        if self.is_static {
            return DpVec3::zero();
        }
        DpVec3::new(
            self.inertia.x * self.angular_velocity.x,
            self.inertia.y * self.angular_velocity.y,
            self.inertia.z * self.angular_velocity.z,
        )
    }
}

// ── 4. DpPhysicsEngine ─────────────────────────────────────────────────────

/// Integration method for physics simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpIntegrator {
    Euler,
    SemiImplicitEuler,
    Verlet,
    Leapfrog,
}

/// Spring connecting two bodies (or a body to a fixed anchor).
#[derive(Debug, Clone)]
pub struct DpSpring {
    pub body_a: usize,
    pub body_b: Option<usize>,
    pub anchor: DpVec3,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
}

/// Configuration for the physics engine.
#[derive(Debug, Clone)]
pub struct DpPhysicsConfig {
    pub gravity: DpVec3,
    pub linear_damping: f64,
    pub angular_damping: f64,
    pub max_velocity: f64,
    pub max_angular_velocity: f64,
}

impl Default for DpPhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: DpVec3::new(0.0, -9.81, 0.0),
            linear_damping: 0.01,
            angular_damping: 0.05,
            max_velocity: 100.0,
            max_angular_velocity: 50.0,
        }
    }
}

/// Differentiable physics engine.
pub struct DpPhysicsEngine {
    pub config: DpPhysicsConfig,
    pub springs: Vec<DpSpring>,
}

impl DpPhysicsEngine {
    pub fn new() -> Self {
        Self {
            config: DpPhysicsConfig::default(),
            springs: Vec::new(),
        }
    }

    pub fn with_config(config: DpPhysicsConfig) -> Self {
        Self {
            config,
            springs: Vec::new(),
        }
    }

    pub fn add_spring(&mut self, spring: DpSpring) {
        self.springs.push(spring);
    }

    fn apply_gravity(&self, bodies: &mut [DpRigidBody]) {
        for body in bodies.iter_mut() {
            if !body.is_static {
                body.force = body.force.add(self.config.gravity.scale(body.mass));
            }
        }
    }

    fn apply_spring_forces(&self, bodies: &mut [DpRigidBody]) {
        for spring in &self.springs {
            let pos_a = bodies[spring.body_a].position;
            let vel_a = bodies[spring.body_a].velocity;
            let (pos_b, vel_b) = if let Some(idx) = spring.body_b {
                (bodies[idx].position, bodies[idx].velocity)
            } else {
                (spring.anchor, DpVec3::zero())
            };
            let diff = pos_b.sub(pos_a);
            let dist = diff.norm();
            if dist < 1e-15 {
                continue;
            }
            let dir = diff.scale(1.0 / dist);
            let f_spring = dir.scale(spring.stiffness * (dist - spring.rest_length));
            let f_damp = dir.scale(spring.damping * vel_b.sub(vel_a).dot(dir));
            let total = f_spring.add(f_damp);
            bodies[spring.body_a].force = bodies[spring.body_a].force.add(total);
            if let Some(idx) = spring.body_b {
                bodies[idx].force = bodies[idx].force.sub(total);
            }
        }
    }

    fn apply_damping(&self, bodies: &mut [DpRigidBody]) {
        for body in bodies.iter_mut() {
            if body.is_static {
                continue;
            }
            body.force = body
                .force
                .sub(body.velocity.scale(self.config.linear_damping));
            body.torque = body
                .torque
                .sub(body.angular_velocity.scale(self.config.angular_damping));
        }
    }

    fn clamp_velocities(&self, bodies: &mut [DpRigidBody]) {
        for body in bodies.iter_mut() {
            let v = body.velocity.norm();
            if v > self.config.max_velocity {
                body.velocity = body.velocity.scale(self.config.max_velocity / v);
            }
            let w = body.angular_velocity.norm();
            if w > self.config.max_angular_velocity {
                body.angular_velocity = body
                    .angular_velocity
                    .scale(self.config.max_angular_velocity / w);
            }
        }
    }

    fn update_orientation(body: &mut DpRigidBody, dt: f64) -> Result<()> {
        let w = body.angular_velocity;
        let wn = w.norm();
        if wn > 1e-15 {
            let dq = DpQuaternion::from_axis_angle(w.scale(1.0 / wn), wn * dt)?;
            body.orientation = dq.mul(body.orientation);
            body.orientation = body.orientation.normalize()?;
        }
        Ok(())
    }

    fn compute_angular_accel(body: &DpRigidBody) -> DpVec3 {
        let inv_i = body.inv_inertia();
        DpVec3::new(
            body.torque.x * inv_i.x,
            body.torque.y * inv_i.y,
            body.torque.z * inv_i.z,
        )
    }

    pub fn step(
        &self,
        bodies: &mut [DpRigidBody],
        dt: f64,
        integrator: DpIntegrator,
    ) -> Result<()> {
        if dt <= 0.0 {
            return Err(dp_err("DpPhysicsEngine::step", "dt must be positive"));
        }
        for body in bodies.iter_mut() {
            body.clear_forces();
        }
        self.apply_gravity(bodies);
        self.apply_spring_forces(bodies);
        self.apply_damping(bodies);
        match integrator {
            DpIntegrator::Euler => self.integrate_euler(bodies, dt)?,
            DpIntegrator::SemiImplicitEuler => self.integrate_semi_implicit(bodies, dt)?,
            DpIntegrator::Verlet => self.integrate_verlet(bodies, dt)?,
            DpIntegrator::Leapfrog => self.integrate_leapfrog(bodies, dt)?,
        }
        self.clamp_velocities(bodies);
        Ok(())
    }

    fn integrate_euler(&self, bodies: &mut [DpRigidBody], dt: f64) -> Result<()> {
        for body in bodies.iter_mut() {
            if body.is_static {
                continue;
            }
            let a = body.force.scale(body.inv_mass());
            body.position = body.position.add(body.velocity.scale(dt));
            body.velocity = body.velocity.add(a.scale(dt));
            let alpha = Self::compute_angular_accel(body);
            Self::update_orientation(body, dt)?;
            body.angular_velocity = body.angular_velocity.add(alpha.scale(dt));
        }
        Ok(())
    }

    fn integrate_semi_implicit(&self, bodies: &mut [DpRigidBody], dt: f64) -> Result<()> {
        for body in bodies.iter_mut() {
            if body.is_static {
                continue;
            }
            let a = body.force.scale(body.inv_mass());
            body.velocity = body.velocity.add(a.scale(dt));
            body.position = body.position.add(body.velocity.scale(dt));
            let alpha = Self::compute_angular_accel(body);
            body.angular_velocity = body.angular_velocity.add(alpha.scale(dt));
            Self::update_orientation(body, dt)?;
        }
        Ok(())
    }

    fn integrate_verlet(&self, bodies: &mut [DpRigidBody], dt: f64) -> Result<()> {
        for body in bodies.iter_mut() {
            if body.is_static {
                continue;
            }
            let a = body.force.scale(body.inv_mass());
            body.position = body
                .position
                .add(body.velocity.scale(dt))
                .add(a.scale(0.5 * dt * dt));
            body.velocity = body.velocity.add(a.scale(dt));
            let alpha = Self::compute_angular_accel(body);
            body.angular_velocity = body.angular_velocity.add(alpha.scale(dt));
            Self::update_orientation(body, dt)?;
        }
        Ok(())
    }

    fn integrate_leapfrog(&self, bodies: &mut [DpRigidBody], dt: f64) -> Result<()> {
        for body in bodies.iter_mut() {
            if body.is_static {
                continue;
            }
            let a = body.force.scale(body.inv_mass());
            body.velocity = body.velocity.add(a.scale(dt));
            body.position = body.position.add(body.velocity.scale(dt));
            let alpha = Self::compute_angular_accel(body);
            body.angular_velocity = body.angular_velocity.add(alpha.scale(dt));
            Self::update_orientation(body, dt)?;
        }
        Ok(())
    }

    /// Simulate for n_steps, returning trajectory snapshots.
    pub fn simulate(
        &self,
        bodies: &mut [DpRigidBody],
        dt: f64,
        n_steps: usize,
        integrator: DpIntegrator,
    ) -> Result<Vec<Vec<DpBodySnapshot>>> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push(snapshot_bodies(bodies));
        for _ in 0..n_steps {
            self.step(bodies, dt, integrator)?;
            trajectory.push(snapshot_bodies(bodies));
        }
        Ok(trajectory)
    }
}

impl Default for DpPhysicsEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of a body's state at one point in time.
#[derive(Debug, Clone)]
pub struct DpBodySnapshot {
    pub position: DpVec3,
    pub velocity: DpVec3,
    pub orientation: DpQuaternion,
    pub angular_velocity: DpVec3,
}

fn snapshot_bodies(bodies: &[DpRigidBody]) -> Vec<DpBodySnapshot> {
    bodies
        .iter()
        .map(|b| DpBodySnapshot {
            position: b.position,
            velocity: b.velocity,
            orientation: b.orientation,
            angular_velocity: b.angular_velocity,
        })
        .collect()
}

// ── 5. DpContactSolver ──────────────────────────────────────────────────────

/// Contact between two objects.
#[derive(Debug, Clone)]
pub struct DpContact {
    pub body_a: usize,
    pub body_b: Option<usize>,
    pub normal: DpVec3,
    pub penetration: f64,
    pub point: DpVec3,
}

/// Ground plane for collision detection.
#[derive(Debug, Clone)]
pub struct DpGroundPlane {
    pub normal: DpVec3,
    pub offset: f64,
}

impl DpGroundPlane {
    pub fn horizontal(height: f64) -> Self {
        Self {
            normal: DpVec3::unit_y(),
            offset: height,
        }
    }
}

/// Contact solver with penalty and impulse methods.
pub struct DpContactSolver {
    pub penalty_stiffness: f64,
    pub penalty_damping: f64,
    pub planes: Vec<DpGroundPlane>,
}

impl DpContactSolver {
    pub fn new() -> Self {
        Self {
            penalty_stiffness: 10000.0,
            penalty_damping: 100.0,
            planes: Vec::new(),
        }
    }

    pub fn add_plane(&mut self, plane: DpGroundPlane) {
        self.planes.push(plane);
    }

    /// Detect sphere-sphere and sphere-plane contacts.
    pub fn detect_contacts(&self, bodies: &[DpRigidBody]) -> Vec<DpContact> {
        let mut contacts = Vec::new();
        let n = bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = bodies[j].position.sub(bodies[i].position);
                let dist = diff.norm();
                let min_dist = bodies[i].radius + bodies[j].radius;
                if dist < min_dist && dist > 1e-15 {
                    // Normal points from B toward A (toward body_a), consistent with plane normals
                    let normal = diff.scale(-1.0 / dist);
                    let penetration = min_dist - dist;
                    let point = bodies[i]
                        .position
                        .add(normal.scale(bodies[i].radius - penetration * 0.5));
                    contacts.push(DpContact {
                        body_a: i,
                        body_b: Some(j),
                        normal,
                        penetration,
                        point,
                    });
                }
            }
        }
        for (i, body) in bodies.iter().enumerate() {
            for plane in &self.planes {
                let dist = body.position.dot(plane.normal) - plane.offset;
                let penetration = body.radius - dist;
                if penetration > 0.0 {
                    let point = body.position.sub(plane.normal.scale(dist));
                    contacts.push(DpContact {
                        body_a: i,
                        body_b: None,
                        normal: plane.normal,
                        penetration,
                        point,
                    });
                }
            }
        }
        contacts
    }

    /// Apply penalty-based contact forces.
    pub fn apply_penalty_forces(&self, bodies: &mut [DpRigidBody], contacts: &[DpContact]) {
        for c in contacts {
            let f_n = c.normal.scale(self.penalty_stiffness * c.penetration);
            let vel_a = bodies[c.body_a].velocity;
            let vel_b = c
                .body_b
                .map(|i| bodies[i].velocity)
                .unwrap_or(DpVec3::zero());
            let f_damp = c
                .normal
                .scale(-self.penalty_damping * vel_a.sub(vel_b).dot(c.normal));
            let total = f_n.add(f_damp);
            // Normal points toward body_a; penalty pushes A in +normal, B in -normal
            bodies[c.body_a].force = bodies[c.body_a].force.add(total);
            if let Some(idx) = c.body_b {
                bodies[idx].force = bodies[idx].force.sub(total);
            }
        }
    }

    /// Resolve contacts using impulse-based method with Coulomb friction.
    pub fn resolve_contacts_impulse(&self, bodies: &mut [DpRigidBody], contacts: &[DpContact]) {
        for c in contacts {
            let ba = &bodies[c.body_a];
            let (im_a, va, rest_a, fric_a) =
                (ba.inv_mass(), ba.velocity, ba.restitution, ba.friction);
            let (im_b, vb, rest_b, fric_b) = if let Some(idx) = c.body_b {
                let bb = &bodies[idx];
                (bb.inv_mass(), bb.velocity, bb.restitution, bb.friction)
            } else {
                (0.0, DpVec3::zero(), 0.5, 0.5)
            };
            let v_rel = va.sub(vb);
            let v_rel_n = v_rel.dot(c.normal);
            if v_rel_n >= 0.0 {
                continue;
            }
            let inv_sum = im_a + im_b;
            if inv_sum < 1e-15 {
                continue;
            }
            let rest = (rest_a + rest_b) * 0.5;
            let j = -(1.0 + rest) * v_rel_n / inv_sum;
            let impulse_n = c.normal.scale(j);
            bodies[c.body_a].velocity = bodies[c.body_a].velocity.add(impulse_n.scale(im_a));
            if let Some(idx) = c.body_b {
                bodies[idx].velocity = bodies[idx].velocity.sub(impulse_n.scale(im_b));
            }
            // Coulomb friction
            let v_tan = v_rel.sub(c.normal.scale(v_rel_n));
            let vtn = v_tan.norm();
            if vtn > 1e-15 {
                let mu = (fric_a + fric_b) * 0.5;
                let j_friction = (vtn / inv_sum).min(mu * j.abs());
                let impulse_t = v_tan.scale(-j_friction / vtn);
                bodies[c.body_a].velocity = bodies[c.body_a].velocity.add(impulse_t.scale(im_a));
                if let Some(idx) = c.body_b {
                    bodies[idx].velocity = bodies[idx].velocity.sub(impulse_t.scale(im_b));
                }
            }
        }
    }

    /// Positional correction (Baumgarte stabilization).
    pub fn apply_positional_correction(
        &self,
        bodies: &mut [DpRigidBody],
        contacts: &[DpContact],
        beta: f64,
        slop: f64,
    ) {
        for c in contacts {
            let cm = (c.penetration - slop).max(0.0) * beta;
            let im_a = bodies[c.body_a].inv_mass();
            let im_b = c.body_b.map(|i| bodies[i].inv_mass()).unwrap_or(0.0);
            let inv_sum = im_a + im_b;
            if inv_sum < 1e-15 {
                continue;
            }
            let corr = c.normal.scale(cm / inv_sum);
            // Normal points toward body_a; correction pushes A in +normal, B in -normal
            if !bodies[c.body_a].is_static {
                bodies[c.body_a].position = bodies[c.body_a].position.add(corr.scale(im_a));
            }
            if let Some(idx) = c.body_b {
                if !bodies[idx].is_static {
                    bodies[idx].position = bodies[idx].position.sub(corr.scale(im_b));
                }
            }
        }
    }
}

impl Default for DpContactSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ── 6. DpSphParticle / DpFluidSim ──────────────────────────────────────────

/// SPH kernel type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpSphKernel {
    CubicSpline,
    WendlandC2,
}

/// Single SPH particle.
#[derive(Debug, Clone)]
pub struct DpSphParticle {
    pub position: DpVec3,
    pub velocity: DpVec3,
    pub density: f64,
    pub pressure: f64,
    pub mass: f64,
    pub force: DpVec3,
}

impl DpSphParticle {
    pub fn new(position: DpVec3, mass: f64) -> Self {
        Self {
            position,
            velocity: DpVec3::zero(),
            density: 0.0,
            pressure: 0.0,
            mass,
            force: DpVec3::zero(),
        }
    }
}

/// SPH fluid simulator.
pub struct DpFluidSim {
    pub h: f64,
    pub rho0: f64,
    pub stiffness_b: f64,
    pub gamma: f64,
    pub viscosity: f64,
    pub gravity: DpVec3,
    pub kernel: DpSphKernel,
}

impl DpFluidSim {
    pub fn new(h: f64) -> Result<Self> {
        if h <= 0.0 {
            return Err(dp_err(
                "DpFluidSim::new",
                "smoothing radius h must be positive",
            ));
        }
        Ok(Self {
            h,
            rho0: 1000.0,
            stiffness_b: 200.0,
            gamma: 7.0,
            viscosity: 0.01,
            gravity: DpVec3::new(0.0, -9.81, 0.0),
            kernel: DpSphKernel::CubicSpline,
        })
    }

    /// Evaluate SPH kernel W(r, h).
    pub fn kernel_w(&self, r: f64) -> f64 {
        let q = r / self.h;
        let h3 = self.h * self.h * self.h;
        match self.kernel {
            DpSphKernel::CubicSpline => {
                let sigma = 1.0 / (std::f64::consts::PI * h3);
                if q < 1.0 {
                    sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
                } else if q < 2.0 {
                    let t = 2.0 - q;
                    sigma * 0.25 * t * t * t
                } else {
                    0.0
                }
            }
            DpSphKernel::WendlandC2 => {
                let sigma = 21.0 / (2.0 * std::f64::consts::PI * h3);
                if q < 2.0 {
                    let t = 1.0 - q * 0.5;
                    sigma * t * t * t * t * (1.0 + 2.0 * q)
                } else {
                    0.0
                }
            }
        }
    }

    /// Gradient of SPH kernel (scalar dW/dr).
    pub fn kernel_grad_scalar(&self, r: f64) -> f64 {
        if r < 1e-15 {
            return 0.0;
        }
        let q = r / self.h;
        let h3 = self.h * self.h * self.h;
        match self.kernel {
            DpSphKernel::CubicSpline => {
                let sigma = 1.0 / (std::f64::consts::PI * h3);
                let dw_dq = if q < 1.0 {
                    sigma * (-3.0 * q + 2.25 * q * q)
                } else if q < 2.0 {
                    let t = 2.0 - q;
                    sigma * (-0.75 * t * t)
                } else {
                    0.0
                };
                dw_dq / self.h
            }
            DpSphKernel::WendlandC2 => {
                let sigma = 21.0 / (2.0 * std::f64::consts::PI * h3);
                if q < 2.0 {
                    let t = 1.0 - q * 0.5;
                    sigma * t * t * t * (-5.0 * q) / self.h
                } else {
                    0.0
                }
            }
        }
    }

    /// Laplacian of kernel (FD approximation for viscosity).
    pub fn kernel_laplacian(&self, r: f64) -> f64 {
        let eps = self.h * 0.001;
        let w_p = self.kernel_w(r + eps);
        let w_c = self.kernel_w(r);
        let w_m = self.kernel_w(if r > eps { r - eps } else { 0.0 });
        (w_p - 2.0 * w_c + w_m) / (eps * eps)
    }

    pub fn compute_densities(&self, particles: &mut [DpSphParticle]) {
        let n = particles.len();
        let pos: Vec<DpVec3> = particles.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        for i in 0..n {
            let mut rho = 0.0;
            for j in 0..n {
                rho += mass[j] * self.kernel_w(pos[i].sub(pos[j]).norm());
            }
            particles[i].density = rho.max(self.rho0 * 0.01);
        }
    }

    pub fn compute_pressures(&self, particles: &mut [DpSphParticle]) {
        for p in particles.iter_mut() {
            p.pressure =
                (self.stiffness_b * ((p.density / self.rho0).powf(self.gamma) - 1.0)).max(0.0);
        }
    }

    pub fn compute_forces(&self, particles: &mut [DpSphParticle]) {
        let n = particles.len();
        let pos: Vec<DpVec3> = particles.iter().map(|p| p.position).collect();
        let vel: Vec<DpVec3> = particles.iter().map(|p| p.velocity).collect();
        let den: Vec<f64> = particles.iter().map(|p| p.density).collect();
        let prs: Vec<f64> = particles.iter().map(|p| p.pressure).collect();
        let mas: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        for i in 0..n {
            let (mut fp, mut fv) = (DpVec3::zero(), DpVec3::zero());
            for j in 0..n {
                if i == j {
                    continue;
                }
                let rv = pos[i].sub(pos[j]);
                let r = rv.norm();
                if r < 1e-15 || r > 2.0 * self.h {
                    continue;
                }
                let dir = rv.scale(1.0 / r);
                let gw = self.kernel_grad_scalar(r);
                let pt = prs[i] / (den[i] * den[i]) + prs[j] / (den[j] * den[j]);
                fp = fp.sub(dir.scale(mas[j] * pt * gw));
                let lw = self.kernel_laplacian(r);
                fv = fv.add(
                    vel[j]
                        .sub(vel[i])
                        .scale(self.viscosity * mas[j] * lw / den[j].max(1e-10)),
                );
            }
            particles[i].force = fp.add(fv).add(self.gravity.scale(den[i]));
        }
    }

    pub fn step(&self, particles: &mut [DpSphParticle], dt: f64) -> Result<()> {
        if dt <= 0.0 {
            return Err(dp_err("DpFluidSim::step", "dt must be positive"));
        }
        if particles.is_empty() {
            return Ok(());
        }
        self.compute_densities(particles);
        self.compute_pressures(particles);
        self.compute_forces(particles);
        for p in particles.iter_mut() {
            let a = p.force.scale(1.0 / p.density.max(1e-10));
            p.velocity = p.velocity.add(a.scale(dt));
            p.position = p.position.add(p.velocity.scale(dt));
        }
        Ok(())
    }
}

// ── 7. DpClothSim ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpClothSpringType {
    Structural,
    Shear,
    Bend,
}

#[derive(Debug, Clone)]
pub struct DpClothParticle {
    pub position: DpVec3,
    pub prev_position: DpVec3,
    pub velocity: DpVec3,
    pub mass: f64,
    pub pinned: bool,
    pub force: DpVec3,
}

#[derive(Debug, Clone)]
pub struct DpClothSpring {
    pub p1: usize,
    pub p2: usize,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub spring_type: DpClothSpringType,
}

pub struct DpClothSim {
    pub gravity: DpVec3,
    pub max_stretch: f64,
    pub wind: DpVec3,
}

impl DpClothSim {
    pub fn new() -> Self {
        Self {
            gravity: DpVec3::new(0.0, -9.81, 0.0),
            max_stretch: 1.5,
            wind: DpVec3::zero(),
        }
    }

    /// Create a rectangular cloth grid in the XZ plane. Returns (particles, springs).
    pub fn create_cloth(
        &self,
        width: usize,
        height: usize,
        spacing: f64,
        stiffness: f64,
        damping: f64,
    ) -> Result<(Vec<DpClothParticle>, Vec<DpClothSpring>)> {
        if width < 2 || height < 2 {
            return Err(dp_err(
                "DpClothSim::create_cloth",
                "width and height must be >= 2",
            ));
        }
        if spacing <= 0.0 {
            return Err(dp_err(
                "DpClothSim::create_cloth",
                "spacing must be positive",
            ));
        }
        let mass = 1.0 / (width * height) as f64;
        let mut particles = Vec::with_capacity(width * height);
        for j in 0..height {
            for i in 0..width {
                let pos = DpVec3::new(i as f64 * spacing, 5.0, j as f64 * spacing);
                particles.push(DpClothParticle {
                    position: pos,
                    prev_position: pos,
                    velocity: DpVec3::zero(),
                    mass,
                    pinned: false,
                    force: DpVec3::zero(),
                });
            }
        }
        let mut springs = Vec::new();
        let idx = |i: usize, j: usize| j * width + i;
        // Structural
        for j in 0..height {
            for i in 0..width {
                if i + 1 < width {
                    springs.push(DpClothSpring {
                        p1: idx(i, j),
                        p2: idx(i + 1, j),
                        rest_length: spacing,
                        stiffness,
                        damping,
                        spring_type: DpClothSpringType::Structural,
                    });
                }
                if j + 1 < height {
                    springs.push(DpClothSpring {
                        p1: idx(i, j),
                        p2: idx(i, j + 1),
                        rest_length: spacing,
                        stiffness,
                        damping,
                        spring_type: DpClothSpringType::Structural,
                    });
                }
            }
        }
        // Shear
        let diag = spacing * std::f64::consts::SQRT_2;
        for j in 0..(height - 1) {
            for i in 0..(width - 1) {
                springs.push(DpClothSpring {
                    p1: idx(i, j),
                    p2: idx(i + 1, j + 1),
                    rest_length: diag,
                    stiffness: stiffness * 0.5,
                    damping: damping * 0.5,
                    spring_type: DpClothSpringType::Shear,
                });
                springs.push(DpClothSpring {
                    p1: idx(i + 1, j),
                    p2: idx(i, j + 1),
                    rest_length: diag,
                    stiffness: stiffness * 0.5,
                    damping: damping * 0.5,
                    spring_type: DpClothSpringType::Shear,
                });
            }
        }
        // Bend
        let bend = spacing * 2.0;
        for j in 0..height {
            for i in 0..width {
                if i + 2 < width {
                    springs.push(DpClothSpring {
                        p1: idx(i, j),
                        p2: idx(i + 2, j),
                        rest_length: bend,
                        stiffness: stiffness * 0.25,
                        damping: damping * 0.25,
                        spring_type: DpClothSpringType::Bend,
                    });
                }
                if j + 2 < height {
                    springs.push(DpClothSpring {
                        p1: idx(i, j),
                        p2: idx(i, j + 2),
                        rest_length: bend,
                        stiffness: stiffness * 0.25,
                        damping: damping * 0.25,
                        spring_type: DpClothSpringType::Bend,
                    });
                }
            }
        }
        Ok((particles, springs))
    }

    fn compute_spring_forces(&self, particles: &mut [DpClothParticle], springs: &[DpClothSpring]) {
        for p in particles.iter_mut() {
            p.force = if p.pinned {
                DpVec3::zero()
            } else {
                self.gravity.scale(p.mass).add(self.wind.scale(p.mass))
            };
        }
        for s in springs {
            let diff = particles[s.p2].position.sub(particles[s.p1].position);
            let dist = diff.norm();
            if dist < 1e-15 {
                continue;
            }
            let dir = diff.scale(1.0 / dist);
            let f_sp = dir.scale(s.stiffness * (dist - s.rest_length));
            let f_dm = dir.scale(
                s.damping
                    * particles[s.p2]
                        .velocity
                        .sub(particles[s.p1].velocity)
                        .dot(dir),
            );
            let total = f_sp.add(f_dm);
            if !particles[s.p1].pinned {
                particles[s.p1].force = particles[s.p1].force.add(total);
            }
            if !particles[s.p2].pinned {
                particles[s.p2].force = particles[s.p2].force.sub(total);
            }
        }
    }

    fn apply_strain_limiting(&self, particles: &mut [DpClothParticle], springs: &[DpClothSpring]) {
        for s in springs {
            let diff = particles[s.p2].position.sub(particles[s.p1].position);
            let dist = diff.norm();
            let max_len = s.rest_length * self.max_stretch;
            if dist > max_len && dist > 1e-15 {
                let corr = diff.scale((dist - max_len) / dist);
                match (particles[s.p1].pinned, particles[s.p2].pinned) {
                    (false, false) => {
                        let h = corr.scale(0.5);
                        particles[s.p1].position = particles[s.p1].position.add(h);
                        particles[s.p2].position = particles[s.p2].position.sub(h);
                    }
                    (true, false) => {
                        particles[s.p2].position = particles[s.p2].position.sub(corr);
                    }
                    (false, true) => {
                        particles[s.p1].position = particles[s.p1].position.add(corr);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn step(
        &self,
        particles: &mut [DpClothParticle],
        springs: &[DpClothSpring],
        dt: f64,
    ) -> Result<()> {
        if dt <= 0.0 {
            return Err(dp_err("DpClothSim::step", "dt must be positive"));
        }
        self.compute_spring_forces(particles, springs);
        for p in particles.iter_mut() {
            if p.pinned {
                continue;
            }
            let a = p.force.scale(1.0 / p.mass);
            p.velocity = p.velocity.add(a.scale(dt));
            p.position = p.position.add(p.velocity.scale(dt));
        }
        self.apply_strain_limiting(particles, springs);
        Ok(())
    }
}

impl Default for DpClothSim {
    fn default() -> Self {
        Self::new()
    }
}

// ── 8. DpAdjointMethod ─────────────────────────────────────────────────────

fn flatten_state(bodies: &[DpRigidBody]) -> Vec<f64> {
    let mut s = Vec::with_capacity(bodies.len() * 6);
    for b in bodies {
        s.extend_from_slice(&[
            b.position.x,
            b.position.y,
            b.position.z,
            b.velocity.x,
            b.velocity.y,
            b.velocity.z,
        ]);
    }
    s
}

fn unflatten_state(state: &[f64], bodies: &mut [DpRigidBody]) -> Result<()> {
    if state.len() != bodies.len() * 6 {
        return Err(dp_err(
            "unflatten_state",
            &format!("expected {} values, got {}", bodies.len() * 6, state.len()),
        ));
    }
    for (i, b) in bodies.iter_mut().enumerate() {
        let o = i * 6;
        b.position = DpVec3::new(state[o], state[o + 1], state[o + 2]);
        b.velocity = DpVec3::new(state[o + 3], state[o + 4], state[o + 5]);
    }
    Ok(())
}

/// Adjoint method for computing gradients through a physics simulation.
pub struct DpAdjointMethod {
    pub fd_epsilon: f64,
}

impl DpAdjointMethod {
    pub fn new() -> Self {
        Self { fd_epsilon: 1e-5 }
    }

    /// Compute the Jacobian of one physics step via finite differences.
    pub fn compute_step_jacobian(
        &self,
        engine: &DpPhysicsEngine,
        bodies: &[DpRigidBody],
        dt: f64,
        integrator: DpIntegrator,
    ) -> Result<Vec<Vec<f64>>> {
        let state = flatten_state(bodies);
        let n = state.len();
        let mut ref_b = bodies.to_vec();
        engine.step(&mut ref_b, dt, integrator)?;
        let ref_next = flatten_state(&ref_b);
        let mut jac = vec![vec![0.0f64; n]; n];
        for j in 0..n {
            let mut ps = state.clone();
            ps[j] += self.fd_epsilon;
            let mut pb = bodies.to_vec();
            unflatten_state(&ps, &mut pb)?;
            engine.step(&mut pb, dt, integrator)?;
            let pn = flatten_state(&pb);
            for i in 0..n {
                jac[i][j] = (pn[i] - ref_next[i]) / self.fd_epsilon;
            }
        }
        Ok(jac)
    }

    /// Compute gradient of a scalar loss w.r.t. initial state via adjoint method.
    pub fn compute_gradient(
        &self,
        engine: &DpPhysicsEngine,
        initial_bodies: &[DpRigidBody],
        dt: f64,
        n_steps: usize,
        integrator: DpIntegrator,
        loss_grad_final: &[f64],
    ) -> Result<Vec<f64>> {
        let ns = initial_bodies.len() * 6;
        if loss_grad_final.len() != ns {
            return Err(dp_err(
                "DpAdjointMethod::compute_gradient",
                &format!(
                    "loss_grad dimension mismatch: expected {}, got {}",
                    ns,
                    loss_grad_final.len()
                ),
            ));
        }
        // Forward: record trajectory
        let mut traj: Vec<Vec<DpRigidBody>> = Vec::with_capacity(n_steps + 1);
        let mut bodies = initial_bodies.to_vec();
        traj.push(bodies.clone());
        for _ in 0..n_steps {
            engine.step(&mut bodies, dt, integrator)?;
            traj.push(bodies.clone());
        }
        // Backward: propagate adjoint through Jacobian chain
        let mut adj = loss_grad_final.to_vec();
        for t in (0..n_steps).rev() {
            let jac = self.compute_step_jacobian(engine, &traj[t], dt, integrator)?;
            let mut new_adj = vec![0.0f64; ns];
            for i in 0..ns {
                for j in 0..ns {
                    new_adj[i] += jac[j][i] * adj[j];
                }
            }
            adj = new_adj;
        }
        Ok(adj)
    }

    /// Validate adjoint gradient against full finite-difference gradient. Returns max abs diff.
    pub fn validate_gradient(
        &self,
        engine: &DpPhysicsEngine,
        initial_bodies: &[DpRigidBody],
        dt: f64,
        n_steps: usize,
        integrator: DpIntegrator,
        loss_fn: &dyn Fn(&[DpRigidBody]) -> f64,
    ) -> Result<f64> {
        let ns = initial_bodies.len() * 6;
        let mut final_b = initial_bodies.to_vec();
        for _ in 0..n_steps {
            engine.step(&mut final_b, dt, integrator)?;
        }
        let ref_loss = loss_fn(&final_b);
        let fs = flatten_state(&final_b);
        let mut lg = vec![0.0f64; ns];
        for i in 0..ns {
            let mut ps = fs.clone();
            ps[i] += self.fd_epsilon;
            let mut pb = final_b.clone();
            unflatten_state(&ps, &mut pb)?;
            lg[i] = (loss_fn(&pb) - ref_loss) / self.fd_epsilon;
        }
        let adj_grad =
            self.compute_gradient(engine, initial_bodies, dt, n_steps, integrator, &lg)?;
        let is = flatten_state(initial_bodies);
        let mut fd_grad = vec![0.0f64; ns];
        for i in 0..ns {
            let mut pi = is.clone();
            pi[i] += self.fd_epsilon;
            let mut pb = initial_bodies.to_vec();
            unflatten_state(&pi, &mut pb)?;
            for _ in 0..n_steps {
                engine.step(&mut pb, dt, integrator)?;
            }
            fd_grad[i] = (loss_fn(&pb) - ref_loss) / self.fd_epsilon;
        }
        Ok(adj_grad
            .iter()
            .zip(fd_grad.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max))
    }
}

impl Default for DpAdjointMethod {
    fn default() -> Self {
        Self::new()
    }
}

// ── 9. DpPhysicsLoss ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpLossType {
    TrajectoryMse,
    EnergyConservation,
    MomentumConservation,
    ContactViolation,
}

/// Physics-informed loss functions.
pub struct DpPhysicsLoss {
    pub traj_weight: f64,
    pub energy_weight: f64,
    pub momentum_weight: f64,
    pub contact_weight: f64,
    pub gravity: DpVec3,
}

impl DpPhysicsLoss {
    pub fn new(gravity: DpVec3) -> Self {
        Self {
            traj_weight: 1.0,
            energy_weight: 0.1,
            momentum_weight: 0.1,
            contact_weight: 0.5,
            gravity,
        }
    }

    /// Trajectory MSE between predicted and target snapshots.
    pub fn trajectory_mse(
        &self,
        predicted: &[Vec<DpBodySnapshot>],
        target: &[Vec<DpBodySnapshot>],
    ) -> Result<f64> {
        if predicted.len() != target.len() {
            return Err(dp_err(
                "DpPhysicsLoss::trajectory_mse",
                "trajectory lengths differ",
            ));
        }
        if predicted.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for (ps, ts) in predicted.iter().zip(target.iter()) {
            if ps.len() != ts.len() {
                return Err(dp_err(
                    "DpPhysicsLoss::trajectory_mse",
                    "body counts differ at some timestep",
                ));
            }
            for (p, t) in ps.iter().zip(ts.iter()) {
                total +=
                    p.position.sub(t.position).norm_sq() + p.velocity.sub(t.velocity).norm_sq();
                count += 2;
            }
        }
        Ok(total / count.max(1) as f64)
    }

    /// Energy conservation loss: penalizes deviation from initial energy.
    pub fn energy_conservation_loss(&self, bt: &[Vec<DpRigidBody>]) -> f64 {
        if bt.len() < 2 {
            return 0.0;
        }
        let e0: f64 = bt[0].iter().map(|b| b.total_energy(self.gravity)).sum();
        let mut pen = 0.0;
        for step in bt.iter().skip(1) {
            let e: f64 = step.iter().map(|b| b.total_energy(self.gravity)).sum();
            let d = e - e0;
            pen += d * d;
        }
        pen / (bt.len() - 1) as f64
    }

    /// Momentum conservation loss.
    pub fn momentum_conservation_loss(&self, bt: &[Vec<DpRigidBody>]) -> f64 {
        if bt.len() < 2 {
            return 0.0;
        }
        let p0 = bt[0]
            .iter()
            .fold(DpVec3::zero(), |a, b| a.add(b.momentum()));
        let mut pen = 0.0;
        for step in bt.iter().skip(1) {
            let p = step.iter().fold(DpVec3::zero(), |a, b| a.add(b.momentum()));
            pen += p.sub(p0).norm_sq();
        }
        pen / (bt.len() - 1) as f64
    }

    /// Contact violation loss.
    pub fn contact_violation_loss(&self, _bodies: &[DpRigidBody], contacts: &[DpContact]) -> f64 {
        contacts
            .iter()
            .map(|c| c.penetration.max(0.0).powi(2))
            .sum::<f64>()
    }

    /// Combined loss.
    pub fn compute_loss(
        &self,
        predicted: &[Vec<DpBodySnapshot>],
        target: &[Vec<DpBodySnapshot>],
        bt: &[Vec<DpRigidBody>],
        contacts: &[DpContact],
        bodies: &[DpRigidBody],
    ) -> Result<f64> {
        Ok(self.traj_weight * self.trajectory_mse(predicted, target)?
            + self.energy_weight * self.energy_conservation_loss(bt)
            + self.momentum_weight * self.momentum_conservation_loss(bt)
            + self.contact_weight * self.contact_violation_loss(bodies, contacts))
    }
}

// ── 10. DpMetrics / DpReport ────────────────────────────────────────────────

pub struct DpMetrics;

impl DpMetrics {
    pub fn energy_drift(init: &[DpRigidBody], fin: &[DpRigidBody], g: DpVec3) -> f64 {
        let ei: f64 = init.iter().map(|b| b.total_energy(g)).sum();
        let ef: f64 = fin.iter().map(|b| b.total_energy(g)).sum();
        (ef - ei).abs() / ei.abs().max(1e-15)
    }

    pub fn trajectory_rmse(
        pred: &[Vec<DpBodySnapshot>],
        tgt: &[Vec<DpBodySnapshot>],
    ) -> Result<f64> {
        if pred.len() != tgt.len() {
            return Err(dp_err(
                "DpMetrics::trajectory_rmse",
                "trajectory lengths differ",
            ));
        }
        if pred.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for (ps, ts) in pred.iter().zip(tgt.iter()) {
            for (p, t) in ps.iter().zip(ts.iter()) {
                total += p.position.sub(t.position).norm_sq();
                count += 1;
            }
        }
        Ok((total / count.max(1) as f64).sqrt())
    }

    pub fn max_penetration(contacts: &[DpContact]) -> f64 {
        contacts
            .iter()
            .map(|c| c.penetration.max(0.0))
            .fold(0.0f64, f64::max)
    }

    pub fn mean_penetration(contacts: &[DpContact]) -> f64 {
        if contacts.is_empty() {
            return 0.0;
        }
        contacts.iter().map(|c| c.penetration.max(0.0)).sum::<f64>() / contacts.len() as f64
    }

    pub fn max_velocity(bodies: &[DpRigidBody]) -> f64 {
        bodies
            .iter()
            .map(|b| b.velocity.norm())
            .fold(0.0f64, f64::max)
    }

    pub fn max_acceleration(bodies: &[DpRigidBody]) -> f64 {
        bodies
            .iter()
            .filter(|b| !b.is_static)
            .map(|b| b.force.scale(b.inv_mass()).norm())
            .fold(0.0f64, f64::max)
    }

    pub fn total_kinetic_energy(bodies: &[DpRigidBody]) -> f64 {
        bodies.iter().map(|b| b.kinetic_energy()).sum()
    }

    pub fn total_momentum_magnitude(bodies: &[DpRigidBody]) -> f64 {
        bodies
            .iter()
            .fold(DpVec3::zero(), |a, b| a.add(b.momentum()))
            .norm()
    }

    pub fn report(
        init: &[DpRigidBody],
        fin: &[DpRigidBody],
        g: DpVec3,
        contacts: &[DpContact],
    ) -> DpReport {
        DpReport {
            energy_drift: Self::energy_drift(init, fin, g),
            max_velocity: Self::max_velocity(fin),
            max_acceleration: Self::max_acceleration(fin),
            max_penetration: Self::max_penetration(contacts),
            mean_penetration: Self::mean_penetration(contacts),
            total_kinetic_energy: Self::total_kinetic_energy(fin),
            total_momentum_magnitude: Self::total_momentum_magnitude(fin),
            num_contacts: contacts.len(),
            num_bodies: fin.len(),
        }
    }
}

/// Comprehensive physics simulation report.
#[derive(Debug, Clone)]
pub struct DpReport {
    pub energy_drift: f64,
    pub max_velocity: f64,
    pub max_acceleration: f64,
    pub max_penetration: f64,
    pub mean_penetration: f64,
    pub total_kinetic_energy: f64,
    pub total_momentum_magnitude: f64,
    pub num_contacts: usize,
    pub num_bodies: usize,
}

impl std::fmt::Display for DpReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Differentiable Physics Report ===")?;
        writeln!(f, "Bodies:            {}", self.num_bodies)?;
        writeln!(f, "Contacts:          {}", self.num_contacts)?;
        writeln!(f, "Energy drift:      {:.6e}", self.energy_drift)?;
        writeln!(f, "Max velocity:      {:.4}", self.max_velocity)?;
        writeln!(f, "Max acceleration:  {:.4}", self.max_acceleration)?;
        writeln!(f, "Max penetration:   {:.6e}", self.max_penetration)?;
        writeln!(f, "Mean penetration:  {:.6e}", self.mean_penetration)?;
        writeln!(f, "Kinetic energy:    {:.4}", self.total_kinetic_energy)?;
        writeln!(
            f,
            "Momentum mag:      {:.6e}",
            self.total_momentum_magnitude
        )?;
        Ok(())
    }
}
