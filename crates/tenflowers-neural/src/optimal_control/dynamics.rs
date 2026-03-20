//! DynamicsModel trait and built-in discrete-time dynamical systems.

use super::utils::{
    mat_add, mat_identity, mat_mul, mat_scale, mat_transpose, mat_vec_mul, vec_add,
};

// DynamicsModel Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Core trait for discrete-time dynamical systems `x_{t+1} = f(x_t, u_t)`.
///
/// Implementors must provide `step` and dimension queries. The `linearize`
/// method has a default implementation via central finite differences (h = 1e-5)
/// but may be overridden for analytical Jacobians.
pub trait DynamicsModel {
    /// Dimensionality of the state vector.
    fn state_dim(&self) -> usize;

    /// Dimensionality of the action / control vector.
    fn action_dim(&self) -> usize;

    /// Advance the state one time step: `x_{t+1} = f(x_t, u_t)`.
    fn step(&self, state: &[f64], action: &[f64]) -> Vec<f64>;

    /// Linearize dynamics around `(state, action)` via central finite differences.
    ///
    /// Returns `(A, B)` where:
    /// - `A` is `state_dim × state_dim` (∂f/∂x)
    /// - `B` is `state_dim × action_dim` (∂f/∂u)
    fn linearize(&self, state: &[f64], action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.state_dim();
        let m = self.action_dim();
        let h = 1e-5;

        // Compute A = ∂f/∂x via central differences
        let mut a = vec![vec![0.0f64; n]; n];
        for j in 0..n {
            let mut xp = state.to_vec();
            let mut xm = state.to_vec();
            xp[j] += h;
            xm[j] -= h;
            let fp = self.step(&xp, action);
            let fm = self.step(&xm, action);
            for i in 0..n {
                a[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }

        // Compute B = ∂f/∂u via central differences
        let mut b = vec![vec![0.0f64; m]; n];
        for j in 0..m {
            let mut up = action.to_vec();
            let mut um = action.to_vec();
            up[j] += h;
            um[j] -= h;
            let fp = self.step(state, &up);
            let fm = self.step(state, &um);
            for i in 0..n {
                b[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }

        (a, b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Dynamics Models
// ─────────────────────────────────────────────────────────────────────────────

/// Linear discrete-time dynamics: `x_{t+1} = A x_t + B u_t`.
pub struct LinearDynamics {
    /// State transition matrix (state_dim × state_dim).
    pub a: Vec<Vec<f64>>,
    /// Control input matrix (state_dim × action_dim).
    pub b: Vec<Vec<f64>>,
}

impl DynamicsModel for LinearDynamics {
    fn state_dim(&self) -> usize {
        self.a.len()
    }

    fn action_dim(&self) -> usize {
        if self.b.is_empty() {
            0
        } else {
            self.b[0].len()
        }
    }

    fn step(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        let ax = mat_vec_mul(&self.a, state);
        let bu = mat_vec_mul(&self.b, action);
        vec_add(&ax, &bu)
    }

    fn linearize(&self, _state: &[f64], _action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        (self.a.clone(), self.b.clone())
    }
}

/// Double integrator in 2D: state = [position, velocity], action = \[acceleration\].
///
/// Euler integration: `x_{t+1} = x_t + v_t * dt`, `v_{t+1} = v_t + a_t * dt`.
pub struct DoubleIntegrator {
    /// Time step size (seconds).
    pub dt: f64,
}

impl DynamicsModel for DoubleIntegrator {
    fn state_dim(&self) -> usize {
        2
    }
    fn action_dim(&self) -> usize {
        1
    }

    fn step(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        let pos = state[0] + state[1] * self.dt;
        let vel = state[1] + action[0] * self.dt;
        vec![pos, vel]
    }

    fn linearize(&self, _state: &[f64], _action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let dt = self.dt;
        // A = [[1, dt], [0, 1]], B = [[0], [dt]]
        let a = vec![vec![1.0, dt], vec![0.0, 1.0]];
        let b = vec![vec![0.0], vec![dt]];
        (a, b)
    }
}

/// CartPole dynamics linearized around the upright equilibrium.
///
/// State: [cart_position, pole_angle, cart_velocity, pole_angular_velocity]
/// Action: [horizontal force on cart]
pub struct CartPole {
    /// Simulation time step (seconds).
    pub dt: f64,
    /// Half-length of the pole (meters). Full length = 2 * pole_length.
    pub pole_length: f64,
    /// Mass of the cart (kg).
    pub mass_cart: f64,
    /// Mass of the pole tip (kg).
    pub mass_pole: f64,
}

impl CartPole {
    /// Linearize the CartPole dynamics analytically around the upright equilibrium
    /// [0, 0, 0, 0] with zero force.
    ///
    /// Returns (A_d, B_d): the discrete-time (Euler-discretized) linearization.
    pub fn linearize_at_equilibrium(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let g = 9.81_f64;
        let l = self.pole_length;
        let mc = self.mass_cart;
        let mp = self.mass_pole;
        let total_mass = mc + mp;
        let dt = self.dt;

        // Continuous-time A and B around upright equilibrium (theta ≈ 0, cos≈1, sin≈0)
        //
        // Simplified linearization of the full CartPole EOM:
        //   x_ddot  = (F + mp*l*theta_ddot) / total_mass  (approx)
        //   theta_ddot = (g/l)*theta + F/(total_mass*l)   (approx)
        //
        // Full linearization:
        let alpha = mp * l / total_mass;
        let denom = l - mp * l * alpha; // ≈ l*(1 - mp/total) = l*mc/total

        // ∂(theta_ddot)/∂theta in continuous time
        let a_theta_theta = g / denom;
        // ∂(theta_ddot)/∂F in continuous time
        let b_theta_f = -1.0 / (total_mass * denom);
        // ∂(x_ddot)/∂theta in continuous time
        let a_x_theta = -mp * l * g / (total_mass * denom);
        // ∂(x_ddot)/∂F in continuous time
        let b_x_f = 1.0 / total_mass + mp * l / (total_mass * total_mass * denom);

        // Continuous A_c (4×4): state = [x, theta, xdot, thetadot]
        // x_dot    = xdot
        // theta_dot= thetadot
        // xddot    = a_x_theta * theta + b_x_f * F
        // tddot    = a_theta_theta * theta + b_theta_f * F
        let a_c = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, a_x_theta, 0.0, 0.0],
            vec![0.0, a_theta_theta, 0.0, 0.0],
        ];
        let b_c = vec![vec![0.0], vec![0.0], vec![b_x_f], vec![b_theta_f]];

        // Euler discretization: A_d = I + dt * A_c, B_d = dt * B_c
        let i4 = mat_identity(4);
        let a_d = mat_add(&i4, &mat_scale(&a_c, dt));
        let b_d = mat_scale(&b_c, dt);
        (a_d, b_d)
    }
}

impl DynamicsModel for CartPole {
    fn state_dim(&self) -> usize {
        4
    }
    fn action_dim(&self) -> usize {
        1
    }

    fn step(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        // Nonlinear CartPole Euler step (simplified formulation)
        let g = 9.81_f64;
        let l = self.pole_length;
        let mc = self.mass_cart;
        let mp = self.mass_pole;
        let dt = self.dt;
        let total_mass = mc + mp;
        let f = action[0];

        let x = state[0];
        let theta = state[1];
        let xdot = state[2];
        let thetadot = state[3];

        let sin_t = theta.sin();
        let cos_t = theta.cos();

        // Equations of motion (standard CartPole)
        let temp = (f + mp * l * thetadot * thetadot * sin_t) / total_mass;
        let theta_acc =
            (g * sin_t - cos_t * temp) / (l * (4.0 / 3.0 - mp * cos_t * cos_t / total_mass));
        let x_acc = temp - mp * l * theta_acc * cos_t / total_mass;

        vec![
            x + dt * xdot,
            theta + dt * thetadot,
            xdot + dt * x_acc,
            thetadot + dt * theta_acc,
        ]
    }

    fn linearize(&self, _state: &[f64], _action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        self.linearize_at_equilibrium()
    }
}

/// Nonlinear pendulum dynamics:
/// `theta_{t+1} = theta_t + omega_t * dt`
/// `omega_{t+1} = omega_t + (-g/l * sin(theta_t) - damping * omega_t + torque_t) * dt`
///
/// State: [theta, omega], Action: \[torque\]
pub struct PendulumDynamics {
    /// Time step (seconds).
    pub dt: f64,
    /// Gravitational acceleration (m/s²).
    pub gravity: f64,
    /// Pendulum length (meters).
    pub length: f64,
    /// Damping coefficient (Nms/rad).
    pub damping: f64,
}

impl DynamicsModel for PendulumDynamics {
    fn state_dim(&self) -> usize {
        2
    }
    fn action_dim(&self) -> usize {
        1
    }

    fn step(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        let theta = state[0];
        let omega = state[1];
        let torque = action[0];
        let alpha = -self.gravity / self.length * theta.sin() - self.damping * omega + torque;
        vec![theta + self.dt * omega, omega + self.dt * alpha]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
