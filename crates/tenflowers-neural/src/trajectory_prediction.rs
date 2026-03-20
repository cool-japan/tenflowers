//! Trajectory Prediction Models
//!
//! Implements trajectory prediction for vehicles, pedestrians, and robots:
//! - [`Trajectory`]: Core trajectory data structures
//! - [`SocialForceModel`]: Physics-based pedestrian prediction (Helbing SFM)
//! - [`LstmTrajPredictor`]: LSTM-based trajectory prediction
//! - [`SocialLstm`]: Social force augmented LSTM (occupancy-grid pooling)
//! - [`TransformerTrajPredictor`]: Attention-based trajectory prediction
//! - [`GmmDecoder`]: Bivariate Gaussian mixture probabilistic decoder
//! - [`TrajMetrics`]: Standard evaluation metrics (ADE, FDE, miss rate)

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// §1  Core Trajectory Data Structures
// ---------------------------------------------------------------------------

/// 2-D position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

impl Point2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Point2D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// 2-D velocity vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity2D {
    pub vx: f32,
    pub vy: f32,
}

impl Velocity2D {
    pub fn new(vx: f32, vy: f32) -> Self {
        Self { vx, vy }
    }

    /// Speed (magnitude of velocity vector).
    pub fn speed(&self) -> f32 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }
}

/// Full state of a single agent at one timestep.
#[derive(Debug, Clone)]
pub struct AgentState {
    pub position: Point2D,
    pub velocity: Velocity2D,
    /// Heading angle in radians.
    pub heading: f32,
    pub agent_id: u32,
}

impl AgentState {
    pub fn new(position: Point2D, velocity: Velocity2D, heading: f32, agent_id: u32) -> Self {
        Self {
            position,
            velocity,
            heading,
            agent_id,
        }
    }
}

/// A sequence of agent states sampled at a fixed timestep.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub states: Vec<AgentState>,
    /// Time between consecutive states (seconds).
    pub dt: f32,
}

impl Trajectory {
    pub fn new(states: Vec<AgentState>, dt: f32) -> Self {
        Self { states, dt }
    }

    /// Total arc length in metres.
    pub fn length(&self) -> f32 {
        if self.states.len() < 2 {
            return 0.0;
        }
        self.states
            .windows(2)
            .map(|w| w[0].position.distance_to(&w[1].position))
            .sum()
    }

    /// Straight-line displacement from first to last state.
    pub fn displacement(&self) -> f32 {
        if self.states.len() < 2 {
            return 0.0;
        }
        let first = &self.states[0].position;
        let last = &self.states[self.states.len() - 1].position;
        first.distance_to(last)
    }

    /// Mean speed over the trajectory.
    pub fn average_speed(&self) -> f32 {
        if self.states.is_empty() {
            return 0.0;
        }
        let total_speed: f32 = self.states.iter().map(|s| s.velocity.speed()).sum();
        total_speed / self.states.len() as f32
    }

    /// Signed curvature at index `i` (finite-difference approximation).
    /// Returns 0.0 at boundary indices.
    pub fn curvature_at(&self, i: usize) -> f32 {
        if i == 0 || i + 1 >= self.states.len() {
            return 0.0;
        }
        let prev = &self.states[i - 1].position;
        let curr = &self.states[i].position;
        let next = &self.states[i + 1].position;
        let dx1 = curr.x - prev.x;
        let dy1 = curr.y - prev.y;
        let dx2 = next.x - curr.x;
        let dy2 = next.y - curr.y;
        let cross = dx1 * dy2 - dy1 * dx2;
        let len1 = (dx1 * dx1 + dy1 * dy1).sqrt().max(1e-9);
        let len2 = (dx2 * dx2 + dy2 * dy2).sqrt().max(1e-9);
        cross / (len1 * len2)
    }
}

/// Scene context: multiple agent trajectories plus static map polygons.
#[derive(Debug, Clone)]
pub struct SceneContext {
    pub agents: Vec<Trajectory>,
    pub map_polygons: Vec<Vec<Point2D>>,
}

impl SceneContext {
    pub fn new(agents: Vec<Trajectory>, map_polygons: Vec<Vec<Point2D>>) -> Self {
        Self {
            agents,
            map_polygons,
        }
    }
}

// ---------------------------------------------------------------------------
// §2  Social Force Model
// ---------------------------------------------------------------------------

/// Configuration for the Social Force Model.
#[derive(Debug, Clone)]
pub struct SfmConfig {
    /// Desired walking speed (m/s).
    pub desired_speed: f32,
    /// Relaxation time τ (seconds).
    pub relaxation_time: f32,
    /// Repulsion strength A.
    pub repulsion_strength: f32,
    /// Repulsion range B (metres).
    pub repulsion_range: f32,
    /// Wall repulsion strength.
    pub wall_strength: f32,
}

impl Default for SfmConfig {
    fn default() -> Self {
        Self {
            desired_speed: 1.34,
            relaxation_time: 0.5,
            repulsion_strength: 2.0,
            repulsion_range: 0.3,
            wall_strength: 10.0,
        }
    }
}

/// Helbing Social Force Model for pedestrian trajectory prediction.
pub struct SocialForceModel {
    pub config: SfmConfig,
}

impl SocialForceModel {
    pub fn new(config: SfmConfig) -> Self {
        Self { config }
    }

    /// Self-propulsion acceleration: a_i = (v0 * e_i - v_i) / τ
    pub fn compute_self_force(&self, state: &AgentState, goal: Point2D) -> Velocity2D {
        let dx = goal.x - state.position.x;
        let dy = goal.y - state.position.y;
        let dist = (dx * dx + dy * dy).sqrt().max(1e-9);
        // Unit direction toward goal
        let ex = dx / dist;
        let ey = dy / dist;
        let ax = (self.config.desired_speed * ex - state.velocity.vx) / self.config.relaxation_time;
        let ay = (self.config.desired_speed * ey - state.velocity.vy) / self.config.relaxation_time;
        Velocity2D::new(ax, ay)
    }

    /// Pairwise exponential repulsion: A * exp((r_ij - d_ij) / B) * n_ij
    pub fn compute_repulsion(&self, agent: &AgentState, others: &[AgentState]) -> Velocity2D {
        let mut fx = 0.0_f32;
        let mut fy = 0.0_f32;
        let r_ij = 0.4_f32; // sum of agent radii (fixed at 0.2m each)

        for other in others {
            if other.agent_id == agent.agent_id {
                continue;
            }
            let dx = agent.position.x - other.position.x;
            let dy = agent.position.y - other.position.y;
            let d_ij = (dx * dx + dy * dy).sqrt().max(1e-9);
            let magnitude = self.config.repulsion_strength
                * ((r_ij - d_ij) / self.config.repulsion_range).exp();
            fx += magnitude * dx / d_ij;
            fy += magnitude * dy / d_ij;
        }
        Velocity2D::new(fx, fy)
    }

    /// Advance all agents by one timestep using Euler integration.
    pub fn step(&self, scene: &SceneContext, goals: &[Point2D], dt: f32) -> Vec<AgentState> {
        let current: Vec<AgentState> = scene
            .agents
            .iter()
            .filter_map(|t| t.states.last().cloned())
            .collect();

        current
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let goal = goals.get(i).copied().unwrap_or(state.position);
                let self_force = self.compute_self_force(state, goal);
                let repulsion = self.compute_repulsion(state, &current);

                let ax = self_force.vx + repulsion.vx;
                let ay = self_force.vy + repulsion.vy;

                let new_vx = state.velocity.vx + ax * dt;
                let new_vy = state.velocity.vy + ay * dt;
                let new_x = state.position.x + new_vx * dt;
                let new_y = state.position.y + new_vy * dt;
                let new_heading = new_vy.atan2(new_vx);

                AgentState::new(
                    Point2D::new(new_x, new_y),
                    Velocity2D::new(new_vx, new_vy),
                    new_heading,
                    state.agent_id,
                )
            })
            .collect()
    }

    /// Predict trajectories for `steps` timesteps starting from `initial` states.
    pub fn predict(
        &self,
        initial: &[AgentState],
        goals: &[Point2D],
        steps: usize,
        dt: f32,
    ) -> Vec<Trajectory> {
        // Build a synthetic SceneContext from the current states
        let mut current_states: Vec<AgentState> = initial.to_vec();
        let mut all_states: Vec<Vec<AgentState>> = vec![Vec::new(); initial.len()];

        // Record initial positions
        for (i, state) in current_states.iter().enumerate() {
            all_states[i].push(state.clone());
        }

        for _ in 0..steps {
            // Build scene from current_states (each as a 1-step trajectory)
            let agent_trajs: Vec<Trajectory> = current_states
                .iter()
                .map(|s| Trajectory::new(vec![s.clone()], dt))
                .collect();
            let scene = SceneContext::new(agent_trajs, vec![]);
            let next = self.step(&scene, goals, dt);
            for (i, state) in next.iter().enumerate() {
                all_states[i].push(state.clone());
            }
            current_states = next;
        }

        all_states
            .into_iter()
            .map(|states| Trajectory::new(states, dt))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// §3  LSTM-based Trajectory Predictor
// ---------------------------------------------------------------------------

/// Configuration for LSTM trajectory predictor.
#[derive(Debug, Clone)]
pub struct LstmConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub n_layers: usize,
}

/// Single LSTM cell with 4-gate formulation.
#[derive(Debug, Clone)]
pub struct TrajLstmCell {
    /// Weight matrices for forget, input, cell, output gates: shape [hidden x (hidden+input)]
    pub wf: Vec<Vec<f32>>,
    pub wi: Vec<Vec<f32>>,
    pub wg: Vec<Vec<f32>>,
    pub wo: Vec<Vec<f32>>,
    pub bf: Vec<f32>,
    pub bi: Vec<f32>,
    pub bg: Vec<f32>,
    pub bo: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}

fn matvec(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>())
        .collect()
}

fn xavier_vec(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f32> {
    let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in)
        .map(|_| rng.random_range(-scale..scale))
        .collect()
}

fn xavier_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let scale = (6.0_f32 / (cols + rows) as f32).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
        .collect()
}

impl TrajLstmCell {
    /// Create a new LSTM cell with Xavier-initialized weights.
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        let total = hidden_dim + input_dim;
        Self {
            wf: xavier_matrix(hidden_dim, total, rng),
            wi: xavier_matrix(hidden_dim, total, rng),
            wg: xavier_matrix(hidden_dim, total, rng),
            wo: xavier_matrix(hidden_dim, total, rng),
            bf: vec![0.0; hidden_dim],
            bi: vec![0.0; hidden_dim],
            bg: vec![0.0; hidden_dim],
            bo: vec![0.0; hidden_dim],
        }
    }

    /// One LSTM step: returns (h_new, c_new).
    pub fn step(&self, h: &[f32], c: &[f32], x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Concatenate [h, x]
        let mut hx = Vec::with_capacity(h.len() + x.len());
        hx.extend_from_slice(h);
        hx.extend_from_slice(x);

        let f_gate: Vec<f32> = matvec(&self.wf, &hx)
            .into_iter()
            .zip(self.bf.iter())
            .map(|(v, b)| sigmoid(v + b))
            .collect();
        let i_gate: Vec<f32> = matvec(&self.wi, &hx)
            .into_iter()
            .zip(self.bi.iter())
            .map(|(v, b)| sigmoid(v + b))
            .collect();
        let g_gate: Vec<f32> = matvec(&self.wg, &hx)
            .into_iter()
            .zip(self.bg.iter())
            .map(|(v, b)| tanh_f32(v + b))
            .collect();
        let o_gate: Vec<f32> = matvec(&self.wo, &hx)
            .into_iter()
            .zip(self.bo.iter())
            .map(|(v, b)| sigmoid(v + b))
            .collect();

        let c_new: Vec<f32> = f_gate
            .iter()
            .zip(c.iter())
            .zip(i_gate.iter().zip(g_gate.iter()))
            .map(|((f, c_prev), (i, g))| f * c_prev + i * g)
            .collect();

        let h_new: Vec<f32> = o_gate
            .iter()
            .zip(c_new.iter())
            .map(|(o, c_val)| o * tanh_f32(*c_val))
            .collect();

        (h_new, c_new)
    }
}

/// Extract feature vector [x, y, vx, vy] from an AgentState.
fn state_to_features(state: &AgentState) -> Vec<f32> {
    vec![
        state.position.x,
        state.position.y,
        state.velocity.vx,
        state.velocity.vy,
    ]
}

/// LSTM-based trajectory predictor.
pub struct LstmTrajPredictor {
    pub cells: Vec<TrajLstmCell>,
    pub output_proj: Vec<Vec<f32>>,
    pub output_bias: Vec<f32>,
    config: LstmConfig,
}

impl LstmTrajPredictor {
    /// Construct with Xavier initialization.
    pub fn new(config: LstmConfig, rng: &mut StdRng) -> Self {
        let mut cells = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            let input = if layer == 0 {
                config.input_dim
            } else {
                config.hidden_dim
            };
            cells.push(TrajLstmCell::new(input, config.hidden_dim, rng));
        }
        let output_proj = xavier_matrix(config.output_dim, config.hidden_dim, rng);
        let output_bias = vec![0.0; config.output_dim];
        Self {
            cells,
            output_proj,
            output_bias,
            config,
        }
    }

    /// Run the LSTM stack on a trajectory and return the final hidden state.
    pub fn encode(&self, history: &Trajectory) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let n_layers = self.config.n_layers;

        let mut hs: Vec<Vec<f32>> = (0..n_layers).map(|_| vec![0.0; hidden_dim]).collect();
        let mut cs: Vec<Vec<f32>> = (0..n_layers).map(|_| vec![0.0; hidden_dim]).collect();

        for state in &history.states {
            let mut x = state_to_features(state);
            for (layer, cell) in self.cells.iter().enumerate() {
                let (h_new, c_new) = cell.step(&hs[layer], &cs[layer], &x);
                hs[layer] = h_new.clone();
                cs[layer] = c_new;
                x = h_new;
            }
        }
        hs.into_iter()
            .last()
            .unwrap_or_else(|| vec![0.0; hidden_dim])
    }

    /// Project hidden state to output (x, y) delta.
    fn project_output(&self, h: &[f32]) -> (f32, f32) {
        let raw = matvec(&self.output_proj, h)
            .into_iter()
            .zip(self.output_bias.iter())
            .map(|(v, b)| v + b)
            .collect::<Vec<_>>();
        let ox = raw.first().copied().unwrap_or(0.0);
        let oy = raw.get(1).copied().unwrap_or(0.0);
        (ox, oy)
    }

    /// Auto-regressively predict `n_steps` future states.
    pub fn predict_steps(&self, history: &Trajectory, n_steps: usize) -> Trajectory {
        let hidden_dim = self.config.hidden_dim;
        let n_layers = self.config.n_layers;

        let mut hs: Vec<Vec<f32>> = (0..n_layers).map(|_| vec![0.0; hidden_dim]).collect();
        let mut cs: Vec<Vec<f32>> = (0..n_layers).map(|_| vec![0.0; hidden_dim]).collect();

        // Encode history
        for state in &history.states {
            let mut x = state_to_features(state);
            for (layer, cell) in self.cells.iter().enumerate() {
                let (h_new, c_new) = cell.step(&hs[layer], &cs[layer], &x);
                hs[layer] = h_new.clone();
                cs[layer] = c_new;
                x = h_new;
            }
        }

        let mut predicted_states = Vec::with_capacity(n_steps);
        let last = history.states.last().cloned().unwrap_or_else(|| {
            AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 0)
        });
        let mut current = last;

        for _ in 0..n_steps {
            let mut x = state_to_features(&current);
            for (layer, cell) in self.cells.iter().enumerate() {
                let (h_new, c_new) = cell.step(&hs[layer], &cs[layer], &x);
                hs[layer] = h_new.clone();
                cs[layer] = c_new;
                x = h_new;
            }
            let h_last = &hs[n_layers - 1];
            let (dx, dy) = self.project_output(h_last);
            let new_x = current.position.x + dx;
            let new_y = current.position.y + dy;
            let vx = dx / history.dt.max(1e-9);
            let vy = dy / history.dt.max(1e-9);
            let heading = vy.atan2(vx);
            let next_state = AgentState::new(
                Point2D::new(new_x, new_y),
                Velocity2D::new(vx, vy),
                heading,
                current.agent_id,
            );
            predicted_states.push(next_state.clone());
            current = next_state;
        }

        Trajectory::new(predicted_states, history.dt)
    }

    /// NLL loss proxy (MSE between predicted and actual positions).
    pub fn nll_loss(&self, history: &Trajectory, future: &Trajectory) -> f32 {
        let predicted = self.predict_steps(history, future.states.len());
        let mut total = 0.0_f32;
        let n = predicted.states.len().min(future.states.len());
        if n == 0 {
            return 0.0;
        }
        for i in 0..n {
            let dx = predicted.states[i].position.x - future.states[i].position.x;
            let dy = predicted.states[i].position.y - future.states[i].position.y;
            total += dx * dx + dy * dy;
        }
        total / n as f32
    }
}

// ---------------------------------------------------------------------------
// §4  Social LSTM
// ---------------------------------------------------------------------------

/// Occupancy-grid-based pooling over neighbor hidden states.
pub struct SocialPooling;

impl SocialPooling {
    /// Pool neighbor hidden states into a flat occupancy-grid feature.
    /// Neighbors closer than `cell_size` contribute their hidden state to the grid cell.
    pub fn pool(
        query_pos: Point2D,
        neighbors: &[(Point2D, Vec<f32>)],
        grid_size: usize,
        cell_size: f32,
    ) -> Vec<f32> {
        if neighbors.is_empty() {
            let hidden_dim = 0;
            return vec![0.0; grid_size * grid_size * hidden_dim.max(1)];
        }
        let hidden_dim = neighbors[0].1.len();
        let mut grid = vec![0.0_f32; grid_size * grid_size * hidden_dim];
        let half = (grid_size as f32 / 2.0) * cell_size;

        for (pos, h) in neighbors {
            let rel_x = pos.x - query_pos.x + half;
            let rel_y = pos.y - query_pos.y + half;
            let col = (rel_x / cell_size) as i32;
            let row = (rel_y / cell_size) as i32;
            if col >= 0 && col < grid_size as i32 && row >= 0 && row < grid_size as i32 {
                let base = (row as usize * grid_size + col as usize) * hidden_dim;
                for (k, &val) in h.iter().enumerate() {
                    if base + k < grid.len() {
                        grid[base + k] += val;
                    }
                }
            }
        }
        grid
    }
}

/// Social-LSTM: LSTM augmented with social pooling of neighbor states.
pub struct SocialLstm {
    pub cell: TrajLstmCell,
    pub social_proj: Vec<Vec<f32>>,
    pub output_proj: Vec<Vec<f32>>,
    pub output_bias: Vec<f32>,
    pub embedding_dim: usize,
    hidden_dim: usize,
    grid_size: usize,
    cell_size: f32,
}

impl SocialLstm {
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        let grid_size = 8usize;
        let pool_feat = grid_size * grid_size * hidden_dim;
        let embedding_dim = 64;
        // Cell input: embedded position + social embedding
        let cell_input = embedding_dim + embedding_dim;
        let cell = TrajLstmCell::new(cell_input, hidden_dim, rng);
        let social_proj = xavier_matrix(embedding_dim, pool_feat.max(1), rng);
        let output_proj = xavier_matrix(input_dim, hidden_dim, rng);
        let output_bias = vec![0.0; input_dim];
        Self {
            cell,
            social_proj,
            output_proj,
            output_bias,
            embedding_dim,
            hidden_dim,
            grid_size,
            cell_size: 1.0,
        }
    }

    fn embed_pos(pos: &Point2D, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        if dim >= 2 {
            v[0] = pos.x;
            v[1] = pos.y;
        }
        v
    }

    /// Predict future trajectories for all agents in the scene.
    pub fn predict_scene(&self, scene: &SceneContext, n_steps: usize) -> Vec<Trajectory> {
        let n_agents = scene.agents.len();
        let mut hs: Vec<Vec<f32>> = (0..n_agents).map(|_| vec![0.0; self.hidden_dim]).collect();
        let mut cs: Vec<Vec<f32>> = (0..n_agents).map(|_| vec![0.0; self.hidden_dim]).collect();

        // Encode histories
        let max_len = scene
            .agents
            .iter()
            .map(|t| t.states.len())
            .max()
            .unwrap_or(0);
        for t in 0..max_len {
            let positions: Vec<Option<Point2D>> = scene
                .agents
                .iter()
                .map(|traj| traj.states.get(t).map(|s| s.position))
                .collect();

            for (i, traj) in scene.agents.iter().enumerate() {
                if let Some(state) = traj.states.get(t) {
                    let pos = state.position;
                    // Collect neighbor hidden states
                    let neighbors: Vec<(Point2D, Vec<f32>)> = positions
                        .iter()
                        .enumerate()
                        .filter(|(j, p)| *j != i && p.is_some())
                        .map(|(j, p)| {
                            let pt = match p {
                                Some(pt) => *pt,
                                None => Point2D::new(0.0, 0.0),
                            };
                            (pt, hs[j].clone())
                        })
                        .collect();
                    let pool = SocialPooling::pool(pos, &neighbors, self.grid_size, self.cell_size);
                    let social_emb = matvec(&self.social_proj, &pool)
                        .into_iter()
                        .map(|v| v.tanh())
                        .collect::<Vec<_>>();
                    let pos_emb = Self::embed_pos(&pos, self.embedding_dim);
                    let mut x = Vec::with_capacity(self.embedding_dim * 2);
                    x.extend_from_slice(&pos_emb);
                    x.extend_from_slice(&social_emb);
                    let (h_new, c_new) = self.cell.step(&hs[i], &cs[i], &x);
                    hs[i] = h_new;
                    cs[i] = c_new;
                }
            }
        }

        // Predict future steps
        let mut current_positions: Vec<Point2D> = scene
            .agents
            .iter()
            .map(|traj| {
                traj.states
                    .last()
                    .map(|s| s.position)
                    .unwrap_or(Point2D::new(0.0, 0.0))
            })
            .collect();
        let dt = scene.agents.first().map(|t| t.dt).unwrap_or(0.1);

        let mut all_future: Vec<Vec<AgentState>> = (0..n_agents).map(|_| Vec::new()).collect();

        for _ in 0..n_steps {
            let prev_positions = current_positions.clone();
            for i in 0..n_agents {
                let pos = prev_positions[i];
                let neighbors: Vec<(Point2D, Vec<f32>)> = prev_positions
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(j, p)| (*p, hs[j].clone()))
                    .collect();
                let pool = SocialPooling::pool(pos, &neighbors, self.grid_size, self.cell_size);
                let social_emb = matvec(&self.social_proj, &pool)
                    .into_iter()
                    .map(|v| v.tanh())
                    .collect::<Vec<_>>();
                let pos_emb = Self::embed_pos(&pos, self.embedding_dim);
                let mut x = Vec::with_capacity(self.embedding_dim * 2);
                x.extend_from_slice(&pos_emb);
                x.extend_from_slice(&social_emb);
                let (h_new, c_new) = self.cell.step(&hs[i], &cs[i], &x);
                hs[i] = h_new;
                cs[i] = c_new;
            }
            for i in 0..n_agents {
                let out: Vec<f32> = matvec(&self.output_proj, &hs[i])
                    .into_iter()
                    .zip(self.output_bias.iter())
                    .map(|(v, b)| v + b)
                    .collect();
                let dx = out.first().copied().unwrap_or(0.0);
                let dy = out.get(1).copied().unwrap_or(0.0);
                let new_pos =
                    Point2D::new(current_positions[i].x + dx, current_positions[i].y + dy);
                let vx = dx / dt.max(1e-9);
                let vy = dy / dt.max(1e-9);
                let heading = vy.atan2(vx);
                let agent_id = scene
                    .agents
                    .get(i)
                    .and_then(|t| t.states.last())
                    .map(|s| s.agent_id)
                    .unwrap_or(i as u32);
                all_future[i].push(AgentState::new(
                    new_pos,
                    Velocity2D::new(vx, vy),
                    heading,
                    agent_id,
                ));
                current_positions[i] = new_pos;
            }
        }

        all_future
            .into_iter()
            .map(|states| Trajectory::new(states, dt))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// §5  Transformer-based Trajectory Predictor
// ---------------------------------------------------------------------------

/// Encodes a trajectory as a sequence of feature vectors.
pub struct MotionEncoder {
    pub d_model: usize,
}

impl MotionEncoder {
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    /// Encode trajectory to a sequence of d_model-dimensional token vectors.
    pub fn encode_trajectory(&self, traj: &Trajectory) -> Vec<Vec<f32>> {
        traj.states
            .iter()
            .enumerate()
            .map(|(t, state)| {
                let mut feat = vec![0.0_f32; self.d_model];
                // Positional + velocity features
                if self.d_model > 0 {
                    feat[0] = state.position.x;
                }
                if self.d_model > 1 {
                    feat[1] = state.position.y;
                }
                if self.d_model > 2 {
                    feat[2] = state.velocity.vx;
                }
                if self.d_model > 3 {
                    feat[3] = state.velocity.vy;
                }
                if self.d_model > 4 {
                    feat[4] = state.heading;
                }
                // Sinusoidal positional encoding
                let pos = t as f32;
                for k in (5..self.d_model).step_by(2) {
                    let freq = 1.0 / (10000.0_f32.powf(k as f32 / self.d_model as f32));
                    feat[k] = (pos * freq).sin();
                    if k + 1 < self.d_model {
                        feat[k + 1] = (pos * freq).cos();
                    }
                }
                feat
            })
            .collect()
    }
}

/// Cross-attention between query agent tokens and context agent tokens.
pub struct SceneAttention;

impl SceneAttention {
    fn softmax_vec(v: &[f32]) -> Vec<f32> {
        let max_v = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = v.iter().map(|x| (x - max_v).exp()).collect();
        let sum = exps.iter().sum::<f32>().max(1e-9);
        exps.iter().map(|e| e / sum).collect()
    }

    /// Multi-head cross-attention: query x context -> attended query.
    pub fn attend(
        query: &[Vec<f32>],
        context: &[Vec<f32>],
        d_model: usize,
        n_heads: usize,
    ) -> Vec<Vec<f32>> {
        if query.is_empty() || context.is_empty() || d_model == 0 || n_heads == 0 {
            return query.to_vec();
        }
        let head_dim = (d_model / n_heads).max(1);
        query
            .iter()
            .map(|q_vec| {
                let mut attended = vec![0.0_f32; d_model];
                for h in 0..n_heads {
                    let start = h * head_dim;
                    let end = (start + head_dim).min(d_model).min(q_vec.len());
                    let q_head: Vec<f32> = q_vec[start..end].to_vec();

                    // Compute scaled dot-product attention scores
                    let scale = (head_dim as f32).sqrt().max(1e-9);
                    let scores: Vec<f32> = context
                        .iter()
                        .map(|ctx| {
                            let ctx_head_end = end.min(ctx.len());
                            let ctx_head = &ctx[start..ctx_head_end];
                            q_head
                                .iter()
                                .zip(ctx_head.iter())
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / scale
                        })
                        .collect();

                    let attn = Self::softmax_vec(&scores);

                    // Weighted sum of context vectors
                    for (j, a) in attn.iter().enumerate() {
                        let ctx = &context[j];
                        for k in start..end.min(ctx.len()) {
                            attended[k] += a * ctx[k];
                        }
                    }
                }
                attended
            })
            .collect()
    }
}

/// Attention-based trajectory predictor.
pub struct TransformerTrajPredictor {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub output_proj: Vec<Vec<f32>>,
    pub output_bias: Vec<f32>,
    motion_encoder: MotionEncoder,
}

impl TransformerTrajPredictor {
    pub fn new(d_model: usize, n_heads: usize, n_layers: usize, rng: &mut StdRng) -> Self {
        let output_proj = xavier_matrix(2, d_model, rng);
        let output_bias = vec![0.0; 2];
        Self {
            d_model,
            n_heads,
            n_layers,
            output_proj,
            output_bias,
            motion_encoder: MotionEncoder::new(d_model),
        }
    }

    /// Predict `n_steps` future positions for the given agent in the scene.
    pub fn predict(
        &self,
        history: &Trajectory,
        scene: &SceneContext,
        n_steps: usize,
    ) -> Trajectory {
        let mut query_tokens = self.motion_encoder.encode_trajectory(history);

        // Gather context from all scene agents
        let context_tokens: Vec<Vec<f32>> = scene
            .agents
            .iter()
            .flat_map(|t| self.motion_encoder.encode_trajectory(t))
            .collect();

        // Stack attention layers
        for _ in 0..self.n_layers {
            let attended =
                SceneAttention::attend(&query_tokens, &context_tokens, self.d_model, self.n_heads);
            // Residual connection
            query_tokens = query_tokens
                .iter()
                .zip(attended.iter())
                .map(|(q, a)| q.iter().zip(a.iter()).map(|(x, y)| x + y).collect())
                .collect();
        }

        // Decode future positions auto-regressively using last token
        let last = history.states.last().cloned().unwrap_or_else(|| {
            AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 0)
        });

        let mut predicted = Vec::with_capacity(n_steps);
        let mut prev_pos = last.position;
        let dt = history.dt;

        for i in 0..n_steps {
            let token_idx = query_tokens.len().saturating_sub(1).min(i);
            let token = query_tokens
                .get(token_idx)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.d_model]);

            let out: Vec<f32> = matvec(&self.output_proj, &token)
                .into_iter()
                .zip(self.output_bias.iter())
                .map(|(v, b)| v + b)
                .collect();

            let dx = out.first().copied().unwrap_or(0.0);
            let dy = out.get(1).copied().unwrap_or(0.0);
            let new_x = prev_pos.x + dx;
            let new_y = prev_pos.y + dy;
            let vx = dx / dt.max(1e-9);
            let vy = dy / dt.max(1e-9);
            let heading = vy.atan2(vx);
            let state = AgentState::new(
                Point2D::new(new_x, new_y),
                Velocity2D::new(vx, vy),
                heading,
                last.agent_id,
            );
            prev_pos = state.position;
            predicted.push(state);
        }

        Trajectory::new(predicted, dt)
    }
}

// ---------------------------------------------------------------------------
// §6  Gaussian Mixture Trajectory Decoder
// ---------------------------------------------------------------------------

/// Bivariate Gaussian with correlation.
#[derive(Debug, Clone)]
pub struct BivariateGaussian {
    pub mu_x: f32,
    pub mu_y: f32,
    pub sigma_x: f32,
    pub sigma_y: f32,
    /// Pearson correlation ρ ∈ (-1, 1).
    pub rho: f32,
}

impl BivariateGaussian {
    pub fn new(mu_x: f32, mu_y: f32, sigma_x: f32, sigma_y: f32, rho: f32) -> Self {
        Self {
            mu_x,
            mu_y,
            sigma_x: sigma_x.max(1e-6),
            sigma_y: sigma_y.max(1e-6),
            rho: rho.clamp(-0.99, 0.99),
        }
    }

    /// Log-probability under the bivariate Gaussian.
    pub fn log_prob(&self, x: f32, y: f32) -> f32 {
        let zx = (x - self.mu_x) / self.sigma_x;
        let zy = (y - self.mu_y) / self.sigma_y;
        let rho2 = self.rho * self.rho;
        let z = zx * zx - 2.0 * self.rho * zx * zy + zy * zy;
        let denom = 2.0 * (1.0 - rho2).max(1e-9);
        let log_norm = -2.0 * std::f32::consts::PI.ln()
            - self.sigma_x.ln()
            - self.sigma_y.ln()
            - 0.5 * (1.0 - rho2).max(1e-9).ln();
        log_norm - z / denom
    }

    /// Sample from the bivariate Gaussian using the Box-Muller transform.
    pub fn sample(&self, rng: &mut StdRng) -> (f32, f32) {
        let u1: f32 = rng.random_range(1e-9_f32..1.0_f32);
        let u2: f32 = rng.random_range(0.0_f32..1.0_f32);
        let two_pi = 2.0_f32 * std::f32::consts::PI;
        let z1 = (-2.0_f32 * u1.ln()).sqrt() * (two_pi * u2).cos();
        let z2 = (-2.0_f32 * u1.ln()).sqrt() * (two_pi * u2).sin();
        let sx = self.mu_x + self.sigma_x * z1;
        let sy =
            self.mu_y + self.sigma_y * (self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * z2);
        (sx, sy)
    }
}

/// Output of the Gaussian Mixture trajectory decoder.
#[derive(Debug, Clone)]
pub struct GmmTrajOutput {
    /// One component per timestep for multi-step output (length = n_steps * n_components),
    /// indexed as [step * n_components + component].
    pub components: Vec<BivariateGaussian>,
    /// Mixture weights per step (length = n_steps * n_components).
    pub weights: Vec<f32>,
    pub n_steps: usize,
    pub n_components: usize,
}

impl GmmTrajOutput {
    pub fn new(
        components: Vec<BivariateGaussian>,
        weights: Vec<f32>,
        n_steps: usize,
        n_components: usize,
    ) -> Self {
        Self {
            components,
            weights,
            n_steps,
            n_components,
        }
    }
}

/// Sample a trajectory from the GMM output by choosing the most likely component.
pub fn sample_trajectory(gmm: &GmmTrajOutput, rng: &mut StdRng) -> Trajectory {
    let mut states = Vec::with_capacity(gmm.n_steps);
    let mut prev_x = 0.0_f32;
    let mut prev_y = 0.0_f32;

    for step in 0..gmm.n_steps {
        let base = step * gmm.n_components;
        // Sample a component index based on weights
        let weight_slice = &gmm.weights[base..(base + gmm.n_components).min(gmm.weights.len())];
        let total: f32 = weight_slice.iter().sum::<f32>().max(1e-9);
        let threshold: f32 = rng.random_range(0.0_f32..1.0_f32) * total;
        let mut cumsum = 0.0_f32;
        let mut chosen = 0usize;
        for (k, &w) in weight_slice.iter().enumerate() {
            cumsum += w;
            if cumsum >= threshold {
                chosen = k;
                break;
            }
        }
        let comp_idx = (base + chosen).min(gmm.components.len().saturating_sub(1));
        let comp = &gmm.components[comp_idx];
        let (sx, sy) = comp.sample(rng);
        let dx = sx - prev_x;
        let dy = sy - prev_y;
        let vx = if step == 0 { 0.0 } else { dx };
        let vy = if step == 0 { 0.0 } else { dy };
        let heading = vy.atan2(vx);
        states.push(AgentState::new(
            Point2D::new(sx, sy),
            Velocity2D::new(vx, vy),
            heading,
            0,
        ));
        prev_x = sx;
        prev_y = sy;
    }
    Trajectory::new(states, 0.1)
}

/// Negative log-likelihood of the target trajectory under the GMM.
pub fn gmm_nll(gmm: &GmmTrajOutput, target: &Trajectory) -> f32 {
    let n = target.states.len().min(gmm.n_steps);
    if n == 0 {
        return 0.0;
    }
    let mut total_nll = 0.0_f32;
    for step in 0..n {
        let x = target.states[step].position.x;
        let y = target.states[step].position.y;
        let base = step * gmm.n_components;
        let weight_end = (base + gmm.n_components).min(gmm.weights.len());
        let weight_slice = &gmm.weights[base..weight_end];
        let total_w: f32 = weight_slice.iter().sum::<f32>().max(1e-9);
        // log sum_k w_k * N(x, y | mu_k, sigma_k)
        let mut log_sum = f32::NEG_INFINITY;
        for (k, &w) in weight_slice.iter().enumerate() {
            let comp_idx = (base + k).min(gmm.components.len().saturating_sub(1));
            let lp = (w / total_w).max(1e-12).ln() + gmm.components[comp_idx].log_prob(x, y);
            log_sum = log_sum_exp_pair(log_sum, lp);
        }
        total_nll -= log_sum;
    }
    total_nll / n as f32
}

fn log_sum_exp_pair(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let mx = a.max(b);
    mx + ((a - mx).exp() + (b - mx).exp()).ln()
}

/// Decoder that maps a hidden state to a GMM over future steps.
pub struct GmmDecoder {
    pub hidden_dim: usize,
    pub n_components: usize,
    pub n_steps: usize,
    /// weights: shape [n_steps * n_components * 6, hidden_dim]
    /// Output per component: [weight_logit, mu_x, mu_y, log_sigma_x, log_sigma_y, atanh_rho]
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl GmmDecoder {
    pub fn new(hidden_dim: usize, n_components: usize, n_steps: usize, rng: &mut StdRng) -> Self {
        let out_dim = n_steps * n_components * 6;
        let weights = xavier_matrix(out_dim, hidden_dim, rng);
        let bias = vec![0.0; out_dim];
        Self {
            hidden_dim,
            n_components,
            n_steps,
            weights,
            bias,
        }
    }

    /// Decode a hidden vector into a GmmTrajOutput.
    pub fn decode(&self, hidden: &[f32]) -> GmmTrajOutput {
        let raw: Vec<f32> = matvec(&self.weights, hidden)
            .into_iter()
            .zip(self.bias.iter())
            .map(|(v, b)| v + b)
            .collect();

        let mut components = Vec::with_capacity(self.n_steps * self.n_components);
        let mut weights = Vec::with_capacity(self.n_steps * self.n_components);

        for step in 0..self.n_steps {
            let step_base = step * self.n_components * 6;
            // Extract weight logits for this step
            let weight_logits: Vec<f32> = (0..self.n_components)
                .map(|k| raw.get(step_base + k * 6).copied().unwrap_or(0.0))
                .collect();
            let softmax_weights = softmax_vec(&weight_logits);

            for k in 0..self.n_components {
                let base = step_base + k * 6;
                let mu_x = raw.get(base + 1).copied().unwrap_or(0.0);
                let mu_y = raw.get(base + 2).copied().unwrap_or(0.0);
                let log_sx = raw.get(base + 3).copied().unwrap_or(0.0);
                let log_sy = raw.get(base + 4).copied().unwrap_or(0.0);
                let atanh_rho = raw.get(base + 5).copied().unwrap_or(0.0);
                let sigma_x = log_sx.exp().max(1e-6);
                let sigma_y = log_sy.exp().max(1e-6);
                let rho = atanh_rho.tanh() * 0.99;
                components.push(BivariateGaussian::new(mu_x, mu_y, sigma_x, sigma_y, rho));
                weights.push(softmax_weights[k]);
            }
        }

        GmmTrajOutput::new(components, weights, self.n_steps, self.n_components)
    }
}

fn softmax_vec(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return vec![];
    }
    let max_v = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max_v).exp()).collect();
    let sum = exps.iter().sum::<f32>().max(1e-9);
    exps.iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// §7  Trajectory Evaluation Metrics
// ---------------------------------------------------------------------------

/// Evaluation metrics for trajectory prediction.
pub struct TrajMetrics;

impl TrajMetrics {
    /// Average Displacement Error: mean L2 distance over all timesteps.
    pub fn ade(predicted: &Trajectory, target: &Trajectory) -> f32 {
        let n = predicted.states.len().min(target.states.len());
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = (0..n)
            .map(|i| {
                predicted.states[i]
                    .position
                    .distance_to(&target.states[i].position)
            })
            .sum();
        sum / n as f32
    }

    /// Final Displacement Error: L2 distance at the last timestep.
    pub fn fde(predicted: &Trajectory, target: &Trajectory) -> f32 {
        let np = predicted.states.len();
        let nt = target.states.len();
        if np == 0 || nt == 0 {
            return 0.0;
        }
        predicted.states[np - 1]
            .position
            .distance_to(&target.states[nt - 1].position)
    }

    /// Miss rate: fraction of predictions where FDE > threshold.
    pub fn miss_rate(predictions: &[Trajectory], target: &Trajectory, threshold: f32) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let misses = predictions
            .iter()
            .filter(|p| Self::fde(p, target) > threshold)
            .count();
        misses as f32 / predictions.len() as f32
    }

    /// Best-of-K ADE: minimum ADE over K predictions.
    pub fn best_of_k_ade(k_preds: &[Trajectory], target: &Trajectory) -> f32 {
        k_preds
            .iter()
            .map(|p| Self::ade(p, target))
            .fold(f32::INFINITY, f32::min)
    }

    /// Best-of-K FDE: minimum FDE over K predictions.
    pub fn best_of_k_fde(k_preds: &[Trajectory], target: &Trajectory) -> f32 {
        k_preds
            .iter()
            .map(|p| Self::fde(p, target))
            .fold(f32::INFINITY, f32::min)
    }

    /// Collision rate: fraction of prediction pairs that come within `min_dist` of each other
    /// at any timestep.
    pub fn collision_rate(predictions: &[Trajectory], min_dist: f32) -> f32 {
        let n = predictions.len();
        if n < 2 {
            return 0.0;
        }
        let mut collision_count = 0usize;
        let total_pairs = n * (n - 1) / 2;
        for i in 0..n {
            for j in (i + 1)..n {
                let max_t = predictions[i].states.len().min(predictions[j].states.len());
                let collides = (0..max_t).any(|t| {
                    predictions[i].states[t]
                        .position
                        .distance_to(&predictions[j].states[t].position)
                        < min_dist
                });
                if collides {
                    collision_count += 1;
                }
            }
        }
        collision_count as f32 / total_pairs.max(1) as f32
    }
}

/// Summary report from batch evaluation.
#[derive(Debug, Clone)]
pub struct TrajEvalReport {
    pub ade: f32,
    pub fde: f32,
    pub miss_rate: f32,
    pub b_ade: f32,
    pub b_fde: f32,
}

impl TrajEvalReport {
    pub fn new(ade: f32, fde: f32, miss_rate: f32, b_ade: f32, b_fde: f32) -> Self {
        Self {
            ade,
            fde,
            miss_rate,
            b_ade,
            b_fde,
        }
    }
}

/// Compute a [`TrajEvalReport`] for a batch of (predictions, target) pairs.
/// Each element of `predictions` is a K-prediction set for the corresponding target.
pub fn evaluate(predictions: &[Trajectory], targets: &[Trajectory]) -> TrajEvalReport {
    let n = predictions.len().min(targets.len());
    if n == 0 {
        return TrajEvalReport::new(0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut ade_sum = 0.0_f32;
    let mut fde_sum = 0.0_f32;
    let mut miss_sum = 0.0_f32;
    let miss_threshold = 2.0_f32;

    for i in 0..n {
        let a = TrajMetrics::ade(&predictions[i], &targets[i]);
        let f = TrajMetrics::fde(&predictions[i], &targets[i]);
        ade_sum += a;
        fde_sum += f;
        if f > miss_threshold {
            miss_sum += 1.0;
        }
    }

    let ade = ade_sum / n as f32;
    let fde = fde_sum / n as f32;
    let miss = miss_sum / n as f32;

    // For B-ADE / B-FDE treat the slice as a set of K candidates
    let b_ade = TrajMetrics::best_of_k_ade(predictions, targets.first().unwrap_or(&predictions[0]));
    let b_fde = TrajMetrics::best_of_k_fde(predictions, targets.first().unwrap_or(&predictions[0]));

    TrajEvalReport::new(ade, fde, miss, b_ade, b_fde)
}

// ---------------------------------------------------------------------------
// §8  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn straight_line_traj(n: usize, dt: f32, agent_id: u32) -> Trajectory {
        let states: Vec<AgentState> = (0..n)
            .map(|i| {
                AgentState::new(
                    Point2D::new(i as f32 * dt, 0.0),
                    Velocity2D::new(1.0, 0.0),
                    0.0,
                    agent_id,
                )
            })
            .collect();
        Trajectory::new(states, dt)
    }

    // ── §1 Basic data structures ────────────────────────────────────────────

    #[test]
    fn test_point2d_creation() {
        let p = Point2D::new(3.0, 4.0);
        assert!((p.x - 3.0).abs() < 1e-6);
        assert!((p.y - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_trajectory_length() {
        let traj = straight_line_traj(5, 1.0, 0);
        // 4 segments of length 1.0 each
        assert!((traj.length() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_trajectory_displacement() {
        let traj = straight_line_traj(5, 1.0, 0);
        // displacement from 0 to 4
        assert!((traj.displacement() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_trajectory_average_speed() {
        let traj = straight_line_traj(5, 1.0, 0);
        assert!((traj.average_speed() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_trajectory_curvature() {
        let traj = straight_line_traj(5, 1.0, 0);
        // Straight line → curvature ≈ 0
        let c = traj.curvature_at(2);
        assert!(c.abs() < 1e-4, "expected near-zero curvature, got {c}");
    }

    #[test]
    fn test_scene_context_creation() {
        let traj = straight_line_traj(3, 0.5, 0);
        let ctx = SceneContext::new(vec![traj], vec![]);
        assert_eq!(ctx.agents.len(), 1);
    }

    // ── §2 Social Force Model ───────────────────────────────────────────────

    fn default_sfm() -> SocialForceModel {
        SocialForceModel::new(SfmConfig::default())
    }

    #[test]
    fn test_sfm_self_force_direction() {
        let sfm = default_sfm();
        let state = AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 0);
        let goal = Point2D::new(10.0, 0.0);
        let force = sfm.compute_self_force(&state, goal);
        assert!(force.vx > 0.0, "force should point toward positive x");
    }

    #[test]
    fn test_sfm_self_force_magnitude() {
        let sfm = default_sfm();
        let state = AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 0);
        let goal = Point2D::new(1.0, 0.0);
        let force = sfm.compute_self_force(&state, goal);
        // magnitude = desired_speed / relaxation_time = 1.34 / 0.5 = 2.68
        let mag = (force.vx * force.vx + force.vy * force.vy).sqrt();
        assert!((mag - 2.68).abs() < 0.1, "expected ~2.68, got {mag}");
    }

    #[test]
    fn test_sfm_repulsion_away_from_neighbor() {
        let sfm = default_sfm();
        let agent = AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 0);
        let neighbor = AgentState::new(Point2D::new(0.5, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 1);
        let rep = sfm.compute_repulsion(&agent, &[neighbor]);
        // Agent should be pushed in -x direction (away from neighbor at +x)
        assert!(rep.vx < 0.0, "repulsion should point away from neighbor");
    }

    #[test]
    fn test_sfm_repulsion_decreases_with_distance() {
        let sfm = default_sfm();
        let agent = AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 0);
        let near = AgentState::new(Point2D::new(0.3, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 1);
        let far = AgentState::new(Point2D::new(3.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 1);
        let rep_near = sfm.compute_repulsion(&agent, &[near]);
        let rep_far = sfm.compute_repulsion(&agent, &[far]);
        assert!(rep_near.vx.abs() > rep_far.vx.abs());
    }

    #[test]
    fn test_sfm_step_returns_states() {
        let sfm = default_sfm();
        let traj = straight_line_traj(3, 0.1, 0);
        let ctx = SceneContext::new(vec![traj], vec![]);
        let new_states = sfm.step(&ctx, &[Point2D::new(10.0, 0.0)], 0.1);
        assert_eq!(new_states.len(), 1);
    }

    #[test]
    fn test_sfm_predict_trajectory_length() {
        let sfm = default_sfm();
        let state = AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.5, 0.0), 0.0, 0);
        let trajs = sfm.predict(&[state], &[Point2D::new(5.0, 0.0)], 10, 0.1);
        assert_eq!(trajs.len(), 1);
        assert_eq!(trajs[0].states.len(), 11); // initial + 10 steps
    }

    #[test]
    fn test_sfm_predict_moves_toward_goal() {
        let sfm = default_sfm();
        let state = AgentState::new(Point2D::new(0.0, 0.0), Velocity2D::new(0.0, 0.0), 0.0, 0);
        let goal = Point2D::new(5.0, 0.0);
        let trajs = sfm.predict(&[state], &[goal], 20, 0.1);
        let last_pos = trajs[0].states.last().expect("should have states").position;
        // After 20 steps the agent should have moved toward the goal (x > 0)
        assert!(
            last_pos.x > 0.0,
            "agent should move toward goal, got x={}",
            last_pos.x
        );
    }

    // ── §3 LSTM Trajectory Predictor ────────────────────────────────────────

    fn default_lstm_config() -> LstmConfig {
        LstmConfig {
            input_dim: 4,
            hidden_dim: 16,
            output_dim: 2,
            n_layers: 1,
        }
    }

    #[test]
    fn test_lstm_cell_step_output_shape() {
        let mut rng = make_rng();
        let cell = TrajLstmCell::new(4, 16, &mut rng);
        let h = vec![0.0_f32; 16];
        let c = vec![0.0_f32; 16];
        let x = vec![1.0_f32; 4];
        let (h_new, c_new) = cell.step(&h, &c, &x);
        assert_eq!(h_new.len(), 16);
        assert_eq!(c_new.len(), 16);
    }

    #[test]
    fn test_lstm_cell_gate_range() {
        let mut rng = make_rng();
        let cell = TrajLstmCell::new(4, 16, &mut rng);
        let h = vec![0.0_f32; 16];
        let c = vec![0.0_f32; 16];
        let x = vec![0.5_f32; 4];
        let (h_new, _c_new) = cell.step(&h, &c, &x);
        // h = o * tanh(c), so |h| <= 1
        for v in &h_new {
            assert!(*v >= -1.0 && *v <= 1.0, "h should be in [-1,1], got {v}");
        }
    }

    #[test]
    fn test_lstm_traj_predictor_creation() {
        let mut rng = make_rng();
        let config = default_lstm_config();
        let predictor = LstmTrajPredictor::new(config, &mut rng);
        assert_eq!(predictor.cells.len(), 1);
    }

    #[test]
    fn test_lstm_encode_output_shape() {
        let mut rng = make_rng();
        let config = default_lstm_config();
        let predictor = LstmTrajPredictor::new(config, &mut rng);
        let traj = straight_line_traj(5, 0.1, 0);
        let hidden = predictor.encode(&traj);
        assert_eq!(hidden.len(), 16);
    }

    #[test]
    fn test_lstm_predict_steps_length() {
        let mut rng = make_rng();
        let config = default_lstm_config();
        let predictor = LstmTrajPredictor::new(config, &mut rng);
        let history = straight_line_traj(5, 0.1, 0);
        let future = predictor.predict_steps(&history, 10);
        assert_eq!(future.states.len(), 10);
    }

    #[test]
    fn test_lstm_nll_loss_finite() {
        let mut rng = make_rng();
        let config = default_lstm_config();
        let predictor = LstmTrajPredictor::new(config, &mut rng);
        let history = straight_line_traj(5, 0.1, 0);
        let future = straight_line_traj(5, 0.1, 0);
        let loss = predictor.nll_loss(&history, &future);
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
    }

    // ── §4 Social LSTM ──────────────────────────────────────────────────────

    #[test]
    fn test_social_pooling_output_shape() {
        let neighbors = vec![
            (Point2D::new(0.5, 0.0), vec![1.0_f32; 8]),
            (Point2D::new(-0.5, 0.0), vec![0.5_f32; 8]),
        ];
        let pool = SocialPooling::pool(Point2D::new(0.0, 0.0), &neighbors, 4, 1.0);
        assert_eq!(pool.len(), 4 * 4 * 8);
    }

    #[test]
    fn test_social_pooling_empty_neighbors() {
        let pool = SocialPooling::pool(Point2D::new(0.0, 0.0), &[], 4, 1.0);
        // When no neighbors, output is all zeros with length grid_size^2 * 1
        assert_eq!(pool.len(), 16);
        assert!(pool.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_social_lstm_creation() {
        let mut rng = make_rng();
        let slstm = SocialLstm::new(2, 32, &mut rng);
        assert_eq!(slstm.hidden_dim, 32);
    }

    #[test]
    fn test_social_lstm_predict_scene_count() {
        let mut rng = make_rng();
        let slstm = SocialLstm::new(2, 16, &mut rng);
        let t1 = straight_line_traj(4, 0.1, 0);
        let t2 = straight_line_traj(4, 0.1, 1);
        let scene = SceneContext::new(vec![t1, t2], vec![]);
        let futures = slstm.predict_scene(&scene, 5);
        assert_eq!(futures.len(), 2);
        for f in &futures {
            assert_eq!(f.states.len(), 5);
        }
    }

    // ── §5 Transformer Trajectory Predictor ─────────────────────────────────

    #[test]
    fn test_motion_encoder_shape() {
        let encoder = MotionEncoder::new(32);
        let traj = straight_line_traj(5, 0.1, 0);
        let tokens = encoder.encode_trajectory(&traj);
        assert_eq!(tokens.len(), 5);
        for t in &tokens {
            assert_eq!(t.len(), 32);
        }
    }

    #[test]
    fn test_scene_attention_output_shape() {
        let query: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0_f32; 16]).collect();
        let context: Vec<Vec<f32>> = (0..6).map(|_| vec![0.5_f32; 16]).collect();
        let out = SceneAttention::attend(&query, &context, 16, 4);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_transformer_traj_creation() {
        let mut rng = make_rng();
        let predictor = TransformerTrajPredictor::new(32, 4, 2, &mut rng);
        assert_eq!(predictor.d_model, 32);
        assert_eq!(predictor.n_heads, 4);
        assert_eq!(predictor.n_layers, 2);
    }

    #[test]
    fn test_transformer_predict_steps_length() {
        let mut rng = make_rng();
        let predictor = TransformerTrajPredictor::new(16, 2, 1, &mut rng);
        let history = straight_line_traj(5, 0.1, 0);
        let ctx = SceneContext::new(vec![], vec![]);
        let future = predictor.predict(&history, &ctx, 8);
        assert_eq!(future.states.len(), 8);
    }

    // ── §6 Gaussian Mixture Decoder ─────────────────────────────────────────

    #[test]
    fn test_bivariate_gaussian_creation() {
        let g = BivariateGaussian::new(1.0, 2.0, 0.5, 0.5, 0.1);
        assert!((g.mu_x - 1.0).abs() < 1e-6);
        assert!((g.mu_y - 2.0).abs() < 1e-6);
        assert!(g.sigma_x > 0.0);
        assert!(g.sigma_y > 0.0);
    }

    #[test]
    fn test_gmm_traj_output_creation() {
        let comp = BivariateGaussian::new(0.0, 0.0, 1.0, 1.0, 0.0);
        let gmm = GmmTrajOutput::new(vec![comp], vec![1.0], 1, 1);
        assert_eq!(gmm.n_steps, 1);
        assert_eq!(gmm.n_components, 1);
    }

    #[test]
    fn test_gmm_sample_length() {
        let mut rng = make_rng();
        let n_steps = 5;
        let n_comp = 2;
        let components: Vec<BivariateGaussian> = (0..n_steps * n_comp)
            .map(|_| BivariateGaussian::new(0.0, 0.0, 1.0, 1.0, 0.0))
            .collect();
        let weights: Vec<f32> = (0..n_steps * n_comp)
            .map(|k| if k % 2 == 0 { 0.6 } else { 0.4 })
            .collect();
        let gmm = GmmTrajOutput::new(components, weights, n_steps, n_comp);
        let traj = sample_trajectory(&gmm, &mut rng);
        assert_eq!(traj.states.len(), n_steps);
    }

    #[test]
    fn test_gmm_nll_finite() {
        let n_steps = 3;
        let n_comp = 2;
        let components: Vec<BivariateGaussian> = (0..n_steps * n_comp)
            .map(|_| BivariateGaussian::new(0.0, 0.0, 1.0, 1.0, 0.0))
            .collect();
        let weights: Vec<f32> = (0..n_steps * n_comp).map(|_| 0.5).collect();
        let gmm = GmmTrajOutput::new(components, weights, n_steps, n_comp);
        let target = straight_line_traj(n_steps, 0.1, 0);
        let nll = gmm_nll(&gmm, &target);
        assert!(nll.is_finite(), "NLL should be finite, got {nll}");
    }

    #[test]
    fn test_gmm_decoder_creation() {
        let mut rng = make_rng();
        let decoder = GmmDecoder::new(32, 3, 5, &mut rng);
        assert_eq!(decoder.n_components, 3);
        assert_eq!(decoder.n_steps, 5);
    }

    #[test]
    fn test_gmm_decode_output() {
        let mut rng = make_rng();
        let decoder = GmmDecoder::new(32, 2, 4, &mut rng);
        let hidden = vec![0.5_f32; 32];
        let out = decoder.decode(&hidden);
        assert_eq!(out.n_steps, 4);
        assert_eq!(out.n_components, 2);
        assert_eq!(out.components.len(), 4 * 2);
    }

    #[test]
    fn test_gmm_weights_sum_to_one() {
        let mut rng = make_rng();
        let decoder = GmmDecoder::new(16, 3, 4, &mut rng);
        let hidden = vec![0.1_f32; 16];
        let out = decoder.decode(&hidden);
        for step in 0..out.n_steps {
            let base = step * out.n_components;
            let sum: f32 = out.weights[base..(base + out.n_components).min(out.weights.len())]
                .iter()
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "weights should sum to 1 at step {step}, got {sum}"
            );
        }
    }

    // ── §7 Trajectory Metrics ────────────────────────────────────────────────

    #[test]
    fn test_ade_zero_perfect() {
        let traj = straight_line_traj(5, 0.1, 0);
        let ade = TrajMetrics::ade(&traj, &traj);
        assert!(
            ade < 1e-6,
            "ADE of perfect prediction should be ~0, got {ade}"
        );
    }

    #[test]
    fn test_ade_positive() {
        let traj1 = straight_line_traj(5, 0.1, 0);
        let states: Vec<AgentState> = (0..5)
            .map(|i| {
                AgentState::new(
                    Point2D::new(i as f32 * 0.1 + 1.0, 0.0),
                    Velocity2D::new(1.0, 0.0),
                    0.0,
                    0,
                )
            })
            .collect();
        let traj2 = Trajectory::new(states, 0.1);
        let ade = TrajMetrics::ade(&traj1, &traj2);
        assert!(
            ade > 0.0,
            "ADE should be positive for different trajectories"
        );
    }

    #[test]
    fn test_fde_zero_perfect() {
        let traj = straight_line_traj(5, 0.1, 0);
        let fde = TrajMetrics::fde(&traj, &traj);
        assert!(
            fde < 1e-6,
            "FDE of perfect prediction should be ~0, got {fde}"
        );
    }

    #[test]
    fn test_fde_positive() {
        let traj1 = straight_line_traj(5, 0.1, 0);
        let states: Vec<AgentState> = (0..5)
            .map(|i| {
                AgentState::new(
                    Point2D::new(i as f32 * 0.1, 1.0),
                    Velocity2D::new(1.0, 0.0),
                    0.0,
                    0,
                )
            })
            .collect();
        let traj2 = Trajectory::new(states, 0.1);
        let fde = TrajMetrics::fde(&traj1, &traj2);
        assert!(fde > 0.0);
    }

    #[test]
    fn test_miss_rate_all_miss() {
        let pred = straight_line_traj(5, 0.1, 0);
        let states: Vec<AgentState> = (0..5)
            .map(|i| {
                AgentState::new(
                    Point2D::new(i as f32 * 0.1, 100.0),
                    Velocity2D::new(1.0, 0.0),
                    0.0,
                    0,
                )
            })
            .collect();
        let target = Trajectory::new(states, 0.1);
        let rate = TrajMetrics::miss_rate(&[pred], &target, 2.0);
        assert!((rate - 1.0).abs() < 1e-6, "all should miss, got {rate}");
    }

    #[test]
    fn test_miss_rate_none_miss() {
        let traj = straight_line_traj(5, 0.1, 0);
        let rate = TrajMetrics::miss_rate(std::slice::from_ref(&traj), &traj, 100.0);
        assert!(rate < 1e-6, "none should miss with large threshold");
    }

    #[test]
    fn test_best_of_k_ade_le_worst() {
        let preds: Vec<Trajectory> = (0..3)
            .map(|k| {
                let states: Vec<AgentState> = (0..5)
                    .map(|i| {
                        AgentState::new(
                            Point2D::new(i as f32 * 0.1 + k as f32, 0.0),
                            Velocity2D::new(1.0, 0.0),
                            0.0,
                            0,
                        )
                    })
                    .collect();
                Trajectory::new(states, 0.1)
            })
            .collect();
        let target = straight_line_traj(5, 0.1, 0);
        let best = TrajMetrics::best_of_k_ade(&preds, &target);
        let worst: f32 = preds
            .iter()
            .map(|p| TrajMetrics::ade(p, &target))
            .fold(0.0, f32::max);
        assert!(best <= worst + 1e-6);
    }

    #[test]
    fn test_best_of_k_fde_le_worst() {
        let preds: Vec<Trajectory> = (0..3)
            .map(|k| {
                let states: Vec<AgentState> = (0..5)
                    .map(|i| {
                        AgentState::new(
                            Point2D::new(i as f32 * 0.1, k as f32 * 2.0),
                            Velocity2D::new(1.0, 0.0),
                            0.0,
                            0,
                        )
                    })
                    .collect();
                Trajectory::new(states, 0.1)
            })
            .collect();
        let target = straight_line_traj(5, 0.1, 0);
        let best = TrajMetrics::best_of_k_fde(&preds, &target);
        let worst: f32 = preds
            .iter()
            .map(|p| TrajMetrics::fde(p, &target))
            .fold(0.0, f32::max);
        assert!(best <= worst + 1e-6);
    }

    #[test]
    fn test_collision_rate_zero_far_apart() {
        let t1 = straight_line_traj(5, 0.1, 0);
        let states: Vec<AgentState> = (0..5)
            .map(|i| {
                AgentState::new(
                    Point2D::new(i as f32 * 0.1, 100.0),
                    Velocity2D::new(1.0, 0.0),
                    0.0,
                    1,
                )
            })
            .collect();
        let t2 = Trajectory::new(states, 0.1);
        let rate = TrajMetrics::collision_rate(&[t1, t2], 1.0);
        assert!(
            rate < 1e-6,
            "should have zero collisions for far-apart trajectories"
        );
    }

    #[test]
    fn test_collision_rate_one_same_position() {
        let t1 = straight_line_traj(5, 0.1, 0);
        let t2 = straight_line_traj(5, 0.1, 1);
        let rate = TrajMetrics::collision_rate(&[t1, t2], 5.0);
        assert!(
            (rate - 1.0).abs() < 1e-6,
            "should have full collision rate, got {rate}"
        );
    }

    #[test]
    fn test_traj_eval_report_fields() {
        let report = TrajEvalReport::new(1.0, 2.0, 0.3, 0.5, 1.0);
        assert!((report.ade - 1.0).abs() < 1e-6);
        assert!((report.fde - 2.0).abs() < 1e-6);
        assert!((report.miss_rate - 0.3).abs() < 1e-6);
        assert!((report.b_ade - 0.5).abs() < 1e-6);
        assert!((report.b_fde - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_batch() {
        let preds: Vec<Trajectory> = (0..4)
            .map(|i| straight_line_traj(5, 0.1, i as u32))
            .collect();
        let targets: Vec<Trajectory> = (0..4)
            .map(|i| straight_line_traj(5, 0.1, i as u32))
            .collect();
        let report = evaluate(&preds, &targets);
        assert!(report.ade >= 0.0);
        assert!(report.fde >= 0.0);
        assert!(report.miss_rate >= 0.0 && report.miss_rate <= 1.0);
        assert!(report.b_ade >= 0.0);
        assert!(report.b_fde >= 0.0);
    }
}
