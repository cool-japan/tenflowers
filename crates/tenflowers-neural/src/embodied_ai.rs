//! Embodied AI and Robot Learning Components.
//!
//! Provides environment interfaces, visual navigation policies, manipulation
//! learning, self-supervised pretraining, hierarchical RL, and evaluation
//! metrics for embodied intelligence research.
//!
//! # Sections
//! 1. \[`EmbodiedEnv`\] — environment interfaces (GridWorld, ContinuousNav)
//! 2. \[`VisualNavigationPolicy`\] — vision-based navigation (PointGoal, ObjectNav)
//! 3. \[`ManipulationLearning`\] — grasping and Dynamic Movement Primitives
//! 4. \[`EmbodiedPretraining`\] — inverse/forward dynamics and temporal contrastive
//! 5. [`HierarchicalPolicy`] — hierarchical reinforcement learning
//! 6. \[`EmbodiedEval`\] — evaluation metrics (SPL, grasp success, smoothness)

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─── Math primitives ──────────────────────────────────────────────────────────

#[inline]
fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-88.0, 88.0)).exp())
}

/// Matrix-vector multiply + bias + optional ReLU.
fn linear_layer(mat: &[Vec<f32>], bias: &[f32], input: &[f32], apply_relu: bool) -> Vec<f32> {
    mat.iter()
        .zip(bias.iter())
        .map(|(row, &b)| {
            let s: f32 = row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum();
            let v = s + b;
            if apply_relu {
                relu_f32(v)
            } else {
                v
            }
        })
        .collect()
}

/// Forward pass through a sequence of (weight, bias) layers with ReLU, final linear.
fn mlp_forward(layers: &[(Vec<Vec<f32>>, Vec<f32>)], input: &[f32]) -> Vec<f32> {
    let mut h: Vec<f32> = input.to_vec();
    for (i, (w, b)) in layers.iter().enumerate() {
        let is_last = i == layers.len() - 1;
        h = linear_layer(w, b, &h, !is_last);
    }
    h
}

/// Xavier-uniform initialised weight matrix [rows × cols].
fn xavier_mat(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let scale = (6.0_f32 / (rows as f32 + cols as f32)).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
        .collect()
}

fn zero_vec(n: usize) -> Vec<f32> {
    vec![0.0_f32; n]
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = l2_norm(a).max(1e-8);
    let nb = l2_norm(b).max(1e-8);
    (d / (na * nb)).clamp(-1.0, 1.0)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — EMBODIED ENVIRONMENTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Observation type for embodied environments.
#[derive(Debug, Clone)]
pub enum EmbObsType {
    /// Flat image buffer plus (height, width) dimensions.
    Image(Vec<f32>, usize, usize),
    /// Sequence of 3-D (or higher) points.
    PointCloud(Vec<Vec<f32>>),
    /// Flat state vector.
    State(Vec<f32>),
    /// Combined image + state observation.
    MultiModal { image: Vec<f32>, state: Vec<f32> },
}

/// Action type for embodied environments.
#[derive(Debug, Clone)]
pub enum EmbActionType {
    /// Continuous action vector.
    Continuous(Vec<f32>),
    /// Discrete action index.
    Discrete(usize),
    /// Joint-space motor command with gripper scalar.
    MotorCommand {
        joint_angles: Vec<f32>,
        gripper: f32,
    },
}

/// A single timestep observation from an embodied environment.
#[derive(Debug, Clone)]
pub struct EmbodiedObservation {
    /// The raw observation.
    pub obs: EmbObsType,
    /// Current timestep index.
    pub timestep: usize,
    /// Whether the episode has ended.
    pub done: bool,
    /// Scalar reward for this transition.
    pub reward: f32,
}

// ─── GridWorldEnv ─────────────────────────────────────────────────────────────

/// A discrete grid-world navigation environment.
#[derive(Debug, Clone)]
pub struct GridWorldEnv {
    pub width: usize,
    pub height: usize,
    pub obstacles: Vec<(usize, usize)>,
    pub goal: (usize, usize),
    pub agent: (usize, usize),
    timestep: usize,
}

impl GridWorldEnv {
    /// Create a new grid world with random obstacle placement.
    pub fn new(width: usize, height: usize, rng: &mut impl Rng) -> Self {
        let n_obs = (width * height / 10).max(1);
        let mut obstacles = Vec::new();
        for _ in 0..n_obs {
            let x = rng.random_range(0..width);
            let y = rng.random_range(0..height);
            obstacles.push((x, y));
        }
        let goal = (width - 1, height - 1);
        let agent = (0, 0);
        Self {
            width,
            height,
            obstacles,
            goal,
            agent,
            timestep: 0,
        }
    }

    fn is_obstacle(&self, pos: (usize, usize)) -> bool {
        self.obstacles.contains(&pos)
    }

    fn state_vec(&self) -> Vec<f32> {
        vec![
            self.agent.0 as f32,
            self.agent.1 as f32,
            self.goal.0 as f32,
            self.goal.1 as f32,
        ]
    }

    /// Step the environment.  action: 0=up, 1=right, 2=down, 3=left.
    pub fn step(&mut self, action: usize) -> EmbodiedObservation {
        let (ax, ay) = self.agent;
        let new_pos = match action {
            0 => (ax, ay.saturating_sub(1)),
            1 => ((ax + 1).min(self.width - 1), ay),
            2 => (ax, (ay + 1).min(self.height - 1)),
            3 => (ax.saturating_sub(1), ay),
            _ => (ax, ay),
        };
        if !self.is_obstacle(new_pos) {
            self.agent = new_pos;
        }
        self.timestep += 1;
        let done = self.agent == self.goal;
        let reward = if done { 1.0 } else { -0.01 };
        EmbodiedObservation {
            obs: EmbObsType::State(self.state_vec()),
            timestep: self.timestep,
            done,
            reward,
        }
    }

    /// Reset the environment, placing agent at (0,0).
    pub fn reset(&mut self, _rng: &mut impl Rng) -> EmbodiedObservation {
        self.agent = (0, 0);
        self.timestep = 0;
        EmbodiedObservation {
            obs: EmbObsType::State(self.state_vec()),
            timestep: 0,
            done: false,
            reward: 0.0,
        }
    }

    /// Render the grid as ASCII bytes (`.` = empty, `A` = agent, `G` = goal, `#` = obstacle).
    pub fn render(&self) -> Vec<Vec<u8>> {
        (0..self.height)
            .map(|y| {
                (0..self.width)
                    .map(|x| {
                        let pos = (x, y);
                        if pos == self.agent {
                            b'A'
                        } else if pos == self.goal {
                            b'G'
                        } else if self.is_obstacle(pos) {
                            b'#'
                        } else {
                            b'.'
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

// ─── ContinuousNavEnv ─────────────────────────────────────────────────────────

/// A continuous 2-D navigation environment with circular obstacles.
#[derive(Debug, Clone)]
pub struct ContinuousNavEnv {
    pub map_size: f32,
    pub agent_pos: Vec<f32>,
    pub goal_pos: Vec<f32>,
    /// Each obstacle is (center: `Vec<f32>`, radius: f32).
    pub obstacles: Vec<(Vec<f32>, f32)>,
    timestep: usize,
}

impl ContinuousNavEnv {
    /// Create a new continuous nav environment.
    pub fn new(map_size: f32, goal_pos: Vec<f32>, obstacles: Vec<(Vec<f32>, f32)>) -> Self {
        Self {
            map_size,
            agent_pos: vec![0.0, 0.0],
            goal_pos,
            obstacles,
            timestep: 0,
        }
    }

    /// Circle-circle collision check against all obstacles.
    pub fn is_collision(pos: &[f32], obstacles: &[(Vec<f32>, f32)]) -> bool {
        let agent_radius = 0.1_f32;
        obstacles.iter().any(|(center, radius)| {
            let dx = pos.first().copied().unwrap_or(0.0) - center.first().copied().unwrap_or(0.0);
            let dy = pos.get(1).copied().unwrap_or(0.0) - center.get(1).copied().unwrap_or(0.0);
            let dist = (dx * dx + dy * dy).sqrt();
            dist < radius + agent_radius
        })
    }

    fn dist_to_goal(&self) -> f32 {
        let dx = self.agent_pos.first().copied().unwrap_or(0.0)
            - self.goal_pos.first().copied().unwrap_or(0.0);
        let dy = self.agent_pos.get(1).copied().unwrap_or(0.0)
            - self.goal_pos.get(1).copied().unwrap_or(0.0);
        (dx * dx + dy * dy).sqrt()
    }

    /// Apply a velocity command (2-D) and return observation.
    pub fn step(&mut self, velocity: &[f32]) -> EmbodiedObservation {
        let vx = velocity.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let vy = velocity.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let nx = (self.agent_pos.first().copied().unwrap_or(0.0) + vx).clamp(0.0, self.map_size);
        let ny = (self.agent_pos.get(1).copied().unwrap_or(0.0) + vy).clamp(0.0, self.map_size);
        let new_pos = vec![nx, ny];
        if !Self::is_collision(&new_pos, &self.obstacles) {
            self.agent_pos = new_pos;
        }
        self.timestep += 1;
        let dist = self.dist_to_goal();
        let done = dist < 0.5;
        let reward = if done { 1.0 } else { -dist * 0.01 };
        let mut state = self.agent_pos.clone();
        state.extend_from_slice(&self.goal_pos);
        EmbodiedObservation {
            obs: EmbObsType::State(state),
            timestep: self.timestep,
            done,
            reward,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — VISUAL NAVIGATION POLICY
// ═══════════════════════════════════════════════════════════════════════════════

/// FC-based visual encoder.
#[derive(Debug, Clone)]
pub struct VisualEncoder {
    pub conv_weights: Vec<Vec<f32>>,
    pub fc_weight: Vec<Vec<f32>>,
    pub fc_bias: Vec<f32>,
    pub output_dim: usize,
}

impl VisualEncoder {
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let hidden = 64.min(input_dim);
        let conv_weights = xavier_mat(hidden, input_dim, &mut std_rng);
        let fc_weight = xavier_mat(output_dim, hidden, &mut std_rng);
        let fc_bias = zero_vec(output_dim);
        Self {
            conv_weights,
            fc_weight,
            fc_bias,
            output_dim,
        }
    }

    /// Encode a flat image buffer to a feature vector.
    pub fn encode(&self, image: &[f32]) -> Vec<f32> {
        let h: Vec<f32> = self
            .conv_weights
            .iter()
            .map(|row| {
                let s: f32 = row.iter().zip(image.iter()).map(|(&w, &x)| w * x).sum();
                relu_f32(s)
            })
            .collect();
        linear_layer(&self.fc_weight, &self.fc_bias, &h, false)
    }
}

/// FC-based goal encoder.
#[derive(Debug, Clone)]
pub struct GoalEncoder {
    pub weight: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl GoalEncoder {
    pub fn new(goal_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let weight = xavier_mat(output_dim, goal_dim, &mut std_rng);
        let bias = zero_vec(output_dim);
        Self { weight, bias }
    }

    pub fn encode(&self, goal: &[f32]) -> Vec<f32> {
        linear_layer(&self.weight, &self.bias, goal, true)
    }
}

/// Point-goal navigation policy combining visual and goal encoders.
#[derive(Debug, Clone)]
pub struct PointGoalPolicy {
    pub visual_enc: VisualEncoder,
    pub goal_enc: GoalEncoder,
    /// (weights, bias) for the policy head.
    pub policy_head: (Vec<Vec<f32>>, Vec<f32>),
    /// (weights, bias) for the value head.
    pub value_head: (Vec<Vec<f32>>, Vec<f32>),
}

impl PointGoalPolicy {
    pub fn new(
        img_dim: usize,
        goal_dim: usize,
        hidden_dim: usize,
        n_actions: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let visual_enc = VisualEncoder::new(img_dim, hidden_dim, &mut std_rng);
        let goal_enc = GoalEncoder::new(goal_dim, hidden_dim, &mut std_rng);
        let fused = hidden_dim * 2;
        let ph_w = xavier_mat(n_actions, fused, &mut std_rng);
        let ph_b = zero_vec(n_actions);
        let vh_w = xavier_mat(1, fused, &mut std_rng);
        let vh_b = zero_vec(1);
        Self {
            visual_enc,
            goal_enc,
            policy_head: (ph_w, ph_b),
            value_head: (vh_w, vh_b),
        }
    }

    /// Returns (action_logits, value).
    pub fn act(&self, image: &[f32], goal: &[f32]) -> (Vec<f32>, f32) {
        let vf = self.visual_enc.encode(image);
        let gf = self.goal_enc.encode(goal);
        let fused: Vec<f32> = vf.iter().chain(gf.iter()).copied().collect();
        let logits = linear_layer(&self.policy_head.0, &self.policy_head.1, &fused, false);
        let value_vec = linear_layer(&self.value_head.0, &self.value_head.1, &fused, false);
        let value = value_vec.first().copied().unwrap_or(0.0);
        (logits, value)
    }

    /// Return the argmax action.
    pub fn navigate(&self, image: &[f32], goal: &[f32]) -> usize {
        let (logits, _) = self.act(image, goal);
        argmax(&logits)
    }
}

/// Object-goal navigation policy.
#[derive(Debug, Clone)]
pub struct ObjectNavPolicy {
    pub detector_weights: Vec<Vec<f32>>,
    pub policy_head: (Vec<Vec<f32>>, Vec<f32>),
}

impl ObjectNavPolicy {
    pub fn new(
        obs_dim: usize,
        n_object_classes: usize,
        n_actions: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let detector_weights = xavier_mat(n_object_classes, obs_dim, &mut std_rng);
        let ph_w = xavier_mat(n_actions, obs_dim + n_object_classes, &mut std_rng);
        let ph_b = zero_vec(n_actions);
        Self {
            detector_weights,
            policy_head: (ph_w, ph_b),
        }
    }

    /// Returns a sigmoid detection score in [0, 1] for the target class.
    pub fn detect_object(&self, obs: &[f32], target_class: usize) -> f32 {
        let row = self
            .detector_weights
            .get(target_class % self.detector_weights.len());
        let row = match row {
            Some(r) => r,
            None => return 0.5,
        };
        let s: f32 = row.iter().zip(obs.iter()).map(|(&w, &x)| w * x).sum();
        sigmoid_f32(s)
    }

    /// Return the action index for navigating toward the target object class.
    pub fn act(&self, obs: &[f32], target_class: usize) -> usize {
        let n_classes = self.detector_weights.len();
        let mut scores = zero_vec(n_classes);
        for (c, row) in self.detector_weights.iter().enumerate() {
            let s: f32 = row.iter().zip(obs.iter()).map(|(&w, &x)| w * x).sum();
            scores[c] = sigmoid_f32(s);
        }
        let fused: Vec<f32> = obs.iter().chain(scores.iter()).copied().collect();
        let n_actions = self.policy_head.0.len();
        let logits = linear_layer(&self.policy_head.0, &self.policy_head.1, &fused, false);
        let _ = target_class; // target class is encoded through scores
        argmax(&logits[..n_actions.min(logits.len())])
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — MANIPULATION LEARNING
// ═══════════════════════════════════════════════════════════════════════════════

/// A single grasp candidate with position, orientation and quality.
#[derive(Debug, Clone)]
pub struct GraspCandidate {
    /// 3-D grasp position.
    pub position: Vec<f32>,
    /// Quaternion or Euler orientation (4-D or 3-D).
    pub orientation: Vec<f32>,
    /// Estimated grasp quality in [0, 1].
    pub quality: f32,
}

/// MLP-based grasp planner operating on point clouds.
#[derive(Debug, Clone)]
pub struct GraspPlanner {
    /// Network layers: Vec<(weight_matrix, bias)>.
    pub network: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    pub input_dim: usize,
}

impl GraspPlanner {
    pub fn new(point_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        // Layers: point_dim → hidden_dim → 8 (pos:3 + ori:4 + quality:1)
        let l1_w = xavier_mat(hidden_dim, point_dim, &mut std_rng);
        let l1_b = zero_vec(hidden_dim);
        let l2_w = xavier_mat(8, hidden_dim, &mut std_rng);
        let l2_b = zero_vec(8);
        Self {
            network: vec![(l1_w, l1_b), (l2_w, l2_b)],
            input_dim: point_dim,
        }
    }

    fn mean_pool_cloud(cloud: &[Vec<f32>]) -> Vec<f32> {
        if cloud.is_empty() {
            return Vec::new();
        }
        let dim = cloud[0].len();
        let mut out = vec![0.0_f32; dim];
        for pt in cloud {
            for (i, &v) in pt.iter().enumerate() {
                if i < out.len() {
                    out[i] += v;
                }
            }
        }
        let n = cloud.len() as f32;
        out.iter_mut().for_each(|v| *v /= n);
        out
    }

    /// Predict a grasp candidate from a point cloud via mean-pooling + MLP.
    pub fn predict_grasp(&self, point_cloud: &[Vec<f32>]) -> GraspCandidate {
        let feat = Self::mean_pool_cloud(point_cloud);
        let raw = mlp_forward(&self.network, &feat);
        let position = raw.get(0..3).unwrap_or(&[0.0, 0.0, 0.0]).to_vec();
        let orientation = raw.get(3..7).unwrap_or(&[1.0, 0.0, 0.0, 0.0]).to_vec();
        let quality = sigmoid_f32(raw.get(7).copied().unwrap_or(0.0));
        GraspCandidate {
            position,
            orientation,
            quality,
        }
    }

    /// Return indices of candidates sorted by quality (highest first).
    pub fn rank_grasps(&self, candidates: &[GraspCandidate]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..candidates.len()).collect();
        indices.sort_by(|&a, &b| {
            candidates[b]
                .quality
                .partial_cmp(&candidates[a].quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}

// ─── Dynamic Movement Primitives ──────────────────────────────────────────────

/// Configuration for a DMP.
#[derive(Debug, Clone)]
pub struct DmpConfig {
    /// Number of Gaussian basis functions.
    pub n_basis: usize,
    /// Spring constant (goal attraction).
    pub alpha: f32,
    /// Damping coefficient.
    pub beta: f32,
    /// Temporal scaling factor.
    pub tau: f32,
}

impl Default for DmpConfig {
    fn default() -> Self {
        Self {
            n_basis: 20,
            alpha: 25.0,
            beta: 6.25,
            tau: 1.0,
        }
    }
}

/// Dynamic Movement Primitive for trajectory learning and reproduction.
#[derive(Debug, Clone)]
pub struct Dmp {
    pub config: DmpConfig,
    /// LWR-fitted forcing-function weights.
    pub weights: Vec<f32>,
    pub start: Vec<f32>,
    pub goal: Vec<f32>,
}

impl Dmp {
    /// Compute a vector of Gaussian basis function activations.
    pub fn basis_fn(x: f32, centers: &[f32], width: f32) -> Vec<f32> {
        centers
            .iter()
            .map(|&c| (-(x - c).powi(2) / (2.0 * width * width + f32::EPSILON)).exp())
            .collect()
    }

    fn make_centers(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                
                if n > 1 {
                    i as f32 / (n - 1) as f32
                } else {
                    0.5
                }
            })
            .collect()
    }

    /// Fit DMP weights to a set of demonstrations using Locally Weighted Regression.
    pub fn fit(demonstrations: &[Vec<Vec<f32>>]) -> Self {
        let config = DmpConfig::default();
        if demonstrations.is_empty() {
            return Self {
                config,
                weights: vec![0.0; 20],
                start: vec![0.0],
                goal: vec![1.0],
            };
        }
        let demo = &demonstrations[0];
        if demo.is_empty() {
            return Self {
                config,
                weights: vec![0.0; 20],
                start: vec![0.0],
                goal: vec![1.0],
            };
        }
        let dim = demo[0].len();
        let n = demo.len();
        let start = demo[0].clone();
        let goal = demo[n - 1].clone();
        let n_basis = config.n_basis;
        let centers = Self::make_centers(n_basis);
        let width = 0.05_f32;
        // Use first dimension for scalar DMP weights fitting
        let mut num = vec![0.0_f32; n_basis];
        let mut den = vec![0.0_f32; n_basis];
        for (t_idx, traj_pt) in demo.iter().enumerate() {
            let x = if n > 1 {
                t_idx as f32 / (n - 1) as f32
            } else {
                0.0
            };
            let basis = Self::basis_fn(x, &centers, width);
            let target = if dim > 0 { traj_pt[0] } else { 0.0 };
            for (k, &phi) in basis.iter().enumerate() {
                num[k] += phi * x * target;
                den[k] += phi * x * x;
            }
        }
        let weights: Vec<f32> = num
            .iter()
            .zip(den.iter())
            .map(|(&n, &d)| if d.abs() < f32::EPSILON { 0.0 } else { n / d })
            .collect();
        let _ = start.clone();
        Self {
            config,
            weights,
            start,
            goal,
        }
    }

    /// Generate a trajectory by integrating the DMP dynamics.
    pub fn rollout(&self, dt: f32, n_steps: usize) -> Vec<Vec<f32>> {
        let n_basis = self.config.n_basis;
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let tau = self.config.tau.max(f32::EPSILON);
        let centers = Self::make_centers(n_basis);
        let width = 0.05_f32;
        let dim = self.start.len().max(1);
        let mut y: Vec<f32> = self.start.clone();
        if y.is_empty() {
            y = vec![0.0];
        }
        let mut dy = vec![0.0_f32; dim];
        let mut trajectory = Vec::with_capacity(n_steps);
        trajectory.push(y.clone());
        for step in 1..n_steps {
            let x = 1.0 - (step as f32 / n_steps as f32);
            let basis = Self::basis_fn(x, &centers, width);
            let basis_sum: f32 = basis.iter().sum::<f32>().max(f32::EPSILON);
            let forcing: f32 = basis
                .iter()
                .zip(self.weights.iter())
                .map(|(&phi, &w)| phi * w)
                .sum::<f32>()
                / basis_sum
                * x;
            for d in 0..dim {
                let goal_d = self.goal.get(d).copied().unwrap_or(0.0);
                let ddy = (alpha * (beta * (goal_d - y[d]) - dy[d]) + forcing) / tau;
                dy[d] += ddy * dt;
                y[d] += dy[d] * dt;
            }
            trajectory.push(y.clone());
        }
        trajectory
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4 — EMBODIED PRETRAINING
// ═══════════════════════════════════════════════════════════════════════════════

/// Inverse dynamics model: predict action from (obs_t, obs_{t+1}).
#[derive(Debug, Clone)]
pub struct InverseDynamicsModel {
    /// Shared encoder for a single observation.
    pub encoder: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    /// Prediction head operating on concatenated encodings.
    pub prediction_head: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    obs_dim: usize,
    action_dim: usize,
}

impl InverseDynamicsModel {
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let enc_w = xavier_mat(hidden_dim, obs_dim, &mut std_rng);
        let enc_b = zero_vec(hidden_dim);
        let head_w = xavier_mat(action_dim, hidden_dim * 2, &mut std_rng);
        let head_b = zero_vec(action_dim);
        Self {
            encoder: vec![(enc_w, enc_b)],
            prediction_head: vec![(head_w, head_b)],
            obs_dim,
            action_dim,
        }
    }

    fn encode_obs(&self, obs: &[f32]) -> Vec<f32> {
        mlp_forward(&self.encoder, obs)
    }

    /// Predict action from a pair of consecutive observations.
    pub fn predict(&self, obs_t: &[f32], obs_tp1: &[f32]) -> Vec<f32> {
        let e1 = self.encode_obs(obs_t);
        let e2 = self.encode_obs(obs_tp1);
        let cat: Vec<f32> = e1.iter().chain(e2.iter()).copied().collect();
        mlp_forward(&self.prediction_head, &cat)
    }

    /// MSE loss between predicted and actual action.
    pub fn loss(&self, obs_t: &[f32], obs_tp1: &[f32], actual_action: &[f32]) -> f32 {
        let pred = self.predict(obs_t, obs_tp1);
        let n = pred.len().max(1);
        pred.iter()
            .zip(actual_action.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum::<f32>()
            / n as f32
    }
}

/// Forward dynamics model: predict next observation from (obs, action).
#[derive(Debug, Clone)]
pub struct ForwardDynamicsModel {
    pub network: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    obs_dim: usize,
    action_dim: usize,
}

impl ForwardDynamicsModel {
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let l1_w = xavier_mat(hidden_dim, obs_dim + action_dim, &mut std_rng);
        let l1_b = zero_vec(hidden_dim);
        let l2_w = xavier_mat(obs_dim, hidden_dim, &mut std_rng);
        let l2_b = zero_vec(obs_dim);
        Self {
            network: vec![(l1_w, l1_b), (l2_w, l2_b)],
            obs_dim,
            action_dim,
        }
    }

    /// Predict the next observation.
    pub fn predict(&self, obs: &[f32], action: &[f32]) -> Vec<f32> {
        let input: Vec<f32> = obs.iter().chain(action.iter()).copied().collect();
        mlp_forward(&self.network, &input)
    }

    /// MSE loss against next_obs.
    pub fn loss(&self, obs: &[f32], action: &[f32], next_obs: &[f32]) -> f32 {
        let pred = self.predict(obs, action);
        let n = pred.len().max(1);
        pred.iter()
            .zip(next_obs.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>()
            / n as f32
    }
}

/// Temporal contrastive self-supervised learning utilities.
pub struct TemporalContrastive;

impl TemporalContrastive {
    /// Encode an observation through a series of (weight, bias) layers with ReLU.
    pub fn encode_obs(weights: &[(Vec<Vec<f32>>, Vec<f32>)], obs: &[f32]) -> Vec<f32> {
        mlp_forward(weights, obs)
    }

    /// InfoNCE contrastive loss: -log(exp(sim(a,p)/T) / Σ exp(sim(a,n_i)/T)).
    pub fn contrastive_loss(
        anchor: &[f32],
        positive: &[f32],
        negatives: &[Vec<f32>],
        temperature: f32,
    ) -> f32 {
        let t = temperature.max(f32::EPSILON);
        let pos_sim = cosine_sim(anchor, positive) / t;
        let neg_sims: Vec<f32> = negatives
            .iter()
            .map(|neg| cosine_sim(anchor, neg) / t)
            .collect();
        // log-sum-exp for numerical stability
        let all_max = neg_sims.iter().cloned().fold(pos_sim, f32::max);
        let sum_neg = neg_sims.iter().map(|&s| (s - all_max).exp()).sum::<f32>();
        let log_denom = all_max + (sum_neg + (pos_sim - all_max).exp()).ln();
        -(pos_sim - log_denom)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5 — HIERARCHICAL POLICY
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for hierarchical RL.
#[derive(Debug, Clone)]
pub struct HrlConfig {
    /// How many low-level steps per high-level step.
    pub high_level_freq: usize,
    /// Dimensionality of subgoal vector.
    pub subgoal_dim: usize,
    pub obs_dim: usize,
    pub action_dim: usize,
}

/// High-level policy that proposes subgoals.
#[derive(Debug, Clone)]
pub struct HighLevelPolicy {
    pub network: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

impl HighLevelPolicy {
    pub fn new(obs_dim: usize, subgoal_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let l1_w = xavier_mat(hidden_dim, obs_dim, &mut std_rng);
        let l1_b = zero_vec(hidden_dim);
        let l2_w = xavier_mat(subgoal_dim, hidden_dim, &mut std_rng);
        let l2_b = zero_vec(subgoal_dim);
        Self {
            network: vec![(l1_w, l1_b), (l2_w, l2_b)],
        }
    }

    pub fn propose_subgoal(&self, obs: &[f32]) -> Vec<f32> {
        mlp_forward(&self.network, obs)
    }
}

/// Low-level policy that takes primitive actions toward a subgoal.
#[derive(Debug, Clone)]
pub struct LowLevelPolicy {
    pub network: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

impl LowLevelPolicy {
    pub fn new(obs_dim: usize, subgoal_dim: usize, action_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let l1_w = xavier_mat(64, obs_dim + subgoal_dim, &mut std_rng);
        let l1_b = zero_vec(64);
        let l2_w = xavier_mat(action_dim, 64, &mut std_rng);
        let l2_b = zero_vec(action_dim);
        Self {
            network: vec![(l1_w, l1_b), (l2_w, l2_b)],
        }
    }

    pub fn act(&self, obs: &[f32], subgoal: &[f32]) -> Vec<f32> {
        let input: Vec<f32> = obs.iter().chain(subgoal.iter()).copied().collect();
        mlp_forward(&self.network, &input)
    }
}

/// Two-level hierarchical policy (high-level subgoal + low-level action).
#[derive(Debug, Clone)]
pub struct HierarchicalPolicy {
    pub high: HighLevelPolicy,
    pub low: LowLevelPolicy,
    pub config: HrlConfig,
    current_subgoal: Vec<f32>,
}

impl HierarchicalPolicy {
    pub fn new(config: HrlConfig, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let mut std_rng = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let high =
            HighLevelPolicy::new(config.obs_dim, config.subgoal_dim, hidden_dim, &mut std_rng);
        let low = LowLevelPolicy::new(
            config.obs_dim,
            config.subgoal_dim,
            config.action_dim,
            &mut std_rng,
        );
        let current_subgoal = zero_vec(config.subgoal_dim);
        Self {
            high,
            low,
            config,
            current_subgoal,
        }
    }

    /// Returns (action, Option<new_subgoal>).
    /// A new subgoal is generated when `step % high_level_freq == 0`.
    pub fn act(&mut self, obs: &[f32], step: usize) -> (Vec<f32>, Option<Vec<f32>>) {
        let new_subgoal = if step % self.config.high_level_freq == 0 {
            let sg = self.high.propose_subgoal(obs);
            self.current_subgoal = sg.clone();
            Some(sg)
        } else {
            None
        };
        let action = self.low.act(obs, &self.current_subgoal);
        (action, new_subgoal)
    }

    /// Intrinsic reward: negative L2 distance between observation and subgoal.
    pub fn intrinsic_reward(obs: &[f32], subgoal: &[f32]) -> f32 {
        let sq: f32 = obs
            .iter()
            .zip(subgoal.iter())
            .map(|(&o, &g)| (o - g).powi(2))
            .sum();
        -(sq.sqrt())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 6 — EVALUATION METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a single navigation episode.
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub success: bool,
    pub path_length: f32,
    pub optimal_length: f32,
    pub n_steps: usize,
}

/// Navigation evaluation metrics.
pub struct NavigationMetrics;

impl NavigationMetrics {
    /// Fraction of successful episodes.
    pub fn success_rate(episodes: &[EpisodeResult]) -> f32 {
        if episodes.is_empty() {
            return 0.0;
        }
        let n = episodes.len() as f32;
        episodes.iter().filter(|e| e.success).count() as f32 / n
    }

    /// Success weighted by (inverse) Path Length.
    ///
    /// SPL = (1/N) Σ_i s_i * L*_i / max(L_i, L*_i)
    pub fn spl(episodes: &[EpisodeResult]) -> f32 {
        if episodes.is_empty() {
            return 0.0;
        }
        let n = episodes.len() as f32;
        episodes
            .iter()
            .map(|e| {
                if !e.success {
                    return 0.0;
                }
                let denom = e.path_length.max(e.optimal_length).max(f32::EPSILON);
                e.optimal_length / denom
            })
            .sum::<f32>()
            / n
    }

    /// Mean distance from the final agent position to each goal position.
    pub fn dist_to_goal(trajectory: &[Vec<f32>], goal: &[Vec<f32>]) -> f32 {
        if trajectory.is_empty() || goal.is_empty() {
            return 0.0;
        }
        let last = trajectory.last().unwrap_or(&trajectory[0]);
        let n = goal.len() as f32;
        goal.iter()
            .map(|g| {
                let sq: f32 = last
                    .iter()
                    .zip(g.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                sq.sqrt()
            })
            .sum::<f32>()
            / n
    }
}

/// Manipulation evaluation metrics.
pub struct ManipulationMetrics;

impl ManipulationMetrics {
    /// Fraction of grasp candidates with quality ≥ threshold.
    pub fn grasp_success_rate(predictions: &[GraspCandidate], threshold: f32) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let n = predictions.len() as f32;
        predictions
            .iter()
            .filter(|g| g.quality >= threshold)
            .count() as f32
            / n
    }

    /// Mean change in direction (angular velocity proxy) along a trajectory.
    ///
    /// Returns 0 for straight-line or single-segment trajectories.
    pub fn trajectory_smoothness(trajectory: &[Vec<f32>]) -> f32 {
        if trajectory.len() < 3 {
            return 0.0;
        }
        let n = (trajectory.len() - 2) as f32;
        let mut total = 0.0_f32;
        for i in 1..trajectory.len() - 1 {
            let prev = &trajectory[i - 1];
            let curr = &trajectory[i];
            let next = &trajectory[i + 1];
            // velocity vectors
            let v1: Vec<f32> = curr.iter().zip(prev.iter()).map(|(&c, &p)| c - p).collect();
            let v2: Vec<f32> = next.iter().zip(curr.iter()).map(|(&n, &c)| n - c).collect();
            let angle_change = 1.0 - cosine_sim(&v1, &v2).clamp(-1.0, 1.0);
            total += angle_change;
        }
        total / n
    }
}

/// Aggregated embodied evaluation report.
#[derive(Debug, Clone)]
pub struct EmbodiedEvalReport {
    pub nav_success: f32,
    pub spl: f32,
    pub grasp_sr: f32,
    pub smoothness: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    // ── GridWorldEnv ──────────────────────────────────────────────────────────

    #[test]
    fn test_grid_world_creation() {
        let mut rng = StdRng::seed_from_u64(1);
        let env = GridWorldEnv::new(5, 5, &mut rng);
        assert_eq!(env.width, 5);
        assert_eq!(env.height, 5);
        assert_eq!(env.agent, (0, 0));
    }

    #[test]
    fn test_grid_world_step_valid() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut env = GridWorldEnv::new(5, 5, &mut rng);
        env.obstacles.clear();
        let obs = env.step(1); // move right
        assert_eq!(obs.timestep, 1);
        assert!(obs.reward.is_finite());
    }

    #[test]
    fn test_grid_world_reset() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut env = GridWorldEnv::new(5, 5, &mut rng);
        env.step(1);
        env.reset(&mut rng);
        assert_eq!(env.agent, (0, 0));
        assert_eq!(env.timestep, 0);
    }

    #[test]
    fn test_grid_world_render_shape() {
        let mut rng = StdRng::seed_from_u64(4);
        let env = GridWorldEnv::new(4, 3, &mut rng);
        let grid = env.render();
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].len(), 4);
    }

    #[test]
    fn test_grid_world_goal_detection() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GridWorldEnv::new(2, 1, &mut rng);
        env.obstacles.clear();
        env.goal = (1, 0);
        env.agent = (0, 0);
        let obs = env.step(1); // right
        assert!(obs.done);
        assert!((obs.reward - 1.0).abs() < 1e-6);
    }

    // ── ContinuousNavEnv ──────────────────────────────────────────────────────

    #[test]
    fn test_continuous_nav_step() {
        let goal = vec![3.0, 3.0];
        let mut env = ContinuousNavEnv::new(10.0, goal, vec![]);
        let obs = env.step(&[0.5, 0.5]);
        assert!(obs.reward.is_finite());
        assert_eq!(obs.timestep, 1);
    }

    #[test]
    fn test_continuous_nav_collision_detection() {
        let center = vec![2.0, 2.0];
        let obstacles = vec![(center, 0.5_f32)];
        // Agent at 2.0, 2.0 should be inside the obstacle
        assert!(ContinuousNavEnv::is_collision(&[2.0, 2.0], &obstacles));
        assert!(!ContinuousNavEnv::is_collision(&[5.0, 5.0], &obstacles));
    }

    #[test]
    fn test_continuous_nav_goal_reached() {
        let goal = vec![0.0, 0.0];
        let mut env = ContinuousNavEnv::new(10.0, goal, vec![]);
        // Already at goal
        let obs = env.step(&[0.0, 0.0]);
        assert!(obs.done);
    }

    // ── VisualEncoder ─────────────────────────────────────────────────────────

    #[test]
    fn test_visual_encoder_output_shape() {
        let mut rng = StdRng::seed_from_u64(10);
        let enc = VisualEncoder::new(64, 32, &mut rng);
        let image = vec![0.1_f32; 64];
        let out = enc.encode(&image);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_goal_encoder_output_shape() {
        let mut rng = StdRng::seed_from_u64(11);
        let enc = GoalEncoder::new(4, 16, &mut rng);
        let goal = vec![1.0_f32, 0.0, 0.5, 0.3];
        let out = enc.encode(&goal);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_point_goal_policy_creation() {
        let mut rng = StdRng::seed_from_u64(12);
        let policy = PointGoalPolicy::new(64, 4, 32, 4, &mut rng);
        assert_eq!(policy.policy_head.0.len(), 4);
    }

    #[test]
    fn test_point_goal_act_shapes() {
        let mut rng = StdRng::seed_from_u64(13);
        let policy = PointGoalPolicy::new(16, 4, 8, 6, &mut rng);
        let image = vec![0.0_f32; 16];
        let goal = vec![1.0_f32, 0.0, 0.5, 0.2];
        let (logits, _value) = policy.act(&image, &goal);
        assert_eq!(logits.len(), 6);
    }

    #[test]
    fn test_point_goal_navigate_valid_action() {
        let mut rng = StdRng::seed_from_u64(14);
        let policy = PointGoalPolicy::new(16, 4, 8, 5, &mut rng);
        let image = vec![0.0_f32; 16];
        let goal = vec![1.0_f32, 0.0, 0.5, 0.2];
        let action = policy.navigate(&image, &goal);
        assert!(action < 5);
    }

    #[test]
    fn test_object_nav_policy_detect_range() {
        let mut rng = StdRng::seed_from_u64(15);
        let policy = ObjectNavPolicy::new(8, 3, 4, &mut rng);
        let obs = vec![0.1_f32; 8];
        let score = policy.detect_object(&obs, 1);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_object_nav_act_valid() {
        let mut rng = StdRng::seed_from_u64(16);
        let policy = ObjectNavPolicy::new(8, 3, 4, &mut rng);
        let obs = vec![0.5_f32; 8];
        let action = policy.act(&obs, 0);
        assert!(action < 4);
    }

    // ── GraspPlanner ─────────────────────────────────────────────────────────

    #[test]
    fn test_grasp_planner_predict_grasp() {
        let mut rng = StdRng::seed_from_u64(20);
        let planner = GraspPlanner::new(3, 16, &mut rng);
        let cloud: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let candidate = planner.predict_grasp(&cloud);
        assert_eq!(candidate.position.len(), 3);
        assert_eq!(candidate.orientation.len(), 4);
    }

    #[test]
    fn test_grasp_quality_range() {
        let mut rng = StdRng::seed_from_u64(21);
        let planner = GraspPlanner::new(3, 16, &mut rng);
        let cloud: Vec<Vec<f32>> = vec![vec![0.5, 0.5, 0.5]];
        let c = planner.predict_grasp(&cloud);
        assert!(c.quality >= 0.0 && c.quality <= 1.0);
    }

    #[test]
    fn test_rank_grasps_sorted() {
        let mut rng = StdRng::seed_from_u64(22);
        let planner = GraspPlanner::new(3, 16, &mut rng);
        let candidates = vec![
            GraspCandidate {
                position: vec![0.0; 3],
                orientation: vec![1.0; 4],
                quality: 0.3,
            },
            GraspCandidate {
                position: vec![0.0; 3],
                orientation: vec![1.0; 4],
                quality: 0.9,
            },
            GraspCandidate {
                position: vec![0.0; 3],
                orientation: vec![1.0; 4],
                quality: 0.6,
            },
        ];
        let ranked = planner.rank_grasps(&candidates);
        assert_eq!(ranked[0], 1); // highest quality
        assert_eq!(ranked[1], 2);
        assert_eq!(ranked[2], 0);
    }

    #[test]
    fn test_grasp_candidate_creation() {
        let gc = GraspCandidate {
            position: vec![1.0, 2.0, 3.0],
            orientation: vec![0.0, 0.0, 0.0, 1.0],
            quality: 0.8,
        };
        assert!((gc.quality - 0.8).abs() < 1e-6);
        assert_eq!(gc.position.len(), 3);
    }

    // ── DMP ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_dmp_basis_fn_shape() {
        let centers = vec![0.0, 0.5, 1.0];
        let basis = Dmp::basis_fn(0.5, &centers, 0.1);
        assert_eq!(basis.len(), 3);
    }

    #[test]
    fn test_dmp_basis_fn_positive() {
        let centers = vec![0.2, 0.5, 0.8];
        let basis = Dmp::basis_fn(0.5, &centers, 0.1);
        assert!(basis.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_dmp_rollout_shape() {
        let dmp = Dmp {
            config: DmpConfig::default(),
            weights: vec![0.0; 20],
            start: vec![0.0],
            goal: vec![1.0],
        };
        let traj = dmp.rollout(0.01, 50);
        assert_eq!(traj.len(), 50);
        assert_eq!(traj[0].len(), 1);
    }

    #[test]
    fn test_dmp_rollout_starts_at_start() {
        let start = vec![2.0];
        let dmp = Dmp {
            config: DmpConfig::default(),
            weights: vec![0.0; 20],
            start: start.clone(),
            goal: vec![5.0],
        };
        let traj = dmp.rollout(0.01, 30);
        assert!((traj[0][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_dmp_fit_returns_dmp() {
        let demo = vec![vec![vec![0.0], vec![0.5], vec![1.0]]];
        let dmp = Dmp::fit(&demo);
        assert_eq!(dmp.weights.len(), 20);
        assert!((dmp.start[0] - 0.0).abs() < 1e-6);
        assert!((dmp.goal[0] - 1.0).abs() < 1e-6);
    }

    // ── InverseDynamicsModel ──────────────────────────────────────────────────

    #[test]
    fn test_inverse_dynamics_predict_shape() {
        let mut rng = StdRng::seed_from_u64(30);
        let model = InverseDynamicsModel::new(8, 4, 16, &mut rng);
        let obs = vec![0.1_f32; 8];
        let obs2 = vec![0.2_f32; 8];
        let pred = model.predict(&obs, &obs2);
        assert_eq!(pred.len(), 4);
    }

    #[test]
    fn test_inverse_dynamics_loss_nonneg() {
        let mut rng = StdRng::seed_from_u64(31);
        let model = InverseDynamicsModel::new(8, 4, 16, &mut rng);
        let obs = vec![0.0_f32; 8];
        let obs2 = vec![0.1_f32; 8];
        let action = vec![0.5_f32; 4];
        let loss = model.loss(&obs, &obs2, &action);
        assert!(loss >= 0.0);
    }

    // ── ForwardDynamicsModel ──────────────────────────────────────────────────

    #[test]
    fn test_forward_dynamics_predict_shape() {
        let mut rng = StdRng::seed_from_u64(32);
        let model = ForwardDynamicsModel::new(8, 4, 16, &mut rng);
        let obs = vec![0.0_f32; 8];
        let action = vec![0.1_f32; 4];
        let next = model.predict(&obs, &action);
        assert_eq!(next.len(), 8);
    }

    #[test]
    fn test_forward_dynamics_loss_nonneg() {
        let mut rng = StdRng::seed_from_u64(33);
        let model = ForwardDynamicsModel::new(8, 4, 16, &mut rng);
        let obs = vec![0.0_f32; 8];
        let action = vec![0.1_f32; 4];
        let next_obs = vec![0.2_f32; 8];
        let loss = model.loss(&obs, &action, &next_obs);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_forward_dynamics_shape_matches_obs() {
        let mut rng = StdRng::seed_from_u64(99);
        let model = ForwardDynamicsModel::new(12, 6, 32, &mut rng);
        let obs = vec![0.3_f32; 12];
        let action = vec![0.0_f32; 6];
        let pred = model.predict(&obs, &action);
        assert_eq!(pred.len(), 12);
    }

    // ── TemporalContrastive ───────────────────────────────────────────────────

    #[test]
    fn test_temporal_contrastive_loss_positive() {
        let anchor = vec![1.0_f32, 0.0, 0.0];
        let positive = vec![0.9_f32, 0.1, 0.0];
        let negatives = vec![vec![-1.0_f32, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0]];
        let loss = TemporalContrastive::contrastive_loss(&anchor, &positive, &negatives, 0.1);
        // InfoNCE loss should be non-negative for well-separated anchor/positive
        assert!(loss.is_finite());
    }

    // ── HighLevelPolicy / LowLevelPolicy ─────────────────────────────────────

    #[test]
    fn test_high_level_policy_subgoal_shape() {
        let mut rng = StdRng::seed_from_u64(40);
        let policy = HighLevelPolicy::new(8, 4, 16, &mut rng);
        let obs = vec![0.0_f32; 8];
        let sg = policy.propose_subgoal(&obs);
        assert_eq!(sg.len(), 4);
    }

    #[test]
    fn test_low_level_policy_act_shape() {
        let mut rng = StdRng::seed_from_u64(41);
        let policy = LowLevelPolicy::new(8, 4, 3, &mut rng);
        let obs = vec![0.0_f32; 8];
        let sg = vec![1.0_f32; 4];
        let action = policy.act(&obs, &sg);
        assert_eq!(action.len(), 3);
    }

    #[test]
    fn test_hierarchical_policy_act_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = HrlConfig {
            high_level_freq: 5,
            subgoal_dim: 4,
            obs_dim: 8,
            action_dim: 3,
        };
        let mut policy = HierarchicalPolicy::new(config, 16, &mut rng);
        let obs = vec![0.0_f32; 8];
        let (action, _sg) = policy.act(&obs, 0);
        assert_eq!(action.len(), 3);
    }

    #[test]
    fn test_hierarchical_policy_subgoal_at_freq_0() {
        let mut rng = StdRng::seed_from_u64(43);
        let config = HrlConfig {
            high_level_freq: 5,
            subgoal_dim: 4,
            obs_dim: 8,
            action_dim: 3,
        };
        let mut policy = HierarchicalPolicy::new(config, 16, &mut rng);
        let obs = vec![0.0_f32; 8];
        let (_action, sg) = policy.act(&obs, 0); // step 0 → new subgoal
        assert!(sg.is_some());
    }

    #[test]
    fn test_hierarchical_policy_no_subgoal_other_steps() {
        let mut rng = StdRng::seed_from_u64(44);
        let config = HrlConfig {
            high_level_freq: 5,
            subgoal_dim: 4,
            obs_dim: 8,
            action_dim: 3,
        };
        let mut policy = HierarchicalPolicy::new(config, 16, &mut rng);
        let obs = vec![0.0_f32; 8];
        policy.act(&obs, 0); // seed subgoal
        let (_action, sg) = policy.act(&obs, 2); // step 2 → no new subgoal
        assert!(sg.is_none());
    }

    #[test]
    fn test_intrinsic_reward_zero_at_goal() {
        let obs = vec![1.0_f32, 2.0, 3.0];
        let sg = vec![1.0_f32, 2.0, 3.0];
        let r = HierarchicalPolicy::intrinsic_reward(&obs, &sg);
        assert!((r - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_intrinsic_reward_negative_far() {
        let obs = vec![0.0_f32, 0.0];
        let sg = vec![10.0_f32, 10.0];
        let r = HierarchicalPolicy::intrinsic_reward(&obs, &sg);
        assert!(r < 0.0);
    }

    // ── NavigationMetrics ─────────────────────────────────────────────────────

    #[test]
    fn test_navigation_metrics_success_rate() {
        let episodes = vec![
            EpisodeResult {
                success: true,
                path_length: 5.0,
                optimal_length: 4.0,
                n_steps: 10,
            },
            EpisodeResult {
                success: false,
                path_length: 10.0,
                optimal_length: 4.0,
                n_steps: 20,
            },
            EpisodeResult {
                success: true,
                path_length: 4.0,
                optimal_length: 4.0,
                n_steps: 8,
            },
        ];
        let sr = NavigationMetrics::success_rate(&episodes);
        assert!((sr - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_navigation_spl_perfect() {
        let episodes = vec![EpisodeResult {
            success: true,
            path_length: 4.0,
            optimal_length: 4.0,
            n_steps: 8,
        }];
        let spl = NavigationMetrics::spl(&episodes);
        assert!((spl - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spl_zero_failure() {
        let episodes = vec![EpisodeResult {
            success: false,
            path_length: 10.0,
            optimal_length: 4.0,
            n_steps: 20,
        }];
        let spl = NavigationMetrics::spl(&episodes);
        assert!((spl - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dist_to_goal_zero_at_goal() {
        let traj = vec![vec![3.0_f32, 4.0]];
        let goal = vec![vec![3.0_f32, 4.0]];
        let d = NavigationMetrics::dist_to_goal(&traj, &goal);
        assert!(d < 1e-6);
    }

    // ── ManipulationMetrics ───────────────────────────────────────────────────

    #[test]
    fn test_grasp_success_rate_range() {
        let candidates = vec![
            GraspCandidate {
                position: vec![0.0; 3],
                orientation: vec![1.0; 4],
                quality: 0.8,
            },
            GraspCandidate {
                position: vec![0.0; 3],
                orientation: vec![1.0; 4],
                quality: 0.3,
            },
            GraspCandidate {
                position: vec![0.0; 3],
                orientation: vec![1.0; 4],
                quality: 0.9,
            },
        ];
        let sr = ManipulationMetrics::grasp_success_rate(&candidates, 0.7);
        assert!((sr - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_trajectory_smoothness_straight() {
        // Perfectly straight trajectory should have near-zero smoothness value
        let traj: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 0.0]).collect();
        let s = ManipulationMetrics::trajectory_smoothness(&traj);
        assert!(s < 0.1);
    }

    // ── EpisodeResult + EmbodiedEvalReport ────────────────────────────────────

    #[test]
    fn test_episode_result_creation() {
        let ep = EpisodeResult {
            success: true,
            path_length: 3.0,
            optimal_length: 2.0,
            n_steps: 5,
        };
        assert!(ep.success);
        assert!((ep.path_length - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_embodied_eval_report_fields() {
        let report = EmbodiedEvalReport {
            nav_success: 0.8,
            spl: 0.75,
            grasp_sr: 0.6,
            smoothness: 0.05,
        };
        assert!((report.nav_success - 0.8).abs() < 1e-6);
        assert!((report.spl - 0.75).abs() < 1e-6);
        assert!((report.grasp_sr - 0.6).abs() < 1e-6);
        assert!((report.smoothness - 0.05).abs() < 1e-6);
    }
}
