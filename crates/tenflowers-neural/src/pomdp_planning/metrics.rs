//! POMDP evaluation metrics and benchmark generators

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

use super::types::{BeliefState, PomdpAction, PomdpModel};

// ─── 13. PomdpMetrics ─────────────────────────────────────────────────────────

/// Evaluation metrics for POMDP policies.
pub struct PomdpMetrics;

impl PomdpMetrics {
    /// Estimate expected discounted reward by simulating `n_episodes` episodes.
    pub fn expected_discounted_reward<F>(
        policy: F,
        model: &PomdpModel,
        n_episodes: usize,
        horizon: usize,
        rng: &mut StdRng,
    ) -> f64
    where
        F: Fn(&BeliefState) -> PomdpAction,
    {
        let mut total = 0.0;
        for _ in 0..n_episodes {
            let mut b = BeliefState::uniform(model.n_states);
            let mut s = b.sample(rng);
            let mut discount = 1.0;
            let mut episode_reward = 0.0;

            for _ in 0..horizon {
                let a = policy(&b);
                episode_reward += discount * model.reward[s][a];
                discount *= model.gamma;

                // Transition
                let u: f64 = rng.random();
                let mut cum = 0.0;
                let mut sp = model.n_states - 1;
                for (spp, &p) in model.transition[s][a].iter().enumerate() {
                    cum += p;
                    if u <= cum {
                        sp = spp;
                        break;
                    }
                }

                // Observation
                let uo: f64 = rng.random();
                let mut cum_o = 0.0;
                let mut o = model.n_obs - 1;
                for (oo, &p) in model.observation[a][sp].iter().enumerate() {
                    cum_o += p;
                    if uo <= cum_o {
                        o = oo;
                        break;
                    }
                }

                b = b.update(a, o, model);
                s = sp;
            }

            total += episode_reward;
        }
        total / n_episodes as f64
    }

    /// Average entropy of the belief state over a simulated trajectory.
    pub fn belief_state_entropy(model: &PomdpModel, horizon: usize, rng: &mut StdRng) -> f64 {
        let mut b = BeliefState::uniform(model.n_states);
        let mut total_entropy = b.entropy();
        for _ in 0..horizon {
            let a = rng.random_range(0..model.n_actions);
            let s = b.sample(rng);
            let u: f64 = rng.random();
            let mut cum = 0.0;
            let mut sp = model.n_states - 1;
            for (spp, &p) in model.transition[s][a].iter().enumerate() {
                cum += p;
                if u <= cum {
                    sp = spp;
                    break;
                }
            }
            let uo: f64 = rng.random();
            let mut cum_o = 0.0;
            let mut o = model.n_obs - 1;
            for (oo, &p) in model.observation[a][sp].iter().enumerate() {
                cum_o += p;
                if uo <= cum_o {
                    o = oo;
                    break;
                }
            }
            b = b.update(a, o, model);
            total_entropy += b.entropy();
        }
        total_entropy / (horizon + 1) as f64
    }

    /// Approximate policy loss vs. a reference policy (measured in discounted reward difference).
    pub fn policy_loss_vs_optimal<F1, F2>(
        policy: F1,
        reference: F2,
        model: &PomdpModel,
        n_episodes: usize,
        horizon: usize,
        rng: &mut StdRng,
    ) -> f64
    where
        F1: Fn(&BeliefState) -> PomdpAction,
        F2: Fn(&BeliefState) -> PomdpAction,
    {
        use scirs2_core::random::SeedableRng;
        let mut rng2 = StdRng::seed_from_u64(rng.random_range(0_u64..u64::MAX));
        let policy_reward =
            Self::expected_discounted_reward(&policy, model, n_episodes, horizon, rng);
        let ref_reward =
            Self::expected_discounted_reward(&reference, model, n_episodes, horizon, &mut rng2);
        ref_reward - policy_reward
    }
}

// ─── 14. PomdpGenerator ───────────────────────────────────────────────────────

/// Classic POMDP benchmark generators.
pub struct PomdpGenerator;

impl PomdpGenerator {
    /// Tiger problem (Cassandra et al. 1994).
    ///
    /// 2 states: Tiger-Left (0) and Tiger-Right (1).
    /// 3 actions: Listen (0), Open-Left (1), Open-Right (2).
    /// 2 observations: Hear-Left (0), Hear-Right (1).
    pub fn tiger_problem() -> PomdpModel {
        let mut m = PomdpModel::new(2, 3, 2, 0.95);

        // Rewards
        // Listen
        m.set_reward(0, 0, -1.0);
        m.set_reward(1, 0, -1.0);
        // Open-Left: tiger on left → -100, tiger on right → +10
        m.set_reward(0, 1, -100.0);
        m.set_reward(1, 1, 10.0);
        // Open-Right: tiger on left → +10, tiger on right → -100
        m.set_reward(0, 2, 10.0);
        m.set_reward(1, 2, -100.0);

        // Transitions: opening a door resets (uniform); listening keeps state
        // Tiger-Left stays Tiger-Left when listening
        m.set_transition(0, 0, 0, 1.0);
        m.set_transition(1, 0, 1, 1.0);
        // Opening a door: uniform reset
        for s in 0..2 {
            for a in 1..3 {
                m.set_transition(s, a, 0, 0.5);
                m.set_transition(s, a, 1, 0.5);
            }
        }

        // Observations: listen gives 85% correct; opening gives 50/50
        // Listen action (a=0): Z[0][s'][o]
        m.set_observation(0, 0, 0, 0.85); // tiger left → hear left
        m.set_observation(0, 0, 1, 0.15);
        m.set_observation(0, 1, 0, 0.15); // tiger right → hear left with low prob
        m.set_observation(0, 1, 1, 0.85);
        // Open actions: 50/50
        for a in 1..3 {
            for s in 0..2 {
                m.set_observation(a, s, 0, 0.5);
                m.set_observation(a, s, 1, 0.5);
            }
        }

        m
    }

    /// Grid maze POMDP: noisy grid world with partial observability.
    ///
    /// States: n*m grid cells, Actions: 4 cardinal directions, Observations: 4 (wall config).
    pub fn grid_maze_pomdp(n: usize, m: usize) -> PomdpModel {
        let n_states = n * m;
        let n_actions = 4; // N=0, E=1, S=2, W=3
        let n_obs = 4; // Simplified: distance to walls
        let mut model = PomdpModel::new(n_states, n_actions, n_obs, 0.95);

        let idx = |r: usize, c: usize| r * m + c;
        let moves = [(0_i32, -1_i32), (0, 1), (-1, 0), (1, 0)]; // N, E, S, W

        // Transitions: 0.8 success, 0.2 stays
        for r in 0..n {
            for c in 0..m {
                let s = idx(r, c);
                for (a, &(dr, dc)) in moves.iter().enumerate() {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < n as i32 && nc >= 0 && nc < m as i32 {
                        let sp = idx(nr as usize, nc as usize);
                        model.set_transition(s, a, sp, 0.8);
                        model.set_transition(s, a, s, 0.2);
                    } else {
                        model.set_transition(s, a, s, 1.0);
                    }
                }
            }
        }

        // Observations: noisy position modulo n_obs
        let obs_correct_prob = 0.7;
        for a in 0..n_actions {
            for sp in 0..n_states {
                let true_obs = sp % n_obs;
                for o in 0..n_obs {
                    let prob = if o == true_obs {
                        obs_correct_prob
                    } else {
                        (1.0 - obs_correct_prob) / (n_obs - 1) as f64
                    };
                    model.set_observation(a, sp, o, prob);
                }
            }
        }

        // Rewards: +10 at goal (last state), -1 elsewhere
        for s in 0..n_states {
            for a in 0..n_actions {
                let r = if s == n_states - 1 { 10.0 } else { -1.0 };
                model.set_reward(s, a, r);
            }
        }

        model
    }

    /// Rock sample POMDP (Smith & Simmons 2004).
    ///
    /// Agent on a 1D line with `n_rocks` rocks. Actions: move left (0), move right (1),
    /// sample at current position (2), sense rock k (3+k).
    pub fn rock_sample_pomdp(n_rocks: usize) -> PomdpModel {
        // States: (position 0..n_rocks+1) × (rock quality bitfield 0..2^n_rocks)
        // Simplified: position only, rocks encoded as binary state offset
        let n_positions = n_rocks + 2; // 0..n_rocks+1
        let n_rock_configs = 1_usize << n_rocks.min(8); // cap at 2^8 for tractability
        let n_states = n_positions * n_rock_configs;
        let n_actions = 3 + n_rocks; // left, right, sample, sense_0..sense_{n-1}
        let n_obs = 3; // null (0), good rock (1), bad rock (2)
        let mut model = PomdpModel::new(n_states, n_actions, n_obs, 0.95);

        let state_idx = |pos: usize, rocks: usize| pos * n_rock_configs + rocks;

        for pos in 0..n_positions {
            for rocks in 0..n_rock_configs {
                let s = state_idx(pos, rocks);

                // Move left (a=0)
                if pos > 0 {
                    model.set_transition(s, 0, state_idx(pos - 1, rocks), 1.0);
                } else {
                    model.set_transition(s, 0, s, 1.0);
                }

                // Move right (a=1)
                if pos + 1 < n_positions {
                    model.set_transition(s, 1, state_idx(pos + 1, rocks), 1.0);
                    // Exiting right = terminal
                    if pos + 1 == n_positions - 1 {
                        model.set_reward(s, 1, 10.0);
                    }
                } else {
                    model.set_transition(s, 1, s, 1.0);
                }

                // Sample (a=2): sample rock at current position (if any)
                let rock_bit = if pos < n_rocks { 1 << pos } else { 0 };
                let rock_good = (rocks & rock_bit) != 0;
                if pos < n_rocks {
                    let reward = if rock_good { 10.0 } else { -10.0 };
                    model.set_reward(s, 2, reward);
                    // After sampling, rock becomes bad
                    let new_rocks = rocks & !rock_bit;
                    model.set_transition(s, 2, state_idx(pos, new_rocks), 1.0);
                } else {
                    model.set_transition(s, 2, s, 1.0);
                }

                // Sense actions (a = 3..3+n_rocks)
                for k in 0..n_rocks {
                    let a = 3 + k;
                    if a < n_actions {
                        model.set_transition(s, a, s, 1.0);
                        model.set_reward(s, a, 0.0);
                    }
                }

                // Default: all remaining transitions stay
                for a in 0..n_actions {
                    let t_sum: f64 = (0..n_states).map(|sp| model.transition[s][a][sp]).sum();
                    if t_sum < 1e-10 {
                        model.set_transition(s, a, s, 1.0);
                    }
                }
            }
        }

        // Observations: sense gives good/bad rock info; others null
        for a in 0..3 {
            for sp in 0..n_states {
                model.set_observation(a, sp, 0, 1.0); // null obs for move/sample
            }
        }
        for k in 0..n_rocks {
            let a = 3 + k;
            if a < n_actions {
                for pos in 0..n_positions {
                    for rocks in 0..n_rock_configs {
                        let sp = state_idx(pos, rocks);
                        let rock_bit = 1_usize << k;
                        let rock_good = (rocks & rock_bit) != 0;
                        // Efficiency drops with distance
                        let dist = pos.abs_diff(k);
                        let correct_prob = 0.5 + 0.5 * (-(dist as f64) / 2.0).exp();
                        if rock_good {
                            model.set_observation(a, sp, 1, correct_prob);
                            model.set_observation(a, sp, 2, 1.0 - correct_prob);
                        } else {
                            model.set_observation(a, sp, 1, 1.0 - correct_prob);
                            model.set_observation(a, sp, 2, correct_prob);
                        }
                    }
                }
            }
        }

        // Reward: -1 step penalty for moves
        for s in 0..n_states {
            model.set_reward(s, 0, -1.0); // move left
        }

        model
    }
}
