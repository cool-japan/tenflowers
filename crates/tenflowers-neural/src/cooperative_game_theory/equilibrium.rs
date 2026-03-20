//! Equilibrium concepts: CorrelatedEquilibrium, MechanismDesign, MeanFieldEquilibrium.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::types::{CgtError, CgtGame, NashEquilibriumSolver};

// ─────────────────────────────────────────────────────────────────────────────
// §9 CorrelatedEquilibrium
// ─────────────────────────────────────────────────────────────────────────────

/// Correlated equilibrium verification and finding.
pub struct CorrelatedEquilibrium;

impl CorrelatedEquilibrium {
    /// Verify that a joint distribution sigma over action profiles is a correlated equilibrium.
    /// sigma is indexed by flat profile index.
    pub fn is_correlated_equilibrium(sigma: &[f64], game: &CgtGame) -> bool {
        let profiles = game.all_profiles();
        if sigma.len() != profiles.len() {
            return false;
        }
        if (sigma.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
            return false;
        }
        if sigma.iter().any(|&s| s < -1e-9) {
            return false;
        }

        for player in 0..game.n_players {
            let n_a = game.actions_per_player[player];
            for ai in 0..n_a {
                for ai2 in 0..n_a {
                    if ai2 == ai {
                        continue;
                    }
                    let mut improvement = 0.0f64;
                    for (idx, prof) in profiles.iter().enumerate() {
                        if prof[player] != ai {
                            continue;
                        }
                        let u_current = game.payoff(player, prof).unwrap_or(0.0);
                        let mut deviated = prof.clone();
                        deviated[player] = ai2;
                        let u_deviated = game.payoff(player, &deviated).unwrap_or(0.0);
                        improvement += sigma[idx] * (u_deviated - u_current);
                    }
                    if improvement > 1e-6 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Find a correlated equilibrium maximising social welfare via linear programming (gradient ascent).
    pub fn find_maximal_welfare_ce(game: &CgtGame) -> Result<Vec<f64>, CgtError> {
        let profiles = game.all_profiles();
        let n_profiles = profiles.len();
        if n_profiles == 0 {
            return Err(CgtError::InvalidInput("Game has no profiles".into()));
        }

        let nash_strategies = NashEquilibriumSolver::find_nash_support(game)?;
        let mut sigma = vec![0.0f64; n_profiles];
        for (idx, prof) in profiles.iter().enumerate() {
            let mut prob = 1.0f64;
            for (p, &a) in prof.iter().enumerate() {
                if p < nash_strategies.len() && a < nash_strategies[p].len() {
                    prob *= nash_strategies[p][a];
                }
            }
            sigma[idx] = prob;
        }

        for _iter in 0..500 {
            let mut grad = vec![0.0f64; n_profiles];
            for (idx, prof) in profiles.iter().enumerate() {
                let sw: f64 = (0..game.n_players)
                    .map(|p| game.payoff(p, prof).unwrap_or(0.0))
                    .sum();
                grad[idx] = sw;
            }
            let lr = 0.001;
            for (s, &g) in sigma.iter_mut().zip(grad.iter()) {
                *s += lr * g;
                *s = s.max(0.0);
            }
            project_onto_simplex(&mut sigma);
            ce_project(&mut sigma, game, &profiles);
        }

        Ok(sigma)
    }
}

pub(crate) fn project_onto_simplex(v: &mut [f64]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut rho = 0usize;
    let mut cumsum = 0.0f64;
    for i in 0..n {
        cumsum += sorted[i];
        if sorted[i] - (cumsum - 1.0) / (i as f64 + 1.0) > 0.0 {
            rho = i;
        }
    }
    let theta = {
        let cum: f64 = sorted[..=rho].iter().sum::<f64>();
        (cum - 1.0) / (rho as f64 + 1.0)
    };
    for x in v.iter_mut() {
        *x = (*x - theta).max(0.0);
    }
}

fn ce_project(sigma: &mut [f64], game: &CgtGame, profiles: &[Vec<usize>]) {
    for player in 0..game.n_players {
        let n_a = game.actions_per_player[player];
        for ai in 0..n_a {
            for ai2 in 0..n_a {
                if ai2 == ai {
                    continue;
                }
                let mut violation = 0.0f64;
                for (idx, prof) in profiles.iter().enumerate() {
                    if prof[player] != ai {
                        continue;
                    }
                    let u_current = game.payoff(player, prof).unwrap_or(0.0);
                    let mut deviated = prof.clone();
                    deviated[player] = ai2;
                    let u_deviated = game.payoff(player, &deviated).unwrap_or(0.0);
                    violation += sigma[idx] * (u_deviated - u_current);
                }
                if violation > 0.0 {
                    for (idx, prof) in profiles.iter().enumerate() {
                        if prof[player] != ai {
                            continue;
                        }
                        let u_c = game.payoff(player, prof).unwrap_or(0.0);
                        let mut dev = prof.clone();
                        dev[player] = ai2;
                        let u_d = game.payoff(player, &dev).unwrap_or(0.0);
                        if u_d > u_c {
                            sigma[idx] = (sigma[idx] - 0.001 * violation).max(0.0);
                        }
                    }
                }
            }
        }
    }
    project_onto_simplex(sigma);
}

// ─────────────────────────────────────────────────────────────────────────────
// §10 MechanismDesign — VCG Mechanism
// ─────────────────────────────────────────────────────────────────────────────

/// VCG (Vickrey-Clarke-Groves) mechanism.
pub struct MechanismDesign;

impl MechanismDesign {
    /// Compute VCG allocation and payments.
    /// `bids[i]` is agent i's valuation vector over possible allocations (1-indexed positions).
    /// Returns (allocation per agent, payments per agent).
    pub fn vcg_allocation(
        bids: &[Vec<f64>],
        value_fn: impl Fn(&[usize]) -> f64,
    ) -> Result<(Vec<usize>, Vec<f64>), CgtError> {
        let n_agents = bids.len();
        if n_agents == 0 {
            return Err(CgtError::InvalidInput("No agents".into()));
        }
        let n_items = bids[0].len();

        let alloc = Self::welfare_max_allocation(bids, &value_fn, n_agents, n_items)?;
        let social_welfare = value_fn(&alloc);

        let mut payments = vec![0.0f64; n_agents];
        for i in 0..n_agents {
            let bids_without_i: Vec<Vec<f64>> = bids
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, b)| b.clone())
                .collect();
            let alloc_without_i =
                Self::welfare_max_allocation(&bids_without_i, &value_fn, n_agents - 1, n_items)?;

            let welfare_without_i: f64 = bids_without_i
                .iter()
                .enumerate()
                .map(|(j_idx, b)| {
                    let a = alloc_without_i.get(j_idx).copied().unwrap_or(0);
                    b.get(a).copied().unwrap_or(0.0)
                })
                .sum();

            let v_i = bids[i].get(alloc[i]).copied().unwrap_or(0.0);
            let _ = social_welfare;
            payments[i] = welfare_without_i
                - (bids
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(j, b)| b.get(alloc[j]).copied().unwrap_or(0.0))
                    .sum::<f64>());
            let _ = v_i;
        }

        Ok((alloc, payments))
    }

    fn welfare_max_allocation(
        bids: &[Vec<f64>],
        _value_fn: &impl Fn(&[usize]) -> f64,
        n_agents: usize,
        n_items: usize,
    ) -> Result<Vec<usize>, CgtError> {
        if n_agents == 0 {
            return Ok(vec![]);
        }
        let mut alloc = vec![0usize; n_agents];
        for (i, bid) in bids.iter().enumerate().take(n_agents) {
            let best = bid
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx.min(n_items.saturating_sub(1)))
                .unwrap_or(0);
            alloc[i] = best;
        }
        Ok(alloc)
    }

    /// Check incentive compatibility of the allocation rule.
    pub fn is_incentive_compatible(bids: &[Vec<f64>]) -> bool {
        let n_agents = bids.len();
        if n_agents == 0 {
            return true;
        }
        let identity_fn = |alloc: &[usize]| -> f64 {
            bids.iter()
                .enumerate()
                .map(|(i, b)| {
                    b.get(alloc.get(i).copied().unwrap_or(0))
                        .copied()
                        .unwrap_or(0.0)
                })
                .sum()
        };

        let (alloc, payments) = match Self::vcg_allocation(bids, identity_fn) {
            Ok(r) => r,
            Err(_) => return false,
        };
        let utilities: Vec<f64> = (0..n_agents)
            .map(|i| {
                bids[i]
                    .get(alloc.get(i).copied().unwrap_or(0))
                    .copied()
                    .unwrap_or(0.0)
                    - payments[i]
            })
            .collect();

        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..n_agents {
            let mut perturbed = bids.to_vec();
            for b in perturbed[i].iter_mut() {
                *b += rng.random::<f64>() * 0.1 - 0.05;
            }
            if let Ok((alloc2, pay2)) = Self::vcg_allocation(&perturbed, |a: &[usize]| {
                bids.iter()
                    .enumerate()
                    .map(|(j, b)| {
                        b.get(a.get(j).copied().unwrap_or(0))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>()
            }) {
                let u2 = bids[i]
                    .get(alloc2.get(i).copied().unwrap_or(0))
                    .copied()
                    .unwrap_or(0.0)
                    - pay2[i];
                if u2 > utilities[i] + 1e-6 {
                    return false;
                }
            }
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §13 MeanFieldEquilibrium
// ─────────────────────────────────────────────────────────────────────────────

/// Mean field game equilibrium (symmetric population game).
pub struct MeanFieldEquilibrium;

impl MeanFieldEquilibrium {
    /// Compute mean field equilibrium via best-response dynamics iteration.
    /// `payoff_fn(action, mean_field_distribution)` → payoff for that action given population dist.
    /// Returns the equilibrium distribution over actions.
    pub fn compute_mfe(
        payoff_fn: impl Fn(usize, &[f64]) -> f64,
        n_actions: usize,
        tolerance: f64,
    ) -> Vec<f64> {
        let mut dist = vec![1.0 / n_actions as f64; n_actions];

        for _iter in 0..10_000 {
            let payoffs: Vec<f64> = (0..n_actions).map(|a| payoff_fn(a, &dist)).collect();
            let max_p = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let best_actions: Vec<usize> = (0..n_actions)
                .filter(|&a| (payoffs[a] - max_p).abs() < 1e-6)
                .collect();
            let n_best = best_actions.len() as f64;
            let mut br_dist = vec![0.0f64; n_actions];
            for &a in &best_actions {
                br_dist[a] = 1.0 / n_best;
            }

            let alpha = 0.1f64;
            let mut new_dist = vec![0.0f64; n_actions];
            for a in 0..n_actions {
                new_dist[a] = (1.0 - alpha) * dist[a] + alpha * br_dist[a];
            }

            let diff: f64 = new_dist
                .iter()
                .zip(dist.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            dist = new_dist;
            if diff < tolerance {
                break;
            }
        }
        dist
    }
}
