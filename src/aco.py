"""Discretized continuous Ant Colony Optimization for tuning the fuzzy
controller parameters. Each of the 18 parameter dimensions is discretized
into a fixed number of levels; every ant builds a solution by choosing one
level per dimension with probability proportional to the pheromone value
on that (dimension, level) pair. Pheromone is updated with the standard
evaporation + deposit rule:
    tau_ij(t+1) = (1 - rho) * tau_ij(t) + delta_tau_ij
where delta_tau_ij is the sum of Q/cost over ants that chose level j for
dimension i.
"""
import numpy as np

from . import config as cfg
from .cost import evaluate, get_bounds, repair


class ACO:
    def __init__(self, n_ants=cfg.ACO_N_ANTS, n_iter=cfg.ACO_ITERATIONS,
                 n_levels=cfg.ACO_N_LEVELS, seed=cfg.RANDOM_SEED):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.n_levels = n_levels
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = get_bounds()
        self.dim = len(self.lo)
        self.levels = np.array([
            np.linspace(self.lo[d], self.hi[d], n_levels) for d in range(self.dim)
        ])
        self.tau = np.full((self.dim, n_levels), cfg.ACO_TAU0)

    def _construct_solution(self):
        idx = np.zeros(self.dim, dtype=int)
        params = np.zeros(self.dim)
        for d in range(self.dim):
            weights = self.tau[d] ** cfg.ACO_ALPHA
            probs = weights / weights.sum()
            i = self.rng.choice(self.n_levels, p=probs)
            idx[d] = i
            params[d] = self.levels[d, i]
        return idx, params

    def optimize(self):
        best_params, best_cost = None, np.inf
        best_idx = None
        history = []

        for it in range(self.n_iter):
            all_idx = np.zeros((self.n_ants, self.dim), dtype=int)
            all_cost = np.zeros(self.n_ants)

            for a in range(self.n_ants):
                idx, params = self._construct_solution()
                repaired = repair(params)
                cost = evaluate(repaired)
                all_idx[a] = idx
                all_cost[a] = cost

                if cost < best_cost:
                    best_cost = cost
                    best_params = repaired
                    best_idx = idx

            # evaporation
            self.tau *= (1 - cfg.ACO_RHO)

            # deposit: cheaper (better) solutions reinforce their levels more
            for a in range(self.n_ants):
                deposit = cfg.ACO_Q / max(all_cost[a], 1e-6)
                for d in range(self.dim):
                    self.tau[d, all_idx[a, d]] += deposit

            # elitist reinforcement of the best-so-far solution
            if best_idx is not None:
                elite_deposit = cfg.ACO_ELITE_BONUS * cfg.ACO_Q / max(best_cost, 1e-6)
                for d in range(self.dim):
                    self.tau[d, best_idx[d]] += elite_deposit

            self.tau = np.clip(self.tau, 1e-6, None)
            history.append(best_cost)
            print(f"[ACO] iter {it + 1}/{self.n_iter}  best_cost={best_cost:.4f}")

        return best_params, best_cost, history
