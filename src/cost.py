"""Cost function: builds a fuzzy controller from a parameter vector and
evaluates it on the traffic simulation. This is the fitness function
minimized by both PSO and ACO."""
import numpy as np

from . import config as cfg
from .fuzzy_system import FuzzyController
from .simulation import TrafficSimulation

SEEDS = list(range(1, cfg.N_SEEDS + 1))


def evaluate(params, seeds=None, return_metrics=False):
    seeds = seeds if seeds is not None else SEEDS
    controller = FuzzyController(params)

    costs, Ws, Qs, Ss = [], [], [], []
    for seed in seeds:
        sim = TrafficSimulation(controller, cfg.ARRIVAL_RATE_1, cfg.ARRIVAL_RATE_2,
                                 n_cycles=cfg.N_CYCLES, seed=seed)
        res = sim.run()
        c = cfg.ALPHA * res["W"] + cfg.BETA * res["Q"] + cfg.GAMMA * res["S"]
        costs.append(c)
        Ws.append(res["W"])
        Qs.append(res["Q"])
        Ss.append(res["S"])

    mean_cost = float(np.mean(costs))
    if return_metrics:
        metrics = {"W": float(np.mean(Ws)), "Q": float(np.mean(Qs)), "S": float(np.mean(Ss))}
        return mean_cost, metrics
    return mean_cost


def get_bounds():
    """Lower/upper bounds for the 18-dim fuzzy parameter vector."""
    lo = np.array([0, 0, 0, 0, 0, 0, cfg.GREEN_MIN, cfg.GREEN_MIN, cfg.GREEN_MIN] + [0.0] * 9)
    hi = np.array([cfg.QUEUE_MAX] * 6 + [cfg.GREEN_MAX] * 3 + [1.0] * 9)
    return lo, hi


def repair(params):
    """Keep the 3 membership-function centers of each variable sorted so
    they always form a valid strong fuzzy partition, and clip to bounds."""
    p = np.array(params, dtype=float)
    p[0:3] = np.sort(p[0:3])
    p[3:6] = np.sort(p[3:6])
    p[6:9] = np.sort(p[6:9])
    lo, hi = get_bounds()
    return np.clip(p, lo, hi)
