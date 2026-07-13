"""Global configuration constants for the traffic simulation,
fuzzy controller and optimization algorithms."""

# --- Fuzzy domain limits ---
# max queue length used in fuzzy input domain (vehicles)
QUEUE_MAX = 20
CYCLE_TIME = 60         # total signal cycle length (seconds)
GREEN_MIN = 5           # minimum green time (seconds)
GREEN_MAX = 55          # maximum green time (seconds)
SERVICE_RATE = 0.5      # vehicles discharged per second during green

# --- Traffic arrival rates (vehicles / second) ---
ARRIVAL_RATE_1 = 0.25
ARRIVAL_RATE_2 = 0.20

# --- Simulation episode length ---
N_CYCLES = 30           # signal cycles per simulation episode
N_SEEDS = 4             # random seeds averaged per fitness evaluation

# --- Cost function weights: C = ALPHA*W + BETA*Q + GAMMA*S ---
ALPHA = 1.0
BETA = 1.0
GAMMA = 0.05

RANDOM_SEED = 97


# ========== PSO Hyperparameters ==========
PSO_N_PARTICLES = 1000
PSO_ITERATIONS = 30
PSO_W_START = 0.9
PSO_W_END = 0.4

PSO_C1_START = 3.5
PSO_C1_END = 0.5
PSO_C2_START = 0.5
PSO_C2_END = 3.5

PSO_VMAX_FRAC = 0.2                    # velocity limit = fraction of range

# ========== Evaluation ==========
PSO_SEEDS_EARLY = 1                    # seeds in first half of iterations
PSO_SEEDS_LATE = 5                    # seeds in second half
PSO_SEEDS_END_MUL_TERM = 0.9
# ========== Stagnation & Mutation ==========
# start mild mutation after this many stagnant steps
PSO_STAGNATION_LIMIT = 3
PSO_HEAVY_STAGNATION_LIMIT = 5         # reset particle after this many steps

PSO_MUTATION_MIN_RATE = 0.01
PSO_MUTATION_MAX_RATE = 0.4
PSO_MUTATION_EXTRA_RATE = 0.08         # extra rate per stagnant step beyond limit

# probability to jump near gbest instead of random reset
PSO_ELITE_LEARN_PROB = 0.15
# ± fraction of range when jumping near gbest
PSO_ELITE_NOISE_FRAC = 0.05
PSO_RESET_VEL_STD = 0.02               # std of velocity after a reset

# ± fraction of dimension range for mutation
PSO_MUTATION_SHAKE_FRAC = 0.1

# ========== Stagnation tolerance (adaptive) ==========
PSO_TOL_ABSOLUTE = 1e-6
PSO_TOL_RELATIVE = 1e-4                # relative to previous personal best cost

# ========== Parallel ==========
# number of processes (set to None for auto)
PSO_N_WORKERS = 8
