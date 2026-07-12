"""Global configuration constants for the traffic simulation,
fuzzy controller and optimization algorithms."""

# --- Fuzzy domain limits ---
QUEUE_MAX = 20          # max queue length used in fuzzy input domain (vehicles)
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

RANDOM_SEED = 42
