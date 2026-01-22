import random

# Configuration for the FO2 playground experiments

# --- Reproducibility ---
# Seed for the random number generator to ensure reproducible sampling.
SEED = 42

# --- Exhaustive Search ---
# The maximum domain size (n) for which we'll run an exhaustive search.
# WARNING: The number of models grows extremely quickly. n=7 is already slow.
N_MAX_EXHAUSTIVE = 5

# The maximum domain size for which we'll perform expensive isomorphism filtering.
# This should not exceed N_MAX_EXHAUSTIVE.
N_MAX_ISOMORPHISM_FILTER = 5

# --- Random Sampler ---
# A list of domain sizes (n) to test using the random sampler.
SAMPLER_N_LIST = [8, 10, 15, 20, 50, 100]

# The number of random models (k) to generate for each domain size in the sampler.
# A larger k gives a more accurate estimate of the satisfiability rate.
SAMPLER_K = 1000

# --- Model Generation ---
# Probability of an edge existing in the random graph for sample_model.
P_EDGE = 0.5
# Probability of a vertex having a color in the random model for sample_model.
P_UNARY = 0.5
