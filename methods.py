
import numpy as np
import matplotlib.pyplot as plt

# Writing the data set - 23 columns
data = [
    (1,  1, 53), (2,  1, 57), (3,  1, 58), (4,  1, 63),
    (5,  0, 66), (6,  0, 67), (7,  0, 67), (8,  0, 67),
    (9,  0, 68), (10, 0, 69), (11, 0, 70), (12, 0, 70),
    (13, 1, 70), (14, 1, 70), (15, 0, 72), (16, 0, 73),
    (17, 1, 75), (18, 0, 75), (19, 0, 76), (20, 0, 76),
    (21, 0, 78), (22, 0, 79), (23, 0, 81),
]
x = np.array([d[1] for d in data], dtype=float)   # failure status (0/1)
T = np.array([d[2] for d in data], dtype=float)   # temperature (deg F)

RATE = 0.001   # for the exponential prior


# Defining the log likelihood, as prescribed by logistic regression
def loglik(beta0, beta1):
    eta = beta0 + beta1 * T
    logp = -np.log1p(np.exp(-eta))          # log(1/(1+exp(-eta)))
    log1mp = -eta - np.log1p(np.exp(-eta))  # log(exp(-eta)/(1+exp(-eta)))
    sum1 = np.sum(x * logp)
    sum2 = np.sum((1 - x) * log1mp)
    return sum1 + sum2

# Defining different possible priors that can be used in the computation of log posterior
def prior_uniform(b):
    return 0.0

def prior_exponential(b, rate=RATE):
    """log-density of Exponential(rate) at b (support b >= 0)."""
    if b < 0:
        return -np.inf
    return np.log(rate) - rate * b

def prior_negexponential(b, rate=RATE):
    """Exponential prior placed on -b, pushing b negative (Section 8)."""
    return prior_exponential(-b, rate)

def prior_normal(b, mean, sd):
    return -0.5 * ((b - mean) / sd) ** 2


# Homemade MCMC sampler, that takes the following inputs:
# N: number of iterations
# sigma0 and sigma1: step sizes for beta0 and beta1
# seed0 and seed1: seed values for beta0 and beta1
# prior0_fn and prior1_fn: choice of priors for beta0 and beta1
def run_mcmc(N, sigma0, sigma1, seed0, seed1, prior0_fn, prior1_fn, rng_seed=1,
             use_log_u=True):
    rng = np.random.default_rng(rng_seed)

    beta0 = np.empty(N)
    beta1 = np.empty(N)
    post = np.empty(N)

    beta0[0] = seed0
    beta1[0] = seed1
    post[0] = loglik(beta0[0], beta1[0]) + prior0_fn(beta0[0]) + prior1_fn(beta1[0])

    n_accept = 0
    for i in range(1, N):
        b0_prop = rng.normal(beta0[i - 1], sigma0)
        b1_prop = rng.normal(beta1[i - 1], sigma1)

        post_prop = loglik(b0_prop, b1_prop) + prior0_fn(b0_prop) + prior1_fn(b1_prop)

        u = rng.uniform(0.0, 1.0)
        threshold = np.log(u) if use_log_u else u
        if post_prop - post[i - 1] >= threshold:
            beta0[i], beta1[i], post[i] = b0_prop, b1_prop, post_prop
            n_accept += 1
        else:
            beta0[i], beta1[i], post[i] = beta0[i - 1], beta1[i - 1], post[i - 1]

    return beta0, beta1, post, n_accept / (N - 1)

# Define the 95% highest probability density interval - these are two lines that we draw in the histogram
# of a parameter, such that the area between the two lines contain 95% of the area of the histogram
def hpd_interval(samples, cred_mass=0.95):
    s = np.sort(samples)
    n = len(s)
    n_included = int(np.ceil(cred_mass * n))
    n_intervals = n - n_included
    if n_intervals <= 0:
        return s[0], s[-1]
    widths = s[n_included:] - s[:n_intervals]
    min_idx = np.argmin(widths)
    return s[min_idx], s[min_idx + n_included]

#Returns the modal values of the values of the parameters
def mode_estimate(samples, bins=100):
    counts, edges = np.histogram(samples, bins=bins)
    i = np.argmax(counts)
    return 0.5 * (edges[i] + edges[i + 1])
