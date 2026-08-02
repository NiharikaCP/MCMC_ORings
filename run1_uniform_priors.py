"""
run1_uniform_priors.py

MCMC run #1: all seeds and step sizes set to 1, uniform priors on both
parameters (i.e. the priors contribute nothing).

"""

import os
from methods import run_mcmc, prior_uniform
import matplotlib.pyplot as plt

os.makedirs('outputs', exist_ok=True)

# Perform MCMC where I set all of the seeds and step sizes to 1, and set the priors to a simple uniform prior.
# We note that there is clearly no convergence in the traces for either beta and so we need to use a different
# choice for the inputs.
N = 500_000
b0_r1, b1_r1, post_r1, acc_r1 = run_mcmc(
    N, sigma0=1.0, sigma1=1.0, seed0=1.0, seed1=1.0,
    prior0_fn=prior_uniform, prior1_fn=prior_uniform, rng_seed=1,
)

fig1, ax1 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1[0].plot(b0_r1, color='purple', lw=0.5)
ax1[0].set_ylabel(r'$\beta_0$')
ax1[1].plot(b1_r1, color='purple', lw=0.5)
ax1[1].set_ylabel(r'$\beta_1$')
ax1[1].set_xlabel('iteration')
fig1.tight_layout()
fig1.suptitle(r'Traces: all uniform priors, step=1/1, seed=1/1')
plt.savefig('outputs/allones_unifs.png')

print(f"acceptance rate: {acc_r1:.4f}")
