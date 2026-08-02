"""
MCMC run #3: beta0 gets step=5, seed=10, and a Normal(mean=15, sd=10)
prior; beta1 gets step=0.1, seed=-0.05, and an Exponential(rate=0.001)
prior placed on -beta1.
"""

import os
from methods import run_mcmc, prior_normal, prior_negexponential, hpd_interval, mode_estimate, x, T
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('outputs', exist_ok=True)

N = 500_000

# We run the MCMC for: beta0: step=5, seed=10, Normal(mean=15, sd=10); beta1: step=0.1, seed=-0.05, Exponential on -beta1
prior0_normal = lambda b: prior_normal(b, mean=15.0, sd=10.0)

b0_r3, b1_r3, post_r3, acc_r3 = run_mcmc(
    N, sigma0=5.0, sigma1=0.1, seed0=10.0, seed1=-0.05,
    prior0_fn=prior0_normal, prior1_fn=prior_negexponential, rng_seed=1,
)

lo0_r3, hi0_r3 = hpd_interval(b0_r3)
lo1_r3, hi1_r3 = hpd_interval(b1_r3)

fig6, ax6 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax6[0].plot(b0_r3, color='purple', lw=0.5)
ax6[0].set_ylabel(r'$\beta_0$')
ax6[1].plot(b1_r3, color='purple', lw=0.5)
ax6[1].set_ylabel(r'$\beta_1$')
ax6[1].set_xlabel('iteration')
fig6.suptitle(r'Traces: normal prior on $\beta_0$ (mean=15, sd=10), exponential prior on $\beta_1$ (rate=0.001), step=5/0.1, seed=10/-0.05')
fig6.tight_layout()
plt.savefig('outputs/5_10_normal_0.1_-0.05_exp_traces.png')

fig4, ax4 = plt.subplots(1, 2, figsize=(11, 4.5))
ax4[0].hist(b0_r3, bins=80, histtype='step', color='black')
ax4[0].axvline(lo0_r3, color='black', lw=1)
ax4[0].axvline(hi0_r3, color='black', lw=1)
ax4[0].set_title(rf'$\beta_0$={lo0_r3:.2f}   $\beta_0$={hi0_r3:.2f}')
ax4[0].set_xlabel(r'$\beta_0$')
ax4[0].set_ylabel('frequency')

ax4[1].hist(b1_r3, bins=80, histtype='step', color='black')
ax4[1].axvline(lo1_r3, color='black', lw=1)
ax4[1].axvline(hi1_r3, color='black', lw=1)
ax4[1].set_title(rf'$\beta_1$={lo1_r3:.3f}   $\beta_1$={hi1_r3:.3f}')
ax4[1].set_xlabel(r'$\beta_1$')

fig4.suptitle(r'Histograms with 95% HPD lines: normal prior on $\beta_0$ (mean=15, sd=10), exponential prior on $\beta_1$ (rate=0.001), step=5/0.1, seed=10/-0.05')
fig4.tight_layout()
plt.savefig('outputs/5_10_normal_0.1_-0.05_exp_hpds.png')

BETA0_MODE = mode_estimate(b0_r3)   # computed as 14.80
BETA1_MODE = mode_estimate(b1_r3)   # computed as -0.232

def pr_fail(Tval, b0=BETA0_MODE, b1=BETA1_MODE):
    return 1.0 / (1.0 + np.exp(-b1 * Tval - b0))

T_grid = np.linspace(0, 70, 300)
p_grid = pr_fail(T_grid)

# Provides the figure of the probability of failure versus temperature
fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.plot(T_grid, p_grid, color='purple')
ax5.set_xlabel('Temperature (F)')
ax5.set_ylabel('Probability of O-ring failure')
fig5.suptitle('Pr(T) against T')
fig5.tight_layout()
plt.savefig('outputs/failure_prob.png')

# This computes the probability of failure of the o-rings at 36F, which was the temperature when the Challenger shuttle was launched
p36 = pr_fail(36.0)
print(f"\nPr(failure at T=36F) using live modal estimates "
      f"(beta0={BETA0_MODE:.2f}, beta1={BETA1_MODE:.3f}) = {p36:.3f}")
