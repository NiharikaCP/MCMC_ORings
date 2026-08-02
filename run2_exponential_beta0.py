"""
MCMC run #2: beta0 gets step=5, seed=10, and an Exponential(rate=0.001)
prior; beta1 gets step=0.1, seed=-0.05, and a uniform (uninformative)
prior.
"""

import os
from methods import run_mcmc, prior_exponential, prior_uniform, hpd_interval
import matplotlib.pyplot as plt

os.makedirs('outputs', exist_ok=True)

N = 500_000

# We run this for: beta0: step=5, seed=10, exponential prior; beta1: step=0.1, seed=-0.05, uniform prior
# Notice that the traces converge but the HPDs are skewed, so this is not the best choice of parameters
b0_r2, b1_r2, post_r2, acc_r2 = run_mcmc(
    N, sigma0=5.0, sigma1=0.1, seed0=10.0, seed1=-0.05,
    prior0_fn=prior_exponential, prior1_fn=prior_uniform, rng_seed=1,
)

fig2, ax2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax2[0].plot(b0_r2, color='purple', lw=0.5)
ax2[0].set_ylabel(r'$\beta_0$')
ax2[1].plot(b1_r2, color='purple', lw=0.5)
ax2[1].set_ylabel(r'$\beta_1$')
ax2[1].set_xlabel('iteration')
fig2.suptitle(r'Traces: exponential prior on $\beta_0$ (rate=0.001), step=5/0.1, seed=10/-0.05')
fig2.tight_layout()
plt.savefig('outputs/5_10_exp_0.1_-0.05_unif_traces.png')

lo0_r2, hi0_r2 = hpd_interval(b0_r2)
lo1_r2, hi1_r2 = hpd_interval(b1_r2)

fig3, ax3 = plt.subplots(1, 2, figsize=(11, 4.5))
ax3[0].hist(b0_r2, bins=80, histtype='step', color='black')
ax3[0].axvline(lo0_r2, color='black', lw=1)
ax3[0].axvline(hi0_r2, color='black', lw=1)
ax3[0].set_title(rf'$\beta_0$={lo0_r2:.2f}   $\beta_0$={hi0_r2:.2f}')
ax3[0].set_xlabel(r'$\beta_0$')
ax3[0].set_ylabel('frequency')

ax3[1].hist(b1_r2, bins=80, histtype='step', color='black')
ax3[1].axvline(lo1_r2, color='black', lw=1)
ax3[1].axvline(hi1_r2, color='black', lw=1)
ax3[1].set_title(rf'$\beta_1$={lo1_r2:.3f}   $\beta_1$={hi1_r2:.3f}')
ax3[1].set_xlabel(r'$\beta_1$')

fig3.suptitle(r'Histograms with 95% HPD lines: exponential prior on $\beta_0$ (rate=0.001), step=5/0.1, seed=10/-0.05')
fig3.tight_layout()
plt.savefig('outputs/5_10_exp_0.1_-0.05_unif_hpds.png')

print(f"acceptance rate: {acc_r2:.4f}")
