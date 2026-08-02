### MCMC_ORings

## Background

On January 28, 1986, the Space Shuttle Challenger broke apart 73 seconds after launch, killing all seven crew members. The cause was traced to a failure of the rubber O-ring seals in the shuttle's solid rocket boosters, which lost their flexibility in the unusually cold conditions (around 36°F / 2°C) at launch. Data from 23 previous shuttle flights recorded, for each flight, the launch temperature and whether at least one O-ring showed thermal distress.

This project fits a logistic regression model to that data:

```
x_i ~ Bernoulli(p_i)
logit(p_i) = beta0 + beta1 * T_i
```

where x_i is 1 if O-ring thermal distress occurred on flight i, and T_i is that flight's launch temperature (°F). The goal is to estimate beta0 and beta1. Then, we can calculate the probability of O-ring failure
at a given temperature, which would allow us to compute the probability at 36F - the temperature Challenger launched at.

Rather than using an existing sampling library, the posterior is fit with a from-scratch random-walk Metropolis-Hastings sampler : propose new (beta0, beta1) values from a normal random walk centered on the current values, then accept or reject them based on how much better (or worse) the proposal makes the log-posterior (log-likelihood + log-priors).


## Files

- **`methods.py`**: the 23-flight
  O-ring dataset, the logistic regression log-likelihood, four prior building blocks (uniform, exponential,      exponential-on-the-negative, normal), the Metropolis-Hastings sampler (`run_mcmc`), and two posterior
  summary helpers (`hpd_interval` for a 95% highest-probability-density interval, `mode_estimate` for a
  histogram-based point estimate).

- **`run1_uniform_priors.py`** Both step sizes and both starting values set to 1, with
  uniform (uninformative) priors on both parameters. The trace plots show no signs of convergence.

- **`run2_exponential_beta0.py`** In this case, `beta0`: step size 5, starting value 10,
  Exponential(rate=0.001) prior. `beta1`: step size 0.1, starting value -0.05, uniform prior. The trace plots
  now converge, but the resulting 95% HPD intervals are visibly skewed - because an Exponential prior only has
  support on positive numbers, it distorts beta0's posterior toward positive values.

- **`run3_normal_negexp.py`**. In the final run, `beta0`: step size 5, starting value 10, Normal(mean=15,
  sd=10) prior. `beta1`: step size 0.1, starting value -0.05, an Exponential(rate=-0.001) prior placed on `
  beta1`. This produces well-behaved histograms, from which modal estimates (`beta0 ≈ 14.8, beta1 ≈ -0.23`)
  are read off. Those estimates are then used to plot the fitted probability of O-ring failure against
  temperature, and specifically to compute the probability of failure at 36°F - the temperature Challenger
  launched at.

## Outputs

- **`allones_unifs.png`**: the traces from the first run.
- **`5_10_exp_0.1_-0.05_unif_traces.png`**: the traces from the second run.
- **`5_10_exp_0.1_-0.05_unif_hpds.png`**: the HPD and histogram plots from the second run.
- **`5_10_normal_0.1_-0.05_exp_traces.png`**: the traces from the third run.
- **`5_10_normal_0.1_-0.05_exp_hpds.png`**: the HPD and histogram plots from the third run.
- **`failure_prob.png`**: the plot of probability of failure versus temperature.

## Data source

The 23-flight Challenger O-ring dataset (Dalal, Fowlkes & Hoadley, 1989).
