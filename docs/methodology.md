# Methodology

## Adversarial control problem

The frozen PPO victim selects the usual discrete CartPole control action. At the same timestep, a GTrXL PPO adversary selects a continuous wind value `w_t` bounded by `[-w_max, w_max]`. The environment applies

```text
F_t = 10 * victim_action_sign + w_t.
```

The adversary observes the complete CartPole state, previous victim action, and previous wind. Attention retains a longer interaction history.

Wind is modeled as zero-mean Gaussian noise with standard deviation `sigma`. Ignoring constants, its per-step negative log likelihood is

```text
L(w_t) = 0.5 * (w_t / sigma)^2.
```

The adversary receives `-L(w_t)` each step and an additional failure bonus when the victim crosses a CartPole boundary. This trades attack success against disturbance plausibility. Discounting also favors earlier failures.

## Failure-conditioned MCMC

For a fixed-length wind history `w`, the sampler targets the Gaussian density restricted to traces that fail:

```text
pi(w) proportional to p(w) * I[victim fails under w]
log p(w) = -sum_t 0.5 * (w_t / sigma_natural)^2.
```

Random-walk proposals perturb every timestep, one timestep, or a contiguous block. A proposal is rejected if it exceeds the wind bounds or does not reproduce failure. Otherwise it is accepted with

```text
alpha = min(1, exp(log p(w_proposed) - log p(w_current))).
```

Rejected proposals repeat the current state in the stored chain. Keeping those repetitions is required for valid Markov-chain statistics. Proposal standard deviation controls mixing; natural-wind standard deviation defines the scientific target distribution.

## Transformer representation analysis

Each distinct failure trace is replayed deterministically while preserving the adversary's 50-step attention state. The extractor records the mean latent, the final-window mean, and the 64-dimensional latent immediately before the failure-causing action.

The final latent vectors are standardized and clustered with K-means for `k=2,...,10`; silhouette score selects `k`. PCA, t-SNE, and UMAP visualize the same assignments rather than creating separate clusters. Labels are mapped back to the original pre-failure wind histories for interpretation.

## Baselines

- Constant wind measures sensitivity to a static force bias.
- Gaussian random wind separates disturbance magnitude from adaptive timing.
- A pure phase-zero sine wave tests whether an observed Fourier peak is sufficient to cause failure.
