# Adversarial Transformer PPO for CartPole

This project studies the robustness of a reinforcement-learning controller by
placing it in a two-agent CartPole system. A frozen PPO policy—the **victim**—
tries to balance the pole, while a Transformer PPO policy—the **adversary**—
adds a bounded wind force to the dynamics. The central question is not simply
whether an adversary can cause failure, but whether it can do so using a wind
sequence that remains plausible under a natural disturbance model.

The repository also explores why memory matters in partially observable
control, and uses Markov chain Monte Carlo (MCMC) to sample the broader set of
plausible wind histories that lead to failure.

## Problem formulation

Standard CartPole has state

```text
s_t = [x_t, x_dot_t, theta_t, theta_dot_t]
```

where `x` is cart position and `theta` is pole angle. At each timestep, the
victim chooses a discrete action that applies a force of magnitude 10 to the
left or right. The adversary simultaneously chooses a continuous wind force
`w_t`, bounded by

```text
-w_max <= w_t <= w_max.
```

The total horizontal force used by the CartPole dynamics is therefore

```text
F_t = 10 * victim_action_sign + w_t.
```

The victim fails when the cart leaves the track or the pole exceeds the usual
CartPole angle threshold. If neither happens, the episode ends at a fixed
horizon.

This is a dynamics attack rather than a direct policy attack: the adversary
does not change the victim's observations, actions, or network parameters. It
changes the physical transition that follows the victim's action.

## Victim policy

The victim is first trained independently with PPO and then frozen. Freezing
the victim separates two learning problems:

1. the victim learns a nominal control strategy; and
2. the adversary learns which disturbances exploit that fixed strategy.

During adversary training, the victim acts deterministically. This makes a
failure attributable to the initial state and wind history rather than to
exploration noise in the victim policy.

Two victim observation models are supported:

- **Fully observable CartPole:** the victim sees all four state variables.
- **Stateless CartPole:** the victim sees only cart position and pole angle.
  Velocity must be inferred from a history of observations.

The stateless case introduces partial observability. A single observation does
not reveal whether the pole is moving toward or away from failure, even though
that distinction is essential for good control.

## Transformer memory experiment

`main.py` compares a GTrXL Transformer policy with a feed-forward PPO baseline
on both observation models.

For stateless CartPole, the Transformer receives the raw two-dimensional
observation

```text
o_t = [x_t, theta_t]
```

and maintains 50 timesteps of internal attention memory. Attention lets the
policy relate the current observation to earlier observations, allowing it to
infer motion from temporal differences.

The plain PPO baseline receives the same information in a different form: the
last 50 observations are concatenated into a fixed 100-dimensional vector.
This avoids comparing a memory-enabled policy against a policy with no access
to history at all. Instead, the experiment compares learned attention over a
sequence with a feed-forward model over an explicit frame stack.

On fully observable CartPole, memory should be less important because position
and velocity are already present in the current state. The stateless
environment is therefore the more meaningful test of the Transformer's
temporal representation.

## Adversary policy

The adversary observes

```text
z_t = [x_t, x_dot_t, theta_t, theta_dot_t,
       previous victim action, previous wind].
```

Its action is the next continuous wind value. Including the previous action
and wind gives the policy information about the recent control interaction,
while GTrXL attention preserves a longer history.

The adversary faces two competing objectives:

- produce a trajectory that makes the victim fail; and
- avoid using wind that is too unlikely under the assumed disturbance model.

Wind is modeled as zero-mean Gaussian noise with standard deviation `sigma`.
Ignoring constants that do not affect optimization, the negative log
likelihood of a wind action is

```text
L(w_t) = 0.5 * (w_t / sigma)^2.
```

The per-step adversary reward begins as

```text
r_t = -L(w_t).
```

When the victim fails, the adversary receives an additional failure bonus:

```text
r_t = failure_bonus - L(w_t).
```

If the victim survives to the horizon, the final reward also subtracts its
normalized distance from the nearest position or angle failure boundary. This
gives a surviving trajectory a slightly better outcome when it at least brings
the system close to failure.

Because PPO uses discounted returns, an earlier failure bonus is worth more
than the same bonus received later. The policy is therefore encouraged to find
failures that are both early and inexpensive in wind likelihood. A large
`sigma` makes strong wind less costly; a small `sigma` imposes a stricter notion
of natural disturbance.

This formulation is useful because unconstrained attack success is usually
not informative: sufficiently large forces can trivially destroy almost any
controller. Penalizing wind magnitude asks whether the controller fails under
disturbances that remain statistically credible.

## What the reported metrics mean

The main adversary metrics describe different aspects of robustness:

- **Failure rate:** fraction of episodes in which the victim crosses a failure
  boundary before the horizon.
- **Episode length:** survival time; shorter failed episodes indicate a faster
  attack.
- **Mean absolute wind:** an interpretable measure of disturbance strength.
- **Wind log-likelihood penalty:** how improbable the applied wind is under the
  assumed Gaussian model.
- **Failure timestep:** when the boundary crossing occurs.

No single metric is sufficient. High failure rate under extreme wind says
little about plausible failures, while very likely wind that never causes
failure says little about the controller's weak points. The reward and analysis
tools are designed to expose this tradeoff.

## Robustness baselines

The learned adversary is compared with simpler disturbance models.

### Constant wind

A constant-wind sweep evaluates the victim under fixed forces across a chosen
range. The zero-wind result establishes nominal performance. Each disturbed
success rate is compared with that baseline, which reveals the force level at
which performance degrades beyond a chosen tolerance.

Constant wind is easy to interpret but cannot adapt to the state. It tests
static bias sensitivity rather than the more general sequential attack learned
by the adversary.

### Pure sine-wave wind

`eval_sine_wind.py` tests whether the frequency peak observed in the MCMC
failure traces is sufficient by itself to cause failure. It applies the
phase-zero trace

```text
w_t = A * sin(2 * pi * 0.23 * t)
```

for amplitudes `0.1, 0.2, ..., 1.0`. Each episode uses seed `1006` and may run
for the complete 500-step horizon. The script does not search over phase and
does not add an offset or noise, so it is a direct test of a pure
`0.23`-frequency disturbance.

If an amplitude causes a true CartPole boundary failure, the script saves and
replays the 500-step trace before evaluating it on 100 additional initial-state
seeds. If no amplitude fails, no failure file is created. With the current
victim checkpoint, every tested amplitude survived all 500 steps. This shows
that the Fourier peak alone is not sufficient to reproduce the MCMC failures.

### Random wind

Gaussian random wind provides a non-adaptive stochastic baseline. Comparing
its episode-length distribution with the learned adversary helps distinguish
failures caused by disturbance magnitude alone from failures caused by the
timing and direction of the disturbance.

If the learned adversary causes earlier or more frequent failures at comparable
wind scale, it has discovered temporal structure in the victim's
vulnerabilities rather than merely applying noise.

## Failure-conditioned MCMC

A trained adversary produces examples of failure, but those examples do not
by themselves characterize the distribution of possible failures.
`mcmc_failure_trace.py` uses random-walk Metropolis sampling to explore wind
traces conditioned on victim failure.

Let a fixed-length wind trace be

```text
w = [w_1, w_2, ..., w_T].
```

Under an independent zero-mean Gaussian natural-wind model, its log density is,
up to an additive constant,

```text
log p(w) = -sum_t 0.5 * (w_t / sigma_natural)^2.
```

The target distribution is this density restricted to traces that cause the
deterministic victim to fail:

```text
pi(w) proportional to p(w) * I[victim fails under w].
```

The chain starts from a known adversarial failure trace. A proposal perturbs
all timesteps, one timestep, or a contiguous block. The proposed trace is then
replayed through the environment.

- A proposal outside the wind bounds is rejected.
- A proposal that does not reproduce failure is rejected.
- A failure-causing proposal is accepted with probability

```text
alpha = min(1, exp(log p(w_proposed) - log p(w_current))).
```

Consequently, the chain remains entirely inside the failure set while favoring
traces that are more plausible under the natural-wind model.

The proposal standard deviation and natural-wind standard deviation have
different meanings. `proposal_sigma` controls how far the Markov chain tries
to move at each iteration and therefore affects mixing and acceptance.
`natural_wind_sigma` defines the target distribution and therefore determines
which traces count as plausible. Conflating the two would tie a computational
tuning parameter to the scientific disturbance model.

Repeated states after rejected proposals are retained in the raw chain, as
required for correct MCMC statistics. The long-form CSV contains only the
initial trace and newly accepted failure traces, making it more convenient for
visual inspection.

## Interpreting the MCMC diagnostics

- **Acceptance rate** measures how often a proposal becomes the next chain
  state. Very low acceptance can indicate proposals that are too large or a
  narrow failure region.
- **Proposal failure rate** measures how often proposed traces still cause
  victim failure, independent of their Metropolis acceptance.
- **Out-of-bounds rate** shows how often proposals violate the allowed wind
  interval.
- **Failure-step distribution** describes when the sampled plausible failures
  occur.

Burn-in can remove early samples that remain strongly influenced by the source
trace. Thinning reduces visual and storage density, although it does not repair
poor mixing. Trace plots, acceptance statistics, and sensitivity to proposal
scale should be considered together before treating samples as representative.

The replay check performed before sampling is important: if the initial trace
does not fail under the selected environment seed, victim checkpoint, and
observation model, then it is not a valid state in the conditional target
distribution.

## Final pre-failure latent-space analysis

`extract_transformer_latents.py` replays each distinct MCMC failure trace and
records the adversary Transformer's internal representation. It preserves the
model's intended 50-step attention memory and saves three 64-dimensional
summaries per trace:

- the mean latent over the complete trace;
- the mean latent over the final 50 timesteps; and
- the final latent immediately before the failure-causing wind action.

`analyze_transformer_latents.py` analyzes the final pre-failure vectors in
`transformer_latent_summaries.npz`. The current result contains 9,627 distinct
failure traces. Each distinct trace is analyzed exactly once; repeated MCMC
rows and MCMC dwell-time weights are not used in clustering, percentages, or
failure-time summaries.

The analysis performs the following steps:

1. Verify that latent extraction completed and that the MCMC data and policy
   checkpoint fingerprints match.
2. Standardize each of the 64 latent dimensions across the 9,627 traces.
3. Run K-means in the complete standardized 64-dimensional space for
   `k=2,...,10`.
4. Select the value of `k` with the highest silhouette score.
5. Generate separate PCA, t-SNE, and UMAP two-dimensional views, coloring each
   point with the same K-means label.
6. Map the labels back to the original wind traces and plot only winds applied
   before each trace's actual failure.

K-means supplies the cluster assignments. PCA, t-SNE, and UMAP do not perform
additional clustering; they provide three different two-dimensional views of
the same 64-dimensional assignments.

With random seed `42`, K-means selects two clusters:

| Cluster | Distinct traces | Trace percentage | Mean failure step | Median failure step |
| --- | ---: | ---: | ---: | ---: |
| 0 | 9,400 | 97.64% | 130.8 | 120 |
| 1 | 227 | 2.36% | 237.0 | 233 |

The first two PCA components explain 44.49% and 27.33% of standardized latent
variance, respectively. The small second cluster is visibly separated in PCA,
t-SNE, and UMAP and is associated with later failures. This is descriptive
structure among failure traces, not evidence that the cluster causes failure.
All traces are failures from one MCMC experiment, so the analysis does not
compare successful and failed trajectories.

The generated `latent_analysis/` directory contains:

- cluster-selection, PCA, t-SNE, and UMAP plots;
- combined and cluster-faceted wind-trace plots;
- `latent_clusters.csv`, with one row and one cluster label per distinct trace;
  and
- `cluster_summary.csv`, with unweighted trace counts and failure-time
  statistics.

## Minimal reproduction

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Train the frozen victim and then the Transformer adversary:

```bash
python train_victim.py --iters 100 --out-dir checkpoints/victim
python train_adversary.py --victim-checkpoint checkpoints/victim --iters 10 --out-dir checkpoints/adversary
```

Generate and visualize a failure-conditioned chain:

```bash
python mcmc_failure_trace.py --victim-checkpoint checkpoints/victim --output episode_6_mcmc.npz
python plot_wind_history.py --input episode_6_mcmc.npz --burn-in 1000 --thin 10 --show-mean --out-path episode_6_mcmc.png
```

Test the pure phase-zero sine-wave disturbance:

```bash
python eval_sine_wind.py
```

After generating `transformer_latent_summaries.npz` with the
checkpoint-compatible Python 3.8 and Ray 2.10 environment, run the latent
analysis in the main environment:

```bash
python analyze_transformer_latents.py
```

The victim observation setting must remain consistent across training,
adversary evaluation, and MCMC replay. Use `--env stateless` when training a
stateless victim and `--victim-env stateless` in downstream commands.

## Repository layout

- `main.py` — Transformer versus frame-stacked PPO memory experiment
- `train_victim.py` — nominal victim training and deterministic evaluation
- `train_adversary.py` — Transformer adversary training and evaluation
- `envs/adversarial_cartpole.py` — attacked dynamics, reward, and metrics
- `eval_constant_wind.py` — fixed-disturbance robustness baseline
- `plot_episode_length_histogram.py` — random/adversarial survival comparison
- `plot_wind_history.py` — NPZ and CSV wind-trace visualization
- `mcmc_failure_trace.py` — failure-conditioned Metropolis sampler
- `eval_sine_wind.py` — fixed-phase pure sine-wave robustness test
- `extract_transformer_latents.py` — replay and pre-failure latent extraction
- `analyze_transformer_latents.py` — K-means, PCA, t-SNE, UMAP, and
  cluster-colored wind traces
- `analysis.ipynb` — exploratory analysis

Run the unit tests with:

```bash
python -m unittest discover -v
```
