# Transformer PPO with Ray RLlib

A minimal example of training PPO with a Transformer (GTrXL attention)
policy network using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This trains PPO on `CartPole-v1` for a few iterations and prints the mean
episode return. The policy uses RLlib's GTrXL attention encoder
(`use_attention=True`) over a window of recent observations.

## Notes

- GTrXL attention is part of RLlib's ModelV2 API, so `main.py` opts out of
  the new RLModule/Learner stack via `.api_stack(...)`.
- Tune the Transformer via the `model` dict in `main.py`
  (`attention_dim`, `attention_num_heads`, `attention_num_transformer_units`,
  `attention_memory_*`, etc.).
- On `stateless`, plain PPO stacks the last 50 observations into a 100-dim
  input vector; the Transformer uses `attention_memory_*: 50` for internal
  memory over 2-dim observations. This makes the comparison fairer on the
  memory task.
- Swap `CartPole-v1` for any Gymnasium env id to experiment further.

## Minimal adversarial setup

Train a frozen stateless PPO victim:

```bash
python train_victim.py --iters 10
```

Then train the Transformer adversary against that checkpoint:

```bash
python train_adversary.py --victim-checkpoint checkpoints/victim --iters 10
```

The adversary controls a one-dimensional wind force added to the CartPole
dynamics. It prints failure rate, episode length, mean absolute wind, and the
average Gaussian wind log-likelihood penalty.

## MCMC failure-trace sampling

`mcmc_failure_trace.py` runs a random-walk Metropolis chain initialized from
episode 6 in `adversary_wind_history_same_seeds.npz`. The 289-step failure is
zero-padded to 350 winds, and proposals use a standard deviation of `0.01`
while the natural-wind target retains its saved standard deviation of `1.0`.

```bash
python mcmc_failure_trace.py
```

The source trace is replayed before sampling, so the command fails clearly if
the selected environment seed or victim checkpoint does not reproduce the
recorded failure. The resulting raw chain can be plotted after optional
burn-in and thinning:

```bash
python plot_wind_history.py --input episode_6_mcmc.npz --burn-in 1000 --thin 10 --show-mean
```
