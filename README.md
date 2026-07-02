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
