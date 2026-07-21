# Adversarial Transformer PPO for CartPole

This project studies the robustness of a frozen PPO CartPole policy (the
**victim**) against a learned Transformer PPO policy (the **adversary**). The
adversary observes the CartPole state and adds a bounded, one-dimensional wind
force to the victim's left/right control force. Its objective balances causing
an early failure against keeping the wind likely under a Gaussian natural-wind
model.

The repository also includes a smaller RLlib experiment comparing a GTrXL
Transformer policy with a feed-forward PPO baseline on fully observable and
stateless CartPole, plus tools for deterministic evaluation, plotting, and
MCMC sampling of failure-causing wind traces.

## Requirements

- Python with virtual-environment support
- Dependencies pinned in `requirements.txt`, including Ray RLlib 2.40,
  PyTorch 2.5, Gymnasium, Matplotlib, W&B, Pillow, and MoviePy
- Optional CUDA-capable GPU; the training scripts automatically use CUDA when
  PyTorch detects it and otherwise run on CPU

Create an environment and install the dependencies:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Train a standard CartPole victim, then train and evaluate an adversary against
the saved victim checkpoint:

```bash
python train_victim.py --iters 100 --out-dir checkpoints/victim
python train_adversary.py --victim-checkpoint checkpoints/victim --iters 10 --out-dir checkpoints/adversary
```

Both commands perform a deterministic evaluation after training. RLlib may
print a more specific checkpoint directory than the requested output directory;
use the printed path when a later command cannot restore the parent directory.

For a quick smoke test, reduce `--iters`, `--train-batch-size`, and evaluation
episodes:

```bash
python train_victim.py --iters 1 --train-batch-size 256 --num-env-runners 0 --eval-episodes 2
python train_adversary.py --victim-checkpoint checkpoints/victim --iters 1 --train-batch-size 256 --eval-episodes 2
```

## Transformer versus PPO baseline

`main.py` compares an RLlib GTrXL policy against a feed-forward PPO policy. It
runs five seeds for 40 training iterations and reports the cross-seed mean and
standard deviation at every iteration.

```bash
python main.py transformer cartpole
python main.py plain cartpole
python main.py transformer stateless
python main.py plain stateless
```

The stateless environment exposes only cart position and pole angle. For a
fairer memory comparison, the Transformer receives the raw two-dimensional
observation with 50 steps of internal attention memory, while the plain PPO
policy receives the last 50 observations as a 100-dimensional stacked input.
The model, seeds, iteration count, and attention settings are currently
configured directly in `main.py`.

RLlib's GTrXL model uses the ModelV2 API, so the scripts disable the newer
RLModule/Learner and ConnectorV2 stacks.

## Train and evaluate the victim

The victim is a feed-forward PPO policy. Standard CartPole is the default:

```bash
python train_victim.py \
  --env cartpole \
  --iters 100 \
  --hidden-sizes 256,256,128 \
  --out-dir checkpoints/victim
```

To train on the partially observable environment, use `--env stateless`.
Evaluate an existing checkpoint without training:

```bash
python train_victim.py \
  --eval-only \
  --checkpoint checkpoints/victim \
  --env cartpole \
  --eval-episodes 20
```

`--env` must match the environment used to train the checkpoint. A mismatch
changes the observation shape and usually produces a tensor-shape error.

Useful training controls include `--seed`, `--lr`, `--train-batch-size`,
`--num-env-runners`, `--num-envs-per-env-runner`, and `--eval-seed`. Run
`python train_victim.py --help` for the complete list.

## Train and evaluate the adversary

The adversary is a PPO policy with GTrXL attention. Its observation is
`[x, x_dot, theta, theta_dot, last_victim_action, previous_wind]`, and its
continuous action is clipped to `[-max_wind, max_wind]` before being added to
the victim's CartPole force.

```bash
python train_adversary.py \
  --victim-checkpoint checkpoints/victim \
  --victim-env cartpole \
  --iters 10 \
  --max-wind 4.0 \
  --wind-sigma 1.0 \
  --failure-bonus 1000 \
  --out-dir checkpoints/adversary
```

The reward encourages victim failure and penalizes unlikely wind through
`0.5 * (wind / wind_sigma)^2`. Training reports the failure rate, mean episode
length, mean absolute wind, wind penalty, and earliest observed failure.

Evaluate a saved adversary and export each evaluation episode's wind history:

```bash
python train_adversary.py \
  --eval-only \
  --checkpoint checkpoints/adversary \
  --victim-checkpoint checkpoints/victim \
  --victim-env cartpole \
  --eval-episodes 100 \
  --wind-history-out adversary_wind_history.npz
```

As with victim evaluation, `--victim-env` must match the victim checkpoint.
Environment parameters such as `--max-wind`, `--wind-sigma`, `--horizon`, and
`--failure-bonus` should also match the adversary's training configuration.

## Robustness analysis and plots

### Constant-wind sweep

Measure victim success under fixed wind values and optionally save the
zero-wind episode-length histogram:

```bash
python eval_constant_wind.py \
  --victim-checkpoint checkpoints/victim \
  --min-wind -4 \
  --max-wind 4 \
  --step 0.5 \
  --episodes 100 \
  --histogram \
  --histogram-path unstressed_reward_histogram.png
```

The script compares each success rate with the zero-wind baseline and prints
the largest tested wind that remains within `--tolerance` of that baseline.

### Random-wind or adversarial episode lengths

Generate a survival-length histogram under Gaussian random wind:

```bash
python plot_episode_length_histogram.py \
  --mode random \
  --victim-checkpoint checkpoints/victim \
  --out-path random_wind_episode_lengths.png
```

Or evaluate a learned adversary:

```bash
python plot_episode_length_histogram.py \
  --mode adversary \
  --victim-checkpoint checkpoints/victim \
  --adversary-checkpoint checkpoints/adversary \
  --out-path adversary_episode_lengths.png
```

### Wind-history plots

Plot an NPZ exported by `train_adversary.py`:

```bash
python plot_wind_history.py \
  --input adversary_wind_history.npz \
  --out-path adversary_wind_history.png \
  --show-mean --show-std --show-failures
```

Use `--episodes 0,1,2` to select specific episodes or `--failures-only` to
show only failed episodes. The same tool accepts the long-form CSV produced by
the MCMC workflow below.

## MCMC failure-trace sampling

`mcmc_failure_trace.py` runs a random-walk Metropolis chain over bounded wind
traces conditioned on deterministic victim failure. By default it starts from
episode 6 in `adversary_wind_history_same_seeds.npz`, pads its 289-step trace
to 350 steps, and verifies that seed 1129 and the selected victim checkpoint
reproduce the recorded failure before sampling.

```bash
python mcmc_failure_trace.py \
  --victim-checkpoint checkpoints/victim \
  --iterations 10000 \
  --proposal-sigma 0.01 \
  --natural-wind-sigma 1.0 \
  --output episode_6_mcmc.npz
```

`--proposal-sigma` controls proposal size; `--natural-wind-sigma` independently
controls the Gaussian target density used in Metropolis acceptance. Proposals
can perturb the whole trace (`--proposal-mode all`), one timestep (`single`),
or a contiguous region (`block`, with `--block-size`).

Every run writes:

- a compressed NPZ containing the full chain, acceptance diagnostics, failure
  steps, and reproducibility metadata; and
- a long-form CSV containing the initial trace and every newly accepted
  failure trace. Its path defaults to the NPZ name with a `.csv` suffix.

Plot the chain after optional burn-in and thinning:

```bash
python plot_wind_history.py \
  --input episode_6_mcmc.npz \
  --burn-in 1000 \
  --thin 10 \
  --show-mean \
  --out-path episode_6_mcmc.png
```

Or plot the accepted traces from CSV:

```bash
python plot_wind_history.py --input episode_6_mcmc.csv --thin 10 --show-mean --out-path episode_6_mcmc_accepted.png
```

If source replay fails, verify `--episode-index`, `--env-seed`,
`--victim-checkpoint`, and `--victim-env`; MCMC deliberately refuses to start
from a trace that does not reproduce a failure.

## Weights & Biases and rollout videos

Add `--wandb` to either training script to log metrics. Use
`--wandb-mode offline` when network logging is not available.

Victim training can log selected CartPole rollout iterations:

```bash
python train_victim.py --wandb --wandb-video --wandb-video-iters 1,5,10
```

Adversary evaluation can log annotated deterministic rollout videos and retain
MP4 copies locally:

```bash
python train_adversary.py \
  --eval-only \
  --checkpoint checkpoints/adversary \
  --victim-checkpoint checkpoints/victim \
  --wandb --wandb-video
```

Video rendering is supported for the standard CartPole victim and may require
the system FFmpeg executable used by MoviePy.

## Tests

The lightweight unit tests cover MCMC proposal/acceptance behavior, source
trace loading, CSV export, and wind-history plotting:

```bash
python -m unittest discover -v
```

## Repository layout

- `main.py` — Transformer versus plain PPO comparison
- `train_victim.py` — victim training, evaluation, checkpointing, and videos
- `train_adversary.py` — Transformer adversary training and evaluation
- `envs/adversarial_cartpole.py` — adversarial CartPole dynamics and reward
- `eval_constant_wind.py` — deterministic constant-wind robustness sweep
- `plot_episode_length_histogram.py` — random/adversarial survival histograms
- `plot_wind_history.py` — NPZ/CSV wind-trace visualization
- `mcmc_failure_trace.py` — failure-conditioned MCMC sampler
- `analysis.ipynb` — exploratory analysis notebook
