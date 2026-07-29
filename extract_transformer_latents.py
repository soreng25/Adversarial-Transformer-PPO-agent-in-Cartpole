"""Extract transformer representations while replaying MCMC failure traces.

This script is intentionally separate from ``analysis.ipynb`` because replaying
the environment is the expensive part.  Run it with the Python 3.8 / Ray 2.10
environment that matches the saved victim and adversary checkpoints.

The MCMC chain contains repeated rows whenever a proposal was rejected.  Those
rows would produce exactly the same environment trajectory and transformer
representation, so this script replays each distinct chain state once and saves
a mapping back to all MCMC rows.

The extractor maintains the model's intended 50-step rolling attention memory,
matching the history supplied by RLlib during training.  This is deliberate:
the older manual evaluation helper in ``train_adversary.py`` carries only the
most recent state output instead of constructing the full shifted memory.
"""

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import ray
import torch
from ray.rllib.policy.policy import Policy

from envs.adversarial_cartpole import AdversarialCartPoleEnv
from mcmc_failure_trace import VictimPolicyRunner


DEFAULT_MCMC_PATH = "500000iteration_episode6_mcmc.npz"
DEFAULT_VICTIM_POLICY = os.path.join(
    "checkpoints",
    "victim_cartpole_50",
    "policies",
    "default_policy",
)
DEFAULT_ADVERSARY_POLICY = os.path.join(
    "checkpoints",
    "adversary_maxwind1_sigma1_bonus1000_iters100",
    "policies",
    "default_policy",
)
DEFAULT_OUTPUT = "transformer_latent_summaries.npz"
FINGERPRINT_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class TraceLatentSummary:
    """Compact transformer representation of one complete failure trace."""

    mean: np.ndarray
    final_window_mean: np.ndarray
    final: np.ndarray
    step_count: int
    replayed_failure_step: int


def content_fingerprint(path):
    """Return a stable SHA-256 fingerprint for a file or directory."""

    path = os.path.abspath(path)
    digest = hashlib.sha256()

    if os.path.isfile(path):
        relative_files = [("", path)]
    elif os.path.isdir(path):
        relative_files = []
        for root, directories, filenames in os.walk(path):
            directories.sort()
            for filename in sorted(filenames):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, path).replace(
                    os.sep,
                    "/",
                )
                relative_files.append((relative_path, full_path))
    else:
        raise FileNotFoundError(path)

    for relative_path, full_path in relative_files:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with open(full_path, "rb") as input_file:
            while True:
                chunk = input_file.read(FINGERPRINT_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")

    return digest.hexdigest()


def build_run_identity(args, mcmc):
    """Describe the exact data, models, and environment used for extraction."""

    return {
        "mcmc_path": os.path.abspath(args.mcmc_path),
        "mcmc_file_sha256": content_fingerprint(args.mcmc_path),
        "victim_policy_checkpoint": os.path.abspath(
            args.victim_policy_checkpoint
        ),
        "victim_policy_sha256": content_fingerprint(
            args.victim_policy_checkpoint
        ),
        "adversary_policy_checkpoint": os.path.abspath(
            args.adversary_policy_checkpoint
        ),
        "adversary_policy_sha256": content_fingerprint(
            args.adversary_policy_checkpoint
        ),
        "env_seed": int(mcmc["env_seed"]),
        "horizon": int(mcmc["horizon"]),
        "max_wind": float(mcmc["max_wind"]),
        "natural_wind_sigma": float(mcmc["natural_wind_sigma"]),
    }


def build_chain_layout(accepted):
    """Return the distinct chain rows and a mapping from every row to them.

    If proposal ``i`` was rejected, chain row ``i + 1`` is identical to chain
    row ``i``.  If it was accepted, row ``i + 1`` starts a new distinct state.
    """

    accepted = np.asarray(accepted, dtype=bool)
    unique_chain_rows = np.concatenate(
        (
            np.array([0], dtype=np.int32),
            np.flatnonzero(accepted).astype(np.int32) + 1,
        )
    )

    chain_state_ids = np.zeros(len(accepted) + 1, dtype=np.int32)
    chain_state_ids[1:] = np.cumsum(accepted, dtype=np.int32)
    chain_weights = np.bincount(
        chain_state_ids,
        minlength=len(unique_chain_rows),
    ).astype(np.int32)

    return unique_chain_rows, chain_state_ids, chain_weights


def validate_rejected_rows(
    chain,
    failure_steps,
    accepted,
    chunk_size=4096,
):
    """Confirm rejected proposals really repeat the preceding chain state."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    rejected_proposals = np.flatnonzero(~np.asarray(accepted, dtype=bool))
    for start in range(0, len(rejected_proposals), chunk_size):
        proposal_indices = rejected_proposals[start : start + chunk_size]
        previous_rows = chain[proposal_indices]
        repeated_rows = chain[proposal_indices + 1]
        same_trace = np.all(previous_rows == repeated_rows, axis=1)
        same_failure_step = (
            failure_steps[proposal_indices]
            == failure_steps[proposal_indices + 1]
        )
        valid = same_trace & same_failure_step
        if not np.all(valid):
            bad_offset = int(np.flatnonzero(~valid)[0])
            proposal_index = int(proposal_indices[bad_offset])
            raise ValueError(
                "rejected proposal {} does not repeat chain row {} and its "
                "failure step".format(
                    proposal_index,
                    proposal_index,
                )
            )


def evenly_spaced_indices(length, count):
    """Choose up to ``count`` deterministic indices spanning a sequence."""

    if length < 1:
        raise ValueError("length must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    count = min(length, count)
    return np.unique(
        np.linspace(0, length - 1, num=count, dtype=np.int32)
    )


def summarize_latent_steps(latent_steps, final_window):
    """Summarize a trace's per-step latent vectors without losing dimensions."""

    latent_steps = np.asarray(latent_steps, dtype=np.float32)
    if latent_steps.ndim != 2 or latent_steps.shape[0] < 1:
        raise ValueError("latent_steps must have shape (steps, latent_dim)")
    if final_window < 1:
        raise ValueError("final_window must be positive")

    return (
        np.mean(latent_steps, axis=0, dtype=np.float64).astype(np.float32),
        np.mean(
            latent_steps[-final_window:],
            axis=0,
            dtype=np.float64,
        ).astype(np.float32),
        latent_steps[-1].copy(),
    )


def load_single_policy(checkpoint, name):
    """Restore one policy directly, without starting a Ray Algorithm."""

    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(
            "{} policy checkpoint was not found: {}".format(name, checkpoint)
        )

    restored = Policy.from_checkpoint(checkpoint)
    if isinstance(restored, dict):
        if "default_policy" in restored:
            restored = restored["default_policy"]
        elif len(restored) == 1:
            restored = next(iter(restored.values()))
        else:
            raise ValueError(
                "{} checkpoint contains multiple policies".format(name)
            )

    restored.model.eval()
    return restored


def current_model_latent(policy, latent_dim):
    """Copy the most recent 64D GTrXL feature produced by model inference."""

    features = getattr(policy.model, "_features", None)
    if features is None:
        raise RuntimeError(
            "the adversary model did not expose its pre-action _features"
        )

    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)

    if features.size != latent_dim:
        raise RuntimeError(
            "expected exactly one {}D transformer feature vector, got shape "
            "{}".format(latent_dim, features.shape)
        )

    # RLlib may retain a singleton batch/time dimension.
    latent = features.reshape(latent_dim).copy()
    if not np.all(np.isfinite(latent)):
        raise RuntimeError("the transformer produced non-finite latent values")
    return latent


def initialize_attention_memory(policy, memory_length, latent_dim):
    """Create zero-padded rolling memory in the shape GTrXL expects."""

    if memory_length < 1:
        raise ValueError("memory_length must be positive")

    initial_state = policy.get_initial_state()
    if not initial_state:
        raise RuntimeError("the adversary policy did not provide attention state")

    memory = []
    for state_number, value in enumerate(initial_state):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.size != latent_dim:
            raise RuntimeError(
                "attention state {} has {} values; expected {}".format(
                    state_number,
                    value.size,
                    latent_dim,
                )
            )
        memory.append(
            np.repeat(value[np.newaxis, :], memory_length, axis=0)
        )
    return memory


def advance_attention_memory(memory, state_out, latent_dim):
    """Append the current state output and discard the oldest memory row."""

    if len(memory) != len(state_out):
        raise RuntimeError(
            "received {} attention outputs for {} memory tensors".format(
                len(state_out),
                len(memory),
            )
        )

    updated = []
    for state_number, (past, current) in enumerate(zip(memory, state_out)):
        past = np.asarray(past, dtype=np.float32)
        current = np.asarray(current, dtype=np.float32)
        if past.ndim != 2 or past.shape[1] != latent_dim:
            raise RuntimeError(
                "attention memory {} has invalid shape {}".format(
                    state_number,
                    past.shape,
                )
            )
        if current.size != latent_dim:
            raise RuntimeError(
                "attention output {} must contain exactly one {}D vector; "
                "got shape {}".format(
                    state_number,
                    latent_dim,
                    current.shape,
                )
            )
        current = current.reshape(latent_dim)
        updated.append(
            np.concatenate((past[1:], current[np.newaxis, :]), axis=0)
        )
    return updated


def replay_and_summarize(
    env,
    adversary_policy,
    trace,
    expected_failure_step,
    env_seed,
    latent_dim,
    memory_length,
    final_window,
):
    """Replay one stored wind trace and collect its pre-action latents."""

    trace = np.asarray(trace, dtype=np.float32)
    expected_failure_step = int(expected_failure_step)
    if expected_failure_step < 1 or expected_failure_step > len(trace):
        raise ValueError(
            "failure step {} is outside trace length {}".format(
                expected_failure_step,
                len(trace),
            )
        )

    observation, _ = env.reset(seed=int(env_seed))
    memory = initialize_attention_memory(
        adversary_policy,
        memory_length,
        latent_dim,
    )
    latent_steps = np.empty(
        (expected_failure_step, latent_dim),
        dtype=np.float32,
    )
    final_info = None

    for timestep in range(expected_failure_step):
        # This action is deliberately ignored.  Calling the policy updates the
        # transformer memory and exposes how it represents the current history.
        _, state_out, _ = adversary_policy.compute_single_action(
            observation,
            state=memory,
            explore=False,
            timestep=timestep,
        )
        latent_steps[timestep] = current_model_latent(
            adversary_policy,
            latent_dim,
        )

        # Force the stored MCMC wind instead of the adversary's proposed action.
        observation, _, terminated, truncated, info = env.step(
            np.array([trace[timestep]], dtype=np.float32)
        )
        memory = advance_attention_memory(
            memory,
            state_out,
            latent_dim,
        )
        final_info = info

        ended = bool(terminated or truncated)
        is_expected_final_step = timestep + 1 == expected_failure_step
        if ended and not is_expected_final_step:
            raise RuntimeError(
                "replay ended at step {}, expected step {}".format(
                    timestep + 1,
                    expected_failure_step,
                )
            )
        if is_expected_final_step and not ended:
            raise RuntimeError(
                "replay did not end at stored failure step {}".format(
                    expected_failure_step
                )
            )

    if final_info is None or not bool(final_info["victim_failed"]):
        raise RuntimeError(
            "trace ended without reproducing the stored victim failure"
        )
    replayed_failure_step = int(final_info["episode_len"])
    if replayed_failure_step != expected_failure_step:
        raise RuntimeError(
            "replayed failure step {} does not match stored step {}".format(
                replayed_failure_step,
                expected_failure_step,
            )
        )

    mean, final_window_mean, final = summarize_latent_steps(
        latent_steps,
        final_window,
    )
    return TraceLatentSummary(
        mean=mean,
        final_window_mean=final_window_mean,
        final=final,
        step_count=len(latent_steps),
        replayed_failure_step=replayed_failure_step,
    )


def load_mcmc_data(path):
    """Load and validate only the arrays required for latent extraction."""

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = {
            "chain",
            "failure_steps",
            "accepted",
            "env_seed",
            "max_wind",
            "horizon",
            "natural_wind_sigma",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                "MCMC file is missing required arrays: {}".format(missing)
            )

        result = {
            "chain": np.asarray(data["chain"], dtype=np.float32),
            "failure_steps": np.asarray(data["failure_steps"], dtype=np.int32),
            "accepted": np.asarray(data["accepted"], dtype=bool),
            "env_seed": int(data["env_seed"]),
            "max_wind": float(data["max_wind"]),
            "horizon": int(data["horizon"]),
            "natural_wind_sigma": float(data["natural_wind_sigma"]),
        }

    chain = result["chain"]
    failure_steps = result["failure_steps"]
    accepted = result["accepted"]
    if chain.ndim != 2:
        raise ValueError("chain must be a two-dimensional array")
    if failure_steps.shape != (chain.shape[0],):
        raise ValueError("failure_steps does not match the chain row count")
    if accepted.shape != (chain.shape[0] - 1,):
        raise ValueError("accepted must contain one entry per MCMC proposal")
    if np.any(failure_steps < 1) or np.any(failure_steps > chain.shape[1]):
        raise ValueError("failure_steps contains an out-of-range value")
    validate_rejected_rows(chain, failure_steps, accepted)

    return result


def make_environment(mcmc, victim_policy):
    """Recreate the deterministic environment used for MCMC replay."""

    env = AdversarialCartPoleEnv(
        {
            "victim_checkpoint": None,
            "victim_env": "cartpole",
            "max_wind": mcmc["max_wind"],
            "wind_sigma": mcmc["natural_wind_sigma"],
            "horizon": mcmc["horizon"],
            "failure_bonus": 0.0,
        }
    )
    env.victim_algo = VictimPolicyRunner(victim_policy)
    return env


def validate_runtime():
    """Fail early when the active environment cannot read these checkpoints."""

    if sys.version_info[:2] != (3, 8):
        raise RuntimeError(
            "use the checkpoint environment with Python 3.8; found {}".format(
                sys.version.split()[0]
            )
        )
    if ray.__version__ != "2.10.0":
        raise RuntimeError(
            "use Ray 2.10.0 for these checkpoints; found {}".format(
                ray.__version__
            )
        )


def output_payload(
    unique_chain_rows,
    chain_state_ids,
    chain_weights,
    unique_failure_steps,
    mean_latents,
    final_window_mean_latents,
    final_latents,
    latent_step_counts,
    replayed_failure_steps,
    completed,
    final_window,
    latent_dim,
    memory_length,
    run_identity,
):
    """Build the stable NPZ interface consumed later by the notebook."""

    payload = {
        "unique_chain_rows": unique_chain_rows,
        "chain_state_ids": chain_state_ids,
        "chain_weights": chain_weights,
        "failure_steps": unique_failure_steps,
        "mean_latents": mean_latents,
        "final_window_mean_latents": final_window_mean_latents,
        "final_latents": final_latents,
        "latent_step_counts": latent_step_counts,
        "replayed_failure_steps": replayed_failure_steps,
        "completed": completed,
        "latent_dim": np.asarray(latent_dim, dtype=np.int32),
        "attention_memory_length": np.asarray(
            memory_length,
            dtype=np.int32,
        ),
        "final_window": np.asarray(final_window, dtype=np.int32),
    }
    for name, value in run_identity.items():
        payload[name] = np.asarray(value)
    return payload


def save_output(path, payload):
    """Atomically replace the output so an interruption cannot corrupt it."""

    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def initialize_output_arrays(number_of_states, latent_dim):
    """Create empty arrays, using NaN to make incomplete rows obvious."""

    shape = (number_of_states, latent_dim)
    return {
        "mean_latents": np.full(shape, np.nan, dtype=np.float32),
        "final_window_mean_latents": np.full(
            shape,
            np.nan,
            dtype=np.float32,
        ),
        "final_latents": np.full(shape, np.nan, dtype=np.float32),
        "latent_step_counts": np.zeros(number_of_states, dtype=np.int32),
        "replayed_failure_steps": np.full(
            number_of_states,
            -1,
            dtype=np.int32,
        ),
        "completed": np.zeros(number_of_states, dtype=bool),
    }


def validate_output_arrays(
    arrays,
    unique_failure_steps,
    latent_dim,
    require_complete=False,
):
    """Validate partial or complete summary arrays before trusting or saving."""

    unique_failure_steps = np.asarray(unique_failure_steps, dtype=np.int32)
    number_of_states = len(unique_failure_steps)
    vector_shape = (number_of_states, latent_dim)
    scalar_shape = (number_of_states,)

    for name in (
        "mean_latents",
        "final_window_mean_latents",
        "final_latents",
    ):
        if arrays[name].shape != vector_shape:
            raise ValueError(
                "output array {} has shape {}, expected {}".format(
                    name,
                    arrays[name].shape,
                    vector_shape,
                )
            )
    for name in (
        "latent_step_counts",
        "replayed_failure_steps",
        "completed",
    ):
        if arrays[name].shape != scalar_shape:
            raise ValueError(
                "output array {} has shape {}, expected {}".format(
                    name,
                    arrays[name].shape,
                    scalar_shape,
                )
            )

    completed = np.asarray(arrays["completed"], dtype=bool)
    incomplete = ~completed
    if require_complete and not np.all(completed):
        raise RuntimeError("not every distinct state has been completed")

    if np.any(completed):
        for name in (
            "mean_latents",
            "final_window_mean_latents",
            "final_latents",
        ):
            if not np.all(np.isfinite(arrays[name][completed])):
                raise ValueError(
                    "completed rows in {} contain non-finite values".format(
                        name
                    )
                )
        if not np.array_equal(
            arrays["latent_step_counts"][completed],
            unique_failure_steps[completed],
        ):
            raise ValueError(
                "completed latent step counts do not match failure steps"
            )
        if not np.array_equal(
            arrays["replayed_failure_steps"][completed],
            unique_failure_steps[completed],
        ):
            raise ValueError(
                "completed replayed failure steps do not match stored steps"
            )

    if np.any(incomplete):
        for name in (
            "mean_latents",
            "final_window_mean_latents",
            "final_latents",
        ):
            if not np.all(np.isnan(arrays[name][incomplete])):
                raise ValueError(
                    "incomplete rows in {} must remain NaN".format(name)
                )
        if np.any(arrays["latent_step_counts"][incomplete] != 0):
            raise ValueError(
                "incomplete rows must have zero latent step counts"
            )
        if np.any(arrays["replayed_failure_steps"][incomplete] != -1):
            raise ValueError(
                "incomplete rows must have replayed failure step -1"
            )


def validate_resume_identity(saved, run_identity):
    """Reject a resume file produced from different data or checkpoints."""

    missing = sorted(set(run_identity).difference(saved.files))
    if missing:
        raise ValueError(
            "resume file is missing identity arrays: {}".format(missing)
        )

    for name, expected in run_identity.items():
        stored_array = np.asarray(saved[name])
        if stored_array.shape != ():
            raise ValueError(
                "resume identity {} must be a scalar".format(name)
            )
        stored = stored_array.item()
        if stored != expected:
            raise ValueError(
                "resume file uses a different {}".format(
                    name.replace("_", " ")
                )
            )


def load_resume_arrays(
    path,
    unique_chain_rows,
    chain_state_ids,
    chain_weights,
    unique_failure_steps,
    latent_dim,
    memory_length,
    final_window,
    run_identity,
):
    """Load a partial output and ensure it belongs to this exact extraction."""

    with np.load(path, allow_pickle=False) as saved:
        required = {
            "unique_chain_rows",
            "chain_state_ids",
            "chain_weights",
            "mean_latents",
            "final_window_mean_latents",
            "final_latents",
            "latent_step_counts",
            "replayed_failure_steps",
            "completed",
            "latent_dim",
            "attention_memory_length",
            "final_window",
            "failure_steps",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError(
                "resume file is missing arrays: {}".format(missing)
            )

        if not np.array_equal(saved["unique_chain_rows"], unique_chain_rows):
            raise ValueError("resume file uses different distinct chain rows")
        if not np.array_equal(saved["chain_state_ids"], chain_state_ids):
            raise ValueError("resume file uses a different MCMC row mapping")
        if not np.array_equal(saved["chain_weights"], chain_weights):
            raise ValueError("resume file uses different MCMC weights")
        if int(saved["latent_dim"]) != latent_dim:
            raise ValueError("resume file uses a different latent dimension")
        if int(saved["attention_memory_length"]) != memory_length:
            raise ValueError("resume file uses a different attention memory")
        if int(saved["final_window"]) != final_window:
            raise ValueError("resume file uses a different final window")
        if not np.array_equal(saved["failure_steps"], unique_failure_steps):
            raise ValueError("resume file uses different failure steps")
        validate_resume_identity(saved, run_identity)

        arrays = {
            "mean_latents": saved["mean_latents"].copy(),
            "final_window_mean_latents": (
                saved["final_window_mean_latents"].copy()
            ),
            "final_latents": saved["final_latents"].copy(),
            "latent_step_counts": saved["latent_step_counts"].copy(),
            "replayed_failure_steps": saved[
                "replayed_failure_steps"
            ].copy(),
            "completed": saved["completed"].astype(bool, copy=True),
        }

    validate_output_arrays(
        arrays,
        unique_failure_steps,
        latent_dim,
    )
    return arrays


def run_validation(
    env,
    adversary_policy,
    mcmc,
    unique_chain_rows,
    latent_dim,
    memory_length,
    args,
):
    """Reproduce a small, evenly distributed set before the expensive run."""

    positions = evenly_spaced_indices(
        len(unique_chain_rows),
        args.validation_traces,
    )
    rows = unique_chain_rows[positions]
    print(
        "Validating {} distinct states at chain rows: {}".format(
            len(rows),
            ", ".join(str(int(row)) for row in rows),
        )
    )

    first_summary = None
    for number, row in enumerate(rows, start=1):
        summary = replay_and_summarize(
            env=env,
            adversary_policy=adversary_policy,
            trace=mcmc["chain"][row],
            expected_failure_step=mcmc["failure_steps"][row],
            env_seed=mcmc["env_seed"],
            latent_dim=latent_dim,
            memory_length=memory_length,
            final_window=args.final_window,
        )
        print(
            "  [{}/{}] row {} reproduced failure at step {}".format(
                number,
                len(rows),
                int(row),
                summary.replayed_failure_step,
            )
        )
        if number == 1:
            first_summary = summary

    # Repeat one row to catch forgotten environment or transformer resets.
    repeated = replay_and_summarize(
        env=env,
        adversary_policy=adversary_policy,
        trace=mcmc["chain"][rows[0]],
        expected_failure_step=mcmc["failure_steps"][rows[0]],
        env_seed=mcmc["env_seed"],
        latent_dim=latent_dim,
        memory_length=memory_length,
        final_window=args.final_window,
    )
    np.testing.assert_array_equal(first_summary.mean, repeated.mean)
    np.testing.assert_array_equal(
        first_summary.final_window_mean,
        repeated.final_window_mean,
    )
    np.testing.assert_array_equal(first_summary.final, repeated.final)
    print("Validation passed, including deterministic repeated replay.")


def run_extraction(
    env,
    adversary_policy,
    mcmc,
    unique_chain_rows,
    chain_state_ids,
    chain_weights,
    latent_dim,
    memory_length,
    run_identity,
    args,
):
    """Extract compact latent summaries, saving periodically for safe resume."""

    unique_failure_steps = mcmc["failure_steps"][unique_chain_rows]
    output_exists = os.path.exists(args.output)
    if output_exists and not args.resume:
        raise FileExistsError(
            "{} already exists; use --resume or choose another --output".format(
                args.output
            )
        )

    if output_exists:
        arrays = load_resume_arrays(
            args.output,
            unique_chain_rows,
            chain_state_ids,
            chain_weights,
            unique_failure_steps,
            latent_dim,
            memory_length,
            args.final_window,
            run_identity,
        )
        print(
            "Resuming with {}/{} distinct states already complete.".format(
                int(np.sum(arrays["completed"])),
                len(unique_chain_rows),
            )
        )
    else:
        arrays = initialize_output_arrays(
            len(unique_chain_rows),
            latent_dim,
        )

    pending_positions = np.flatnonzero(~arrays["completed"])
    if args.max_unique_traces is not None:
        pending_positions = pending_positions[: args.max_unique_traces]
    if len(pending_positions) == 0:
        validate_output_arrays(
            arrays,
            unique_failure_steps,
            latent_dim,
            require_complete=True,
        )
        if int(np.sum(chain_weights)) != len(chain_state_ids):
            raise RuntimeError(
                "saved MCMC weights do not cover all chain rows"
            )
        print("No pending distinct states to extract.")
        return

    started = time.time()
    processed_this_run = 0

    for unique_position in pending_positions:
        chain_row = int(unique_chain_rows[unique_position])
        summary = replay_and_summarize(
            env=env,
            adversary_policy=adversary_policy,
            trace=mcmc["chain"][chain_row],
            expected_failure_step=unique_failure_steps[unique_position],
            env_seed=mcmc["env_seed"],
            latent_dim=latent_dim,
            memory_length=memory_length,
            final_window=args.final_window,
        )

        arrays["mean_latents"][unique_position] = summary.mean
        arrays["final_window_mean_latents"][
            unique_position
        ] = summary.final_window_mean
        arrays["final_latents"][unique_position] = summary.final
        arrays["latent_step_counts"][
            unique_position
        ] = summary.step_count
        arrays["replayed_failure_steps"][
            unique_position
        ] = summary.replayed_failure_step
        arrays["completed"][unique_position] = True
        processed_this_run += 1

        should_report = (
            processed_this_run == 1
            or processed_this_run % args.progress_every == 0
            or processed_this_run == len(pending_positions)
        )
        if should_report:
            elapsed = time.time() - started
            rate = processed_this_run / max(elapsed, 1e-9)
            print(
                "Processed {}/{} this run; {}/{} total "
                "({:.2f} traces/second).".format(
                    processed_this_run,
                    len(pending_positions),
                    int(np.sum(arrays["completed"])),
                    len(unique_chain_rows),
                    rate,
                )
            )

        should_save = (
            processed_this_run % args.save_every == 0
            or processed_this_run == len(pending_positions)
        )
        if should_save:
            is_complete = bool(np.all(arrays["completed"]))
            validate_output_arrays(
                arrays,
                unique_failure_steps,
                latent_dim,
                require_complete=is_complete,
            )
            if (
                is_complete
                and int(np.sum(chain_weights)) != len(chain_state_ids)
            ):
                raise RuntimeError(
                    "saved MCMC weights do not cover all chain rows"
                )
            payload = output_payload(
                unique_chain_rows=unique_chain_rows,
                chain_state_ids=chain_state_ids,
                chain_weights=chain_weights,
                unique_failure_steps=unique_failure_steps,
                final_window=args.final_window,
                latent_dim=latent_dim,
                memory_length=memory_length,
                run_identity=run_identity,
                **arrays
            )
            save_output(args.output, payload)

    complete_count = int(np.sum(arrays["completed"]))
    print(
        "Saved {}/{} distinct latent summaries to {}".format(
            complete_count,
            len(unique_chain_rows),
            os.path.abspath(args.output),
        )
    )
    if complete_count < len(unique_chain_rows):
        print("Run again with --resume to continue the partial extraction.")
    else:
        validate_output_arrays(
            arrays,
            unique_failure_steps,
            latent_dim,
            require_complete=True,
        )
        if int(np.sum(chain_weights)) != len(chain_state_ids):
            raise RuntimeError("saved MCMC weights do not cover all chain rows")
        print(
            "Extraction complete.  The {} distinct states represent all {} "
            "MCMC rows through chain_state_ids.".format(
                len(unique_chain_rows),
                len(chain_state_ids),
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay distinct MCMC failure traces and save compact 64D "
            "transformer latent summaries."
        )
    )
    parser.add_argument("--mcmc-path", default=DEFAULT_MCMC_PATH)
    parser.add_argument(
        "--victim-policy-checkpoint",
        default=DEFAULT_VICTIM_POLICY,
    )
    parser.add_argument(
        "--adversary-policy-checkpoint",
        default=DEFAULT_ADVERSARY_POLICY,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="replay a small sample and do not write an output file",
    )
    parser.add_argument("--validation-traces", type=int, default=12)
    parser.add_argument(
        "--final-window",
        type=int,
        default=50,
        help="number of final pre-failure latents in the second summary",
    )
    parser.add_argument(
        "--max-unique-traces",
        type=int,
        default=None,
        help="process only this many pending states, then save for --resume",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing partial output",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads used by PyTorch inference",
    )
    return parser.parse_args()


def validate_args(args):
    for name in (
        "validation_traces",
        "final_window",
        "progress_every",
        "save_every",
        "torch_threads",
    ):
        if getattr(args, name) < 1:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))
    if args.max_unique_traces is not None and args.max_unique_traces < 1:
        raise ValueError("--max-unique-traces must be positive")
    if args.validate_only and args.resume:
        raise ValueError("--validate-only and --resume cannot be combined")


def main():
    args = parse_args()
    validate_args(args)
    validate_runtime()
    torch.set_num_threads(args.torch_threads)

    print(
        "Runtime: Python {}, Ray {}, Torch {}".format(
            sys.version.split()[0],
            ray.__version__,
            torch.__version__,
        )
    )
    mcmc = load_mcmc_data(args.mcmc_path)
    print("Fingerprinting MCMC data and policy checkpoints...")
    run_identity = build_run_identity(args, mcmc)
    unique_chain_rows, chain_state_ids, chain_weights = build_chain_layout(
        mcmc["accepted"]
    )
    print(
        "Loaded {} MCMC rows containing {} distinct chain states.".format(
            len(chain_state_ids),
            len(unique_chain_rows),
        )
    )

    print("Restoring victim policy...")
    victim_policy = load_single_policy(
        args.victim_policy_checkpoint,
        "victim",
    )
    print("Restoring adversary transformer policy...")
    adversary_policy = load_single_policy(
        args.adversary_policy_checkpoint,
        "adversary",
    )

    model_config = adversary_policy.config.get("model", {})
    latent_dim = int(model_config.get("attention_dim", 0))
    memory_length = int(model_config.get("attention_memory_inference", 0))
    if latent_dim != 64 or not bool(model_config.get("use_attention")):
        raise RuntimeError(
            "expected a 64D attention model, got use_attention={} and "
            "attention_dim={}".format(
                model_config.get("use_attention"),
                latent_dim,
            )
        )
    if memory_length != 50:
        raise RuntimeError(
            "expected 50-step attention memory, got {}".format(memory_length)
        )
    if int(model_config.get("attention_use_n_prev_actions", 0)) != 0:
        raise RuntimeError(
            "this extractor does not supply previous actions to attention"
        )
    if int(model_config.get("attention_use_n_prev_rewards", 0)) != 0:
        raise RuntimeError(
            "this extractor does not supply previous rewards to attention"
        )
    print(
        "Adversary transformer latent dimension: {}; memory: {} steps "
        "across {} tensor(s).".format(
            latent_dim,
            memory_length,
            len(adversary_policy.get_initial_state()),
        )
    )

    env = make_environment(mcmc, victim_policy)
    try:
        if args.validate_only:
            run_validation(
                env,
                adversary_policy,
                mcmc,
                unique_chain_rows,
                latent_dim,
                memory_length,
                args,
            )
        else:
            run_extraction(
                env,
                adversary_policy,
                mcmc,
                unique_chain_rows,
                chain_state_ids,
                chain_weights,
                latent_dim,
                memory_length,
                run_identity,
                args,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
