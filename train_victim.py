"""Train and save a frozen PPO victim for adversarial CartPole."""

import argparse
import os
import time

import gymnasium as gym
import ray
import torch
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from envs.victim_history import make_stacked_stateless_cartpole


#Assign name and register the environment
NORMAL_VICTIM_ENV_ID = "victim_cartpole"
STATELESS_VICTIM_ENV_ID = "victim_stateless_cartpole"


def make_cartpole(env_config=None):
    base = gym.make("CartPole-v1")
    base_step = base.step

    def step(action):
        obs, reward, terminated, truncated, info = base_step(action)
        info["victim_failed"] = bool(terminated)
        return obs, reward, terminated, truncated, info

    base.step = step
    return base


def register_env():
    tune.register_env(NORMAL_VICTIM_ENV_ID, make_cartpole)
    tune.register_env(STATELESS_VICTIM_ENV_ID, make_stacked_stateless_cartpole)


class VictimMetricsCallback(DefaultCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["victim_failed"] = 0.0

    def on_episode_step(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if info and info.get("victim_failed"):
            episode.user_data["victim_failed"] = 1.0

    def on_episode_end(self, *, episode, **kwargs):
        episode.custom_metrics["victim_failure_rate"] = episode.user_data[
            "victim_failed"
        ]


# Handles checkpoints
def checkpoint_path(checkpoint):
    if hasattr(checkpoint, "checkpoint") and hasattr(checkpoint.checkpoint, "path"):
        return checkpoint.checkpoint.path
    return getattr(checkpoint, "path", str(checkpoint))


def parse_hidden_sizes(value):
    return [int(size.strip()) for size in value.split(",") if size.strip()]


def cuda_status():
    if not torch.cuda.is_available():
        return "cuda_available=False"
    return f"cuda_available=True  gpu={torch.cuda.get_device_name(0)}"


def configure_api_stack(config):
    if not hasattr(config, "api_stack"):
        return config

    return config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )


def configure_rollout_workers(config, args):
    if hasattr(config, "env_runners"):
        return config.env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
        )

    return config.rollouts(
        num_rollout_workers=args.num_env_runners,
        num_envs_per_worker=args.num_envs_per_env_runner,
    )


def victim_env_id(args):
    if args.env == "cartpole":
        return NORMAL_VICTIM_ENV_ID
    return STATELESS_VICTIM_ENV_ID


def make_eval_env(args):
    if args.env == "cartpole":
        return make_cartpole()
    return make_stacked_stateless_cartpole()


def build_config(args):
    num_gpus = 1 if torch.cuda.is_available() else 0
    config = (
        configure_api_stack(PPOConfig()) #starts a PPO object
        .resources(num_gpus=num_gpus)
        .environment(victim_env_id(args))
        .framework("torch")
        .debugging(seed=args.seed) #sets random seed
        .callbacks(VictimMetricsCallback)
        .training(
            gamma=0.99, #considers future rewards almost as important as short-term ones
            lr=args.lr, #learning rate step size
            train_batch_size=args.train_batch_size,
            model={
                #model params
                "use_attention": False,
                "fcnet_hiddens": args.hidden_sizes,
                "fcnet_activation": "relu",
            },
        )
    )
    return configure_rollout_workers(config, args)

#helper function to return result
def mean_episode_return(result):
    return result.get("env_runners", {}).get(
        "episode_return_mean",
        result.get("episode_reward_mean"),
    )


def result_value(result, key):
    env_runners = result.get("env_runners", {})
    custom = result.get("custom_metrics", {})
    env_custom = env_runners.get("custom_metrics", {})

    candidates = [
        (custom, key),
        (custom, f"{key}_mean"),
        (env_custom, key),
        (env_custom, f"{key}_mean"),
    ]
    for source, name in candidates:
        if name in source:
            return source[name]
    return None


def evaluate(algo, args):
    if args.eval_episodes <= 0:
        return

    env = make_eval_env(args)
    episode_returns = []
    failures = 0

    try:
        for episode_idx in range(args.eval_episodes):
            obs, _ = env.reset(seed=args.eval_seed + episode_idx)
            done = False
            episode_return = 0.0
            final_info = {}

            while not done:
                action = algo.compute_single_action(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += reward
                final_info = info

            episode_returns.append(episode_return)
            if final_info.get("victim_failed"):
                failures += 1
    finally:
        env.close()

    mean_return = sum(episode_returns) / len(episode_returns)
    failure_rate = failures / len(episode_returns)
    print("final victim evaluation:")
    print(f"  episodes={len(episode_returns)}")
    print(f"  mean_episode_return={mean_return:.2f}")
    print(f"  failure_rate={failure_rate:.3f}")


def load_checkpoint(checkpoint):
    return Algorithm.from_checkpoint(os.path.abspath(checkpoint))


#defines CLI arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--env", choices=["cartpole", "stateless"], default="cartpole")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--num-envs-per-env-runner", type=int, default=1)
    parser.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=parse_hidden_sizes("256,256,128"),
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out-dir", default="checkpoints/victim")
    return parser.parse_args()


def main():
    args = parse_args() #reads CLI arguments
    register_env() #gives RLlib the environment
    num_gpus = 1 if torch.cuda.is_available() else 0
    print(cuda_status())
    print(
        "victim config: "
        f"env={args.env}  "
        f"iters={args.iters}  "
        f"train_batch_size={args.train_batch_size}  "
        f"num_env_runners={args.num_env_runners}  "
        f"num_envs_per_env_runner={args.num_envs_per_env_runner}  "
        f"hidden_sizes={args.hidden_sizes}  "
        f"eval_only={args.eval_only}"
    )
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus) #starts Ray to run RLlib
    algo = None
    try:
        if args.eval_only:
            if not args.checkpoint:
                raise ValueError("--eval-only requires --checkpoint")
            print(
                "loading victim checkpoint for deterministic evaluation: "
                f"{args.checkpoint}"
            )
            print(
                "make sure --env matches the environment used to train the checkpoint"
            )
            algo = load_checkpoint(args.checkpoint)
            try:
                evaluate(algo, args)
            except RuntimeError as exc:
                raise RuntimeError(
                    "evaluation failed; if this is a tensor shape error, "
                    "--env probably does not match the checkpoint's training env"
                ) from exc
            return

        algo = build_config(args).build() #build the PPO algorithm, assign it to algo
        #loops over iterations. ONLY RUNS ON 1 SEED
        for i in range(args.iters):
            start_time = time.perf_counter()
            result = algo.train()
            iter_seconds = time.perf_counter() - start_time
            reward = mean_episode_return(result)
            print(
                f"victim iter {i + 1}: "
                f"mean_episode_return={reward}  "
                f"failure_rate={result_value(result, 'victim_failure_rate')}  "
                f"iter_seconds={iter_seconds:.2f}"
            )

        os.makedirs(args.out_dir, exist_ok=True) #creates folder to save the trained victimPPO policy
        checkpoint = algo.save(args.out_dir) # saves the checkpoint policy
        print(f"victim checkpoint: {checkpoint_path(checkpoint)}") #prints the path we pass into the adversarial transformer training script
        evaluate(algo, args)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
