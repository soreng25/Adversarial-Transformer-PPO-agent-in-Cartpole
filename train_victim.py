"""Train and save a frozen PPO victim for adversarial CartPole."""

import argparse
import os

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.envs.classes.stateless_cartpole import StatelessCartPole


VICTIM_ENV_ID = "victim_stateless_cartpole"


def register_env():
    tune.register_env(VICTIM_ENV_ID, lambda env_config: StatelessCartPole())


def checkpoint_path(checkpoint):
    if hasattr(checkpoint, "checkpoint") and hasattr(checkpoint.checkpoint, "path"):
        return checkpoint.checkpoint.path
    return getattr(checkpoint, "path", str(checkpoint))


def build_config(args):
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(VICTIM_ENV_ID)
        .framework("torch")
        .debugging(seed=args.seed)
        .env_runners(num_env_runners=args.num_env_runners)
        .training(
            gamma=0.99,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
            model={
                "use_attention": False,
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            },
        )
    )


def mean_episode_return(result):
    return result.get("env_runners", {}).get(
        "episode_return_mean",
        result.get("episode_reward_mean"),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--out-dir", default="checkpoints/victim")
    return parser.parse_args()


def main():
    args = parse_args()
    register_env()
    ray.init(ignore_reinit_error=True)
    algo = None
    try:
        algo = build_config(args).build()
        for i in range(args.iters):
            result = algo.train()
            reward = mean_episode_return(result)
            print(f"victim iter {i + 1}: mean_episode_return={reward}")

        os.makedirs(args.out_dir, exist_ok=True)
        checkpoint = algo.save(args.out_dir)
        print(f"victim checkpoint: {checkpoint_path(checkpoint)}")
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
