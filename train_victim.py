"""Train and save a frozen PPO victim for adversarial CartPole."""

import argparse
import os

import ray
import torch
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig

from envs.victim_history import make_stacked_stateless_cartpole


#Assign name and register the environment
VICTIM_ENV_ID = "victim_stateless_cartpole"
def register_env():
    tune.register_env(VICTIM_ENV_ID, make_stacked_stateless_cartpole)


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


def build_config(args):
    num_gpus = 1 if torch.cuda.is_available() else 0
    return (
        PPOConfig() #starts a PPO object
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(num_gpus=num_gpus)
        .environment(VICTIM_ENV_ID)
        .framework("torch")
        .debugging(seed=args.seed) #sets random seed
        .env_runners(num_env_runners=args.num_env_runners) #sets number of parallel environment runners
        .callbacks(VictimMetricsCallback)
        .training(
            gamma=0.99, #considers future rewards almost as important as short-term ones
            lr=args.lr, #learning rate step size
            train_batch_size=args.train_batch_size,
            model={
                #model params
                "use_attention": False,
                "fcnet_hiddens": [64, 64, 64], 
                "fcnet_activation": "relu",
            },
        )
    )

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

#defines CLI arguments
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
    args = parse_args() #reads CLI arguments
    register_env() #gives RLlib the environment
    ray.init(ignore_reinit_error=True, num_gpus=1 if torch.cuda.is_available() else 0) #starts Ray to run RLlib
    algo = None
    try:
        algo = build_config(args).build() #build the PPO algorithm, assign it to algo
        #loops over iterations. ONLY RUNS ON 1 SEED
        for i in range(args.iters):
            result = algo.train()
            reward = mean_episode_return(result)
            print(
                f"victim iter {i + 1}: "
                f"mean_episode_return={reward}  "
                f"failure_rate={result_value(result, 'victim_failure_rate')}"
            )

        os.makedirs(args.out_dir, exist_ok=True) #creates folder to save the trained victimPPO policy
        checkpoint = algo.save(args.out_dir) # saves the checkpoint policy
        print(f"victim checkpoint: {checkpoint_path(checkpoint)}") #prints the path we pass into the adversarial transformer training script
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
