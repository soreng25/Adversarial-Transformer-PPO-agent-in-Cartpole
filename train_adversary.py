"""Train a Transformer PPO adversary against a frozen PPO victim."""

import argparse
import math
import os
import time

import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig

from envs.adversarial_cartpole import AdversarialCartPoleEnv

try:
    import wandb
except ImportError:
    wandb = None


ADVERSARY_ENV_ID = "adversarial_cartpole"
HISTORY_LEN = 50  #how many history steps of memory the transformer keeps

# We want to collect metrics over each episode during the adversary's training
#RLlib owns the loop of start episode, step, end, update weights, repeat. Callbacks are designed so when episode starts, RLlib wil run this code which we used to collect metrics
# RLlib expects callbacks to be managed by a class that inherits DefaultCallBacks
class AdversaryMetricsCallback(DefaultCallbacks):


    #We care about: EPISODE START, EPISODE STEP, EPISODE END. RLlib automatically calls these events

    #Create empty lists for the metrics we want to collect
    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["abs_winds"] = []
        episode.user_data["wind_penalties"] = []
        episode.user_data["victim_failed"] = 0.0
        episode.user_data["failure_timestep"] = math.inf

    # During each envstep, environment returns an INFO dictionary that includes our metrics
    def on_episode_step(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if not info:
            return

        #Append each metric we want to our lists
        episode.user_data["abs_winds"].append(info["abs_wind"])
        episode.user_data["wind_penalties"].append(
            info["wind_log_likelihood_penalty"]
        )
        if info["victim_failed"]:
            episode.user_data["victim_failed"] = 1.0
            episode.user_data["failure_timestep"] = info["episode_len"]

    # On episode end, summarize each metric across the episode and give back to RLlib
    def on_episode_end(self, *, episode, **kwargs):
        #save them
        abs_winds = episode.user_data["abs_winds"]
        wind_penalties = episode.user_data["wind_penalties"]
        victim_failed = episode.user_data["victim_failed"]
        failure_timestep = episode.user_data["failure_timestep"]
        
        #RBlib collects the metrics again
        episode.custom_metrics["victim_failure_rate"] = victim_failed
        episode.custom_metrics["mean_abs_wind"] = (
            float(np.mean(abs_winds)) if abs_winds else 0.0
        )
        episode.custom_metrics["mean_wind_log_likelihood_penalty"] = (
            float(np.mean(wind_penalties)) if wind_penalties else 0.0
        )
        episode.custom_metrics["failure_timestep"] = (
            failure_timestep if victim_failed else 10**9
        )

# Register env and handle checkpoint paths
def register_env():
    tune.register_env(
        ADVERSARY_ENV_ID,
        lambda env_config: AdversarialCartPoleEnv(env_config),
    )
def checkpoint_path(checkpoint):
    if hasattr(checkpoint, "checkpoint") and hasattr(checkpoint.checkpoint, "path"):
        return checkpoint.checkpoint.path
    return getattr(checkpoint, "path", str(checkpoint))

# Helper to establish parameters for how the adversary interacts with the environment
# E.g., needs to know the victim checkpoint, strongest wind it can use, failure bonus, etc
def env_config(args):
    return {
        "victim_checkpoint": args.victim_checkpoint,
        "victim_env": args.victim_env,
        "max_wind": args.max_wind,
        "wind_sigma": args.wind_sigma,
        "horizon": args.horizon,
        "failure_bonus": args.failure_bonus,
    }

# Transformer architecture settings
def build_config(args):
    num_gpus = 1 if torch.cuda.is_available() else 0
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(num_gpus=num_gpus)
        .environment(ADVERSARY_ENV_ID, env_config=env_config(args))
        .framework("torch")
        .debugging(seed=args.seed)
        .env_runners(num_env_runners=args.num_env_runners)
        .callbacks(AdversaryMetricsCallback)
        .training(
            gamma=0.99,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
            model={
                "use_attention": True,
                "attention_num_transformer_units": 1, #1 transformer block
                "attention_dim": 64, #size of attention representation
                "attention_num_heads": 1, #number of attention heads
                "attention_head_dim": 32, #size of each attention head
                "attention_memory_inference": HISTORY_LEN, #training memory
                "attention_memory_training": HISTORY_LEN, #inference memory
                "fcnet_hiddens": [64], #1 hidden layer, just for initial testing
                "fcnet_activation": "relu",
            },
        )
    )

#helper function to get final results after each iteration
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

# helper function to get averaged adversary reward after each iteration
def result_reward(result):
    return result.get("env_runners", {}).get(
        "episode_return_mean",
        result.get("episode_reward_mean"),
    )

# Given the current observation, computes what action should be taken next
#During training, RLlib automatically does this. But after we train it when we want to evaluate it, 
#we have to manually compute the action ourselves. 
def compute_action(algo, obs, state, explore):
    #State is transformer memory
    state = [
        np.expand_dims(s, axis=0) if np.asarray(s).ndim == 1 else s
        for s in state
    ]
    out = algo.compute_single_action(obs, state=state, explore=explore) #pick an action
    action, state_out, _ = out #returns the action chosen and the updated state
    return action, state_out

# tests adversary after training
def evaluate(algo, args):
    env = AdversarialCartPoleEnv(env_config(args)) #creates new environment for evaluation
    policy = algo.get_policy() #get the trained adversarial policy

    failures = 0
    failure_steps = []
    abs_wind_sum = 0.0
    penalty_sum = 0.0
    total_steps = 0

    for episode_idx in range(args.eval_episodes):
        #seed changes every episode so we don't evaluate adversary on the same episode every time
        obs, _ = env.reset(seed=args.seed + 1000 + episode_idx) # starts new episode
        state = policy.get_initial_state() #gets blank memory for start of episode
        done = False
        final_info = None

        while not done: #keep running until the episode is done
            #explore=false means don't add randomness, use training policy cleanly
            action, state = compute_action(algo, obs, state, explore=False) #compute next action and state. 
            #apply action to environment, and return observation, if victim failed, if horizoin was reached
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated #done if victim failed or horizon reached
            final_info = info #save info

            #update metrics
            total_steps += 1
            abs_wind_sum += info["abs_wind"]
            penalty_sum += info["wind_log_likelihood_penalty"]
            
        #increase failure count if victim failed
        if final_info and final_info["victim_failed"]:
            failures += 1
            failure_steps.append(final_info["episode_len"])
    env.close()

    #compute average failure timestep, absolute wind, and wind penalty
    avg_failure_timestep = (
        sum(failure_steps) / len(failure_steps) if failure_steps else None
    )
    avg_abs_wind = abs_wind_sum / total_steps if total_steps else 0.0
    avg_wind_penalty = penalty_sum / total_steps if total_steps else 0.0

    #print eval metrics
    print("final evaluation:")
    print(f"  episodes={args.eval_episodes}")
    print(f"  failure_count={failures}")
    print(f"  failure_rate={failures / args.eval_episodes:.3f}")
    print(f"  average_failure_timestep={avg_failure_timestep}")
    print(f"  average_abs_wind={avg_abs_wind:.4f}")
    print(f"  average_wind_log_likelihood_penalty={avg_wind_penalty:.4f}")
    return {
        "episodes": args.eval_episodes,
        "failure_count": failures,
        "failure_rate": failures / args.eval_episodes,
        "average_failure_timestep": avg_failure_timestep,
        "average_abs_wind": avg_abs_wind,
        "average_wind_log_likelihood_penalty": avg_wind_penalty,
    }


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` "
            "or remove the --wandb flag."
        )

    config = {
        "victim_checkpoint": args.victim_checkpoint,
        "victim_env": args.victim_env,
        "iters": args.iters,
        "seed": args.seed,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "num_env_runners": args.num_env_runners,
        "out_dir": args.out_dir,
        "eval_episodes": args.eval_episodes,
        "max_wind": args.max_wind,
        "wind_sigma": args.wind_sigma,
        "horizon": args.horizon,
        "failure_bonus": args.failure_bonus,
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    )

    #get CLI arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--victim-checkpoint", required=True)
    parser.add_argument(
        "--victim-env",
        choices=["cartpole", "stateless"],
        default="cartpole",
        help="Observation format used by the frozen victim checkpoint.",
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--out-dir", default="checkpoints/adversary")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-wind", type=float, default=4.0)
    parser.add_argument("--wind-sigma", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--failure-bonus", type=float, default=1000.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cartpole-adversary")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    return parser.parse_args()


# main loop for TRAINING
def main():
    args = parse_args()
    register_env()
    wandb_run = maybe_init_wandb(args)
    ray.init(ignore_reinit_error=True, num_gpus=1 if torch.cuda.is_available() else 0)
    algo = None
    best_failure_timestep = None
    try:
        algo = build_config(args).build()
        for i in range(args.iters):
            start_time = time.perf_counter()
            result = algo.train()
            iter_seconds = time.perf_counter() - start_time
            reward = result_reward(result)
            failure_rate = result_value(result, "victim_failure_rate")
            episode_len_mean = result.get("env_runners", {}).get(
                "episode_len_mean",
                result.get("episode_len_mean"),
            )
            mean_abs_wind = result_value(result, "mean_abs_wind")
            mean_wind_penalty = result_value(
                result,
                "mean_wind_log_likelihood_penalty",
            )
            failure_timestep = result_value(result, "failure_timestep_min")
            if failure_timestep is not None and failure_timestep < 10**9:
                best_failure_timestep = (
                    failure_timestep
                    if best_failure_timestep is None
                    else min(best_failure_timestep, failure_timestep)
                )

            print(
                f"adversary iter {i + 1}: "
                f"reward_mean={reward}  "
                f"failure_rate={failure_rate}  "
                f"episode_len_mean={episode_len_mean}  "
                f"mean_abs_wind={mean_abs_wind}  "
                f"mean_wind_log_likelihood_penalty={mean_wind_penalty}  "
                f"best_failure_timestep={best_failure_timestep}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/iteration": i + 1,
                        "train/reward_mean": reward,
                        "train/failure_rate": failure_rate,
                        "train/episode_len_mean": episode_len_mean,
                        "train/mean_abs_wind": mean_abs_wind,
                        "train/mean_wind_log_likelihood_penalty": mean_wind_penalty,
                        "train/failure_timestep_min": failure_timestep,
                        "train/best_failure_timestep": best_failure_timestep,
                        "train/iter_seconds": iter_seconds,
                    },
                    step=i + 1,
                )

        os.makedirs(args.out_dir, exist_ok=True)
        checkpoint = algo.save(args.out_dir)
        print(f"adversary checkpoint: {checkpoint_path(checkpoint)}")
        if wandb_run is not None:
            wandb_run.summary["checkpoint_path"] = checkpoint_path(checkpoint)
        eval_metrics = evaluate(algo, args)
        if wandb_run is not None:
            wandb_run.log(
                {f"eval/{key}": value for key, value in eval_metrics.items()}
            )
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
