"""Train and save a frozen PPO victim for adversarial CartPole."""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from envs.victim_history import make_stacked_stateless_cartpole

try:
    import wandb
except ImportError:
    wandb = None


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


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def make_render_env(args):
    if args.env != "cartpole":
        return None
    return gym.make("CartPole-v1", render_mode="rgb_array")


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
        return None

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
    return {
        "episodes": len(episode_returns),
        "mean_episode_return": mean_return,
        "failure_rate": failure_rate,
    }


def render_policy_episode(algo, args, seed):
    env = make_render_env(args)
    if env is None:
        print("wandb video logging is only available for --env cartpole")
        return None

    frames = []
    episode_return = 0.0
    final_info = {}

    try:
        obs, _ = env.reset(seed=seed)
        try:
            frame = env.render()
        except gym.error.DependencyNotInstalled:
            print(
                "CartPole video rendering needs pygame; install it with "
                "`pip install \"gymnasium[classic-control]\"`"
            )
            return None
        if frame is not None:
            frames.append(frame)

        for step in range(args.wandb_video_max_steps):
            action = algo.compute_single_action(
                obs,
                explore=args.wandb_video_explore,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            final_info = info

            try:
                frame = env.render()
            except gym.error.DependencyNotInstalled:
                print(
                    "CartPole video rendering needs pygame; install it with "
                    "`pip install \"gymnasium[classic-control]\"`"
                )
                return None
            if frame is not None:
                frames.append(frame)

            if terminated or truncated:
                break
    finally:
        env.close()

    if not frames:
        return None

    video = np.stack(frames).transpose(0, 3, 1, 2)
    return {
        "video": video,
        "episode_return": episode_return,
        "episode_len": len(frames) - 1,
        "victim_failed": float(final_info.get("victim_failed", False)),
    }


def maybe_log_wandb_video(wandb_run, algo, args, iteration):
    if wandb_run is None or not args.wandb_video:
        return
    if iteration not in args.wandb_video_iters:
        return

    rendered = render_policy_episode(
        algo,
        args,
        seed=args.eval_seed + iteration,
    )
    if rendered is None:
        return

    wandb_run.log(
        {
            "video/policy_rollout": wandb.Video(
                rendered["video"],
                fps=args.wandb_video_fps,
                format="mp4",
            ),
            "video/iteration": iteration,
            "video/episode_return": rendered["episode_return"],
            "video/episode_len": rendered["episode_len"],
            "video/victim_failed": rendered["victim_failed"],
        },
        step=iteration,
    )


def load_checkpoint(checkpoint):
    return Algorithm.from_checkpoint(os.path.abspath(checkpoint))


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` "
            "or remove the --wandb flag."
        )

    config = {
        "env": args.env,
        "iters": args.iters,
        "seed": args.seed,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "num_env_runners": args.num_env_runners,
        "num_envs_per_env_runner": args.num_envs_per_env_runner,
        "hidden_sizes": args.hidden_sizes,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "eval_only": args.eval_only,
        "checkpoint": args.checkpoint,
        "out_dir": args.out_dir,
        "wandb_video": args.wandb_video,
        "wandb_video_iters": args.wandb_video_iters,
        "wandb_video_explore": args.wandb_video_explore,
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    )


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
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cartpole-victim")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-video", action="store_true")
    parser.add_argument(
        "--wandb-video-iters",
        type=parse_int_list,
        default=parse_int_list("1,5,10,25,50,100"),
    )
    parser.add_argument("--wandb-video-fps", type=int, default=30)
    parser.add_argument("--wandb-video-max-steps", type=int, default=500)
    parser.add_argument("--wandb-video-explore", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args() #reads CLI arguments
    register_env() #gives RLlib the environment
    num_gpus = 1 if torch.cuda.is_available() else 0
    wandb_run = maybe_init_wandb(args)
    print(cuda_status())
    print(
        "victim config: "
        f"env={args.env}  "
        f"iters={args.iters}  "
        f"train_batch_size={args.train_batch_size}  "
        f"num_env_runners={args.num_env_runners}  "
        f"num_envs_per_env_runner={args.num_envs_per_env_runner}  "
        f"hidden_sizes={args.hidden_sizes}  "
        f"eval_only={args.eval_only}  "
        f"wandb={args.wandb}"
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
                eval_metrics = evaluate(algo, args)
                if wandb_run is not None and eval_metrics is not None:
                    wandb_run.log(
                        {f"eval/{key}": value for key, value in eval_metrics.items()}
                    )
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
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/iteration": i + 1,
                        "train/mean_episode_return": reward,
                        "train/failure_rate": result_value(
                            result, "victim_failure_rate"
                        ),
                        "train/iter_seconds": iter_seconds,
                    },
                    step=i + 1,
                )
                maybe_log_wandb_video(wandb_run, algo, args, i + 1)

        os.makedirs(args.out_dir, exist_ok=True) #creates folder to save the trained victimPPO policy
        checkpoint = algo.save(args.out_dir) # saves the checkpoint policy
        print(f"victim checkpoint: {checkpoint_path(checkpoint)}") #prints the path we pass into the adversarial transformer training script
        if wandb_run is not None:
            wandb_run.summary["checkpoint_path"] = checkpoint_path(checkpoint)
        eval_metrics = evaluate(algo, args)
        if wandb_run is not None and eval_metrics is not None:
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
