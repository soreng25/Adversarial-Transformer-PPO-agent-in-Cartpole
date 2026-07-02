"""Minimal Transformer (attention) PPO example using Ray RLlib.

Trains PPO on CartPole-v1 with RLlib's GTrXL attention network enabled,
so the policy uses a Transformer-based encoder over a window of recent
observations instead of a plain feedforward MLP.

Run:
    pip install -r requirements.txt
    python main.py
"""

import sys
import ray
import time
import statistics
import numpy as np
import gymnasium as gym
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.envs.classes.stateless_cartpole import StatelessCartPole

# How many past timesteps each model can use.
# Transformer: attention_memory_inference / attention_memory_training
# Plain MLP: number of [x, theta] pairs concatenated into one input vector
HISTORY_LEN = 50


def stacked_observation(history, num_frames, obs_dim):
    # history is list of observations so far: [ [a,b], [c, d] ]
    # num_frames is how many observations we want total (50)
    # obs_dim is dimension of a single observation (2: x-pos, theta)

    # creates a fake observation of zeros. zero=[0, 0]. We need to pad with zeros because the MLP always needs
    # the same input size, so pad old missing observations with zeros.
    zero = np.zeros(obs_dim, dtype=np.float32)

    # missing represents how many fake zero observations we need. if theres only 1 obs in history, then 50-1=49 fake observations
    missing = num_frames - len(history)

    # frames=[ [0,0],[0,0], ... [a, b],[c,d] ]
    frames = [zero] * missing + history

    # concatenates into one long vector of scalars, dimension 100x1
    return np.concatenate(frames)


def make_env(env_config):
    # use_transformer is some boolean from the dictionary if the user selected transformer or MLP
    use_transformer = env_config["use_transformer"]

    # returns current observation [x, theta] from the Stateless CartPole environment
    base = StatelessCartPole()

    # Transformer: raw 2-dim obs since the transformer feeds itself memory on its own
    if use_transformer:
        return base

    # Plain MLP: remember last HISTORY_LEN frames in a local list.
    history = []  # stores recent observations
    obs_dim = base.observation_space.shape[0]  # obs_dim=2

    # stacked space produces ranges for each x value and each theta value
    # The raw stateless environment looks like: shape=(2,), low=[x_min, theta_min], high=[x_max, theta_max]
    # the stacked space just repeats it HISTORY_LEN times, so 50 times to create two vectors of length 100 that describe the ranges for the x-pos, theta values
    stacked_space = gym.spaces.Box(
        low=np.tile(base.observation_space.low, HISTORY_LEN),  # low=[xmin, thetamin] * 50
        high=np.tile(base.observation_space.high, HISTORY_LEN),  # high=[xmax, thetamax] * 50
        dtype=np.float32,
    )

    # called at start of new episode to clear old frame history and get first raw obs from environment
    def reset(seed=None, options=None):
        nonlocal history
        # base.reset(seed, options) gets a new CartPole episode and returns first observation. obs=[x0, theta0]
        obs, info = base.reset(seed=seed, options=options)
        history = [obs]  # resets history to just the current new observation
        return stacked_observation(history, HISTORY_LEN, obs_dim), info  # repads with 0s and returns

    def step(action):
        obs, reward, terminated, truncated, info = base.step(action)
        history.append(obs)  # adds the observation to the history vector
        if len(history) > HISTORY_LEN:  # if theres more than 50 obs,
            history[:] = history[-HISTORY_LEN:]  # gets last 50 elements of array since we append to the end
        return (
            stacked_observation(history, HISTORY_LEN, obs_dim),
            reward,
            terminated,
            truncated,
            info,
        )

    # Bare gym.Env with our functions attached — no custom class.
    # history lives inside make_env; each RLlib worker calls make_env again and gets its own list.
    env = gym.Env()
    env.observation_space = stacked_space
    env.action_space = base.action_space
    env.reset = reset
    env.step = step
    return env


# RLlib needs a registered env id (not a bare function) for custom env creators.
STATELESS_ENV_ID = "stateless_cartpole_with_history"
tune.register_env(STATELESS_ENV_ID, make_env)


# configue PPO to set parameters for PPO algorithm.
# use_transformer=True  -> PPO + Transformer (GTrXL attention)
# use_transformer=False -> plain PPO (regular MLP, no attention)
# env_name="cartpole"   -> standard CartPole-v1 (fully observable, memory useless)
# env_name="stateless"  -> StatelessCartPole (velocities hidden, REQUIRES memory)
def build_config(use_transformer, env_name, seed):
    # both versions share the same model dict except for the attention part
    if use_transformer:
        model = {
            "use_attention": True, #this uses the transformer GTrXL model. it looks at short history of observations
            "attention_num_transformer_units": 1,
            "attention_dim": 64,
            "attention_num_heads": 1,
            "attention_head_dim": 32,
            "attention_memory_inference": HISTORY_LEN,
            "attention_memory_training": HISTORY_LEN,
            # Small feedforward stack feeding the attention layers.
            "fcnet_hiddens": [64],
            "fcnet_activation": "relu",
        }
    else:
        # plain PPO: just a regular feedforward network, no attention
        model = {
            "use_attention": False,
            "fcnet_hiddens": [141, 141],
            "fcnet_activation": "relu",
        }

    # pick the environment:
    # - "stateless" hides the velocities, so the agent must remember past
    #   observations to infer them -> a memory task where attention helps.
    # - otherwise use the standard fully-observable CartPole-v1.
    if env_name == "stateless":
        env = STATELESS_ENV_ID
        env_config = {"use_transformer": use_transformer}
    else:
        env = "CartPole-v1"
        env_config = {}

    # configuration for PPO algorithm
    config = (
        PPOConfig() 
        .api_stack( 
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )

        #picks the RL environment we want to use (chosen above). 
        #cartpolev1 is standard gymnasium environment where goal of agent is to prevent the pole from falling over by moving the cart left/right to balance the pole puright
        .environment(env, env_config=env_config)

        # specifies Pytorch to train neural networks for actor/critic in PPO 
        .framework("torch")

        # one master seed makes the whole pipeline reproducible:
        # weight initialization, per-worker env resets, and action sampling.
        # RLlib derives a different seed per env runner from this master seed,
        # so the 4 runners still explore differently but the run as a whole repeats.
        .debugging(seed=seed)

        # sets number of environments to run in parallel
        .env_runners(num_env_runners=4)

        # training parameters
        .training(
            gamma=0.99, #discount factor for future rewards
            #determines how much agent values immediate rewrad vs long term reward
            # higher value means agent cares more about long term reward
            lr=3e-4, #learning rate. this is the step size taken during gradient descent
            #when updating the neraul network weights
            train_batch_size=4000, #number of environment steps taken before NN is updated
            model=model, # the neural network chosen above (transformer or plain)
        )
    )
    return config


def main(use_transformer, env_name):
    if use_transformer:
        mode = "PPO + Transformer"
    else:
        mode = "plain PPO"
    print(f"=== Running {mode} on {env_name} ===")

    seeds = [42, 43, 44, 45, 46]   # always run multiple seeds; edit this list to add/remove
    num_iters = 40

    ray.init() #intialize ray. in this case im not doing parallel computing but thats what its used for
    try: #try to train the model
        start_time = time.perf_counter()

        #train one full run per seed and remember every iteration's reward
        per_seed_rewards = []  # one list of per-iteration rewards per seed
        for seed in seeds:
            print(f"--- seed {seed} ---")
            algo = build_config(use_transformer, env_name, seed).build()

            #train model for num_iters iterations, each iteration is full training step
            #so for each iteration, agent interacts with environment 4000 times, then updates policy
            rewards = []
            for i in range(num_iters):
                #internally, current policy chooses action, gets reward, computes advantage, and updates
                #actor and critic weights, and returns updated policy
                result = algo.train()

                #mean reward over all the episodes in the given iteration
                reward = result.get("env_runners", {}).get("episode_return_mean", result.get("episode_reward_mean"))
                rewards.append(reward)
            algo.stop()
            per_seed_rewards.append(rewards)

        # average each iteration across seeds (expanded for readability)
        print("=== averaged across seeds ===")
        num_seeds = len(seeds)
        for i in range(num_iters):
            # gather this iteration's reward from every seed's run
            iteration_rewards = []
            for s in range(num_seeds):
                # gets 1st iteration of 1st seed, 1st iteration from 2nd seed, 1st iteration from 3rd seed. Then that for loop ends and moves onto the 2nd iteration.
                iteration_rewards.append(per_seed_rewards[s][i])

            # mean = add up the seeds' rewards, then divide by how many seeds
            mean_reward = sum(iteration_rewards) / num_seeds

            # std = how spread out the seeds' rewards are around that mean
            #       (population standard deviation across the seeds)
            std_reward = statistics.pstdev(iteration_rewards)

            print(f"iter {i + 1}: "
                  f"mean={mean_reward:.2f}  "
                  f"std={std_reward:.2f}  "
                  f"(rewards from each seed: {iteration_rewards})")

        end_time = time.perf_counter()
        print(f"Training time: {end_time - start_time} seconds")

    finally: #even if the try block hits an error, this block will always run to shutdown ray
        ray.shutdown()


'''in the CartPole environment, agent gets +1 reward for every timestep that the pole stays balanced
Episode ends when pole tips too far, cart runs off edge, or it hits max of 500 steps. 
Reward = number of steps the agent survived. worst case is 8-10, best case is 500 if it balanced the whole time

In each iteration, the environment runs 4000 timesteps. This is made up of multiple episodes where the agent
attempts to balance the pole. When it fails (or maxes out at 500 steps) then a new episode starts. The reward
value that is being printed is the average reward among those episodes during a given iteration of 4000 steps before
the neural network has been updated. 
'''

if __name__ == "__main__":
    # choose mode and environment from the command line:
    #   arg 1 = model:  transformer (default) | plain
    #   arg 2 = env:    cartpole (default)     | stateless
    #
    # examples:
    #   python main.py transformer stateless   -> Transformer with 50-step internal memory
    #   python main.py plain stateless         -> plain PPO with 50-frame stack (fair baseline)
    #   python main.py transformer cartpole    -> Transformer on standard CartPole
    #   python main.py plain                   -> plain PPO on standard CartPole
    #
    # the interesting comparison is "transformer stateless" vs "plain stateless":
    # plain PPO stacks the last HISTORY_LEN raw observations (100-dim input);
    # the Transformer keeps 2-dim obs and uses attention_memory_* for internal memory.
    # arg 1 = model choice (default: transformer)
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
    else:
        model_arg = "transformer"

    # turn the model argument into a True/False flag
    if model_arg == "transformer":
        use_transformer = True
    else:
        use_transformer = False

    # arg 2 = environment choice (default: cartpole)
    if len(sys.argv) > 2:
        env_arg = sys.argv[2]
    else:
        env_arg = "cartpole"

    main(use_transformer=use_transformer, env_name=env_arg)
