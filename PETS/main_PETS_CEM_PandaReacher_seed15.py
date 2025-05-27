# from IPython import display
# %matplotlib inline
# import matplotlib as mpl
# import matplotlib.pyplot as plt

import sys
print("Python Version:", sys.version)

import numpy as np
import torch
import omegaconf
# import gym
import gymnasium as gym

# import mbrl.env.cartpole_continuous as cartpole_env

# import mbrl.env.reward_fns as reward_fns
# import mbrl.env.termination_fns as termination_fns

import reward_fns
import termination_fns

import mbrl.models as models
# from model_env import ModelEnv

import mbrl.planning as planning
import mbrl.util.common as common_util
# import common as common_util
import mbrl.util as util

import panda_gym

import os
def save_data(seed, prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_{seed}.npz")

    np.savez(
    save_path,  # Save to the same folder as this script
    # f"{prob}_{method_name}_results_{seed}.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )

# %load_ext autoreload
# %autoreload 2

# mpl.rcParams.update({"font.size": 16})

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from gym import spaces
# import gym
import numpy as np

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

# class DictToBoxWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         # Get ordered keys to flatten
#         self.obs_keys = list(env.observation_space.spaces.keys())

#         # Compute new low/high by concatenating the original spaces
#         lows = []
#         highs = []
#         for key in self.obs_keys:
#             space = env.observation_space.spaces[key]
#             assert isinstance(space, Box), f"Expected Box for key '{key}', got {type(space)}"
#             lows.append(space.low.flatten())
#             highs.append(space.high.flatten())
        
#         low = np.concatenate(lows)
#         high = np.concatenate(highs)
#         self.observation_space = Box(low=low, high=high, dtype=np.float32)

#     def observation(self, obs):
#         # Flatten and connate([obs[key].flatten() for key in self.obs_keys]).astype(np.float32)

class DictToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # print("observation_space ", env.observation_space, "\n")
        # print("spaces.Dict ", spaces.Dict, "\n")
        
        # # Original space must be a Dict
        # assert isinstance(env.observation_space, spaces.Dict), "Expected Dict observation space!"
        
        # Extract the "observation" subspace
        self.observation_space = env.observation_space["observation"]
        
        # print("observation_space ", env.observation_space, "\n")
    
    def observation(self, obs):
        # Return only the "observation" key
        return obs["observation"]

class ResetStepWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        # print("self.env.reset(**kwargs) ", self.env.reset(**kwargs), "\n")
        # obs, info = self.env.reset(**kwargs) # Discard info
        
        # result = self.env.reset(**kwargs) # Discard info
        # obs, info = self.env.reset(**kwargs) # Discard info
        
        # if isinstance(result, tuple) and len(result) == 2:
        #     obs = result[0]  # drop the infos
        # else:
        #     obs = result  # keep as is
        # print("result ", result, "\n")
        
        # print("self.env.reset(**kwargs)[0] ", self.env.reset(**kwargs)[0], "\n")
        return self.env.reset(**kwargs)[0] # obs
    
    def step(self, action):
        result = self.env.step(action)
        
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        # obs, reward, terminated, truncated, info = self.env.step(action)
        
        return obs, reward, done, info


# prob = "MountainCarContinuous"
# prob = "LunarLanderContinuous"
# prob = "Pendulum"
# prob = "InvertedPendulum"
# prob = "Reacher"
prob = "PandaReacher"
# prob = "PandaReacherDense"
# prob = "CartPoleContinuous" # Not used for my tests but implemented by the authors

# seeds =  [0, 8 ,15]
seed = 15
seeds = [seed]

# if prob == "CartPoleContinuous":
#     import mbrl.env.cartpole_continuous as cartpole_env
#     env = cartpole_env.CartPoleEnv()
#     reward_fn = reward_fns.cartpole
#     term_fn = termination_fns.cartpole
#     trial_length = 200
#     num_trials = 100 # 10

if prob == "MountainCarContinuous":
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
    reward_fn = reward_fns.mountaincar_continuous
    term_fn = termination_fns.mountaincar_continuous
    trial_length = 1000
    num_trials = 300 # 10

if prob == "LunarLanderContinuous":
    env = gym.make('LunarLanderContinuous-v3', render_mode='rgb_array')
    reward_fn = reward_fns.lunarlander_continuous
    term_fn = termination_fns.lunarlander_continuous
    trial_length = 1000
    num_trials = 300 # 10
        
if prob == "Pendulum":
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    reward_fn = reward_fns.pendulum
    term_fn = termination_fns.pendulum
    trial_length = 200
    num_trials = 300 # 10
    
if prob == "InvertedPendulum":
    env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    reward_fn = reward_fns.inverted_pendulum
    term_fn = termination_fns.inverted_pendulum
    trial_length = 1000
    num_trials = 300 # 10
    
if prob == "Reacher":
    env = gym.make('Reacher-v5', render_mode='rgb_array')
    reward_fn = reward_fns.reacher
    term_fn = termination_fns.reacher
    trial_length = 50
    num_trials = 300 # 10

if prob == "PandaReacher" or prob == "PandaReacherDense":
    
    if prob == "PandaReacher":
        env = gym.make('PandaReach-v3') # , render_mode='rgb_array'
    else:
        env = gym.make('PandaReachDense-v3') # , render_mode='rgb_array'
    
    reward_fn = reward_fns.panda_reach_seed0
    term_fn = termination_fns.panda_reach_seed0
    trial_length = 50
    num_trials = 300 # 10

# if prob == "PandaReacher":
#     import panda_gym
#     env = gym.make('PandaReach-v3', render_mode='rgb_array')
#     reward_fn = reward_fns.panda_reach
#     # term_fn = termination_fns.panda_reach
#     trial_length = 50
#     num_trials = 300 # 10
    
# if prob == "PandaReacherDense":
#     import panda_gym
#     env = gym.make('PandaReachDense-v3', render_mode='rgb_array')
#     reward_fn = reward_fns.panda_reach
#     term_fn = termination_fns.panda_reach
#     trial_length = 50
#     num_trials = 300 # 10

method_name = "PETS_CEM"

# env = gym.make('LunarLanderContinuous-v3', render_mode='rgb_array')

episodic_return_seeds = []
for seed in seeds:
    env = DictToBoxWrapper(env)  # Converts Dict → Box
    # env = DoneWrapper(env)
    env = ResetStepWrapper(env)
    
    print("env.reset(seed=seed) ", env.reset(seed=seed), "\n")
    print("env.step(env.action_space.sample()) ", env.step(env.action_space.sample()), "\n")
    
    # env.seed(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # if prob == "PandaReacher" or prob == "PandaReacherDense":
    #     obs_shape = env.observation_space['observation'].shape # len(.low) #env.observation_space.shape
    # else:
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    print("obs_shape", obs_shape, "\n")
    print("act_shape", act_shape, "\n")
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)

    ensemble_size = 5

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the 
    # environment information
    # cfg_dict = {
    #     # dynamics model configuration
    #     "dynamics_model": {
    #         "_target_": "mbrl.models.GaussianMLP",
    #         # "model": {
    #             # "_target_": "mbrl.models.GaussianMLP",
    #             "device": device,
    #             "num_layers": 3,
    #             "ensemble_size": ensemble_size,
    #             "hid_size": 200,
    #             "use_silu": True,
    #             "in_size": "???", # 9
    #             "out_size": "???", # 6
    #             "deterministic": False,
    #             "propagation_method": "fixed_model"
    #         # }
    #     },
    #     # options for training the dynamics model
    #     "algorithm": {
    #         "learned_rewards": False,
    #         "target_is_delta": True,
    #         "normalize": True,
    #     },
    #     # these are experiment specific options
    #     "overrides": {
    #         "trial_length": trial_length,
    #         "num_steps": num_trials * trial_length,
    #         "model_batch_size": 32,
    #         "validation_ratio": 0.05
    #     }
    # }
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "model": {
                "_target_": "mbrl.models.GaussianMLP",
                "device": device,
                "num_layers": 3,
                "ensemble_size": ensemble_size,
                "hid_size": 200,
                "use_silu": True,
                "in_size": "???",
                "out_size": "???",
                "deterministic": False,
                "propagation_method": "fixed_model"
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }
    
    cfg = omegaconf.OmegaConf.create(cfg_dict)

    print("obs_shape", obs_shape, "\n")
    print("act_shape", act_shape, "\n")
    print("cfg ", cfg, "\n")

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # # Create a gym-like environment to encapsulate the model
    # if prob == "PandaReacher" or prob == "PandaReacherDense":
    #     obs = env.reset(seed=seed) # Added a wrapper to discard info
    #     goal_state = obs#['desired_goal']
    #     model_env = ModelEnv(env, dynamics_model, reward_fn, generator=generator, prob=prob, goal_state=goal_state)
    # else:
    #     model_env = ModelEnv(env, dynamics_model, reward_fn, generator=generator) # term_fn
    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)

    replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)

    common_util.rollout_agent_trajectories(
        env,
        trial_length, # initial exploration steps
        planning.RandomAgent(env),
        {}, # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=trial_length
        )
    
    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 15,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.1,
            "population_size": 500,
            "alpha": 0.1,
            "device": device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True
                }
            })

    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
        )
    
    train_losses = []
    val_scores = []

    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(val_score.mean().item())   # this returns val score per ensemble model


    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

    # Create visualization objects
    # fig, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
    # ax_text = axs[0].text(300, 50, "")
        
    # import tqdm
    # from tqdm import trange
    
    # Main PETS loop
    # all_rewards = [0]
    # all_rewards = []
    episodic_return = []
    from tqdm import trange
    for trial in trange(num_trials):
    # for trial in range(num_trials):
        obs = env.reset(seed=seed) # Added a wrapper to discard info
        obs = obs#['observation']
        agent.reset()
        
        done = False
        total_reward = 0.0
        steps_trial = 0
        # # Render mode is define when creating the env instead when using gymnasium
        # # update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)
        
        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not done:
            # --------------- Model Training -----------------
            if steps_trial == 0:
                dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats
                
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )
                    
                model_trainer.train(
                    dataset_train, 
                    dataset_val=dataset_val, 
                    num_epochs=50, 
                    patience=50, 
                    callback=train_callback)

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer)
            
            # print("next_obs ", next_obs, "\n")    
            
            # update_axes(
            #     axs, env.render(), ax_text, trial, steps_trial, all_rewards)
                # axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)
            
            obs = next_obs
            total_reward += reward
            steps_trial += 1
            
            if steps_trial == trial_length:
                break
        
        episodic_return.append(total_reward)

    episodic_return_seeds.append(episodic_return)
    # update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards, force_update=True)
    # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards, force_update=True)

print("prob ", prob, "\n")
print("seed ", seed, "\n")

episodic_return_seeds = np.array(episodic_return_seeds)

mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
std_episodic_return = np.std(episodic_return_seeds, axis=0)

# print("max_episodes", max_episodes, "\n")
print("episodic_return_seeds.shape ", episodic_return_seeds.shape, "\n")
print("mean_episodic_return ", mean_episodic_return.shape, "\n")
print("std_episodic_return.shape ", std_episodic_return.shape, "\n")

save_data(seed, prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
print("Saved data \n")
env.close()

