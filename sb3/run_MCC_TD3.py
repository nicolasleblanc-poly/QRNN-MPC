import gymnasium as gym
# from gymnasium import ObservationWrapper
# import panda_gym
import numpy as np
# import matplotlib.pyplot as plt
from sb3_contrib import TQC
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
# from funcs import run
from funcs_June27 import run

env_seeds = [0, 8, 15]

steps_per_episode = 1000
max_episodes = 600
prob = "MountainCarContinuous"
# # A2C
# run(env_seeds, prob, "A2C", steps_per_episode, max_episodes)
# # PPO
# run(env_seeds, prob, "PPO", steps_per_episode, max_episodes)
# # DDPG
# run(env_seeds, prob, "DDPG", steps_per_episode, max_episodes)
# # SAC
# run(env_seeds, prob, "SAC", steps_per_episode, max_episodes)
# TD3
run(env_seeds, prob, "TD3", steps_per_episode, max_episodes)
# # TQC
# run(env_seeds, prob, "TQC", steps_per_episode, max_episodes)


