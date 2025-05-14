import gymnasium as gym
# from gymnasium import ObservationWrapper
# import panda_gym
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import TQC
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback


class EpisodicReturnLogger(BaseCallback):
    def __init__(self, max_episodes):
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episodic_returns = []  # Stores episodic returns
        self.timesteps = []         # Stores corresponding timesteps

    def _on_step(self) -> bool:
        # Log at the end of each episode (SB3 updates `ep_info_buffer` after an episode ends)
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                episodic_return = info['episode']['r']  # Cumulative reward of the episode
                self.episodic_returns.append(episodic_return)
                self.timesteps.append(self.num_timesteps)  # Current timestep
                
                if self.episode_count >= self.max_episodes:
                    return False
                
        return True
    