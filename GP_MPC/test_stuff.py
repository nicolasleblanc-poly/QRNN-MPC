import panda_gym
import gymnasium as gym

env_name = 'PandaReach-v3'
env = gym.make(env_name, render_mode='human')
print("env ", env, "\n")

