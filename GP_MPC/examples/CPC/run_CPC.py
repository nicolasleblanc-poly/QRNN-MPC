import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

import gymnasium as gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.run_env_function import run_env
from config_CPC import get_config

# import cartpole_continuous as cartpole_env
from cartpole_continuous import CartPoleEnv

def run_CPC():
    num_steps = 200 # 150
    random_actions_init = 10
    num_repeat_actions = 1
    len_horizon = 15
    verbose = True

    # prob = "Pendulum"
    env_name = 'CartPoleContinuous'
    # env = gym.make(env_name, render_mode='human')
    # env = gym.make(env_name)
    env = CartPoleEnv() # cartpole_env.
    # env = cartpole_env.CartPoleContinuousEnv(render_mode="rgb_array").unwrapped
    control_config = get_config(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    seed = 0
    visu_config = VisuConfig()
    episodic_return = run_env(env, env_name, seed, control_config, visu_config, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose) 
    env.close()
    return episodic_return


if __name__ == '__main__':
    episodic_return = run_CPC()
    print("episodic_return ", episodic_return, "\n")
    