import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

import gymnasium as gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.run_env_function import run_env
from config_pendulum import get_config

def run_pendulum():
    num_steps = 200 # 150
    random_actions_init = 10
    num_repeat_actions = 1
    len_horizon = 15
    verbose = True

    # prob = "Pendulum"
    env_name = 'Pendulum-v1'
    env = gym.make(env_name, render_mode='human')
    # env = gym.make(env_name)
    control_config = get_config(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    seed = 0
    visu_config = VisuConfig()
    episodic_return = run_env(env, env_name, seed, control_config, visu_config, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose) 
    env.close()
    return episodic_return


if __name__ == '__main__':
    episodic_return = run_pendulum()
    print("episodic_return ", episodic_return, "\n")
    