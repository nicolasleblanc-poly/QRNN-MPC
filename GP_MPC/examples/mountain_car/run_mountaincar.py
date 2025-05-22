import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

import gymnasium as gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.run_env_function import run_env
from config_mountaincar import get_config


def run_mountain_car(num_steps=500, random_actions_init=20, num_repeat_actions=5):
    len_horizon = 12
    verbose = True
    action_dim = 1

    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name, render_mode='human')
    control_config = get_config(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    
    # visu_config = VisuConfig(render_live_plot_2d=True, render_env=True, save_render_env=True)
    visu_config = VisuConfig()
    episodic_return = run_env(env, action_dim, control_config, visu_config, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose) 
    env.close()
    return episodic_return


if __name__ == '__main__':
    episodic_return = run_mountain_car(num_steps=500, num_repeat_actions=5)
    print("episodic_return ", episodic_return, "\n")
 
