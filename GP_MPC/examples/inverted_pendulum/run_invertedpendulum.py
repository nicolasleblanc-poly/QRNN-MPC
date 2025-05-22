import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

import gymnasium as gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.run_env_function import run_env
from config_invertedpendulum import get_config
# inverted_pendulum.

def run_inverted_pendulum(num_steps=1000, random_actions_init=20, num_repeat_actions=5):
    len_horizon = 30
    # verbose = True
    verbose = False

    env_name = 'InvertedPendulum-v5'
    env = gym.make(env_name, render_mode='human')
    # env = gym.make(env_name)
    control_config = get_config(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    seed = 0
    visu_config = VisuConfig(render_live_plot_2d=False, render_env=False, save_render_env=False)
    # visu_config = VisuConfig()
    episodic_return = run_env(env, env_name, seed, control_config, visu_config, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose) 
    env.close()
    return episodic_return


if __name__ == '__main__':
    episodic_return = run_inverted_pendulum(num_steps=1000, num_repeat_actions=5)
    print("episodic_return ", episodic_return, "\n")
    