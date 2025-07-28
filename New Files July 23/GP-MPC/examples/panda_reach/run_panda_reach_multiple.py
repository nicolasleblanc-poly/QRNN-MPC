import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))
import panda_gym
import gymnasium as gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
# from config_pandareach import get_config
from config_pandareach_seed0 import get_config_seed0
from config_pandareach_seed8 import get_config_seed8
from config_pandareach_seed15 import get_config_seed15

from rl_gp_mpc.run_env_function import run_env_multiple

if __name__ == '__main__':
    num_episodes = 400 # 5 #20
    random_actions_init=10
    num_repeat_actions=5
    num_steps = 50
    len_horizon = 12
    verbose = False
    
    seeds = [0, 8, 15]

    env_name = 'PandaReach-v3'
    # env_name = 'PandaReachDense-v3'
    # env = gym.make(env_name, render_mode='human')
    env = gym.make(env_name)
    # control_config = get_config(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    control_config_seed0 = get_config_seed0(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    control_config_seed8 = get_config_seed8(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    control_config_seed15 = get_config_seed15(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    control_config = [control_config_seed0, control_config_seed8, control_config_seed15]
    
    visu_config = VisuConfig(render_live_plot_2d=False, render_env=False, save_render_env=False)
    # visu_config = VisuConfig()
    
    run_env_multiple(env, env_name, control_config, visu_config, seeds, num_episodes, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose)
