import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress DeprecationWarnings
from evotorch import Problem, Solution, SolutionBatch
from evotorch.algorithms import CEM
from evotorch.logging import StdOutLogger

import logging

logging.getLogger("evotorch").disabled = True
logging.getLogger("evotorch").setLevel(logging.ERROR)  # or logging.CRITICAL

from main_funcs import main_UsingEnv_MPC
from ASNN import ReplayBuffer_ASNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from setup import setup_class
# import panda_gym # Need to comment when running on the cluster!

import os

# Problem setup
# prob = "CartPole"
# prob = "Acrobot"
prob = "MountainCar"
# prob = "LunarLander"
# prob = "Pendulum"
# prob = "Pendulum_xyomega"
# prob = "MountainCarContinuous"
# prob = "LunarLanderContinuous"
# prob = "PandaReacher"
# prob = "PandaReacherDense"
# prob = "PandaPusher"
# prob = "PandaPusherDense"
# prob = "MuJoCoReacher"
# prob = "MuJoCoPusher"

print("prob ", prob, "\n")

prob_vars = setup_class(prob)

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):
    
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_July21.npz")
    # print("save_path ", save_path, "\n")
    np.savez(
    save_path,
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )

# --------------------------------------------------------------------

# 1. Run PF algos

# 1.1 UsingEnv-PF methods
# First run
# """
# if method_name == "MPC_QRNN_ASNN_mid":
# 1.1.1 Run QRNN-ASNN-PF
do_RS = False
use_ASNN = True
do_QRNN_step_rnd = False
method_name = "MPC_UsingEnv_ASNN_mid"

replay_buffer_ASN = ReplayBuffer_ASNN(10000)
model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher" or prob == "PandaReacherDense" or prob == "PandaPusherDense":
    episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid, std_episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_UsingEnv_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_UsingEnv_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_UsingEnv_WithASNN_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

else:
    episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid, std_episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid, std_episode_rep_rewards_MPC_PF_UsingEnv_WithASNN_mid)
print("episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid saved \n")

# 1.1.2 Run UsingEnv-basic-PF
do_RS = False
use_ASNN = False
do_QRNN_step_rnd = False
method_name = "MPC_UsingEnv_basic_mid"

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher" or prob == "PandaReacherDense" or prob == "PandaPusherDense":
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid, episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid)
print("episode_rep_rewards_MPC_PF_QRNN_basic_mid saved \n")
# """

# # ###################################################################################
# # 1.1.3 Run UsingEnv-rnd-PF
# do_RS = False
# use_ASNN = False
# do_QRNN_step_rnd = True
# method_name = "MPC_UsingEnv_random_mid"

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher" or prob == "PandaReacherDense" or prob == "PandaPusherDense":
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
# print("episode_rep_rewards_MPC_PF_QRNN_random_mid saved \n")

# # 1.1.4 Run Random shooting using Env for simulations (UsingEnv-RS)
# # No need to run QRNN-RS twice since they do not optimize the particles (so no difference between PF and CEM)
# do_RS = True
# use_ASNN = False
# do_QRNN_step_rnd = False
# method_name = "RS_mid_UsingEnv"

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher" or prob == "PandaReacherDense" or prob == "PandaPusherDense":
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# else:
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# print("episode_rep_rewards_RS_mid_QRNN saved \n")

# # 2. Run CEM algos

# use_CEM = True
# prob_vars = setup_class(prob, use_CEM)

# def save_data_CEM(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):
    
#     origin_folder = os.path.dirname(os.path.abspath(__file__))
#     save_path = os.path.join(origin_folder, f"{prob}_{method_name}_July21.npz")

#     np.savez(
#     save_path,
#     episode_rewards=episodic_rep_returns,
#     mean_rewards=mean_episodic_returns,
#     std_rewards=std_episodic_returns
#     )

# # First run
# # """
# # 2.1 UsingEnv-CEM methods

# # 2.1.1 Run UsingEnv-ASNN-CEM
# do_RS = False
# use_ASNN = True
# do_QRNN_step_rnd = False
# method_name = "MPC_UsingEnv_ASNN_mid"

# replay_buffer_ASN = ReplayBuffer_ASNN(10000)
# model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
# optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher" or prob == "PandaReacherDense" or prob == "PandaPusherDense":
#     episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)


# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid)
# print("episode_rep_rewards_MPC_UsingEnv_WithASNN_mid_CEM saved \n")

# # 2.1.2 Run QRNN-basic-CEM using mid quantile
# # if method_name == "MPC_QRNN_basic_mid":
# # Run MPC-QRNN basic mid
# do_RS = False
# use_ASNN = False
# do_QRNN_step_rnd = False
# method_name = "MPC_QRNN_basic_mid"

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher" or prob == "PandaReacherDense" or prob == "PandaPusherDense":
#     episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid, episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid)
# print("episode_rep_rewards_MPC_UsingEnv_basic_mid_CEM saved \n")
# # """

# ###################################################################################
# # Second run
# # """
# # if method_name == "MPC_QRNN_random_mid":
# # 2.1.3 Run MPC-QRNN-rnd-CEM using mid quantile
# # Run MPC-QRNN random mid
# do_RS = False
# use_ASNN = False
# do_QRNN_step_rnd = True
# method_name = "MPC_QRNN_random_mid"

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher" or prob == "PandaReacherDense" or prob == "PandaPusherDense":
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
# print("episode_rep_rewards_MPC_UsingEnv_random_mid_CEM saved \n")

print("all done \n")

