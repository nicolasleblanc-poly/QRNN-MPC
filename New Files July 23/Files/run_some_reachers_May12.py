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

from state_pred_models import NextStateQuantileNetwork, quantile_loss, NextStateSinglePredNetwork, quantile_loss_median, mse_loss
from main_funcs import main_QRNN_MPC, main_CEM, main_50NN_MSENN_MPC, main_CEM_50NN_MSENN, main_QRNN_MPC_LBFGSB, main_50NN_MSENN_MPC_LBFGSB
from ASNN import ReplayBuffer_ASNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from setup import setup_class

# Problem setup
# prob = "CartPole"
# prob = "Acrobot"
# prob = "MountainCar"
# prob = "LunarLander"
# prob = "Pendulum"
# prob = "Pendulum_xyomega"
# prob = "MountainCarContinuous"
# prob = "LunarLanderContinuous"
# prob = "PandaReacher"
# prob = "PandaReacherDense"
prob = "PandaPusher"
# prob = "PandaPusherDense"
# prob = "MuJoCoReacher"
# prob = "MuJoCoPusher"

print("prob ", prob, "\n")

prob_vars = setup_class(prob)

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

    # data = {
    #     'episode': np.arange(len(episodic_rep_returns)),
    #     'episodic_rep_returns': episodic_rep_returns,
    #     'mean_episodic_returns': mean_episodic_returns,
    #     'std_episodic_returns': std_episodic_returns
    # }

    np.savez(
    f"{prob}_{method_name}_July21.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )
# # Run MPC-QRNN random mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = True
# method_name = "MPC_QRNN_random_mid"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
# print("episode_rep_rewards_MPC_PF_QRNN_random_mid saved \n")


# # if method_name == "RS_mid_MSENN":
# # Run Random shooting (RS)
# do_RS = True
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "RS_mid_MSENN"
# use_QRNN = False
# use_50NN = False
# use_MSENN = True

# model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_MSENN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None
# loss_MSENN = mse_loss

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# print("episode_rep_rewards_RS_mid_MSENN saved \n")

# # Run CEM algos

# use_CEM = True
# prob_vars = setup_class(prob, use_CEM)

# def save_data_CEM(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

#     # data = {
#     #     'episode': np.arange(len(episodic_rep_returns)),
#     #     'episodic_rep_returns': episodic_rep_returns,
#     #     'mean_episodic_returns': mean_episodic_returns,
#     #     'std_episodic_returns': std_episodic_returns
#     # }

#     np.savez(
#     f"{prob}_{method_name}_May12_CEM.npz",
#     episode_rewards=episodic_rep_returns,
#     mean_rewards=mean_episodic_returns,
#     std_rewards=std_episodic_returns
#     )

# # if method_name == "MPC_QRNN_random_mid":
# # Run MPC-QRNN random mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = True
# method_name = "MPC_QRNN_random_mid"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
# print("episode_rep_rewards_MPC_PF_QRNN_random_mid_CEM saved \n")


# First run
# """
# if method_name == "MPC_QRNN_ASNN_mid":
# Run MPC-QRNN-ASNN mid
do_RS = False
use_ASNN = True
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "MPC_QRNN_ASNN_mid"
use_QRNN = True
use_50NN = False
use_MSENN = False

model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_QRNN = []

replay_buffer_ASN = ReplayBuffer_ASNN(10000)
model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# print("episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid ", episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, "\n")
# print("mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid ", mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, "\n")
# print("std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid ", std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, "\n")

save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid)
print("episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid saved \n")

# if method_name == "MPC_QRNN_basic_mid":
# Run MPC-QRNN basic mid
do_RS = False
use_ASNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "MPC_QRNN_basic_mid"
use_QRNN = True
use_50NN = False
use_MSENN = False

model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_QRNN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid, episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid)
print("episode_rep_rewards_MPC_PF_QRNN_basic_mid saved \n")
# """

# ###################################################################################
# # Second run
# """
# if method_name == "MPC_QRNN_random_mid":
# Run MPC-QRNN random mid
do_RS = False
use_ASNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = True
method_name = "MPC_QRNN_random_mid"
use_QRNN = True
use_50NN = False
use_MSENN = False

model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_QRNN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
print("episode_rep_rewards_MPC_PF_QRNN_random_mid saved \n")

# # # if method_name == "MPC_QRNN_CEM_mid":
# # # Run MPC-QRNN-CEM mid
# # if not prob_vars.discrete:
# #     do_RS = False
# #     use_ASNN = False
# #     model_QRNN_pretrained = None
# #     optimizer_QRNN_pretrained = None
# #     use_QRNN = True
# #     use_50NN = False
# #     use_MSENN = False

# #     method_name = "MPC_QRNN_CEM_mid"

# #     # Experience replay buffer
# #     replay_buffer_QRNN_pretrained = None

# #     use_sampling = False
# #     use_mid = True

# #     model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# #     optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# #     # Experience replay buffer
# #     replay_buffer_QRNN = []

# #     if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
# #         episode_rep_rewards_QRNN_MPC_CEM_mid, mean_episode_rep_rewards_QRNN_MPC_CEM_mid, std_episode_rep_rewards_QRNN_MPC_CEM_mid, episode_rep_SuccessRate_QRNN_MPC_CEM_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_sampling_mid, std_episode_rep_SuccessRate_QRNN_MPC_CEM_mid = main_CEM(prob_vars, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)

# #     else:
# #         episode_rep_rewards_QRNN_MPC_CEM_mid, mean_episode_rep_rewards_QRNN_MPC_CEM_mid, std_episode_rep_rewards_QRNN_MPC_CEM_mid = main_CEM(prob_vars, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)

# #     save_data(prob, method_name, episode_rep_rewards_QRNN_MPC_CEM_mid, mean_episode_rep_rewards_QRNN_MPC_CEM_mid, std_episode_rep_rewards_QRNN_MPC_CEM_mid)
# #     print("episode_rep_rewards_QRNN_MPC_CEM_mid saved \n")

# # if method_name == "RS_mid_QRNN":
# # Run Random shooting using QRNN (RS-QRNN)
# do_RS = True
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "RS_mid_QRNN"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# print("episode_rep_rewards_RS_mid_QRNN saved \n")


# # # # Run MPC-QRNN L-BFGS-B mid
# # # # use_ASNN = False
# # # # use_sampling = False
# # # # use_mid = True
# # # # method_name = "MPC_QRNN_LBFGSB_mid"
# # # # use_QRNN = True
# # # # use_50NN = False
# # # # use_MSENN = False

# # # # model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# # # # optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # # # # Experience replay buffer
# # # # replay_buffer_QRNN = []

# # # # replay_buffer_ASN = None
# # # # model_ASN = None
# # # # optimizer_ASN = None

# # # # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
# # # #     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_QRNN_MPC_LBFGSB(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)

# # # # else:
# # # #     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_QRNN_MPC_LBFGSB(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)
                                                                                                                                                                                
# # # # save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
# """

# # Third run
# """
# '''
# -----------------------------------------------------
# Only 50% quantile used in predictions
# ------------------------------------------------------
# '''
# # if method_name == "MPC_50NN_ASNN_mid":
# # Run MPC-50-ASNN mid
# do_RS = False
# use_ASNN = True
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_50NN_ASNN_mid"
# # use_QRNN = False
# use_50NN = True
# use_MSENN = False

# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = ReplayBuffer_ASNN(10000)
# model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
# optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid)
# print("episode_rep_rewards_MPC_PF_50NN_WithASNN_mid saved \n")

# # if method_name == "MPC_50NN_basic_mid":
# # Run MPC-50NN basic mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_50NN_basic_mid"
# # use_QRNN = False
# use_50NN = True
# use_MSENN = False


# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_QRNN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid, episode_rep_SuccessRate_MPC_PF_50NN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid)
# print("episode_rep_rewards_MPC_PF_50NN_basic_mid saved \n")
# """

# # Fourth run
# """
# # if method_name == "MPC_50NN_random_mid":
# # Run MPC-QRNN random mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = True
# method_name = "MPC_50NN_random_mid"
# # use_QRNN = False
# use_50NN = True
# use_MSENN = False

# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid, episode_rep_SuccessRate_MPC_PF_50NN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_random_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid)
# print("episode_rep_rewards_MPC_PF_50NN_random_mid saved \n")

# # if method_name == "MPC_50NN_CEM_mid":
# # # Run MPC-QRNN-CEM mid
# # if not prob_vars.discrete:
# #     do_RS = False
# #     use_ASNN = False
# #     model_QRNN_pretrained = None
# #     optimizer_QRNN_pretrained = None
# #     use_QRNN = False
# #     use_50NN = True
# #     use_MSENN = False

# #     method_name = "MPC_50NN_CEM_mid"

# #     # Experience replay buffer
# #     replay_buffer_QRNN_pretrained = None

# #     use_sampling = False
# #     use_mid = True

# #     model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# #     optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# #     loss_50NN = quantile_loss_median

# #     # Experience replay buffer
# #     replay_buffer_50NN = []

# #     if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
# #         episode_rep_rewards_50NN_MPC_CEM_mid, mean_episode_rep_rewards_50NN_MPC_CEM_mid, std_episode_rep_rewards_50NN_MPC_CEM_mid, episode_rep_SuccessRate_50NN_MPC_CEM_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_sampling_mid, std_episode_rep_SuccessRate_50NN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

# #     else:
# #         episode_rep_rewards_50NN_MPC_CEM_mid, mean_episode_rep_rewards_50NN_MPC_CEM_mid, std_episode_rep_rewards_50NN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

# #     save_data(prob, method_name, episode_rep_rewards_50NN_MPC_CEM_mid, mean_episode_rep_rewards_50NN_MPC_CEM_mid, std_episode_rep_rewards_50NN_MPC_CEM_mid)
# #     print("episode_rep_rewards_50NN_MPC_CEM_mid saved \n")

# # if method_name == "RS_mid_50NN":
# # Run Random shooting (RS)
# do_RS = True
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "RS_mid_50NN"
# use_QRNN = False
# use_50NN = True
# use_MSENN = False

# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# print("episode_rep_rewards_RS_mid_50NN saved \n")

# # Run MPC-QRNN L-BFGS-B mid
# # use_sampling = False
# # use_mid = True
# # method_name = "MPC_50NN_LBFGSB_mid"
# # use_QRNN = False
# # use_50NN = True
# # use_MSENN = False

# # model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# # optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# # loss_50NN = quantile_loss_median

# # # Experience replay buffer
# # replay_buffer_50NN = []

# # replay_buffer_ASN = None
# # model_ASN = None
# # optimizer_ASN = None

# # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
# #     episode_rep_rewards_MPC_505NN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_50NN_MSENN_MPC_LBFGSB(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

# # else:
# #     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_50NN_MSENN_MPC_LBFGSB(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

# # save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
# """

# ##################################


# # NEED TO RUN ALL ENVS FROM HERE
# # '''
# # -----------------------------------------------------
# # Predict next state using classic MSE error as loss
# # ------------------------------------------------------
# # '''

# # Run 5
# """
# # if method_name == "MPC_MSENN_ASNN_mid":
# # Run MPC-MSE-ASNN mid
# do_RS = False
# use_ASNN = True
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_MSENN_ASNN_mid"
# use_QRNN = False
# use_50NN = False
# use_MSENN = True

# model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
# loss_MSENN = mse_loss

# # Experience replay buffer
# replay_buffer_MSENN = []

# replay_buffer_ASN = ReplayBuffer_ASNN(10000)
# model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
# optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN,loss_MSENN,  model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid)
# print("episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid saved \n")

# # if method_name == "MPC_MSENN_basic_mid":
# # Run MPC-QRNN basic mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_MSENN_basic_mid"
# use_QRNN = False
# use_50NN = False
# use_MSENN = True

# model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
# loss_MSENN = mse_loss

# # Experience replay buffer
# replay_buffer_MSENN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_MSENN_basic_mid, mean_episode_rep_rewards_MPC_PF_MSENN_basic_mid, std_episode_rep_rewards_MPC_PF_MSENN_basic_mid, episode_rep_SuccessRate_MPC_PF_MSENN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_MSENN_basic_mid, mean_episode_rep_rewards_MPC_PF_MSENN_basic_mid, std_episode_rep_rewards_MPC_PF_MSENN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_basic_mid, mean_episode_rep_rewards_MPC_PF_MSENN_basic_mid, std_episode_rep_rewards_MPC_PF_MSENN_basic_mid)
# print("episode_rep_rewards_MPC_PF_MSENN_basic_mid saved \n")
# """

# # # Run 6
# # # """
# # # if method_name == "MPC_MSENN_random_mid":
# # # Run MPC-QRNN random mid
# # do_RS = False
# # use_ASNN = False
# # use_sampling = False
# # use_mid = True
# # do_QRNN_step_rnd = True
# # method_name = "MPC_MSENN_random_mid"
# # use_QRNN = False
# # use_50NN = False
# # use_MSENN = True

# # model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# # optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
# # loss_MSENN = mse_loss

# # # Experience replay buffer
# # replay_buffer_MSENN = []

# # replay_buffer_ASN = None
# # model_ASN = None
# # optimizer_ASN = None

# # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
# #     episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid, episode_rep_SuccessRate_MPC_PF_MSENN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_random_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# # else:
# #     episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# # save_data(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid)
# # print("episode_rep_rewards_MPC_PF_MSENN_random_mid saved \n")

# # if method_name == "MPC_MSENN_CEM_mid":
# # Run MPC-MSENN-CEM mid
# if not prob_vars.discrete:
#     do_RS = False
#     use_ASNN = False
#     model_QRNN_pretrained = None
#     optimizer_QRNN_pretrained = None
#     use_QRNN = False
#     use_50NN = False
#     use_MSENN = True

#     method_name = "MPC_MSENN_CEM_mid"

#     # Experience replay buffer
#     replay_buffer_QRNN_pretrained = None

#     use_sampling = False
#     use_mid = True

#     model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
#     optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
#     loss_MSENN = mse_loss

#     # Experience replay buffer
#     replay_buffer_MSENN = []

#     if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#         episode_rep_rewards_MSENN_MPC_CEM_mid, mean_episode_rep_rewards_MSENN_MPC_CEM_mid, std_episode_rep_rewards_MSENN_MPC_CEM_mid, episode_rep_SuccessRate_MSENN_MPC_CEM_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_sampling_mid, std_episode_rep_SuccessRate_MSENN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, use_sampling, use_mid)

#     else:
#         episode_rep_rewards_MSENN_MPC_CEM_mid, mean_episode_rep_rewards_MSENN_MPC_CEM_mid, std_episode_rep_rewards_MSENN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, use_sampling, use_mid)

#     save_data(prob, method_name, episode_rep_rewards_MSENN_MPC_CEM_mid, mean_episode_rep_rewards_MSENN_MPC_CEM_mid, std_episode_rep_rewards_MSENN_MPC_CEM_mid)

# # if method_name == "RS_mid_MSENN":
# # # Run Random shooting (RS)
# # do_RS = True
# # use_ASNN = False
# # use_sampling = False
# # use_mid = True
# # do_QRNN_step_rnd = False
# # method_name = "RS_mid_MSENN"
# # use_QRNN = False
# # use_50NN = False
# # use_MSENN = True

# # model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# # optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)

# # # Experience replay buffer
# # replay_buffer_MSENN = []

# # replay_buffer_ASN = None
# # model_ASN = None
# # optimizer_ASN = None
# # loss_MSENN = mse_loss

# # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
# #     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# # else:
# #     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# # save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# # print("episode_rep_rewards_RS_mid_MSENN saved \n")



# # Run CEM algos

# use_CEM = True
# prob_vars = setup_class(prob, use_CEM)

# def save_data_CEM(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

#     # data = {
#     #     'episode': np.arange(len(episodic_rep_returns)),
#     #     'episodic_rep_returns': episodic_rep_returns,
#     #     'mean_episodic_returns': mean_episodic_returns,
#     #     'std_episodic_returns': std_episodic_returns
#     # }

#     np.savez(
#     f"{prob}_{method_name}_May12_CEM.npz",
#     episode_rewards=episodic_rep_returns,
#     mean_rewards=mean_episodic_returns,
#     std_rewards=std_episodic_returns
#     )

# # First run
# # """
# # if method_name == "MPC_QRNN_ASNN_mid":
# # Run MPC-QRNN-ASNN mid
# do_RS = False
# use_ASNN = True
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_QRNN_ASNN_mid"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = ReplayBuffer_ASNN(10000)
# model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
# optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_WithASNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# # print("episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid ", episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, "\n")
# # print("mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid ", mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, "\n")
# # print("std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid ", std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, "\n")

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid)
# print("episode_rep_rewards_MPC_PF_QRNN_WithASNN_mid_CEM saved \n")

# # if method_name == "MPC_QRNN_basic_mid":
# # Run MPC-QRNN basic mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_QRNN_basic_mid"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid, episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid)
# print("episode_rep_rewards_MPC_PF_QRNN_basic_mid_CEM saved \n")
# # """

# ###################################################################################
# # Second run
# # """
# # if method_name == "MPC_QRNN_random_mid":
# # Run MPC-QRNN random mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = True
# method_name = "MPC_QRNN_random_mid"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
# print("episode_rep_rewards_MPC_PF_QRNN_random_mid_CEM saved \n")

# # if method_name == "RS_mid_QRNN":
# # Run Random shooting using QRNN (RS-QRNN)
# do_RS = True
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "RS_mid_QRNN"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# print("episode_rep_rewards_RS_mid_QRNN_CEM saved \n")

# # Run MPC-QRNN-CEM mid
# if not prob_vars.discrete:
#     do_RS = False
#     use_ASNN = False
#     model_QRNN_pretrained = None
#     optimizer_QRNN_pretrained = None
#     use_QRNN = True
#     use_50NN = False
#     use_MSENN = False

#     method_name = "MPC_QRNN_EvoCEM_mid"

#     model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
#     optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

#     # Experience replay buffer
#     replay_buffer_QRNN = []

#     replay_buffer_ASN = None
#     model_ASN = None
#     optimizer_ASN = None

#     # Experience replay buffer
#     replay_buffer_QRNN = []

#     if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#         episode_rep_rewards_QRNN_MPC_EvoCEM_mid, mean_episode_rep_rewards_QRNN_MPC_EvoCEM_mid, std_episode_rep_rewards_QRNN_MPC_EvoCEM_mid, episode_rep_SuccessRate_QRNN_MPC_EvoCEM_mid, mean_episode_rep_SuccessRate_QRNN_MPC_EvoCEM_mid, std_episode_rep_SuccessRate_QRNN_MPC_EvoCEM_mid = main_CEM(prob_vars, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)

#     else:
#         episode_rep_rewards_QRNN_MPC_EvoCEM_mid, mean_episode_rep_rewards_QRNN_MPC_EvoCEM_mid, std_episode_rep_rewards_QRNN_MPC_EvoCEM_mid = main_CEM(prob_vars, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)

#     save_data(prob, method_name, episode_rep_rewards_QRNN_MPC_EvoCEM_mid, mean_episode_rep_rewards_QRNN_MPC_EvoCEM_mid, std_episode_rep_rewards_QRNN_MPC_EvoCEM_mid)
#     print("episode_rep_rewards_QRNN_MPC_EvoCEM_mid_CEM saved \n")

# # Third run
# # """
# '''
# -----------------------------------------------------
# Only 50% quantile used in predictions
# ------------------------------------------------------
# '''
# # if method_name == "MPC_50NN_ASNN_mid":
# # Run MPC-50-ASNN mid
# do_RS = False
# use_ASNN = True
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_50NN_ASNN_mid"
# # use_QRNN = False
# use_50NN = True
# use_MSENN = False

# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = ReplayBuffer_ASNN(10000)
# model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
# optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASNN_mid)
# print("episode_rep_rewards_MPC_PF_50NN_WithASNN_mid_CEM saved \n")

# # if method_name == "MPC_50NN_basic_mid":
# # Run MPC-50NN basic mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_50NN_basic_mid"
# # use_QRNN = False
# use_50NN = True
# use_MSENN = False


# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_QRNN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid, episode_rep_SuccessRate_MPC_PF_50NN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid)
# print("episode_rep_rewards_MPC_PF_50NN_basic_mid_CEM saved \n")


# # Fourth run
# # """
# # if method_name == "MPC_50NN_random_mid":
# # Run MPC-QRNN random mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = True
# method_name = "MPC_50NN_random_mid"
# # use_QRNN = False
# use_50NN = True
# use_MSENN = False

# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid, episode_rep_SuccessRate_MPC_PF_50NN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_random_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid)
# print("episode_rep_rewards_MPC_PF_50NN_random_mid_CEM saved \n")

# # if method_name == "MPC_50NN_CEM_mid":
# # Run MPC-50NN-CEM mid
# if not prob_vars.discrete:
#     do_RS = False
#     use_ASNN = False
#     model_QRNN_pretrained = None
#     optimizer_QRNN_pretrained = None
#     use_QRNN = False
#     use_50NN = True
#     use_MSENN = False

#     method_name = "MPC_50NN_EvoCEM_mid"

#     # Experience replay buffer
#     replay_buffer_50NN_pretrained = None

#     use_sampling = False
#     use_mid = True

#     model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
#     optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
#     loss_50NN = quantile_loss_median

#     # Experience replay buffer
#     replay_buffer_50NN = []

#     if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#         episode_rep_rewards_50NN_MPC_EvoCEM_mid, mean_episode_rep_rewards_50NN_MPC_EvoCEM_mid, std_episode_rep_rewards_50NN_MPC_EvoCEM_mid, episode_rep_SuccessRate_50NN_MPC_EvoCEM_mid, mean_episode_rep_SuccessRate_50NN_MPC_EvoCEM_mid, std_episode_rep_SuccessRate_50NN_MPC_EvoCEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

#     else:
#         episode_rep_rewards_50NN_MPC_EvoCEM_mid, mean_episode_rep_rewards_50NN_MPC_EvoCEM_mid, std_episode_rep_rewards_50NN_MPC_EvoCEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

#     save_data(prob, method_name, episode_rep_rewards_50NN_MPC_EvoCEM_mid, mean_episode_rep_rewards_50NN_MPC_EvoCEM_mid, std_episode_rep_rewards_50NN_MPC_EvoCEM_mid)
#     print("episode_rep_rewards_50NN_MPC_EvoCEM_mid saved \n")

# # if method_name == "RS_mid_50NN":
# # Run Random shooting (RS)
# do_RS = True
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "RS_mid_50NN"
# use_QRNN = False
# use_50NN = True
# use_MSENN = False

# model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
# loss_50NN = quantile_loss_median

# # Experience replay buffer
# replay_buffer_50NN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# print("episode_rep_rewards_RS_mid_50NN_CEM saved \n")


# # Run 5
# # """
# # if method_name == "MPC_MSENN_ASNN_mid":
# # Run MPC-MSE-ASNN mid
# do_RS = False
# use_ASNN = True
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_MSENN_ASNN_mid"
# use_QRNN = False
# use_50NN = False
# use_MSENN = True

# model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
# loss_MSENN = mse_loss

# # Experience replay buffer
# replay_buffer_MSENN = []

# replay_buffer_ASN = ReplayBuffer_ASNN(10000)
# model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
# optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN,loss_MSENN,  model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid)
# print("episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid_CEM saved \n")

# # if method_name == "MPC_MSENN_basic_mid":
# # Run MPC-QRNN basic mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_MSENN_basic_mid"
# use_QRNN = False
# use_50NN = False
# use_MSENN = True

# model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
# loss_MSENN = mse_loss

# # Experience replay buffer
# replay_buffer_MSENN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_MSENN_basic_mid, mean_episode_rep_rewards_MPC_PF_MSENN_basic_mid, std_episode_rep_rewards_MPC_PF_MSENN_basic_mid, episode_rep_SuccessRate_MPC_PF_MSENN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_MSENN_basic_mid, mean_episode_rep_rewards_MPC_PF_MSENN_basic_mid, std_episode_rep_rewards_MPC_PF_MSENN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_basic_mid, mean_episode_rep_rewards_MPC_PF_MSENN_basic_mid, std_episode_rep_rewards_MPC_PF_MSENN_basic_mid)
# print("episode_rep_rewards_MPC_PF_MSENN_basic_mid_CEM saved \n")

# # Run 6
# # """
# # if method_name == "MPC_MSENN_random_mid":
# # Run MPC-QRNN random mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = True
# method_name = "MPC_MSENN_random_mid"
# use_QRNN = False
# use_50NN = False
# use_MSENN = True

# model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
# loss_MSENN = mse_loss

# # Experience replay buffer
# replay_buffer_MSENN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid, episode_rep_SuccessRate_MPC_PF_MSENN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_random_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid)
# print("episode_rep_rewards_MPC_PF_MSENN_random_mid_CEM saved \n")

# # if method_name == "MPC_MSENN_CEM_mid":
# # Run MPC-MSENN-CEM mid
# if not prob_vars.discrete:
#     do_RS = False
#     use_ASNN = False
#     model_QRNN_pretrained = None
#     optimizer_QRNN_pretrained = None
#     use_QRNN = False
#     use_50NN = False
#     use_MSENN = True

#     method_name = "MPC_MSENN_EvoCEM_mid"

#     # Experience replay buffer
#     replay_buffer_MSENN_pretrained = None

#     use_sampling = False
#     use_mid = True

#     model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
#     optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
#     loss_MSENN = mse_loss

#     # Experience replay buffer
#     replay_buffer_MSENN = []

#     if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#         episode_rep_rewards_MSENN_MPC_EvoCEM_mid, mean_episode_rep_rewards_MSENN_MPC_EvoCEM_mid, std_episode_rep_rewards_MSENN_MPC_EvoCEM_mid, episode_rep_SuccessRate_MSENN_MPC_EvoCEM_mid, mean_episode_rep_SuccessRate_MSENN_MPC_EvoCEM_mid, std_episode_rep_SuccessRate_MSENN_MPC_EvoCEM_mid = main_CEM_50NN_MSENN(prob_vars, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, use_sampling, use_mid)

#     else:
#         episode_rep_rewards_MSENN_MPC_EvoCEM_mid, mean_episode_rep_rewards_MSENN_MPC_EvoCEM_mid, std_episode_rep_rewards_MSENN_MPC_EvoCEM_mid = main_CEM_50NN_MSENN(prob_vars, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, use_sampling, use_mid)

#     save_data(prob, method_name, episode_rep_rewards_MSENN_MPC_EvoCEM_mid, mean_episode_rep_rewards_MSENN_MPC_EvoCEM_mid, std_episode_rep_rewards_MSENN_MPC_EvoCEM_mid)
#     print("episode_rep_rewards_MSENN_MPC_EvoCEM_mid saved \n")

# # if method_name == "RS_mid_MSENN":
# # Run Random shooting (RS)
# do_RS = True
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "RS_mid_MSENN"
# use_QRNN = False
# use_50NN = False
# use_MSENN = True

# model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
# optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_MSENN = []

# replay_buffer_ASN = None
# model_ASN = None
# optimizer_ASN = None
# loss_MSENN = mse_loss

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# else:
#     episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

# save_data_CEM(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
# print("episode_rep_rewards_RS_mid_MSENN_CEM saved \n")

# print("all done \n")

