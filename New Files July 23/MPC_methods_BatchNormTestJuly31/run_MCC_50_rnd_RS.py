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
from main_funcs import main_QRNN_MPC, main_50NN_MSENN_MPC
from ASNN import ReplayBuffer_ASNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from setup import setup_class

# Problem setup
prob = "MountainCarContinuous"


print("prob ", prob, "\n")
print("all methods \n")
# print("method_name ", method_name, "\n")


prob_vars = setup_class(prob)

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

    # data = {
    #     'episode': np.arange(len(episodic_rep_returns)),
    #     'episodic_rep_returns': episodic_rep_returns,
    #     'mean_episodic_returns': mean_episodic_returns,
    #     'std_episodic_returns': std_episodic_returns
    # }

    np.savez(
    f"{prob}_{method_name}_July31_4repeat.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )

# Fourth run
# """
# if method_name == "MPC_50NN_random_mid":
# Run MPC-QRNN random mid
do_RS = False
use_ASNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = True
# method_name = "MPC_50NN_random_mid"
method_name = "MPC_50NN_random_mid_ChangeHorizonTo30From12_AddedVelocity"
# use_QRNN = False
use_50NN = True
use_MSENN = False

model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
loss_50NN = quantile_loss_median

# Experience replay buffer
replay_buffer_50NN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid, episode_rep_SuccessRate_MPC_PF_50NN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_random_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

else:
    episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_50NN_random_mid, mean_episode_rep_rewards_MPC_PF_50NN_random_mid, std_episode_rep_rewards_MPC_PF_50NN_random_mid)
print("episode_rep_rewards_MPC_PF_50NN_random_mid saved \n")

# if method_name == "MPC_50NN_CEM_mid":
# # Run MPC-QRNN-CEM mid
# if not prob_vars.discrete:
#     do_RS = False
#     use_ASNN = False
#     model_QRNN_pretrained = None
#     optimizer_QRNN_pretrained = None
#     use_QRNN = False
#     use_50NN = True
#     use_MSENN = False

#     method_name = "MPC_50NN_CEM_mid"

#     # Experience replay buffer
#     replay_buffer_QRNN_pretrained = None

#     use_sampling = False
#     use_mid = True

#     model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
#     optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
#     loss_50NN = quantile_loss_median

#     # Experience replay buffer
#     replay_buffer_50NN = []

#     if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#         episode_rep_rewards_50NN_MPC_CEM_mid, mean_episode_rep_rewards_50NN_MPC_CEM_mid, std_episode_rep_rewards_50NN_MPC_CEM_mid, episode_rep_SuccessRate_50NN_MPC_CEM_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_WithASNN_sampling_mid, std_episode_rep_SuccessRate_50NN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

#     else:
#         episode_rep_rewards_50NN_MPC_CEM_mid, mean_episode_rep_rewards_50NN_MPC_CEM_mid, std_episode_rep_rewards_50NN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

#     save_data(prob, method_name, episode_rep_rewards_50NN_MPC_CEM_mid, mean_episode_rep_rewards_50NN_MPC_CEM_mid, std_episode_rep_rewards_50NN_MPC_CEM_mid)
#     print("episode_rep_rewards_50NN_MPC_CEM_mid saved \n")

# if method_name == "RS_mid_50NN":
# Run Random shooting (RS)
do_RS = True
use_ASNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "RS_mid_50NN_ChangeHorizonTo30From12_AddedVelocity"
use_QRNN = False
use_50NN = True
use_MSENN = False

model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
loss_50NN = quantile_loss_median

# Experience replay buffer
replay_buffer_50NN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

else:
    episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
print("episode_rep_rewards_RS_mid_50NN saved \n")

# Run MPC-QRNN L-BFGS-B mid
# use_sampling = False
# use_mid = True
# method_name = "MPC_50NN_LBFGSB_mid"
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
#     episode_rep_rewards_MPC_505NN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_50NN_MSENN_MPC_LBFGSB(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_50NN_MSENN_MPC_LBFGSB(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid)
