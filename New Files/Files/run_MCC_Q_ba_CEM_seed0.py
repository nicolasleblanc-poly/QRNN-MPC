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
from ASNN import ReplayBuffer_ASGNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from setup_seed0 import setup_class

# Problem setup
prob = "MountainCarContinuous"

print("prob ", prob, "\n")
print("all methods \n")
# print("method_name ", method_name, "\n")

use_CEM = True
prob_vars = setup_class(prob, use_CEM)

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

    # data = {
    #     'episode': np.arange(len(episodic_rep_returns)),
    #     'episodic_rep_returns': episodic_rep_returns,
    #     'mean_episodic_returns': mean_episodic_returns,
    #     'std_episodic_returns': std_episodic_returns
    # }

    np.savez(
    f"{prob}_{method_name}_June12_CEM_seed0.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )

# # First run
# # """
# # if method_name == "MPC_QRNN_ASGNN_mid":
# # Run MPC-QRNN-ASGNN mid
# do_RS = False
# use_ASGNN = True
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_QRNN_ASGNN_mid"
# use_QRNN = True
# use_50NN = False
# use_MSENN = False

# model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
# optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# # Experience replay buffer
# replay_buffer_QRNN = []

# replay_buffer_ASN = ReplayBuffer_ASGNN(10000)
# model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
# optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

# if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
#     episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, episode_rep_SuccessRate_MPC_PF_QRNN_WithASGNN_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_WithASGNN_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_WithASGNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

# else:
#     episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

# # print("episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid ", episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, "\n")
# # print("mean_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid ", mean_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, "\n")
# # print("std_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid ", std_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, "\n")

# save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid)
# print("episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid saved \n")

# if method_name == "MPC_QRNN_basic_mid":
# Run MPC-QRNN basic mid
do_RS = False
use_ASGNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
# method_name = "MPC_QRNN_basic_mid"
method_name = "MPC_QRNN_basic_mid_ChangeHorizonTo30From12_AddedVelocity"
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
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid, episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid)
print("episode_rep_rewards_MPC_PF_QRNN_basic_mid saved \n")
# """

print("all done \n")

