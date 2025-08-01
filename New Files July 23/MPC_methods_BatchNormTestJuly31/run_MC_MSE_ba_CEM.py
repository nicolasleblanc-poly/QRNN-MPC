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
prob = "MountainCar"

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
    f"{prob}_{method_name}_July31_CEM_4repeat.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )



# Run 5
# """
# if method_name == "MPC_MSENN_ASNN_mid":
# Run MPC-MSE-ASNN mid
do_RS = False
use_ASNN = True
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
# method_name = "MPC_MSENN_ASNN_mid"
method_name = "MPC_MSENN_ASNN_mid_ChangeHorizonTo30From70_AddedVelocity"
use_QRNN = False
use_50NN = False
use_MSENN = True

model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
loss_MSENN = mse_loss

# Experience replay buffer
replay_buffer_MSENN = []

replay_buffer_ASN = ReplayBuffer_ASNN(10000)
model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

else:
    episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN,loss_MSENN,  model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, mean_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid, std_episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid)
print("episode_rep_rewards_MPC_PF_MSENN_WithASNN_mid saved \n")

# # # if method_name == "MPC_MSENN_basic_mid":
# # Run MPC-QRNN basic mid
# do_RS = False
# use_ASNN = False
# use_sampling = False
# use_mid = True
# do_QRNN_step_rnd = False
# method_name = "MPC_MSENN_basic_mid_ChangeHorizonTo30From70_AddedVelocity"
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


# print("all done \n")

