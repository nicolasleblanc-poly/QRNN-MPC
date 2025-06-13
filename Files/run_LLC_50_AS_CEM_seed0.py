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
from ASGNN import ReplayBuffer_ASGNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from setup_seed0 import setup_class

prob = "LunarLanderContinuous"

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
    f"{prob}_{method_name}_May6_CEM_seed0.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )

# Third run
# """
'''
-----------------------------------------------------
Only 50% quantile used in predictions
------------------------------------------------------
'''
# if method_name == "MPC_50NN_ASGNN_mid":
# Run MPC-50-ASGNN mid
do_RS = False
use_ASGNN = True
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "MPC_50NN_ASGNN_mid"
# use_QRNN = False
use_50NN = True
use_MSENN = False

model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
loss_50NN = quantile_loss_median

# Experience replay buffer
replay_buffer_50NN = []

replay_buffer_ASN = ReplayBuffer_ASGNN(10000)
model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid, episode_rep_SuccessRate_MPC_PF_50NN_WithASGNN_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_WithASGNN_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_WithASGNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid)
print("episode_rep_rewards_MPC_PF_50NN_WithASGNN_mid saved \n")

# if method_name == "MPC_50NN_basic_mid":
# Run MPC-50NN basic mid
do_RS = False
use_ASGNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "MPC_50NN_basic_mid"
# use_QRNN = False
use_50NN = True
use_MSENN = False


model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
optimizer_QRNN = optim.Adam(model_50NN.parameters(), lr=1e-3)
loss_50NN = quantile_loss_median

# Experience replay buffer
replay_buffer_50NN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid, episode_rep_SuccessRate_MPC_PF_50NN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_50NN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_50NN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_50NN_basic_mid, mean_episode_rep_rewards_MPC_PF_50NN_basic_mid, std_episode_rep_rewards_MPC_PF_50NN_basic_mid)
print("episode_rep_rewards_MPC_PF_50NN_basic_mid saved \n")
# """
