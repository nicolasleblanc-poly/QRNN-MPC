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
from setup import setup_class

# Problem setup
prob = "CartPoleContinuous"

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
    f"{prob}_{method_name}_June2.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )


# Run 6
# """
# if method_name == "MPC_MSENN_random_mid":
# Run MPC-QRNN random mid
do_RS = False
use_ASGNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = True
method_name = "MPC_MSENN_random_mid"
use_QRNN = False
use_50NN = False
use_MSENN = True

model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)
loss_MSENN = mse_loss

# Experience replay buffer
replay_buffer_MSENN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid, episode_rep_SuccessRate_MPC_PF_MSENN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_random_mid, std_episode_rep_SuccessRate_MPC_PF_MSENN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

save_data(prob, method_name, episode_rep_rewards_MPC_PF_MSENN_random_mid, mean_episode_rep_rewards_MPC_PF_MSENN_random_mid, std_episode_rep_rewards_MPC_PF_MSENN_random_mid)
print("episode_rep_rewards_MPC_PF_MSENN_random_mid saved \n")

# # if method_name == "MPC_MSENN_CEM_mid":
# # Run MPC-QRNN-CEM mid
# if not prob_vars.discrete:
#     do_RS = False
#     use_ASGNN = False
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
#         episode_rep_rewards_MSENN_MPC_CEM_mid, mean_episode_rep_rewards_MSENN_MPC_CEM_mid, std_episode_rep_rewards_MSENN_MPC_CEM_mid, episode_rep_SuccessRate_MSENN_MPC_CEM_mid, mean_episode_rep_SuccessRate_MPC_PF_MSENN_WithASGNN_sampling_mid, std_episode_rep_SuccessRate_MSENN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, use_sampling, use_mid)

#     else:
#         episode_rep_rewards_MSENN_MPC_CEM_mid, mean_episode_rep_rewards_MSENN_MPC_CEM_mid, std_episode_rep_rewards_MSENN_MPC_CEM_mid = main_CEM_50NN_MSENN(prob_vars, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, use_sampling, use_mid)

#     save_data(prob, method_name, episode_rep_rewards_MSENN_MPC_CEM_mid, mean_episode_rep_rewards_MSENN_MPC_CEM_mid, std_episode_rep_rewards_MSENN_MPC_CEM_mid)

# if method_name == "RS_mid_MSENN":
# Run Random shooting (RS)
do_RS = True
use_ASGNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "RS_mid_MSENN"
use_QRNN = False
use_50NN = False
use_MSENN = True

model_MSENN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
optimizer_MSENN = optim.Adam(model_MSENN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_MSENN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None
loss_MSENN = mse_loss

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
    episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_50NN_MSENN_MPC(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

save_data(prob, method_name, episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid)
print("episode_rep_rewards_RS_mid_MSENN saved \n")

# # # Run MPC-MSENN L-BFGS-B mid
# # use_sampling = False
# # use_mid = True
# # method_name = "MPC_MSENN_LBFGSB_mid"
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
# #     episode_rep_rewards_MPC_MSENN_LBFGSB_mid, mean_episode_rep_rewards_MPC_MSENN_LBFGSB_mid, std_episode_rep_rewards_MPC_MSENN_LBFGSB_mid, episode_rep_SuccessRate_MPC_PF_MSENN_LBFGSB_mid, mean_episode_rep_SuccessRate_MPC_MSENN_LBFGSB_mid, std_episode_rep_SuccessRate_MPC_MSENN_LBFGSB_mid = main_50NN_MSENN_MPC_LBFGSB(prob_vars, method_name, model_MSENN, replay_buffer_MSENN, optimizer_MSENN, loss_MSENN, use_sampling, use_mid)

# # else:
# #     episode_rep_rewards_MPC_MSENN_LBFGSB_mid, mean_episode_rep_rewards_MPC_MSENN_LBFGSB_mid, std_episode_rep_rewards_MPC_MSENN_LBFGSB_mid = main_50NN_MSENN_MPC_LBFGSB(prob_vars, method_name, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

# # save_data(prob, method_name, episode_rep_rewards_MPC_MSENN_LBFGSB_mid, mean_episode_rep_rewards_MPC_MSENN_LBFGSB_mid, std_episode_rep_rewards_MPC_MSENN_LBFGSB_mid)

# print("part 1 QRNN \n")
# print("part 2 50NN \n")
# print("part 3 MSENN \n")
# print("only RS MSENN \n")
# print("CEM for real now \n")
# """
print("all done \n")

