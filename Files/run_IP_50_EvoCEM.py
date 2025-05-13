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
from setup import setup_class

# Problem setup
prob = "InvertedPendulum"


print("prob ", prob, "\n")
print("all methods \n")
# print("method_name ", method_name, "\n")


prob_vars = setup_class(prob)

def save_data_EvoCEM(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

    # data = {
    #     'episode': np.arange(len(episodic_rep_returns)),
    #     'episodic_rep_returns': episodic_rep_returns,
    #     'mean_episodic_returns': mean_episodic_returns,
    #     'std_episodic_returns': std_episodic_returns
    # }

    np.savez(
    f"{prob}_{method_name}_May6_EvoCEM.npz",
    episode_rewards=episodic_rep_returns,
    mean_rewards=mean_episodic_returns,
    std_rewards=std_episodic_returns
    )

# Run MPC-50NN-CEM mid
if not prob_vars.discrete:
    do_RS = False
    use_ASGNN = False
    model_QRNN_pretrained = None
    optimizer_QRNN_pretrained = None
    use_QRNN = False
    use_50NN = True
    use_MSENN = False

    method_name = "MPC_50NN_EvoCEM_mid"

    # Experience replay buffer
    replay_buffer_50NN_pretrained = None

    use_sampling = False
    use_mid = True

    model_50NN = NextStateSinglePredNetwork(prob_vars.state_dim, prob_vars.action_dim)
    optimizer_50NN = optim.Adam(model_50NN.parameters(), lr=1e-3)
    loss_50NN = quantile_loss_median

    # Experience replay buffer
    replay_buffer_50NN = []

    if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
        episode_rep_rewards_50NN_MPC_EvoCEM_mid, mean_episode_rep_rewards_50NN_MPC_EvoCEM_mid, std_episode_rep_rewards_50NN_MPC_EvoCEM_mid, episode_rep_SuccessRate_50NN_MPC_EvoCEM_mid, mean_episode_rep_SuccessRate_50NN_MPC_EvoCEM_mid, std_episode_rep_SuccessRate_50NN_MPC_EvoCEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

    else:
        episode_rep_rewards_50NN_MPC_EvoCEM_mid, mean_episode_rep_rewards_50NN_MPC_EvoCEM_mid, std_episode_rep_rewards_50NN_MPC_EvoCEM_mid = main_CEM_50NN_MSENN(prob_vars, model_50NN, replay_buffer_50NN, optimizer_50NN, loss_50NN, use_sampling, use_mid)

    save_data_EvoCEM(prob, method_name, episode_rep_rewards_50NN_MPC_EvoCEM_mid, mean_episode_rep_rewards_50NN_MPC_EvoCEM_mid, std_episode_rep_rewards_50NN_MPC_EvoCEM_mid)
    print("episode_rep_rewards_50NN_MPC_EvoCEM_mid saved \n")

