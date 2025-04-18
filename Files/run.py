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

from models import NextStateQuantileNetwork, quantile_loss, NextStateSinglePredNetwork, quantile_loss_median, mse_loss
from main_funcs import main_QRNN_MPC, main_CEM
from ASGNN import ReplayBuffer_ASGNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from setup import setup_class

# Problem setup
# prob = "CartPole"
prob = "Acrobot"
# prob = "MountainCar"
# prob = "LunarLander"
# prob = "Pendulum"
# prob = "Pendulum_xyomega"
# prob = "MountainCarContinuous"
# prob = "LunarLanderContinuous"
# prob = "PandaReacher"
# prob = "PandaPusher"
# prob = "MuJoCoReacher"
# prob = "MuJoCoPusher"

prob_vars = setup_class(prob)


# Run MPC-QRNN-ASGNN mid
do_RS = False
use_ASGNN = True
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "MPC_QRNN_ASGNN_mid"

model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_QRNN = []

replay_buffer_ASN = ReplayBuffer_ASGNN(10000)
model_ASN = ActionSequenceNN(prob_vars.state_dim, prob_vars.goal_state_dim, prob_vars.action_dim, discrete=prob_vars.discrete, nb_actions=prob_vars.nb_actions)
optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
    episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, episode_rep_SuccessRate_MPC_PF_QRNN_WithASGNN_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_WithASGNN_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_WithASGNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, mean_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid, std_episode_rep_rewards_MPC_PF_QRNN_WithASGNN_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)



# Run MPC-QRNN basic mid
do_RS = False
use_ASGNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "MPC_QRNN_basic_mid"

model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_QRNN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid, episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_basic_mid, mean_episode_rep_rewards_MPC_PF_QRNN_basic_mid, std_episode_rep_rewards_MPC_PF_QRNN_basic_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)


# Run MPC-QRNN random mid
do_RS = False
use_ASGNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = True
method_name = "MPC_QRNN_random_mid"

model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_QRNN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
    episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid, episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid, std_episode_rep_SuccessRate_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_MPC_PF_QRNN_random_mid, mean_episode_rep_rewards_MPC_PF_QRNN_random_mid, std_episode_rep_rewards_MPC_PF_QRNN_random_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

# Run MPC-QRNN-CEM mid
if not prob_vars.discrete:
    do_RS = False
    use_ASGNN = False
    model_QRNN_pretrained = None
    optimizer_QRNN_pretrained = None

    # Experience replay buffer
    replay_buffer_QRNN_pretrained = None

    use_sampling = False
    use_mid = True

    model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
    optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

    # Experience replay buffer
    replay_buffer_QRNN = []

    if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher" or prob == "MuJoCoPusher":
        episode_rep_rewards_QRNN_MPC_CEM_mid, mean_episode_rep_rewards_QRNN_MPC_CEM_mid, std_episode_rep_rewards_QRNN_MPC_CEM_mid, episode_rep_SuccessRate_QRNN_MPC_CEM_mid, mean_episode_rep_SuccessRate_MPC_PF_QRNN_WithASGNN_sampling_mid, std_episode_rep_SuccessRate_QRNN_MPC_CEM_mid = main_CEM(prob, std, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid, horizon, max_episodes, max_steps, random_seeds, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state)

    else:
        episode_rep_rewards_QRNN_MPC_CEM_mid, mean_episode_rep_rewards_QRNN_MPC_CEM_mid, std_episode_rep_rewards_QRNN_MPC_CEM_mid = main_CEM(prob_vars, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)

# Run Random shooting (RS)
do_RS = True
use_ASGNN = False
use_sampling = False
use_mid = True
do_QRNN_step_rnd = False
method_name = "RS_mid"

model_QRNN = NextStateQuantileNetwork(prob_vars.state_dim, prob_vars.action_dim, prob_vars.num_quantiles)
optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer_QRNN = []

replay_buffer_ASN = None
model_ASN = None
optimizer_ASN = None

if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
    episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid, episode_rep_SuccessRate_RS_mid, mean_episode_rep_SuccessRate_RS_mid, std_episode_rep_SuccessRate_RS_mid = main_QRNN_MPC(prob, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)

else:
    episode_rep_rewards_RS_mid, mean_episode_rep_rewards_RS_mid, std_episode_rep_rewards_RS_mid = main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)




