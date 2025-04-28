import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import copy
import torch
import torch.nn as nn
import torch.optim as optim
# import panda_gym

import pandas as pd

# from tqdm import tqdm

from collections import deque, namedtuple

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress DeprecationWarnings


from evotorch import Problem, Solution, SolutionBatch
from evotorch.algorithms import CEM
from evotorch.logging import StdOutLogger

import logging

logging.getLogger("evotorch").disabled = True
logging.getLogger("evotorch").setLevel(logging.ERROR)  # or logging.CRITICAL

from MPC_QRNN_funcs import start_QRNN_MPC_wASGNN, start_QRNNrand_RS
from MPC_50NN_MSENN_funcs import start_50NN_MSENN_MPC_wASGNN, start_50NN_MSENNrand_RS
from ASGNN import ReplayBuffer_ASGNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from state_pred_models import NextStateQuantileNetwork, quantile_loss, NextStateSinglePredNetwork, quantile_loss_median, mse_loss
from cem_continuous import start_QRNN_MPC_CEM
from cem_continuous_50NN_MSENN import start_50NN_MSENN_MPC_CEM

# Only record every 100th episode
def should_record(episode_id):
    return episode_id % 50 == 0

def main_50NN_MSENN_MPC(prob_vars, method_name, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN):
    # std is the std for continuous action spaces or the change prob
    # for discrete action spaces
    episode_rep_rewards = np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        episode_rep_SuccessRate= np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    env = prob_vars.env
        
    # for rep_ep in tqdm(range(prob_vars.nb_rep_episodes)):
    for rep_ep in range(prob_vars.nb_rep_episodes):
    # for seed in tqdm(random_seeds):
        seed = int(prob_vars.random_seeds[rep_ep]) # Seed for episode

        # Wrap the env to record videos into the 'videos/' folder
        # print("prob ", prob, "\n")
        # print("std ", std, "\n")
        # print("method_name ", method_name, "\n")
        # print("seed ", seed, "\n")
        
        """
        Gymnasium integrated recording documentation:
        https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
        """
        # if prob_vars.change_prob is None and prob_vars.std is not None:
        #     env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.std}_{method_name}_seed{seed}", video_length=prob_vars.max_steps)
        # elif prob_vars.change_prob is not None and prob_vars.std is None:
        #     env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.change_prob}_{method_name}_seed{seed}", video_length=prob_vars.max_steps)

        # Reset models
        # model_QRNN = NextStateQuantileNetwork(state_dim, action_dim, num_quantiles)
        # optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

        # # Experience replay buffer
        # replay_buffer_QRNN = []


        if not use_ASGNN:
            # replay_buffer_ASN = None # ReplayBuffer(10000)
            # model_ASN = None # ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = None # optim.Adam(model_ASN.parameters(), lr=1e-3)
            
            # episode_reward_list_RS_sampling = start_QRNNrand_RS(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, goal_state)

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
                episode_reward_list, episode_SuccessRate = start_50NN_MSENNrand_RS(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)
            
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:    
                episode_reward_list = start_50NN_MSENNrand_RS(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)
            
            episode_rep_rewards[rep_ep] = episode_reward_list

        else:
            # replay_buffer_ASN = ReplayBuffer(10000)
            # model_ASN = ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
                episode_reward_list, episode_SuccessRate = start_50NN_MSENN_MPC_wASGNN(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN)
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:
                episode_reward_list = start_50NN_MSENN_MPC_wASGNN(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN)
            
            episode_rep_rewards[rep_ep] = episode_reward_list
        
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards

def main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN):
    # std is the std for continuous action spaces or the change prob
    # for discrete action spaces
    episode_rep_rewards = np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        episode_rep_SuccessRate= np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    env = prob_vars.env
        
    # for rep_ep in tqdm(range(prob_vars.nb_rep_episodes)):
    for rep_ep in range(prob_vars.nb_rep_episodes):
    # for seed in tqdm(random_seeds):
        seed = int(prob_vars.random_seeds[rep_ep]) # Seed for episode

        # Wrap the env to record videos into the 'videos/' folder
        # print("prob ", prob, "\n")
        # print("std ", std, "\n")
        # print("method_name ", method_name, "\n")
        # print("seed ", seed, "\n")
        
        """
        Gymnasium integrated recording documentation:
        https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
        """
        # if prob_vars.change_prob is None and prob_vars.std is not None:
        #     env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.std}_{method_name}_seed{seed}", video_length=prob_vars.max_steps)
        # elif prob_vars.change_prob is not None and prob_vars.std is None:
        #     env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.change_prob}_{method_name}_seed{seed}", video_length=prob_vars.max_steps)

        # Reset models
        # model_QRNN = NextStateQuantileNetwork(state_dim, action_dim, num_quantiles)
        # optimizer_QRNN = optim.Adam(model_QRNN.parameters(), lr=1e-3)

        # # Experience replay buffer
        # replay_buffer_QRNN = []


        if not use_ASGNN:
            # replay_buffer_ASN = None # ReplayBuffer(10000)
            # model_ASN = None # ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = None # optim.Adam(model_ASN.parameters(), lr=1e-3)
            
            # episode_reward_list_RS_sampling = start_QRNNrand_RS(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, goal_state)

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
                episode_reward_list, episode_SuccessRate = start_QRNNrand_RS(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)
            
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:    
                episode_reward_list = start_QRNNrand_RS(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN)
            
            episode_rep_rewards[rep_ep] = episode_reward_list

        else:
            # replay_buffer_ASN = ReplayBuffer(10000)
            # model_ASN = ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
                episode_reward_list, episode_SuccessRate = start_QRNN_MPC_wASGNN(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN)
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:
                episode_reward_list = start_QRNN_MPC_wASGNN(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN)
            
            episode_rep_rewards[rep_ep] = episode_reward_list
        
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards
    
    
def main_CEM(prob_vars, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid):
    
    episode_rep_rewards = np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    env = prob_vars.env

    if prob_vars == "PandaReacher" or prob_vars == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        episode_rep_SuccessRate= np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))
        
    # for rep_ep in tqdm(range(prob_vars.nb_rep_episodes)):
    for rep_ep in range(prob_vars.nb_rep_episodes):
    # for seed in tqdm(random_seeds):
        seed = int(prob_vars.random_seeds[rep_ep]) # Seed for episode

        # # Wrap the env to record videos into the 'videos/' folder
        # if prob_vars.change_prob is None and prob_vars.std is not None: # Continuous action space so use std
        #     env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.std}_{seed}_MPC_CEM_mid_seed{seed}", video_length=prob_vars.max_steps)
        # elif prob_vars.change_prob is not None and prob_vars.std is None:
        #     env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.change_prob}_{seed}_MPC_CEM_mid_seed{seed}", video_length=prob_vars.max_steps)
            
        # if not use_ASGNN:
            # replay_buffer_ASN = None # ReplayBuffer(10000)
            # model_ASN = None # ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = None # optim.Adam(model_ASN.parameters(), lr=1e-3)
            
            # episode_reward_list_RS_sampling = start_QRNNrand_RS(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, goal_state)

        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
            episode_reward_list, episode_SuccessRate = start_QRNN_MPC_CEM(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)
        
            episode_rep_SuccessRate[rep_ep] = np.array(episode_SuccessRate)
        else:
            episode_reward_list = start_QRNN_MPC_CEM(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid)
        
        episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards
    
    
    
def main_CEM_50NN_MSENN(prob_vars, model_state, replay_buffer_state, optimizer_state, loss_state, use_sampling, use_mid):
    
    episode_rep_rewards = np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    if prob_vars == "PandaReacher" or prob_vars == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        episode_rep_SuccessRate= np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))
        
    env = prob_vars.env
    
    # for rep_ep in tqdm(range(prob_vars.nb_rep_episodes)):
    for rep_ep in range(prob_vars.nb_rep_episodes):
    # for seed in tqdm(random_seeds):
        seed = int(prob_vars.random_seeds[rep_ep]) # Seed for episode

        # # Wrap the env to record videos into the 'videos/' folder
        # if prob_vars.change_prob is None and prob_vars.std is not None: # Continuous action space so use std
        #     env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.std}_{seed}_MPC_CEM_mid_seed{seed}", video_length=prob_vars.max_steps)
        # elif prob_vars.change_prob is not None and prob_vars.std is None:
        #     env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.change_prob}_{seed}_MPC_CEM_mid_seed{seed}", video_length=prob_vars.max_steps)
            
        # if not use_ASGNN:
            # replay_buffer_ASN = None # ReplayBuffer(10000)
            # model_ASN = None # ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = None # optim.Adam(model_ASN.parameters(), lr=1e-3)
            
            # episode_reward_list_RS_sampling = start_QRNNrand_RS(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, goal_state)

        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
            episode_reward_list, episode_SuccessRate = start_50NN_MSENN_MPC_CEM(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, use_sampling, use_mid)
        
            episode_rep_SuccessRate[rep_ep] = np.array(episode_SuccessRate)
        else:
            episode_reward_list = start_50NN_MSENN_MPC_CEM(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, use_sampling, use_mid)
        
        episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards
    
