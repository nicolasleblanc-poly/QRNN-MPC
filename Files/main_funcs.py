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
import panda_gym

import pandas as pd

from tqdm import tqdm

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

# Only record every 100th episode
def should_record(episode_id):
    return episode_id % 50 == 0


def main_QRNN_MPC(prob, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, random_seeds, nb_rep_episodes, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state):
    # std is the std for continuous action spaces or the change prob
    # for discrete action spaces
    episode_rep_rewards = np.zeros((nb_rep_episodes, max_episodes))

    if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
        episode_rep_SuccessRate= np.zeros((nb_rep_episodes, max_episodes))
        
    for rep_ep in tqdm(range(nb_rep_episodes)):
    # for seed in tqdm(random_seeds):
        seed = int(random_seeds[rep_ep]) # Seed for episode

        # Wrap the env to record videos into the 'videos/' folder
        # print("prob ", prob, "\n")
        # print("std ", std, "\n")
        # print("method_name ", method_name, "\n")
        # print("seed ", seed, "\n")
        
        """
        Gymnasium integrated recording documentation:
        https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
        """
        if change_prob is None and std is not None:
            env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob}_{std}_{method_name}_seed{seed}", video_length=max_steps)
        elif change_prob is not None and std is None:
            env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob}_{change_prob}_{method_name}_seed{seed}", video_length=max_steps)

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

            if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                episode_reward_list, episode_SuccessRate = start_QRNNrand_RS(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state)
            
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:    
                episode_reward_list = start_QRNNrand_RS(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state)
            
            episode_rep_rewards[rep_ep] = episode_reward_list

        else:
            # replay_buffer_ASN = ReplayBuffer(10000)
            # model_ASN = ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = optim.Adam(model_ASN.parameters(), lr=1e-3)

            if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                episode_reward_list, episode_SuccessRate = start_QRNN_MPC_wASGNN(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_reps_MPC, nb_top_particles, nb_random, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state)
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:
                episode_reward_list = start_QRNN_MPC_wASGNN(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_reps_MPC, nb_top_particles, nb_random, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state)
            
            episode_rep_rewards[rep_ep] = episode_reward_list
        
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards
    
    
def main_CEM(prob, std, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid, horizon, max_episodes, max_steps, seed, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state):
    
    episode_rep_rewards = np.zeros((nb_rep_episodes, max_episodes))

    if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
        episode_rep_SuccessRate= np.zeros((nb_rep_episodes, max_episodes))
        
    for rep_ep in tqdm(range(nb_rep_episodes)):
    # for seed in tqdm(random_seeds):
        seed = int(random_seeds[rep_ep]) # Seed for episode

        # Wrap the env to record videos into the 'videos/' folder
        if change_prob is None and std is not None: # Continuous action space so use std
            env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob}_{std}_{seed}_MPC_CEM_mid_seed{seed}", video_length=max_steps)
        elif change_prob is not None and std is None:
            env = RecordVideo(env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob}_{change_prob}_{seed}_MPC_CEM_mid_seed{seed}", video_length=max_steps)
            
        # if not use_ASGNN:
            # replay_buffer_ASN = None # ReplayBuffer(10000)
            # model_ASN = None # ActionSequenceNN(state_dim, goal_state_dim, action_dim, discrete=discrete, nb_actions=nb_actions)
            # optimizer_ASN = None # optim.Adam(model_ASN.parameters(), lr=1e-3)
            
            # episode_reward_list_RS_sampling = start_QRNNrand_RS(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN, horizon, max_episodes, max_steps, std, change_prob, seed, nb_top_particles, nb_random, nb_reps_MPC, env, state_dim, action_dim, action_low, action_high, goal_state)

        if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
            episode_reward_list, episode_SuccessRate = start_QRNN_MPC_CEM(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid, horizon, max_episodes, max_steps, seed, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state)
        
            episode_rep_SuccessRate[rep_ep] = np.array(episode_SuccessRate)
        else:
            episode_reward_list = start_QRNN_MPC_CEM(prob, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, use_sampling, use_mid, horizon, max_episodes, max_steps, seed, env, state_dim, action_dim, action_low, action_high, states_low, states_high, goal_state)
        
        episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(max_episodes)
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards
    
    

