import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm # If you want to use tqdm for progress bar
import pandas as pd

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

from MPC_QRNN_funcs import start_QRNN_MPC_wASNN, start_QRNNrand_RS
from MPC_50NN_MSENN_funcs import start_50NN_MSENN_MPC_wASNN, start_50NN_MSENNrand_RS
from MPC_UsingEnv_funcs import start_UsingEnv_MPC_wASNN, start_UsingEnvrand_RS
from ASNN import ReplayBuffer_ASNN, ActionSequenceNN, gaussian_nll_loss, categorical_cross_entropy_loss, train_ActionSequenceNN
from state_pred_models import NextStateQuantileNetwork, quantile_loss, NextStateSinglePredNetwork, quantile_loss_median, mse_loss

# Only record every 100th episode
def should_record(episode_id):
    return episode_id % 100 == 0

def main_50NN_MSENN_MPC(prob_vars, method_name, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN):
    
    episode_rep_rewards = np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        episode_rep_SuccessRate= np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    env = prob_vars.env
        
    # for rep_ep in tqdm(range(prob_vars.nb_rep_episodes)): # If you want to use tqdm for progress bar
    for rep_ep in range(prob_vars.nb_rep_episodes):
        seed = int(prob_vars.random_seeds[rep_ep]) # Seed for episode
        
        """
        Gymnasium integrated recording documentation:
        https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
        """
        # If you want to record videos, uncomment the following lines
        # if prob_vars.prob != "MuJoCoReacher" and prob_vars.prob != "MuJoCoPusher" and prob_vars.prob != "InvertedPendulum":
            # if prob_vars.change_prob is None and prob_vars.std is not None:
            #     env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.std}_{method_name}_seed{seed}", video_length=prob_vars.max_steps)
            # elif prob_vars.change_prob is not None and prob_vars.std is None:
            #     env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.change_prob}_{method_name}_seed{seed}", video_length=prob_vars.max_steps)

        if not use_ASNN:

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                episode_reward_list, episode_SuccessRate = start_50NN_MSENNrand_RS(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)
            
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:    
                episode_reward_list = start_50NN_MSENNrand_RS(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)
            episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)

        else:

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                episode_reward_list, episode_SuccessRate = start_50NN_MSENN_MPC_wASNN(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASNN)
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:
                episode_reward_list = start_50NN_MSENN_MPC_wASNN(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASNN)
            
            episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)        
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards

def main_QRNN_MPC(prob_vars, method_name, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN):

    episode_rep_rewards = np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        episode_rep_SuccessRate= np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    env = prob_vars.env
        
    # for rep_ep in tqdm(range(prob_vars.nb_rep_episodes)): # If you want to use tqdm for progress bar
    for rep_ep in range(prob_vars.nb_rep_episodes):
        seed = int(prob_vars.random_seeds[rep_ep]) # Seed for episode

        
        """
        Gymnasium integrated recording documentation:
        https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
        """
        
        # If you want to record videos, uncomment the following lines
        # if prob_vars.prob != "MuJoCoReacher" and prob_vars.prob != "MuJoCoPusher" and prob_vars.prob != "InvertedPendulum":
        #     if prob_vars.change_prob is None and prob_vars.std is not None:
        #         env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.std}_{method_name}_seed{seed}") # , video_length=prob_vars.max_steps
        #     elif prob_vars.change_prob is not None and prob_vars.std is None:
        #         env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.change_prob}_{method_name}_seed{seed}") # , video_length=prob_vars.max_steps

        if not use_ASNN:

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                episode_reward_list, episode_SuccessRate = start_QRNNrand_RS(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)
            
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:    
                episode_reward_list = start_QRNNrand_RS(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASNN)
            
            episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)

        else:

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                episode_reward_list, episode_SuccessRate = start_QRNN_MPC_wASNN(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASNN)
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:
                episode_reward_list = start_QRNN_MPC_wASNN(prob_vars, env, seed, model_QRNN, replay_buffer_QRNN, optimizer_QRNN, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASNN)
            
            episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)    
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards
    
def main_UsingEnv_MPC(prob_vars, method_name, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN):

    episode_rep_rewards = np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        episode_rep_SuccessRate= np.zeros((prob_vars.nb_rep_episodes, prob_vars.max_episodes))

    env = prob_vars.env
        
    # for rep_ep in tqdm(range(prob_vars.nb_rep_episodes)): # If you want to use tqdm for progress bar
    for rep_ep in range(prob_vars.nb_rep_episodes):
        seed = int(prob_vars.random_seeds[rep_ep]) # Seed for episode

        
        """
        Gymnasium integrated recording documentation:
        https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
        """
        
        # If you want to record videos, uncomment the following lines
        # if prob_vars.prob != "MuJoCoReacher" and prob_vars.prob != "MuJoCoPusher" and prob_vars.prob != "InvertedPendulum":
        #     if prob_vars.change_prob is None and prob_vars.std is not None:
        #         env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.std}_{method_name}_seed{seed}") # , video_length=prob_vars.max_steps
        #     elif prob_vars.change_prob is not None and prob_vars.std is None:
        #         env = RecordVideo(prob_vars.env, video_folder="videos", episode_trigger=should_record, name_prefix=f"{prob_vars.prob}_{prob_vars.change_prob}_{method_name}_seed{seed}") # , video_length=prob_vars.max_steps

        if not use_ASNN:

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                episode_reward_list, episode_SuccessRate = start_UsingEnvrand_RS(prob_vars, env, seed, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)
            
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:    
                episode_reward_list = start_UsingEnvrand_RS(prob_vars, env, seed, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_ASNN)
            
            episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)

        else:

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                episode_reward_list, episode_SuccessRate = start_UsingEnv_MPC_wASNN(prob_vars, env, seed, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_ASNN)
                episode_rep_SuccessRate[rep_ep] = episode_SuccessRate
            else:
                episode_reward_list = start_UsingEnv_MPC_wASNN(prob_vars, env, seed, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_ASNN)
            
            episode_rep_rewards[rep_ep] = np.array(episode_reward_list).reshape(prob_vars.max_episodes)    
        
    mean_episode_rep_rewards = np.mean(episode_rep_rewards, axis=0)
    std_episode_rep_rewards = np.std(episode_rep_rewards, axis=0)

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        mean_episode_rep_SuccessRate = np.mean(episode_rep_SuccessRate, axis=0)
        std_episode_rep_SuccessRate = np.std(episode_rep_SuccessRate, axis=0)

        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards, episode_rep_SuccessRate, mean_episode_rep_SuccessRate, std_episode_rep_SuccessRate

    else:
        return episode_rep_rewards, mean_episode_rep_rewards, std_episode_rep_rewards
