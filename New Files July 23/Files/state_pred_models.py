import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim

'''
QRNN model and quantile regression loss
'''
class NextStateQuantileNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles):
        super(NextStateQuantileNetwork, self).__init__()
        self.num_quantiles = num_quantiles

        # Input layer (state + action concatenation)
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, state_dim * num_quantiles)  # Output quantiles for each state dimension

    def forward(self, state, action):
        
        if len(state.shape) == 1:
            x = torch.cat((action, state))
        else:
            x = torch.cat((action, state), dim=1)
            
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1, self.num_quantiles, state.size(-1))


def quantile_loss(predicted, target, quantiles, batch_size=32):
    """
    Calculate Quantile regression Loss.
    predicted: Predicted quantiles, shape (batch_size, state_dim, num_quantiles)
    target: Target next state, shape (batch_size, state_dim)
    quantiles: Quantiles (e.g., [0.1, 0.3, 0.7, 0.9]), shape (num_quantiles,)
    """
    
    target = target.unsqueeze(1).repeat(1,len(quantiles),1)
    
    error = target - predicted  # Shape: (batch_size, state_dim, num_quantiles)

    quantiles = quantiles.view(1, -1, 1)
    quantiles = quantiles.repeat(batch_size, 1, target.shape[-1])

    # Standard Quantile regression Loss
    quantile_loss = torch.max(
        quantiles * error,
        (quantiles - 1) * error
    )

    return quantile_loss.mean()

def get_mid_quantile(num_quantiles, predicted_quantiles):
    lower_quantile = predicted_quantiles[:, 0, :]  # Shape: (1, state_dim)
    mid_quantile = predicted_quantiles[:, int(num_quantiles/2), :]#.detach().numpy()  # Shape: (1, state_dim)
    upper_quantile = predicted_quantiles[:, -1, :]  # Shape: (1, state_dim)
    return mid_quantile

def train_QRNN(prob_vars, model_QRNN, replay_buffer_QRNN, optimizer_QRNN):
    
    batch = random.sample(replay_buffer_QRNN, prob_vars.batch_size)
    states, actions_train, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions_train, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    states = torch.clip(states, prob_vars.states_low, prob_vars.states_high)
    actions_tensor = torch.clip(actions_tensor, prob_vars.action_low, prob_vars.action_high)
    
    # Predict next state quantiles
    predicted_quantiles = model_QRNN(states, actions_tensor)  # Shape: (batch_size, num_quantiles, state_dim)
    
    # Use next state as target (can be improved with target policy)
    target_quantiles = next_states

    # Compute Quantile Loss
    loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)
    
    # Optimize the model_QRNN
    optimizer_QRNN.zero_grad()
    loss.backward()
    optimizer_QRNN.step()

'''
Neural network for predicting a single value for the next state.
Can either use the quantile regression loss (only using the 50% quantile)
or the MSE loss.
'''

class NextStateSinglePredNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NextStateSinglePredNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, state_dim)  # Single output per state dim

    def forward(self, state, action):
        if len(state.shape) == 1:
            x = torch.cat((action, state))
        else:
            x = torch.cat((action, state), dim=1) # .unsqueeze(1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
# Uses quantile regression loss only with 0.5 quantile
def quantile_loss_median(predicted, target):
    error = target - predicted
    quantile = torch.tensor(0.5)
    loss = torch.max(
        quantile * error,
        (quantile - 1) * error
    )
    return loss.mean()

# Use MSE as loss 
def mse_loss(predicted, target):
    error = target - predicted
    return error.pow(2).mean()

def train_50NN_MSENN(prob_vars, model_state, replay_buffer_state, optimizer_state, loss_state):
    
    batch = random.sample(replay_buffer_state, prob_vars.batch_size)
    states, actions_train, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions_train, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    states = torch.clip(states, prob_vars.states_low, prob_vars.states_high)
    actions_tensor = torch.clip(actions_tensor, prob_vars.action_low, prob_vars.action_high)
    
    predicted_next_states = model_state(states, actions_tensor)

    # Compute Loss
    loss = loss_state(predicted_next_states, next_states)
    
    # Optimize the model
    optimizer_state.zero_grad()
    loss.backward()
    optimizer_state.step()
