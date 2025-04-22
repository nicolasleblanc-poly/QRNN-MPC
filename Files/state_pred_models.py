import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim


# QRNN model and loss
# Uses quantile regression loss
class NextStateQuantileNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles):
        super(NextStateQuantileNetwork, self).__init__()
        self.num_quantiles = num_quantiles

        # Input layer (state + action concatenation)
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, state_dim * num_quantiles)  # Output quantiles for each state dimension
        # self.layer3 = torch.tanh(256, state_dim * num_quantiles)  # Output quantiles for each state dimension

    def forward(self, state, action):
        # Concatenate state and action
        # x = torch.cat((action, state))
        # print("action ", action, "\n")
        
        # print("state ", state, "\n")
        # print("state.shape ", state.shape, "\n")
        # print("action ", action, "\n")
        # print("action.shape ", action.shape, "\n")
        
        if len(state.shape) == 1:
            x = torch.cat((action, state))
        else:
            x = torch.cat((action, state), dim=1) # .unsqueeze(1)
            
        # print("x ", x, "\n")
        # print("x.shape ", x.shape, "\n")
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1, self.num_quantiles, state.size(-1))


def quantile_loss(predicted, target, quantiles, batch_size=32):
    """
    Calculate Quantile Huber Loss.
    :param predicted: Predicted quantiles, shape (batch_size, state_dim, num_quantiles)
    :param target: Target next state, shape (batch_size, state_dim)
    :param quantiles: Quantiles (e.g., [0.1, 0.3, 0.7, 0.9]), shape (num_quantiles,)
    """
    
    # print("target.shape ", target.shape, "\n")
    # print("target ", target, "\n")
    # target = target.unsqueeze(-1)  # Shape: (batch_size, state_dim, 1)
    target = target.unsqueeze(1).repeat(1,len(quantiles),1)
    # print("target ", target, "\n")
    # print("predicted ", predicted, "\n")
    # print("target.shape ", target.shape, "\n")
    # print("predicted.shape ", predicted.shape, "\n")
    
    error = target - predicted  # Shape: (batch_size, state_dim, num_quantiles)
    
    # print("error ", error, "\n")
    quantiles = quantiles.view(1, -1, 1)
    quantiles = quantiles.repeat(batch_size, 1, target.shape[-1])  # Shape: [3, 4, 2]
    # quantiles = quantiles.repeat(batch_size, target.shape[-1], 1) 
    # quantiles = quantiles.transpose(1, 2)  # Shape: [3, 2, 4]
    # print("quantiles ", quantiles, "\n")
    
    # # Make delta adaptive by scaling it based on quantiles
    # delta = 1.0
    # adaptive_delta = delta * (1.0 + torch.abs(quantiles - 0.5))  # Give more tolerance to extreme quantiles
    
    # # print("quantiles.shape ", quantiles.shape, "\n")
    # # Calculate loss
    # huber_loss = torch.where(
    #     error.abs() <= 0.0, # 1.0
    #     0.5 * error.pow(2),
    #     error.abs() - 0.5
    # )

    # # huber_loss = error.abs()-0.5  # Simple L1 loss
    
    # # print("huber_loss ", huber_loss, "\n")
    # # print("huber_loss.shape ", huber_loss.shape, "\n")
    
    # # Quantile loss computation
    # quantile_loss = (quantiles - (error < 0).float()).abs() * huber_loss

    # Standard Quantile Loss (Pinball Loss)
    quantile_loss = torch.max(
        quantiles * error,
        (quantiles - 1) * error
    )

    return quantile_loss.mean()


# Network only the median

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
    
# Uses quantile regression loss with 0.5 only
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


# def model_predict_and_train():

    




#     pass
#     # return pass


