import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from state_pred_models import quantile_loss
import numpy as np

# Define a named tuple for transitions
Transition = namedtuple('Transition', ('states', 'goal_states', 'outputs'))

class ReplayBuffer_ASNN:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        sampled_transitions = random.sample(self.memory, batch_size)

        states = torch.stack([torch.tensor(t.states, dtype=torch.float32) for t in sampled_transitions])
        goal_states = torch.stack([torch.tensor(t.goal_states, dtype=torch.float32) for t in sampled_transitions])
        outputs = torch.stack([torch.tensor(t.outputs, dtype=torch.float32) for t in sampled_transitions])

        return states, goal_states, outputs

    def __len__(self):
        return len(self.memory)

    def view_memory(self):
        """View all transitions in memory"""
        for idx, transition in enumerate(self.memory):
            print(f"Transition {idx}: states={transition.states}, goal_states={transition.goal_states}, outputs={transition.outputs}")

class ActionSequenceNN(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, discrete=False, nb_actions = None):
        super(ActionSequenceNN, self).__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.nb_actions = nb_actions

        self.fc = nn.Sequential(
                nn.Linear(state_dim + goal_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                )
        
        if discrete:
            self.action_output_layer = nn.Linear(64, self.nb_actions)
            # self.action_output_layer = nn.Linear(64, action_dim) # Output logits for discrete actions
        else:
            self.mu_output_layer = nn.Linear(64, action_dim) # Mean for continuous actions
            self.sigma_output_layer = nn.Linear(64, action_dim) # Uncertainty for continuous actions

    def combine_tensors(self, s, s_g):
        if s.dim() == 1:
            s = s.view(1, -1) 
        if s_g.dim() == 1:
            s_g = s_g.view(1, -1) 

        if s.size(0) != s_g.size(0):
            if s.size(0) == 1:
                s = s.repeat(s_g.size(0), 1)
            elif s_g.size(0) == 1:
                s_g = s_g.repeat(s.size(0), 1)
            else:
                raise ValueError("Cannot combine tensors with mismatched rows.")

        return torch.cat([s, s_g], dim=1)

    def forward(self, s, s_g):
        x = self.combine_tensors(s, s_g)
        x = self.fc(x)

        if self.discrete:
            logits = self.action_output_layer(x)
            # print("logits ", logits, "\n")
            action_probs = torch.softmax(logits, dim=-1) # -1
            return action_probs # [0] # , None
            # return logits
        else:
            mu = self.mu_output_layer(x)
            sigma = self.sigma_output_layer(x)
            sigma = torch.exp(sigma)
            return mu, sigma

def gaussian_nll_loss(mus, sigmas, a):
    log_likelihood = 0.5 * torch.log(2 * torch.pi * sigmas**2) + 0.5 * ((a - mus)**2) / (sigmas**2)
    return log_likelihood.mean()

# def categorical_cross_entropy_loss(logits, actions):
#     actions = actions.long().squeeze()  # Convert to shape [batch_size]
#     return torch.nn.functional.cross_entropy(logits, actions)

def categorical_cross_entropy_loss(action_probs, actions):
    actions = actions.long().squeeze()  # Convert to shape [batch_size]
    log_probs = torch.log(action_probs + 1e-10)  # Small epsilon to avoid log(0)
    return torch.nn.functional.nll_loss(log_probs, actions)

# def categorical_cross_entropy_loss(action_probs, actions):
#     actions = actions.long().squeeze()  # Convert to shape [batch_size]
#     return torch.nn.functional.cross_entropy(torch.log(action_probs + 1e-10), actions)

    # return -torch.sum(actions * torch.log(action_probs + 1e-10), dim=-1).mean()

def train_ActionSequenceNN(model, replay_buffer, batch_size, optimizer, num_epochs):
    for _ in range(num_epochs):
        states, goal_states, actions = replay_buffer.sample(batch_size)

        if model.discrete:
            outputs = model(states, goal_states)
            # print("outputs ", outputs, "\n")
            # logits = model(states, goal_states)
            # print("logits ", logits, "\n")
            # print("actions ",9 actions, "\n")
            loss = categorical_cross_entropy_loss(outputs, actions)
            # loss = categorical_cross_entropy_loss(logits, actions)
        else:
            outputs, uncertainties = model(states, goal_states)
            
            # print("outputs ", outputs, "\n")
            # print("uncertainties ", uncertainties, "\n")
            # print("actions ", actions, "\n")
            loss = gaussian_nll_loss(outputs, uncertainties, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


