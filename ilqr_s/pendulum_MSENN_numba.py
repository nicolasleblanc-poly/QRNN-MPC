'''
Swing up pendulum with limited torque using Gymnasium Pendulum environment
with Numba-compatible dynamics
'''
import gymnasium as gym
import numpy as np
from numba import jit
from ilqr_numba_MSENN import iLQR
from ilqr_numba_MSENN.containers import Dynamics, Cost
from ilqr_numba_MSENN.utils import GetSyms, Constrain, Bounded
from ilqr_numba_MSENN.controller import MPC

import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Initialize Gymnasium Pendulum environment
# env = gym.make('Pendulum-v1', render_mode='human')
env = gym.make('Pendulum-v1', render_mode='rgb_array')
n_x = 3  # [cos(theta), sin(theta), angular velocity]
n_u = 1  # [torque]
# dt = env.dt  # Use the environment's time step (0.05 by default)
dt = 0.05

# Physics parameters
g = 10.0
m = 1.0
l = 1.0
max_torque = 2.0
max_speed = 8.0

class NextStateSinglePredNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NextStateSinglePredNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, state_dim)  # Single output per state dim

    def forward(self, state, action):
        # print("state.shape ", state.shape, "\n")
        # print("action.shape ", action.shape, "\n")
        if len(state.shape) == 1:
            x = torch.cat((action, state))
        else:
            x = torch.cat((action, state), dim=1) # .unsqueeze(1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Use MSE as loss 
def mse_loss(predicted, target):
    error = target - predicted
    return error.pow(2).mean()

# Numba-compatible dynamics function
# @jit(nopython=True)
# def f_numba(x, u, model):
#     # Convert state to angle and angular velocity
#     theta = np.arctan2(x[1], x[0])
#     thdot = x[2]

#     # Clip the action
#     u_clipped = min(max(u[0], -max_torque), max_torque)
    
#     # Calculate angular acceleration
#     thdot_dot = 3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u_clipped
#     # (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u_clipped
    
#     # Update angular velocity with manual clipping
#     new_thdot = thdot + thdot_dot * dt
#     if new_thdot > max_speed:
#         new_thdot = max_speed
#     elif new_thdot < -max_speed:
#         new_thdot = -max_speed
    
#     # Update angle
#     new_theta = theta + new_thdot * dt
    
#     # Return state in Gymnasium format
#     return np.array([np.cos(new_theta), np.sin(new_theta), new_thdot])

# def f_numba(x, u, model):
    
#     next_state = model(x, u)
    
#     return next_state.detach().numpy()  # Convert to numpy array

# # Wrapper function to match expected interface
# def f(x, u, model):
#     return f_numba(x, u, model)

# @jit(nopython=True)
def f(x, u, model):
    x = torch.tensor(x, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    next_state = model(x, u)
    
    return next_state.detach().numpy()  # Convert to numpy array

# Create dynamics container
Pendulum = Dynamics.Discrete(f)

# Construct cost function
x, u = GetSyms(n_x, n_u)
x_goal = np.array([1, 0, 0])  # Upright position
Q = np.diag([1, 1, 0.1])      # State cost weights
R = np.diag([0.1])             # Control cost weight
QT = np.diag([10, 10, 1])      # Terminal state cost weights

# Add torque constraints
cons = Bounded(u, high=[max_torque], low=[-max_torque])
SwingUpCost = Cost.QR(Q, R, QT, x_goal, cons)

# Initialize the controller
controller = iLQR(Pendulum, SwingUpCost)

# Run simulation
observation, _ = env.reset()
states = []
actions = []

# # Initial state (pointing downward)
# # x0 = np.array([-1, 0, 0])
# x0 = observation

# # Initial guess for controls
# max_steps = 200
# us_init = np.random.uniform(low=-2, high=2, size=(max_steps, n_u))
# # us_init = np.random.randn(max_steps, n_u) * 0.01
# # us_init = np.random.randn(200, n_u) * 0.01

# # Get optimal states and actions
# xs, us, cost_trace = controller.fit(x0, us_init)

# Set initial state to match our starting point
# env.unwrapped.state = [np.pi, 0]  # [theta, theta_dot]

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

        # Get the folder where this script is located
        origin_folder = os.path.dirname(os.path.abspath(__file__))
        # Construct full path to save
        save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results.npz")

        np.savez(
        save_path,
        # f"{prob}_{method_name}_results.npz",
        episode_rewards=episodic_rep_returns,
        mean_rewards=mean_episodic_returns,
        std_rewards=std_episodic_returns
        )

# env_seeds = [0, 8, 15]
env_seeds = [0]
episodic_return_seeds = []
max_steps = 200
max_episodes = 1#00
method_name = "iLQR"
prob = "Pendulum"
horizon = 15
batch_size = 32

action_low = env.action_space.low[0]
action_high = env.action_space.high[0]

states_low = torch.tensor([-1, -1, -8])
states_high = torch.tensor([1, 1, 8])

model = NextStateSinglePredNetwork(n_x, n_u)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Experience replay buffer
replay_buffer = []

mpc = MPC(controller, control_horizon=horizon, model=model)

# env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)
from tqdm import trange
for seed in env_seeds:
    episodic_return = []
    
    for episode in range(max_episodes):
    # for episode in trange(max_episodes):
        total_reward = 0
        observation, _ = env.reset(seed=seed)

        # Initial guess for controls
        us_init = np.random.uniform(low=-2, high=2, size=(max_steps, n_u))
        
        # for step in range(max_steps):
        for step in trange(max_steps):
            state = observation
            # Get optimal states and actions
            # xs, us, cost_trace = controller.fit(x0, us_init)
            
            # Get optimal action using MPC and iLQR
            mpc.set_initial(us_init)
            us = mpc.control(state)
            
            # for i in range(len(us)):
            action = us[0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            # states.append(next_state)
            # actions.append(action)
            env.render()
            total_reward += reward
            
            if prob == "CartPole" or prob == "Acrobot" or prob == "MountainCar" or prob == "LunarLander":
                replay_buffer.append((state, np.array([action]), reward, next_state, terminated))
            else:
                replay_buffer.append((state, action, reward, next_state, terminated))
            
            if len(replay_buffer) < batch_size:
                pass
            else:
                batch = random.sample(replay_buffer, batch_size)
                states, actions_train, rewards, next_states, dones = zip(*batch)
                # print("batch states ", states, "\n")
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions_tensor = torch.tensor(actions_train, dtype=torch.float32)
                # print("actions.shape ", actions_tensor, "\n")
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                #     # Clip states to ensure they are within the valid range
                #     # before inputting them to the model (sorta like normalization)
                states = torch.clip(states, states_low, states_high)
                # states = 2 * ((states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
                actions_tensor = torch.clip(actions_tensor, action_low, action_high)
                
                # Predict next state quantiles
                # predicted_quantiles = model_QRNN(states, actions_tensor)  # Shape: (batch_size, num_quantiles, state_dim)
                predicted_next_states = model(states, actions_tensor)
                
                # Use next state as target (can be improved with target policy)
                # target_quantiles = next_states
                
                # Compute the target quantiles (e.g., replicate next state across the quantile dimension)
                # target_quantiles = next_states.unsqueeze(-1).repeat(1, 1, num_quantiles)

                # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)
                loss = mse_loss(predicted_next_states, next_states)
                
                # # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)
                
                # Optimize the model_QRNN
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if terminated or truncated:
                break

            # Shift the actions for the next step
            us_init = np.roll(us_init, -1, axis=0)
            us_init[-1] = env.action_space.sample()  # Sample a random action for the last step
            # print("us_init", us_init, "\n")
            
            state = next_state

        episodic_return.append(total_reward)
        # print("Total reward:", total_reward, "\n")
        
    episodic_return_seeds.append(episodic_return)
        
episodic_return_seeds = np.array(episodic_return_seeds)

mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
std_episodic_return = np.std(episodic_return_seeds, axis=0)
print("max_episodes", max_episodes, "\n")
print("episodic_return_seeds.shape ", episodic_return_seeds.shape, "\n")
print("mean_episodic_return ", mean_episodic_return.shape, "\n")
print("std_episodic_return.shape ", std_episodic_return.shape, "\n")

# import matplotlib.pyplot as plt
# plt.plot(mean_episodic_return, label='Mean Episodic Return')
# plt.show()

save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
print("Saved data \n")
# print("Total reward:", total_reward, "\n")
env.close()


# # Plot results
# import matplotlib.pyplot as plt
# theta = np.arctan2(np.array(states)[:, 1], np.array(states)[:, 0])
# theta = np.where(theta < 0, 2*np.pi + theta, theta)

# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(theta)
# plt.title('Pendulum Angle')
# plt.ylabel('Radians')
# plt.axhline(y=0, color='r', linestyle='--', label='Target (upright)')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(actions)
# plt.title('Control Torque')
# plt.ylabel('Nm')
# plt.xlabel('Time step')
# plt.axhline(y=max_torque, color='r', linestyle=':', label='Torque limit')
# plt.axhline(y=-max_torque, color='r', linestyle=':')
# plt.tight_layout()
# plt.show()