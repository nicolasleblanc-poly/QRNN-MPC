'''
Swing up pendulum with limited torque using Gymnasium Pendulum environment
with Numba-compatible dynamics
'''
import gymnasium as gym
import numpy as np
from numba import jit
from ilqr_numba import iLQR
from ilqr_numba.containers import Dynamics, Cost
from ilqr_numba.utils import GetSyms, Constrain, Bounded
from ilqr_numba.controller import MPC

import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

import cartpole_continuous as cartpole_env

# Initialize Gymnasium Pendulum environment
# env = gym.make('Pendulum-v1', render_mode='human')
# env = gym.make('Pendulum-v1', render_mode='rgb_array')
env = cartpole_env.CartPoleContinuousEnv(render_mode="rgb_array").unwrapped
n_x = 4  # [x, v, theta, theta_dot]
n_u = 1  # [torque]
# dt = env.dt  # Use the environment's time step (0.05 by default)
dt = 0.05

# Physics parameters
g = 10.0
m = 1.0
l = 1.0
max_torque = 2.0
max_speed = 8.0


# Numba-compatible dynamics function
@jit(nopython=True)
def f_numba(state, u):
    # Constants
    g = 9.8  # gravity
    m_cart = 1.0  # mass of the cart
    m_pole = 0.1  # mass of the pole
    total_mass = m_cart + m_pole
    l = 0.5  # half the pole length
    mu = 0.0  # friction (can be 0)
    dt = 0.02  # time step (default Gym)

    # Unpack state
    x, x_dot, theta, theta_dot = state

    # Force input
    force = np.clip(u[0], -10.0, 10.0)  # adjust limits if needed

    # Equations of motion from OpenAI Gym CartPole
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    temp = (force + m_pole * l * theta_dot ** 2 * sin_theta) / total_mass
    theta_acc = (g * sin_theta - cos_theta * temp) / (l * (4.0/3.0 - m_pole * cos_theta**2 / total_mass))
    x_acc = temp - m_pole * l * theta_acc * cos_theta / total_mass

    # Integrate using Euler method
    x = x + dt * x_dot
    x_dot = x_dot + dt * x_acc
    theta = theta + dt * theta_dot
    theta_dot = theta_dot + dt * theta_acc

    return np.array([x, x_dot, theta, theta_dot])

# def f_numba(x, u, model):
    
#     next_state = model(x, u)
    
#     return next_state.detach().numpy()  # Convert to numpy array

# Wrapper function to match expected interface
def f(x, u):
    return f_numba(x, u)


# def f(x, u, model):
#     x = torch.tensor(x, dtype=torch.float32)
#     u = torch.tensor(u, dtype=torch.float32)
#     next_state = model(x, u)
    
#     return next_state.detach().numpy()  # Convert to numpy array

# Create dynamics container
Pendulum = Dynamics.Discrete(f)

# Construct cost function
x, u = GetSyms(n_x, n_u)
x_goal = np.array([0, 0, 0, 0])  # Upright position
Q = np.diag([0.1, 0.1, 1, 0])      # State cost weights
R = np.diag([0.1])             # Control cost weight
QT = np.diag([1, 1, 10, 0])      # Terminal state cost weights

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

env_seeds = [0, 8, 15]
# env_seeds = [0]
episodic_return_seeds = []
max_steps = 200
max_episodes = 100
method_name = "iLQR"
prob = "Pendulum"
horizon = 15
batch_size = 32

action_low = env.action_space.low[0]
action_high = env.action_space.high[0]

states_low = torch.tensor([-4.8, -torch.inf, -0.41887903, -torch.inf])
states_high = torch.tensor([4.8, torch.inf, 0.41887903, torch.inf])

mpc = MPC(controller, control_horizon=horizon)

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
          
            if terminated or truncated:
                break

            # Shift the actions for the next step
            us_init = np.roll(us_init, -1, axis=0)
            us_init[-1] = env.action_space.sample()  # Sample a random action for the last step
            # print("us_init", us_init, "\n")
            
            state = next_state

        episodic_return.append(total_reward)
        print("Total reward:", total_reward, "\n")
        
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