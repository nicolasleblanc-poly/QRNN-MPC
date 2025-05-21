'''
Swing up pendulum with limited torque using Gymnasium Pendulum environment
with Numba-compatible dynamics
'''
import gymnasium as gym
import numpy as np
from numba import jit
from ilqr import iLQR
from ilqr.containers import Dynamics, Cost
from ilqr.utils import GetSyms, Constrain, Bounded
import os 
import copy

# Initialize Gymnasium Pendulum environment
# env = gym.make('Pendulum-v1', render_mode='human')
env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
n_x = 4  # [x, v]
n_u = 1  # [force]
# dt = env.dt  # Use the environment's time step (0.05 by default)
dt = 0.02

# # Physics parameters
# g = 10.0
# m = 1.0
# l = 1.0
# max_torque = 2.0
# max_speed = 8.0

# Numba-compatible dynamics function
# @jit(nopython=True)
# def f_numba(x, u, env):
def f(x, u, copy_env):
    copy_env.unwrapped.set_state(x[:2], x[2:])
    state, reward, terminated, truncated, info = copy_env.step(u)

    return state

# Wrapper function to match expected interface
def f(x, u, copy_env):
    # return f_numba(x, u, env)
    return f(x, u, copy_env)

# Create dynamics container
InvertedPendulum = Dynamics.Discrete(f)

# Construct cost function
x, u = GetSyms(n_x, n_u)
x_goal = np.array([0.0, 0.0, 0.0, 0.0])
Q = np.diag([10.0, 0.1, 100.0, 0.1]) # Penalize position error more
R = np.diag([0.01]) # Penalize control effort lightly
QT = np.diag([100.0, 1.0, 1000.0, 1.0]) # Strong terminal cost
cons = Bounded(u, high=[3.0], low=[-3.0])
InvertedPendulumCost = Cost.QR(Q, R, QT, x_goal, cons)

# Initialize the controller
controller = iLQR(InvertedPendulum, InvertedPendulumCost)

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
        f"{prob}_{method_name}_results.npz",
        episode_rewards=episodic_rep_returns,
        mean_rewards=mean_episodic_returns,
        std_rewards=std_episodic_returns
        )

env_seeds = [0]#, 8, 15]
episodic_return_seeds = []
max_steps = 1000
max_episodes = 300
method_name = "iLQR"
prob = "InvertedPendulum"

# env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

for seed in env_seeds:
    episodic_return = []
    # Initial guess for controls
    us_init = np.random.uniform(low=-3, high=3, size=(max_steps, n_u))
    for episode in range(max_episodes):
        total_reward = 0
        observation, _ = env.reset(seed=seed)
        if episode > 0:
            us_init = us
        x0 = observation
        # Get optimal states and actions
        # print("x0", x0.shape, "\n")
        # print("us_init", us_init.shape, "\n")
        copy_env = copy.deepcopy(env)
        xs, us, cost_trace = controller.fit(x0, us_init, copy_env)
        
        for i in range(len(us)):
            action = us[i]
            observation, reward, terminated, truncated, _ = env.step(action)
            states.append(observation)
            actions.append(action)
            env.render()
            total_reward += reward
            
            if terminated or truncated:
                break
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

# save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
print("Saved data \n")
# print("Total reward:", total_reward, "\n")
env.close()

# import matplotlib.pyplot as plt
# plt.plot(mean_episodic_return, label='Mean Episodic Return')
# plt.show()


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