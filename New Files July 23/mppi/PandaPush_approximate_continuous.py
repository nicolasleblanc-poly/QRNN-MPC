"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""
import gymnasium as gym
import numpy as np
import torch
import logging
import math
from pytorch_mppi_folder import mppi_modified as mppi
import os
import panda_gym

if __name__ == "__main__":
    ENV_NAME = "PandaPush-v3"
    TIMESTEPS = 15  # T
    N_SAMPLES = 50  # K
    ACTION_LOW = [-1.0, -1.0, -1.0]
    ACTION_HIGH = [1.0, 1.0, 1.0]

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    noise_sigma = torch.eye(3, device=d, dtype=dtype)
    lambda_ = 1e-2

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    global goal_state 

    nx = 18
    nu = 3
    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, nx)
    ).double().to(device=d)

    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW[0], ACTION_HIGH[0])
        if state.dim() == 1 or u.dim() == 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        # if u.shape[1] > 1:
        #     u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)
        # print("xu.shape ", xu.shape, "\n")
        state_residual = network(xu)
        next_state = state + state_residual

        return next_state

    def running_cost(state, action):
        # Need the goal state based on the seed
        # goal_state = np.array([0.04108851, -0.06906398,  0.02]) # seed = 0
        # goal_state = np.array([-0.05190832,  0.14618306,  0.02]) # seed = 8
        goal_state = np.array([0.05782301, 0.09474514, 0.02]) # seed = 15
        goal_state = torch.tensor(goal_state, dtype=torch.float32, device=state.device).reshape(1, 3)
        cost = torch.norm(state[:, :3] - goal_state, dim=1)

        return cost

    dataset = None

    def train(new_data):
        global dataset
        # not normalized inside the simulator
        # new_data[:, 0] = angle_normalize(new_data[:, 0])
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -nu:] = torch.clamp(new_data[:, -nu:], ACTION_LOW[0], ACTION_HIGH[0])
        new_data = new_data.to(device=d)
        # append data to whole dataset
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)
        
        # train on the whole dataset (assume small enough we can train on all together)
        XU = dataset
        
        Y = XU[1:, :nx] - XU[:-1, :nx]
        xu = XU[:-1, :]  # make same size as Y
    
        # thaw network
        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters())
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            # MSE loss
            Yhat = network(xu)
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            loss.mean().backward()
            optimizer.step()

        # freeze network
        for param in network.parameters():
            param.requires_grad = False

        
    env = gym.make(ENV_NAME)
    observation, info = env.reset()
    state  = observation['observation']
    goal_state_boostrap = observation['desired_goal']

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = state # env.state
            action = np.random.uniform(low=ACTION_LOW[0], high=ACTION_HIGH[0], size=nu)
            state, reward, terminated, truncated, info = env.step(action)
            state = state['observation']
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        train(new_data)
        print("bootstrapping finished \n")

        # Save the initial weights after bootstrapping
        initial_state_dict = network.state_dict()

    # env_seeds = [0, 8, 15]
    # seed = 0
    # seed = 8
    seed = 15
    print("seed ", seed, "\n")
    episodic_return_seeds = []
    max_episodes = 400
    method_name = "MPPI"
    prob = "PandaPush"
    max_steps = 50
    
    # for seed in env_seeds:
    episodic_return = []
    # Reset network to initial pretrained weights
    network.load_state_dict(initial_state_dict)
    
    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
        lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
        u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
    
    for episode in range(max_episodes):
        state, info = env.reset(seed=seed)
        
        goal_state = state['desired_goal']

        total_reward, data = mppi.run_mppi(mppi_gym, seed, env, train, iter=max_steps, render=False, prob=prob) #  # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)
        episodic_return.append(total_reward)
    
    episodic_return = np.array(episodic_return)
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_seed{seed}_July26.npz")
    np.savez(save_path, episodic_return)
    
    print("Saved data \n")
    env.close()
   