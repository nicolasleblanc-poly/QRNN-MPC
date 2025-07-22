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

if __name__ == "__main__":
    ENV_NAME = "LunarLanderContinuous-v3"
    TIMESTEPS = 15  # T
    N_SAMPLES = 1000  # K
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    noise_sigma = torch.eye(2, device=d, dtype=dtype)
    lambda_ = 1e-2

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 8
    nu = 2
    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, nx)
    ).double().to(device=d)

    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() == 1 or u.dim() == 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        xu = torch.cat((state, u), dim=1)
        
        state_residual = network(xu)
        next_state = state + state_residual

        return next_state

    def running_cost(state, action):
        x, y = state[:, 0], state[:, 1]
        vx, vy = state[:, 2], state[:, 3]
        theta, dtheta = state[:, 4], state[:, 5]
        left_leg, right_leg = state[:, 6], state[:, 7]
        a1, a2 = action[:, 0], action[:, 1]

        # Penalize distance from origin, velocity, tilt, rotation speed, and engine usage
        cost = (x ** 2 + y ** 2) \
            + 0.1 * (vx ** 2 + vy ** 2) \
            + 0.3 * (theta ** 2 + dtheta ** 2) \
            + 0.001 * (a1 ** 2 + a2 ** 2)\
                - 10 * (left_leg + right_leg)
        return cost

    dataset = None

    def train(new_data):
        global dataset
        # not normalized inside the simulator
        # new_data[:, 0] = angle_normalize(new_data[:, 0])
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        new_data = new_data.to(device=d)
        # append data to whole dataset
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)
        
        # train on the whole dataset (assume small enough we can train on all together)
        XU = dataset

        dx = XU[1:, :nx] - XU[:-1, :nx]
        Y = dx
        xu = XU[:-1]  # make same size as Y
        xu = torch.cat((xu, ), dim=1)


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

    # downward_start = True
    env = gym.make(ENV_NAME)
    state, info = env.reset()

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=nu)
            state, _, terminated, truncated, _ = env.step(action) # env.step([action])
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action
            
            if terminated:
                state, info = env.reset()
                env.reset()

        train(new_data)
        print("bootstrapping finished \n")

        # Save the initial weights after bootstrapping
        initial_state_dict = network.state_dict()

    # env_seeds = [0, 8, 15]
    seed = 8
    episodic_return_seeds = []
    max_episodes = 300
    method_name = "MPPI"
    prob = "LunarLanderContinuous"
    max_steps = 1000
    
    # for seed in env_seeds:
    episodic_return = []
    # Reset network to initial pretrained weights
    network.load_state_dict(initial_state_dict)
    
    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
    
    for episode in range(max_episodes):
        env.reset(seed=seed)

        total_reward, data = mppi.run_mppi(mppi_gym, seed, env, train, iter=max_steps, render=False) # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)
        episodic_return.append(total_reward)
    
    episodic_return = np.array(episodic_return)
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_seed{seed}_June27.npz")
    np.savez(save_path, episodic_return)
    
    print("Saved data \n")
    env.close()

   