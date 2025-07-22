"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""
import gymnasium as gym
import numpy as np
import torch
import logging
import math
from pytorch_cem import cem
import os

if __name__ == "__main__":
    ENV_NAME = "LunarLanderContinuous-v3"
    TIMESTEPS = 15  # T
    N_SAMPLES = 1000  # K
    N_ELITES = 10
    SAMPLE_ITER = 3
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    noise_sigma = torch.tensor([1.0, 1.0], device=d, dtype=dtype)
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


    def running_cost(state, action, horizon, t):
        # LunarLander cost function
        x = state[..., 0]  # horizontal position
        y = state[..., 1]  # vertical position
        vx = state[..., 2]  # horizontal velocity
        vy = state[..., 3]  # vertical velocity
        angle = state[..., 4]  # angle
        ang_vel = state[..., 5]  # angular velocity
        left_leg = state[..., 6]  # left leg contact
        right_leg = state[..., 7]  # right leg contact
        
        # Target is to land at (0,0) with low speed and upright
        cost = (
            x.pow(2) +              # Distance from center
            y.pow(2) +               # Altitude cost (want to be near ground)
            0.1 * vx.pow(2) +            # Horizontal velocity
            0.1 * vy.pow(2) +            # Vertical velocity
            0.3 * angle.pow(2) +         # Angle (want to be upright)
            0.3 * ang_vel.pow(2) +      # Angular velocity
            0.001 * action.pow(2).sum(-1) # Action penalty
            - 10 * (left_leg + right_leg)
        )
            
        return cost


    dataset = None

    def train(new_data):
        global dataset
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
    seed = 15
    episodic_return_seeds = []
    max_episodes = 300
    method_name = "CEM"
    prob = "LunarLanderContinuous"
    max_steps = 1000
    
    # for seed in env_seeds:
    episodic_return = []
    # Reset network to initial pretrained weights
    network.load_state_dict(initial_state_dict)

    for episode in range(max_episodes):
        env.reset(seed=seed)

        ctrl = cem.CEM(dynamics, running_cost, nx, nu, num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER,
                        horizon=TIMESTEPS, device=d, num_elite=N_ELITES,
                        u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d), init_cov_diag=1)

        total_reward, data = cem.run_cem(ctrl, seed, env, train, iter=max_steps, render=False) # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)

        
        episodic_return.append(total_reward)
    
    episodic_return = np.array(episodic_return)
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_seed{seed}_June27.npz")
    np.savez(save_path, episodic_return)
    
    print("Saved data \n")
    env.close()
