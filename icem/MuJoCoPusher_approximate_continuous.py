"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""
import gymnasium as gym
import numpy as np
import torch
import logging
import math
from pytorch_icem import icem
import os

if __name__ == "__main__":
    ENV_NAME = "Pusher-v5"
    TIMESTEPS = 15  # T
    N_SAMPLES = 50  # K
    ACTION_LOW = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
    ACTION_HIGH = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double
    noise_sigma = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=d, dtype=dtype)
    lambda_ = 1e-2

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 23
    nu = 7
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
        xu = torch.cat((state, u), dim=1)
        # print("xu.shape ", xu.shape, "\n")
        state_residual = network(xu)
        next_state = state + state_residual

        return next_state

    def running_cost(state, action):
        cost = torch.norm(state[:, 14:17]-state[:, 17:20], dim=1)+torch.norm(state[:, 17:20]-state[:, 20:], dim=1)+0.1*(torch.norm(actions, dim=1))**2

        return cost

    def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

        # Get the folder where this script is located
        origin_folder = os.path.dirname(os.path.abspath(__file__))
        # Construct full path to save
        save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_July28.npz")

        np.savez(
        save_path,
        episode_rewards=episodic_rep_returns,
        mean_rewards=mean_episodic_returns,
        std_rewards=std_episodic_returns
        )

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
    state, info = env.reset()

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = state # env.state
            action = np.random.uniform(low=ACTION_LOW[0], high=ACTION_HIGH[0], size=nu)
            state, reward, terminated, truncated, info = env.step(action)
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        train(new_data)
        print("bootstrapping finished \n")

        # Save the initial weights after bootstrapping
        initial_state_dict = network.state_dict()

    env_seeds = [0, 8, 15]
    episodic_return_seeds = []
    max_episodes = 400
    method_name = "iCEM"
    prob = "MuJoCoPusher"
    max_steps = 100
    for seed in env_seeds:
        episodic_return = []
        # Reset network to initial pretrained weights
        network.load_state_dict(initial_state_dict)
        
        for episode in range(max_episodes):
            env.reset(seed=seed)

            icem_gym = icem.iCEM(dynamics, icem.accumulate_running_cost(running_cost), nx, nu, sigma=noise_sigma,
                     warmup_iters=5, online_iters=5,
                     num_samples=N_SAMPLES, num_elites=10, horizon=TIMESTEPS, device=d, )
        
            total_reward, data = icem.run_icem(icem_gym, seed, env, train, iter=max_steps, render=False)
                        
            episodic_return.append(total_reward)
            
        episodic_return_seeds.append(episodic_return)
        
    episodic_return_seeds = np.array(episodic_return_seeds)

    mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
    std_episodic_return = np.std(episodic_return_seeds, axis=0)
    
    print("max_episodes", max_episodes, "\n")
    print("episodic_return_seeds.shape ", episodic_return_seeds.shape, "\n")
    print("mean_episodic_return ", mean_episodic_return.shape, "\n")
    print("std_episodic_return.shape ", std_episodic_return.shape, "\n")
    
    save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
    print("Saved data \n")
    env.close()
    
    