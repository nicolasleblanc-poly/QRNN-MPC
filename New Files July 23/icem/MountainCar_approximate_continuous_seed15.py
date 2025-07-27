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

def run(ctrl: icem.iCEM, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True):
    dataset = torch.zeros((retrain_after_iter, ctrl.nx + ctrl.nu), device=ctrl.device)
    total_reward = 0
    for i in range(iter):
        state = env.unwrapped.state.copy()
        action = ctrl.command(state)
        res = env.step(action.cpu().numpy())
        s, r = res[0], res[1]
        total_reward += r
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :ctrl.nx] = torch.tensor(state, device=ctrl.device)
        dataset[di, ctrl.nx:] = action
    return total_reward, dataset

if __name__ == "__main__":
    ENV_NAME = "MountainCarContinuous-v0"
    TIMESTEPS = 30  # T
    N_SAMPLES = 200  # K
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    d = torch.device("cpu")  # use CPU for now
    dtype = torch.double

    noise_sigma = torch.tensor([1], device=d, dtype=dtype)
    lambda_ = 1e-2

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 2
    nu = 1
    # network output is state residualcc
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
        if u.shape[1] > 1:
            u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)
        
        state_residual = network(xu)
        next_state = state + state_residual

        # Clip position and velocity
        next_state[:, 0] = next_state[:, 0].clamp(-1.2, 0.6)  # position
        next_state[:, 1] = next_state[:, 1].clamp(-0.07, 0.07)  # velocity

        return next_state

    def true_dynamics(state, action):
        position = state[:, 0].view(-1, 1)
        velocity = state[:, 1].view(-1, 1)
        force = torch.clamp(action[:, 0].view(-1, 1), ACTION_LOW, ACTION_HIGH)

        velocity += 0.001 * force - 0.0025 * torch.cos(3 * position)
        velocity = torch.clamp(velocity, -0.07, 0.07)
        position += velocity
        position = torch.clamp(position, -1.2, 0.6)

        # reset velocity if at left wall
        velocity[position == -1.2] = 0

        return torch.cat((position, velocity), dim=1)

    def running_cost(state, action):
        gamma = 0.5
        horizon = 30
        goal = 0.45
        position = state[:, 0]
        velocity = state[:, 1]
        force = action[:, 0]

        positions = state[:, :, 0]  # Assuming states is of shape (batch_size, time_steps, state_dim)
        costs = torch.zeros(positions.shape[0], horizon)  # Initialize cost accumulator
        
        for t in range(horizon):
            position_t = positions[:, t]
            cost_t = (goal - position_t) ** 2
            reverse_discount_factor = gamma ** (horizon - t - 1)
            distance_reward_t = reverse_discount_factor * cost_t
            costs[:, t] = distance_reward_t
        
        return costs

    dataset = None
    # create some true dynamics validation set to compare model against
    Nv = 1000
    statev = torch.cat((
            torch.rand(Nv, 1, dtype=torch.double, device=d) * (0.6 + 1.2) - 1.2,  # position in [-1.2, 0.6]
            torch.rand(Nv, 1, dtype=torch.double, device=d) * 0.14 - 0.07         # velocity in [-0.07, 0.07]
        ), dim=1)

    actionv = torch.rand(Nv, 1, dtype=torch.double, device=d) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW


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
        
        dx = XU[1:, 0] - XU[:-1, 0]
        dv = XU[1:, 1] - XU[:-1, 1]
        Y = torch.cat((dx.view(-1, 1), dv.view(-1, 1)), dim=1)  # x' - x residual
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

        # evaluate network against true dynamics
        yt = true_dynamics(statev, actionv)
        yp = dynamics(statev, actionv)
        dx = yp[:, 0] - yt[:, 0]
        dv = yp[:, 1] - yt[:, 1]
        E = torch.cat((dx.view(-1, 1), dv.view(-1, 1)), dim=1).norm(dim=1)
        
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    state, info = env.reset()

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = env.state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
            state, _, terminated, truncated, info = env.step([action])
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

            done = terminated or truncated
            if done:
                state, info = env.reset()

        train(new_data)
        print("bootstrapping finished \n")
        
        # Save the initial weights after bootstrapping
        initial_state_dict = network.state_dict()

    # env_seeds = [0, 8, 15]
    seed = 15
    episodic_return_seeds = []
    max_episodes = 300
    method_name = "iCEM"
    prob = "MountainCarContinuous"
    max_steps = 1000
    
    episodic_return = []
    # Reset network to initial pretrained weights
    network.load_state_dict(initial_state_dict)
    
    for episode in range(max_episodes):
        env.reset(seed=seed)

        icem_gym = icem.iCEM(dynamics, icem.accumulate_running_cost(running_cost), nx, nu, sigma=noise_sigma,
                     warmup_iters=5, online_iters=5,
                     num_samples=N_SAMPLES, num_elites=10, horizon=TIMESTEPS, device=d, )
    
        total_reward, data = icem.run_icem(icem_gym, seed, env, train, iter=max_steps, render=False, prob=prob) # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)
        print("total_reward ", total_reward, "\n")
        episodic_return.append(total_reward)
    
    episodic_return = np.array(episodic_return)
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_seed{seed}_June27.npz")
    np.savez(save_path, episodic_return)
    
    print("Saved data \n")
    env.close()
