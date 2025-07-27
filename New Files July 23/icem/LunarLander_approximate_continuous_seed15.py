"""
Adapted for LunarLander-v2 continuous environment
"""
import gymnasium as gym
import numpy as np
import torch
import math
from pytorch_icem import icem
import os


if __name__ == "__main__":
    ENV_NAME = "LunarLander-v3"
    TIMESTEPS = 20  # Increased horizon for lunar lander
    N_SAMPLES = 200  # More samples for more complex environment
    ACTION_LOW = -1.0  # LunarLander continuous action space is [-1, 1]
    ACTION_HIGH = 1.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    # Different sigma for lunar lander (one for each action dimension)
    noise_sigma = torch.tensor([1.0, 1.0], device=d, dtype=dtype)
    lambda_ = 1e-2

    # Network parameters
    H_UNITS = 64  # Increased network size for more complex dynamics
    TRAIN_EPOCH = 200
    BOOT_STRAP_ITER = 200

    # LunarLander state space: 8 dimensions
    # (x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact)
    nx = 8
    # Action space: 2 dimensions (main engine, side engine)
    nu = 2
    
    # Define the network architecture
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, nx)
    ).double().to(device=d)

    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() == 1 or u.dim() == 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        # if u.shape[1] > 1:
        #     u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)
        
        state_residual = network(xu)
        next_state = state + state_residual

        return next_state

    def running_cost(state, action):
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
        new_data[:, -nu:] = torch.clamp(new_data[:, -nu:], ACTION_LOW, ACTION_HIGH)
        new_data = new_data.to(device=d)
        
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)

        # Prepare training data
        XU = dataset
        Y = XU[1:, :nx] - XU[:-1, :nx]  # State residual
        xu = XU[:-1]  # Input is current state + action

        # Train network
        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            Yhat = network(xu)
            loss = (Y - Yhat).pow(2).mean()
            loss.backward()
            optimizer.step()

        # Freeze network
        for param in network.parameters():
            param.requires_grad = False

    # Create environment
    # env = gym.make(ENV_NAME, continuous=True, render_mode="human")
    env = gym.make(ENV_NAME, continuous=True, render_mode="rgb_array")
    state, info = env.reset()

    # Bootstrap network with random actions
    if BOOT_STRAP_ITER:
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            # pre_action_state = env.unwrapped.state.copy()
            pre_action_state = state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(2,))
            state, _, terminated, truncated, info = env.step(action)
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action
            if terminated or truncated:
                state, info = env.reset()
                env.reset()

        train(new_data)

        # Save the initial weights after bootstrapping
        initial_state_dict = network.state_dict()


    # env_seeds = [0, 8, 15]
    seed = 15
    episodic_return_seeds = []
    max_episodes = 300
    method_name = "iCEM"
    prob = "LunarLanderContinuous"
    max_steps = 1000

    env.reset(seed=seed)
    
    # for seed in env_seeds:
    episodic_return = []
    # Reset network to initial pretrained weights
    network.load_state_dict(initial_state_dict)
    
    for episode in range(max_episodes):
        # print(f"Episode {episode + 1}/{max_episodes}")
        env.reset(seed=seed)

        icem_gym = icem.iCEM(dynamics, icem.accumulate_running_cost(running_cost), nx, nu, sigma=noise_sigma,
                     warmup_iters=5, online_iters=5,
                     num_samples=N_SAMPLES, num_elites=10, horizon=TIMESTEPS, device=d, )
    
        total_reward, data = icem.run_icem(icem_gym, seed, env, train, iter=max_steps, render=False) # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)
        episodic_return.append(total_reward)
    
    episodic_return = np.array(episodic_return)
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_seed{seed}_June25.npz")
    np.savez(save_path, episodic_return)
    
    print("Saved data \n")
    env.close()

    