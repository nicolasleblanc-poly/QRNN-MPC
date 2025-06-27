"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""
import gymnasium as gym
# import gym
import numpy as np
import torch
import logging
import math
from pytorch_mppi_folder import mppi_modified as mppi
import os
import panda_gym
# from gym import logger as gym_log

# gym_log.set_level(gym_log.INFO)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,
#                     format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
#                     datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "PandaReachDense-v3"
    TIMESTEPS = 15  # T
    N_SAMPLES = 50  # K
    # ACTION_LOW = -1.0
    # ACTION_HIGH = 1.0
    ACTION_LOW = [-1.0, -1.0, -1.0]
    ACTION_HIGH = [1.0, 1.0, 1.0]

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    # noise_sigma = torch.tensor(1, device=d, dtype=dtype)
    noise_sigma = torch.eye(3, device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1e-2
    # lambda_ = 1.

    import random

    # randseed = 24
    # if randseed is None:
    #     randseed = random.randint(0, 1000000)
    # random.seed(randseed)
    # np.random.seed(randseed)
    # torch.manual_seed(randseed)
    # logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    global goal_state 

    nx = 6
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

        # # Clip position and velocity
        # next_state[:, 0] = next_state[:, 0].clamp(-1.2, 0.6)  # position
        # next_state[:, 1] = next_state[:, 1].clamp(-0.07, 0.07)  # velocity

        return next_state

    # def true_dynamics(state, action):
    #     position = state[:, 0].view(-1, 1)
    #     velocity = state[:, 1].view(-1, 1)
    #     force = torch.clamp(action[:, 0].view(-1, 1), ACTION_LOW, ACTION_HIGH)

    #     velocity += 0.001 * force - 0.0025 * torch.cos(3 * position)
    #     velocity = torch.clamp(velocity, -0.07, 0.07)
    #     position += velocity
    #     position = torch.clamp(position, -1.2, 0.6)

    #     # reset velocity if at left wall
    #     velocity[position == -1.2] = 0

    #     return torch.cat((position, velocity), dim=1)

    def running_cost(state, action): # goal_state
        # print("goal_state ", goal_state, "\n")
        # goal_state = np.array([0.04108851, -0.06906398,  0.01229206]) # seed = 0
        # goal_state = np.array([-0.05190832,  0.14618306,  0.09561325]) # seed = 8
        goal_state = np.array([0.05782301, 0.09474514, 0.10332203]) # seed = 15
        goal_state = torch.tensor(goal_state, dtype=torch.float32, device=state.device).reshape(1, 3)
        cost = torch.norm(state[:, :3] - goal_state, dim=1)

        return cost

    # def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

    #     # Get the folder where this script is located
    #     origin_folder = os.path.dirname(os.path.abspath(__file__))
    #     # Construct full path to save
    #     save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_June27.npz")

    #     np.savez(
    #     save_path,
    #     # f"{prob}_{method_name}_results.npz",
    #     episode_rewards=episodic_rep_returns,
    #     mean_rewards=mean_episodic_returns,
    #     std_rewards=std_episodic_returns
    #     )

    dataset = None
    # # create some true dynamics validation set to compare model against
    # Nv = 1000
    # statev = torch.cat((
    #         torch.rand(Nv, 1, dtype=torch.double, device=d) * (0.6 + 1.2) - 1.2,  # position in [-1.2, 0.6]
    #         torch.rand(Nv, 1, dtype=torch.double, device=d) * 0.14 - 0.07         # velocity in [-0.07, 0.07]
    #     ), dim=1)

    # actionv = torch.rand(Nv, 1, dtype=torch.double, device=d) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW


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
        # XU = dataset
        # dtheta = angular_diff_batch(XU[1:, 0], XU[:-1, 0])
        # dtheta_dt = XU[1:, 1] - XU[:-1, 1]
        # Y = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)  # x' - x residual
        # xu = XU[:-1]  # make same size as Y
        # xu = torch.cat((torch.sin(xu[:, 0]).view(-1, 1), torch.cos(xu[:, 0]).view(-1, 1), xu[:, 1:]), dim=1)

        # Y = dataset[1:, :nx] - dataset[:-1, :nx]
        # xu = dataset[:-1, :]
        
        # train on the whole dataset (assume small enough we can train on all together)
        XU = dataset
        # dx = XU[1:, 0], XU[:-1, 0]
        # dv = XU[1:, 1] - XU[:-1, 1]
        # Y = torch.cat((dx.view(-1, 1), dv.view(-1, 1)), dim=1)  # x' - x residual
        # xu = XU[:-1]  # make same size as Y
        # xu = torch.cat(dx.view(-1, 1), dv.view(-1, 1), xu[:, 1:], dim=1)
        
        Y = XU[1:, :nx] - XU[:-1, :nx]
        xu = XU[:-1, :]  # make same size as Y
        
        # dx = XU[1:, :nx] - XU[:-1, 0]
        # dv = XU[1:, 1] - XU[:-1, 1]
        # Y = torch.cat((dx.view(-1, 1), dv.view(-1, 1)), dim=1)  # x' - x residual
        # xu = XU[:-1]  # make same size as Y
        # xu = torch.cat((xu, ), dim=1)


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
            # logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False

        # # evaluate network against true dynamics
        # yt = true_dynamics(statev, actionv)
        # yp = dynamics(statev, actionv)
        # # print("yt.shape ", yt.shape, "\n")
        # # print("yp.shape ", yp.shape, "\n")
        # print(yp[:, 0].shape, yt[:, 0].shape, "\n")
        # dx = yp[:, 0] - yt[:, 0]
        # dv = yp[:, 1] - yt[:, 1]
        # # print("dx.shape ", type(dx[0]), "\n")
        # # print("dv.shape ", dv, "\n")
        # E = torch.cat((dx.view(-1, 1), dv.view(-1, 1)), dim=1).norm(dim=1)
        # logger.info("Error with true dynamics dx %f dv %f norm %f", dx.abs().mean(),
        #             dv.abs().mean(), E.mean())
        # logger.debug("Start next collection sequence")
        
        


    # downward_start = True
    env = gym.make(ENV_NAME) # , render_mode="human"  # bypass the default TimeLimit wrapper
    observation, info = env.reset()
    # print("state ", observation, "\n")
    state  = observation['observation']
    goal_state_boostrap = observation['desired_goal']
    
    # state, info = env.reset()
    # print("state", state)
    # print("env.state", env.state)
    # if downward_start:
    #     # env.state = env.unwrapped.state = [np.pi, 1]
    #     env.state = env.unwrapped.state = [0, 0]

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        # logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = state # env.state
            action = np.random.uniform(low=ACTION_LOW[0], high=ACTION_HIGH[0], size=nu)
            state, reward, terminated, truncated, info = env.step(action)
            state = state['observation']
            # env.render()
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

            done = terminated or truncated
            if done:
                state, info = env.reset()  # reset environment if done
                state  = observation['observation']
                goal_state_boostrap = observation['desired_goal']

        train(new_data)
        # logger.info("bootstrapping finished")
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
    prob = "PandaReachDense"
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
        # print("goal_state ", goal_state, "\n")

        # N_SAMPLES = 200 is the number of steps per episode
        # mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
        #                     lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
        #                     u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
        total_reward, data = mppi.run_mppi(mppi_gym, seed, env, train, iter=max_steps, render=False, prob=prob) #  # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)
        episodic_return.append(total_reward)
        
        # logger.info("Total reward %f", total_reward)

    # episodic_return_seeds.append(episodic_return)
    
    episodic_return = np.array(episodic_return)
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_seed{seed}_June27.npz")
    np.savez(save_path, episodic_return)
    # episodic_return_seeds = np.array(episodic_return_seeds)

    # mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
    # std_episodic_return = np.std(episodic_return_seeds, axis=0)
    
    # print("max_episodes", max_episodes, "\n")
    # print("episodic_return_seeds.shape ", episodic_return_seeds.shape, "\n")
    # print("mean_episodic_return ", mean_episodic_return.shape, "\n")
    # print("std_episodic_return.shape ", std_episodic_return.shape, "\n")
    
    # save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
    print("Saved data \n")
    env.close()
    
    """
    for seed in env_seeds:
        episodic_return = []
        # Reset network to initial pretrained weights
        network.load_state_dict(initial_state_dict)
        
        for episode in range(max_episodes):
            state, info = env.reset(seed=seed)
            
            goal_state = state['desired_goal']
            # print("goal_state ", goal_state, "\n")

            # N_SAMPLES = 200 is the number of steps per episode
            mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                                lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                                u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
            total_reward, data = mppi.run_mppi(mppi_gym, seed, env, train, iter=max_steps, render=False) # , prob=prob # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)
            episodic_return.append(total_reward)
            
            # logger.info("Total reward %f", total_reward)

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
    """
    
    # # state, info = env.reset()
    # # if downward_start:
    # #     env.state = env.unwrapped.state = [np.pi, 1]
    # seed = 0
    # max_steps = 50
    # mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
    #                      lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
    #                      u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
    # # Changed source code to reset env in the function below
    # total_reward, data = mppi.run_mppi(mppi_gym, seed, env, train, iter=max_steps, render=True)
    # # logger.info("Total reward %f", total_reward)
    # print("Total reward %f" % total_reward, "\n")
    # env.close()