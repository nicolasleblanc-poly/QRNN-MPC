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
# from gym import logger as gym_log

# gym_log.set_level(gym_log.INFO)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,
#                     format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
#                     datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "LunarLanderContinuous-v3"
    TIMESTEPS = 15  # T
    N_SAMPLES = 1000  # K
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    # noise_sigma = torch.tensor(1, device=d, dtype=dtype)
    noise_sigma = torch.eye(2, device=d, dtype=dtype)
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
        # if u.shape[1] > 1:
        #     u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)
        
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

    # def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

    #     # Get the folder where this script is located
    #     origin_folder = os.path.dirname(os.path.abspath(__file__))
    #     # Construct full path to save
    #     save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results.npz")

    #     np.savez(
    #     save_path,
    #     f"{prob}_{method_name}_results.npz",
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
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
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
        
        # dx = XU[1:, 0] - XU[:-1, 0]
        # dv = XU[1:, 1] - XU[:-1, 1]
        # Y = torch.cat((dx.view(-1, 1), dv.view(-1, 1)), dim=1)  # x' - x residual
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
            # logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False

        # # evaluate network against true dynamics
        # yt = true_dynamics(statev, actionv)
        # yp = dynamics(statev, actionv)
        # # print("yt.shape ", yt.shape, "\n")
        # # print("yp.shape ", yp.shape, "\n")
        # # print(yp[:, 0].shape, yt[:, 0].shape, "\n")
        # dx = yp[:, 0] - yt[:, 0]
        # dv = yp[:, 1] - yt[:, 1]
        # # print("dx.shape ", type(dx[0]), "\n")
        # # print("dv.shape ", dv, "\n")
        # E = torch.cat((dx.view(-1, 1), dv.view(-1, 1)), dim=1).norm(dim=1)
        # logger.info("Error with true dynamics dx %f dv %f norm %f", dx.abs().mean(),
        #             dv.abs().mean(), E.mean())
        # logger.debug("Start next collection sequence")
        
    # from gym import Wrapper
    # class LunarLanderStateWrapper(Wrapper):
    #     """Wrapper that adds state access and continuous angle representation"""
    #     def __init__(self, env):
    #         super().__init__(env)
    #         self.original_observation_space = env.observation_space
    #         # Modify observation space to include sin/cos of angle
    #         low = env.observation_space.low
    #         high = env.observation_space.high
    #         # low = np.append(env.observation_space.low[:4], [-1.0, -1.0, env.observation_space.low[5:]])
    #         # high = np.append(env.observation_space.high[:4], [1.0, 1.0, env.observation_space.high[5:]])
    #         self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            
    #     @property
    #     def state(self):
    #         """Returns the full physics state as a numpy array"""
    #         return np.array([
    #             self.env.lander.position.x,          # x position
    #             self.env.lander.position.y,          # y position
    #             self.env.lander.linearVelocity.x,    # x velocity
    #             self.env.lander.linearVelocity.y,    # y velocity
    #             self.env.lander.angle,               # angle (radians)
    #             self.env.lander.angularVelocity,    # angular velocity
    #             float(self.env.legs[0].ground_contact),  # left leg contact
    #             float(self.env.legs[1].ground_contact)   # right leg contact
    #         ])
        
    #     @state.setter
    #     def state(self, state_array):
    #         """Sets the physics state from a numpy array"""
    #         self.env.lander.position = (state_array[0], state_array[1])
    #         self.env.lander.linearVelocity = (state_array[2], state_array[3])
    #         self.env.lander.angle = state_array[4]
    #         self.env.lander.angularVelocity = state_array[5]
        
    #     def reset(self, **kwargs):
    #         obs, info = self.env.reset(**kwargs)
    #         return obs, info
        
    #     def step(self, action):
    #         obs, reward, terminated, truncated, info = self.env.step(action)
    #         info['true_state'] = self.state  # Store full state in info
    #         return obs, reward, terminated, truncated, info

    # def make_lunar_lander_with_state():
    #     """Creates a LunarLander environment with state access"""
    #     env = gym.make('LunarLander-v3', continuous=True)
    #     wrapped_env = LunarLanderStateWrapper(env)
    #     return wrapped_env


    # downward_start = True
    env = gym.make(ENV_NAME) # , render_mode="human"  # bypass the default TimeLimit wrapper
    # env = make_lunar_lander_with_state()
    state, info = env.reset()
    # print("env.state", env.state)
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
            # pre_action_state = env.state
            pre_action_state = state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=nu)
            state, _, terminated, truncated, _ = env.step(action) # env.step([action])
            # env.step(action)
            # truncated
            # env.render()
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action
            
            if terminated:
                state, info = env.reset()
                env.reset()

        train(new_data)
        # logger.info("bootstrapping finished")
        print("bootstrapping finished \n")

        # Save the initial weights after bootstrapping
        initial_state_dict = network.state_dict()

    # env_seeds = [0, 8, 15]
    seed = 15
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

        # mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
        #                     lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
        #                     u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
        total_reward, data = mppi.run_mppi(mppi_gym, seed, env, train, iter=max_steps, render=False) # mppi.run_mppi(mppi_gym, seed, env, train, iter=max_episodes, render=False)
        episodic_return.append(total_reward)
        
        # logger.info("Total reward %f", total_reward)

    # episodic_return_seeds.append(episodic_return)
        
    # episodic_return_seeds = np.array(episodic_return_seeds)

    # mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
    # std_episodic_return = np.std(episodic_return_seeds, axis=0)
    
    episodic_return = np.array(episodic_return)
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results_seed{seed}_June27.npz")
    np.savez(save_path, episodic_return)
    
    # print("max_episodes", max_episodes, "\n")
    # print("episodic_return_seeds.shape ", episodic_return_seeds.shape, "\n")
    # print("mean_episodic_return ", mean_episodic_return.shape, "\n")
    # print("std_episodic_return.shape ", std_episodic_return.shape, "\n")
    
    # save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
    print("Saved data \n")
    env.close()

    # # env.reset()
    # # if downward_start:
    # #     env.state = env.unwrapped.state = [np.pi, 1]
    # seed = 0
    # max_steps = 1000
    # mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
    #                      lambda_=lambda_, device=d, u_min=torch.tensor([ACTION_LOW, ACTION_LOW], dtype=torch.double, device=d),
    #                      u_max=torch.tensor([ACTION_HIGH, ACTION_HIGH], dtype=torch.double, device=d))
    # total_reward, data = mppi.run_mppi(mppi_gym, seed, env, train, iter=max_steps, render=True) # mppi.run_mppi(mppi_gym, seed, env, train)
    # # logger.info("Total reward %f", total_reward)
    # print("Total reward %f", total_reward)
    # env.close()