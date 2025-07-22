"""
Modified version that works with cartesian coordinates (x, y, omega) instead of angle theta
"""
import gym
import numpy as np
import torch
import logging
import math
from pytorch_cem import cem
from gym import wrappers, logger as gym_log

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 15  # T
    N_ELITES = 10
    N_SAMPLES = 100  # K
    SAMPLE_ITER = 3  # M
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    import random

    randseed = 24
    if randseed is None:
        randseed = random.randint(0, 1000000)
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 3  # Now using x, y, omega
    nu = 1
    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),  # Removed the +1 since we're not using sin/cos anymore
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
        
        # Normalize x,y to keep them on the unit circle
        xy = next_state[:, :2]
        xy_norm = xy / xy.norm(dim=1).view(-1, 1)
        next_state[:, :2] = xy_norm
        
        # Clamp angular velocity
        next_state[:, 2] = torch.clamp(next_state[:, 2], -8, 8)
        return next_state


    def true_dynamics(state, perturbed_action):
        # Convert cartesian to angle for true dynamics
        x = state[:, 0]
        y = state[:, 1]
        omega = state[:, 2]
        
        th = torch.atan2(y, x)
        thdot = omega

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        # Convert back to cartesian
        newx = torch.cos(newth)
        newy = torch.sin(newth)
        
        state = torch.cat((newx.view(-1, 1), newy.view(-1, 1), newthdot.view(-1, 1)), dim=1)
        return state


    def running_cost(state, action):
        # Convert cartesian to angle for cost calculation
        x = state[:, 0]
        y = state[:, 1]
        omega = state[:, 2]
        
        theta = torch.atan2(y, x)
        theta_dt = omega
        action = action[:, 0]
        
        # Normalize theta to [-pi, pi]
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
        
        cost = theta ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action ** 2
        return cost


    dataset = None
    # create some true dynamics validation set to compare model against
    Nv = 1000
    # Generate random states on the unit circle with random angular velocity
    angles = (torch.rand(Nv, 1, dtype=torch.double, device=d) - 0.5) * 2 * math.pi
    x = torch.cos(angles)
    y = torch.sin(angles)
    omega = (torch.rand(Nv, 1, dtype=torch.double, device=d) - 0.5) * 16
    statev = torch.cat((x, y, omega), dim=1)
    actionv = (torch.rand(Nv, 1, dtype=torch.double, device=d) - 0.5) * (ACTION_HIGH - ACTION_LOW)


    def train(new_data):
        global dataset
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        new_data = new_data.to(device=d)
        
        # Normalize x,y to unit circle
        xy = new_data[:, :2]
        xy_norm = xy / xy.norm(dim=1).view(-1, 1)
        new_data[:, :2] = xy_norm
        
        # append data to whole dataset
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)

        # train on the whole dataset (assume small enough we can train on all together)
        XU = dataset
        dx = XU[1:, 0] - XU[:-1, 0]
        dy = XU[1:, 1] - XU[:-1, 1]
        domega = XU[1:, 2] - XU[:-1, 2]
        Y = torch.cat((dx.view(-1, 1), dy.view(-1, 1), domega.view(-1, 1)), dim=1)  # x' - x residual
        xu = XU[:-1]  # make same size as Y

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
            logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False

        # evaluate network against true dynamics
        yt = true_dynamics(statev, actionv)
        yp = dynamics(statev, actionv)
        
        # Convert to angles for error calculation
        # th_t = torch.atan2(yt[:, 1], yt[:, 0])
        # th_p = torch.atan2(yp[:, 1], yp[:, 0])
        # dtheta = ((th_p - th_t + math.pi) % (2 * math.pi)) - math.pi
        # domega = yp[:, 2] - yt[:, 2]
        
        # E = torch.cat((dtheta.view(-1, 1), domega.view(-1, 1)), dim=1).norm(dim=1)
        # logger.info("Error with true dynamics theta %f omega %f norm %f", dtheta.abs().mean(),
        #             domega.abs().mean(), E.mean())
        # logger.debug("Start next collection sequence")


    downward_start = True
    env = gym.make(ENV_NAME, render_mode='human').env  # bypass the default TimeLimit wrapper
    
    # Modify reset to return cartesian coordinates
    # def modified_reset():
    #     state = env.reset()
    #     if downward_start:
    #         env.state = [np.pi, 1]  # Start pointing down
    #     # Convert to cartesian
    #     theta = env.state[0]
    #     omega = env.state[1]
    #     x = np.cos(theta)
    #     y = np.sin(theta)
    #     return np.array([x, y, omega])
    
    # env.reset = modified_reset
    
    # Modify step to return cartesian coordinates
    # original_step = env.step
    # def modified_step(action):
    #     obs, reward, done, info = original_step(action)
    #     # Convert to cartesian
    #     theta = env.state[0]
    #     omega = env.state[1]
    #     x = np.cos(theta)
    #     y = np.sin(theta)
    #     return np.array([x, y, omega]), reward, done, info
    # env.step = modified_step
    
    state, info = env.reset()
    
    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = state #np.array([np.cos(env.state[0]), np.sin(env.state[0]), env.state[1]])
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
            state, _, _, _, _ = env.step([action])
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action
            
        train(new_data)
        logger.info("bootstrapping finished")

    # env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    env.reset()

    ctrl = cem.CEM(dynamics, running_cost, nx, nu, num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER,
                      horizon=TIMESTEPS, device=d, num_elite=N_ELITES,
                      u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d), init_cov_diag=1)
    total_reward, data = cem.run_cem(ctrl, env, train)
    logger.info("Total reward %f", total_reward)