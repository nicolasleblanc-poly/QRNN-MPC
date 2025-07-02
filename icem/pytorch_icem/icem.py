import torch
import colorednoise
from arm_pytorch_utilities import handle_batch_input
import numpy as np
import logging

logger = logging.getLogger(__name__)

def accumulate_running_cost(running_cost, terminal_state_weight=10.0):
    def _accumulate_running_cost(x, u):
        cost = running_cost(x, u)
        terminal_cost = cost[:, -1]
        cost = torch.sum(cost, dim=-1)
        cost += terminal_cost * (terminal_state_weight - 1)
        # cost[:, -1] += (terminal_state_weight - 1) * cost[:, -1]
        return cost

    return _accumulate_running_cost

class iCEM:

    def __init__(self, dynamics, trajectory_cost, nx, nu, sigma=None, num_samples=100, num_elites=10, horizon=15,
                 elites_keep_fraction=0.5,
                 alpha=0.05,
                 noise_beta=3,
                 warmup_iters=100, online_iters=100,
                 includes_x0=False,
                 fixed_H=True,
                 device="cpu"):

        self.dynamics = dynamics
        self.trajectory_cost = trajectory_cost

        self.nx = nx
        self.nu = nu
        self.H = horizon
        self.fixed_H = fixed_H
        self.N = num_samples
        self.device = device

        if sigma is None:
            sigma = torch.ones(self.nu, device=self.device).float()
        elif isinstance(sigma, float):
            sigma = torch.ones(self.nu, device=self.device).float() * sigma

        # if self.nu == 1:
        #     sigma = sigma.view(-1, 1)
        # print(f"Sigma shape: {sigma.shape}, nu: {self.nu}")

        if isinstance(sigma, torch.Tensor):
            if sigma.shape != (self.nu,):
                raise ValueError(f"If sigma is a tensor, it must have shape (nu,). Got {sigma.shape} for nu={nu}")
        elif not np.isscalar(sigma):
            raise ValueError(f"Sigma must be a scalar or a tensor of shape (nu,). Got {type(sigma)}")

        # if len(sigma.shape) != nu:
        #     raise ValueError(f"Sigma must be either a scalar or a vector of length nu {nu}")
        self.sigma = sigma
        self.dtype = self.sigma.dtype

        self.warmup_iters = warmup_iters
        self.online_iters = online_iters
        self.includes_x0 = includes_x0
        self.noise_beta = noise_beta
        self.K = num_elites
        self.alpha = alpha
        self.keep_fraction = elites_keep_fraction

        self.sigma = torch.tensor(self.sigma).to(device=self.device)
        self.std = self.sigma.clone()

        # initialise mean and std of actions
        self.mean = torch.zeros(self.H, self.nu, device=self.device)
        self.kept_elites = None
        self.warmed_up = False

    def reset(self):
        self.warmed_up = False
        self.mean = torch.zeros(self.H, self.nu, device=self.device)
        self.std = self.sigma.clone()
        self.kept_elites = None

    def sample_action_sequences(self, state, N):
        # colored noise
        if self.noise_beta > 0:
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
            samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(N, self.nu,
                                                                                self.H)).transpose(
                [0, 2, 1])
            samples = torch.from_numpy(samples).to(device=self.device, dtype=self.dtype)
        else:
            samples = torch.randn(N, self.H, self.nu, device=self.device, dtype=self.dtype)

        # print("self.mean shape: ", self.mean.shape, "self.std shape: ", self.std.shape, "\n")
        # print("samples shape: ", samples.shape, "\n")
        U = self.mean + self.std * samples
        return U

    def update_distribution(self, elites):
        """
        param: elites - K x H x du number of best K control sequences by cost
        """

        # fit around mean of elites
        new_mean = elites.mean(dim=0)
        new_std = elites.std(dim=0)

        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean  # [h,d]
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std

    @handle_batch_input(n=3)
    def _cost(self, x, u):
        # print("x.shape = ", x.shape, "u.shape = ", u.shape, "\n")
        return self.trajectory_cost(x, u)
        # xu = torch.cat((x, u), dim=-1)
        # return self.problem.objective(xu)

    @handle_batch_input(n=2)
    def _dynamics(self, x, u):
        return self.dynamics(x, u)

    def _rollout_dynamics(self, x0, u):
        N, H, du = u.shape
        assert H == self.H
        assert du == self.nu

        x = [x0.reshape(1, self.nx).repeat(N, 1)]
        for t in range(self.H):
            x.append(self._dynamics(x[-1], u[:, t]))

        if self.includes_x0:
            return torch.stack(x[:-1], dim=1)
        return torch.stack(x[1:], dim=1)

    def command(self, state, shift_nominal_trajectory=True, return_full_trajectories=False, **kwargs):
        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self.device, dtype=self.dtype)
        x = state

        if self.fixed_H or (not self.warmed_up):
            new_T = None
        else:
            # new_T = self.problem.H - 1
            new_T = self.H - 1
            self.H = new_T

        # self.problem.update(x, T=new_T, **kwargs)

        if self.warmed_up:
            iterations = self.online_iters
        else:
            iterations = self.warmup_iters
            self.warmed_up = True

        # Shift the keep elites

        for i in range(iterations):
            if self.kept_elites is None:
                # Sample actions
                U = self.sample_action_sequences(x, self.N)
            else:
                # reuse the elites from the previous iteration
                U = self.sample_action_sequences(x, self.N - len(self.kept_elites))
                U = torch.cat((U, self.kept_elites), dim=0)

            # evaluate costs and update the distribution!
            pred_x = self._rollout_dynamics(x, U)
            costs = self._cost(pred_x, U)
            sorted, indices = torch.sort(costs)
            elites = U[indices[:self.K]]
            self.update_distribution(elites)
            # save kept elites fraction
            self.kept_elites = U[indices[:int(self.K * self.keep_fraction)]]

        # Return best sampled trajectory
        out_U = elites[0].clone()
        if shift_nominal_trajectory:
            self.shift()

        if return_full_trajectories:
            out_X = self._rollout_dynamics(x, out_U.reshape(1, self.H, self.nu)).reshape(self.H, self.nx)
            out_trajectory = torch.cat((out_X, out_U), dim=-1)

            # Top N // 20 sampled trajectories - for visualization
            sampled_trajectories = torch.cat((pred_x, U), dim=-1)
            # only return best 10% trajectories for visualization
            sampled_trajectories = sampled_trajectories[torch.argsort(costs, descending=False)][:64]
            return out_trajectory, sampled_trajectories
        else:
            return out_U[0]

    def shift(self):
        # roll distribution
        self.mean = torch.roll(self.mean, -1, dims=0)
        self.mean[-1] = torch.zeros(self.nu, device=self.device)
        self.std = self.sigma.clone()
        # Also shift the elites
        if self.kept_elites is not None:
            self.kept_elites = torch.roll(self.kept_elites, -1, dims=1)
            self.kept_elites[:, -1] = self.sigma * torch.randn(len(self.kept_elites), self.nu, device=self.device)


def run_icem(ctrl: iCEM, seed, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True, prob = None):
    dataset = torch.zeros((retrain_after_iter, ctrl.nx + ctrl.nu), device=ctrl.device)
    total_reward = 0
    state, info = env.reset(seed=seed)
    if prob == "PandaReach" or prob == "PandaReachDense":
        # # global goal_state
        # goal_state = state['desired_goal']
        state  = state['observation']

    for i in range(iter):
        # state = env.unwrapped.state.copy()
        if prob == "Pendulum" or prob == "MountainCarContinuous":
            state = env.unwrapped.state.copy()
        # command_start = time.perf_counter()
        # print("state: ", state, "\n")
        action = ctrl.command(state)
        # elapsed = time.perf_counter() - command_start
        # res = env.step(action.cpu().numpy())
        # s, r = res[0], res[1]
        # action
        
        # action = action
        if prob != "PandaReach" and prob != "PandaReachDense":
            # action = torch.clip(action, torch.tensor(env.action_space.low, dtype=torch.float32, device='cuda'), torch.tensor(env.action_space.high, dtype=torch.float32, device='cuda'))
            action = torch.clip(action, torch.tensor(env.action_space.low, dtype=torch.float32, device='cpu'), torch.tensor(env.action_space.high, dtype=torch.float32, device='cpu'))
        
        next_state, r, terminated, truncated, info = env.step(action.cpu().numpy())

        if prob == "Pendulum" or prob == "MountainCarContinuous":
            next_state = env.unwrapped.state.copy()
        elif prob == "PandaReach" or prob == "PandaReachDense":
            next_state  = next_state['observation']

        total_reward += r
        # logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

        done = terminated or truncated
        # print("truncated ", truncated, "\n")
        # print("terminated ", terminated, "\n")
        if done:
            break

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :ctrl.nx] = torch.tensor(state, device=ctrl.device)
        dataset[di, ctrl.nx:] = action
        
        # print("state ", state, "action ", action, "next_state ", next_state, "\n")
        # print("r ", r, "total_reward ", total_reward, "i ", i, "\n")
        # print("info ", info, "\n")
        state = next_state
    return total_reward, dataset
