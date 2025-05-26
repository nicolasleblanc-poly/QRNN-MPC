# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

# from . import termination_fns
import PETS.termination_fns_old as termination_fns_old

######################## My own cost functions ########################

def lunarlander_continuous(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [x, y, vx, vy, theta, vtheta, leg1_contact, leg2_contact]
    # assert len(next_obs.shape) == len(act.shape) == 2

    pos = next_obs[:, :2]
    vel = next_obs[:, 2:4]
    angle = next_obs[:, 4]
    ang_vel = next_obs[:, 5]
    legs = next_obs[:, 6:8]

    reward = (
        -10 * torch.norm(pos, dim=1)  # distance to center
        - torch.norm(vel, dim=1)  # penalize velocity
        - torch.abs(angle)       # penalize angle
        - torch.abs(ang_vel)      # penalize angular velocity
        + 10 * legs.sum(dim=1)         # bonus for contact with legs
        + 0.1 * (act**2).sum(dim=1)  # penalize action
    )
    return reward.view(-1, 1)

def mountaincar_continuous(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [position, velocity]
    # assert len(next_obs.shape) == len(act.shape) == 2

    # pos = next_obs[:, 0]
    # reward = -0.1 * (act**2).sum(dim=1)  # control cost
    # reward += (pos >= 0.45).float() * 100.0  # bonus for reaching goal
    # return reward.view(-1, 1)
    act = act.reshape(-1)
    goal_position = 0.45  # Position goal in Mountain Car
    cost = (next_obs[:, 0]-goal_position)**2
    cost += 0.05*(act)**2
    
    return -cost.view(-1, 1)

def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    
    cart_position = next_obs[:, 0]
    pole_angle = next_obs[:, 1]
    cart_velocity = next_obs[:, 2]
    # force = action[:, 0]
    cost = pole_angle**2 + 0.1 * cart_position**2 + 0.1 * cart_velocity**2

    return -cost.view(-1, 1)

def pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [cos(theta), sin(theta), theta_dot]
    assert len(next_obs.shape) == len(act.shape) == 2
    
    # cos_theta = next_obs[:, 0]
    # sin_theta = next_obs[:, 1]
    # theta_dot = next_obs[:, 2]

    # theta = torch.atan2(sin_theta, cos_theta)
    # cost = - (theta**2 + 0.1 * theta_dot**2 + 0.001 * (act**2).sum(dim=1))
    
    x = next_obs[:, 0]
    y = next_obs[:, 1]
    omega = next_obs[:, 2]

    cost = (1-x)**2 + y**2 + 0.1 * omega**2 + 0.01 * act.reshape(len(act)) ** 2
    
    return -cost.view(-1, 1)

def reacher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), vel1, vel2, fingertip_x-target_x, fingertip_y-target_y]
    assert len(next_obs.shape) == len(act.shape) == 2

    dist = next_obs[:, 6:8].pow(2).sum(dim=1).sqrt()

    reward_dist = -dist
    reward_ctrl = -0.1 * (act**2).sum(dim=1)

    return (reward_dist + reward_ctrl).view(-1, 1)

def panda_reach(act: torch.Tensor, next_obs: torch.Tensor, goal_obs) -> torch.Tensor:
    # next_obs: contains positions of end effector and object
    assert len(next_obs.shape) == len(act.shape) == 2
    
    goal_state = torch.tensor(goal_state, dtype=torch.float32, device=next_obs.device).reshape(1, 3)

    cost = torch.norm(next_obs[:, :3] - goal_state, dim=1)
    
    return -cost.view(-1, 1)

############### Already implementated termination functions ################
def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns_old.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
    act_cost = -0.01 * torch.sum(act**2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


# def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
#     assert len(next_obs.shape) == len(act.shape) == 2

#     return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)



def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)

def panda_push(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: contains positions of end effector and object
    assert len(next_obs.shape) == len(act.shape) == 2

    end_effector = next_obs[:, :3]
    obj_pos = next_obs[:, 3:6]
    goal_pos = torch.tensor([0.6, 0.0, 0.0], device=next_obs.device)

    ee_obj_dist = (end_effector - obj_pos).norm(dim=1)
    obj_goal_dist = (obj_pos - goal_pos).norm(dim=1)

    reward = -1.0 * ee_obj_dist - 2.0 * obj_goal_dist - 0.1 * (act**2).sum(dim=1)
    return reward.view(-1, 1)



