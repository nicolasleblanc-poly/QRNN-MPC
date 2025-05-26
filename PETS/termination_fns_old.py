# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch

# TODO remove act from all of these, it's not needed

############### Already implementated termination functions ################
def hopper(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.isfinite(next_obs).all(-1)
        * (next_obs[:, 1:].abs() < 100).all(-1)
        * (height > 0.7)
        * (angle.abs() < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    not_done = torch.isfinite(next_obs).all(-1) * (next_obs[:, 1].abs() <= 0.2)
    done = ~not_done

    done = done[:, None]

    return done


def no_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs)).bool().to(next_obs.device)
    done = done[:, None]
    return done


def walker2d(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def ant(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    x = next_obs[:, 0]
    not_done = torch.isfinite(next_obs).all(-1) * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    done = done[:, None]
    return done


def humanoid(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    z = next_obs[:, 0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:, None]
    return done

######################## My own functions ########################
def lunarlander_continuous(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [x, y, vx, vy, theta, vtheta, leg1_contact, leg2_contact]
    assert len(next_obs.shape) == 2

    x, y = next_obs[:, 0], next_obs[:, 1]
    theta = next_obs[:, 4]

    out_of_bounds = (x.abs() > 1.0) + (y < 0.0)
    angle_limit = theta.abs() > math.pi / 2

    done = out_of_bounds + angle_limit
    return done[:, None]


def mountaincar_continuous(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [position, velocity]
    assert len(next_obs.shape) == 2

    pos = next_obs[:, 0]
    done = (pos >= 0.45)  # Episode ends when car reaches the goal
    return done[:, None]


def reacher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: includes fingertip and target positions at the end
    assert len(next_obs.shape) == 2

    # This is a continuous task with no early termination in standard Reacher
    done = torch.zeros(len(next_obs), dtype=torch.bool, device=next_obs.device)
    return done[:, None]


def pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [cos(theta), sin(theta), theta_dot]
    assert len(next_obs.shape) == 2

    # Pendulum typically runs indefinitely
    done = torch.zeros(len(next_obs), dtype=torch.bool, device=next_obs.device)
    return done[:, None]


def panda_push(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # next_obs: [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z]
    assert len(next_obs.shape) == 2

    goal = torch.tensor([0.6, 0.0, 0.0], device=next_obs.device)
    obj_pos = next_obs[:, 3:6]

    dist = torch.norm(obj_pos - goal, dim=1)
    done = (dist < 0.05)  # consider successful push if within 5cm

    return done[:, None]
