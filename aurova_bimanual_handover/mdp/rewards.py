# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from .dq import dq_distance, q_mul

from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat
from math import pi


# Compute error as a dual quaternion distance
def dual_quaternion_error(pose1: torch.Tensor, pose2: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pose1 / pose2 - torch.Tensor(N, 7): poses to compute the distance in translation(3) + rotation in quaternions(4).
        - device - str: Device into which the environment is stored.

    Out:
        - distance - torch.Tensor(N, 3): distance[:, 0] in dual quaternions, translation module[:, 1] and rotation module[:, 2].
    '''
    # Convert position and orientation (pose) to dual quaternion
    pose1_dq = pose2dq(pose = pose1, device = device)
    pose2_dq = pose2dq(pose = pose2, device = device)

    # Compute dual quaternion distance
    distance = dq_distance(pose2_dq, pose1_dq)

    return distance

# Compute error as a dual quaternion distance
def cartesian_error(pose1: torch.Tensor, pose2: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pose1 / pose2 - torch.Tensor(N, 7): poses to compute the distance in translation(3) + rotation in quaternions(4).
        - device - str: Device into which the environment is stored.

    Out:
        - distance - torch.Tensor(N, 3): distance[:, 0] in translation and Euler angles, translation module[:, 1] and rotation module[:, 2].
    '''
    # Convert position and orientation (pose) to dual quaternion
    pos_1, pos_2 = pose1[:, :3], pose2[:, :3]

    r1, p1, y1 = euler_xyz_from_quat(quat = pose1[:, 3:])
    r2, p2, y2 = euler_xyz_from_quat(quat = pose2[:, 3:])

    euler_1 = torch.cat((r1.unsqueeze(-1), p1.unsqueeze(-1), y1.unsqueeze(-1)), dim=-1).to(device)
    euler_2 = torch.cat((r2.unsqueeze(-1), p2.unsqueeze(-1), y2.unsqueeze(-1)), dim=-1).to(device)

    # euler_1 = torch.tensor(euler_1).to(device)
    # euler_2 = torch.tensor(euler_2).to(device)

    euler_1 = torch.where(euler_1 > 0.0, euler_1, -euler_1)
    euler_2 = torch.where(euler_2 > 0.0, euler_2, -euler_2)

    t_dist = (pos_1 - pos_2).norm(dim = -1) / 0.46
    r_dist = (euler_1 - euler_2).norm(dim = -1) / 3.1

    distance = t_dist + r_dist

    return torch.cat((distance.unsqueeze(-1), t_dist.unsqueeze(-1), r_dist.unsqueeze(-1)), dim = -1)


def SE3_error(pose1: torch.Tensor, pose2: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pose1 / pose2 - torch.Tensor(N, 7): poses to compute the distance in translation(3) + rotation in quaternions(4).
        - device - str: Device into which the environment is stored.

    Out:
        - distance - torch.Tensor(N, 3): distance[:, 0] in SE(3), translation module[:, 1] and rotation module[:, 2].
    '''

    pos_1, pos_2 = pose1[:, :3], pose2[:, :3]

    R1 = matrix_from_quat(quaternions = pose1[:, 3:])
    R2 = matrix_from_quat(quaternions = pose2[:, 3:])

    R_diff = torch.matmul(torch.transpose(R1, -1, -2), R2)
    trace_diff = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]

    t_dist = (pos_1 - pos_2).norm(dim = -1) / 0.46

    r_dist = torch.abs(torch.acos(torch.clamp((trace_diff - 1) / 2, -1.0, 1.0))) / pi

    distance = t_dist + r_dist

    return torch.cat((distance.unsqueeze(-1), t_dist.unsqueeze(-1), r_dist.unsqueeze(-1)), dim = -1)


##
# UTILS
##

# Transforms a pose / frame into a dual quaternion
def pose2dq(pose: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pose - torch.Tensor(N, 7): pose to be converted into a dual quaternion in translation(3) + rotation in quaternions(4).
        - device - str: Device into which the environment is stored.

    Out:
        - return - torch.Tensor(N, 8): frame represented as a dual quaternion.
    '''

    # Separate position and orientation
    pos = pose[:, :3]
    orient = pose[:, 3:]

    # Converts position to a simple quaternion
    pos = trans2q(pos = pos, device = device)

    # Shape comprobation
    assert pos.shape[-1] == 4
    assert orient.shape[-1] == 4

    # From translation and orientation to DQ
    return torch.cat((orient, 0.5 * q_mul(orient, pos)), dim = -1)


# Transforms a linear translation to simple quaternion
def trans2q(pos: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pos - torch.Tensor(N, 3): position to be converted into a simple quaternion.
        - device - str: Device into which the environment is stored.

    Out:
        - return - torch.Tensor(N, 4): translation represented as a simple quaternion.
    '''

    # Shape comprobation
    assert pos.shape[-1] == 3

    # Add zero to the imaginary part of the translation quaternion
    return torch.cat((torch.zeros((pos.size()[0], 1)).to(device), pos), dim = -1)
