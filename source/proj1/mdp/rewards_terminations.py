from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================
# OBSERVATION HELPERS
# ============================================================
def feet_height_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, asset_cfg.body_ids, 2]


def base_height_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2:3]


def feet_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if hasattr(contact_sensor.data, "net_forces_w_history"):
        forces = contact_sensor.data.net_forces_w_history
        force_mag = forces.norm(dim=-1).max(dim=1)[0]
        return (force_mag[:, sensor_cfg.body_ids] > float(force_threshold)).float()

    if hasattr(contact_sensor.data, "net_forces_w"):
        forces = contact_sensor.data.net_forces_w
        force_mag = forces.norm(dim=-1)
        return (force_mag[:, sensor_cfg.body_ids] > float(force_threshold)).float()

    return torch.zeros(
        (env.num_envs, len(sensor_cfg.body_ids)),
        device=env.device,
        dtype=torch.float32,
    )


def feet_speed_xy_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return vel_xy.reshape(env.num_envs, -1)

def feet_speed_xy_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(vel_xy), dim=(1, 2))



def height_scan(
    env: ManagerBasedRLEnv,
    num_rays: int = 8,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, num_rays), device=env.device, dtype=torch.float32)


def surrounding_height_offsets(
    env: ManagerBasedRLEnv,
    num_points: int = 8,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, num_points), device=env.device, dtype=torch.float32)


# ============================================================
# REWARDS
# ============================================================
def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.terminated.float()

def root_lin_vel_error_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_lin_vel_w[:, :3]), dim=1)

def root_ang_vel_error_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_w[:, :3]), dim=1)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    g_b = robot.data.projected_gravity_b
    err = torch.abs(g_b[:, 2] + 1.0)
    return torch.exp(-6.0 * err)

def base_yaw_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_ang_vel_w[:, 2])


def base_z_lin_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_lin_vel_w[:, 2])


def joint_pos_limit_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    lim = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    q_min = lim[..., 0]
    q_max = lim[..., 1]

    below = torch.clamp(q_min - q, min=0.0)
    above = torch.clamp(q - q_max, min=0.0)
    return torch.sum(below + above, dim=1)


def root_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    err = torch.abs(z - float(target_height))
    return torch.exp(-12.0 * err)


def undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if hasattr(contact_sensor.data, "net_forces_w_history"):
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1).max(dim=1)[0]
    elif hasattr(contact_sensor.data, "net_forces_w"):
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1)
    else:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    return torch.sum((force_mag > float(force_threshold)).float(), dim=1)

def joint_pos_target_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.mean(torch.abs(q - q0), dim=1)
    return torch.exp(-1.0 * err)

def zero_velocity_command(env: ManagerBasedRLEnv, dim: int = 3) -> torch.Tensor:
    return torch.zeros((env.num_envs, dim), device=env.device)

def waist_yaw_abs_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.square(q[:, 0])


# ============================================================
# TERMINATIONS
# ============================================================

def fall_by_height(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < float(min_height)


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    gravity_z = robot.data.projected_gravity_b[:, 2]
    angle = torch.acos(-gravity_z) 
    return angle > float(limit_angle)


def undesired_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if hasattr(contact_sensor.data, "net_forces_w_history"):
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1).max(dim=1)[0]
    elif hasattr(contact_sensor.data, "net_forces_w"):
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1)
    else:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    return torch.any(force_mag > float(force_threshold), dim=1)


# ============================================================
# EVENTS 
# ============================================================


def push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot = env.scene[asset_cfg.name]
    vel = robot.data.root_vel_w[env_ids].clone()

    vel[:, 0] += torch.empty(len(env_ids), device=env.device).uniform_(
        velocity_range["x"][0], velocity_range["x"][1]
    )
    vel[:, 1] += torch.empty(len(env_ids), device=env.device).uniform_(
        velocity_range["y"][0], velocity_range["y"][1]
    )

    robot.write_root_velocity_to_sim(vel, env_ids)

def reset_joints_by_scale(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    position_range: tuple, 
    velocity_range: tuple, 
    asset_cfg: SceneEntityCfg
):
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    
    scales = torch.distributions.Uniform(position_range[0], position_range[1]).sample(joint_pos.shape).to(env.device)
    joint_pos *= scales
    
    joint_vel = torch.zeros_like(joint_pos) 
    robot.write_joint_state_to_sim(joint_pos, joint_vel, asset_cfg.joint_ids, env_ids)