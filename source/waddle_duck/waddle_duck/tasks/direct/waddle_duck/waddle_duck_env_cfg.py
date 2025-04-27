# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import dataclasses

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from .waddle_duck import WADDLEDUCK_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.sim import SimulationCfg, RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class WaddleDuckEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 4
    observation_space = 14
    state_space = 0

    # simulation
    # sim: SimulationCfg = SimulationCfg(
    #         dt=1 / 120,
    #         render_interval=decimation,
    #         physics_material=RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),)
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = WADDLEDUCK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    imu_cfg: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Robot/base")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # - action scale
    action_scale = 1.0
    # - reward scales
    rew_scale_alive = 0
    rew_scale_terminated = -5.0
    rew_scale_y_pos_progression = 2
    rew_scale_x_pos_deviation = -0.1
    rew_scale_action_to_prev_dif = -1.0
    rew_scale_joint_limit_approach = 1.0 # not implemented yet
    # - reset states/conditions
    initial_servo_pos_range = [-0.1, 0.1]
    max_side_grav_lin_acc = 80.0  # reset if robot exceeds abs x or y lin acc (checks if the robot has fallen over)
    min_y_pos_progress = -1.0 