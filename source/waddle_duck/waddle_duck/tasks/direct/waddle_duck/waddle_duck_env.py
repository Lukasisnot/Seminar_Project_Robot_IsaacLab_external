# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import Imu

from .waddle_duck_env_cfg import WaddleDuckEnvCfg


class WaddleDuckEnv(DirectRLEnv):
    cfg: WaddleDuckEnvCfg

    def __init__(self, cfg: WaddleDuckEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        self.robot_dof_ids, _ = self.robot.find_joints(".*")

        # self.joint_vel = self.robot.data.joint_vel
        self.joint_pos = self.robot.data.joint_pos

        self.imu_lin_acc = self.imu.data.lin_acc_b
        self.imu_ang_acc = self.imu.data.ang_acc_b

        self.previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.target_pos = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.y_pos_progress = torch.zeros(self.num_envs, device=self.device)

        self.body_base_id, _ = self.robot.find_bodies("base")
        self.robot_last_y_pos = self.robot.data.body_pos_w[:, self.body_base_id[0], 2]
        self.robot_init_x_pos = self.robot.data.body_pos_w[:, self.body_base_id[0], 1]

    def _setup_scene(self):
        # add robot and sensors
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.imu = Imu(self.cfg.imu_cfg)
        self.scene.sensors["imu"] = self.imu
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.target_pos = self.actions * self.cfg.action_scale + self.joint_pos
        # self.actions = torch.tensor([[45, 45, 45, 45]] * self.num_envs, device=self.device)
        # self.actions = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * self.num_envs, device=self.device)
        # self.actions = torch.tensor([[1.5, 2.5, 1.5, 2.5]] * self.num_envs, device=self.device)

    def _apply_action(self) -> None:
        # print("ACTIONS")
        # print(self.actions)
        # print("JOINT POS")
        # print(self.joint_pos)
        # print("TARGET POS")
        # print(target_pos)
        self.robot.set_joint_position_target(self.target_pos, joint_ids=self.robot_dof_ids)

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.imu_lin_acc = self.imu.data.lin_acc_b
        self.imu_ang_acc = self.imu.data.ang_acc_b
        obs = torch.cat(
            (
                self.joint_pos[:, self.robot_dof_ids[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.robot_dof_ids[1]].unsqueeze(dim=1),
                self.joint_pos[:, self.robot_dof_ids[2]].unsqueeze(dim=1),
                self.joint_pos[:, self.robot_dof_ids[3]].unsqueeze(dim=1),
                self.imu_lin_acc[:, 0].unsqueeze(dim=1),
                self.imu_lin_acc[:, 1].unsqueeze(dim=1),
                self.imu_lin_acc[:, 2].unsqueeze(dim=1),
                self.imu_ang_acc[:, 0].unsqueeze(dim=1),
                self.imu_ang_acc[:, 1].unsqueeze(dim=1),
                self.imu_ang_acc[:, 2].unsqueeze(dim=1),
                self.previous_actions[:, 0].unsqueeze(dim=1),
                self.previous_actions[:, 1].unsqueeze(dim=1),
                self.previous_actions[:, 2].unsqueeze(dim=1),
                self.previous_actions[:, 3].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        # print(obs)
        # print("-------------------------------------------------------")
        # print(self.imu_lin_acc[:, 0].unsqueeze(dim=1))
        # print(self.imu_lin_acc[:, 1].unsqueeze(dim=1))
        # print(self.imu_lin_acc[:, 2].unsqueeze(dim=1))
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        rew_termination = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        self.y_pos_progress = torch.sum(torch.mul(torch.sub(self.robot_last_y_pos, self.robot.data.body_pos_w[:, self.body_base_id[0], 2]), -1).unsqueeze(dim=1), dim=-1)
        rew_y_pos_progression = self.cfg.rew_scale_y_pos_progression * self.y_pos_progress
        rew_x_pos_deviation = self.cfg.rew_scale_x_pos_deviation * torch.sum(torch.square(torch.sub(torch.abs(self.robot.data.body_pos_w[:, self.body_base_id[0], 1]), torch.abs(self.robot_init_x_pos))).unsqueeze(dim=1), dim=-1)
        rew_action_to_prev_dif = self.cfg.rew_scale_action_to_prev_dif * torch.sum(torch.square(self.actions - self.previous_actions), dim=-1)
        total_reward = rew_alive + rew_termination + rew_y_pos_progression + rew_x_pos_deviation + rew_action_to_prev_dif

        print("-------------------------------------------------------")
        print("LAST POS")
        print(self.robot_last_y_pos[0])
        print("CURR POS")
        print(self.robot.data.body_pos_w[0, self.body_base_id[0], 2])
        print("POS DIF")
        print(rew_y_pos_progression[0] / self.cfg.rew_scale_y_pos_progression)
        print("Y PROGRESS")
        print(self.y_pos_progress[0])
        print("TERMINATED REW")
        print(rew_termination)
        print("Y PROGRESS REW")
        print(rew_y_pos_progression[0])
        print("X DEVIATION REW")
        print(rew_x_pos_deviation[0])
        print("ACTION TO PREV DIF")
        print(rew_action_to_prev_dif[0])
        print("TOTAL REWARD")
        print(total_reward[0])
        print("X & Y LIN ACC")
        print(self.imu_lin_acc[:, 0])
        print(self.imu_lin_acc[:, 1])
        print("ACTIONS")
        print(self.actions[0])
        print("PREV ACTIONS")
        print(self.previous_actions[0])
        print("TARGET POSITIONS")
        print(self.target_pos[0])
        print("JOINT POSITIONS")
        print(self.joint_pos[0])
        
        self.robot_last_y_pos = self.robot.data.body_pos_w[:, self.body_base_id[0], 2]
        self.previous_actions = self.actions.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # time_out = torch.empty(self.num_envs, dtype=torch.int, device=self.device)

        # out_of_bounds = torch.any(torch.abs(self.imu_lin_acc[:, 0]) > self.cfg.max_side_grav_lin_acc, dim=-1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.imu_lin_acc[:, 1]) > self.cfg.max_side_grav_lin_acc, dim=-1)

        # out_of_bounds = torch.any(torch.gt(torch.abs(self.imu_lin_acc[:, 0]), self.cfg.max_side_grav_lin_acc), dim=-1)
        # out_of_bounds = out_of_bounds | torch.any(torch.gt(torch.abs(self.imu_lin_acc[:, 1]), self.cfg.max_side_grav_lin_acc), dim=-1)
        out_of_bounds = torch.zeros_like(time_out)

        # out_of_bounds = out_of_bounds | torch.any(torch.lt(self.y_pos_progress, self.cfg.min_y_pos_progress), dim=-1)
        # out_of_bounds = torch.empty(self.num_envs, dtype=torch.int, device=self.device)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_vel = self.robot.data.default_joint_vel[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self.robot_dof_ids] += sample_uniform(
            self.cfg.initial_servo_pos_range[0],
            self.cfg.initial_servo_pos_range[1],
            joint_pos[:, self.robot_dof_ids].shape,
            joint_pos.device,
        )

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
