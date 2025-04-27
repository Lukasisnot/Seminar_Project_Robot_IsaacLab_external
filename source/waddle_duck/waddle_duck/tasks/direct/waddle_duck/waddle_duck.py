# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Construct the absolute path to the USD file relative to this script's location
_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WADDLEDUCK_USD_PATH = os.path.join(_THIS_SCRIPT_DIR, "asset/Waddle_Duck", "MiniRobo_Simple_edit.usd")

WADDLEDUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/lukas/dev/waddle_duck/waddle_duck/source/waddle_duck/waddle_duck/tasks/direct/waddle_duck/asset/waddle-duck.usd",
        # usd_path="/home/lukas/Documents/Waddle_Duck/MiniRobo_Simple_edit.usd",
        usd_path=WADDLEDUCK_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
    # Using default soft limits
    soft_joint_pos_limit_factor=1.0,
)