"""Configuration for the UR5e and Kinova GEN3 robots (Isaac Lab 3.0 / PhysX backend)."""

import os
from math import pi

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Paths
##

# Portable asset path: resolved relative to this file instead of a hardcoded
# absolute path from a previous cluster user (was "/home/hapi039h/isaaclab/...").
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "config", "usd")

##
# Configuration
##

GEN3_4f_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSETS_DIR, "gen3_4f.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_joint_1": 0.0,
            "arm_joint_2": -pi / 8,
            "arm_joint_3": 0.0,
            "arm_joint_4": 3 * pi / 4,
            "arm_joint_5": 0.0,
            "arm_joint_6": -pi / 6,
            "arm_joint_7": 5 * pi / 4 - pi,
            "joint_12_0": 0.263,
        },
        pos=(-1.25, 0.0, 0.0),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit=100.0,
            effort_limit=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
)

UR5e_4f_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSETS_DIR, "ur5e_4f_ros2.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_shoulder_pan_joint": 0.0,
            "arm_shoulder_lift_joint": -pi / 2,
            "arm_elbow_joint": -pi / 2,
            "arm_wrist_1_joint": 0.0,
            "arm_wrist_2_joint": pi / 2,
            "arm_wrist_3_joint": pi / 4,
            "joint_0_0": 0.0,
            "joint_1_0": 0.263 * 6,
            "joint_2_0": 0.263 * 5,
            "joint_3_0": 0.263 * 2.3,
            "joint_4_0": 0.0,
            "joint_5_0": 0.263 * 6,
            "joint_6_0": 0.263 * 5,
            "joint_7_0": 0.263 * 2.3,
            "joint_8_0": 0.0,
            "joint_9_0": 0.263 * 6,
            "joint_10_0": 0.263 * 5,
            "joint_11_0": 0.263 * 2.3,
            "joint_12_0": 0.263,  # zero position is 0.263
            "joint_13_0": 0.0,
            "joint_14_0": 0.263 * 5,
            "joint_15_0": 0.263 * 5,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit=100.0,
            effort_limit=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
)
