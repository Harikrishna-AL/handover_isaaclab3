from __future__ import annotations

import os
import torch
import torch.nn as nn
from collections.abc import Sequence
import copy
import random

from .mdp.utils import compute_rewards, save_images_grid
from .mdp.rewards import dual_quaternion_error, cartesian_error, SE3_error
from .bimanual_direct_env_cfg import BimanualDirectCfg, update_cfg, update_collisions

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import RigidObject


'''
                    ############## IMPORTANT #################
   The whole environment is build for two robots: the UR5e and Kinova GEN3-7dof.
   These two variables (cfg.UR5e and cfg.GEN3) serve as an abstraction to treat the robots during the episodes. In fact,
all the methods need an index to differentiate from which robot get the information.
   Also, data storage is performed using lists, not tensors because the joint space of the robots is
different from one another.

   NOTE (Isaac Lab 3.0 port): asset ``.data.*`` properties now return a
   ``ProxyArray`` instead of a raw ``torch.Tensor``; ``.torch`` is appended at
   every read site below to get the cached zero-copy tensor view explicitly
   (ProxyArray also has a transparent torch bridge, but it emits a
   DeprecationWarning per-call, so we use the explicit accessor).
'''

# Class for the Bimanual Direct Environment
class BimanualDirect(DirectRLEnv):
    cfg: BimanualDirectCfg

    # --- init function ---
    def __init__(self, cfg: BimanualDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initial poses sampled in reset for both robots
        self.reset_joint_positions = [torch.zeros((self.num_envs, 6+16)).to(self.device), torch.zeros((self.num_envs, 7+16)).to(self.device)]

        # Debug poses for the object and end effector of the GEN3 robot. These poses
        # are used to draw the markers in the simulation
        self.debug_GEN3_ee_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.debug_grasp_point_obj_pose_w = copy.deepcopy(self.debug_GEN3_ee_pose_w)
        self.debug_tips_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.debug_tips_back_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)

        # Poses for the object and GEN3 robot
        self.GEN3_rot_ee_pose_r = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.grasp_point_obj_pose_r = copy.deepcopy(self.GEN3_rot_ee_pose_r)
        self.tips_pose_r = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.tips_pose_r_back = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)

        # Indexes for: robot joints, hand joints, all joints, finger tips, end effector's jacobian
        self._robot_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self._hand_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.hand_joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self._all_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.all_joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self.finger_tips = torch.tensor([self.scene.articulations[key].find_bodies(self.cfg.finger_tips[idx])[0] for idx, key in enumerate(self.cfg.keys)]).to(self.device)
        self.ee_jacobi_idx = torch.tensor([self.scene.articulations[key].find_bodies(self.cfg.ee_link[idx])[0][0] - 1 for idx, key in enumerate(self.cfg.keys)]).to(self.device)


        # IK Controller
        controller_cfg = DifferentialIKControllerCfg(command_type = "pose", use_relative_mode = False, ik_method = "dls")
        # DifferentialIKControllerCfg: Configuration for differential inverse kinematics controller.
        #    command_type: Type of task-space command to control the articulation's body.
        #    use_relative_mode: Whether to use relative mode for the controller. --> Use increments for the positions.
        #    ik_method: Method for computing inverse of Jacobian.

        self.controller = DifferentialIKController(controller_cfg, num_envs = self.num_envs, device = self.device)
        # DifferentialIKController: Differential inverse kinematics (IK) controller.
        #    num_envs: Number of environments handled by the controller.
        #    device: Device into which the controller is stored.

        # List for the default joint poses of both robots --> As a list due to the different joints of the arms (6 and 7)
        self.default_joint_pos = [self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.default_joint_pos.torch,
                                  self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.default_joint_pos.torch]

        # Default joints to open the hand
        self.open_hand_joints = torch.zeros((1, 16)).to(self.device)
        self.open_hand_joints[:, 1] = 0.263  # this value is the zero for the joint0 of the thumb

        # List of joint actions
        self.actions = copy.deepcopy(self.default_joint_pos)

        # Poses obtained at reset
        self.reset_robot_poses_r = [torch.zeros((self.num_envs, 7)).to(self.device), torch.zeros((self.num_envs, 7)).to(self.device)]

        # Update configuration class
        self.cfg = update_cfg(cfg = cfg, num_envs = self.num_envs, device = self.device)

        # Obtain the ranges in which sample reset positions
        self.ee_pose_ranges = torch.tensor([[ [(i + cfg.apply_range[idx]*inc[0]), (i + cfg.apply_range[idx]*inc[1])] for i, inc in zip(poses, cfg.ee_pose_incs)] for idx, poses in enumerate(cfg.ee_init_pose)]).to(self.device)

        # Obtain the number of contact sensors per environment
        self.num_contacts = 0
        for __ in self.cfg.contact_sensors_dict:
            self.num_contacts += 1

        # Variable to store contacts between prims
        self.contacts = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)
        # Force magnitude tracking (raw sensor force norms, current and previous step)
        self.force_norms = torch.zeros(self.num_envs, len(self.cfg.contact_sensors_dict)).to(self.device)
        self.prev_force_norms = torch.zeros_like(self.force_norms)
        self.prev_contacts = torch.zeros_like(self.contacts).float()

        # Create output directory to save images
        if self.cfg.save_imgs:
            self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
            os.makedirs(self.output_dir, exist_ok=True)

        # Previous distances
        self.prev_dist = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)
        self.prev_dist_target = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)

        # Reached flags
        self.obj_reached = torch.zeros(self.num_envs).to(self.device).bool()
        self.obj_reached_target = torch.zeros(self.num_envs).to(self.device).bool()


        # --------- COMPROBAR VALIDEZ ---------
        self.err_aux = torch.zeros((self.num_envs, 3)).to(self.device)
        self.cont_err = torch.zeros(self.num_envs).to(self.device)

        self.aux_info = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0]


    # Method to add all the prims to the scene --> Overrides method of DirectRLEnv
    def _setup_scene(self, ):
        '''
        NOTE: The "self.scene" variable is declared at "super().__init__(cfg, render_mode, **kwargs)" in __init__
        '''

        # Add ground plane
        # NOTE (Isaac Lab 3.0 kit-less port): GroundPlaneCfg()'s default usd_path
        # points at a Nucleus-hosted asset ({ISAAC_NUCLEUS_DIR}/Environments/Grid/...).
        # With no Isaac Sim / omni.client available, that can't be resolved, so this
        # spawns a plain static procedural collider instead (same pattern already
        # used for `object_cfg` below -- CuboidCfg never touches check_file_path()).
        ground_cfg = sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.05))

        # Clone, filter and replicate
        self.scene.clone_environments(copy_from_source=False)
        # clone_environments: Creates clones of the environment /World/envs/env_0.
        #     if "copy_from_source" is False, clones inherit from /World/envs/env_0 and mirror its changes.

        self.scene.filter_collisions(global_prim_paths=[])
        # filter_collisions: Disables collisions between the environments in /World/envs/env_.* and enables collisions with the prims in global prim paths (e.g. ground plane).
        #     if "global_prim_paths" is None, environments do not collide with each other.


        # Add articulations to scene
        self.scene.articulations[self.cfg.keys[self.cfg.UR5e]] = Articulation(self.cfg.robot_cfg_1)
        self.scene.articulations[self.cfg.keys[self.cfg.GEN3]] = Articulation(self.cfg.robot_cfg_2)


        # Correct collision sensors
        self.cfg = update_collisions(self.cfg, num_envs = self.num_envs)
        for idx, sensor_cfg in self.cfg.contact_sensors_dict.items():
            self.scene.sensors[idx] = ContactSensor(sensor_cfg)

        # Add bodies
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)

        # Add extras (markers, ...)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # Obtain the end effector pose of the index robot in the base frame
    def _get_ee_pose(self, idx):
        '''
        In:
            - idx - int(0,1): index of the robot.

        Out:
            - ee_pos_r - torch.tensor(N, 3): position of the end effector in the base frame for each environment.
            - ee_quat_r - torch.tensor(N, 4): orientation as a quaternions of the end effector in the base frame for each environment.
            - jacobian - torch.tensor(N, 6, n_joints (6 or 7)): jacobian of all robots' end effector.
            - joint_pos - torch.tensor(N, n_joints(6 or 7)): joint position of the robot.
        '''

        # Obtains the jacobian of the end effector of the robot
        # NOTE: `root_physx_view` is deprecated in favor of `root_view`, but kept here
        # since it still works (emits a DeprecationWarning) and its indexing semantics
        # are unchanged; `data.body_link_jacobian_w` is the new recommended accessor but
        # was not verified against this code's indexing convention without a live GPU run.
        jacobian = self.scene.articulations[self.cfg.keys[idx]].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx[idx], :, self._robot_joints_idx[idx]]

        # Obtains the pose of the end effector in the world frame
        ee_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.body_state_w.torch[:, self.ee_jacobi_idx[idx]+1, 0:7]

        # Obtains the pose of the base of the robot in the world frame
        root_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.root_state_w.torch[:, 0:7]

        # Obtains the joint position
        joint_pos = self.scene.articulations[self.cfg.keys[idx]].data.joint_pos.torch[:, self._robot_joints_idx[idx]]

        # Transforms end effector frame coordinates (in world) into root (local / base) coordinates
        ee_pos_r, ee_quat_r = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
        # root = T01 // ee = T02 -> substract = (T01)^-1 * T02 = T10 * T02 = T12

        return ee_pos_r, ee_quat_r, jacobian, joint_pos


    # Method to preprocess the actions so they have a proper format
    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        '''
        In:
            - actions - torch.Tensor: raw actions. --> rotation is in the form of a quaternion.
                Format: [x, y, z, rotation]:
                    actions[:3]: translation.
                    actions[3]: rotation angle of a quaternion.
                    actions[4:]: rotation as a: (1) vector of a quaternion or (2) euler angles.

        Out:
            - actions - torch.Tensor: preprocessed actions.
        '''

        # Clamp actions
        actions = torch.clamp(actions, -1, 1)

        # Scale actions
        actions[:, :3]  *= self.cfg.translation_scale

        # Action in quaternion form
        actions_quat = torch.zeros((self.num_envs, 7+16)).to(self.device)
        actions_quat[:, :3] = actions[:, :3]

        # Determines the index of the hand joints
        hand_joint_index = 6 + int(not self.cfg.euler_flag)

        # Obtains extended action
        val = (actions[:, hand_joint_index:] * self.cfg.hand_joint_scale).repeat_interleave(4, dim = -1)

        # Assigns the values to the respective joints
        actions_quat[:, 7] = 0
        actions_quat[:, 8] = val[:, 0] * 2
        actions_quat[:, 9] = 0
        actions_quat[:, 10] = 0
        actions_quat[:, 11] = val[:, 1] * 5
        actions_quat[:, 12] = 0.0
        actions_quat[:, 13] = val[:, 2] * 5
        actions_quat[:, 14] = val[:, 3] * 5

        actions_quat[:, 15] = val[:, 4]*4
        actions_quat[:, 16] = val[:, 5] * 3
        actions_quat[:, 17] = val[:, 6]*4
        actions_quat[:, 18] = val[:, 7]*4

        actions_quat[:, 19] = val[:, 8] * 2.5
        actions_quat[:, 20] = val[:, 9] * 2.5
        actions_quat[:, 21] = val[:, 10] * 2.5
        actions_quat[:, 22] = val[:, 11] * 2.5

        # Assigns hand opening if the object is reached
        self.reset_joint_positions[self.cfg.UR5e][self.obj_reached.bool(), 6:] = self.open_hand_joints

        # If the actions are in euler, transform them to quaternion
        if self.cfg.euler_flag:
            actions[:, 3:6] *= self.cfg.angle_scale

            actions_quat[:, 3:7] = quat_from_euler_xyz(roll = actions[:, 3],
                                                    pitch = actions[:, 4],
                                                    yaw = actions[:, 5])

        # Else, the actions are already in quaternion form
        else:
            # Scale angle and rotation vector
            actions_quat[:, 3] *= self.cfg.angle_scale
            actions_quat[:, 4:7] = torch.nn.functional.normalize(actions_quat[:, 4:7])

            # Real part of the quaternion
            w = torch.cos(actions_quat[:, 3]/2).unsqueeze(dim = 0).T

            # Imaginary part of the quaternion
            v = actions_quat[:, 4:7]
            sin_a = torch.sin(actions_quat[:, 3] / 2).unsqueeze(dim=0).T

            # Build the quaternion
            q = sin_a * v

            # Reassign quaternion
            actions_quat[:, 3:7] = torch.cat((w, q), dim = 1)

        return actions_quat


    # Performs the action increment
    def perform_increment(self, idx, actions):
        '''
        In:
            - idx - int(0,1): index of the robot.
            - actions - torch.tensor(N, 7 + 16): the increment to be performed to the actual pose and hand joint position.

        Out:
            - None
        '''

        # Obtains the poses
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose(idx)

        # Perform an increment on the robot end effector in the root frame
        new_act_pos, new_act_quat = combine_frame_transforms(t01 = ee_pos_r, q01 = ee_quat_r,
                                                             t12 = actions[:, 0:3], q12 = actions[:, 3:7])
        new_poses = torch.cat((new_act_pos, new_act_quat), dim = -1)


        # Set the command for the IKDifferentialController
        self.controller.set_command(new_poses)

        # Perform the increment for the hand
        new_hand_joint_pos = self.scene.articulations[self.cfg.keys[idx]].data.joint_pos.torch[:, self._hand_joints_idx[idx]] + actions[:, 7:]
        new_hand_joint_pos[:, 5] = -0.65

        # Get the actions for the UR5e. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.actions[idx] = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos),
                                       new_hand_joint_pos),
                                       dim = -1)


    # Method called before executing control actions on the simulation --> Overrides method of DirecRLEnv
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        '''
        In:
            - actions - torch.tensor(N, 6+3): actions to apply to the environment (robot and hand actions).

        Out:
            - None
        '''

        # Preprocessing actions
        actions = self._preprocess_actions(actions)

        # --- GEN3 actions ---
        # Obtains the increments and the poses for the GEN3 robot
        self.perform_increment(idx = self.cfg.GEN3, actions = actions)


    # Applies the preprocessed action in the environment --> Overrides method of DirecRLEnv
    def _apply_action(self) -> None:
        '''
        In:
            - None

        Out:
            - None
        '''

        # Moves the holder if required
        self.reset_joint_positions[self.cfg.UR5e][:, 0] += self.dir[0]*0.0005 * int(self.cfg.move_holder)
        self.reset_joint_positions[self.cfg.UR5e][:, 1] += self.dir[1]*0.00015 * int(self.cfg.move_holder)
        self.reset_joint_positions[self.cfg.UR5e][:, 2] += self.dir[2]*0.00015 * int(self.cfg.move_holder)
        self.reset_joint_positions[self.cfg.UR5e][:, 3] += self.dir[3]*0.0005 * int(self.cfg.move_holder)
        self.reset_joint_positions[self.cfg.UR5e][:, 4] += self.dir[4]*0.0007 * int(self.cfg.move_holder)
        self.reset_joint_positions[self.cfg.UR5e][:, 5] += self.dir[5]*0.00175 * int(self.cfg.move_holder)

        # Applies joint actions to the robots
        self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].set_joint_position_target_index(target=self.reset_joint_positions[self.cfg.UR5e], joint_ids=self._all_joints_idx[self.cfg.UR5e])
        self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].set_joint_position_target_index(target=self.actions[self.cfg.GEN3], joint_ids=self._all_joints_idx[self.cfg.GEN3])


    # Update the position of the markers with debug purposes
    def update_markers(self):
        '''
        Current markers:
            - End effector of the UR5e.
            - End effector of the GEN3.
            - Grasping position for the object (transformated to match GEN3's).
            - Tips of the fingers of the GEN3.
            - Tips of the fingers of the GEN3 (displaced in front of the hand).
        '''

        # Obtains the positions of the of the robots
        ee_pose_w_UR5e = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w.torch[:, self.finger_tips[self.cfg.UR5e], 0:7].mean(1)

        # Obtains a tensor of indices (a tensor containing tensors from 0 to the number of markers)
        marker_indices = torch.arange(self.scene.extras["markers"].num_prototypes).repeat(self.num_envs)

        # Updates poses in simulation
        self.scene.extras["markers"].visualize(translations = torch.cat((ee_pose_w_UR5e[:, :3],
                                                                         self.debug_GEN3_ee_pose_w[:, :3],
                                                                         self.debug_grasp_point_obj_pose_w[:, :3],
                                                                         self.debug_tips_pose_w[:, :3],
                                                                         self.debug_tips_back_pose_w[:, :3]),),
                                                orientations = torch.cat((ee_pose_w_UR5e[:, 3:],
                                                                          self.debug_GEN3_ee_pose_w[:, 3:],
                                                                          self.debug_grasp_point_obj_pose_w[:,3:],
                                                                          self.debug_tips_pose_w[:, 3:],
                                                                          self.debug_tips_back_pose_w[:, 3:]),),
                                                marker_indices=marker_indices)


    # Method to filter collisions according to the force matrix
    def filter_collisions(self):
        '''
        In:
            - None

        Out:
            - None
        '''

        # Loop through all the contact sensors configuration for the indexes
        for idx, (key, __) in enumerate(self.cfg.contact_sensors_dict.items()):

            # Obtain the matrix -> reshape it -> sum the last two dimensions ->
            #    -> if the value is greater than 0, there is force so  there is contact
            self.contacts[:, idx] = torch.abs(self.scene.sensors[key].data.force_matrix_w.torch).view(self.num_envs, -1, 3).sum(dim = (1,2), keepdim = True).squeeze((-2, -1)) > 0.0
            self.force_norms[:, idx] = (
                torch.abs(self.scene.sensors[key].data.force_matrix_w.torch)
                    .view(self.num_envs, -1, 3)
                    .norm(dim=-1)
                    .sum(dim=-1)
            )


    # Updates the poses of the object and robots so they can match when performing the grasp
    def update_new_poses(self):
        '''
        In:
            - None

        Out:
            - None
        '''

        # ---- GEN3 transformations ----
        # Obtain the pose of the GEN3 end effector in world frame
        self.debug_GEN3_ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w.torch[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 0:7]

        # Obtains the pose of the base of the GEN3 robot in the world frame
        GEN3_root_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.root_state_w.torch[:, 0:7]

        # Obtain the pose of the end effector in GEN3 root frame
        GEN3_rot_ee_pos_r, GEN3_rot_ee_quat_r = subtract_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                                              t02 = self.debug_GEN3_ee_pose_w[:, :3], q02 = self.debug_GEN3_ee_pose_w[:, 3:])

        self.GEN3_rot_ee_pose_r = torch.cat((GEN3_rot_ee_pos_r, GEN3_rot_ee_quat_r), dim = -1)




        # ---- Tips transformations ----
        # Obtains the pose of the finger tips in world frame and performs the mean
        self.debug_tips_pose_w = torch.mean(self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w.torch[:, self.finger_tips[self.cfg.GEN3], 0:7], dim = -2)

        # Transform tips pose to GEN3 root frame
        tip_pos_r, tip_or_r = subtract_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                        t02 = self.debug_tips_pose_w[:, :3], q02 = self.debug_tips_pose_w[:, 3:])
        self.tips_pose_r = torch.cat((tip_pos_r, tip_or_r), dim = -1)

        # Replaces the orientation with GEN3 ee orientation
        self.tips_pose_r[:, 3:] = self.GEN3_rot_ee_pose_r[:, 3:]




        # Clones the tips original pose
        self.tips_pose_r_back = self.tips_pose_r.clone()

        # Displaces the tips pose in front of the hand
        tip_pos_r, tip_or_r = combine_frame_transforms(t01 = self.tips_pose_r[:, :3], q01 = self.tips_pose_r[:, 3:],
                                                        t12 = self.cfg.tips_displacement, q12 = torch.tensor([1,0,0,0]).to(self.device).repeat(self.num_envs, 1))
        self.tips_pose_r = torch.cat((tip_pos_r, tip_or_r), dim = -1)




        # Transforms the modified tips pose to the world frame
        tips_pos_w, tips_or_w = combine_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                         t12 = self.tips_pose_r[:, :3], q12 = self.tips_pose_r[:, 3:])
        self.debug_tips_pose_w = torch.cat((tips_pos_w, tips_or_w), dim = -1)

        # Transforms the modified tips back pose to the world frame
        tips_pos_w, tips_or_w = combine_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                         t12 = self.tips_pose_r_back[:, :3], q12 = self.tips_pose_r_back[:, 3:])
        self.debug_tips_back_pose_w = torch.cat((tips_pos_w, tips_or_w), dim = -1)




        # ---- Object transformations ----
        # Obtains the pose of the object in the world frame
        obj_pose_w = self.scene.rigid_objects["object"].data.body_state_w.torch[:, 0, :7]

        # Transforms the object frame so as to generate a more suitable frame for grasping
        grasp_point_obj_pos_w, grasp_point_obj_quat_w = combine_frame_transforms(t01 = obj_pose_w[:, :3], q01 = obj_pose_w[:, 3:],
                                                                             t12 = self.cfg.grasp_obs_obj_pos_trans, q12 = self.cfg.grasp_obs_obj_quat_trans)
        grasp_point_obj_pos_w, grasp_point_obj_quat_w = combine_frame_transforms(t01 = grasp_point_obj_pos_w, q01 = grasp_point_obj_quat_w,
                                                                             t12 = torch.zeros_like(grasp_point_obj_pos_w), q12 = self.cfg.rot_225_z_pos_quat)
        self.debug_grasp_point_obj_pose_w = torch.cat((grasp_point_obj_pos_w, grasp_point_obj_quat_w), dim=-1)

        # Apply transformation to get the grasping point in the GEN3 root frame
        grasp_point_obj_pos_r, grasp_point_obj_quat_r = subtract_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                                              t02 = grasp_point_obj_pos_w, q02 = grasp_point_obj_quat_w)
        self.grasp_point_obj_pose_r = torch.cat((grasp_point_obj_pos_r, grasp_point_obj_quat_r), dim = -1)


    # Getter for the observations for the environment --> Overrides method of DirectRLEnv
    def _get_observations(self) -> dict:
        '''
        In:
            - None

        Out:
            - observations - dict: observations from the environment --> Needs to be with "policy" key.
        '''

        # Obtain boolean values for collisions
        self.filter_collisions()

        # Updates the poses of the GEN3 end effector and the object so they match
        self.update_new_poses()

        # Render images every certain amount of steps
        if self.count % self.cfg.render_steps == 0 and self.cfg.render_imgs:

            # Obtain images from the sensor
            image_tensor = [self.scene["camera"].data.output["rgb"][0, ..., :3]]

            # Function to save images (in utils)
            if self.cfg.save_imgs:
                save_images_grid(images = image_tensor,
                                 subtitles = ["Camera"],
                                 title = "RGB Image: Cam0",
                                 filename = os.path.join(self.output_dir, "rgb", f"{self.count:04d}.jpg"))

        # Builds the tensor with all the observations in a single row tensor (N, 7+16)
        obs = torch.cat(
            (
                self.GEN3_rot_ee_pose_r,
                self.grasp_point_obj_pose_r,
            ),
            dim = -1
        )

        # Obtains the joint positions for the hand
        # hand_joint_pos_1 = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.joint_pos.torch[:, self._hand_joints_idx[self.cfg.UR5e]]
        hand_joint_pos_2 = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.joint_pos.torch[:, self._hand_joints_idx[self.cfg.GEN3]]

        # Selects the hand joints to be observed
        sel_hand_joint = torch.round(torch.cat((hand_joint_pos_2[:, 1].unsqueeze(-1), hand_joint_pos_2[:, 4].unsqueeze(-1), hand_joint_pos_2[:, 6:]), dim = -1), decimals = 1)

        # Concatenates the mean of selected hand joint positions
        obs = torch.cat(
            (
                obs,
                sel_hand_joint.view(-1, 3,4).mean(dim = -1).view(-1, 3),
            ),
            dim = -1
        )

        # Builds the dictionary
        observations = {"policy": obs,
                        "dist": self.aux_info[0],
                        "phase":self.obj_reached.int()}

        # Updates markers
        if self.cfg.debug_markers:
            self.update_markers()

        return observations


    # Computes the reward of the transition --> Overrides method of DirectRLEnv
    def _get_rewards(self) -> torch.Tensor:
        '''
        In:
            - None

        Out:
            - compute_rewards() - torch.tensor(N,1): reward for each environment.
        '''

        # Snapshot previous-step force/contact state before this step overwrites it
        self.prev_force_norms = self.force_norms.clone()
        self.prev_contacts = self.contacts.clone().float()

        # ---- Variable assignments ----
        rew_scale_hand_obj = self.cfg.rew_scale_hand_obj * torch.ones(self.num_envs).to(self.device)
        rew_scale_obj_target = self.cfg.rew_scale_obj_target
        ee_pose = self.tips_pose_r
        ee_pose_bask = self.tips_pose_r_back
        obj_pose = self.grasp_point_obj_pose_r
        prev_dist = self.prev_dist
        prev_dist_target = self.prev_dist_target
        device = self.device
        target_pose = self.cfg.target_pose

        tips_ur5e = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w.torch[:, self.finger_tips[self.cfg.UR5e], 0:7].mean(1)[:, :3]
        tips_gen3 = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w.torch[:, self.finger_tips[self.cfg.GEN3], 0:7].mean(1)[:, :3]
        obj = self.scene.rigid_objects["object"].data.body_state_w.torch[:, 0, :3]

        # ---- Distance computation ----
        # Dual quaternion distance between GEN3 hand tips and object
        hand_obj_dist = dual_quaternion_error(ee_pose, obj_pose, device)

        # Dual quaternion distance between GEN3 back hand and object
        hand_obj_dist_back = dual_quaternion_error(ee_pose_bask, obj_pose, device)

        # Dual quaternion distance between object and target pose
        obj_target_dist = dual_quaternion_error(obj_pose, target_pose, device)

        # Distance between hand tips of the robots
        tips_dist = torch.norm((tips_ur5e - tips_gen3), dim = -1)

        # Distance between object and GEN3 hand tips
        obj_dist = torch.norm((obj - tips_gen3), dim = -1)


        # ---- Contact computation ----
        # Obtain the weighted contacts
        contacts_w = self.contacts * self.cfg.contact_matrix

        # Thumb contact
        thumb_w = contacts_w[:, -8:-3].clone()
        thumb_con = thumb_w.sum(-1) > 0.0


        # ---- Flag ----
        # There is contact if the thumb and the fingers (finger collide without the thumb) are in contact
        if self.cfg.use_force_rewards and self.cfg.F_threshold > 0:
            force_normalized = torch.tanh(self.force_norms / self.cfg.F_ref)
            force_w_trans = force_normalized * self.cfg.contact_matrix
            thumb_force_w = force_w_trans[:, 10:15].clone()
            contacts_flag = torch.logical_and(
                force_w_trans[:, :-1].sum(-1) - thumb_force_w.sum(-1) > self.cfg.F_threshold,
                thumb_force_w.sum(-1) > 0.0)
        else:
            contacts_flag = torch.logical_and(contacts_w[:, :-1].sum(-1) - thumb_w.sum(-1) > 0.4, thumb_con)

        # Detect the exact step the phase transition fires, for metrics (before obj_reached is latched)
        just_transitioned = torch.logical_and(contacts_flag, torch.logical_not(self.obj_reached))

        # Reached flag pre-conditions: if if it did not reach the object befor ...
        bonus = self.obj_reached.clone().bool()

        # ... but does it now, ...
        self.obj_reached = torch.logical_or(contacts_flag, self.obj_reached)

        # ... the bonus must be activated if the object is reached at this step
        bonus = torch.logical_and(self.obj_reached.clone().bool(), torch.logical_not(bonus))

        # Check if the object has reached the target
        self.obj_reached_target = (obj_pose[:, 0] < 0.05).bool() # (obj_target_dist[:, 1] < obj_reach_target_thres).bool()


        # ---- Distance evaluation ----
        # Obtains the distance according to the object reached flag
        dist = hand_obj_dist[:, 0] * torch.logical_not(self.obj_reached).int() + obj_target_dist[:, 0] * self.obj_reached.int()
        prev_dist = prev_dist * torch.logical_not(self.obj_reached).int() + prev_dist_target * self.obj_reached.int()

        # Obtains wether the agent is approaching or not
        pre_mod = torch.logical_and(tips_dist > obj_dist, hand_obj_dist_back[:,0] > hand_obj_dist[:,0])
        mod = 2*(torch.logical_and(dist < prev_dist, pre_mod)) - 1

        # Modifies scalation according to the contacts detected
        rew_scale_hand_obj = rew_scale_hand_obj / (self.contacts[:, 1:-2].sum(-1) + 1)


        # ---- Distance reward ----
        # Reward for the first phase --> Approaching (mod) hand-obj distance divided by wether the object is approaching with the palm
        reward_1 = mod * rew_scale_hand_obj * torch.exp(-2*hand_obj_dist[:, 0]) / (1 + 2*(torch.logical_not(pre_mod)).int())

        # Reward for the second phase --> Object-target distance the target
        reward_2 = rew_scale_obj_target * torch.exp(-2*obj_target_dist[:, 0])


        # ---- Reward composition ----
        # Phase reward plus phase 1 bonuses
        reward = (reward_1) * torch.logical_not(self.obj_reached) + 10*(reward_2) * self.obj_reached + self.cfg.bonus_obj_reach * bonus / 2

        # Reward for the contacts
        reward = reward + contacts_w[:, 1:-1].sum(-1)

        # Signal 1: grasp firmness (force-magnitude-based contact reward)
        if self.cfg.use_signal_1:
            force_normalized = torch.tanh(self.force_norms / self.cfg.F_ref)
            force_w = force_normalized * self.cfg.contact_matrix
            signal_1_value = force_w[:, 1:-1].sum(-1)
            reward = reward + signal_1_value

        # Signal 2: thumb opposition balance
        if self.cfg.use_signal_2:
            force_normalized = torch.tanh(self.force_norms / self.cfg.F_ref)
            force_w = force_normalized * self.cfg.contact_matrix
            thumb_force = force_w[:, 10:15].sum(-1)
            finger_force = force_w[:, 1:10].sum(-1) + self.cfg.palm_weight * force_w[:, 15:16].sum(-1)
            balance_ratio = thumb_force / (finger_force + 1e-6)
            total_grasp_force = force_w[:, 1:15].sum(-1)
            balance_active = (total_grasp_force > self.cfg.F_threshold).float()
            r_balance = -self.cfg.lambda_balance * torch.abs(balance_ratio - 1.0) * self.obj_reached.float() * balance_active
            reward = reward + r_balance

        # Signal instability: penalize sudden contact-pattern changes with force drop
        if self.cfg.use_signal_instability:
            pattern_change = (self.contacts[:, 1:-1].float() - self.prev_contacts[:, 1:-1]).abs().sum(-1)
            current_total = self.force_norms[:, 1:-1].sum(-1)
            prev_total = self.prev_force_norms[:, 1:-1].sum(-1)
            force_drop = torch.relu(prev_total - current_total)
            r_instability = -self.cfg.lambda_instability * pattern_change * force_drop * self.obj_reached.float()
            reward = reward + r_instability

        # Reward for reaching target
        bonus_term = self.cfg.bonus_obj_reach * self.obj_reached_target * (contacts_w[:, 1:].sum(-1) > 0.0).int()
        reward = reward + bonus_term

        # 1. Obtain linear velocity norm of the GEN3 end effector (useful for logging regardless of penalty)
        ee_vel_norm = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w.torch[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 7:10], dim=-1)

        if self.cfg.vel_rew:
            # 2. Define the safe velocity threshold
            v_min = self.cfg.v_min

            # 3. Calculate excess velocity using ReLU
            excess_vel = torch.nn.functional.relu(ee_vel_norm - v_min)

            # 4. Calculate the distance-conditioned penalty
            lambda_vel = self.cfg.lambda_vel
            k_decay = self.cfg.k_decay
            velocity_penalty = lambda_vel * (excess_vel ** 2) * torch.exp(-k_decay * hand_obj_dist[:, 0]) * torch.logical_not(self.obj_reached).float()

            # 5. Subtract from final reward
            reward = reward - velocity_penalty

        # Excessive force penalty (master-switch gated, applies regardless of which signals are active)
        if self.cfg.use_force_rewards:
            excess_force = torch.relu(self.force_norms[:, 1:-1] - self.cfg.F_safe)
            r_excess = -self.cfg.lambda_force_excess * excess_force.sum(-1)
            reward = reward + r_excess

        # Update previous distances
        self.prev_dist = hand_obj_dist[:, 0]
        self.prev_dist_target = obj_target_dist[:, 0]

        # Log metrics to WandB
        if "log" not in self.extras:
            self.extras["log"] = dict()

        self.extras["log"]["reward_terms/approach"] = reward_1.mean().item()
        self.extras["log"]["reward_terms/target"] = reward_2.mean().item()
        self.extras["log"]["reward_terms/bonus"] = bonus_term.float().mean().item()
        self.extras["log"]["metrics/hand_obj_dist"] = hand_obj_dist[:, 0].mean().item()
        self.extras["log"]["metrics/obj_target_dist"] = obj_target_dist[:, 0].mean().item()
        self.extras["log"]["metrics/gen3_ee_vel_norm"] = ee_vel_norm.mean().item()
        self.extras["log"]["metrics/contact_forces"] = contacts_w.sum(-1).mean().item()
        self.extras["log"]["metrics/success_rate"] = self.obj_reached_target.float().mean().item()

        if self.cfg.vel_rew:
            self.extras["log"]["reward_terms/velocity_penalty"] = velocity_penalty.mean().item()

        if self.cfg.use_signal_1:
            self.extras["log"]["reward_terms/signal_1_firmness"] = signal_1_value.mean().item()

        if self.cfg.use_signal_2:
            self.extras["log"]["reward_terms/signal_2_balance"] = r_balance.mean().item()

        if self.cfg.use_signal_instability:
            self.extras["log"]["reward_terms/signal_instability"] = r_instability.mean().item()

        if self.cfg.use_force_rewards:
            self.extras["log"]["reward_terms/force_excess"] = r_excess.mean().item()

        self.extras["log"]["metrics/force_at_transition"] = (
            self.force_norms[:, 1:15][just_transitioned].sum(-1).mean().item() if just_transitioned.any() else 0.0)
        self.extras["log"]["metrics/velocity_at_transition"] = (
            ee_vel_norm[just_transitioned].mean().item() if just_transitioned.any() else 0.0)

        return reward


    # Verifies when to reset the environment --> Overrides method of DirecRLEnv
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        In:
            - None

        Out:
            - truncated - torch.tensor(N, 1): tensor of boolean indicating if the episodes was truncated (finished badly).
            - terminated - torch.tensor(N, 1): tensor of boolean indicating if the episodes was terminated (finished).
        '''

        # Computes time out indicators
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Checks out of bounds in velocity
        out_of_bounds_1 = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w.torch[:, self.ee_jacobi_idx[self.cfg.UR5e]+1, 7:], dim = -1) > self.cfg.velocity_limit
        out_of_bounds_2 = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w.torch[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 7:], dim = -1) > self.cfg.velocity_limit
        out_of_bounds = torch.logical_or(out_of_bounds_1, out_of_bounds_2)

        # Falling conditions
        GEN3_falling = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w.torch[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 2] < self.cfg.gen3_height_limit
        object_falling = self.scene.rigid_objects["object"].data.body_state_w.torch[:, 0, 2] < self.cfg.object_height_limit
        falling = torch.logical_or(GEN3_falling, object_falling)

        # Contact conditions
        GEN3_ground_contact = self.contacts[:, 0]

        # Truncated and terminated variables
        truncated = torch.logical_or(torch.logical_or(falling, out_of_bounds), GEN3_ground_contact)
        terminated = torch.logical_or(time_out, self.obj_reached_target)

        # Outcome classification for logging
        success = self.obj_reached_target & ~truncated
        indet = self.obj_reached_target & truncated
        fail = ~self.obj_reached_target & (terminated | truncated)

        if "log" not in self.extras:
            self.extras["log"] = dict()

        self.extras["log"]["metrics/outcome_success"] = success.float().mean().item()
        self.extras["log"]["metrics/outcome_indetermination"] = indet.float().mean().item()
        self.extras["log"]["metrics/outcome_fail"] = fail.float().mean().item()

        if "log" in self.extras:
            self.extras["log"]["metrics/out_of_bounds"] = out_of_bounds.float().mean().item()

        return truncated, terminated


    # Resets the index robot JOINT positions
    def reset_robot(self, idx, env_ids):
        '''
        In:
            - idx - int(0 or 1): index for the robot.
            - env_ids - torch.tensor(m): IDs for the 'm' environments that need to be resetted.

        Out:
            - None
        '''

        # Default joint position for the robots
        joint_pos = self.default_joint_pos[idx][env_ids]
        joint_vel = self.scene.articulations[self.cfg.keys[idx]].data.default_joint_vel.torch[env_ids]

        # Write the joint positions to the environments
        # NOTE (Isaac Lab 3.0 port): the combined `write_joint_state_to_sim` helper was
        # removed; it is now two separate index-suffixed calls.
        self.scene.articulations[self.cfg.keys[idx]].write_joint_position_to_sim_index(position=joint_pos, joint_ids=None, env_ids=env_ids)
        self.scene.articulations[self.cfg.keys[idx]].write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=None, env_ids=env_ids)


    # Resets the index robot according to their END EFFECTOR
    def reset_robot_ee(self, idx, env_ids):
        '''
        In:
            - idx - int(0 or 1): index for the robot.
            - env_ids - torch.tensor(m): IDs for the 'm' environments that need to be resetted.

        Out:
            - None
        '''

        # Sample a random position using the end effector ranges with the shape of all environmnets
        ee_init_pose = sample_uniform(
            self.ee_pose_ranges[idx, :, 0],
            self.ee_pose_ranges[idx, :, 1],
            [self.num_envs, self.ee_pose_ranges[idx, :, 0].shape[0]],
            self.device,
        )

        # Transforms Euler to quaternion
        quat = quat_from_euler_xyz(roll = ee_init_pose[:, 3],
                                    pitch = ee_init_pose[:, 4],
                                    yaw = ee_init_pose[:, 5])

        # Builds the new initial pose
        ee_init_pose = torch.cat((ee_init_pose[:, :3], quat), dim = -1)

        # Save sampled pose
        self.reset_robot_poses_r[idx][env_ids] = ee_init_pose[env_ids]

        # Sets the command to the DifferentialIKController
        self.controller.set_command(ee_init_pose)

        # Obtains current poses for the robot
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose(idx)

        # Obtains the joint positions to reset. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.reset_joint_positions[idx][env_ids] = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos),
                                         self.default_joint_pos[idx][:, (6+idx):]),
                                         dim=-1)[env_ids]
        joint_pos = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos),
                               self.default_joint_pos[idx][:, (6+idx):]),
                               dim=-1)[env_ids]

        # Obtains the joint velocities
        joint_vel = self.scene.articulations[self.cfg.keys[idx]].data.default_joint_vel.torch[env_ids]

        # Writes the state to the simulation
        self.scene.articulations[self.cfg.keys[idx]].write_joint_position_to_sim_index(position=joint_pos, joint_ids=None, env_ids=env_ids)
        self.scene.articulations[self.cfg.keys[idx]].write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=None, env_ids=env_ids)


    # Resets the simulation --> Overrides method of DirectRLEnv
    def _reset_idx(self, env_ids: Sequence[int] | None):
        '''
        In:
            - env_ids - Sequence(m): 'm' indexes of the environments that need to be resetted.

        Out:
            - None
        '''

        # Reset method from DirectRLEnv
        super()._reset_idx(env_ids)

        # Reset the count
        self.count = 0

        # --- Reset the robots ---
        # Reset the robots first to the default joint position so the IK is easier to compute after
        self.reset_robot(idx = self.cfg.UR5e, env_ids = env_ids)
        self.reset_robot(idx = self.cfg.GEN3, env_ids = env_ids)

        # Reset the robot to a random Euclidean position
        self.reset_robot_ee(idx = self.cfg.UR5e, env_ids = env_ids)
        self.reset_robot_ee(idx = self.cfg.GEN3, env_ids = env_ids)

        # --- Reset controller ---
        self.controller.reset()

        # --- Reset object ---
        # Obtains the end effector position for the UR5e
        ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w.torch[env_ids, self.ee_jacobi_idx[self.cfg.UR5e]+1, 0:7]

        # Transforms the translation and orientation of the object pose (in the end effector frame) to the world frame
        obj_pos, obj_quat = combine_frame_transforms(t01 = ee_pose_w[:, :3], q01 = ee_pose_w[:, 3:],
                                                     t12 = self.cfg.obj_pos_trans[env_ids], q12 = self.cfg.obj_quat_trans[env_ids])

        # Writes the new object position to the simulation
        self.scene.rigid_objects["object"].write_root_pose_to_sim_index(root_pose = torch.cat((obj_pos, obj_quat), dim = -1), env_ids = env_ids)

        # Updates the poses of the GEN3 end effector and the object in the reset
        self.update_new_poses()

        # Reset previous distances
        self.prev_dist[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.prev_dist_target[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.obj_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]
        self.obj_reached_target[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]

        self.contacts = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)

        self.dir = torch.randint(low = -1,high = 1,size=(6,)).to(self.device)*-2-1
