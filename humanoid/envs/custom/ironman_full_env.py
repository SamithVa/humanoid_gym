# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain

CONTACT_FORCE = 5.  # threshold for contact force to consider foot in contact
FEET_OFFSET = 0.02  # height of the ground plane

class IronmanFullFreeEnv(LeggedRobot):
    '''
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        
        # Curriculum learning initialization
        self.curriculum_stage = 1
        self.curriculum_iteration = 0
        self.episode_reward_sums = torch.zeros(self.num_envs, device=self.device)
        self.stage_start_iteration = 0
        
        # Print curriculum status
        if hasattr(cfg, 'curriculum') and cfg.curriculum.enabled:
            print(f"\n{'='*60}")
            print(f"CURRICULUM LEARNING ENABLED")
            print(f"Stage 1: {cfg.curriculum.stage1_iterations} iterations (LEGS ONLY - loin+arms frozen)")
            print(f"  → Trains exactly like lower-body config")
            print(f"Stage 2: {cfg.curriculum.stage2_iterations} iterations (loin+arms at {cfg.curriculum.stage2_arm_action_scale*100:.0f}%)")
            print(f"Stage 3: Full body control")
            print(f"{'='*60}\n")
        
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

        

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        """ Compute the gait phase based on the episode length and cycle time."""
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        """
        Compute a per-environment stance mask for left/right feet.

        Uses the gait phase (self._get_phase()) and sin(2*pi*phase) to determine which foot
        is in stance (1.0) or swing (0.0). A small window around the phase crossing
        (|sin| < 0.1) is treated as double-support and both feet are set to stance.

        Returns:
            torch.Tensor of shape (num_envs, 2) with float values 1.0 (stance) or 0.0 (swing).
            Column 0 = left foot stance flag, column 1 = right foot stance flag.

        Example:
            # assume num_envs == 3 and _get_phase() returns tensor([0.25, 0.75, 0.0])
            # sin_pos = [1.0, -1.0, 0.0]
            # stance_mask -> [[1,0], [0,1], [1,1]]
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    
    def _reward_shoulder_assist_balance(self):
        """
        Rewards shoulder movement that helps stabilize the base orientation.
        Isaac Gym Order: 0: l_shoulder_pitch, 15: r_shoulder_pitch
        """
        shoulder_pos = torch.stack([self.dof_pos[:, 0], self.dof_pos[:, 15]], dim=1)  # Both shoulder pitch joints
        pitch_error = torch.abs(self.base_euler_xyz[:, 1])
        
        # Reward: if pitch is high, shoulders should move; if pitch is low, shoulders should be small
        return torch.exp(-torch.abs(torch.norm(shoulder_pos, dim=1) - pitch_error) * 5)

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        # Isaac Gym Alphabetical Order:
        # 0-3: l_arm, 4: loin_yaw, 5-9: l_leg, 10-14: r_leg, 15-18: r_arm
        
        # --------------- LEFT LEG (indices 5-9) ---------------
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 5] = sin_pos_l * scale_1      # leg_l1 (hip pitch)
        self.ref_dof_pos[:, 8] = -sin_pos_l * scale_2     # leg_l4 (knee pitch)
        self.ref_dof_pos[:, 9] = sin_pos_l * scale_1      # leg_l5 (ankle pitch)
        
        # --------------- RIGHT LEG (indices 10-14) ---------------
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1    # leg_r1 (hip pitch)
        self.ref_dof_pos[:, 13] = sin_pos_r * scale_2     # leg_r4 (knee pitch)
        self.ref_dof_pos[:, 14] = -sin_pos_r * scale_1    # leg_r5 (ankle pitch)

        # --------------- LOIN + ARMS (curriculum-aware) ---------------
        if hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled:
            # Stage 1: No loin/arm reference (train like lower-body only)
            if self.curriculum_stage >= 2:
                # Stage 2+: Enable loin counter-rotation for balance
                self.ref_dof_pos[:, 4] = sin_pos_r * scale_1 * 0.5  # loin_yaw (index 4)
                
                # Arm reference motion
                arm_scale = scale_1 if self.curriculum_stage == 2 else scale_2
                self.ref_dof_pos[:, 15] = -sin_pos_l * arm_scale  # r_shoulder_pitch (index 15)
                self.ref_dof_pos[:, 0] = sin_pos_r * arm_scale    # l_shoulder_pitch (index 0)
        else:
            # No curriculum: always use full reference
            self.ref_dof_pos[:, 4] = sin_pos_r * scale_1 * 0.5    # loin_yaw
            self.ref_dof_pos[:, 15] = -sin_pos_l * scale_2        # r_shoulder_pitch
            self.ref_dof_pos[:, 0] = sin_pos_r * scale_2          # l_shoulder_pitch
        
        # Double support phase [NOTE]
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0 

        self.ref_action = 2 * self.ref_dof_pos

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands (sin, cos, lin vel x,y, ang vel yaw)
        noise_vec[5: 24] = noise_scales.dof_pos * self.obs_scales.dof_pos # dof pos , 19 actions
        noise_vec[24: 43] = noise_scales.dof_vel * self.obs_scales.dof_vel # dof vel , 19 actions
        noise_vec[43: 62] = 0.  # previous actions
        noise_vec[62: 65] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel (roll, pitch, yaw)
        noise_vec[65: 68] = noise_scales.quat * self.obs_scales.quat         # euler x,y,z (roll, pitch, yaw)
        return noise_vec

    def step(self, actions):
        # Compute reference state for rewards (must be before action masking)
        self.compute_ref_state()
        
        # Apply curriculum learning action masking
        if hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled:
            actions = self._apply_curriculum_action_mask(actions)

        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        
        return super().step(actions)
    
    def _apply_curriculum_action_mask(self, actions):
        """
        Apply action masking based on curriculum stage.
        Isaac Gym Order: 0-3: l_arm, 4: loin_yaw, 5-9: l_leg, 10-14: r_leg, 15-18: r_arm
        Stage 1: Freeze loin_yaw (4) + arms (0-3, 15-18) - train ONLY legs
        Stage 2: Allow partial arm movement, enable loin_yaw
        Stage 3: Full control
        """
        if self.curriculum_stage == 1:
            # Stage 1: Freeze loin_yaw AND arms - train EXACTLY like lower-body
            actions[:, 0:4] = 0.0    # Freeze left arm (indices 0-3)
            actions[:, 4] = 0.0      # Freeze loin_yaw (index 4)
            actions[:, 15:19] = 0.0  # Freeze right arm (indices 15-18)
        elif self.curriculum_stage == 2:
            # Stage 2: Enable loin_yaw, scale down arm actions
            actions[:, 0:4] *= self.cfg.curriculum.stage2_arm_action_scale    # Left arm
            actions[:, 15:19] *= self.cfg.curriculum.stage2_arm_action_scale  # Right arm
            
        # Stage 3: no modification, full control
        return actions
    
    def update_curriculum(self, current_iteration):
        """
        Update curriculum stage based on training iteration (NOT step count).
        Should be called once per training iteration by the training loop.
        
        Args:
            current_iteration: Current training iteration number
        """
        if not hasattr(self.cfg, 'curriculum') or not self.cfg.curriculum.enabled:
            return
        
        # Update current iteration
        self.curriculum_iteration = current_iteration
        iterations_in_stage = self.curriculum_iteration - self.stage_start_iteration
        
        # Log progress every 50 iterations
        if self.curriculum_iteration % 50 == 0:
            avg_reward = torch.mean(self.rew_buf).item()
            print(f"[CURRICULUM] Stage {self.curriculum_stage} | Iter: {iterations_in_stage}/{self._get_stage_target_iterations()} | Avg Reward: {avg_reward:.2f}")
        
        # Check if we should progress to next stage
        if self.curriculum_stage == 1:
            if iterations_in_stage >= self.cfg.curriculum.stage1_iterations:
                # Progress to stage 2 after sufficient iterations
                print(f"\n{'='*70}")
                print(f"[CURRICULUM] ✓ STAGE 1 COMPLETE - Advancing to Stage 2")
                print(f"Iterations: {iterations_in_stage} | Learned stable walking with legs")
                print(f"Now enabling: loin_yaw rotation + arms at 50% scale")
                print(f"{'='*70}\n")
                self.curriculum_stage = 2
                self.stage_start_iteration = self.curriculum_iteration
                    
        elif self.curriculum_stage == 2:
            if iterations_in_stage >= self.cfg.curriculum.stage2_iterations:
                # Progress to stage 3 after sufficient iterations
                print(f"\n{'='*70}")
                print(f"[CURRICULUM] ✓ STAGE 2 COMPLETE - Advancing to Stage 3")
                print(f"Iterations: {iterations_in_stage} | FULL BODY CONTROL ENABLED")
                print(f"{'='*70}\n")
                self.curriculum_stage = 3
                self.stage_start_iteration = self.curriculum_iteration
    
    def get_curriculum_stage(self):
        """Returns current curriculum stage for logging/monitoring"""
        if not hasattr(self.cfg, 'curriculum') or not self.cfg.curriculum.enabled:
            return 0
        return self.curriculum_stage


    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3 = 5
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 19
            self.dof_vel * self.obs_scales.dof_vel,  # 19
            self.actions,  # 19
            diff,  # 19
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1) 

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 19D
            dq,  # 19D
            self.actions,   # 19D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)


        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
    
    def _get_stage_target_iterations(self):
        """Helper to get target iterations for current stage"""
        if self.curriculum_stage == 1:
            return self.cfg.curriculum.stage1_iterations
        elif self.curriculum_stage == 2:
            return self.cfg.curriculum.stage2_iterations
        return float('inf')  # Stage 3 has no limit

# ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        Curriculum-aware: only penalizes leg joints in stage 1 (like lower-body training).
        Isaac Gym Order: 0-3: l_arm, 4: loin_yaw, 5-9: l_leg, 10-14: r_leg, 15-18: r_arm
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone() # target joint pos
        diff = joint_pos - pos_target
        
        # Apply curriculum-aware masking
        if hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled:
            if self.curriculum_stage == 1:
                # Stage 1: Only reward leg joint tracking (indices 5-14)
                # Exclude loin (4) and arms (0-3, 15-18) - EXACTLY like lower-body training
                diff[:, 0:4] = 0.0    # Ignore left arm
                diff[:, 4] = 0.0      # Ignore loin_yaw  
                diff[:, 15:19] = 0.0  # Ignore right arm
            # Stage 2+: Reward all joints
            
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    # TODO : test this reward
    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > CONTACT_FORCE
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > CONTACT_FORCE
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    # TODO : test this reward
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > CONTACT_FORCE
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        # print("contact: ", contact)
        # print("stance_mask: ", stance_mask)
        # print("reward: ", reward)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    # TODO: test this reward
    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.

        Returns:
            torch.Tensor (shape: num_envs,)
                Per-environment penalty computed as the sum over feet of the positive excess
                contact magnitude above cfg.rewards.max_contact_force:

                    penalty_env = sum_over_feet max(0, ||F_foot|| - max_contact_force)

                - ||F_foot|| is the Euclidean norm of the contact force vector (N) for that foot.
                - Values are in Newtons; larger values indicate larger penalty for hard impacts.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clamp(0, 150), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions.
        Isaac Gym Order: 5-9: l_leg (l1,l2,l3,l4,l5), 10-14: r_leg (r1,r2,r3,r4,r5)
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, 6:8]   # leg_l2 and leg_l3 (indices 6,7)
        right_yaw_roll = joint_diff[:, 11:13] # leg_r2 and leg_r3 (indices 11,12)
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        # if stance_mask.sum() == 2:
        #     print("feet z cord: ", self.rigid_state[:, self.feet_indices, 2], stance_mask) # [0.02, 0.05], this coordinate is relative to the feet origin?
        #     print("root z cord: ", self.root_states[:, 2]) # 0.454 -> 0.456
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        # base_height = self.root_states[:, 2] - (measured_heights - 0.05) # why there is a 0.05 offset here? NOTE
        base_height = self.root_states[:, 2] - (measured_heights - FEET_OFFSET)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    # TODO : test this reward
    # def _reward_feet_clearance(self): # ORIGINAL
    #     """
    #     Calculates reward based on the clearance of the swing leg from the ground during movement.
    #     Encourages appropriate lift of the feet during the swing phase of the gait.
    #     """
    #     # Compute feet contact mask (1 if in contact, 0 if not)
    #     contact = self.contact_forces[:, self.feet_indices, 2] > CONTACT_FORCE 
    #     # [num_envs, 2] e.g [[1,0]] -> left foot contact, right foot swing

    #     # Get the z-position of the feet and compute the change in z-position
    #     feet_z = self.rigid_state[:, self.feet_indices, 2] - FEET_OFFSET
    #     delta_z = feet_z - self.last_feet_z
    #     self.feet_height += delta_z
    #     self.last_feet_z = feet_z

    #     # Compute swing mask
    #     swing_mask = 1 - self._get_gait_phase()

    #     # feet height should be closed to target feet height at the peak
    #     rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
    #     rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    #     self.feet_height *= ~contact
    #     return rew_pos
    
    def _reward_feet_clearance(self): 
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Get the current z-position of the feet
        contact = self.contact_forces[:, self.feet_indices, 2] > CONTACT_FORCE
        feet_z = self.rigid_state[:, self.feet_indices, 2] # [0.02, 0.05]
        
        # Get swing mask (1 if foot is in swing phase, 0 if in stance)
        swing_mask = 1 - self._get_gait_phase()

        # Reward swinging feet for being near the target height.
        # use smoother reward function, Gaussian centered at target height 
        target_h = self.cfg.rewards.target_feet_height
        # print(f"feet z coord: {feet_z[0].cpu().numpy()}, target height: {target_h}, swing mask: {swing_mask[0].cpu().numpy()}")
        rew_pos = torch.exp(-((feet_z - target_h) ** 2) / (2 * 0.02 ** 2))   # sigma = 0.02 

        # Only reward feet that are swinging AND not in contact with ground
        rew_pos = rew_pos * swing_mask * (~contact).float()
        rew_pos = torch.sum(rew_pos, dim=1)

        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    # ============= Curriculum-Specific Rewards ============= #
    
    def _reward_arm_at_target_pos(self):
        """
        Reward for keeping arms at default position during stage 1.
        Isaac Gym Order: 0-3: l_arm, 15-18: r_arm
        """
        if not hasattr(self.cfg, 'curriculum') or not self.cfg.curriculum.enabled:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Only apply in stage 1
        if self.curriculum_stage == 1:
            # Check both left arm (0-3) and right arm (15-18)
            l_arm_diff = self.dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]
            r_arm_diff = self.dof_pos[:, 15:19] - self.default_dof_pos[:, 15:19]
            arm_pos_diff = torch.cat([l_arm_diff, r_arm_diff], dim=1)
            return torch.exp(-torch.norm(arm_pos_diff, dim=1) * 10)
        return torch.zeros(self.num_envs, device=self.device)
    
    def _reward_arm_smoothness(self):
        """
        Reward for smooth arm movements in stages 2 and 3.
        Isaac Gym Order: 0-3: l_arm, 15-18: r_arm
        """
        if not hasattr(self.cfg, 'curriculum') or not self.cfg.curriculum.enabled:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Apply in stages 2 and 3
        if self.curriculum_stage >= 2:
            # Check both left arm (0-3) and right arm (15-18)
            l_arm_diff = self.actions[:, 0:4] - self.last_actions[:, 0:4]
            r_arm_diff = self.actions[:, 15:19] - self.last_actions[:, 15:19]
            arm_action_diff = torch.cat([l_arm_diff, r_arm_diff], dim=1)
            smoothness_penalty = torch.sum(torch.square(arm_action_diff), dim=1)
            return torch.exp(-smoothness_penalty * 5)
        return torch.zeros(self.num_envs, device=self.device)
