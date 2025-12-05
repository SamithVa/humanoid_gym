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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class IronmanFullCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 68 # 5(cmd) + 19(q) + 19(dq) + 19(act) + 3(ang_vel) + 3(euler) = 68
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 101 # 5+19+19+19+19+3+3+3+2+3+1+1+2+2 = 101
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 19
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0 
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/i1/urdf/i1_full_body.urdf'
        name = "i1_1101_full_body"

        foot_name = "5" # the fifth link e.g "leg_l5"
        knee_name = "4" # the fourth link e.g "leg_l4"

        terminate_after_contacts_on = ['base_link', 'head'] 
        penalize_contacts_on = ['base_link', 'head'] 
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter, TODO
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class curriculum:
        """Curriculum learning configuration for progressive skill learning"""
        enabled = True  # Enable curriculum learning
        
        # Stage 1: Learn basic walking with legs only (arms frozen)
        # Note: iterations = training iterations (60 steps @ 4096 envs each)
        stage1_iterations = 1000  # ~1000 training iterations for basic walking
        stage1_reward_threshold = 50.0  # Not used - kept for reference
        
        # Stage 2: Introduce small arm movements
        stage2_iterations = 1000  # Gradual arm introduction
        stage2_arm_action_scale = 0.5  # Less restrictive (was 0.3)
        stage2_reward_threshold = 60.0  # Not used - kept for reference
        
        # Stage 3: Full arm control enabled
        stage3_arm_action_scale = 1.0  # Full action scale
        
        # Arm joint indices (Isaac Gym ALPHABETICAL ORDER!)
        # 0-3: l_arm, 4: loin_yaw, 5-9: l_leg, 10-14: r_leg, 15-18: r_arm
        arm_joint_start_idx = 0  # Left arm starts at index 0 (alphabetical)
        arm_joint_end_idx = 4    # Up to (not including) index 4 (loin_yaw)
        # Note: Right arm is 15-19 (handled separately in code)
        
        # Reward scaling per stage
        leg_reward_scale_stage1 = 1.0
        arm_reward_scale_stage1 = 0.0  # No arm reward in stage 1
        arm_reward_scale_stage2 = 0.3  # Partial arm reward in stage 2
        arm_reward_scale_stage3 = 1.0  # Full arm reward in stage 3

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.48] # x,y,z [m]

        # initialAngle = np.array([0.6,0.,0.,-0.9,0.28,0.6,0.,0.,-0.9,0.28])

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # ISAAC GYM ALPHABETICAL ORDER (critical!)
            'l_shoulder_pitch_joint': 0.,  # [0]
            'l_shoulder_roll_joint': 0.,   # [1]
            'l_shoulder_yaw_joint': 0.,    # [2]
            'l_arm_pitch_joint': 0.,       # [3]
            
            'loin_yaw_joint': 0.,          # [4] loin_yaw_joint [-0.5,0.5]
            
            'leg_l1_joint': 0.,            # [5] left_leg_pitch_joint [-0.9,0.9]
            'leg_l2_joint': 0.,            # [6] left_leg_roll_joint [-0.05,0.7]
            'leg_l3_joint': 0.,            # [7] left_leg_yaw_joint [-0.8,0.8]
            'leg_l4_joint': 0.,            # [8] left_knee_joint [-0.1,1.8]
            'leg_l5_joint': 0.,            # [9] left_ankle_pitch_joint [-0.9,0.3]

            'leg_r1_joint': 0.,            # [10] right_leg_pitch_joint [-0.7,1.0]
            'leg_r2_joint': 0.,            # [11] right_leg_roll_joint [-0.9,0.7]
            'leg_r3_joint': 0.,            # [12] right_leg_yaw_joint [-0.8,0.8]
            'leg_r4_joint': 0.,            # [13] right_knee_joint [-1.6,0.1] 
            'leg_r5_joint': 0.,            # [14] right_ankle_pitch_joint [-0.2,1.]

            'r_shoulder_pitch_joint': 0.,  # [15]
            'r_shoulder_roll_joint': 0.,   # [16]
            'r_shoulder_yaw_joint': 0.,    # [17]
            'r_arm_pitch_joint': 0.,       # [18]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters: #TODO
        # high stiffness for precise tracking, but big spikes in torque
        # low stiffness for smooth torque, but less precise tracking

        stiffness = {'1': 35, '2': 20, '3': 20,'4': 35, '5': 2, 
                     'loin_yaw': 2, 
                     'shoulder_pitch': 30, 'shoulder_roll': 10, 'shoulder_yaw': 10, 'arm_pitch': 10}

        damping = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0, 
               'loin_yaw': 1.0, 
               'shoulder_pitch': 1.0, 'shoulder_roll': 1.0, 'shoulder_yaw': 1.0, 'arm_pitch': 1.0}  

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz, which mean the physics engine updates the robot's state every 0.001 seconds.
        substeps = 1 
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-3, 3]  # [kg]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.46 # from base_link to ground # 0.42941 m (walking init), 0.4566
        min_dist = 0.2 # feet min distance to other feet (standing: 0.23m, allow 0.18m for dynamic walking) # NOTE (original 0.2)
        max_dist = 0.5 # feet max distance to other feet (standing: 0.23m, allow up to 0.32m for extended stride) # NOTE (original 0.5)
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.20    # (rad) # original 0.26 （Best result 0.15)
        target_feet_height = 0.06       # original 0.06       # (m) target feet height when swinging
        cycle_time = 0.64          # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 200  # [N] Forces above this value are penalized # TODO According to the robot weight, let assume our robot is 10kg, peak around 50-150N

        class scales: # reward scales
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2 
            # gait
            feet_air_time = 1.
            foot_slip = -0.05 # penalize foot slip
            # foot_slip = -0.2 # penalize foot slip
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01 # penalize high contact forces
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4 # penalize high joint velocity
            dof_acc = -1e-7 # penalize high joint acceleration
            collision = -1.

            # UPPER BODY REWARDS
            shoulder_assist_balance = 0.05
            # Curriculum-based rewards
            arm_at_target_pos = 0.3  # Reward for keeping arms at default pos (stage 1)
            arm_smoothness = 0.2  # Reward for smooth arm movements (stage 2+)

    

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class IronmanFullCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]  # Deeper for 19 DOF (was [512, 256, 128])
        critic_hidden_dims = [768, 256, 128]  # Keep critic deep for value estimation

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001  # Encourages exploration
        learning_rate = 1e-3  # Higher learning rate for std decay (was 5e-5)
        schedule = 'adaptive'  # Adaptive learning rate based on KL divergence
        desired_kl = 0.01  # Target KL divergence for adaptive schedule
        num_learning_epochs = 5  # More epochs for better std parameter updates (was 2)
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        max_grad_norm = 1.0  # Gradient clipping for stability

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'Ironman_ppo_full_body'
        run_name = ''
        
        # Video recording during training
        video_interval = None  # Record video every N iterations (set to None to disable)
        
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
