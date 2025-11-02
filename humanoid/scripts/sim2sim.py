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


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import IronmanCfg
import torch
import csv
from datetime import datetime
import os


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

# def quaternion_to_euler_array(quat):
#     """Convert quaternion to Euler angles in ZYX order (yaw, pitch, roll)"""
#     # quat is in [x, y, z, w] format
#     r = R.from_quat(quat)
#     # as_euler('zyx') returns [yaw, pitch, roll] matching MuJoCo's convention
#     euler = r.as_euler('zyx', degrees=False)
#     return euler

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double) 
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    # print("Quaternion:", quat)  # Debugging: Print quaternion
    r = R.from_quat(quat)
    # print("Rotation Matrix:\n", r.as_matrix())  # Debugging: Print rotation matrix
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    # print("Linear Velocity in Base Frame:", v)  # Debugging: Print linear velocity
    omega = data.sensor('angular-velocity').data.astype(np.double)
    # print("Angular Velocity:", omega)  # Debugging: Print angular velocity
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

# Debugging 
# Function to retrieve the name of a body given its ID
def print_model_structure(model):
    """Prints the structure of the MuJoCo model."""
    print("="*30)
    print("MuJoCo Model Structure")
    print("="*30)

    print(f"Number of bodies: {model.nbody}")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  Body {i}: {name}")

    print("\n" + "-"*30 + "\n")

    print(f"Number of joints: {model.njnt}")
    
    print("\n" + "-"*30 + "\n")

    print(f"Number of geoms: {model.ngeom}")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"  Geom {i}: {name}")

    print("\n" + "-"*30 + "\n")

    print(f"Number of actuators: {model.nu}")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Actuator {i}: {name}")

    print("\n" + "-"*30 + "\n")

    print(f"Number of sensors: {model.nsensor}")
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        print(f"  Sensor {i}: {name}")
    
    print("\n" + "="*30)

def print_actuator_mapping(model):
    """Print actuator to joint mapping for debugging"""
    print("\nActuator Order (for kps/kds arrays):")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(f"  [{i}] {actuator_name} -> {joint_name}")
# End of Debugging

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)

    # print_actuator_mapping(model)  # Debugging: Print actuator mapping
    # print(f"Gravity: {model.opt.gravity}")  # Should be [0, 0, -9.81]
    # print(f"Mass : {[model.body_mass[i] for i in range(len(model.body_mass))]}")  # Check base_link mass
    # print(f"Total Mass: {np.sum(model.body_mass)}")  # Total mass of the robot

    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # Setup CSV logging
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'debug_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'sim2sim_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    csv_file = open(log_file, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write header
    header = ['timestep']
    # Observation components
    header.extend([f'obs_{i}' for i in range(cfg.env.num_single_obs)])
    # Action components
    header.extend([f'action_{i}' for i in range(cfg.env.num_actions)])
    # State information
    header.extend([f'q_{i}' for i in range(cfg.env.num_actions)])
    header.extend([f'dq_{i}' for i in range(cfg.env.num_actions)])
    header.extend([f'target_q_{i}' for i in range(cfg.env.num_actions)])
    header.extend([f'tau_{i}' for i in range(cfg.env.num_actions)])
    
    csv_writer.writerow(header)
    csv_file.flush()

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # state observation
            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:15] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 15:25] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 25:35] = action
            obs[0, 35:38] = omega
            obs[0, 38:41] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            # print(f"Observation at step {count_lowlevel}: {obs}")  # Debugging: Print observation

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            # print(f"policy Output at step {count_lowlevel}: {action}")  # Debugging: Print raw action
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            # init target_q with zero for first 500 steps
            if count_lowlevel < 300:
                target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
            else:
                target_q = action * cfg.control.action_scale



        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        # Log to CSV at each control step
        if count_lowlevel % cfg.sim_config.decimation == 0:
            row = [count_lowlevel]
            row.extend(obs[0, :].tolist())
            row.extend(action.tolist())
            row.extend(q.tolist())
            row.extend(dq.tolist())
            row.extend(target_q.tolist())
            row.extend(tau.tolist())
            csv_writer.writerow(row)
            csv_file.flush()

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()
    csv_file.close()
    print(f"\nDebug log saved to: {log_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(IronmanCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml' # use original terrain
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/i1/mjcf/i1_1101.xml'
            sim_duration = 2.0
            dt = 0.001
            decimation = 10

        class robot_config:
            
            # stiffness = {'2': 40., '1': 60., '3': 40.,
            #          '4': 60., '5': 15.}
            # damping = {'2': 5, '1': 5, '3':
            #         5, '4': 5, '5': 3}
            # actuator robot parameters
            kps = np.array([20, 60, 20, 40, 1, 20, 60, 20, 60, 1], dtype=np.double) # leg_roll, leg_pitch, leg_yaw, knee, ankle_pitch
            kds = np.array(np.ones(10) * 3, dtype=np.double)
            tau_limit = 200. * np.ones(10, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
