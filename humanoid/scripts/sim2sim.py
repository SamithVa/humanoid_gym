# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import argparse
import csv
import math
import os
from collections import deque
from datetime import datetime

import mujoco
import mujoco_viewer
import numpy as np
from tqdm import tqdm

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import IronmanCfg

from scipy.spatial.transform import Rotation as R


import torch


def get_obs(data):
    """Extract observation from mujoco data structure."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg):
    """Run the Mujoco simulation using the provided policy and configuration."""
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    debug_mode = cfg.sim_config.debug_mode
    csv_file, csv_writer = None, None
    if debug_mode:
        log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'debug_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'sim2sim_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        csv_file = open(log_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        header = ['timestep']
        header += [f'obs_{i}' for i in range(cfg.env.num_single_obs)]
        header += [f'action_{i}' for i in range(cfg.env.num_actions)]
        header += [f'q_{i}' for i in range(cfg.env.num_actions)]
        header += [f'dq_{i}' for i in range(cfg.env.num_actions)]
        header += [f'torque_{i}' for i in range(cfg.env.num_actions)]
        csv_writer.writerow(header)
        csv_file.flush()

    num_actions = cfg.env.num_actions
    num_single_obs = cfg.env.num_single_obs
    target_q = np.zeros(num_actions, dtype=np.double)
    action = np.zeros(num_actions, dtype=np.double)
    target_dq = np.zeros(num_actions, dtype=np.double)

    hist_obs = deque([np.zeros([1, num_single_obs], dtype=np.double) for _ in range(cfg.env.frame_stack)])
    count_lowlevel = 0
    num_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    for _ in tqdm(range(num_steps), desc="Simulating..."):
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-num_actions:]
        dq = dq[-num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, num_single_obs], dtype=np.float32)
            eu_ang = R.from_quat(quat).as_euler('xyz')
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            phase = 2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64
            obs[0, 0] = math.sin(phase)
            obs[0, 1] = math.cos(phase)
            obs[0, 2] = cfg.cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cfg.cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cfg.cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:15] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 15:25] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 25:35] = action
            obs[0, 35:38] = omega
            obs[0, 38:41] = eu_ang
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * num_single_obs:(i + 1) * num_single_obs] = hist_obs[i][0, :]
            
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q = action * cfg.control.action_scale

            if debug_mode:
                q_clipped_debug = np.clip(q, cfg.robot_config.q_limits_low, cfg.robot_config.q_limits_high)
                tau_log = pd_control(target_q, q_clipped_debug, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
                tau_log = np.clip(tau_log, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
                row = [count_lowlevel] + obs[0].tolist() + action.tolist() + q.tolist() + dq.tolist() + tau_log.tolist()
                csv_writer.writerow(row)
                csv_file.flush()

        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        
        # Enforce joint limits
        q_clipped = np.clip(q, cfg.robot_config.q_limits_low, cfg.robot_config.q_limits_high)
        if np.any(q != q_clipped):
            tau = pd_control(target_q, q_clipped, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()
    if debug_mode:
        csv_file.close()
        print(f"\nDebug log saved to: {log_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sim2sim deployment script.')
    parser.add_argument('--load_model', type=str, default='/home/embodied/wanshan/humanoid_gym/logs/Ironman_ppo/exported/policies/policy_1.pt',
                        help='Path to policy model')
    parser.add_argument('--terrain', action='store_true', help='Use terrain instead of plane')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with CSV logging')
    args = parser.parse_args()

    class Sim2simCfg(IronmanCfg):
        class sim_config:
            mujoco_model_path = (f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
                                 if args.terrain else
                                 f'{LEGGED_GYM_ROOT_DIR}/resources/robots/i1/mjcf/i1_1101.xml')
            sim_duration = 5.0
            dt = 0.005
            decimation = 4
            debug_mode = args.debug

        class robot_config:
            kps = np.array([40, 18, 18, 40, 40, 40, 18, 18, 40, 40], dtype=np.double)
            kds = np.array([3, 3, 3, 3, 10, 3, 3, 3, 3, 10], dtype=np.double)
            tau_limit = 20.0 * np.ones(10, dtype=np.double)
            # Joint position limits [rad]: (low, high) for each joint
            q_limits_low = np.array([-0.9, -0.1, -0.8, -0.1, -0.2, -0.9, -0.1, -0.8, -0.1, -0.2], dtype=np.double)
            q_limits_high = np.array([0.7, 0.7, 0.8, 1.6, 0.2, 0.7, 0.7, 0.8, 1.6, 0.2], dtype=np.double)
            
        class cmd:
            vx = 0.2
            vy = 0.0
            dyaw = 0.0

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
