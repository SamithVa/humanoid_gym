"""
Test script to verify each joint's effect on the robot.
This helps identify URDF issues by systematically testing each joint.
"""

import numpy as np
from isaacgym import gymapi, gymutil
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from humanoid.envs.custom.ironman_full_config import IronmanFullCfg

def test_joint_effects():
    """Test each joint individually to verify URDF correctness"""
    
    # Test mode: 
    # True = Fixed base (good for verifying joint axes/directions)
    # False = Free base (good for detecting inertia problems - watch for spinning/tilting)
    FIX_BASE = False  # Set to False to test inertia
    
    # Initialize Isaac Gym
    gym = gymapi.acquire_gym()
    
    # Parse arguments
    args = gymutil.parse_arguments(description="Joint Effect Tester")
    
    # Create simulation
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.01  # 100 Hz (10x faster than 1000 Hz)
    sim_params.substeps = 2  # More substeps for stability at higher dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.use_gpu = True  # Enable GPU acceleration
    
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, 
                         gymapi.SIM_PHYSX, sim_params)
    
    # Create ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)
    
    # Load robot asset
    cfg = IronmanFullCfg()
    asset_root = os.path.join(os.path.dirname(__file__), '../resources/robots/i1')
    asset_file = "urdf/i1_full_body.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    asset_options.collapse_fixed_joints = False
    asset_options.fix_base_link = FIX_BASE
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 100.0
    
    print(f"\nLoading robot from: {os.path.join(asset_root, asset_file)}")
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    # Get DOF names and properties
    dof_names = gym.get_asset_dof_names(robot_asset)
    num_dofs = gym.get_asset_dof_count(robot_asset)
    
    print(f"\n{'='*80}")
    print(f"ROBOT DOF INFORMATION")
    print(f"{'='*80}")
    print(f"Total DOFs: {num_dofs}")
    print(f"\nJoint order from URDF:")
    for i, name in enumerate(dof_names):
        print(f"  [{i:2d}] {name}")
    
    # Create environment
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    # Add robot to environment - spawn in air to avoid ground contact during testing
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.48)  # 0.48 meters above ground
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    actor_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 1)
    
    # Set DOF properties
    dof_props = gym.get_actor_dof_properties(env, actor_handle)
    for i in range(num_dofs):
        dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        dof_props['stiffness'][i] = 50.0
        dof_props['damping'][i] = 2.0
    gym.set_actor_dof_properties(env, actor_handle, dof_props)
    
    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        return
    
    # Set camera
    cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    print(f"\n{'='*80}")
    print(f"JOINT EFFECT TEST - AUTONOMOUS MODE")
    print(f"{'='*80}")
    print(f"Mode: {'FIXED BASE (testing joint axes)' if FIX_BASE else 'FREE BASE (testing inertia)'}")
    print(f"Testing each joint for 0.5 seconds, then auto-advancing...")
    if not FIX_BASE:
        print(f"Watch for: spinning, tilting, or unstable motion = inertia problem!")
    print(f"Close window to quit")
    print(f"{'='*80}\n")
    
    # Test all joints (arms, loin, legs)
    # Legs will be kept at initial position for stability
    joints_to_test = list(range(num_dofs))
    
    current_joint_idx = 0
    current_joint = joints_to_test[current_joint_idx]
    test_angle = 0.5  # radians (~28 degrees) - larger for visibility
    hold_time = 0.5  # seconds per joint (faster!)
    frame_count = 0
    frames_to_hold = int(hold_time / sim_params.dt)
    
    # Find symmetric joint pairs (left/right arm joints)
    # Based on alphabetical ordering: l_arm (0-3), r_arm (15-18)
    symmetric_pairs = {
        0: 15,  # l_shoulder_pitch <-> r_shoulder_pitch
        1: 16,  # l_shoulder_roll <-> r_shoulder_roll  
        2: 17,  # l_shoulder_yaw <-> r_shoulder_yaw
        3: 18,  # l_arm_pitch <-> r_arm_pitch
    }
    
    print(f"\n>>> Testing Joint [{current_joint}]: {dof_names[current_joint]}")
    
    # Get initial joint positions - legs will stay at these positions
    initial_dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
    initial_positions = initial_dof_states['pos'].copy()
    
    # Leg joint indices (alphabetical): l_leg (5-9), r_leg (10-14)
    leg_joints = list(range(5, 15))
    
    while not gym.query_viewer_has_closed(viewer):
        
        # Start with initial positions (keeps legs stable)
        dof_positions = initial_positions.copy().astype(np.float32)
        
        # Modify the test joint (unless it's a leg joint)
        if current_joint not in leg_joints:
            progress = frame_count / frames_to_hold
            dof_positions[current_joint] = test_angle * np.sin(2 * np.pi * progress * 2)  # 2 full cycles
            
            # If testing an arm joint, move the symmetric joint too for balance
            if current_joint in symmetric_pairs:
                symmetric_joint = symmetric_pairs[current_joint]
                dof_positions[symmetric_joint] = test_angle * np.sin(2 * np.pi * progress * 2)
        
        gym.set_actor_dof_position_targets(env, actor_handle, dof_positions)
        
        # Display info every 50 frames
        if frame_count % 50 == 0:
            dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
            actual_pos = dof_states['pos'][current_joint]
            target_pos = dof_positions[current_joint]
            actual_vel = dof_states['vel'][current_joint]
            joint_info = f"Joint [{current_joint:2d}] {dof_names[current_joint]:25s}"
            
            # Add symmetric joint info if applicable
            if current_joint in symmetric_pairs:
                sym_joint = symmetric_pairs[current_joint]
                joint_info += f" + [{sym_joint:2d}] {dof_names[sym_joint]:25s}"
            
            print(f"  Time: {frame_count*sim_params.dt:4.1f}s | {joint_info} | "
                  f"Target: {target_pos:+.3f} rad | Actual: {actual_pos:+.3f} rad | Vel: {actual_vel:+.3f} rad/s")
        
        # Step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        frame_count += 1
        
        # Auto-advance to next joint after hold time
        if frame_count >= frames_to_hold:
            current_joint_idx = (current_joint_idx + 1) % len(joints_to_test)
            current_joint = joints_to_test[current_joint_idx]
            frame_count = 0
            if current_joint_idx == 0:
                print(f"\n{'='*80}")
                print(f"COMPLETED FULL CYCLE - Starting over...")
                print(f"{'='*80}\n")
            print(f"\n>>> Testing Joint [{current_joint}]: {dof_names[current_joint]}")
    
    print("\nCleaning up...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done!")

if __name__ == "__main__":
    test_joint_effects()
