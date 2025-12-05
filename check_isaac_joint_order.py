import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
import torch

# Initialize gym
gym = gymapi.acquire_gym()

# Create sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Load robot
asset_root = "./resources/robots/i1/"
asset_file = "urdf/i1_full_body_balanced.urdf"
asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
asset_options.fix_base_link = False

robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Get DOF names
num_dofs = gym.get_asset_dof_count(robot_asset)
print(f"\nIsaac Gym Joint Order (Total: {num_dofs} DOFs):")
print("="*60)
for i in range(num_dofs):
    dof_name = gym.get_asset_dof_name(robot_asset, i)
    print(f"Index {i}: {dof_name}")

print("\n" + "="*60)
print("Expected training order:")
print("Index 0: loin_yaw_joint")
print("Index 1-5: leg_l1-l5 (left leg)")
print("Index 6-10: leg_r1-r5 (right leg)")
print("Index 11-18: arms")

gym.destroy_sim(sim)
