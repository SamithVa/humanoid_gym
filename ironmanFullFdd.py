import numpy as np
import pinocchio
import crocoddyl
import time
from pinocchio import visualize
from pinocchio.robot_wrapper import RobotWrapper

def estimate_scales(robot_model, rdata, q0_standing):
    import numpy as np
    import pinocchio

    leftFoot = 'leg_l5'   # or rightFoot
    frame_id = robot_model.getFrameId(leftFoot)

    # base config used in your script (q0_init / q0_standing)
    q0 = q0_standing.copy()    # your standing configuration used for display
    # compute baseline foot pos
    pinocchio.forwardKinematics(robot_model, rdata, q0)
    pinocchio.updateFramePlacements(robot_model, rdata)
    pos0 = rdata.oMf[frame_id].translation.copy()

    desired_dz = 0.06   # target vertical lift (m), adjust to cfg.rewards.target_feet_height
    dq = 1e-6           # small finite diff step (rad)
    joint_start = 7     # in your script leg joints are q[7:17]

    # compute dz/dq for each leg joint (single-joint sensitivity)
    sens = []
    for qi in range(joint_start, joint_start + 10):
        q_plus = q0.copy()
        q_plus[qi] += dq
        pinocchio.forwardKinematics(robot_model, rdata, q_plus)
        pinocchio.updateFramePlacements(robot_model, rdata)
        pos_plus = rdata.oMf[frame_id].translation
        dz_dq = (pos_plus[2] - pos0[2]) / dq
        sens.append(dz_dq)

    sens = np.array(sens)  # shape (10,)
    print("dz/dq per joint (m/rad):", sens)

    # single-joint estimate (avoid near-zero sensitivities)
    for i, s in enumerate(sens):
        if abs(s) < 1e-8:
            print(f"joint {i} nearly no effect on z")
        else:
            required_delta_q = desired_dz / s
            print(f"joint {i} (q index {joint_start+i}): required delta_q = {required_delta_q:.4f} rad")

    # multi-joint least-squares: use the 10 sensitivities as a 1x10 row (Jz)
    Jz = sens.reshape(1, -1)  # (1,10)
    if np.linalg.norm(Jz) > 1e-8:
        # minimum-norm delta_q solving Jz @ dq = desired_dz
        dq_multi = (Jz.T * desired_dz) / (Jz @ Jz.T)
        print("multi-joint delta_q (rad):", dq_multi.ravel())
        # suggested scale_1 if you move only hip/ankle (choose entries corresponding to those joints)
    else:
        print("Jacobian too small, try different base pose or consider larger movement")

    HIP_IDX = 0        # dq for left hip pitch (q7)
    ANKLE_IDX = 4      # dq for left ankle pitch (q11)  (may be 4 or different; adjust to your mapping)
    KNEE_IDX = 3       # dq for left knee (q10)

    raw = np.abs(dq_multi.ravel())          # make sure shape (10,)
    # choose scale_1 as the amplitude for hip/ankle motion:
    # option A: use hip amplitude
    scale_1_candidate = raw[HIP_IDX]

    # option B: use max of hip and ankle (safer if both contribute)
    scale_1_candidate = max(raw[HIP_IDX], raw[ANKLE_IDX])

    # enforce joint limits and safety factor
    joint_limits_low = robot_model.lowerPositionLimit[joint_start:joint_start+10]
    joint_limits_high = robot_model.upperPositionLimit[joint_start:joint_start+10]
    # maximum allowed delta from current angle (avoid exceeding limits)
    current_angles = q0[joint_start:joint_start+10]
    max_allowed = np.minimum(current_angles - joint_limits_low, joint_limits_high - current_angles)
    max_allowed = np.abs(max_allowed)

    # clamp candidate to allowed range (avoid > limit)
    scale_1_clamped = min(scale_1_candidate, max_allowed[HIP_IDX], max_allowed[ANKLE_IDX])

    SAFETY = 0.5    # scale down to be conservative (pick 0.3..0.7)
    scale_1 = float(scale_1_clamped * SAFETY)

    # determine knee amplitude; you can keep scale_2 = 2*scale_1 or use measured knee dq
    scale_2 = float(2.0 * scale_1)
    # or use measured knee: scale_2 = float(min(raw[KNEE_IDX]*SAFETY, max_allowed[KNEE_IDX]))

    print("suggested scale_1:", scale_1, " scale_2:", scale_2)

# --- Configuration ---
# URDF model path
modelPath = './resources/robots/i1/'
URDF_FILENAME = "urdf/i1_full_body_balanced.urdf"

# --- Model Loading ---
# Load the full model with a floating base (JointModelFreeFlyer)
robot = RobotWrapper.BuildFromURDF(
    modelPath + URDF_FILENAME,
    [modelPath],
    pinocchio.JointModelFreeFlyer()
)

robot_model = robot.model

# Define frame names for the feet (used for display and kinematics)
rightFoot = 'leg_r5'
leftFoot = 'leg_l5'

# --- Display Setup ---
# Initialize Meshcat display for the robot, tracking the specified foot frames
display = crocoddyl.MeshcatDisplay(
    robot,
    # frameNames=[rightFoot, leftFoot]
)

# --- Initial State Display and Kinematics ---

# Create a zero configuration vector
q0_init = pinocchio.utils.zero(robot_model.nq)
# Perform forward kinematics and update frame placements for the initial configuration
rdata = robot_model.createData()
pinocchio.forwardKinematics(robot_model, rdata, q0_init)
pinocchio.updateFramePlacements(robot_model, rdata)

# Get the frame IDs for the feet
rfId = robot_model.getFrameId(rightFoot)
lfId = robot_model.getFrameId(leftFoot)

# Store the initial position of the feet
rfFootPos0 = rdata.oMf[rfId].translation
lfFootPos0 = rdata.oMf[lfId].translation

# Calculate and store the initial center of mass (CoM) position
comRef = pinocchio.centerOfMass(robot_model, rdata, q0_init)

# --- Define Base Pose (Standing Configuration) ---
# A vector for the initial joint angles:
# 1 loin_yaw + 10 leg joints + 8 arm joints = 19 DoFs total
initialAngle_legs = np.zeros(10)
initialAngle_arms = np.zeros(8)  # 8 arm joints at neutral

# Define the base 'standing' configuration (q0)
q0_standing = pinocchio.utils.zero(robot.model.nq)
q0_standing[6] = 1.0     # Quaternion w component (q.w) for identity rotation
q0_standing[2] = 0.475  # Initial vertical position (z)
q0_standing[7] = 0.0     # loin_yaw_joint at neutral
q0_standing[8:18] = initialAngle_legs  # Set the leg joint angles
q0_standing[18:26] = initialAngle_arms  # Set the arm joint angles

# Display the robot in the defined standing configuration
print("\nDisplaying robot in standing configuration...")
display.display([q0_standing])

print("\nRobot model has {} DoFs and {} joints.".format(robot.model.nq, robot.model.njoints))
# --- Joint Exploration Loop 1: Displaying each joint one by one ---
# Loop through all 19 movable joints (loin_yaw + 10 legs + 8 arms)
print("\nExploring positive displacement for each joint...")
joint_names = ['loin_yaw'] + [f'leg_joint_{i}' for i in range(10)] + [f'arm_joint_{i}' for i in range(8)]
for i in range(19):
    q_explore = pinocchio.utils.zero(robot.model.nq)
    q_explore[6] = 1.0  # Identity rotation
    q_explore[2] = 0.475  # Base z position at standing height
    q_explore[i + 7] = 0.5  # Set the i-th movable joint to 0.5 radian
    print(f"  Joint {i}: {joint_names[i]}")
    display.display([q_explore])
# --- Animation Loop: Full Body Sinusoidal 'Walk' with Arms ---
dt = 0.001  # Time step for each display update
cycle_time = 0.64  # Duration of one gait cycle (seconds)
frame_rate = 50  # Display at 50 FPS

print("\nStarting full-body sinusoidal walking animation (loin + legs + arms)...")
print(f"Animation: {5000 * dt:.1f} seconds of motion at {frame_rate} FPS")
for i in range(5000): # Animation frames
    # Time progression
    phase = i * dt / cycle_time
    # Sinusoidal trajectory for joint displacement
    sin_pos = np.sin(2 * np.pi * phase)
    sin_pos_l = sin_pos.copy()
    sin_pos_r = sin_pos.copy()

    # Joint positions vector (19 DoFs: Pinocchio alphabetical order)
    # Pinocchio order: [l_arm(0-3), loin(4), l_leg(5-9), r_leg(10-14), r_arm(15-18)]
    # Training order: [loin(0), l_leg(1-5), r_leg(6-10), arms(11-18)]
    ref_dof_pos = pinocchio.utils.zero(19)
    
    # Scaling factors for joint movement
    scale_1 = 0.3  # For hip/ankle pitch joints (increased from 0.2 for more visible motion)
    scale_2 = 2 * scale_1  # For knee joints

    # --- Loin Movement (Counter-rotation for balance) ---
    # Pinocchio index 4 = loin_yaw
    ref_dof_pos[4] = 0.0  # loin_yaw - FIXED at 0 for now

    # --- Left Leg Movement (Swing Phase when sin_pos_l < 0) ---
    # Left foot stance phase when sin_pos_l > 0
    sin_pos_l_clamped = sin_pos_l.copy()
    if sin_pos_l_clamped > 0:
        sin_pos_l_clamped = 0.0

    # Apply motion to left leg joints - Pinocchio indices 5-9
    ref_dof_pos[5] = sin_pos_l_clamped * scale_1 + initialAngle_legs[0]   # leg_l1 (hip pitch)
    ref_dof_pos[6] = 0.0 + initialAngle_legs[1]                           # leg_l2 (hip roll) - keep neutral
    ref_dof_pos[7] = 0.0 + initialAngle_legs[2]                           # leg_l3 (hip yaw) - keep neutral
    ref_dof_pos[8] = -sin_pos_l_clamped * scale_2 + initialAngle_legs[3]  # leg_l4 (knee pitch)
    ref_dof_pos[9] = sin_pos_l_clamped * scale_1 + initialAngle_legs[4]   # leg_l5 (ankle pitch)

    # --- Right Leg Movement (Swing Phase when sin_pos_r > 0) ---
    # Right foot stance phase when sin_pos_r < 0
    sin_pos_r_clamped = sin_pos_r.copy()
    if sin_pos_r_clamped < 0:
        sin_pos_r_clamped = 0.0

    # Apply motion to right leg joints - Pinocchio indices 10-14
    ref_dof_pos[10] = -sin_pos_r_clamped * scale_1 + initialAngle_legs[5]  # leg_r1 (hip pitch)
    ref_dof_pos[11] = 0.0 + initialAngle_legs[6]                           # leg_r2 (hip roll) - keep neutral
    ref_dof_pos[12] = 0.0 + initialAngle_legs[7]                           # leg_r3 (hip yaw) - keep neutral
    ref_dof_pos[13] = sin_pos_r_clamped * scale_2 + initialAngle_legs[8]   # leg_r4 (knee pitch)
    ref_dof_pos[14] = -sin_pos_r_clamped * scale_1 + initialAngle_legs[9]  # leg_r5 (ankle pitch)

    # --- Arm Movement (Natural walking swing) ---
    # Arms swing opposite to legs: right arm forward when left leg forward
    arm_scale = scale_2  # Same amplitude as knee movement
    # Left arm: Pinocchio indices 0-3
    ref_dof_pos[0] = sin_pos_r * arm_scale   # left_shoulder_pitch (opposite to right leg)
    ref_dof_pos[1] = 0.0  # left_shoulder_roll
    ref_dof_pos[2] = 0.0  # left_shoulder_yaw
    ref_dof_pos[3] = 0.0  # left_arm_pitch
    
    # Right arm: Pinocchio indices 15-18
    ref_dof_pos[15] = -sin_pos_l * arm_scale  # right_shoulder_pitch (opposite to left leg)
    ref_dof_pos[16] = 0.0  # right_shoulder_roll
    ref_dof_pos[17] = 0.0  # right_shoulder_yaw
    ref_dof_pos[18] = 0.0  # right_arm_pitch

    # --- Double Support Phase Cleanup ---
    # During transition, reset leg reference positions
    if np.abs(sin_pos) < 0.1:
        ref_dof_pos[:] = 0.0

    # --- Display Current State ---
    q_step = pinocchio.utils.zero(robot.model.nq)
    q_step[6] = 1.0  # Identity rotation
    q_step[2] = 0.475  # Base z at standing height
    q_step[7:26] = ref_dof_pos  # Set all 19 joint angles (loin + legs + arms)
    display.display([q_step])
    
    # Control display speed - run at ~50 FPS for smooth visualization
    time.sleep(1.0 / frame_rate)
    
    # Progress indicator
    if i % 100 == 0:
        print(f"  Frame {i}/5000 ({i*dt:.2f}s / {5000*dt:.1f}s)")
        # Debug: Print joint values at this frame
        if i % 500 == 0:
            print(f"    Left leg pitch (Pinocchio index 5): {ref_dof_pos[5]:.3f}")
            print(f"    Left knee (Pinocchio index 8): {ref_dof_pos[8]:.3f}")
            print(f"    Right leg pitch (Pinocchio index 10): {ref_dof_pos[10]:.3f}")
            print(f"    Right knee (Pinocchio index 13): {ref_dof_pos[13]:.3f}")

print("\nAnimation complete!")

# --- Joint Exploration Loop 2: Repeating the single-joint display ---
print("\nRepeating exploration of all 19 joints...")
for i in range(19):
    q_explore = pinocchio.utils.zero(robot.model.nq)
    q_explore[6] = 1.0  # Identity rotation
    q_explore[2] = 0.475  # Base z at standing height
    q_explore[i + 7] = 0.5  # Set the i-th movable joint
    print(f"  Joint {i}: {joint_names[i]}")
    display.display([q_explore])
    time.sleep(0.5)  # Pause 0.5s between each joint display

print("\nVisualization complete!")

