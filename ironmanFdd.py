import numpy as np
import pinocchio
import crocoddyl
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
modelPath = './resources/robots/body_0926_walk_init/'
URDF_FILENAME = "urdf/body_0926.urdf"

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
# A vector for the initial joint angles (10 DoFs for the legs)
initialAngle = np.array([0.35,0.,0.,-0.7,0.35,0.35,0.,0.,-0.7,0.35])
# initialAngle = initialAngle * -1  # Invert the angles for correct orientation

# Define the base 'standing' configuration (q0)
q0_standing = pinocchio.utils.zero(robot.model.nq)
q0_standing[6] = 1.0     # Quaternion w component (q.w) for identity rotation
q0_standing[2] = 0.475  # Initial vertical position (z)
q0_standing[7:17] = initialAngle  # Set the leg joint angles

# Display the robot in the defined standing configuration
print("\nDisplaying robot in standing configuration...")
display.display([q0_standing])

print("\nRobot model has {} DoFs and {} joints.".format(robot.model.nq, robot.model.njoints))
# --- Joint Exploration Loop 1: Displaying each joint one by one ---
# Loop through the 10 movable joints (indices 7 to 16)
print("\nExploring positive displacement for each joint...")
for i in range(robot.model.nq - 7):
    q_explore = pinocchio.utils.zero(robot.model.nq)
    q_explore[6] = 1.0  # Identity rotation
    q_explore[2] = 0.0  # Base z position (for simplicity in this loop)
    q_explore[i + 7] = 1.0  # Set the i-th movable joint to 1 radian/meter
    display.display([q_explore])
    # The original code had a commented print(1) here.

# --- Animation Loop: Simple Sinusoidal 'Walk' ---
dt = 0.001  # Time step for each display update
cycle_time = 0.64  # Duration of one gait cycle (seconds)

print("\nStarting simple sinusoidal walking animation...")
for i in range(5000): # i here is episode_length_buff
    # Time progression (0.005 seconds per step)
    phase = i * dt / cycle_time # dt/cycle_time, cycle_time=0.64, dt=0.001, dt/cycle_time=0.001/0.64=0.0015625
    # Sinusoidal trajectory for joint displacement
    sin_pos = np.sin(2 * np.pi * phase)
    sin_pos_l = sin_pos.copy()
    sin_pos_r = sin_pos.copy()

    # Joint positions vector (1x10 for the 10 leg joints)
    ref_dof_pos = pinocchio.utils.zero(10)
    # Scaling factors for joint movement
    scale_1 = 0.2  # For hip/ankle pitch joints
    scale_2 = 2 * scale_1  # For knee joints

    # --- Left Leg Movement (Swing Phase when sin_pos_l < 0) ---
    # Left foot is considered in 'stance' (default pos) when sin_pos_l > 0
    if sin_pos_l > 0:
        sin_pos_l = 0.0 # Stance phase: keep joint at zero

    # Apply motion to left leg joints (indices 0, 3, 4 of ref_dof_pos -> 7, 10, 11 of q0)
    ref_dof_pos[0] = sin_pos_l * scale_1 + initialAngle[0] # left_leg_pitch_joint
    ref_dof_pos[3] = -sin_pos_l * scale_2 + initialAngle[3]  # left_knee_joint
    ref_dof_pos[4] = sin_pos_l * scale_1 + initialAngle[4]  # left_ankle_pitch_joint

    # --- Right Leg Movement (Swing Phase when sin_pos_r > 0) ---
    # Right foot is considered in 'stance' (default pos) when sin_pos_r < 0
    if sin_pos_r < 0:
        sin_pos_r = 0.0 # Stance phase: keep joint at zero

    # Apply motion to right leg joints (indices 5, 8, 9 of ref_dof_pos -> 12, 15, 16 of q0)
    ref_dof_pos[5] = -sin_pos_r * scale_1 + initialAngle[5]  # right_leg_pitch_joint
    ref_dof_pos[8] = sin_pos_r * scale_2 + initialAngle[8] # right_knee_joint
    ref_dof_pos[9] = -sin_pos_r * scale_1 + initialAngle[9] # right_ankle_pitch_joint

    # --- Double Support Phase Cleanup ---
    # If the absolute value of the sine wave is close to zero, set all joints to zero
    # This enforces a brief double-support/neutral phase at the transition points.
    if np.abs(sin_pos) < 0.1:
         ref_dof_pos[:] = 0.0


    # --- Display Current State ---
    # Construct the configuration vector (q) for the display
    q_step = pinocchio.utils.zero(robot.model.nq)
    q_step[6] = 1.0  # Identity rotation
    q_step[2] = 0.0  # Base z position (for simplicity)
    q_step[7:17] = ref_dof_pos  # Set the computed leg joint angles
    display.display([q_step])
    # The original code had a commented print(1) here.

# --- Joint Exploration Loop 2: Repeating the single-joint display ---
# This is a duplicate of the first exploration loop, likely for repeated viewing.
print("\nRepeating exploration of positive displacement for each joint...")
for i in range(robot.model.nq - 7):
    q_explore = pinocchio.utils.zero(robot.model.nq)
    q_explore[6] = 1.0  # Identity rotation
    q_explore[2] = 0.0  # Base z position
    q_explore[i + 7] = 1.0  # Set the i-th movable joint to 1 radian/meter
    display.display([q_explore])
    # The original code had a commented print(1) here.

# quick helper: estimate required joint amplitudes to lift foot by desired_dz
# place in ironmanFdd.py near top (after robot_model, rdata, q0_init defined)

