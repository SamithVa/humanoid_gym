"""
Analyze deployment logs for sim2sim debugging.
Usage: python analyze_log.py <log_file.npz>
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_log(filepath):
    """Load log data from npz file"""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


def print_summary(data):
    """Print summary of logged data"""
    print("\n" + "="*60)
    print("LOG SUMMARY")
    print("="*60)
    
    n_frames = len(data['timestamp'])
    duration = data['timestamp'][-1] - data['timestamp'][0] if n_frames > 1 else 0
    
    print(f"Total frames: {n_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average FPS: {n_frames/duration:.2f}" if duration > 0 else "N/A")
    print(f"Control dt: {data['control_dt']}")
    print(f"Action scale: {data['action_scale']}")
    print(f"Num obs: {data['num_obs']}")
    print(f"Frame stack: {data['frame_stack']}")
    
    print("\nData shapes:")
    for key in ['observations', 'policy_input', 'actions', 'target_q', 'qj', 'dqj', 'imu_quat', 'imu_euler', 'imu_gyro', 'commands']:
        if key in data:
            print(f"  {key}: {data[key].shape}")
    
    print("\nDefault angles:")
    print(f"  {data['default_angles']}")


def plot_joint_data(data, save_path=None):
    """Plot joint positions and velocities"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    t = data['timestamp'] - data['timestamp'][0]
    qj = data['qj']
    dqj = data['dqj']
    target_q = data['target_q']
    
    n_joints = qj.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    
    # Plot joint positions
    ax = axes[0]
    for i in range(n_joints):
        ax.plot(t, qj[:, i], label=f'q{i}', color=colors[i], alpha=0.7)
        ax.plot(t, target_q[:, i], '--', color=colors[i], alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Position (rad)')
    ax.set_title('Joint Positions (solid=actual, dashed=target)')
    ax.legend(loc='upper right', ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot joint velocities
    ax = axes[1]
    for i in range(n_joints):
        ax.plot(t, dqj[:, i], label=f'dq{i}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Velocity (rad/s)')
    ax.set_title('Joint Velocities')
    ax.legend(loc='upper right', ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.npz', '_joints.png'), dpi=150)
    plt.show()


def plot_imu_data(data, save_path=None):
    """Plot IMU data"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    t = data['timestamp'] - data['timestamp'][0]
    euler = data['imu_euler']
    gyro = data['imu_gyro'].squeeze()
    
    # Plot euler angles
    ax = axes[0]
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax.plot(t, np.degrees(euler[:, i]), label=labels[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('IMU Euler Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot gyroscope
    ax = axes[1]
    labels = ['Gyro X', 'Gyro Y', 'Gyro Z']
    for i in range(3):
        ax.plot(t, gyro[:, i], label=labels[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('IMU Gyroscope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.npz', '_imu.png'), dpi=150)
    plt.show()


def plot_actions(data, save_path=None):
    """Plot actions"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    t = data['timestamp'] - data['timestamp'][0]
    actions = data['actions']
    
    n_actions = actions.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_actions))
    
    for i in range(n_actions):
        ax.plot(t, actions[:, i], label=f'a{i}', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action')
    ax.set_title('Policy Actions')
    ax.legend(loc='upper right', ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.npz', '_actions.png'), dpi=150)
    plt.show()


def plot_commands(data, save_path=None):
    """Plot remote controller commands"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    
    t = data['timestamp'] - data['timestamp'][0]
    commands = data['commands']
    
    ax.plot(t, commands[:, 0], label='lx (forward)')
    ax.plot(t, commands[:, 1], label='ly (lateral)')
    ax.plot(t, commands[:, 2], label='rx (rotation)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Command')
    ax.set_title('Remote Controller Commands')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.npz', '_commands.png'), dpi=150)
    plt.show()


def plot_observation_heatmap(data, save_path=None):
    """Plot observation heatmap over time"""
    obs = data['observations'].squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    im = ax.imshow(obs.T, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.percentile(np.abs(obs), 95),
                   vmax=np.percentile(np.abs(obs), 95))
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Observation dimension')
    ax.set_title('Observation Heatmap')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.npz', '_obs_heatmap.png'), dpi=150)
    plt.show()


def export_for_sim2sim(data, output_path):
    """Export data in a format suitable for sim2sim replay"""
    export_data = {
        'observations': data['observations'],
        'actions': data['actions'],
        'target_q': data['target_q'],
        'qj': data['qj'],
        'dqj': data['dqj'],
        'imu_euler': data['imu_euler'],
        'imu_gyro': data['imu_gyro'],
        'commands': data['commands'],
        'default_angles': data['default_angles'],
        'control_dt': data['control_dt'],
        'action_scale': data['action_scale'],
    }
    
    np.savez(output_path, **export_data)
    print(f"Exported sim2sim data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze deployment logs for sim2sim debugging')
    parser.add_argument('log_file', type=str, help='Path to the log file (.npz)')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting')
    parser.add_argument('--export-sim2sim', type=str, help='Export data for sim2sim to specified path')
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return
    
    data = load_log(args.log_file)
    print_summary(data)
    
    if args.export_sim2sim:
        export_for_sim2sim(data, args.export_sim2sim)
    
    if not args.no_plots:
        plot_joint_data(data, args.log_file)
        plot_imu_data(data, args.log_file)
        plot_actions(data, args.log_file)
        plot_commands(data, args.log_file)
        plot_observation_heatmap(data, args.log_file)


if __name__ == "__main__":
    main()
