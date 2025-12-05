"""
Compare real deployment logs (.npz) with sim2sim logs (.csv)
Usage: python compare_real_sim2sim.py <real_log.npz> <sim2sim_log.csv>
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def load_npz(filepath):
    """Load data from npz file (real deployment log)"""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_csv(filepath):
    """Load data from csv file (sim2sim log)"""
    # obs 0 -> 1: phase sin/cos
    # obs 2 -> 4: cmd vx, vy, dyaw
    # obs 5 -> 14: q
    # obs 15 -> 24: dq
    # obs 25 -> 34: previous action
    # obs 35 -> 37: omega
    # obs 38 -> 40: eu_ang
    return pd.read_csv(filepath)


def print_npz_summary(data, name="NPZ"):
    """Print summary of npz data"""
    print(f"\n{'='*60}")
    print(f"{name} DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Available keys: {list(data.keys())}")
    print("\nData shapes:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key}: {value}")


def print_csv_summary(df, name="CSV"):
    """Print summary of csv data"""
    print(f"\n{'='*60}")
    print(f"{name} DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    

def extract_npz_arrays(data):
    """Extract relevant arrays from npz for comparison"""
    result = {}
    
    # Common keys to extract
    keys_to_extract = [
        'observations', 'actions', 'target_q', 'qj', 'dqj',
        'imu_euler', 'imu_gyro', 'commands', 'timestamp'
    ]
    
    for key in keys_to_extract:
        if key in data:
            arr = data[key]
            if isinstance(arr, np.ndarray):
                result[key] = arr.squeeze()  # Remove singleton dimensions
    
    return result


def extract_csv_arrays(df):
    """Extract relevant arrays from sim2sim csv for comparison"""
    result = {}
    
    # Extract observations (obs_0, obs_1, ...)
    obs_cols = [c for c in df.columns if c.startswith('obs_')]
    if obs_cols:
        result['observations'] = df[sorted(obs_cols, key=lambda x: int(x.split('_')[1]))].values
    
    # Extract actions (action_0, action_1, ...)
    action_cols = [c for c in df.columns if c.startswith('action_')]
    if action_cols:
        result['actions'] = df[sorted(action_cols, key=lambda x: int(x.split('_')[1]))].values
    
    # Extract joint positions (q_0, q_1, ...)
    q_cols = [c for c in df.columns if c.startswith('q_') and not c.startswith('q_t')]
    if q_cols:
        result['qj'] = df[sorted(q_cols, key=lambda x: int(x.split('_')[1]))].values
    
    # Extract joint velocities (dq_0, dq_1, ...)
    dq_cols = [c for c in df.columns if c.startswith('dq_')]
    if dq_cols:
        result['dqj'] = df[sorted(dq_cols, key=lambda x: int(x.split('_')[1]))].values
    
    # Extract target joint positions (target_q_0, ...)
    target_cols = [c for c in df.columns if c.startswith('target_q_')]
    if target_cols:
        result['target_q'] = df[sorted(target_cols, key=lambda x: int(x.split('_')[2]))].values
    
    # Extract torques (tau_0, tau_1, ...)
    tau_cols = [c for c in df.columns if c.startswith('tau_')]
    if tau_cols:
        result['tau'] = df[sorted(tau_cols, key=lambda x: int(x.split('_')[1]))].values
    
    # Timestep
    if 'timestep' in df.columns:
        result['timestep'] = df['timestep'].values
    
    return result


def compare_observations_detailed(real_data, sim_data, n_samples=None):
    """Compare observations by semantic groups based on observation structure:
    obs 0 -> 1: phase sin/cos
    obs 2 -> 4: cmd vx, vy, dyaw
    obs 5 -> 14: q (joint positions)
    obs 15 -> 24: dq (joint velocities)
    obs 25 -> 34: previous action
    obs 35 -> 37: omega (angular velocity)
    obs 38 -> 40: eu_ang (euler angles)
    """
    if 'observations' not in real_data or 'observations' not in sim_data:
        print("Observations not found in both datasets")
        return None
    
    real_obs = real_data['observations']
    sim_obs = sim_data['observations']
    
    # Align lengths
    min_len = min(len(real_obs), len(sim_obs))
    if n_samples:
        min_len = min(min_len, n_samples)
    
    real_obs = real_obs[:min_len]
    sim_obs = sim_obs[:min_len]
    
    # Define observation groups
    obs_groups = {
        'phase_sin_cos': (0, 2),      # obs 0-1
        'cmd_vx_vy_dyaw': (2, 5),     # obs 2-4
        'joint_pos_q': (5, 15),        # obs 5-14
        'joint_vel_dq': (15, 25),      # obs 15-24
        'prev_action': (25, 35),       # obs 25-34
        'omega': (35, 38),             # obs 35-37
        'euler_ang': (38, 41),         # obs 38-40
    }
    
    print(f"\n{'='*60}")
    print("DETAILED OBSERVATION COMPARISON")
    print(f"{'='*60}")
    print(f"Real obs shape: {real_data['observations'].shape}")
    print(f"Sim obs shape: {sim_data['observations'].shape}")
    print(f"Compared samples: {min_len}")
    
    results = {}
    for group_name, (start, end) in obs_groups.items():
        if end <= real_obs.shape[1] and end <= sim_obs.shape[1]:
            real_group = real_obs[:, start:end]
            sim_group = sim_obs[:, start:end]
            
            diff = real_group - sim_group
            mae = np.mean(np.abs(diff), axis=0)
            rmse = np.sqrt(np.mean(diff**2, axis=0))
            
            print(f"\n--- {group_name} (obs {start}-{end-1}) ---")
            print(f"  MAE per dim: {np.round(mae, 6)}")
            print(f"  Mean MAE: {np.mean(mae):.6f}")
            print(f"  RMSE per dim: {np.round(rmse, 6)}")
            print(f"  Mean RMSE: {np.mean(rmse):.6f}")
            
            results[group_name] = {'mae': mae, 'rmse': rmse, 'diff': diff}
    
    return results


def plot_observations_comparison(real_data, sim_data, save_path=None):
    """Plot detailed observation comparison by groups - one subplot per dimension"""
    if 'observations' not in real_data or 'observations' not in sim_data:
        print("Observations not found in both datasets")
        return
    
    real_obs = real_data['observations']
    sim_obs = sim_data['observations']
    
    min_len = min(len(real_obs), len(sim_obs))
    real_obs = real_obs[:min_len]
    sim_obs = sim_obs[:min_len]
    
    obs_groups = {
        'phase': (0, 2),           # sin, cos
        'cmd': (2, 5),             # vx, vy, dyaw
        'joint_pos_q': (5, 15),    # 10 joints
        'joint_vel_dq': (15, 25),  # 10 joints
        'prev_action': (25, 35),   # 10 actions
        'omega': (35, 38),         # 3 axes
        'euler_ang': (38, 41),     # roll, pitch, yaw
    }
    
    # Create separate figure for each group
    for group_name, (start, end) in obs_groups.items():
        n_dims = end - start
        
        if n_dims <= 3:
            fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5*n_dims), sharex=True)
        else:
            # For larger groups, use 2 columns
            n_rows = (n_dims + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2*n_rows), sharex=True)
            axes = axes.flatten()
        
        if n_dims == 1:
            axes = [axes]
        
        for i in range(n_dims):
            obs_idx = start + i
            ax = axes[i]
            
            if obs_idx < real_obs.shape[1] and obs_idx < sim_obs.shape[1]:
                ax.plot(real_obs[:, obs_idx], 'b-', label='Real', alpha=0.8, linewidth=1)
                ax.plot(sim_obs[:, obs_idx], 'r--', label='Sim', alpha=0.8, linewidth=1)
                ax.set_ylabel(f'obs[{obs_idx}]', fontsize=9)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        if n_dims > 3:
            for i in range(n_dims, len(axes)):
                axes[i].set_visible(False)
        
        axes[-1].set_xlabel('Timestep') if n_dims <= 3 else None
        fig.suptitle(f'{group_name} (obs {start}-{end-1})', fontsize=11)
        plt.tight_layout()
        
        if save_path:
            group_save_path = save_path.replace('.png', f'_{group_name}.png')
            plt.savefig(group_save_path, dpi=150)
            print(f"Saved {group_name} plot to {group_save_path}")
        plt.show()


def plot_observations_summary(real_data, sim_data, save_path=None):
    """Plot a cleaner summary with MAE per observation group over time"""
    if 'observations' not in real_data or 'observations' not in sim_data:
        print("Observations not found in both datasets")
        return
    
    real_obs = real_data['observations']
    sim_obs = sim_data['observations']
    
    min_len = min(len(real_obs), len(sim_obs))
    real_obs = real_obs[:min_len]
    sim_obs = sim_obs[:min_len]
    
    obs_groups = {
        'phase': (0, 2),
        'cmd': (2, 5),
        'joint_pos_q': (5, 15),
        'joint_vel_dq': (15, 25),
        'prev_action': (25, 35),
        'omega': (35, 38),
        'euler_ang': (38, 41),
    }
    
    fig, axes = plt.subplots(len(obs_groups), 1, figsize=(12, 2*len(obs_groups)), sharex=True)
    
    for idx, (group_name, (start, end)) in enumerate(obs_groups.items()):
        ax = axes[idx]
        
        real_group = real_obs[:, start:end]
        sim_group = sim_obs[:, start:end]
        
        # Compute MAE over dimensions at each timestep
        mae_per_step = np.mean(np.abs(real_group - sim_group), axis=1)
        
        ax.plot(mae_per_step, 'b-', linewidth=1)
        ax.fill_between(range(len(mae_per_step)), mae_per_step, alpha=0.3)
        ax.set_ylabel(f'{group_name}\nMAE', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add mean MAE text
        mean_mae = np.mean(mae_per_step)
        ax.axhline(y=mean_mae, color='r', linestyle='--', alpha=0.5)
        ax.text(0.98, 0.85, f'mean={mean_mae:.4f}', transform=ax.transAxes, 
                fontsize=8, ha='right', color='red')
    
    axes[-1].set_xlabel('Timestep')
    fig.suptitle('Observation MAE over Time (Real vs Sim2Sim)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        summary_path = save_path.replace('.png', '_summary.png')
        plt.savefig(summary_path, dpi=150)
        print(f"Saved summary plot to {summary_path}")
    plt.show()


def compare_arrays(real_data, sim_data, key, n_samples=None):
    """Compare a specific array between real and sim data"""
    if key not in real_data or key not in sim_data:
        print(f"Key '{key}' not found in both datasets")
        return None
    
    real_arr = real_data[key]
    sim_arr = sim_data[key]
    
    # Align lengths
    min_len = min(len(real_arr), len(sim_arr))
    if n_samples:
        min_len = min(min_len, n_samples)
    
    real_arr = real_arr[:min_len]
    sim_arr = sim_arr[:min_len]
    
    # Compute statistics
    diff = real_arr - sim_arr
    mae = np.mean(np.abs(diff), axis=0)
    mse = np.mean(diff**2, axis=0)
    rmse = np.sqrt(mse)
    
    print(f"\n--- Comparison for '{key}' ---")
    print(f"Real shape: {real_data[key].shape}, Sim shape: {sim_data[key].shape}")
    print(f"Compared samples: {min_len}")
    
    if real_arr.ndim == 1:
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
    else:
        print(f"MAE per dim: {mae}")
        print(f"Mean MAE: {np.mean(mae):.6f}")
        print(f"RMSE per dim: {rmse}")
        print(f"Mean RMSE: {np.mean(rmse):.6f}")
    
    return {'mae': mae, 'rmse': rmse, 'diff': diff}


def plot_comparison(real_data, sim_data, key, n_dims=None, save_path=None):
    """Plot comparison between real and sim data"""
    if key not in real_data or key not in sim_data:
        print(f"Key '{key}' not found in both datasets")
        return
    
    real_arr = real_data[key]
    sim_arr = sim_data[key]
    
    # Align lengths
    min_len = min(len(real_arr), len(sim_arr))
    real_arr = real_arr[:min_len]
    sim_arr = sim_arr[:min_len]
    
    if real_arr.ndim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(real_arr, label='Real', alpha=0.7)
        ax.plot(sim_arr, label='Sim2Sim', alpha=0.7)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(key)
        ax.set_title(f'{key} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        n_total_dims = real_arr.shape[1]
        if n_dims is None:
            n_dims = min(n_total_dims, 10)  # Plot up to 10 dimensions
        
        fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3*n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i < n_total_dims:
                ax.plot(real_arr[:, i], label=f'Real {key}[{i}]', alpha=0.7)
                ax.plot(sim_arr[:, i], label=f'Sim {key}[{i}]', alpha=0.7, linestyle='--')
                ax.set_ylabel(f'{key}[{i}]')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Timestep')
        fig.suptitle(f'{key} Comparison: Real vs Sim2Sim')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare real deployment logs with sim2sim logs')
    parser.add_argument('real_log', type=str, help='Path to real deployment log (.npz)')
    parser.add_argument('sim2sim_log', type=str, help='Path to sim2sim log (.csv)')
    parser.add_argument('--keys', type=str, nargs='+', default=['actions', 'qj', 'dqj', 'target_q'],
                        help='Keys to compare')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting')
    parser.add_argument('--output-dir', type=str, default="/data/wanshan/Desktop/robotics/humanoid-gym/debug_logs/output", help='Directory to save plots')
    args = parser.parse_args()
    
    # Load data
    print("Loading real deployment log...")
    npz_data = load_npz(args.real_log)
    print_npz_summary(npz_data, "Real Deployment")
    
    print("\nLoading sim2sim log...")
    csv_df = load_csv(args.sim2sim_log)
    print_csv_summary(csv_df, "Sim2Sim")
    
    # Extract arrays
    real_arrays = extract_npz_arrays(npz_data)
    sim_arrays = extract_csv_arrays(csv_df)
    
    print(f"\nExtracted real arrays: {list(real_arrays.keys())}")
    print(f"Extracted sim arrays: {list(sim_arrays.keys())}")
    
    # Compare
    for key in args.keys:
        compare_arrays(real_arrays, sim_arrays, key)
    
    # Compare observations in detail
    compare_observations_detailed(real_arrays, sim_arrays)
    
    # Plot
    if not args.no_plots:
        for key in args.keys:
            save_path = None
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, f'compare_{key}.png')
            plot_comparison(real_arrays, sim_arrays, key, save_path=save_path)
        
        # Plot observations comparison
        if args.output_dir:
            obs_save_path = os.path.join(args.output_dir, 'compare_observations_detailed.png')
        else:
            obs_save_path = None
        plot_observations_comparison(real_arrays, sim_arrays, save_path=obs_save_path)
        plot_observations_summary(real_arrays, sim_arrays, save_path=obs_save_path)


if __name__ == "__main__":
    main()
