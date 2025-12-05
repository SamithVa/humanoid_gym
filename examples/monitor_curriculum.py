#!/usr/bin/env python3
"""
Curriculum Learning Monitor - Visualize training progress across curriculum stages
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_curriculum_progress(log_file_path):
    """
    Parse training logs and visualize curriculum progression.
    
    Looks for patterns like:
    - [CURRICULUM] Stage 1 | Iteration X/Y
    - [CURRICULUM] Advancing to Stage 2
    - Episode reward values
    """
    stages = []
    iterations = []
    rewards = []
    
    # Parse log file
    with open(log_file_path, 'r') as f:
        current_stage = 1
        for line in f:
            # Detect stage transitions
            if "Advancing to Stage 2" in line:
                current_stage = 2
            elif "Advancing to Stage 3" in line:
                current_stage = 3
            
            # Extract iteration and reward info (customize based on your log format)
            # This is a template - adjust regex/parsing based on actual logs
            if "episode_reward" in line:
                try:
                    # Example: "Iteration 100 | episode_reward: 0.65"
                    parts = line.split()
                    iteration = int(parts[1])
                    reward = float(parts[-1])
                    stages.append(current_stage)
                    iterations.append(iteration)
                    rewards.append(reward)
                except:
                    pass
    
    return np.array(stages), np.array(iterations), np.array(rewards)


def create_curriculum_visualization(stages, iterations, rewards, output_path="curriculum_progress.png"):
    """
    Create a comprehensive visualization of curriculum training progress.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Color map for stages
    stage_colors = {1: 'blue', 2: 'orange', 3: 'green'}
    stage_names = {1: 'Stage 1: Legs Only', 2: 'Stage 2: Partial Arms', 3: 'Stage 3: Full Control'}
    
    # Plot 1: Rewards over time, colored by stage
    ax1 = axes[0]
    for stage_num in [1, 2, 3]:
        mask = stages == stage_num
        if mask.any():
            ax1.scatter(iterations[mask], rewards[mask], 
                       c=stage_colors[stage_num], label=stage_names[stage_num],
                       alpha=0.6, s=20)
            
            # Add trend line
            if mask.sum() > 10:
                z = np.polyfit(iterations[mask], rewards[mask], 2)
                p = np.poly1d(z)
                ax1.plot(iterations[mask], p(iterations[mask]), 
                        c=stage_colors[stage_num], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Curriculum Learning Progress: Reward Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stage timeline
    ax2 = axes[1]
    ax2.plot(iterations, stages, drawstyle='steps-post', linewidth=2, color='purple')
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Curriculum Stage')
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['Stage 1\n(Legs)', 'Stage 2\n(Partial Arms)', 'Stage 3\n(Full)'])
    ax2.set_title('Curriculum Stage Progression')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 3.5)
    
    # Plot 3: Reward improvement per stage
    ax3 = axes[2]
    stage_rewards = []
    stage_labels = []
    for stage_num in [1, 2, 3]:
        mask = stages == stage_num
        if mask.any():
            stage_rewards.append(rewards[mask])
            stage_labels.append(f'Stage {stage_num}')
    
    if stage_rewards:
        bp = ax3.boxplot(stage_rewards, labels=stage_labels, patch_artist=True)
        for patch, stage_num in zip(bp['boxes'], [1, 2, 3]):
            patch.set_facecolor(stage_colors[stage_num])
            patch.set_alpha(0.6)
    
    ax3.set_xlabel('Curriculum Stage')
    ax3.set_ylabel('Episode Reward Distribution')
    ax3.set_title('Reward Distribution by Stage')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def print_curriculum_statistics(stages, iterations, rewards):
    """
    Print summary statistics for curriculum training.
    """
    print("\n" + "="*60)
    print("CURRICULUM TRAINING STATISTICS")
    print("="*60 + "\n")
    
    for stage_num in [1, 2, 3]:
        mask = stages == stage_num
        if mask.any():
            stage_iters = iterations[mask]
            stage_rewards = rewards[mask]
            
            print(f"Stage {stage_num}:")
            print(f"  Duration: {stage_iters[-1] - stage_iters[0]} iterations")
            print(f"  Initial Reward: {stage_rewards[0]:.4f}")
            print(f"  Final Reward: {stage_rewards[-1]:.4f}")
            print(f"  Average Reward: {np.mean(stage_rewards):.4f}")
            print(f"  Improvement: {stage_rewards[-1] - stage_rewards[0]:.4f}")
            print()
    
    print(f"Total Training Iterations: {iterations[-1] - iterations[0]}")
    print(f"Overall Reward Improvement: {rewards[-1] - rewards[0]:.4f}")
    print("="*60 + "\n")


def generate_mock_data():
    """
    Generate mock curriculum training data for demonstration.
    """
    np.random.seed(42)
    
    # Stage 1: 0-1000 iterations
    stage1_iters = np.arange(0, 1000, 10)
    stage1_rewards = 0.3 + 0.4 * (stage1_iters / 1000) + np.random.normal(0, 0.05, len(stage1_iters))
    stage1_stages = np.ones(len(stage1_iters))
    
    # Stage 2: 1000-1800 iterations
    stage2_iters = np.arange(1000, 1800, 10)
    stage2_rewards = 0.65 + 0.15 * ((stage2_iters - 1000) / 800) + np.random.normal(0, 0.04, len(stage2_iters))
    stage2_stages = np.ones(len(stage2_iters)) * 2
    
    # Stage 3: 1800-3000 iterations
    stage3_iters = np.arange(1800, 3000, 10)
    stage3_rewards = 0.78 + 0.1 * ((stage3_iters - 1800) / 1200) + np.random.normal(0, 0.03, len(stage3_iters))
    stage3_stages = np.ones(len(stage3_iters)) * 3
    
    stages = np.concatenate([stage1_stages, stage2_stages, stage3_stages])
    iterations = np.concatenate([stage1_iters, stage2_iters, stage3_iters])
    rewards = np.concatenate([stage1_rewards, stage2_rewards, stage3_rewards])
    
    return stages, iterations, rewards


def main():
    """
    Main function - can parse real logs or generate demo visualization.
    """
    print("\nCurriculum Learning Monitor")
    print("="*60)
    
    # Try to find actual log file
    log_paths = [
        Path("logs/Ironman_ppo_full_body/train.log"),
        Path("../logs/Ironman_ppo_full_body/train.log"),
        Path("humanoid/logs/Ironman_ppo_full_body/train.log"),
    ]
    
    log_file = None
    for path in log_paths:
        if path.exists():
            log_file = path
            break
    
    if log_file:
        print(f"Found log file: {log_file}")
        print("Parsing training logs...")
        try:
            stages, iterations, rewards = plot_curriculum_progress(log_file)
            print(f"Loaded {len(iterations)} data points")
        except Exception as e:
            print(f"Error parsing log: {e}")
            print("Generating mock data for demonstration...")
            stages, iterations, rewards = generate_mock_data()
    else:
        print("No log file found - generating mock data for demonstration")
        stages, iterations, rewards = generate_mock_data()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_curriculum_visualization(stages, iterations, rewards)
    
    # Print statistics
    print_curriculum_statistics(stages, iterations, rewards)
    
    print("✅ Complete! Check 'curriculum_progress.png' for visualization.")


if __name__ == "__main__":
    main()
