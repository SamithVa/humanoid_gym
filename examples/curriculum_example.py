#!/usr/bin/env python3
"""
Example script demonstrating curriculum learning usage for Ironman robot.
Shows how to configure, run, and monitor curriculum-based training.
"""

from humanoid.envs.custom.ironman_full_config import IronmanFullCfg, IronmanFullCfgPPO

def print_curriculum_config(cfg):
    """Print curriculum configuration for verification"""
    print("\n" + "="*60)
    print("CURRICULUM LEARNING CONFIGURATION")
    print("="*60)
    
    if hasattr(cfg, 'curriculum'):
        curr = cfg.curriculum
        print(f"Enabled: {curr.enabled}")
        print(f"\nStage 1 (Legs Only):")
        print(f"  - Target Iterations: {curr.stage1_iterations}")
        print(f"  - Reward Threshold: {curr.stage1_reward_threshold}")
        print(f"  - Arm Joints Frozen: [{curr.arm_joint_start_idx}:{curr.arm_joint_end_idx}]")
        
        print(f"\nStage 2 (Partial Arms):")
        print(f"  - Target Iterations: {curr.stage2_iterations}")
        print(f"  - Arm Action Scale: {curr.stage2_arm_action_scale}")
        print(f"  - Reward Threshold: {curr.stage2_reward_threshold}")
        
        print(f"\nStage 3 (Full Control):")
        print(f"  - Arm Action Scale: {curr.stage3_arm_action_scale}")
        print(f"  - Duration: Until training ends")
    else:
        print("Curriculum learning not configured!")
    
    print("\nCurriculum Rewards:")
    print(f"  - arm_at_target_pos: {cfg.rewards.scales.arm_at_target_pos}")
    print(f"  - arm_smoothness: {cfg.rewards.scales.arm_smoothness}")
    print(f"  - shoulder_assist_balance: {cfg.rewards.scales.shoulder_assist_balance}")
    print("="*60 + "\n")


def modify_curriculum_for_quick_test(cfg):
    """
    Modify curriculum config for quick testing.
    Reduces iteration requirements to see stage progression faster.
    """
    print("\n⚠️  QUICK TEST MODE: Reducing curriculum iteration requirements\n")
    cfg.curriculum.stage1_iterations = 100  # Normally 1000
    cfg.curriculum.stage2_iterations = 80   # Normally 800
    cfg.curriculum.stage1_reward_threshold = 0.3  # Normally 0.7
    cfg.curriculum.stage2_reward_threshold = 0.35  # Normally 0.75
    return cfg


def modify_curriculum_for_stable_training(cfg):
    """
    Modify curriculum config for more stable, slower progression.
    Good for actual training runs.
    """
    print("\n✅ STABLE TRAINING MODE: Extended curriculum stages\n")
    cfg.curriculum.stage1_iterations = 1500  # More leg practice
    cfg.curriculum.stage2_iterations = 1000
    cfg.curriculum.stage1_reward_threshold = 0.75  # Higher mastery
    cfg.curriculum.stage2_reward_threshold = 0.80
    cfg.curriculum.stage2_arm_action_scale = 0.25  # More conservative
    return cfg


def disable_curriculum(cfg):
    """Disable curriculum for baseline comparison"""
    print("\n🚫 CURRICULUM DISABLED: Training with full control from start\n")
    cfg.curriculum.enabled = False
    return cfg


def example_usage():
    """
    Example showing different ways to configure curriculum learning.
    """
    print("\n" + "="*60)
    print("CURRICULUM LEARNING - EXAMPLE CONFIGURATIONS")
    print("="*60 + "\n")
    
    # Load base configuration
    env_cfg = IronmanFullCfg()
    train_cfg = IronmanFullCfgPPO()
    
    # Example 1: Default curriculum
    print("Example 1: Default Curriculum Configuration")
    print_curriculum_config(env_cfg)
    
    # Example 2: Quick test mode
    print("\nExample 2: Quick Test Mode (for debugging)")
    env_cfg_test = IronmanFullCfg()
    env_cfg_test = modify_curriculum_for_quick_test(env_cfg_test)
    print_curriculum_config(env_cfg_test)
    
    # Example 3: Stable training mode
    print("\nExample 3: Stable Training Mode (recommended)")
    env_cfg_stable = IronmanFullCfg()
    env_cfg_stable = modify_curriculum_for_stable_training(env_cfg_stable)
    print_curriculum_config(env_cfg_stable)
    
    # Example 4: Disabled curriculum
    print("\nExample 4: Curriculum Disabled (baseline)")
    env_cfg_no_curr = IronmanFullCfg()
    env_cfg_no_curr = disable_curriculum(env_cfg_no_curr)
    print_curriculum_config(env_cfg_no_curr)
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("""
To use these configurations in training:

1. Quick Test (see stage transitions quickly):
   - Modify ironman_full_config.py with quick test values
   - Run: python humanoid/scripts/train.py --task=ironman_full_ppo
   - Expect stage transitions within 100-200 iterations

2. Production Training (recommended):
   - Use default or stable configuration
   - Run: python humanoid/scripts/train.py --task=ironman_full_ppo
   - Monitor console for [CURRICULUM] messages
   - Expect ~1000-2000 iterations per stage

3. Baseline Comparison:
   - Set curriculum.enabled = False in config
   - Train for same total iterations
   - Compare final performance and training stability

4. Monitor Progress:
   - Watch for [CURRICULUM] log messages
   - Check episode reward trends
   - Verify smooth transitions between stages
    """)
    print("="*60 + "\n")


if __name__ == "__main__":
    example_usage()
    
    print("Tips for tuning curriculum:")
    print("  1. If robot struggles in Stage 1:")
    print("     - Increase stage1_iterations (more practice)")
    print("     - Lower stage1_reward_threshold (easier to progress)")
    print("     - Check leg-specific rewards are well-tuned")
    print()
    print("  2. If Stage 2 is unstable:")
    print("     - Reduce stage2_arm_action_scale (0.2-0.25)")
    print("     - Increase stage1_reward_threshold (better foundation)")
    print("     - Extend stage2_iterations")
    print()
    print("  3. If arms don't move in Stage 3:")
    print("     - Verify compute_ref_state() includes arm movements")
    print("     - Check arm reward scales are non-zero")
    print("     - Ensure action masking is removed")
    print()
