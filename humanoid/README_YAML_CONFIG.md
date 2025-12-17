# YAML Configuration Guide for Humanoid Gym

## Overview

The training and play scripts now support YAML-based configuration management. Training runs automatically save their complete configuration as `config.yaml`, and the play script can load these configs to reproduce exact training conditions.

## Table of Contents
- [Quick Start](#quick-start)
- [Training: Auto-Save Configuration](#training-auto-save-configuration)
- [Play: Loading Configurations](#play-loading-configurations)
- [YAML File Format](#yaml-file-format)
- [Configuration Options](#configuration-options)
- [Examples](#examples)

---

## Quick Start

### Training (Automatic Config Save)
```bash
python ./scripts/train.py --task ironman_ppo --run_name my_experiment
# Output: Configuration saved to: logs/Ironman_ppo/Dec17_12-34-56_my_experiment/config.yaml
```

### Playing with Saved Config
```bash
python ./scripts/play.py --task ironman_ppo \
  --load_run Dec17_12-34-56_my_experiment \
  --config_file logs/Ironman_ppo/Dec17_12-34-56_my_experiment/config.yaml
```

---

## Training: Auto-Save Configuration

### What Gets Saved

Every training run automatically saves a `config.yaml` file containing:
- **Environment configurations**: terrain, noise, rewards, domain randomization, etc.
- **Training configurations**: algorithm, policy, runner settings, seed, etc.
- **All hyperparameters** used during that specific training run

### File Location

```
logs/
└── Ironman_ppo/
    └── Dec17_12-34-56_my_experiment/
        ├── config.yaml          ← Automatically saved at training start
        ├── model_200.pt
        ├── model_400.pt
        └── ...
```

### Benefits

✅ **Reproducibility**: Exact training configuration preserved with model checkpoints  
✅ **Easy Testing**: Load exact training settings during play/evaluation  
✅ **Version Control**: Track which configs produced good results  
✅ **Sharing**: Share complete configurations with team members  
✅ **Debugging**: Compare configs between different training runs  

---

## Play: Loading Configurations

### Usage Options

#### Option 1: Use Saved Training Config (Recommended)
```bash
python ./scripts/play.py --task ironman_ppo \
  --load_run Dec17_11-52-59_smaller_action_scale_50hz \
  --config_file logs/Ironman_ppo/Dec17_11-52-59_smaller_action_scale_50hz/config.yaml
```

#### Option 2: Use Custom Config
```bash
python ./scripts/play.py --task ironman_ppo \
  --config_file play_config_example.yaml
```

#### Option 3: Use Wandb Config (Legacy)
```bash
python ./scripts/play.py --task ironman_ppo \
  --config_file logs/Ironman_ppo/wandb/run-20251217_011557-xamobjb4/files/config.yaml
```

#### Option 4: No Config (Default Behavior)
```bash
python ./scripts/play.py --task ironman_ppo
```

### Configuration Priority

When multiple configuration sources are present:

1. **Base config** from task class (e.g., `IronmanCfg`)
2. **YAML config** (if `--config_file` is provided) ← Overrides base
3. **Hardcoded overrides** in `play.py` ← Applied last

---

## YAML File Format

### Wandb-Style Format (Used in Auto-Saved Configs)

```yaml
terrain:
  value:
    mesh_type: plane
    curriculum: false
    num_rows: 5
    num_cols: 5

noise:
  value:
    add_noise: true
    noise_level: 0.5

env:
  value:
    num_envs: 1
    episode_length_s: 24
```

### Simplified Format (For Custom Configs)

Both formats work with play.py:

```yaml
terrain:
  mesh_type: plane
  curriculum: false
  num_rows: 5
  num_cols: 5

noise:
  add_noise: true
  noise_level: 0.5
```

---

## Configuration Options

### Environment Configuration Sections

#### `env`
- `num_envs`: Number of parallel environments (default: 4096, typically 1 for play)
- `episode_length_s`: Episode length in seconds
- `env_spacing`: Spacing between environments
- `num_observations`, `num_actions`: Observation/action dimensions

#### `terrain`
- `mesh_type`: `'plane'`, `'trimesh'`, or `'heightfield'`
- `curriculum`: Enable/disable terrain curriculum (boolean)
- `num_rows`, `num_cols`: Terrain grid dimensions
- `max_init_terrain_level`: Starting terrain difficulty
- `static_friction`, `dynamic_friction`: Surface friction values

#### `noise`
- `add_noise`: Enable/disable observation noise (boolean)
- `noise_level`: Noise magnitude (0.0 to 1.0)
- `curriculum`: Enable noise curriculum (boolean)
- `noise_scales`: Per-observation-type noise scales (dict)

#### `domain_rand`
- `push_robots`: Enable random pushes (boolean)
- `randomize_friction`: Randomize surface friction (boolean)
- `randomize_base_mass`: Randomize robot mass (boolean)
- `joint_angle_noise`: Joint angle noise magnitude
- `action_delay`: Action delay in seconds

#### `control`
- `action_scale`: Scale factor for actions
- `decimation`: Control frequency decimation
- `stiffness`, `damping`: PD controller gains (dicts by joint)

#### `rewards`
- `scales`: Reward term weights (dict)
- `only_positive_rewards`: Clip negative rewards (boolean)
- `tracking_sigma`: Tracking reward sigma value

#### `viewer`
- `pos`: Camera position [x, y, z]
- `lookat`: Camera look-at point [x, y, z]
- `ref_env`: Reference environment index

#### `sim`
- `dt`: Simulation timestep
- `gravity`: Gravity vector [x, y, z]
- `physx`: PhysX-specific parameters (dict)

### Training Configuration Sections

#### `algorithm`
- `clip_param`: PPO clip parameter
- `entropy_coef`: Entropy coefficient
- `learning_rate`: Learning rate
- `num_learning_epochs`: Epochs per update
- `gamma`, `lam`: GAE parameters

#### `policy`
- `actor_hidden_dims`: Actor network layer sizes (list)
- `critic_hidden_dims`: Critic network layer sizes (list)
- `init_noise_std`: Initial action noise std

#### `runner`
- `num_steps_per_env`: Steps per environment per iteration
- `max_iterations`: Total training iterations
- `save_interval`: Model save frequency
- `experiment_name`, `run_name`: Logging names
- `resume`, `load_run`, `checkpoint`: Resume settings

#### `seed`
- Random seed for reproducibility (integer)

---

## Examples

### Example 1: Replay with Exact Training Config

```bash
# Train a model
python ./scripts/train.py --task ironman_ppo --run_name experiment_v1

# Play using the exact training configuration
python ./scripts/play.py --task ironman_ppo \
  --load_run Dec17_12-34-56_experiment_v1 \
  --config_file logs/Ironman_ppo/Dec17_12-34-56_experiment_v1/config.yaml
```

### Example 2: Test on Different Terrain

Create `custom_terrain.yaml`:
```yaml
terrain:
  value:
    mesh_type: trimesh
    curriculum: true
    max_init_terrain_level: 10
```

Run:
```bash
python ./scripts/play.py --task ironman_ppo \
  --load_run Dec17_12-34-56_experiment_v1 \
  --config_file custom_terrain.yaml
```

### Example 3: Disable Noise for Clean Testing

Create `no_noise.yaml`:
```yaml
noise:
  value:
    add_noise: false

domain_rand:
  value:
    push_robots: false
    randomize_friction: false
```

Run:
```bash
python ./scripts/play.py --task ironman_ppo \
  --config_file no_noise.yaml
```

### Example 4: Custom Camera View

Create `camera_view.yaml`:
```yaml
viewer:
  value:
    pos: [2, -2, 1.5]
    lookat: [0, 0, 0.5]
```

Run:
```bash
python ./scripts/play.py --task ironman_ppo \
  --config_file camera_view.yaml
```

### Example 5: Minimal Custom Config (Partial Override)

You only need to include settings you want to change:

```yaml
# minimal_config.yaml
terrain:
  value:
    mesh_type: plane

noise:
  value:
    noise_level: 0.8
```

All other settings will use defaults from the task class configuration.

---

## Tips and Best Practices

### For Training
✅ Config is saved automatically - no action needed  
✅ Each training run gets its own config file  
✅ Config includes all merged parameters (base + overrides)  

### For Testing/Play
✅ Use saved training configs for exact reproduction  
✅ Create minimal custom configs with only the changes you need  
✅ Combine with `--debug` flag for detailed logging  
✅ The script warns about invalid/unknown config keys  

### For Experimentation
✅ Save custom configs with descriptive names  
✅ Keep configs in version control alongside code  
✅ Document what each custom config is for  
✅ Compare configs between runs to understand differences  

### Debugging
✅ Check the console output - it shows which config values are being updated  
✅ Compare saved configs between good and bad runs  
✅ Use `--debug` mode with play.py to log detailed behavior  

---

## File Reference

- **Auto-saved configs**: `logs/<experiment_name>/<run_timestamp>/config.yaml`
- **Wandb configs**: `logs/<experiment_name>/wandb/<run_id>/files/config.yaml`
- **Example config**: `play_config_example.yaml` (in humanoid directory)

---

## Troubleshooting

**Q: Config file not found**  
A: Make sure the training run completed initialization. Config is saved at training start.

**Q: Some settings not being applied**  
A: Check the configuration priority - hardcoded overrides in play.py are applied last and will override YAML settings.

**Q: Warning about unknown config keys**  
A: The key doesn't exist in the config class. Check spelling and refer to the available config options above.

**Q: Want to use both saved config and custom overrides**  
A: Currently not supported. You can copy the saved config and modify it, or modify the play.py hardcoded overrides.

---

## Summary

This YAML configuration system provides:
- **Automatic configuration preservation** during training
- **Easy configuration loading** during testing
- **Reproducible experiments** with saved configs
- **Flexible customization** with custom YAML files
- **Clear documentation** of all parameters

For questions or issues, refer to the config class definitions in `humanoid/envs/custom/` and `humanoid/envs/base/legged_robot_config.py`.
