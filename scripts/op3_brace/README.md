# OP3 Brace Training & Rendering Scripts

## Overview

This directory contains training and rendering scripts for the OP3 Full-Body Brace environment (12 DOF: 6 arms + 6 legs).

## Training

### Start a new training run

```bash
python train.py --run_label my_training_run
```

This will:
- Create a `checkpoints/my_training_run/` directory
- Train the PPO agent on the Op3BraceEnv
- Save best model as `best_op3_brace_model.pth`
- Save periodic checkpoints every 10 iterations
- Generate a training progress plot: `training_progress.png`

### Resume training from a checkpoint

```bash
python train.py --run_label my_training_run --checkpoint checkpoints/my_training_run/best_op3_brace_model.pth
```

### Key hyperparameters (defined in `train.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max iterations | 1000 | Training stops early if ≥95% success maintained for 100 episodes |
| Steps per iteration | 2048 | Environment steps per PPO update |
| Learning rate | 3e-4 | Adam optimizer |
| Gamma (discount) | 0.99 | Standard RL discount factor |
| PPO epsilon (clip) | 0.2 | Clipping range for policy updates |
| Entropy coef (initial) | 0.01 | Exploration bonus, decayed over time |
| Entropy decay | 0.995 | Per-iteration decay factor |
| Value loss weight | 0.5 | Balance between policy and value function |

## Rendering

### Render trained episodes

```bash
python render.py --checkpoint checkpoints/my_training_run/best_op3_brace_model.pth --output_dir demo_videos --episodes 3
```

This will:
- Load the trained model
- Run 3 episodes with deterministic policy
- Generate MP4 videos from 3 viewpoints (left, front, top) per episode
- Save videos to `demo_videos/episode_1/`, `demo_videos/episode_2/`, etc.

### Rendering options

```bash
python render.py \
  --checkpoint checkpoints/my_training_run/best_op3_brace_model.pth \
  --output_dir demo_videos \
  --episodes 2 \
  --max_steps 200 \
  --width 640 \
  --height 480 \
  --limp_scale 0.01
```

| Argument | Default | Notes |
|----------|---------|-------|
| `--checkpoint` | Required | Path to trained model checkpoint |
| `--output_dir` | Required | Output directory for videos |
| `--episodes` | 1 | Number of episodes to render |
| `--max_steps` | env.episode_length (200) | Max steps per episode |
| `--width` | 640 | Video frame width in pixels |
| `--height` | 480 | Video frame height in pixels |
| `--limp_scale` | 0.01 | Compliance scale after impact (lower = more limp) |

## Training Output

```
checkpoints/my_training_run/
├── best_op3_brace_model.pth              # Best model so far
├── op3_brace_model_iter_10.pth           # Checkpoint at iter 10
├── op3_brace_model_iter_20.pth           # Checkpoint at iter 20
├── ...
└── training_progress.png                 # Plot of reward and success rate
```

## Video Output

```
demo_videos/
├── episode_1/
│   ├── povleft.mp4    # Left-side viewpoint
│   ├── povfront.mp4   # Front viewpoint
│   └── povtop.mp4     # Top-down viewpoint
├── episode_2/
│   ├── povleft.mp4
│   ├── povfront.mp4
│   └── povtop.mp4
└── ...
```

## Monitoring Training

During training, the script prints:
- **iter**: Current iteration / total iterations
- **eps**: Number of completed episodes (in current 10-episode window)
- **steps**: Total environment steps so far
- **mean_rew**: Mean episode reward (10-ep window)
- **max_mean_rew**: Best mean reward seen so far
- **win_rate**: Current success rate
- **consec_success**: Consecutive successful episodes
- **entropy_coef**: Current entropy coefficient (decaying)

Early stopping occurs when:
- Success rate ≥ 95% maintained for ≥ 100 consecutive episodes

## Environment Details

**Observation Space:** 31 dimensions
- Body orientation (roll, pitch)
- Body angular velocity (roll rate, pitch rate)
- 12 joint angles (6 arms + 6 legs)
- 12 joint velocities (6 arms + 6 legs)
- 3 contact flags (arms, head, feet)

**Action Space:** 12 dimensions (normalized to [-1, 1])
- 6 arm actions (left/right shoulder pitch/roll, elbow)
- 6 leg actions (left/right hip roll/pitch, knee)

**Reward Function:**
- Arm-first contact reward
- Arm synchronization bonus
- Knee timing reward
- Head impact penalty
- Torque efficiency penalty
- Action jitter penalty
- Success bonus (+100 if episode succeeds)

See `envs/op3_brace/system_explanation.md` for complete details.
