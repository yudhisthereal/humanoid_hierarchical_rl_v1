# Humanoid Hierarchical RL (MuJoCo Warp + PPO)

This repository contains a GPU-first reinforcement learning setup for humanoid fall-response training, with:

- **MuJoCo Warp** physics (`mujoco-warp==3.6.0`)
- **PyTorch PPO**
- Two separable training pipelines:
  - **Strategy Selector** (discrete action)
  - **Goal Executor** (continuous action)

All rollout tensors are intended to remain on CUDA tensors during training.

---

## Project Structure

- `main.py` — CLI entrypoint
- `agents/ppo/ppo.py` — PPO agent + policy/value model
- `envs/strategy_selector/env.py` — strategy selector environment
- `envs/goal_conditioned/env.py` — goal-conditioned executor environment
- `envs/robot_env.py` — wrapper env (kept for compatibility)
- `scripts/train.py` — training pipelines (`selector` and `executor`)
- `scripts/test.py` — inference test script
- `assets/` — MuJoCo XML models/assets
- `requirements.txt` — Python dependencies

---

## Requirements

Install dependencies:

- `mujoco-warp==3.6.0`
- `torch`
- `warp-lang`
- `tensorboard`

Use the existing `requirements.txt` in this repo.

---

## How to Run

### 1) Train Strategy Selector (discrete PPO)

```bash
python main.py train --env selector
```

### 2) Train Goal Executor (continuous PPO)

```bash
python main.py train --env executor
```

### 3) Run test/inference

```bash
python main.py test --checkpoint <path_to_checkpoint.pt> --steps 1000
```

---

## Logging and Checkpoints

### TensorBoard logs

- Selector: `report/tensorboard/strategy_selector`
- Executor: `report/tensorboard/goal_executor`

Start TensorBoard:

```bash
tensorboard --logdir report/tensorboard
```

### Checkpoints

- Selector: `report/checkpoints/strategy_selector`
- Executor: `report/checkpoints/goal_executor`

Saved periodically and on interrupt/final stop.

---

## Training Monitor (CLI)

Live monitor updates in place (tqdm-like) and shows:

- `iter`, `eps`, `steps`
- `rew`, `max_rew`
- `win_rate`, `max_win_rate`
- `consec_success`, `max_consec_success`
- `entropy_coef`

`win_rate` is computed from successful completed episodes within the current iteration.

---

## PPO Stabilization Features Implemented

In `agents/ppo/ppo.py` and `scripts/train.py`:

1. **Entropy Annealing**
   - Linear schedule from `entropy_coef_initial` to `entropy_coef_final` over `anneal_steps`
2. **Learning Rate Decay**
   - Linear schedule from initial LR to `initial_lr * lr_final_factor`
3. **KL Early Stopping**
   - Approximate KL computed during updates
   - PPO minibatch/epoch updates stop early when KL exceeds threshold
4. **Observation Normalization (GPU)**
   - Running mean/variance (Welford-style) on CUDA
   - Policy/value use normalized observations
5. **Advantage Normalization (GPU)**
   - Advantages normalized before PPO updates
6. **Best Policy Rollback**
   - Tracks best mean reward
   - On collapse (`current_mean_reward < 0.8 * best_mean_reward`), restores best model+optimizer and damps LR/entropy

---

## TensorBoard Metrics Added

Core metrics include:

- `policy_loss`, `value_loss`, `entropy`
- `entropy_coef`
- `learning_rate`
- `approx_kl`
- `current_mean_reward`
- `best_mean_reward`
- `consecutive_successes`
- `max_consecutive_successes`

Metrics are logged on three x-axis variants:

- `.../by_iteration`
- `.../by_episode`
- `.../by_timestep`

---

## Notes

- Training termination currently uses a consecutive-success threshold in `scripts/train.py`.
- Both training pipelines are intentionally separated so selector and executor can be trained independently.
- MuJoCo Warp graph capture is used in the goal executor environment for faster repeated stepping.

---

## Troubleshooting

1. **CUDA not available**
   - Training intentionally raises if CUDA is unavailable.
2. **TensorBoard command fails**
   - Ensure `tensorboard` is installed in the active Python environment.
3. **No log updates**
   - Confirm you started training with `python main.py train --env ...` and check `report/tensorboard/...` paths.
4. **Environment/package mismatch**
   - Use the same Python environment for install + run.

---

## Quick Start Recap

```bash
python main.py train --env selector
python main.py train --env executor
tensorboard --logdir report/tensorboard
```
