# OP3 Brace Environment System Design

## Recent Changes
- 2026-05-21: `op3_brace` fully implemented with full-body control (12 DOF: 6 arms + 6 legs), 31-dim observation space.
- 2026-05-21: Leg joint control ranges finalized: hip roll/pitch, knee joints with specified bounds.
- 2026-05-21: Reward structure: arm-first contact (primary), arm synchronization, head-impact penalty, torque efficiency, jitter smoothness.


## Overview

The OP3 Brace environment (`Op3BraceEnv`) is a single-env, MuJoCo-based RL task where the OP3 humanoid robot learns to brace itself when pushed, by controlling both **arms and legs** to absorb impact while preventing the head from touching the ground. This full-body variant enables more sophisticated bracing strategies involving coordinated leg stabilization and arm extension.

**Key Design Principle:** No Warp GPU parallelization; single-env, pure MuJoCo CPU simulation with PyTorch for RL training. Full-body control enables complex bracing dynamics.

---

## Observation Space

**Dimension: 31** (continuous, clipped to [−50, +50] per element)

| Index | Name | Unit | Description |
|-------|------|------|-------------|
| 0 | `body_roll` | rad | Body roll angle (rotation around X) |
| 1 | `body_pitch` | rad | Body pitch angle (rotation around Y) |
| 2–3 | `body_angvel_rp` | rad/s | Angular velocity (roll rate, pitch rate) |
| 4–15 | `joint_angles` | rad | 12 actuated joint positions: `[l_sho_pitch, l_sho_roll, l_el, r_sho_pitch, r_sho_roll, r_el, l_hip_roll, l_hip_pitch, l_knee, r_hip_roll, r_hip_pitch, r_knee]` |
| 16–27 | `joint_velocities` | rad/s | 12 actuated joint velocities (corresponding to joint angles above) |
| 28–30 | `contact_flags` | {0,1} | 3 binary contact indicators: `[arms_contact, head_contact, feet_contact]` (1 = actual MuJoCo contact with floor, 0 = no contact) |

### Notes
- **Body orientation** extracted from the root (free) joint quaternion via Euler angle conversion (roll-pitch-yaw).
- **Body angular velocity** (roll, pitch rates) extracted from body frame angular velocity vector.
- **Joint angles and velocities** are absolute positions and velocities in radians/rad-per-second from the OP3 model's joint structure (6 arm + 6 leg joints).
- **Contact flags** determined by iterating MuJoCo contact array (`data.ncon`, `data.contact[i]`) and checking for collision pairs where one geom is in the limb group (arms/head/feet) and the other is the floor.
- **Arm contact:** True if **either** left or right arm geoms contact the floor.
- **Head contact:** True if any head geoms contact the floor.
- **Feet contact:** True if any leg geoms contact the floor.

---

## Action Space

**Dimension: 12** (continuous, normalized to [−1, 1])

Normalized actions are mapped to absolute joint position targets via affine scaling:

$$\text{target}_j = \text{ctrl\_min}_j + 0.5 \cdot (\text{action}_j + 1.0) \cdot (\text{ctrl\_max}_j - \text{ctrl\_min}_j)$$

### Actuated Joints (Arms + Legs)

| Index | Joint Name | Control Range [rad] | Notes |
|-------|-----------|---|---|
| 0 | `l_sho_pitch` | [−1.7, 1.7] | Left shoulder pitch |
| 1 | `l_sho_roll` | [−1.0, 1.0] | Left shoulder roll |
| 2 | `l_el` | [−1.0, 1.0] | Left elbow |
| 3 | `r_sho_pitch` | [−1.7, 1.7] | Right shoulder pitch |
| 4 | `r_sho_roll` | [−1.0, 1.0] | Right shoulder roll |
| 5 | `r_el` | [−1.0, 1.0] | Right elbow |
| 6 | `l_hip_roll` | [−0.5, 0.0] | Left hip roll |
| 7 | `l_hip_pitch` | [−1.0, 0.0] | Left hip pitch |
| 8 | `l_knee` | [0.0, 1.57] | Left knee |
| 9 | `r_hip_roll` | [0.0, 0.5] | Right hip roll |
| 10 | `r_hip_pitch` | [0.0, 1.0] | Right hip pitch |
| 11 | `r_knee` | [−1.57, 0.0] | Right knee |

### Fixed (Non-Actuated) Joints

| Joint | Value [rad] | Notes |
|-------|---|---|
| `head_pan` | 0.0 | Always straight ahead |
| `head_tilt` | 0.0 | Always fixed level |
| `l_hip_yaw`, `r_hip_yaw` | 0.0 | No yaw control for stability |
| `l_ank_pitch`, `l_ank_roll` | 0.0 | Ankles locked |
| `r_ank_pitch`, `r_ank_roll` | 0.0 | Ankles locked |

**Implementation:** Fixed joints are explicitly set to their target values in the control input via the `FIXED_ACTUATORS` dictionary before calling `mujoco.mj_step()`.

---

## Policy Network Architecture

**Class: `Op3PPONetwork(nn.Module)` (or custom network in training script)**

```
Input: obs (shape [batch] or [])  # 31-dim observation
  ↓
Shared Backbone:
  Linear(31 → 64) + Tanh
  Linear(64 → 64) + Tanh
  ↓
Policy Head:
  Linear(64 → 12)  →  mean (action means, μ)
  
Value Head:
  Linear(64 → 1)   →  value (scalar critic estimate)

Learned Parameters:
  policy_logstd: Parameter(shape=[12])  (log std of Gaussian policy)
```

**Policy Distribution:** Gaussian with state-dependent mean and learned fixed variance (diagonal covariance).

**Forward Pass:**
```python
shared = backbone(obs)
action_mean = policy_head(shared)
value = value_head(shared)

if deterministic:
    action = tanh(action_mean)  # Deterministic policy
else:
    std = exp(clip(policy_logstd, -5, 2))  # Bounded std
    action = tanh(action_mean + std * randn_like(action_mean))  # Tanh-squashed
```

**Key Points:**
- Actions are **tanh-squashed** to enforce [-1, 1] bounds.
- Log-probability is corrected for tanh transformation: $\log \pi(a|s) = \log \mu(u|s) - \sum_i \log(1 - a_i^2)$ where $a_i = \tanh(u_i)$.

---

## Training Hyperparameters

All hyperparameters are optimized for the arm-only bracing task based on `op3_arm_brace_v2` defaults.

| Category | Parameter | Value |
|----------|-----------|-------|
| **Optimization** | Learning rate | 3e−4 (Adam) |
| | Gradient clipping | 0.5 (L2 norm) |
| **PPO** | Clipping range (ε) | 0.2 |
| | Value loss coefficient | 0.5 |
| | Entropy coefficient (initial) | 0.01 |
| **Exploration** | Entropy decay (per iter) | 0.995 |
| | Minimum entropy coef | 0.001 |
| **Sampling** | Steps per iteration | 2048 |
| | Trajectory GAE λ | 0.95 |
| | Discount factor (γ) | 0.99 |
| **Episode** | Episode length (timeout) | 200 steps |
| | MuJoCo substeps per step | 5 (frame_skip) |
| **Stopping** | Target success rate | ≥95% (over 10-episode window) |
| | Early stop condition | Maintain ≥95% for 100+ consecutive episodes |
| **Checkpointing** | Save interval | Every 10 iterations |
| | Maximum iterations | 1000+ |

---

## Reward Function

Per-step reward is composed of multiple components designed to encourage arm-first contact, synchronized bracing, and energy efficiency. All contact timing is **MuJoCo contact array-based**.

### 1. Arm-First Contact Reward ($R_{arm\_first}$)

Reward for arms contacting before head or torso:

$$R_{arm\_first} = \tanh\left(\frac{\max(t_H - t_A^{min}, 0)}{1.0}\right)$$

where $t_A^{min} = \min(t_A^L, t_A^R)$ and $t_H$ is head contact time (from MuJoCo array). If no arms contact, $R_{arm\_first} = 0$.

**Weight:** $w_{arm\_first} = 5.0$ (Primary objective)

### 2. Arm Synchronization Reward ($R_{arm\_sync}$)

Bonus for synchronized left/right arm contact:

$$R_{arm\_sync} = 1.0 - \tanh(|t_A^L - t_A^R|)$$

Evaluated only if both arms have contacted; else $R_{arm\_sync} = 0$.

**Weight:** $w_{arm\_sync} = 1.0$ (Secondary objective)

### 3. Knee Timing Reward ($R_{knee\_timing}$)

Penalty for early knee contact relative to arms:

$$R_{knee\_timing} = 1.0 - \tanh\left(\frac{|t_K - t_A^{min}|}{0.2}\right)$$

where $t_K$ is first knee contact time from MuJoCo array. Evaluated if knees contact; else $R_{knee\_timing} = 0$.

**Weight:** $w_{knee\_timing} = 1.0$

### 4. Head Impact Penalty ($C_{head\_impact}$)

Per-step penalty based on actual MuJoCo contact forces when any head geom contacts the floor. This uses the MuJoCo contact array rather than a height heuristic.

Compute per-contact force magnitude from the contact frame vectors reported by MuJoCo (example implementation uses `c.frame[0:3]` for force components). Then scale and cap each contact contribution:

$$C_{head\_impact} = \sum_{i\in\mathcal{C}_{head}} \min\big(20.0,\; 2.0 \cdot \|f_i\|\big)$$

where $\mathcal{C}_{head}$ is the set of head–floor contacts at the current step and $f_i$ is the contact frame force vector for contact $i$. The scaling factor (2.0) and cap (20.0) reflect implementation choices to keep the per-step penalty bounded.

**Interpretation:** This is an actual contact-force-based penalty (higher when head hits harder). In the implementation head contact also immediately terminates the episode (failure), so this term reinforces the hard constraint during intermediate steps.

### 5. Torque Load Cost ($C_{torque}$)

Per-step positive cost on cumulative actuation effort to encourage energy-efficient bracing:

$$C_{torque} = \sum_{j=0}^{11} |\tau_j|$$

where $\tau_j = \text{data.actuator\_force}[j]$ is the actuator force/torque of actuator $j$. This value is treated as a cost (non-negative) and subtracted from the reward via its weight.

**Weight:** $w_{torque} = 0.2$ (tertiary - efficiency improvement)

### 6. Jitter Cost ($C_{jitter}$)

Per-step positive count-based cost on action sign-flips to encourage smooth, coherent movement:

$$C_{jitter} = n_{jitter}$$

where $n_{jitter}$ is the count of actuators experiencing sign-flip jitter:
- For each of the 12 actuators, check if the normalized action value changed sign between consecutive steps.
- **Condition for jitter:** $|\text{action}_{t-1}[j]| \geq 0.5$ AND $|\text{action}_{t}[j]| \geq 0.5$ AND $\text{sign}(\text{action}_{t-1}[j]) \neq \text{sign}(\text{action}_{t}[j])$
- $n_{jitter} \in [0, 12]$.

**Weight:** $w_{jitter} = 0.2$ (tertiary - smooth motion)

### 7. Success Bonus

Applied **only at episode termination** if success criteria are met:

$$R_{success} = 100.0 \text{ if episode succeeds, else } 0.0$$


### Total Per-Step Reward

Per-step reward combines positive objective terms and subtracts cost terms:

$$
R_t = w_{arm\_first} R_{arm\_first} + w_{arm\_sync} R_{arm\_sync} + w_{knee\_timing} R_{knee\_timing} - w_{head\_impact} C_{head\_impact} - w_{torque} C_{torque} - w_{jitter} C_{jitter} + R_{success}
$$

Where $w_{head\_impact}$ controls how strongly contact forces on the head reduce reward (implementation treats head contact as immediate termination, so this term reinforces safety before termination). Typical weights used in the codebase: $w_{arm\_first}=5.0$, $w_{arm\_sync}=1.0$, $w_{knee\_timing}=1.0$, $w_{torque}=0.2$, $w_{jitter}=0.2$.

**Clipping:** Final per-step reward is clipped to [−100, +100].

**Note (recorded value):** The scalar value returned by `env.step()` is the per-step reward $R_t$ computed above (the weighted sum of positive objectives minus positive costs), plus any immediate success bonus applied at termination. Training scripts in `scripts/` additionally record the raw per-component values each step and append a `step_reward` entry (the `env.step()` return). The per-component visualizer (`scripts/reward_component_tracker.py`) uses these recorded values so it can display both the raw component contributions and the actual `step_reward` for direct comparison.

**Component Priority:**

| Component | Weight | Priority | Intuition |
|-----------|--------|----------|-----------|
| $R_{arm\_first}$ | 5.0 | **PRIMARY** | Arm-first contact is the core defensive task objective. |
| $R_{arm\_sync}$ | 1.0 | **SECONDARY** | Synchronized arms maximize bracing effectiveness. |
| $R_{knee\_timing}$ | 1.0 | **SECONDARY** | Timely knee contact supports the bracing strategy. |
| $C_{torque}$ | 0.2 | **TERTIARY** | Energy efficiency encourages controlled bracing. |
| $C_{jitter}$ | 0.2 | **TERTIARY** | Smooth movement prevents oscillation and jerky behavior. |
| $C_{head\_impact}$ | (scaled) | **HARD CONSTRAINT** | Head contact penalized by actual contact force; head must never touch. |
| $R_{success}$ | 100.0 | **EPISODE BONUS** | Final reward applied only at termination if all criteria met. |

---

## Episode Termination & Success

### Termination Conditions (done = True)

Episode ends if **any** of the following occur:

1. **Head Contact (Failure):** Head collision geom makes actual MuJoCo contact with floor geom.
   - Detection: Iterate `data.contact[0:data.ncon]` and check if any contact pair includes a head geom (`h1c`, `h2c`, `h21c`, `h22c`) and the floor geom.
   - Immediate episode termination, success = False.

2. **Timeout:** Step count reaches or exceeds 200 steps.
   - No automatic failure; success evaluated based on contact sequence.

3. **Invalid State:** NaN values in `data.qpos` or `data.qvel`.
   - Indicates simulation instability; episode terminates with success = False.

### Success Criteria

An episode is considered **successful** if **all** of the following hold at episode termination:

1. **Head never touches ground** during the episode (no MuJoCo contact between head geoms and floor at any step): $t_H = \infty$
2. **Left arm contacts ground before head:** $t_A^L < t_H$
3. **Right arm contacts ground before head:** $t_A^R < t_H$
4. **Arms synchronized:** $|t_A^L - t_A^R| \leq 0.1$ seconds (timing from MuJoCo contact array)

All contact timing is based on MuJoCo contact array detection only.

### Training Stopping Condition

Training halts early when:
- Success rate ≥ 95% is achieved over a 10-episode rolling window, **AND**
- That ≥95% success rate is **maintained for at least 100 consecutive additional episodes**.

This is stricter than achieving 100% success once; it requires sustained high performance.

---

## Initialization & Reset

### Episode Reset (`env.reset()`)

1. Reset all joint positions and velocities to zero (via `mujoco.mj_resetData()`).
2. Raise COM slightly (+0.05 m in Z) to avoid initial collision.
3. **Apply initial push impulse** (V2 approach):
   - Fixed push force: **90.0 N** in +X direction (forward).
   - Applied for **5 steps** (~0.025 s wall-clock time) at episode start.
   - Applied via both direct velocity impulse and external force application on the body.
4. Set fixed controls explicitly:
   - `head_pan`, `head_tilt` → 0.0
   - All hip yaw, ankle joints → 0.0
5. Set initial arm targets to neutral (zeros).
6. **Clear contact timers after push phase:** $t_A^L, t_A^R, t_H \leftarrow \infty$
   - This ensures success/reward metrics only evaluate the **recovery phase**, not the push itself.
7. Call `mujoco.mj_forward()` to finalize state.

### Reset Returns
- Initial observation (19-dim) from post-push state.

### Push Mechanism Details

The push is applied **during** reset before the policy gains control:

```python
push_impulse = float(push_force) * (dt * push_kick_scale)  # Scale factor ~0.02
data.qvel[root_dof_adr + 0] += push_impulse  # Direct velocity boost

for _ in range(push_steps):  # 5 steps
    data.xfrc_applied[body_id, 0] = push_force  # 90 N forward
    for _ in range(frame_skip):  # 5 substeps per step
        mujoco.mj_step(model, data)

data.xfrc_applied[:] = 0.0  # Clear applied forces

# Reset contact timers so recovery phase is tracked separately
t_arms_l, t_arms_r, t_head = inf, inf, inf
```

**Design rationale:** The push simulates a disturbance or collision that the policy must respond to. By applying it during reset and then clearing contact timers, the policy learns to react to the perturbation within the 200-step episode window.

---

## Geometry & Collision

### Key Collision Geoms

| Category | OP3 Geom Names | Purpose |
|----------|---|---|
| **Head** | `h1c`, `h2c`, `h21c`, `h22c` | Head collision meshes |
| **Arms (Left)** | `la1c`, `la2c`, `la3c` | Left arm links |
| **Arms (Right)** | `ra1c`, `ra2c`, `ra3c` | Right arm links |
| **Legs (Left)** | `ll1c`–`ll6c` | Left leg link meshes |
| **Legs (Right)** | `rl1c`–`rl6c` | Right leg link meshes |
| **Body** | `bodyc`, `body1c`–`body4c` | Torso collision meshes |
| **Floor** | `floor` (world geom) | Ground plane |

### Contact Detection Algorithm

**All contact timing and detection must use MuJoCo contact array (`data.contact[0:data.ncon]`).**

For each step:
1. Iterate through all contacts: `for i in range(int(data.ncon))`
2. Get contact pair: `c = data.contact[i]; geom1 = int(c.geom1); geom2 = int(c.geom2)`
3. Determine if floor is involved: `floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor'))`
4. Check limb membership: if `geom1 == floor_id` and `geom2` in arm/head/knee set, record contact time for that limb. (Or vice versa for `geom2 == floor_id`.)
5. Record first contact time for each limb; ignore subsequent contacts.

**No position-based estimation:** Contact is only registered when MuJoCo's contact solver detects a collision (normal force > 0 implies contact).

---

## Implementation Notes

### Simulation Timestep
- MuJoCo base timestep: `model.opt.timestep` (default ~0.002 s).
- Per environment step: **5 MuJoCo substeps** (cumulative ~0.01 s per action step).
- Episode length: 200 steps → ~2.0 seconds of wall-clock simulation.

### Contact Detection Timing
- Contact detection occurs **after** each `mujoco.mj_step()` call, when `data.contact` is populated.
- Contact times are recorded as the step number (in simulation time) when first contact is detected.
- **Critical:** Use MuJoCo contact array only; never infer contact from position or height thresholds.

### Position Control
OP3 uses position-servo actuators. Each actuator is a proportional controller targeting a desired joint position. The policy outputs normalized actions [−1, 1] which map to control targets via the joint's `ctrlrange`.

### Actuator Force / Torque
- `data.actuator_force[actuator_id]` gives the scalar force applied by the actuator at current step.
- For position servo: $\text{force} = k_p \cdot (\text{target} - \text{current\_pos})$ (simplified).
- Used for torque penalty: penalizes high actuation effort.

### Clipping & Bounds
- Observations clipped to [−50, +50].
- Rewards clipped to [−100, +100].
- Actions clipped to [−1, +1] before mapping to control ranges.

---

## File Structure

```
envs/op3_brace/
├── __init__.py                  # Package init, exports Op3BraceEnv
├── env.py                       # Core environment class (Op3BraceEnv)
├── op3.xml                      # OP3 model reference (or symlink from assets/)
└── system_explanation.md        # This document
```

**Note:** `op3_brace` is the full-body environment. For the arm-only variant, see `envs/op3_arm_brace_v2/`.

---

## Differences from `op3_arm_brace_v2` (Arm-Only Variant)

| Aspect | Full-Body (`op3_brace`) | Arm-Only (`op3_arm_brace_v2`) |
|--------|---|---|
| **Actuated DOF** | 12 (6 arms + 6 legs) | 6 (arms only) |
| **Obs Dim** | 31 | 19 |
| **Action Dim** | 12 | 6 |
| **Obs Components** | Body pose, 12 joint angles/vels, 3 contact flags | Body pose, 6 arm angles/vels, 3 contact flags |
| **Controlled Joints** | Shoulders, elbows, hips, knees | Shoulders, elbows only |
| **Fixed Joints** | Head, ankles, hip yaw | Head, all leg joints, ankles, hip yaw |
| **Push Force** | Fixed | 90 N (fixed) |
| **Arm Contact Reward** | Component of larger contact reward | 5.0 (immediate, high priority) |
| **Head Impact Penalty** | 5.0× | 10.0× (stricter) |
| **Torque Penalty** | −(full sum of all 12) | −0.01×Στ (lighter, arm-only) |
| **Jitter Penalty** | −(count of all 12) | −0.1×count (lighter, arm-only) |

**Key Advantage of Full-Body:** Enables more sophisticated bracing strategies with coordinated leg stabilization, better generalization to varied push directions and magnitudes.

---

## Validation Checklist

Before training:
- [ ] OP3 XML loads without errors; all joint names resolve.
- [ ] Observation shape = 31; no NaN/Inf during reset.
- [ ] Action space maps cleanly to control targets.
- [ ] Contact detection logic identifies floor collisions correctly.
- [ ] First episode runs for 200 steps without crashes.
- [ ] Reward components compute without NaN.
- [ ] Success flag evaluates correctly after episode.
- [ ] Network forward pass and policy sampling work on CPU.

---

## References

- Baseline: `scripts/brace_only_single/brace_only_single.py`
- OP3 Model: `assets/robotis_op3/op3.xml`
- MuJoCo Python API: https://mujoco.readthedocs.io/
