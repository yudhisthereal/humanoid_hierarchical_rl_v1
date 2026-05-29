# OP3 Arm-Only Brace Environment System Design

## Recent Changes
- 2026-05-19: Default `head_tilt` changed from -1.0 to 0.0 (level).
- 2026-05-19: Reward shaping updated: immediate arm-contact reward, survival bonus for avoiding head contact, stricter arm-sync tolerance, and tuned torque/jitter penalties.
- 2026-05-19: Training fixes applied (PPONetwork outputs now tanh-squashed; sampling/log-prob corrected; PPO old_log_probs bug fixed).
- 2026-05-21: Push applied during reset (before episode starts), not during episode steps.

## Overview

The OP3 Arm-Only Brace environment is a simplified variant of the full OP3 Brace task. The robot learns to control **only its arms** to brace itself when pushed, without any hip or knee actuation. The legs remain rigid (fixed in place), and all bracing effort must come from arm extension. This is a more constrained, arm-focused variant useful for studying upper-body impact absorption.

**Key Design Principle:** No Warp GPU parallelization; single-env, pure MuJoCo CPU simulation with PyTorch for learning. Arm-only control; legs locked at zero angles.

---

## Observation Space

**Dimension: 19** (continuous, clipped to [−50, +50] per element)

| Index | Name | Unit | Description |
|-------|------|------|-------------|
| 0 | `body_roll` | rad | Body roll angle (rotation around X) |
| 1 | `body_pitch` | rad | Body pitch angle (rotation around Y) |
| 2–3 | `body_angvel_rp` | rad/s | Angular velocity (roll rate, pitch rate) |
| 4–9 | `joint_angles` | rad | 6 actuated arm joint positions: `[l_sho_pitch, l_sho_roll, l_el, r_sho_pitch, r_sho_roll, r_el]` |
| 10–15 | `joint_velocities` | rad/s | 6 actuated arm joint velocities: `[l_sho_pitch_vel, l_sho_roll_vel, l_el_vel, r_sho_pitch_vel, r_sho_roll_vel, r_el_vel]` |
| 16–18 | `contact_flags` | {0,1} | 3 binary contact indicators: `[arms_contact, head_contact, feet_contact]` (1 = actual MuJoCo contact with floor, 0 = no contact) |

### Notes
- **Body orientation** extracted from the root (free) joint quaternion via Euler angle conversion.
- **Body angular velocity** (roll, pitch rates) extracted from body frame angular velocity.
- **Arm joint angles and velocities** are absolute positions and velocities for the 6 shoulder/elbow joints.
- **Contact flags:** arms_contact (arms hit floor), head_contact (head hits floor = failure), feet_contact (feet/legs hit floor).
- **Contact detection:** Via MuJoCo contact array (`data.ncon`, `data.contact[i]`).

---

## Action Space

**Dimension: 6** (continuous, normalized to [−1, 1])

Normalized actions are mapped to absolute joint position targets via affine scaling:

$$\text{target}_j = \text{ctrl\_min}_j + 0.5 \cdot (\text{action}_j + 1.0) \cdot (\text{ctrl\_max}_j - \text{ctrl\_min}_j)$$

### Actuated Joints (Arms Only)

| Index | Joint Name | Control Range [rad] | Notes |
|-------|-----------|---|---|
| 0 | `l_sho_pitch` | [−1.7, 1.7] | Left shoulder pitch |
| 1 | `l_sho_roll` | [−1, 1] | Left shoulder roll |
| 2 | `l_el` | [−1, 1] | Left elbow |
| 3 | `r_sho_pitch` | [−1.7, 1.7] | Right shoulder pitch |
| 4 | `r_sho_roll` | [−1, 1] | Right shoulder roll |
| 5 | `r_el` | [−1, 1] | Right elbow |

### Fixed (Non-Actuated) Joints

| Joint | Value [rad] | Notes |
|-------|---|---|
| `head_pan` | 0.0 | Always straight ahead |
| `head_tilt` | 0.0 | Always fixed level |
| `l_hip_yaw` | 0.0 | Locked |
| `r_hip_yaw` | 0.0 | Locked |
| `l_hip_roll` | 0.0 | Locked |
| `r_hip_roll` | 0.0 | Locked |
| `l_hip_pitch` | 0.0 | Locked |
| `r_hip_pitch` | 0.0 | Locked |
| `l_knee` | 0.0 | Locked |
| `r_knee` | 0.0 | Locked |
| `l_ank_pitch` | 0.0 | Locked |
| `l_ank_roll` | 0.0 | Locked |
| `r_ank_pitch` | 0.0 | Locked |
| `r_ank_roll` | 0.0 | Locked |

**Implementation:** During environment step, all fixed joints are explicitly set to their target values in the control input before calling `mujoco.mj_step()`.

---

## Policy Network Architecture

**Class: `Op3ArmPPONetwork(nn.Module)`**

```
Input: obs (shape [batch] or [])
  ↓
Shared Backbone:
  Linear(19 → 64) + Tanh
  Linear(64 → 64) + Tanh
  ↓
Policy Head:
  Linear(64 → 6)  →  mean (action means)
  
Value Head:
  Linear(64 → 1)   →  value (scalar critic estimate)

Learned Parameters:
  policy_logstd: Parameter(shape=[6])  (log std of Gaussian policy)
```

**Policy Distribution:** Gaussian with state-dependent mean and learned fixed variance (diagonal covariance).

---

## Training Hyperparameters

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
| | MuJoCo substeps per step | 5 |
| **Stopping** | Target success rate | ≥95% (over 10-episode window) |
| | Early stop condition | Maintain ≥95% for 100+ consecutive episodes |
| **Checkpointing** | Save interval | Every 10 iterations |
| | Maximum iterations | 1000 |

---

## Reward Function

Per-step reward is computed as a combination of contact-sequence reward, torque penalty, jitter penalty, and a success bonus applied at episode end.

### 1. Arm Contact Reward ($r_{arms\_contact}$)

Encourages both arms to make contact with the ground:

$$r_{arms\_contact} = \begin{cases} 5.0 & \text{if both arms have contacted floor} \\ 0 & \text{otherwise} \end{cases}$$

Contact detection uses MuJoCo contact array (actual geometric collision with floor geom).

### 2. Arm Synchronization Reward ($r_{arm\_sync}$)

Rewards synchronized left/right arm contact when both arms have contacted:

$$r_{arm\_sync} = 1.0 - \tanh\left(\frac{|t_{arms\_left} - t_{arms\_right}|}{0.2}\right)$$

where $t_{arms\_left}$ and $t_{arms\_right}$ are first contact times from MuJoCo contact array.

- Returns 0 if not both arms have contacted
- Stricter sync tolerance (0.2 denominator) encourages near-simultaneous contact

**Weight in total reward:** $1.0$

### 3. Head Impact Penalty ($c_{head\_impact}$)

Binary penalty applied when head makes actual contact with floor:

$$c_{head\_impact} = \begin{cases} 100.0 & \text{if head contacts floor (MuJoCo contact)} \\ 0 & \text{otherwise} \end{cases}$$

**Note:** Contact detection uses MuJoCo contact array. Head contact also triggers immediate episode termination.

### 4. Survival Bonus ($r_{survival}$)

Small positive reward per step for keeping head off the ground:

$$r_{survival} = \begin{cases} 0.01 & \text{if no head contact} \\ 0 & \text{if head contact} \end{cases}$$

Encourages the robot to keep its head up.

### 5. Torque Cost ($c_{torque}$)

Positive penalty for high actuator effort:

$$c_{torque} = 0.01 \times \sum_{j=1}^{6} |\tau_j|$$

where $\tau_j$ is the actuator force for arm joint $j$ (`data.actuator_force[actuator_id]`).

This cost is **subtracted** from the reward.

### 6. Jitter Cost ($c_{jitter}$)

Penalty for action sign-flips that indicate oscillatory behavior:

$$c_{jitter} = 0.1 \times n_{jitter}$$

where $n_{jitter}$ is the count of actuators (0-6) satisfying:
- Previous action magnitude ≥ 0.5
- Current action magnitude ≥ 0.5  
- Sign(prev) ≠ Sign(curr)

This cost is **subtracted** from the reward.

## Reward Function

### Total Per-Step Reward Formula

$$R_t = r_{\text{arms\_contact}} + r_{\text{arm\_sync}} - c_{\text{head\_impact}} + r_{\text{survival}} - c_{\text{torque}} - c_{\text{jitter}}$$

**Success Bonus (added at episode termination only):**

$$R_{\text{success}} = \begin{cases} 100.0 & \text{if } done = \text{True and } success = \text{True} \\ 0 & \text{otherwise} \end{cases}$$

**Final reward returned by `step()`:**

$$\text{reward} = \text{clip}(R_t + R_{\text{success}}, -100.0, 100.0)$$

### Reward Components

| Component | Symbol | Weight | Description |
|-----------|--------|--------|-------------|
| Arm Contact | `r_arms_contact` | 5.0 | Reward given when both arms have made contact with the floor. Contact is detected via MuJoCo contact array (actual geometric collision with floor geom). Once both arms have contacted, this reward is given on every subsequent step of the episode. |
| Arm Synchronization | `r_arm_sync` | 1.0 | Reward for near-simultaneous arm contact. Calculated as $1.0 - \tanh(\lvert t_L - t_R \rvert / 0.2)$ where $t_L$ and $t_R$ are the first contact times from the MuJoCo contact array. Only applies after both arms have contacted the floor. The denominator of 0.2 creates a stricter synchronization requirement, heavily rewarding contacts within ~0.2 seconds of each other. |
| Head Impact Penalty | `c_head_impact` | 100.0 | Penalty applied when the head makes contact with the floor. Contact is detected via MuJoCo contact array. This penalty is applied on the step where head contact occurs, and the episode also terminates immediately with success = False. |
| Survival Bonus | `r_survival` | 0.01 | Small positive reward given on each step where the head is NOT in contact with the floor. Encourages the robot to keep its head elevated throughout the episode. |
| Torque Penalty | `c_torque` | 0.01 | Penalty for high actuator effort. Calculated as $0.01 \times \sum_{j=1}^{6} \lvert \tau_j \rvert$ summed over the 6 actuated arm joints, where $\tau_j$ is the actuator force from `data.actuator_force`. Encourages energy-efficient bracing motions. |
| Jitter Penalty | `c_jitter` | 0.1 | Penalty for action sign-flips that indicate oscillatory or unstable control. Calculated as $0.1 \times n_{\text{jitter}}$ where $n_{\text{jitter}}$ counts the number of actuators (0-6) satisfying: previous action magnitude $\geq 0.5$, current action magnitude $\geq 0.5$, and $\text{sign}(a_{t-1}) \neq \text{sign}(a_t)$. Encourages smooth, coherent arm movements. |
| Success Bonus | `R_success` | 100.0 | One-time bonus applied only at episode termination when all success criteria are met. Success requires: head never touched floor during episode, both arms contacted floor, arms contacted before head, and arm contact times synchronized within 0.1 seconds. |

### Component Return Values (via `info` dict)

| Key | Description |
|-----|-------------|
| `r_arms_contact` | Arm contact reward value (5.0 or 0.0) |
| `r_arm_sync` | Synchronization reward value (0.0 to 1.0) |
| `r_survival` | Survival bonus value (0.01 or 0.0) |
| `c_torque` | Torque penalty value (≥ 0.0) |
| `c_jitter` | Jitter penalty value (≥ 0.0) |
| `c_head_impact` | Head impact penalty value (100.0 or 0.0) |
| `success_bonus` | Success bonus value (100.0 or 0.0, non-zero only at episode termination) |

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

1. **Head never touches ground** during the episode (no MuJoCo contact between head geoms and floor at any step).
2. **Left arm contacted:** $t_{arms\_left}$ is finite (arm made contact).
3. **Right arm contacted:** $t_{arms\_right}$ is finite (arm made contact).
4. **Arms contact before head:** $t_{arms\_left} < t_{head}$ AND $t_{arms\_right} < t_{head}$.
5. **Arms synchronized:** $|t_{arms\_left} - t_{arms\_right}| \leq 0.1$ seconds.

**Note:** Contact times are recorded only when actual MuJoCo contact is detected with the floor geom. No contact occurs before episode starts (contact timers are reset after push).

### Training Stopping Condition

Training halts early when:
- Success rate ≥ 95% is achieved over a 10-episode rolling window, **AND**
- That ≥95% success rate is **maintained for at least 100 consecutive additional episodes**.

---

## Initialization & Reset

### Episode Reset (`env.reset()`)

The push is applied **during reset** before the policy receives the first observation. The policy only sees the post-push state.

Reset sequence:
1. Reset all joint positions and velocities to zero (via `mujoco.mj_resetData()`).
2. Raise COM slightly (+0.05 m in Z) to avoid initial floor penetration.
3. Set fixed joints explicitly to zero (head pan/tilt, all hip/knee/ankle joints).
4. **Apply push disturbance:**
   - Forward force: **90.0 N** applied to body
   - Forward velocity kick: `push_force * (dt * push_kick_scale)`
   - Simulated for `push_steps = 5` steps (~0.025 seconds)
5. Reset contact timers to infinity **after** push (so only recovery phase counts for success).
6. Return initial observation (19-dim).

**Key:** No push is applied during `step()` calls. The push is entirely contained within `reset()`.

### Reset Returns
- Initial observation (19-dim) from post-push state.

---

## Geometry & Collision

### Key Collision Geoms

| Category | OP3 Geom Names | Purpose |
|----------|---|---|
| **Head** | `h1c`, `h2c`, `h21c`, `h22c` | Head collision meshes |
| **Arms (Left)** | `la1c`, `la2c`, `la3c` | Left arm links |
| **Arms (Right)** | `ra1c`, `ra2c`, `ra3c` | Right arm links |
| **Legs (Left)** | `ll1c`–`ll6c` | Left leg link meshes (rigid; not actuated) |
| **Legs (Right)** | `rl1c`–`rl6c` | Right leg link meshes (rigid; not actuated) |
| **Body** | `bodyc`, `body1c`–`body4c` | Torso collision meshes |
| **Floor** | `floor` (world geom) | Ground plane |

### Contact Detection Algorithm

**All contact timing and detection must use MuJoCo contact array (`data.contact[0:data.ncon]`).**

For each step:
1. Iterate through all contacts: `for i in range(int(data.ncon))`
2. Get contact pair: `c = data.contact[i]; geom1 = int(c.geom1); geom2 = int(c.geom2)`
3. Determine if floor is involved: `floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor'))`
4. Check limb membership: if `geom1 == floor_id` and `geom2` in arm/head set, record contact time for that limb. (Or vice versa for `geom2 == floor_id`.)
5. Record **first** contact time for each limb; ignore subsequent contacts.

**No position-based estimation:** Contact is only registered when MuJoCo's contact solver detects a collision.

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

### Clipping & Bounds
- Observations clipped to [−50, +50].
- Rewards clipped to [−100, +100].
- Actions clipped to [−1, +1] before mapping to control ranges.

---

## File Structure

```
envs/op3_arm_brace_v2/
├── __init__.py           # Package init, exports Op3ArmBraceEnv
├── env.py                # Core environment class (this file)
├── op3.xml               # OP3 model (copy or reference from assets/)
└── system_explanation.md # This document
```

---

## Differences from `op3_brace`

| Aspect | OP3 Brace (Full) | OP3 Arm Brace (v2) |
|--------|---|---|
| **Actuated Joints** | 12 (6 arms + 6 legs) | 6 (arms only) |
| **Action Dim** | 12 | 6 |
| **Obs Dim** | 31 | 19 |
| **Obs Components** | Body pose (R/P), angular velocity, 12 joint angles/vels, contact flags | Body pose (R/P), angular velocity, 6 arm angles/vels, contact flags |
| **Fixed Joints** | Head, hip yaw, ankles | Head, all legs (hip yaw, pitch, roll, knee, ankle) |
| **Contact Reward** | Arm-first, arm-sync, knee timing | Arm-first, arm-sync only (no leg contact) |
| **Success Condition** | Arms first + arm sync + no head touch + knees? | Arms first + arm sync + no head touch |
| **Task Difficulty** | Robot can use legs to stabilize | Arms only; more difficult; pure upper-body bracing |
| **Push Timing** | During reset only | During reset only |

---

## Validation Checklist

Before training:
- [ ] OP3 XML loads without errors; all arm joint names resolve.
- [ ] All leg joints are fixed to zero and cannot be actuated.
- [ ] Observation shape = 19; no NaN/Inf during reset.
- [ ] Action space maps cleanly to 6 arm control targets.
- [ ] Contact detection logic identifies floor collisions correctly.
- [ ] Push is applied during reset, not during steps.
- [ ] Contact timers reset after push (t_arms_l/r/t_head = inf after reset).
- [ ] First episode runs for 200 steps without crashes.
- [ ] Reward components compute without NaN.
- [ ] Success flag evaluates correctly after episode.
- [ ] Network forward pass and policy sampling work on CPU.

---

## References

- Baseline: `scripts/brace_only_single/brace_only_single.py`
- Full OP3 Brace: `envs/op3_brace/`
- OP3 Model: `assets/robotis_op3/op3.xml`
- MuJoCo Python API: https://mujoco.readthedocs.io/
