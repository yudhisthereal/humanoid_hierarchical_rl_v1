from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PROJECT_ROOT / "assets" / "robotis_op3" / "op3_scene.xml"


class Op3BraceEnv:
    """Single-environment OP3 full-body brace task (no Warp).

    Observation (31):
    [roll, pitch, angvel_roll, angvel_pitch,
     12 joint angles (6 arms + 6 legs),
     12 joint velocities,
     arms_contact, head_contact, feet_contact]

    Action (12): normalized in [-1, 1], mapped to arm + leg joint targets.
    
    REWARD WEIGHTS (SINGLE SOURCE OF TRUTH):
    =========================================
    r_arm_first:        5.0  (Primary: arms contacting before head)
    r_arm_sync:         1.0  (Secondary: both arms contact together)
    r_knee_timing:      1.0  (Tertiary: knees contact after arms)
    c_head_impact:    100.0  (Catastrophic: head contact = episode failure)
    c_torque:           0.2  (Efficiency: penalize high joint torques)
    c_jitter:           0.2  (Smoothness: penalize action sign flips)
    torso_pitch:       10.0  (Stability: penalize leaning beyond 17deg)
    success_bonus:    100.0  (Success: only at episode termination)
    """

    ARM_ACTUATORS = [
        "l_sho_pitch_act",
        "l_sho_roll_act",
        "l_el_act",
        "r_sho_pitch_act",
        "r_sho_roll_act",
        "r_el_act",
    ]

    LEG_ACTUATORS = [
        "l_hip_roll_act",
        "l_hip_pitch_act",
        "l_knee_act",
        "r_hip_roll_act",
        "r_hip_pitch_act",
        "r_knee_act",
    ]

    ARM_JOINTS = [
        "l_sho_pitch",
        "l_sho_roll",
        "l_el",
        "r_sho_pitch",
        "r_sho_roll",
        "r_el",
    ]

    LEG_JOINTS = [
        "l_hip_roll",
        "l_hip_pitch",
        "l_knee",
        "r_hip_roll",
        "r_hip_pitch",
        "r_knee",
    ]

    # Arm control limits
    ARM_CTRL_MIN = np.array([-1.7, -1.0, -1.0, -1.7, -1.0, -1.0], dtype=np.float32)
    ARM_CTRL_MAX = np.array([1.7, 1.0, 1.0, 1.7, 1.0, 1.0], dtype=np.float32)

    # Leg control limits
    LEG_CTRL_MIN = np.array([-0.5, -1.0, 0.0, 0.0, 0.0, -1.57], dtype=np.float32)
    LEG_CTRL_MAX = np.array([0.0, 0.0, 1.57, 0.5, 1.0, 0.0], dtype=np.float32)

    # Fixed controls (not actuated by policy)
    FIXED_ACTUATORS = {
        "head_pan_act": 0.0,
        "head_tilt_act": 0.0,
        "l_hip_yaw_act": 0.0,
        "l_ank_pitch_act": 0.0,
        "l_ank_roll_act": 0.0,
        "r_hip_yaw_act": 0.0,
        "r_ank_pitch_act": 0.0,
        "r_ank_roll_act": 0.0,
    }

    HEAD_GEOMS = ["h1c", "h2c", "h21c", "h22c"]
    LEFT_ARM_GEOMS = ["la1c", "la2c", "la3c"]
    RIGHT_ARM_GEOMS = ["ra1c", "ra2c", "ra3c"]
    LEG_GEOMS = [
        "ll1c",
        "ll2c",
        "ll3c",
        "ll4c",
        "ll5c",
        "ll6c",
        "rl1c",
        "rl2c",
        "rl3c",
        "rl4c",
        "rl5c",
        "rl6c",
    ]

    def __init__(self, xml_path: Path | str = XML_PATH):
        self.xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = 5
        self.dt = float(self.model.opt.timestep) * float(self.frame_skip)

        self.action_dim = 12
        self.obs_dim = 31

        self.obs_clip = 50.0
        self.reward_clip = 100.0
        self.episode_length = 200

        self.push_force = 90.0
        self.push_steps = 5
        self.push_kick_scale = 0.02

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.done = False
        self.last_success = False
        self.step_count = 0

        # Map actuator and joint names
        self.arm_actuator_ids = [self._must_actuator_id(n) for n in self.ARM_ACTUATORS]
        self.leg_actuator_ids = [self._must_actuator_id(n) for n in self.LEG_ACTUATORS]
        self.all_actuator_ids = self.arm_actuator_ids + self.leg_actuator_ids

        self.arm_joint_ids = [self._must_joint_id(n) for n in self.ARM_JOINTS]
        self.leg_joint_ids = [self._must_joint_id(n) for n in self.LEG_JOINTS]
        self.all_joint_ids = self.arm_joint_ids + self.leg_joint_ids

        self.arm_qpos_idx = [int(self.model.jnt_qposadr[j]) for j in self.arm_joint_ids]
        self.leg_qpos_idx = [int(self.model.jnt_qposadr[j]) for j in self.leg_joint_ids]
        self.all_qpos_idx = self.arm_qpos_idx + self.leg_qpos_idx

        self.arm_qvel_idx = [int(self.model.jnt_dofadr[j]) for j in self.arm_joint_ids]
        self.leg_qvel_idx = [int(self.model.jnt_dofadr[j]) for j in self.leg_joint_ids]
        self.all_qvel_idx = self.arm_qvel_idx + self.leg_qvel_idx

        # Fixed actuators
        self.fixed_actuator_ids: list[int] = []
        self.fixed_targets: list[float] = []
        for name, value in self.FIXED_ACTUATORS.items():
            aid = self._must_actuator_id(name)
            lo = float(self.model.actuator_ctrlrange[aid, 0])
            hi = float(self.model.actuator_ctrlrange[aid, 1])
            self.fixed_actuator_ids.append(aid)
            self.fixed_targets.append(float(np.clip(value, lo, hi)))

        # Geometry IDs for contact detection
        self.floor_geom_id = self._must_geom_id("floor")
        self.head_geom_ids = self._collect_geom_ids(self.HEAD_GEOMS)
        self.arm_left_geom_ids = self._collect_geom_ids(self.LEFT_ARM_GEOMS)
        self.arm_right_geom_ids = self._collect_geom_ids(self.RIGHT_ARM_GEOMS)
        self.leg_geom_ids = self._collect_geom_ids(self.LEG_GEOMS)

        # Body IDs
        self.body_id = self._must_body_id("body_link")
        self.root_dof_adr = int(self.model.body_dofadr[self.body_id])

        # Contact timing trackers
        self.t_arms_l = float("inf")
        self.t_arms_r = float("inf")
        self.t_head = float("inf")

    def _must_actuator_id(self, name: str) -> int:
        aid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        if aid < 0:
            raise ValueError(f"Actuator not found: {name}")
        return aid

    def _must_joint_id(self, name: str) -> int:
        jid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise ValueError(f"Joint not found: {name}")
        return jid

    def _must_geom_id(self, name: str) -> int:
        gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if gid < 0:
            raise ValueError(f"Geom not found: {name}")
        return gid

    def _must_body_id(self, name: str) -> int:
        bid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
        if bid < 0:
            raise ValueError(f"Body not found: {name}")
        return bid

    def _collect_geom_ids(self, names: list[str]) -> set[int]:
        out: set[int] = set()
        for name in names:
            gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
            if gid >= 0:
                out.add(gid)

        if not out:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(self.xml_path)
                root = tree.getroot()
                mesh_names: list[str] = []
                for geom in root.findall(".//geom"):
                    mesh_attr = geom.get("mesh")
                    name_attr = geom.get("name")
                    mesh_names.append(mesh_attr if mesh_attr is not None else (name_attr if name_attr is not None else ""))

                n = min(len(mesh_names), int(self.model.ngeom))
                for i in range(n):
                    if mesh_names[i] in names:
                        out.add(i)
            except Exception:
                out = set()

        if not out:
            raise ValueError(f"No geoms resolved from names: {names}")
        return out

    @staticmethod
    def _quat_to_roll_pitch_yaw(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
        w, x, y, z = [float(v) for v in quat_wxyz]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = np.pi / 2.0 if sinp >= 1.0 else (-np.pi / 2.0 if sinp <= -1.0 else np.arcsin(sinp))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return float(roll), float(pitch), float(yaw)

    def map_action_to_ctrl(self, action_unit: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(np.asarray(action_unit, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        a = np.clip(a, -1.0, 1.0)
        
        arm_ctrl = self.ARM_CTRL_MIN + 0.5 * (a[:6] + 1.0) * (self.ARM_CTRL_MAX - self.ARM_CTRL_MIN)
        leg_ctrl = self.LEG_CTRL_MIN + 0.5 * (a[6:12] + 1.0) * (self.LEG_CTRL_MAX - self.LEG_CTRL_MIN)
        
        return np.concatenate([arm_ctrl, leg_ctrl])

    def _has_floor_contact(self, geom_ids: set[int]) -> bool:
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 == self.floor_geom_id and g2 in geom_ids) or (g2 == self.floor_geom_id and g1 in geom_ids):
                return True
        return False

    def _update_contact_times(self) -> None:
        t = float(self.step_count) * self.dt
        if np.isinf(self.t_arms_l) and self._has_floor_contact(self.arm_left_geom_ids):
            self.t_arms_l = t
        if np.isinf(self.t_arms_r) and self._has_floor_contact(self.arm_right_geom_ids):
            self.t_arms_r = t
        if np.isinf(self.t_head) and self._has_floor_contact(self.head_geom_ids):
            self.t_head = t

    def _get_obs(self) -> np.ndarray:
        quat = np.asarray(self.data.xquat[self.body_id], dtype=np.float64)
        roll, pitch, _ = self._quat_to_roll_pitch_yaw(quat)

        cvel = np.asarray(self.data.cvel[self.body_id], dtype=np.float64)
        angvel_roll = float(cvel[0])
        angvel_pitch = float(cvel[1])

        joint_pos = np.array([float(self.data.qpos[i]) for i in self.all_qpos_idx], dtype=np.float32)
        joint_vel = np.array([float(self.data.qvel[i]) for i in self.all_qvel_idx], dtype=np.float32)

        arms_contact = 1.0 if (self._has_floor_contact(self.arm_left_geom_ids) or self._has_floor_contact(self.arm_right_geom_ids)) else 0.0
        head_contact = 1.0 if self._has_floor_contact(self.head_geom_ids) else 0.0
        feet_contact = 1.0 if self._has_floor_contact(self.leg_geom_ids) else 0.0

        obs = np.concatenate(
            [
                np.array([roll, pitch, angvel_roll, angvel_pitch], dtype=np.float32),
                joint_pos,
                joint_vel,
                np.array([arms_contact, head_contact, feet_contact], dtype=np.float32),
            ]
        )
        return np.clip(obs, -self.obs_clip, self.obs_clip)

    def _apply_controls(self, all_ctrl: np.ndarray) -> None:
        ctrl = np.array(self.data.ctrl, copy=True)
        
        for i, aid in enumerate(self.all_actuator_ids):
            lo = float(self.model.actuator_ctrlrange[aid, 0])
            hi = float(self.model.actuator_ctrlrange[aid, 1])
            ctrl[aid] = float(np.clip(float(all_ctrl[i]), lo, hi))

        for aid, target in zip(self.fixed_actuator_ids, self.fixed_targets):
            ctrl[aid] = float(target)

        self.data.ctrl[:] = ctrl

    def _compute_reward(self, action: np.ndarray, success: bool, done: bool, pitch: float) -> tuple[float, dict[str, float]]:
        """
        REWARD COMPUTATION - WEIGHTS ARE APPLIED HERE (SINGLE SOURCE OF TRUTH)
        
        Raw components (logged for analysis):
        - r_arm_first_raw:  arms contact timing [0, 1]
        - r_arm_sync_raw:   arm synchronization [0, 1]
        - r_knee_timing_raw: knee timing [0, 1]
        - c_head_impact_raw: 1 if head contact, else 0 (binary)
        - c_torque_raw:     sum of absolute joint torques (capped at 50 offset)
        - c_jitter_raw:     number of action sign flips
        - torso_pitch_raw:  radians of lean beyond 17deg [0, inf)
        - success_raw:      1 if success, else 0
        
        Weighted contributions (what actually affects reward):
        - r_arm_first:      5.0 * r_arm_first_raw
        - r_arm_sync:       1.0 * r_arm_sync_raw
        - r_knee_timing:    1.0 * r_knee_timing_raw
        - c_head_impact:   -100.0 * c_head_impact_raw (episode ends immediately)
        - c_torque:         -0.2 * c_torque_raw
        - c_jitter:         -0.2 * c_jitter_raw
        - torso_pitch:     -10.0 * torso_pitch_raw
        - success_bonus:   +100.0 if success and done else 0
        """
        
        # ===== RAW COMPONENT CALCULATIONS =====
        
        # R1: Arm-first strategy (raw: 0 to 1)
        arms_contacted = not np.isinf(self.t_arms_l) and not np.isinf(self.t_arms_r)
        t_arms_min = min(self.t_arms_l, self.t_arms_r)
        r_arm_first_raw = float(np.tanh(max(self.t_head - t_arms_min, 0.0) / 1.0)) if arms_contacted else 0.0
        
        # R2: Arm synchronization (raw: 0 to 1)
        if arms_contacted:
            r_arm_sync_raw = 1.0 - float(np.tanh(abs(self.t_arms_l - self.t_arms_r)))
        else:
            r_arm_sync_raw = 0.0
        
        # R3: Knee timing (raw: 0 to 1)
        legs_contacted = self._has_floor_contact(self.leg_geom_ids)
        if legs_contacted and arms_contacted:
            r_knee_timing_raw = 1.0 - float(np.tanh(abs(max(self.t_head - t_arms_min, 0.0)) / 0.2))
        else:
            r_knee_timing_raw = 0.0
        
        # C1: Head impact (raw: 0 or 1)
        head_contact = self._has_floor_contact(self.head_geom_ids)
        c_head_impact_raw = 1.0 if head_contact else 0.0
        
        # C2: Torque cost (raw: sum of torques, capped at 50 offset)
        torque_sum = float(np.sum(np.abs(np.asarray(self.data.actuator_force[self.all_actuator_ids], dtype=np.float32))))
        c_torque_raw = max(0.0, torque_sum - 50.0)
        
        # C3: Torso pitch penalty (raw: radians beyond 0.3 rad threshold)
        torso_pitch_raw = max(0.0, abs(pitch) - 0.3) * 10.0  # Multiplied by 10 here for raw value
        
        # C4: Action jitter (raw: count of sign flips)
        prev = np.asarray(self.prev_action, dtype=np.float32)
        curr = np.asarray(action, dtype=np.float32)
        mask = (np.abs(prev) >= 0.5) & (np.abs(curr) >= 0.5) & (np.sign(prev) != np.sign(curr))
        c_jitter_raw = float(np.sum(mask))
        
        # Success bonus (raw: 0 or 1)
        success_raw = 1.0 if (done and success) else 0.0
        
        # ===== WEIGHTED REWARD COMPUTATION (SINGLE SOURCE OF TRUTH) =====
        # Weights are applied HERE, not in the tracker
        WEIGHT_ARM_FIRST = 5.0
        WEIGHT_ARM_SYNC = 1.0
        WEIGHT_KNEE_TIMING = 1.0
        WEIGHT_HEAD_IMPACT = 100.0  # Catastrophic penalty
        WEIGHT_TORQUE = 0.2
        WEIGHT_JITTER = 0.2
        WEIGHT_TORSO_PITCH = 10.0
        WEIGHT_SUCCESS_BONUS = 100.0
        
        reward = (WEIGHT_ARM_FIRST * r_arm_first_raw +
                  WEIGHT_ARM_SYNC * r_arm_sync_raw +
                  WEIGHT_KNEE_TIMING * r_knee_timing_raw)
        
        reward -= (WEIGHT_HEAD_IMPACT * c_head_impact_raw +
                   WEIGHT_TORQUE * c_torque_raw +
                   WEIGHT_JITTER * c_jitter_raw +
                   WEIGHT_TORSO_PITCH * torso_pitch_raw)
        
        reward += WEIGHT_SUCCESS_BONUS * success_raw
        
        # Clip final reward
        reward = float(np.clip(np.nan_to_num(reward, nan=0.0), -self.reward_clip, self.reward_clip))
        
        # Return reward and RAW components for logging (tracker will NOT apply weights)
        return reward, {
            # Raw components (what the tracker will log)
            "r_arm_first_raw": r_arm_first_raw,
            "r_arm_sync_raw": r_arm_sync_raw,
            "r_knee_timing_raw": r_knee_timing_raw,
            "c_head_impact_raw": c_head_impact_raw,
            "c_torque_raw": c_torque_raw,
            "c_jitter_raw": c_jitter_raw,
            "torso_pitch_raw": torso_pitch_raw,
            "success_raw": success_raw,
            # Weighted contributions (for debugging, but tracker won't use these)
            "r_arm_first_weighted": WEIGHT_ARM_FIRST * r_arm_first_raw,
            "r_arm_sync_weighted": WEIGHT_ARM_SYNC * r_arm_sync_raw,
            "r_knee_timing_weighted": WEIGHT_KNEE_TIMING * r_knee_timing_raw,
            "c_head_impact_weighted": -WEIGHT_HEAD_IMPACT * c_head_impact_raw,
            "c_torque_weighted": -WEIGHT_TORQUE * c_torque_raw,
            "c_jitter_weighted": -WEIGHT_JITTER * c_jitter_raw,
            "torso_pitch_weighted": -WEIGHT_TORSO_PITCH * torso_pitch_raw,
            "success_bonus_weighted": WEIGHT_SUCCESS_BONUS * success_raw,
        }

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0
        self.done = False
        self.last_success = False
        self.prev_action[:] = 0.0

        self.t_arms_l = float("inf")
        self.t_arms_r = float("inf")
        self.t_head = float("inf")

        self.data.qpos[2] += 0.05
        self._apply_controls(np.zeros(self.action_dim, dtype=np.float32))

        if hasattr(self.data, "xfrc_applied"):
            self.data.xfrc_applied[:] = 0.0
        push_impulse = float(self.push_force) * (self.dt * self.push_kick_scale)
        self.data.qvel[self.root_dof_adr + 0] += push_impulse
        for _ in range(int(self.push_steps)):
            if hasattr(self.data, "xfrc_applied"):
                self.data.xfrc_applied[:] = 0.0
                self.data.xfrc_applied[self.body_id, 0] = float(self.push_force)
            self._apply_controls(np.zeros(self.action_dim, dtype=np.float32))
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)

        if hasattr(self.data, "xfrc_applied"):
            self.data.xfrc_applied[:] = 0.0

        self.t_arms_l = float("inf")
        self.t_arms_r = float("inf")
        self.t_head = float("inf")

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self._get_obs(), 0.0, True, {"success": self.last_success}

        action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

        all_ctrl = self.map_action_to_ctrl(action)
        self._apply_controls(all_ctrl)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._update_contact_times()

        head_contact = self._has_floor_contact(self.head_geom_ids)
        timeout = self.step_count >= (self.episode_length - 1)
        invalid_state = (not np.isfinite(self.data.qpos).all()) or (not np.isfinite(self.data.qvel).all())

        success = (
            np.isinf(self.t_head)
            and np.isfinite(self.t_arms_l)
            and np.isfinite(self.t_arms_r)
            and (self.t_arms_l < self.t_head)
            and (self.t_arms_r < self.t_head)
            and (abs(self.t_arms_l - self.t_arms_r) <= 0.1)
        )

        done = bool(head_contact or timeout or invalid_state)
        
        quat = np.asarray(self.data.xquat[self.body_id], dtype=np.float64)
        _, pitch, _ = self._quat_to_roll_pitch_yaw(quat)
        
        reward, parts = self._compute_reward(action=action, success=bool(success), done=done, pitch=pitch)

        self.last_success = bool(success and done)
        self.done = done
        self.step_count += 1
        self.prev_action = action.copy()

        info = {
            "success": self.last_success,
            "head_contact": bool(head_contact),
            "timeout": bool(timeout),
            "invalid_state": bool(invalid_state),
            **parts,
        }
        return self._get_obs(), reward, done, info

    def get_success(self) -> bool:
        return bool(self.last_success)