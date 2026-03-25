from __future__ import annotations

from contextlib import nullcontext
import math
import xml.etree.ElementTree as ET
from enum import IntEnum
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp


class GoalId(IntEnum):
    BRACE = 0
    ROLL = 1


class GoalConditionedExecutorEnv:
    def __init__(
        self,
        model_xml: str,
        num_envs: int = 4096,
        dt: float = 0.02,
        episode_length: int = 100,
        device: str = "cuda",
        enable_step_timing: bool = False,
        nconmax: int = 512,
        njmax: int = 1024,
    ):
        wp.init()
        self.num_envs = num_envs
        self.dt = dt
        self.episode_length = episode_length
        self.device = torch.device(device)
        self.enable_step_timing = enable_step_timing
        self.obs_clip = 50.0
        self.reward_clip = 100.0
        self.nconmax = int(nconmax)
        self.njmax = int(njmax)

        ctrl_min, ctrl_max = self._parse_ctrl_limits(model_xml)
        self.ctrl_min = ctrl_min.to(self.device)
        self.ctrl_max = ctrl_max.to(self.device)
        self.action_dim = self.ctrl_min.numel()

        # Goal-conditioned split: 1024 BRACE, 3072 ROLL
        self.goal_id = torch.full((self.num_envs,), GoalId.ROLL, device=self.device, dtype=torch.long)
        self.goal_id[:1024] = GoalId.BRACE

        self.success_counter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.achieved_goal = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)

        self.push_force = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.push_steps_left = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        self.step_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.done = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)

        # State tensors
        self.vx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.omega = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.rotation = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.com_z = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.head_z = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.waist_angle = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.knees_angle = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # Contacts / timing
        inf = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.t_arms_l = inf.clone()
        self.t_arms_r = inf.clone()
        self.t_knees = inf.clone()
        self.t_head = inf.clone()
        self.t_torso = inf.clone()
        self.t_waist = inf.clone()

        self.obs_torch = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.obs_wp = wp.from_torch(self.obs_torch, dtype=wp.float32)

        self.last_action = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)

        # MuJoCo Warp batched physics setup
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.mj_data = mujoco.MjData(self.mj_model)
        if int(self.mj_model.nu) != int(self.action_dim):
            raise ValueError(
                f"Actuator mismatch: parsed action_dim={self.action_dim} from XML ctrlranges, "
                f"but model.nu={int(self.mj_model.nu)}"
            )
        self._validate_actuator_joint_range_match()
        self.model = mjw.put_model(self.mj_model)
        try:
            self.data = mjw.put_data(
                self.mj_model,
                self.mj_data,
                nworld=self.num_envs,
                nconmax=self.nconmax,
                njmax=self.njmax,
            )
        except TypeError:
            # Compatibility fallback for older/newer mjw.put_data signatures.
            self.data = mjw.put_data(self.mj_model, self.mj_data, nworld=self.num_envs)

        self.qpos_torch = wp.to_torch(self.data.qpos)
        self.qvel_torch = wp.to_torch(self.data.qvel)
        self.ctrl_torch = wp.to_torch(self.data.ctrl)
        self.xfrc_torch = wp.to_torch(self.data.xfrc_applied) if hasattr(self.data, "xfrc_applied") else None

        self._qpos_default = self.qpos_torch[0].clone()
        self._qvel_default = self.qvel_torch[0].clone()
        self._ctrl_default = self.ctrl_torch[0].clone()

        self._rootx_qvel_idx = self._joint_qvel_idx("rootx")
        self._rootz_qpos_idx = self._joint_qpos_idx("rootz")
        self._rootz_qvel_idx = self._joint_qvel_idx("rootz")
        self._rooty_qpos_idx = self._joint_qpos_idx("rooty")
        self._rooty_qvel_idx = self._joint_qvel_idx("rooty")
        self._waist_qpos_idx = self._joint_qpos_idx("waist_joint")
        self._leg_qpos_idx = self._joint_qpos_idx_any("leg_joint", "shin_joint")
        self._leg_left_qpos_idx = self._joint_qpos_idx_any("leg_left_joint", "shin_left_joint")
        self._torso_body_id = self._body_id("torso")
        self._head_geom_id = self._geom_id("head")

        # Training-time push profile (forward x-direction).
        self.push_force_choices = torch.tensor([20.0, 35.0, 70.0], device=self.device)
        self.push_steps = 5
        self.push_kick_scale = 0.02

        self.geom_xpos_torch = wp.to_torch(self.data.geom_xpos) if hasattr(self.data, "geom_xpos") else None

        self._step_graph = None
        with wp.ScopedCapture() as capture:
            mjw.step(self.model, self.data)
        self._step_graph = capture.graph
        mjw.reset_data(self.model, self.data)

    @staticmethod
    def _parse_ctrl_limits(model_xml: str) -> Tuple[torch.Tensor, torch.Tensor]:
        xml_path = Path(model_xml)
        root = ET.parse(xml_path).getroot()
        mins = []
        maxs = []
        for actuator in root.findall("./actuator/position"):
            rng = actuator.attrib.get("ctrlrange", "-1 1").split()
            lo = float(rng[0])
            hi = float(rng[1])
            mins.append(lo)
            maxs.append(hi)
        return torch.tensor(mins, dtype=torch.float32), torch.tensor(maxs, dtype=torch.float32)

    def _joint_qpos_idx(self, name: str) -> int:
        j_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise ValueError(f"Joint not found in model: {name}")
        return int(self.mj_model.jnt_qposadr[j_id])

    def _joint_qpos_idx_any(self, *names: str) -> int:
        for name in names:
            j_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id >= 0:
                return int(self.mj_model.jnt_qposadr[j_id])
        raise ValueError(f"None of joints found in model: {names}")

    def _joint_qvel_idx(self, name: str) -> int:
        j_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise ValueError(f"Joint not found in model: {name}")
        return int(self.mj_model.jnt_dofadr[j_id])

    def _body_id(self, name: str) -> int:
        b_id = int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name))
        if b_id < 0:
            raise ValueError(f"Body not found in model: {name}")
        return b_id

    def _geom_id(self, name: str) -> int:
        g_id = int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if g_id < 0:
            raise ValueError(f"Geom not found in model: {name}")
        return g_id

    def map_action_to_ctrl(self, action_unit: torch.Tensor) -> torch.Tensor:
        # action_unit in [-1, 1] -> actuator ctrlrange
        a = torch.nan_to_num(action_unit, nan=0.0, posinf=1.0, neginf=-1.0)
        a = torch.clamp(a, -1.0, 1.0)
        return self.ctrl_min + 0.5 * (a + 1.0) * (self.ctrl_max - self.ctrl_min)

    def _validate_actuator_joint_range_match(self, atol: float = 1e-6) -> None:
        """Ensure each position actuator ctrlrange matches its target joint range."""
        for act_id in range(int(self.mj_model.nu)):
            j_id = int(self.mj_model.actuator_trnid[act_id, 0])
            if j_id < 0:
                continue

            # Skip joints that are not range-limited.
            if int(self.mj_model.jnt_limited[j_id]) == 0:
                continue

            ctrl_lo = float(self.mj_model.actuator_ctrlrange[act_id, 0])
            ctrl_hi = float(self.mj_model.actuator_ctrlrange[act_id, 1])
            jnt_lo = float(self.mj_model.jnt_range[j_id, 0])
            jnt_hi = float(self.mj_model.jnt_range[j_id, 1])

            if abs(ctrl_lo - jnt_lo) > atol or abs(ctrl_hi - jnt_hi) > atol:
                act_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or str(act_id)
                jnt_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or str(j_id)
                raise ValueError(
                    "Actuator ctrlrange must match joint range: "
                    f"actuator={act_name} ctrlrange=({ctrl_lo}, {ctrl_hi}) "
                    f"joint={jnt_name} range=({jnt_lo}, {jnt_hi})"
                )

    def _update_goal_buffer(self):
        reached = self.success_counter >= 1000
        if not torch.any(reached):
            return

        idx = torch.nonzero(reached, as_tuple=False).squeeze(-1)
        current_goal = self.goal_id[idx]
        self.achieved_goal[idx, current_goal] = True

        # assign next unachieved goal (BRACE -> ROLL or ROLL -> BRACE)
        next_goal = 1 - current_goal
        has_next = ~self.achieved_goal[idx, next_goal]
        self.goal_id[idx] = torch.where(has_next, next_goal, current_goal)
        self.success_counter[idx] = 0

    def reset(self, env_mask: torch.Tensor | None = None) -> torch.Tensor:
        if env_mask is None:
            env_mask = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)

        idx = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return wp.to_torch(self.obs_wp)

        self.qpos_torch[idx] = self._qpos_default
        self.qvel_torch[idx] = self._qvel_default
        self.ctrl_torch[idx] = self._ctrl_default
        if self.xfrc_torch is not None:
            self.xfrc_torch[idx] = 0.0

        # robot starts slightly above ground
        self.qpos_torch[idx, self._rootz_qpos_idx] = self._qpos_default[self._rootz_qpos_idx] + 0.05
        self.qvel_torch[idx, self._rootx_qvel_idx] = 0.0
        self.qvel_torch[idx, self._rootz_qvel_idx] = 0.0
        self.qvel_torch[idx, self._rooty_qvel_idx] = 0.0

        self.step_count[idx] = 0
        self.done[idx] = False

        self.com_z[idx] = self.qpos_torch[idx, self._rootz_qpos_idx]
        self.head_z[idx] = self.com_z[idx] + 0.3
        self.vx[idx] = self.qvel_torch[idx, self._rootx_qvel_idx]
        self.omega[idx] = self.qvel_torch[idx, self._rooty_qvel_idx]
        self.rotation[idx] = self.qpos_torch[idx, self._rooty_qpos_idx]
        self.waist_angle[idx] = self.qpos_torch[idx, self._waist_qpos_idx]
        self.knees_angle[idx] = 0.5 * (
            self.qpos_torch[idx, self._leg_qpos_idx] + self.qpos_torch[idx, self._leg_left_qpos_idx]
        )

        inf = torch.full((idx.numel(),), float("inf"), device=self.device)
        self.t_arms_l[idx] = inf
        self.t_arms_r[idx] = inf
        self.t_knees[idx] = inf
        self.t_head[idx] = inf
        self.t_torso[idx] = inf
        self.t_waist[idx] = inf

        # Disturbance: forward torso push starts immediately and lasts multiple steps.
        sampled = torch.randint(0, self.push_force_choices.numel(), (idx.numel(),), device=self.device)
        self.push_force[idx] = self.push_force_choices[sampled]
        self.push_steps_left[idx] = self.push_steps
        self.qvel_torch[idx, self._rootx_qvel_idx] = (
            self.qvel_torch[idx, self._rootx_qvel_idx] + self.push_force[idx] * (self.dt * self.push_kick_scale)
        )

        self.obs_torch[idx, 0] = self.vx[idx]
        self.obs_torch[idx, 1] = self.omega[idx]
        self.obs_torch[idx, 2] = self.rotation[idx]
        self.obs_torch[idx, 3] = self.com_z[idx]
        self.obs_torch[idx, 4] = self.head_z[idx]
        self.obs_torch[idx, 5] = self.waist_angle[idx]
        self.obs_torch[idx, 6] = self.knees_angle[idx]
        return wp.to_torch(self.obs_wp)

    def _refresh_state_from_sim(self):
        self.vx = self.qvel_torch[:, self._rootx_qvel_idx]
        self.omega = self.qvel_torch[:, self._rooty_qvel_idx]
        self.rotation = self.qpos_torch[:, self._rooty_qpos_idx]
        self.com_z = self.qpos_torch[:, self._rootz_qpos_idx]

        if self.geom_xpos_torch is not None and self._head_geom_id >= 0:
            self.head_z = self.geom_xpos_torch[:, self._head_geom_id, 2]
        else:
            self.head_z = self.com_z + 0.3

        self.waist_angle = self.qpos_torch[:, self._waist_qpos_idx]
        self.knees_angle = 0.5 * (
            self.qpos_torch[:, self._leg_qpos_idx] + self.qpos_torch[:, self._leg_left_qpos_idx]
        )

    def _roll_reward(self, action_unit: torch.Tensor, success: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Normalize all terms to [-1, 1] range
        r_vel = torch.tanh(self.vx / 3.0)  # peak velocity ~3 m/s
        
        v_rot = -self.omega
        r_rot = torch.tanh(torch.clamp(v_rot, min=0.0) / 5.0)  # peak rotation ~5 rad/s
        
        r_tuck = torch.tanh((torch.abs(self.waist_angle) + torch.abs(self.knees_angle)) / 4.0)  # typical range ~4 rad
        
        # Smooth impact penalty: penalize head contact, not binary spike
        c_impact = torch.relu(0.15 - self.head_z) * 5.0  # smooth penalty ramping
        
        action_norm = torch.clamp(action_unit, -1.0, 1.0)
        c_ctrl = torch.sum(action_norm * action_norm, dim=-1)  # typical max ~13 for 13 actions
        c_ctrl = torch.tanh(c_ctrl / 13.0)
        
        # Balanced weights: each term ~[-1, 1]
        reward = (1.0 * r_vel) + (1.0 * r_rot) + (0.5 * r_tuck) - (0.5 * c_ctrl) - (1.0 * c_impact)
        
        # Success bonus
        reward = reward + success.float() * 100.0
        
        return reward, {
            "r_vel": r_vel,
            "r_rot": r_rot,
            "r_tuck": r_tuck,
            "c_ctrl": c_ctrl,
            "c_impact": c_impact,
        }

    def _brace_reward(self, t: torch.Tensor, success: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Continuous timing rewards: penalize deviation from ideal sequence
        # Ideal: arms touch first, then knees ~0.1s later
        t_arms_min = torch.minimum(self.t_arms_l, self.t_arms_r)
        t_arms_max = torch.maximum(self.t_arms_l, self.t_arms_r)
        
        # Reward arms contacting first (before other body parts)
        t_head_arm_diff = self.t_head - t_arms_min
        t_torso_arm_diff = self.t_torso - t_arms_min
        t_waist_arm_diff = self.t_waist - t_arms_min
        
        r_arm_first = torch.tanh((t_head_arm_diff + t_torso_arm_diff + t_waist_arm_diff) / 3.0)  # normalized
        r_arm_first = torch.where(torch.isinf(t_arms_min), torch.zeros_like(r_arm_first), r_arm_first)  # no arms contact = 0
        
        # Reward arm pair synchronization (timing within 0.1s)
        arm_sync_error = torch.abs(self.t_arms_l - self.t_arms_r)
        r_arm_sync = torch.tanh(torch.where(torch.isinf(arm_sync_error), torch.full_like(arm_sync_error, 1.0), arm_sync_error))
        r_arm_sync = torch.where(torch.isinf(t_arms_min), torch.zeros_like(r_arm_sync), 1.0 - r_arm_sync)  # no contact = 0
        
        # Reward knee timing relative to arms (should contact ~0.1s after arms)
        t_arm_knee_diff = torch.abs(self.t_knees - t_arms_min)
        r_knee_timing = 1.0 - torch.tanh(t_arm_knee_diff / 0.2)  # peaks at ~0.1s difference
        r_knee_timing = torch.where(torch.isinf(self.t_knees), torch.zeros_like(r_knee_timing), r_knee_timing)  # no knee contact = 0
        
        # Smooth head impact penalty
        c_impact = torch.relu(0.15 - self.head_z) * 5.0
        
        # Balanced weights: each term ~[-1, 1]
        reward = (1.0 * r_arm_first) + (0.8 * r_arm_sync) + (1.0 * r_knee_timing) - (1.0 * c_impact)
        
        # Success bonus
        reward = reward + success.float() * 100.0
        
        return reward, {
            "r_arm_first": r_arm_first,
            "r_arm_sync": r_arm_sync,
            "r_knee_timing": r_knee_timing,
            "c_head_impact": c_impact,
        }

    def step(self, action_unit: torch.Tensor) -> Dict[str, torch.Tensor]:
        timer_ctx = wp.ScopedTimer("goal_conditioned_step") if self.enable_step_timing else nullcontext()
        with timer_ctx:
            alive = ~self.done
            action_unit = torch.nan_to_num(action_unit, nan=0.0, posinf=1.0, neginf=-1.0)
            action_unit = torch.clamp(action_unit, -1.0, 1.0)
            ctrl = self.map_action_to_ctrl(action_unit)
            self.last_action = ctrl

            self.ctrl_torch[:] = 0.0
            self.ctrl_torch[alive] = ctrl[alive]

            push_active = alive & (self.push_steps_left > 0)
            if self.xfrc_torch is not None:
                self.xfrc_torch[:] = 0.0
                self.xfrc_torch[push_active, self._torso_body_id, 0] = self.push_force[push_active]

            # Always apply a small velocity kick while push is active to guarantee visible perturbation.
            self.qvel_torch[push_active, self._rootx_qvel_idx] = (
                self.qvel_torch[push_active, self._rootx_qvel_idx]
                + self.push_force[push_active] * (self.dt * self.push_kick_scale)
            )

            if self._step_graph is not None:
                wp.capture_launch(self._step_graph)
            else:
                mjw.step(self.model, self.data)

            state_valid = torch.isfinite(self.qpos_torch).all(dim=1) & torch.isfinite(self.qvel_torch).all(dim=1)
            invalid_state = ~state_valid
            if torch.any(invalid_state):
                self.qpos_torch[invalid_state] = self._qpos_default
                self.qvel_torch[invalid_state] = self._qvel_default
                self.ctrl_torch[invalid_state] = self._ctrl_default
                if self.xfrc_torch is not None:
                    self.xfrc_torch[invalid_state] = 0.0
                self.step_count[invalid_state] = self.episode_length - 1
                self.done[invalid_state] = True

            self._refresh_state_from_sim()

            roll_goal = self.goal_id == GoalId.ROLL
            brace_goal = self.goal_id == GoalId.BRACE

            # Contacts (vectorized, no env loop)
            t = self.step_count.float() * self.dt
            arms_contact_l = self.com_z < 0.33
            arms_contact_r = self.com_z < 0.33
            knees_contact = self.com_z < 0.28
            head_contact = self.head_z < 0.15
            torso_contact = self.com_z < 0.20
            waist_contact = self.com_z < 0.23

            self.t_arms_l = torch.where(arms_contact_l & torch.isinf(self.t_arms_l), t, self.t_arms_l)
            self.t_arms_r = torch.where(arms_contact_r & torch.isinf(self.t_arms_r), t, self.t_arms_r)
            self.t_knees = torch.where(knees_contact & torch.isinf(self.t_knees), t, self.t_knees)
            self.t_head = torch.where(head_contact & torch.isinf(self.t_head), t, self.t_head)
            self.t_torso = torch.where(torso_contact & torch.isinf(self.t_torso), t, self.t_torso)
            self.t_waist = torch.where(waist_contact & torch.isinf(self.t_waist), t, self.t_waist)

            roll_success = (self.rotation >= (1.5 * math.pi)) & ((torch.relu(0.15 - self.head_z) * 100.0) < 0.1)
            brace_success = (
                (self.t_arms_l < self.t_head)
                & (self.t_arms_r < self.t_head)
                & (self.t_arms_l < self.t_torso)
                & (self.t_arms_r < self.t_torso)
                & (self.t_arms_l < self.t_waist)
                & (self.t_arms_r < self.t_waist)
                & (torch.abs(self.t_arms_l - self.t_arms_r) <= 0.1)
            )

            success = torch.where(roll_goal, roll_success, brace_success)

            roll_reward, roll_parts = self._roll_reward(action_unit, success & roll_goal)
            brace_reward, brace_parts = self._brace_reward(t, success & brace_goal)

            reward = torch.where(roll_goal, roll_reward, brace_reward)
            reward = torch.where(alive, reward, torch.zeros_like(reward))
            reward = torch.nan_to_num(reward, nan=0.0, posinf=self.reward_clip, neginf=-self.reward_clip)
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            self.success_counter = torch.where(success, self.success_counter + 1, torch.zeros_like(self.success_counter))

            stopped = (torch.abs(self.vx) < 0.02) & (torch.abs(self.omega) < 0.02) & (self.com_z < 0.20)
            timeout = self.step_count >= (self.episode_length - 1)
            self.done = self.done | stopped | timeout | success | invalid_state

            self.step_count = self.step_count + alive.long()
            self.push_steps_left = torch.clamp(self.push_steps_left - alive.long(), min=0)

            self.obs_torch[:, 0] = torch.clamp(torch.nan_to_num(self.vx, nan=0.0), -self.obs_clip, self.obs_clip)
            self.obs_torch[:, 1] = torch.clamp(torch.nan_to_num(self.omega, nan=0.0), -self.obs_clip, self.obs_clip)
            self.obs_torch[:, 2] = torch.clamp(torch.nan_to_num(self.rotation, nan=0.0), -self.obs_clip, self.obs_clip)
            self.obs_torch[:, 3] = torch.clamp(torch.nan_to_num(self.com_z, nan=0.0), -self.obs_clip, self.obs_clip)
            self.obs_torch[:, 4] = torch.clamp(torch.nan_to_num(self.head_z, nan=0.0), -self.obs_clip, self.obs_clip)
            self.obs_torch[:, 5] = torch.clamp(torch.nan_to_num(self.waist_angle, nan=0.0), -self.obs_clip, self.obs_clip)
            self.obs_torch[:, 6] = torch.clamp(torch.nan_to_num(self.knees_angle, nan=0.0), -self.obs_clip, self.obs_clip)

            self._update_goal_buffer()

            info = {
                "success": success,
                "goal_id": self.goal_id,
                "reward_roll": torch.where(roll_goal, roll_reward, torch.zeros_like(roll_reward)),
                "reward_brace": torch.where(brace_goal, brace_reward, torch.zeros_like(brace_reward)),
                "roll_r_vel": roll_parts["r_vel"],
                "roll_r_rot": roll_parts["r_rot"],
                "roll_r_tuck": roll_parts["r_tuck"],
                "roll_c_ctrl": roll_parts["c_ctrl"],
                "roll_c_impact": roll_parts["c_impact"],
                "brace_r_arm_first": brace_parts["r_arm_first"],
                "brace_r_arm_sync": brace_parts["r_arm_sync"],
                "brace_r_knee_timing": brace_parts["r_knee_timing"],
                "brace_c_head_impact": brace_parts["c_head_impact"],
            }

            return {
                "obs": wp.to_torch(self.obs_wp),
                "reward": reward,
                "done": self.done,
                "info": info,
            }
