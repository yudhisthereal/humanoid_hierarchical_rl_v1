from __future__ import annotations

from contextlib import nullcontext
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp


class SanityCheckEnv:
    """Simple pose-tracking environment for motor-angle sanity checks.

    Reward definition:
        r = -err - jit

    where:
      - err: mean absolute joint-angle error to target end-pose
      - jit: mean penalty for sudden velocity reversal magnitude
    """

    def __init__(
        self,
        model_xml: str,
        num_envs: int = 4096,
        dt: float = 0.02,
        episode_length: int = 300,
        device: str = "cuda",
        enable_step_timing: bool = False,
        nconmax: int = 256,
        njmax: int = 512,
    ):
        wp.init()
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.episode_length = int(episode_length)
        self.device = torch.device(device)
        self.enable_step_timing = bool(enable_step_timing)
        self.obs_clip = 50.0
        self.reward_clip = 100.0
        self.success_err_thresh = 0.08
        self.reset_noise = 0.05
        self.jit_scale = 0.05
        self.nconmax = int(nconmax)
        self.njmax = int(njmax)

        ctrl_min, ctrl_max = self._parse_ctrl_limits(model_xml)
        self.ctrl_min = ctrl_min.to(self.device)
        self.ctrl_max = ctrl_max.to(self.device)
        self.action_dim = int(self.ctrl_min.numel())

        # End-pose target in radians:
        # waist=1, leg=-1.21, shin=2, arm=-1.29, forearm=5.19
        self.target_pose = torch.tensor([1.0, -1.21, 2.0, -1.29, 5.19], device=self.device, dtype=torch.float32)
        if self.target_pose.numel() != self.action_dim:
            raise ValueError(
                f"Expected 5 actuators for sanity check, got {self.action_dim}. "
                "Use humanoid_2d_half.xml actuator layout."
            )

        self.step_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.done = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.prev_joint_vel = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)

        self.obs_torch = torch.zeros((self.num_envs, self.action_dim * 2), device=self.device, dtype=torch.float32)
        self.obs_wp = wp.from_torch(self.obs_torch, dtype=wp.float32)

        self.mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.mj_data = mujoco.MjData(self.mj_model)
        if int(self.mj_model.nu) != self.action_dim:
            raise ValueError(
                f"Actuator mismatch: parsed action_dim={self.action_dim}, model.nu={int(self.mj_model.nu)}"
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
            self.data = mjw.put_data(self.mj_model, self.mj_data, nworld=self.num_envs)

        self.qpos_torch = wp.to_torch(self.data.qpos)
        self.qvel_torch = wp.to_torch(self.data.qvel)
        self.ctrl_torch = wp.to_torch(self.data.ctrl)
        self.xfrc_torch = wp.to_torch(self.data.xfrc_applied) if hasattr(self.data, "xfrc_applied") else None

        act_joint_ids_np = self.mj_model.actuator_trnid[:, 0].astype("int64")
        if (act_joint_ids_np < 0).any():
            raise ValueError("Sanity check env expects actuators to target valid joints.")

        self.actuated_joint_ids = torch.as_tensor(act_joint_ids_np, device=self.device, dtype=torch.long)

        qpos_idx = [int(self.mj_model.jnt_qposadr[jid]) for jid in act_joint_ids_np]
        qvel_idx = [int(self.mj_model.jnt_dofadr[jid]) for jid in act_joint_ids_np]
        self._act_qpos_idx = torch.as_tensor(qpos_idx, device=self.device, dtype=torch.long)
        self._act_qvel_idx = torch.as_tensor(qvel_idx, device=self.device, dtype=torch.long)

        self._qpos_default = self.qpos_torch[0].clone()
        self._qvel_default = self.qvel_torch[0].clone()
        self._ctrl_default = self.ctrl_torch[0].clone()
        self._target_qpos = self._qpos_default[self._act_qpos_idx].clone()

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
            mins.append(float(rng[0]))
            maxs.append(float(rng[1]))
        return torch.tensor(mins, dtype=torch.float32), torch.tensor(maxs, dtype=torch.float32)

    def _validate_actuator_joint_range_match(self, atol: float = 1e-6) -> None:
        for act_id in range(int(self.mj_model.nu)):
            j_id = int(self.mj_model.actuator_trnid[act_id, 0])
            if j_id < 0:
                continue

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

    def map_action_to_ctrl(self, action_unit: torch.Tensor) -> torch.Tensor:
        a = torch.nan_to_num(action_unit, nan=0.0, posinf=1.0, neginf=-1.0)
        a = torch.clamp(a, -1.0, 1.0)
        return self.ctrl_min + 0.5 * (a + 1.0) * (self.ctrl_max - self.ctrl_min)

    def _refresh_obs_from_sim(self) -> None:
        joint_pos = self.qpos_torch[:, self._act_qpos_idx]
        joint_vel = self.qvel_torch[:, self._act_qvel_idx]

        self.obs_torch[:, : self.action_dim] = torch.clamp(
            torch.nan_to_num(joint_pos, nan=0.0),
            -self.obs_clip,
            self.obs_clip,
        )
        self.obs_torch[:, self.action_dim :] = torch.clamp(
            torch.nan_to_num(joint_vel, nan=0.0),
            -self.obs_clip,
            self.obs_clip,
        )

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

        noise = (torch.rand((idx.numel(), self.action_dim), device=self.device) * 2.0 - 1.0) * self.reset_noise
        self.qpos_torch[idx[:, None], self._act_qpos_idx] = self._target_qpos.unsqueeze(0) + noise
        self.qvel_torch[idx[:, None], self._act_qvel_idx] = 0.0

        self.step_count[idx] = 0
        self.done[idx] = False
        self.success[idx] = False

        current_vel = self.qvel_torch[idx][:, self._act_qvel_idx]
        self.prev_joint_vel[idx] = current_vel

        self._refresh_obs_from_sim()
        return wp.to_torch(self.obs_wp)

    def step(self, action_unit: torch.Tensor) -> Dict[str, torch.Tensor]:
        timer_ctx = wp.ScopedTimer("sanity_check_step") if self.enable_step_timing else nullcontext()
        with timer_ctx:
            alive = ~self.done
            action_unit = torch.nan_to_num(action_unit, nan=0.0, posinf=1.0, neginf=-1.0)
            action_unit = torch.clamp(action_unit, -1.0, 1.0)
            ctrl = self.map_action_to_ctrl(action_unit)

            self.ctrl_torch[:] = 0.0
            self.ctrl_torch[alive] = ctrl[alive]

            if self.xfrc_torch is not None:
                self.xfrc_torch[:] = 0.0

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

            joint_pos = self.qpos_torch[:, self._act_qpos_idx]
            joint_vel = self.qvel_torch[:, self._act_qvel_idx]

            err_per_joint = torch.abs(joint_pos - self.target_pose.unsqueeze(0))
            err = torch.mean(err_per_joint, dim=1)

            vel_delta = torch.abs(joint_vel - self.prev_joint_vel)
            reversed_dir = (joint_vel * self.prev_joint_vel) < 0.0
            jit_per_joint = torch.where(reversed_dir, vel_delta, torch.zeros_like(vel_delta))
            jit = torch.mean(jit_per_joint, dim=1)

            reward = 1.0 - err - (self.jit_scale * jit)
            reward = torch.where(alive, reward, torch.zeros_like(reward))
            reward = torch.nan_to_num(reward, nan=0.0, posinf=self.reward_clip, neginf=-self.reward_clip)
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            success = err <= self.success_err_thresh
            timeout = self.step_count >= (self.episode_length - 1)
            self.done = self.done | timeout | success | invalid_state
            self.success = success

            self.step_count = self.step_count + alive.long()
            self.prev_joint_vel = joint_vel.clone()

            self._refresh_obs_from_sim()

            info = {
                "success": success,
                "err": err,
                "jit": jit,
                "err_per_joint": err_per_joint,
            }

            return {
                "obs": wp.to_torch(self.obs_wp),
                "reward": reward,
                "done": self.done,
                "info": info,
            }
