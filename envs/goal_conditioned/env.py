from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from enum import IntEnum
from pathlib import Path
from typing import Dict, Tuple

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
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.episode_length = episode_length
        self.device = torch.device(device)

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

        self.obs_torch = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.obs_wp = wp.from_torch(self.obs_torch, dtype=wp.float32)

        self.last_action = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)

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
            # XML uses degrees in this model, policy outputs radians.
            lo = math.radians(lo)
            hi = math.radians(hi)
            mins.append(lo)
            maxs.append(hi)
        return torch.tensor(mins, dtype=torch.float32), torch.tensor(maxs, dtype=torch.float32)

    def map_action_to_ctrl(self, action_unit: torch.Tensor) -> torch.Tensor:
        # action_unit in [-1, 1] -> actuator ctrlrange
        a = torch.clamp(action_unit, -1.0, 1.0)
        return self.ctrl_min + 0.5 * (a + 1.0) * (self.ctrl_max - self.ctrl_min)

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

        self.step_count[idx] = 0
        self.done[idx] = False

        # robot starts slightly above ground
        self.com_z[idx] = 1.12
        self.head_z[idx] = 1.42
        self.vx[idx] = 0.0
        self.omega[idx] = 0.0
        self.rotation[idx] = 0.0
        self.waist_angle[idx] = 0.0
        self.knees_angle[idx] = 0.0

        inf = torch.full((idx.numel(),), float("inf"), device=self.device)
        self.t_arms_l[idx] = inf
        self.t_arms_r[idx] = inf
        self.t_knees[idx] = inf
        self.t_head[idx] = inf
        self.t_torso[idx] = inf
        self.t_waist[idx] = inf

        # Disturbance: torso push for 5 steps, random per env in {20, 35, 70}
        push_choices = torch.tensor([20.0, 35.0, 70.0], device=self.device)
        sampled = torch.randint(0, 3, (idx.numel(),), device=self.device)
        self.push_force[idx] = push_choices[sampled]
        self.push_steps_left[idx] = 5

        self.obs_torch[idx, 0] = self.vx[idx]
        self.obs_torch[idx, 1] = self.omega[idx]
        return wp.to_torch(self.obs_wp)

    def _roll_reward(self, ctrl: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vx = self.vx
        r_vel = vx

        v_rot = -self.omega
        r_rot = torch.clamp(v_rot, min=0.0)

        r_tuck = torch.abs(self.waist_angle) + torch.abs(self.knees_angle)

        c_impact = torch.where(self.head_z < 0.15, torch.full_like(self.head_z, 10.0), torch.zeros_like(self.head_z))
        c_ctrl = torch.sum(ctrl * ctrl, dim=-1)

        reward = (1.5 * r_vel) + (2.0 * r_rot) + (0.5 * r_tuck) - (0.1 * c_ctrl) - c_impact
        return reward, {
            "r_vel": r_vel,
            "r_rot": r_rot,
            "r_tuck": r_tuck,
            "c_ctrl": c_ctrl,
            "c_impact": c_impact,
        }

    def _brace_reward(self, t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        arm_first = (self.t_arms_l < self.t_head) & (self.t_arms_l < self.t_torso) & (self.t_arms_l < self.t_waist)
        arm_pair = torch.abs(self.t_arms_l - self.t_arms_r) <= 0.1

        knee_timing = torch.abs(self.t_knees - torch.minimum(self.t_arms_l, self.t_arms_r)) <= 0.1

        high = torch.where(arm_first & arm_pair, torch.full_like(t, 3.0), torch.zeros_like(t))
        med = torch.where(knee_timing, torch.full_like(t, 1.5), torch.zeros_like(t))

        normal_force_head = torch.relu(0.15 - self.head_z) * 100.0
        penalty = torch.exp(normal_force_head * 0.05) - 1.0

        reward = high + med - penalty
        return reward, {
            "r_arm_first": high,
            "r_knee_timing": med,
            "c_head_impact": penalty,
            "head_normal_force": normal_force_head,
        }

    def step(self, action_unit: torch.Tensor) -> Dict[str, torch.Tensor]:
        with wp.ScopedTimer("goal_conditioned_step"):
            alive = ~self.done
            ctrl = self.map_action_to_ctrl(action_unit)
            self.last_action = ctrl

            # lightweight vectorized dynamics proxy with push disturbance
            push_active = (self.push_steps_left > 0).float()
            push_acc = self.push_force * push_active * 0.003
            roll_goal = self.goal_id == GoalId.ROLL
            brace_goal = self.goal_id == GoalId.BRACE

            self.vx = self.vx + push_acc - 0.02 * self.vx - 0.01 * torch.sum(ctrl, dim=-1)
            self.omega = self.omega - 0.03 * torch.sum(ctrl[:, :2], dim=-1) - 0.01 * self.omega
            self.rotation = self.rotation + self.omega * self.dt

            self.waist_angle = ctrl[:, 0]
            self.knees_angle = 0.5 * (ctrl[:, 2] + ctrl[:, 5])

            tuck = torch.abs(self.waist_angle) + torch.abs(self.knees_angle)
            self.com_z = self.com_z - 0.004 * torch.abs(self.vx) - 0.003 * torch.abs(self.omega) + 0.0008 * tuck
            self.com_z = torch.clamp(self.com_z, min=0.03)
            self.head_z = self.com_z + 0.3 - 0.15 * torch.abs(self.waist_angle)

            # Contacts (vectorized, no env loop)
            t = self.step_count.float() * self.dt
            arms_contact_l = (self.com_z < 0.35) & (ctrl[:, 7] > 0.4)
            arms_contact_r = (self.com_z < 0.35) & (ctrl[:, 9] > 0.4)
            knees_contact = (self.com_z < 0.30) & (torch.abs(self.knees_angle) > 0.35)
            head_contact = self.head_z < 0.15
            torso_contact = self.com_z < 0.20
            waist_contact = (self.com_z < 0.23) & (torch.abs(self.waist_angle) > 0.4)

            self.t_arms_l = torch.where(arms_contact_l & torch.isinf(self.t_arms_l), t, self.t_arms_l)
            self.t_arms_r = torch.where(arms_contact_r & torch.isinf(self.t_arms_r), t, self.t_arms_r)
            self.t_knees = torch.where(knees_contact & torch.isinf(self.t_knees), t, self.t_knees)
            self.t_head = torch.where(head_contact & torch.isinf(self.t_head), t, self.t_head)
            self.t_torso = torch.where(torso_contact & torch.isinf(self.t_torso), t, self.t_torso)
            self.t_waist = torch.where(waist_contact & torch.isinf(self.t_waist), t, self.t_waist)

            roll_reward, roll_parts = self._roll_reward(ctrl)
            brace_reward, brace_parts = self._brace_reward(t)

            reward = torch.where(roll_goal, roll_reward, brace_reward)
            reward = torch.where(alive, reward, torch.zeros_like(reward))

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
            self.success_counter = torch.where(success, self.success_counter + 1, torch.zeros_like(self.success_counter))

            stopped = (torch.abs(self.vx) < 0.02) & (torch.abs(self.omega) < 0.02) & (self.com_z < 0.20)
            timeout = self.step_count >= (self.episode_length - 1)
            self.done = self.done | stopped | timeout | success

            self.step_count = self.step_count + alive.long()
            self.push_steps_left = torch.clamp(self.push_steps_left - alive.long(), min=0)

            self.obs_torch[:, 0] = self.vx
            self.obs_torch[:, 1] = self.omega

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
                "brace_r_knee_timing": brace_parts["r_knee_timing"],
                "brace_c_head_impact": brace_parts["c_head_impact"],
            }

            return {
                "obs": wp.to_torch(self.obs_wp),
                "reward": reward,
                "done": self.done,
                "info": info,
            }
