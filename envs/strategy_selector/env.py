from __future__ import annotations

from enum import IntEnum
from typing import Dict

import torch
import warp as wp


class StrategyAction(IntEnum):
    DO_NOTHING = 0
    BRACE = 1
    ROLL = 2


class StrategySelectorEnv:
    def __init__(self, num_envs: int = 4096, episode_length: int = 5, device: str = "cuda"):
        wp.init()
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.device = torch.device(device)

        self.push_values = torch.tensor([20.0, 35.0, 70.0], device=self.device)
        self.target_action_for_force = torch.tensor(
            [StrategyAction.DO_NOTHING, StrategyAction.BRACE, StrategyAction.ROLL],
            device=self.device,
            dtype=torch.long,
        )

        self.step_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.done = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.force_bucket = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.push_force = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.has_acted = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.obs_torch = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.obs_wp = wp.from_torch(self.obs_torch, dtype=wp.float32)

    def reset(self, env_mask: torch.Tensor | None = None) -> torch.Tensor:
        if env_mask is None:
            env_mask = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)

        idx = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return wp.to_torch(self.obs_wp)

        sampled = torch.randint(0, 3, (idx.numel(),), device=self.device)
        self.force_bucket[idx] = sampled
        self.push_force[idx] = self.push_values[sampled]
        self.obs_torch[idx, 0] = self.push_force[idx]

        self.step_count[idx] = 0
        self.done[idx] = False
        self.success[idx] = False
        self.has_acted[idx] = False
        return wp.to_torch(self.obs_wp)

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        # agent acts only once at timestep 0
        is_t0 = self.step_count == 0
        apply_mask = is_t0 & (~self.has_acted) & (~self.done)

        target_actions = self.target_action_for_force[self.force_bucket]

        clamped_actions = torch.clamp(actions.long(), 0, 2)
        dist = torch.abs(clamped_actions - target_actions)

        reward_small = torch.full_like(dist, -0.5, dtype=torch.float32)
        reward_large = torch.full_like(dist, -1.0, dtype=torch.float32)
        reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        correct = dist == 0
        reward = torch.where(correct, torch.ones_like(reward), reward)
        reward = torch.where(dist == 1, reward_small, reward)
        reward = torch.where(dist >= 2, reward_large, reward)

        reward = torch.where(apply_mask, reward, torch.zeros_like(reward))
        self.success = torch.where(apply_mask, correct, self.success)
        self.has_acted = self.has_acted | apply_mask

        self.step_count = self.step_count + (~self.done).long()

        timed_out = self.step_count >= self.episode_length
        self.done = self.done | timed_out | self.success

        info = {
            "target_action": target_actions,
            "chosen_action": clamped_actions,
            "success": self.success,
        }
        return {
            "obs": wp.to_torch(self.obs_wp),
            "reward": reward,
            "done": self.done,
            "info": info,
        }
