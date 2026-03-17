from __future__ import annotations

from typing import Dict

import torch

from envs.goal_conditioned import GoalConditionedExecutorEnv
from envs.strategy_selector import StrategySelectorEnv


class RobotHierarchicalEnv:
	def __init__(
		self,
		model_xml: str,
		num_envs: int = 4096,
		device: str = "cuda",
	):
		self.num_envs = num_envs
		self.device = torch.device(device)

		self.strategy_selector = StrategySelectorEnv(num_envs=num_envs, episode_length=5, device=device)
		self.goal_executor = GoalConditionedExecutorEnv(
			model_xml=model_xml,
			num_envs=num_envs,
			dt=0.02,
			episode_length=100,
			device=device,
		)

	@property
	def obs_dim(self) -> int:
		return 2

	@property
	def action_dim(self) -> int:
		return self.goal_executor.action_dim

	def reset(self, env_mask: torch.Tensor | None = None) -> torch.Tensor:
		self.strategy_selector.reset(env_mask)
		return self.goal_executor.reset(env_mask)

	def step(self, action: torch.Tensor) -> Dict[str, torch.Tensor]:
		out = self.goal_executor.step(action)
		done = out["done"]
		if torch.any(done):
			self.reset(done)
		return out
