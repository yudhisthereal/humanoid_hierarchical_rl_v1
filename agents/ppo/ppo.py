from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


@dataclass
class PPOConfig:
	obs_dim: int
	action_dim: int
	hidden_dim: int = 256
	lr: float = 3e-4
	gamma: float = 0.99
	gae_lambda: float = 0.95
	clip_eps: float = 0.2
	entropy_coef: float = 0.01
	value_coef: float = 0.5
	max_grad_norm: float = 0.5
	minibatch_size: int = 128
	update_epochs: int = 4
	is_discrete: bool = False
	device: str = "cuda"


class ActorCritic(nn.Module):
	def __init__(self, cfg: PPOConfig):
		super().__init__()
		self.cfg = cfg

		self.backbone = nn.Sequential(
			nn.Linear(cfg.obs_dim, cfg.hidden_dim),
			nn.Tanh(),
			nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
			nn.Tanh(),
		)

		if cfg.is_discrete:
			self.policy_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)
			self.log_std = None
		else:
			self.policy_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)
			self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

		self.value_head = nn.Linear(cfg.hidden_dim, 1)

	def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.backbone(obs)
		logits_or_mean = self.policy_head(h)
		value = self.value_head(h).squeeze(-1)
		return logits_or_mean, value

	def _dist(self, obs: torch.Tensor):
		logits_or_mean, _ = self.forward(obs)
		if self.cfg.is_discrete:
			return Categorical(logits=logits_or_mean)
		std = self.log_std.exp().expand_as(logits_or_mean)
		return Normal(logits_or_mean, std)

	def sample_action(
		self,
		obs: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		logits_or_mean, value = self.forward(obs)
		if self.cfg.is_discrete:
			dist = Categorical(logits=logits_or_mean)
			action = dist.sample()
			log_prob = dist.log_prob(action)
		else:
			std = self.log_std.exp().expand_as(logits_or_mean)
			dist = Normal(logits_or_mean, std)
			raw_action = dist.rsample()
			action = torch.tanh(raw_action)
			log_prob = dist.log_prob(raw_action).sum(-1)
		return action, log_prob, value

	def evaluate_actions(
		self,
		obs: torch.Tensor,
		actions: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		logits_or_mean, value = self.forward(obs)
		if self.cfg.is_discrete:
			dist = Categorical(logits=logits_or_mean)
			log_prob = dist.log_prob(actions.long())
			entropy = dist.entropy()
		else:
			std = self.log_std.exp().expand_as(logits_or_mean)
			dist = Normal(logits_or_mean, std)
			clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
			raw_actions = 0.5 * torch.log((1 + clipped_actions) / (1 - clipped_actions))
			log_prob = dist.log_prob(raw_actions).sum(-1)
			entropy = dist.entropy().sum(-1)
		return log_prob, entropy, value


class PPOAgent:
	def __init__(self, cfg: PPOConfig):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.model = ActorCritic(cfg).to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

	@torch.no_grad()
	def act(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
		action, log_prob, value = self.model.sample_action(obs)
		return {
			"action": action,
			"log_prob": log_prob,
			"value": value,
		}

	def compute_gae(
		self,
		rewards: torch.Tensor,
		dones: torch.Tensor,
		values: torch.Tensor,
		last_value: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		# rewards, dones, values: [T, N]
		t_steps, n_env = rewards.shape
		advantages = torch.zeros((t_steps, n_env), device=self.device, dtype=torch.float32)
		gae = torch.zeros((n_env,), device=self.device, dtype=torch.float32)
		next_value = last_value
		for t in reversed(range(t_steps)):
			not_done = 1.0 - dones[t]
			delta = rewards[t] + self.cfg.gamma * next_value * not_done - values[t]
			gae = delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * gae
			advantages[t] = gae
			next_value = values[t]
		returns = advantages + values
		return advantages, returns

	def update(
		self,
		obs: torch.Tensor,
		actions: torch.Tensor,
		old_log_probs: torch.Tensor,
		returns: torch.Tensor,
		advantages: torch.Tensor,
	) -> Dict[str, float]:
		# Flatten [T, N, ...] -> [T*N, ...]
		obs = obs.reshape(-1, obs.shape[-1])
		if self.cfg.is_discrete:
			actions = actions.reshape(-1)
		else:
			actions = actions.reshape(-1, actions.shape[-1])
		old_log_probs = old_log_probs.reshape(-1)
		returns = returns.reshape(-1)
		advantages = advantages.reshape(-1)
		advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

		batch_size = obs.shape[0]
		mb = self.cfg.minibatch_size

		total_policy_loss = torch.zeros((), device=self.device)
		total_value_loss = torch.zeros((), device=self.device)
		total_entropy = torch.zeros((), device=self.device)
		n_updates = 0

		for _ in range(self.cfg.update_epochs):
			perm = torch.randperm(batch_size, device=self.device)
			for start in range(0, batch_size, mb):
				idx = perm[start : start + mb]
				mb_obs = obs[idx]
				mb_actions = actions[idx]
				mb_old_log_probs = old_log_probs[idx]
				mb_returns = returns[idx]
				mb_advantages = advantages[idx]

				new_log_probs, entropy, values = self.model.evaluate_actions(mb_obs, mb_actions)

				ratio = torch.exp(new_log_probs - mb_old_log_probs)
				surr1 = ratio * mb_advantages
				surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_advantages
				policy_loss = -torch.min(surr1, surr2).mean()
				value_loss = F.mse_loss(values, mb_returns)
				entropy_loss = entropy.mean()

				loss = (
					policy_loss
					+ self.cfg.value_coef * value_loss
					- self.cfg.entropy_coef * entropy_loss
				)

				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
				self.optimizer.step()

				total_policy_loss = total_policy_loss + policy_loss.detach()
				total_value_loss = total_value_loss + value_loss.detach()
				total_entropy = total_entropy + entropy_loss.detach()
				n_updates += 1

		return {
			"policy_loss": float((total_policy_loss / n_updates).item()),
			"value_loss": float((total_value_loss / n_updates).item()),
			"entropy": float((total_entropy / n_updates).item()),
		}

	def save(self, path: str, iteration: int):
		torch.save(
			{
				"iteration": iteration,
				"model": self.model.state_dict(),
				"optimizer": self.optimizer.state_dict(),
				"config": self.cfg.__dict__,
			},
			path,
		)

	def load(self, path: str):
		checkpoint = torch.load(path, map_location=self.device)
		self.model.load_state_dict(checkpoint["model"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		return checkpoint.get("iteration", 0)
