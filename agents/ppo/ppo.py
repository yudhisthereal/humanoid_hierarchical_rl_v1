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
	entropy_coef_initial: float = 0.02
	entropy_coef_final: float = 0.0005
	anneal_steps: int = 1_000_000
	lr_final_factor: float = 0.1
	kl_threshold: float = 0.02
	min_lr: float = 1e-6
	min_entropy: float = 1e-5


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
		logits_or_mean = torch.nan_to_num(logits_or_mean, nan=0.0, posinf=1.0, neginf=-1.0)
		value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
		if self.cfg.is_discrete:
			dist = Categorical(logits=logits_or_mean)
			action = dist.sample()
			log_prob = dist.log_prob(action)
		else:
			std = torch.exp(torch.clamp(self.log_std, min=-5.0, max=2.0)).expand_as(logits_or_mean)
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
		logits_or_mean = torch.nan_to_num(logits_or_mean, nan=0.0, posinf=1.0, neginf=-1.0)
		value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
		if self.cfg.is_discrete:
			dist = Categorical(logits=logits_or_mean)
			log_prob = dist.log_prob(actions.long())
			entropy = dist.entropy()
		else:
			std = torch.exp(torch.clamp(self.log_std, min=-5.0, max=2.0)).expand_as(logits_or_mean)
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
		self.total_env_steps = 0
		self.lr_initial = cfg.lr
		self.lr_final = cfg.lr * cfg.lr_final_factor
		self.current_lr = cfg.lr
		self.current_entropy_coef = cfg.entropy_coef_initial
		self.current_approx_kl = 0.0
		self.lr_scale = 1.0
		self.entropy_scale = 1.0

		self.obs_mean = torch.zeros((cfg.obs_dim,), device=self.device, dtype=torch.float32)
		self.obs_var = torch.ones((cfg.obs_dim,), device=self.device, dtype=torch.float32)
		self.obs_count = torch.tensor(1e-4, device=self.device, dtype=torch.float32)

	def _update_schedules(self, total_env_steps: int, success_rate: float = 0.0) -> None:
		self.total_env_steps = int(total_env_steps)
		progress = min(1.0, float(self.total_env_steps) / float(self.cfg.anneal_steps))

		# LR: anneal based on total steps (unchanged)
		base_lr = self.lr_initial * (1.0 - progress) + self.lr_final * progress

		# Entropy: map success_rate [25%, 100%] to entropy_annealed [0, initial - final]
		# Then entropy_coeff = initial - entropy_annealed
		success_clamped = torch.clamp(torch.tensor(success_rate), min=0.25, max=1.0)
		success_progress = (success_clamped - 0.25) / (1.0 - 0.25)  # [0.25, 1.0] -> [0, 1]
		entropy_range = self.cfg.entropy_coef_initial - self.cfg.entropy_coef_final
		entropy_annealed = success_progress * entropy_range
		base_entropy = self.cfg.entropy_coef_initial - entropy_annealed.item()

		self.current_entropy_coef = max(self.cfg.min_entropy, base_entropy * self.entropy_scale)
		self.current_lr = max(self.cfg.min_lr, base_lr * self.lr_scale)
		self.cfg.entropy_coef = self.current_entropy_coef

		for param_group in self.optimizer.param_groups:
			param_group["lr"] = self.current_lr

	@torch.no_grad()
	def update_obs_rms(self, obs: torch.Tensor) -> None:
		assert obs.is_cuda
		flat_obs = obs.reshape(-1, obs.shape[-1]).float()
		batch_mean = flat_obs.mean(dim=0)
		batch_var = flat_obs.var(dim=0, unbiased=False)
		batch_count = torch.tensor(float(flat_obs.shape[0]), device=self.device)

		delta = batch_mean - self.obs_mean
		total_count = self.obs_count + batch_count
		new_mean = self.obs_mean + delta * (batch_count / total_count)

		m_a = self.obs_var * self.obs_count
		m_b = batch_var * batch_count
		m2 = m_a + m_b + delta * delta * (self.obs_count * batch_count / total_count)
		new_var = m2 / total_count

		self.obs_mean = new_mean
		self.obs_var = torch.clamp(new_var, min=1e-8)
		self.obs_count = total_count

	def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
		assert obs.is_cuda
		return (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)

	def apply_rollback_damping(self, success_rate: float = 0.0) -> None:
		# Only dampen learning rate if success rate >= 90% (near convergence)
		# Otherwise, keep learning rate high to allow exploration and learning
		if success_rate >= 0.90:
			self.lr_scale *= 0.5
			self.current_lr = max(self.cfg.min_lr, self.current_lr * 0.5)
			for param_group in self.optimizer.param_groups:
				param_group["lr"] = self.current_lr

	@torch.no_grad()
	def act(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
		assert obs.is_cuda
		obs = torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
		self.update_obs_rms(obs)
		obs_norm = torch.nan_to_num(self.normalize_obs(obs), nan=0.0, posinf=10.0, neginf=-10.0)
		action, log_prob, value = self.model.sample_action(obs_norm)
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
		total_env_steps: int,
	) -> Dict[str, float]:
		assert obs.is_cuda
		assert advantages.is_cuda
		self._update_schedules(total_env_steps)

		# Flatten [T, N, ...] -> [T*N, ...]
		obs = torch.nan_to_num(obs.reshape(-1, obs.shape[-1]), nan=0.0, posinf=1e3, neginf=-1e3)
		self.update_obs_rms(obs)
		obs = torch.nan_to_num(self.normalize_obs(obs), nan=0.0, posinf=10.0, neginf=-10.0)
		if self.cfg.is_discrete:
			actions = actions.reshape(-1)
		else:
			actions = torch.nan_to_num(actions.reshape(-1, actions.shape[-1]), nan=0.0, posinf=1.0, neginf=-1.0)
		old_log_probs = torch.nan_to_num(old_log_probs.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
		returns = torch.nan_to_num(returns.reshape(-1), nan=0.0, posinf=1e3, neginf=-1e3)
		advantages = torch.nan_to_num(advantages.reshape(-1), nan=0.0, posinf=1e3, neginf=-1e3)
		advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

		batch_size = obs.shape[0]
		mb = self.cfg.minibatch_size

		total_policy_loss = torch.zeros((), device=self.device)
		total_value_loss = torch.zeros((), device=self.device)
		total_entropy = torch.zeros((), device=self.device)
		total_approx_kl = torch.zeros((), device=self.device)
		n_updates = 0
		stop_update = False
		approx_kl = torch.zeros((), device=self.device)

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
				approx_kl = (mb_old_log_probs - new_log_probs).mean()
				if approx_kl > self.cfg.kl_threshold:
					stop_update = True
					break

				ratio = torch.exp(new_log_probs - mb_old_log_probs)
				surr1 = ratio * mb_advantages
				surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_advantages
				policy_loss = -torch.min(surr1, surr2).mean()
				value_loss = F.mse_loss(values, mb_returns)
				entropy_loss = entropy.mean()

				loss = (
					policy_loss
					+ self.cfg.value_coef * value_loss
					- self.current_entropy_coef * entropy_loss
				)
				if not torch.isfinite(loss):
					continue

				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()
				has_bad_grad = False
				for param in self.model.parameters():
					if param.grad is not None and (not torch.isfinite(param.grad).all()):
						has_bad_grad = True
						break
				if has_bad_grad:
					self.optimizer.zero_grad(set_to_none=True)
					continue
				nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
				self.optimizer.step()
				with torch.no_grad():
					for param in self.model.parameters():
						if not torch.isfinite(param).all():
							param.copy_(torch.nan_to_num(param, nan=0.0, posinf=1.0, neginf=-1.0))

				total_policy_loss = total_policy_loss + policy_loss.detach()
				total_value_loss = total_value_loss + value_loss.detach()
				total_entropy = total_entropy + entropy_loss.detach()
				total_approx_kl = total_approx_kl + approx_kl.detach()
				n_updates += 1

			if stop_update:
				break

		denom = max(1, n_updates)
		self.current_approx_kl = float((total_approx_kl / denom).item()) if n_updates > 0 else float(approx_kl.item())

		return {
			"policy_loss": float((total_policy_loss / denom).item()),
			"value_loss": float((total_value_loss / denom).item()),
			"entropy": float((total_entropy / denom).item()),
			"approx_kl": self.current_approx_kl,
			"entropy_coef": float(self.current_entropy_coef),
			"learning_rate": float(self.current_lr),
		}

	def save(self, path: str, iteration: int):
		torch.save(
			{
				"iteration": iteration,
				"model": self.model.state_dict(),
				"optimizer": self.optimizer.state_dict(),
				"config": self.cfg.__dict__,
				"obs_mean": self.obs_mean,
				"obs_var": self.obs_var,
				"obs_count": self.obs_count,
				"total_env_steps": self.total_env_steps,
				"lr_scale": self.lr_scale,
				"entropy_scale": self.entropy_scale,
			},
			path,
		)

	def load(self, path: str):
		checkpoint = torch.load(path, map_location=self.device)
		self.model.load_state_dict(checkpoint["model"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		if "obs_mean" in checkpoint:
			self.obs_mean = checkpoint["obs_mean"].to(self.device)
			self.obs_var = checkpoint["obs_var"].to(self.device)
			self.obs_count = checkpoint["obs_count"].to(self.device)
		self.total_env_steps = int(checkpoint.get("total_env_steps", 0))
		self.lr_scale = float(checkpoint.get("lr_scale", 1.0))
		self.entropy_scale = float(checkpoint.get("entropy_scale", 1.0))
		return checkpoint.get("iteration", 0)
