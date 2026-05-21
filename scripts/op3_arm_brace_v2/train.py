from __future__ import annotations

import os
os.environ['MUJOCO_GL'] = 'osmesa'
import sys
from collections import deque
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from envs.op3_arm_brace_v2 import Op3ArmBraceEnv

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"


def kv(label, value, label2="", value2=""):
	s = f"{label}: \x1b[1;36m{value}\x1b[0m"
	if label2:
		s += f"  {label2}: \x1b[1;36m{value2}\x1b[0m"
	return s


class PPONetwork(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim=64):
		super().__init__()
		self.shared = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
		)
		self.policy_mean = nn.Sequential(
			nn.Linear(hidden_dim, action_dim),
			nn.Tanh()  # Constrain to [-1, 1]
		)
		self.policy_logstd = nn.Parameter(torch.zeros(action_dim))
		self.value = nn.Linear(hidden_dim, 1)

	def forward(self, x):
		shared_out = self.shared(x)
		return self.policy_mean(shared_out), self.value(shared_out)

	def get_action(self, obs, deterministic=False):
		if not isinstance(obs, torch.Tensor):
			obs = torch.FloatTensor(obs)
		if obs.dim() == 1:
			obs = obs.unsqueeze(0)

		shared_out = self.shared(obs)
		mean = self.policy_mean(shared_out)  # Already in [-1, 1] from tanh
		value = self.value(shared_out)
		
		if deterministic:
			action = mean
		else:
			std = self.policy_logstd.exp()
			# Sample in Gaussian space and apply tanh
			raw_action = mean + std * torch.randn_like(mean)
			action = torch.tanh(raw_action)
		return action.squeeze(0), value.squeeze(0)


class PPO:
	def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, entropy_coef=0.01, value_coef=0.5):
		self.network = PPONetwork(obs_dim, action_dim)
		self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.value_coef = value_coef
		self.entropy_coef = entropy_coef

	def update(self, trajectories):
		obs = torch.FloatTensor(np.array(trajectories["obs"]))
		actions = torch.FloatTensor(np.array(trajectories["actions"]))
		rewards = torch.FloatTensor(np.array(trajectories["rewards"]))
		dones = torch.FloatTensor(np.array(trajectories["dones"]))

		with torch.no_grad():
			_, values = self.network(obs)
			values = values.squeeze()

		returns = []
		advantages = []
		gae = 0.0
		next_value = 0.0

		for t in reversed(range(len(rewards))):
			if dones[t]:
				delta = rewards[t] - values[t]
				gae = delta
			else:
				delta = rewards[t] + self.gamma * next_value - values[t]
				gae = delta + self.gamma * 0.95 * gae
			next_value = values[t]
			returns.insert(0, gae + values[t])
			advantages.insert(0, gae)

		returns = torch.FloatTensor(returns)
		advantages = torch.FloatTensor(advantages)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		# Use detached forward pass to compute old_log_probs
		with torch.no_grad():
			mean, _ = self.network(obs)
			old_log_probs = self._log_prob(mean, self.network.policy_logstd, actions).detach()

		for _ in range(4):
			mean, values = self.network(obs)
			values = values.squeeze()
			new_log_probs = self._log_prob(mean, self.network.policy_logstd, actions)

			ratio = (new_log_probs - old_log_probs).exp()
			surr1 = ratio * advantages
			surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
			policy_loss = -torch.min(surr1, surr2).mean()

			value_loss = nn.MSELoss()(values, returns)
			entropy = (0.5 * (1 + np.log(2 * np.pi)) + self.network.policy_logstd).sum()
			entropy_loss = -entropy.mean()

			loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
			self.optimizer.step()

	def _log_prob(self, mean, logstd, action):
		"""Compute log probability with tanh squashing correction.
		
		Note: mean is already tanh-squashed to [-1, 1].
		We need to account for the change of variables from the 
		unsquashed Gaussian space to the tanh-squashed action space.
		"""
		# Ensure tensors have compatible shapes for batch computations
		# action: [B, A] or [A]
		action_clamped = torch.clamp(action, -0.999999, 0.999999)
		# Expand mean/logstd to match action batch if necessary
		if mean.dim() == 1:
			# mean: [A] -> [1, A]
			mean_exp = mean.unsqueeze(0).expand(action_clamped.size(0), -1)
		else:
			mean_exp = mean

		if logstd.dim() == 1:
			logstd_exp = logstd.unsqueeze(0).expand(action_clamped.size(0), -1)
		else:
			logstd_exp = logstd

		std_exp = logstd_exp.exp()

		# Inverse tanh: arctanh(a) = 0.5 * ln((1 + a) / (1 - a))
		raw_action = 0.5 * torch.log((1.0 + action_clamped) / (1.0 - action_clamped))

		# Elementwise Gaussian log-prob in raw space
		var = std_exp ** 2
		log_prob_element = -((raw_action - mean_exp) ** 2) / (2 * var) - logstd_exp - 0.5 * np.log(2 * np.pi)

		# Sum over action dims to get per-sample log-prob
		log_prob = log_prob_element.sum(dim=-1)

		# Correction for tanh transformation: subtract sum(log(1 - a^2)) per sample
		correction = torch.sum(torch.log(1.0 - action_clamped ** 2 + 1e-8), dim=-1)
		log_prob = log_prob - correction
		return log_prob


class LivePlot:
	def __init__(self, title: str):
		plt.ion()
		self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
		self.fig.suptitle(title, fontsize=14, fontweight="bold")

		self.ax1.set_title("Episode Reward")
		self.ax1.set_xlabel("Iteration")
		self.ax1.set_ylabel("Mean Reward")
		self.ax1.grid(True, alpha=0.3)
		self.reward_line, = self.ax1.plot([], [], "b-", linewidth=2, label="Mean Reward")
		self.reward_scatter = self.ax1.scatter([], [], c="blue", alpha=0.3, s=10, label="Episode Rewards")
		self.ax1.legend()

		self.ax2.set_title("Success Rate")
		self.ax2.set_xlabel("Iteration")
		self.ax2.set_ylabel("Success Rate (%)")
		self.ax2.grid(True, alpha=0.3)
		self.ax2.set_ylim(0, 100)
		self.success_line, = self.ax2.plot([], [], "g-", linewidth=2, label="Success Rate")
		self.ax2.legend()

		self.reward_history = []
		self.success_history = []
		self.iteration_history = []
		self.episode_rewards = []

		plt.tight_layout()
		plt.pause(0.1)

	def update(self, iteration, mean_reward, success_rate, episode_rewards_list):
		self.iteration_history.append(iteration)
		self.reward_history.append(mean_reward)
		self.success_history.append(success_rate * 100)
		self.episode_rewards = episode_rewards_list

		self.reward_line.set_data(self.iteration_history, self.reward_history)
		if len(self.episode_rewards) > 0:
			scatter_x = [iteration] * len(self.episode_rewards)
			self.reward_scatter.set_offsets(np.c_[scatter_x, self.episode_rewards])
		self.success_line.set_data(self.iteration_history, self.success_history)

		if len(self.iteration_history) > 0:
			self.ax1.set_xlim(0, max(50, iteration + 10))
			self.ax2.set_xlim(0, max(50, iteration + 10))
			if len(self.reward_history) > 0:
				y_min = min(min(self.reward_history), min(self.episode_rewards) if self.episode_rewards else 0)
				y_max = max(max(self.reward_history), max(self.episode_rewards) if self.episode_rewards else 1)
				margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
				self.ax1.set_ylim(y_min - margin, y_max + margin)

		plt.pause(0.01)

	def close(self):
		plt.ioff()
		plt.close(self.fig)


def main():
	parser = argparse.ArgumentParser(description="Train OP3 Arm-Only Brace PPO v2 (single env).")
	parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file to continue training from")
	parser.add_argument("--run_label", required=True, help="Label for this training run; used to name checkpoint folder")
	args = parser.parse_args()

	run_dir = CHECKPOINT_DIR / args.run_label
	run_dir.mkdir(parents=True, exist_ok=True)

	env = Op3ArmBraceEnv()
	agent = PPO(env.obs_dim, env.action_dim, entropy_coef=0.01)

	if args.checkpoint:
		agent.network.load_state_dict(torch.load(args.checkpoint))
	plot = LivePlot("OP3 Arm-Only Brace PPO v2 Training Progress")

	max_iters = 1000
	steps_per_iter = 2048
	entropy_decay = 0.995
	min_entropy = 0.001
	consecutive_success_threshold = 100

	episode_rewards = deque(maxlen=10)
	success_history = deque(maxlen=10)
	max_mean_reward = -np.inf
	max_win_rate = 0.0
	consec_success = 0
	max_consec = 0
	latest_iter_mean_reward = 0.0

	total_steps = 0
	done_iter = 0

	try:
		for iteration in range(max_iters):
			trajectories = {"obs": [], "actions": [], "rewards": [], "dones": []}
			iter_rewards = []
			iter_successes = []

			obs = env.reset()
			episode_reward = 0.0

			for step in range(steps_per_iter):
				action, _ = agent.network.get_action(obs)
				action = action.detach().numpy()

				next_obs, reward, done, info = env.step(action)

				trajectories["obs"].append(obs)
				trajectories["actions"].append(action)
				trajectories["rewards"].append(reward)
				trajectories["dones"].append(done)

				episode_reward += reward
				obs = next_obs

				if done or step == steps_per_iter - 1:
					episode_rewards.append(episode_reward)
					iter_rewards.append(episode_reward)
					success = env.get_success()
					iter_successes.append(success)
					success_history.append(success)

					if success:
						consec_success += 1
						max_consec = max(max_consec, consec_success)
					else:
						consec_success = 0

					obs = env.reset()
					episode_reward = 0.0
					done_iter += 1

			agent.update(trajectories)
			total_steps += steps_per_iter

			mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
			latest_success_rate = np.mean(success_history) if success_history else 0.0
			latest_iter_mean_reward = float(np.mean(iter_rewards)) if iter_rewards else 0.0

			agent.entropy_coef = max(agent.entropy_coef * entropy_decay, min_entropy)

			if mean_reward > max_mean_reward:
				max_mean_reward = mean_reward
				best_model_path = run_dir / "best_op3_arm_brace_v2_model.pth"
				torch.save(agent.network.state_dict(), best_model_path)

			if (iteration + 1) % 10 == 0:
				checkpoint_path = run_dir / f"op3_arm_brace_v2_model_iter_{iteration + 1}.pth"
				torch.save(agent.network.state_dict(), checkpoint_path)

			max_win_rate = max(max_win_rate, latest_success_rate)
			plot.update(iteration, mean_reward, latest_success_rate, list(iter_rewards))

			prefix = "\x1b[1;32m[OP3 ARM BRACE V2 PPO]\x1b[0m"
			iter_txt = f"{iteration + 1:3d}/{max_iters}"
			eps_txt = f"{len(episode_rewards):3d}"
			steps_txt = f"{total_steps:,}"
			done_iter_txt = f"{done_iter}"
			mean_rew_txt = f"{mean_reward:.3f}"
			max_rew_txt = f"{max_mean_reward:.3f}"

			if iteration > 0:
				print("\x1b[7A", end="")

			print(
				f"\r\x1b[K{prefix}\n"
				f"\x1b[K{kv('iter', iter_txt, 'eps', eps_txt)}\n"
				f"\x1b[K{kv('steps', steps_txt, 'done_iter', done_iter_txt)}\n"
				f"\x1b[K{kv('mean_rew', mean_rew_txt, 'max_mean_rew', max_rew_txt)}\n"
				f"\x1b[K{kv('win_rate', f'{latest_success_rate * 100.0:.2f}%', 'max_win_rate', f'{max_win_rate:.2%}')}\n"
				f"\x1b[K{kv('consec_success', f'{consec_success}', 'max_consec', f'{max_consec}')}\n"
				f"\x1b[K{kv('entropy_coef', f'{agent.entropy_coef:.6f}', '', '')}",
				end="",
				flush=True,
			)

			# Early stop: success rate >= 95% for 100+ consecutive iterations
			if latest_success_rate >= 0.95 and consec_success >= consecutive_success_threshold:
				print(f"\n\nTarget reached! Success rate: {latest_success_rate:.2%} maintained for {consec_success} iterations")
				break

		best_model_path = run_dir / "best_op3_arm_brace_v2_model.pth"
		print(f"\n\nTraining completed! Best model saved to {best_model_path}")
		print(f"Max mean reward: {max_mean_reward:.3f}, Max win rate: {max_win_rate:.2%}")
		print(f"Last iteration mean episode reward: {latest_iter_mean_reward:.3f}")

	except KeyboardInterrupt:
		print("\n\nTraining interrupted by user!")

	finally:
		plot.close()
		training_plot_path = run_dir / "training_progress.png"
		plot.fig.savefig(training_plot_path, dpi=150, bbox_inches="tight")
		print(f"Plot saved as '{training_plot_path.name}' in {run_dir}")


if __name__ == "__main__":
	main()
