from __future__ import annotations

import copy
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Literal

import torch
import warp as wp
from torch.utils.tensorboard import SummaryWriter

# wp.config.quiet = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo.ppo import PPOAgent, PPOConfig
from envs.goal_conditioned import GoalConditionedExecutorEnv
from envs.goal_conditioned import GoalId
from envs.strategy_selector import StrategySelectorEnv


_progress_started = False
_progress_lines = 8


def _format_compact(value: float, decimals: int = 2) -> str:
	abs_v = abs(float(value))
	units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
	for scale, suffix in units:
		if abs_v >= scale:
			return f"{value / scale:.{decimals}f}{suffix}"
	return f"{value:.{decimals}f}"


def _all_env_success(success_buf: list[torch.Tensor]) -> bool:
	success_matrix = torch.stack(success_buf, dim=0)
	return bool(torch.all(torch.any(success_matrix, dim=0)).item())


def _print_progress(
	iteration: int,
	episodes: int,
	timesteps: int,
	completed_eps_iter: int,
	latest_mean_reward: float,
	max_mean_reward: float,
	max_mean_reward_iteration: int,
	latest_success_rate: float,
	max_success_rate: float,
	max_success_rate_iteration: int,
	consecutive_successes: int,
	max_consecutive_successes: int,
	entropy_coef: float,
) -> None:
	global _progress_started
	prefix = ""
	if _progress_started:
		prefix = f"\r\x1b[{_progress_lines - 1}A"
	_progress_started = True
	col1 = 32
	col2 = 30
	line_w = col1 + col2 + 5
	sep = "─" * line_w
	max_rew_compact = f"{max_mean_reward:.4f}@{max_mean_reward_iteration}"
	max_win_compact = f"{max_success_rate * 100.0:.2f}%@{max_success_rate_iteration}"
	iter_txt = _format_compact(float(iteration), decimals=0)
	eps_txt = _format_compact(float(episodes), decimals=0)
	steps_txt = _format_compact(float(timesteps), decimals=0)
	done_iter_txt = _format_compact(float(completed_eps_iter), decimals=0)
	mean_rew_txt = _format_compact(float(latest_mean_reward), decimals=4 if abs(latest_mean_reward) < 1e3 else 2)
	if abs(max_mean_reward) >= 1e3:
		max_rew_compact = f"{_format_compact(max_mean_reward, decimals=2)}@{max_mean_reward_iteration}"

	def kv(left_k: str, left_v: str, right_k: str, right_v: str) -> str:
		left = f"{left_k:<14}: {left_v:>15}"
		right = f"{right_k:<14}: {right_v:>13}"
		return f"│ {left:<{col1}} │ {right:<{col2}} │"

	print(
		f"{prefix}\x1b[2K┌{sep}┐\n"
		f"\x1b[2K{kv('iter', iter_txt, 'eps', eps_txt)}\n"
		f"\x1b[2K{kv('steps', steps_txt, 'done_iter', done_iter_txt)}\n"
		f"\x1b[2K{kv('mean_rew', mean_rew_txt, 'max_mean_rew', max_rew_compact)}\n"
		f"\x1b[2K{kv('win_rate', f'{latest_success_rate * 100.0:.2f}%', 'max_win_rate', max_win_compact)}\n"
		f"\x1b[2K{kv('consec_success', f'{consecutive_successes}', 'max_consec', f'{max_consecutive_successes}')}\n"
		f"\x1b[2K{kv('entropy_coef', f'{entropy_coef:.6f}', '', '')}\n"
		f"\x1b[2K└{sep}┘",
		end="",
		flush=True,
	)


def _add_scalar_with_all_axes(
	writer: SummaryWriter,
	name: str,
	value: torch.Tensor | float,
	iteration: int,
	episodes: int,
	timesteps: int,
) -> None:
	writer.add_scalar(f"{name}/by_iteration", value, iteration)


def train_strategy_selector(best_state_tracking: str = "reward") -> None:
	global _progress_started
	CONSECUTIVE_SUCCESS_THRES = 100
	_progress_started = False
	if best_state_tracking not in ["reward", "success_rate"]:
		raise ValueError(f"best_state_tracking must be 'reward' or 'success_rate', got '{best_state_tracking}'")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required. CPU fallback is disabled.")

	device = "cuda"
	print(f"Training strategy selector with best_state_tracking={best_state_tracking}")
	project_root = PROJECT_ROOT
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	num_envs = 12_000
	horizon = 5
	max_episode_timesteps = 100

	env = StrategySelectorEnv(num_envs=num_envs, episode_length=max_episode_timesteps, device=device)

	ppo_cfg = PPOConfig(
		obs_dim=1,
		action_dim=3,
		minibatch_size=128,
		is_discrete=True,
		device=device,
	)
	agent = PPOAgent(ppo_cfg)

	ckpt_dir = project_root / "report" / "checkpoints" / "strategy_selector" / timestamp
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	log_dir = project_root / "report" / "tensorboard" / "strategy_selector" / timestamp
	log_dir.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(log_dir=str(log_dir))

	iteration = 0
	consecutive_successes = 0
	max_consecutive_successes = 0
	episodes = 0
	timesteps = 0
	max_mean_reward = float("-inf")
	max_mean_reward_iteration = 0
	max_success_rate = 0.0
	max_success_rate_iteration = 0
	best_success_rate = 0.0
	best_mean_reward = float("-inf")
	best_model_state: dict[str, torch.Tensor] | None = None
	best_optimizer_state: dict | None = None
	total_env_steps = 0

	obs = env.reset()

	try:
		while True:
			obs_buf = []
			actions_buf = []
			logp_buf = []
			values_buf = []
			rewards_buf = []
			dones_buf = []
			success_buf = []
			done_count_iter = torch.zeros((), device=device)
			success_done_count_iter = torch.zeros((), device=device)

			for _ in range(horizon):
				assert obs.is_cuda
				act = agent.act(obs)
				next_step = env.step(act["action"])
				assert next_step["obs"].is_cuda
				assert next_step["reward"].is_cuda

				obs_buf.append(obs.clone())
				actions_buf.append(act["action"].clone())
				logp_buf.append(act["log_prob"].clone())
				values_buf.append(act["value"].clone())
				rewards_buf.append(next_step["reward"].clone())
				dones_buf.append(next_step["done"].float().clone())
				success_buf.append(next_step["info"]["success"].clone())
				done_mask = next_step["done"]
				done_count_iter = done_count_iter + done_mask.float().sum()
				success_done_count_iter = success_done_count_iter + (
					(next_step["info"]["success"] & done_mask).float().sum()
				)
				done = done_mask
				obs = env.reset(done) if torch.any(done) else next_step["obs"]

			with torch.no_grad():
				last_value = agent.act(obs)["value"]

			obs_t = torch.stack(obs_buf, dim=0)
			actions_t = torch.stack(actions_buf, dim=0)
			logp_t = torch.stack(logp_buf, dim=0)
			values_t = torch.stack(values_buf, dim=0)
			rewards_t = torch.stack(rewards_buf, dim=0)
			dones_t = torch.stack(dones_buf, dim=0)

			advantages, returns = agent.compute_gae(
				rewards=rewards_t,
				dones=dones_t,
				values=values_t,
				last_value=last_value,
			)
			assert advantages.is_cuda
			total_env_steps += num_envs * horizon

			loss_dict = agent.update(
				obs=obs_t,
				actions=actions_t,
				old_log_probs=logp_t,
				returns=returns,
				advantages=advantages,
				total_env_steps=total_env_steps,
			)

			# 1 iteration = 1 PPO update using all env data
			iteration += 1

			total_reward = rewards_t.sum()
			# Per-iteration mean across envs:
			# 1) sum rewards over rollout horizon for each env, then
			# 2) average those per-env iteration returns across all envs.
			# Equivalent to total_reward / num_envs.
			current_mean_reward = float((total_reward / num_envs).item())
			max_mean_reward = max(max_mean_reward, current_mean_reward)
			if max_mean_reward == current_mean_reward:
				max_mean_reward_iteration = iteration

			completed_eps_iter = int(done_count_iter.item())
			# Always calculate latest_success_rate: 0 if no episodes complete this iter, else actual rate
			if completed_eps_iter > 0:
				latest_success_rate = float((success_done_count_iter / done_count_iter).item())
				# Update consecutive successes when episodes actually complete
				consecutive_successes = consecutive_successes + 1 if latest_success_rate >= 0.95 else 0
			else:
				latest_success_rate = 0.0  # Show 0 when no episodes complete this iteration
				consecutive_successes = 0    # Reset when no episodes complete
			
			max_success_rate = max(max_success_rate, latest_success_rate)
			if max_success_rate == latest_success_rate:
				max_success_rate_iteration = iteration

			episodes += int(dones_t.sum().item())
			timesteps += num_envs * horizon
			max_consecutive_successes = max(max_consecutive_successes, consecutive_successes)

			# Update entropy schedule based on success rate
			agent._update_schedules(total_env_steps, success_rate=latest_success_rate)

			# Save best state based on configured tracking method
			should_save = False
			if best_state_tracking == "reward":
				should_save = current_mean_reward > best_mean_reward
			elif best_state_tracking == "success_rate":
				should_save = latest_success_rate > best_success_rate
			
			if should_save:
				best_success_rate = latest_success_rate
				best_mean_reward = current_mean_reward
				best_model_state = copy.deepcopy(agent.model.state_dict())
				best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())

			# Reward-collapse rollback: trigger only on mean_reward collapse
			if (
				best_mean_reward > float("-inf")
				and current_mean_reward < (best_mean_reward * 0.8)
				and best_model_state is not None
				and best_optimizer_state is not None
			):
				agent.model.load_state_dict(best_model_state)
				agent.optimizer.load_state_dict(best_optimizer_state)
				agent.apply_rollback_damping(success_rate=latest_success_rate)
				for group in agent.optimizer.param_groups:
					group["lr"] = max(1e-6, group["lr"])
				agent.cfg.entropy_coef = max(1e-5, agent.cfg.entropy_coef)

			_print_progress(
				iteration,
				episodes,
				timesteps,
				completed_eps_iter,
				current_mean_reward,
				max_mean_reward,
				max_mean_reward_iteration,
				latest_success_rate,
				max_success_rate,
				max_success_rate_iteration,
				consecutive_successes,
				max_consecutive_successes,
				float(loss_dict["entropy_coef"]),
			)

			_add_scalar_with_all_axes(writer, "selector/mean_reward", current_mean_reward, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/policy_loss", loss_dict["policy_loss"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/value_loss", loss_dict["value_loss"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/entropy", loss_dict["entropy"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/entropy_coef", loss_dict["entropy_coef"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/learning_rate", loss_dict["learning_rate"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/approx_kl", loss_dict["approx_kl"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/current_mean_reward", current_mean_reward, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "selector/best_mean_reward", best_mean_reward, iteration, episodes, timesteps)

			_add_scalar_with_all_axes(writer, "train/consecutive_successes", consecutive_successes, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/max_consecutive_successes", max_consecutive_successes, iteration, episodes, timesteps)

			if iteration % 1000 == 0:
				ckpt_path = ckpt_dir / f"ppo_iter_{iteration}.pt"
				agent.save(str(ckpt_path), iteration)

			if consecutive_successes >= CONSECUTIVE_SUCCESS_THRES:
				final_ckpt = ckpt_dir / f"ppo_iter_{iteration}_final.pt"
				agent.save(str(final_ckpt), iteration)
				print()
				break

	except KeyboardInterrupt:
		print()
		interrupt_ckpt = ckpt_dir / f"ppo_iter_{iteration}_interrupt.pt"
		agent.save(str(interrupt_ckpt), iteration)
	finally:
		writer.flush()
		writer.close()


def train_goal_executor(best_state_tracking: str = "reward") -> None:
	global _progress_started
	_progress_started = False
	CONSECUTIVE_SUCCESS_THRES = 100
	if best_state_tracking not in ["reward", "success_rate"]:
		raise ValueError(f"best_state_tracking must be 'reward' or 'success_rate', got '{best_state_tracking}'")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required. CPU fallback is disabled.")

	device = "cuda"
	print(f"Training goal executor with best_state_tracking={best_state_tracking}")
	project_root = PROJECT_ROOT
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	model_xml = project_root / "assets" / "humanoid_2d" / "humanoid_2d.xml"

	num_envs = 12_000
	horizon = 100
	max_episode_timesteps = 7000

	env = GoalConditionedExecutorEnv(model_xml=str(model_xml), num_envs=num_envs, dt=0.02, episode_length=max_episode_timesteps, device=device)

	ppo_cfg = PPOConfig(
		obs_dim=7,
		action_dim=env.action_dim,
		minibatch_size=128,
		is_discrete=False,
		device=device,
	)
	agent = PPOAgent(ppo_cfg)

	ckpt_dir = project_root / "report" / "checkpoints" / "goal_executor" / timestamp
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	log_dir = project_root / "report" / "tensorboard" / "goal_executor" / timestamp
	log_dir.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(log_dir=str(log_dir))

	iteration = 0
	consecutive_successes = 0
	max_consecutive_successes = 0
	episodes = 0
	timesteps = 0
	max_mean_reward = float("-inf")
	max_mean_reward_iteration = 0
	max_success_rate = 0.0
	max_success_rate_iteration = 0
	best_success_rate = 0.0
	best_mean_reward = float("-inf")
	best_model_state: dict[str, torch.Tensor] | None = None
	best_optimizer_state: dict | None = None
	total_env_steps = 0

	obs = env.reset()

	try:
		while True:
			obs_buf = []
			actions_buf = []
			logp_buf = []
			values_buf = []
			rewards_buf = []
			dones_buf = []
			success_buf = []
			done_count_iter = torch.zeros((), device=device)
			success_done_count_iter = torch.zeros((), device=device)

			brace_reward_total = torch.zeros((), device=device)
			roll_reward_total = torch.zeros((), device=device)
			brace_count = torch.zeros((), device=device)
			roll_count = torch.zeros((), device=device)

			brace_arm_first = torch.zeros((), device=device)
			brace_arm_sync = torch.zeros((), device=device)
			brace_knee_timing = torch.zeros((), device=device)
			brace_head_impact = torch.zeros((), device=device)

			roll_r_vel = torch.zeros((), device=device)
			roll_r_rot = torch.zeros((), device=device)
			roll_r_tuck = torch.zeros((), device=device)
			roll_c_ctrl = torch.zeros((), device=device)
			roll_c_impact = torch.zeros((), device=device)

			for _ in range(horizon):
				assert obs.is_cuda
				act = agent.act(obs)
				next_step = env.step(act["action"])
				assert next_step["obs"].is_cuda
				assert next_step["reward"].is_cuda

				obs_buf.append(obs.clone())
				actions_buf.append(act["action"].clone())
				logp_buf.append(act["log_prob"].clone())
				values_buf.append(act["value"].clone())
				rewards_buf.append(next_step["reward"].clone())
				dones_buf.append(next_step["done"].float().clone())
				success_buf.append(next_step["info"]["success"].clone())
				done_mask = next_step["done"]
				done_count_iter = done_count_iter + done_mask.float().sum()
				success_done_count_iter = success_done_count_iter + (
					(next_step["info"]["success"] & done_mask).float().sum()
				)

				goal_id = next_step["info"]["goal_id"]
				is_brace = goal_id == GoalId.BRACE
				is_roll = goal_id == GoalId.ROLL

				brace_reward_total = brace_reward_total + next_step["info"]["reward_brace"].sum()
				roll_reward_total = roll_reward_total + next_step["info"]["reward_roll"].sum()
				brace_count = brace_count + is_brace.float().sum()
				roll_count = roll_count + is_roll.float().sum()

				brace_arm_first = brace_arm_first + next_step["info"]["brace_r_arm_first"].sum()
				brace_arm_sync = brace_arm_sync + next_step["info"]["brace_r_arm_sync"].sum()
				brace_knee_timing = brace_knee_timing + next_step["info"]["brace_r_knee_timing"].sum()
				brace_head_impact = brace_head_impact + next_step["info"]["brace_c_head_impact"].sum()

				roll_r_vel = roll_r_vel + next_step["info"]["roll_r_vel"].sum()
				roll_r_rot = roll_r_rot + next_step["info"]["roll_r_rot"].sum()
				roll_r_tuck = roll_r_tuck + next_step["info"]["roll_r_tuck"].sum()
				roll_c_ctrl = roll_c_ctrl + next_step["info"]["roll_c_ctrl"].sum()
				roll_c_impact = roll_c_impact + next_step["info"]["roll_c_impact"].sum()

				done = done_mask
				obs = env.reset(done) if torch.any(done) else next_step["obs"]

			with torch.no_grad():
				last_value = agent.act(obs)["value"]

			obs_t = torch.stack(obs_buf, dim=0)
			actions_t = torch.stack(actions_buf, dim=0)
			logp_t = torch.stack(logp_buf, dim=0)
			values_t = torch.stack(values_buf, dim=0)
			rewards_t = torch.stack(rewards_buf, dim=0)
			dones_t = torch.stack(dones_buf, dim=0)

			advantages, returns = agent.compute_gae(
				rewards=rewards_t,
				dones=dones_t,
				values=values_t,
				last_value=last_value,
			)
			assert advantages.is_cuda
			total_env_steps += num_envs * horizon

			loss_dict = agent.update(
				obs=obs_t,
				actions=actions_t,
				old_log_probs=logp_t,
				returns=returns,
				advantages=advantages,
				total_env_steps=total_env_steps,
			)

			iteration += 1

			total_reward = rewards_t.sum()
			# Per-iteration mean across envs:
			# 1) sum rewards over rollout horizon for each env, then
			# 2) average those per-env iteration returns across all envs.
			# Equivalent to total_reward / num_envs.
			current_mean_reward = float((total_reward / num_envs).item())
			max_mean_reward = max(max_mean_reward, current_mean_reward)
			if max_mean_reward == current_mean_reward:
				max_mean_reward_iteration = iteration

			completed_eps_iter = int(done_count_iter.item())
			# Always calculate latest_success_rate: 0 if no episodes complete this iter, else actual rate
			if completed_eps_iter > 0:
				latest_success_rate = float((success_done_count_iter / done_count_iter).item())
				# Update consecutive successes when episodes actually complete
				consecutive_successes = consecutive_successes + 1 if latest_success_rate >= 0.95 else 0
			else:
				latest_success_rate = 0.0  # Show 0 when no episodes complete this iteration
				consecutive_successes = 0    # Reset when no episodes complete
			
			max_success_rate = max(max_success_rate, latest_success_rate)
			if max_success_rate == latest_success_rate:
				max_success_rate_iteration = iteration

			# Update entropy schedule based on success rate
			agent._update_schedules(total_env_steps, success_rate=latest_success_rate)

			episodes += int(dones_t.sum().item())
			timesteps += num_envs * horizon
			max_consecutive_successes = max(max_consecutive_successes, consecutive_successes)

			# Save best state based on configured tracking method
			should_save = False
			if best_state_tracking == "reward":
				should_save = current_mean_reward > best_mean_reward
			elif best_state_tracking == "success_rate":
				should_save = latest_success_rate > best_success_rate
			
			if should_save:
				best_success_rate = latest_success_rate
				best_mean_reward = current_mean_reward
				best_model_state = copy.deepcopy(agent.model.state_dict())
				best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())

			# Reward-collapse rollback: trigger only on mean_reward collapse
			if (
				best_mean_reward > float("-inf")
				and current_mean_reward < (best_mean_reward * 0.8)
				and best_model_state is not None
				and best_optimizer_state is not None
			):
				agent.model.load_state_dict(best_model_state)
				agent.optimizer.load_state_dict(best_optimizer_state)
				agent.apply_rollback_damping(success_rate=latest_success_rate)
				for group in agent.optimizer.param_groups:
					group["lr"] = max(1e-6, group["lr"])
				agent.cfg.entropy_coef = max(1e-5, agent.cfg.entropy_coef)

			_print_progress(
				iteration,
				episodes,
				timesteps,
				completed_eps_iter,
				current_mean_reward,
				max_mean_reward,
				max_mean_reward_iteration,
				latest_success_rate,
				max_success_rate,
				max_success_rate_iteration,
				consecutive_successes,
				max_consecutive_successes,
				float(loss_dict["entropy_coef"]),
			)

			_add_scalar_with_all_axes(writer, "train/mean_reward", current_mean_reward, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/policy_loss", loss_dict["policy_loss"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/value_loss", loss_dict["value_loss"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/entropy", loss_dict["entropy"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/entropy_coef", loss_dict["entropy_coef"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/learning_rate", loss_dict["learning_rate"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/approx_kl", loss_dict["approx_kl"], iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/current_mean_reward", current_mean_reward, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/best_mean_reward", best_mean_reward, iteration, episodes, timesteps)

			brace_den = torch.clamp(brace_count, min=1.0)
			roll_den = torch.clamp(roll_count, min=1.0)

			_add_scalar_with_all_axes(writer, "Goal BRACE/total_reward", brace_reward_total / brace_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal BRACE/r_arm_first", brace_arm_first / brace_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal BRACE/r_arm_sync", brace_arm_sync / brace_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal BRACE/r_knee_timing", brace_knee_timing / brace_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal BRACE/c_head_impact", brace_head_impact / brace_den, iteration, episodes, timesteps)

			_add_scalar_with_all_axes(writer, "Goal ROLL/total_reward", roll_reward_total / roll_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal ROLL/r_vel", roll_r_vel / roll_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal ROLL/r_rot", roll_r_rot / roll_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal ROLL/r_tuck", roll_r_tuck / roll_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal ROLL/c_ctrl", roll_c_ctrl / roll_den, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "Goal ROLL/c_impact", roll_c_impact / roll_den, iteration, episodes, timesteps)

			_add_scalar_with_all_axes(writer, "train/consecutive_successes", consecutive_successes, iteration, episodes, timesteps)
			_add_scalar_with_all_axes(writer, "train/max_consecutive_successes", max_consecutive_successes, iteration, episodes, timesteps)

			if iteration % 1000 == 0:
				ckpt_path = ckpt_dir / f"ppo_iter_{iteration}.pt"
				agent.save(str(ckpt_path), iteration)

			if consecutive_successes >= CONSECUTIVE_SUCCESS_THRES:
				final_ckpt = ckpt_dir / f"ppo_iter_{iteration}_final.pt"
				agent.save(str(final_ckpt), iteration)
				print()
				break

	except KeyboardInterrupt:
		print()
		interrupt_ckpt = ckpt_dir / f"ppo_iter_{iteration}_interrupt.pt"
		agent.save(str(interrupt_ckpt), iteration)
	finally:
		writer.flush()
		writer.close()


def train(target_env: Literal["selector", "executor"] = "executor", best_state_tracking: str = "reward") -> None:
	if best_state_tracking not in ["reward", "success_rate"]:
		raise ValueError(f"best_state_tracking must be 'reward' or 'success_rate', got '{best_state_tracking}'")
	if target_env == "selector":
		train_strategy_selector(best_state_tracking=best_state_tracking)
		return
	if target_env == "executor":
		train_goal_executor(best_state_tracking=best_state_tracking)
		return
	raise ValueError(f"Unknown training target: {target_env}")


if __name__ == "__main__":
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
	train()
