from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
import warp as wp
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo.ppo import PPOAgent, PPOConfig
from envs.goal_conditioned import GoalId
from envs.robot_env import RobotHierarchicalEnv


def train() -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required. CPU fallback is disabled.")

	device = "cuda"
	project_root = PROJECT_ROOT
	model_xml = project_root / "assets" / "humanoid_2d" / "humanoid_2d.xml"

	num_envs = 4096
	horizon = 100

	env = RobotHierarchicalEnv(model_xml=str(model_xml), num_envs=num_envs, device=device)

	ppo_cfg = PPOConfig(
		obs_dim=env.obs_dim,
		action_dim=env.action_dim,
		minibatch_size=128,
		is_discrete=False,
		device=device,
	)
	agent = PPOAgent(ppo_cfg)

	ckpt_dir = project_root / "report" / "checkpoints"
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	log_dir = project_root / "report" / "tensorboard"
	log_dir.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(log_dir=str(log_dir))

	iteration = 0
	consecutive_successes = 0

	obs = env.reset()

	try:
		while True:
			with wp.ScopedTimer("ppo_iteration"):
				obs_buf = []
				actions_buf = []
				logp_buf = []
				values_buf = []
				rewards_buf = []
				dones_buf = []
				success_buf = []

				# metric accumulators (GPU)
				brace_reward_total = torch.zeros((), device=device)
				roll_reward_total = torch.zeros((), device=device)
				brace_count = torch.zeros((), device=device)
				roll_count = torch.zeros((), device=device)

				brace_arm_first = torch.zeros((), device=device)
				brace_knee_timing = torch.zeros((), device=device)
				brace_head_impact = torch.zeros((), device=device)

				roll_r_vel = torch.zeros((), device=device)
				roll_r_rot = torch.zeros((), device=device)
				roll_r_tuck = torch.zeros((), device=device)
				roll_c_ctrl = torch.zeros((), device=device)
				roll_c_impact = torch.zeros((), device=device)

				for _ in range(horizon):
					act = agent.act(obs)
					next_step = env.step(act["action"])

					obs_buf.append(obs)
					actions_buf.append(act["action"])
					logp_buf.append(act["log_prob"])
					values_buf.append(act["value"])
					rewards_buf.append(next_step["reward"])
					dones_buf.append(next_step["done"].float())
					success_buf.append(next_step["info"]["success"])

					goal_id = next_step["info"]["goal_id"]
					is_brace = goal_id == GoalId.BRACE
					is_roll = goal_id == GoalId.ROLL

					brace_reward_total = brace_reward_total + next_step["info"]["reward_brace"].sum()
					roll_reward_total = roll_reward_total + next_step["info"]["reward_roll"].sum()
					brace_count = brace_count + is_brace.float().sum()
					roll_count = roll_count + is_roll.float().sum()

					brace_arm_first = brace_arm_first + next_step["info"]["brace_r_arm_first"].sum()
					brace_knee_timing = brace_knee_timing + next_step["info"]["brace_r_knee_timing"].sum()
					brace_head_impact = brace_head_impact + next_step["info"]["brace_c_head_impact"].sum()

					roll_r_vel = roll_r_vel + next_step["info"]["roll_r_vel"].sum()
					roll_r_rot = roll_r_rot + next_step["info"]["roll_r_rot"].sum()
					roll_r_tuck = roll_r_tuck + next_step["info"]["roll_r_tuck"].sum()
					roll_c_ctrl = roll_c_ctrl + next_step["info"]["roll_c_ctrl"].sum()
					roll_c_impact = roll_c_impact + next_step["info"]["roll_c_impact"].sum()

					obs = next_step["obs"]

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

				loss_dict = agent.update(
					obs=obs_t,
					actions=actions_t,
					old_log_probs=logp_t,
					returns=returns,
					advantages=advantages,
				)

				# 1 iteration = 1 PPO update using all env data
				iteration += 1

				total_reward = rewards_t.sum()
				writer.add_scalar("train/total_reward", total_reward, iteration)
				writer.add_scalar("train/policy_loss", loss_dict["policy_loss"], iteration)
				writer.add_scalar("train/value_loss", loss_dict["value_loss"], iteration)
				writer.add_scalar("train/entropy", loss_dict["entropy"], iteration)

				brace_den = torch.clamp(brace_count, min=1.0)
				roll_den = torch.clamp(roll_count, min=1.0)

				writer.add_scalar("Goal BRACE/total_reward", brace_reward_total / brace_den, iteration)
				writer.add_scalar("Goal BRACE/r_arm_first", brace_arm_first / brace_den, iteration)
				writer.add_scalar("Goal BRACE/r_knee_timing", brace_knee_timing / brace_den, iteration)
				writer.add_scalar("Goal BRACE/c_head_impact", brace_head_impact / brace_den, iteration)

				writer.add_scalar("Goal ROLL/total_reward", roll_reward_total / roll_den, iteration)
				writer.add_scalar("Goal ROLL/r_vel", roll_r_vel / roll_den, iteration)
				writer.add_scalar("Goal ROLL/r_rot", roll_r_rot / roll_den, iteration)
				writer.add_scalar("Goal ROLL/r_tuck", roll_r_tuck / roll_den, iteration)
				writer.add_scalar("Goal ROLL/c_ctrl", roll_c_ctrl / roll_den, iteration)
				writer.add_scalar("Goal ROLL/c_impact", roll_c_impact / roll_den, iteration)

				success_matrix = torch.stack(success_buf, dim=0)
				all_env_success = torch.all(torch.any(success_matrix, dim=0))
				consecutive_successes = consecutive_successes + 1 if bool(all_env_success.item()) else 0
				writer.add_scalar("train/consecutive_successes", consecutive_successes, iteration)

				if iteration % 1000 == 0:
					ckpt_path = ckpt_dir / f"ppo_iter_{iteration}.pt"
					agent.save(str(ckpt_path), iteration)

				if consecutive_successes >= 1000:
					final_ckpt = ckpt_dir / f"ppo_iter_{iteration}_final.pt"
					agent.save(str(final_ckpt), iteration)
					break

	except KeyboardInterrupt:
		interrupt_ckpt = ckpt_dir / f"ppo_iter_{iteration}_interrupt.pt"
		agent.save(str(interrupt_ckpt), iteration)
	finally:
		writer.flush()
		writer.close()


if __name__ == "__main__":
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
	train()
