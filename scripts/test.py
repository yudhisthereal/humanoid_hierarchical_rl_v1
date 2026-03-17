from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import warp as wp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo.ppo import PPOAgent, PPOConfig
from envs.robot_env import RobotHierarchicalEnv


def run_test(checkpoint: str, steps: int = 1000) -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required. CPU fallback is disabled.")

	device = "cuda"
	project_root = PROJECT_ROOT
	model_xml = project_root / "assets" / "humanoid_2d" / "humanoid_2d.xml"

	env = RobotHierarchicalEnv(model_xml=str(model_xml), num_envs=4096, device=device)
	cfg = PPOConfig(
		obs_dim=env.obs_dim,
		action_dim=env.action_dim,
		minibatch_size=128,
		is_discrete=False,
		device=device,
	)
	agent = PPOAgent(cfg)
	agent.load(checkpoint)
	agent.model.eval()

	obs = env.reset()

	with torch.no_grad():
		for _ in range(steps):
			with wp.ScopedTimer("test_inference_step"):
				act = agent.act(obs)
				out = env.step(act["action"])
				obs = out["obs"]


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--steps", type=int, default=1000)
	args = parser.parse_args()
	run_test(args.checkpoint, args.steps)
