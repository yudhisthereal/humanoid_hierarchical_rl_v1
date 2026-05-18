from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import re
import sys

import mediapy as media
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo.ppo import PPOAgent, PPOConfig
from envs.goal_conditioned import GoalConditionedExecutorEnv, GoalId
from envs.sanity_check import SanityCheckEnv
from envs.strategy_selector import StrategySelectorEnv


wp.config.verbose = False
LOGGER = logging.getLogger("render")


def _resolve_render_size(model: mujoco.MjModel, width: int, height: int) -> tuple[int, int]:
	offwidth = width
	offheight = height
	try:
		offwidth = int(model.vis.global_.offwidth)
		offheight = int(model.vis.global_.offheight)
	except Exception:
		# Fallback when API/fields are unavailable; renderer will validate if needed.
		return width, height

	# MuJoCo renderer cannot exceed offscreen framebuffer dimensions.
	return max(1, min(width, offwidth)), max(1, min(height, offheight))


def _build_agent(env_name: str, obs_dim: int, action_dim: int, device: str, checkpoint: str | None) -> PPOAgent:
	if env_name == "selector":
		cfg = PPOConfig(
			obs_dim=obs_dim,
			action_dim=3,
			minibatch_size=128,
			is_discrete=True,
			device=device,
		)
	else:
		cfg = PPOConfig(
			obs_dim=obs_dim,
			action_dim=action_dim,
			minibatch_size=128,
			is_discrete=False,
			device=device,
		)

	agent = PPOAgent(cfg)
	if checkpoint is not None:
		agent.load(checkpoint)
	agent.model.eval()
	return agent


def _resolve_checkpoint(env_name: str, checkpoint: str | None) -> str | None:
	# Only use checkpoint if explicitly provided via CLI parameter.
	return checkpoint


def _selector_frame(push_force: float, action: int, reward: float, success: bool, done: bool) -> np.ndarray:
	h, w = 1080, 1920
	frame = np.zeros((h, w, 3), dtype=np.uint8)
	frame[:] = (20, 20, 20)

	# Push-force bar (left)
	force_norm = float(np.clip(push_force / 70.0, 0.0, 1.0))
	bar_h = int((h - 240) * force_norm)
	frame[h - 120 - bar_h : h - 120, 180:540] = np.array([60, 150, 255], dtype=np.uint8)

	# Action block (right)
	action_colors = {
		0: np.array([120, 120, 120], dtype=np.uint8),
		1: np.array([80, 220, 120], dtype=np.uint8),
		2: np.array([220, 140, 80], dtype=np.uint8),
	}
	frame[240:840, 1080:1740] = action_colors.get(int(action), np.array([180, 180, 180], dtype=np.uint8))

	# Status strip
	if success:
		frame[0:90, :] = np.array([40, 180, 90], dtype=np.uint8)
	elif done:
		frame[0:90, :] = np.array([180, 50, 50], dtype=np.uint8)
	else:
		frame[0:90, :] = np.array([90, 90, 90], dtype=np.uint8)

	# Reward intensity marker
	reward_clip = float(np.clip((reward + 1.0) / 2.0, 0.0, 1.0))
	reward_w = int((w - 120) * reward_clip)
	frame[h - 60 : h - 30, 60 : 60 + reward_w] = np.array([255, 220, 90], dtype=np.uint8)

	return frame


def _load_single_env_module(env_name: str):
	module_path = PROJECT_ROOT / "scripts" / env_name / f"{env_name}.py"
	if not module_path.exists():
		raise FileNotFoundError(f"Unable to find single-env module for {env_name}: {module_path}")

	spec = importlib.util.spec_from_file_location(f"render_{env_name}", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Unable to load module spec for {env_name} from {module_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _zero_ctrl_action_from_env(env) -> np.ndarray:
	"""Return a normalized action that maps to zero actuator ctrl if possible."""
	if hasattr(env, "ctrl_min") and hasattr(env, "ctrl_max"):
		ctrl_min = np.asarray(env.ctrl_min, dtype=np.float32)
		ctrl_max = np.asarray(env.ctrl_max, dtype=np.float32)
		denom = ctrl_max - ctrl_min
		with np.errstate(divide="ignore", invalid="ignore"):
			action = np.where(
				denom != 0.0,
				2.0 * (0.0 - ctrl_min) / denom - 1.0,
				0.0,
			)
		return np.clip(action.astype(np.float32), -1.0, 1.0)
	return np.zeros((int(getattr(env, "action_dim", 0)),), dtype=np.float32)


def _make_brace_env_limp(env) -> None:
	"""Reduce actuator stiffness so the robot goes limp after success."""
	factor = 0.01
	try:
		if hasattr(env, "model") and hasattr(env.model, "actuator_gainprm"):
			env.model.actuator_gainprm[:] *= factor
		if hasattr(env, "model") and hasattr(env.model, "actuator_biasprm"):
			env.model.actuator_biasprm[:] *= factor
		if hasattr(env, "data") and hasattr(env.data, "ctrl"):
			env.data.ctrl[:] = 0.0
	except Exception as exc:
		LOGGER.exception("[%s] Failed to apply limp mode: %s", getattr(env, "__class__", type(env)).__name__, exc)


def _render_brace_single_episode(
	env_name: str,
	video_name: str,
	checkpoint: str | None = None,
	timesteps: int = 300,
	fps: int = 30,
	no_perturb: bool = False,
	no_control: bool = False,
	push_force: float | None = None,
) -> Path:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required. CPU fallback is disabled.")

	device = "cuda"
	module = _load_single_env_module(env_name)
	env = module.HumanoidEnv()
	out_dir = PROJECT_ROOT / "report" / "video" / env_name
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{video_name}.mp4"

	if no_control:
		agent = None
	else:
		agent = module.PPO(env.obs_dim, env.action_dim)
		if checkpoint is not None:
			state_dict = torch.load(checkpoint, map_location=device)
			agent.network.load_state_dict(state_dict)
		agent.network.to(device)
		agent.network.eval()

	render_w, render_h = _resolve_render_size(env.model, width=1920, height=1080)
	renderer = mujoco.Renderer(env.model, width=render_w, height=render_h)
	frames: list[np.ndarray] = []

	try:
		obs = env.reset()
		success_limp = False
		limp_action = _zero_ctrl_action_from_env(env)
		if no_perturb:
			env.push_force = 0.0
			env.push_steps_left = 0
			if push_force is not None:
				_tqdm_write(f"[INFO] [{env_name}] --push-force is ignored because --no-perturb is enabled.")
		elif push_force is not None:
			env.push_force = float(push_force)

		with torch.no_grad():
			for step_idx in _tqdm_range(max(1, timesteps), f"Rendering {env_name}"):
				if success_limp:
					action = limp_action
				elif no_control:
					action = np.zeros((env.action_dim,), dtype=np.float32)
				else:
					obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
					action_tensor, _ = agent.network.get_action(obs_tensor, deterministic=True)
					action = action_tensor.detach().cpu().numpy()

				obs, reward, done, _ = env.step(action)
				success = bool(env.get_success())
				if success and not success_limp:
					_make_brace_env_limp(env)
					success_limp = True
					action = limp_action

				renderer.update_scene(env.data)
				frames.append(renderer.render())

				if step_idx % 100 == 0:
					_tqdm_write(
						f"[INFO] [{env_name}] step={int(step_idx)} "
						f"reward={float(reward):.4f} "
						f"success={success} "
						f"done={bool(done)}"
					)

				if done and not success:
					obs = env.reset()
					success_limp = False
					limp_action = _zero_ctrl_action_from_env(env)
					if no_perturb:
						env.push_force = 0.0
						env.push_steps_left = 0
					elif push_force is not None:
						env.push_force = float(push_force)

		media.write_video(str(out_path), frames, fps=fps)
	finally:
		renderer.close()

	return out_path


def _tqdm_range(total: int, desc: str):
	try:
		tqdm_mod = importlib.import_module("tqdm")
		return tqdm_mod.tqdm(range(total), desc=desc, unit="step", leave=True, dynamic_ncols=True)
	except Exception:
		return range(total)


def _tqdm_write(message: str) -> None:
	try:
		tqdm_mod = importlib.import_module("tqdm")
		tqdm_mod.tqdm.write(message)
	except Exception:
		print(message)


def _disable_perturbation_safely(
	env: GoalConditionedExecutorEnv | StrategySelectorEnv | SanityCheckEnv,
	env_name: str,
) -> None:
	try:
		env.push_force.zero_()
		if hasattr(env, "push_steps_left"):
			env.push_steps_left.zero_()
	except Exception as exc:
		LOGGER.exception("[%s] Failed to disable perturbation: %s", env_name, exc)


def _override_push_force_safely(
	env: GoalConditionedExecutorEnv | StrategySelectorEnv | SanityCheckEnv,
	env_name: str,
	push_force: float | None,
) -> None:
	if push_force is None:
		return
	try:
		env.push_force.fill_(float(push_force))
	except Exception as exc:
		LOGGER.exception("[%s] Failed to override push force: %s", env_name, exc)


def _policy_action_or_zero(
	agent: PPOAgent | None,
	obs: torch.Tensor,
	no_control: bool,
	action_shape: tuple[int, ...],
	dtype: torch.dtype,
	device: str,
	env_name: str,
) -> torch.Tensor:
	if no_control:
		return torch.zeros(action_shape, device=device, dtype=dtype)
	if agent is None:
		LOGGER.error("[%s] Agent is None while control is enabled. Using zero action fallback.", env_name)
		return torch.zeros(action_shape, device=device, dtype=dtype)
	try:
		return agent.act(obs)["action"]
	except Exception as exc:
		LOGGER.exception("[%s] Policy action inference failed. Using zero action fallback: %s", env_name, exc)
		return torch.zeros(action_shape, device=device, dtype=dtype)


def render_episode(
	env_name: str,
	video_name: str,
	checkpoint: str | None = None,
	timesteps: int = 300,
	fps: int = 30,
	no_perturb: bool = False,
	no_control: bool = False,
	goal: str | None = None,
	push_force: float | None = None,
) -> Path:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required. CPU fallback is disabled.")

	device = "cuda"
	out_dir = PROJECT_ROOT / "report" / "video" / env_name
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{video_name}.mp4"

	if env_name == "executor":
		model_xml = PROJECT_ROOT / "assets" / "humanoid_2d" / "humanoid_2d.xml"
		env = GoalConditionedExecutorEnv(
			model_xml=str(model_xml),
			num_envs=1,
			dt=0.02,
			episode_length=max(1, timesteps),
			device=device,
		)
		# Set goal if specified
		if goal is not None:
			goal_upper = goal.upper()
			if goal_upper == "BRACE":
				env.goal_id[0] = GoalId.BRACE
			elif goal_upper == "ROLL":
				env.goal_id[0] = GoalId.ROLL
			else:
				raise ValueError(f"Invalid goal: {goal}. Must be 'brace' or 'roll'.")
		resolved_ckpt = _resolve_checkpoint("executor", checkpoint)
		random_control = (resolved_ckpt is None) and (not no_control)
		act_joint_ids = env.mj_model.actuator_trnid[:, 0].astype(np.int64)
		act_qvel_idx = torch.as_tensor(env.mj_model.jnt_dofadr[act_joint_ids], device=env.device, dtype=torch.long)
		frame_skip = max(1, int(round(env.dt / max(float(env.mj_model.opt.timestep), 1e-9))))
		if no_control:
			agent = None
		else:
			agent = _build_agent("executor", env.obs_torch.shape[1], env.action_dim, device, resolved_ckpt)
			if random_control:
				_tqdm_write("[INFO] [executor] No checkpoint provided. Using exploratory random control for visible joint motion.")

		render_w, render_h = _resolve_render_size(env.mj_model, width=1920, height=1080)
		renderer = mujoco.Renderer(env.mj_model, width=render_w, height=render_h)
		frames: list[np.ndarray] = []

		try:
			obs = env.reset()
			if no_perturb:
				_disable_perturbation_safely(env, env_name)
				if push_force is not None:
					_tqdm_write("[INFO] [executor] --push-force is ignored because --no-perturb is enabled.")
			else:
				_override_push_force_safely(env, env_name, push_force)
				_tqdm_write(
					f"[INFO] [{env_name}] Initial push: force={float(env.push_force[0].item()):.2f} "
					f"steps={int(env.push_steps_left[0].item())}"
				)
			try:
				mjw.get_data_into(env.mj_data, env.mj_model, env.data)
			except Exception as exc:
				LOGGER.exception("[%s] Failed initial sync from Warp to MuJoCo: %s", env_name, exc)
				raise

			# Local MJ-side push state for authoritative control/render path.
			push_force = float(env.push_force[0].item()) if (not no_perturb) else 0.0
			push_steps_left = int(env.push_steps_left[0].item()) if (not no_perturb) else 0

			# MuJoCo-style recording: sample frames by simulation time, not by step count.
			frame_dt = 1.0 / float(max(1, fps))
			next_frame_t = float(env.mj_data.time)

			renderer.update_scene(env.mj_data)
			frames.append(renderer.render())
			next_frame_t += frame_dt

			with torch.no_grad():
				for step_idx in _tqdm_range(max(1, timesteps), "Rendering executor"):
					# Use the actual environment observation
					obs = env.obs_torch[:1].clone()
					
					if random_control:
						action = torch.empty((1, env.action_dim), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
					else:
						action = _policy_action_or_zero(
							agent=agent,
							obs=obs,
							no_control=no_control,
							action_shape=(1, env.action_dim),
							dtype=torch.float32,
							device=device,
							env_name=env_name,
						)

					ctrl = env.map_action_to_ctrl(action)
					env.mj_data.ctrl[:] = ctrl[0].detach().cpu().numpy()

					if (not no_perturb) and push_steps_left > 0:
						env.mj_data.xfrc_applied[:] = 0.0
						env.mj_data.xfrc_applied[env._head_body_id, 0] = push_force
						env.mj_data.qvel[env._rootx_qvel_idx] += push_force * env.dt * 0.02
						push_steps_left -= 1
					else:
						env.mj_data.xfrc_applied[:] = 0.0

					for _ in range(frame_skip):
						mujoco.mj_step(env.mj_model, env.mj_data)

					# Refresh observation buffer from simulation state
					env._refresh_state_from_sim()
					env.obs_torch[0, 0] = env.vx[0]
					env.obs_torch[0, 1] = env.omega[0]
					env.obs_torch[0, 2] = env.rotation[0]
					env.obs_torch[0, 3] = env.com_z[0]
					env.obs_torch[0, 4] = env.head_z[0]
					env.obs_torch[0, 5] = env.waist_angle[0]
					env.obs_torch[0, 6] = env.knees_angle[0]

					if step_idx % 100 == 0:
						act_qvel_abs = float(np.mean(np.abs(env.mj_data.qvel[act_joint_ids])))
						ctrl_abs = float(np.mean(np.abs(env.mj_data.ctrl)))
						_tqdm_write(
							f"[INFO] [{env_name}] step={int(step_idx)} "
							f"action_abs_mean={float(torch.mean(torch.abs(action)).item()):.4f} "
							f"ctrl_abs_mean={ctrl_abs:.4f} "
							f"act_qvel_abs_mean={act_qvel_abs:.4f} "
							f"vx={float(env.mj_data.qvel[env._rootx_qvel_idx]):.4f} "
							f"push_force={push_force:.2f} "
							f"push_left={push_steps_left}"
						)

					sim_t = float(env.mj_data.time)
					while sim_t + 1e-12 >= next_frame_t:
						try:
							renderer.update_scene(env.mj_data)
							frames.append(renderer.render())
						except Exception as exc:
							LOGGER.exception("[%s] Frame render failed at step %s: %s", env_name, step_idx, exc)
							raise
						next_frame_t += frame_dt
					# In pure MuJoCo render path we intentionally do not early-break on env.done.

			try:
				media.write_video(str(out_path), frames, fps=fps)
			except Exception as exc:
				LOGGER.exception("[%s] Video writing failed for %s: %s", env_name, out_path, exc)
				raise
		finally:
			renderer.close()

	elif env_name in {"brace_only_single", "brace_arm_only_single"}:
		out_path = _render_brace_single_episode(
			env_name=env_name,
			video_name=video_name,
			checkpoint=checkpoint,
			timesteps=timesteps,
			fps=fps,
			no_perturb=no_perturb,
			no_control=no_control,
			push_force=push_force,
		)

	elif env_name == "selector":
		env = StrategySelectorEnv(num_envs=1, episode_length=max(1, timesteps), device=device)
		resolved_ckpt = _resolve_checkpoint("selector", checkpoint)
		if no_control:
			agent = None
		else:
			agent = _build_agent("selector", 1, action_dim=3, device=device, checkpoint=resolved_ckpt)
		frames: list[np.ndarray] = []
		obs = env.reset()
		if no_perturb:
			_disable_perturbation_safely(env, env_name)
			if push_force is not None:
				_tqdm_write("[INFO] [selector] --push-force is ignored because --no-perturb is enabled.")
		else:
			_override_push_force_safely(env, env_name, push_force)

		with torch.no_grad():
			for step_idx in _tqdm_range(max(1, timesteps), "Rendering selector"):
				action = _policy_action_or_zero(
					agent=agent,
					obs=obs,
					no_control=no_control,
					action_shape=(1,),
					dtype=torch.long,
					device=device,
					env_name=env_name,
				)
				try:
					out = env.step(action)
				except Exception as exc:
					LOGGER.exception("[%s] env.step failed at step %s: %s", env_name, step_idx, exc)
					raise
				obs = out["obs"]
				if no_perturb:
					_disable_perturbation_safely(env, env_name)

				frames.append(
					_selector_frame(
						push_force=float(env.push_force[0].item()),
						action=int(action[0].item()),
						reward=float(out["reward"][0].item()),
						success=bool(out["info"]["success"][0].item()),
						done=bool(out["done"][0].item()),
					)
				)
				if bool(out["done"][0].item()):
					obs = env.reset(out["done"])
					if no_perturb:
						_disable_perturbation_safely(env, env_name)
					else:
						_override_push_force_safely(env, env_name, push_force)
		try:
			media.write_video(str(out_path), frames, fps=fps)
		except Exception as exc:
			LOGGER.exception("[%s] Video writing failed for %s: %s", env_name, out_path, exc)
			raise
	elif env_name == "sanity_check":
		model_xml = PROJECT_ROOT / "assets" / "humanoid_2d" / "humanoid_2d_half.xml"
		env = SanityCheckEnv(
			model_xml=str(model_xml),
			num_envs=1,
			dt=0.02,
			episode_length=max(1, timesteps),
			device=device,
		)
		resolved_ckpt = _resolve_checkpoint("sanity_check", checkpoint)
		random_control = (resolved_ckpt is None) and (not no_control)
		if no_control:
			agent = None
		else:
			agent = _build_agent(
				"sanity_check",
				obs_dim=env.obs_torch.shape[1],
				action_dim=env.action_dim,
				device=device,
				checkpoint=resolved_ckpt,
			)
			if random_control:
				_tqdm_write("[INFO] [sanity_check] No checkpoint provided. Using exploratory random control.")

		render_w, render_h = _resolve_render_size(env.mj_model, width=1920, height=1080)
		renderer = mujoco.Renderer(env.mj_model, width=render_w, height=render_h)
		frames: list[np.ndarray] = []

		try:
			obs = env.reset()
			if no_perturb:
				_tqdm_write("[INFO] [sanity_check] --no-perturb has no effect for sanity_check env.")
			if push_force is not None:
				_tqdm_write("[INFO] [sanity_check] --push-force has no effect for sanity_check env.")

			with torch.no_grad():
				for step_idx in _tqdm_range(max(1, timesteps), "Rendering sanity_check"):
					if random_control:
						action = torch.empty((1, env.action_dim), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
					else:
						action = _policy_action_or_zero(
							agent=agent,
							obs=obs,
							no_control=no_control,
							action_shape=(1, env.action_dim),
							dtype=torch.float32,
							device=device,
							env_name=env_name,
						)

					out = env.step(action)
					obs = out["obs"]

					mjw.get_data_into(env.mj_data, env.mj_model, env.data)
					renderer.update_scene(env.mj_data)
					frames.append(renderer.render())

					if step_idx % 100 == 0:
						_tqdm_write(
							f"[INFO] [{env_name}] step={int(step_idx)} "
							f"reward={float(out['reward'][0].item()):.4f} "
							f"err={float(out['info']['err'][0].item()):.4f} "
							f"jit={float(out['info']['jit'][0].item()):.4f}"
						)

					if bool(out["done"][0].item()):
						obs = env.reset(out["done"])

			media.write_video(str(out_path), frames, fps=fps)
		finally:
			renderer.close()
	elif env_name == "sanity_check_single":
		# Import the exact environment from sanity_check.py
		import importlib.util
		spec = importlib.util.spec_from_file_location(
			"sanity_check", 
			PROJECT_ROOT / "scripts" / "singleenvs" / "sanity_check.py"
		)
		sanity_check_module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(sanity_check_module)
		
		# Use the exact HumanoidEnv class
		model_xml = PROJECT_ROOT / "assets" / "humanoid_2d" / "humanoid_2d_half.xml"
		env = sanity_check_module.HumanoidEnv(xml_path=str(model_xml))
		
		# Use the original PPONetwork class directly
		agent = None
		if checkpoint and not no_control:
			agent = sanity_check_module.PPONetwork(env.obs_dim, env.action_dim)
			agent.load_state_dict(torch.load(checkpoint, map_location=device))
			agent.to(device)
			agent.eval()
			_tqdm_write(f"[INFO] [sanity_check_single] Loaded checkpoint: {checkpoint}")
		elif not no_control:
			_tqdm_write("[INFO] [sanity_check_single] No checkpoint provided. Using zero control.")
		
		render_w, render_h = _resolve_render_size(env.model, width=1920, height=1080)
		renderer = mujoco.Renderer(env.model, width=render_w, height=render_h)
		frames: list[np.ndarray] = []
		
		try:
			obs = env.reset()
			
			with torch.no_grad():
				for step_idx in _tqdm_range(max(1, timesteps), "Rendering sanity_check_single"):
					if agent is not None:
						action, _ = agent.get_action(torch.FloatTensor(obs).to(device), deterministic=True)
						action = action.cpu().numpy()
					else:
						action = np.zeros(env.action_dim)
					
					next_obs, reward, done, _ = env.step(action)
					obs = next_obs
					
					renderer.update_scene(env.data)
					frames.append(renderer.render())
					
					if step_idx % 100 == 0:
						success = env.get_success()
						current_pos = np.array([env.data.qpos[env.model.jnt_qposadr[jid]] 
											for jid in env.joint_ids])
						distance = np.linalg.norm(current_pos - sanity_check_module.TARGET_POS)
						_tqdm_write(
							f"[INFO] [sanity_check_single] step={int(step_idx)} "
							f"reward={reward:.4f} "
							f"distance={distance:.4f} "
							f"success={success}"
						)
					
					if done:
						obs = env.reset()
			
			media.write_video(str(out_path), frames, fps=fps)
			_tqdm_write(f"[INFO] [sanity_check_single] Video saved: {out_path}")
		finally:
			renderer.close()

	return out_path


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", choices=["selector", "executor", "sanity_check", "sanity_check_single", "brace_only_single", "brace_arm_only_single"], required=True)
	parser.add_argument("--video-name", type=str, default="episode")
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--timesteps", type=int, default=300)
	parser.add_argument("--fps", type=int, default=30)
	parser.add_argument("--no-perturb", action="store_true")
	parser.add_argument("--no-control", action="store_true")
	parser.add_argument("--goal", type=str, choices=["brace", "roll"], default=None,
					   help="Goal for executor environment (brace or roll). Only used when --env executor.")
	parser.add_argument("--push-force", type=float, default=None,
					   help="Fixed perturbation force to apply (ignored with --no-perturb).")
	args = parser.parse_args()

	video_path = render_episode(
		env_name=args.env,
		video_name=args.video_name,
		checkpoint=args.checkpoint,
		timesteps=args.timesteps,
		fps=args.fps,
		no_perturb=args.no_perturb,
		no_control=args.no_control,
		goal=args.goal,
		push_force=args.push_force,
	)
	print(f"Saved video: {video_path}")


if __name__ == "__main__":
	main()
