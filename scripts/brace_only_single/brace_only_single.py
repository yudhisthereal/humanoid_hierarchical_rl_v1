from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


SCRIPT_DIR = Path(__file__).resolve().parent
XML_PATH = SCRIPT_DIR / "humanoid_2d_half.xml"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_brace_model.pth"
TRAINING_PLOT_PATH = SCRIPT_DIR / "training_progress.png"

ACTUATOR_NAMES = ["pos_waist", "pos_leg", "pos_shin", "pos_arm", "pos_forearm"]
JOINT_NAMES = ["waist_joint", "leg_joint", "shin_joint", "arm_joint", "forearm_joint"]


def kv(label, value, label2="", value2=""):
	s = f"{label}: \x1b[1;36m{value}\x1b[0m"
	if label2:
		s += f"  {label2}: \x1b[1;36m{value2}\x1b[0m"
	return s


class HumanoidEnv:
	def __init__(self, xml_path: Path | str = XML_PATH):
		self.model = mujoco.MjModel.from_xml_path(str(xml_path))
		self.data = mujoco.MjData(self.model)

		self.actuator_ids = []
		for name in ACTUATOR_NAMES:
			aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
			if aid < 0:
				raise ValueError(f"Actuator not found in model: {name}")
			self.actuator_ids.append(int(aid))

		self.joint_ids = []
		for name in JOINT_NAMES:
			jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
			if jid < 0:
				raise ValueError(f"Joint not found in model: {name}")
			self.joint_ids.append(int(jid))

		self.head_geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head"))
		if self.head_geom_id < 0:
			raise ValueError("Geom not found in model: head")
		self.floor_geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
		if self.floor_geom_id < 0:
			raise ValueError("Geom not found in model: floor")

		def _collect_geom_ids(names: list[str]) -> set[int]:
			ids: set[int] = set()
			for n in names:
				gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n))
				if gid >= 0:
					ids.add(gid)
			return ids

		self.arm_right_geom_ids = _collect_geom_ids(["uarm_geom", "larm_geom"])
		self.arm_left_geom_ids = _collect_geom_ids(["uarm_left_geom", "larm_left_geom"])
		self.knee_right_geom_ids = _collect_geom_ids(["shin_geom", "leg_geom"])
		self.knee_left_geom_ids = _collect_geom_ids(["shin_left_geom", "leg_left_geom"])
		self.torso_geom_ids = _collect_geom_ids(["lower_torso_geom", "upper_torso_geom"])
		self.waist_geom_ids = _collect_geom_ids(["upper_torso_geom"])

		self.rootx_qvel_idx = int(
			self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rootx")]
		)
		self.rootz_qpos_idx = int(
			self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rootz")]
		)
		self.rootz_qvel_idx = int(
			self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rootz")]
		)
		self.rooty_qpos_idx = int(
			self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rooty")]
		)
		self.rooty_qvel_idx = int(
			self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rooty")]
		)
		self.arm_qpos_idx = int(
			self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint")]
		)
		self.arm_qvel_idx = int(
			self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint")]
		)
		self.forearm_qpos_idx = int(
			self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "forearm_joint")]
		)
		self.forearm_qvel_idx = int(
			self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "forearm_joint")]
		)
		self.leg_qpos_idx = int(
			self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "leg_joint")]
		)
		self.leg_qvel_idx = int(
			self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "leg_joint")]
		)
		self.leg_left_qpos_idx = int(
			self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "leg_left_joint")]
		)
		self.leg_left_qvel_idx = int(
			self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "leg_left_joint")]
		)

		self.action_dim = len(self.actuator_ids)
		# BRACE-specific observation with joint velocities for jitter-awareness
		# [com_z, knees_angle, arm_angle, forearm_angle,
		#  arms_contact, head_contact, torso_contact,
		#  leg_vel, leg_left_vel, arm_vel, forearm_vel]
		self.obs_dim = 11

		self.dt = float(self.model.opt.timestep) * 5.0
		self.obs_clip = 50.0
		self.reward_clip = 100.0
		self.episode_length = 200

		self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

		# actuator ctrl limits (for mapping normalized actions -> actuator ctrl units)
		self.ctrl_min = np.array([float(self.model.actuator_ctrlrange[aid, 0]) for aid in self.actuator_ids], dtype=np.float32)
		self.ctrl_max = np.array([float(self.model.actuator_ctrlrange[aid, 1]) for aid in self.actuator_ids], dtype=np.float32)
		self.step_count = 0
		self.done = False
		self.last_success = False

		self.t_arms_l = float("inf")
		self.t_arms_r = float("inf")
		self.t_knees = float("inf")
		self.t_head = float("inf")
		self.t_torso = float("inf")
		self.t_waist = float("inf")

		# Use a fixed uniform push force for training (user requested)
		self.push_force_choices = np.array([90.0], dtype=np.float32)
		self.push_steps = 5
		self.push_kick_scale = 0.02
		self.push_force = 0.0
		self.push_steps_left = 0

		self.com_z = 0.0
		self.head_z = 0.0
		self.vx = 0.0
		self.omega = 0.0
		self.rotation = 0.0
		self.knees_angle = 0.0
		self.arm_angle = 0.0
		self.forearm_angle = 0.0
		self.leg_vel = 0.0
		self.leg_left_vel = 0.0
		self.arm_vel = 0.0
		self.forearm_vel = 0.0

		self.torso_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"))
		if self.torso_body_id < 0:
			raise ValueError("Body not found in model: torso")
		
		self.head_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "upper_torso"))
		if self.head_body_id < 0:
			raise ValueError("Body not found in model: upper_torso")

	def _get_joint_pos(self, idx: int) -> float:
		return float(self.data.qpos[idx])

	def map_action_to_ctrl(self, action_unit):
		"""Map normalized action in [-1,1] to actuator ctrl ranges.
		Accepts numpy array or torch tensor; returns same type as input.
		"""
		try:
			import torch
		except Exception:
			torch = None

		if torch is not None and isinstance(action_unit, torch.Tensor):
			a = torch.nan_to_num(action_unit, nan=0.0, posinf=1.0, neginf=-1.0)
			a = torch.clamp(a, -1.0, 1.0)
			ctrl_min_t = torch.as_tensor(self.ctrl_min, device=a.device, dtype=a.dtype)
			ctrl_max_t = torch.as_tensor(self.ctrl_max, device=a.device, dtype=a.dtype)
			return ctrl_min_t + 0.5 * (a + 1.0) * (ctrl_max_t - ctrl_min_t)

		# assume numpy-like
		a = np.nan_to_num(np.asarray(action_unit, dtype=np.float32), nan=0.0)
		a = np.clip(a, -1.0, 1.0)
		return self.ctrl_min + 0.5 * (a + 1.0) * (self.ctrl_max - self.ctrl_min)

	def _get_joint_vel(self, idx: int) -> float:
		return float(self.data.qvel[idx])

	def _refresh_state(self) -> None:
		self.vx = self._get_joint_vel(self.rootx_qvel_idx)
		self.omega = self._get_joint_vel(self.rooty_qvel_idx)
		self.rotation = self._get_joint_pos(self.rooty_qpos_idx)
		self.com_z = self._get_joint_pos(self.rootz_qpos_idx)
		self.head_z = float(self.data.geom_xpos[self.head_geom_id, 2])
		self.knees_angle = 0.5 * (self._get_joint_pos(self.leg_qpos_idx) + self._get_joint_pos(self.leg_left_qpos_idx))
		self.arm_angle = self._get_joint_pos(self.arm_qpos_idx)
		self.forearm_angle = self._get_joint_pos(self.forearm_qpos_idx)
		self.leg_vel = self._get_joint_vel(self.leg_qvel_idx)
		self.leg_left_vel = self._get_joint_vel(self.leg_left_qvel_idx)
		self.arm_vel = self._get_joint_vel(self.arm_qvel_idx)
		self.forearm_vel = self._get_joint_vel(self.forearm_qvel_idx)

	def _has_floor_contact(self, geom_ids: set[int]) -> bool:
		if not geom_ids:
			return False
		for i in range(int(self.data.ncon)):
			c = self.data.contact[i]
			g1 = int(c.geom1)
			g2 = int(c.geom2)
			if ((g1 in geom_ids) and (g2 == self.floor_geom_id)) or ((g2 in geom_ids) and (g1 == self.floor_geom_id)):
				return True
		return False

	def _get_obs(self) -> np.ndarray:
		arms_contact = 1.0 if (self._has_floor_contact(self.arm_right_geom_ids) or self._has_floor_contact(self.arm_left_geom_ids)) else 0.0
		head_contact = 1.0 if self._has_floor_contact({self.head_geom_id}) else 0.0
		torso_contact = 1.0 if self._has_floor_contact(self.torso_geom_ids) else 0.0

		obs = np.array(
			[
				self.com_z,
				self.knees_angle,
				self.arm_angle,
				self.forearm_angle,
				arms_contact,
				head_contact,
				torso_contact,
				self.leg_vel,
				self.leg_left_vel,
				self.arm_vel,
				self.forearm_vel,
			],
			dtype=np.float32,
		)
		return np.clip(obs, -self.obs_clip, self.obs_clip)

	def reset(self) -> np.ndarray:
		mujoco.mj_resetData(self.model, self.data)

		self.data.qpos[self.rootz_qpos_idx] += 0.05
		self.data.qvel[self.rootx_qvel_idx] = 0.0
		self.data.qvel[self.rootz_qvel_idx] = 0.0
		self.data.qvel[self.rooty_qvel_idx] = 0.0
		self.data.ctrl[:] = 0.0

		# Fixed initial push force (uniform)
		self.push_force = float(self.push_force_choices[0])
		self.push_steps_left = self.push_steps
		self.data.qvel[self.rootx_qvel_idx] += self.push_force * (self.dt * self.push_kick_scale)

		# Set deterministic initial arm poses (including left side if present)
		try:
			# primary/right arm
			j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint")
			if j_id >= 0:
				qidx = int(self.model.jnt_qposadr[j_id])
				self.data.qpos[qidx] = 0.129
			# left arm (if exists)
			j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_left_joint")
			if j_id >= 0:
				qidx = int(self.model.jnt_qposadr[j_id])
				self.data.qpos[qidx] = 0.129
			# forearm
			j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "forearm_joint")
			if j_id >= 0:
				qidx = int(self.model.jnt_qposadr[j_id])
				self.data.qpos[qidx] = 6.0
				# also zero velocity for that joint
				dofidx = int(self.model.jnt_dofadr[j_id])
				self.data.qvel[dofidx] = 0.0
			# left forearm (if exists)
			j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "forearm_left_joint")
			if j_id >= 0:
				qidx = int(self.model.jnt_qposadr[j_id])
				self.data.qpos[qidx] = 6.0
		except Exception:
			pass

		# Ensure actuators target the same initial pose so position actuators reflect it immediately
		try:
			aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_arm")
			if aid >= 0:
				self.data.ctrl[aid] = 0.129
			aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_forearm")
			if aid >= 0:
				self.data.ctrl[aid] = 6.0
		except Exception:
			pass

		mujoco.mj_forward(self.model, self.data)

		self.step_count = 0
		self.done = False
		self.last_success = False
		self.prev_action[:] = 0.0

		self.t_arms_l = float("inf")
		self.t_arms_r = float("inf")
		self.t_knees = float("inf")
		self.t_head = float("inf")
		self.t_torso = float("inf")
		self.t_waist = float("inf")

		self._refresh_state()
		return self._get_obs()

	def _brace_reward(self, success: bool) -> tuple[float, dict[str, float]]:
		# support multiple reward versions; default to original behaviour
		reward_ver = getattr(self, "reward_ver", "orig")

		if reward_ver != "v1":
			# original reward (copied from goal-conditioned implementation)
			t_arms_min = min(self.t_arms_l, self.t_arms_r)
			t_head_arm_diff = self.t_head - t_arms_min
			t_torso_arm_diff = self.t_torso - t_arms_min
			t_waist_arm_diff = self.t_waist - t_arms_min

			if np.isinf(t_arms_min):
				r_arm_first = 0.0
			else:
				r_arm_first = float(np.tanh((t_head_arm_diff + t_torso_arm_diff + t_waist_arm_diff) / 3.0))

			if np.isinf(t_arms_min):
				r_arm_sync = 0.0
			else:
				arm_sync_error = abs(self.t_arms_l - self.t_arms_r)
				arm_sync_error = 1.0 if np.isinf(arm_sync_error) else arm_sync_error
				r_arm_sync = 1.0 - float(np.tanh(arm_sync_error))

			if np.isinf(self.t_knees):
				r_knee_timing = 0.0
			else:
				t_arm_knee_diff = abs(self.t_knees - t_arms_min)
				r_knee_timing = 1.0 - float(np.tanh(t_arm_knee_diff / 0.2))

			c_impact = max(0.0, 0.15 - self.head_z) * 5.0
			reward = r_arm_first + 0.8 * r_arm_sync + r_knee_timing - c_impact
			if success:
				reward += 100.0

			return reward, {
				"r_arm_first": r_arm_first,
				"r_arm_sync": r_arm_sync,
				"r_knee_timing": r_knee_timing,
				"c_head_impact": c_impact,
			}

		# v1: R_BRACE = w_r0*r_arms_first - w_c0*c_head - w_c1*c_jitter
		# constants per user request
		w_r0 = 10.0
		w_c0 = 10.0
		w_c1 = 0.1

		# r_arms_first: +1 if arms contact before torso/head, else -1
		t_arms_min = min(self.t_arms_l, self.t_arms_r)
		if np.isinf(t_arms_min) or np.isinf(self.t_head) or np.isinf(self.t_torso):
			r_arms_first = -1.0
		else:
			if (t_arms_min < self.t_head) and (t_arms_min < self.t_torso):
				r_arms_first = 1.0
			else:
				r_arms_first = -1.0

		# c_head: 1 if head touches ground (head_z < 0.15), else 0
		c_head = 1.0 if (self.head_z < 0.15) else 0.0

		# c_jitter: count sign-flip 'jitters' between previous and current action per actuator
		# threshold for considering an actuator movement significant
		jitter_thr = 0.5
		prev = np.asarray(self.prev_action, dtype=np.float32)
		curr = np.asarray(getattr(self, "_current_action_for_reward", np.zeros_like(prev)), dtype=np.float32)
		# sign change where both magnitudes exceed threshold
		sign_prev = np.sign(prev)
		sign_curr = np.sign(curr)
		mask = (np.abs(prev) >= jitter_thr) & (np.abs(curr) >= jitter_thr) & (sign_prev != sign_curr)
		c_jitter = int(np.sum(mask))

		reward = float(w_r0 * r_arms_first - w_c0 * c_head - w_c1 * c_jitter)

		return reward, {"r_arms_first": r_arms_first, "c_head": c_head, "c_jitter": float(c_jitter)}

	def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
		# Accept normalized actions in [-1,1] (preferred) or direct ctrl units.
		action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0)
		# Map normalized actions to ctrl units using ctrl ranges
		ctrls = self.map_action_to_ctrl(action)
		# Write to mj data ctrl
		for i, aid in enumerate(self.actuator_ids):
			self.data.ctrl[aid] = float(np.clip(float(ctrls[i]), float(self.model.actuator_ctrlrange[aid, 0]), float(self.model.actuator_ctrlrange[aid, 1])))

		alive = not self.done
		self.data.xfrc_applied[:] = 0.0
		if alive and self.push_steps_left > 0:
			self.data.xfrc_applied[self.head_body_id, 0] = self.push_force
			self.data.qvel[self.rootx_qvel_idx] += self.push_force * (self.dt * self.push_kick_scale)

		for _ in range(5):
			mujoco.mj_step(self.model, self.data)

		self._refresh_state()

		t = self.step_count * self.dt
		arms_contact_l = self._has_floor_contact(self.arm_left_geom_ids)
		arms_contact_r = self._has_floor_contact(self.arm_right_geom_ids)
		knees_contact = self._has_floor_contact(self.knee_left_geom_ids) or self._has_floor_contact(self.knee_right_geom_ids)
		head_contact = self._has_floor_contact({self.head_geom_id})
		torso_contact = self._has_floor_contact(self.torso_geom_ids)
		waist_contact = self._has_floor_contact(self.waist_geom_ids)

		if arms_contact_l and np.isinf(self.t_arms_l):
			self.t_arms_l = t
		if arms_contact_r and np.isinf(self.t_arms_r):
			self.t_arms_r = t
		if knees_contact and np.isinf(self.t_knees):
			self.t_knees = t
		if head_contact and np.isinf(self.t_head):
			self.t_head = t
		if torso_contact and np.isinf(self.t_torso):
			self.t_torso = t
		if waist_contact and np.isinf(self.t_waist):
			self.t_waist = t

		success = (
			(self.t_arms_l < self.t_head)
			and (self.t_arms_r < self.t_head)
			and (self.t_arms_l < self.t_torso)
			and (self.t_arms_r < self.t_torso)
			and (self.t_arms_l < self.t_waist)
			and (self.t_arms_r < self.t_waist)
			and (abs(self.t_arms_l - self.t_arms_r) <= 0.1)
		)

		# expose current action for reward calculations (e.g., jitter)
		self._current_action_for_reward = action.copy()
		reward, reward_parts = self._brace_reward(success)
		reward = float(np.nan_to_num(reward, nan=0.0, posinf=self.reward_clip, neginf=-self.reward_clip))
		reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

		stopped = (abs(self.vx) < 0.02) and (abs(self.omega) < 0.02) and (self.com_z < 0.20)
		timeout = self.step_count >= (self.episode_length - 1)
		invalid_state = (not np.isfinite(self.data.qpos).all()) or (not np.isfinite(self.data.qvel).all())

		self.last_success = bool(success)
		self.done = bool(success or stopped or timeout or invalid_state)
		self.step_count += 1
		self.push_steps_left = max(0, self.push_steps_left - 1)
		self.prev_action = action.copy()

		obs = self._get_obs()
		info = {
			"success": self.last_success,
			"reward_brace": reward,
			**reward_parts,
		}
		return obs, reward, self.done, info

	def get_success(self) -> bool:
		return self.last_success


class PPONetwork(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim=64):
		super().__init__()
		self.shared = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
		)
		self.policy_mean = nn.Linear(hidden_dim, action_dim)
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

		mean, value = self.forward(obs)
		if deterministic:
			action = mean
		else:
			std = self.policy_logstd.exp()
			action = mean + std * torch.randn_like(mean)
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

		mean, _ = self.network(obs)
		old_log_probs = self._log_prob(mean, self.network.policy_logstd, actions).sum(dim=-1).detach()

		for _ in range(4):
			mean, values = self.network(obs)
			values = values.squeeze()
			new_log_probs = self._log_prob(mean, self.network.policy_logstd, actions).sum(dim=-1)

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
		std = logstd.exp()
		var = std**2
		return -((action - mean) ** 2) / (2 * var) - logstd - 0.5 * np.log(2 * np.pi)


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
	parser = argparse.ArgumentParser(description="Train brace-only PPO (single env).")
	parser.add_argument("--run_label", required=True, help="Label for this training run; used to name checkpoint folder")
	parser.add_argument("--reward_ver", choices=["orig", "v1"], default="orig", help="Which brace reward version to use")
	args = parser.parse_args()

	run_dir = CHECKPOINT_DIR / args.run_label
	run_dir.mkdir(parents=True, exist_ok=True)

	env = HumanoidEnv()
	# set reward version on env for use inside _brace_reward
	env.reward_ver = args.reward_ver
	agent = PPO(env.obs_dim, env.action_dim, entropy_coef=0.01)
	plot = LivePlot("Brace Only PPO Training Progress")

	max_iters = 1000
	steps_per_iter = 2048
	entropy_decay = 0.995
	min_entropy = 0.001

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

				next_obs, reward, done, _ = env.step(action)

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
				best_model_path = run_dir / "best_brace_model.pth"
				torch.save(agent.network.state_dict(), best_model_path)

			if (iteration + 1) % 10 == 0:
				checkpoint_path = run_dir / f"brace_model_iter_{iteration + 1}.pth"
				torch.save(agent.network.state_dict(), checkpoint_path)

			max_win_rate = max(max_win_rate, latest_success_rate)
			plot.update(iteration, mean_reward, latest_success_rate, list(iter_rewards))

			prefix = "\x1b[1;32m[BRACE PPO]\x1b[0m"
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

			if latest_success_rate > 0.95:
				print(f"\n\nTarget reached! Success rate: {latest_success_rate:.2%}")
				break

		best_model_path = run_dir / "best_brace_model.pth"
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
