"""Simple render script for Op3ArmBraceEnv v2.

Usage:
    python render.py --checkpoint checkpoints/arm_brace_v2/best_op3_arm_brace_v2_model.pth --output_dir demo_v2
"""
from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import mujoco
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import imageio
from tqdm import tqdm
import traceback

from envs.op3_arm_brace_v2 import Op3ArmBraceEnv

SCRIPT_DIR = Path(__file__).resolve().parent


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
            nn.Tanh(),
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
        mean = self.policy_mean(shared_out)
        value = self.value(shared_out)

        if deterministic:
            action = mean
        else:
            std = self.policy_logstd.exp()
            raw_action = mean + std * torch.randn_like(mean)
            action = torch.tanh(raw_action)
        return action.squeeze(0), value.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Render Op3ArmBraceEnv v2 with a trained checkpoint (three POVs).")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint model file (.pth)")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for videos")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode (overrides env default)")
    parser.add_argument("--limp_scale", type=float, default=0.01, help="Scale to apply to actuator gains/bias on impact (smaller = more limp)")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the single XML scene with named POV cameras
    env = Op3ArmBraceEnv()

    # Load network
    network = PPONetwork(env.obs_dim, env.action_dim)
    network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    network.eval()

    # Setup renderer for the shared model; switch POV via camera name in update_scene()
    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    camera_left = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "pov_left")
    camera_front = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "pov_front")
    camera_top = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "pov_top")

    print(f"Rendering {args.episodes} episode(s) to {output_dir}...")

    total_frames = 0

    for ep in range(args.episodes):
        # Create per-episode video writers to stream frames to disk
        ep_output_dir = output_dir / f"episode_{ep+1}"
        ep_output_dir.mkdir(parents=True, exist_ok=True)

        writers = {
            "left": imageio.get_writer(str(ep_output_dir / "povleft.mp4"), fps=30),
            "front": imageio.get_writer(str(ep_output_dir / "povfront.mp4"), fps=30),
            "top": imageio.get_writer(str(ep_output_dir / "povtop.mp4"), fps=30),
        }

        # If we previously scaled actuator params for a limp episode, restore originals now
        try:
            if hasattr(env, "_orig_actuator_gainprm") and hasattr(env.model, "actuator_gainprm"):
                env.model.actuator_gainprm[:] = env._orig_actuator_gainprm
                print(f"[render.py] Restored original actuator_gainprm before episode {ep+1}")
        except Exception as e:
            print("[render.py] Warning: failed to restore actuator_gainprm:", e)
            traceback.print_exc()
        try:
            if hasattr(env, "_orig_actuator_biasprm") and hasattr(env.model, "actuator_biasprm"):
                env.model.actuator_biasprm[:] = env._orig_actuator_biasprm
                print(f"[render.py] Restored original actuator_biasprm before episode {ep+1}")
        except Exception as e:
            print("[render.py] Warning: failed to restore actuator_biasprm:", e)
            traceback.print_exc()

        obs = env.reset()
        done = False
        step = 0

        # Determine max steps: prefer user arg, then env attribute, then default 500
        if args.max_steps is not None:
            episode_max = int(args.max_steps)
        else:
            episode_max = int(getattr(env, "episode_length", 500))

        pbar = tqdm(total=episode_max, desc=f"Ep {ep+1}", unit="step")
        impacted = False
        while step < episode_max:
            if not impacted:
                # Get deterministic action from policy
                with torch.no_grad():
                    action_tensor, _ = network.get_action(obs, deterministic=True)
                action = action_tensor.cpu().numpy()

                # Step environment once
                obs, _, done, info = env.step(action)

                # If env signaled done (e.g., impact), stop controller and go limp
                if done:
                    impacted = True
                    limp_scale = float(args.limp_scale)
                    # Try scaling actuator gainprm and biasprm to make joints compliant
                    try:
                        if hasattr(env.model, "actuator_gainprm"):
                            if not hasattr(env, "_orig_actuator_gainprm"):
                                env._orig_actuator_gainprm = env.model.actuator_gainprm.copy()
                            env.model.actuator_gainprm[:] = env._orig_actuator_gainprm * limp_scale
                    except Exception as e:
                        print("[render.py] Warning: failed to scale actuator_gainprm:", e)
                        traceback.print_exc()
                    try:
                        if hasattr(env.model, "actuator_biasprm"):
                            if not hasattr(env, "_orig_actuator_biasprm"):
                                env._orig_actuator_biasprm = env.model.actuator_biasprm.copy()
                            env.model.actuator_biasprm[:] = env._orig_actuator_biasprm * limp_scale
                    except Exception as e:
                        print("[render.py] Warning: failed to scale actuator_biasprm:", e)
                        traceback.print_exc()
                    # As an extra safety, apply zero actions once
                    zero_action = np.zeros(env.action_dim, dtype=np.float32)
                    try:
                        arm_ctrl = env.map_action_to_ctrl(zero_action)
                        env._apply_controls(arm_ctrl)
                    except Exception as e:
                        print("[render.py] Warning: failed to apply zero actions via env.map_action_to_ctrl/_apply_controls:", e)
                        traceback.print_exc()
                        # Fallback: directly zero the control buffer
                        try:
                            env.data.ctrl[:] = 0.0
                        except Exception as e2:
                            print("[render.py] Error: failed to zero env.data.ctrl:", e2)
                            traceback.print_exc()

            else:
                # After impact, keep limbs limp by applying zero actions each step
                zero_action = np.zeros(env.action_dim, dtype=np.float32)
                obs, _, done, info = env.step(zero_action)

            # Render and write frames immediately
            renderer.update_scene(env.data, camera=camera_left)
            writers["left"].append_data(renderer.render())

            renderer.update_scene(env.data, camera=camera_front)
            writers["front"].append_data(renderer.render())

            renderer.update_scene(env.data, camera=camera_top)
            writers["top"].append_data(renderer.render())

            total_frames += 1
            step += 1
            pbar.update(1)

        # Close writers to flush files to disk
        for w in writers.values():
            w.close()
        pbar.close()

        print(f"  Episode {ep + 1}/{args.episodes}: {step} steps, videos saved to {ep_output_dir}")

    print(f"\nAll episodes complete. Videos saved under: {output_dir}")


if __name__ == "__main__":
    main()
