# brace_v1

from __future__ import annotations

import os
os.environ['MUJOCO_GL'] = 'osmesa'
import sys
from collections import deque
from pathlib import Path
import random
import csv
from datetime import datetime

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from envs.op3_brace import Op3BraceEnv
from reward_component_tracker import RewardComponentTracker

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


class TrainingLogger:
    """Logs training metrics to CSV files for analysis."""
    
    def __init__(self, log_dir: Path, run_label: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main training log
        self.main_log_path = log_dir / f"{run_label}_training_log.csv"
        self.main_fieldnames = [
            "iteration", "total_steps", "episodes_completed", 
            "mean_reward", "max_mean_reward", "latest_iter_mean_reward",
            "success_rate", "max_win_rate", "consecutive_successes", "max_consecutive_successes",
            "entropy_coef", "learning_rate", "replay_buffer_size", "entropy_adjustment"
        ]
        
        with open(self.main_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.main_fieldnames)
            writer.writeheader()
        
        # Episode-level log
        self.episode_log_path = log_dir / f"{run_label}_episode_log.csv"
        self.episode_fieldnames = [
            "iteration", "episode_idx", "episode_reward", "success",
            "consecutive_successes", "total_steps_so_far"
        ]
        
        with open(self.episode_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.episode_fieldnames)
            writer.writeheader()
        
        # Component log
        self.component_log_path = log_dir / f"{run_label}_components_log.csv"
        
        # Entropy adjustment log
        self.entropy_log_path = log_dir / f"{run_label}_entropy_history.csv"
        with open(self.entropy_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "success_rate", "entropy_coef", "adjustment", "target_entropy", "curriculum_scale"])
        
        self.episode_counter = 0
    
    def log_iteration(self, iteration: int, total_steps: int, episodes_completed: int,
                     mean_reward: float, max_mean_reward: float, latest_iter_mean_reward: float,
                     success_rate: float, max_win_rate: float, consec_success: int, max_consec: int,
                     entropy_coef: float, lr: float, replay_buffer_size: int, entropy_adjustment: str):
        with open(self.main_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.main_fieldnames)
            writer.writerow({
                "iteration": iteration,
                "total_steps": total_steps,
                "episodes_completed": episodes_completed,
                "mean_reward": f"{mean_reward:.4f}",
                "max_mean_reward": f"{max_mean_reward:.4f}",
                "latest_iter_mean_reward": f"{latest_iter_mean_reward:.4f}",
                "success_rate": f"{success_rate:.4f}",
                "max_win_rate": f"{max_win_rate:.4f}",
                "consecutive_successes": consec_success,
                "max_consecutive_successes": max_consec,
                "entropy_coef": f"{entropy_coef:.6f}",
                "learning_rate": f"{lr:.2e}",
                "replay_buffer_size": replay_buffer_size,
                "entropy_adjustment": entropy_adjustment
            })
    
    def log_episode(self, iteration: int, episode_reward: float, success: bool,
                   consec_success: int, total_steps: int):
        self.episode_counter += 1
        with open(self.episode_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.episode_fieldnames)
            writer.writerow({
                "iteration": iteration,
                "episode_idx": self.episode_counter,
                "episode_reward": f"{episode_reward:.4f}",
                "success": 1 if success else 0,
                "consecutive_successes": consec_success,
                "total_steps_so_far": total_steps
            })
    
    def log_components(self, iteration: int, component_summary: dict):
        file_exists = self.component_log_path.exists()
        with open(self.component_log_path, 'a', newline='') as f:
            component_summary["iteration"] = iteration
            fieldnames = ["iteration"] + [k for k in component_summary.keys() if k != "iteration"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(component_summary)
    
    def log_entropy(self, iteration: int, success_rate: float, entropy_coef: float, 
                   adjustment: str, target_entropy: float, curriculum_scale: float):
        with open(self.entropy_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, f"{success_rate:.4f}", f"{entropy_coef:.6f}", 
                           adjustment, f"{target_entropy:.6f}", f"{curriculum_scale:.3f}"])


class RunningNorm:
    """Running mean and variance normalization for observations."""
    
    def __init__(self, shape: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4
        self.clip = clip
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation using current statistics."""
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip, self.clip
        )


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for PPO (stores trajectories)."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def add(self, trajectory: dict, td_error: float):
        """Add a trajectory with priority based on TD error."""
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = trajectory
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample a batch of trajectories based on priorities."""
        if len(self.buffer) == 0:
            return [], [], []
        
        probs = np.array(self.priorities) / np.sum(self.priorities)
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs)
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        samples = [self.buffer[i] for i in indices]
        return samples, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled transitions."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)


class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

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


class PPOWithReplay:
    """PPO with Prioritized Experience Replay (PPO + PER) and adaptive entropy."""
    
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.995, gae_lambda=0.95,
                 eps_clip=0.2, entropy_coef=0.02, value_coef=0.5, 
                 replay_capacity=5000, replay_batch_size=512, replay_epochs=2):
        self.network = PPONetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Replay buffer settings
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity)
        self.replay_batch_size = replay_batch_size
        self.replay_epochs = replay_epochs
        
        # Track entropy history
        self.entropy_history = []
    
    def get_current_lr(self):
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]
    
    def update_entropy_adaptive(self, success_rate: float, iteration: int):
        """
        Adaptively adjust entropy coefficient (MODIFIES IN PLACE).
        
        This function directly modifies self.entropy_coef based on:
        - Current success rate
        - Training iteration (curriculum)
        
        Returns:
            tuple: (adjustment_type, new_entropy_value)
        """
        # Base entropy bounds
        max_entropy = 0.05
        min_entropy = 0.0005
        
        # Iteration-based curriculum scaling
        if iteration < 200:
            curriculum_scale = 1.2  # Early: more exploration
        elif iteration < 500:
            curriculum_scale = 1.0  # Mid: normal
        elif iteration < 800:
            curriculum_scale = 0.8  # Late: less exploration
        else:
            curriculum_scale = 0.6  # Final: fine-tuning
        
        # Determine target entropy based on success rate
        if success_rate < 0.3:
            base_target = max_entropy
            adjustment = "EXPLORE+"
        elif success_rate < 0.5:
            base_target = 0.025
            adjustment = "EXPLORE"
        elif success_rate < 0.7:
            base_target = 0.015
            adjustment = "EXPLOIT"
        elif success_rate < 0.85:
            base_target = 0.008
            adjustment = "REFINE"
        elif success_rate < 0.95:
            base_target = 0.003
            adjustment = "TUNE"
        else:
            # Success rate >= 0.95
            # CAUTIOUS: If we're still early in training (< 1000 iterations), 
            # this success is likely a fluke - maintain higher entropy
            if iteration < 1000:
                base_target = 0.01  # Keep moderate exploration
                adjustment = "CAUTIOUS"
            else:
                base_target = min_entropy
                adjustment = "FINE_TUNE"
        
        # Apply curriculum scaling
        target_entropy = base_target * curriculum_scale
        target_entropy = np.clip(target_entropy, min_entropy, max_entropy)
        
        # Smooth transition to target entropy (modifies IN PLACE)
        if iteration < 200:
            smoothing = 0.90  # Fast adaptation early
        elif iteration < 500:
            smoothing = 0.95  # Moderate mid-training
        else:
            smoothing = 0.98  # Stable late training
        
        # DIRECTLY MODIFY self.entropy_coef
        self.entropy_coef = self.entropy_coef * smoothing + target_entropy * (1 - smoothing)
        self.entropy_coef = np.clip(self.entropy_coef, min_entropy, max_entropy)
        
        # Store history
        self.entropy_history.append({
            'iteration': iteration,
            'entropy_coef': self.entropy_coef,
            'success_rate': success_rate,
            'adjustment': adjustment,
            'target_entropy': target_entropy,
            'curriculum_scale': curriculum_scale
        })
        
        return adjustment, self.entropy_coef
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            next_value = values[t]
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return returns, advantages
    
    def update_from_trajectory(self, trajectories):
        """Update policy using collected trajectories (on-policy update)."""
        obs = torch.FloatTensor(np.array(trajectories["obs"]))
        actions = torch.FloatTensor(np.array(trajectories["actions"]))
        rewards = torch.FloatTensor(np.array(trajectories["rewards"]))
        dones = torch.FloatTensor(np.array(trajectories["dones"]))
        
        with torch.no_grad():
            _, values = self.network(obs)
            values = values.squeeze().numpy()
        
        returns, advantages = self.compute_gae(rewards.numpy(), values, dones.numpy())
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store trajectory in replay buffer with mean TD error as priority
        mean_td_error = float(np.mean(np.abs(advantages.numpy())))
        self.replay_buffer.add(trajectories.copy(), mean_td_error)
        
        # Perform PPO update
        self._update_policy(obs, actions, returns, advantages)
    
    def update_from_replay(self):
        """Update policy using replay buffer (off-policy correction with importance sampling)."""
        if len(self.replay_buffer) < self.replay_batch_size:
            return
        
        for _ in range(self.replay_epochs):
            samples, weights, indices = self.replay_buffer.sample(self.replay_batch_size)
            if not samples:
                continue
            
            # Aggregate samples
            all_obs = []
            all_actions = []
            all_returns = []
            all_advantages = []
            
            for sample in samples:
                all_obs.extend(sample["obs"])
                all_actions.extend(sample["actions"])
                all_returns.extend(sample.get("returns", []))
                all_advantages.extend(sample.get("advantages", []))
            
            if not all_returns:
                continue
            
            obs = torch.FloatTensor(np.array(all_obs))
            actions = torch.FloatTensor(np.array(all_actions))
            returns = torch.FloatTensor(np.array(all_returns))
            advantages = torch.FloatTensor(np.array(all_advantages))
            weights_tensor = torch.FloatTensor(weights[:len(obs)])
            
            # Normalize advantages for replay buffer samples
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            self._update_policy(obs, actions, returns, advantages, weights_tensor)
    
    def _update_policy(self, obs, actions, returns, advantages, sample_weights=None):
        """Core PPO policy update with optional importance sampling weights."""
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
            policy_loss = -torch.min(surr1, surr2)
            
            if sample_weights is not None:
                policy_loss = (policy_loss * sample_weights).mean()
            else:
                policy_loss = policy_loss.mean()
            
            value_loss = nn.MSELoss(reduction='none')(values, returns)
            if sample_weights is not None:
                value_loss = (value_loss * sample_weights).mean()
            else:
                value_loss = value_loss.mean()
            
            entropy = (0.5 * (1 + np.log(2 * np.pi)) + self.network.policy_logstd).sum()
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.scheduler.step()
    
    def _log_prob(self, mean, logstd, action):
        action_clamped = torch.clamp(action, -0.999999, 0.999999)
        if mean.dim() == 1:
            mean_exp = mean.unsqueeze(0).expand(action_clamped.size(0), -1)
        else:
            mean_exp = mean

        if logstd.dim() == 1:
            logstd_exp = logstd.unsqueeze(0).expand(action_clamped.size(0), -1)
        else:
            logstd_exp = logstd

        std_exp = logstd_exp.exp()
        raw_action = 0.5 * torch.log((1.0 + action_clamped) / (1.0 - action_clamped))
        var = std_exp ** 2
        log_prob_element = -((raw_action - mean_exp) ** 2) / (2 * var) - logstd_exp - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob_element.sum(dim=-1)
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
        self.success_line.set_data(self.iteration_history, self.success_history)

        if len(self.iteration_history) > 0:
            self.ax1.set_xlim(0, max(50, iteration + 10))
            self.ax2.set_xlim(0, max(50, iteration + 10))
            if len(self.reward_history) > 0:
                y_min = min(self.reward_history)
                y_max = max(self.reward_history)
                margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
                self.ax1.set_ylim(y_min - margin, y_max + margin)

        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description="Train OP3 Full-Body Brace PPO with PER (single env, 12 DOF).")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file to continue training from")
    parser.add_argument("--run_label", required=True, help="Label for this training run; used to name checkpoint folder")
    args = parser.parse_args()

    run_dir = CHECKPOINT_DIR / args.run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger = TrainingLogger(run_dir, args.run_label)

    env = Op3BraceEnv()
    agent = PPOWithReplay(
        obs_dim=env.obs_dim, 
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        eps_clip=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        replay_capacity=5000,
        replay_batch_size=512,
        replay_epochs=2
    )
    
    obs_normalizer = RunningNorm(shape=env.obs_dim)
    
    # NO WEIGHTS PASSED TO TRACKER - weights are ONLY in env.py
    reward_tracker = RewardComponentTracker(max_history=50)

    if args.checkpoint:
        agent.network.load_state_dict(torch.load(args.checkpoint))
    plot = LivePlot("OP3 Full-Body Brace PPO Training Progress")

    max_iters = 1000
    steps_per_iter = 2048
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

    # Warm-up the normalizer
    print("Warming up observation normalizer...")
    warmup_obs = []
    for _ in range(100):
        obs = env.reset()
        warmup_obs.append(obs)
        for _ in range(50):
            action = np.random.uniform(-1, 1, env.action_dim)
            obs, _, done, _ = env.step(action)
            warmup_obs.append(obs)
            if done:
                obs = env.reset()
    obs_normalizer.update(np.array(warmup_obs))
    print("Normalizer warm-up complete.")

    try:
        for iteration in range(max_iters):
            trajectories = {"obs": [], "actions": [], "rewards": [], "dones": []}
            iter_rewards = []
            iter_successes = []

            obs = env.reset()
            obs = obs_normalizer.normalize(obs)
            episode_reward = 0.0

            for step in range(steps_per_iter):
                action, _ = agent.network.get_action(obs)
                action = action.detach().numpy()

                next_obs_raw, reward, done, info = env.step(action)
                next_obs = obs_normalizer.normalize(next_obs_raw)

                trajectories["obs"].append(obs)
                trajectories["actions"].append(action)
                trajectories["rewards"].append(reward)
                trajectories["dones"].append(done)

                episode_reward += reward
                obs = next_obs

                obs_normalizer.update(next_obs_raw.reshape(1, -1))

                # Record RAW reward components from env (NO weights applied)
                component_dict = {k: float(v) for k, v in info.items() 
                                if k not in ["success", "head_contact", "timeout", "invalid_state"] 
                                and isinstance(v, (int, float))
                                and not k.endswith("_weighted")}  # Skip weighted versions
                if component_dict:
                    reward_tracker.record_episode(component_dict)

                if done:
                    episode_rewards.append(episode_reward)
                    iter_rewards.append(episode_reward)
                    success = env.get_success()
                    iter_successes.append(success)
                    success_history.append(success)

                    logger.log_episode(iteration, episode_reward, success, consec_success, total_steps + step)

                    if success:
                        consec_success += 1
                        max_consec = max(max_consec, consec_success)
                    else:
                        consec_success = 0

                    obs_raw = env.reset()
                    obs = obs_normalizer.normalize(obs_raw)
                    episode_reward = 0.0
                    done_iter += 1

            agent.update_from_trajectory(trajectories)
            agent.update_from_replay()
            total_steps += steps_per_iter

            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            latest_success_rate = np.mean(success_history) if success_history else 0.0
            latest_iter_mean_reward = float(np.mean(iter_rewards)) if iter_rewards else 0.0

            entropy_adjustment, new_entropy = agent.update_entropy_adaptive(latest_success_rate, iteration)
            
            last_entropy = agent.entropy_history[-1] if agent.entropy_history else {}
            logger.log_entropy(
                iteration, latest_success_rate, agent.entropy_coef,
                entropy_adjustment,
                last_entropy.get('target_entropy', 0.0),
                last_entropy.get('curriculum_scale', 1.0)
            )

            if mean_reward > max_mean_reward:
                max_mean_reward = mean_reward
                best_model_path = run_dir / "best_op3_brace_model.pth"
                torch.save(agent.network.state_dict(), best_model_path)

            if (iteration + 1) % 10 == 0:
                checkpoint_path = run_dir / f"op3_brace_model_iter_{iteration + 1}.pth"
                torch.save(agent.network.state_dict(), checkpoint_path)

            max_win_rate = max(max_win_rate, latest_success_rate)
            
            reward_tracker.finalize_iteration(iteration)
            
            component_summary = reward_tracker.get_summary()
            if component_summary:
                flat_summary = {}
                for component, stats in component_summary.items():
                    flat_summary[f"{component}_mean"] = stats.get("mean", 0.0)
                    flat_summary[f"{component}_min"] = stats.get("min", 0.0)
                    flat_summary[f"{component}_max"] = stats.get("max", 0.0)
                logger.log_components(iteration, flat_summary)
            
            logger.log_iteration(
                iteration=iteration,
                total_steps=total_steps,
                episodes_completed=done_iter,
                mean_reward=mean_reward,
                max_mean_reward=max_mean_reward,
                latest_iter_mean_reward=latest_iter_mean_reward,
                success_rate=latest_success_rate,
                max_win_rate=max_win_rate,
                consec_success=consec_success,
                max_consec=max_consec,
                entropy_coef=agent.entropy_coef,
                lr=agent.get_current_lr(),
                replay_buffer_size=len(agent.replay_buffer),
                entropy_adjustment=entropy_adjustment
            )
            
            plot.update(iteration, mean_reward, latest_success_rate, list(iter_rewards))

            prefix = "\x1b[1;32m[OP3 FULL-BODY BRACE PPO+PER]\x1b[0m"
            iter_txt = f"{iteration + 1:3d}/{max_iters}"
            eps_txt = f"{len(episode_rewards):3d}"
            steps_txt = f"{total_steps:,}"
            done_iter_txt = f"{done_iter}"
            mean_rew_txt = f"{mean_reward:.3f}"
            max_rew_txt = f"{max_mean_reward:.3f}"
            replay_size_txt = f"{len(agent.replay_buffer)}"
            entropy_status = f"{agent.entropy_coef:.6f} [{entropy_adjustment}]"

            if iteration > 0:
                print("\x1b[9A", end="")

            print(
                f"\r\x1b[K{prefix}\n"
                f"\x1b[K{kv('iter', iter_txt, 'eps', eps_txt)}\n"
                f"\x1b[K{kv('steps', steps_txt, 'done_iter', done_iter_txt)}\n"
                f"\x1b[K{kv('mean_rew', mean_rew_txt, 'max_mean_rew', max_rew_txt)}\n"
                f"\x1b[K{kv('win_rate', f'{latest_success_rate * 100.0:.2f}%', 'max_win_rate', f'{max_win_rate:.2%}')}\n"
                f"\x1b[K{kv('consec_success', f'{consec_success}', 'max_consec', f'{max_consec}')}\n"
                f"\x1b[K{kv('entropy', entropy_status, 'replay_size', replay_size_txt)}\n"
                f"\x1b[K{kv('lr', f'{agent.get_current_lr():.2e}', '', '')}",
                end="",
                flush=True,
            )

            if latest_success_rate >= 0.95 and consec_success >= consecutive_success_threshold:
                print(f"\n\nTarget reached! Success rate: {latest_success_rate:.2%} maintained for {consec_success} iterations")
                break

        best_model_path = run_dir / "best_op3_brace_model.pth"
        print(f"\n\nTraining completed! Best model saved to {best_model_path}")
        print(f"Max mean reward: {max_mean_reward:.3f}, Max win rate: {max_win_rate:.2%}")
        print(f"Last iteration mean episode reward: {latest_iter_mean_reward:.3f}")
        print(f"\nLogs saved to: {run_dir}/")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    finally:
        plot.close()
        training_plot_path = run_dir / "training_progress.png"
        plot.fig.savefig(training_plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved as '{training_plot_path.name}' in {run_dir}")
        
        reward_tracker.close()
        reward_component_plot_path = run_dir / "reward_components.png"
        reward_tracker.save(reward_component_plot_path, title="OP3 Full-Body Brace - RAW Reward Components")
        print(f"Reward component plot saved to {reward_component_plot_path.name}")


if __name__ == "__main__":
    main()