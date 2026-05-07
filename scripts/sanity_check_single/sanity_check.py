import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import mujoco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set headless mode
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# Target joint positions
TARGET_POS = np.array([1.0, -1.21, 2.0, -1.29, 5.19])  # waist, leg, shin, arm, forearm

class HumanoidEnv:
    def __init__(self, xml_path="humanoid_2d_half.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get actuator names from XML
        self.actuator_names = ["pos_waist", "pos_leg", "pos_shin", "pos_arm", "pos_forearm"]
        self.actuator_ids = []
        for name in self.actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.actuator_ids.append(aid)
        
        # Get joint names for observation
        self.joint_names = ["waist_joint", "leg_joint", "shin_joint", "arm_joint", "forearm_joint"]
        self.joint_ids = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_ids.append(jid)
        
        self.action_dim = len(self.actuator_ids)  # 5
        # Observation: joint positions (5) + joint velocities (5) = 10
        self.obs_dim = 2 * self.action_dim  # 10
        
        self.prev_action = None
        self.min_distance = float('inf')  # Track minimum distance in episode
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Randomize initial pose slightly
        for jid in self.joint_ids:
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr] += np.random.uniform(-0.1, 0.1)
        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.action_dim)
        self.min_distance = float('inf')
        return self._get_obs()
    
    def _get_obs(self):
        obs = []
        # Current joint positions
        for jid in self.joint_ids:
            qpos_addr = self.model.jnt_qposadr[jid]
            obs.append(self.data.qpos[qpos_addr])
        
        # Current joint velocities
        for jid in self.joint_ids:
            dof_addr = self.model.jnt_dofadr[jid]
            obs.append(self.data.qvel[dof_addr])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        # Apply position control to actuators
        for i, aid in enumerate(self.actuator_ids):
            ctrl_range = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(action[i], ctrl_range[0], ctrl_range[1])
        
        # Simulate 5 steps per action for stability
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = False
        
        self.prev_action = action.copy()
        
        return obs, reward, done, {}
    
    def _compute_reward(self, action):
        # Distance-based reward (primary)
        current_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] 
                               for jid in self.joint_ids])
        distance = np.linalg.norm(current_pos - TARGET_POS)
        
        # Update minimum distance
        self.min_distance = min(self.min_distance, distance)
        
        distance_reward = np.exp(-2.0 * distance)
        
        # Smoothness reward (small bonus for smooth actions)
        if self.prev_action is not None:
            action_diff = np.linalg.norm(action - self.prev_action)
            smoothness_reward = np.exp(-5.0 * action_diff)  # Penalize jerky actions
        else:
            smoothness_reward = 1.0
            
        # Velocity penalty (optional - discourages unnecessary movement)
        velocities = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] 
                              for jid in self.joint_ids])
        velocity_penalty = np.exp(-0.1 * np.linalg.norm(velocities))
        
        # Combined reward
        reward = distance_reward + 0.05 * smoothness_reward + 0.02 * velocity_penalty
        
        return reward
    
    def get_success(self):
        return self.min_distance < 0.7  # Success threshold
    
    def get_min_distance(self):
        return self.min_distance

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
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
            obs = obs.unsqueeze(0)  # Add batch dimension
            
        mean, value = self.forward(obs)
        if deterministic:
            action = mean
        else:
            std = self.policy_logstd.exp()
            action = mean + std * torch.randn_like(mean)
        return action.squeeze(0), value.squeeze(0)

class PPO:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 entropy_coef=0.01, value_coef=0.5):
        self.network = PPONetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def update(self, trajectories):
        obs = torch.FloatTensor(np.array(trajectories['obs']))
        actions = torch.FloatTensor(np.array(trajectories['actions']))
        rewards = torch.FloatTensor(np.array(trajectories['rewards']))
        dones = torch.FloatTensor(np.array(trajectories['dones']))
        
        # Compute returns and advantages
        with torch.no_grad():
            _, values = self.network(obs)
            values = values.squeeze()
            
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        
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
        
        # PPO update
        mean, _ = self.network(obs)
        old_log_probs = self._log_prob(mean, self.network.policy_logstd, actions).sum(dim=-1).detach()
        
        for _ in range(4):  # PPO epochs
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
        var = std ** 2
        return -((action - mean) ** 2) / (2 * var) - logstd - 0.5 * np.log(2 * np.pi)

def kv(label, value, label2="", value2=""):
    """Format key-value pairs for display"""
    s = f"{label}: \x1b[1;36m{value}\x1b[0m"
    if label2:
        s += f"  {label2}: \x1b[1;36m{value2}\x1b[0m"
    return s

class LivePlot:
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Humanoid PPO Training Progress', fontsize=14, fontweight='bold')
        
        # Reward plot
        self.ax1.set_title('Episode Reward')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Mean Reward')
        self.ax1.grid(True, alpha=0.3)
        self.reward_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Mean Reward')
        self.reward_scatter = self.ax1.scatter([], [], c='blue', alpha=0.3, s=10, label='Episode Rewards')
        self.ax1.legend()
        
        # Success rate plot
        self.ax2.set_title('Success Rate')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Success Rate (%)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 100)
        self.success_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='Success Rate')
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
        
        # Update reward line
        self.reward_line.set_data(self.iteration_history, self.reward_history)
        
        # Update scatter plot for episode rewards
        if len(self.episode_rewards) > 0:
            scatter_x = [iteration] * len(self.episode_rewards)
            self.reward_scatter.set_offsets(np.c_[scatter_x, self.episode_rewards])
        
        # Update success rate line
        self.success_line.set_data(self.iteration_history, self.success_history)
        
        # Adjust axes limits
        if len(self.iteration_history) > 0:
            self.ax1.set_xlim(0, max(50, iteration + 10))
            self.ax2.set_xlim(0, max(50, iteration + 10))
            
            if len(self.reward_history) > 0:
                y_min = min(min(self.reward_history), min(self.episode_rewards) if self.episode_rewards else 0)
                y_max = max(max(self.reward_history), max(self.episode_rewards) if self.episode_rewards else 1)
                margin = (y_max - y_min) * 0.1
                self.ax1.set_ylim(max(0, y_min - margin), y_max + margin)
        
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.show(block=True)

def main():
    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = HumanoidEnv()
    agent = PPO(env.obs_dim, env.action_dim, entropy_coef=0.01)
    
    # Initialize live plot
    plot = LivePlot()
    
    # Training parameters
    max_iters = 500
    steps_per_iter = 2048
    entropy_decay = 0.995
    min_entropy = 0.001
    
    # Statistics
    episode_rewards = deque(maxlen=10)
    success_history = deque(maxlen=10)
    min_distance_history = deque(maxlen=10)  # Track min distances
    max_mean_reward = -np.inf
    max_win_rate = 0.0
    consec_success = 0
    max_consec = 0
    best_model_path = os.path.join(checkpoint_dir, "best_humanoid_model.pth")
    
    total_steps = 0
    done_iter = 0
    
    try:
        for iteration in range(max_iters):
            # Collect trajectories
            trajectories = {'obs': [], 'actions': [], 'rewards': [], 'dones': []}
            iter_rewards = []
            iter_successes = []
            iter_min_distances = []
            
            obs = env.reset()
            episode_reward = 0
            
            for step in range(steps_per_iter):
                action, _ = agent.network.get_action(obs)
                action = action.detach().numpy()
                
                next_obs, reward, done, _ = env.step(action)
                
                trajectories['obs'].append(obs)
                trajectories['actions'].append(action)
                trajectories['rewards'].append(reward)
                trajectories['dones'].append(done)
                
                episode_reward += reward
                obs = next_obs
                
                if done or step == steps_per_iter - 1:
                    episode_rewards.append(episode_reward)
                    iter_rewards.append(episode_reward)
                    success = env.get_success()
                    iter_successes.append(success)
                    success_history.append(success)
                    min_dist = env.get_min_distance()
                    iter_min_distances.append(min_dist)
                    min_distance_history.append(min_dist)
                    
                    if success:
                        consec_success += 1
                        max_consec = max(max_consec, consec_success)
                    else:
                        consec_success = 0
                        
                    obs = env.reset()
                    episode_reward = 0
                    done_iter += 1
            
            # Update agent
            agent.update(trajectories)
            total_steps += steps_per_iter
            
            # Compute statistics
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0
            latest_success_rate = np.mean(success_history) if success_history else 0
            current_success_rate = np.mean(iter_successes) if iter_successes else 0
            mean_min_distance = np.mean(iter_min_distances) if iter_min_distances else float('inf')
            best_min_distance = min(min_distance_history) if min_distance_history else float('inf')
            
            # Decay entropy
            agent.entropy_coef = max(agent.entropy_coef * entropy_decay, min_entropy)
            
            # Save best model (silently)
            if mean_reward > max_mean_reward:
                max_mean_reward = mean_reward
                torch.save(agent.network.state_dict(), best_model_path)
            
            # Save checkpoint every 10 iterations (silently)
            if (iteration + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"humanoid_model_iter_{iteration+1}.pth")
                torch.save(agent.network.state_dict(), checkpoint_path)

            max_win_rate = max(max_win_rate, latest_success_rate)
            
            # Update plot
            plot.update(iteration, mean_reward, latest_success_rate, list(iter_rewards))
            
            # Display progress - in-place update
            prefix = "\x1b[1;32m[HUMANOID PPO]\x1b[0m"
            
            iter_txt = f"{iteration+1:3d}/{max_iters}"
            eps_txt = f"{len(episode_rewards):3d}"
            steps_txt = f"{total_steps:,}"
            done_iter_txt = f"{done_iter}"
            mean_rew_txt = f"{mean_reward:.3f}"
            max_rew_compact = f"{max_mean_reward:.3f}"
            min_dpos_txt = f"{mean_min_distance:.4f}"
            best_min_dpos_txt = f"{best_min_distance:.4f}"
            
            # Clear and redraw everything
            if iteration > 0:
                # Move up 7 lines (the data lines, not the prefix since it's part of first line)
                print("\x1b[7A", end="")
            
            # Print everything, clearing each line
            print(
                f"\r\x1b[K{prefix}\n"
                f"\x1b[K{kv('iter', iter_txt, 'eps', eps_txt)}\n"
                f"\x1b[K{kv('steps', steps_txt, 'done_iter', done_iter_txt)}\n"
                f"\x1b[K{kv('mean_rew', mean_rew_txt, 'max_mean_rew', max_rew_compact)}\n"
                f"\x1b[K{kv('win_rate', f'{latest_success_rate * 100.0:.2f}%', 'max_win_rate', f'{max_win_rate:.2%}')}\n"
                f"\x1b[K{kv('min_dpos', min_dpos_txt, 'best_min_dpos', best_min_dpos_txt)}\n"
                f"\x1b[K{kv('consec_success', f'{consec_success}', 'max_consec', f'{max_consec}')}\n"
                f"\x1b[K{kv('entropy_coef', f'{agent.entropy_coef:.6f}', '', '')}",
                end="",
                flush=True,
            )
            
            # Early stopping if target reached
            if latest_success_rate > 0.95:
                print(f"\n\nTarget reached! Success rate: {latest_success_rate:.2%}")
                break
        
        print(f"\n\nTraining completed! Best model saved to {best_model_path}")
        print(f"Max mean reward: {max_mean_reward:.3f}, Max win rate: {max_win_rate:.2%}")
        print(f"Best min distance: {best_min_distance:.4f}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    finally:
        plot.close()
        # Save final plot
        plot.fig.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'training_progress.png'")

if __name__ == "__main__":
    main()