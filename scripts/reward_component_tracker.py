"""Utility for tracking and visualizing per-component rewards across iterations."""

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class RewardComponentTracker:
    """Tracks reward components per episode and per iteration.
    
    NOTE: This tracker does NOT apply any weights. It logs raw component values
    directly from the environment. Weights are ONLY applied in env.py.
    """

    def __init__(self, max_history: int = 50):
        """
        Args:
            max_history: Max number of iterations to keep in rolling window.
        """
        self.max_history = max_history
        
        # History storage
        self.iteration_history = []
        self.component_lines = {}  # {component_name: line_object}
        self.component_data = {}   # {component_name: [values]}
        
        # Create persistent live plot figure with TWO subplots (rewards and penalties)
        plt.ion()
        self.fig, (self.ax_rewards, self.ax_penalties) = plt.subplots(2, 1, figsize=(14, 10))
        self.fig.suptitle("Reward Components (RAW Values - No Weights Applied)", fontsize=14, fontweight="bold")
        
        # Rewards subplot
        self.ax_rewards.set_title("Rewards (Positive Components)", fontsize=12, fontweight="bold")
        self.ax_rewards.set_xlabel("Iteration", fontweight="bold")
        self.ax_rewards.set_ylabel("Raw Value", fontweight="bold")
        self.ax_rewards.grid(True, alpha=0.3)
        self.ax_rewards.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        
        # Penalties subplot
        self.ax_penalties.set_title("Penalties/Costs (Negative Components)", fontsize=12, fontweight="bold")
        self.ax_penalties.set_xlabel("Iteration", fontweight="bold")
        self.ax_penalties.set_ylabel("Raw Penalty Value", fontweight="bold")
        self.ax_penalties.grid(True, alpha=0.3)
        self.ax_penalties.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plt.pause(0.1)
        
        # Episode tracking
        self.current_iter_components: Dict[str, List[float]] = {}

    def record_episode(self, reward_components: Dict[str, float]):
        """Record reward components from a single episode.
        
        Args:
            reward_components: Dict mapping component names to their RAW values from env.
        """
        for name, value in reward_components.items():
            if name not in self.current_iter_components:
                self.current_iter_components[name] = []
            self.current_iter_components[name].append(float(value))

    def finalize_iteration(self, iteration: int):
        """Finalize the current iteration: compute mean, update plot."""
        self.iteration_history.append(iteration)
        
        # Compute means for each component (NO WEIGHTS APPLIED)
        for name, values in self.current_iter_components.items():
            if name not in self.component_data:
                self.component_data[name] = []
            mean_val = float(np.mean(values)) if values else 0.0
            self.component_data[name].append(mean_val)
        
        self.current_iter_components.clear()
        
        # Update the live plot EVERY iteration
        self.update()
    
    def update(self):
        """Update the live plot, separating positive rewards from penalties."""
        if not self.component_data:
            plt.pause(0.01)
            return
        
        self.ax_rewards.clear()
        self.ax_penalties.clear()
        
        # Separate components based on name convention
        rewards_data = {}   # Positive components (r_*, success_*)
        penalties_data = {} # Negative components (c_*, torso_*)
        
        for component_name, values in self.component_data.items():
            # Components starting with 'r_' or 'success' are positive rewards
            if component_name.startswith('r_') or component_name.startswith('success'):
                rewards_data[component_name] = values
            else:
                # Components starting with 'c_', 'torso_', etc. are penalties
                penalties_data[component_name] = values
        
        # Plot rewards
        if rewards_data:
            reward_colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(rewards_data))))
            if len(rewards_data) > 10:
                extra_colors = plt.cm.tab20(np.linspace(0, 1, len(rewards_data) - 10))
                reward_colors = np.vstack([reward_colors, extra_colors])
            
            for idx, (component_name, values) in enumerate(rewards_data.items()):
                self.ax_rewards.plot(self.iteration_history, values, color=reward_colors[idx],
                                    linewidth=2, label=component_name, alpha=0.9)
            
            self.ax_rewards.set_xlabel("Iteration", fontweight="bold")
            self.ax_rewards.set_ylabel("Raw Value", fontweight="bold")
            self.ax_rewards.grid(True, alpha=0.3)
            self.ax_rewards.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
            self.ax_rewards.legend(loc="best", fontsize=9, framealpha=0.95)
            self.ax_rewards.set_title("Rewards (Positive Components - RAW Values)", fontsize=12, fontweight="bold")
            
            if len(self.iteration_history) > 0:
                self.ax_rewards.set_xlim(0, max(50, self.iteration_history[-1] + 10))
        
        # Plot penalties (shown as positive values for readability)
        if penalties_data:
            penalty_colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(penalties_data))))
            if len(penalties_data) > 10:
                extra_colors = plt.cm.tab20(np.linspace(0, 1, len(penalties_data) - 10))
                penalty_colors = np.vstack([penalty_colors, extra_colors])
            
            for idx, (component_name, values) in enumerate(penalties_data.items()):
                # Show raw penalty values (positive = bad)
                self.ax_penalties.plot(self.iteration_history, values, color=penalty_colors[idx],
                                      linewidth=2, label=component_name, alpha=0.9)
            
            self.ax_penalties.set_xlabel("Iteration", fontweight="bold")
            self.ax_penalties.set_ylabel("Raw Penalty Value", fontweight="bold")
            self.ax_penalties.grid(True, alpha=0.3)
            self.ax_penalties.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
            self.ax_penalties.legend(loc="best", fontsize=9, framealpha=0.95)
            self.ax_penalties.set_title("Penalties/Costs (RAW Values)", fontsize=12, fontweight="bold")
            
            if len(self.iteration_history) > 0:
                self.ax_penalties.set_xlim(0, max(50, self.iteration_history[-1] + 10))
        
        plt.tight_layout()
        plt.pause(0.01)

    def close(self):
        """Close the live plot figure."""
        plt.ioff()
        plt.close(self.fig)

    def save(self, output_path: Path | str, title: str = "Reward Components (Raw Values)"):
        """Save the current per-component figure to a file."""
        try:
            self.fig.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
        except Exception:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary stats for each component (mean, min, max)."""
        summary = {}
        for component_name, values in self.component_data.items():
            if values:
                summary[component_name] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        return summary