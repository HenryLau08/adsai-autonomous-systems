"""
Rollout buffer, organized timestep-major (paper Section IV, Fig. 6 /
Algorithm 2): each "row" holds all trajectories' data for one
timestep, which is exactly the access pattern GAE's backward pass
needs (fetch one full timestep block, process all trajectories in
that block in parallel, move to t-1).

This is a plain in-memory (CPU/GPU tensor) version of that layout --
no actual BRAM/FILO stack involved, but the indexing mirrors
RMB[t][i], VMB[t][i] etc. from Algorithm 2.
"""

from __future__ import annotations
import torch


class RolloutBuffer:
    def __init__(self, T: int, N: int, obs_dim: int, device: str | torch.device = "cpu"):
        """T = timesteps per rollout, N = number of parallel trajectories
        (num_envs * num_agents)."""
        self.T, self.N, self.obs_dim = T, N, obs_dim
        self.device = device

        self.obs = torch.zeros((T, N, obs_dim), dtype=torch.uint8, device=device)
        self.actions = torch.zeros((T, N), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=device)       # standardized-scale values, as used by the critic
        self.rewards_raw = torch.zeros((T, N), dtype=torch.float32, device=device)  # raw env rewards, pre-standardization
        self.dones = torch.zeros((T, N), dtype=torch.float32, device=device)

        self.bootstrap_value = torch.zeros(N, dtype=torch.float32, device=device)
        self.ptr = 0

    def push(self, t: int, obs, action, log_prob, value, reward_raw, done):
        """Insert one timestep's data across all N trajectories
        (analogous to Algorithm 2's `Push reward[i][t] into RMB[t][i]`,
        done here for a whole row i=0..N-1 at once)."""
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards_raw[t] = reward_raw
        self.dones[t] = done

    def reset(self):
        self.ptr = 0
