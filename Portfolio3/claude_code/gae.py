"""
Generalized Advantage Estimation (paper Section I-A / Eq. 2-5).

HEPPO's hardware processes all trajectories for a given timestep in
parallel, then steps backward through time (the FILO memory layout of
Section IV). This module mirrors that data-parallel-over-trajectories,
backward-over-time structure in plain PyTorch:

    for t in reversed(range(T)):
        for all trajectories i (vectorized):
            delta[i] = r[i,t] + gamma * V(s[i,t+1]) * (1-done) - V(s[i,t])
            A[i,t]   = delta[i] + gamma*lambda*(1-done)*A[i,t+1]   (Eq. 4)
    RTG = V + A                                                   (Eq. 5)

We don't implement the k-step lookahead pipelining trick (Section
III-B) -- that's purely a hardware technique to break a feedback loop
across clock cycles; it computes the exact same recursion, just
re-associated to allow register retiming. The recursion below is
mathematically identical to Eq. (10)/(11) with k=1.
"""

from __future__ import annotations
import torch


def compute_gae(
    rewards: torch.Tensor,      # (T, N) standardized rewards
    values: torch.Tensor,       # (T+1, N) values incl. bootstrap at T
    dones: torch.Tensor,        # (T, N) 1.0 if episode ended at this step
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """Compute GAE advantages and rewards-to-go for a batch of trajectories.

    Shapes: T = timesteps, N = number of parallel trajectories
    (e.g. N = num_envs * num_agents for warlords_v3 with 4 agents).

    `values` has T+1 rows: values[t] is V(s_t) for t=0..T-1, and
    values[T] is the bootstrap value V(s_T) used to seed the backward
    recursion (paper's "future values" dependency, Table II).

    Returns:
        advantages: (T, N) float tensor, GAE estimate A^GAE_t (Eq. 3/4)
        returns:    (T, N) float tensor, rewards-to-go (Eq. 5)
    """
    T, N = rewards.shape
    assert values.shape == (T + 1, N), f"expected values shape ({T+1},{N}), got {tuple(values.shape)}"
    assert dones.shape == (T, N)

    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

    # Backward pass, t = T-1 .. 0 (matches the paper's FILO / reverse-order
    # iteration, Section III-B and Algorithm 2, lines 9-15)
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]   # Eq. 2 (TD residual)
        last_adv = delta + gamma * lam * not_done * last_adv                # Eq. 4
        advantages[t] = last_adv

    returns = values[:-1] + advantages   # Eq. 5: Rewards-to-Go = V_t + A^GAE_t
    return advantages, returns
