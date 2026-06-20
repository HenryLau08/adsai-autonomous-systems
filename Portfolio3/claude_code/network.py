"""
Actor-Critic network for Atari RAM observations.

`obs_type='ram'` in PettingZoo/ALE gives a length-128 uint8 vector (the
console's 1024-bit RAM, packed as 128 bytes) instead of an image, so a
small MLP is sufficient -- no CNN needed. Warlords' 4 players are
symmetric (same rules, mirrored roles), so we share one network across
all agents (a common multi-agent PPO simplification); each agent's
RAM observation/action is still a separate row in the batch.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = 1.4142135623730951, bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for discrete action spaces.

    Input: RAM observation, shape (B, 128), raw bytes in [0, 255].
    """

    def __init__(self, obs_dim: int = 128, n_actions: int = 6, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        self.critic_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    @staticmethod
    def preprocess(obs_uint8: torch.Tensor) -> torch.Tensor:
        """Map raw RAM bytes [0,255] -> roughly [-1, 1] floats."""
        return obs_uint8.float() / 127.5 - 1.0

    def forward(self, obs_uint8: torch.Tensor):
        x = self.preprocess(obs_uint8)
        h = self.trunk(x)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs_uint8: torch.Tensor):
        """Sample an action; returns (action, log_prob, value)."""
        logits, value = self.forward(obs_uint8)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs_uint8: torch.Tensor, actions: torch.Tensor):
        """Used during the PPO update: recompute log_prob/entropy/value
        for previously-collected (obs, action) pairs under the *current*
        policy parameters."""
        logits, value = self.forward(obs_uint8)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
