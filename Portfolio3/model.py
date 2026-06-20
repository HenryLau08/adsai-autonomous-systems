import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, act_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)