import torch
import torch.nn as nn
import copy


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.net(x)
        return self.actor(x), self.critic(x)


def clone_model(model):
    return copy.deepcopy(model)