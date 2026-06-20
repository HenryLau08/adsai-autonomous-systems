import torch
import numpy as np
from utils import compute_gae


class PPO:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config

        self.reward_mean = 0
        self.reward_var = 0
        self.count = 1e-8

    def normalize_reward(self, r):
        self.count += 1
        delta = r - self.reward_mean
        self.reward_mean += delta / self.count
        self.reward_var += delta * (r - self.reward_mean)
        std = (self.reward_var / self.count) ** 0.5
        return (r - self.reward_mean) / (std + 1e-8)

    def update(self, batch):
        obs, actions, old_logprobs, rewards, values, dones = batch

        # -----------------------
        # Dynamic reward norm
        # -----------------------
        rewards = np.array([self.normalize_reward(r) for r in rewards])

        # -----------------------
        # GAE
        # -----------------------
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            self.config.GAMMA,
            self.config.LAMBDA
        )

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # advantage normalization (recommended even in paper practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -----------------------
        # PPO update
        # -----------------------
        for _ in range(self.config.EPOCHS):

            logits, value = self.model(obs)

            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.CLIP_EPS,
                1 + self.config.CLIP_EPS
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((returns - value.squeeze()) ** 2).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()