import torch
import numpy as np


class PPO:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def compute_gae(self, rewards, values, dones):
        gamma = self.config["gamma"]
        lam = self.config["lam"]

        adv = []
        gae = 0

        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv.insert(0, gae)

        returns = np.array(adv) + np.array(values[:-1])

        return np.array(adv), returns

    def update(self, batch):
        obs, actions, old_logp, rewards, values, dones = batch

        adv, ret = self.compute_gae(rewards, values, dones)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions)
        old_logp = torch.tensor(old_logp)

        ret = torch.tensor(ret, dtype=torch.float32)
        adv = torch.tensor(adv, dtype=torch.float32)

        for _ in range(self.config["epochs"]):

            logits, value = self.model(obs)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            ratio = torch.exp(log_prob - old_logp)

            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - self.config["clip"], 1 + self.config["clip"]) * adv

            policy_loss = -torch.min(s1, s2).mean()
            value_loss = ((ret - value.squeeze()) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()