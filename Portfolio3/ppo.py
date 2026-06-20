import torch
import numpy as np


class PPO:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def compute_gae(self, rewards, values, dones):
        gamma = self.cfg["gamma"]
        lam = self.cfg["lam"]

        values = values + [0]

        gae = 0
        adv = []

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv.insert(0, gae)

        returns = np.array(adv) + np.array(values[:-1])
        return np.array(adv), returns

    def update(self, batch):
        obs, actions, logp_old, rewards, values, dones = batch

        adv, ret = self.compute_gae(rewards, values, dones)

        # normalization (IMPORTANT for sparse + delayed rewards)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions)
        logp_old = torch.tensor(logp_old)
        returns = torch.tensor(ret, dtype=torch.float32)
        adv = torch.tensor(adv, dtype=torch.float32)

        for _ in range(self.cfg["epochs"]):

            logits, value = self.model(obs)

            dist = torch.distributions.Categorical(logits=logits)

            logp = dist.log_prob(actions)

            ratio = torch.exp(logp - logp_old)

            clip = self.cfg["clip"]

            loss1 = ratio * adv
            loss2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv

            policy_loss = -torch.min(loss1, loss2).mean()
            value_loss = ((returns - value.squeeze()) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()