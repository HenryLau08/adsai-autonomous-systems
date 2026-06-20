import torch
import numpy as np

from env import WarlordsEnv
from model import PolicyNet
from ppo import PPO


def train():

    env = WarlordsEnv()

    obs_dim = 128   # RAM obs (approx)
    act_dim = 6     # warlords action range [0,5]

    policy = PolicyNet(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    ppo = PPO(policy, optimizer, {
        "gamma": 0.99,
        "lam": 0.95,
        "clip": 0.2,
        "epochs": 4
    })

    buffer = {
        "obs": [],
        "actions": [],
        "logp": [],
        "rewards": [],
        "values": [],
        "dones": []
    }

    for ep in range(10000):

        obs = env.reset()
        done = False

        ep_reward = 0

        while not done:

            actions = {}

            for agent_id in env.agents:

                o = torch.tensor(obs[agent_id], dtype=torch.float32)

                logits, value = policy(o)

                dist = torch.distributions.Categorical(logits=logits)

                action = dist.sample()

                logp = dist.log_prob(action)

                action = action.item()

                # IMPORTANT: correct bounds [0,5]
                action = max(0, min(action, 5))

                actions[agent_id] = action

                buffer["obs"].append(obs[agent_id])
                buffer["actions"].append(action)
                buffer["logp"].append(logp.item())
                buffer["values"].append(value.item())

            obs, rewards, dones, infos, done = env.step(actions)

            # sparse reward handling (IMPORTANT)
            r = sum(rewards.values())

            buffer["rewards"].append(r)
            buffer["dones"].append(done)

            ep_reward += r

        # PPO update
        ppo.update((
            buffer["obs"],
            buffer["actions"],
            buffer["logp"],
            buffer["rewards"],
            buffer["values"],
            buffer["dones"]
        ))

        buffer = {k: [] for k in buffer}

        print(f"Episode {ep} | Reward {ep_reward}")


if __name__ == "__main__":
    train()