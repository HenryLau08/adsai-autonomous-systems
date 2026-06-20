import torch
import torch.optim as optim

from env import PettingZooRAMEnv
from model import ActorCritic
from buffer import Buffer
from ppo import PPO
import config as cfg


def train():

    env = PettingZooRAMEnv()

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    agent = PPO(model, optimizer, cfg)
    buffer = Buffer()

    obs = env.reset()

    for step in range(1_000_000):

        action, logprob, value = model.act(obs)

        next_obs, reward, done, _ = env.step(action)

        buffer.store(obs, action, logprob.item(), reward, value.item(), done)

        obs = next_obs

        if done or len(buffer.obs) >= cfg.STEPS_PER_UPDATE:

            batch = buffer.get()
            agent.update(batch)
            buffer.clear()

            obs = env.reset()

            print(f"Update at step {step}")


if __name__ == "__main__":
    train()