import torch
import torch.optim as optim

import config as cfg
from env import PettingZooRAMEnv
from model import ActorCritic
from buffer import Buffer
from ppo import PPO
from checkpoint import save_checkpoint, load_checkpoint


def train():

    # -------------------------
    # Environment
    # -------------------------
    env = PettingZooRAMEnv()

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # -------------------------
    # Model + Optimizer
    # -------------------------
    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    # -------------------------
    # PPO Agent
    # -------------------------
    agent = PPO(model, optimizer, cfg)
    buffer = Buffer()

    # -------------------------
    # Load checkpoint (if exists)
    # -------------------------
    step = load_checkpoint(cfg.CHECKPOINT_PATH, model, optimizer)

    # -------------------------
    # Reset env
    # -------------------------
    obs = env.reset()

    episode_reward = 0
    episode_count = 0

    # -------------------------
    # Training loop
    # -------------------------
    while step < 1_000_000:

        action, logprob, value = model.act(obs)

        next_obs, reward, done, _ = env.step(action)

        buffer.store(
            obs,
            action,
            logprob.item(),
            reward,
            value.item(),
            done
        )

        obs = next_obs
        episode_reward += reward
        step += 1

        # -------------------------
        # PPO update trigger
        # -------------------------
        if len(buffer.obs) >= cfg.STEPS_PER_UPDATE:

            batch = buffer.get()
            agent.update(batch)
            buffer.clear()

            # -------------------------
            # Save checkpoint
            # -------------------------
            if step % cfg.SAVE_EVERY == 0:
                save_checkpoint(
                    cfg.CHECKPOINT_PATH,
                    model,
                    optimizer,
                    step
                )
                print(f"[Checkpoint] Saved at step {step}")

        # -------------------------
        # Episode end handling
        # -------------------------
        if done:
            episode_count += 1

            print(
                f"Episode {episode_count} | "
                f"Reward: {episode_reward:.2f} | "
                f"Step: {step}"
            )

            obs = env.reset()
            episode_reward = 0


if __name__ == "__main__":
    train()