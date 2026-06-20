"""
Train a shared-policy PPO agent on PettingZoo Atari Warlords (RAM obs),
using HEPPO's dynamic reward standardization + block value
standardization + uniform quantization pipeline ahead of GAE
(paper Sections II, V; "Experiment 5" config from Table III).

Usage:
    pip install "pettingzoo[atari]" autorom torch numpy
    AutoROM --accept-license          # installs Atari ROMs (one-time, needs internet)
    python train_warlords.py --total-steps 2_000_000

Notes on the environment:
    - warlords_v3 is a 4-player Atari game (last-fortress-standing).
      With obs_type='ram', each agent's observation is the console's
      128-byte RAM vector (not pixels), so a small MLP suffices --
      see heppo/network.py.
    - Action space is Discrete(6): noop, fire, up, right, left, down.
    - We use the *parallel* PettingZoo API (all 4 agents step
      simultaneously each frame) and treat the 4 agents as 4 extra
      parallel "trajectories" sharing one policy (a standard
      parameter-sharing simplification for symmetric multi-agent games).
    - If you run multiple Warlords env instances (--num-envs > 1),
      total parallel trajectories N = num_envs * 4.
"""

from __future__ import annotations
import argparse
import time
from collections import deque

import numpy as np
import torch

from heppo import HEPPOAgent, PPOConfig, RolloutBuffer

try:
    from pettingzoo.atari import warlords_v3
except ImportError as e:
    raise ImportError(
        "pettingzoo[atari] is required. Install with:\n"
        '  pip install "pettingzoo[atari]" autorom\n'
        "  AutoROM --accept-license"
    ) from e


OBS_DIM = 128       # RAM observation size
N_ACTIONS = 6        # warlords_v3 minimal action space
AGENT_NAMES = ["first_0", "second_0", "third_0", "fourth_0"]


class WarlordsVecEnv:
    """Wraps `num_envs` independent warlords_v3 parallel_env instances
    and exposes a single batched step/reset over N = num_envs * 4
    trajectories (agent order is fixed: AGENT_NAMES, repeated per env).

    PettingZoo's parallel API drops agents from the returned dicts once
    they're done (fortress destroyed) until the whole episode resets,
    so we track per-(env, agent) "active" masks and feed zeros /
    last-known-done for inactive slots, auto-resetting an env once all
    4 of its agents are done.
    """

    def __init__(self, num_envs: int = 8, max_cycles: int = 10_000, seed: int = 0):
        self.num_envs = num_envs
        self.N = num_envs * 4
        self.envs = [
            warlords_v3.parallel_env(obs_type="ram", max_cycles=max_cycles)
            for _ in range(num_envs)
        ]
        self._obs_buf = np.zeros((self.N, OBS_DIM), dtype=np.uint8)
        for i, env in enumerate(self.envs):
            obs, _infos = env.reset(seed=seed + i)
            self._scatter_obs(i, obs)

    def _scatter_obs(self, env_idx: int, obs_dict: dict):
        base = env_idx * 4
        for j, name in enumerate(AGENT_NAMES):
            if name in obs_dict:
                self._obs_buf[base + j] = obs_dict[name]
            # if name missing (agent already done this episode), keep last obs

    def get_obs(self) -> torch.Tensor:
        return torch.from_numpy(self._obs_buf.copy())

    def step(self, actions: torch.Tensor):
        """actions: (N,) long tensor. Returns (rewards, dones) each (N,) float."""
        actions_np = actions.cpu().numpy()
        rewards = np.zeros(self.N, dtype=np.float32)
        dones = np.zeros(self.N, dtype=np.float32)

        for i, env in enumerate(self.envs):
            base = i * 4
            action_dict = {
                name: int(actions_np[base + j])
                for j, name in enumerate(AGENT_NAMES)
                if name in env.agents
            }
            if not action_dict:
                # all 4 agents finished -> episode already over, reset fresh
                obs, _infos = env.reset()
                self._scatter_obs(i, obs)
                continue

            obs, rew, term, trunc, _infos = env.step(action_dict)
            for j, name in enumerate(AGENT_NAMES):
                if name in rew:
                    rewards[base + j] = rew[name]
                    dones[base + j] = float(term.get(name, False) or trunc.get(name, False))
                else:
                    # agent already finished earlier this episode: no-op, marked done
                    dones[base + j] = 1.0
            self._scatter_obs(i, obs)

            if len(env.agents) == 0:
                # episode fully ended (all 4 fortresses fell / max_cycles) -> reset
                obs, _infos = env.reset()
                self._scatter_obs(i, obs)

        return torch.from_numpy(rewards), torch.from_numpy(dones)

    def close(self):
        for env in self.envs:
            env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=8, help="parallel Warlords instances (N=4*this)")
    p.add_argument("--rollout-len", type=int, default=128, help="timesteps per rollout (T)")
    p.add_argument("--total-steps", type=int, default=2_000_000, help="total env frames across all trajectories")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--quant-bits", type=int, default=8, help="HEPPO uniform quantization bit-width (paper: 8+ is stable)")
    p.add_argument("--no-quantization", action="store_true", help="disable HEPPO quantization (ablation)")
    p.add_argument("--no-adv-standardize", action="store_true", help="disable advantage standardization (ablation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint", type=str, default="heppo_warlords.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    vec_env = WarlordsVecEnv(num_envs=args.num_envs, seed=args.seed)
    N = vec_env.N
    T = args.rollout_len

    cfg = PPOConfig(
        gamma=args.gamma,
        lam=args.lam,
        clip_coef=args.clip_coef,
        lr=args.lr,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        quant_bits=args.quant_bits,
        use_quantization=not args.no_quantization,
        standardize_advantages=not args.no_adv_standardize,
    )
    agent = HEPPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, config=cfg, device=device)
    rollout = RolloutBuffer(T=T, N=N, obs_dim=OBS_DIM, device=device)

    episode_returns = np.zeros(N, dtype=np.float32)
    recent_returns = deque(maxlen=100)

    total_iters = args.total_steps // (T * N)
    obs = vec_env.get_obs().to(device)

    start_time = time.time()
    global_step = 0

    for it in range(1, total_iters + 1):
        for t in range(T):
            with torch.no_grad():
                action, log_prob, value = agent.net.act(obs)

            rewards, dones = vec_env.step(action.cpu())
            rewards = rewards.to(device)
            dones = dones.to(device)

            rollout.push(t, obs, action, log_prob, value, rewards, dones)

            episode_returns += rewards.cpu().numpy()
            done_np = dones.cpu().numpy()
            for idx in np.nonzero(done_np)[0]:
                recent_returns.append(episode_returns[idx])
                episode_returns[idx] = 0.0

            obs = vec_env.get_obs().to(device)
            global_step += N

        with torch.no_grad():
            _, _, bootstrap_value = agent.net.act(obs)
        rollout.bootstrap_value = bootstrap_value

        advantages, returns = agent.compute_advantages(rollout)
        stats = agent.update(rollout, advantages, returns)
        rollout.reset()

        if it % args.log_every == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
            print(
                f"iter {it:5d} | step {global_step:9d} | sps {sps:6d} | "
                f"mean_ep_return(last100) {mean_ret:7.3f} | "
                f"policy_loss {stats['policy_loss']:7.4f} | value_loss {stats['value_loss']:7.4f} | "
                f"entropy {stats['entropy']:6.4f} | clip_frac {stats['clip_frac']:5.3f} | "
                f"reward_std_running {agent.reward_standardizer.std.item():7.4f}"
            )

    torch.save(
        {
            "model_state_dict": agent.net.state_dict(),
            "reward_standardizer": agent.reward_standardizer.state_dict(),
            "config": vars(args),
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    vec_env.close()


if __name__ == "__main__":
    main()
