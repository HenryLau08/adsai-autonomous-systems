"""
Reproduce the paper's Table III ablation (Section V-C): compare
different standardization/quantization configurations against each
other on the same environment. This doesn't reproduce the paper's
exact numbers (different env, network, hyperparameters) but lets you
run the same *qualitative* comparison they describe:

    Experiment 1: baseline PPO, no quantization, no standardization.
    Experiment 2: dynamic standardization of rewards only.
    Experiment 3: rewards + values standardized & quantized, with
                  rewards de-standardized back before use in GAE.
    Experiment 4: rewards + values standardized & quantized, with
                  rewards kept quantized/standardized with NO
                  de-standardization step (paper found this performs
                  poorly -- Section V-C).
    Experiment 5: dynamic standardization for rewards (kept in
                  standardized form, no de-standardization) + block
                  standardization for values (de-standardized back).
                  Paper's best-performing config.

Usage:
    python run_ablation.py --total-steps 500000
"""

from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass

import numpy as np
import torch

from heppo import HEPPOAgent, PPOConfig, RolloutBuffer
from heppo.standardization import (
    DynamicRewardStandardizer,
    block_standardize_values,
    block_destandardize_values,
    uniform_quantize,
)
from heppo.gae import compute_gae


# ---- Experiment-specific reward/value pipelines (Table III) ----

class Experiment1Agent(HEPPOAgent):
    """Baseline: no standardization, no quantization at all."""
    def standardize_and_quantize_rewards(self, rewards_raw):
        return rewards_raw

    def standardize_and_quantize_values(self, values_raw):
        return values_raw


class Experiment2Agent(HEPPOAgent):
    """Dynamic standardization of rewards only; values untouched."""
    def standardize_and_quantize_values(self, values_raw):
        return values_raw


class Experiment3Agent(HEPPOAgent):
    """Both standardized + quantized; rewards DE-standardized back
    before GAE (i.e. GAE sees raw-scale rewards, just quantization-noisy)."""
    def standardize_and_quantize_rewards(self, rewards_raw):
        self.reward_standardizer.update(rewards_raw)
        r_std = self.reward_standardizer.standardize(rewards_raw)
        r_std = uniform_quantize(r_std, n_bits=self.cfg.quant_bits)
        # de-standardize back to raw scale (unlike Experiment 5)
        return r_std * self.reward_standardizer.std + self.reward_standardizer.mean


class Experiment4Agent(HEPPOAgent):
    """Both standardized + quantized; rewards kept standardized AND
    NOT de-standardized -- but unlike Exp.5, values ALSO stay in
    quantized/standardized form (no final de-standardization of values
    either). Paper found this performs poorly."""
    def standardize_and_quantize_values(self, values_raw):
        v_std, mu_v, sigma_v = block_standardize_values(values_raw)
        v_std = uniform_quantize(v_std, n_bits=self.cfg.quant_bits)
        return v_std  # NOTE: no de-standardization back to raw scale


# Experiment 5 is exactly HEPPOAgent's default behavior (see ppo_agent.py).
Experiment5Agent = HEPPOAgent


EXPERIMENTS = {
    1: ("Baseline (no std, no quant)", Experiment1Agent),
    2: ("Dynamic reward standardization only", Experiment2Agent),
    3: ("Std+quant both, rewards de-standardized before GAE", Experiment3Agent),
    4: ("Std+quant both, no de-standardization at all", Experiment4Agent),
    5: ("Dynamic reward std (kept) + block value std (de-std'd) [paper's best]", Experiment5Agent),
}


def run_experiment(exp_id: int, env_factory, total_steps: int, T: int, N: int,
                    obs_dim: int, n_actions: int, device: str, seed: int = 0):
    name, agent_cls = EXPERIMENTS[exp_id]
    print(f"\n=== Experiment {exp_id}: {name} ===")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = PPOConfig(quant_bits=8, use_quantization=True, standardize_advantages=True)
    agent = agent_cls(obs_dim=obs_dim, n_actions=n_actions, config=cfg, device=device)
    rollout = RolloutBuffer(T=T, N=N, obs_dim=obs_dim, device=device)
    env = env_factory()

    obs = env.get_obs().to(device)
    rolling_returns = []
    episode_returns = np.zeros(N, dtype=np.float32)
    global_step = 0
    iters = total_steps // (T * N)

    for it in range(iters):
        for t in range(T):
            with torch.no_grad():
                action, log_prob, value = agent.net.act(obs)
            rewards, dones = env.step(action.cpu())
            rewards, dones = rewards.to(device), dones.to(device)
            rollout.push(t, obs, action, log_prob, value, rewards, dones)

            episode_returns += rewards.cpu().numpy()
            for idx in np.nonzero(dones.cpu().numpy())[0]:
                rolling_returns.append(episode_returns[idx])
                episode_returns[idx] = 0.0

            obs = env.get_obs().to(device)
            global_step += N

        with torch.no_grad():
            _, _, bootstrap_value = agent.net.act(obs)
        rollout.bootstrap_value = bootstrap_value

        advantages, returns = agent.compute_advantages(rollout)
        stats = agent.update(rollout, advantages, returns)
        rollout.reset()

        if it % max(1, iters // 10) == 0:
            mean_ret = np.mean(rolling_returns[-50:]) if rolling_returns else float("nan")
            print(f"  iter {it:4d}/{iters} | step {global_step:8d} | mean_return {mean_ret:7.3f} | "
                  f"value_loss {stats['value_loss']:.4f}")

    env.close()
    return rolling_returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-len", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--experiments", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = p.parse_args()

    from train_warlords import WarlordsVecEnv, OBS_DIM, N_ACTIONS

    def env_factory():
        return WarlordsVecEnv(num_envs=args.num_envs)

    N = args.num_envs * 4
    results = {}
    for exp_id in args.experiments:
        rets = run_experiment(
            exp_id, env_factory, args.total_steps, args.rollout_len, N,
            OBS_DIM, N_ACTIONS, args.device,
        )
        results[exp_id] = rets

    print("\n=== Summary (mean of last 50 episode returns) ===")
    for exp_id in args.experiments:
        name, _ = EXPERIMENTS[exp_id]
        rets = results[exp_id]
        mean_ret = np.mean(rets[-50:]) if rets else float("nan")
        print(f"Experiment {exp_id} ({name}): {mean_ret:.3f}")


if __name__ == "__main__":
    main()
