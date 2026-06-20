"""
PPO clipped-objective update (paper Algorithm 1), using HEPPO's
Experiment 5 configuration (Section V-C / Table III), which the paper
found performs best:

    - Dynamic standardization of REWARDS (running mean/std, Eqs. 6-9),
      kept in standardized form through GAE computation (Section II-C-1,
      "leaving the rewards in their standardized form enhances the
      cumulative rewards by around 50%").
    - Block standardization of VALUES (per-rollout mean/std, Section
      II-B), de-standardized back after quantization for use in the
      critic loss (Section II-C-2).
    - Uniform n-bit quantization (n>=8 recommended, Section V-B) applied
      to both, simulating the precision loss of HEPPO's fixed-point
      hardware pipeline even though this is a float reference impl.
    - Advantage normalization (paper Section V-A: "in most PPO
      implementations, the final calculated advantage vector is
      standardized to stabilize gradient updates"; both with/without
      this layered on top performed well, we expose it as a toggle).
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from .standardization import (
    DynamicRewardStandardizer,
    block_standardize_values,
    block_destandardize_values,
    uniform_quantize,
)
from .gae import compute_gae
from .network import ActorCritic


@dataclass
class PPOConfig:
    # NOTE on gamma/lambda for Warlords' delayed (+1 win / -1 lose,
    # 0 every other step) reward structure: episodes can run for
    # hundreds-to-thousands of frames before the terminal reward
    # lands. With the "dense-task" default of gamma=0.99, a reward
    # 500 steps in the past has decayed by 0.99^500 ~= 0.007 -- it's
    # essentially invisible to GAE by the time it reaches early
    # actions, so the agent gets almost no signal for opening-game
    # play. Raising gamma to 0.999 keeps that same reward at
    # 0.999^500 ~= 0.61, i.e. still a meaningful credit-assignment
    # signal across the whole episode.
    #
    # lambda is raised alongside it (0.95 -> 0.97) for a related
    # reason: lambda trades off bias (low lambda, short effective
    # trace, cuts off long-horizon credit) against variance (high
    # lambda, full Monte-Carlo-like trace, noisier but unbiased over
    # the whole episode). The usual argument for keeping lambda
    # moderate is that dense per-step rewards already carry most of
    # the local signal, so a short trace is "good enough" and lower
    # variance helps. With a SPARSE terminal reward there's no local
    # per-step signal to lean on in the first place -- truncating the
    # trace just throws away the only information that exists. So the
    # usual bias/variance argument tips toward a longer trace despite
    # the variance cost.
    #
    # These aren't tuned for warlords_v3 specifically (no FPGA/cluster
    # access to sweep here) -- they're the standard adjustment used
    # for any sparse/delayed-reward PPO setup (e.g. similar values are
    # common in sparse-reward MuJoCo/Atari-with-clipped-score work).
    # Sweep gamma in {0.995, 0.999, 0.9995} and lambda in {0.95, 0.97,
    # 0.99} if you have the compute; this is a reasonable starting
    # point, not a proven-optimal one.
    gamma: float = 0.999
    lam: float = 0.97

    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    n_epochs: int = 4
    n_minibatches: int = 4
    quant_bits: int = 8          # paper: 8-bit+ is the stable threshold (Section V-B)
    standardize_advantages: bool = True
    use_quantization: bool = True  # toggle HEPPO's quantization step on/off

    # Optional, OFF BY DEFAULT: a small per-step shaping bonus added to
    # the raw reward before standardization (e.g. survival time, ball
    # proximity, etc.), for anyone who wants to densify the signal
    # further. Left off by default because shaping a competitive
    # win/lose objective risks the agent optimizing the shaping term
    # (e.g. "survive longer" without actually trying to win) instead of
    # the true objective -- a classic reward-hacking failure mode in
    # self-play. If you do enable it, keep its magnitude small relative
    # to the +-1 terminal reward and verify win rate (not just return)
    # still improves. See HEPPOAgent.set_reward_shaping_fn().
    use_reward_shaping: bool = False


class HEPPOAgent:
    """Bundles the policy network, optimizer, and the HEPPO-style
    standardization/quantization pipeline around a standard PPO update.

    Self-play note: dynamic reward standardization (Section II-A) uses
    a running mean/std over ALL rewards seen since training start. Under
    self-play, the opponent distribution is non-stationary (it gets
    stronger over time, and which past checkpoints you're matched
    against shifts), so the *marginal* win rate -- and hence the raw
    reward mean -- drifts for reasons unrelated to absolute skill. The
    running standardizer will adapt to this drift, which is actually
    the correct behavior (it's the same non-stationarity-handling the
    paper designed it for, Section II-A: "accounting for all previously
    attained rewards"). The practical implication: don't use raw
    standardized-reward-derived metrics (like average advantage
    magnitude) as your progress signal under self-play -- use the
    OpponentPool's ELO instead (see opponent_pool.py and
    train_warlords_selfplay.py), since ELO is explicitly designed to
    stay meaningful against a moving target.
    """

    def __init__(self, obs_dim: int, n_actions: int, config: PPOConfig,
                 device: str | torch.device = "cpu"):
        self.cfg = config
        self.device = device
        self.net = ActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.reward_standardizer = DynamicRewardStandardizer(device=device)
        self._reward_shaping_fn = None  # optional callable(rollout) -> (T,N) shaping bonus

    def set_reward_shaping_fn(self, fn) -> None:
        """Register an optional shaping function applied to raw rewards
        BEFORE standardization, only used if cfg.use_reward_shaping=True.
        fn(rollout) -> torch.Tensor of shape (T, N), added elementwise to
        rollout.rewards_raw. Keep magnitudes small relative to +-1."""
        self._reward_shaping_fn = fn

    # ---- Step 1: reward standardization (Section II-A) ----
    def standardize_and_quantize_rewards(self, rewards_raw: torch.Tensor) -> torch.Tensor:
        """Update running reward stats with this batch, then return the
        standardized (and optionally quantized) rewards used for GAE."""
        self.reward_standardizer.update(rewards_raw)
        r_std = self.reward_standardizer.standardize(rewards_raw)
        if self.cfg.use_quantization:
            r_std = uniform_quantize(r_std, n_bits=self.cfg.quant_bits)
        return r_std

    # ---- Step 2: block standardization of values (Section II-B) ----
    def standardize_and_quantize_values(self, values_raw: torch.Tensor):
        """Block-standardize values, quantize, then immediately
        de-standardize back to original scale (paper Section II-C-2:
        rewards stay standardized for GAE, but values get a final
        de-standardization step). Returns values on the original scale,
        with quantization error baked in."""
        v_std, mu_v, sigma_v = block_standardize_values(values_raw)
        if self.cfg.use_quantization:
            v_std = uniform_quantize(v_std, n_bits=self.cfg.quant_bits)
        v_recon = block_destandardize_values(v_std, mu_v, sigma_v)
        return v_recon

    # ---- GAE + rewards-to-go ----
    def compute_advantages(self, rollout) -> tuple[torch.Tensor, torch.Tensor]:
        """rollout: a RolloutBuffer (see buffer.py) that has already been
        filled by the collection loop."""
        rewards = rollout.rewards_raw
        if self.cfg.use_reward_shaping and self._reward_shaping_fn is not None:
            rewards = rewards + self._reward_shaping_fn(rollout)

        r_std = self.standardize_and_quantize_rewards(rewards)

        # values_raw has T rows from the buffer + 1 bootstrap row appended
        values_full = torch.cat([rollout.values, rollout.bootstrap_value.unsqueeze(0)], dim=0)
        values_full = self.standardize_and_quantize_values(values_full)

        advantages, returns = compute_gae(
            rewards=r_std,
            values=values_full,
            dones=rollout.dones,
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )

        if self.cfg.standardize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ---- PPO clipped update (Algorithm 1) ----
    def update(self, rollout, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        T, N = rollout.T, rollout.N
        obs = rollout.obs.reshape(T * N, rollout.obs_dim)
        actions = rollout.actions.reshape(T * N)
        old_log_probs = rollout.log_probs.reshape(T * N)
        adv_flat = advantages.reshape(T * N)
        ret_flat = returns.reshape(T * N)

        batch_size = T * N
        minibatch_size = max(1, batch_size // self.cfg.n_minibatches)
        indices = torch.arange(batch_size, device=self.device)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        n_updates = 0

        for _ in range(self.cfg.n_epochs):
            perm = indices[torch.randperm(batch_size, device=self.device)]
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start:start + minibatch_size]

                log_prob, entropy, value = self.net.evaluate_actions(obs[mb_idx], actions[mb_idx])
                ratio = torch.exp(log_prob - old_log_probs[mb_idx])

                mb_adv = adv_flat[mb_idx]

                # PPO-Clip objective (Algorithm 1, line 8)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value regression (Algorithm 1, line 9)
                value_loss = nn.functional.mse_loss(value, ret_flat[mb_idx])

                entropy_loss = entropy.mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["clip_frac"] += clip_frac
                n_updates += 1

        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats
