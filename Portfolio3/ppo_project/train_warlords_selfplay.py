"""
Self-play training for warlords_v3 (RAM obs) with ELO-weighted opponent
sampling from a pool of past checkpoints.

WHY THIS IS STRUCTURED DIFFERENTLY FROM train_warlords.py:

train_warlords.py treats all 4 agent slots in every env as "the
trainee" -- they all push experience into the same rollout buffer and
get the same PPO update. That's fine for a cooperative or
shared-objective game, but Warlords is competitive (last fortress
standing wins): if all 4 slots are simultaneously being trained by the
SAME gradient step, the "opponents" a trainee faces are changing out
from under it mid-rollout in a way that has no stable opponent
identity to measure progress against, and there's no natural place to
hang an ELO rating (whose ELO -- the shared policy's? against whom?).

Here, each env has exactly ONE trainee slot (always 'first_0') that
collects rollout data and gets PPO-updated; the other 3 slots
('second_0','third_0','fourth_0') are filled by FROZEN opponent
snapshots sampled (independently per env, per episode) from the
OpponentPool via ELO-weighted sampling. This gives:
  - A single, well-defined "current policy" with a single live ELO
    rating to track as the primary progress metric.
  - A standard self-play curriculum: opponents are resampled every
    episode, weighted toward the trainee's current skill level, so
    the trainee is consistently facing competitive (~50/50) matches
    rather than steamrolling stale weak opponents or hopelessly
    losing to far-future versions of itself.
  - Periodic snapshots of the trainee get added back into the pool
    (--snapshot-every iterations), so the pool grows alongside the
    trainee's own improvement (this is what makes it self-play rather
    than training against a fixed fixed set of opponents).

The other 3 non-trainee slots' actions are sampled from their
respective frozen networks but NOT pushed into the rollout buffer and
NOT used in the PPO update -- only first_0's transitions count.

Usage:
    pip install "pettingzoo[atari]" autorom torch numpy
    AutoROM --accept-license
    python train_warlords_selfplay.py --total-steps 5_000_000
"""

from __future__ import annotations
import argparse
import copy
import random
import time
from collections import deque

import numpy as np
import torch

from heppo import HEPPOAgent, PPOConfig, RolloutBuffer
from heppo.network import ActorCritic
from heppo.opponent_pool import OpponentPool

try:
    from pettingzoo.atari import warlords_v3
except ImportError as e:
    raise ImportError(
        "pettingzoo[atari] is required. Install with:\n"
        '  pip install "pettingzoo[atari]" autorom\n'
        "  AutoROM --accept-license"
    ) from e


OBS_DIM = 128
N_ACTIONS = 6
AGENT_NAMES = ["first_0", "second_0", "third_0", "fourth_0"]
TRAINEE_SLOT = "first_0"   # the slot whose experience is collected/trained


class SelfPlayWarlordsVecEnv:
    """Like WarlordsVecEnv, but only the TRAINEE_SLOT agent's
    (obs, action, reward, done) per env is exposed for training. The
    other 3 slots are driven by externally-supplied opponent networks
    (one independently-sampled opponent network per OTHER slot per
    env, resampled at the start of each episode for that env).

    Episode outcome tracking: when an env's episode ends, we report
    whether the TRAINEE won (got +1, i.e. was the last fortress
    standing), lost (-1), or it was ambiguous/simultaneous (treated as
    a draw, score=0.5) so ELO can be updated.
    """

    def __init__(self, num_envs: int, opponent_pool: OpponentPool,
                 device: str | torch.device, max_cycles: int = 10_000, seed: int = 0):
        self.num_envs = num_envs
        self.N = num_envs  # ONE trajectory (the trainee) per env now, not 4
        self.device = device
        self.pool = opponent_pool
        self.envs = [
            warlords_v3.parallel_env(obs_type="ram", max_cycles=max_cycles)
            for _ in range(num_envs)
        ]
        self._trainee_obs = np.zeros((num_envs, OBS_DIM), dtype=np.uint8)
        # per-env opponent networks for the 3 non-trainee slots
        self._opponent_nets: list[dict[str, ActorCritic]] = [dict() for _ in range(num_envs)]
        self._opponent_checkpoints: list[dict] = [dict() for _ in range(num_envs)]
        # per-env, per-OTHER-slot obs cache (needed to act with opponent nets)
        self._other_obs = [dict() for _ in range(num_envs)]

        self.rng = random.Random(seed)

        for i, env in enumerate(self.envs):
            obs, _infos = env.reset(seed=seed + i)
            self._trainee_obs[i] = obs[TRAINEE_SLOT]
            for name in AGENT_NAMES:
                if name != TRAINEE_SLOT:
                    self._other_obs[i][name] = obs.get(name, np.zeros(OBS_DIM, dtype=np.uint8))
            self._resample_opponents(i)

    def _build_net_from_checkpoint(self, cp) -> ActorCritic:
        net = ActorCritic(obs_dim=OBS_DIM, n_actions=N_ACTIONS).to(self.device)
        if cp is not None:
            net.load_state_dict(cp.state_dict)
        net.eval()
        return net

    def _resample_opponents(self, env_idx: int):
        """Pick a fresh opponent checkpoint independently for each of the
        3 non-trainee slots in this env (ELO-weighted). If the pool is
        empty (very start of training), opponents act randomly."""
        for name in AGENT_NAMES:
            if name == TRAINEE_SLOT:
                continue
            cp = self.pool.sample_opponent(self.rng)
            self._opponent_checkpoints[env_idx][name] = cp
            self._opponent_nets[env_idx][name] = (
                self._build_net_from_checkpoint(cp) if cp is not None else None
            )

    def get_trainee_obs(self) -> torch.Tensor:
        return torch.from_numpy(self._trainee_obs.copy()).to(self.device)

    @torch.no_grad()
    def _opponent_actions(self, env_idx: int, env) -> dict[str, int]:
        actions = {}
        for name in AGENT_NAMES:
            if name == TRAINEE_SLOT or name not in env.agents:
                continue
            net = self._opponent_nets[env_idx][name]
            obs = torch.from_numpy(self._other_obs[env_idx][name]).unsqueeze(0).to(self.device)
            if net is None:
                actions[name] = self.rng.randrange(N_ACTIONS)
            else:
                action, _, _ = net.act(obs)
                actions[name] = int(action.item())
        return actions

    def step(self, trainee_actions: torch.Tensor):
        """trainee_actions: (num_envs,) long tensor (action for TRAINEE_SLOT
        in each env). Returns (rewards, dones) each (num_envs,) float, for
        the TRAINEE only, plus a list of finished-episode outcomes
        [(env_idx, score_for_trainee), ...] (score in {0,1,0.5}) for ELO
        updates, only populated on the step(s) where an episode fully ends."""
        trainee_actions_np = trainee_actions.cpu().numpy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        outcomes = []

        for i, env in enumerate(self.envs):
            if TRAINEE_SLOT not in env.agents:
                # trainee already finished this episode (died earlier than
                # others) -- step remaining agents with no-op until episode
                # ends, then reset. We don't collect trainee reward here
                # since done was already reported on the step it died.
                if len(env.agents) > 0:
                    action_dict = self._opponent_actions(i, env)
                    obs, rew, term, trunc, _infos = env.step(action_dict)
                    for name in AGENT_NAMES:
                        if name != TRAINEE_SLOT and name in obs:
                            self._other_obs[i][name] = obs[name]
                if len(env.agents) == 0:
                    obs, _infos = env.reset()
                    self._trainee_obs[i] = obs[TRAINEE_SLOT]
                    for name in AGENT_NAMES:
                        if name != TRAINEE_SLOT:
                            self._other_obs[i][name] = obs.get(name, np.zeros(OBS_DIM, dtype=np.uint8))
                    self._resample_opponents(i)
                continue

            action_dict = {TRAINEE_SLOT: int(trainee_actions_np[i])}
            action_dict.update(self._opponent_actions(i, env))

            obs, rew, term, trunc, _infos = env.step(action_dict)

            trainee_done = term.get(TRAINEE_SLOT, False) or trunc.get(TRAINEE_SLOT, False)
            if TRAINEE_SLOT in rew:
                rewards[i] = rew[TRAINEE_SLOT]
                dones[i] = float(trainee_done)
            else:
                dones[i] = 1.0

            if TRAINEE_SLOT in obs:
                self._trainee_obs[i] = obs[TRAINEE_SLOT]
            for name in AGENT_NAMES:
                if name != TRAINEE_SLOT and name in obs:
                    self._other_obs[i][name] = obs[name]

            if trainee_done:
                # +1 / -1 reward directly tells us the outcome; treat
                # exactly-0 terminal reward (e.g. truncation/draw) as a draw
                r = rewards[i]
                score = 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
                outcomes.append((i, score))

            if len(env.agents) == 0:
                obs, _infos = env.reset()
                self._trainee_obs[i] = obs[TRAINEE_SLOT]
                for name in AGENT_NAMES:
                    if name != TRAINEE_SLOT:
                        self._other_obs[i][name] = obs.get(name, np.zeros(OBS_DIM, dtype=np.uint8))
                self._resample_opponents(i)
            elif trainee_done:
                # trainee died but others still playing -- trainee's slot
                # is now inactive until the whole episode resets; next
                # step() calls will hit the "TRAINEE_SLOT not in env.agents"
                # branch above. We still need an opponent resample lined up
                # for when reset happens, which the reset branch handles.
                pass

        return (
            torch.from_numpy(rewards).to(self.device),
            torch.from_numpy(dones).to(self.device),
            outcomes,
        )

    def close(self):
        for env in self.envs:
            env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16, help="parallel Warlords instances; N = this (1 trainee/env)")
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--gamma", type=float, default=0.999, help="raised for delayed win/lose reward (see PPOConfig docstring)")
    p.add_argument("--lam", type=float, default=0.97)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--quant-bits", type=int, default=8)
    p.add_argument("--no-quantization", action="store_true")
    p.add_argument("--no-adv-standardize", action="store_true")
    # self-play specific
    p.add_argument("--snapshot-every", type=int, default=20, help="add trainee to opponent pool every N PPO iterations")
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--elo-k", type=float, default=16.0)
    p.add_argument("--elo-temperature", type=float, default=200.0, help="lower = sample opponents more tightly around current ELO")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint", type=str, default="heppo_warlords_selfplay.pt")
    p.add_argument("--pool-checkpoint", type=str, default="heppo_warlords_pool.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = args.device

    pool = OpponentPool(
        max_pool_size=args.pool_size,
        k_factor=args.elo_k,
        temperature=args.elo_temperature,
        device=device,
    )

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

    # Seed the pool with the initial random policy so envs have something
    # to sample from from the very first iteration (otherwise opponents
    # fall back to fully-random actions, which is a degenerate "opponent").
    pool.add_checkpoint(agent.net.state_dict(), inherit_elo=False)

    vec_env = SelfPlayWarlordsVecEnv(num_envs=args.num_envs, opponent_pool=pool, device=device, seed=args.seed)
    N = vec_env.N
    T = args.rollout_len

    rollout = RolloutBuffer(T=T, N=N, obs_dim=OBS_DIM, device=device)

    recent_results = deque(maxlen=200)  # 1=win, 0=loss, 0.5=draw, for win-rate logging
    total_iters = args.total_steps // (T * N)
    obs = vec_env.get_trainee_obs()

    start_time = time.time()
    global_step = 0

    for it in range(1, total_iters + 1):
        for t in range(T):
            with torch.no_grad():
                action, log_prob, value = agent.net.act(obs)

            rewards, dones, outcomes = vec_env.step(action.cpu())
            rollout.push(t, obs, action, log_prob, value, rewards, dones)

            for env_idx, score in outcomes:
                opp_cp = vec_env._opponent_checkpoints[env_idx].get("second_0")
                # NOTE: with 3 distinct opponents per env (one per
                # non-trainee slot), we attribute the ELO update to
                # 'second_0's sampled checkpoint as a representative
                # opponent for this match. A more elaborate scheme could
                # update against all 3 simultaneously; we keep it simple
                # and slightly conservative (single representative
                # opponent) since over many episodes with independent
                # resampling this still converges to sensible ratings.
                if opp_cp is not None:
                    pool.report_result(opp_cp, score)
                else:
                    # opponent was a random-action placeholder (pool was
                    # empty) -- still update current_elo against a fixed
                    # nominal "random policy" rating of 0 for bookkeeping,
                    # but don't pollute pool ratings.
                    from heppo.opponent_pool import elo_update
                    pool.current_elo, _ = elo_update(pool.current_elo, 0.0, score, pool.k_factor)
                recent_results.append(score)

            obs = vec_env.get_trainee_obs()
            global_step += N

        with torch.no_grad():
            _, _, bootstrap_value = agent.net.act(obs)
        rollout.bootstrap_value = bootstrap_value

        advantages, returns = agent.compute_advantages(rollout)
        stats = agent.update(rollout, advantages, returns)
        rollout.reset()

        if it % args.snapshot_every == 0:
            pool.add_checkpoint(agent.net.state_dict(), inherit_elo=True)

        if it % args.log_every == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            win_rate = float(np.mean([1.0 if s == 1.0 else 0.0 for s in recent_results])) if recent_results else float("nan")
            print(
                f"iter {it:5d} | step {global_step:9d} | sps {sps:6d} | "
                f"win_rate(last200) {win_rate:5.3f} | current_elo {pool.current_elo:6.0f} | "
                f"{pool.pool_summary()} | "
                f"policy_loss {stats['policy_loss']:7.4f} | value_loss {stats['value_loss']:7.4f} | "
                f"entropy {stats['entropy']:6.4f} | clip_frac {stats['clip_frac']:5.3f}"
            )

    torch.save(
        {
            "model_state_dict": agent.net.state_dict(),
            "reward_standardizer": agent.reward_standardizer.state_dict(),
            "current_elo": pool.current_elo,
            "config": vars(args),
        },
        args.checkpoint,
    )
    pool.save(args.pool_checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}, pool to {args.pool_checkpoint}")
    vec_env.close()


if __name__ == "__main__":
    main()
