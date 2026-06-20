"""
Self-play opponent pool with ELO-based skill tracking.

Why ELO here: under self-play, raw episode return stops being a
useful progress signal (see module docstring discussion in
train_warlords_selfplay.py) because the opponent distribution is
non-stationary -- a flat win rate against an ever-strengthening pool
can still mean real improvement. ELO normalizes for "who you played"
the same way it does in chess: beating a stronger opponent earns more
than beating a weaker one.

Why ELO-WEIGHTED sampling (not uniform, not always-latest):
  - Uniform sampling over the whole pool wastes time on opponents the
    current policy already crushes or has no hope against -- both
    give a near-zero learning gradient (advantage estimates collapse
    toward a constant once one side wins/loses almost every episode).
  - Always playing the latest checkpoint risks "chasing your own tail"
    / cyclic strategies (rock-paper-scissors non-transitivity) and
    catastrophic forgetting of how to beat earlier styles.
  - Weighting sample probability by closeness in ELO to the *current*
    policy's live rating concentrates training on the opponents the
    policy is roughly competitive with right now -- the matches with
    the highest information content (closest to 50/50 win probability,
    where the policy gradient has the most room to move things).
This is a lightweight version of the "Prioritized Fictitious Self-Play"
(PFSP) idea used in AlphaStar / OpenAI Five-style training, simplified
to a single-population pool rather than a league with exploiters.
"""

from __future__ import annotations
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Standard ELO expected score for A vs B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 16.0):
    """One ELO update step. score_a in {0, 0.5, 1} (loss/draw/win).
    Returns (new_rating_a, new_rating_b)."""
    expected_a = elo_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


@dataclass
class OpponentCheckpoint:
    checkpoint_id: int
    state_dict: dict
    elo: float = 1000.0
    games_played: int = 0


class OpponentPool:
    """Maintains a pool of past policy checkpoints with live ELO ratings,
    and supports ELO-weighted sampling of an opponent for a given
    "current policy" rating.

    The CURRENT (training) policy also has an ELO rating tracked here
    (self.current_elo), which is what you should log/plot as the
    primary progress metric instead of raw return.
    """

    def __init__(
        self,
        max_pool_size: int = 20,
        k_factor: float = 16.0,
        temperature: float = 200.0,
        seed_elo: float = 1000.0,
        device: str | torch.device = "cpu",
    ):
        """
        max_pool_size: oldest checkpoints are evicted once exceeded
            (keeps eval/sampling cost bounded; ELO history isn't lost,
            just the snapshot you'd play against).
        k_factor: ELO K-factor (paper-agnostic; 16 is a common, fairly
            conservative choice -- big enough to track real skill
            changes, small enough not to be noise-dominated by single
            4-player free-for-all outcomes).
        temperature: controls how strongly sampling concentrates on
            ELO-close opponents. Sampling weight for opponent i is
            softmax(-|elo_i - current_elo| / temperature). Smaller
            temperature = more sharply concentrated on closest-skill
            opponents; larger = closer to uniform.
        """
        self.pool: list[OpponentCheckpoint] = []
        self.max_pool_size = max_pool_size
        self.k_factor = k_factor
        self.temperature = temperature
        self.current_elo = seed_elo
        self.device = device
        self._next_id = 0

    def add_checkpoint(self, state_dict: dict, inherit_elo: bool = True) -> int:
        """Snapshot the current policy into the pool. Call this
        periodically (e.g. every N PPO iterations) during training.

        inherit_elo=True seeds the new checkpoint's rating at the
        current policy's live ELO (a reasonable prior: "this snapshot
        was about this good when it was saved"), rather than resetting
        to 1000 and having to re-discover its strength from scratch.
        """
        cp = OpponentCheckpoint(
            checkpoint_id=self._next_id,
            state_dict={k: v.detach().cpu().clone() for k, v in state_dict.items()},
            elo=self.current_elo if inherit_elo else 1000.0,
        )
        self._next_id += 1
        self.pool.append(cp)
        if len(self.pool) > self.max_pool_size:
            # evict the OLDEST checkpoint (lowest id), not the weakest --
            # keeps a spread of "eras" of the policy rather than collapsing
            # onto only recent/strong ones, which helps avoid forgetting
            # how to beat earlier strategies (a known self-play failure mode).
            self.pool.sort(key=lambda c: c.checkpoint_id)
            self.pool.pop(0)
        return cp.checkpoint_id

    def sample_opponent(self, rng: random.Random | None = None) -> OpponentCheckpoint | None:
        """ELO-weighted sample: prefer opponents close in rating to the
        current policy. Returns None if the pool is empty (caller
        should fall back to self-play against the live policy, or pure
        random/heuristic agents, for the very first checkpoints)."""
        if not self.pool:
            return None
        rng = rng or random
        diffs = np.array([abs(c.elo - self.current_elo) for c in self.pool], dtype=np.float64)
        logits = -diffs / max(self.temperature, 1e-6)
        logits -= logits.max()  # numerical stability
        weights = np.exp(logits)
        weights /= weights.sum()
        idx = rng.choices(range(len(self.pool)), weights=weights.tolist(), k=1)[0]
        return self.pool[idx]

    def report_result(self, opponent: OpponentCheckpoint, score_current: float) -> None:
        """Update both the current policy's live ELO and the opponent
        checkpoint's ELO after an episode. score_current in {0, 0.5, 1}
        from the CURRENT policy's perspective (1=won, 0=lost, 0.5=draw/
        ambiguous -- e.g. timed-out episode with no winner)."""
        new_current, new_opp = elo_update(self.current_elo, opponent.elo, score_current, self.k_factor)
        self.current_elo = new_current
        opponent.elo = new_opp
        opponent.games_played += 1

    def pool_summary(self) -> str:
        if not self.pool:
            return "pool empty"
        ratings = sorted(c.elo for c in self.pool)
        return (f"pool_size={len(self.pool)} "
                f"elo_min={ratings[0]:.0f} elo_median={ratings[len(ratings)//2]:.0f} "
                f"elo_max={ratings[-1]:.0f} current_elo={self.current_elo:.0f}")

    def save(self, path: str) -> None:
        torch.save(
            {
                "pool": [
                    {"checkpoint_id": c.checkpoint_id, "state_dict": c.state_dict,
                     "elo": c.elo, "games_played": c.games_played}
                    for c in self.pool
                ],
                "current_elo": self.current_elo,
                "next_id": self._next_id,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.pool = [
            OpponentCheckpoint(d["checkpoint_id"], d["state_dict"], d["elo"], d["games_played"])
            for d in data["pool"]
        ]
        self.current_elo = data["current_elo"]
        self._next_id = data["next_id"]
