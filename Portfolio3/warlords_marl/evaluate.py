"""Evaluatie: toernooien spelen, metrieken berekenen en resultaten plotten.

Dit module vergelijkt het getrainde RL-beleid met de baselines (opdracht 3b/3c)
en levert de visualisaties voor het rapport: leercurves, win-percentages en
overlevingsduur.
"""

from __future__ import annotations

import glob
import os
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

from . import AGENT_ORDER

Policy = Callable[[np.ndarray], int]


# ---------------------------------------------------------------------------
# Eén wedstrijd en een volledig toernooi spelen.
# ---------------------------------------------------------------------------
def play_match(
    parallel_env,
    policies: Dict[str, Policy],
    seed: int = 0,
    max_steps: int = 100_000,
    frame_skip: int = 4,
    record_frames: bool = False,
):
    """Speel één Warlords-wedstrijd met vier policies.

    Parameters
    ----------
    parallel_env:
        Een ``warlords_v3.parallel_env`` (eventueel met ``render_mode="rgb_array"``
        als ``record_frames=True``).
    policies:
        Dict ``{agent_naam: callable(ruwe_ram) -> actie}`` voor elke hoek.
    max_steps:
        Bovengrens op het aantal emulator-frames (voorkomt extreem lange games).
    frame_skip:
        Elke policy beslist één keer per ``frame_skip`` frames; de actie wordt
        herhaald. Dit versnelt de evaluatie (~4x minder policy-evaluaties) en houdt
        het gedrag consistent met de training.

    Returns
    -------
    dict met ``winner``, ``rewards`` (per hoek), ``survival`` (frames overleefd)
    en optioneel ``frames``.
    """
    # Stateful baselines (bijv. RuleBasedPolicy) terugzetten naar beginstand.
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset()

    obs, _ = parallel_env.reset(seed=seed)
    total_reward: Dict[str, float] = defaultdict(float)
    survival: Dict[str, int] = defaultdict(int)
    frames: List[np.ndarray] = []
    steps = 0

    while parallel_env.agents and steps < max_steps:
        # Beslis één keer en herhaal de actie gedurende frame_skip frames.
        actions = {}
        for agent in parallel_env.agents:
            ram = np.asarray(obs[agent], dtype=np.uint8)
            actions[agent] = int(policies[agent](ram))

        for _ in range(frame_skip):
            if not parallel_env.agents or steps >= max_steps:
                break
            step_actions = {a: actions[a] for a in parallel_env.agents}
            obs, rewards, _, _, _ = parallel_env.step(step_actions)
            for agent, rew in rewards.items():
                total_reward[agent] += rew
            for agent in parallel_env.agents:  # nog levende agenten na deze stap
                survival[agent] += 1
            if record_frames:
                frames.append(parallel_env.render())
            steps += 1

    parallel_env.close()

    # Winnaar = hoogste totale beloning; bij gelijkspel de langst overlevende.
    winner = max(
        AGENT_ORDER,
        key=lambda a: (total_reward.get(a, 0.0), survival.get(a, 0)),
    )
    result = {
        "winner": winner,
        "rewards": {a: float(total_reward.get(a, 0.0)) for a in AGENT_ORDER},
        "survival": {a: int(survival.get(a, 0)) for a in AGENT_ORDER},
        "steps": steps,
    }
    if record_frames:
        result["frames"] = frames
    return result


def run_tournament(
    parallel_env_fn,
    policies: Dict[str, Policy],
    n_games: int = 20,
    base_seed: int = 0,
    max_steps: int = 8_000,
    frame_skip: int = 4,
    verbose: bool = True,
):
    """Speel ``n_games`` wedstrijden en aggregeer de resultaten.

    ``parallel_env_fn`` is een fabriek (bijv.
    ``warlords_marl.env_wrapper.make_parallel_env``) zodat elke wedstrijd een
    verse omgeving krijgt. ``max_steps`` begrenst de lengte van een wedstrijd en
    ``frame_skip`` versnelt het spelen; samen voorkomen ze extreem lange, stille
    cellen (en daarmee time-outs op gratis Colab).

    Returns een dict met win-aantallen, win-percentages, gemiddelde beloning en
    gemiddelde overlevingsduur per hoek.
    """
    wins: Counter = Counter()
    reward_sum: Dict[str, float] = defaultdict(float)
    survival_sum: Dict[str, float] = defaultdict(float)

    for game in range(n_games):
        env = parallel_env_fn()
        result = play_match(
            env, policies, seed=base_seed + game,
            max_steps=max_steps, frame_skip=frame_skip,
        )
        wins[result["winner"]] += 1
        for agent in AGENT_ORDER:
            reward_sum[agent] += result["rewards"][agent]
            survival_sum[agent] += result["survival"][agent]
        if verbose:
            print(f"game {game + 1:3d}/{n_games}: winnaar = {result['winner']}, "
                  f"frames = {result['steps']}")

    summary = {
        "n_games": n_games,
        "wins": {a: int(wins.get(a, 0)) for a in AGENT_ORDER},
        "win_rate": {a: wins.get(a, 0) / n_games for a in AGENT_ORDER},
        "mean_reward": {a: reward_sum[a] / n_games for a in AGENT_ORDER},
        "mean_survival": {a: survival_sum[a] / n_games for a in AGENT_ORDER},
    }
    return summary


# ---------------------------------------------------------------------------
# Visualisaties.
# ---------------------------------------------------------------------------
def moving_average(values, window: int = 50):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _load_monitor_rewards(monitor_dir: str):
    """Lees alle episode-beloningen uit de monitor-CSV's in een map (op volgorde)."""
    import pandas as pd

    rewards = []
    for path in sorted(glob.glob(os.path.join(monitor_dir, "*.monitor.csv"))):
        # Monitor-CSV's beginnen met een #-commentaarregel met JSON-metadata.
        df = pd.read_csv(path, skiprows=1)
        rewards.append(df)
    if not rewards:
        return None
    merged = pd.concat(rewards, ignore_index=True)
    merged = merged.sort_values("t")  # chronologisch op wandkloktijd
    return merged["r"].to_numpy()


def plot_learning_curves(
    monitor_root: str,
    agents: Optional[List[str]] = None,
    window: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Plot de geglade trainingsbeloning per hoek over alle generaties.

    Verwacht de mappenstructuur die :func:`train.train_independent` aanmaakt:
    ``<monitor_root>/<agent>_gen<g>/*.monitor.csv``.

    Als ``window`` niet wordt opgegeven, schaalt het smoothing-venster mee met het
    aantal episodes (zodat ook een korte run niet onnodig rommelig oogt).
    """
    import matplotlib.pyplot as plt

    agents = agents or AGENT_ORDER

    # Verzamel eerst per hoek de beloningen, zodat we het venster kunnen afstemmen.
    rewards_per_agent = {}
    for agent in agents:
        gen_dirs = sorted(glob.glob(os.path.join(monitor_root, f"{agent}_gen*")))
        all_rewards = [r for r in (_load_monitor_rewards(d) for d in gen_dirs) if r is not None]
        if all_rewards:
            rewards_per_agent[agent] = np.concatenate(all_rewards)

    if window is None:
        max_eps = max((len(r) for r in rewards_per_agent.values()), default=0)
        window = max(1, min(50, max_eps // 5))  # adaptief: ~1/5 van de episodes, max 50

    plt.figure(figsize=(9, 5))
    for agent, rewards in rewards_per_agent.items():
        plt.plot(moving_average(rewards, window), label=agent)

    plt.title("Leercurve per hoek (geglad over episodes)")
    plt.xlabel("Episode")
    plt.ylabel(f"Gem. episode-beloning (window={window})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"opgeslagen: {save_path}")
    return plt.gcf()


def plot_winrates(
    summaries: Dict[str, dict],
    save_path: Optional[str] = None,
):
    """Vergelijk win-percentages tussen configuraties met een staafdiagram.

    ``summaries`` is een dict ``{label: run_tournament-resultaat}``, bijv.
    ``{"4x random": ..., "4x PPO": ...}``.
    """
    import matplotlib.pyplot as plt

    labels = list(summaries.keys())
    x = np.arange(len(AGENT_ORDER))
    width = 0.8 / max(1, len(labels))

    plt.figure(figsize=(9, 5))
    for i, label in enumerate(labels):
        win_rate = summaries[label]["win_rate"]
        heights = [win_rate[a] for a in AGENT_ORDER]
        plt.bar(x + i * width, heights, width=width, label=label)

    plt.xticks(x + width * (len(labels) - 1) / 2, AGENT_ORDER)
    plt.ylabel("Win-percentage")
    plt.title("Win-percentage per hoek en configuratie")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"opgeslagen: {save_path}")
    return plt.gcf()
