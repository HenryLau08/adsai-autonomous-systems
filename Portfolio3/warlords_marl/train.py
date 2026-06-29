"""Trainingslogica: independent-learners (IPPO) met PPO voor Warlords.

We trainen vier afzonderlijke PPO-policies, één per hoek (independent learners).
De niet-stationariteit van gelijktijdig lerende agenten wordt getemd door de
tegenstanders per *generatie* te bevriezen:

1. Aan het begin van een generatie maken we een snapshot van elk huidig beleid.
2. Vervolgens traint elk beleid een vast aantal stappen tegen de bevroren
   snapshots van de andere drie hoeken.
3. Herhaal voor meerdere generaties; de tegenstanders worden zo steeds sterker.

Dit is een gangbare, stabiele benadering van independent multi-agent RL en nauw
verwant aan *fictitious self-play* (Heinrich & Silver, 2016) en IPPO
(de Witt et al., 2020). PPO zelf volgt Schulman et al. (2017).

Referenties (zie het rapport voor de volledige APA-lijst):
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
- de Witt, C. S., et al. (2020). Is Independent Learning All You Need...? (IPPO)
- Raffin, A., et al. (2021). Stable-Baselines3.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from . import AGENT_ORDER
from .baselines import RandomPolicy, PolicyOpponent
from .env_wrapper import (
    make_vec_env_for_agent,
    make_vec_env_corner_mixed,
)

# Standaard PPO-hyperparameters, afgestemd op een kleine MLP op 128-byte RAM.
# Pas deze aan in het notebook voor de hyperparameter-experimenten (opdracht 3a).
DEFAULT_PPO_KWARGS = dict(
    learning_rate=2.5e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0,
)


def _make_ppo(env, seed: int, device: str = "auto", ppo_overrides: Optional[dict] = None):
    """Maak een PPO-model met de standaard-hyperparameters (plus overrides)."""
    from stable_baselines3 import PPO

    kwargs = dict(DEFAULT_PPO_KWARGS)
    if ppo_overrides:
        kwargs.update(ppo_overrides)
    return PPO("MlpPolicy", env, seed=seed, device=device, **kwargs)


def _snapshot_path(save_dir: str, agent: str, generation: int) -> str:
    return os.path.join(save_dir, "snapshots", f"{agent}_gen{generation}.zip")


def train_independent(
    generations: int = 4,
    steps_per_agent: int = 200_000,
    n_envs: int = 8,
    survival_bonus: float = 0.01,
    frame_skip: int = 4,
    seed: int = 0,
    save_dir: str = "tournament/models",
    device: str = "auto",
    progress_bar: bool = True,
    ppo_overrides: Optional[dict] = None,
) -> Dict[str, object]:
    """Train vier independent PPO-policies met generatie-gewijs bevroren opponents.

    Returns een dict ``{agent_naam: getraind PPO-model}``. De eindmodellen worden
    opgeslagen als ``<save_dir>/ppo_<agent>.zip``.
    """
    from stable_baselines3 import PPO

    os.makedirs(os.path.join(save_dir, "snapshots"), exist_ok=True)
    monitor_root = os.path.join(save_dir, "monitor")

    # 1) Initialiseer een PPO-model per hoek. Generatie 0 traint tegen een
    #    willekeurige baseline (een natuurlijke, eenvoudige startcurriculum).
    models: Dict[str, object] = {}
    for agent in AGENT_ORDER:
        opponents = {b: RandomPolicy(seed=seed) for b in AGENT_ORDER if b != agent}
        # Wegwerp-env enkel om het PPO-model te construeren; geen monitor-logs hier
        # (de echte logs komen uit de generatie-lus hieronder).
        env = make_vec_env_for_agent(
            agent, opponents, n_envs=n_envs, survival_bonus=survival_bonus,
            frame_skip=frame_skip, seed=seed, monitor_dir=None,
        )
        models[agent] = _make_ppo(env, seed=seed, device=device, ppo_overrides=ppo_overrides)

    # 2) Generatie-lus.
    for gen in range(generations):
        # Bevries de huidige policies door ze naar schijf te schrijven.
        for agent in AGENT_ORDER:
            models[agent].save(_snapshot_path(save_dir, agent, gen))

        # Bouw gedeelde, bevroren opponent-callables (alleen vanaf generatie 1;
        # generatie 0 gebruikt de random baseline als tegenstander).
        frozen: Dict[str, PolicyOpponent] = {}
        if gen > 0:
            for agent in AGENT_ORDER:
                frozen_model = PPO.load(
                    _snapshot_path(save_dir, agent, gen), device="cpu"
                )
                frozen[agent] = PolicyOpponent(frozen_model, deterministic=False)

        for agent in AGENT_ORDER:
            if gen == 0:
                opponents = {b: RandomPolicy(seed=seed + gen) for b in AGENT_ORDER if b != agent}
            else:
                opponents = {b: frozen[b] for b in AGENT_ORDER if b != agent}

            env = make_vec_env_for_agent(
                agent, opponents, n_envs=n_envs, survival_bonus=survival_bonus,
                frame_skip=frame_skip, seed=seed + gen,
                monitor_dir=os.path.join(monitor_root, f"{agent}_gen{gen}"),
            )
            models[agent].set_env(env)
            print(f"[generatie {gen}] train {agent} voor {steps_per_agent} stappen...")
            models[agent].learn(
                total_timesteps=steps_per_agent,
                reset_num_timesteps=False,
                progress_bar=progress_bar,
            )

    # 3) Sla de eindmodellen op onder een stabiele naam.
    for agent in AGENT_ORDER:
        final_path = os.path.join(save_dir, f"ppo_{agent}.zip")
        models[agent].save(final_path)
        print(f"opgeslagen: {final_path}")

    return models


def train_corner_robust(
    total_timesteps: int = 800_000,
    n_envs: int = 8,
    survival_bonus: float = 0.01,
    frame_skip: int = 4,
    seed: int = 0,
    save_dir: str = "tournament/models",
    device: str = "auto",
    progress_bar: bool = True,
    ppo_overrides: Optional[dict] = None,
):
    """Train één gedeeld beleid dat afwisselend alle vier de hoeken bestuurt.

    Bedoeld voor het klassentoernooi, waar onze agent in een willekeurige hoek
    geplaatst kan worden. Tegenstanders zijn de random baseline. Zie het rapport
    voor de bewuste afweging robuustheid-vs-specialisatie.
    """
    os.makedirs(save_dir, exist_ok=True)
    opponents_per_corner = {
        corner: {b: RandomPolicy(seed=seed) for b in AGENT_ORDER if b != corner}
        for corner in AGENT_ORDER
    }
    env = make_vec_env_corner_mixed(
        opponents_per_corner, n_envs=n_envs, survival_bonus=survival_bonus,
        frame_skip=frame_skip, seed=seed,
        monitor_dir=os.path.join(save_dir, "monitor", "corner_robust"),
    )
    model = _make_ppo(env, seed=seed, device=device, ppo_overrides=ppo_overrides)
    print(f"train hoek-robuust beleid voor {total_timesteps} stappen...")
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    final_path = os.path.join(save_dir, "ppo_corner_robust.zip")
    model.save(final_path)
    print(f"opgeslagen: {final_path}")
    return model


if __name__ == "__main__":
    # Korte sanity-run; gebruik in de praktijk het Colab-notebook met meer stappen.
    train_independent(generations=2, steps_per_agent=20_000, n_envs=4, progress_bar=False)
