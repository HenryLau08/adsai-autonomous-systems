"""Hulpmiddelen om de 128-byte RAM-observatie van Warlords te interpreteren.

De exacte betekenis van elke RAM-byte is spel-specifiek en niet gedocumenteerd.
Met de functies hieronder kun je empirisch achterhalen welke bytes overeenkomen
met de bal- en paddleposities, door te kijken welke bytes het meest veranderen
tijdens het spel. De starterscode bevat hiervoor al een eenvoudige diff-tool;
dit module bouwt daarop voort met aggregatie over een hele episode.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import RAM_SIZE


def collect_ram_trace(parallel_env, policies, max_steps: int = 2000, seed: int = 0):
    """Speel een episode en verzamel de RAM-observatie bij elke stap.

    Parameters
    ----------
    parallel_env:
        Een PettingZoo parallel-omgeving (``warlords_v3.parallel_env``).
    policies:
        Dict ``{agent_naam: callable(obs) -> actie}`` voor de vier spelers.
    max_steps:
        Maximaal aantal stappen dat wordt gespeeld.
    seed:
        Seed voor reproduceerbaarheid.

    Returns
    -------
    np.ndarray van vorm ``(T, 128)`` met de RAM-bytes per tijdstap.
    """
    obs, _ = parallel_env.reset(seed=seed)
    trace = []
    for _ in range(max_steps):
        if not parallel_env.agents:
            break
        actions = {}
        for agent in parallel_env.agents:
            policy = policies.get(agent)
            ram = np.asarray(obs[agent], dtype=np.uint8)
            actions[agent] = int(policy(ram)) if policy is not None else 0
        obs, _, _, _, _ = parallel_env.step(actions)
        if obs:
            # Alle agenten zien dezelfde globale RAM; pak er een.
            any_agent = next(iter(obs))
            trace.append(np.asarray(obs[any_agent], dtype=np.uint8))
    parallel_env.close()
    return np.asarray(trace, dtype=np.uint8)


def rank_changing_bytes(trace: np.ndarray, top_k: int = 16):
    """Rangschik RAM-bytes op variabiliteit gedurende een episode.

    Bytes die veel veranderen zijn kandidaten voor dynamische game-state zoals
    bal- en paddleposities; bytes die nauwelijks veranderen zijn meestal
    statische configuratie.

    Returns
    -------
    Lijst van ``(byte_index, aantal_unieke_waarden, standaarddeviatie)`` tuples,
    aflopend gesorteerd op aantal unieke waarden.
    """
    trace = np.asarray(trace, dtype=np.int32)
    if trace.ndim != 2 or trace.shape[1] != RAM_SIZE:
        raise ValueError(f"trace moet vorm (T, {RAM_SIZE}) hebben, kreeg {trace.shape}")

    stats = []
    for idx in range(RAM_SIZE):
        col = trace[:, idx]
        stats.append((idx, int(np.unique(col).size), float(np.std(col))))

    stats.sort(key=lambda row: (row[1], row[2]), reverse=True)
    return stats[:top_k]


def summarize_byte(trace: np.ndarray, byte_index: int):
    """Geef een korte statistiek van een enkele RAM-byte over de episode."""
    col = np.asarray(trace, dtype=np.int32)[:, byte_index]
    return {
        "index": byte_index,
        "min": int(col.min()),
        "max": int(col.max()),
        "mean": float(col.mean()),
        "std": float(col.std()),
        "unique": int(np.unique(col).size),
    }
